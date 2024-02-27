import copy
import torch
import torch.nn.functional as F

def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   args):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(handle_points.shape[1]):
            pi0, pi = handle_points_init[:,i,:], handle_points[:,i,:]
            # f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            f0 = []
            for j in range(F0.shape[0]):
                f0.append(F0[j,:,int(pi0[j,0]),int(pi0[j,1])])
            f0 = torch.stack(f0, dim=0)
            r1, r2 = torch.max(torch.zeros(4), pi[:,0].long()-args.r_p).long(), torch.min(torch.full_like(pi[:,0], max_r),pi[:,0].long()+args.r_p+1).long()
            c1, c2 = torch.max(torch.zeros(4) ,pi[:,1].long()-args.r_p).long(), torch.min(torch.full_like(pi[:, 1], max_c), pi[:,1].long()+args.r_p+1).long()
            # F1_neighbor = F1[:, :, r1:r2, c1:c2]
            F1_neighbor = []
            for j in range(F1.shape[0]):
                ans = F1[j, :, r1[j]:r2[j], c1[j]:c2[j]]
                F1_neighbor.append(ans)
            F1_neighbor = torch.stack(F1_neighbor, dim=0)
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            # all_dist = all_dist.squeeze(dim=0)
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            for j in range(F1.shape[0]):
                row, col = divmod(all_dist[j].argmin().item(), all_dist[j].shape[-1])
                handle_points[j,i,0] = r1[j] + row
                handle_points[j,i,1] = c1[j] + col
        return handle_points
    
def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def drag_step(model,
        init_code,
        text_embeddings,
        t,
        handle_points,
        handle_points_init,
        target_points,
        mask,
        step_idx,
        F0,
        using_mask,
        x_prev_0,
        interp_mask,
        args):
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam([init_code], lr=args.drag_diffusion_lr)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        unet_output, F1 = model.forward_unet_features(init_code, t, encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_updated,_ = model.step(unet_output, t, init_code)

        # do point tracking to update handle points before computing motion supervision loss
        if step_idx != 0:
            handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
            # print('new handle points', handle_points)

        # break if all handle points have reached the targets
        if check_handle_reach_target(handle_points, target_points):
            return None

        loss = 0.0
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi, ti = handle_points[:,i,:], target_points[:,i,:]
            # skip if the distance between target and source is less than 1
            if (ti - pi).norm() < 2.:
                continue

            di = (ti - pi) / (ti - pi).norm()

            # motion supervision
            # with boundary protection
            r1, r2 = torch.max(torch.zeros(4), pi[:,0].long()-args.r_m).long(), torch.min(torch.full_like(pi[:,0], max_r),pi[:,0].long()+args.r_m+1).long()
            c1, c2 = torch.max(torch.zeros(4) ,pi[:,1].long()-args.r_m).long(), torch.min(torch.full_like(pi[:, 1], max_c), pi[:,1].long()+args.r_m+1).long()
            f0_patch = []
            f1_patch = []
            for i in range(F1.shape[0]):
                ans = F1[i, :, r1[i]:r2[i], c1[i]:c2[i]]
                f0_patch.append(ans.detach().reshape((-1)))
                ans = interpolate_feature_patch(F1[i,:,:,:].unsqueeze(0),r1[i]+di[i, 0],r2[i]+di[i,0],c1[i]+di[i,1],c2[i]+di[i,1])[0]
                f1_patch.append(ans.reshape((-1)))
            f0_patch = torch.cat(f0_patch, dim=0)
            f1_patch = torch.cat(f1_patch, dim=0)
            f0_patch = F1[:,:,r1:r2, c1:c2].detach()
            f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])

            # original code, without boundary protection
            # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
            # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
            loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()