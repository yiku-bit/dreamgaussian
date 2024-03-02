import copy
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union


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
    
def point_tracking_without_batch(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   args):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            print("f0:", f0)

            r1, r2 = max(0,int(pi[0])-args.r_p), min(max_r,int(pi[0])+args.r_p+1)
            print("r1:", r1, "r2:", r2)
            c1, c2 = max(0,int(pi[1])-args.r_p), min(max_c,int(pi[1])+args.r_p+1)
            print("c1:", c1, "c2:", c2)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]

            print("F1_neighbor:", F1_neighbor)
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # print(all_dist)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
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
        # scaler,
        optimizer,
        args):


    with torch.autocast(device_type='cuda', dtype=torch.float16):
        unet_output, F1 = model.forward_unet_features(init_code, t, encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_updated,_ = model.step(unet_output, t, init_code)

        # do point tracking to update handle points before computing motion supervision loss
        if step_idx != 0:
            handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
            print('new handle points', handle_points)

        # break if all handle points have reached the targets
        if check_handle_reach_target(handle_points, target_points):
            return None

        loss = 0.0
        _, _, max_r, max_c = F0.shape
        for i in range(handle_points.shape[1]):
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
            # f0_patch = F1[:,:,r1:r2, c1:c2].detach()
            # f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])

            # original code, without boundary protection
            # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
            # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
            loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

    # scaler.scale(loss).backward(retain_graph=True)
    # scaler.step(optimizer)
    # scaler.update()
        
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()    

    print("init_code:", init_code[0][0])
    print("handle_points:", handle_points)
    return init_code, handle_points


def drag_step_without_batch(model,
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
        scaler,
        optimizer,
        cur_batch,
        args):

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        unet_output, F1 = model.forward_unet_features(init_code.unsqueeze(0), t, encoder_hidden_states=text_embeddings.unsqueeze(0),
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_updated,_ = model.step(unet_output, t, init_code)

        # do point tracking to update handle points before computing motion supervision loss
        if step_idx != 0:
            handle_points = point_tracking_without_batch(F0, F1, handle_points, handle_points_init, args)
            # print('new handle points', handle_points)
        else:
            print('handle_points:', handle_points)
        # break if all handle points have reached the targets
        # if check_handle_reach_target(handle_points, target_points):
        #     break

        loss = 0.0
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi, ti = handle_points[i], target_points[i]
            # skip if the distance between target and source is less than 1
            if (ti - pi).norm() < 2.:
                continue

            di = (ti - pi) / (ti - pi).norm()

            # motion supervision
            # with boundary protection
            r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
            c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
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
        print('batch=', cur_batch, '  loss total=%f'%(loss.item()))

        # scaler.scale(loss).backward(retain_graph=True)
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        return init_code, handle_points

# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)
        # print("sample:",sample.shape)
        # print(emb.shape)
        # print(encoder_hidden_states.shape)
        # 3. down 
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward