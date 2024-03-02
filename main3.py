import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import sys
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from guidance.sd_utils import StableDiffusion

from drag import drag_step, override_forward
import copy
import json
from einops import rearrange
import gradio as gr
from copy import deepcopy

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # text embeds
        self.text_embeds = None
        self.pos_embeds = None

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.dragging_steps = 50
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # load pre-trained gaussian
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            print("Warning: please load pre-trained gaussian.")

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")


        # # prepare embeddings
        # with torch.no_grad():

        #     if self.enable_sd:
        #         if self.opt.imagedream:
        #             self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
        #         else:
        #             self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

        #     if self.enable_zero123:
        #         self.guidance_zero123.get_img_embeds(self.input_img_torch)

        # prepare text_embeddings
        self.pos_embeds = self.guidance_sd.encode_text(self.prompt)  # [1, 77, 768]
        self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

    def get_2d_mask(self, mask_dir):
        source_images, imgname2idx = load_and_preprocess_images(args.folder_path, device="cuda")
        if mask_dir is not None:
            idx2name = dict(zip(imgname2idx.values(), imgname2idx.keys()))
            with open(mask_dir) as f:
                masks_dict = json.load(f)
            masks = []
            for i in range(1, 5):
                print(idx2name)
                mask = masks_dict[idx2name[i]+'.png']
                masks.append(mask)
                # Visualization
                plt.figure(figsize=(6, 6))
                plt.imshow(1 - np.array(mask), cmap='gray')
                plt.colorbar()
                plt.title("Visualization of the 2D Mask")
                plt.axis('off')  # Hide the axis

                # Save the visualization result
                file_path = "mask_visualization_{}.png".format(i)
                plt.savefig(file_path)
                
                
            mask = 1 - torch.tensor(masks)
            # print(mask)
            mask = rearrange(mask, "n h w -> n 1 h w").float().cuda()
        else:
            mask = torch.ones_like(source_images[0, 0, :, :])
            mask = rearrange(mask, "h w -> 1 1 h w").cuda()

        return mask

    def train_step(self):

        torch.autograd.set_detect_anomaly(True)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # per-iteration training
        # sample 4 random camera poses
        self.step += 1
        step_ratio = min(1, self.step / self.opt.iters)

        # update lr
        self.renderer.gaussians.update_learning_rate(self.step)
        
        # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        render_resolution = 256
        images = []
        poses = []  
        vers, hors, radii = [], [], []
        start_points = torch.tensor([[[390, 154]],[[227,127]],[[360, 123]],[[560, 110]]])
        start_points = (start_points * 128 // 800).float()
        end_points = torch.tensor([[[390, 84]],[[227, 61]],[[360, 62]],[[560, 34]]])
        end_points = (end_points * 128 // 800).float() 
        latents_before_editing = []
        masks = []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

        start_hor = -180
        start_ver = -180

        # initialize model
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        #                   beta_schedule="scaled_linear", clip_sample=False,
        #                   set_alpha_to_one=False, steps_offset=1)
        # self.guidance_sd = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

        self.pos_embeds = self.pos_embeds.repeat(4, 1, 1)

        #vers = [0, 0, 0, 0]
        #hors = [-180, -90, 0, 90]
        vers = [0, 0, 0, 0]
        hors = [-90, -90, 90, 90]
        poses = [[
                [
                    0.3260957896709442e+00,
                    0.14048941433429718e+00,
                    -0.934839129447937e+00,
                    -3.768457174301147e+00
                ],
                [
                    -0.9453368186950684e+00,
                    0.04846210405230522e+00,
                    -0.3224746286869049e+00,
                    -1.2999367713928223e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.9888953566551208e+00,
                    0.1486130952835083e+00,
                    0.5990785360336304e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00
                ]],
                [
                [
                    0.462623655796051e+00,
                    -0.4831503927707672e+00,
                    0.7433337569236755e+00,
                    2.996474266052246e+00
                ],
                [
                    0.8865548372268677e+00,
                    0.25211840867996216e+00,
                    -0.38788774609565735e+00,
                    -1.5636255741119385e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.8384521007537842e+00,
                    0.544975221157074e+00,
                    2.1968653202056885e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00
                ]],
                [
                [
                    -0.9964723587036133e+00,
                    0.0291383545845747e+00,
                    -0.07870139926671982e+00,
                    -0.31725549697875977e+00
                ],
                [
                    -0.08392231911420822e+00,
                    -0.3459814488887787e+00,
                    0.9344804883003235e+00,
                    3.7670114040374756e+00
                ],
                [
                    0.000000000000000e+00,
                    0.937788724899292e+00,
                    0.3472062945365906e+00,
                    1.3996332883834839e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00
                ]],
                [
                [
                    0.9830045104026794e+00,
                    0.045349299907684326e+00,
                    -0.17789188027381897e+00,
                    -0.7171050906181335e+00
                ],
                [
                    -0.18358123302459717e+00,
                    0.24282744526863098e+00,
                    -0.9525401592254639e+00,
                    -3.8398122787475586e+00
                ],
                [
                    3.725290298461914e-09,
                    0.9690088033676147e+00,
                    0.2470257729291916e+00,
                    0.9957927465438843e+00
                ],
                [
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    0.0000000000000000e+00,
                    1.0000000000000000e+00
                ]]
                ]
        

        for i in range(self.opt.batch_size):
            # set batch_size = 4

            radius = 0

            radii.append(radius)

            # pose = orbit_camera(self.opt.elevation + vers[i], hors[i], self.opt.radius + radius)

            poses = torch.tensor(poses, dtype=torch.float32)
            cur_cam = MiniCam(poses[i], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
            # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            bg_color = torch.tensor([0, 0, 0],  dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)
            # load start & end points

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1] normalized
            images.append(image)

            # ***images visualize
            # image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            # image_np = (image_np * 255).astype('uint8')
            # image_pil = Image.fromarray(image_np)
            # image_pil.save(f"rendered_images/visualized_image_before_DDIMinversion_{i}.png")
            
        images = torch.cat(images, dim=0)    


        # latent_before_editing = DDIM_inversion(source_images=image,
        #                                    text_embeddings=self.guidance_sd.get_text_embeds)
        inversion_strength = 0.7
        n_actual_inference_step = round(inversion_strength * self.opt.n_inference_step)
        print("Start DDIM inversion...")
        latents_before_editing = self.guidance_sd.invert(images,
                                                    prompt=self.prompt,
                                                    text_embeddings=self.pos_embeds,
                                                    guidance_scale=self.opt.guidance_scale,
                                                    num_inference_steps=self.opt.n_inference_step,
                                                    num_actual_inference_steps=n_actual_inference_step)
        print("invert code shape:", latents_before_editing.shape)
  
        
        # double_pos_embeds = self.pos_embeds.repeat(2, 1, 1)
        # gen_image = self.guidance_sd.sampling(prompt=self.prompt,
        #                                 batch_size=self.opt.batch_size,
        #                                 text_embeddings=double_pos_embeds,
        #                                 latents=torch.cat([init_code, init_code], dim=0),
        #                                 guidance_scale=self.opt.guidance_scale,
        #                                 num_inference_steps=self.opt.n_inference_step,
        #                                 num_actual_inference_steps=n_actual_inference_step)

        # ***test sampling
        # test_prompt = "An apple."
        # test_text_embeds = self.guidance_sd.encode_text(test_prompt)
        # gen_image = self.guidance_sd.sampling(prompt= test_prompt,
        #                                       guidance_scale=self.opt.guidance_scale,
        #                                       text_embeddings=test_text_embeds)
        
        # print("gen_image:", gen_image.shape)

        # ***images visualize
        # for i in range(8):
        #     image_np = gen_image[i].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        #     image_np = (image_np * 255).astype('uint8')
        #     image_pil = Image.fromarray(image_np)
        #     image_pil.save(f"rendered_images/sampling_results_{i}.png")

        # sys.exit()      
        self.guidance_sd.unet.forward = override_forward(self.guidance_sd.unet)


        self.guidance_sd.scheduler.set_timesteps(self.opt.n_inference_step)
        t = self.guidance_sd.scheduler.timesteps[self.opt.n_inference_step - n_actual_inference_step]

        assert len(start_points[0]) == len(end_points[0]), \
            "number of handle point must equals target points"
        if self.pos_embeds is None:
            # text_embeddings = self.guidance_sd.get_text_embeddings(args.prompt)
            print("Warning: please input text prompts.")


        print("init_code:", init_code.shape)

        # the init output feature of unet
        with torch.no_grad():
            unet_output, F0 = self.guidance_sd.forward_unet_features(init_code, t, encoder_hidden_states=self.pos_embeds,
                layer_idx=self.opt.unet_feature_idx, interp_res_h=self.opt.sup_res_h, interp_res_w=self.opt.sup_res_w)
            x_prev_0,_ = self.guidance_sd.step(unet_output, t, init_code)
            # init_code_orig = copy.deepcopy(init_code)

        # prepare optimizable init_code
        init_code.requires_grad_(True)
        init_code = init_code.to(torch.float32)
        optimizer = torch.optim.Adam([init_code.detach()], lr=self.opt.drag_diffusion_lr)

        # prepare for point tracking and background regularization
        handle_points_init = copy.deepcopy(start_points)
        # mask = self.get_2d_mask(self.opt.mask_dir)
        # print("input mask shape:", mask.shape)
        # interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
        # using_mask = interp_mask.sum() != 0.0
        

        # prepare amp scaler for mixed-precision training
        # scaler = torch.cuda.amp.GradScaler()
        for i in range(self.dragging_steps):
            
            latents_after_editing, new_handle_points = drag_step(
                model=self.guidance_sd,
                init_code=latents_before_editing,
                text_embeddings=self.pos_embeds,
                t=t,
                handle_points=start_points,
                handle_points_init=handle_points_init,
                target_points=end_points,
                mask=None,
                step_idx=i,
                F0=F0,
                using_mask=False,
                x_prev_0=x_prev_0,
                interp_mask = None,
                # scaler = scaler,
                optimizer = optimizer,
                args = self.opt)
            
            latents_before_editing = latents_after_editing
            start_points = new_handle_points

            loss = 0.0
            # one step motion supervision & point tracking
            for j in range(self.opt.batch_size):

                # *** loss calculation: add latent after editing
                latent_for_sds = latents_after_editing[j].unsqueeze(0)
                image_for_sds = images[j].unsqueeze(0)
                loss = loss + self.opt.lambda_sd * self.guidance_sd.draggs_train_step(latent_for_sds, image_for_sds, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(image_for_sds, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            print("sds_loss=", loss.item())  

            self.optimizer.zero_grad()    
            loss.backward()
            self.optimizer.step()

            images = []
            for j in range(self.opt.batch_size):
                # set batch_size = 4
                out = self.renderer.render(cur_cam, bg_color=bg_color)
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1] normalized
                images.append(image)
                
            images = torch.cat(images, dim=0) 
            



        # densify and prune
        # if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
        #     viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
        #     self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        #     self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        #     if self.step % self.opt.densification_interval == 0:
        #         self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
        #     if self.step % self.opt.opacity_reset_interval == 0:
        #         self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)   

        self.need_update = True     
        
    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
        path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
        self.renderer.gaussians.save_ply(path)
        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
