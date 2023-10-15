import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from skimage import exposure
from safetensors.torch import load_file
import einops
from pytorch_lightning import seed_everything
import torch.nn.functional as F
import torchvision.transforms as T

from .blend import BlendType, blendLayers
from .config import RenderConfig
from .ddim_v_hacked import DDIMVSampler
from .freeu import freeu_forward
from .controller import AttentionControl

from diffusers.src.controlnet.annotator.canny import CannyDetector
from diffusers.src.controlnet.annotator.hed import HEDdetector
from diffusers.src.controlnet.annotator.util import HWC3
from diffusers.src.controlnet.cldm.cldm import ControlLDM
from diffusers.src.controlnet.cldm.model import create_model, load_state_dict

from diffusers.src.flow.flow_utils import get_warped_and_mask
from diffusers.src.gmflow.gmflow import GMFlow


def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(cv2.cvtColor(np.asarray(original_image),  cv2.COLOR_RGB2LAB), correction, channel_axis=2), cv2.COLOR_LAB2RGB).astype('uint8'))
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)
    return image


@torch.no_grad()
def find_flat_region(mask):
    device = mask.device
    kernel_x = torch.Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    mask_ = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    grad_x = torch.nn.functional.conv2d(mask_, kernel_x)
    grad_y = torch.nn.functional.conv2d(mask_, kernel_y)
    return ((abs(grad_x) + abs(grad_y)) == 0).float()[0]


def numpy2tensor(img):
    x0 = torch.from_numpy(img.copy()).float().cuda() / 255.0 * 2.0 - 1.
    x0 = torch.stack([x0], dim=0)
    return einops.rearrange(x0, 'b h w c -> b c h w').clone()


def render(cfg: RenderConfig, args, masks, frame_files_with_interval, sd_model_path, controlnet_model_path, vae_model_path,
           gmflow_model_path, frame_path, mask_path):
    # Load models
    if cfg.control_type == 'hed':
        detector = HEDdetector()
    elif cfg.control_type == 'canny':
        canny_detector = CannyDetector()
        low_threshold = cfg.canny_low
        high_threshold = cfg.canny_high

        def apply_canny(x):
            return canny_detector(x, low_threshold, high_threshold)

        detector = apply_canny
    else:
        raise "Undefined control_type"

    # load controlnet
    diffusers_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cldm_v15 = os.path.join(diffusers_src, "controlnet", "models", "cldm_v15.yaml")
    model: ControlLDM = create_model(cldm_v15).cpu()

    if cfg.control_type == 'hed':
        model.load_state_dict(load_state_dict(controlnet_model_path, location='cuda'))
    elif cfg.control_type == 'canny':
        model.load_state_dict(load_state_dict(controlnet_model_path, location='cuda'))

    model = model.cuda()
    model.control_scales = [cfg.control_strength] * 13

    if sd_model_path:
        model_ext = os.path.splitext(sd_model_path)[1]
        if model_ext == '.safetensors':
            model.load_state_dict(load_file(sd_model_path), strict=False)
        elif model_ext == '.ckpt' or model_ext == '.pth':
            model.load_state_dict(torch.load(sd_model_path)['state_dict'], strict=False)

    try:
        model.first_stage_model.load_state_dict(torch.load(vae_model_path)['state_dict'],strict=False)
    except Exception:
        print('Warning: We suggest you download the fine-tuned VAE',
              'otherwise the generation quality will be degraded')

    model.model.diffusion_model.forward = freeu_forward(model.model.diffusion_model, *cfg.freeu_args)
    ddim_v_sampler = DDIMVSampler(model)

    flow_model = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6,
    ).to('cuda')

    checkpoint = torch.load(gmflow_model_path, map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval()

    num_samples = 1
    ddim_steps = 20
    eta = 0.0

    blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
    totensor = T.PILToTensor()

    style_update_freq = cfg.style_update_freq

    mask_period = cfg.mask_period
    firstx0 = True
    controller = AttentionControl(cfg.inner_strength, cfg.mask_period, cfg.cross_period, cfg.ada_period, cfg.warp_period, cfg.loose_cfattn)

    for mask_id in masks.keys():
        # for each mask
        mask_files_path = masks[mask_id]["frame_files_path"]
        mask_files_path = sorted(os.listdir(mask_files_path))
        common_mask_files = sorted(list(set(mask_files_path) & set(frame_files_with_interval)))

        prompt = masks[mask_id]["prompt"]
        a_prompt = args.a_prompt
        n_prompt = masks[mask_id]["n_prompt"] + args.n_prompt
        prompt = prompt + ', ' + a_prompt

        x0_strength = 1 - float(masks[mask_id]["input_strength"])
        scale = float(masks[mask_id]["input_scale"])

        seed = int(masks[mask_id]["input_seed"])
        if seed == -1:
            seed = random.randint(0, 65535)

        with torch.no_grad():
            first_frame_file = os.path.join(frame_path, common_mask_files[0])
            frame = cv2.imread(first_frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = HWC3(frame)
            H, W, C = img.shape

            img_ = numpy2tensor(img)
            encoder_posterior = model.encode_first_stage(img_.cuda())
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = detector(img)
            detected_map = HWC3(detected_map)
            # For visualization
            detected_img = 255 - detected_map

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            cond = {
                'c_concat': [control],
                'c_crossattn':
                    [model.get_learned_conditioning([prompt] * num_samples)]
            }
            un_cond = {
                'c_concat': [control],
                'c_crossattn':
                    [model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)

            controller.set_task('initfirst')
            seed_everything(seed)
            samples, _ = ddim_v_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                               unconditional_guidance_scale=scale, unconditional_conditioning=un_cond,
                                               controller=controller, x0=x0, strength=x0_strength)

            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img
            first_result = pre_result
            first_img = pre_img
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,255).astype(np.uint8)

        color_corrections = setup_color_correction(Image.fromarray(x_samples[0]))
        Image.fromarray(x_samples[0]).save(os.path.join(cfg.first_dir, 'first.jpg'))
        cv2.imwrite(os.path.join(cfg.first_dir, 'first_edge.jpg'), detected_img)

        for common_frame_name in common_mask_files:
            # load frame
            frame = cv2.imread(os.path.join(frame_path, common_frame_name))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = HWC3(frame)

            # load mask
            inpaint_mask_frame = cv2.imread(os.path.join(mask_path, f"mask_{mask_id}", common_frame_name), cv2.IMREAD_GRAYSCALE)
            # Binarize the image
            _, binary_mask = cv2.threshold(inpaint_mask_frame, 128, 1, cv2.THRESH_BINARY)
            # Resize to desired shape
            resized_mask = cv2.resize(binary_mask, (shape[2], shape[1]))
            # Convert to tensor and adjust dimensions
            inpaint_mask = torch.tensor(resized_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

            if cfg.color_preserve:
                img_ = numpy2tensor(img)
            else:
                img_ = apply_color_correction(color_corrections, Image.fromarray(img))
                img_ = totensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1

            encoder_posterior = model.encode_first_stage(img_.cuda())
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = detector(img)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            cond['c_concat'] = [control]
            un_cond['c_concat'] = [control]

            image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
            image2 = torch.from_numpy(img).permute(2, 0, 1).float()
            warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(flow_model, image1, image2, pre_result, False)
            blend_mask_pre = blur(F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
            blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

            image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
            warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(flow_model, image1, image2, first_result, False)
            blend_mask_0 = blur(F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
            blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

            if firstx0:
                mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
                controller.set_warp(F.interpolate(bwd_flow_0 / 8.0, scale_factor=1. / 8, mode='bilinear'), mask)
            else:
                mask = 1 - F.max_pool2d(blend_mask_pre, kernel_size=8)
                controller.set_warp(F.interpolate(bwd_flow_pre / 8.0, scale_factor=1. / 8, mode='bilinear'), mask)

            controller.set_task('keepx0, keepstyle')
            seed_everything(seed)
            samples, intermediates = ddim_v_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                inpaint_mask=inpaint_mask,
                x0=x0,
                strength=x0_strength)
            direct_result = model.decode_first_stage(samples)

            # not pixelfusion
            blend_results = (1 - blend_mask_pre) * warped_pre + blend_mask_pre * direct_result
            blend_results = (1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

            bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
            blend_mask = blur(F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
            blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

            encoder_posterior = model.encode_first_stage(blend_results)
            xtrg = model.get_first_stage_encoding(encoder_posterior).detach()  # * mask
            blend_results_rec = model.decode_first_stage(xtrg)
            encoder_posterior = model.encode_first_stage(blend_results_rec)
            xtrg_rec = model.get_first_stage_encoding(encoder_posterior).detach()
            xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
            blend_results_rec_new = model.decode_first_stage(xtrg_)
            tmp = (abs(blend_results_rec_new - blend_results).mean(dim=1, keepdims=True) > 0.25).float()
            mask_x = F.max_pool2d((F.interpolate(tmp, scale_factor=1 / 8., mode='bilinear') > 0).float(), kernel_size=3, stride=1, padding=1)
            mask = (1 - F.max_pool2d(1 - blend_mask, kernel_size=8))  # * (1-mask_x)
            # noise rescale
            noise_rescale = find_flat_region(mask)

            masks_list = []
            for i in range(ddim_steps):
                if i <= ddim_steps * mask_period[0] or i >= ddim_steps * mask_period[1]:
                    masks_list += [None]
                else:
                    masks_list += [mask * cfg.mask_strength]

            xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1

            tasks = 'keepstyle, keepx0'
            if not firstx0:
                tasks += ', updatex0'
            if i % style_update_freq == 0:
                tasks += ', updatestyle'
            controller.set_task(tasks, 1.0)

            seed_everything(seed)
            samples, _ = ddim_v_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=x0_strength,
                xtrg=xtrg,
                mask=masks_list,
                inpaint_mask=inpaint_mask,
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            pil_created_image = Image.fromarray(viz[0])

            pil_created_image.save(os.path.join(cfg.key_subdir, common_frame_name))
            pil_created_image.save(os.path.join(frame_path, common_frame_name))

    # empty cache
    del model, ddim_v_sampler, flow_model, checkpoint, weights, controller
    torch.cuda.empty_cache()