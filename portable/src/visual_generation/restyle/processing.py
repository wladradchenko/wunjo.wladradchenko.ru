import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from skimage import exposure
from safetensors.torch import load_file
import einops
import torch.nn.functional as F

from .blend import BlendType, blendLayers

from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor


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


def load_state_dict_compat(model, state_dict_path, device='cpu'):
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model_state_dict = model.state_dict()

    # Filter out keys that do not match the model's expected keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Identify any keys in the state dict that don't match the model's keys
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())

    if unexpected_keys:
        print(f"Warning: The following keys are unexpected and will be ignored: {unexpected_keys}")

    if missing_keys:
        print(f"Warning: The following keys are missing in the state_dict: {missing_keys}")

    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)


def inpaint2img(init_image, mask_image, controlnet_path="lllyasviel/sd-controlnet-canny", controlnet_type="canny",
                sd_path="runwayml/stable-diffusion-v1-5", prompt=None, n_prompt=None, weights_path=None, strength=0.7,
                scale=7.5, seed=1, save_path=None, device="cuda", control_strength=0.8, guess_mode=False, eta=1.0,
                blur_factor=10, control_guidance_end=1.0, model_cache=None):
    prompt = prompt if prompt else "colorful"
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def make_canny_condition(image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    if controlnet_type == "canny":
        control_image = make_canny_condition(init_image)
    else:
        control_image = init_image

    # Load model from cache or initialize it
    pipe = model_cache.get_model(f"inpaint2img_{controlnet_path}_{weights_path}") if model_cache is not None else None
    controlnet = None

    if pipe is None:  # TODO clear cache?
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)

        # Define the model components
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", torch_dtype=torch.float16)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(sd_path, subfolder="safety_checker", torch_dtype=torch.float16)  # to filter naked content
        feature_extractor = CLIPImageProcessor.from_pretrained(sd_path, subfolder="feature_extractor")

        # Load the weights from the safetensors file
        if weights_path:
            weights = load_file(weights_path)  # weights_path
            unet.load_state_dict(weights, strict=False)

        pipe = StableDiffusionControlNetInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler"),
            safety_checker=None,  # safety_checker,
            requires_safety_checker=False,
            feature_extractor=feature_extractor,
        ).to(device)

        if device == "cpu":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_vae_tiling()
            pipe.enable_xformers_memory_efficient_attention()

        # Update cache
        model_cache.update_model(f"inpaint2img_{controlnet_path}_{weights_path}", pipe) if model_cache else None

    # generate image
    image = pipe(
        prompt,
        negative_prompt=n_prompt,
        num_inference_steps=20,
        generator=generator,
        eta=eta,
        guess_mode=guess_mode,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        strength=strength,
        guidance_scale=scale,
        controlnet_conditioning_scale=control_strength,
        control_guidance_end=control_guidance_end,
    ).images[0]

    del pipe, controlnet

    # Improve image result
    # Create a mask using the binary image for blending
    mask = mask_image.convert('L').filter(ImageFilter.GaussianBlur(blur_factor))
    # Blend the images based on the mask
    image_resized = image.resize((init_image.size), Image.Resampling.LANCZOS)
    blended_image = Image.composite(image_resized, init_image, mask)

    if save_path:
        blended_image.save(save_path)
        return save_path
    else:
        return blended_image
