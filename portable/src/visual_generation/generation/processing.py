import os
import torch
import numpy as np
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from .omnigen import OmniGenPipeline


def supports_bfloat16():
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def imgs2img(omnigen_path="Shitao/OmniGen-v1", input_images=None, prompt=None, width=512, height=512, offload_model=True,
             guidance_scale=2.5, seed=1, save_path=None, device="cuda", model_cache=None, progress_callback=None):
    prompt = prompt if prompt else "colorful"
    # Get validate images in one line
    i = 0
    valid_input_images = ([input_image for input_image in (input_images or []) if (file_name := os.path.basename(input_image)) in prompt and (prompt := prompt.replace(file_name, f"<img><|image_{i + 1}|></img>")) and (i:=i+1)] or None)

    # Load model from cache or initialize it
    pipe = model_cache.get_model("omnigen") if model_cache is not None else None

    if pipe is None:
        if progress_callback:
            progress_callback(5, "Load model in memory...")

        pipe = OmniGenPipeline.from_pretrained(model_name=omnigen_path, device=device)

        if device != "cpu":
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

    # Update cache for fast generation
    model_cache.update_model("omnigen", pipe) if model_cache else None

    width = int(np.round(width / 16.0)) * 16
    height = int(np.round(height / 16.0)) * 16

    # generate image
    images = pipe(
        prompt=prompt,
        input_images=valid_input_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=1.6,
        offload_model=offload_model,
        seed=seed,
        use_kv_cache=False,
        offload_kv_cache=False,
        callback=progress_callback
    )

    del pipe

    if save_path:
        images[0].save(save_path)
        return save_path
    else:
        return images[0]


def txt2img(sd_path="runwayml/stable-diffusion-v1-5", prompt=None, n_prompt=None, weights_path=None, width=512, height=512, strength=0.7, scale=7.5, seed=1, save_path=None, device="cuda"):
    prompt = prompt if prompt else "colorful"

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Define the model components
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", torch_dtype=torch.float16)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(sd_path, subfolder="safety_checker", torch_dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained(sd_path, subfolder="feature_extractor")

    # Load the weights from the safetensors file
    if weights_path:
        weights = load_file(weights_path)  # weights_path
        unet.load_state_dict(weights, strict=False)

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
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

    width = int(np.round(width / 64.0)) * 64
    height = int(np.round(height / 64.0)) * 64

    # generate image
    image = pipe(
        prompt,
        negative_prompt=n_prompt,
        width=width,
        height=height,
        num_inference_steps=20,
        generator=generator,
        eta=1.0,
        strength=strength,
        guidance_scale=scale,
        controlnet_conditioning_scale=0.75,
    ).images[0]

    del pipe

    if save_path:
        image.save(save_path)
        return save_path
    else:
        return image
