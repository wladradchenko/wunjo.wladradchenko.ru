import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from safetensors.torch import load_file

from diffusers.utils import export_to_video
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline,StableVideoDiffusionPipeline, EulerDiscreteScheduler, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection


def inpaint2img(init_image, mask_image, controlnet_path="lllyasviel/sd-controlnet-canny", controlnet_type="canny",
                sd_path="runwayml/stable-diffusion-v1-5", prompt=None, n_prompt=None, weights_path=None, strength=0.7,
                scale=7.5, seed=1, save_path=None, device="cuda", control_strength=0.8, guess_mode=False, eta=1.0,
                blur_factor=10, control_guidance_end=1.0):
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


def img2vid(init_image, save_path=None, sd_path="stabilityai/stable-video-diffusion-img2vid-xt", width=1024, height=512,
            fps=12, duration=2, aspect_ratio=1.77, num_frames=12, decode_chunk_size=2):
    # Define the model components
    vae = AutoencoderKLTemporalDecoder.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_path, subfolder="image_encoder", torch_dtype=torch.float16)
    scheduler = EulerDiscreteScheduler.from_pretrained(sd_path, subfolder="scheduler")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(sd_path, subfolder="unet", torch_dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained(sd_path, subfolder="feature_extractor")
    # Init
    pipe = StableVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        scheduler=scheduler,
        unet=unet,
        feature_extractor=feature_extractor
    )
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    pipe.enable_xformers_memory_efficient_attention()  # TODO use if found and not using enable_model_cpu_offload()?

    # Load the conditioning image
    image = init_image.resize((width, height))
    original_image = image

    generator = torch.manual_seed(42)
    if aspect_ratio > 1:
        crop_width = int(width)
        crop_height = int(width/aspect_ratio)
    else:
        crop_width = int(height*aspect_ratio)
        crop_height = int(height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    cropped_frames = []
    # for i in range(int(fps / min(num_frames, fps) * duration)):  # problem with quality after second iter
    motion_bucket_id = [120, 140]
    noise_aug_strength = [0.1, 0.2]
    for i in range(duration):  # duration 2 times the best result
        frames = pipe(image, num_inference_steps=50, decode_chunk_size=decode_chunk_size, generator=generator, num_frames=num_frames, motion_bucket_id=motion_bucket_id[i % len(motion_bucket_id)], noise_aug_strength=noise_aug_strength[i % len(noise_aug_strength)]).frames[0]
        for j, frame in enumerate(frames):
            cropped_frames.append(frame.crop((left, top, right, bottom)))
        else:
            image = frames[-1]
    else:
        cropped_frames = list(reversed(cropped_frames))
        image = original_image
    for i in range(duration):  # duration 2 times the best result
        frames = pipe(image, num_inference_steps=50, decode_chunk_size=decode_chunk_size, generator=generator, num_frames=num_frames, motion_bucket_id=motion_bucket_id[i % len(motion_bucket_id)], noise_aug_strength=noise_aug_strength[i % len(noise_aug_strength)]).frames[0]
        for j, frame in enumerate(frames):
            cropped_frames.append(frame.crop((left, top, right, bottom)))
        else:
            image = frames[-1]

    del pipe

    if save_path:
        video_path = os.path.join(save_path, "generated.mp4")
        export_to_video(cropped_frames, video_path, fps=fps)
        return video_path
    else:
        return cropped_frames
