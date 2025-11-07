import os
import cv2
import json
import uuid

from tqdm import tqdm


from visual_processing.utils.videoio import VideoManipulation
from backend.config import Settings
from backend.folders import ALL_MODEL_FOLDER
from backend.download import get_nested_url, is_connected, download_model, check_download_size


def content_enhancer(media_path, save_folder, method='gfpgan', device='cpu', fps=30, progress_callback=None):
    if os.path.isfile(os.path.join(ALL_MODEL_FOLDER, 'all_models.json')):
        with open(os.path.join(ALL_MODEL_FOLDER, 'all_models.json'), 'r', encoding="utf8") as file:
            config_deepfake = json.load(file)
    else:
        raise "[Error] Config file all_models.json is not exist"

    local_model_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
    save_frame_folder = os.path.join(save_folder, "enhancer")
    os.makedirs(save_frame_folder, exist_ok=True)

    if method == 'gfpgan':
        from .gfpganer import GFPGANer

        model_path = os.path.join(local_model_path, 'GFPGANv1.4.pth')
        url = get_nested_url(config_deepfake, ["gfpgan", "GFPGANv1.4.pth"])

        if not os.path.isfile(model_path):
            # check what is internet access
            if is_connected(model_path):
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": True})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = True
                # download pre-trained models from url
                model_path = url
            else:
                print(f"Not internet connection to download model {url} in {model_path}.")
                raise

        restorer = GFPGANer(
            model_path=model_path,
            root_dir=local_model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
    elif method == 'animesgan':
        # https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inference_realesrgan_video.py
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from .realesrgan import RealESRGANer

        model_path = os.path.join(local_model_path, 'realesr-animevideov3.pth')
        url = get_nested_url(config_deepfake, ["gfpgan", "realesr-animevideov3.pth"])

        if device == "cpu":  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            return media_path

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        if not os.path.isfile(model_path):
            # check what is internet access
            if is_connected(model_path):
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN": True})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN = True
                # download pre-trained models from url
                download_model(model_path, url, progress_callback=progress_callback)
            else:
                print(f"Not internet connection to download model {url} in {model_path}.")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN:
                check_download_size(model_path, url)

        restorer = RealESRGANer(
            scale=4,  # 4
            model_path=model_path,
            root_dir=local_model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
    elif method == 'realesrgan':
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from .realesrgan import RealESRGANer

        general_model_path = os.path.join(local_model_path, 'realesr-general-x4v3.pth')
        general_url = get_nested_url(config_deepfake, ["gfpgan", "realesr-general-x4v3.pth"])
        wdn_model_path = os.path.join(local_model_path, 'realesr-general-wdn-x4v3.pth')
        wdn_url = get_nested_url(config_deepfake, ["gfpgan", "realesr-general-wdn-x4v3.pth"])
        models_path = [general_model_path, wdn_model_path]

        if device == "cpu":  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            return media_path

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        denoise_strength = 0.5
        dni_weight = [denoise_strength, 1 - denoise_strength]

        if not os.path.isfile(general_model_path):
            # check what is internet access
            if is_connected(general_model_path):
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN": True})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN = True
                # download pre-trained models from url
                download_model(models_path[0], general_url, progress_callback=progress_callback)
            else:
                print(f"Not internet connection to download model {general_url} in {models_path[0]}.")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN:
                check_download_size(models_path[0], general_url)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        if not os.path.isfile(wdn_model_path):
            # check what is internet access
            if is_connected(wdn_model_path):
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN": True})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN = True
                # download pre-trained models from url
                download_model(models_path[1], wdn_url, progress_callback=progress_callback)
            else:
                print(f"Not internet connection to download model {general_url} in {models_path[1]}.")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN:
                check_download_size(models_path[1], wdn_url)

        restorer = RealESRGANer(
            scale=4,  # 4
            model_path=models_path,
            root_dir=local_model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            dni_weight=dni_weight,
        )
    else:
        raise ValueError(f'Wrong model version {method}.')

    # Updating progress
    if progress_callback:
        progress_callback(100, "Inspection models...")

    if media_path.endswith(('.mp4', '.avi', '.mov', '.gif')):  # Video or GIF
        cap = cv2.VideoCapture(media_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            if method == 'gfpgan':
                _, _, output = restorer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            elif method in ['animesgan', 'realesrgan']:
                # downgrade quality before upscale
                frame = resize_frame_downscale(frame, scale=0.5)
                output, _ = restorer.enhance(frame)
                # downscale after upscale
                output = resize_frame_downscale(output, scale=0.5)
            else:
                raise ValueError(f'Wrong model version {method}.')

            # Create a unique filename based on the current frame index
            file_name = f"{idx:04d}.png"
            save_path = os.path.join(save_frame_folder, file_name)
            cv2.imwrite(save_path, output)

            # Updating progress
            if progress_callback:
                progress_callback(round(idx / total_frames * 100, 0), "Enhancement...")

        cap.release()
        file_name = VideoManipulation.save_video_from_frames(frame_names="%04d.png", save_path=save_frame_folder, fps=fps, alternative_save_path=save_folder)
    else:  # Image
        file_name = str(uuid.uuid4()) + '.png'
        save_path = os.path.join(save_folder, file_name)
        img = cv2.imread(media_path)
        if method == 'gfpgan':
            _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            print("GFPGANer finished")
        elif method in ['animesgan', 'realesrgan']:
            output, _ = restorer.enhance(img)
            print("RealESRGANer finished")
        else:
            raise ValueError(f'Wrong model version {method}.')

        cv2.imwrite(save_path, output)

    # Updating progress
    if progress_callback:
        progress_callback(100, "Enhancement...")

    # Update config about models if process was successfully
    if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN and method == 'animesgan':
        Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN": False})
        Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN = False
    if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN and method == 'realesrgan':
        Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN": False})
        Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN = False

    return file_name


def resize_frame_downscale(frame, scale=0.25, min_side=320):
    # Calculate new dimensions
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    # Ensure one side is not less than min_side
    if new_width < min_side and new_height < min_side:
        if new_width < new_height:
            aspect_ratio = float(width) / float(height)
            new_width = min_side
            new_height = int(new_width / aspect_ratio)
        else:
            aspect_ratio = float(height) / float(width)
            new_height = min_side
            new_width = int(new_height / aspect_ratio)
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame