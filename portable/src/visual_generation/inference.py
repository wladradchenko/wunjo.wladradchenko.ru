import os
import re
import sys
import cv2
import uuid
import json
import torch
import random
import shutil
import requests
from PIL import Image, ImageFilter
from time import strftime
from scenedetect import detect, ContentDetector

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modules_root_path = os.path.join(root_path, "visual_generation")
sys.path.insert(0, modules_root_path)

from restyle import inpaint2img
from generation import txt2img, imgs2img, Outpaint
from highlights import SmolVLM2, Whisper
from visual_generation.utils.mediaio import (
    save_video_frames_cv2, save_image_frame_cv2, vram_limit_device_resolution_diffusion, resize_and_save_image,
    save_empty_mask, get_new_dimensions, resize_and_save_video, vram_limit_device_resolution_only_ebsynth, resize_pil_image64
)

sys.path.pop(0)

from backend.folders import TMP_FOLDER, ALL_MODEL_FOLDER, MEDIA_FOLDER
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity, get_best_device
from backend.download import download_model, unzip, check_download_size, get_nested_url, is_connected, download_huggingface_repo, is_admin
from backend.config import ModelsConfig, Settings, HuggingfaceRepoConfig
from visual_processing.utils.audio import cut_audio
from visual_processing.utils.videoio import VideoManipulation


class DiffusionSafeTensorConfig:
    DIFFUSION_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/diffusion.json"
    DEFAULT_CONFIG = []
    CUSTOM_CONFIG = []

    def __call__(self):
        if len(self.DEFAULT_CONFIG) == 0:
            print("Diffusion model config updated")
            self.get_default_config()
        self.get_custom_config()
        if isinstance(self.DEFAULT_CONFIG, list) and isinstance(self.CUSTOM_CONFIG, list):
            return self.DEFAULT_CONFIG + self.CUSTOM_CONFIG
        if isinstance(self.DEFAULT_CONFIG, list):
            return self.DEFAULT_CONFIG
        if isinstance(self.CUSTOM_CONFIG, list):
            return self.CUSTOM_CONFIG
        return []

    def get_default_config(self):
        if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
            try:
                response = requests.get(self.DIFFUSION_JSON_URL)
                with open(os.path.join(ALL_MODEL_FOLDER, 'diffusion.json'), 'wb') as file:
                    file.write(response.content)
            except:
                print("Not internet connection to get actual versions of diffuser models")

        if not os.path.isfile(os.path.join(ALL_MODEL_FOLDER, 'diffusion.json')):
            self.DEFAULT_CONFIG = []
        else:
            with open(os.path.join(ALL_MODEL_FOLDER, 'diffusion.json'), 'r', encoding="utf8") as file:
                try:
                    self.DEFAULT_CONFIG = json.load(file)
                except:
                    self.DEFAULT_CONFIG = []

    def get_custom_config(self):
        if not os.path.isfile(os.path.join(ALL_MODEL_FOLDER, 'custom_diffusion.json')):
            self.CUSTOM_CONFIG = []
        else:
            with open(os.path.join(ALL_MODEL_FOLDER, 'custom_diffusion.json'), 'r', encoding="utf8") as file:
                try:
                    self.CUSTOM_CONFIG = json.load(file)
                except:
                    self.CUSTOM_CONFIG = []


file_models_config = ModelsConfig()
file_diffusion_config = DiffusionSafeTensorConfig()
repo_config = HuggingfaceRepoConfig()


class BaseModule:
    @staticmethod
    def get_media_url(path: str):
        return path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_file(file, formats):
        """
        Validates if the file matches one of the specified formats.
        :param file: File to be checked
        :param formats: List of formats (e.g., ['audio', 'image', 'video'])
        :return: True if the file is valid for at least one format, otherwise False
        """
        for format_type in formats:
            if format_type == 'audio' and BaseModule.is_valid_audio(file):
                return True
            elif format_type == 'image' and BaseModule.is_valid_image(file):
                return True
            elif format_type == 'video' and BaseModule.is_valid_video(file):
                return True
        return False

    @staticmethod
    def is_valid_audio(file):
        import librosa
        # Check if the file has a recognized audio extension
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']  # Add more if needed
        _, ext = os.path.splitext(file)
        if ext.lower() in audio_extensions:
            return True
        # Try to load the file as audio and check if it's valid
        try:
            # Using librosa
            librosa.load(file, sr=None)
            return True
        except Exception as err:
            pass
        return False

    @staticmethod
    def is_valid_image(file):
        return VideoManipulation.check_media_type(file) == "static"

    @staticmethod
    def is_valid_video(file):
        return VideoManipulation.check_media_type(file) == "animated"

    @staticmethod
    def set_seed(seed: int = -1):
        if isinstance(seed, int) and seed >= 0:
            return seed
        elif isinstance(seed, str) and seed.isdigit():
            return int(seed)
        else:
            return random.randint(1, 1000000)

    @staticmethod
    def is_img_url(img_url) -> bool:
        # Check if the URL starts with "https://" or "http://"
        if not re.match(r'^https?://', img_url):
            return False

        # Check if the URL ends with ".jpeg", ".jpg", or ".png"
        if re.search(r'\.(jpeg|jpg|png)$', img_url):
            return True
        else:
            return False


    @staticmethod
    def get_sd_models():
        config = []
        diffuser_folder = os.path.join(ALL_MODEL_FOLDER, "diffusion")

        for c in file_diffusion_config():
            if c.get("model") is None:
                continue
            else:
                sd_model_path = os.path.join(diffuser_folder, str(c["model"]))
                if not os.path.exists(sd_model_path) and c.get("url") is None:
                    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
                        continue  # model not found, url not found and user set offline mode
            if c.get("img") is not None:
                if BaseModule.is_img_url(c["img"]):  # if url
                    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
                        c["img"] = None
                else:  # if media folder file
                    sd_img_path = os.path.join(diffuser_folder, str(c["img"]))
                    if not os.path.exists(sd_img_path):
                        c["img"] = None
                    else:
                        # cover path
                        c["img"] = BaseModule.get_media_url(sd_img_path)
            if c.get("name") is None:
                c["name"] = "Unknown"

            config.append(c)

        return config

    @staticmethod
    def get_model_info(model_path, url_key):
        return {
            "path": model_path,
            "url": get_nested_url(file_models_config(), url_key) if isinstance(url_key, list) else url_key,
            "is_exist": os.path.exists(model_path)
        }

    @staticmethod
    def validate_models(model_path, model_url, user_name, method_name: str, progress_callback=None):
        """
        Helper function to check if the model exists, validate its size, and download if needed.
        """
        if not os.path.exists(model_path):
            if is_connected(model_url):
                lprint(msg=f"Model {model_path} not found. Downloading model.", user=user_name, level="info")
                BaseModule.update_model_download_checker(method_name)
                download_model(model_path, model_url, progress_callback=progress_callback)
            else:
                lprint(msg=f"No internet connection to download model {model_url}.", user=user_name, level="error")
                raise ConnectionError(f"Cannot download model {model_path}")
        else:
            # Only check the download size if the setting is enabled
            if getattr(Settings, method_name, False):
                check_download_size(model_path, model_url, progress_callback=progress_callback)

    @staticmethod
    def validate_zip(unzip_path, zip_path, zip_url, user_name, method_name: str, progress_callback=None):
        """
        Helper function to check if the model exists, validate its size, and download if needed.
        """
        if not os.path.exists(unzip_path):
            if is_connected(zip_url):
                lprint(msg=f"Zip not found. Downloading zip.", user=user_name, level="info")
                BaseModule.update_model_download_checker(method_name)
                download_model(zip_path, zip_url, progress_callback=progress_callback)
                unzip(zip_path, unzip_path)
            else:
                lprint(msg=f"No internet connection to download zip {zip_url}.", user=user_name, level="error")
                raise ConnectionError(f"Cannot download zip {zip_path}")
        else:
            # Only check the download size if the setting is enabled
            if getattr(Settings, method_name, False):
                check_download_size(zip_path, zip_url, progress_callback=progress_callback)
                if not os.listdir(unzip_path):
                    unzip(zip_path, unzip_path)

    @staticmethod
    def update_model_download_checker(method_name: str):
        """
        Update settings to enable model download size checking.
        """
        lprint(msg=f"Updating download size checker for method: {method_name}.")
        # Update config with method-specific download size check flag
        Settings.config_manager.update_config({method_name: True})
        # Set the dynamic attribute in Settings to True
        setattr(Settings, method_name, True)


class SceneDetect(BaseModule):
    """Detect scene change"""
    @staticmethod
    def detecting(user_name: str, source_file: str, interval: int = 10, threshold: float = 50.0, is_path: bool = False):
        method = "Scenedetect"
        lprint(msg=f"[{method}] Start content scenedetect.", user=user_name, level="info")
        # save dir
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        os.makedirs(local_save_dir, exist_ok=True)

        # Get the list of detected scenes.
        scene_list = detect(source_file, ContentDetector(threshold=threshold))

        # read and save frames
        cap = cv2.VideoCapture(source_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_frames = []
        scene_files = []

        if len(scene_list) > 0:
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                if interval:
                    scene_frames.extend(range(start_frame, end_frame, interval))
                    scene_frames.append(end_frame - 1)
                else:
                    scene_frames.extend([start_frame, end_frame - 1])
                lprint(msg=f"[{method}] Scene {i + 1}: {scene[0]} [{start_frame}] to {scene[1]} [{end_frame - 1}].", user=user_name, level="info")
            else:
                scene_frames.append(total_frames - 1)
        else:
            if interval:
                scene_frames = list(range(0, total_frames - 1, interval))
                scene_frames.append(total_frames - 1)
            else:
                scene_frames = [0, total_frames - 1]

        # Ensure unique frames and save them
        for i in sorted(set(scene_frames)):
            if i <= total_frames - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    file_path = os.path.join(local_save_dir, f"{i}.jpg")
                    scene_files.append({"file": file_path if is_path else SceneDetect.get_media_url(file_path), "frame": i + 1})
                    cv2.imwrite(file_path, frame)

        # Release the video manager resources.
        cap.release()
        lprint(msg=f"[{method}] Finished content scenedetect.", user=user_name, level="info")

        return {"scenes": scene_files}


class IdentityPreservingGeneration(BaseModule):
    @staticmethod
    def image_composition(user_name: str, input_images: list, prompt: str, progress_callback=None, seed: int = 1,
                          scale: float = 2.5, width: int = 1024, height: int = 1024, model_cache=None):
        # method
        method = "Image composition"
        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # params
        scale = float(scale) if scale else 2.5
        seed = IdentityPreservingGeneration.set_seed(seed)

        lprint(msg=f"[{method}] Processing.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb < Settings.MIN_VRAM_IMAGE_COMPOSITION_GB:
            raise RuntimeError(f"Can not run {method} because of set MIN_VRAM_IMAGE_COMPOSITION_GB {Settings.MIN_VRAM_IMAGE_COMPOSITION_GB} Gb but now is {available_vram_gb} Gb memory")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # load sd default repo
        omnigen_repo = get_nested_url(repo_config(), ["omnigen"])
        omnigen_path = os.path.join(ALL_MODEL_FOLDER, "omnigen")
        download_huggingface_repo(repo_id=omnigen_repo, download_dir=omnigen_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_OMNIGEN, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        if progress_callback:
            progress_callback(0, "Image composition generate...")

        offload_model = False if available_vram_gb > 20 else True

        output = imgs2img(omnigen_path=omnigen_path, input_images=input_images, prompt=prompt, width=width,
                          height=height, offload_model=offload_model, guidance_scale=scale, seed=seed,
                          device=device, model_cache=model_cache, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Image composition generate finished...")

        lprint(msg=f"[{method}] Image composition generate finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = IdentityPreservingGeneration.get_media_url(output_path)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_OMNIGEN:
            lprint(msg="Update checker of download size controlnet canny models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_OMNIGEN": False})
            Settings.CHECK_DOWNLOAD_SIZE_OMNIGEN = False

        return {"filename": output_media}


class ImageGeneration(BaseModule):
    @staticmethod
    def inpaint(user_name: str, source_file: str, inpaint_file: str, sd_model_name: str = None, progress_callback=None,
                params: dict = None, control_type: str = None, max_size: int = 1280, eta: float = 1, model_cache=None):
        # method
        method = "Inpaint"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # update parameters
        eta = float(eta) if eta is not None else 1  # If 0 than with one params will do similar results else add some noise
        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        strength = float(params.get("strength")) if params.get("strength") else 0.75
        control_strength = float(params.get("controlStrength")) if params.get("controlStrength") else 0.75
        blur_factor = int(params.get("blurFactor")) if params.get("blurFactor") else 10
        seed = params.get("seed")
        seed = ImageGeneration.set_seed(seed)
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "colorful"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

        # valid params
        control_type = control_type if control_type is not None else "canny"

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # load diffusion model
        diffuser_folder = os.path.join(ALL_MODEL_FOLDER, "diffusion")
        os.environ['TORCH_HOME'] = diffuser_folder
        if not os.path.exists(diffuser_folder):
            os.makedirs(diffuser_folder)

        if sd_model_name is None:
            lprint(msg=f"[{method}] Stable diffusion model is not selected.", user=user_name, level="error")
            return

        for c in file_diffusion_config():
            if c.get("model") == sd_model_name:
                sd_model_path = os.path.join(diffuser_folder, str(c["model"]))
                link_sd_model = c.get("url")
                if not os.path.exists(sd_model_path):
                    # check what is internet access
                    if is_connected(sd_model_path) and link_sd_model is not None:
                        # download pre-trained models from url
                        download_model(sd_model_path, link_sd_model, progress_callback=progress_callback)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name, level="error")
                        else:
                            lprint(msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.", user=user_name, level="error")
                        raise ConnectionError("Not internet connection or not found link")
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model, progress_callback=progress_callback)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise ValueError("Not found selected model")

        # Updating progress
        if progress_callback:
            progress_callback(25, "Inspection models...")

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        exclude_files = ["v1-5-pruned.ckpt", "v1-5-pruned-emaonly.ckpt", "v1-5-pruned-emaonly.safetensors"]
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER, progress_callback=progress_callback, exclude_files=exclude_files)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        # load controlnet model
        controlnet_repo = get_nested_url(repo_config(), ["controlnet_canny"])
        controlnet_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_canny")
        download_huggingface_repo(repo_id=controlnet_repo, download_dir=controlnet_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(75, "Processing...")

        lprint(msg=f"[{method}] Processing.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb < Settings.MIN_VRAM_DIFFUSER_GB:
            device = "cpu"
            lprint(msg=f"[{method}] Memory out of necessary for {method} during load diffusion models on GPU. Will be using CPU.", level="warning", user=user_name)

        # Resize image
        original_width, original_height, image = resize_pil_image64(Image.open(source_file), max_size)

        # Mask image
        mask_image = Image.open(inpaint_file)
        if mask_image.size[0] != image.size[0] or mask_image.size[1] != image.size[1]:
            lprint(msg=f"[{method}] Resize mask image with restore ratio.", user=user_name, level="info")
            mask_image = mask_image.resize((image.size[0], image.size[1]), Image.Resampling.LANCZOS)
        # Convert the image to RGB if it's not already
        mask_image = mask_image.convert("RGB")
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_factor))
        # Create a new list for the binary pixels using a list comprehension
        color_threshold = 10
        binary_pixels = [255 if max(pixel) > color_threshold else 0 for pixel in mask_image.getdata()]
        # Create a new binary image
        binary_image = Image.new('L', mask_image.size)
        binary_image.putdata(binary_pixels)
        # Convert to '1' mode to strictly get black and white
        binary_image = binary_image.convert('1')

        output = inpaint2img(
            weights_path=sd_model_path, sd_path=sd_path, controlnet_path=controlnet_path, device=device,
            scale=scale, strength=strength, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt,
            init_image=image, mask_image=binary_image, control_strength=control_strength, eta=eta,
            blur_factor=blur_factor, model_cache=model_cache
        )

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inpaint finished...")

        lprint(msg=f"[{method}] Inpaint finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        # Save in tmp folder to not return bytes
        if output.size[0] != original_width or output.size[1] != original_height:
            lprint(msg=f"[{method}] Resize image with restore ratio.", user=user_name, level="info")
            output = output.resize((original_width, original_height), Image.Resampling.LANCZOS)
        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = ImageGeneration.get_media_url(output_path)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY and control_type == "canny":
            lprint(msg="Update checker of download size controlnet canny models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY": False})
            Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY = False

        return {"filename": output_media}

    @staticmethod
    def outpaint(user_name: str, source_file: str,  sd_model_name: str = None, params: dict = None, blur_factor: int=10,
                 width=1024, height=576, model_cache=None, progress_callback=None):
        # method
        method = "Outpaint"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # update parameters
        blur_factor = int(blur_factor) if str(blur_factor).isdigit() else 10
        width = int(width) if str(width).isdigit() else 1024
        height = int(height) if str(height).isdigit() else 576

        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        seed = params.get("seed")
        seed = ImageGeneration.set_seed(seed)
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "the best quality"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # load diffusion model
        diffuser_folder = os.path.join(ALL_MODEL_FOLDER, "diffusion")
        os.environ['TORCH_HOME'] = diffuser_folder
        if not os.path.exists(diffuser_folder):
            os.makedirs(diffuser_folder)

        if sd_model_name is None:
            lprint(msg=f"[{method}] Stable diffusion model is not selected.", user=user_name, level="error")
            return

        for c in file_diffusion_config():
            if c.get("model") == sd_model_name:
                sd_model_path = os.path.join(diffuser_folder, str(c["model"]))
                link_sd_model = c.get("url")
                if not os.path.exists(sd_model_path):
                    # check what is internet access
                    if is_connected(sd_model_path) and link_sd_model is not None:
                        # download pre-trained models from url
                        download_model(sd_model_path, link_sd_model, progress_callback=progress_callback)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name, level="error")
                        else:
                            lprint(msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.", user=user_name, level="error")
                        raise ConnectionError("Not internet connection or not found link")
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model, progress_callback=progress_callback)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise ValueError("Not found selected model")

        # Updating progress
        if progress_callback:
            progress_callback(20, "Inspection models...")

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        exclude_files = ["v1-5-pruned.ckpt", "v1-5-pruned-emaonly.ckpt", "v1-5-pruned-emaonly.safetensors"]
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER, progress_callback=progress_callback, exclude_files=exclude_files)

        # Updating progress
        if progress_callback:
            progress_callback(30, "Inspection models...")

        # load controlnet model
        controlnet_tile_repo = get_nested_url(repo_config(), ["controlnet_tile"])
        controlnet_tile_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_tile")
        commit = "7e2607811c97dc77a6c98a4256bb7a44f5abc8b3"
        download_huggingface_repo(repo_id=controlnet_tile_repo, download_dir=controlnet_tile_path,compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE, progress_callback=progress_callback, commit=commit)

        # Updating progress
        if progress_callback:
            progress_callback(40, "Inspection models...")

        controlnet_canny_repo = get_nested_url(repo_config(), ["controlnet_canny"])
        controlnet_canny_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_canny")
        download_huggingface_repo(repo_id=controlnet_canny_repo, download_dir=controlnet_canny_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(60, "Outpaint...")

        lprint(msg=f"[{method}] Processing.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb < Settings.MIN_VRAM_DIFFUSER_GB:
            device = "cpu"
            lprint(msg=f"[{method}] Memory out of necessary for {method} during load diffusion models on GPU. Will be using CPU.", level="warning", user=user_name)

        # Outpaint image
        outpaint_image, mask_rgb = Outpaint().decode_image(img_path=source_file, padding=blur_factor, max_height=height, max_width=width)
        # Convert OpenCV BGR format to RGB
        outpaint_image = cv2.cvtColor(outpaint_image, cv2.COLOR_BGR2RGB)
        # Convert NumPy array (RGB) to Pillow Image
        outpaint_image = Image.fromarray(outpaint_image)
        outpaint_image = outpaint_image.resize((width, height), Image.Resampling.LANCZOS)
        # Convert NumPy array (RGB) to Pillow Image
        mask_image = Image.fromarray(mask_rgb)
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_factor))
        mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

        # Create a new list for the binary pixels using a list comprehension
        color_threshold = 10
        if mask_image.mode == 'L':  # Grayscale image
            binary_pixels = [255 if pixel > color_threshold else 0 for pixel in mask_image.getdata()]
        else:  # Multi-channel image (RGB, RGBA, etc.)
            binary_pixels = [255 if max(pixel) > color_threshold else 0 for pixel in mask_image.getdata()]
        # Create a new binary image
        binary_image = Image.new('L', mask_image.size)
        binary_image.putdata(binary_pixels)
        # Convert to '1' mode to strictly get black and white
        binary_image = binary_image.convert('1')

        # Inpaint
        inpaint_image = inpaint2img(
            weights_path=sd_model_path, sd_path=sd_path, controlnet_path=controlnet_canny_path, device=device,
            scale=scale, strength=0.75, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt, blur_factor=blur_factor,
            init_image=outpaint_image, mask_image=binary_image, control_strength=0.8, guess_mode=True,
            control_guidance_end=0.8, model_cache=model_cache
        )

        # Updating progress
        if progress_callback:
            progress_callback(80, "Improve image after outpain...")

        # Improve image after outpaint
        output = inpaint2img(
            weights_path=sd_model_path, sd_path=sd_path, controlnet_path=controlnet_tile_path, device=device,
            scale=scale, strength=0.3, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt, blur_factor=blur_factor,
            init_image=inpaint_image, mask_image=binary_image, control_strength=0.9, guess_mode=True, model_cache=model_cache
        )

        # Updating progress
        if progress_callback:
            progress_callback(100, "Outpaint finished.")

        lprint(msg=f"[{method}] Outpaint finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        # Save in tmp folder to not return bytes
        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = ImageGeneration.get_media_url(output_path)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY:
            lprint(msg="Update checker of download size controlnet canny models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY": False})
            Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY = False

        if Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE:
            lprint(msg="Update checker of download size controlnet canny models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE": False})
            Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE = False

        return {"filename": output_media, "path": output_path}

    @staticmethod
    def text_to_image(user_name: str, sd_model_name: str = None, params: dict = None, width: int = 512, height: int = 512, progress_callback=None):
        # method
        method = "Text to Image"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # update parameters
        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        strength = float(params.get("strength")) if params.get("strength") else 0.75
        seed = params.get("seed")
        seed = ImageGeneration.set_seed(seed)
        width = int(width) if width else 512
        height = int(height) if height else 512
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "colorful"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # load diffusion model
        diffuser_folder = os.path.join(ALL_MODEL_FOLDER, "diffusion")
        os.environ['TORCH_HOME'] = diffuser_folder
        if not os.path.exists(diffuser_folder):
            os.makedirs(diffuser_folder)

        if sd_model_name is None:
            lprint(msg=f"[{method}] Stable diffusion model is not selected.", user=user_name, level="error")
            return

        for c in file_diffusion_config():
            if c.get("model") == sd_model_name:
                sd_model_path = os.path.join(diffuser_folder, str(c["model"]))
                link_sd_model = c.get("url")
                if not os.path.exists(sd_model_path):
                    # check what is internet access
                    if is_connected(sd_model_path) and link_sd_model is not None:
                        # download pre-trained models from url
                        download_model(sd_model_path, link_sd_model, progress_callback=progress_callback)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name, level="error")
                        else:
                            lprint(msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.", user=user_name, level="error")
                        raise ConnectionError("Not internet connection or not found link")
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model, progress_callback=progress_callback)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise ValueError("Not found selected model")

        # Updating progress
        if progress_callback:
            progress_callback(25, "Inspection models...")

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        exclude_files = ["v1-5-pruned.ckpt", "v1-5-pruned-emaonly.ckpt", "v1-5-pruned-emaonly.safetensors"]
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER, progress_callback=progress_callback, exclude_files=exclude_files)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        lprint(msg=f"[{method}] Generate image by stable diffusion.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb < Settings.MIN_VRAM_DIFFUSER_GB:
            device = "cpu"
            lprint(msg=f"[{method}] Memory out of necessary for {method} during load diffusion models on GPU. Will be using CPU.", level="warning", user=user_name)

        # Resize image
        max_size = 1080

        if width > max_size or height > max_size:
            if width > height:
                width = max_size
                height = int((max_size / width) * height)
            else:
                height = max_size
                width = int((max_size / height) * width)

        # Updating progress
        if progress_callback:
            progress_callback(75, "Image generation...")

        output = txt2img(
            weights_path=sd_model_path, sd_path=sd_path, device=device, width=width, height=height,
            scale=scale, strength=strength, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt
        )

        # Updating progress
        if progress_callback:
            progress_callback(100, "Image generation finished.")

        lprint(msg=f"[{method}] Image generation finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = ImageGeneration.get_media_url(output_path)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        return {"filename": output_media, "path": output_path}


class Highlights(BaseModule):
    @staticmethod
    def assistant(source_file: str, user_name: str, prompt: str, source_start: float = None, source_end: float = None,
                  mode: str = None, is_transcribe: bool = False, is_transcribe_words: bool = False, model_cache=None, progress_callback=None):
        # method
        method = "Assistant"

        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        os.makedirs(local_save_dir, exist_ok=True)

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb > 5.5 and FLASH_ATTN_2_AVAILABLE:
            smolvlm2_name = "SmolVLM2-2.2B"
        elif available_vram_gb > 12 and not FLASH_ATTN_2_AVAILABLE:
            smolvlm2_name = "SmolVLM2-2.2B"
        else:
            smolvlm2_name = "SmolVLM2-500M"

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # load sd default repo
        smolvlm2_repo = get_nested_url(repo_config(), [smolvlm2_name])
        smolvlm2_path = os.path.join(ALL_MODEL_FOLDER, smolvlm2_name)
        download_huggingface_repo(repo_id=smolvlm2_repo, download_dir=smolvlm2_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS, progress_callback=progress_callback)

        # params
        duration = VideoManipulation.get_video_duration_seconds(source_file)
        source_start = float(source_start) if source_start is not None and Highlights.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and Highlights.is_float(source_end) else 0

        if source_start != 0 or source_end != duration and source_end != 0:
            lprint(msg=f"[{method}] Cut video segment.", user=user_name, level="info")
            if Highlights.is_valid_video(source_file):
                source_file = VideoManipulation.cut_video_ffmpeg(source_file, source_start, source_end, save_dir=local_save_dir)
            elif Highlights.is_valid_audio(source_file):
                source_file = cut_audio(source_file, source_start, source_end, local_save_dir)

        if mode == "sound" or is_transcribe or is_transcribe_words:
            transcription_text, _ = Highlights.whisper_transcribe(source_file, is_words=is_transcribe_words, device=device, local_save_dir=local_save_dir, progress_callback=progress_callback)
            if not transcription_text:
                return {"text": "Not found speech"}
            if is_transcribe or is_transcribe_words:
                return {"text": transcription_text}
        else:
            transcription_text = None

        # Updating progress
        if progress_callback:
            progress_callback(50, "Generate...")

        smolvlm2 = model_cache.get_model(f"smolvlm2") if model_cache is not None else None
        if smolvlm2 is None:
            smolvlm2 = SmolVLM2(model_path=smolvlm2_path, device=device)

        if mode == "sound":
            text = smolvlm2.process_text_analyser(transcription_text, prompt=prompt)
        else:
            text = smolvlm2.process_video_analyser(source_file, prompt=prompt)

        lprint(msg=f"Synthesis finished.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS:
            lprint(msg="Update checker of download size smol vlm models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_HIGHLIGHTS": False})
            Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS = False

        # remove tmp files
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        return {"text": text}

    @staticmethod
    def whisper_transcribe(source_file: str, is_words: bool = False, segment_length: int = 10, device: str = "cpu", local_save_dir=None, progress_callback=None):
        if local_save_dir is None:
            local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
            os.makedirs(local_save_dir, exist_ok=True)

        _, available_vram_gb, _ = get_vram_gb(device=device)
        if available_vram_gb < 3:
            device = "cpu"

        # load whisper default repo
        whisper_name = "whisper"
        whisper_repo = get_nested_url(repo_config(), [whisper_name])
        whisper_path = os.path.join(ALL_MODEL_FOLDER, whisper_name)
        download_huggingface_repo(repo_id=whisper_repo, download_dir=whisper_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(20, "Audio analysis...")

        transcriber = Whisper(download_root=ALL_MODEL_FOLDER, device=device)

        if Highlights.is_valid_video(source_file):
            source_file = transcriber.get_audio(source_file, local_save_dir)
            if not os.path.exists(source_file):
                return None, None

        transcription = transcriber(audio_path=source_file, language=None, task="transcribe")
        if is_words:
            transcription_text = transcriber.get(transcription, segments_length=segment_length, mode="words")
            transcription_timestamp = None
        else:
            transcription_text = transcriber.get(transcription, segments_length=segment_length, mode=None)
            transcription_timestamp = transcriber.get(transcription, segments_length=segment_length, mode="timestamp")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Audio analysis...")

        del transcriber
        torch.cuda.empty_cache()
        return transcription_text, transcription_timestamp


    @staticmethod
    def synthesis(folder_name: str, user_name: str, user_save_name: str, source_file: str, mode: str = None, segment_length: int = 10,
                  duration_limit: int = None, highlight_description: str = None, model_cache=None, progress_callback=None):
        # method
        method = "Highlights"

        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb > 5.5 and FLASH_ATTN_2_AVAILABLE:
            smolvlm2_name = "SmolVLM2-2.2B"
        elif available_vram_gb > 12 and not FLASH_ATTN_2_AVAILABLE:
            smolvlm2_name = "SmolVLM2-2.2B"
        else:
            smolvlm2_name = "SmolVLM2-500M"

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # load smolvlm2_ default repo
        smolvlm2_repo = get_nested_url(repo_config(), [smolvlm2_name])
        smolvlm2_path = os.path.join(ALL_MODEL_FOLDER, smolvlm2_name)
        download_huggingface_repo(repo_id=smolvlm2_repo, download_dir=smolvlm2_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        if mode == "sound":
            transcription_text, transcription_timestamp = Highlights.whisper_transcribe(source_file, is_words=False, segment_length=segment_length, device=device, local_save_dir=local_save_dir, progress_callback=progress_callback)
        else:
            transcription_text, transcription_timestamp = None, None

        smolvlm2 = model_cache.get_model(f"smolvlm2") if model_cache is not None else None
        if smolvlm2 is None:
            smolvlm2 = SmolVLM2(model_path=smolvlm2_path, device=device, download_root=ALL_MODEL_FOLDER)

        if mode == "sound":
            kept_segments = smolvlm2.generate_text_highlight_moments(highlight_description=highlight_description, video_path=source_file, text=transcription_text, timestamp=transcription_timestamp, segment_length=segment_length, duration_limit_min=duration_limit, progress_callback=progress_callback)
        else:
            kept_segments = smolvlm2.generate_visual_highlight_moments(save_dir=local_save_dir, highlight_description=highlight_description, video_path=source_file, segment_length=segment_length, duration_limit_min=duration_limit, progress_callback=progress_callback)

        # create name for file
        user_save_name = str(user_save_name) + '.mp4' if user_save_name is not None else str(uuid.uuid4()) + '.mp4'
        output_file = os.path.join(user_save_dir, user_save_name)

        if not kept_segments:  # without highlights
            shutil.move(source_file, output_file)

            # Updating progress
            if progress_callback:
                progress_callback(100, "Finished without highlights.")

            output_media = Highlights.get_media_url(output_file)
            return {"file": output_file, "media": output_media}

        smolvlm2.concatenate_scenes(source_file, kept_segments, output_file)
        lprint(msg=f"Video saved.", user=user_name, level="info")
        del smolvlm2

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS:
            lprint(msg="Update checker of download size smol vlm models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_HIGHLIGHTS": False})
            Settings.CHECK_DOWNLOAD_SIZE_HIGHLIGHTS = False

        # remove tmp files
        if not sys.platform == 'win32' and os.path.exists(local_save_dir):
           shutil.rmtree(local_save_dir)

        lprint(msg=f"Synthesis finished.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        output_media = Highlights.get_media_url(output_file)
        return {"file": output_file, "media": output_media}

    @staticmethod
    def smart_cut(folder_name: str, user_name: str, user_save_name: str, source_file: str,
                  kept_segments: list, progress_callback=None):
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(user_save_dir, exist_ok=True)

        if progress_callback:
            progress_callback(10, "Start cut...")

        kept_segments = [(float(s.get("start", 0)), float(s.get("end", 0))) for s in kept_segments]
        if Highlights.is_valid_video(source_file):
            # create name for file
            user_save_name = str(user_save_name) + '.mp4' if user_save_name is not None else str(uuid.uuid4()) + '.mp4'
            output_file = os.path.join(user_save_dir, user_save_name)

            SmolVLM2.concatenate_scenes(source_file, kept_segments, output_file)
        elif Highlights.is_valid_audio(source_file):
            # create name for file
            user_save_name = str(user_save_name) + '.wav' if user_save_name is not None else str(uuid.uuid4()) + '.wav'
            output_file = os.path.join(user_save_dir, user_save_name)

            SmolVLM2.concatenate_audio(source_file, kept_segments, output_file)
        else:
            lprint(msg=f"Is not support file format.", user=user_name, level="error")
            raise Exception("Is not support file format.")

        lprint(msg=f"Synthesis finished.", user=user_name, level="info")

        if progress_callback:
            progress_callback(100, "Finished.")

        output_media = Highlights.get_media_url(output_file)
        return {"file": output_file, "media": output_media}


class NotEnoughMemoryError(Exception):
    pass


def memory_message(method: str, msg: str, user: str, level: str, device: str="cuda:0"):
    _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)