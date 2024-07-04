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

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modules_root_path = os.path.join(root_path, "generation")
sys.path.insert(0, modules_root_path)

from pipeline import txt2img, img2vid, Outpaint, inpaint2img
from generation.utils.mediaio import resize_pil_image64

sys.path.pop(0)

from backend.folders import TMP_FOLDER, ALL_MODEL_FOLDER, MEDIA_FOLDER
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity
from backend.download import download_model, unzip, check_download_size, get_nested_url, is_connected, download_huggingface_repo
from backend.config import ModelsConfig, Settings, HuggingfaceRepoConfig
from deepfake.src.utils.videoio import VideoManipulation


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


class VideoGeneration:
    @staticmethod
    def is_valid_image(file):
        return VideoManipulation.check_media_type(file) == "static"

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
                if VideoGeneration.is_img_url(c["img"]):  # if url
                    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
                        c["img"] = None
                else:  # if media folder file
                    sd_img_path = os.path.join(diffuser_folder, str(c["img"]))
                    if not os.path.exists(sd_img_path):
                        c["img"] = None
                    else:
                        # cover path
                        c["img"] = sd_img_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
            if c.get("name") is None:
                c["name"] = "Unknown"

            config.append(c)

        return config

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def inspect() -> dict:
        return {}

    @staticmethod
    def synthesis(folder_name: str, user_name: str, user_save_name: str, source_file: str = None, positive_prompt: str = None,
                  negative_prompt: str = None, seed: int = None, scale: float = None, strength: float = None, fps: int = 12,
                  sd_model_name: str = None, aspect_ratio: float = None):
        # method
        method = "Video Generation"

        scale = float(scale) if str(scale).isdigit() else 7.5
        strength = float(strength) if str(strength).isdigit() else 0.75
        seed = int(seed) if seed and str(seed).isdigit() else random.randint(0, 1000000)
        positive_prompt = positive_prompt if positive_prompt and len(positive_prompt) > 0 else "colorful"
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"
        fps = int(fps) if str(fps).isdigit() else 12

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE','cpu') else True

        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing can not run on CPU.", user=user_name, level="error")
            return

        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

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
                        download_model(sd_model_path, link_sd_model)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name,
                                   level="error")
                        else:
                            lprint(
                                msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.",
                                user=user_name, level="error")
                        raise
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise

        # load sd default repo
        sd_video_repo = get_nested_url(repo_config(), ["stable_video_diffusion"])
        sd_video_path = os.path.join(ALL_MODEL_FOLDER, "stabilityai")
        download_huggingface_repo(repo_id=sd_video_repo, download_dir=sd_video_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER)

        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb()
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        if available_vram_gb < Settings.MIN_VRAM_VIDEO_DIFFUSER_GB:
            lprint(msg=f"[{method}] Memory out of necessary for {method} during load diffusion models on GPU.", level="warning", user=user_name)
            return

        if source_file is None or not os.path.exists(source_file):
            if aspect_ratio > 1:
                width = int(1024)
                height = int(1024 / aspect_ratio)
            else:
                width = int(576 * aspect_ratio)
                height = int(576)

            # Generate image before if empty
            lprint(msg=f"[{method}] Generate image before processing.", user=user_name, level="info")
            params = {"positivePrompt": positive_prompt, "negativePrompt": negative_prompt, "seed": seed, "scale": scale, "strength": strength}
            response = ImageGeneration.text_to_image(user_name=user_name, sd_model_name=sd_model_name, params=params, width=width, height=height)
            source_file = response.get("path")

        with Image.open(source_file) as source_image:
            width, height = source_image.size

        aspect_ratio = width / height
        if width != 1024 or height != 576:
            lprint(msg=f"[{method}] Outpaint image before processing.", user=user_name, level="info")
            params = {"positivePrompt": positive_prompt, "negativePrompt": negative_prompt, "seed": seed, "scale": scale, "strength": strength}
            response = ImageGeneration.outpaint(user_name=user_name, source_file=source_file,  sd_model_name=sd_model_name, params=params, blur_factor=30, width=1024, height=576)
            output_path = response.get("path")
        else:
            output_path = source_file

        init_image = Image.open(output_path)
        init_image.convert("RGB")

        _, available_vram_gb, busy_vram_gb = get_vram_gb()

        if available_vram_gb > 20:
            num_frames = 24
            decode_chunk_size = 8
        elif available_vram_gb > 15:
            num_frames = 24
            decode_chunk_size = 4
        else:
            num_frames = 16
            decode_chunk_size = 4

        lprint(msg=f"[{method}] Generate video.", user=user_name, level="info")

        frames = img2vid(init_image=init_image, sd_path=sd_video_path, width=1024, height=576,
                         fps=fps, aspect_ratio=aspect_ratio, num_frames=num_frames, decode_chunk_size=decode_chunk_size)

        lprint(msg=f"Video saved.", user=user_name, level="info")

        for i, frame in enumerate(frames):
            frame.save(os.path.join(local_save_dir, f"{i}.jpg"))
        else:
            video_name = VideoManipulation.save_video_from_frames(frame_names="%d.jpg", save_path=local_save_dir, fps=fps)

        # create name for file
        user_save_name = str(user_save_name) + '.mp4' if user_save_name is not None else str(uuid.uuid4()) + '.mp4'
        output_file = os.path.join(user_save_dir, user_save_name)

        shutil.move(os.path.join(local_save_dir, video_name), output_file)

        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"Synthesis finished.", user=user_name, level="info")

        if Settings.CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER:
            lprint(msg="Update checker of download size video diffuser models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER": False})
            Settings.CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER = False

        return {"file": output_file}

class ImageGeneration:
    @staticmethod
    def is_valid_image(file):
        return VideoManipulation.check_media_type(file) == "static"

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
                if ImageGeneration.is_img_url(c["img"]):  # if url
                    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
                        c["img"] = None
                else:  # if media folder file
                    sd_img_path = os.path.join(diffuser_folder, str(c["img"]))
                    if not os.path.exists(sd_img_path):
                        c["img"] = None
                    else:
                        # cover path
                        c["img"] = sd_img_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
            if c.get("name") is None:
                c["name"] = "Unknown"

            config.append(c)

        return config

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def inpaint(user_name: str, source_file: str, inpaint_file: str, sd_model_name: str = None,
                params: dict = None, control_type: str = None, max_size: int = 1280, eta: float = 1):
        # method
        method = "Inpaint"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"

        # update parameters
        eta = float(eta) if eta is not None else 1  # If 0 than with one params will do similar results else add some noise
        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        strength = float(params.get("strength")) if params.get("strength") else 0.75
        control_strength = float(params.get("controlStrength")) if params.get("controlStrength") else 0.75
        blur_factor = int(params.get("blurFactor")) if params.get("blurFactor") else 10
        seed = params.get("seed")
        seed = int(seed) if seed and str(seed).isdigit() else random.randint(0, 1000000)
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "colorful"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

        # valid params
        control_type = control_type if control_type is not None else "canny"

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
                        download_model(sd_model_path, link_sd_model)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name,
                                   level="error")
                        else:
                            lprint(
                                msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.",
                                user=user_name, level="error")
                        raise
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER)

        # load controlnet model
        controlnet_repo = get_nested_url(repo_config(), ["controlnet_canny"])
        controlnet_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_canny")
        download_huggingface_repo(repo_id=controlnet_repo, download_dir=controlnet_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY)

        lprint(msg=f"[{method}] Processing.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb()
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
            init_image=image, mask_image=binary_image, control_strength=control_strength, eta=eta, blur_factor=blur_factor
        )

        lprint(msg=f"[{method}] Inpaint finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        # Save in tmp folder to not return bytes
        if output.size[0] != original_width or output.size[1] != original_height:
            lprint(msg=f"[{method}] Resize image with restore ratio.", user=user_name, level="info")
            output = output.resize((original_width, original_height), Image.Resampling.LANCZOS)
        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = output_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY and control_type == "canny":
            lprint(msg="Update checker of download size controlnet canny models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY": False})
            Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY = False

        return {"filename": output_media}

    @staticmethod
    def outpaint(user_name: str, source_file: str,  sd_model_name: str = None, params: dict = None, blur_factor: int=10,
                 width=1024, height=576):
        # method
        method = "Outpaint"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"

        # update parameters
        blur_factor = int(blur_factor) if str(blur_factor).isdigit() else 10
        width = int(width) if str(width).isdigit() else 1024
        height = int(height) if str(height).isdigit() else 576

        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        seed = params.get("seed")
        seed = int(seed) if seed and str(seed).isdigit() else random.randint(0, 1000000)
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "the best quality"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

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
                        download_model(sd_model_path, link_sd_model)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name,
                                   level="error")
                        else:
                            lprint(
                                msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.",
                                user=user_name, level="error")
                        raise
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER)

        # load controlnet model
        controlnet_tile_repo = get_nested_url(repo_config(), ["controlnet_tile"])
        controlnet_tile_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_tile")
        download_huggingface_repo(repo_id=controlnet_tile_repo, download_dir=controlnet_tile_path,compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE)

        controlnet_canny_repo = get_nested_url(repo_config(), ["controlnet_canny"])
        controlnet_canny_path = os.path.join(ALL_MODEL_FOLDER, "controlnet_canny")
        download_huggingface_repo(repo_id=controlnet_canny_repo, download_dir=controlnet_canny_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY)

        lprint(msg=f"[{method}] Processing.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb()
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
            init_image=outpaint_image, mask_image=binary_image, control_strength=0.8, guess_mode=True, control_guidance_end=0.8
        )

        # Improve image after outpaint
        output = inpaint2img(
            weights_path=sd_model_path, sd_path=sd_path, controlnet_path=controlnet_tile_path, device=device,
            scale=scale, strength=0.3, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt, blur_factor=blur_factor,
            init_image=inpaint_image, mask_image=binary_image, control_strength=0.9, guess_mode=True
        )

        lprint(msg=f"[{method}] Outpaint finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        # Save in tmp folder to not return bytes
        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = output_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

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
    def text_to_image(user_name: str, sd_model_name: str = None, params: dict = None, width: int = 512, height: int = 512):
        # method
        method = "Text to Image"

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="error")
            device = "cpu"

        # update parameters
        params = params if params else {}
        scale = float(params.get("scale")) if params.get("scale") else 7.5
        strength = float(params.get("strength")) if params.get("strength") else 0.75
        seed = params.get("seed")
        seed = int(seed) if seed and str(seed).isdigit() else random.randint(0, 1000000)
        width = int(width) if width else 512
        height = int(height) if height else 512
        positive_prompt = params.get("positivePrompt")
        positive_prompt = params.get("positivePrompt") if positive_prompt and len(positive_prompt) > 0 else "colorful"
        negative_prompt = params.get("negativePrompt")
        negative_prompt = negative_prompt if negative_prompt and len(negative_prompt) > 0 else "ugly, bad, terrible"

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
                        download_model(sd_model_path, link_sd_model)
                    else:
                        if link_sd_model is None:
                            lprint(msg=f"[{method}] Not found url for {sd_model_path} to download.", user=user_name,
                                   level="error")
                        else:
                            lprint(
                                msg=f"[{method}] Not internet connection to download model {link_sd_model} in {sd_model_path}.",
                                user=user_name, level="error")
                        raise
                else:
                    if link_sd_model is not None:
                        if c.get("update"):  # key update
                            check_download_size(sd_model_path, link_sd_model)
                break
        else:
            lprint(msg=f"[{method}] Not found selected model {sd_model_name}.", user=user_name, level="error")
            raise

        # load sd default repo
        sd_repo = get_nested_url(repo_config(), ["stable_diffusion"])
        sd_path = os.path.join(ALL_MODEL_FOLDER, "runwayml")
        download_huggingface_repo(repo_id=sd_repo, download_dir=sd_path, compare_size=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER)

        lprint(msg=f"[{method}] Generate image by stable diffusion.", user=user_name, level="info")
        # diffusion frames
        # delay for load model if a few processes
        _, available_vram_gb, busy_vram_gb = get_vram_gb()
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

        output = txt2img(
            weights_path=sd_model_path, sd_path=sd_path, device=device, width=width, height=height,
            scale=scale, strength=strength, seed=seed, prompt=positive_prompt, n_prompt=negative_prompt
        )

        lprint(msg=f"[{method}] Image generation finished.", user=user_name, level="info")

        if isinstance(output, str):
            output = Image.open(output)

        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        output.save(output_path)
        output_media = output_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        return {"filename": output_media, "path": output_path}


def memory_message(method: str, msg: str, user: str, level: str):
    _, available_vram_gb, busy_vram_gb = get_vram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)