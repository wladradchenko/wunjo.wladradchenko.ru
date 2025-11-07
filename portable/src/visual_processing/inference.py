import os
import gc
import cv2
import sys
import uuid
import math
import torch
import shutil
import random
import numpy as np
from tqdm import tqdm
from time import strftime, sleep

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modules_root_path = os.path.join(root_path, "visual_processing")
sys.path.insert(0, modules_root_path)

"""Media Manipulation"""
from utils.videoio import VideoManipulation
from utils.imageio import save_image_cv2, read_image_cv2, read_image_pil
from utils.audio import cut_audio
"""Media Manipulation"""
"""Lip Sync"""
from lip_sync import MelProcessor, GenerateWave2Lip
"""Lip Sync"""
"""Enhancer"""
from enhancement.face_enhancer import content_enhancer
"""Enhancer"""
"""Face Swap"""
from face_swap import FaceSwapProcessing
"""Face Swap"""
"""Remover"""
from remover import InpaintModel, RemoveObjectUtils, RemoveObjectProcessor
"""Remover"""
"""Segmentation"""
from segmentation import SegmentAnything, SegmentText
"""Segmentation"""
"""Portrait Animation"""
from portrait_animation import LivePortrait
"""Portrait Animation"""

sys.path.pop(0)

from backend.folders import ALL_MODEL_FOLDER, TMP_FOLDER, MEDIA_FOLDER
from backend.download import download_model, check_download_size, get_nested_url, is_connected, download_huggingface_repo
from backend.config import ModelsConfig, Settings, HuggingfaceRepoConfig
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity, get_best_device


file_models_config = ModelsConfig()
repo_config = HuggingfaceRepoConfig()


class BaseModule:
    @staticmethod
    def get_media_url(path: str):
        return path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

    @staticmethod
    def handle_retries_synthesis(min_delay=20, max_delay=120, user_name=None, progress_callback=None):
        lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=str(user_name))
        delay = random.randint(min_delay, max_delay)
        # Updating progress
        if progress_callback:
            progress_callback(0, f"Not enough memory available. The process will resume after {delay} sec...")
        # Clear memory and wait for a while before retry
        gc.collect()
        torch.cuda.empty_cache()
        sleep(random.randint(min_delay, max_delay))
        # Updating progress
        if progress_callback:
            progress_callback(0, "Wait until all GPU operations are completed.")
        # Waiting while process free
        torch.cuda.synchronize()  # TODO test this will have problem or not?
        torch.cuda.empty_cache()
        # Updating progress
        if progress_callback:
            progress_callback(0, "Run processing again after clear cache.")

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def valid_value(val: float = 0, max_val: float = 0, min_val: float = 0) -> float:
        val = 0 if val is None else val
        return max(min_val, min(max_val, val))

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
    def is_valid_video_area(areas: list, offset_width: int, offset_height: int, natural_width: int, natural_height: int) -> list:
        """
        Set valid format
        Source area is list of dict and has x, y, width, height and start_time and end_time.
        Time need to use in face swap with a few faces boxes with difference time
        """
        area = []
        for i, a in enumerate(areas):
            if a is None:
                continue

            # Calculate the scale factor
            scale_factor_x = int(natural_width) / int(offset_width)
            scale_factor_y = int(natural_height) / int(offset_height)

            area.append({
                "x": int(float(a.get("x", 0)) * scale_factor_x),
                "y": int(float(a.get("y", 0)) * scale_factor_y),
                "width": int(float(a.get("width", 0)) * scale_factor_x),
                "height": int(float(a.get("height", 0)) * scale_factor_y),
                "naturalWidth": int(natural_width),
                "naturalHeight": int(natural_height),
                "startTime": a.get("startTime", None),
                "endTime": a.get("endTime", None)
            })
        return area

    @staticmethod
    def is_valid_image_area(areas: list, natural_width: int, natural_height: int) -> list:
        """
            Set valid format image (naturalWidth and naturalHeight used)
            Avatar area is list of dict and has x, y, width, height
            """
        area = []
        for i, a in enumerate(areas):
            if a is None:
                continue

            area.append({
                "x": int(a.get("x", 0)),
                "y": int(a.get("y", 0)),
                "width": int(a.get("width", 0)),
                "height": int(a.get("height", 0)),
                "naturalWidth": int(natural_width),
                "naturalHeight": int(natural_height),
            })
        return area

    @staticmethod
    def scale_coord(area, updated_width, updated_height):
        x1 = int(area.get("x", 0))
        y1 = int(area.get("y", 0))
        width = int(area.get("width", 0))
        height = int(area.get("height", 0))
        natural_height = int(area.get("naturalHeight", 0))
        natural_width = int(area.get("naturalWidth", 0))

        # Calculate the scaling factors
        scale_x = updated_width / natural_width
        scale_y = updated_height / natural_height

        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(new_x1 + width * scale_x)
        new_y2 = int(new_y1 + height * scale_y)

        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def get_resize_factor(width, height, max_size: int = 1280):
        """If need to set max size"""
        resize_factor = max_size / max(width, height) if max_size is not None else 1
        return resize_factor if resize_factor < 1 else 1

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
    def update_model_download_checker(method_name: str):
        """
        Update settings to enable model download size checking.
        """
        lprint(msg=f"Updating download size checker for method: {method_name}.")
        # Update config with method-specific download size check flag
        Settings.config_manager.update_config({method_name: True})
        # Set the dynamic attribute in Settings to True
        setattr(Settings, method_name, True)


class LivePortraitRetargeting(BaseModule):
    @staticmethod
    def inspect() -> dict:
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        portrait_dir = os.path.join(checkpoint_folder, "portrait")
        model_dir_full = os.path.join(ALL_MODEL_FOLDER, "models")
        model_data = {}

        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'

        model_info_configs = [
            ("Live portrait", "Face alignment", os.path.join(portrait_dir, "landmark.onnx"),["portrait", "landmark.onnx"]),
            ("Live portrait", "Face alignment", os.path.join(model_dir_full, "buffalo_l"),"%s/%s.zip" % (BASE_REPO_URL, 'buffalo_l')),
            ("Live portrait", "Retarget", os.path.join(portrait_dir, "stitching_retargeting_module.pth"),["portrait", "stitching_retargeting_module.pth"]),
            ("Live portrait", "Retarget", os.path.join(portrait_dir, "warping_module.pth"),["portrait", "warping_module.pth"]),
            ("Live portrait", "Retarget", os.path.join(portrait_dir, "spade_generator.pth"),["portrait", "spade_generator.pth"]),
            ("Live portrait", "Retarget", os.path.join(portrait_dir, "motion_extractor.pth"),["portrait", "motion_extractor.pth"]),
            ("Live portrait", "Retarget", os.path.join(portrait_dir, "appearance_feature_extractor.pth"),["portrait", "appearance_feature_extractor.pth"])
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = LivePortraitRetargeting.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def load_model(method, user_name, progress_callback=None):
        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        msg = f"Before loading model in memory"
        memory_message(method, msg, user_name, "info", device=device)

        # Load models
        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        portrait_dir = os.path.join(checkpoint_dir_full, "portrait")

        landmark = os.path.join(portrait_dir, 'landmark.onnx')
        link_landmark = get_nested_url(file_models_config(), ["portrait", "landmark.onnx"])
        LivePortraitRetargeting.validate_models(landmark, link_landmark, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        stitching_retargeting_module = os.path.join(portrait_dir, 'stitching_retargeting_module.pth')
        link_stitching_retargeting_module = get_nested_url(file_models_config(), ["portrait", "stitching_retargeting_module.pth"])
        LivePortraitRetargeting.validate_models(stitching_retargeting_module, link_stitching_retargeting_module, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        warping_module = os.path.join(portrait_dir, 'warping_module.pth')
        link_warping_module = get_nested_url(file_models_config(), ["portrait", "warping_module.pth"])
        LivePortraitRetargeting.validate_models(warping_module, link_warping_module, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        spade_generator = os.path.join(portrait_dir, 'spade_generator.pth')
        link_spade_generator = get_nested_url(file_models_config(), ["portrait", "spade_generator.pth"])
        LivePortraitRetargeting.validate_models(spade_generator, link_spade_generator, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        motion_extractor = os.path.join(portrait_dir, 'motion_extractor.pth')
        link_motion_extractor = get_nested_url(file_models_config(), ["portrait", "motion_extractor.pth"])
        LivePortraitRetargeting.validate_models(motion_extractor, link_motion_extractor, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        appearance_feature_extractor = os.path.join(portrait_dir, 'appearance_feature_extractor.pth')
        link_appearance_feature_extractor = get_nested_url(file_models_config(), ["portrait", "appearance_feature_extractor.pth"])
        LivePortraitRetargeting.validate_models(appearance_feature_extractor, link_appearance_feature_extractor, user_name, "CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT", progress_callback=progress_callback)

        insightface_root = os.path.join(ALL_MODEL_FOLDER, "models")

        live_portrait = LivePortrait(
            source_max_dim=Settings.MAX_RESOLUTION_LIVE_PORTRAIT,
            checkpoint_F=appearance_feature_extractor, checkpoint_M=motion_extractor, checkpoint_G=spade_generator,
            checkpoint_W=warping_module, checkpoint_S=stitching_retargeting_module, insightface_root=insightface_root,
            landmark_ckpt_path=landmark, device=device
        )

        return live_portrait

    @staticmethod
    def move(input_file, folder_name, output_name):
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(user_save_dir, exist_ok=True)
        output_file = os.path.join(user_save_dir, output_name)
        shutil.move(input_file, os.path.join(user_save_dir, output_file))
        return output_file

    @staticmethod
    def retargeting_image(source: str, user_name: str, eye_ratio: float = 0, lip_ratio: float = 0, pitch: float = 0,
                          yaw: float = 0, roll: float = 0, x: float = 0, y: float = 0, z: float = 1,
                          eyeball_direction_x: float = 0, eyeball_direction_y: float = 0, progress_callback=None,
                          smile: float = 0, wink: float = 0, eyebrow: float = 0, grin: float = 0,
                          pursing: float = 0, pouting: float = 0, lip_expression: float = 0,
                          crops: list = None, model_cache=None):
        # method
        method = "Portrait retargeting"

        # valid values
        eye_ratio = float(eye_ratio)
        lip_ratio = float(lip_ratio)
        pitch = float(pitch)
        yaw = float(yaw)
        roll = float(roll)
        x = LivePortraitRetargeting.valid_value(x, max_val=0.19, min_val=-0.19)
        y = LivePortraitRetargeting.valid_value(y, max_val=0.19, min_val=-0.19)
        z = LivePortraitRetargeting.valid_value(z, max_val=1.2, min_val=0.9)
        eyeball_direction_x = LivePortraitRetargeting.valid_value(eyeball_direction_x, max_val=30.0, min_val=-30.0)
        eyeball_direction_y = LivePortraitRetargeting.valid_value(eyeball_direction_y, max_val=60.0, min_val=-60.0)
        smile = LivePortraitRetargeting.valid_value(smile, max_val=1.3, min_val=-0.3)
        wink = LivePortraitRetargeting.valid_value(wink, max_val=39.0, min_val=0.0)
        eyebrow = LivePortraitRetargeting.valid_value(eyebrow, max_val=30.0, min_val=-30.0)
        grin = LivePortraitRetargeting.valid_value(grin, max_val=15.0, min_val=0.0)
        pursing = LivePortraitRetargeting.valid_value(pursing, max_val=15.0, min_val=-20.0)
        pouting = LivePortraitRetargeting.valid_value(pouting, max_val=0.09, min_val=-0.09)
        lip_expression = LivePortraitRetargeting.valid_value(lip_expression, max_val=90, min_val=-90)
        lprint(msg=f"[{method}] Got retarget parameters.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        live_portrait = model_cache.get_model("live_portrait") if model_cache is not None else None
        if live_portrait is None:
            live_portrait = LivePortraitRetargeting.load_model(method=method, user_name=user_name, progress_callback=progress_callback)
            # Update cache
            model_cache.update_model("live_portrait", live_portrait) if model_cache else None
        live_portrait_pipeline = live_portrait.live_portrait_pipeline

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Retarget image...")

        # Retargeting
        output_path = os.path.join(TMP_FOLDER, f"{str(uuid.uuid4())}.jpg")
        # Clear windows for cache
        live_portrait.reset_face_analysis(live_portrait_pipeline)
        # Crops
        for i, crop in enumerate(crops):
            crop = crop if isinstance(crop, dict) else {}
            # Extract parameters
            eye_ratio_real, lip_ratio_real, pitch_real, yaw_real, roll_real = live_portrait.extract_portrait_parameters(img_src=source, pipeline=live_portrait_pipeline, crop=crop.copy())
            eye_ratio_updated = LivePortraitRetargeting.valid_value(eye_ratio + eye_ratio_real, max_val=0.8, min_val=0.0)
            lip_ratio_updated = LivePortraitRetargeting.valid_value(lip_ratio + lip_ratio_real, max_val=0.8, min_val=0.0)
            pitch_updated = LivePortraitRetargeting.valid_value(pitch + pitch_real, max_val=90, min_val=-90)
            yaw_updated = LivePortraitRetargeting.valid_value(yaw + yaw_real, max_val=90, min_val=-90)
            roll_updated = LivePortraitRetargeting.valid_value(roll + roll_real, max_val=90, min_val=-90)
            lprint(msg=f"[{method}] Update image portrait parameters {i + 1}.", user=user_name, level="info")
            output = live_portrait.update_image_portrait_parameters(
                img_src=source, pipeline=live_portrait_pipeline, eye_ratio=eye_ratio_updated, lip_ratio=lip_ratio_updated,
                pitch=pitch_updated, yaw=yaw_updated, roll=roll_updated, x=x, y=y, z=z,
                eyeball_direction_x=eyeball_direction_x, eyeball_direction_y=eyeball_direction_y, smile=smile, wink=wink,
                eyebrow=eyebrow, grin=grin, pursing=pursing, pouting=pouting, lip_expression=lip_expression, crop=crop.copy()
            )
            cv2.imwrite(output_path, output[..., ::-1])
            source = output_path  # if a few crops faces for one image

            # Updating progress
            if progress_callback:
                progress_callback(round((i + 1) / len(crops), 0), "Retarget image...")

        # Updating progress
        if progress_callback:
            progress_callback(80, "Encrypted...")

        lprint(msg=f"[{method}] Encrypted.", user=user_name, level="info")
        try:
            file_name = VideoManipulation.encrypted(output_path, TMP_FOLDER)
            output_path = os.path.join(TMP_FOLDER, file_name)
        except Exception as err:
            print(f"Error with VideoManipulation.encrypted {err}")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Encrypted...")

        output_media = LivePortraitRetargeting.get_media_url(output_path)

        if Settings.CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT:
            lprint(msg="Update checker of download size live portrait models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT": False})
            Settings.CHECK_DOWNLOAD_SIZE_LIVE_PORTRAIT = False

        lprint(msg=f"[{method}] Finished.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        return {"filename": output_media, "file": output_path, "media": output_media}


class SAMGetSegment:
    @staticmethod
    def is_model_sam():
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        vit_quantized = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        vit_quantized_media = vit_quantized.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
        return vit_quantized_media

    @staticmethod
    def load_model(progress_callback=None):
        device = "cpu"

        checkpoint_dir = "checkpoints"
        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, checkpoint_dir)
        os.environ['TORCH_HOME'] = checkpoint_dir_full

        # load model segmentation
        model_type = "vit_b"

        sam_vit_checkpoint = os.path.join(checkpoint_dir_full, 'sam_vit_b.pth')
        link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_b.pth"])
        if not os.path.exists(sam_vit_checkpoint):
            # check what is internet access
            if is_connected(sam_vit_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                # download pre-trained models from url
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint, progress_callback=progress_callback)
            else:
                lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", level="error")
                raise ConnectionError("Not internet connection or not found link")
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(25, "Inspection models...")

        onnx_vit_checkpoint = os.path.join(checkpoint_dir_full, 'vit_b_quantized.onnx')
        link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_b_quantized.onnx"])
        if not os.path.exists(onnx_vit_checkpoint):
            # check what is internet access
            if is_connected(onnx_vit_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                # download pre-trained models from url
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint, progress_callback=progress_callback)
            else:
                lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", level="error")
                raise ConnectionError("Not internet connection or not found link")
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint, progress_callback=progress_callback)

        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_CPU_TASK_GB, device=device, retries=10):
            lprint(msg="Memory out of necessary for SAM segment load model task.", level="error")
            raise RuntimeError("Memory out of necessary for SAM segment load model task.")

        # Updating progress
        if progress_callback:
            progress_callback(40, "Inspection models...")

        segmentation = SegmentAnything()
        predictor = segmentation.init_sam(sam_vit_checkpoint, model_type, device)

        return predictor

    @staticmethod
    def read(file):
        return read_image_pil(file)

    @staticmethod
    def clear(file):
        if os.path.exists(file) and os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def extract_embedding(image: bytes, model_cache=None, progress_callback=None):
        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")
        predictor = model_cache.get_model("extract_embedding") if model_cache is not None else None
        if predictor is None:
            predictor = SAMGetSegment.load_model(progress_callback)
            # Update cache
            model_cache.update_model("extract_embedding", predictor) if model_cache else None
        # Updating progress
        if progress_callback:
            progress_callback(50, "Get segment...")
        # segmentation
        segmentation = SegmentAnything()
        image = segmentation.preprocess({"image": image}, device="cpu")
        # Updating progress
        if progress_callback:
            progress_callback(75, "Predict...")
        # Run the inference.
        embedding = predictor.image_encoder(image)
        embedding = embedding.cpu().detach().numpy()
        embedding_decoded = segmentation.postprocess(embedding)
        del segmentation, predictor
        torch.cuda.empty_cache()
        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished...")
        return embedding_decoded

    @staticmethod
    def is_valid_image(file):
        return VideoManipulation.check_media_type(file) == "static"

    @staticmethod
    def zero_embedding():
        zero_embedding_decoded = {
            "image_embedding": "",
            "image_embedding_shape": (0, 0, 0, 0),
        }
        return zero_embedding_decoded


class Analyser(BaseModule):
    """Analysing content"""
    @staticmethod
    def analysis(user_name: str, source_file: str):
        method = "Analysis"
        lprint(msg=f"[{method}] Start content analysing.", user=user_name, level="info")
        is_original = VideoManipulation.decrypted(source_file)
        if is_original:
            lprint(msg=f"[{method}] Content is original.", user=user_name, level="info")
        else:
            lprint(msg=f"[{method}] Content is Wunjo fake.", user=user_name, level="info")

        return {"is_original": is_original}


class RemoveBackground(BaseModule):
    """Remove background for object in image or video"""
    @staticmethod
    def is_valid_point_list(points: list, offset_width: int, offset_height: int) -> dict:
        points_dict = {}
        for i, point in enumerate(points):
            if point is None:
                continue
            points_dict[str(i)] = {
                "point_list": [],
                "start_time": None,
                "end_time": None
            }
            for p in point:
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
                click_type = int(p.get("clickType", 1))
                points_dict[str(i)]["point_list"].append({
                    "canvasHeight": int(offset_height),
                    "canvasWidth": int(offset_width),
                    "color": click_type,
                    "x": x,
                    "y": y
                })
        return points_dict

    @staticmethod
    def inspect() -> dict:
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        model_data = {}

        model_info_configs = [
            ("Remove background", "SAM GPU", os.path.join(checkpoint_folder, "sam_vit_h.pth"), ["checkpoints", "sam_vit_h.pth"]),
            ("Remove background", "SAM GPU", os.path.join(checkpoint_folder, "vit_h_quantized.onnx"), ["checkpoints", "vit_h_quantized.onnx"]),
            ("Remove background", "SAM CPU", os.path.join(checkpoint_folder, "sam_vit_b.pth"), ["checkpoints", "sam_vit_b.pth"]),
            ("Remove background", "SAM CPU", os.path.join(checkpoint_folder, "vit_h_quantized.onnx"), ["checkpoints", "vit_h_quantized.onnx"])
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = RemoveBackground.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, masks: dict, user_save_name: str = None,
                          predictor=None, session=None, segment_percentage: int = 25, inverse_mask: bool = False,
                          source_start: float = 0, source_end: float = 0, max_size: int = 1280,
                          mask_color: str = "#ffffff", progress_callback=None):
        min_delay = 20
        max_delay = 120
        try:
            response = RemoveBackground.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                       masks=masks, user_save_name=user_save_name, predictor=predictor, session=session,
                                       segment_percentage=segment_percentage, source_start=source_start, inverse_mask=inverse_mask,
                                       source_end=source_end, max_size=max_size, mask_color=mask_color, progress_callback=progress_callback)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                RemoveBackground.handle_retries_synthesis(min_delay, max_delay, user_name, progress_callback)
                # Run again
                response = RemoveBackground.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                           masks=masks, user_save_name=user_save_name, predictor=predictor,
                                           session=session, inverse_mask=inverse_mask, progress_callback=progress_callback,
                                           segment_percentage=segment_percentage, source_start=source_start,
                                           source_end=source_end, max_size=max_size, mask_color=mask_color)
            else:
                raise RuntimeError(err)
        return response

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, masks: dict, user_save_name: str = None,
                  predictor=None, session=None, segment_percentage: int = 25, inverse_mask: bool = False,
                  source_start: float = 0, source_end: float = 0, max_size: int = 1280, mask_color: str = "#ffffff",
                  progress_callback=None):

        # log message
        method = "Remove background"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)
        # create tmp folder
        tmp_dir = os.path.join(local_save_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        # frame folder
        frame_dir = os.path.join(tmp_dir, "frame")
        os.makedirs(frame_dir, exist_ok=True)
        # resized frame folder
        resized_dir = os.path.join(tmp_dir, "resized")
        os.makedirs(resized_dir, exist_ok=True)

        # get source type
        is_video = RemoveBackground.is_valid_video(source_file)
        is_image = RemoveBackground.is_valid_image(source_file) if not is_video else False

        # params
        source_start = float(source_start) if source_start is not None and RemoveBackground.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and RemoveBackground.is_float(source_end) else 0

        # cut video in os
        if is_video:
            duration = VideoManipulation.get_video_duration_seconds(source_file)
            if source_start != 0 or source_end != duration and source_end != 0:
                source_file = VideoManipulation.cut_video_ffmpeg(source_file, source_start, source_end)

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        msg = f"Before load models in memory"
        memory_message(method, msg, user_name, "info", device=device)

        _, available_vram_gb, busy_vram_gb = get_vram_gb()
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # Determine the model type based on available VRAM
        vit_model_type = "vit_h" if device != "cpu" and available_vram_gb > 7 else "vit_b"

        # Set the download size check message based on device type
        download_size_check = "CHECK_DOWNLOAD_SIZE_SAM_GPU" if device != "cpu" and available_vram_gb > 7 else "CHECK_DOWNLOAD_SIZE_SAM_CPU"

        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
        RemoveBackground.validate_models(sam_vit_checkpoint, link_sam_vit_checkpoint, user_name, download_size_check, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        RemoveBackground.validate_models(onnx_vit_checkpoint, link_onnx_vit_checkpoint, user_name, download_size_check, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        # segment
        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_BACKGROUND_GB, limit_vram=Settings.MIN_VRAM_REMOVE_BACKGROUND_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load segmentation model.", level="error", user=user_name)
            raise RuntimeError(f"Memory out of necessary for {method}.")

        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage, inverse_mask)
        if session is None:
            session = segmentation.init_onnx(os.path.join(checkpoint_folder, f'{vit_model_type}_quantized.onnx'), device)
            lprint(msg=f"[{method}] Model session is loaded.", user=user_name, level="info")
        if predictor is None:
            predictor = segmentation.init_vit(os.path.join(checkpoint_folder, f'sam_{vit_model_type}.pth'), vit_model_type, device)
            lprint(msg=f"[{method}] Model predictor is loaded.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(50, "Init models...")

        msg = f"Load SAM models {device}"
        memory_message(method, msg, user_name, "info", device=device)

        if is_image:
            fps = 0
            # Save frame to frame_dir
            frame_files = ["static_frame.png"]
            cv2.imwrite(os.path.join(frame_dir, frame_files[0]), read_image_cv2(source_file))
        elif is_video:
            fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
            frame_files = sorted(os.listdir(frame_dir))
        else:
            lprint(msg=f"[{method}] Source is not detected as image or video.", user=user_name, level="error")
            raise ValueError("Source is not detected as image or video")

        # if user not set init time for object, set this
        for key in masks.keys():
            if masks[key].get("start_time") is None:
                masks[key]["start_time"] = source_start
            if masks[key].get("end_time") is None:
                masks[key]["end_time"] = source_end

        # Remove keys with time after cut
        remove_keys = []
        for key in masks.keys():
            if masks[key].get("start_time") < source_start:
                remove_keys += [key]
                continue
            if masks[key].get("start_time") > source_end:
                remove_keys += [key]
                continue
            if masks[key].get("end_time") > source_end:
                # if video was cut in end
                masks[key]["end_time"] = source_end
            # if start cud need to reduce end time
            masks[key]["end_time"] = masks[key]["end_time"] - source_start
            # if video was cut in start
            masks[key]["start_time"] = masks[key]["start_time"] - source_start
        else:
            for remove_key in remove_keys:
                masks.pop(remove_key)

        first_frame = read_image_cv2(os.path.join(frame_dir, frame_files[0]))
        orig_height, orig_width, _ = first_frame.shape

        # Resize frames while maintaining aspect ratio
        # Updating progress
        if progress_callback:
            progress_callback(0, "Resize frames...")

        for i, frame_file in enumerate(frame_files):
            frame = read_image_cv2(os.path.join(frame_dir, frame_file))
            h, w, _ = frame.shape
            if max(h, w) > max_size:
                if h > w:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                else:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                resized_frame = cv2.resize(frame, (new_w, new_h))
                cv2.imwrite(os.path.join(resized_dir, frame_file), resized_frame)
            else:
                cv2.imwrite(os.path.join(resized_dir, frame_file), frame)

            # Updating progress
            if progress_callback:
                progress_callback(round(i / len(frame_files)), "Resize frames...")

        # read again after resized
        work_dir = resized_dir
        first_work_frame = read_image_cv2(os.path.join(work_dir, frame_files[0]))
        work_height, work_width, _ = first_work_frame.shape

        # get segmentation frames as maks
        segmentation.load_models(predictor=predictor, session=session)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Init models...")

        # change parameter
        mask_color = segmentation.check_valid_hex(mask_color)
        if is_video and mask_color == "transparent":  # to create chromo key video
            mask_color = "#ffffff"

        for key in masks.keys():
            lprint(msg=f"[{method}] Processing ID: {key}", user=user_name, level="info")
            start_time = masks[key]["start_time"]
            end_time = masks[key]["end_time"]
            start_frame = math.floor(start_time * fps)
            end_frame = math.ceil(end_time * fps) + 1
            # list of frames of list of one frame
            filter_frames_files = frame_files[start_frame:end_frame]
            # set progress bar
            progress_bar = tqdm(total=len(filter_frames_files), unit='it', unit_scale=True)
            filter_frame_file_name = filter_frames_files[0]
            # read mot resized files

            filter_frame = cv2.imread(os.path.join(work_dir, filter_frame_file_name))  # read first frame file after filter
            # get segment mask
            segment_mask = segmentation.set_obj(point_list=masks[key]["point_list"], frame=filter_frame)
            # save first frame as 0001
            color = segmentation.hex_to_rgba(mask_color)
            os.makedirs(os.path.join(local_save_dir, key), exist_ok=True)
            orig_filter_frame = cv2.imread(os.path.join(frame_dir, filter_frame_file_name))
            saving_mask = segmentation.apply_mask_on_frame(segment_mask, orig_filter_frame, color, orig_width, orig_height)
            saving_mask.save(os.path.join(local_save_dir, key, filter_frame_file_name))

            # update progress bar
            progress_bar.update(1)

            # Updating progress
            if progress_callback:
                progress_callback(0, f"Processing ID ({key})...")

            if len(filter_frames_files) > 1:
                for i, filter_frame_file_name in enumerate(filter_frames_files[1:]):
                    filter_frame = cv2.imread(os.path.join(work_dir, filter_frame_file_name))  # read frame file after filter
                    segment_mask = segmentation.draw_mask_frames(frame=filter_frame)
                    if segment_mask is None:
                        lprint(msg=f"[{method}] Encountered None mask. Breaking the loop.", user=user_name, level="info")
                        break
                    # save frames after 0002
                    color = segmentation.hex_to_rgba(mask_color)
                    orig_filter_frame = cv2.imread(os.path.join(frame_dir, filter_frame_file_name))
                    saving_mask = segmentation.apply_mask_on_frame(segment_mask, orig_filter_frame, color, orig_width, orig_height)
                    saving_mask.save(os.path.join(local_save_dir, key, filter_frame_file_name))

                    progress_bar.update(1)

                    # Updating progress
                    if progress_callback:
                        progress_callback(round(i / len(filter_frames_files) * 100, 0), f"Processing ID ({key})...")

            # set new key
            masks[key]["mask_path"] = os.path.join(local_save_dir, key)
            # close progress bar for key
            progress_bar.close()
            msg = f"After processing SAM for {key}"
            memory_message(method, msg, user_name, "info", device=device)

        del segmentation, predictor, session
        torch.cuda.empty_cache()

        for i, key in enumerate(masks.keys()):
            frame_files_path = masks[key]["mask_path"]
            # If files more than 1, create video
            list_files = os.listdir(frame_files_path)
            if len(list_files) > 1:
                # save_name witch create user
                user_save_name = f"{user_save_name}{i}.mp4" if user_save_name is not None else None  # if None will random name generated
                user_save_name = VideoManipulation.save_video_from_list_frames(list_files, frame_files_path, fps, alternative_save_path=user_save_dir, user_file_name=user_save_name)
                # Update code
                if progress_callback:
                    progress_callback(99, "Update codec.")
                user_save_name = VideoManipulation.convert_yuv420p(local_save_dir, os.path.join(user_save_dir, user_save_name))
            elif len(list_files) == 1:
                # move file to user_save_dir
                user_save_name =  f"{user_save_name}{i}.jpg" if user_save_name is not None else str(uuid.uuid4())
                source_path = os.path.join(frame_files_path, list_files[0])
                destination_path = os.path.join(user_save_dir, user_save_name)
                shutil.move(source_path, destination_path)

            # Updating progress
            if progress_callback:
                progress_callback(round(i / len(masks.keys()) * 100, 0), f"Combine...")

        # remove tmp files
        if tmp_source_file != source_file and os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        msg = f"Synthesis finished"
        memory_message(method, msg, user_name, "info", device=device)

        # Update config about models if process was successfully
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU and vit_model_type == "vit_b":
            lprint(msg="Update checker of download size SAM CPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = False
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU and vit_model_type == "vit_h":
            lprint(msg="Update checker of download size SAM GPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = False

        # Updating progress
        if progress_callback:
            progress_callback(100, f"Finished.")

        output_path = os.path.join(user_save_dir, user_save_name)
        output_media = RemoveBackground.get_media_url(output_path)

        return {"file": output_path, "media": output_media}


class RemoveObject(BaseModule):
    """Remove object in image or video"""
    @staticmethod
    def is_valid_point_list(points: list, offset_width: int, offset_height: int) -> dict:
        points_dict = {}
        for i, point in enumerate(points):
            if point is None:
                continue
            points_dict[str(i)] = {
                "point_list": [],
                "start_time": None,
                "end_time": None
            }
            for p in point:
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
                click_type = int(p.get("clickType", 1))
                points_dict[str(i)]["point_list"].append({
                    "canvasHeight": int(offset_height),
                    "canvasWidth": int(offset_width),
                    "color": click_type,
                    "x": x,
                    "y": y
                })
        return points_dict

    @staticmethod
    def inspect() -> dict:
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        model_data = {}

        model_info_configs = [
            ("Remove object", "Improved remove", os.path.join(checkpoint_folder, "ProPainter.pth"), ["checkpoints", "ProPainter.pth"]),
            ("Remove object", "Improved remove", os.path.join(checkpoint_folder, "raft-things.pth"), ["checkpoints", "raft-things.pth"]),
            ("Remove object", "Improved remove", os.path.join(checkpoint_folder, "recurrent_flow_completion.pth"), ["checkpoints", "recurrent_flow_completion.pth"]),
            ("Remove object", "Simple remove", os.path.join(checkpoint_folder, "retouch_object.pth"), ["checkpoints", "retouch_object.pth"]),
            ("Remove object", "SAM GPU", os.path.join(checkpoint_folder, "sam_vit_h.pth"), ["checkpoints", "sam_vit_h.pth"]),
            ("Remove object", "SAM GPU", os.path.join(checkpoint_folder, "vit_h_quantized.onnx"), ["checkpoints", "vit_h_quantized.onnx"]),
            ("Remove object", "SAM CPU", os.path.join(checkpoint_folder, "sam_vit_b.pth"), ["checkpoints", "sam_vit_b.pth"]),
            ("Remove object", "SAM CPU", os.path.join(checkpoint_folder, "vit_h_quantized.onnx"), ["checkpoints", "vit_h_quantized.onnx"]),
            ("Remove object", "Text recognition", os.path.join(checkpoint_folder, "vgg16_baseline.pth"), ["checkpoints", "vgg16_baseline.pth"]),
            ("Remove object", "Text recognition", os.path.join(checkpoint_folder, "vgg16_east.pth"), ["checkpoints", "vgg16_east.pth"])
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = RemoveObject.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, masks: dict = None, user_save_name: str = None,
                          predictor=None, session=None, segment_percentage: int = 25, use_improved_model: bool = False,
                          source_start: float = 0, source_end: float = 0, max_size: int = 1280, text_area: list = None,
                          use_remove_text: bool = False, inverse_mask: bool = False, progress_callback=None):
        min_delay = 20
        max_delay = 120
        try:
            response = RemoveObject.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                   masks=masks, user_save_name=user_save_name, predictor=predictor, inverse_mask=inverse_mask,
                                   session=session, segment_percentage=segment_percentage, source_start=source_start,
                                   source_end=source_end, max_size=max_size, use_improved_model=use_improved_model,
                                   text_area=text_area, use_remove_text=use_remove_text, progress_callback=progress_callback)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                RemoveObject.handle_retries_synthesis(min_delay, max_delay, user_name, progress_callback)
                # Run again
                response = RemoveObject.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                       masks=masks, user_save_name=user_save_name, predictor=predictor, inverse_mask=inverse_mask,
                                       session=session, segment_percentage=segment_percentage, source_start=source_start,
                                       source_end=source_end, max_size=max_size, use_improved_model=use_improved_model,
                                       text_area=text_area, use_remove_text=use_remove_text, progress_callback=progress_callback)
            else:
                raise RuntimeError(err)
        return response

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, masks: dict = None, user_save_name: str = None,
                  predictor=None, session=None, segment_percentage: int = 25, use_improved_model: bool = False,
                  source_start: float = 0, source_end: float = 0, max_size: int = 1280, text_area: list = None,
                  use_remove_text: bool = False, inverse_mask: bool = False, progress_callback=None):

        # method
        method = "Remove object"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)
        # create tmp folder
        tmp_dir = os.path.join(local_save_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        # frame folder
        frame_dir = os.path.join(tmp_dir, "frame")
        os.makedirs(frame_dir, exist_ok=True)
        # resized frame folder
        resized_dir = os.path.join(tmp_dir, "resized")
        os.makedirs(resized_dir, exist_ok=True)

        # mask
        if masks is None or use_remove_text:
            masks = {}

        # get source type
        is_video = RemoveObject.is_valid_video(source_file)
        is_image = RemoveObject.is_valid_image(source_file) if not is_video else False

        # params
        source_start = float(source_start) if source_start is not None and RemoveObject.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and RemoveObject.is_float(source_end) else 0

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")
        use_half = False if device == "cpu" else True

        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        msg = f"Before load model in {device}"
        memory_message(method, msg, user_name, "info", device=device)

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # download models
        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        if is_video and use_improved_model and device != "cpu":
            # load model improved remove object
            model_pro_painter_path = os.path.join(checkpoint_folder, "ProPainter.pth")
            link_pro_painter = get_nested_url(file_models_config(), ["checkpoints", "ProPainter.pth"])
            RemoveObject.validate_models(model_pro_painter_path, link_pro_painter, user_name, "CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT", progress_callback=progress_callback)

            # Updating progress
            if progress_callback:
                progress_callback(14, "Inspection models...")

            model_raft_things_path = os.path.join(checkpoint_folder, "raft-things.pth")
            link_raft_things = get_nested_url(file_models_config(), ["checkpoints", "raft-things.pth"])
            RemoveObject.validate_models(model_raft_things_path, link_raft_things, user_name, "CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT", progress_callback=progress_callback)

            # Updating progress
            if progress_callback:
                progress_callback(28, "Inspection models...")

            model_recurrent_flow_path = os.path.join(checkpoint_folder, "recurrent_flow_completion.pth")
            link_recurrent_flow = get_nested_url(file_models_config(), ["checkpoints", "recurrent_flow_completion.pth"])
            RemoveObject.validate_models(model_recurrent_flow_path, link_recurrent_flow, user_name, "CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT", progress_callback=progress_callback)

            # Empty path
            model_retouch_path = None
        else:
            model_retouch_path = os.path.join(checkpoint_folder, "retouch_object.pth")
            link_model_retouch = get_nested_url(file_models_config(), ["checkpoints", "retouch_object.pth"])
            RemoveObject.validate_models(model_retouch_path, link_model_retouch, user_name, "CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT", progress_callback=progress_callback)

            # Empty paths
            model_raft_things_path = model_recurrent_flow_path = model_pro_painter_path = None

        # Updating progress
        if progress_callback:
            progress_callback(32, "Inspection models...")

        if device != "cpu":
            gpu_vram = available_vram_gb
        else:
            gpu_vram = 0

        # Determine the model type based on available VRAM
        vit_model_type = "vit_h" if device != "cpu" and available_vram_gb > 7 else "vit_b"

        # Set the download size check message based on device type
        download_size_check = "CHECK_DOWNLOAD_SIZE_SAM_GPU" if device != "cpu" and available_vram_gb > 7 else "CHECK_DOWNLOAD_SIZE_SAM_CPU"

        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
        RemoveObject.validate_models(sam_vit_checkpoint, link_sam_vit_checkpoint, user_name, download_size_check, progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        RemoveObject.validate_models(onnx_vit_checkpoint, link_onnx_vit_checkpoint, user_name, download_size_check, progress_callback=progress_callback)

        if use_remove_text:
            # Get models for text detection
            vgg16_baseline_path = os.path.join(checkpoint_folder, "vgg16_baseline.pth")
            link_vgg16_baseline = get_nested_url(file_models_config(), ["checkpoints", "vgg16_baseline.pth"])
            RemoveObject.validate_models(vgg16_baseline_path, link_vgg16_baseline, user_name, "CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT", progress_callback=progress_callback)

            # Updating progress
            if progress_callback:
                progress_callback(74, "Inspection models...")

            vgg16_east_path = os.path.join(checkpoint_folder, "vgg16_east.pth")
            link_vgg16_east = get_nested_url(file_models_config(), ["checkpoints", "vgg16_east.pth"])
            RemoveObject.validate_models(vgg16_east_path, link_vgg16_east, user_name, "CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT", progress_callback=progress_callback)
        else:
            vgg16_baseline_path = vgg16_east_path = None

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        # segment
        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage, inverse_mask)
        if session is None and not use_remove_text:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load segmentation session model.", level="error",  user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")
            session = segmentation.init_onnx(onnx_vit_checkpoint, device)
            msg = f"Model session loaded"
            memory_message(method, msg, user_name, "info", device=device)
        else:
            lprint(msg=f"[{method}] Model session will not be loaded because of not need to use for text.", user=user_name, level="info")
        if predictor is None and not use_remove_text:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load segmentation predictor model.", level="error", user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")
            predictor = segmentation.init_vit(sam_vit_checkpoint, vit_model_type, device)
            msg = f"Model predictor loaded"
            memory_message(method, msg, user_name, "info", device=device)
        else:
            lprint(msg=f"[{method}] Model predictor will not be loaded because of not need to use for text.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Ensure memory capacity...")

        if is_image:
            fps = 0
            # Save frame to frame_dir
            frame_files = ["static_frame.png"]
            cv2.imwrite(os.path.join(frame_dir, frame_files[0]), read_image_cv2(source_file))
        elif is_video:
            fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
            frame_files = sorted(os.listdir(frame_dir))
        else:
            raise ValueError("Source is not detected as image or video")

        # if user not set init time for object, set this
        if not use_remove_text:
            for key in masks.keys():
                if masks[key].get("start_time") is None:
                    masks[key]["start_time"] = source_start
                if masks[key].get("end_time") is None:
                    masks[key]["end_time"] = source_end

        first_frame = read_image_cv2(os.path.join(frame_dir, frame_files[0]))
        orig_height, orig_width, _ = first_frame.shape
        frame_batch_size = 50  # for improve remove object to limit VRAM, for CPU slowly, but people have a lot RAM

        if is_video and use_improved_model and device != "cpu":
            # table of gpu memory
            gpu_table = {32: 1280, 23: 1080, 15: 768, 7: 720, 6: 640, 2: 320}
            # get user resize for gpu
            new_max_size = max(val for key, val in gpu_table.items() if key <= gpu_vram)
            if new_max_size < max_size:
                max_size = new_max_size
                lprint(msg=f"[{method}] Limit available VRAM is {gpu_vram} Gb. Video will resize before {max_size} for max size.", user=user_name, level="warn")

        # Resize frames while maintaining aspect ratio
        for i, frame_file in enumerate(frame_files):
            frame = read_image_cv2(os.path.join(frame_dir, frame_file))
            h, w, _ = frame.shape
            if max(h, w) > max_size:
                if h > w:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                else:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                resized_frame = cv2.resize(frame, (new_w, new_h))
                cv2.imwrite(os.path.join(resized_dir, frame_file), resized_frame)
            else:
                cv2.imwrite(os.path.join(resized_dir, frame_file), frame)
            # Updating progress
            if progress_callback:
                progress_callback(round(i / len(frame_files) * 100, 0), "Resize frames...")
        work_dir = resized_dir
        lprint(msg=f"[{method}] Files resized", user=user_name, level="info")

        # read again after resized
        first_work_frame = read_image_cv2(os.path.join(work_dir, frame_files[0]))
        work_height, work_width, _ = first_work_frame.shape

        # Updating progress
        if progress_callback:
            progress_callback(0, "Init model...")

        # get segmentation frames as maks
        segmentation.load_models(predictor=predictor, session=session)
        thickness_mask = 10

        if not use_remove_text and session is not None and predictor is not None:
            for key in masks.keys():
                lprint(msg=f"[{method}] Processing ID: {key}", user=user_name, level="info")
                mask_key_save_path = os.path.join(tmp_dir, f"mask_{key}")
                os.makedirs(mask_key_save_path, exist_ok=True)
                start_time = masks[key]["start_time"]
                end_time = masks[key]["end_time"]
                start_frame = math.floor(start_time * fps)
                end_frame = math.ceil(end_time * fps) + 1
                # list of frames of list of one frame
                filter_frames_files = frame_files[start_frame:end_frame]
                # set progress bar
                progress_bar = tqdm(total=len(filter_frames_files), unit='it', unit_scale=True)
                filter_frame_file_name = filter_frames_files[0]
                # read mot resized files
                filter_frame = cv2.imread(os.path.join(work_dir, filter_frame_file_name))  # read first frame file after filter
                # get segment mask
                segment_mask = segmentation.set_obj(point_list=masks[key]["point_list"], frame=filter_frame)
                # save first frame as 0001
                segmentation.save_black_mask(filter_frame_file_name, segment_mask, mask_key_save_path, kernel_size=thickness_mask, width=work_width, height=work_height)

                # update progress bar
                progress_bar.update(1)

                # Updating progress
                if progress_callback:
                    progress_callback(0, f"Processing ID ({key})...")

                if len(filter_frames_files) > 1:
                    for i, filter_frame_file_name in enumerate(filter_frames_files[1:]):
                        filter_frame = cv2.imread(os.path.join(work_dir, filter_frame_file_name))  # read frame file after filter
                        segment_mask = segmentation.draw_mask_frames(frame=filter_frame)
                        if segment_mask is None:
                            lprint(msg=f"[{method}] Encountered None mask for {key}. Breaking the loop.", user=user_name, level="info")
                            break
                        # save frames after 0002
                        segmentation.save_black_mask(filter_frame_file_name, segment_mask, mask_key_save_path, kernel_size=thickness_mask, width=work_width, height=work_height)

                        progress_bar.update(1)
                        # Updating progress
                        if progress_callback:
                            progress_callback(round(i / len(filter_frames_files) * 100, 0), f"Processing ID ({key})...")

                # set new key
                masks[key]["mask_path"] = mask_key_save_path
                # close progress bar for key
                progress_bar.close()

                msg = f"After processing SAM for {key}"
                memory_message(method, msg, user_name, "info", device=device)

        if use_remove_text and vgg16_baseline_path is not None and vgg16_east_path is not None:
            lprint(msg=f"[{method}] Processing text", user=user_name, level="info")
            mask_text_save_path = os.path.join(tmp_dir, f"mask_text")
            os.makedirs(mask_text_save_path, exist_ok=True)
            start_time = source_start * fps
            end_time = source_end * fps
            start_frame = math.floor(start_time)
            end_frame = math.ceil(end_time) + 1
            # list of frames of list of one frame
            filter_frames_files = frame_files[start_frame:end_frame]
            # load model
            # Updating progress
            if progress_callback:
                progress_callback(0, "Ensure memory capacity...")
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load find text model.", level="error", user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")
            segment_text = SegmentText(device=device, vgg16_path=vgg16_baseline_path, east_path=vgg16_east_path)
            # Updating progress
            if progress_callback:
                progress_callback(50, "Init model...")
            text_area = [None] if text_area is None else text_area

            # set progress bar
            progress_bar = tqdm(total=len(filter_frames_files) * len(text_area), unit='it', unit_scale=True)
            for area in text_area:
                for i, frame_file in enumerate(filter_frames_files):
                    frame_file_path = os.path.join(frame_dir, frame_file)
                    mask_text_frame = segment_text.detect_text(img_path=frame_file_path)
                    if area is not None and isinstance(area, dict):
                        w, h = mask_text_frame.size
                        x1, y1, x2, y2 = RemoveObject.scale_coord(area, w, h)
                        coord = [x1, y1, x2, y2]
                    else:
                        coord = None
                    mask_textarea_frame = segment_text.crop(mask_text_frame, coord)
                    mask_text_frame_path = os.path.join(mask_text_save_path, frame_file)
                    mask_textarea_frame.save(mask_text_frame_path)
                    progress_bar.update(1)

                    # Updating progress
                    if progress_callback:
                        progress_callback(round(i / len(filter_frames_files) * 100, 0), "Text search...")
            # close progress bar for text mask
            progress_bar.close()

            del segment_text
            # Set mask path in masks
            masks["text"] = {'mask_path': mask_text_save_path, 'start_time': start_time, 'end_time': end_time}

            msg = f"After processing text recognition"
            memory_message(method, msg, user_name, "info", device=device)

        del segmentation, predictor, session
        torch.cuda.empty_cache()

        msg = f"After SAM model"
        memory_message(method, msg, user_name, "info", device=device)

        if is_video and use_improved_model and device != "cpu":
            if model_raft_things_path is None or model_recurrent_flow_path is None or model_pro_painter_path is None:
                lprint(msg=f"[{method}] Path for improved remove object models is None.", user=user_name, level="error")
                return

            # raft
            # Updating progress
            if progress_callback:
                progress_callback(0, "Ensure memory capacity...")
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load improve remove object model.", level="error",user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")
            retouch_processor = RemoveObjectProcessor(device, model_raft_things_path, model_recurrent_flow_path, model_pro_painter_path)
            overlap = int(0.2 * frame_batch_size)
            # Updating progress
            if progress_callback:
                progress_callback(100, "Init models...")

            for key in masks.keys():
                mask_files = sorted(os.listdir(masks[key]["mask_path"]))

                # processing video with batch size because of limit VRAM
                if len(mask_files) < 3:
                    continue

                for i in range(0, len(mask_files), frame_batch_size - overlap):
                    if len(mask_files) - i < 3:  # not less than 3 frames for batch
                        continue
                    # read mask by batch_size and convert back from file image
                    current_mask_files = mask_files[i: i + frame_batch_size]
                    inpaint_mask_frames = [cv2.imread(os.path.join(tmp_dir, f"mask_{key}", current_mask_file), cv2.IMREAD_GRAYSCALE) for current_mask_file in current_mask_files]
                    # Binarize the image
                    binary_masks = []
                    for idx, inpaint_mask_frame in enumerate(inpaint_mask_frames):
                        binary_mask = cv2.threshold(inpaint_mask_frame, 128, 1, cv2.THRESH_BINARY)[1]
                        # If it's not the first batch and the frame is within the overlap range, set the mask to false
                        if i > 0 and idx < overlap:
                            binary_mask = np.zeros_like(binary_mask)
                        bool_mask = binary_mask.astype(bool)
                        reshaped_mask = bool_mask[np.newaxis, np.newaxis, ...]
                        binary_masks.append(reshaped_mask)
                    # read frames by name of current_mask_files
                    current_frame_files = current_mask_files
                    current_frames = [cv2.imread(os.path.join(work_dir, current_frame_file)) for current_frame_file in current_frame_files]

                    # convert frame to pillow
                    current_frames = [retouch_processor.transfer_to_pil(f) for f in current_frames]
                    resized_frames, size, out_size = retouch_processor.resize_frames(current_frames)
                    width, height = size
                    blur = 1

                    flow_masks, masks_dilated = retouch_processor.read_retouch_mask(binary_masks, width, height, kernel_size=blur)
                    update_frames, flow_masks, masks_dilated, masked_frame_for_save, frames_inp = retouch_processor.process_frames_with_retouch_mask( frames=resized_frames, masks_dilated=masks_dilated, flow_masks=flow_masks, height=height, width=width)
                    comp_frames = retouch_processor.process_video_with_mask(update_frames, masks_dilated, flow_masks, frames_inp, width, height, use_half=use_half)
                    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
                    comp_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in comp_frames]
                    # update frames in work dir
                    for j in range(len(comp_frames_bgr)):
                        cv2.imwrite(os.path.join(work_dir, current_frame_files[j]), comp_frames_bgr[j])

                    # Updating progress
                    if progress_callback:
                        progress_callback(round(i / len(mask_files) * 100, 0), "Improve remover...")

                msg = f"Improve remove object {key}"
                memory_message(method, msg, user_name, "info", device=device)

            # empty cache
            del retouch_processor
            torch.cuda.empty_cache()
        else:
            if model_retouch_path is None:
                lprint(msg=f"[{method}] Path for remove object model is None.", user=user_name, level="error")
                return

            # retouch
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load simple remove object model.", level="error", user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")
            model_retouch = InpaintModel(model_path=model_retouch_path)

            for key in masks.keys():
                mask_files = sorted(os.listdir(masks[key]["mask_path"]))
                # set progress bar
                progress_bar = tqdm(total=len(mask_files), unit='it', unit_scale=True)
                for i, file_name in enumerate(mask_files):
                    # read mask to pillow
                    segment_mask = cv2.imread(os.path.join(tmp_dir, f"mask_{key}", file_name), cv2.IMREAD_GRAYSCALE)
                    binary_mask = cv2.threshold(segment_mask, 128, 1, cv2.THRESH_BINARY)[1]
                    bool_mask = binary_mask.astype(bool)
                    reshaped_mask = bool_mask[np.newaxis, np.newaxis, ...]
                    segment_mask_pil = RemoveObjectUtils.convert_colored_mask_thickness_cv2(reshaped_mask)
                    # read frame to pillow
                    current_frame = cv2.imread(os.path.join(work_dir, file_name))
                    frame_pil = RemoveObjectUtils.convert_cv2_to_pil(current_frame)
                    # retouch_frame
                    retouch_frame = RemoveObjectUtils.process_remover(img=frame_pil, mask=segment_mask_pil, model=model_retouch)
                    # update frame
                    cv2.imwrite(os.path.join(work_dir, file_name), RemoveObjectUtils.pil_to_cv2(retouch_frame))
                    progress_bar.update(1)

                    # Updating progress
                    if progress_callback:
                        progress_callback(round(i / len(mask_files) * 100, 0), "Simple remover...")
                # close progress bar for key
                progress_bar.close()

                msg = f"Simple remove object {key}"
                memory_message(method, msg, user_name, "info", device=device)

            # empty cache
            del model_retouch
            torch.cuda.empty_cache()

        msg = f"After finished remove model clear memory"
        memory_message(method, msg, user_name, "info", device=device)

        # Updating progress
        if progress_callback:
            progress_callback(0, "After processing...")

        if is_video:
            # get saved file as merge frames to video
            video_name = VideoManipulation.save_video_from_frames(frame_names="frame%04d.png", save_path=work_dir, fps=fps, alternative_save_path=local_save_dir)
            # get audio from video target
            audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
            # combine audio and video
            save_name = VideoManipulation.save_video_with_audio(os.path.join(local_save_dir, video_name), os.path.join(local_save_dir, str(audio_file_name)), local_save_dir)
            # Update code
            if progress_callback:
                progress_callback(99, "Update codec.")
            save_name = VideoManipulation.convert_yuv420p(local_save_dir, os.path.join(local_save_dir, save_name))
            # transfer video to user path
            user_save_name = f"{user_save_name}.mp4" if user_save_name is not None else save_name
            shutil.move(os.path.join(local_save_dir, save_name), os.path.join(user_save_dir, user_save_name))
        else:
            save_name = frame_files[0]
            user_save_name = f"{user_save_name}.jpg" if user_save_name is not None else save_name
            retouched_frame = read_image_cv2(os.path.join(work_dir, save_name))
            cv2.imwrite(os.path.join(user_save_dir, user_save_name), retouched_frame)

        # Updating progress
        if progress_callback:
            progress_callback(100, "After processing...")

        # remove tmp files
        if tmp_source_file != source_file and os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis remove object finished.", user=user_name, level="info")

        # Update config about models if process was successfully
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU and vit_model_type == "vit_b":
            lprint(msg="Update checker of download size SAM CPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = False
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU and vit_model_type == "vit_h":
            lprint(msg="Update checker of download size SAM GPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = False
        if Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT and use_remove_text:
            lprint(msg="Update checker of download size detector text models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT": False})
            Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT = False
        if Settings.CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT and is_video and use_improved_model and device != "cpu":
            lprint(msg="Update checker of download size improve remove object models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT": False})
            Settings.CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT = False
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT:
                lprint(msg="Update checker of download size simple remove object models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT": False})
                Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT = False

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        output_path = os.path.join(user_save_dir, user_save_name)
        output_media = RemoveObject.get_media_url(output_path)

        return {"file": os.path.join(user_save_dir, user_save_name), "media": output_media}


class FaceSwap(BaseModule):
    """
    Face swap by one photo
    """

    @staticmethod
    def inspect() -> dict:
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        model_dir_full = os.path.join(ALL_MODEL_FOLDER, "models")
        model_data = {}

        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'

        model_info_configs = [
            ("Face swap", "Swapper", os.path.join(checkpoint_folder, "faceswap.onnx"), ["checkpoints", "faceswap.onnx"]),
            ("Face swap", "Face alignment", os.path.join(model_dir_full, "buffalo_l"), "%s/%s.zip" % (BASE_REPO_URL, 'buffalo_l'))
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = FaceSwap.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, avatar_file: str, user_save_name: str = None,
                          source_crop_area: list = None, avatar_crop_area: list = None, progress_callback=None,
                          source_start: float = 0, source_end: float = 0, resize_factor: float = 1, is_gemini_faces: bool = False,
                          replace_all_faces: bool = False):
        min_delay = 20
        max_delay = 120
        try:
            response = FaceSwap.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                               avatar_file=avatar_file, user_save_name=user_save_name, source_crop_area=source_crop_area,
                               avatar_crop_area=avatar_crop_area, resize_factor=resize_factor, source_start=source_start,
                               source_end=source_end, is_gemini_faces=is_gemini_faces, replace_all_faces=replace_all_faces,
                               progress_callback=progress_callback)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                FaceSwap.handle_retries_synthesis(min_delay, max_delay, user_name, progress_callback)
                # Run again
                response = FaceSwap.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                   avatar_file=avatar_file, user_save_name=user_save_name,
                                   source_crop_area=source_crop_area,
                                   avatar_crop_area=avatar_crop_area, resize_factor=resize_factor,
                                   source_start=source_start, progress_callback=progress_callback,
                                   source_end=source_end, is_gemini_faces=is_gemini_faces,
                                   replace_all_faces=replace_all_faces)
            else:
                raise RuntimeError(err)
        return response

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, avatar_file: str, user_save_name: str = None,
                  source_crop_area: list = None, avatar_crop_area: list = None, progress_callback=None,
                  source_start: float = 0, source_end: float = 0, resize_factor: float = 1, is_gemini_faces: bool = False,
                  replace_all_faces: bool = False):

        # method
        method = "Face swap"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        # Parameters
        resize_factor = 1 if resize_factor is None else resize_factor
        crop = [0, -1, 0, -1]
        rotate = False

        # get source type
        is_video = FaceSwap.is_valid_video(source_file)

        # params
        saved_file = None
        source_start = float(source_start) if source_start is not None and FaceSwap.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and FaceSwap.is_float(source_end) else 0

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        msg = f"Before loading model in memory"
        memory_message(method, msg, user_name, "info", device=device)

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # Load model
        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_dir_full
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
        link_faceswap_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "faceswap.onnx"])
        FaceSwap.validate_models(faceswap_checkpoint, link_faceswap_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_FACE_SWAP", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        # get fps and calculate current frame and get that frame for source
        avatar_frame = VideoManipulation.get_first_frame(avatar_file, 0)

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        # Init face swap
        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_FACE_SWAP_GB, limit_vram=Settings.MIN_VRAM_FACE_SWAP_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load face swap model.", level="error", user=user_name)
            raise RuntimeError(f"Memory out of necessary for {method}.")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Init model...")

        faceswap = FaceSwapProcessing(ALL_MODEL_FOLDER, faceswap_checkpoint, is_gemini_faces, device, progress_callback=progress_callback)

        for j in range(len(source_crop_area)):  # To change a few faces
            lprint(msg=f"[{method}] Processing area {j + 1}.", user=user_name, level="info")

            current_source_crop_area = source_crop_area[j]
            current_avatar_crop_area = avatar_crop_area[j % len(avatar_crop_area)]  # Repeat elements if necessary

            source_face = faceswap.face_detect_with_alignment_from_source_frame(avatar_frame, current_avatar_crop_area)
            lprint(msg=f"[{method}] Got avatar face {j + 1}.", user=user_name, level="info")

            if is_video:
                # get video frame for source if type is video
                frame_dir = os.path.join(local_save_dir, f"frames_{j}")
                os.makedirs(frame_dir, exist_ok=True)

                fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=rotate, crop=crop,  resize_factor=resize_factor)
                start_frame = int(source_crop_area[j]["startTime"] * fps) if source_crop_area[j].get("startTime") else int(source_start * fps)
                end_frame = int(source_crop_area[j]["endTime"] * fps + 1) if source_crop_area[j].get("endTime") else int(source_end * fps + 1)
                # create face swap
                file_name = faceswap.swap_video(frame_dir, source_face, current_source_crop_area, local_save_dir, replace_all_faces, fps, start_frame, end_frame)
                saved_file = os.path.join(local_save_dir, file_name)
            else:  # static file
                # create face swap on image
                frame = VideoManipulation.get_first_frame(source_file)
                image = faceswap.swap_image(frame, source_face, current_source_crop_area, local_save_dir, replace_all_faces)
                file_name = "swap_result.png"
                saved_file = save_image_cv2(os.path.join(local_save_dir, file_name), image)

            lprint(msg=f"[{method}] Completed face swap for {j + 1}.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(100, "Face swap...")

        # after generation
        if saved_file is not None:
            # empty cache
            del faceswap
            torch.cuda.empty_cache()

            msg = f"After finished and clear memory"
            memory_message(method, msg, user_name, "info", device=device)

            # Updating progress
            if progress_callback:
                progress_callback(0, "Encrypted...")

            try:
                file_name = VideoManipulation.encrypted(saved_file, local_save_dir, progress_callback=progress_callback)
                saved_file = os.path.join(local_save_dir, file_name)
            except Exception as err:
                print(f"Error with VideoManipulation.encrypted {err}")

            # Updating progress
            if progress_callback:
                progress_callback(100, "Encrypted...")

            if is_video:
                # get audio from video target
                audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
                # combine audio and video
                file_name = VideoManipulation.save_video_with_audio(saved_file, os.path.join(local_save_dir, str(audio_file_name)), local_save_dir)
                # Update code
                if progress_callback:
                    progress_callback(99, "Update codec.")
                file_name = VideoManipulation.convert_yuv420p(local_save_dir, os.path.join(local_save_dir, file_name))
                saved_file = os.path.join(local_save_dir, file_name)

            if is_video:
                user_save_name += ".mp4"
            else:
                user_save_name += ".jpg"

            shutil.move(saved_file, os.path.join(user_save_dir, user_save_name))

            # remove tmp files
            if os.path.exists(avatar_file):
                os.remove(avatar_file)
            if tmp_source_file != source_file and os.path.exists(source_file):
                os.remove(source_file)
            if os.path.exists(local_save_dir):
                shutil.rmtree(local_save_dir)

            lprint(msg=f"[{method}] Synthesis face swap finished.", user=user_name, level="info")

            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP:
                lprint(msg="Update checker of download size face swap models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_FACE_SWAP": False})
                Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP = False

            # Updating progress
            if progress_callback:
                progress_callback(100, "Finished.")

            output_path = os.path.join(user_save_dir, user_save_name)
            output_media = FaceSwap.get_media_url(output_path)

            return {"file": os.path.join(user_save_dir, user_save_name), "media": output_media}
        else:
            lprint(msg=f"[{method}] Synthesis face swap finished but not created save file.", user=user_name, level="error")
            return {"file": None}


class Enhancement(BaseModule):
    """Enhancement"""

    @staticmethod
    def inspect() -> dict:
        gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        model_data = {}

        model_info_configs = [
            ("Enhancement", "Enhancement faces", os.path.join(gfpgan_path, "GFPGANv1.4.pth"), ["gfpgan", "GFPGANv1.4.pth"]),
            ("Enhancement", "Enhancement draw video", os.path.join(gfpgan_path, "realesr-animevideov3.pth"), ["gfpgan", "realesr-animevideov3.pth"]),
            ("Enhancement", "Enhancement simple video", os.path.join(gfpgan_path, 'realesr-general-x4v3.pth'), ["gfpgan", 'realesr-general-x4v3.pth']),
            ("Enhancement", "Enhancement simple video", os.path.join(gfpgan_path, 'realesr-general-wdn-x4v3.pth'), ["gfpgan", 'realesr-general-wdn-x4v3.pth']),
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = Enhancement.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None,
                          enhancer: str="gfpgan", progress_callback=None):
        min_delay = 20
        max_delay = 120
        try:
            response = Enhancement.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                  enhancer=enhancer, user_save_name=user_save_name, progress_callback=progress_callback)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                Enhancement.handle_retries_synthesis(min_delay, max_delay, user_name, progress_callback)
                # Run again
                response = Enhancement.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                      enhancer=enhancer, user_save_name=user_save_name, progress_callback=progress_callback)
            else:
                raise RuntimeError(err)
        return response

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None,
                  enhancer: str="gfpgan", progress_callback=None):
        # method
        method = "Enhancement"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        # get source type
        is_video = Enhancement.is_valid_video(source_file)

        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        enhancer = "gfpgan" if device == "cpu" else enhancer

        if enhancer not in ["gfpgan", "animesgan", "realesrgan"]:
            lprint(msg=f"[{method}] Enhancer {enhancer} not found in list.", user=user_name, level="info")
            return {"file": None}

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        # gfpgan
        local_gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_gfpgan_path, 'GFPGANv1.4.pth')
        link_gfpgan_checkpoint = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
        Enhancement.validate_models(gfpgan_model_path, link_gfpgan_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(25, "Inspection models...")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        if is_video:
            audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
            stream = cv2.VideoCapture(source_file)
            fps = stream.get(cv2.CAP_PROP_FPS)
            stream.release()
            lprint(msg=f"[{method}] Start video processing.", user=user_name, level="info")
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_ENHANCEMENT_GB, limit_vram=Settings.MIN_VRAM_ENHANCEMENT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load {enhancer} model.", level="error", user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")

            enhanced_name = content_enhancer(source_file, save_folder=local_save_dir, method=enhancer, fps=float(fps), device=device, progress_callback=progress_callback)
            enhanced_path = os.path.join(local_save_dir, enhanced_name)
            save_name = VideoManipulation.save_video_with_audio(enhanced_path, os.path.join(local_save_dir, audio_file_name), local_save_dir)
            # Update code
            if progress_callback:
                progress_callback(99, "Update codec.")
            save_name = VideoManipulation.convert_yuv420p(local_save_dir, os.path.join(local_save_dir, save_name))
        else:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_ENHANCEMENT_GB, limit_vram=Settings.MIN_VRAM_ENHANCEMENT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load {enhancer} model.", level="error", user=user_name)
                raise RuntimeError(f"Memory out of necessary for {method}.")

            lprint(msg="Starting improve image", user=user_name, level="info")
            save_name = content_enhancer(source_file, save_folder=local_save_dir, method=enhancer, device=device, progress_callback=progress_callback)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if is_video:
            user_save_name = f"{user_save_name}.mp4" if user_save_name else save_name
        else:
            user_save_name += f"{user_save_name}.jpg" if user_save_name else save_name

        shutil.move(os.path.join(local_save_dir, save_name), os.path.join(user_save_dir, user_save_name))

        # remove tmp files
        if tmp_source_file != source_file and os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis enhancement finished.", user=user_name, level="info")

        if enhancer == "gfpgan":
            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN and device != "cpu":
                lprint(msg="Update checker of download size gfpgan models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": False})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = False

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        output_path = os.path.join(user_save_dir, user_save_name)
        output_media = Enhancement.get_media_url(output_path)

        return {"file": os.path.join(user_save_dir, user_save_name), "media": output_media}


class LipSync(BaseModule):
    @staticmethod
    def inspect() -> dict:
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        model_dir_full = os.path.join(ALL_MODEL_FOLDER, "models")
        gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        model_data = {}

        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'

        model_info_configs = [
            ("Lip sync", "Mouth animation", os.path.join(checkpoint_folder, "wav2lip.pth"), ["checkpoints", "wav2lip.pth"]),
            ("Lip sync", "Face alignment", os.path.join(model_dir_full, "buffalo_l"), "%s/%s.zip" % (BASE_REPO_URL, 'buffalo_l')),
            ("Lip sync", "Face alignment", os.path.join(checkpoint_folder, "RetinaFace-R50.pth"), ["checkpoints", "RetinaFace-R50.pth"]),
            ("Lip sync", "Face alignment", os.path.join(checkpoint_folder, "ParseNet-latest.pth"), ["checkpoints", "ParseNet-latest.pth"]),
            ("Lip sync", "Enhancement face", os.path.join(gfpgan_path, "GFPGANv1.4.pth"), ["gfpgan", "GFPGANv1.4.pth"]),
        ]

        for category, sub_category, file_path, url_key in model_info_configs:
            model_info = LipSync.get_model_info(file_path, url_key)
            if not model_info.get("is_exist"):
                if model_data.get(category) is None:
                    model_data[category] = {}
                if model_data[category].get(sub_category) is None:
                    model_data[category][sub_category] = []
                model_data[category][sub_category].append(model_info)

        return model_data

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, audio_file: str, progress_callback=None,
                          user_save_name: str = None, source_crop_area: list = None, source_start: float = 0,
                          source_end: float = 0, audio_start: float = 0, audio_end: float = 0, overlay_audio: bool = False,
                          resize_factor: float = 1, use_gfpgan: bool = False, use_smooth: bool = False, use_diffuser: bool = False):
        min_delay = 20
        max_delay = 120
        try:
            response = LipSync.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file, progress_callback=progress_callback,
                              audio_file=audio_file, user_save_name=user_save_name, source_crop_area=source_crop_area,
                              source_start=source_start, source_end=source_end, audio_start=audio_start, use_diffuser=use_diffuser,
                              audio_end=audio_end, overlay_audio=overlay_audio, use_gfpgan=use_gfpgan, use_smooth=use_smooth,
                              resize_factor=resize_factor)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                LipSync.handle_retries_synthesis(min_delay, max_delay, user_name, progress_callback)
                # Run again
                response = LipSync.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                  audio_file=audio_file, user_save_name=user_save_name,
                                  source_crop_area=source_crop_area, use_gfpgan=use_gfpgan, use_smooth=use_smooth,
                                  source_start=source_start, source_end=source_end, audio_start=audio_start,
                                  audio_end=audio_end, overlay_audio=overlay_audio, use_diffuser=use_diffuser,
                                  resize_factor=resize_factor, progress_callback=progress_callback)
            else:
                raise RuntimeError(err)
        return response

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, audio_file: str, progress_callback=None,
                  user_save_name: str = None, source_crop_area: list = None, source_start: float = 0,
                  source_end: float = 0, audio_start: float = 0, audio_end: float = 0, overlay_audio: bool = False,
                  resize_factor: float = 1, use_gfpgan: bool = False, use_smooth: bool = False, use_diffuser: bool = False):
        # method
        method = "Lip sync"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        # Params
        batch_size = 128
        img_size = 96
        source_crop_area = source_crop_area[0] if len(source_crop_area) > 0 else {}
        source_start = float(source_start) if source_start is not None and LipSync.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and LipSync.is_float(source_end) else 0
        audio_start = float(audio_start) if audio_start is not None and LipSync.is_float(audio_start) else 0
        audio_end = float(audio_end) if audio_end is not None and LipSync.is_float(audio_end) else 0

        # get source type
        lprint(msg=f"[{method}] Get media type.", user=user_name, level="info")
        is_video = LipSync.is_valid_video(source_file)

        # Extract full audio
        if is_video:
            lprint(msg=f"[{method}] Extract audio.", user=user_name, level="info")
            # get audio from video target
            source_audio_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
        else:
            source_audio_name = None

        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        device = get_best_device(use_cpu)
        lprint(msg=f"[{method}] Processing will run on {device}.", user=user_name, level="info")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Inspection models...")

        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_dir_full
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        sd_vae_path, latent_sync_path = None, None

        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        link_wav2lip_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "wav2lip.pth"])
        LipSync.validate_models(wav2lip_checkpoint, link_wav2lip_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_LIP_SYNC", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(25, "Inspection models...")

        # get models to smooth face
        retina_face_checkpoint = os.path.join(checkpoint_dir_full, 'RetinaFace-R50.pth')
        link_retina_face_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "RetinaFace-R50.pth"])
        LipSync.validate_models(retina_face_checkpoint, link_retina_face_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_LIP_SYNC", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(50, "Inspection models...")

        face_parse_checkpoint = os.path.join(checkpoint_dir_full, 'ParseNet-latest.pth')
        link_face_parse_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "ParseNet-latest.pth"])
        LipSync.validate_models(face_parse_checkpoint, link_face_parse_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_LIP_SYNC", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(75, "Inspection models...")

        # gfpgan
        local_gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_gfpgan_path, 'GFPGANv1.4.pth')
        link_gfpgan_checkpoint = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
        LipSync.validate_models(gfpgan_model_path, link_gfpgan_checkpoint, user_name, "CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN", progress_callback=progress_callback)

        # Updating progress
        if progress_callback:
            progress_callback(100, "Inspection models...")

        # Cut audio
        if audio_start is not None and audio_end is not None:
            if audio_start == audio_end:
                lprint(msg=f"[{method}] Audio can not have equal start and end time.", user=user_name, level="error")
                return
            else:
                lprint(msg=f"[{method}] Audio cut started.", user=user_name, level="info")
                audio_file = cut_audio(audio_file, audio_start, audio_end, local_save_dir)

        if is_video:
            # get video frame for source if type is video
            frame_dir = os.path.join(local_save_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            lprint(msg=f"[{method}] Save video to frames.", user=user_name, level="info")
            fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=False, crop=[0, -1, 0, -1], resize_factor=resize_factor)
            # Get a list of all frame files in the target_frames_path directory
            frame_files = sorted([os.path.join(frame_dir, fname) for fname in os.listdir(frame_dir) if fname.endswith('.png')])
            frame_files_slice = frame_files[int(source_start * fps): int((source_end * fps) + 1)]
            frame_files_reverse = frame_files_slice[::-1]  # Reverse the sliced list
            frame_files_range = frame_files_slice + frame_files_reverse
        else:
            fps = 25
            if resize_factor != 1:
                frame = read_image_cv2(source_file)
                frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))
                # Write the frame to the image file
                source_file = os.path.join(local_save_dir, f"{uuid.uuid4()}.jpg")
                cv2.imwrite(source_file, frame)
            frame_files_slice = [source_file]
            frame_files_range = [source_file]

        # get mel of audio
        if progress_callback:
            progress_callback(0, "Get mel audio...")

        mel_processor = MelProcessor(audio=audio_file, save_output=local_save_dir, fps=fps)
        mel_chunks = mel_processor.process()

        if progress_callback:
            progress_callback(100, "Get mel audio...")

        # create wav to lip
        full_frames_files = frame_files_range[:len(mel_chunks)]

        # Updating progress
        if progress_callback:
            progress_callback(0, "Ensure memory capacity...")

        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_LIP_SYNC_GB, limit_vram=Settings.MIN_VRAM_LIP_SYNC_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load lip sync models.", level="error", user=user_name)
            raise RuntimeError(f"Memory out of necessary for {method}.")

        # Updating progress
        if progress_callback:
            progress_callback(0, "Init model...")

        wav2lip = GenerateWave2Lip(
            ALL_MODEL_FOLDER, retina_face=retina_face_checkpoint, face_parse=face_parse_checkpoint,
            local_gfpgan_path=local_gfpgan_path, gfpgan_model_path=gfpgan_model_path, device=device,
            use_gfpgan=use_gfpgan, use_smooth=use_smooth, progress_callback=progress_callback
        )

        if wav2lip is None:
            lprint(msg="List index out of range in datagen of lip sync for.", level="error", user=user_name)
            raise ValueError("List index out of range in datagen of lip sync for, you can try set another timeline or face.")

        lprint(msg=f"[{method}] Face detection started.", user=user_name, level="info")
        gen = wav2lip.datagen(full_frames_files, mel_chunks, img_size, batch_size, crop=source_crop_area)
        # load wav2lip
        lprint(msg=f"[{method}] Lip sync started.", user=user_name, level="info")
        wav2lip_processed_video = wav2lip.generate_video_from_chunks(gen, mel_chunks, batch_size, wav2lip_checkpoint, device, local_save_dir, fps)
        if wav2lip_processed_video is None:
            lprint(msg=f"[{method}] Lip sync video in not created.", user=user_name, level="error")
            return
        wav2lip_result_video = wav2lip_processed_video

        del wav2lip

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        msg = f"After finished and clear memory"
        memory_message(method, msg, user_name, "info", device=device)

        # Updating progress
        if progress_callback:
            progress_callback(0, "Encrypted...")

        try:
            file_name = VideoManipulation.encrypted(wav2lip_result_video, local_save_dir, progress_callback=progress_callback)
            mp4_file_path = os.path.join(local_save_dir, file_name)
        except Exception as err:
            print(f"Error with VideoManipulation.encrypted {err}")
            mp4_file_path = wav2lip_result_video

        # Updating progress
        if progress_callback:
            progress_callback(100, "Encrypted...")

        lprint(msg=f"[{method}] Save video with audio.", user=user_name, level="info")
        mp4_file_name = VideoManipulation.save_video_with_audio(mp4_file_path, audio_file, local_save_dir)
        mp4_file_path = os.path.join(local_save_dir, mp4_file_name)

        if os.path.exists(mp4_file_path) and os.path.getsize(mp4_file_path) > 1024:  # 1 Kb
            if overlay_audio and source_audio_name:
                # Updating progress
                if progress_callback:
                    progress_callback(0, "Overlay audio...")
                lprint(msg=f"[{method}] Overlay audio.", user=user_name, level="info")
                output_file = VideoManipulation.overlay_audio_on_video(mp4_file_path, os.path.join(local_save_dir, source_audio_name), save_dir=local_save_dir)
                if not os.path.exists(output_file):
                    output_file = mp4_file_path
                # Updating progress
                if progress_callback:
                    progress_callback(100, "Overlay audio...")
            else:
                lprint(msg=f"[{method}] Concatenated video have small size.", user=user_name, level="warn")
                output_file = mp4_file_path
        else:  # if video concatenate this error and have small size
            output_file = mp4_file_path

        # Update code
        if progress_callback:
            progress_callback(99, "Update codec.")
        file_name = VideoManipulation.convert_yuv420p(local_save_dir, output_file)
        output_file = os.path.join(local_save_dir, file_name)

        user_save_name = f"{user_save_name}.mp4" if user_save_name else f"{uuid.uuid4()}.mp4"

        shutil.move(output_file, os.path.join(user_save_dir, user_save_name))

        # remove tmp files
        if tmp_source_file != source_file and os.path.exists(source_file):
            os.remove(source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis lip sync finished.", user=user_name, level="info")

        # Update config about models if process was successfully
        if Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC:
            lprint(msg="Update checker of download size lip sync models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIP_SYNC": False})
            Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC = False
        if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN and device != "cpu":
            lprint(msg="Update checker of download size gfpgan models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": False})
            Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = False

        # Updating progress
        if progress_callback:
            progress_callback(100, "Finished.")

        output_path = os.path.join(user_save_dir, user_save_name)
        output_media = LipSync.get_media_url(output_path)

        return {"file": os.path.join(user_save_dir, user_save_name), "media": output_media}


def memory_message(method: str, msg: str, user: str, level: str, device: str="cuda:0"):
    _, available_vram_gb, busy_vram_gb = get_vram_gb(device=device)
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)
