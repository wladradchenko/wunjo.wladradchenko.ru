import os
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
deepfake_root_path = os.path.join(root_path, "deepfake")
sys.path.insert(0, deepfake_root_path)

"""Video and image"""
from src.utils.videoio import VideoManipulation
from src.utils.imageio import save_image_cv2, read_image_cv2, read_image_pil
from src.utils.audio import cut_audio
"""Video and image"""
"""Wav2Lip"""
from src.wav2mel import MelProcessor
from src.utils.face_enhancer import enhancer as content_enhancer
from src.utils.sync_lip import GenerateWave2Lip
"""Wav2Lip"""
"""Face Swap"""
from src.utils.faceswap import FaceSwapDeepfake
"""Face Swap"""
"""Retouch"""
from src.retouch import InpaintModel, process_retouch, pil_to_cv2, convert_cv2_to_pil, convert_colored_mask_thickness_cv2
"""Retouch"""
"""Segmentation"""
from src.utils.segment import SegmentAnything
from src.east.detect import SegmentText
"""Segmentation"""

sys.path.pop(0)

from backend.folders import ALL_MODEL_FOLDER, TMP_FOLDER, MEDIA_FOLDER
from backend.download import download_model, unzip, check_download_size, get_nested_url, is_connected
from backend.config import ModelsConfig, Settings
from backend.general_utils import lprint, get_vram_gb, get_ram_gb, ensure_memory_capacity


file_models_config = ModelsConfig()


class SAMGetSegment:
    @staticmethod
    def load_model():
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
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

        onnx_vit_checkpoint = os.path.join(checkpoint_dir_full, 'vit_b_quantized.onnx')
        link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_b_quantized.onnx"])
        if not os.path.exists(onnx_vit_checkpoint):
            # check what is internet access
            if is_connected(onnx_vit_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                # download pre-trained models from url
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_CPU_TASK_GB, device=device, retries=10):
            lprint(msg="Memory out of necessary for SAM segment load model task.", level="error")
            raise "Memory out of necessary for SAM segment load model task."

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
    def extract_embedding(image: bytes):
        predictor = SAMGetSegment.load_model()
        # segmentation
        segmentation = SegmentAnything()
        image = segmentation.preprocess({"image": image}, device="cpu")
        # Run the inference.
        embedding = predictor.image_encoder(image)
        embedding = embedding.cpu().detach().numpy()
        embedding_decoded = segmentation.postprocess(embedding)
        del segmentation, predictor
        torch.cuda.empty_cache()
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


class Analyser:
    """Analysing content"""
    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

    @staticmethod
    def analysis(user_name: str, source_file: str):
        method = "Analysis"
        lprint(msg=f"[{method}] Start content analysing.", user=user_name, level="info")
        is_original = VideoManipulation.decrypted(source_file)
        if is_original:
            lprint(msg=f"[{method}] Content is original.", user=user_name, level="info")
        else:
            lprint(msg=f"[{method}] Content is Wunjo fake.", user=user_name, level="info")
        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        return {"is_original": is_original}


class RemoveBackground:
    """Remove background for object in image or video"""

    @staticmethod
    def is_model_sam():
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        vit_quantized = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        vit_quantized_media = vit_quantized.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
        return vit_quantized_media

    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

    @staticmethod
    def is_valid_point_list(points: list, offset_width: int, offset_height: int):
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
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def inspect() -> dict:
        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        models_information = {
            "Remove background":
                {
                    "SAM CPU": [], "SAM GPU": [],
                }
        }

        # SAM GPU
        sam_vit_gpu_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        link_sam_vit_gpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
        if not os.path.exists(sam_vit_gpu_checkpoint):
            sam_vit_gpu_information = {"path": sam_vit_gpu_checkpoint, "url": link_sam_vit_gpu_checkpoint, "is_exist": False}
        else:
            sam_vit_gpu_information = {"path": sam_vit_gpu_checkpoint, "url": link_sam_vit_gpu_checkpoint, "is_exist": True}
        models_information["Remove background"]["SAM GPU"].append(sam_vit_gpu_information)

        onnx_vit_gpu_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_gpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        if not os.path.exists(onnx_vit_gpu_checkpoint):
            onnx_vit_gpu_information = {"path": onnx_vit_gpu_checkpoint, "url": link_onnx_vit_gpu_checkpoint, "is_exist": False}
        else:
            onnx_vit_gpu_information = {"path": onnx_vit_gpu_checkpoint, "url": link_onnx_vit_gpu_checkpoint, "is_exist": True}
        models_information["Remove background"]["SAM GPU"].append(onnx_vit_gpu_information)

        # SAM CPU
        sam_vit_cpu_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        link_sam_vit_cpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_b.pth"])
        if not os.path.exists(sam_vit_cpu_checkpoint):
            sam_vit_cpu_information = {"path": sam_vit_cpu_checkpoint, "url": link_sam_vit_cpu_checkpoint, "is_exist": False}
        else:
            sam_vit_cpu_information = {"path": sam_vit_cpu_checkpoint, "url": link_sam_vit_cpu_checkpoint, "is_exist": True}
        models_information["Remove background"]["SAM CPU"].append(sam_vit_cpu_information)

        onnx_vit_cpu_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_cpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        if not os.path.exists(onnx_vit_cpu_checkpoint):
            onnx_vit_cpu_information = {"path": onnx_vit_cpu_checkpoint, "url": link_onnx_vit_cpu_checkpoint, "is_exist": False}
        else:
            onnx_vit_cpu_information = {"path": onnx_vit_cpu_checkpoint, "url": link_onnx_vit_cpu_checkpoint, "is_exist": True}
        models_information["Remove background"]["SAM CPU"].append(onnx_vit_cpu_information)

        return models_information

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, masks: dict, user_save_name: str = None,
                          predictor=None, session=None, segment_percentage: int = 25, inverse_mask: bool = False,
                          source_start: float = 0, source_end: float = 0, max_size: int = 1280, mask_color: str = "#ffffff"):
        min_delay = 20
        max_delay = 180
        try:
            RemoveBackground.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                       masks=masks, user_save_name=user_save_name, predictor=predictor, session=session,
                                       segment_percentage=segment_percentage, source_start=source_start, inverse_mask=inverse_mask,
                                       source_end=source_end, max_size=max_size, mask_color=mask_color)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=user_name)
                torch.cuda.empty_cache()
                sleep(random.randint(min_delay, max_delay))
                torch.cuda.empty_cache()
                RemoveBackground.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                           masks=masks, user_save_name=user_save_name, predictor=predictor,
                                           session=session, inverse_mask=inverse_mask,
                                           segment_percentage=segment_percentage, source_start=source_start,
                                           source_end=source_end, max_size=max_size, mask_color=mask_color)
            else:
                raise err

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, masks: dict, user_save_name: str = None,
                  predictor=None, session=None, segment_percentage: int = 25, inverse_mask: bool = False,
                  source_start: float = 0, source_end: float = 0, max_size: int = 1280, mask_color: str = "#ffffff"):

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
        source_media_type = VideoManipulation.check_media_type(source_file)

        # params
        source_start = float(source_start) if source_start is not None and RemoveBackground.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and RemoveBackground.is_float(source_end) else 0

        # cut video in os
        if source_media_type == "animated":
            source_file = VideoManipulation.cut_video_ffmpeg(source_file, source_start, source_end)

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"

        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        msg = f"Before load models in memory"
        memory_message(method, msg, user_name, "info")

        _, available_vram_gb, busy_vram_gb = get_vram_gb()
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        # load model segmentation
        if device == "cuda" and available_vram_gb > 7:  # min 7 Gb VRAM
            vit_model_type = "vit_h"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
            link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
            if not os.path.exists(sam_vit_checkpoint):
                # check what is internet access
                if is_connected(sam_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = True
                    # download pre-trained models from url
                    download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU:
                    check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
            link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
            if not os.path.exists(onnx_vit_checkpoint):
                # check what is internet access
                if is_connected(onnx_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = True
                    # download pre-trained models from url
                    download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU:
                    check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
        else:
            vit_model_type = "vit_b"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
            link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_b.pth"])
            if not os.path.exists(sam_vit_checkpoint):
                # check what is internet access
                if is_connected(sam_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                    # download pre-trained models from url
                    download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                    check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
            link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_b_quantized.onnx"])
            if not os.path.exists(onnx_vit_checkpoint):
                # check what is internet access
                if is_connected(onnx_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                    # download pre-trained models from url
                    download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                    check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        # segment
        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_BACKGROUND_GB, limit_vram=Settings.MIN_VRAM_REMOVE_BACKGROUND_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load segmentation model.", level="error", user=user_name)
            raise f"Memory out of necessary for {method}."

        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage, inverse_mask)
        if session is None:
            session = segmentation.init_onnx(onnx_vit_checkpoint, device)
            lprint(msg=f"[{method}] Model session is loaded.", user=user_name, level="info")
        if predictor is None:
            predictor = segmentation.init_vit(sam_vit_checkpoint, vit_model_type, device)
            lprint(msg=f"[{method}] Model predictor is loaded.", user=user_name, level="info")

        msg = f"Load SAM models {device}"
        memory_message(method, msg, user_name, "info")

        if source_media_type == "static":
            fps = 0
            # Save frame to frame_dir
            frame_files = ["static_frame.png"]
            cv2.imwrite(os.path.join(frame_dir, frame_files[0]), read_image_cv2(source_file))
        elif source_media_type == "animated":
            fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
            frame_files = sorted(os.listdir(frame_dir))
        else:
            lprint(msg=f"[{method}] Source is not detected as image or video.", user=user_name, level="error")
            raise "Source is not detected as image or video"

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
        for frame_file in frame_files:
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

        # read again after resized
        work_dir = resized_dir
        first_work_frame = read_image_cv2(os.path.join(work_dir, frame_files[0]))
        work_height, work_width, _ = first_work_frame.shape

        # get segmentation frames as maks
        segmentation.load_models(predictor=predictor, session=session)

        # change parameter
        mask_color = segmentation.check_valid_hex(mask_color)
        if source_media_type == "animated" and mask_color == "transparent":  # to create chromo key video
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

            if len(filter_frames_files) > 1:
                for filter_frame_file_name in filter_frames_files[1:]:
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
            # set new key
            masks[key]["mask_path"] = os.path.join(local_save_dir, key)
            # close progress bar for key
            progress_bar.close()
            msg = f"After processing SAM for {key}"
            memory_message(method, msg, user_name, "info")

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
            elif len(list_files) == 1:
                # move file to user_save_dir
                user_save_name =  f"{user_save_name}{i}.jpg" if user_save_name is not None else str(uuid.uuid4())
                source_path = os.path.join(frame_files_path, list_files[0])
                destination_path = os.path.join(user_save_dir, user_save_name)
                shutil.move(source_path, destination_path)

        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if tmp_source_file != source_file and os.path.exists(tmp_source_file):
            os.remove(tmp_source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        msg = f"Synthesis finished"
        memory_message(method, msg, user_name, "info")

        # Update config about models if process was successfully
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU and vit_model_type == "vit_b":
            lprint(msg="Update checker of download size SAM CPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = False
        if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU and vit_model_type == "vit_h":
            lprint(msg="Update checker of download size SAM GPU models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": False})
            Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = False

        return {"files": os.path.join(user_save_dir, user_save_name)}


class RemoveObject:
    """Remove object in image or video"""

    @staticmethod
    def is_model_sam():
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        vit_quantized = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        vit_quantized_media = vit_quantized.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
        return vit_quantized_media

    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

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
    def is_valid_text_area(fields: list, offset_width: int, offset_height: int) -> list:
        text_area = []
        for i, field in enumerate(fields):
            if field is None:
                continue
            text_area.append({
                "x": int(field.get("x", 0)),
                "y": int(field.get("y", 0)),
                "width": int(field.get("width", 0)),
                "height": int(field.get("height", 0)),
                "offsetHeight": int(offset_height),
                "offsetWidth": int(offset_width)
            })
        return text_area

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def resize_text_area(field, natural_width, natural_height):
        x1 = int(field.get("x", 0))
        y1 = int(field.get("y", 0))
        width = int(field.get("width", 0))
        height = int(field.get("height", 0))
        offset_height = int(field.get("offsetHeight", 0))
        offset_width = int(field.get("offsetWidth", 0))

        # Calculate the scaling factors
        scale_x = natural_width / offset_width
        scale_y = natural_height / offset_height

        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(new_x1 + width * scale_x)
        new_y2 = int(new_y1 + height * scale_y)

        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def inspect() -> dict:
        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        models_information = {
            "Remove object":
                {
                    "Simple remove model": [], "SAM CPU": [], "SAM GPU": [], "Text recognition": []
                }
        }

        # Simple
        model_retouch_path = os.path.join(checkpoint_folder, "retouch_object.pth")
        link_model_retouch = get_nested_url(file_models_config(), ["checkpoints", "retouch_object.pth"])
        if not os.path.exists(model_retouch_path):
            model_retouch_information = {"path": model_retouch_path, "url": link_model_retouch, "is_exist": False}
        else:
            model_retouch_information = {"path": model_retouch_path, "url": link_model_retouch, "is_exist": True}
        models_information["Remove object"]["Simple remove model"].append(model_retouch_information)

        # SAM GPU
        sam_vit_gpu_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        link_sam_vit_gpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
        if not os.path.exists(sam_vit_gpu_checkpoint):
            sam_vit_gpu_information = {"path": sam_vit_gpu_checkpoint, "url": link_sam_vit_gpu_checkpoint, "is_exist": False}
        else:
            sam_vit_gpu_information = {"path": sam_vit_gpu_checkpoint, "url": link_sam_vit_gpu_checkpoint, "is_exist": True}
        models_information["Remove object"]["SAM GPU"].append(sam_vit_gpu_information)

        onnx_vit_gpu_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_gpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        if not os.path.exists(onnx_vit_gpu_checkpoint):
            onnx_vit_gpu_information = {"path": onnx_vit_gpu_checkpoint, "url": link_onnx_vit_gpu_checkpoint, "is_exist": False}
        else:
            onnx_vit_gpu_information = {"path": onnx_vit_gpu_checkpoint, "url": link_onnx_vit_gpu_checkpoint, "is_exist": True}
        models_information["Remove object"]["SAM GPU"].append(onnx_vit_gpu_information)

        # SAM CPU
        sam_vit_cpu_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        link_sam_vit_cpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_b.pth"])
        if not os.path.exists(sam_vit_cpu_checkpoint):
            sam_vit_cpu_information = {"path": sam_vit_cpu_checkpoint, "url": link_sam_vit_cpu_checkpoint, "is_exist": False}
        else:
            sam_vit_cpu_information = {"path": sam_vit_cpu_checkpoint, "url": link_sam_vit_cpu_checkpoint, "is_exist": True}
        models_information["Remove object"]["SAM CPU"].append(sam_vit_cpu_information)

        onnx_vit_cpu_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_cpu_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
        if not os.path.exists(onnx_vit_cpu_checkpoint):
            onnx_vit_cpu_information = {"path": onnx_vit_cpu_checkpoint, "url": link_onnx_vit_cpu_checkpoint, "is_exist": False}
        else:
            onnx_vit_cpu_information = {"path": onnx_vit_cpu_checkpoint, "url": link_onnx_vit_cpu_checkpoint, "is_exist": True}
        models_information["Remove object"]["SAM CPU"].append(onnx_vit_cpu_information)

        # Text recognition
        # Get models for text detection
        vgg16_baseline_path = os.path.join(checkpoint_folder, "vgg16_baseline.pth")
        link_vgg16_baseline = get_nested_url(file_models_config(), ["checkpoints", "vgg16_baseline.pth"])
        if not os.path.exists(vgg16_baseline_path):
            vgg16_baseline_information = {"path": vgg16_baseline_path, "url": link_vgg16_baseline, "is_exist": False}
        else:
            vgg16_baseline_information = {"path": vgg16_baseline_path, "url": link_vgg16_baseline, "is_exist": True}
        models_information["Remove object"]["Text recognition"].append(vgg16_baseline_information)

        vgg16_east_path = os.path.join(checkpoint_folder, "vgg16_east.pth")
        link_vgg16_east = get_nested_url(file_models_config(), ["checkpoints", "vgg16_east.pth"])
        if not os.path.exists(vgg16_east_path):
            vgg16_east_information = {"path": vgg16_east_path, "url": link_vgg16_east, "is_exist": False}
        else:
            vgg16_east_information = {"path": vgg16_east_path, "url": link_vgg16_east, "is_exist": True}
        models_information["Remove object"]["Text recognition"].append(vgg16_east_information)

        return models_information

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, masks: dict = None, user_save_name: str = None,
                          predictor=None, session=None, segment_percentage: int = 25, use_improved_model: bool = False,
                          source_start: float = 0, source_end: float = 0, max_size: int = 1280, text_area: list = None,
                          use_remove_text: bool = False, inverse_mask: bool = False):
        min_delay = 20
        max_delay = 180
        try:
            RemoveObject.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                   masks=masks, user_save_name=user_save_name, predictor=predictor, inverse_mask=inverse_mask,
                                   session=session, segment_percentage=segment_percentage, source_start=source_start,
                                   source_end=source_end, max_size=max_size, use_improved_model=use_improved_model,
                                   text_area=text_area, use_remove_text=use_remove_text)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=user_name)
                torch.cuda.empty_cache()
                sleep(random.randint(min_delay, max_delay))
                torch.cuda.empty_cache()
                RemoveObject.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                       masks=masks, user_save_name=user_save_name, predictor=predictor, inverse_mask=inverse_mask,
                                       session=session, segment_percentage=segment_percentage, source_start=source_start,
                                       source_end=source_end, max_size=max_size, use_improved_model=use_improved_model,
                                       text_area=text_area, use_remove_text=use_remove_text)
            else:
                raise err

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, masks: dict = None, user_save_name: str = None,
                  predictor=None, session=None, segment_percentage: int = 25, use_improved_model: bool = False,
                  source_start: float = 0, source_end: float = 0, max_size: int = 1280, text_area: list = None,
                  use_remove_text: bool = False, inverse_mask: bool = False):

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
        source_media_type = VideoManipulation.check_media_type(source_file)

        # params
        source_start = float(source_start) if source_start is not None and RemoveObject.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and RemoveObject.is_float(source_end) else 0

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
            use_half = True
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"
            use_half = False

        _, available_vram_gb, busy_vram_gb = get_vram_gb()
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        msg = f"Before load model in {device}"
        memory_message(method, msg, user_name, "info")

        # download models
        # get models path
        checkpoint_folder = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        model_retouch_path = os.path.join(checkpoint_folder, "retouch_object.pth")
        link_model_retouch = get_nested_url(file_models_config(), ["checkpoints", "retouch_object.pth"])
        if not os.path.exists(model_retouch_path):
            # check what is internet access
            if is_connected(model_retouch_path):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT": True})
                Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT = True
                # download pre-trained models from url
                download_model(model_retouch_path, link_model_retouch)
            else:
                lprint(msg=f"Not internet connection to download model {link_model_retouch} in {model_retouch_path}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT:
                check_download_size(model_retouch_path, link_model_retouch)

        if device == "cuda":
            gpu_vram = available_vram_gb
        else:
            gpu_vram = 0

        # load model segmentation
        if device == "cuda" and gpu_vram > 7:  # min 7 Gb VRAM
            vit_model_type = "vit_h"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
            link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_h.pth"])
            if not os.path.exists(sam_vit_checkpoint):
                # check what is internet access
                if is_connected(sam_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = True
                    # download pre-trained models from url
                    download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU:
                    check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
            link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_h_quantized.onnx"])
            if not os.path.exists(onnx_vit_checkpoint):
                # check what is internet access
                if is_connected(onnx_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_GPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU = True
                    # download pre-trained models from url
                    download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_GPU:
                    check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
        else:
            vit_model_type = "vit_b"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
            link_sam_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "sam_vit_b.pth"])
            if not os.path.exists(sam_vit_checkpoint):
                # check what is internet access
                if is_connected(sam_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                    # download pre-trained models from url
                    download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_sam_vit_checkpoint} in {sam_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                    check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
            link_onnx_vit_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "vit_b_quantized.onnx"])
            if not os.path.exists(onnx_vit_checkpoint):
                # check what is internet access
                if is_connected(onnx_vit_checkpoint):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SAM_CPU": True})
                    Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU = True
                    # download pre-trained models from url
                    download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {link_onnx_vit_checkpoint} in {onnx_vit_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_SAM_CPU:
                    check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        if use_remove_text:
            # Get models for text detection
            vgg16_baseline_path = os.path.join(checkpoint_folder, "vgg16_baseline.pth")
            link_vgg16_baseline = get_nested_url(file_models_config(), ["checkpoints", "vgg16_baseline.pth"])
            if not os.path.exists(vgg16_baseline_path):
                # check what is internet access
                if is_connected(vgg16_baseline_path):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT": True})
                    Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT = True
                    # download pre-trained models from url
                    download_model(vgg16_baseline_path, link_vgg16_baseline)
                else:
                    lprint(msg=f"Not internet connection to download model {link_vgg16_baseline} in {vgg16_baseline_path}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT:
                    check_download_size(vgg16_baseline_path, link_vgg16_baseline)

            vgg16_east_path = os.path.join(checkpoint_folder, "vgg16_east.pth")
            link_vgg16_east = get_nested_url(file_models_config(), ["checkpoints", "vgg16_east.pth"])
            if not os.path.exists(vgg16_east_path):
                # check what is internet access
                if is_connected(vgg16_east_path):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT": True})
                    Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT = True
                    # download pre-trained models from url
                    download_model(vgg16_east_path, link_vgg16_east)
                else:
                    lprint(msg=f"Not internet connection to download model {link_vgg16_east} in {vgg16_east_path}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT:
                    check_download_size(vgg16_east_path, link_vgg16_east)
        else:
            vgg16_baseline_path = None
            vgg16_east_path = None

        # segment
        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage, inverse_mask)
        if session is None and not use_remove_text:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load segmentation session model.", level="error",  user=user_name)
                raise f"Memory out of necessary for {method}."
            session = segmentation.init_onnx(onnx_vit_checkpoint, device)
            msg = f"Model session loaded"
            memory_message(method, msg, user_name, "info")
        else:
            lprint(msg=f"[{method}] Model session will not be loaded because of not need to use for text.", user=user_name, level="info")
        if predictor is None and not use_remove_text:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load segmentation predictor model.", level="error", user=user_name)
                raise f"Memory out of necessary for {method}."
            predictor = segmentation.init_vit(sam_vit_checkpoint, vit_model_type, device)
            msg = f"Model predictor loaded"
            memory_message(method, msg, user_name, "info")
        else:
            lprint(msg=f"[{method}] Model predictor will not be loaded because of not need to use for text.", user=user_name, level="info")

        if source_media_type == "static":
            fps = 0
            # Save frame to frame_dir
            frame_files = ["static_frame.png"]
            cv2.imwrite(os.path.join(frame_dir, frame_files[0]), read_image_cv2(source_file))
        elif source_media_type == "animated":
            fps, frame_dir = VideoManipulation.save_frames(video=source_file, output_dir=frame_dir, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
            frame_files = sorted(os.listdir(frame_dir))
        else:
            raise "Source is not detected as image or video"

        # if user not set init time for object, set this
        if not use_remove_text:
            for key in masks.keys():
                if masks[key].get("start_time") is None:
                    masks[key]["start_time"] = source_start
                if masks[key].get("end_time") is None:
                    masks[key]["end_time"] = source_end

        first_frame = read_image_cv2(os.path.join(frame_dir, frame_files[0]))
        orig_height, orig_width, _ = first_frame.shape

        # Resize frames while maintaining aspect ratio
        for frame_file in frame_files:
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
        work_dir = resized_dir
        lprint(msg=f"[{method}] Files resized", user=user_name, level="info")

        # read again after resized
        first_work_frame = read_image_cv2(os.path.join(work_dir, frame_files[0]))
        work_height, work_width, _ = first_work_frame.shape

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

                if len(filter_frames_files) > 1:
                    for filter_frame_file_name in filter_frames_files[1:]:
                        filter_frame = cv2.imread(os.path.join(work_dir, filter_frame_file_name))  # read frame file after filter
                        segment_mask = segmentation.draw_mask_frames(frame=filter_frame)
                        if segment_mask is None:
                            lprint(msg=f"[{method}] Encountered None mask for {key}. Breaking the loop.", user=user_name, level="info")
                            break
                        # save frames after 0002
                        segmentation.save_black_mask(filter_frame_file_name, segment_mask, mask_key_save_path, kernel_size=thickness_mask, width=work_width, height=work_height)

                        progress_bar.update(1)
                # set new key
                masks[key]["mask_path"] = mask_key_save_path
                # close progress bar for key
                progress_bar.close()

                msg = f"After processing SAM for {key}"
                memory_message(method, msg, user_name, "info")

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
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load find text model.", level="error", user=user_name)
                raise f"Memory out of necessary for {method}."
            segment_text = SegmentText(device=device, vgg16_path=vgg16_baseline_path, east_path=vgg16_east_path)
            text_area = [None] if text_area is None else text_area

            # set progress bar
            progress_bar = tqdm(total=len(filter_frames_files) * len(text_area), unit='it', unit_scale=True)
            for area in text_area:
                for frame_file in filter_frames_files:
                    frame_file_path = os.path.join(frame_dir, frame_file)
                    mask_text_frame = segment_text.detect_text(img_path=frame_file_path)
                    if area is not None and isinstance(area, dict):
                        w, h = mask_text_frame.size
                        x1, y1, x2, y2 = RemoveObject.resize_text_area(area, w, h)
                        coord = [x1, y1, x2, y2]
                    else:
                        coord = None
                    mask_textarea_frame = segment_text.crop(mask_text_frame, coord)
                    mask_text_frame_path = os.path.join(mask_text_save_path, frame_file)
                    mask_textarea_frame.save(mask_text_frame_path)
                    progress_bar.update(1)
            # close progress bar for text mask
            progress_bar.close()

            del segment_text
            # Set mask path in masks
            masks["text"] = {'mask_path': mask_text_save_path, 'start_time': start_time, 'end_time': end_time}

            msg = f"After processing text recognition"
            memory_message(method, msg, user_name, "info")

        del segmentation, predictor, session
        torch.cuda.empty_cache()

        msg = f"After SAM model"
        memory_message(method, msg, user_name, "info")

        if model_retouch_path is None:
            lprint(msg=f"[{method}] Path for remove object model is None.", user=user_name, level="error")
            return

        # retouch
        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_REMOVE_OBJECT_GB, limit_vram=Settings.MIN_VRAM_REMOVE_OBJECT_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load simple remove object model.", level="error", user=user_name)
            raise f"Memory out of necessary for {method}."
        model_retouch = InpaintModel(model_path=model_retouch_path)

        for key in masks.keys():
            mask_files = sorted(os.listdir(masks[key]["mask_path"]))
            # set progress bar
            progress_bar = tqdm(total=len(mask_files), unit='it', unit_scale=True)
            for file_name in mask_files:
                # read mask to pillow
                segment_mask = cv2.imread(os.path.join(tmp_dir, f"mask_{key}", file_name), cv2.IMREAD_GRAYSCALE)
                binary_mask = cv2.threshold(segment_mask, 128, 1, cv2.THRESH_BINARY)[1]
                bool_mask = binary_mask.astype(bool)
                reshaped_mask = bool_mask[np.newaxis, np.newaxis, ...]
                segment_mask_pil = convert_colored_mask_thickness_cv2(reshaped_mask)
                # read frame to pillow
                current_frame = cv2.imread(os.path.join(work_dir, file_name))
                frame_pil = convert_cv2_to_pil(current_frame)
                # retouch_frame
                retouch_frame = process_retouch(img=frame_pil, mask=segment_mask_pil, model=model_retouch)
                # update frame
                cv2.imwrite(os.path.join(work_dir, file_name), pil_to_cv2(retouch_frame))
                progress_bar.update(1)
            # close progress bar for key
            progress_bar.close()

            msg = f"Simple remove object {key}"
            memory_message(method, msg, user_name, "info")

        # empty cache
        del model_retouch
        torch.cuda.empty_cache()

        msg = f"After finished remove model clear memory"
        memory_message(method, msg, user_name, "info")

        if source_media_type == "animated":
            # get saved file as merge frames to video
            video_name = VideoManipulation.save_video_from_frames(frame_names="frame%04d.png", save_path=work_dir, fps=fps, alternative_save_path=local_save_dir)
            # get audio from video target
            audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
            # combine audio and video
            save_name = VideoManipulation.save_video_with_audio(os.path.join(local_save_dir, video_name), os.path.join(local_save_dir, str(audio_file_name)), local_save_dir)
            # transfer video to user path
            user_save_name = f"{user_save_name}.mp4" if user_save_name is not None else save_name
            shutil.move(os.path.join(local_save_dir, save_name), os.path.join(user_save_dir, user_save_name))
        else:
            save_name = frame_files[0]
            user_save_name = f"{user_save_name}.jpg" if user_save_name is not None else save_name
            retouched_frame = read_image_cv2(os.path.join(work_dir, save_name))
            cv2.imwrite(os.path.join(user_save_dir, user_save_name), retouched_frame)

        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if tmp_source_file != source_file and os.path.exists(tmp_source_file):
            os.remove(tmp_source_file)
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
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT:
                lprint(msg="Update checker of download size simple remove object models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT": False})
                Settings.CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT = False

        return {"files": os.path.join(user_save_dir, user_save_name)}


class FaceSwap:
    """
    Face swap by one photo
    """

    @staticmethod
    def is_valid_source_area(areas: list, offset_width: int, offset_height: int) -> list:
        """
        Set valid format
        Source area is list of dict and has x, y, width, height and start_time and end_time.
        Time need to use in face swap with a few faces boxes with difference time
        """
        source_area = []
        for i, a in enumerate(areas):
            if a is None:
                continue
            source_area.append({
                "x": int(a.get("x", 0) + int(a.get("width", 0)) / 2),
                "y": int(a.get("y", 0) + int(a.get("height", 0)) / 2),
                "canvasHeight": int(offset_height),
                "canvasWidth": int(offset_width),
                "startTime": a.get("startTime", None),
                "endTime": a.get("endTime", None)
            })
        return source_area

    @staticmethod
    def is_valid_avatar_area(areas: list, offset_width: int, offset_height: int) -> list:
        """
            Set valid format
            Avatar area is list of dict and has x, y, width, height
            """
        avatar_area = []
        for i, a in enumerate(areas):
            if a is None:
                continue

            avatar_area.append({
                "x": int(a.get("x", 0) + int(a.get("width", 0)) / 2),
                "y": int(a.get("y", 0) + int(a.get("height", 0)) / 2),
                "canvasHeight": int(offset_height),
                "canvasWidth": int(offset_width)
            })
        return avatar_area

    @staticmethod
    def is_valid_avatar_image(file):
        return VideoManipulation.check_media_type(file) == "static"

    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_resize_factor(width, height, max_size: int = 1280):
        """If need to set max size"""
        resize_factor = max_size / max(width, height) if max_size is not None else 1
        return resize_factor if resize_factor < 1 else 1

    @staticmethod
    def inspect() -> dict:
        models_information = {"Face swap": {"Swapper": [], "Face alignment": []}}

        model_dir_full = os.path.join(ALL_MODEL_FOLDER, "models")
        face_recognition = os.path.join(model_dir_full, 'buffalo_l')
        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
        name = 'buffalo_l'
        link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
        if not os.path.exists(face_recognition):
            face_recognition_information = {"path": face_recognition, "url": link_face_recognition, "is_exist": False}
        else:
            face_recognition_information = {"path": face_recognition, "url": link_face_recognition, "is_exist": True}
        models_information["Face swap"]["Face alignment"].append(face_recognition_information)

        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
        link_faceswap_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "faceswap.onnx"])
        if not os.path.exists(faceswap_checkpoint):
            faceswap_information = {"path": faceswap_checkpoint, "url": link_faceswap_checkpoint, "is_exist": False}
        else:
            faceswap_information = {"path": faceswap_checkpoint, "url": link_faceswap_checkpoint, "is_exist": True}
        models_information["Face swap"]["Swapper"].append(faceswap_information)

        return models_information

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, avatar_file: str, user_save_name: str = None,
                          source_crop_area: list = None, avatar_crop_area: list = None,
                          source_start: float = 0, source_end: float = 0, resize_factor: float = 1, is_gemini_faces: bool = False,
                          replace_all_faces: bool = False, face_discrimination_factor: float = 1.5):
        min_delay = 20
        max_delay = 180
        try:
            FaceSwap.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                               avatar_file=avatar_file, user_save_name=user_save_name, source_crop_area=source_crop_area,
                               avatar_crop_area=avatar_crop_area, resize_factor=resize_factor, source_start=source_start,
                               source_end=source_end, is_gemini_faces=is_gemini_faces, replace_all_faces=replace_all_faces,
                               face_discrimination_factor=face_discrimination_factor)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=user_name)
                torch.cuda.empty_cache()
                sleep(random.randint(min_delay, max_delay))
                torch.cuda.empty_cache()
                FaceSwap.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                   avatar_file=avatar_file, user_save_name=user_save_name,
                                   source_crop_area=source_crop_area,
                                   avatar_crop_area=avatar_crop_area, resize_factor=resize_factor,
                                   source_start=source_start,
                                   source_end=source_end, is_gemini_faces=is_gemini_faces,
                                   replace_all_faces=replace_all_faces,
                                   face_discrimination_factor=face_discrimination_factor)
            else:
                raise err

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, avatar_file: str, user_save_name: str = None,
                  source_crop_area: list = None, avatar_crop_area: list = None,
                  source_start: float = 0, source_end: float = 0, resize_factor: float = 1, is_gemini_faces: bool = False,
                  replace_all_faces: bool = False, face_discrimination_factor: float = 1.5):

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
        source_media_type = VideoManipulation.check_media_type(source_file)

        # params
        saved_file = None
        source_start = float(source_start) if source_start is not None and FaceSwap.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and FaceSwap.is_float(source_end) else 0

        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"

        _, available_vram_gb, busy_vram_gb = get_vram_gb()
        _, available_ram_gb, busy_ram_gb = get_ram_gb()

        msg = f"Before loading model in memory"
        memory_message(method, msg, user_name, "info")

        # Load model
        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_dir_full
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
        link_faceswap_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "faceswap.onnx"])
        if not os.path.exists(faceswap_checkpoint):
            # check what is internet access
            if is_connected(faceswap_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_FACE_SWAP": True})
                Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP = True
                # download pre-trained models from url
                download_model(faceswap_checkpoint, link_faceswap_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {link_faceswap_checkpoint} in {faceswap_checkpoint}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP:
                check_download_size(faceswap_checkpoint, link_faceswap_checkpoint)

        # get fps and calculate current frame and get that frame for source
        avatar_frame = VideoManipulation.get_first_frame(avatar_file, 0)

        # Init face swap
        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_FACE_SWAP_GB, limit_vram=Settings.MIN_VRAM_FACE_SWAP_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load face swap model.", level="error", user=user_name)
            raise f"Memory out of necessary for {method}."

        faceswap = FaceSwapDeepfake(ALL_MODEL_FOLDER, faceswap_checkpoint, is_gemini_faces, face_discrimination_factor, device)

        for j in range(len(source_crop_area)):  # To change a few faces
            lprint(msg=f"[{method}] Processing area {j + 1}.", user=user_name, level="info")

            current_source_crop_area = source_crop_area[j]
            current_avatar_crop_area = avatar_crop_area[j % len(avatar_crop_area)]  # Repeat elements if necessary

            source_face = faceswap.face_detect_with_alignment_from_source_frame(avatar_frame, current_avatar_crop_area)
            lprint(msg=f"[{method}] Got avatar face {j + 1}.", user=user_name, level="info")

            if source_media_type == "animated":
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

        # after generation
        if saved_file is not None:
            # empty cache
            del faceswap
            torch.cuda.empty_cache()

            msg = f"After finished and clear memory"
            memory_message(method, msg, user_name, "info")

            try:
                file_name = VideoManipulation.encrypted(saved_file, local_save_dir)  # VideoManipulation.encrypted
                saved_file = os.path.join(local_save_dir, file_name)
            except Exception as err:
                print(f"Error with VideoManipulation.encrypted {err}")

            if source_media_type == "animated":
                # get audio from video target
                audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
                # combine audio and video
                file_name = VideoManipulation.save_video_with_audio(saved_file, os.path.join(local_save_dir, str(audio_file_name)), local_save_dir)
                saved_file = os.path.join(local_save_dir, file_name)

            if source_media_type == "animated":
                user_save_name += ".mp4"
            else:
                user_save_name += ".jpg"

            shutil.move(saved_file, os.path.join(user_save_dir, user_save_name))

            # remove tmp files
            if os.path.exists(avatar_file):
                os.remove(avatar_file)
            if os.path.exists(source_file):
                os.remove(source_file)
            if tmp_source_file != source_file and os.path.exists(tmp_source_file):
                os.remove(tmp_source_file)
            if os.path.exists(local_save_dir):
                shutil.rmtree(local_save_dir)

            lprint(msg=f"[{method}] Synthesis face swap finished.", user=user_name, level="info")

            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP:
                lprint(msg="Update checker of download size face swap models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_FACE_SWAP": False})
                Settings.CHECK_DOWNLOAD_SIZE_FACE_SWAP = False

            return {"files": os.path.join(user_save_dir, user_save_name)}
        else:
            lprint(msg=f"[{method}] Synthesis face swap finished but not created save file.", user=user_name, level="error")
            return {"files": None}


class Enhancement:
    """Enhancement"""

    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

    @staticmethod
    def inspect() -> dict:
        models_information = {"Enhancement": {"Enhancement faces": [], "Enhancement simple video": [], "Enhancement draw video": []}}

        local_model_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_model_path, 'GFPGANv1.4.pth')
        gfpgan_url = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
        if not os.path.exists(gfpgan_model_path):
            gfpgan_information = {"path": local_model_path, "url": gfpgan_url, "is_exist": False}
        else:
            gfpgan_information = {"path": local_model_path, "url": gfpgan_url, "is_exist": True}
        models_information["Enhancement"]["Enhancement faces"].append(gfpgan_information)

        animesgan_model_path = os.path.join(local_model_path, 'realesr-animevideov3.pth')
        animesgan_url = get_nested_url(file_models_config(), ["gfpgan", "realesr-animevideov3.pth"])
        if not os.path.exists(animesgan_model_path):
            animesgan_information = {"path": animesgan_model_path, "url": animesgan_url, "is_exist": False}
        else:
            animesgan_information = {"path": animesgan_model_path, "url": animesgan_url, "is_exist": True}
        models_information["Enhancement"]["Enhancement draw video"].append(animesgan_information)

        general_model_path = os.path.join(local_model_path, 'realesr-general-x4v3.pth')
        general_url = get_nested_url(file_models_config(), ["gfpgan", "realesr-general-x4v3.pth"])
        if not os.path.exists(general_model_path):
            general_information = {"path": general_model_path, "url": general_url, "is_exist": False}
        else:
            general_information = {"path": general_model_path, "url": general_url, "is_exist": True}
        models_information["Enhancement"]["Enhancement simple video"].append(general_information)

        wdn_model_path = os.path.join(local_model_path, 'realesr-general-wdn-x4v3.pth')
        wdn_url = get_nested_url(local_model_path, ["gfpgan", "realesr-general-wdn-x4v3.pth"])
        if not os.path.exists(wdn_model_path):
            wdn_information = {"path": wdn_model_path, "url": wdn_url, "is_exist": False}
        else:
            wdn_information = {"path": wdn_model_path, "url": wdn_url, "is_exist": True}
        models_information["Enhancement"]["Enhancement simple video"].append(wdn_information)

        return models_information

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None, enhancer: str="gfpgan"):
        min_delay = 20
        max_delay = 180
        try:
            Enhancement.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                  enhancer=enhancer, user_save_name=user_save_name)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=user_name)
                torch.cuda.empty_cache()
                sleep(random.randint(min_delay, max_delay))
                torch.cuda.empty_cache()
                Enhancement.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                      enhancer=enhancer, user_save_name=user_save_name)
            else:
                raise err

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, user_save_name: str = None, enhancer: str="gfpgan"):
        # default for ce
        enhancer = "gfpgan"
        # method
        method = "Enhancement"
        # create local folder and user
        tmp_source_file = source_file
        local_save_dir = os.path.join(TMP_FOLDER, str(uuid.uuid4()))
        user_save_dir = os.path.join(folder_name, strftime("%Y_%m_%d_%H%M%S"))
        os.makedirs(local_save_dir, exist_ok=True)
        os.makedirs(user_save_dir, exist_ok=True)

        # get source type
        source_media_type = VideoManipulation.check_media_type(source_file)

        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"
            enhancer = "gfpgan"

        if enhancer not in ["gfpgan", "animesgan", "realesrgan"]:
            lprint(msg=f"[{method}] Enhancer {enhancer} not found in list.", user=user_name, level="info")
            return {"files": None}

        if enhancer == "gfpgan":
            # gfpgan
            local_gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
            gfpgan_model_path = os.path.join(local_gfpgan_path, 'GFPGANv1.4.pth')
            link_gfpgan_checkpoint = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
            if not os.path.exists(gfpgan_model_path):
                # check what is internet access
                if is_connected(gfpgan_model_path):
                    lprint(msg="Model not found. Update checker of download size model.")
                    Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": True})
                    Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = True
                    # download pre-trained models from url
                    download_model(gfpgan_model_path, link_gfpgan_checkpoint)
                else:
                    lprint(msg=f"Not internet connection to download model {gfpgan_model_path} in {link_gfpgan_checkpoint}.", user=user_name, level="error")
                    raise
            else:
                if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN:
                    check_download_size(gfpgan_model_path, link_gfpgan_checkpoint)

        if source_media_type == "animated":
            audio_file_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
            stream = cv2.VideoCapture(source_file)
            fps = stream.get(cv2.CAP_PROP_FPS)
            stream.release()
            lprint(msg=f"[{method}] Start video processing.", user=user_name, level="info")
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_ENHANCEMENT_GB, limit_vram=Settings.MIN_VRAM_ENHANCEMENT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load {enhancer} model.", level="error", user=user_name)
                raise f"Memory out of necessary for {method}."

            enhanced_name = content_enhancer(source_file, save_folder=local_save_dir, method=enhancer, fps=float(fps), device=device)
            enhanced_path = os.path.join(local_save_dir, enhanced_name)
            save_name = VideoManipulation.save_video_with_audio(enhanced_path, os.path.join(local_save_dir, audio_file_name), local_save_dir)
        else:
            # delay for load model if a few processes
            if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_ENHANCEMENT_GB, limit_vram=Settings.MIN_VRAM_ENHANCEMENT_GB, device=device, retries=50):
                lprint(msg=f"Memory out of necessary for {method} during load {enhancer} model.", level="error", user=user_name)
                raise f"Memory out of necessary for {method}."

            lprint(msg="Starting improve image", user=user_name, level="info")
            save_name = content_enhancer(source_file, save_folder=local_save_dir, method=enhancer, device=device)

        lprint(msg=f"[{method}] Clear cuda cache.", user=user_name, level="info")
        torch.cuda.empty_cache()  # this also will work if cou version

        if source_media_type == "animated":
            user_save_name = f"{user_save_name}.mp4" if user_save_name else save_name
        else:
            user_save_name += f"{user_save_name}.jpg" if user_save_name else save_name

        shutil.move(os.path.join(local_save_dir, save_name), os.path.join(user_save_dir, user_save_name))

        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if tmp_source_file != source_file and os.path.exists(tmp_source_file):
            os.remove(tmp_source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis enhancement finished.", user=user_name, level="info")

        if enhancer == "gfpgan":
            # Update config about models if process was successfully
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN and device == "cuda":
                lprint(msg="Update checker of download size gfpgan models.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": False})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = False

        return {"files": os.path.join(user_save_dir, user_save_name)}


class LipSync:
    @staticmethod
    def is_valid(file):
        return VideoManipulation.check_media_type(file) != "other"

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
        except Exception as e:
            pass
        return False

    @staticmethod
    def is_valid_source_area(areas: list, offset_width: int, offset_height: int) -> dict:
        """
        Set valid format
        Source area is dict and has x, y, width, height and start_time and end_time.
        This will work only with one face for one run
        """
        for i, a in enumerate(areas):
            if a is None:
                continue
            return {
                "x": int(a.get("x", 0) + int(a.get("width", 0)) / 2),
                "y": int(a.get("y", 0) + int(a.get("height", 0)) / 2),
                "canvasHeight": int(offset_height),
                "canvasWidth": int(offset_width),
                "startTime": a.get("startTime", None),
                "endTime": a.get("endTime", None)
            }
        return {}

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_resize_factor(width, height, max_size: int = 1280):
        """If need to set max size"""
        resize_factor = max_size / max(width, height) if max_size is not None else 1
        return resize_factor if resize_factor < 1 else 1

    @staticmethod
    def inspect() -> dict:
        models_information = {"Lip sync": {"Mouth animation": [], "Face alignment": [], "Enhancement": []}}

        model_dir_full = os.path.join(ALL_MODEL_FOLDER, "models")
        face_recognition = os.path.join(model_dir_full, 'buffalo_l')
        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
        name = 'buffalo_l'
        link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
        if not os.path.exists(face_recognition):
            face_recognition_information = {"path": face_recognition, "url": link_face_recognition, "is_exist": False}
        else:
            face_recognition_information = {"path": face_recognition, "url": link_face_recognition, "is_exist": True}
        models_information["Lip sync"]["Face alignment"].append(face_recognition_information)

        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        link_wav2lip_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "wav2lip.pth"])
        if not os.path.exists(wav2lip_checkpoint):
            wav2lip_information = {"path": wav2lip_checkpoint, "url": link_wav2lip_checkpoint, "is_exist": False}
        else:
            wav2lip_information = {"path": wav2lip_checkpoint, "url": link_wav2lip_checkpoint, "is_exist": True}
        models_information["Lip sync"]["Mouth animation"].append(wav2lip_information)

        # get models to smooth face
        retina_face_checkpoint = os.path.join(checkpoint_dir_full, 'RetinaFace-R50.pth')
        link_retina_face_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "RetinaFace-R50.pth"])
        if not os.path.exists(retina_face_checkpoint):
            retina_face_information = {"path": retina_face_checkpoint, "url": link_retina_face_checkpoint, "is_exist": False}
        else:
            retina_face_information = {"path": retina_face_checkpoint, "url": link_retina_face_checkpoint, "is_exist": True}
        models_information["Lip sync"]["Face alignment"].append(retina_face_information)

        face_parse_checkpoint = os.path.join(checkpoint_dir_full, 'ParseNet-latest.pth')
        link_face_parse_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "ParseNet-latest.pth"])
        if not os.path.exists(face_parse_checkpoint):
            face_parse_information = {"path": face_parse_checkpoint, "url": link_face_parse_checkpoint, "is_exist": False}
        else:
            face_parse_information = {"path": face_parse_checkpoint, "url": link_face_parse_checkpoint, "is_exist": True}
        models_information["Lip sync"]["Face alignment"].append(face_parse_information)

        # gfpgan
        local_gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_gfpgan_path, 'GFPGANv1.4.pth')
        link_gfpgan_checkpoint = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
        if not os.path.exists(gfpgan_model_path):
            gfpgan_information = {"path": gfpgan_model_path, "url": link_gfpgan_checkpoint, "is_exist": False}
        else:
            gfpgan_information = {"path": gfpgan_model_path, "url": link_gfpgan_checkpoint, "is_exist": True}
        models_information["Lip sync"]["Enhancement"].append(gfpgan_information)

        return models_information

    @staticmethod
    def retries_synthesis(folder_name: str, user_name: str, source_file: str, audio_file: str,
                          user_save_name: str = None, source_crop_area: dict = None, source_start: float = 0,
                          source_end: float = 0, audio_start: float = 0, audio_end: float = 0, overlay_audio: bool = False,
                          face_discrimination_factor: float = 1.5, resize_factor: float = 1):
        min_delay = 20
        max_delay = 180
        try:
            LipSync.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                              audio_file=audio_file, user_save_name=user_save_name, source_crop_area=source_crop_area,
                              source_start=source_start, source_end=source_end, audio_start=audio_start,
                              audio_end=audio_end, overlay_audio=overlay_audio,
                              face_discrimination_factor=face_discrimination_factor, resize_factor=resize_factor)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                # Handle the out-of-memory error (e.g., re-run task)
                lprint(msg=f"CUDA out of memory error. Attempting to re-run task...", level="error", user=user_name)
                torch.cuda.empty_cache()
                sleep(random.randint(min_delay, max_delay))
                torch.cuda.empty_cache()
                LipSync.synthesis(folder_name=folder_name, user_name=user_name, source_file=source_file,
                                  audio_file=audio_file, user_save_name=user_save_name,
                                  source_crop_area=source_crop_area,
                                  source_start=source_start, source_end=source_end, audio_start=audio_start,
                                  audio_end=audio_end, overlay_audio=overlay_audio,
                                  face_discrimination_factor=face_discrimination_factor,
                                  resize_factor=resize_factor)
            else:
                raise err

    @staticmethod
    def synthesis(folder_name: str, user_name: str, source_file: str, audio_file: str,
                  user_save_name: str = None, source_crop_area: dict = None, source_start: float = 0,
                  source_end: float = 0, audio_start: float = 0, audio_end: float = 0, overlay_audio: bool = False,
                  face_discrimination_factor: float = 1.5, resize_factor: float = 1):
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
        source_start = float(source_start) if source_start is not None and LipSync.is_float(source_start) else 0
        source_end = float(source_end) if source_end is not None and LipSync.is_float(source_end) else 0
        audio_start = float(audio_start) if audio_start is not None and LipSync.is_float(audio_start) else 0
        audio_end = float(audio_end) if audio_end is not None and LipSync.is_float(audio_end) else 0

        # get source type
        lprint(msg=f"[{method}] Get media type.", user=user_name, level="info")
        source_media_type = VideoManipulation.check_media_type(source_file)

        # Extract fist part of video to merge
        if source_start != 0 and source_media_type == "animated":
            lprint(msg=f"[{method}] Get first part of source video.", user=user_name, level="info")
            start_part_source_file = VideoManipulation.cut_video_ffmpeg(source_file, 0, source_start, local_save_dir, True)
        else:
            start_part_source_file = None

        # Extract full audio
        if source_media_type == "animated":
            lprint(msg=f"[{method}] Extract audio.", user=user_name, level="info")
            # get audio from video target
            source_audio_name = VideoManipulation.extract_audio_from_video(source_file, local_save_dir)
        else:
            source_audio_name = None

        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            lprint(msg=f"[{method}] Processing will run on GPU.", user=user_name, level="info")
            device = "cuda"
        else:
            lprint(msg=f"[{method}] Processing will run on CPU.", user=user_name, level="info")
            device = "cpu"

        checkpoint_dir_full = os.path.join(ALL_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_dir_full
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        link_wav2lip_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "wav2lip.pth"])
        if not os.path.exists(wav2lip_checkpoint):
            # check what is internet access
            if is_connected(wav2lip_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIP_SYNC": True})
                Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC = True
                # download pre-trained models from url
                download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {link_wav2lip_checkpoint} in {wav2lip_checkpoint}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC:
                check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)

        # get models to smooth face
        retina_face_checkpoint = os.path.join(checkpoint_dir_full, 'RetinaFace-R50.pth')
        link_retina_face_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "RetinaFace-R50.pth"])
        if not os.path.exists(retina_face_checkpoint):
            # check what is internet access
            if is_connected(retina_face_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIP_SYNC": True})
                Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC = True
                # download pre-trained models from url
                download_model(retina_face_checkpoint, link_retina_face_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {link_retina_face_checkpoint} in {retina_face_checkpoint}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC:
                check_download_size(retina_face_checkpoint, link_retina_face_checkpoint)

        face_parse_checkpoint = os.path.join(checkpoint_dir_full, 'ParseNet-latest.pth')
        link_face_parse_checkpoint = get_nested_url(file_models_config(), ["checkpoints", "ParseNet-latest.pth"])
        if not os.path.exists(face_parse_checkpoint):
            # check what is internet access
            if is_connected(face_parse_checkpoint):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIP_SYNC": True})
                Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC = True
                # download pre-trained models from url
                download_model(face_parse_checkpoint, link_face_parse_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {face_parse_checkpoint} in {link_face_parse_checkpoint}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC:
                check_download_size(face_parse_checkpoint, link_face_parse_checkpoint)

        # gfpgan
        local_gfpgan_path = os.path.join(ALL_MODEL_FOLDER, 'gfpgan', 'weights')
        gfpgan_model_path = os.path.join(local_gfpgan_path, 'GFPGANv1.4.pth')
        link_gfpgan_checkpoint = get_nested_url(file_models_config(), ["gfpgan", "GFPGANv1.4.pth"])
        if not os.path.exists(gfpgan_model_path):
            # check what is internet access
            if is_connected(gfpgan_model_path):
                lprint(msg="Model not found. Update checker of download size model.")
                Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": True})
                Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = True
                # download pre-trained models from url
                download_model(gfpgan_model_path, link_gfpgan_checkpoint)
            else:
                lprint(msg=f"Not internet connection to download model {gfpgan_model_path} in {link_gfpgan_checkpoint}.", user=user_name, level="error")
                raise
        else:
            if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN:
                check_download_size(gfpgan_model_path, link_gfpgan_checkpoint)

        # Cut audio
        if audio_start is not None and audio_end is not None:
            if audio_start == audio_end:
                lprint(msg=f"[{method}] Audio can not have equal start and end time.", user=user_name, level="error")
                return
            else:
                lprint(msg=f"[{method}] Audio cut started.", user=user_name, level="info")
                audio_file = cut_audio(audio_file, audio_start, audio_end, local_save_dir)

        if source_media_type == "animated":
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
            frame_files_range = [source_file]

        # get mel of audio
        mel_processor = MelProcessor(audio=audio_file, save_output=local_save_dir, fps=fps)
        mel_chunks = mel_processor.process()
        # create wav to lip
        full_frames_files = frame_files_range[:len(mel_chunks)]

        # delay for load model if a few processes
        if not ensure_memory_capacity(limit_ram=Settings.MIN_RAM_LIP_SYNC_GB, limit_vram=Settings.MIN_VRAM_LIP_SYNC_GB, device=device, retries=50):
            lprint(msg=f"Memory out of necessary for {method} during load lip sync models.", level="error", user=user_name)
            raise f"Memory out of necessary for {method}."

        wav2lip = GenerateWave2Lip(
            ALL_MODEL_FOLDER, retina_face=retina_face_checkpoint, face_parse=face_parse_checkpoint,
            local_gfpgan_path=local_gfpgan_path, gfpgan_model_path=gfpgan_model_path, device=device,
            emotion_label=None, similar_coeff=face_discrimination_factor
        )

        if wav2lip is None:
            lprint(msg="List index out of range in datagen of lip sync for.", level="error", user=user_name)
            raise "List index out of range in datagen of lip sync for, you can try set another timeline or face."

        wav2lip.face_fields = source_crop_area

        lprint(msg=f"[{method}] Face detection started.", user=user_name, level="info")
        gen = wav2lip.datagen(full_frames_files, mel_chunks, img_size, batch_size)
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
        memory_message(method, msg, user_name, "info")

        try:
            file_name = VideoManipulation.encrypted(wav2lip_result_video, local_save_dir)  # VideoManipulation.encrypted
            mp4_file_path = os.path.join(local_save_dir, file_name)
        except Exception as err:
            print(f"Error with VideoManipulation.encrypted {err}")
            mp4_file_path = wav2lip_result_video

        lprint(msg=f"[{method}] Save video with audio.", user=user_name, level="info")
        mp4_file_name = VideoManipulation.save_video_with_audio(mp4_file_path, audio_file, local_save_dir)
        mp4_file_path = os.path.join(local_save_dir, mp4_file_name)


        if start_part_source_file is not None:
            lprint(msg=f"[{method}] Concatenate videos.", user=user_name, level="info")
            concatenated_video = VideoManipulation.concatenate_videos(start_part_source_file, mp4_file_path, local_save_dir)
        else:
            concatenated_video = mp4_file_path

        if os.path.exists(concatenated_video) and os.path.getsize(concatenated_video) > 1024:  # 1 Kb
            if overlay_audio and source_audio_name:
                lprint(msg=f"[{method}] Overlay audio.", user=user_name, level="info")
                output_file = VideoManipulation.overlay_audio_on_video(concatenated_video, os.path.join(local_save_dir, source_audio_name), save_dir=local_save_dir)
                if not os.path.exists(output_file):
                    output_file = mp4_file_path
            else:
                lprint(msg=f"[{method}] Concatenated video have small size.", user=user_name, level="warn")
                output_file = concatenated_video
        else:  # if video concatenate this error and have small size
            output_file = mp4_file_path

        user_save_name = f"{user_save_name}.mp4" if user_save_name else f"{uuid.uuid4()}.mp4"

        shutil.move(output_file, os.path.join(user_save_dir, user_save_name))

        # remove tmp files
        if os.path.exists(source_file):
            os.remove(source_file)
        if tmp_source_file != source_file and os.path.exists(tmp_source_file):
            os.remove(tmp_source_file)
        if os.path.exists(local_save_dir):
            shutil.rmtree(local_save_dir)

        lprint(msg=f"[{method}] Synthesis lip sync finished.", user=user_name, level="info")

        # Update config about models if process was successfully
        if Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC:
            lprint(msg="Update checker of download size lip sync models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_LIP_SYNC": False})
            Settings.CHECK_DOWNLOAD_SIZE_LIP_SYNC = False
        if Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN and device == "cuda":
            lprint(msg="Update checker of download size gfpgan models.")
            Settings.config_manager.update_config({"CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": False})
            Settings.CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = False

        return {"files": os.path.join(user_save_dir, user_save_name)}



def memory_message(method: str, msg: str, user: str, level: str):
    _, available_vram_gb, busy_vram_gb = get_vram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_vram_gb} GB and busy {busy_vram_gb} GB VRAM.", user=user, level=level)
    _, available_ram_gb, busy_ram_gb = get_ram_gb()
    lprint(msg=f"[{method}] {msg}. Available {available_ram_gb} GB and busy {busy_ram_gb} GB RAM.", user=user, level=level)
