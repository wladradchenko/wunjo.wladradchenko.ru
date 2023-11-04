import os
import sys
import cv2
import json
import torch
import shutil
import requests
from tqdm import tqdm
from time import strftime
from argparse import Namespace

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
diffusers_root_path = os.path.join(root_path, "diffusers")
sys.path.insert(0, diffusers_root_path)

from src.utils.video2render import render
from src.utils.config import RenderConfig

sys.path.pop(0)

from cog import Input
from backend.folders import TMP_FOLDER, DEEPFAKE_MODEL_FOLDER
from backend.download import download_model, unzip, check_download_size, get_nested_url, is_connected
from deepfake.src.utils.segment import SegmentAnything
from deepfake.src.utils.videoio import (
    cut_start_video, get_frames, check_media_type, save_video_from_frames, extract_audio_from_video, save_video_with_audio
)
from diffusers.src.utils.mediaio import (
    save_video_frames_cv2, save_image_frame_cv2, vram_limit_device_resolution_diffusion, save_empty_mask
)
from diffusers.src.utils.ebsynth import Ebsynth

DEEPFAKE_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/deepfake.json"


def create_diffusion_instruction():
    # create default.json
    custom_diffusion_file = os.path.join(DEEPFAKE_MODEL_FOLDER, "custom_diffusion.json")
    # Check if file exists
    if not os.path.exists(custom_diffusion_file):
        # If the file doesn't exist, create and write to it
        with open(custom_diffusion_file, 'w') as f:
            json.dump({}, f)
    else:
        print(f"{custom_diffusion_file} already exists!")

    with open(custom_diffusion_file, 'r') as f:
        custom_diffusion = json.load(f)
    custom_diffusion = {**{"Realistic_Vision_V5.1.safetensors": None}, **custom_diffusion}
    return custom_diffusion



def get_config_deepfake() -> dict:
    try:
        response = requests.get(DEEPFAKE_JSON_URL)
        with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'wb') as file:
            file.write(response.content)
    except:
        print("Not internet connection to get actual versions of deepfake models")
    finally:
        if not os.path.isfile(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json')):
            deepfake = {}
        else:
            with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'r', encoding="utf8") as file:
                deepfake = json.load(file)

    return deepfake


file_deepfake_config = get_config_deepfake()


class Video2Video:
    @staticmethod
    def main_video_render(source: str, output_folder: str, masks: dict = None, interval: int = 10, source_start: float = 0, sd_model_name: str = None,
                          source_end: float = 0, source_type: str = "video", control_strength: float = 0.7, thickness_mask: int = 10,
                          use_limit_device_resolution: bool = True, predictor=None, session=None, segment_percentage: int = 25,
                          control_type: str = Input(description="Choose a controlnet", choices=["canny", "hed"], default="canny",),
                          translation: list = None):
        args = Video2Video.load_video2video_default()
        # create config
        cfg = RenderConfig()

        # folders and files
        cfg.input_path = source  # input video
        cfg.work_dir = os.path.join(output_folder, strftime("%Y_%m_%d_%H%M%S"))  # output folder
        os.makedirs(cfg.work_dir, exist_ok=True)

        frame_save_path = os.path.join(cfg.work_dir, "media")  # frames folder
        os.makedirs(frame_save_path, exist_ok=True)

        source_frame_folder_name = "media_original"
        source_frame_folder_path = os.path.join(cfg.work_dir, source_frame_folder_name)  # original frames folder
        os.makedirs(source_frame_folder_path, exist_ok=True)

        mask_save_path = os.path.join(cfg.work_dir, "masks")  # mask folder
        os.makedirs(mask_save_path, exist_ok=True)

        cfg.first_dir = os.path.join(cfg.work_dir, "first_key")
        os.makedirs(cfg.first_dir, exist_ok=True)

        cfg.output_path = os.path.join(cfg.work_dir, "blend.mp4")  # output video
        cfg.key_subdir = os.path.join(cfg.work_dir, "keys")  # frame folder
        os.makedirs(cfg.key_subdir, exist_ok=True)
        # values
        cfg.interval = interval
        cfg.control_strength = control_strength
        if translation is None:
            translation = []
        if "loose_cfattn" in translation:
            cfg.loose_cfattn = True
        if "freeu" in translation:
            cfg.freeu_args = (1.1, 1.2, 1.0, 0.2)

        cfg.use_limit_device_resolution = use_limit_device_resolution
        cfg.control_type = control_type
        cfg.canny_low = None if cfg.control_type != "canny" else cfg.canny_low
        cfg.canny_high = None if cfg.control_type != "canny" else cfg.canny_high

        # get models path
        checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # load model segmentation TODO small or big, I don't know yet
        vit_model_type = "vit_h"

        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        link_sam_vit_checkpoint = get_nested_url(file_deepfake_config, ["checkpoints", "sam_vit_h.pth"])
        if not os.path.exists(sam_vit_checkpoint):
            # check what is internet access
            is_connected(sam_vit_checkpoint)
            # download pre-trained models from url
            download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
        else:
            check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        link_onnx_vit_checkpoint = get_nested_url(file_deepfake_config, ["checkpoints", "vit_h_quantized.onnx"])
        if not os.path.exists(onnx_vit_checkpoint):
            # check what is internet access
            is_connected(onnx_vit_checkpoint)
            # download pre-trained models from url
            download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
        else:
            check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        # load diffuser
        diffuser_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "diffusion")
        os.environ['TORCH_HOME'] = diffuser_folder
        if not os.path.exists(diffuser_folder):
            os.makedirs(diffuser_folder)

        if sd_model_name is None:
            sd_model_path = os.path.join(diffuser_folder, "Realistic_Vision_V5.1.safetensors")
            link_sd_model = get_nested_url(file_deepfake_config, ["diffusion", "Realistic_Vision_V5.1.safetensors"])
            if not os.path.exists(sd_model_path):
                # check what is internet access
                is_connected(sd_model_path)
                # download pre-trained models from url
                download_model(sd_model_path, link_sd_model)
            else:
                check_download_size(sd_model_path, link_sd_model)
            # set default prompts to improve quality
            args.a_prompt = "RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"  # improve prompt
            args.n_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"  # default negative prompt
        else:
            sd_model_path = os.path.join(diffuser_folder, sd_model_name)
            print("Will use sd model ", sd_model_path)

        # load controlnet model
        if control_type == "hed":
            controlnet_model_path = os.path.join(diffuser_folder, "control_sd15_hed.pth")
            link_controlnet_model = get_nested_url(file_deepfake_config, ["diffusion", "control_sd15_hed.pth"])
            if not os.path.exists(controlnet_model_path):
                # check what is internet access
                is_connected(controlnet_model_path)
                # download pre-trained models from url
                download_model(controlnet_model_path, link_controlnet_model)
            else:
                check_download_size(controlnet_model_path, link_controlnet_model)
            # load hed annotator
            controlnet_model_annotator_path = os.path.join(diffuser_folder, "ControlNetHED.pth")
            link_controlnet_model_annotator = get_nested_url(file_deepfake_config, ["diffusion", "ControlNetHED.pth"])
            if not os.path.exists(controlnet_model_annotator_path):
                # check what is internet access
                is_connected(controlnet_model_annotator_path)
                # download pre-trained models from url
                download_model(controlnet_model_annotator_path, link_controlnet_model_annotator)
            else:
                check_download_size(controlnet_model_annotator_path, link_controlnet_model_annotator)
        elif control_type == "canny":
            controlnet_model_path = os.path.join(diffuser_folder, "control_sd15_canny.pth")
            link_controlnet_model = get_nested_url(file_deepfake_config, ["diffusion", "control_sd15_canny.pth"])
            if not os.path.exists(controlnet_model_path):
                # check what is internet access
                is_connected(controlnet_model_path)
                # download pre-trained models from url
                download_model(controlnet_model_path, link_controlnet_model)
            else:
                check_download_size(controlnet_model_path, link_controlnet_model)
        else:
            raise Exception("Error... undefined controlnet type")

        # load vae model
        vae_model_path = os.path.join(diffuser_folder, "vae-ft-mse-840000-ema-pruned.ckpt")
        link_vae_model = get_nested_url(file_deepfake_config, ["diffusion", "vae-ft-mse-840000-ema-pruned.ckpt"])
        if not os.path.exists(vae_model_path):
            # check what is internet access
            is_connected(vae_model_path)
            # download pre-trained models from url
            download_model(vae_model_path, link_vae_model)
        else:
            check_download_size(vae_model_path, link_vae_model)

        # load gmflow model
        gmflow_model_path = os.path.join(diffuser_folder, "gmflow_sintel-0c07dcb3.pth")
        link_gmflow_model = get_nested_url(file_deepfake_config, ["diffusion", "gmflow_sintel-0c07dcb3.pth"])
        if not os.path.exists(gmflow_model_path):
            # check what is internet access
            is_connected(vae_model_path)
            # download pre-trained models from url
            download_model(gmflow_model_path, link_gmflow_model)
        else:
            check_download_size(gmflow_model_path, link_gmflow_model)

        # segmentation mask
        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage)
        if session is None:
            session = segmentation.init_onnx(onnx_vit_checkpoint, "cuda")  # TODO or small for CPU?
        if predictor is None:
            predictor = segmentation.init_vit(sam_vit_checkpoint, vit_model_type, "cuda")  # TODO or small for CPU?

        # cut video
        if source_type == "video":
            source = cut_start_video(source, source_start, source_end)
            # get audio from video target
            audio_file_name = extract_audio_from_video(source, cfg.work_dir)
        else:
            audio_file_name = None

        remove_keys = []
        for key in masks.keys():
            if key == "background":
                # for background image
                masks[key]["end_time"] = source_end - source_start
                continue
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

        source_media_type = check_media_type(source)
        if source_media_type == "static":
            fps, num_frames, width, height = save_image_frame_cv2(source, frame_save_path, '%04d.png', vram_limit_device_resolution_diffusion, "cuda")
        elif source_media_type == "animated":
            fps, num_frames, width, height = save_video_frames_cv2(source, frame_save_path, '%04d.png', vram_limit_device_resolution_diffusion, "cuda")
            save_video_frames_cv2(source, source_frame_folder_path, '%04d.png', vram_limit_device_resolution_diffusion, "cuda")
        else:
            raise Exception("Source is not detected as image or video")

        # read frames files
        frame_files = sorted(os.listdir(frame_save_path))
        frame_files_with_interval = frame_files[::interval] + [frame_files[-1]]

        if source_media_type == "animated":
            source = save_video_from_frames(frame_names="%04d.png", save_path=frame_save_path, alternative_save_path=cfg.work_dir, fps=fps)
            source = os.path.join(cfg.work_dir, source)

        # Extract the background data and remove it from the original dictionary
        background_mask = masks.pop('background', None)
        if background_mask:
            # Create the directory if it doesn't exist
            mask_key_save_path = os.path.join(mask_save_path, "mask_background")
            os.makedirs(mask_key_save_path, exist_ok=True)
            # Generate and save black images
            for i in range(num_frames):
                if i == num_frames:
                    break
                save_empty_mask(i + 1, width, height, mask_key_save_path, '%04d.png')
            background_mask["frame_files_path"] = mask_key_save_path

        # get segmentation frames as maks and save
        segmentation.load_models(predictor=predictor, session=session)

        for key in masks.keys():
            print(f"Processing ID: {key}")
            mask_key_save_path = os.path.join(mask_save_path, f"mask_{key}")
            os.makedirs(mask_key_save_path, exist_ok=True)
            start_time = masks[key]["start_time"]
            end_time = masks[key]["end_time"]
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) + 1
            # list of frames of list of one frame
            filter_frames_files = frame_files[start_frame:end_frame]
            # set progress bar
            progress_bar = tqdm(total=len(filter_frames_files), unit='it', unit_scale=True)
            filter_frame_file_name = filter_frames_files[0]
            filter_frame = cv2.imread(os.path.join(frame_save_path, filter_frame_file_name))  # read first frame file after filter
            # get segment mask
            segment_mask = segmentation.set_obj(point_list=masks[key]["point_list"], frame=filter_frame)
            # save first frame as 0001
            segmentation.save_black_mask(filter_frame_file_name, segment_mask, mask_key_save_path, kernel_size=thickness_mask, width=width, height=height)

            # update background
            if background_mask:
                background_mask_frame_file_path = os.path.join(mask_save_path, "mask_background", filter_frame_file_name)
                background_mask_frame = cv2.imread(background_mask_frame_file_path)
                segmentation.save_white_background_mask(background_mask_frame_file_path, segment_mask, background_mask_frame, kernel_size=thickness_mask, width=width, height=height)

            # update progress bar
            progress_bar.update(1)
            # update interval frame
            start_filter_frame = filter_frame_file_name
            end_filter_frame = filter_frame_file_name

            if len(filter_frames_files) > 1:
                for filter_frame_file_name in filter_frames_files[1:]:
                    filter_frame = cv2.imread(os.path.join(frame_save_path, filter_frame_file_name))  # read frame file after filter
                    segment_mask = segmentation.draw_mask_frames(frame=filter_frame)
                    if segment_mask is None:
                        print(key, "Encountered None mask. Breaking the loop.")
                        break
                    # update infor about end mask frame
                    end_filter_frame = filter_frame_file_name
                    # save frames after 0002
                    segmentation.save_black_mask(filter_frame_file_name, segment_mask, mask_key_save_path, kernel_size=thickness_mask, width=width, height=height)
                    # update background
                    if background_mask:
                        background_mask_frame_file_path = os.path.join(mask_save_path, "mask_background", filter_frame_file_name)
                        background_mask_frame = cv2.imread(background_mask_frame_file_path)
                        segmentation.save_white_background_mask(background_mask_frame_file_path, segment_mask, background_mask_frame, kernel_size=thickness_mask, width=width, height=height)

                    progress_bar.update(1)
            # update interval list frames by start and last frame, and next after last frame
            if end_filter_frame != frame_files[-1]:
                if start_filter_frame != frame_files[0]:
                    frame_files_with_interval += [frame_files[frame_files.index(start_filter_frame) - 1], start_filter_frame, end_filter_frame, frame_files[frame_files.index(end_filter_frame) + 1]]
                else:
                    frame_files_with_interval += [start_filter_frame, end_filter_frame, frame_files[frame_files.index(end_filter_frame) + 1]]
            else:
                frame_files_with_interval += [start_filter_frame, end_filter_frame]
            # set new key
            masks[key]["frame_files_path"] = mask_key_save_path
            # close progress bar for key
            progress_bar.close()

        del segmentation, predictor, session  # remove
        torch.cuda.empty_cache()

        # combine masks and background mask
        if background_mask:
            # masks = {**{"background": background_mask}, **masks}
            masks = {**masks, **{"background": background_mask}}

        # generate diffusion images by prompt
        frame_files_with_interval = sorted(list(set(frame_files_with_interval)))
        # diffusion frames
        render(
            cfg=cfg, args=args, masks=masks, frame_files_with_interval=frame_files_with_interval, sd_model_path=sd_model_path,
            controlnet_model_path=controlnet_model_path, vae_model_path=vae_model_path, gmflow_model_path=gmflow_model_path,
            frame_path=frame_save_path, mask_path=mask_save_path
        )

        if source_media_type == "static":
            # remove only tmp folder
            for f in os.listdir(TMP_FOLDER):
                os.remove(os.path.join(TMP_FOLDER, f))
            # return changed image
            return os.path.join(cfg.key_subdir, frame_files[0])

        # download if ebsynth app is not exist
        ebsynth_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "ebsynth")
        os.environ['TORCH_HOME'] = ebsynth_folder
        if not os.path.exists(ebsynth_folder):
            os.makedirs(ebsynth_folder)

        if sys.platform == 'win32':
            ebsynth_path = os.path.join(ebsynth_folder, "EbSynth.exe")
            ebsynth_path_zip = os.path.join(ebsynth_folder, "EbSynth-Beta-Win.zip")
            link_ebsynth = get_nested_url(file_deepfake_config, ["ebsynth", "EbSynth-Beta-Win.zip"])
            if not os.path.exists(ebsynth_path):
                # check what is internet access
                is_connected(ebsynth_path)
                # download pre-trained models from url
                download_model(ebsynth_path_zip, link_ebsynth)
                unzip(ebsynth_path_zip, ebsynth_folder)
            else:
                check_download_size(ebsynth_path_zip, link_ebsynth)
                if not os.listdir(ebsynth_folder):
                    unzip(ebsynth_path_zip, ebsynth_folder)
            # access read app
            # username = os.environ.get('USERNAME')
            username = os.environ.get('USERNAME') or os.environ.get('USER')
            os.system(f'icacls "{ebsynth_path}" /grant:r "{username}:(R,W)" /T')
            ebsynth_path = os.path.join(ebsynth_folder, "EbSynth.exe")
            os.system(f'icacls "{ebsynth_path}" /grant:r "{username}:(R,W)" /T')
        elif sys.platform == 'linux':
            ebsynth_path = os.path.join(ebsynth_folder, "ebsynth_linux_cu118")
            link_ebsynth = get_nested_url(file_deepfake_config, ["ebsynth", "ebsynth_linux_cu118"])
            if not os.path.exists(ebsynth_path):
                # check what is internet access
                is_connected(ebsynth_path)
                # download pre-trained models from url
                download_model(ebsynth_path, link_ebsynth)
            else:
                check_download_size(ebsynth_path, link_ebsynth)
            # access read app
            os.system(f"chmod +x {ebsynth_path}")
        else:
            raise Exception("Ebsynth is not support this platform")

        # before start set to will be all keys
        frame_dir = os.path.join(cfg.work_dir, source_frame_folder_name)
        # Iterate through each file in frame_files_with_interval
        for frame_file in frame_files_with_interval:
            # Destination path for the key
            key_path = os.path.join(cfg.key_subdir, frame_file)
            # Check if the file doesn't exist in the key directory
            if not os.path.exists(key_path):
                # Source path for the frame file
                frame_path = os.path.join(frame_dir, frame_file)
                # Copy the file from frame directory to key directory
                shutil.copy(frame_path, key_path)

        # processing ebsynth
        ebsynth = Ebsynth(gmflow_model_path=gmflow_model_path, ebsynth_path=ebsynth_path)
        output_frame_folder_path, output_names = ebsynth.processing_ebsynth(
            frames_path=cfg.key_subdir, frames=frame_files_with_interval,
            base_folder=cfg.work_dir, input_subdir=source_frame_folder_name
        )

        # remove first frame
        os.remove(os.path.join(output_frame_folder_path, output_names % 1))
        # get saved file as merge frames to video
        save_name = save_video_from_frames(frame_names=output_names, save_path=output_frame_folder_path, fps=fps, alternative_save_path=cfg.work_dir)
        # combine audio and video
        if os.path.exists(os.path.join(cfg.work_dir, str(audio_file_name))):
            save_name = save_video_with_audio(os.path.join(cfg.work_dir, save_name), os.path.join(cfg.work_dir, str(audio_file_name)), cfg.work_dir)
        # remove files
        for f in os.listdir(cfg.work_dir):
            if save_name == f:
                save_name = os.path.join(cfg.work_dir, f)
            else:
                if os.path.isfile(os.path.join(cfg.work_dir, f)):
                    os.remove(os.path.join(cfg.work_dir, f))
                elif os.path.isdir(os.path.join(cfg.work_dir, f)):
                    shutil.rmtree(os.path.join(cfg.work_dir, f))

        for f in os.listdir(TMP_FOLDER):
            os.remove(os.path.join(TMP_FOLDER, f))

        return save_name

    @staticmethod
    def load_video2video_default():
        return Namespace(
            cfg=None,  # str Path to config file
            input=None,  # str The input path to video.
            output=None,  # str
            prompt=None,  # str
            key_video_path=None,  # str
            one=None,  # bool Run the first frame with ControlNet only
            nr=None,  # bool Do not run rerender and do postprocessing only
            nb=None,  # bool Do not run postprocessing and run rerender only
            ne=None,  # bool Do not run ebsynth (use previous ebsynth temporary output)
            nps=None,  # bool Do not run poisson gradient blending
            n_proc=1,  # int The max process count (removed multiprocessing because duplicate)
            tmp=None,  # bool Keep ebsynth temporary output,
            a_prompt="",  # improve prompt
            n_prompt="",  # default negative prompt
            crop=[0, 0, 0, 0],  # crop video turn off
            canny_low=50,
            canny_high=100,
            warp_period=[0, 0.1],
            ada_period=[0.8, 1],
            freeu_args=[1.1, 1.2, 1.0, 0.2],
        )