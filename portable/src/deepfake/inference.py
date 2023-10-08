import os
import cv2
import sys
import torch
import imageio
import subprocess
import numpy as np
from tqdm import tqdm
from time import strftime
from argparse import Namespace

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deepfake_root_path = os.path.join(root_path, "deepfake")
sys.path.insert(0, deepfake_root_path)

"""Video and image"""
from src.utils.videoio import (
    save_video_with_audio, cut_start_video, get_frames, get_first_frame, encrypted,
    check_media_type, extract_audio_from_video, save_video_from_frames, video_to_frames
)
from src.utils.imageio import save_image_cv2, read_image_cv2, save_colored_mask_cv2
"""Video and image"""
"""Animation Face"""
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
"""Animation Face"""
"""Wav2Lip"""
from src.wav2mel import MelProcessor
from src.utils.face_enhancer import enhancer as face_enhancer
from src.utils.video2fake import GenerateFakeVideo2Lip
"""Wav2Lip"""
"""Face Swap"""
from src.utils.faceswap import FaceSwapDeepfake
"""Face Swap"""
"""Retouch"""
from src.retouch import InpaintModel, process_retouch, pil_to_cv2, convert_cv2_to_pil, VideoRemoveObjectProcessor, convert_colored_mask_thickness_cv2, upscale_retouch_frame
"""Retouch"""
"""Segmentation"""
from src.utils.segment import SegmentAnything
"""Segmentation"""

sys.path.pop(0)

from cog import Input
import requests
import shutil
import json

from backend.folders import DEEPFAKE_MODEL_FOLDER, TMP_FOLDER
from backend.download import download_model, unzip, check_download_size


DEEPFAKE_JSON_URL = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/deepfake.json"


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


class AnimationFaceTalk:
    """
    Animation face talk on image or gif
    """
    @staticmethod
    def main_img_deepfake(source_image: str, driven_audio: str, deepfake_dir: str, still: bool = True, face_fields: list = None,
                          enhancer: str = Input(description="Choose a face enhancer", choices=["gfpgan", "RestoreFormer"], default="gfpgan",),
                          preprocess: str = Input(description="how to preprocess the images", choices=["crop", "resize", "full"], default="full",),
                          expression_scale=1.0, input_yaw=None, input_pitch=None, input_roll=None, background_enhancer=None, pose_style=0):

        if file_deepfake_config == {}:
            raise "[Error] Config file deepfake.json is not exist"
        # torch.backends.cudnn.enabled = False
        args = AnimationFaceTalk.load_img_default()
        args.checkpoint_dir = "checkpoints"
        args.source_image = source_image
        args.driven_audio = driven_audio
        args.result_dir = deepfake_dir

        cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True

        if torch.cuda.is_available() and not cpu:
            print("Processing will run on GPU")
            args.device = "cuda"
        else:
            print("Processing will run on CPU")
            args.device = "cpu"

        args.still = still
        args.enhancer = enhancer

        args.expression_scale = expression_scale
        args.input_yaw = input_yaw
        args.input_pitch = input_pitch
        args.input_roll = input_roll
        args.background_enhancer = "realesrgan" if background_enhancer else None

        if preprocess is None:
            preprocess = "crop"
        args.preprocess = preprocess

        pic_path = args.source_image
        audio_path = args.driven_audio
        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll

        current_root_path = deepfake_root_path  # os.path.split(current_code_path)[0]
        model_user_path = DEEPFAKE_MODEL_FOLDER

        os.environ['TORCH_HOME'] = os.path.join(model_user_path, args.checkpoint_dir)
        checkpoint_dir_full = os.path.join(model_user_path, args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        path_of_lm_croper = os.path.join(checkpoint_dir_full, 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(path_of_lm_croper):
            link_of_lm_croper = file_deepfake_config["checkpoints"]["shape_predictor_68_face_landmarks.dat"]
            download_model(path_of_lm_croper, link_of_lm_croper)
        else:
            link_of_lm_croper = file_deepfake_config["checkpoints"]["shape_predictor_68_face_landmarks.dat"]
            check_download_size(path_of_lm_croper, link_of_lm_croper)

        path_of_net_recon_model = os.path.join(checkpoint_dir_full, 'epoch_20.pth')
        if not os.path.exists(path_of_net_recon_model):
            link_of_net_recon_model = file_deepfake_config["checkpoints"]["epoch_20.pth"]
            download_model(path_of_net_recon_model, link_of_net_recon_model)
        else:
            link_of_net_recon_model = file_deepfake_config["checkpoints"]["epoch_20.pth"]
            check_download_size(path_of_net_recon_model, link_of_net_recon_model)

        dir_of_BFM_fitting = os.path.join(checkpoint_dir_full, 'BFM_Fitting')
        args.bfm_folder = dir_of_BFM_fitting
        if not os.path.exists(dir_of_BFM_fitting):
            link_of_BFM_fitting = file_deepfake_config["checkpoints"]["BFM_Fitting.zip"]
            download_model(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), link_of_BFM_fitting)
            unzip(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), checkpoint_dir_full)
        else:
            link_of_BFM_fitting = file_deepfake_config["checkpoints"]["BFM_Fitting.zip"]
            check_download_size(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), link_of_BFM_fitting)
            if not os.listdir(dir_of_BFM_fitting):
                unzip(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), checkpoint_dir_full)

        dir_of_hub_models = os.path.join(checkpoint_dir_full, 'hub')
        if not os.path.exists(dir_of_hub_models):
            link_of_hub_models = file_deepfake_config["checkpoints"]["hub.zip"]
            download_model(os.path.join(checkpoint_dir_full, 'hub.zip'), link_of_hub_models)
            unzip(os.path.join(checkpoint_dir_full, 'hub.zip'), checkpoint_dir_full)
        else:
            link_of_hub_models = file_deepfake_config["checkpoints"]["hub.zip"]
            check_download_size(os.path.join(checkpoint_dir_full, 'hub.zip'), link_of_hub_models)
            if not os.listdir(dir_of_hub_models):
                unzip(os.path.join(checkpoint_dir_full, 'hub.zip'), checkpoint_dir_full)

        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        if not os.path.exists(wav2lip_checkpoint):
            link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
            download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
        else:
            link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
            check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)

        audio2pose_checkpoint = os.path.join(checkpoint_dir_full, 'auido2pose_00140-model.pth')
        if not os.path.exists(audio2pose_checkpoint):
            link_audio2pose_checkpoint = file_deepfake_config["checkpoints"]["auido2pose_00140-model.pth"]
            download_model(audio2pose_checkpoint, link_audio2pose_checkpoint)
        else:
            link_audio2pose_checkpoint = file_deepfake_config["checkpoints"]["auido2pose_00140-model.pth"]
            check_download_size(audio2pose_checkpoint, link_audio2pose_checkpoint)
        audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

        audio2exp_checkpoint = os.path.join(checkpoint_dir_full, 'auido2exp_00300-model.pth')
        if not os.path.exists(audio2exp_checkpoint):
            link_audio2exp_checkpoint = file_deepfake_config["checkpoints"]["auido2exp_00300-model.pth"]
            download_model(audio2exp_checkpoint, link_audio2exp_checkpoint)
        else:
            link_audio2exp_checkpoint = file_deepfake_config["checkpoints"]["auido2exp_00300-model.pth"]
            check_download_size(audio2exp_checkpoint, link_audio2exp_checkpoint)
        audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

        free_view_checkpoint = os.path.join(checkpoint_dir_full, 'facevid2vid_00189-model.pth.tar')
        if not os.path.exists(free_view_checkpoint):
            link_free_view_checkpoint = file_deepfake_config["checkpoints"]["facevid2vid_00189-model.pth.tar"]
            download_model(free_view_checkpoint, link_free_view_checkpoint)
        else:
            link_free_view_checkpoint = file_deepfake_config["checkpoints"]["facevid2vid_00189-model.pth.tar"]
            check_download_size(free_view_checkpoint, link_free_view_checkpoint)

        if args.preprocess == 'full':
            mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00109-model.pth.tar')
            if not os.path.exists(mapping_checkpoint):
                link_mapping_checkpoint = file_deepfake_config["checkpoints"]["mapping_00109-model.pth.tar"]
                download_model(mapping_checkpoint, link_mapping_checkpoint)
            else:
                link_mapping_checkpoint = file_deepfake_config["checkpoints"]["mapping_00109-model.pth.tar"]
                check_download_size(mapping_checkpoint, link_mapping_checkpoint)
            facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
        else:
            mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00229-model.pth.tar')
            if not os.path.exists(mapping_checkpoint):
                link_mapping_checkpoint = file_deepfake_config["checkpoints"]["mapping_00229-model.pth.tar"]
                download_model(mapping_checkpoint, link_mapping_checkpoint)
            else:
                link_mapping_checkpoint = file_deepfake_config["checkpoints"]["mapping_00229-model.pth.tar"]
                check_download_size(mapping_checkpoint, link_mapping_checkpoint)
            facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

        # init model
        print("Starting to crop and extract frames")
        preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device, face_fields)

        print("Starting get audio coefficient")
        audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path, wav2lip_checkpoint, device)

        print("Starting animate face from audio coefficient")
        animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, device)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)

        print('Extraction 3DMM for source image')
        pic_path_type = check_media_type(pic_path)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_path_type=pic_path_type)
        if first_coeff_path is None:
            print("Can't get the coefficients by 3DMM of the input")
            return

        ref_eyeblink_coeff_path = None
        ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # coeff2video
        args.enhancer = "gfpgan" if background_enhancer else enhancer
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, input_yaw_list, input_pitch_list, input_roll_list, expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess)
        mp4_path = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, pic_path_type=pic_path_type, device=device)

        for f in os.listdir(save_dir):
            if mp4_path == f:
                mp4_path = os.path.join(save_dir, f)
            else:
                if os.path.isfile(os.path.join(save_dir, f)):
                    os.remove(os.path.join(save_dir, f))
                elif os.path.isdir(os.path.join(save_dir, f)):
                    shutil.rmtree(os.path.join(save_dir, f))

        return mp4_path

    @staticmethod
    def load_img_default():
        return Namespace(
            batch_size=2,
            ref_eyeblink=None,
            ref_pose=None,
            net_recon="resnet50",
            init_path=None,
            use_last_fc=False,
            bfm_folder=os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints", "BFM_Fitting"),
            bfm_model="BFM_model_front.mat",
            focal=1015.0,
            center=112.0,
            camera_d=10.0,
            z_near=5.0,
            z_far=15.0,
        )


class AnimationMouthTalk:
    """
    Animation mouth talk on video
    """
    @staticmethod
    def main_video_deepfake(deepfake_dir: str, face: str, audio: str, static: bool = False, face_fields: list = None,
                            enhancer: str = Input(description="Choose a face enhancer", choices=["gfpgan", "RestoreFormer"], default="gfpgan",),
                            box: list = [-1, -1, -1, -1], background_enhancer: str = None, video_start: float = 0, video_end: float = 0, emotion_label: int = None, similar_coeff: float = 0.96):
        args = AnimationMouthTalk.load_video_default()
        args.checkpoint_dir = "checkpoints"
        args.result_dir = deepfake_dir
        args.video_start = float(video_start)
        args.video_end = float(video_end)
        args.face = face
        args.audio = audio
        args.static = static  # ok, we will check what it is video, not img
        args.box = box  # a constant bounding box for the face. Use only as a last resort if the face is not detected.
        # Also (about box), might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
        args.face_fields = face_fields
        args.enhancer = enhancer
        args.background_enhancer = background_enhancer

        cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True

        if torch.cuda.is_available() and not cpu:
            print("Processing will run on GPU")
            args.device = "cuda"
        else:
            print("Processing will run on CPU")
            args.device = "cpu"

        args.background_enhancer = "realesrgan" if background_enhancer else None

        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        model_user_path = DEEPFAKE_MODEL_FOLDER

        os.environ['TORCH_HOME'] = os.path.join(model_user_path, args.checkpoint_dir)
        checkpoint_dir_full = os.path.join(model_user_path, args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        if emotion_label is None:
            wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
            if not os.path.exists(wav2lip_checkpoint):
                link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
                download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
            else:
                link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
                check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)
        else:
            wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'emo2lip.pth')
            if not os.path.exists(wav2lip_checkpoint):
                link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["emo2lip.pth"]
                download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
            else:
                link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["emo2lip.pth"]
                check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)

        # If video_start is not 0 when cut video from start
        args.face = cut_start_video(args.face, args.video_start, args.video_end)

        # get video frames
        frames, fps = get_frames(video=args.face, rotate=args.rotate, crop=args.crop, resize_factor=args.resize_factor)
        # get mel of audio
        mel_processor = MelProcessor(args=args, save_output=save_dir, fps=fps)
        mel_chunks = mel_processor.process()
        # create wav to lip
        full_frames = frames[:len(mel_chunks)]
        batch_size = args.wav2lip_batch_size
        wav2lip = GenerateFakeVideo2Lip(DEEPFAKE_MODEL_FOLDER, emotion_label=emotion_label, similar_coeff=similar_coeff)
        wav2lip.face_fields = args.face_fields

        print("Face detect starting")
        gen = wav2lip.datagen(
            full_frames.copy(), mel_chunks, args.box, args.static, args.img_size, args.wav2lip_batch_size,
            args.pads, args.nosmooth
        )
        # load wav2lip
        print("Starting mouth animate")
        wav2lip_processed_video = wav2lip.generate_video_from_chunks(gen, mel_chunks, batch_size, wav2lip_checkpoint, args.device, save_dir, fps)
        if wav2lip_processed_video is None:
            return
        wav2lip_result_video = wav2lip_processed_video

        # after face or background enchanter
        enhancer = "gfpgan" if background_enhancer else enhancer
        if enhancer:
            video_name_enhancer = 'wav2lip_video_enhanced.mp4'
            enhanced_path = os.path.join(save_dir, 'temp_' + video_name_enhancer)
            enhanced_images = face_enhancer(wav2lip_processed_video, method=enhancer, bg_upsampler=background_enhancer, device=args.device)
            imageio.mimsave(enhanced_path, enhanced_images, fps=float(fps))
            wav2lip_result_video = enhanced_path

        try:
            file_name = encrypted(wav2lip_result_video, save_dir)  # encrypted
            wav2lip_result_video = os.path.join(save_dir, file_name)
        except Exception as err:
            print(f"Error with encrypted {err}")
        mp4_path = save_video_with_audio(wav2lip_result_video, args.audio, save_dir)

        for f in os.listdir(save_dir):
            if mp4_path == f:
                mp4_path = os.path.join(save_dir, f)
            else:
                if os.path.isfile(os.path.join(save_dir, f)):
                    os.remove(os.path.join(save_dir, f))
                elif os.path.isdir(os.path.join(save_dir, f)):
                    shutil.rmtree(os.path.join(save_dir, f))

        for f in os.listdir(TMP_FOLDER):
            os.remove(os.path.join(TMP_FOLDER, f))

        return mp4_path

    @staticmethod
    def load_video_default():
        return Namespace(
            pads=[0, 10, 0, 0],
            face_det_batch_size=16,
            wav2lip_batch_size=128,
            resize_factor=1,
            crop=[0, -1, 0, -1],
            rotate=False,
            nosmooth=False,
            img_size=96
        )



class FaceSwap:
    """
    Face swap by one photo
    """
    @staticmethod
    def main_faceswap(deepfake_dir: str, target: str, target_face_fields: str, source: str, source_face_fields: str,
                      type_file_target: str, type_file_source: str, target_video_start: float = 0, target_video_end: float = 0, source_video_frame: float = 0, source_video_end: float = 0,
                      enhancer: str = Input(description="Choose a face enhancer", choices=["gfpgan", "RestoreFormer"], default="gfpgan",),
                      background_enhancer: str = None, multiface: bool = False, similarface: bool = False, similar_coeff: float = 0.95):
        args = FaceSwap.load_faceswap_default()
        # Folders
        args.checkpoint_dir = "checkpoints"
        args.result_dir = deepfake_dir
        # Target
        args.target = target
        args.target_face_fields = target_face_fields
        args.type_file_target = type_file_target
        args.target_video_start = float(target_video_start)
        args.target_video_end = float(target_video_end)
        # Source
        args.source = source
        args.source_face_fields = source_face_fields
        args.type_file_source = type_file_source
        args.source_video_frame = float(source_video_frame)
        args.source_video_end = float(source_video_end)
        # Additionally processing
        args.multiface = multiface
        args.enhancer = enhancer
        args.background_enhancer = background_enhancer

        cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True

        if torch.cuda.is_available() and not cpu:
            print("Processing will run on GPU")
            args.device = "cuda"
        else:
            print("Processing will run on CPU")
            args.device = "cpu"

        args.background_enhancer = "realesrgan" if background_enhancer else None
        enhancer = "gfpgan" if background_enhancer else enhancer

        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        model_user_path = DEEPFAKE_MODEL_FOLDER

        os.environ['TORCH_HOME'] = os.path.join(model_user_path, args.checkpoint_dir)
        checkpoint_dir_full = os.path.join(model_user_path, args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
        if not os.path.exists(faceswap_checkpoint):
            link_faceswap_checkpoint = file_deepfake_config["checkpoints"]["faceswap.onnx"]
            download_model(faceswap_checkpoint, link_faceswap_checkpoint)
        else:
            link_faceswap_checkpoint = file_deepfake_config["checkpoints"]["faceswap.onnx"]
            check_download_size(faceswap_checkpoint, link_faceswap_checkpoint)

        faceswap = FaceSwapDeepfake(DEEPFAKE_MODEL_FOLDER, faceswap_checkpoint, similarface, similar_coeff)

        # get source for face
        if args.type_file_source == "video":
            args.source = cut_start_video(args.source, args.source_video_frame, args.source_video_end)
        # get frame
        source_frame = get_first_frame(args.source)
        source_face = faceswap.face_detect_with_alignment_from_source_frame(source_frame, args.source_face_fields)

        # if this is video target
        args.type_file_target = check_media_type(args.target)
        if args.type_file_target == "animated":
            # If video_start for target is not 0 when cut video from start
            args.target = cut_start_video(args.target, args.target_video_start, args.target_video_end)

            # get video frame for source if type is video
            target_frames, fps = get_frames(video=args.target, rotate=args.rotate, crop=args.crop, resize_factor=args.resize_factor)
            # create face swap
            file_name = "swap_result.mp4"
            saved_file = os.path.join(save_dir, file_name)
            faceswap.swap_video(target_frames, source_face, args.target_face_fields, saved_file, args.multiface, fps)

            # after face or background enchanter
            if enhancer:
                enhanced_images = face_enhancer(saved_file, method=enhancer, bg_upsampler=background_enhancer, device=args.device)
                file_name = "swap_result_enhanced.mp4"
                saved_file = os.path.join(save_dir, file_name)
                imageio.mimsave(saved_file, enhanced_images, fps=float(fps))

            try:
                file_name = encrypted(saved_file, save_dir)  # encrypted
                saved_file = os.path.join(save_dir, file_name)
            except Exception as err:
                print(f"Error with encrypted {err}")

            # get audio from video target
            audio_file_name = extract_audio_from_video(args.target, save_dir)
            # combine audio and video
            file_name = save_video_with_audio(saved_file, os.path.join(save_dir, str(audio_file_name)), save_dir)

        else:  # static file
            # create face swap on image
            target_frame = get_first_frame(args.target)
            target_image = faceswap.swap_image(target_frame, source_face, args.target_face_fields, save_dir, args.multiface)
            file_name = "swap_result.png"
            saved_file = save_image_cv2(os.path.join(save_dir, file_name), target_image)
            if enhancer:
                enhanced_image = face_enhancer(saved_file, method=enhancer, bg_upsampler=background_enhancer, device=args.device)
                file_name = "swap_result_enhanced.png"
                saved_file = os.path.join(save_dir, file_name)
                imageio.imsave(saved_file, enhanced_image[0])

            try:
                file_name = encrypted(saved_file, save_dir)  # encrypted
            except Exception as err:
                print(f"Error with encrypted {err}")

        for f in os.listdir(save_dir):
            if file_name == f:
                file_name = os.path.join(save_dir, f)
            else:
                if os.path.isfile(os.path.join(save_dir, f)):
                    os.remove(os.path.join(save_dir, f))
                elif os.path.isdir(os.path.join(save_dir, f)):
                    shutil.rmtree(os.path.join(save_dir, f))

        for f in os.listdir(TMP_FOLDER):
            os.remove(os.path.join(TMP_FOLDER, f))

        return file_name

    @staticmethod
    def load_faceswap_default():
        return Namespace(
            resize_factor=1,
            crop=[0, -1, 0, -1],
            rotate=False,
        )


class Retouch:
    """Retouch image or video"""

    @staticmethod
    def main_retouch(output: str, source: str, masks: dict, retouch_model_type: str = "retouch_object", predictor=None,
                     session=None, source_start: float = 0, source_end: float = 0, source_type: str = "img",
                     mask_color: str = None, upscale: bool = True, blur: int = 1, segment_percentage: int = 25):
        # create folder
        save_dir = os.path.join(output, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
        # get device
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if torch.cuda.is_available() and not use_cpu:
            print("Processing will run on GPU")
            device = "cuda"
            use_half = True
        else:
            print("Processing will run on CPU")
            device = "cpu"
            use_half = False
        # get models path
        checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
        os.environ['TORCH_HOME'] = checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # load model improved remove object
        model_pro_painter_path = os.path.join(checkpoint_folder, "ProPainter.pth")

        if not os.path.exists(model_pro_painter_path):
            link_pro_painter = file_deepfake_config["checkpoints"]["ProPainter.pth"]
            download_model(model_pro_painter_path, link_pro_painter)
        else:
            link_pro_painter = file_deepfake_config["checkpoints"]["ProPainter.pth"]
            check_download_size(model_pro_painter_path, link_pro_painter)

        model_raft_things_path = os.path.join(checkpoint_folder, "raft-things.pth")

        if not os.path.exists(model_raft_things_path):
            link_raft_things = file_deepfake_config["checkpoints"]["raft-things.pth"]
            download_model(model_raft_things_path, link_raft_things)
        else:
            link_raft_things = file_deepfake_config["checkpoints"]["raft-things.pth"]
            check_download_size(model_raft_things_path, link_raft_things)

        model_recurrent_flow_path = os.path.join(checkpoint_folder, "recurrent_flow_completion.pth")

        if not os.path.exists(model_recurrent_flow_path):
            link_recurrent_flow = file_deepfake_config["checkpoints"]["recurrent_flow_completion.pth"]
            download_model(model_recurrent_flow_path, link_recurrent_flow)
        else:
            link_recurrent_flow = file_deepfake_config["checkpoints"]["recurrent_flow_completion.pth"]
            check_download_size(model_recurrent_flow_path, link_recurrent_flow)

        # load model retouch
        if retouch_model_type == "retouch_face":
            retouch_model_name = "retouch_face"
            model_retouch_path = os.path.join(checkpoint_folder, "retouch_face.pth")
        else:
            retouch_model_name = "retouch_object"
            model_retouch_path = os.path.join(checkpoint_folder, "retouch_object.pth")

        if not os.path.exists(model_retouch_path):
            link_model_retouch = file_deepfake_config["checkpoints"][f"{retouch_model_name}.pth"]
            download_model(model_retouch_path, link_model_retouch)
        else:
            link_model_retouch = file_deepfake_config["checkpoints"][f"{retouch_model_name}.pth"]
            check_download_size(model_retouch_path, link_model_retouch)

        # load model segmentation
        if device == "cuda":
            vit_model_type = "vit_h"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
            if not os.path.exists(sam_vit_checkpoint):
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_h.pth"]
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
            else:
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_h.pth"]
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
            if not os.path.exists(onnx_vit_checkpoint):
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_h_quantized.onnx"]
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
            else:
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_h_quantized.onnx"]
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
        else:
            vit_model_type = "vit_b"

            sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
            if not os.path.exists(sam_vit_checkpoint):
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_b.pth"]
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
            else:
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_b.pth"]
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
            if not os.path.exists(onnx_vit_checkpoint):
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_b_quantized.onnx"]
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
            else:
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_b_quantized.onnx"]
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        # segment
        segment_percentage = segment_percentage / 100
        segmentation = SegmentAnything(segment_percentage)
        if session is None:
            session = segmentation.init_onnx(onnx_vit_checkpoint, device)
        if predictor is None:
            predictor = segmentation.init_vit(sam_vit_checkpoint, vit_model_type, device)

        # cut video
        if source_type == "video":
            source = cut_start_video(source, source_start, source_end)

        remove_keys = []
        for key in masks.keys():
            if masks[key].get("start_time") < source_start:
                remove_keys += [key]
                continue
            if masks[key].get("start_time") > source_end:
                remove_keys += [key]
                continue
            if masks[key].get("end_time") > source_end:
                masks[key]["end_time"] = source_end
        else:
            for remove_key in remove_keys:
                masks.pop(remove_key)

        source_media_type = check_media_type(source)
        if source_media_type == "static":
            fps = 0
            frames = [read_image_cv2(source)]
        elif source_media_type == "animated":
            frames, fps = get_frames(video=source, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
        else:
            raise "Source is not detected as image or video"

        orig_height, orig_width, _ = frames[0].shape
        orig_frames = frames.copy()
        frame_batch_size = 50  # for improve remove object to limit VRAM, for CPU slowly, but people have a lot RAM

        if source_media_type == "animated" and retouch_model_type == "improved_retouch_object" and device == "cuda":
            # resize only for VRAM
            gpu_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            # table of gpu memory
            gpu_table = {19: 1280, 7: 720, 6: 640, 2: 320}
            # get user resize for gpu
            max_size = max(val for key, val in gpu_table.items() if key <= gpu_vram)
            print(f"Limit VRAM is {gpu_vram} Gb. Video will resize before {max_size} for max size")
            # Resize frames while maintaining aspect ratio
            resized_frames = []
            for frame in frames:
                h, w, _ = frame.shape
                if max(h, w) > max_size:
                    if h > w:
                        new_h = max_size
                        new_w = int(w * (max_size / h))
                    else:
                        new_w = max_size
                        new_h = int(h * (max_size / w))
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    resized_frames.append(resized_frame)
                else:
                    resized_frames.append(frame)
            frames = resized_frames


        # get segmentation frames as maks
        segmentation.load_models(predictor=predictor, session=session)
        segment_mask = {}

        for key in masks.keys():
            print(f"Processing ID: {key}")
            start_time = masks[key]["start_time"] - source_start
            end_time = masks[key]["end_time"]
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) + 1
            # list of frames of list of one frame
            filter_frames = frames[start_frame:end_frame]
            # set progress bar
            progress_bar = tqdm(total=len(filter_frames), unit='it', unit_scale=True)
            segment_mask[key] = [segmentation.set_obj(point_list=masks[key]["point_list"], frame=filter_frames[0])]
            # update progress bar
            progress_bar.update(1)
            if len(filter_frames) > 1:
                for i, frame in enumerate(filter_frames[1:]):
                    segment_mask[key] += [segmentation.draw_mask_frames(frame=frame)]
                    if segment_mask[key][-1] is None:
                        segment_mask[key] = segment_mask[key][:-1]
                        print(key, "Encountered None mask. Breaking the loop.")
                        break
                    progress_bar.update(1)
            # close progress bar for key
            progress_bar.close()

        if mask_color:
            print("Save mask")
            color = segmentation.hex_to_rgba(mask_color)
            for key in masks.keys():
                os.makedirs(os.path.join(save_dir, key), exist_ok=True)
                start_time = masks[key]["start_time"] - source_start
                end_time = masks[key]["end_time"]
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps) + 1
                # list of frames of list of one frame
                filter_frames = orig_frames[start_frame:end_frame]
                filter_segment_mask = segment_mask[key]
                for i, mask in enumerate(filter_segment_mask):
                    saving_mask = segmentation.apply_mask_on_frame(filter_segment_mask[i], filter_frames[i], color, orig_width, orig_height)
                    saving_mask.save(os.path.join(save_dir, key, f"{i}.png"))

            print("Mask save is finished. Open folder")
            if os.path.exists(save_dir):
                if sys.platform == 'win32':
                    # Open folder for Windows
                    subprocess.Popen(r'explorer /select,"{}"'.format(save_dir))
                elif sys.platform == 'darwin':
                    # Open folder for MacOS
                    subprocess.Popen(['open', save_dir])
                elif sys.platform == 'linux':
                    # Open folder for Linux
                    subprocess.Popen(['xdg-open', save_dir])

        del segmentation, predictor, session

        if retouch_model_type is None:
            return save_dir

        if source_media_type == "animated" and retouch_model_type == "improved_retouch_object":
            # raft
            retouch_processor = VideoRemoveObjectProcessor(device, model_raft_things_path, model_recurrent_flow_path, model_pro_painter_path)

            for key in masks.keys():
                start_time = masks[key]["start_time"] - source_start
                start_frame = int(start_time * fps)
                end_frame = start_frame + len(segment_mask[key])
                # transfer to pil list of frames of list of one frame
                filter_frames = [retouch_processor.transfer_to_pil(f) for f in frames[start_frame:end_frame]]
                filter_frames, size, out_size = retouch_processor.resize_frames(filter_frames)
                width, height = size
                # processing video with batch size because of limit VRAM
                if len(segment_mask[key]) < 3:
                    continue
                comp_frames_bgr = []
                for i in range(0, len(filter_frames), frame_batch_size):
                    if len(filter_frames) - i < 3:  # not less than 3 frames for batch
                        continue
                    flow_masks, masks_dilated = retouch_processor.read_retouch_mask(segment_mask[key][i: i + frame_batch_size], width, height, kernel_size=blur)
                    update_frames, flow_masks, masks_dilated, masked_frame_for_save, frames_inp = retouch_processor.process_frames_with_retouch_mask(frames=filter_frames[i: i + frame_batch_size], masks_dilated=masks_dilated, flow_masks=flow_masks, height=height, width=width)
                    comp_frames = retouch_processor.process_video_with_mask(update_frames, masks_dilated, flow_masks, frames_inp, width, height, use_half=use_half)
                    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
                    comp_frames_bgr += [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in comp_frames]
                # update list of frames of list of one frame
                frames[start_frame:end_frame] = comp_frames_bgr
                # upscale to restore size as this retouch with gpu require a lot VRAM
                filter_segment_mask = segment_mask[key]
                filter_orig_frame = orig_frames[start_frame:end_frame]
                for j in range(len(filter_segment_mask)):
                    orig_frames[start_frame + j] = upscale_retouch_frame(mask=filter_segment_mask[j], frame=comp_frames_bgr[j], original_frame=filter_orig_frame[j], width=orig_width, height=orig_height)

            if upscale:
                frames = orig_frames  # set restored frames
        else:
            # retouch
            model_retouch = InpaintModel(model_path=model_retouch_path)

            for key in masks.keys():
                start_time = masks[key]["start_time"] - source_start
                end_time = masks[key]["end_time"]
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps) + 1
                # list of frames of list of one frame
                filter_frames = frames[start_frame:end_frame]
                segment_mask_pil = [convert_colored_mask_thickness_cv2(mask) for mask in segment_mask[key]]
                # set progress bar
                progress_bar = tqdm(total=len(segment_mask_pil), unit='it', unit_scale=True)
                for i in range(len(segment_mask_pil)):
                    frame_pil = convert_cv2_to_pil(filter_frames[i])
                    # retouch_frame
                    retouch_frame = process_retouch(img=frame_pil, mask=segment_mask_pil[i], model=model_retouch)
                    # update frame
                    frames[start_frame + i] = pil_to_cv2(retouch_frame)
                    progress_bar.update(1)
                # close progress bar for key
                progress_bar.close()


        for i, frame in enumerate(frames):
            save_name = "retouch_image_%s.png" % i
            save_image_cv2(os.path.join(save_dir, save_name), frame)

        if source_media_type == "animated":
            # get saved file as merge frames to video
            video_name = save_video_from_frames(frame_names="retouch_image_%d.png", save_path=save_dir, fps=fps)
            # get audio from video target
            audio_file_name = extract_audio_from_video(source, save_dir)
            # combine audio and video
            save_name = save_video_with_audio(os.path.join(save_dir, video_name), os.path.join(save_dir, str(audio_file_name)), save_dir)

        for f in os.listdir(save_dir):
            if save_name == f:
                save_name = os.path.join(save_dir, f)
            else:
                if os.path.isfile(os.path.join(save_dir, f)):
                    os.remove(os.path.join(save_dir, f))
                elif os.path.isdir(os.path.join(save_dir, f)):
                    pass

        for f in os.listdir(TMP_FOLDER):
            os.remove(os.path.join(TMP_FOLDER, f))

        return save_name


class VideoEdit:
    """Edit video"""
    @staticmethod
    def main_video_work(output, source, enhancer, enhancer_background, is_get_frames, media_start=0, media_end=0):
        save_dir = os.path.join(output, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        media_start = float(media_start)
        media_end = float(media_end)

        import time
        time.sleep(5)

        enhancer_background = "realesrgan" if enhancer_background else None
        enhancer = "gfpgan" if enhancer_background else enhancer
        audio_file_name = extract_audio_from_video(source, save_dir)

        if enhancer:
            cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
            if torch.cuda.is_available() and not cpu:
                print("Processing will run on GPU")
                device = "cuda"
            else:
                print("Processing will run on CPU")
                device = "cpu"

            print("Get audio and video frames")
            _, fps = get_frames(video=source, rotate=False, crop=[0, -1, 0, -1], resize_factor=1)
            if check_media_type(source) == "animated":
                source = cut_start_video(source, media_start, media_end)
                name_enhancer = 'video_enhanced.mp4'
                enhanced_path = os.path.join(output, 'temp_' + name_enhancer)
                print("Starting improve video")
                enhanced_video = face_enhancer(source, method=enhancer, bg_upsampler=enhancer_background, device=device)
                imageio.mimsave(enhanced_path, enhanced_video, fps=float(fps))
                save_name = save_video_with_audio(enhanced_path, os.path.join(save_dir, audio_file_name), save_dir)
            else:
                save_name = 'image_enhanced.png'
                print("Starting improve image")
                enhanced_image = face_enhancer(source, method=enhancer, bg_upsampler=enhancer_background, device=device)
                saved_file = os.path.join(save_dir, save_name)
                imageio.imsave(saved_file, enhanced_image[0])

            for f in os.listdir(save_dir):
                if save_name == f:
                    save_name = os.path.join(save_dir, f)
                else:
                    if os.path.isfile(os.path.join(save_dir, f)):
                        os.remove(os.path.join(save_dir, f))
                    elif os.path.isdir(os.path.join(save_dir, f)):
                        shutil.rmtree(os.path.join(save_dir, f))

            for f in os.listdir(TMP_FOLDER):
                os.remove(os.path.join(TMP_FOLDER, f))

            return save_name

        if is_get_frames:
            source = cut_start_video(source, media_start, media_end)
            video_to_frames(source, save_dir)
            for f in os.listdir(TMP_FOLDER):
                os.remove(os.path.join(TMP_FOLDER, f))

            if os.path.exists(save_dir):
                if sys.platform == 'win32':
                    # Open folder for Windows
                    subprocess.Popen(r'explorer /select,"{}"'.format(save_dir))
                elif sys.platform == 'darwin':
                    # Open folder for MacOS
                    subprocess.Popen(['open', save_dir])
                elif sys.platform == 'linux':
                    # Open folder for Linux
                    subprocess.Popen(['xdg-open', save_dir])

            print(f"You will find images in {save_dir}")
            return os.path.join(save_dir, audio_file_name)

        print("Error... Not get parameters for video edit!")
        return "Error"

    @staticmethod
    def main_merge_frames(output, source_folder, audio_path, fps):
        save_dir = os.path.join(output, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        # get saved file as merge frames to video
        video_name = save_video_from_frames(frame_names="%d.png", save_path=source_folder, fps=fps)

        # combine audio and video
        video_name = save_video_with_audio(os.path.join(source_folder, video_name), str(audio_path), save_dir)

        for f in os.listdir(TMP_FOLDER):
            os.remove(os.path.join(TMP_FOLDER, f))

        return os.path.join(save_dir, video_name)


class GetSegment:
    @staticmethod
    def load_model():
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True

        if torch.cuda.is_available() and not use_cpu:
            print("Processing will run on GPU")
            device = "cuda"
        else:
            print("Processing will run on CPU")
            device = "cpu"

        checkpoint_dir = "checkpoints"
        os.environ['TORCH_HOME'] = os.path.join(DEEPFAKE_MODEL_FOLDER, checkpoint_dir)
        checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, checkpoint_dir)

        if device == "cuda":
            model_type = "vit_h"

            sam_vit_checkpoint = os.path.join(checkpoint_dir_full, 'sam_vit_h.pth')
            if not os.path.exists(sam_vit_checkpoint):
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_h.pth"]
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
            else:
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_h.pth"]
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_dir_full, 'vit_h_quantized.onnx')
            if not os.path.exists(onnx_vit_checkpoint):
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_h_quantized.onnx"]
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
            else:
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_h_quantized.onnx"]
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
        else:
            model_type = "vit_b"

            sam_vit_checkpoint = os.path.join(checkpoint_dir_full, 'sam_vit_b.pth')
            if not os.path.exists(sam_vit_checkpoint):
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_b.pth"]
                download_model(sam_vit_checkpoint, link_sam_vit_checkpoint)
            else:
                link_sam_vit_checkpoint = file_deepfake_config["checkpoints"]["sam_vit_b.pth"]
                check_download_size(sam_vit_checkpoint, link_sam_vit_checkpoint)

            onnx_vit_checkpoint = os.path.join(checkpoint_dir_full, 'vit_b_quantized.onnx')
            if not os.path.exists(onnx_vit_checkpoint):
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_b_quantized.onnx"]
                download_model(onnx_vit_checkpoint, link_onnx_vit_checkpoint)
            else:
                link_onnx_vit_checkpoint = file_deepfake_config["checkpoints"]["vit_b_quantized.onnx"]
                check_download_size(onnx_vit_checkpoint, link_onnx_vit_checkpoint)

        segmentation = SegmentAnything()
        predictor = segmentation.init_vit(sam_vit_checkpoint, model_type, device)
        session = segmentation.init_onnx(onnx_vit_checkpoint, device)

        return {"predictor": predictor, "session": session}

    @staticmethod
    def get_segment_mask_file(predictor, session, source: str, point_list: list):
        # set time sleep else file will not still loaded
        import time
        time.sleep(5)
        # read frame
        frame = read_image_cv2(source)
        # segmentation
        segmentation = SegmentAnything()
        mask = segmentation.draw_mask(predictor=predictor, session=session, frame=frame, point_list=point_list)
        # save mask in tmp?
        mask_file_name = save_colored_mask_cv2(TMP_FOLDER, mask)
        return mask_file_name
