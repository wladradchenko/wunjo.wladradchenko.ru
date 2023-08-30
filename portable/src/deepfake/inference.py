import os
import sys
import torch
import imageio
from time import strftime
from argparse import Namespace

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deepfake_root_path = os.path.join(root_path, "deepfake")
sys.path.insert(0, deepfake_root_path)

"""SadTalker"""
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
"""SadTalker"""
"""Wav22Lip"""
from src.wav2mel import MelProcessor
from src.utils.videoio import save_video_with_audio, cut_start_video
from src.utils.face_enhancer import enhancer as face_enhancer
from src.utils.video2fake import get_frames, GenerateFakeVideo2Lip
"""Wav22Lip"""

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
        print("Not internet connection")
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
                          expression_scale=1.0, input_yaw=None, input_pitch=None, input_roll=None, background_enhancer=None):

        if file_deepfake_config == {}:
            raise "[Error] Config file deepfake.json is not exist"
        # torch.backends.cudnn.enabled = False
        args = AnimationFaceTalk.load_img_default()
        args.checkpoint_dir = "checkpoints"
        args.source_image = source_image
        args.driven_audio = driven_audio
        args.result_dir = deepfake_dir

        cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        print("cpu", cpu)
        if torch.cuda.is_available() and not cpu:
            args.device = "cuda"
        else:
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
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

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
        print(path_of_net_recon_model)
        preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device, face_fields)

        print(audio2pose_checkpoint)
        print(audio2exp_checkpoint)
        audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path, wav2lip_checkpoint, device)

        print(free_view_checkpoint)
        print(mapping_checkpoint)
        animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, device)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir)
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir)
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # 3dface render
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

        # coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, input_yaw_list, input_pitch_list, input_roll_list, expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess)
        mp4_path = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess)

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
    def load_img_default():
        return Namespace(
            pose_style=0,
            batch_size=2,
            ref_eyeblink=None,
            ref_pose=None,
            face3dvis=False,
            net_recon="resnet50",
            init_path=None,
            use_last_fc=False,
            bfm_folder="./checkpoints/BFM_Fitting/",
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
                            box: list = [-1, -1, -1, -1], background_enhancer: str = None, video_start: float = 0):
        args = AnimationMouthTalk.load_video_default()
        args.checkpoint_dir = "checkpoints"
        args.result_dir = deepfake_dir
        args.face = face
        args.audio = audio
        args.static = static  # ok, we will check what it is video, not img
        args.box = box  # a constant bounding box for the face. Use only as a last resort if the face is not detected.
        # Also (about box), might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
        args.face_fields = face_fields
        args.enhancer = enhancer
        args.background_enhancer = background_enhancer

        cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        print("cpu", cpu)
        if torch.cuda.is_available() and not cpu:
            args.device = "cuda"
        else:
            args.device = "cpu"

        args.background_enhancer = "realesrgan" if background_enhancer else None

        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        current_root_path = deepfake_root_path  # os.path.split(current_code_path)[0]
        model_user_path = DEEPFAKE_MODEL_FOLDER

        os.environ['TORCH_HOME'] = os.path.join(model_user_path, args.checkpoint_dir)
        checkpoint_dir_full = os.path.join(model_user_path, args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir_full):
            os.makedirs(checkpoint_dir_full)

        wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
        if not os.path.exists(wav2lip_checkpoint):
            link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
            download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
        else:
            link_wav2lip_checkpoint = file_deepfake_config["checkpoints"]["wav2lip.pth"]
            check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)

        # If video_start is not 0 when cut video from start
        if video_start != 0:
            args.face = cut_start_video(args.face, video_start)

        # get video frames
        frames, fps = get_frames(video=args.face, rotate=args.rotate, crop=args.crop, resize_factor=args.resize_factor)
        # get mel of audio
        mel_processor = MelProcessor(args=args, save_output=save_dir, fps=fps)
        mel_chunks = mel_processor.process()
        # test
        full_frames = frames[:len(mel_chunks)]
        batch_size = args.wav2lip_batch_size
        wav2lip = GenerateFakeVideo2Lip(DEEPFAKE_MODEL_FOLDER)
        wav2lip.face_fields = args.face_fields
        print("Face detected start")
        gen = wav2lip.datagen(
            full_frames.copy(), mel_chunks, args.box, args.static, args.img_size, args.wav2lip_batch_size,
            args.device, args.pads, args.nosmooth
        )
        # load wav2lip
        print("Create mouth move start")
        wav2lip_processed_video = wav2lip.generate_video_from_chunks(gen, mel_chunks, batch_size, wav2lip_checkpoint, args.device, save_dir, fps)
        if wav2lip_processed_video is None:
            return
        wav2lip_result_video = wav2lip_processed_video
        # after face or background enchanter
        if enhancer:
            video_name_enhancer = 'wav2lip_video_enhanced.mp4'
            enhanced_path = os.path.join(save_dir, 'temp_' + video_name_enhancer)
            enhanced_images = face_enhancer(wav2lip_processed_video, method=enhancer, bg_upsampler=background_enhancer)
            imageio.mimsave(enhanced_path, enhanced_images, fps=float(fps))
            wav2lip_result_video = enhanced_path

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
