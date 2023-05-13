import torch
from time import  strftime
import os, sys, time
from argparse import Namespace

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
talker_root_path = f"{root_path}/talker"
sys.path.insert(0, f"{root_path}/talker")

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

sys.path.pop(0)

from cog import Input
import requests
import shutil
import json

from backend.folders import TALKER_MODEL_FOLDER, TMP_FOLDER
from backend.extension import download_model, unzip, check_download_size


TALKER_JSON_URL = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/talker.json"


def get_config_talker() -> dict:
    try:
        response = requests.get(TALKER_JSON_URL)
        with open(os.path.join(TALKER_MODEL_FOLDER, 'talker.json'), 'wb') as file:
            file.write(response.content)
    except:
        print("Not internet connection")
    finally:
        if not os.path.isfile(os.path.join(TALKER_MODEL_FOLDER, 'talker.json')):
            talker = {}
        else:
            with open(os.path.join(TALKER_MODEL_FOLDER, 'talker.json'), 'r', encoding="utf8") as file:
                talker = json.load(file)

    return talker


file_talker_config = get_config_talker()


def main_talker(source_image: str, driven_audio: str, talker_dir: str, cpu: bool = True, still: bool = True, face_fields: list = None,
                enhancer: str = Input(description="Choose a face enhancer",choices=["gfpgan", "RestoreFormer"],default="gfpgan",),
                preprocess: str = Input(description="how to preprocess the images", choices=["crop", "resize", "full"], default="full",),
                expression_scale=1.0, input_yaw=None, input_pitch=None, input_roll=None, background_enhancer=None):

    if file_talker_config == {}:
        raise "[Error] Config file talker.json is not exist"
    # torch.backends.cudnn.enabled = False
    args = load_default()
    args.checkpoint_dir = "checkpoints"
    args.source_image = source_image
    args.driven_audio = driven_audio
    args.result_dir = talker_dir
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

    current_root_path = talker_root_path  # os.path.split(current_code_path)[0]
    model_user_path = TALKER_MODEL_FOLDER

    os.environ['TORCH_HOME'] = os.path.join(model_user_path, args.checkpoint_dir)
    checkpoint_dir_full = os.path.join(model_user_path, args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir_full):
        os.makedirs(checkpoint_dir_full)

    path_of_lm_croper = os.path.join(checkpoint_dir_full, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(path_of_lm_croper):
        link_of_lm_croper = file_talker_config["checkpoints"]["shape_predictor_68_face_landmarks.dat"]
        download_model(path_of_lm_croper, link_of_lm_croper)
    else:
        link_of_lm_croper = file_talker_config["checkpoints"]["shape_predictor_68_face_landmarks.dat"]
        check_download_size(path_of_lm_croper, link_of_lm_croper)

    path_of_net_recon_model = os.path.join(checkpoint_dir_full, 'epoch_20.pth')
    if not os.path.exists(path_of_net_recon_model):
        link_of_net_recon_model = file_talker_config["checkpoints"]["epoch_20.pth"]
        download_model(path_of_net_recon_model, link_of_net_recon_model)
    else:
        link_of_net_recon_model = file_talker_config["checkpoints"]["epoch_20.pth"]
        check_download_size(path_of_net_recon_model, link_of_net_recon_model)

    dir_of_BFM_fitting = os.path.join(checkpoint_dir_full, 'BFM_Fitting')
    args.bfm_folder = dir_of_BFM_fitting
    if not os.path.exists(dir_of_BFM_fitting):
        link_of_BFM_fitting = file_talker_config["checkpoints"]["BFM_Fitting.zip"]
        download_model(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), link_of_BFM_fitting)
        unzip(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), checkpoint_dir_full)
    else:
        link_of_BFM_fitting = file_talker_config["checkpoints"]["BFM_Fitting.zip"]
        check_download_size(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), link_of_BFM_fitting)
        if not os.listdir(dir_of_BFM_fitting):
            unzip(os.path.join(checkpoint_dir_full, 'BFM_Fitting.zip'), checkpoint_dir_full)

    dir_of_hub_models = os.path.join(checkpoint_dir_full, 'hub')
    if not os.path.exists(dir_of_hub_models):
        link_of_hub_models = file_talker_config["checkpoints"]["hub.zip"]
        download_model(os.path.join(checkpoint_dir_full, 'hub.zip'), link_of_hub_models)
        unzip(os.path.join(checkpoint_dir_full, 'hub.zip'), checkpoint_dir_full)
    else:
        link_of_hub_models = file_talker_config["checkpoints"]["hub.zip"]
        check_download_size(os.path.join(checkpoint_dir_full, 'hub.zip'), link_of_hub_models)
        if not os.listdir(dir_of_hub_models):
            unzip(os.path.join(checkpoint_dir_full, 'hub.zip'), checkpoint_dir_full)

    wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
    if not os.path.exists(wav2lip_checkpoint):
        link_wav2lip_checkpoint = file_talker_config["checkpoints"]["wav2lip.pth"]
        download_model(wav2lip_checkpoint, link_wav2lip_checkpoint)
    else:
        link_wav2lip_checkpoint = file_talker_config["checkpoints"]["wav2lip.pth"]
        check_download_size(wav2lip_checkpoint, link_wav2lip_checkpoint)

    audio2pose_checkpoint = os.path.join(checkpoint_dir_full, 'auido2pose_00140-model.pth')
    if not os.path.exists(audio2pose_checkpoint):
        link_audio2pose_checkpoint = file_talker_config["checkpoints"]["auido2pose_00140-model.pth"]
        download_model(audio2pose_checkpoint, link_audio2pose_checkpoint)
    else:
        link_audio2pose_checkpoint = file_talker_config["checkpoints"]["auido2pose_00140-model.pth"]
        check_download_size(audio2pose_checkpoint, link_audio2pose_checkpoint)
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

    audio2exp_checkpoint = os.path.join(checkpoint_dir_full, 'auido2exp_00300-model.pth')
    if not os.path.exists(audio2exp_checkpoint):
        link_audio2exp_checkpoint = file_talker_config["checkpoints"]["auido2exp_00300-model.pth"]
        download_model(audio2exp_checkpoint, link_audio2exp_checkpoint)
    else:
        link_audio2exp_checkpoint = file_talker_config["checkpoints"]["auido2exp_00300-model.pth"]
        check_download_size(audio2exp_checkpoint, link_audio2exp_checkpoint)
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(checkpoint_dir_full, 'facevid2vid_00189-model.pth.tar')
    if not os.path.exists(free_view_checkpoint):
        link_free_view_checkpoint = file_talker_config["checkpoints"]["facevid2vid_00189-model.pth.tar"]
        download_model(free_view_checkpoint, link_free_view_checkpoint)
    else:
        link_free_view_checkpoint = file_talker_config["checkpoints"]["facevid2vid_00189-model.pth.tar"]
        check_download_size(free_view_checkpoint, link_free_view_checkpoint)

    if args.preprocess == 'full':
        mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00109-model.pth.tar')
        if not os.path.exists(mapping_checkpoint):
            link_mapping_checkpoint = file_talker_config["checkpoints"]["mapping_00109-model.pth.tar"]
            download_model(mapping_checkpoint, link_mapping_checkpoint)
        else:
            link_mapping_checkpoint = file_talker_config["checkpoints"]["mapping_00109-model.pth.tar"]
            check_download_size(mapping_checkpoint, link_mapping_checkpoint)
        facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
    else:
        mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00229-model.pth.tar')
        if not os.path.exists(mapping_checkpoint):
            link_mapping_checkpoint = file_talker_config["checkpoints"]["mapping_00229-model.pth.tar"]
            download_model(mapping_checkpoint, link_mapping_checkpoint)
        else:
            link_mapping_checkpoint = file_talker_config["checkpoints"]["mapping_00229-model.pth.tar"]
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

    animate_from_coeff.generate(data, save_dir, pic_path, crop_info, enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess)

    mp4_path = None
    for f in os.listdir(save_dir):
        if "enhanced.mp4" in f:
            mp4_path = os.path.join(save_dir, f)
        else:
            if os.path.isfile(os.path.join(save_dir, f)):
                os.remove(os.path.join(save_dir, f))
            elif os.path.isdir(os.path.join(save_dir, f)):
                shutil.rmtree(os.path.join(save_dir, f))

    for f in os.listdir(TMP_FOLDER):
        os.remove(os.path.join(TMP_FOLDER, f))

    return mp4_path

def load_default():
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
