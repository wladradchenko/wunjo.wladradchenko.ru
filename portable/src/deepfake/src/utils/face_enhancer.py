import os
import json
import torch 

# from gfpgan import GFPGANer
from src.utils.gfpganer import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2
from backend.folders import DEEPFAKE_MODEL_FOLDER
from backend.download import download_model, check_download_size

import cv2


def enhancer(images, method='gfpgan', bg_upsampler='realesrgan'):
    print('face enhancer....')
    if os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    if os.path.isfile(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json')):
        with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'r', encoding="utf8") as file:
            config_deepfake = json.load(file)
    else:
        raise "[Error] Config file deepfake.json is not exist"

    local_model_path = os.path.join(DEEPFAKE_MODEL_FOLDER, './gfpgan/weights')

    # ------------------------ set up GFPGAN restorer ------------------------

    if method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = config_deepfake["gfpgan"]["GFPGANv1.4.pth"]
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = config_deepfake["gfpgan"]["RestoreFormer.pth"]
    elif method == 'codeformer':
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = config_deepfake["gfpgan"]["codeformer.pth"]
    else:
        raise ValueError(f'Wrong model version {method}.')


    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            # from realesrgan import RealESRGANer
            from src.utils.realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            os.path.join(DEEPFAKE_MODEL_FOLDER, './gfpgan/weights')
            model_realesrgan_path = os.path.join(local_model_path, "RealESRGAN_x2plus.pth" + '.pth')
            url_realesrgan = config_deepfake["gfpgan"]["RealESRGAN_x2plus.pth"]

            if not os.path.isfile(model_realesrgan_path):
                # download pre-trained models from url
                model_realesrgan_path = url_realesrgan

            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=model_realesrgan_path,
                root_dir=local_model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join(local_model_path, model_name + '.pth')

    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        root_dir=local_model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    restored_img = [] 
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        
        # restore faces and background if necessary
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True)
        
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        
        restored_img += [r_img]
       
    return restored_img
