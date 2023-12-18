import os
import sys
import json
import torch
import requests
from denoiser.pretrained import MASTER_64_URL

from backend.folders import DEEPFAKE_MODEL_FOLDER, VOICE_FOLDER, MEDIA_FOLDER, RTVC_VOICE_FOLDER, SETTING_FOLDER, AVATAR_FOLDER, CUSTOM_VOICE_FOLDER
from backend.download import get_nested_url

# Params
SUB_CUSTOM_VOICE = "(Custom)"  # sub name for custom voice to do uniq
DEEPFAKE_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/deepfake.json"
VOICE_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/voice.json"
RTVC_MODELS_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/rtvc.json"


def get_deepfake_config() -> dict:
    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
        try:
            response = requests.get(DEEPFAKE_JSON_URL)
            with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'wb') as file:
                file.write(response.content)
        except:
            print("Not internet connection to get actual versions of deepfake models")

    if not os.path.isfile(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json')):
        deepfake = {}
    else:
        with open(os.path.join(DEEPFAKE_MODEL_FOLDER, 'deepfake.json'), 'r', encoding="utf8") as file:
            deepfake = json.load(file)

    return deepfake


def get_diffusion_config():
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
    return custom_diffusion


def cover_windows_media_path_from_str(path_str: str):
    path_sys = MEDIA_FOLDER
    for folder_or_file in path_str.split("/"):
        if folder_or_file:
            path_sys = os.path.join(path_sys, folder_or_file)
    return path_sys


def _get_avatars(voices: dict) -> None:
    if not os.path.exists(AVATAR_FOLDER):
        os.makedirs(AVATAR_FOLDER)

    for avatar in voices.keys():
        # Check if the avatar file exists
        if not os.path.exists(os.path.join(AVATAR_FOLDER, f"{avatar}.png")):
            # If it doesn't exist, download it from the URL
            request_png = requests.get(voices[avatar]["avatar_download"])
            with open(os.path.join(AVATAR_FOLDER, f"{avatar}.png"), 'wb') as file:
                file.write(request_png.content)


def _create_config_voice(voices: dict) -> dict:
    general = {
        "general": {
            "device": "cuda",
            "pause_type": "silence",
            "sample_rate": 22050,

            "logging": {
                "log_level": "INFO",
                "log_file": None
            }
        }
    }
    return {**general, **voices}


def get_tts_config():
    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
        try:
            response = requests.get(VOICE_JSON_URL)
            with open(os.path.join(VOICE_FOLDER, 'voice.json'), 'wb') as file:
                file.write(response.content)
        except:
            print("Not internet connection to update available server voice list")

    if not os.path.isfile(os.path.join(VOICE_FOLDER, 'voice.json')):
        voices_config = {}
    else:
        with open(os.path.join(VOICE_FOLDER, 'voice.json'), 'r', encoding="utf8") as file:
            voices_config = json.load(file)

    _get_avatars(voices=voices_config)

    if voices_config.get("Unknown") is not None:
       voices_config.pop("Unknown")

    voice_names = []
    for key in voices_config.keys():
        if voices_config[key].get("engine") and voices_config[key].get("vocoder"):
            tacotron2 = voices_config[key]["engine"]["tacotron2"]["model_path"]
            full_tacotron2_path = cover_windows_media_path_from_str(tacotron2)
            voices_config[key]["engine"]["tacotron2"]["model_path"] = full_tacotron2_path

            waveglow = voices_config[key]["vocoder"]["waveglow"]["model_path"]
            full_waveglow_path = cover_windows_media_path_from_str(waveglow)
            voices_config[key]["vocoder"]["waveglow"]["model_path"] = full_waveglow_path

            voice_names += [key]

    return _create_config_voice(voices_config), voice_names


def get_custom_tts_config():
    if not os.path.isfile(os.path.join(CUSTOM_VOICE_FOLDER, 'custom_user.json')):
        return {}, []
    else:
        with open(os.path.join(CUSTOM_VOICE_FOLDER, 'custom_user.json'), 'r', encoding="utf8") as file:
            voices = json.load(file)
            file_custom_voice_config = {}
            custom_voice_names = []
            for key in voices.keys():
                voices[key]['avatar_download'] = None
                voices[key]['checkpoint_download'] = None
                voices[key]['waveglow_download'] = None

                try:
                    tacotron2 = voices[key]["engine"]["tacotron2"]["model_path"]
                    full_tacotron2_path = cover_windows_media_path_from_str(tacotron2)
                    voices[key]["engine"]["tacotron2"]["model_path"] = full_tacotron2_path

                    waveglow = voices[key]["vocoder"]["waveglow"]["model_path"]
                    full_waveglow_path = cover_windows_media_path_from_str(waveglow)
                    voices[key]["vocoder"]["waveglow"]["model_path"] = full_waveglow_path
                    if not os.path.exists(full_tacotron2_path) or not os.path.exists(full_waveglow_path):
                        print(f"Waveglow or Tactoron2 models don't exist for user voice {key}")
                    else:
                        new_key = key + " " + SUB_CUSTOM_VOICE
                        file_custom_voice_config[new_key] = voices[key]
                        custom_voice_names += [new_key]
                except Exception as err:
                    print(f"Error when load custom voices ... {err}")
        return file_custom_voice_config, custom_voice_names


def get_rtvc_config() -> dict:
    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
        try:
            response = requests.get(RTVC_MODELS_JSON_URL)
            with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'wb') as file:
                file.write(response.content)
        except:
            print("Not internet connection to get clone voice models")

    if not os.path.isfile(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json')):
        rtvc_models = {}
    else:
        with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'r', encoding="utf8") as file:
            rtvc_models = json.load(file)
    return rtvc_models


"""INSPECT MODELS CONFIG"""


def inspect_face_animation_config() -> tuple:
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()

    if not deepfake_config:
        return offline_status, models_is_not_exist

    checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
    path_of_net_recon_model = os.path.join(checkpoint_dir_full, 'epoch_20.pth')
    if not os.path.exists(path_of_net_recon_model):
        link_of_net_recon_model = get_nested_url(deepfake_config, ["checkpoints", "epoch_20.pth"])
        models_is_not_exist += [(path_of_net_recon_model, link_of_net_recon_model)]

    dir_of_BFM_fitting = os.path.join(checkpoint_dir_full, 'BFM_Fitting')
    if not os.path.exists(dir_of_BFM_fitting):
        link_of_BFM_fitting = get_nested_url(deepfake_config, ["checkpoints", "BFM_Fitting.zip"])
        models_is_not_exist += [(dir_of_BFM_fitting, link_of_BFM_fitting)]

    dir_of_hub_models = os.path.join(checkpoint_dir_full, 'hub')
    if not os.path.exists(dir_of_hub_models):
        link_of_hub_models = get_nested_url(deepfake_config, ["checkpoints", "hub.zip"])
        models_is_not_exist += [(dir_of_hub_models, link_of_hub_models)]

    wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
    if not os.path.exists(wav2lip_checkpoint):
        link_wav2lip_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "wav2lip.pth"])
        models_is_not_exist += [(wav2lip_checkpoint, link_wav2lip_checkpoint)]

    audio2pose_checkpoint = os.path.join(checkpoint_dir_full, 'auido2pose_00140-model.pth')
    if not os.path.exists(audio2pose_checkpoint):
        link_audio2pose_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "auido2pose_00140-model.pth"])
        models_is_not_exist += [(audio2pose_checkpoint, link_audio2pose_checkpoint)]

    audio2exp_checkpoint = os.path.join(checkpoint_dir_full, 'auido2exp_00300-model.pth')
    if not os.path.exists(audio2exp_checkpoint):
        link_audio2exp_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "auido2exp_00300-model.pth"])
        models_is_not_exist += [(audio2exp_checkpoint, link_audio2exp_checkpoint)]

    free_view_checkpoint = os.path.join(checkpoint_dir_full, 'facevid2vid_00189-model.pth.tar')
    if not os.path.exists(free_view_checkpoint):
        link_free_view_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "facevid2vid_00189-model.pth.tar"])
        models_is_not_exist += [(free_view_checkpoint, link_free_view_checkpoint)]

    mapping_checkpoint = os.path.join(checkpoint_dir_full, 'mapping_00109-model.pth.tar')
    if not os.path.exists(mapping_checkpoint):
        link_mapping_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "mapping_00109-model.pth.tar"])
        models_is_not_exist += [(mapping_checkpoint, link_mapping_checkpoint)]

    mapping_checkpoint_crop = os.path.join(checkpoint_dir_full, 'mapping_00229-model.pth.tar')
    if not os.path.exists(mapping_checkpoint_crop):
        link_mapping_checkpoint_crop = get_nested_url(deepfake_config, ["checkpoints", "mapping_00229-model.pth.tar"])
        models_is_not_exist += [(mapping_checkpoint_crop, link_mapping_checkpoint_crop)]

    model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "model")
    face_recognition = os.path.join(model_dir_full, 'buffalo_l')
    if not os.path.exists(face_recognition):
        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
        name = 'buffalo_l'
        link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
        models_is_not_exist += [(face_recognition, link_face_recognition)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_mouth_animation_config() -> tuple:
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()

    if not deepfake_config:
        return offline_status, models_is_not_exist

    checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
    wav2lip_checkpoint = os.path.join(checkpoint_dir_full, 'wav2lip.pth')
    if not os.path.exists(wav2lip_checkpoint):
        link_wav2lip_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "wav2lip.pth"])
        models_is_not_exist += [(wav2lip_checkpoint, link_wav2lip_checkpoint)]

    emo2lip_checkpoint = os.path.join(checkpoint_dir_full, 'emo2lip.pth')
    if not os.path.exists(emo2lip_checkpoint):
        link_emo2lip_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "emo2lip.pth"])
        models_is_not_exist += [(emo2lip_checkpoint, link_emo2lip_checkpoint)]

    model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "models")
    face_recognition = os.path.join(model_dir_full, 'buffalo_l')
    if not os.path.exists(face_recognition):
        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
        name = 'buffalo_l'
        link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
        models_is_not_exist += [(face_recognition, link_face_recognition)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_face_swap_config():
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()

    if not deepfake_config:
        return offline_status, models_is_not_exist

    checkpoint_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
    faceswap_checkpoint = os.path.join(checkpoint_dir_full, 'faceswap.onnx')
    if not os.path.exists(faceswap_checkpoint):
        link_faceswap_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "faceswap.onnx"])
        models_is_not_exist += [(faceswap_checkpoint, link_faceswap_checkpoint)]

    model_dir_full = os.path.join(DEEPFAKE_MODEL_FOLDER, "models")
    face_recognition = os.path.join(model_dir_full, 'buffalo_l')
    if not os.path.exists(face_recognition):
        BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
        name = 'buffalo_l'
        link_face_recognition = "%s/%s.zip" % (BASE_REPO_URL, name)
        models_is_not_exist += [(face_recognition, link_face_recognition)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_retouch_config():
    # retouch and remove objects
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()

    if not deepfake_config:
        return offline_status, models_is_not_exist

    checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
    model_pro_painter_path = os.path.join(checkpoint_folder, "ProPainter.pth")
    if not os.path.exists(model_pro_painter_path):
        link_pro_painter = get_nested_url(deepfake_config, ["checkpoints", "ProPainter.pth"])
        models_is_not_exist += [(model_pro_painter_path, link_pro_painter)]

    model_raft_things_path = os.path.join(checkpoint_folder, "raft-things.pth")
    if not os.path.exists(model_raft_things_path):
        link_raft_things = get_nested_url(deepfake_config, ["checkpoints", "raft-things.pth"])
        models_is_not_exist += [(model_raft_things_path, link_raft_things)]

    model_recurrent_flow_path = os.path.join(checkpoint_folder, "recurrent_flow_completion.pth")
    if not os.path.exists(model_recurrent_flow_path):
        link_recurrent_flow = get_nested_url(deepfake_config, ["checkpoints", "recurrent_flow_completion.pth"])
        models_is_not_exist += [(model_recurrent_flow_path, link_recurrent_flow)]

    model_retouch_face_path = os.path.join(checkpoint_folder, "retouch_face.pth")
    if not os.path.exists(model_retouch_face_path):
        link_model_retouch_face = get_nested_url(deepfake_config, ["checkpoints", f"retouch_face.pth"])
        models_is_not_exist += [(model_retouch_face_path, link_model_retouch_face)]

    model_retouch_object_path = os.path.join(checkpoint_folder, "retouch_object.pth")
    if not os.path.exists(model_retouch_object_path):
        link_model_retouch_object = get_nested_url(deepfake_config, ["checkpoints", f"retouch_object.pth"])
        models_is_not_exist += [(model_retouch_object_path, link_model_retouch_object)]

    # Inspect models by GPU or CPU
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if not use_cpu:
        # Large model
        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        if not os.path.exists(sam_vit_checkpoint):
            link_sam_vit_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "sam_vit_h.pth"])
            models_is_not_exist += [(sam_vit_checkpoint, link_sam_vit_checkpoint)]

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        if not os.path.exists(onnx_vit_checkpoint):
            link_onnx_vit_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "vit_h_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_checkpoint, link_onnx_vit_checkpoint)]
    else:
        # Small model
        sam_vit_small_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        if not os.path.exists(sam_vit_small_checkpoint):
            link_sam_vit_small_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "sam_vit_b.pth"])
            models_is_not_exist += [(sam_vit_small_checkpoint, link_sam_vit_small_checkpoint)]

        onnx_vit_small_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        if not os.path.exists(onnx_vit_small_checkpoint):
            link_onnx_vit_small_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "vit_b_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_small_checkpoint, link_onnx_vit_small_checkpoint)]

    # Get models for text detection
    vgg16_baseline_path = os.path.join(checkpoint_folder, "vgg16_baseline.pth")
    if not os.path.exists(vgg16_baseline_path):
        link_vgg16_baseline = get_nested_url(deepfake_config, ["checkpoints", "vgg16_baseline.pth"])
        models_is_not_exist += [(vgg16_baseline_path, link_vgg16_baseline)]

    vgg16_east_path = os.path.join(checkpoint_folder, "vgg16_east.pth")
    if not os.path.exists(vgg16_east_path):
        link_vgg16_east = get_nested_url(deepfake_config, ["checkpoints", "vgg16_east.pth"])
        models_is_not_exist += [(vgg16_east_path, link_vgg16_east)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_media_edit_config():
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()

    if not deepfake_config:
        return offline_status, models_is_not_exist

    local_model_path = os.path.join(DEEPFAKE_MODEL_FOLDER, 'gfpgan', 'weights')
    gfpgan_model_path = os.path.join(local_model_path, 'GFPGANv1.4.pth')
    if not os.path.exists(gfpgan_model_path):
        gfpgan_url = get_nested_url(deepfake_config, ["gfpgan", "GFPGANv1.4.pth"])
        models_is_not_exist += [(gfpgan_model_path, gfpgan_url)]

    # Inspect models by GPU or CPU
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if not use_cpu:
        animesgan_model_path = os.path.join(local_model_path, 'realesr-animevideov3.pth')
        if not os.path.exists(animesgan_model_path):
            animesgan_url = get_nested_url(deepfake_config, ["gfpgan", "realesr-animevideov3.pth"])
            models_is_not_exist += [(animesgan_model_path, animesgan_url)]

        general_model_path = os.path.join(local_model_path, 'realesr-general-x4v3.pth')
        if not os.path.exists(general_model_path):
            general_url = get_nested_url(deepfake_config, ["gfpgan", "realesr-general-x4v3.pth"])
            models_is_not_exist += [(general_model_path, general_url)]

        wdn_model_path = os.path.join(local_model_path, 'realesr-general-wdn-x4v3.pth')
        if not os.path.exists(wdn_model_path):
            wdn_url = get_nested_url(deepfake_config, ["gfpgan", "realesr-general-wdn-x4v3.pth"])
            models_is_not_exist += [(wdn_model_path, wdn_url)]

    # Audio models
    general_models = os.path.join(RTVC_VOICE_FOLDER, "general")
    rtvc_config = get_rtvc_config()

    trimed = os.path.join(general_models, "trimed.pt")
    if not os.path.exists(trimed):
        link_trimed = get_nested_url(rtvc_config, ["general", "trimed.pt"])
        models_is_not_exist += [(trimed, link_trimed)]

    voicefixer = os.path.join(general_models, "voicefixer.ckpt")
    if not os.path.exists(voicefixer):
        link_voicefixer = get_nested_url(rtvc_config, ["general", "voicefixer.ckpt"])
        models_is_not_exist += [(voicefixer, link_voicefixer)]

    hub_dir = torch.hub.get_dir()
    hub_dir_checkpoints = os.path.join(hub_dir, "checkpoints")
    unmix_vocals_voice = os.path.join(hub_dir_checkpoints, "vocals-bccbd9aa.pth")
    if not os.path.exists(unmix_vocals_voice):
        link_unmix_vocals_voice = get_nested_url(rtvc_config, ["general", "vocals-bccbd9aa.pth"])
        models_is_not_exist += [(unmix_vocals_voice, link_unmix_vocals_voice)]

    master_64_path = os.path.join(hub_dir_checkpoints, MASTER_64_URL.split("/")[-1])
    if not os.path.exists(master_64_path):
        models_is_not_exist += [(master_64_path, MASTER_64_URL)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_diffusion_config():
    models_is_not_exist = []
    offline_status = False
    deepfake_config = get_deepfake_config()
    diffusion_config = get_diffusion_config()

    # Check models of deepfake
    if not deepfake_config:
        return offline_status, models_is_not_exist
    # Check models of stable diffusion
    if not diffusion_config:
        return offline_status, models_is_not_exist

    checkpoint_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "checkpoints")
    # Inspect models by GPU or CPU
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if not use_cpu:
        # Large model
        sam_vit_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_h.pth')
        if not os.path.exists(sam_vit_checkpoint):
            link_sam_vit_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "sam_vit_h.pth"])
            models_is_not_exist += [(sam_vit_checkpoint, link_sam_vit_checkpoint)]

        onnx_vit_checkpoint = os.path.join(checkpoint_folder, 'vit_h_quantized.onnx')
        if not os.path.exists(onnx_vit_checkpoint):
            link_onnx_vit_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "vit_h_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_checkpoint, link_onnx_vit_checkpoint)]
    else:
        # Small model
        sam_vit_small_checkpoint = os.path.join(checkpoint_folder, 'sam_vit_b.pth')
        if not os.path.exists(sam_vit_small_checkpoint):
            link_sam_vit_small_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "sam_vit_b.pth"])
            models_is_not_exist += [(sam_vit_small_checkpoint, link_sam_vit_small_checkpoint)]

        onnx_vit_small_checkpoint = os.path.join(checkpoint_folder, 'vit_b_quantized.onnx')
        if not os.path.exists(onnx_vit_small_checkpoint):
            link_onnx_vit_small_checkpoint = get_nested_url(deepfake_config, ["checkpoints", "vit_b_quantized.onnx"])
            models_is_not_exist += [(onnx_vit_small_checkpoint, link_onnx_vit_small_checkpoint)]

    diffuser_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "diffusion")
    controlnet_model_path = os.path.join(diffuser_folder, "control_sd15_hed.pth")
    if not os.path.exists(controlnet_model_path):
        link_controlnet_model = get_nested_url(deepfake_config, ["diffusion", "control_sd15_hed.pth"])
        models_is_not_exist += [(controlnet_model_path, link_controlnet_model)]

    controlnet_model_annotator_path = os.path.join(diffuser_folder, "ControlNetHED.pth")
    if not os.path.exists(controlnet_model_annotator_path):
        link_controlnet_model_annotator = get_nested_url(deepfake_config, ["diffusion", "ControlNetHED.pth"])
        models_is_not_exist += [(controlnet_model_annotator_path, link_controlnet_model_annotator)]

    controlnet_model_canny_path = os.path.join(diffuser_folder, "control_sd15_canny.pth")
    if not os.path.exists(controlnet_model_canny_path):
        link_controlnet_model_canny = get_nested_url(deepfake_config, ["diffusion", "control_sd15_canny.pth"])
        models_is_not_exist += [(controlnet_model_canny_path, link_controlnet_model_canny)]

    vae_model_path = os.path.join(diffuser_folder, "vae-ft-mse-840000-ema-pruned.ckpt")
    if not os.path.exists(vae_model_path):
        link_vae_model = get_nested_url(deepfake_config, ["diffusion", "vae-ft-mse-840000-ema-pruned.ckpt"])
        models_is_not_exist += [(vae_model_path, link_vae_model)]

    # load gmflow model
    gmflow_model_path = os.path.join(diffuser_folder, "gmflow_sintel-0c07dcb3.pth")
    if not os.path.exists(gmflow_model_path):
        link_gmflow_model = get_nested_url(deepfake_config, ["diffusion", "gmflow_sintel-0c07dcb3.pth"])
        models_is_not_exist += [(gmflow_model_path, link_gmflow_model)]

    ebsynth_folder = os.path.join(DEEPFAKE_MODEL_FOLDER, "ebsynth")
    if sys.platform == 'win32':
        ebsynth_path = os.path.join(ebsynth_folder, "EbSynth.exe")
        if not os.path.exists(ebsynth_path):
            link_ebsynth = get_nested_url(deepfake_config, ["ebsynth", "EbSynth-Beta-Win.zip"])
            models_is_not_exist += [(ebsynth_folder, link_ebsynth)]
    elif sys.platform == 'linux':
        ebsynth_path = os.path.join(ebsynth_folder, "ebsynth_linux_cu118")
        if not os.path.exists(ebsynth_path):
            link_ebsynth = get_nested_url(deepfake_config, ["ebsynth", "ebsynth_linux_cu118"])
            models_is_not_exist += [(ebsynth_folder, link_ebsynth)]

    for key, val in diffusion_config.items():
        if os.path.exists(os.path.join(diffuser_folder, str(val))):
            print("Is minimum one model for stable diffusion")
            break
    else:
        sd_model_path = os.path.join(diffuser_folder, "Realistic_Vision_V5.1.safetensors")
        if not os.path.exists(sd_model_path):
            link_sd_model = get_nested_url(deepfake_config, ["diffusion", "Realistic_Vision_V5.1.safetensors"])
            models_is_not_exist += [(sd_model_path, link_sd_model)]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist


def inspect_rtvc_config():
    offline_status = False
    models_is_not_exist = []
    offline_language = []
    online_language = []
    rtvc_config = get_rtvc_config()

    if not rtvc_config:
        return offline_status, models_is_not_exist, online_language, offline_language
    # English
    english_models_is_not_exist = []
    english_models = os.path.join(RTVC_VOICE_FOLDER, "en")
    english_encoder = os.path.join(english_models, "encoder.pt")
    if not os.path.exists(english_encoder):
        link_english_encoder = get_nested_url(rtvc_config, ["en", "encoder"])
        english_models_is_not_exist += [(english_encoder, link_english_encoder)]

    english_synthesizer = os.path.join(english_models, "synthesizer.pt")
    if not os.path.exists(english_synthesizer):
        link_english_synthesizer = get_nested_url(rtvc_config, ["en", "synthesizer"])
        english_models_is_not_exist += [(english_synthesizer, link_english_synthesizer)]

    english_vocoder = os.path.join(english_models, "vocoder.pt")
    if not os.path.exists(english_vocoder):
        link_english_vocoder = get_nested_url(rtvc_config, ["en", "vocoder"])
        english_models_is_not_exist += [(english_vocoder, link_english_vocoder)]

    # Russian
    russian_models_is_not_exist = []
    russian_models = os.path.join(RTVC_VOICE_FOLDER, "ru")
    russian_encoder = os.path.join(russian_models, "encoder.pt")
    if not os.path.exists(russian_encoder):
        link_russian_encoder = get_nested_url(rtvc_config, ["ru", "encoder"])
        russian_models_is_not_exist += [(russian_encoder, link_russian_encoder)]

    russian_synthesizer = os.path.join(russian_models, "synthesizer.pt")
    if not os.path.exists(russian_synthesizer):
        link_russian_synthesizer = get_nested_url(rtvc_config, ["ru", "synthesizer"])
        russian_models_is_not_exist += [(russian_synthesizer, link_russian_synthesizer)]

    russian_vocoder = os.path.join(russian_models, "vocoder.pt")
    if not os.path.exists(russian_vocoder):
        link_russian_vocoder = get_nested_url(rtvc_config, ["ru", "vocoder"])
        russian_models_is_not_exist += [(russian_vocoder, link_russian_vocoder)]

    # Chinese
    chinese_models_is_not_exist = []
    chinese_models = os.path.join(RTVC_VOICE_FOLDER, "zh")
    chinese_encoder = os.path.join(chinese_models, "encoder.pt")
    if not os.path.exists(chinese_encoder):
        link_chinese_encoder = get_nested_url(rtvc_config, ["zh", "encoder"])
        chinese_models_is_not_exist += [(chinese_encoder, link_chinese_encoder)]

    chinese_synthesizer = os.path.join(chinese_models, "synthesizer.pt")
    if not os.path.exists(chinese_synthesizer):
        link_chinese_synthesizer = get_nested_url(rtvc_config, ["zh", "synthesizer"])
        chinese_models_is_not_exist += [(chinese_synthesizer, link_chinese_synthesizer)]

    chinese_vocoder = os.path.join(chinese_models, "vocoder.pt")
    if not os.path.exists(chinese_vocoder):
        link_chinese_vocoder = get_nested_url(rtvc_config, ["zh", "vocoder"])
        chinese_models_is_not_exist += [(chinese_vocoder, link_chinese_vocoder)]

    # General
    general_models = os.path.join(RTVC_VOICE_FOLDER, "general")
    signature = os.path.join(general_models, "signature.pt")
    if not os.path.exists(signature):
        link_signature = get_nested_url(rtvc_config, ["general", "signature"])
        models_is_not_exist += [(signature, link_signature)]

    hubert = os.path.join(general_models, "hubert.pt")
    if not os.path.exists(hubert):
        link_hubert = get_nested_url(rtvc_config, ["general", "hubert"])
        models_is_not_exist += [(hubert, link_hubert)]

    trimed = os.path.join(general_models, "trimed.pt")
    if not os.path.exists(trimed):
        link_trimed = get_nested_url(rtvc_config, ["general", "trimed.pt"])
        models_is_not_exist += [(trimed, link_trimed)]

    voicefixer = os.path.join(general_models, "voicefixer.ckpt")
    if not os.path.exists(voicefixer):
        link_voicefixer = get_nested_url(rtvc_config, ["general", "voicefixer.ckpt"])
        models_is_not_exist += [(voicefixer, link_voicefixer)]

    hub_dir = torch.hub.get_dir()
    hub_dir_checkpoints = os.path.join(hub_dir, "checkpoints")
    unmix_vocals_voice = os.path.join(hub_dir_checkpoints, "vocals-bccbd9aa.pth")
    if not os.path.exists(unmix_vocals_voice):
        link_unmix_vocals_voice = get_nested_url(rtvc_config, ["general", "vocals-bccbd9aa.pth"])
        models_is_not_exist += [(unmix_vocals_voice, link_unmix_vocals_voice)]

    master_64_path = os.path.join(hub_dir_checkpoints, MASTER_64_URL.split("/")[-1])
    if not os.path.exists(master_64_path):
        models_is_not_exist += [(master_64_path, MASTER_64_URL)]

    # Language
    if len(english_models_is_not_exist) > 0:
        online_language += ["English"]
    else:
        offline_language += ["English"]
    if len(russian_models_is_not_exist) > 0:
        online_language += ["Russian"]
    else:
        offline_language += ["Russian"]
    if len(chinese_models_is_not_exist) > 0:
        online_language += ["Chinese"]
    else:
        offline_language += ["Chinese"]

    offline_status = False if len(models_is_not_exist) > 0 else True

    return offline_status, models_is_not_exist, online_language, offline_language

