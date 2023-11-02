import os
import gc
import sys
import json

import torch
import random
import subprocess
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
from flask_cors import CORS, cross_origin
from flaskwebgui import FlaskUI

from deepfake.inference import AnimationMouthTalk, AnimationFaceTalk, FaceSwap, Retouch, VideoEdit, GetSegment
try:
    from diffusers.inference import Video2Video, create_diffusion_instruction
    VIDEO2VIDEO_AVAILABLE = True
    diffusion_models = create_diffusion_instruction()
except ImportError:
    VIDEO2VIDEO_AVAILABLE = False
    diffusion_models = {}
from speech.interface import TextToSpeech, VoiceCloneTranslate, AudioSeparatorVoice
from speech.tts_models import load_voice_models, voice_names, file_voice_config, file_custom_voice_config, custom_voice_names
from speech.rtvc_models import load_rtvc, rtvc_models_config
from backend.folders import MEDIA_FOLDER, WAVES_FOLDER, DEEPFAKE_FOLDER, TMP_FOLDER, SETTING_FOLDER, CUSTOM_VOICE_FOLDER
from backend.translator import get_translate
from backend.general_utils import get_version_app, set_settings, current_time, is_ffmpeg_installed, get_folder_size, format_dir_time, clean_text_by_language

import logging

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

app.config['DEBUG'] = False
app.config['SYNTHESIZE_STATUS'] = {"status_code": 200, "message": ""}
app.config['SYNTHESIZE_RESULT'] = []
app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT'] = {}  # get segment result
app.config['SEGMENT_ANYTHING_MODEL'] = {}  # load segment anything model to dynamically change
app.config['RTVC_LOADED_MODELS'] = {}  # in order to not load model again if it was loaded in prev synthesize (faster)
app.config['TTS_LOADED_MODELS'] = {}  # in order to not load model again if it was loaded in prev synthesize (faster)
app.config['USER_LANGUAGE'] = "en"
app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}

logging.getLogger('werkzeug').disabled = True

version_app = get_version_app()


def clear_cache():
    # empty cache before big gpu models
    del app.config['SEGMENT_ANYTHING_MODEL'], app.config['RTVC_LOADED_MODELS'], app.config['TTS_LOADED_MODELS']
    app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT'] = {}  # clear segment data
    app.config['SEGMENT_ANYTHING_MODEL'], app.config['RTVC_LOADED_MODELS'], app.config['TTS_LOADED_MODELS'] = {}, {}, {}
    torch.cuda.empty_cache()
    gc.collect()


def get_avatars_static():
    # not need to use os.path.join for windows because it is frontend path to media file with normal symbol use
    standard_voices = {voice_name: url_for("media_file", filename=f"avatar/{voice_name}.png") for voice_name in voice_names}
    custom_voices = {voice_name: url_for("media_file", filename=f"avatar/Unknown.png") for voice_name in custom_voice_names}
    return {**standard_voices, **custom_voices}


def split_input_deepfake(input_param):
    if input_param is not None:
        input_param = input_param.split(",")
        params = []
        for param in input_param:
            if len(param) == 0:
                continue
            if param[0] == '-' and param[1:].isdigit() or param.isdigit():
                param = int(param)
                if param > 30:
                    param = 30
                elif param < -30:
                    param = -30
                params += [param]

        if params:
            return params
    return None


def get_print_translate(text):
    return get_translate(text=text, targetLang=app.config['USER_LANGUAGE'])


@app.route("/update_translation", methods=["POST"])
@cross_origin()
def update_translation():
    # create here file if not exist
    localization_path = os.path.join(SETTING_FOLDER, "localization.json")
    req = request.get_json()
    with open(localization_path, 'w', encoding='utf-8') as f:
            json.dump(req, f, ensure_ascii=False)

    return {"status": 200}


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    # Define a dictionary of languages
    settings = set_settings()
    lang_user = settings["user_language"]
    lang_code = lang_user.get("code", "en")
    lang_name = lang_user.get("name", "English")
    default_lang = settings["default_language"]
    if default_lang.get(lang_name) is not None:
        default_lang.pop(lang_name)
    ordered_lang = {**{lang_name: lang_code}, **default_lang}
    app.config['USER_LANGUAGE'] = lang_code  # set user language for translate system print
    return render_template(
        "index.html", version=json.dumps(version_app), existing_langs=ordered_lang,
        user_lang=lang_code, existing_models=get_avatars_static(), diffusion_models=json.dumps(diffusion_models)
    )


@app.route('/current_processor', methods=["GET"])
@cross_origin()
def current_processor():
    if torch.cuda.is_available():
        return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": True}
    return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": False}


@app.route('/upload_tmp', methods=['POST'])
@cross_origin()
def upload_file_deepfake():
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    if 'file' not in request.files:
        return {"status": 'No file uploaded'}
    file = request.files['file']
    if file.filename == '':
        return {"status": 'No file selected'}
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(TMP_FOLDER, filename))
        return {"status": 'File uploaded'}

@app.route('/open_folder', methods=["POST"])
@cross_origin()
def open_folder():
    if os.path.exists(MEDIA_FOLDER):
        path = MEDIA_FOLDER
        if sys.platform == 'win32':
            # Open folder for Windows
            subprocess.Popen(r'explorer /select,"{}"'.format(path))
        elif sys.platform == 'darwin':
            # Open folder for MacOS
            subprocess.Popen(['open', path])
        elif sys.platform == 'linux':
            # Open folder for Linux
            subprocess.Popen(['xdg-open', path])
        return {"status_code": 200} #redirect('/')
    return {"status_code": 300} #redirect('/')


@app.route('/record_settings', methods=["POST"])
@cross_origin()
def record_lang_setting():
    req = request.get_json()
    lang_code = req.get("code", "en")
    lang_name = req.get("name", "English")
    settings = set_settings()
    setting_file = os.path.join(SETTING_FOLDER, "settings.json")
    with open(setting_file, 'w') as f:
        settings["user_language"] = {
            "code": lang_code,
            "name": lang_name
        }
        json.dump(settings, f)

    app.config['USER_LANGUAGE'] = lang_code

    return {
        "response_code": 0,
        "response": "Set new language"
    }


@app.route("/voice_status/", methods=["GET"])
@cross_origin()
def get_voice_list():
    voice_models_status = {}
    for voice_name in voice_names:
        voice_models_status[voice_name] = {}
        checkpoint_path = file_voice_config[voice_name]["engine"]["tacotron2"]["model_path"]
        if not os.path.exists(checkpoint_path):
            voice_models_status[voice_name]["checkpoint"] = False
        else:
            voice_models_status[voice_name]["checkpoint"] = True
        waveglow = file_voice_config[voice_name]["vocoder"]["waveglow"]["model_path"]
        if not os.path.exists(waveglow):
            voice_models_status[voice_name]["waveglow"] = False
        else:
            voice_models_status[voice_name]["waveglow"] = True
    return voice_models_status


@app.route("/create_segment_anything/", methods=["POST"])
@cross_origin()
def create_segment_anything():
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # empty cache without func clear_cache()
    torch.cuda.empty_cache()
    gc.collect()

    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}

    if not app.config['SEGMENT_ANYTHING_MODEL']:
        app.config['SEGMENT_ANYTHING_MODEL'] = GetSegment.load_model()

    # get parameters
    request_list = request.get_json()
    source = request_list.get("source")
    point_list = request_list.get("point_list")
    current_time = request_list.get("current_time", 0)
    obj_id = request_list.get("obj_id", 1)

    # call get segment anything
    segment_models = app.config['SEGMENT_ANYTHING_MODEL']
    predictor = segment_models.get("predictor")
    session = segment_models.get("session")
    result_filename = GetSegment.get_segment_mask_file(
        predictor=predictor, session=session, source=os.path.join(TMP_FOLDER, source), point_list=point_list
    )

    # Set new data to send in frontend
    app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT'] = {}
    app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT'][str(current_time)] = {}
    app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT'][str(current_time)][str(obj_id)] = {
        "mask": url_for("media_file", filename="/tmp/" + result_filename),
        "point_list": point_list
    }

    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}

    return {"status": 200}


@app.route("/get_segment_anything/", methods=["GET"])
@cross_origin()
def get_segment_anything():
    segment_preview = app.config['SEGMENT_ANYTHING_MASK_PREVIEW_RESULT']
    return {
        "response_code": 0, "response": segment_preview
    }

@app.route("/synthesize_video_merge/", methods=["POST"])
@cross_origin()
def synthesize_video_merge():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    # get parameters
    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print("Please wait... Processing is started")

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    source_folder = request_list.get("source_folder")
    audio_name = request_list.get("audio_name")
    audio_path = os.path.join(TMP_FOLDER, audio_name) if audio_name else None
    fps = request_list.get("fps", 30)

    request_time = current_time()
    request_date = format_dir_time(request_time)

    try:
        result = VideoEdit.main_merge_frames(
            output=DEEPFAKE_FOLDER, source_folder=source_folder,
            audio_path=audio_path, fps=fps
        )
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Image to video"), "request_mode": "deepfake", "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    result_filename = "/video/" + result.replace("\\", "/").split("/video/")[-1]
    url = url_for("media_file", filename=result_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Image to video"), "request_mode": "deepfake", "response_url": url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Merge frames to video completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}

    return {"status": 200}


@app.route("/synthesize_media_editor/", methods=["POST"])
@cross_origin()
def synthesize_media_editor():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    # get parameters
    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print("Please wait... Processing is started")

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    source = request_list.get("source")
    gfpgan = request_list.get("gfpgan", False)
    animesgan = request_list.get("animesgan", False)
    realesrgan = request_list.get("realesrgan", False)
    vocals = request_list.get("vocals", False)
    residual = request_list.get("residual", False)
    # List of enhancer options in preferred order
    enhancer_options = [gfpgan, animesgan, realesrgan]
    separator_options = [vocals, residual]
    # Find the first non-False option
    enhancer = next((enhancer_option for enhancer_option in enhancer_options if enhancer_option), False)
    audio_separator = next((separator_option for separator_option in separator_options if separator_option), False)
    is_get_frames = request_list.get("get_frames", False)
    media_start = request_list.get("media_start", 0)
    media_end = request_list.get("media_end", 0)
    media_type = request_list.get("media_type", "img")

    request_time = current_time()
    request_date = format_dir_time(request_time)

    if enhancer and not audio_separator:
        request_mode = "deepfake"
        mode_msg = get_print_translate("Content improve")
    elif not enhancer and audio_separator:
        request_mode = "speech"
        mode_msg = get_print_translate("Separator audio")
    else:
        request_mode = "deepfake"
        mode_msg = get_print_translate("Video to images")

    try:
        if not audio_separator and media_type in ["img", "video"]:
            result = VideoEdit.main_video_work(
                output=DEEPFAKE_FOLDER, source=os.path.join(TMP_FOLDER, source),
                enhancer=enhancer, is_get_frames=is_get_frames,
                media_start=media_start, media_end=media_end
            )
            result_filename = "/video/" + result.replace("\\", "/").split("/video/")[-1]
        elif audio_separator and media_type in ["audio", "video"]:
            dir_time = current_time()
            result_path = AudioSeparatorVoice.get_audio_separator(
                source=os.path.join(TMP_FOLDER, source), output_path=os.path.join(WAVES_FOLDER, dir_time), file_type=media_type,
                converted_wav=True, target=audio_separator, trim_silence=False, resample=False
            )
            result_filename = "/waves/" + result_path.replace("\\", "/").split("/waves/")[-1]
        else:
            raise Exception("Not recognition options for media content editor")
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": mode_msg, "voice": mode_msg, "request_mode": request_mode, "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    url = url_for("media_file", filename=result_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": mode_msg, "voice": mode_msg, "request_mode": request_mode, "response_url": url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Edit media completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}

    return {"status": 200}


@app.route("/synthesize_diffuser/", methods=["POST"])
@cross_origin()
def synthesize_diffuser():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    # has to work only with GPU
    current_processor = os.environ.get('WUNJO_TORCH_DEVICE', "cpu")
    if current_processor == "cpu":
        print("You need to use GPU for this function")
        return {"status": 400}

    # check what module is exist
    if not VIDEO2VIDEO_AVAILABLE:
        print("In this app version is not module diffusion")
        return {"status": 400}

    # get parameters
    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print("Please wait... Processing is started")

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    source = request_list.get("source")
    source_start = float(request_list.get("source_start", 0))
    source_end = float(request_list.get("source_end", 0))
    source_type = request_list.get("source_type", "video")
    masks = request_list.get("masks", {})
    interval_generation = int(request_list.get("interval_generation", 10))
    controlnet = request_list.get("controlnet", "canny")
    loose_cfattn = request_list.get("preprocessor_loose_cfattn", False)
    freeu = request_list.get("preprocessor_freeu", False)
    preprocessor = [loose_cfattn, freeu]
    segment_percentage = int(request_list.get("segment_percentage", 25))
    thickness_mask = int(request_list.get("thickness_mask", 10))
    sd_model_name = request_list.get("sd_model_name", None)

    request_time = current_time()
    request_date = format_dir_time(request_time)

    clear_cache()  # clear empty, because will be better load segment models again and after empty cache

    try:
        diffusion_result = Video2Video.main_video_render(
            source=os.path.join(TMP_FOLDER, source), output_folder=DEEPFAKE_FOLDER, source_start=source_start, sd_model_name=sd_model_name,
            source_end=source_end, source_type=source_type, masks=masks, interval=interval_generation, thickness_mask=thickness_mask,
            control_type=controlnet, translation=preprocessor, predictor=None, session=None, segment_percentage=segment_percentage
        )
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Diffusion"), "request_mode": "deepfake", "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    diffusion_result_filename = "/video/" + diffusion_result.replace("\\", "/").split("/video/")[-1]
    diffusion_url = url_for("media_file", filename=diffusion_result_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Diffusion"), "request_mode": "deepfake", "response_url": diffusion_url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Diffusion synthesis completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # empty cache
    torch.cuda.empty_cache()

    return {"status": 200}


@app.route("/synthesize_retouch/", methods=["POST"])
@cross_origin()
def synthesize_retouch():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    # get parameters
    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print("Please wait... Processing is started")

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    source = request_list.get("source")
    source_start = float(request_list.get("source_start", 0))
    source_end = float(request_list.get("source_end", 0))
    source_type = request_list.get("source_type", "img")
    masks = request_list.get("masks", {})
    model_type = request_list.get("model_type", "retouch_object")
    mask_color = request_list.get("mask_color", None)
    blur = int(request_list.get("blur", 1))
    upscale = request_list.get("upscale", False)
    segment_percentage = int(request_list.get("segment_percentage", 25))

    request_time = current_time()
    request_date = format_dir_time(request_time)
    # clear empty, because will be better load segment models again and after empty cache, else not empty full
    clear_cache()

    try:
        retouch_result = Retouch.main_retouch(
            output=DEEPFAKE_FOLDER, source=os.path.join(TMP_FOLDER, source), source_start=source_start,
            masks=masks, retouch_model_type=model_type, source_end=source_end, source_type=source_type,
            predictor=None, session=None, mask_color=mask_color, blur=blur, upscale=upscale, segment_percentage=segment_percentage
        )
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Content clean-up"), "request_mode": "deepfake", "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    retouch_result_filename = "/video/" + retouch_result.replace("\\", "/").split("/video/")[-1]
    retouch_url = url_for("media_file", filename=retouch_result_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Content clean-up"), "request_mode": "deepfake", "response_url": retouch_url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Retouch synthesis completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}
    # empty cache
    torch.cuda.empty_cache()

    return {"status": 200}


@app.route("/synthesize_face_swap/", methods=["POST"])
@cross_origin()
def synthesize_face_swap():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    # get parameters
    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print("Please wait... Processing is started")

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    target_content = request_list.get("target_content")
    face_target_fields = request_list.get("face_target_fields")
    source_face_fields = request_list.get("face_source_fields")
    source_content = request_list.get("source_content")
    type_file_source = request_list.get("type_file_source")
    video_start_target = request_list.get("video_start_target", 0)
    video_end_target = request_list.get("video_end_target", 0)
    video_current_time_source = request_list.get("video_current_time_source", 0)
    video_end_source = request_list.get("video_end_source", 0)
    multiface = request_list.get("multiface", False)
    similarface = request_list.get("similarface", False)
    similar_coeff = float(request_list.get("similar_coeff", 0.95))

    request_time = current_time()
    request_date = format_dir_time(request_time)

    clear_cache()  # clear empty

    try:
        face_swap_result = FaceSwap.main_faceswap(
            deepfake_dir=DEEPFAKE_FOLDER,
            target=os.path.join(TMP_FOLDER, target_content),
            target_face_fields=face_target_fields,
            source=os.path.join(TMP_FOLDER, source_content),
            source_face_fields=source_face_fields,
            type_file_source=type_file_source,
            target_video_start=video_start_target,
            target_video_end=video_end_target,
            source_current_time=video_current_time_source,
            source_video_end=video_end_source,
            multiface=multiface,
            similarface=similarface,
            similar_coeff=similar_coeff
        )
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Face swap"), "request_mode": "deepfake", "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    face_swap_result_filename = "/video/" + face_swap_result.replace("\\", "/").split("/video/")[-1]
    face_swap_url = url_for("media_file", filename=face_swap_result_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": get_print_translate("Face swap"), "request_mode": "deepfake", "response_url": face_swap_url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Face swap synthesis completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}
    # empty cache
    torch.cuda.empty_cache()

    return {"status": 200}


@app.route("/synthesize_deepfake/", methods=["POST"])
@cross_origin()
def synthesize_deepfake():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print(get_print_translate("Please wait... Processing is started"))

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    face_fields = request_list.get("face_fields")
    source_image = os.path.join(TMP_FOLDER, request_list.get("source_media"))
    driven_audio = os.path.join(TMP_FOLDER, request_list.get("driven_audio"))
    preprocess = request_list.get("preprocess")
    still = request_list.get("still")
    expression_scale = float(request_list.get("expression_scale", 1.0))
    input_yaw = split_input_deepfake(request_list.get("input_yaw"))
    input_pitch = split_input_deepfake(request_list.get("input_pitch"))
    input_roll = split_input_deepfake(request_list.get("input_roll"))
    type_file = request_list.get("type_file")
    media_start = request_list.get("media_start", 0)
    media_end = request_list.get("media_end", 0)
    emotion_label = request_list.get("emotion_label", None)
    similar_coeff = float(request_list.get("similar_coeff", 0.95))

    request_time = current_time()
    request_date = format_dir_time(request_time)

    clear_cache()  # clear empty

    if type_file == "img":
        mode_msg = get_print_translate("Face in sync")
    else:
        mode_msg = get_print_translate("Lip in sync")

    try:
        if type_file == "img":
            deepfake_result = AnimationFaceTalk.main_img_deepfake(
                deepfake_dir=DEEPFAKE_FOLDER,
                source_image=source_image,
                driven_audio=driven_audio,
                still=still,
                preprocess=preprocess,
                face_fields=face_fields,
                expression_scale=expression_scale,
                input_yaw=input_yaw,
                input_pitch=input_pitch,
                input_roll=input_roll,
                pose_style=random.randint(0, 45)
                )
        elif type_file == "video":
            deepfake_result = AnimationMouthTalk.main_video_deepfake(
                deepfake_dir=DEEPFAKE_FOLDER,
                face=source_image,
                audio=driven_audio,
                face_fields=face_fields,
                video_start=float(media_start),
                video_end=float(media_end),
                emotion_label=emotion_label,
                similar_coeff=similar_coeff
            )

        torch.cuda.empty_cache()
    except Exception as err:
        app.config['SYNTHESIZE_RESULT'] += [{"mode": mode_msg, "request_mode": "deepfake", "response_url": "", "request_date": request_date, "request_information": get_print_translate("Error")}]
        print(f"Error ... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}


    deepfake_filename = "/video/" + deepfake_result.replace("\\", "/").split("/video/")[-1]
    deepfake_url = url_for("media_file", filename=deepfake_filename)

    app.config['SYNTHESIZE_RESULT'] += [{"mode": mode_msg, "request_mode": "deepfake", "response_url": deepfake_url, "request_date": request_date, "request_information": get_print_translate("Successfully")}]

    print("Deepfake synthesis completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}
    # empty cache
    torch.cuda.empty_cache()

    return {"status": 200}

@app.route("/synthesize_result/", methods=["GET"])
@cross_origin()
def get_synthesize_result():
    general_results = app.config['SYNTHESIZE_RESULT']
    return {
        "response_code": 0,
        "response": general_results
    }

@app.route("/synthesize_speech/", methods=["POST"])
@cross_origin()
def synthesize():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    request_list = request.get_json()
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 300}
    print(get_print_translate("Please wait... Processing is started"))

    dir_time = current_time()
    request_date = format_dir_time(dir_time)

    for request_json in request_list:
        text = request_json["text"]
        model_type = request_json["voice"]

        options = {
            "rate": float(request_json.get("rate", 1.0)),
            "pitch": float(request_json.get("pitch", 1.0)),
            "volume": float(request_json.get("volume", 0.0))
        }

        auto_translation = request_json.get("auto_translation", False)
        lang_translation = request_json.get("lang_translation")
        rtvc_models_lang = lang_translation  # try to use user lang for rtvc models

        use_voice_clone_on_audio = request_json.get("use_voice_clone_on_audio", False)
        rtvc_audio_clone_voice = request_json.get("rtvc_audio_clone_voice", "")

        if not rtvc_audio_clone_voice:
            use_voice_clone_on_audio = False

        # rtvc doesn't have models by user lang, set english models
        if rtvc_models_config.get(lang_translation) is None:
            rtvc_models_lang = "en"

        # init only one time the models for voice clone if it is needs
        if (auto_translation or rtvc_audio_clone_voice) and app.config['RTVC_LOADED_MODELS'].get(rtvc_models_lang) is None:
            # Check ffmpeg
            is_ffmpeg = is_ffmpeg_installed()
            if not is_ffmpeg:
                app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
                return {"status": 400}
            # init models
            encoder, synthesizer, signature, vocoder = load_rtvc(rtvc_models_lang)
            app.config['RTVC_LOADED_MODELS'][rtvc_models_lang] = {"encoder": encoder, "synthesizer": synthesizer, "signature": signature, "vocoder": vocoder}

        if model_type:
            app.config['TTS_LOADED_MODELS'] = load_voice_models(model_type, app.config['TTS_LOADED_MODELS'])

        for model in model_type:
            # if set auto translate, when get clear translation for source of models to clear synthesis audio
            # get tacotron2 lang from engine
            tacotron2_lang = app.config['TTS_LOADED_MODELS'][model].engine.charset
            if auto_translation:
                print("User use auto translation. Translate text before TTS.")
                tts_text = get_translate(text=text, targetLang=tacotron2_lang)
            else:
                tts_text = text

            response_code, results = TextToSpeech.get_synthesized_audio(tts_text, model, app.config['TTS_LOADED_MODELS'], os.path.join(WAVES_FOLDER, dir_time), **options)

            if response_code == 0:
                for result in results:
                    filename = result.pop("filename")

                    # translated audio if user choose lang not equal for model and set auto translation
                    # or text has not tacotron lang fonts
                    # (1) remove clean_text_by_language if I will want to delete multilanguage
                    if (tacotron2_lang != lang_translation and auto_translation) or clean_text_by_language(text, tacotron2_lang) != clean_text_by_language(text, None, True):
                        # voice clone on tts audio result
                        # init models if not defined
                        if app.config['RTVC_LOADED_MODELS'].get(rtvc_models_lang) is None:
                            encoder, synthesizer, signature, vocoder = load_rtvc(rtvc_models_lang)
                            app.config['RTVC_LOADED_MODELS'][rtvc_models_lang] = {"encoder": encoder, "synthesizer": synthesizer, "signature": signature, "vocoder": vocoder}
                        # get models
                        encoder = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["encoder"]
                        synthesizer = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["synthesizer"]
                        signature = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["signature"]
                        vocoder = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["vocoder"]

                        # text translated inside get_synthesized_audio
                        response_code, clone_result = VoiceCloneTranslate.get_synthesized_audio(
                            audio_file=filename, encoder=encoder, synthesizer=synthesizer, signature=signature,
                            vocoder=vocoder, text=text, src_lang=lang_translation, need_translate=auto_translation,
                            save_folder=os.path.join(WAVES_FOLDER, dir_time), tts_model_name=model, converted_wav=False
                        )

                        if response_code == 0:
                            result = clone_result
                            # get new filename
                            filename = result.pop("filename")
                        else:
                            print("Error...during clone synthesized voice")

                    result["file_name"] = os.path.basename(filename)
                    filename = "/waves/" + filename.replace("\\", "/").split("/waves/")[-1]
                    print("Synthesized file: ", filename)
                    result.pop("response_audio")
                    result["response_url"] = url_for("media_file", filename=filename)
                    # result["response_audio"] = b64encode(audio_bytes).decode("utf-8")
                    result["response_code"] = response_code
                    result["request_date"] = request_date
                    result["request_information"] = text
                    result["request_mode"] = "speech"
                    result["voice"] = get_print_translate(result.get("voice"))
                    # Add result in frontend
                    app.config['SYNTHESIZE_RESULT'] += [result]

        # here use for voice cloning of audio file without tts
        if use_voice_clone_on_audio:
            encoder = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["encoder"]
            synthesizer = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["synthesizer"]
            signature = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["signature"]
            vocoder = app.config['RTVC_LOADED_MODELS'][rtvc_models_lang]["vocoder"]
            rtvc_audio_clone_path = os.path.join(TMP_FOLDER, rtvc_audio_clone_voice)

            response_code, result = VoiceCloneTranslate.get_synthesized_audio(
                audio_file=rtvc_audio_clone_path, encoder=encoder, synthesizer=synthesizer, signature=signature, vocoder=vocoder,
                text=text, src_lang=lang_translation, need_translate=auto_translation, save_folder= os.path.join(WAVES_FOLDER, dir_time)
            )
            if response_code == 0:
                filename = result.pop("filename")
                result["file_name"] = os.path.basename(filename)
                filename = "/waves/" + filename.replace("\\", "/").split("/waves/")[-1]
                print("Synthesized file: ", filename)
                result.pop("response_audio")
                result["response_url"] = url_for("media_file", filename=filename)
                # result["response_audio"] = b64encode(audio_bytes).decode("utf-8")
                result["response_code"] = response_code
                result["request_date"] = request_date
                result["request_information"] = text
                result["request_mode"] = "speech"
                result["voice"] = get_print_translate(result.get("voice"))
                # Add result in frontend
                app.config['SYNTHESIZE_RESULT'] += [result]

    # remove subfiles
    try:
        result_filenames = []
        for result in app.config['SYNTHESIZE_RESULT']:
            if result.get("file_name"):
                result_filenames += [result["file_name"]]
        current_result_folder = os.path.join(WAVES_FOLDER, dir_time)
        for f in os.listdir(current_result_folder):
            if f not in result_filenames:
                os.remove(os.path.join(current_result_folder, f))
    except Exception as err:
        print("Some error during remove files for speech synthesis")

    app.config['RTVC_LOADED_MODELS'] = {}  # remove RTVC models
    print("Text to speech synthesis completed successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    # Update disk space size
    app.config['FOLDER_SIZE_RESULT'] = {"audio": get_folder_size(WAVES_FOLDER), "video": get_folder_size(DEEPFAKE_FOLDER)}
    # empty cache
    torch.cuda.empty_cache()

    return {"status": 200}


@app.route("/synthesize_process/", methods=["GET"])
@cross_origin()
def get_synthesize_status():
    status = app.config['SYNTHESIZE_STATUS']
    return status


@app.route("/disk_space_used/", methods=["GET"])
@cross_origin()
def get_disk_space_used_status():
    status = app.config['FOLDER_SIZE_RESULT']
    return jsonify(status)


@app.route('/change_processor', methods=["POST"])
@cross_origin()
def change_processor():
    current_processor = os.environ.get('WUNJO_TORCH_DEVICE', "cpu")
    if app.config['SYNTHESIZE_STATUS'].get("status_code") == 200:
        if not torch.cuda.is_available():
            print("No GPU driver was found on your computer. Working on the GPU will speed up the generation of content several times.")
            print("Visit the documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki to learn how to install drivers for your computer.")
            return {"current_processor": 'none'}  # will be message how install cuda or hust hidden changer
        if current_processor == "cpu":
            print("You turn on GPU")
            os.environ['WUNJO_TORCH_DEVICE'] = 'cuda'
            return {"current_processor": 'cuda'}
        else:
            print("You turn on CPU")
            os.environ['WUNJO_TORCH_DEVICE'] = 'cpu'
            return {"current_processor": 'cpu'}
    return {"current_processor": current_processor}


if not app.config['DEBUG']:
    from time import time
    from io import StringIO


    class TimestampedIO(StringIO):
        def __init__(self):
            super().__init__()
            self.logs = []

        def write(self, msg):
            timestamped_msg = (time(), msg)
            self.logs.append(timestamped_msg)
            super().write(msg)


    console_stdout = TimestampedIO()
    console_stderr = TimestampedIO()
    sys.stdout = console_stdout  # prints
    sys.stderr = console_stderr  # errors and warnings


    @app.route('/console_log', methods=['GET'])
    def console_log():
        max_len_logs = 30

        replace_phrases = [
            "* Debug mode: off", "* Serving Flask app 'wunjo.app'",
            "WARNING:waitress.queue:Task queue depth is 1", "WARNING:waitress.queue:Task queue depth is 2",
            "WARNING:waitress.queue:Task queue depth is 3", "WARNING:waitress.queue:Task queue depth is 4"
        ]

        combined_logs = console_stdout.logs + console_stderr.logs
        combined_logs.sort(key=lambda x: x[0])  # Sort by timestamp

        # Filter out unwanted phrases and extract log messages
        filtered_logs = [
            log[1] for log in combined_logs
            if not any(phrase in log[1] for phrase in replace_phrases)
        ]

        # Send the last max_len_logs lines to the frontend
        return jsonify(filtered_logs[-max_len_logs:])
else:
    os.environ['WUNJO_TORCH_DEVICE'] = 'cuda'

    @app.route('/console_log', methods=['GET'])
    def console_log():
        logs = ["Debug mode. Console is turned off"]
        return jsonify(logs)


@app.route('/console_log_print', methods=['POST'])
def console_log_print():
    resp = request.get_json()
    msg = resp.get("print", None)
    if msg:
        print(msg)  # show message in backend console
    return jsonify({"status": 200})


"""TRAIN MODULE"""
@app.route('/training_voice', methods=["POST"])
@cross_origin()
def common_training_route():
    # check what it is not repeat button click
    if app.config['SYNTHESIZE_STATUS'].get("status_code") != 200:
        print("The process is already running... ")
        return {"status": 400}

    try:
        # clear keep models
        app.config['RTVC_LOADED_MODELS'] = {}
        app.config['TTS_LOADED_MODELS'] = {}
        # get params and send
        print("Sending parameters to route... ")
        param = request.get_json()

        from train.utils import training_route
        training_route(param)
    except Exception as err:
        print(f"Error training... {err}")
        app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
        return {"status": 400}

    print("Trained finished successfully!")
    app.config['SYNTHESIZE_STATUS'] = {"status_code": 200}
    return {"status": 200}
"""TRAIN MODULE"""


class InvalidVoice(Exception):
    pass


@app.route("/media/<path:filename>", methods=["GET"])
@cross_origin()
def media_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=False)


def main():
    if not app.config['DEBUG'] and sys.platform != 'darwin':
        FlaskUI(app=app, server="flask").run()
    else:
        app.run()
