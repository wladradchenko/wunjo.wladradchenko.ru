import os
import sys
import json
import torch
import requests
import subprocess
from time import gmtime, strftime
from base64 import b64encode
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_from_directory, url_for, abort
from flask_cors import CORS, cross_origin
from flaskwebgui import FlaskUI

from deepfake.inference import main_img_deepfake, main_video_deepfake
from speech.file_handler import FileHandler
from speech.models import load_voice_models, voice_names, file_voice_config
from backend.folders import MEDIA_FOLDER, WAVES_FOLDER, DEEPFAKE_FOLDER, TMP_FOLDER, EXTENSIONS_FOLDER, SETTING_FOLDER
from backend.download import download_model, unzip, check_download_size, get_download_filename


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config['DEBUG'] = True
app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": ""}
app.config['SYSNTHESIZE_SPEECH_RESULT'] = []
app.config['SYSNTHESIZE_DEEPFAKE_RESULT'] = []

app.config['models'], app.config['_valid_model_types'] = {}, []
# get list of all directories in folder


def set_settings():
    default_language = {
            "English": "en",
            "Русский": "ru",
            "Portugal": "pt",
            "中文": "zh",
            "한국어": "ko"
        }
    standard_language = {
            "code": "en",
            "name": "English"
        }

    default_settings = {
        "user_language": standard_language,
        "default_language": default_language
    }
    setting_file = os.path.join(SETTING_FOLDER, "settings.json")
    # Check if the SETTING_FILE exists
    if not os.path.exists(setting_file):
        # If not, create it with default settings
        with open(setting_file, 'w') as f:
            json.dump(default_settings, f)
        return default_settings
    else:
        # If it exists, read its content
        with open(setting_file, 'r') as f:
            try:
                user_settings = json.load(f)
                if user_settings.get("user_language") is None:
                    user_settings["user_language"] = standard_language
                if user_settings.get("default_language") is None:
                    user_settings["default_language"] = default_language
                return user_settings
            except Exception as e:
                print(e)
                return default_settings


def get_avatars_static():
    return {voice_name: url_for("media_file", filename=f"avatar/{voice_name}.png") for voice_name in voice_names}


def current_time(format: str = None):
    if format:
        return strftime(format, gmtime())
    return strftime("%Y%m%d%H%M%S", gmtime())


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

@app.route("/", methods=["GET"])
@cross_origin()
def index():
    settings = set_settings()
    lang_user = settings["user_language"]
    lang_code = lang_user.get("code", "en")
    lang_name = lang_user.get("name", "English")
    default_lang = settings["default_language"]
    if default_lang.get(lang_name) is not None:
        default_lang.pop(lang_name)
    ordered_lang = {**{lang_name: lang_code}, **default_lang}
    # Define a dictionary of languages
    return render_template("index.html", existing_langs=ordered_lang, user_lang=lang_code, existing_models=get_avatars_static(), extensions_html=extensions_html)


@app.route('/current_processor', methods=["GET"])
@cross_origin()
def current_processor():
    if torch.cuda.is_available():
        return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": True}
    return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": False}

@app.route('/upload_tmp_deepfake', methods=['POST'])
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
    lang_name = req.get("name", "en")
    settings = set_settings()
    setting_file = os.path.join(SETTING_FOLDER, "settings.json")
    with open(setting_file, 'w') as f:
        settings["user_language"] = {
            "code": lang_code,
            "name": lang_name
        }
        json.dump(settings, f)

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


@app.route("/synthesize_deepfake_result/", methods=["GET"])
@cross_origin()
def get_synthesize_deepfake_result():
    general_results = app.config['SYSNTHESIZE_DEEPFAKE_RESULT']
    return {
        "response_code": 0,
        "response": general_results
    }

@app.route("/synthesize_deepfake/", methods=["POST"])
@cross_origin()
def synthesize_deepfake():
    request_list = request.get_json()
    app.config['SYSNTHESIZE_STATUS'] = {"status_code": 300, "message": "Подождите... Происходит обработка"}

    if not os.path.exists(DEEPFAKE_FOLDER):
        os.makedirs(DEEPFAKE_FOLDER)

    face_fields = request_list.get("face_fields")


    source_image = os.path.join(TMP_FOLDER, request_list.get("source_image"))
    driven_audio = os.path.join(TMP_FOLDER, request_list.get("driven_audio"))
    preprocess = request_list.get("preprocess")
    still = request_list.get("still")
    enhancer = request_list.get("enhancer")
    expression_scale = float(request_list.get("expression_scale", 1.0))
    input_yaw = split_input_deepfake(request_list.get("input_yaw"))
    input_pitch = split_input_deepfake(request_list.get("input_pitch"))
    input_roll = split_input_deepfake(request_list.get("input_roll"))
    background_enhancer = request_list.get("background_enhancer")
    type_file = request_list.get("type_file")
    video_start = request_list.get("video_start", 0)

    try:
        if type_file == "img":
            deepfake_result = main_img_deepfake(
                deepfake_dir=DEEPFAKE_FOLDER,
                source_image=source_image,
                driven_audio=driven_audio,
                still=still,
                enhancer=enhancer,
                preprocess=preprocess,
                face_fields=face_fields,
                expression_scale=expression_scale,
                input_yaw=input_yaw,
                input_pitch=input_pitch,
                input_roll=input_roll,
                background_enhancer=background_enhancer
                )
        elif type_file == "video":
            deepfake_result = main_video_deepfake(
                deepfake_dir=DEEPFAKE_FOLDER,
                face=source_image,
                audio=driven_audio,
                face_fields=face_fields,
                enhancer=enhancer,
                background_enhancer=background_enhancer,
                video_start=float(video_start)
            )

        torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        app.config['SYSNTHESIZE_DEEPFAKE_RESULT'] += [{"response_video_url": "", "response_video_date": "Лицо не найдено"}]
        app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": ""}
        return {"status": 400}


    deepfake_filename = "/video/" + deepfake_result.replace("\\", "/").split("/video/")[-1]
    deepfake_url = url_for("media_file", filename=deepfake_filename)
    deepfake_date = current_time("%H:%M:%S")  # maybe did in js parse of date

    app.config['SYSNTHESIZE_DEEPFAKE_RESULT'] += [{"response_video_url": deepfake_url, "response_video_date": deepfake_date}]

    app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": ""}

    return {"status": 200}

@app.route("/synthesize_speech_result/", methods=["GET"])
@cross_origin()
def get_synthesize_result():
    general_results = app.config['SYSNTHESIZE_SPEECH_RESULT']
    return {
        "response_code": 0,
        "response": general_results
    }

@app.route("/synthesize_speech/", methods=["POST"])
@cross_origin()
def synthesize():
    request_list = request.get_json()
    app.config['SYSNTHESIZE_STATUS'] = {"status_code": 300, "message": "Подождите... Происходит обработка"}
    app.config['SYSNTHESIZE_SPEECH_RESULT'] = []

    dir_time = current_time()

    for request_json in request_list:
        text = request_json["text"]
        model_type = request_json["voice"]

        options = {
            "rate": float(request_json.get("rate", 1.0)),
            "pitch": float(request_json.get("pitch", 1.0)),
            "volume": float(request_json.get("volume", 0.0))
        }

        app.config['models'] = load_voice_models(model_type, app.config['models'])

        for model in model_type:
            response_code, results = FileHandler.get_synthesized_audio(text, model, app.config['models'], os.path.join(WAVES_FOLDER, dir_time), **options)

            if response_code == 0:
                for result in results:
                    filename = result.pop("filename")
                    filename = "/waves/" + filename.replace("\\", "/").split("/waves/")[-1]
                    print(filename)
                    audio_bytes = result.pop("response_audio")
                    result["response_audio_url"] = url_for("media_file", filename=filename)
                    result["response_audio"] = b64encode(audio_bytes).decode("utf-8")
                    result["response_code"] = response_code
                    result["recognition_text"] = text

                    app.config['SYSNTHESIZE_SPEECH_RESULT'] += results

    app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": ""}

    return {"status": 200}


@app.route("/synthesize_process/", methods=["GET"])
@cross_origin()
def get_synthesize_status():
    status = app.config['SYSNTHESIZE_STATUS']
    return status


"""EXTENSIONS"""
EXTENSIONS_JSON_URL = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/extensions.json"

@app.route('/list_extensions/', methods=["GET"])
@cross_origin()
def list_extensions():
    try:
        response = requests.get(EXTENSIONS_JSON_URL)
        return response.json()
    except:
        return {}

@app.route('/get_extensions/', methods=["POST"])
@cross_origin()
def get_extensions():
    app.config['SYSNTHESIZE_STATUS'] = {"status_code": 300, "message": "Подождите... Происходит скачивание доп. пакетов"}

    request_list = request.get_json()  # get key
    extension_name = request_list.get("extension_name", None)
    extension_url = request_list.get("extension_url", '')
    if len(extension_url) > 0:
        if extension_name:
            path_extension = os.path.join(EXTENSIONS_FOLDER, extension_name)
        else:
            extension_name = get_download_filename(extension_url)
            if extension_name:
                path_extension = os.path.join(EXTENSIONS_FOLDER, extension_name)

        if extension_name is None:
            app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": "Не является файлом"}
            return {"status_code": 404}

        support_format = [".zip", ".tar", ".rar", ".7z"]
        for s in support_format:
            if s in extension_url:
                extension_name = extension_name.rsplit(".", 1)[0]
                break
        else:
            app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": "Не является поддерживаемым форматом zip, tar, rar или 7z"}
            return {"status_code": 404}

        try:
            if not os.path.exists(path_extension):
                if ".zip" in extension_url:
                    download_model(f"{path_extension}.zip", extension_url)
                    unzip(f"{path_extension}.zip", EXTENSIONS_FOLDER, extension_name)
                else:
                    download_model(path_extension, extension_url)
            else:
                if ".zip" in extension_url:
                    check_download_size(f"{path_extension}.zip", extension_url)
                    if not os.listdir(path_extension):
                        unzip(f"{path_extension}.zip", EXTENSIONS_FOLDER, extension_name)
                else:
                    check_download_size(path_extension, extension_url)

            app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": "Доп. пакеты скачаны. Перезагрузите приложение"}
        except:
            app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": "Проблемы с соединением"}
    else:
        app.config['SYSNTHESIZE_STATUS'] = {"status_code": 200, "message": "Ссылка пустая"}

    return {"status_code": 200}


from importlib.util import spec_from_file_location, module_from_spec


extensions_html = []

for entry in os.listdir(EXTENSIONS_FOLDER):
    entry_path = os.path.join(EXTENSIONS_FOLDER, entry)
    if os.path.isdir(entry_path):
        # Get the list of files in the extensions folder
        extension_files = [
            filename
            for filename in os.listdir(entry_path)
            if filename.endswith(".py") and filename != "__init__.py"
        ]

        # Iterate over the extension files and load them as modules
        for filename in extension_files:
            module_name = os.path.splitext(filename)[0]
            module_path = os.path.join(entry_path, filename)

            # Create a module object
            spec = spec_from_file_location(module_name, module_path)
            module = module_from_spec(spec)

            # Load the module
            spec.loader.exec_module(module)

            # Execute the module's code
            if hasattr(module, "run"):
                # Read the HTML content from the file
                with open(os.path.join(entry_path, 'templates', 'index.html'), 'r') as file:
                    html_content = file.read()

                # Append the HTML content to the extensions_html list
                extensions_html.append(html_content)

                module.run(MEDIA_FOLDER, entry_path, app)  # Pass the variables as arguments


# Route for accessing static files from the extensions folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'js', 'css', 'mp3', 'wav', 'mp4', 'yaml', 'json'}
@app.route('/extensions/static/<path:filename>')
def serve_extensions_static(filename):
    if '.' in filename:
        extension = filename.rsplit('.', 1)[1].lower()
        if extension in ALLOWED_EXTENSIONS:
            return send_from_directory(EXTENSIONS_FOLDER, filename)
    abort(404)

"""EXTENSIONS"""


class InvalidVoice(Exception):
    pass


@app.route("/media/<path:filename>", methods=["GET"])
@cross_origin()
def media_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=False)


def main():
    # FlaskUI(app=app, server="flask").run()
    app.run(port=5005)