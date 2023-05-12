import os
import sys
import requests
import subprocess
from time import gmtime, strftime
from base64 import b64encode
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_from_directory, url_for, redirect
from flask_cors import CORS, cross_origin
from flaskwebgui import FlaskUI

from talker.inference import main_talker
from backend.file_handler import FileHandler
from backend.models import models, keys as CONFIG_KEYS


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config['DEBUG'] = False
app.config['SYSNTHESIZE_STATUS'] = 200
app.config['SYSNTHESIZE_RESULT'] = []
app.config['SYSNTHESIZE_TALKER_RESULT'] = []

_valid_model_types = [key for key in models]
# get list of all directories in folder

HOME_FOLDER = os.path.expanduser('~')
MEDIA_FOLDER = os.path.join(HOME_FOLDER, '.voiceai')
if not os.path.exists(MEDIA_FOLDER):
    os.makedirs(MEDIA_FOLDER)

WAVES_FOLDER = os.path.join(MEDIA_FOLDER, 'waves')
AVATAR_FOLDER = os.path.join(MEDIA_FOLDER, "avatar")
TALKER_FOLDER = os.path.join(MEDIA_FOLDER, 'talker')
TMP_FOLDER = os.path.join(MEDIA_FOLDER, 'tmp')

def valid_model_types():
    return [key for key in models]

def get_png_files():
    if not os.path.exists(AVATAR_FOLDER):
        os.makedirs(AVATAR_FOLDER)

    png_files = {}
    for voice_name in CONFIG_KEYS:
        if voice_name in ['general']:
            voice_name = "Unknown"

        # Check if the avatar file exists
        if not os.path.exists(os.path.join(AVATAR_FOLDER, f"{voice_name}.png")):
            # If it doesn't exist, download it from the URL
            request_png = requests.get(f'https://wladradchenko.ru/static/voiceai.wladradchenko.ru/{voice_name}.png')
            with open(os.path.join(AVATAR_FOLDER, f"{voice_name}.png"), 'wb') as file:
                file.write(request_png.content)

        png_files[voice_name] = url_for("media_file", filename=f"avatar/{voice_name}.png")

    return png_files


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    model_avatars = get_png_files()
    model_avatars.pop("Unknown")
    return render_template("index.html", existing_models=model_avatars)

@app.route('/upload_tmp_talker', methods=['POST'])
@cross_origin()
def upload_file_talker():
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

@app.route("/synthesize_talker_result/", methods=["GET"])
@cross_origin()
def get_synthesize_talker_result():
    general_results = app.config['SYSNTHESIZE_TALKER_RESULT']
    return {
        "response_code": 0,
        "response": general_results
    }

@app.route("/synthesize_talker/", methods=["POST"])
@cross_origin()
def synthesize_talker():
    request_list = request.get_json()
    app.config['SYSNTHESIZE_STATUS'] = 300

    if not os.path.exists(TALKER_FOLDER):
        os.makedirs(TALKER_FOLDER)

    face_fields = request_list.get("face_fields")


    source_image = os.path.join(TMP_FOLDER, request_list.get("source_image"))
    driven_audio = os.path.join(TMP_FOLDER, request_list.get("driven_audio"))
    preprocess = request_list.get("preprocess")
    still = request_list.get("still")
    enhancer = request_list.get("enhancer")

    try:
        talker_result = main_talker(talker_dir=TALKER_FOLDER,
                                    source_image=source_image,
                                    driven_audio=driven_audio,
                                    still=still,
                                    enhancer=enhancer,
                                    preprocess=preprocess,
                                    face_fields=face_fields)
    except BaseException:
        app.config['SYSNTHESIZE_TALKER_RESULT'] += [{"response_video_url": "", "response_video_date": "Лицо не найдено"}]
        app.config['SYSNTHESIZE_STATUS'] = 200
        return {"status": 400}


    talker_filename = "/talker/" + talker_result.replace("\\", "/").split("/talker/")[-1]
    talker_url = url_for("media_file", filename=talker_filename)
    talker_date = strftime("%H:%M:%S", gmtime())

    app.config['SYSNTHESIZE_TALKER_RESULT'] += [{"response_video_url": talker_url, "response_video_date": talker_date}]

    app.config['SYSNTHESIZE_STATUS'] = 200

    return {"status": 200}

@app.route("/synthesize_result/", methods=["GET"])
@cross_origin()
def get_synthesize_result():
    general_results = app.config['SYSNTHESIZE_RESULT']
    return {
        "response_code": 0,
        "response": general_results
    }

@app.route("/synthesize_process/", methods=["GET"])
@cross_origin()
def get_synthesize_status():
    status = app.config['SYSNTHESIZE_STATUS']
    return {"status_code": status}

@app.route("/synthesize/", methods=["POST"])
@cross_origin()
def synthesize():
    request_list = request.get_json()
    app.config['SYSNTHESIZE_STATUS'] = 300
    app.config['SYSNTHESIZE_RESULT'] = []

    dir_time = strftime("%Y%m%d%H%M%S", gmtime())

    for request_json in request_list:
        text = request_json["text"]
        model_type = request_json["voice"]

        options = {
            "rate": float(request_json.get("rate", 1.0)),
            "pitch": float(request_json.get("pitch", 1.0)),
            "volume": float(request_json.get("volume", 0.0))
        }

        for model in model_type:
            response_code, results = FileHandler.get_synthesized_audio(text, model, os.path.join(WAVES_FOLDER, dir_time), **options)

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

                    app.config['SYSNTHESIZE_RESULT'] += results

    app.config['SYSNTHESIZE_STATUS'] = 200

    return {"status": 200}


class InvalidVoice(Exception):
    pass

@app.route("/get_user_dict/", methods=["POST"])
@cross_origin()
def get_user_dict():
    request_json = request.get_json()

    model_type = request_json.get("voice")

    response_code = 1
    try:
        if model_type not in _valid_model_types:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(_valid_model_types))

        model = models[model_type]
        result = model.get_user_dict()

        response_code = 0
    except InvalidVoice as e:
        result = str(e)
    except Exception as e:
        result = str(e)

    return {
        "response_code": response_code,
        "response": result
    }


@app.route("/update_user_dict/", methods=["POST"])
@cross_origin()
def update_user_dict():
    request_json = request.get_json()

    model_type = request_json.get("voice")
    user_dict = request_json.get("user_dict")

    response_code = 1
    try:
        if model_type not in _valid_model_types:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(_valid_model_types))

        model = models[model_type]
        model.update_user_dict(user_dict)

        result = "User dictionary has been updated"
        response_code = 0
    except InvalidVoice as e:
        result = str(e)
    except Exception as e:
        result = str(e)

    return {
        "response_code": response_code,
        "response": result
    }


@app.route("/replace_user_dict/", methods=["POST"])
@cross_origin()
def replace_user_dict():
    request_json = request.get_json()

    model_type = request_json.get("voice")
    user_dict = request_json.get("user_dict")

    response_code = 1
    try:
        if model_type not in _valid_model_types:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(_valid_model_types))

        model = models[model_type]
        model.replace_user_dict(user_dict)

        result = "User dictionary has been replaced"
        response_code = 0
    except InvalidVoice as e:
        result = str(e)
    except Exception as e:
        result = str(e)

    return {
        "response_code": response_code,
        "response": result
    }


@app.route("/media/<path:filename>", methods=["GET"])
@cross_origin()
def media_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=False)


def main():
    FlaskUI(app=app, server="flask").run()
    # app.run()
