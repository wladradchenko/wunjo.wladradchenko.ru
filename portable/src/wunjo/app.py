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
from backend.models import load_voice_models, voice_names, file_voice_config
from backend.folders import MEDIA_FOLDER, WAVES_FOLDER, TALKER_FOLDER, TMP_FOLDER


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config['DEBUG'] = False
app.config['SYSNTHESIZE_STATUS'] = 200
app.config['SYSNTHESIZE_RESULT'] = []
app.config['SYSNTHESIZE_TALKER_RESULT'] = []

app.config['models'], app.config['_valid_model_types'] = {}, []
# get list of all directories in folder

def get_avatars_static():
    return {voice_name: url_for("media_file", filename=f"avatar/{voice_name}.png") for voice_name in voice_names}

def split_input_talker(input_param):
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
    return render_template("index.html", existing_models=get_avatars_static())

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
    expression_scale = float(request_list.get("expression_scale", 1.0))
    input_yaw = split_input_talker(request_list.get("input_yaw"))
    input_pitch = split_input_talker(request_list.get("input_pitch"))
    input_roll = split_input_talker(request_list.get("input_roll"))
    background_enhancer = request_list.get("background_enhancer")

    try:
        talker_result = main_talker(
            talker_dir=TALKER_FOLDER,
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
    #except BaseException:
    except Exception as e:
        print(e)
        app.config['SYSNTHESIZE_TALKER_RESULT'] += [{"response_video_url": "", "response_video_date": "Лицо не найдено"}]
        app.config['SYSNTHESIZE_STATUS'] = 200
        return {"status": 400}


    talker_filename = "/video/" + talker_result.replace("\\", "/").split("/video/")[-1]
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
        if model_type not in app.config['_valid_model_types']:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(app.config['_valid_model_types']))

        model = app.config['models'][model_type]
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
        if model_type not in app.config['_valid_model_types']:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(app.config['_valid_model_types']))

        model = app.config['models'][model_type]
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
        if model_type not in app.config['_valid_model_types']:
            raise InvalidVoice("Parameter 'voice' must be one of the following: {}".format(app.config['_valid_model_types']))

        model = app.config['models'][model_type]
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