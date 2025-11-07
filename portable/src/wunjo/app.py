# Copyright (c) 2024 Wladislav Radchenko
# All rights reserved.
# This file is proprietary and may not be reproduced, distributed, or modified without express permission.

import os
import gc
import sys
import uuid
import json
import torch
import socket
import signal
import shutil
import base64
import psutil
import requests
import threading
from apscheduler.schedulers.background import BackgroundScheduler

from time import sleep
from datetime import datetime, timedelta
from cryptography.fernet import InvalidToken
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # remove msg
from werkzeug.utils import secure_filename

import re
from flask import Flask, Response, render_template, request, g, url_for, jsonify, redirect, session
from flask_babel import Babel, _ as _b
from flask_cors import CORS, cross_origin
from flasgger import Swagger, swag_from
if (sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32':
    from flaskwebgui import FlaskUI, close_application
else:
    FlaskUI, close_application = None, None
from ipaddress import ip_address

imports_finished = False
lprint = lambda msg, is_server=True, level=None, user=None: print(msg)


def delayed_import():
    global imports_finished, DIFFUSERS_AVAILABLE
    global FaceSwap, Enhancement, SAMGetSegment, RemoveBackground, RemoveObject, LipSync, Analyser, LivePortraitRetargeting
    global LipSync, Analyser, LivePortraitRetargeting
    global SceneDetect, ImageGeneration, Separator, IdentityPreservingGeneration, Highlights
    global version_app, current_time, is_ffmpeg_installed, get_folder_size, lprint, get_vram_gb, get_ram_gb, move_folder, format_dir_time, check_tmp_file_uploaded, generate_file_tree, model_cache

    lprint("Start delay imports")
    if sys.platform == 'win32':
        import transformers.utils.logging  # fix _default_handler.flush = sys.stderr.flush is None
        transformers.utils.logging._configure_library_root_logger = lambda: None

    # Imports
    from visual_processing.inference import FaceSwap, Enhancement, SAMGetSegment, RemoveBackground, RemoveObject, LipSync, Analyser, LivePortraitRetargeting

    try:
        from visual_generation.inference import SceneDetect, ImageGeneration, IdentityPreservingGeneration, Highlights
        DIFFUSERS_AVAILABLE = True
    except Exception as err:
        DIFFUSERS_AVAILABLE = False
    from sound_processing.inference import Separator

    from backend.general_utils import (
        get_version_app, current_time, is_ffmpeg_installed, get_folder_size, lprint, get_vram_gb, get_ram_gb,
        move_folder, format_dir_time, check_tmp_file_uploaded, generate_file_tree, model_cache
    )

    # Control parameters
    version_app = get_version_app()
    imports_finished = True
    lprint("Finished delay imports")


from backend.folders import (
    MEDIA_FOLDER, TMP_FOLDER, SETTING_FOLDER, CONTENT_FOLDER, CONTENT_REMOVE_OBJECT_FOLDER_NAME,
    CONTENT_ENHANCEMENT_FOLDER_NAME, CONTENT_FACE_SWAP_FOLDER_NAME, CONTENT_AVATARS_FOLDER_NAME, CONTENT_SMART_CUT_NAME,
    CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_LIP_SYNC_FOLDER_NAME, ALL_MODEL_FOLDER, CONTENT_ANALYSIS_NAME,
    CONTENT_RESTYLE_FOLDER_NAME, CONTENT_SEPARATOR_FOLDER_NAME, CONTENT_CLONE_VOICE_FOLDER_NAME, CONTENT_OUTPAINT_NAME,
    CONTENT_SCENEDETECT_NAME, CONTENT_EBSYNTH_NAME, CONTENT_GENERATION_FOLDER_NAME, CONTENT_IMG2IMG_NAME, CONTENT_IMAGE_COMPOSITION_NAME,
    CONTENT_TXT2IMG_NAME, CONTENT_LIVE_PORTRAIT_FOLDER_NAME, CONTENT_RETARGET_PORTRAIT_NAME, CONTENT_HIGHLIGHTS_FOLDER_NAME
)
from backend.config import Settings, WEBGUI_PATH
from backend.translations import msgids

import logging

# pybabel extract -F babel.cfg -o messages.pot .
# pybabel update -i messages.pot -d translations -l ru (pybabel init -i messages.pot -d translations -l ru)
# pybabel compile -d translations
# briefcase package windows VisualStudio -p msi
# https://briefcase.readthedocs.io/en/stable/how-to/ci.html


def get_locale():
    # if a user is logged in, use the locale from the user settings
    return session.get('LANGUAGE_INTERFACE_CODE', Settings.LANGUAGE_INTERFACE_CODE) if session.get('LANGUAGE_INTERFACE_CODE', Settings.LANGUAGE_INTERFACE_CODE) in ['en', 'ru', 'es', 'zh', 'ko'] else 'en'


def get_timezone():
    user = getattr(g, 'user', None)
    if user is not None:
        return user.timezone


app = Flask(__name__)
# Configuring Swagger
app.config['SWAGGER'] = {
    'title': 'Wunjo', 'openapi': '3.0.2', 'uiversion': 3, 'swagger_ui_css': '/static/flasgger/css/swagger-ui.css',
    'jquery_js': '/static/flasgger/js/jquery.min.js', 'swagger_ui_standalone_preset_js': '/static/flasgger/js/swagger-ui-standalone-preset.js',
    'swagger_ui_bundle_js': '/static/flasgger/js/swagger-ui-bundle.js', 'favicon': '/static/basic/icon/square/green/logo-32.png',
    'static_folder': os.path.join(os.path.dirname(__file__), 'static'), 'static_url_path': '/static', 'tour': Settings.TOUR,
    'template_folder': os.path.join(os.path.dirname(__file__), 'templates'), 'hide_top_bar': True, 'version': Settings.VERSION,
    'termsOfService': '/term-of-service', 'description': 'Official website wunjo.online.',
    'host': Settings.FRONTEND_HOST, 'token': base64.b64encode(f"api:{Settings.SECRET_KEY}".encode()).decode()
}
swagger = Swagger(app)
# Cors
cors = CORS(app)
# Set translation
translation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'translations'))
babel = Babel(app, locale_selector=get_locale, timezone_selector=get_timezone, default_translation_directories=translation_path)
# TODO https://briefcase.readthedocs.io/en/stable/reference/configuration.html
scheduler = BackgroundScheduler()


def set_available_port():
    """Check if the given port is free (available)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", Settings.STATIC_PORT))
        return Settings.STATIC_PORT
    except OSError as e:
        print(f"Static port {Settings.STATIC_PORT} is not free: {e}")
        print(f"Using fallback port: {Settings.DYNAMIC_PORT}")
        return Settings.DYNAMIC_PORT


app.config["CORS_HEADERS"] = "Content-Type"
app.config["SECRET_KEY"] = Settings.SECRET_KEY
os.environ['DEBUG'] = 'False'  # str False or True
app.config['DEBUG'] = os.environ.get('DEBUG', 'False') == 'True'
app.config['PORT'] = set_available_port() if not app.config['DEBUG'] else Settings.STATIC_PORT
app.config['QUEUE_CPU'] = {}
app.config["QUEUE_MODULE"] = {}
app.config['SYNTHESIZE_STATUS'] = {}

logging.getLogger('werkzeug').disabled = True

# Threading
timer = threading.Timer(1.0, delayed_import)
timer.start()

# CODE
## 100 - Busy
## 200 - OK
## 300 - Warn
## 404 - Error

# If need to limit number CPU
# num_cores = os.cpu_count() // 2  # Half from total
# torch.set_num_threads(num_cores)
# briefcase package web static


def clear_cache():
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
    gc.collect()


"""FRONTEND METHODS"""


@app.context_processor
def inject_locale():
    return dict(get_locale=get_locale)


if app.config['DEBUG']:
    @app.before_request
    def reload_templates():
        """Control reload page after HTML change if Debug"""
        app.jinja_env.cache.clear()


@app.after_request
def add_cors_headers(response):
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    return response


# Custom 404 error handler
@app.errorhandler(404)
def page_not_found(error):
    return render_content_page(None, '404.html')


@app.route("/inspect-import")
def inspect_import():
    return {"imports_finished": imports_finished}


@app.route("/login", methods=["GET", "POST"])
@cross_origin()
def login_page():
    if not imports_finished:  # delay heavy imports
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0 if sys.platform == 'win32' else True
        return render_template("loading.html", is_admin=bool(is_admin))

    user_id = user_valid()
    if user_id is None:
        return render_template("login.html", host=Settings.FRONTEND_HOST)
    if request.method == "POST":
        if getattr(Settings, 'SESSION', None) is None:
            setattr(Settings, 'SESSION', True)
            processor("cuda")  # Init processor
        return redirect(url_for('index_page'))
    else:
        return render_template("login.html", host=Settings.FRONTEND_HOST)


def authorization_api_token(token):
    return True


@app.route("/authorization", methods=["GET", "POST"])
@cross_origin()
def authorization():
    user_id = user_valid()
    if user_id is None:
        return {"status": 400, "email": '', "password": '', "token": '', "host": Settings.FRONTEND_HOST, "identifier": Settings.SECRET_KEY}
    return {"status": 200, "email": '', "password": '', "token": '', "host": Settings.FRONTEND_HOST, "identifier": Settings.SECRET_KEY}


@app.route("/logout", methods=["GET"])
@cross_origin()
def logout():
    return {"status": 200}


@app.route("/", methods=["GET"])
@cross_origin()
def index_page():
    # Inspect that path without invalid characters or space
    return render_content_page(None, "index.html", invalid_path=(not all(ord(char) < 128 for char in MEDIA_FOLDER) or ' ' in MEDIA_FOLDER) and sys.platform == 'win32')


def user_valid():
    user_ids = request.headers.get('X-Forwarded-For', request.remote_addr)
    if isinstance(user_ids, str):
        try:
            user_id = user_ids.split(',')[0]
            ip_address(user_id)  # Check if user_id is a valid IP address
            return user_id
        except ValueError:
            return None
    else:
        return None


def render_content_page(content_folder_name, template_name, **kwargs):
    user_id = user_valid()
    if user_id is None or getattr(Settings, "SESSION", None) is None:
        return redirect(url_for('login_page'))
    user_folder_name = user_id.replace(".", "_")
    user_folder = os.path.join(CONTENT_FOLDER, user_folder_name)
    user_content_folder = os.path.join(user_folder, content_folder_name) if content_folder_name else user_folder
    if not os.path.exists(user_content_folder):
        os.makedirs(user_content_folder)
    user_content_file_tree = generate_file_tree(user_content_folder)

    return render_template(
        template_name,
        host=Settings.FRONTEND_HOST,
        version=Settings.VERSION,
        token=base64.b64encode(f'None:None'.encode()).decode(),
        identifier=Settings.SECRET_KEY,
        changelog=json.dumps(version_app),
        user_file_tree=user_content_file_tree,
        user_content_id=user_folder_name,
        folder_name=content_folder_name,
        tour=session.get('TOUR', Settings.TOUR),
        language_tooltip_translation=Settings.LANGUAGE_TOOLTIP_TRANSLATION,
        enable_tooltip_translation=Settings.ENABLE_TOOLTIP_TRANSLATION,
        **kwargs
    )


@app.route("/content", methods=["GET"])
@cross_origin()
def content_page():
    return render_content_page(None, "content.html")


@app.route("/profile", methods=["GET", "POST"])
@cross_origin()
def profile_page():
    user_id = user_valid()
    if request.method == "POST" and "login" not in request.form:
        if "logout" in request.form:
            session.pop(user_id, None)
            logout()
            return redirect(url_for('profile_page'))
        elif "exit" in request.form:
            if user_id != Settings.ADMIN_ID:  # this will be id admin
                session.pop(user_id, None)
                logout()
                return redirect(url_for('profile_page'))
            else:
                if not app.config['DEBUG'] and sys.platform != 'darwin' and ((sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32'):
                    close_application()
                else:
                    port = app.config['PORT']
                    # Stop port
                    for proc in psutil.process_iter():
                        try:
                            for conns in proc.connections(kind="inet"):
                                if conns.laddr.port == port:
                                    proc.send_signal(signal.SIGTERM)
                        except psutil.AccessDenied:
                            continue

    public_link = f"http://0.0.0.0:{app.config['PORT']}"

    email = 'support@wunjo.online'
    expiration_date = None

    return render_content_page(None, "profile.html", user_id=Settings.SECRET_KEY, public_link=public_link, email=email, expiration_date=expiration_date)


@app.route("/settings", methods=["GET"])
@cross_origin()
def settings_page():
    models_dict = {
        **LivePortraitRetargeting.inspect(),
        **FaceSwap.inspect(),
        **LipSync.inspect(),
        **RemoveObject.inspect(),
        **RemoveBackground.inspect(),
        **Enhancement.inspect(),
        **Separator.inspect(),
    }
    return render_content_page(None, "settings.html", models=models_dict)


@app.route("/faq", methods=["GET"])
@cross_origin()
def faq_page():
    return render_content_page(None, "faq.html")


@app.route("/face-swap", methods=["GET"])
@cross_origin()
def face_swap_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_FACE_SWAP
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_FACE_SWAP_FOLDER_NAME, "face_swap.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size, analysis_method=CONTENT_ANALYSIS_NAME)


@app.route("/remove-background", methods=["GET"])
@cross_origin()
def remove_background_page():
    model_sam_media_path = SAMGetSegment.is_model_sam()
    max_duration_sec = Settings.MAX_DURATION_SEC_REMOVE_BACKGROUND
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, "remove_background.html", model_sam=model_sam_media_path, max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/remove-object", methods=["GET"])
@cross_origin()
def remove_object_page():
    model_sam_media_path = SAMGetSegment.is_model_sam()
    max_duration_sec = Settings.MAX_DURATION_SEC_REMOVE_OBJECT
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_REMOVE_OBJECT_FOLDER_NAME, "remove_object.html", model_sam=model_sam_media_path, max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/enhancement", methods=["GET"])
@cross_origin()
def enhancement_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_ENHANCEMENT
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_ENHANCEMENT_FOLDER_NAME, "enhancement.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/lip-sync", methods=["GET"])
@cross_origin()
def lip_sync_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_LIP_SYNC
    max_files_size = Settings.MAX_FILE_SIZE
    total_vram_gb, _, _ = get_vram_gb()
    total_vram_gb = round(total_vram_gb)
    return render_content_page(CONTENT_LIP_SYNC_FOLDER_NAME, "lip_sync.html", total_vram_gb=total_vram_gb, max_duration_sec=max_duration_sec, max_files_size=max_files_size, analysis_method=CONTENT_ANALYSIS_NAME)


@app.route("/restyling", methods=["GET"])
@cross_origin()
def restyling_page():
    if getattr(Settings, "SESSION", None) is None:
        processor('cuda')

    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if use_cpu:
        return redirect(url_for('index_page', message='Restyle requires a GPU to function. If your system supports it, please switch from CPU to GPU in your settings.'))
    if not DIFFUSERS_AVAILABLE:
        return redirect(url_for('index_page', message='Restyle not available on this device.'))
    total_vram_gb, _, _ = get_vram_gb()
    total_vram_gb = round(total_vram_gb)
    if total_vram_gb < 6:
        return redirect(url_for('index_page', message=f"You only have {str(total_vram_gb)} GB of video memory. A minimum of 6 GB is required to operate."))

    max_duration_sec = Settings.MAX_DURATION_SEC_DIFFUSER
    max_files_size = Settings.MAX_FILE_SIZE
    model_sam_media_path = ""
    sd_models = []
    return render_content_page(CONTENT_RESTYLE_FOLDER_NAME, "restyling.html", model_sam=model_sam_media_path, max_duration_sec=max_duration_sec, max_files_size=max_files_size, sd_models=sd_models, scenedetect_method=CONTENT_SCENEDETECT_NAME, control_method=CONTENT_EBSYNTH_NAME, inpaint_method=CONTENT_IMG2IMG_NAME, download_models=Settings.CHECK_DOWNLOAD_SIZE_DIFFUSER)


@app.route("/highlights", methods=["GET"])
@cross_origin()
def highlights_page():
    if getattr(Settings, "SESSION", None) is None:
        processor('cuda')

    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if use_cpu:
        return redirect(url_for('index_page', message='Highlights requires a GPU to function. If your system supports it, please switch from CPU to GPU in your settings.'))
    if not DIFFUSERS_AVAILABLE:
        return redirect(url_for('index_page', message='Highlights not available on this device.'))
    total_vram_gb, _, _ = get_vram_gb()
    total_vram_gb = round(total_vram_gb)
    if total_vram_gb < 6:
        return redirect(url_for('index_page', message=f"You only have {str(total_vram_gb)} GB of video memory. A minimum of 6 GB is required to operate."))

    max_duration_sec = Settings.MAX_DURATION_SEC_HIGHLIGHTS
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_HIGHLIGHTS_FOLDER_NAME, "highlights.html", max_files_size=max_files_size, max_duration_sec=max_duration_sec, cut_method=CONTENT_SMART_CUT_NAME)


@app.route("/generation", methods=["GET"])
@cross_origin()
def generation_page():
    if getattr(Settings, "SESSION", None) is None:
        processor('cuda')

    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if use_cpu:
        return redirect(url_for('index_page', message='Generation requires a GPU to function. If your system supports it, please switch from CPU to GPU in your settings.'))
    if not DIFFUSERS_AVAILABLE:
        return redirect(url_for('index_page', message='Generation not available on this device.'))
    total_vram_gb, _, _ = get_vram_gb()
    total_vram_gb = round(total_vram_gb)
    if total_vram_gb < 6:
        return redirect(url_for('index_page', message=f"You only have {str(total_vram_gb)} GB of video memory. A minimum of 6 GB is required to operate."))

    max_files_size = Settings.MAX_FILE_SIZE
    sd_models = []
    video_models = []
    return render_content_page(CONTENT_GENERATION_FOLDER_NAME, "generation.html", max_files_size=max_files_size, sd_models=sd_models, video_models=video_models, image_composition_method=CONTENT_IMAGE_COMPOSITION_NAME, inpaint_method=CONTENT_IMG2IMG_NAME, txt2img_method=CONTENT_TXT2IMG_NAME, outpaint_method=CONTENT_OUTPAINT_NAME, download_models=Settings.CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER)


@app.route("/separator", methods=["GET"])
@cross_origin()
def separator_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_SEPARATOR
    max_files_size = 40 * 1024 * 1024  # 40 Mb
    return render_content_page(CONTENT_SEPARATOR_FOLDER_NAME, "separator.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/clone-voice", methods=["GET"])
@cross_origin()
def clone_voice_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_CLONE_VOICE
    max_files_size = 40 * 1024 * 1024  # 40 Mb
    max_number_symbols = Settings.MAX_NUMBER_SYNBOLS_CLONE_VOICE
    return render_content_page(CONTENT_CLONE_VOICE_FOLDER_NAME, "clone_voice.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size, max_number_symbols=max_number_symbols, analysis_method=CONTENT_ANALYSIS_NAME)


@app.route("/live-portrait", methods=["GET"])
@cross_origin()
def live_portrait_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_LIVE_PORTRAIT
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_LIVE_PORTRAIT_FOLDER_NAME, "live_portrait.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size, scenedetect_method=CONTENT_SCENEDETECT_NAME, analysis_method=CONTENT_ANALYSIS_NAME, retarget_method=CONTENT_RETARGET_PORTRAIT_NAME)


@app.route('/components_template/<template_name>', methods=['GET'])
def components_template_route(template_name=None):
    if template_name is None:
        return jsonify({'error': 'Template name is required', 'status': 400}), 400
    # Ensure that the template path is safe
    safe_template_name = os.path.basename(template_name)
    html = render_template(f"components/{safe_template_name}")
    return jsonify({'html': html})


"""FRONTEND METHODS"""

"""API LOGICAL"""

"""WORK WITH CONTENT FOLDER FILES"""


@app.route('/open-cache', methods=['GET'])
@app.route('/open-cache/<folder_name>', methods=['GET'])
def open_cache(folder_name=None):
    import platform
    import subprocess
    if folder_name:
        folder_name = os.path.dirname(os.path.abspath(__file__))
    if not folder_name:
        folder_name = MEDIA_FOLDER
    if not os.access(folder_name, os.R_OK):
        return jsonify({"status": "error", "message": "No permission to access the directory."}), 403
    try:
        if platform.system() == "Windows":
            # os.startfile(folder_name)  # Windows
            subprocess.run(f'start explorer /select,"{os.path.join(folder_name)}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", folder_name])
        else:  # Linux
            subprocess.run(["xdg-open", folder_name])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/content-rename', methods=['POST'])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'content_rename.yaml'), methods=['POST'], validation=False)
def content_secure_rename():
    data = request.json
    file_relative_path = data.get('filePath')
    file_name = data.get('fileName')
    new_file_name = data.get('newFileName')

    # Check if file_relative_path and file_name are strings and not empty
    if not isinstance(file_relative_path, str) or not file_relative_path or not isinstance(file_name, str) or not file_name:
        return {"status": 404, "message": "Invalid file path or name."}

    if not isinstance(new_file_name, str) or not new_file_name:
        return {"status": 404, "message": "Invalid new file name."}

    # Secure the file name to prevent directory traversal attacks
    file_name = secure_filename(file_name)
    new_file_name = secure_filename(new_file_name)

    _, file_extension = os.path.splitext(file_name)
    _, new_file_extension = os.path.splitext(new_file_name)

    if file_extension != new_file_extension or len(new_file_name) - len(new_file_extension) == 0:
        return {"status": 404, "message": "Invalid new file name.\nName has to be include file extension in original format"}

    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}

    user_folder_name = user_id.replace(".", "_")
    user_folder = os.path.join(CONTENT_FOLDER, user_folder_name)

    # Validate the file path to prevent directory traversal attacks
    file_path = user_folder
    for relative_path in file_relative_path.replace("\\", "/").split("/"):
        file_path = os.path.join(file_path, relative_path)

    if not os.path.exists(file_path):
        return {"status": 404, "message": "File path does not exist."}

    file_absolute_path = os.path.join(file_path, file_name)
    if not file_absolute_path.startswith(CONTENT_FOLDER):
        return {"status": 404, "message": "Invalid file path."}

    new_file_absolute_path = os.path.join(file_path, new_file_name)
    if not new_file_absolute_path.startswith(CONTENT_FOLDER):
        return {"status": 404, "message": "Invalid new file path."}

    if os.path.exists(new_file_absolute_path):
        return {"status": 404, "message": "File already exist."}

    if os.path.isfile(file_absolute_path):
        try:
            # Rename the file
            os.rename(file_absolute_path, new_file_absolute_path)
            # Rename file change ctime and this reason why calendar filter after rename file will be have today date
            # Return success message
            return {"status": 200, "message": "File renamed successfully. For file set current date.", "newFileName": new_file_name}
        except Exception as e:
            # Handle any errors that may occur during the renaming process
            return {"status": 500, "message": f"Error renaming file: {str(e)}"}
    else:
        # If the file does not exist, return an error message
        return {"status": 404, "message": "File not found."}


@app.route('/content-delete', methods=['POST'])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'content_delete.yaml'), methods=['POST'], validation=False)
def content_secure_delete():
    data = request.json
    file_relative_path = data.get('filePath')
    file_name = data.get('fileName')

    # Check if file_relative_path and file_name are strings and not empty
    if not isinstance(file_relative_path, str) or not file_relative_path or not isinstance(file_name, str) or not file_name:
        return {"status": 404, "message": "Invalid file path or name."}

    # Secure the file name to prevent directory traversal attacks
    file_name = secure_filename(file_name)

    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}

    user_folder_name = user_id.replace(".", "_")
    user_folder = os.path.join(CONTENT_FOLDER, user_folder_name)

    # Validate the file path to prevent directory traversal attacks
    file_path = user_folder
    for relative_path in file_relative_path.replace("\\", "/").split("/"):
        file_path = os.path.join(file_path, relative_path)

    if not os.path.exists(file_path):
        return {"status": 404, "message": "File path does not exist."}

    file_absolute_path = os.path.join(file_path, file_name)
    if not file_absolute_path.startswith(CONTENT_FOLDER):
        return {"status": 404, "message": "Invalid file path."}

    if os.path.isfile(file_absolute_path):
        # Perform file deletion
        try:
            os.remove(file_absolute_path)
            return {"status": 200, "message": "File deleted successfully."}
        except Exception as e:
            return {"status": 500, "message": f"Error deleting file: {str(e)}"}
    else:
        return {"status": 404, "message": "File not found."}


@app.route('/content-reload', methods=['GET'])
@app.route('/content-reload/<folder_name>', methods=['GET'])  # Define the route with a parameter
@cross_origin()
def content_reload(folder_name=None):
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}
    else:
        user_folder_name = user_id.replace(".", "_")
        user_folder = os.path.join(CONTENT_FOLDER, user_folder_name)
        if folder_name is not None:
            user_folder = os.path.join(user_folder, folder_name)
            if not os.path.exists(user_folder):
                return {"status": 404, "message": "File path does not exist."}
        user_file_tree = generate_file_tree(user_folder)  # Get new tree
        return {"status": 200, "message": "Files reloaded.", "user_file_tree": user_file_tree}


@app.route('/upload-tmp', methods=['POST'])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'upload_tmp.yaml'), validation=False)
def upload_file_media():
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    if 'file' not in request.files:
        return {"status": 200, "message": 'No file uploaded'}
    chunk = request.files['file']
    if chunk.filename == '':
        return {"status": 200, "message": 'No file selected'}
    filename = secure_filename(chunk.filename)

    file_path = os.path.join(TMP_FOLDER, filename)
    with open(file_path, 'ab') as f:  # Open in append-binary mode to append chunks
        f.write(chunk.read())  # Write the received chunk to the file

    return {"status": 200, "message": 'Chunk uploaded successfully'}


@app.route("/get-avatars", methods=["GET"])
@cross_origin()
def get_avatars():
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})
    user_folder_name = user_id.replace(".", "_")
    user_avatar_folder = os.path.join(CONTENT_FOLDER, user_folder_name, CONTENT_AVATARS_FOLDER_NAME)
    if not os.path.exists(user_avatar_folder):
        os.makedirs(user_avatar_folder)
    user_avatar_file_tree = generate_file_tree(user_avatar_folder)
    # Message to user if folder is not created yet
    return jsonify({"status": 200, "user_avatars": user_avatar_file_tree, "folder_name": CONTENT_AVATARS_FOLDER_NAME})

"""WORK WITH CONTENT FOLDER FILES"""


"""FEATURE MODELS"""

@app.route("/generate-avatar", methods=["POST"])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'generate_avatar.yaml'), methods=['POST'], validation=False)
def generate_avatar():
    import io
    from PIL import Image

    # Generate a unique ID for the request
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    if request.content_type != 'application/json':
        # If Content-Type is not application/json, extract parameters from query
        gender = request.args.get("gender", "all")
        age = request.args.get("age", "all")
        ethnic = request.args.get("ethnic", "all")
    else:
        # If Content-Type is application/json, extract data from JSON
        data = request.get_json() or {}
        gender = data.get("gender") or request.args.get("gender", "all")
        age = data.get("age") or request.args.get("age", "all")
        ethnic = data.get("ethnic") or request.args.get("ethnic", "all")
    user_folder_name = user_id.replace(".", "_")
    user_avatar_folder = os.path.join(CONTENT_FOLDER, user_folder_name, CONTENT_AVATARS_FOLDER_NAME)

    if not os.path.exists(user_avatar_folder):
        os.makedirs(user_avatar_folder)

    url = f"https://this-person-does-not-exist.com/new?gender={gender}&age={age}&ethnic={ethnic}"

    try:
        response = requests.get(url, timeout=1)
        response.raise_for_status()  # Raise an error for bad responses

        if response.status_code == 200:
            # Parse the JSON response
            json_data = response.json()
            if json_data.get("generated") == "true":
                src = json_data.get("src")
                image_url = f"https://this-person-does-not-exist.com{src}"
                image_response = requests.get(image_url, timeout=1)

                # Check if the response contains image data
                if image_response.headers.get("Content-Type", "").startswith("image"):
                    # Open the image using PIL
                    image = Image.open(io.BytesIO(image_response.content))

                    # Crop the image to around 90% of the actual height and width
                    width, height = image.size
                    crop_width = int(width * 0.89)
                    crop_height = int(height * 0.89)
                    left = (width - crop_width) // 2
                    top = (height - crop_height) // 2
                    right = left + crop_width
                    bottom = top + crop_height
                    cropped_image = image.crop((left, top, right, bottom))

                    # Create a new image with white background
                    new_width, new_height = crop_width + 250, crop_height + 250
                    background_color = (255, 255, 255)  # White color
                    background = Image.new('RGB', (new_width, new_height), background_color)

                    # Paste the cropped image onto the white background
                    paste_left = (new_width - crop_width) // 2
                    paste_top = (new_height - crop_height) // 2
                    background.paste(cropped_image, (paste_left, paste_top))

                    # Save the image to the user's folder
                    image_name = str(uuid.uuid4()) + '.jpg'
                    image_file_path = os.path.join(user_avatar_folder, image_name)
                    background.save(image_file_path, format='JPEG')

                    media_file_path = image_file_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")
                    return jsonify({"status": 200, "media_file": media_file_path})

                else:
                    # Return default image if the response does not contain image data
                    return jsonify({"status": 300, "message": "Invalid image data received"})
            else:
                return jsonify({"status": 300, "message": "Invalid image data received"})
        else:
            return jsonify({"status": 300, "message": "Invalid image data received"})

    except Exception as err:
        return jsonify({"status": 400, "message": str(err)})

"""CPU TASKS"""


def is_ffmpeg(user) -> bool:
    # Check ffmpeg
    if not is_ffmpeg_installed():
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return False
    return True


@app.route("/submit-cpu-task", methods=["POST"])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'submit_cpu_task.yaml'), methods=['POST'], validation=False)
def submit_cpu_task():
    # Generate a unique ID for the request
    request_id = str(uuid.uuid4())
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    api_token = request.headers.get("token")
    if api_token is not None:
        authorization_api_status = authorization_api_token(api_token)
        if not authorization_api_status:
            return jsonify({"status": 404, "message": "Unauthorized"})

    data = request.get_json()
    method = data.pop('method') if data.get('method') is not None else None
    filename = data.get('filename')
    # Keep datetime to clear old sessions
    queue_data = {user_id: {request_id: {"method": method, "data": data, "completed": False, "busy": None, "filename": filename, "response": {}, "date": datetime.now()}}}

    if method not in [CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_REMOVE_OBJECT_FOLDER_NAME, CONTENT_RESTYLE_FOLDER_NAME, CONTENT_ANALYSIS_NAME, CONTENT_SCENEDETECT_NAME, CONTENT_IMG2IMG_NAME, CONTENT_IMAGE_COMPOSITION_NAME, CONTENT_TXT2IMG_NAME, CONTENT_OUTPAINT_NAME, CONTENT_RETARGET_PORTRAIT_NAME, CONTENT_HIGHLIGHTS_FOLDER_NAME]:
        return jsonify({"status": 404, "message": "Method not found for CPU task."})

    for user in queue_data.keys():
        if app.config["QUEUE_CPU"].get(user) is None:
            app.config["QUEUE_CPU"][user] = {}
        for request_id in queue_data[user].keys():
            app.config["QUEUE_CPU"][user][request_id] = queue_data[user][request_id]
            lprint(msg=f"Add in background processing on CPU {request_id}")

    return jsonify({"status": 200, "message": "Updated delay CPU task.", "id": request_id})


@app.route("/query-cpu-task/<request_id>", methods=["GET"])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'query_cpu_task.yaml'), methods=['GET'], validation=False)
def query_cpu_task(request_id):
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    if app.config["QUEUE_CPU"].get(user_id) is not None:
        if app.config["QUEUE_CPU"][user_id].get(request_id) is not None:
            queue_task = app.config["QUEUE_CPU"][user_id][request_id]
            completed = queue_task.get("completed", False) if isinstance(queue_task, dict) else False
            progress_message = queue_task.get("progress_message", "Prepare...")
            progress = queue_task.get("progress", 0)

            if isinstance(queue_task, dict) and completed:
                response = queue_task.get("response")
                app.config["QUEUE_CPU"][user_id].pop(request_id)  # Clear
                return jsonify({"status": 200, "message": "Successfully!", "response": response })
            if completed == "cancel":
                # Canceled task
                app.config["QUEUE_CPU"][user_id].pop(request_id)  # Clear
                return jsonify({"status": 404, "message": "Task is canceled.", "response": None })
            return jsonify({"status": 100, "message": "Busy.", "response": None, "progress": progress, "progress_message": progress_message })
    return jsonify({"status": 300, "message": "Not found task.", "response": None})


def get_retarget_portrait(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        parameters = send_data.get("data")
        filename = parameters.get("filename")
        file_path = os.path.join(TMP_FOLDER, filename)

        data = parameters.get("retargetParameters", {})
        eye_ratio = float(data.get("eyeRatio", 0))
        lip_ratio = float(data.get("lipRatio", 0))
        pitch = float(data.get("pitch", 0))
        yaw = float(data.get("yaw", 0))
        roll = float(data.get("roll", 0))
        x = float(data.get("headX", 0))
        y = float(data.get("headY", 0))
        z = float(data.get("headZ", 1))
        eyeball_direction_x = float(data.get("eyeballX", 0))
        eyeball_direction_y = float(data.get("eyeballY", 0))
        smile = float(data.get("smile", 0))
        wink = float(data.get("wink", 0))
        eyebrow = float(data.get("eyebrow", 0))
        grin = float(data.get("grin", 0))
        pursing = float(data.get("pursing", 0))
        pouting = float(data.get("pouting", 0))
        lip_expression = float(data.get("lipExpression", 0))
        # Crop parameters
        natural_width = int(data.get("naturalWidth", 0))
        natural_height = int(data.get("naturalHeight", 0))
        changing_crop_face = data.get("cropField", [])  # list of dict(x, y, width, height)
        valid_crops = LivePortraitRetargeting.is_valid_image_area(changing_crop_face, natural_width, natural_height)

        if check_tmp_file_uploaded(file_path) and LivePortraitRetargeting.is_valid_image(file_path):
            lprint(msg="Start retarget portrait image.", user=user)
            response = LivePortraitRetargeting.retargeting_image(
                source=file_path, user_name=user, model_cache=model_cache, eye_ratio=eye_ratio, lip_ratio=lip_ratio,
                pitch=pitch, yaw=yaw, roll=roll, x=x, y=y, z=z, eyeball_direction_x=eyeball_direction_x,
                eyeball_direction_y=eyeball_direction_y, smile=smile, wink=wink, eyebrow=eyebrow, grin=grin,
                pursing=pursing, pouting=pouting, lip_expression=lip_expression,
                crops=valid_crops, progress_callback=progress_update
            )
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_sam_embedding(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        filename = send_data.get("filename")
        file_path = os.path.join(TMP_FOLDER, filename)
        if check_tmp_file_uploaded(file_path) and SAMGetSegment.is_valid_image(file_path):
            image_data = SAMGetSegment.read(file_path)  # Read data from path
            lprint(msg="Start to get embedding by SAM.", user=user)
            response = SAMGetSegment.extract_embedding(image_data, model_cache=model_cache, progress_callback=progress_update)
            SAMGetSegment.clear(file_path)  # Remove tpm file
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_analysis(user, request_id) -> None:
    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        filename = send_data.get("filename")
        file_path = os.path.join(TMP_FOLDER, filename)

        if check_tmp_file_uploaded(file_path) and Analyser.is_valid_file(file_path, ["video", "image"]):
            lprint(msg="Start to get analysis content.", user=user)
            response = Analyser.analysis(user_name=user, source_file=file_path)
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_scenedetect(user, request_id) -> None:
    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        filename = send_data.get("filename")
        data = send_data.get("data")
        interval = int(data["interval"]) if data is not None and data.get("interval") else None
        threshold = float(data["threshold"]) if data is not None and data.get("threshold") else 50  # 50%
        file_path = os.path.join(TMP_FOLDER, filename)

        if check_tmp_file_uploaded(file_path) and SceneDetect.is_valid_video(file_path):
            lprint(msg="Start to get scenedetect content.", user=user)
            response = SceneDetect.detecting(user_name=user, source_file=file_path, interval=interval, threshold=threshold)
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)


def get_inpaint(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        data = send_data.get("data") if send_data.get("data") else None
        filename = data.get("filename")
        file_path = os.path.join(TMP_FOLDER, str(filename))
        inpaint_filename = data.get("canvasFileName")
        inpaint_file_path = os.path.join(TMP_FOLDER, str(inpaint_filename))
        sd_model = data.get("sdModel")
        eta = data.get("eta", 1)
        diffusion_parameters = data.get("diffusionParameters") if data.get("diffusionParameters") else {}

        if check_tmp_file_uploaded(file_path) and ImageGeneration.is_valid_image(file_path) and check_tmp_file_uploaded(inpaint_file_path) and ImageGeneration.is_valid_image(inpaint_file_path):
            lprint(msg="Start to inpaint content.", user=user)
            response = ImageGeneration.inpaint(user_name=user, source_file=file_path, sd_model_name=sd_model, params=diffusion_parameters, inpaint_file=inpaint_file_path, eta=eta, model_cache=model_cache, progress_callback=progress_update)
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_image_composition(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        data = send_data.get("data") if send_data.get("data") else None
        seed = int(data.get("seed", 1))
        scale = float(data.get("scale", 2.5))
        width = int(data.get("width", 512))
        height = int(data.get("height", 512))
        prompt = data.get("prompt")
        input_images = data.get("inputImages", [])
        input_path_images = []
        total_vram_gb, _, _ = get_vram_gb()

        for input_image in input_images:
            file_path = os.path.join(TMP_FOLDER, str(input_image))
            if not check_tmp_file_uploaded(file_path) or not IdentityPreservingGeneration.is_valid_image(file_path):
                handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
                return
            input_path_images.append(file_path)
        else:
            lprint(msg="Start to generate image composition content.", user=user)
            response = IdentityPreservingGeneration.image_composition(
                user_name=user, input_images=input_path_images, prompt=prompt, seed=seed, scale=scale, width=width,
                height=height, model_cache=model_cache, progress_callback=progress_update
            )
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_txt2img(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        data = send_data.get("data") if send_data.get("data") else None
        sd_model = data.get("sdModel")
        width = data.get("width")
        height = data.get("height")
        diffusion_parameters = data.get("diffusionParameters") if data.get("diffusionParameters") else {}

        lprint(msg="Start to generate image content.", user=user)
        response = ImageGeneration.text_to_image(user_name=user, sd_model_name=sd_model, params=diffusion_parameters, width=width, height=height, progress_callback=progress_update)
        handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_outpaint(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        data = send_data.get("data") if send_data.get("data") else None
        filename = data.get("filename")
        file_path = os.path.join(TMP_FOLDER, str(filename))
        sd_model = data.get("sdModel")
        width = data.get("width")
        height = data.get("height")
        blur_factor = data.get("blurFactor")
        diffusion_parameters = data.get("diffusionParameters") if data.get("diffusionParameters") else {}

        if check_tmp_file_uploaded(file_path) and ImageGeneration.is_valid_image(file_path):
            lprint(msg="Start to outpaint image content.", user=user)
            response = ImageGeneration.outpaint(source_file=file_path, user_name=user, sd_model_name=sd_model, params=diffusion_parameters, width=width, height=height, blur_factor=blur_factor, model_cache=model_cache, progress_callback=progress_update)
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


def get_chat(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message, queue_key="QUEUE_CPU")

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        data = send_data.get("data") if send_data.get("data") else None
        filename = data.get("filename")
        file_path = os.path.join(TMP_FOLDER, str(filename))
        prompt = data.get("prompt")
        is_transcribe = data.get("isTranscribe", False)
        is_transcribe_words = data.get("isTranscribeWords")
        start_time = data.get("startTime")
        end_time = data.get("endTime")
        mode = "sound" if data.get("isSoundAnalysis") else "visual"

        if check_tmp_file_uploaded(file_path) and (Highlights.is_valid_video(file_path) or Highlights.is_valid_audio(file_path)):
            lprint(msg="Start to analysis content and chat.", user=user)
            response = Highlights.assistant(source_file=file_path, user_name=user, prompt=prompt, mode=mode, source_start=start_time,
                                            source_end=end_time, is_transcribe=is_transcribe, is_transcribe_words=is_transcribe_words,
                                            model_cache=model_cache, progress_callback=progress_update)
            handle_task_completion(user, request_id, response, queue_key="QUEUE_CPU")
        else:
            handle_invalid_content(user, request_id, queue_key="QUEUE_CPU")
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()


"""CPU TASKS"""
"""GPU TASKS"""


@app.route("/submit-module-task", methods=["POST"])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'submit_module_task.yaml'), methods=['POST'], validation=False)
def submit_module_task():
    # Generate a unique ID for the request
    request_id = str(uuid.uuid4())
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    if getattr(Settings, 'SESSION', None) is None:
        setattr(Settings, 'SESSION', True)
        processor("cuda")  # Init processor

    api_token = request.headers.get("token")
    if api_token is not None:
        authorization_api_status = authorization_api_token(api_token)
        if not authorization_api_status:
            return jsonify({"status": 404, "message": "Unauthorized"})

    data = request.get_json()
    method = data.pop('method') if data.get('method') is not None else None
    sub_method = data.pop('subMethod') if data.get('subMethod') is not None else method
    # Keep datetime to clear old sessions
    queue_data = {user_id: {request_id: {"method": method, "sub_method": sub_method, "data": data, "completed": False, "busy": None, "response": {}, "date": datetime.now()}}}

    module_methods = [CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_REMOVE_OBJECT_FOLDER_NAME,
                      CONTENT_FACE_SWAP_FOLDER_NAME, CONTENT_ENHANCEMENT_FOLDER_NAME, CONTENT_CLONE_VOICE_FOLDER_NAME,
                      CONTENT_LIP_SYNC_FOLDER_NAME, CONTENT_RESTYLE_FOLDER_NAME, CONTENT_SEPARATOR_FOLDER_NAME,
                      CONTENT_EBSYNTH_NAME, CONTENT_GENERATION_FOLDER_NAME, CONTENT_LIVE_PORTRAIT_FOLDER_NAME,
                      CONTENT_RETARGET_PORTRAIT_NAME, CONTENT_HIGHLIGHTS_FOLDER_NAME, CONTENT_SMART_CUT_NAME]

    if method not in module_methods:
        return jsonify({"status": 404, "message": "Method not found for module task."})

    for user in queue_data.keys():
        if app.config["QUEUE_MODULE"].get(user) is None:
            app.config["QUEUE_MODULE"][user] = {}
        for request_id in queue_data[user].keys():
            app.config["QUEUE_MODULE"][user][request_id] = queue_data[user][request_id]
            # print(app.config["QUEUE_MODULE"][user][request_id])  # TODO get datat for test here
            lprint(msg=f"Add in background module processing on {str(os.environ.get('WUNJO_TORCH_DEVICE', 'cpu')).upper()} {request_id}", user=user_id)

    return jsonify({"status": 200, "message": "Updated delay module task.", "id": request_id})


@app.route("/query-module-task/<method>", methods=["GET"])
@cross_origin()
@swag_from(os.path.join(os.path.dirname(__file__), 'api', 'query_module_task.yaml'), methods=['GET'], validation=False)
def query_module_task(method):
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})
    user_busy_task = []

    if app.config["QUEUE_MODULE"].get(user_id) is not None:
        user_queue_module = app.config["QUEUE_MODULE"][user_id].copy()
        request_ids = list(user_queue_module.keys())
        for request_id in request_ids:
            user_method = user_queue_module[request_id].get("method")
            completed = user_queue_module[request_id].get("completed", False)
            progress_message = user_queue_module[request_id].get("progress_message", "Prepare...")
            progress = user_queue_module[request_id].get("progress", 0)

            if method == user_method and completed:
                output = app.config["QUEUE_MODULE"][user_id][request_id].get("response")
                app.config["QUEUE_MODULE"][user_id].pop(request_id)  # Clear completed task
                return jsonify({"status": 201, "message": "Successfully!", "response": [{"request_id": request_id, "output": output}]})
            if completed == "cancel":
                # Remove canceled task
                app.config["QUEUE_MODULE"][user_id].pop(request_id)  # Clear
            if method == user_method and not completed:
                created = user_queue_module[request_id].get("date")  # date created
                data = user_queue_module[request_id].get("data")  # parameters data
                module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
                user_save_name = module_parameters.get("nameFile")
                user_busy_task.append({"request_id": request_id, "name": user_save_name, "created": created, "progress": progress, "progress_message": progress_message})
        else:
            if len(user_busy_task) > 0:
                return jsonify({"status": 100, "message": "Busy.", "response": user_busy_task})
            return jsonify({"status": 200, "message": f"Not found module {method} tasks.", "response": None})
    return jsonify({"status": 200, "message": "Not found task for user.", "response": None})


def save_metadata(response: dict, metadata: dict) -> None:
    """
    Stores JSON metadata alongside the result file.
    :param response: A dictionary containing the key 'file' with the path to the result
    :param metadata: Metadata to save in JSON
    """
    import yaml
    output_list = response.get("file")
    if not output_list:
        return
    if isinstance(output_list, str):
        output_list = [output_list]
    if isinstance(output_list, list):
        for output_file in output_list:
            yaml_file = os.path.splitext(output_file)[0] + ".yaml"
            try:
                with open(yaml_file, "w", encoding="utf-8") as f:
                    yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, indent=4)
                lprint(f"Metadata saved: {yaml_file}")
            except Exception as e:
                lprint(f"Failed to save metadata: {e}")


def handle_task_completion(user, request_id, response, queue_key=None):
    if queue_key is None or queue_key not in ["QUEUE_MODULE", "QUEUE_CPU"]:
        queue_key = "QUEUE_MODULE"
    if app.config[queue_key][user].get(request_id):
        app.config[queue_key][user][request_id]["response"] = response
        app.config[queue_key][user][request_id]["busy"] = False
        app.config[queue_key][user][request_id]["completed"] = True
        app.config[queue_key][user][request_id]["completed_date"] = datetime.now()
    # Calculate only QUEUE_MODULE
    is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
    if any(is_busy_list):
        lprint(msg=f"Task {request_id} successfully finished but process is still busy.", user=user)
    else:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id}."}


def handle_invalid_content(user, request_id, queue_key=None):
    if queue_key is None or queue_key not in ["QUEUE_MODULE", "QUEUE_CPU"]:
        queue_key = "QUEUE_MODULE"
    if app.config[queue_key][user].get(request_id):
        app.config[queue_key][user][request_id]["response"] = None
        app.config[queue_key][user][request_id]["busy"] = False
        app.config[queue_key][user][request_id]["completed"] = "cancel"
        app.config[queue_key][user][request_id]["completed_date"] = datetime.now()
    # Calculate only QUEUE_MODULE
    is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
    if any(is_busy_list):
        lprint(msg=f"This is not a right file format {request_id}, but the process is still busy.", user=user, level="warn")
    else:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not a right file format {request_id}."}


def handle_exception(user, request_id, err, queue_key=None):
    if queue_key is None or queue_key not in ["QUEUE_MODULE", "QUEUE_CPU"]:
        queue_key = "QUEUE_MODULE"
    lprint(msg=str(err), level="error", user=user)

    if app.config[queue_key][user].get(request_id):
        app.config[queue_key][user].pop(request_id)

    app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with task {request_id}: {str(err)}."}


def handle_progress_update(user, request_id, progress, progress_message, queue_key=None):
    if queue_key is None or queue_key not in ["QUEUE_MODULE", "QUEUE_CPU"]:
        queue_key = "QUEUE_MODULE"
    if app.config[queue_key][user].get(request_id):
        app.config[queue_key][user][request_id]["progress"] = progress
        app.config[queue_key][user][request_id]["progress_message"] = progress_message


def synthesize_remove_background(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_REMOVE_BACKGROUND_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_REMOVE_BACKGROUND_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        background_color = module_parameters.get("backgroundColor")
        inverse_mask = module_parameters.get("reverseSelect", False)
        point_list = module_parameters.get("SAM", [])  # list of dict(x, y, clickType)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_offset_width = source_file_parameters.get("offsetWidth")
        source_offset_height = source_file_parameters.get("offsetHeight")
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_REMOVE_BACKGROUND)

        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        if check_tmp_file_uploaded(source_file_path) and RemoveBackground.is_valid_file(source_file_path, ["video", "image"]):
            valid_point_list = RemoveBackground.is_valid_point_list(point_list, source_offset_width, source_offset_height)
            if len(valid_point_list.keys()) == 0:
                raise

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
            model_cache.clear()  # Unload CPU task models
            response = RemoveBackground.retries_synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, inverse_mask=inverse_mask,
                masks=valid_point_list, user_save_name=user_save_name,  segment_percentage=25,
                source_start=source_start_time, source_end=source_end_time, progress_callback=progress_update,
                max_size=Settings.MAX_RESOLUTION_REMOVE_BACKGROUND, mask_color=background_color
            )
            handle_task_completion(user, request_id, response)
            # Stories metadata and history
            save_metadata(response, data)
            update_history(response.get("file"), response.get("media"), CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, user)
        else:
            handle_invalid_content(user, request_id)

    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_remove_object(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_REMOVE_OBJECT_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_REMOVE_OBJECT_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        use_improved_model = module_parameters.get("modelImproved", False)
        use_remove_text = module_parameters.get("useRemoveText", False)
        inverse_mask = module_parameters.get("reverseSelect", False)
        point_list = module_parameters.get("SAM", [])  # list of dict(x, y, clickType)
        crop_text = module_parameters.get("cropField", [])  # list of dict(x, y, width, height)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_offset_width = source_file_parameters.get("offsetWidth")
        source_offset_height = source_file_parameters.get("offsetHeight")
        source_natural_width = source_file_parameters.get("naturalWidth")
        source_natural_height = source_file_parameters.get("naturalHeight")
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_REMOVE_OBJECT)

        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        # valid data
        if check_tmp_file_uploaded(source_file_path):
            if RemoveObject.is_valid_image(source_file_path):
                valid_text_area = RemoveObject.is_valid_image_area(crop_text, source_natural_width, source_natural_height)
            elif RemoveObject.is_valid_video(source_file_path):
                valid_text_area = RemoveObject.is_valid_video_area(crop_text, source_offset_width, source_offset_height, source_natural_width, source_natural_height)
            else:
                raise Exception("Invalid content")
        else:
            raise Exception("Content not uploaded")

        valid_point_list = RemoveObject.is_valid_point_list(point_list, source_offset_width, source_offset_height)

        if (len(valid_point_list.keys()) == 0 and not use_remove_text) and (len(valid_text_area.keys()) == 0 and use_remove_text):
            raise Exception("Invalid points or crop field")

        processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
        model_cache.clear()  # Unload CPU task models
        response = RemoveObject.retries_synthesis(
            folder_name=folder_name, user_name=user_folder, source_file=source_file_path, use_improved_model=use_improved_model,
            masks=valid_point_list, user_save_name=user_save_name,  segment_percentage=25, source_start=source_start_time,
            inverse_mask=inverse_mask, source_end=source_end_time, max_size=Settings.MAX_RESOLUTION_REMOVE_OBJECT,
            text_area=valid_text_area, use_remove_text=use_remove_text, progress_callback=progress_update
        )
        handle_task_completion(user, request_id, response)
        # Stories metadata and history
        save_metadata(response, data)
        update_history(response.get("file"), response.get("media"), CONTENT_REMOVE_OBJECT_FOLDER_NAME, user)
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_face_swap(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_FACE_SWAP_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_FACE_SWAP_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        is_gemini_faces = module_parameters.get("isGeminiFaces", False)
        replace_all_faces = module_parameters.get("replaceAllFaces", False)
        avatar_crop_face = module_parameters.get("avatarCropField", [])  # list of dict(x, y, width, height)
        changing_crop_face = module_parameters.get("cropField", [])  # list of dict(x, y, width, height)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        source_offset_width = int(source_file_parameters.get("offsetWidth"))
        source_offset_height = int(source_file_parameters.get("offsetHeight"))
        source_natural_width = int(source_file_parameters.get("naturalWidth"))
        source_natural_height = int(source_file_parameters.get("naturalHeight"))
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_FACE_SWAP)

        avatar_file_parameters = data.get("avatarFile") if isinstance(data, dict) else {}
        avatar_file_name = avatar_file_parameters.get("nameFile")
        avatar_file_path = os.path.join(TMP_FOLDER, str(avatar_file_name))
        avatar_natural_width = int(source_file_parameters.get("naturalWidth"))
        avatar_natural_height = int(source_file_parameters.get("naturalHeight"))

        # valid data
        if check_tmp_file_uploaded(source_file_path):
            if FaceSwap.is_valid_image(source_file_path):
                valid_source_area = FaceSwap.is_valid_image_area(changing_crop_face, source_natural_width, source_natural_height)
            elif FaceSwap.is_valid_video(source_file_path):
                valid_source_area = FaceSwap.is_valid_video_area(changing_crop_face, source_offset_width, source_offset_height, source_natural_width, source_natural_height)
            else:
                raise Exception("Invalid content")
        else:
            raise Exception("Content not uploaded")

        if check_tmp_file_uploaded(avatar_file_path):
            if FaceSwap.is_valid_image(avatar_file_path):
                valid_avatar_area = FaceSwap.is_valid_image_area(avatar_crop_face, avatar_natural_width, avatar_natural_height)
            else:
                raise Exception("Invalid content")
        else:
            raise Exception("Content not uploaded")

        resize_factor = FaceSwap.get_resize_factor(source_natural_width, source_natural_height, max_size=Settings.MAX_RESOLUTION_FACE_SWAP)

        processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
        model_cache.clear()  # Unload CPU task models
        response = FaceSwap.synthesis(
            folder_name=folder_name, user_name=user_folder, source_file=source_file_path, avatar_file=avatar_file_path,
            user_save_name=user_save_name, source_crop_area=valid_source_area, avatar_crop_area=valid_avatar_area,
            source_start=source_start_time, source_end=source_end_time, is_gemini_faces=is_gemini_faces,
            replace_all_faces=replace_all_faces, resize_factor=resize_factor, progress_callback=progress_update
        )
        handle_task_completion(user, request_id, response)
        # Stories metadata and history
        save_metadata(response, data)
        update_history(response.get("file"), response.get("media"), CONTENT_FACE_SWAP_FOLDER_NAME, user)
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_enhancement(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_ENHANCEMENT_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_ENHANCEMENT_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        enhancement_model = module_parameters.get("model", "gfpgan")

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))

        is_valid_source = True if check_tmp_file_uploaded(source_file_path) and Enhancement.is_valid_file(source_file_path, ["video", "image"]) else False

        if is_valid_source:
            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
            model_cache.clear()  # Unload CPU task models
            response = Enhancement.retries_synthesis(
                folder_name=folder_name, user_name=user, source_file=source_file_path,
                user_save_name=user_save_name, enhancer=enhancement_model, progress_callback=progress_update
            )
            handle_task_completion(user, request_id, response)
            # Stories metadata and history
            save_metadata(response, data)
            update_history(response.get("file"), response.get("media"), CONTENT_ENHANCEMENT_FOLDER_NAME, user)
        else:
            handle_invalid_content(user, request_id)

    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_lip_sync(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_LIP_SYNC_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_LIP_SYNC_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        changing_crop_face = module_parameters.get("cropField", [])  # list of dict(x, y, width, height)
        overlay_audio = module_parameters.get("overlayAudio", False)
        use_gfpgan = module_parameters.get("useGfpgan", False)
        use_smooth = module_parameters.get("useSmooth", False)
        use_diffuser = module_parameters.get("useDiffuser", False)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        source_offset_width = int(source_file_parameters.get("offsetWidth"))
        source_offset_height = int(source_file_parameters.get("offsetHeight"))
        source_natural_width = int(source_file_parameters.get("naturalWidth"))
        source_natural_height = int(source_file_parameters.get("naturalHeight"))
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_LIP_SYNC)

        audio_file_parameters = data.get("audioFile") if isinstance(data, dict) else {}
        audio_file_name = audio_file_parameters.get("nameFile")
        audio_file_path = os.path.join(TMP_FOLDER, str(audio_file_name))
        audio_start_time = audio_file_parameters.get("startTime")
        audio_end_time = audio_file_parameters.get("endTime")

        if audio_start_time and audio_end_time:  # set what time not more limits
            audio_start_time = float(audio_start_time)
            audio_end_time = min(float(audio_end_time), float(audio_start_time) + Settings.MAX_DURATION_SEC_LIP_SYNC)

        # valid data
        if check_tmp_file_uploaded(source_file_path):
            if LipSync.is_valid_image(source_file_path):
                valid_source_area = LipSync.is_valid_image_area(changing_crop_face, source_natural_width, source_natural_height)
            elif LipSync.is_valid_video(source_file_path):
                valid_source_area = LipSync.is_valid_video_area(changing_crop_face, source_offset_width, source_offset_height, source_natural_width, source_natural_height)
            else:
                raise Exception("Invalid content")
        else:
            raise Exception("Content not uploaded")

        if not (check_tmp_file_uploaded(audio_file_path) and LipSync.is_valid_audio(audio_file_path)):
            raise Exception("Audio not uploaded")

        resize_factor = LipSync.get_resize_factor(source_natural_width, source_natural_height, max_size=Settings.MAX_RESOLUTION_FACE_SWAP)

        processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
        model_cache.clear()  # Unload CPU task models
        response = LipSync.retries_synthesis(
            folder_name=folder_name, user_name=user_folder, source_file=source_file_path, audio_file=audio_file_path,
            user_save_name=user_save_name, source_crop_area=valid_source_area, source_start=source_start_time,
            source_end=source_end_time, audio_start=audio_start_time, audio_end=audio_end_time, overlay_audio=overlay_audio,
            resize_factor=resize_factor, use_gfpgan=use_gfpgan, use_smooth=use_smooth, use_diffuser=use_diffuser, progress_callback=progress_update
        )
        handle_task_completion(user, request_id, response)
        # Stories metadata and history
        save_metadata(response, data)
        update_history(response.get("file"), response.get("media"), CONTENT_LIP_SYNC_FOLDER_NAME, user)
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_separator(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_SEPARATOR_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_SEPARATOR_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        extract_method = module_parameters.get("extractMethod", "vocals")
        trim_silence = module_parameters.get("trimSilence", False)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_SEPARATOR)

        # valid data
        is_valid_source = True if check_tmp_file_uploaded(source_file_path) and Separator.is_valid_audio(source_file_path) else False

        if is_valid_source:
            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
            model_cache.clear()  # Unload CPU task models
            response = Separator.synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, trim_silence=trim_silence,
                user_save_name=user_save_name, target=extract_method, progress_callback=progress_update
            )
            handle_task_completion(user, request_id, response)
            # Stories metadata and history
            save_metadata(response, data)
            update_history(response.get("file"), response.get("media"), CONTENT_SEPARATOR_FOLDER_NAME, user)
        else:
            handle_invalid_content(user, request_id)

    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_highlights(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_HIGHLIGHTS_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_HIGHLIGHTS_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        segment_length = int(module_parameters.get("segmentLength"))
        duration_limit = int(module_parameters.get("durationLimit"))
        prompt = module_parameters.get("highlightDescription")
        mode = "sound" if module_parameters.get("isSoundAnalysis") else "visual"

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        # valid data
        if check_tmp_file_uploaded(source_file_path) and Highlights.is_valid_video(source_file_path):
            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))
            model_cache.clear()  # Unload CPU task models

            response = Highlights.synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, segment_length=segment_length, mode=mode,
                highlight_description=prompt, user_save_name=user_save_name, duration_limit=duration_limit, progress_callback=progress_update
            )
            handle_task_completion(user, request_id, response)
            # Stories metadata and history
            save_metadata(response, data)
            update_history(response.get("file"), response.get("media"), CONTENT_HIGHLIGHTS_FOLDER_NAME, user)
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


def synthesize_cut(user, request_id) -> None:
    def progress_update(progress, progress_message):
        handle_progress_update(user, request_id, progress, progress_message)

    # Check ffmpeg
    if not is_ffmpeg(user):
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_HIGHLIGHTS_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_HIGHLIGHTS_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        kept_segments = list(module_parameters.get("segments", []))

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_format = source_file_parameters.get("formatFile")
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        # valid data
        if check_tmp_file_uploaded(source_file_path) and (Highlights.is_valid_video(source_file_path) or Highlights.is_valid_audio(source_file_path)):
            response = Highlights.smart_cut(folder_name=folder_name, user_save_name=user_save_name, source_file=source_file_path, user_name=user, kept_segments=kept_segments, progress_callback=progress_update)
            handle_task_completion(user, request_id, response)
            # Stories metadata and history
            save_metadata(response, data)
            update_history(response.get("file"), response.get("media"), CONTENT_HIGHLIGHTS_FOLDER_NAME, user)
    except Exception as err:
        handle_exception(user, request_id, err)
    finally:
        clear_cache()
        setattr(Settings, 'HISTORY_POLICY', False)


"""GPU TASKS"""
"""SYSTEM LOGICAL WITH LIMITS"""


@app.route('/get-resolution/<module_id>', methods=["GET"])
@cross_origin()
def get_resolution(module_id):
    max_content_size = 1920  # max size more than will be resize down
    if module_id == CONTENT_REMOVE_BACKGROUND_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_REMOVE_BACKGROUND
    if module_id == CONTENT_REMOVE_OBJECT_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_REMOVE_OBJECT
    if module_id == CONTENT_FACE_SWAP_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_FACE_SWAP
    if module_id == CONTENT_ENHANCEMENT_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_ENHANCEMENT
    if module_id == CONTENT_LIP_SYNC_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_LIP_SYNC
    if module_id == CONTENT_LIVE_PORTRAIT_FOLDER_NAME:
        max_content_size = Settings.MAX_RESOLUTION_LIVE_PORTRAIT
    if module_id == CONTENT_RESTYLE_FOLDER_NAME:
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if use_cpu:
            max_content_size = 0
        else:
            max_content_size = Settings.get_max_resolution_diffuser()
    if module_id == CONTENT_GENERATION_FOLDER_NAME:
        use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
        if use_cpu:
            max_content_size = 0
        else:
            max_content_size = 1080  # limit for method of generation
    return jsonify({"status": 200, "max_content_size": max_content_size})


def background_task_cpu():
    max_queue_cpu_task = Settings.CPU_MAX_QUEUE_ALL_USER  # max number task
    num_queue_one_user = Settings.CPU_MAX_QUEUE_ONE_USER  # max number task for one user
    cancel_delay_time = datetime.now() - timedelta(minutes=Settings.CPU_QUEUE_CANCEL_TIME_MIN)
    # Filter old task and calculate number busy task
    tasks = []
    current_queue_cpu = app.config["QUEUE_CPU"].copy()
    for user in current_queue_cpu.keys():
        user_tasks = []
        # Create a copy of the keys before iterating over them
        request_ids = list(current_queue_cpu[user].keys())
        for request_id in request_ids:
            busy = current_queue_cpu[user][request_id].get("busy", False)  # run only with None
            date = current_queue_cpu[user][request_id].get("date")
            method = current_queue_cpu[user][request_id].get("method", None)
            if not busy and date is not None and date < cancel_delay_time:
                app.config["QUEUE_CPU"][user].pop(request_id)
                continue
            if busy:
                max_queue_cpu_task -= 1
            elif busy is None:
                user_tasks.append((date, user, request_id, method))
        else:
            tasks.extend(user_tasks[:num_queue_one_user])

    sorted_tasks = sorted(tasks, key=lambda x: x[0])

    if max_queue_cpu_task > 0:
        for task in sorted_tasks[:max_queue_cpu_task]:
            _, available_ram_gb, busy_ram_gb = get_ram_gb()

            print(f"RAM for task CPU before {available_ram_gb} and busy {busy_ram_gb}.")

            date, user, request_id, method = task

            if app.config["QUEUE_CPU"][user].get(request_id) is None:
                continue

            if method in [CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_REMOVE_OBJECT_FOLDER_NAME, CONTENT_RESTYLE_FOLDER_NAME]:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_sam_embedding(user, request_id)
            elif method == CONTENT_ANALYSIS_NAME:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_analysis(user, request_id)
            elif method == CONTENT_SCENEDETECT_NAME:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_scenedetect(user, request_id)
            elif method == CONTENT_IMG2IMG_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_inpaint(user, request_id)
            elif method == CONTENT_IMAGE_COMPOSITION_NAME:
                if available_ram_gb > Settings.MIN_VRAM_IMAGE_COMPOSITION_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_image_composition(user, request_id)
            elif method == CONTENT_TXT2IMG_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_txt2img(user, request_id)
            elif method == CONTENT_OUTPAINT_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_outpaint(user, request_id)
            elif method == CONTENT_RETARGET_PORTRAIT_NAME:
                if available_ram_gb > Settings.MIN_RAM_LIVE_PORTRAIT_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_retarget_portrait(user, request_id)
            elif method == CONTENT_HIGHLIGHTS_FOLDER_NAME:
                if available_ram_gb > Settings.MIN_RAM_HIGHLIGHTS_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_chat(user, request_id)
            else:
                app.config["QUEUE_CPU"][user].pop(request_id)  # Remove task if not found method and enough RAM

            # Set delay to load models in RAM or VRAM and get real memory
            sleep(Settings.CPU_DELAY_TIME_SEC)


def background_task_module():
    """
    Can be on GPU or CPU
    """
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    max_queue_module_task = Settings.MODULE_MAX_QUEUE_ALL_USER
    num_queue_one_user = Settings.MODULE_MAX_QUEUE_ONE_USER  # max number task for one user
    cancel_delay_time = datetime.now() - timedelta(minutes=Settings.MODULE_QUEUE_CANCEL_TIME_MIN)
    # Filter old task and calculate number busy task
    tasks = []
    current_queue_module = app.config["QUEUE_MODULE"].copy()

    for user in current_queue_module.keys():
        user_tasks = []
        # Create a copy of the keys before iterating over them
        request_ids = list(current_queue_module[user].keys())
        for request_id in request_ids:
            busy = current_queue_module[user][request_id].get("busy", False)  # run only with None
            date = current_queue_module[user][request_id].get("date")
            sub_method = current_queue_module[user][request_id].get("sub_method", None)
            if not busy and date is not None and date < cancel_delay_time:
                app.config["QUEUE_MODULE"][user].pop(request_id)
                continue
            if busy:
                max_queue_module_task -= 1
            elif busy is None:
                user_tasks.append((date, user, request_id, sub_method))
        else:
            tasks.extend(user_tasks[:num_queue_one_user])

    sorted_tasks = sorted(tasks, key=lambda x: x[0])

    # Check min RAM and VRAM from settings if max queue module task can be work a few task else run without check
    if max_queue_module_task > 0:
        for task in sorted_tasks[:max_queue_module_task]:
            _, available_ram_gb, busy_ram_gb = get_ram_gb()
            _, available_vram_gb, busy_vram_gb = get_vram_gb()

            print(f"RAM for task module before {available_ram_gb} and busy {busy_ram_gb}.")
            print(f"VRAM for task module before {available_vram_gb} and busy {busy_vram_gb}.")

            date, user, request_id, sub_method = task

            if app.config["QUEUE_MODULE"][user].get(request_id) is None:
                continue

            if sub_method == CONTENT_REMOVE_BACKGROUND_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_REMOVE_BACKGROUND_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_background(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_VRAM_REMOVE_BACKGROUND_GB
                    if (available_ram_gb > Settings.MIN_RAM_REMOVE_BACKGROUND_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_background(user, request_id)
            elif sub_method == CONTENT_REMOVE_OBJECT_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_REMOVE_OBJECT_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_object(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_VRAM_REMOVE_OBJECT_GB
                    if (available_ram_gb > Settings.MIN_RAM_REMOVE_OBJECT_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_object(user, request_id)
            elif sub_method == CONTENT_FACE_SWAP_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_FACE_SWAP_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_face_swap(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_VRAM_FACE_SWAP_GB
                    if (available_ram_gb > Settings.MIN_RAM_FACE_SWAP_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_face_swap(user, request_id)
            elif sub_method == CONTENT_ENHANCEMENT_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_ENHANCEMENT_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_enhancement(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_RAM_ENHANCEMENT_GB
                    if (available_ram_gb > Settings.MIN_RAM_ENHANCEMENT_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_enhancement(user, request_id)
            elif sub_method == CONTENT_LIP_SYNC_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_LIP_SYNC_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_lip_sync(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_VRAM_LIP_SYNC_GB
                    if (available_ram_gb > Settings.MIN_RAM_LIP_SYNC_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_lip_sync(user, request_id)
            elif sub_method == CONTENT_SEPARATOR_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_SEPARATOR_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_separator(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = Settings.MIN_VRAM_SEPARATOR_GB
                    if (available_ram_gb > Settings.MIN_RAM_SEPARATOR_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_separator(user, request_id)
            elif sub_method == CONTENT_HIGHLIGHTS_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_HIGHLIGHTS_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_highlights(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    if (available_ram_gb > Settings.MIN_RAM_HIGHLIGHTS_GB) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_highlights(user, request_id)
            elif sub_method == CONTENT_SMART_CUT_NAME:
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                synthesize_cut(user, request_id)
            else:
                app.config["QUEUE_MODULE"][user].pop(request_id)  # Remove task if not found method

            # Set delay to load models in RAM or VRAM and get real memory
            sleep(Settings.MODULE_DELAY_TIME_SEC)


def background_clear_memory():
    clear_delay_time = datetime.now() - timedelta(minutes=3)  # minute

    current_queue_cpu = app.config["QUEUE_CPU"].copy()
    for user in current_queue_cpu.keys():
        # Create a copy of the keys before iterating over them
        request_ids = list(current_queue_cpu[user].keys())
        for request_id in request_ids:
            completed_date = current_queue_cpu[user][request_id].get("completed_date")
            if completed_date is not None and completed_date < clear_delay_time:
                app.config["QUEUE_CPU"][user].pop(request_id)

    current_queue_module = app.config["QUEUE_MODULE"].copy()
    for user in current_queue_module.keys():
        # Create a copy of the keys before iterating over them
        request_ids = list(current_queue_module[user].keys())
        for request_id in request_ids:
            completed_date = current_queue_module[user][request_id].get("completed_date")
            if completed_date is not None and completed_date < clear_delay_time:
                app.config["QUEUE_MODULE"][user].pop(request_id)

    # clear_cache()  # clear cache


def background_clear_tmp():
    # Check if any CPU tasks are pending in the queue
    if any(cpu_task for cpu_task in app.config["QUEUE_CPU"].values() if cpu_task):
        return

    # Check if any module tasks are pending in the queue
    if any(module_task for module_task in app.config["QUEUE_MODULE"].values() if module_task):
        return

    # Clear the temporary folder (TMP_FOLDER) if it's not empty
    if os.path.exists(TMP_FOLDER):
        # Iterate over items in TMP_FOLDER
        for item in os.listdir(TMP_FOLDER):
            item_path = os.path.join(TMP_FOLDER, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Recursively remove directory
            except Exception as e:
                print(f"Failed to remove {item_path}: {e}")


"""FEATURE MODELS"""

@app.route('/get-access', methods=["GET"])
@cross_origin()
def get_access():
    user_id = user_valid()
    full_access = True if user_id == Settings.ADMIN_ID else False  # user or admin access
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    return {"status": 200, "full_access": full_access, "use_cpu": use_cpu, "session": getattr(Settings, "SESSION", None)}


@app.route("/synthesize-process/", methods=["GET"])
@cross_origin()
def get_synthesize_status():
    user_id = user_valid()
    if user_id is None:
        return {"status": 404, "message": "Oops! System couldn't identify your IP address.\nDon't worry, click here to download the log!"}

    user_status = app.config['SYNTHESIZE_STATUS'].get(user_id, None)
    if user_status is None:
        user_status = {"status": 200, "message": ""}
        app.config['SYNTHESIZE_STATUS'][user_id] = user_status
    return user_status


@app.route("/system-resources", methods=["GET"])
@cross_origin()
def get_system_resources_status():
    total_vram_gb, available_vram_gb, _ = get_vram_gb()
    total_ram_gb, available_ram_gb, _ = get_ram_gb()
    return jsonify({"status": 200, "models_drive": get_folder_size(ALL_MODEL_FOLDER) / 1024, "content_drive": get_folder_size(CONTENT_FOLDER) / 1024, "total_vram": total_vram_gb, "available_vram": available_vram_gb, "total_ram": total_ram_gb, "available_ram": available_ram_gb})


@app.route('/set-init-config', methods=["POST"])
@cross_origin()
def set_init_config():
    data = request.get_json()

    # Disable tour for simple user
    if data.get("TOUR") is not None:
        session['TOUR'] = data['TOUR']
        # app.config['SWAGGER']['tour'] = data['TOUR']  # Tour for API
    if data.get("LANGUAGE_INTERFACE_CODE") is not None:
        session['LANGUAGE_INTERFACE_CODE'] = data['LANGUAGE_INTERFACE_CODE']

    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}

    # Remove keys with None values
    filtered_data = {key: value for key, value in data.items() if value is not None}
    # Set settings
    Settings.config_manager.update_config(filtered_data)
    for k, v in filtered_data.items():
        setattr(Settings, k, v)
    return {"status": 200}


@app.route('/get-init-config', methods=["GET"])
@cross_origin()
def get_init_config():
    all_parameters = Settings.config_manager.get_all_parameters().copy()
    all_parameters.pop("SECRET_KEY", None)
    all_parameters.pop("ADMIN_ID", None)
    return {"status": 200, "all_parameters": all_parameters}


@app.route('/set-public-share', methods=["POST"])
@cross_origin()
def set_public_share():
    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}
    return jsonify({"status": 200, "is_public": False})


@app.route('/get-public-share', methods=["GET"])
@cross_origin()
def get_public_share():
    share_link = f"http://0.0.0.0:{app.config['PORT']}"
    return {"status": 200, "share_link": share_link, "share_bool": False}


@app.route('/set-port', methods=["POST"])
@cross_origin()
def set_port():
    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}
    data = request.get_json()
    port = data.get('port', Settings.STATIC_PORT)
    if port and isinstance(port, int):
        Settings.config_manager.update_config({"STATIC_PORT": port})  # update config file
        Settings.STATIC_PORT = port
        lprint(f"New port {port} is set.", user=user_id, level="info")
        return jsonify({"status": 200, "port": port})
    return jsonify({"status": 300, "message": ""})


@app.route('/get-port', methods=["GET"])
@cross_origin()
def get_port():
    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}
    return {"status": 200, "port": Settings.STATIC_PORT, "run_port": app.config['PORT']}


@app.route('/set-home', methods=["POST"])
@cross_origin()
def set_home():
    import subprocess

    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}

    data = request.get_json()
    # For portable version not change os.path.expanduser("~")
    wunjo_home = os.environ.get('WUNJO_HOME', os.path.expanduser("~"))
    wunjo_cache = os.path.join(wunjo_home, '.cache', 'wunjo')
    new_home = data.get('home', wunjo_home)
    new_cache = os.path.join(new_home, '.cache', 'wunjo')

    if not all(ord(char) < 128 for char in new_home):
        return {"status": 404, "message": f"Folder {new_home} must contain only Latin characters."}

    if not os.path.exists(new_home):
        return {"status": 404, "message": f"Folder {new_home} not found."}
    elif not os.path.isdir(new_home):
        return {"status": 400, "message": f"{new_home} is not a directory."}

    if not os.access(new_home, os.W_OK):
        return {"status": 404, "message": f"Not found permission for {new_home}."}

    if sys.platform == 'win32':
        try:
            subprocess.run(["setx", "WUNJO_HOME", new_home], check=True)
            print(f"Environment variable WUNJO_HOME set to '{new_home}' persistently on Windows.")
            move_folder(wunjo_cache, new_cache)
            return {"status": 200, "message": "Folder changed and files moved. The software will shut down automatically after 5 seconds for the changes to take effect."}
        except subprocess.CalledProcessError:
            return {"status": 404, "message": "Failed to set persistent environment variable on Windows."}
    elif sys.platform in ['darwin', 'linux']:
        shell_profile = os.path.expanduser("~/.bashrc") if os.path.exists(os.path.expanduser("~/.bashrc")) else os.path.expanduser("~/.zshrc")
        export_line = f"export WUNJO_HOME={new_home}\n"

        with open(shell_profile, 'r') as file:
            lines = file.readlines()

        with open(shell_profile, 'w') as file:
            for line in lines:
                if not line.startswith("export WUNJO_HOME="):
                    file.write(line)
            file.write(export_line)

        print(f"Environment variable WUNJO_HOME set to '{new_home}' persistently.")
        move_folder(wunjo_cache, new_cache)
        return {"status": 200, "message": "Folder changed and files moved. The software will shut down automatically after 5 seconds for the changes to take effect."}

    return {"status": 404, "message": "Failed to set environment variable."}



@app.route('/get-processor', methods=["GET"])
@cross_origin()
def get_processor():
    if torch.cuda.is_available():
        return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": True}
    return {"current_processor": os.environ.get('WUNJO_TORCH_DEVICE', "cpu"), "upgrade_gpu": False}


@app.route('/set-processor', methods=["POST"])
@cross_origin()
def set_processor():
    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}

    if getattr(Settings, 'SESSION', None) is None:
        setattr(Settings, 'SESSION', True)

    current_processor = os.environ.get('WUNJO_TORCH_DEVICE', "cpu")
    current_users_status = app.config['SYNTHESIZE_STATUS'].copy()

    # 200, 300, 404 - processed finished, 100 is busy
    if any(u.get("status") not in [200, 300, 404] for u in current_users_status.values()):
        lprint(msg="At the moment, the model is running on some processor. You cannot change the processor during processing. Wait for the process to complete.", user=user_id, level="info")
        return {"current_processor": current_processor}
    else:
        if not torch.cuda.is_available():
            lprint(msg="No GPU driver was found on your computer. Working on the GPU will speed up the generation of content several times.", level="warn", user=user_id)
            lprint(msg="Visit the documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki to learn how to install drivers for your computer.", level="warn", user=user_id)
            return {"current_processor": 'none'}  # will be message how install cuda or hust hidden changer
        if current_processor == "cpu":
            lprint(msg="You turn on GPU", user=user_id)
            processor('cuda')
            return {"current_processor": 'cuda'}
        else:
            lprint(msg="You turn on CPU", user=user_id)
            processor('cpu')
            return {"current_processor": 'cpu'}


def update_history(paths, urls, method, user):
    paths_list = paths if isinstance(paths, list) else [paths]
    urls_list = urls if isinstance(urls, list) else [urls]
    max_len = max(len(paths_list), len(urls_list))
    paths_list += [None] * (max_len - len(paths_list))
    urls_list += [None] * (max_len - len(urls_list))
    [Settings.history_manager.update_history({"date": datetime.now().isoformat(), "path": path, "url": url, "method": method, "flag": False, "user": user}) for path, url in list(zip(paths_list, urls_list))]


@app.route('/set-history', methods=["GET"])
@cross_origin()
def set_history():
    setattr(Settings, 'HISTORY_POLICY', True)
    return {"status": 200}


@app.route('/get-history', methods=["POST"])
@cross_origin()
def get_history():
    user_id = user_valid()
    data = request.get_json()
    if data.get("is_existing"):
        method = data.get("method")
        return {"status": 200, "history": Settings.history_manager.existing_history(method), "user_id": user_id}
    return {"status": 200, "history": Settings.history_manager.history_data, "user_id": user_id}


@app.route('/session', methods=["POST"])
@cross_origin()
def set_session():
    return {"status": 200}


def processor(val):
    os.environ['WUNJO_TORCH_DEVICE'] = val if 'cuda' == val and torch.cuda.is_available() else 'cpu'


if not app.config['DEBUG']:
    from time import time
    from io import StringIO

    lprint(f"http://127.0.0.1:{app.config['PORT']}")
    lprint(f"http://127.0.0.1:{app.config['PORT']}/console-log")

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


    @app.route('/console-log', methods=['GET'])
    @swag_from(os.path.join(os.path.dirname(__file__), 'api', 'console_log.yaml'), methods=['GET'], validation=False)
    def console_log():
        max_len_logs = 1000

        replace_phrases = [
            "* Debug mode: off", "* Serving Flask app 'wunjo.app'",
            "WARNING:waitress.queue:Task queue depth is 1", "WARNING:waitress.queue:Task queue depth is 2",
            "WARNING:waitress.queue:Task queue depth is 3", "WARNING:waitress.queue:Task queue depth is 4"
        ]

        combined_logs = console_stdout.logs + console_stderr.logs
        combined_logs.sort(key=lambda x: x[0])  # Sort by timestamp

        # Escape sequences to remove
        escape_sequence = '\u001b'

        # Get user id for log
        user_id = user_valid()
        if user_id is None:
            return jsonify([])

        # Covert bytes log
        combined_logs = [(log[0], log[1].decode('utf-8')) if isinstance(log[1], bytes) else log for log in combined_logs]

        # Filter out unwanted phrases and extract log messages and show user only user log if not admin
        if user_id != Settings.ADMIN_ID:
            user_folder_name = user_id.replace(".", "_")
            filtered_logs = [
                log[1] for log in combined_logs
                if (escape_sequence not in log[1]) and not any(phrase in log[1] for phrase in replace_phrases) and (user_folder_name in log[1])
            ]
        else:
            filtered_logs = [
                log[1] for log in combined_logs
                if (escape_sequence not in log[1]) and not any(phrase in log[1] for phrase in replace_phrases)
            ]

        # Send the last max_len_logs lines to the frontend
        return jsonify(filtered_logs[-max_len_logs:])
else:
    @app.route('/console-log', methods=['GET'])
    @swag_from(os.path.join(os.path.dirname(__file__), 'api', 'console_log.yaml'), methods=['GET'], validation=False)
    def console_log():
        logs = ["Debug mode. Console is turned off"]
        return jsonify(logs)


@app.route('/console-log-print', methods=['POST'])
def console_log_print():
    resp = request.get_json()
    msg = resp.get("print", None)
    if msg:
        user_id = user_valid()
        lprint(msg=msg, is_server=False, user=user_id if user_id else None)  # show message in backend console
    return jsonify({"status": 200})


class InvalidVoice(Exception):
    pass


def get_range_header(request):
    range_header = request.headers.get('Range', None)
    if range_header is not None:
        # Parse the range header and return start and end
        range_match = re.search(r'(\d+)-(\d*)', range_header)
        start = int(range_match.group(1))
        end = range_match.group(2)
        end = int(end) if end else None
        return start, end
    return None, None


@app.route("/media/<path:filename>", methods=["GET"])
@cross_origin()
def stream_media(filename):
    file_path = os.path.normpath(os.path.join(MEDIA_FOLDER, filename))

    if not os.path.exists(file_path):  # if file not found, send empty
        mimetype = 'application/octet-stream'
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"',
        }
        return Response(b"", headers=headers, mimetype=mimetype, status=200)

    # Handle range requests
    range_header = request.headers.get('Range', None)
    if range_header:
        try:
            start, end = get_range_header(request)
            return partial_response(file_path, start, end)
        except Exception as e:
            return Response("Error while streaming media", status=500)

    # If no range header, stream the file using a generator
    def generate():
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                yield chunk

    # Determine content type for full file response
    if filename.endswith(('.mp4', '.mov', '.avi')):
        mimetype = 'video/mp4'
    elif filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        mimetype = 'image/jpeg'
    elif filename.endswith(('.mp3', '.wav', '.ogg')):
        mimetype = 'audio/mpeg'
    else:
        mimetype = 'application/octet-stream'  # Default type

    return Response(generate(), mimetype=mimetype)


def partial_response(file_path, start, end=None):
    file_size = os.path.getsize(file_path)
    if end is None:
        end = file_size - 1

    length = end - start + 1
    with open(file_path, 'rb') as f:
        f.seek(start)
        data = f.read(length)

    # Determine content type
    if file_path.endswith(('.mp4', '.mov', '.avi')):
        mimetype = 'video/mp4'
    elif file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        mimetype = 'image/jpeg'
    elif file_path.endswith(('.mp3', '.wav', '.ogg')):
        mimetype = 'audio/mpeg'
    else:
        mimetype = 'application/octet-stream'  # Default type

    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': mimetype,
    }

    return Response(data, headers=headers, status=206)


"""TRANSLATION"""
@app.route('/translations.json')
def translations():
    translations = {m: _b(m) for m in msgids}
    return jsonify(translations)
"""TRANSLATION"""


def main():
    if not app.config['DEBUG']:
        background_clear_tmp()

    # Scheduler
    scheduler.add_job(background_task_cpu, 'interval', seconds=10, max_instances=Settings.CPU_MAX_QUEUE_ALL_USER, coalesce=False, id="background_task_cpu")
    scheduler.add_job(background_task_module, 'interval', seconds=30, max_instances=Settings.MODULE_MAX_QUEUE_ALL_USER, coalesce=False, id="background_task_module")
    scheduler.add_job(background_clear_memory, 'interval', seconds=30, max_instances=1, coalesce=False, id="background_clear_memory")
    scheduler.start()

    def server(**server_kwargs):
        server_app = server_kwargs.pop("app", None)
        server_kwargs.pop("debug", None)
        server_app.run(host='0.0.0.0', **server_kwargs)

    # Init app by mode, OS and is display or not
    port = app.config['PORT']  # get set port
    if not app.config['DEBUG'] and sys.platform != 'darwin' and ((sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32'):
        FlaskUI(server=server, browser_path=WEBGUI_PATH, server_kwargs={"app": app, "port": port}).run()
    else:
        lprint(f"http://127.0.0.1:{port}")
        app.run(host="0.0.0.0", port=port)
