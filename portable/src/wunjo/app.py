import os
import gc
import sys
import uuid
import json
import torch
import socket
import shutil
import base64
import requests

from time import sleep
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # remove msg
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_from_directory, url_for, jsonify, redirect, session
from flask_cors import CORS, cross_origin
from flaskwebgui import FlaskUI, kill_port, close_application
from ipaddress import ip_address
from apscheduler.schedulers.background import BackgroundScheduler

from deepfake.inference import FaceSwap, Enhancement, SAMGetSegment, RemoveBackground, RemoveObject, LipSync, Analyser
from speech.inference import Separator

try:
    from generation.inference import VideoGeneration, ImageGeneration
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from backend.folders import (
    MEDIA_FOLDER, TMP_FOLDER, SETTING_FOLDER, CONTENT_FOLDER, CONTENT_REMOVE_OBJECT_FOLDER_NAME,
    CONTENT_ENHANCEMENT_FOLDER_NAME, CONTENT_FACE_SWAP_FOLDER_NAME, CONTENT_AVATARS_FOLDER_NAME,
    CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_LIP_SYNC_FOLDER_NAME, ALL_MODEL_FOLDER, CONTENT_ANALYSIS_NAME,
    CONTENT_SEPARATOR_FOLDER_NAME, CONTENT_GENERATION_FOLDER_NAME, CONTENT_IMG2IMG_NAME, CONTENT_TXT2IMG_NAME, CONTENT_OUTPAINT_NAME
)
from backend.config import Settings, WEBGUI_PATH
from backend.general_utils import (
    get_version_app, current_time, is_ffmpeg_installed, get_folder_size, lprint, get_vram_gb, get_ram_gb,
    format_dir_time, check_tmp_file_uploaded, generate_file_tree, decrypted_val, decrypted_key
)

import logging

scheduler = BackgroundScheduler()
app = Flask(__name__)
cors = CORS(app)
# TODO https://briefcase.readthedocs.io/en/stable/reference/configuration.html


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
app.config['LOGIN_PAGE'] = os.environ.get('DEBUG', 'False') == 'True'

logging.getLogger('werkzeug').disabled = True

version_app = get_version_app()

# CODE
## 100 - Busy
## 200 - OK
## 300 - Warn
## 404 - Error


def clear_cache():
    torch.cuda.empty_cache()
    # TODO torch.cuda.ipc_collect()
    gc.collect()


"""FRONTEND METHODS"""


@app.after_request
def add_cors_headers(response):
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    return response


# Custom 404 error handler
@app.errorhandler(404)
def page_not_found(error):
    return render_content_page(None, '404.html')


@app.route("/login", methods=["GET", "POST"])
@cross_origin()
def login_page():
    if request.method == "POST":
        user_id = user_valid()
        app.config['LOGIN_PAGE'] = True
        if user_id is None:
            return render_template("login.html")
        return redirect(url_for('index_page'))
    return render_template("login.html", host=Settings.FRONTEND_HOST, token=Settings.SECRET_KEY)


@app.route("/logout")
@cross_origin()
def logout():
    return redirect(url_for('login_page'))


@app.route("/", methods=["GET"])
@cross_origin()
def index_page():
    return render_content_page(None, "index.html")


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
    if user_id is None or not app.config['LOGIN_PAGE']:
        return redirect(url_for('login_page'))
    user_folder_name = user_id.replace(".", "_")
    user_folder = os.path.join(CONTENT_FOLDER, user_folder_name)
    user_content_folder = os.path.join(user_folder, content_folder_name) if content_folder_name else user_folder
    if not os.path.exists(user_content_folder):
        os.makedirs(user_content_folder)
    user_content_file_tree = generate_file_tree(user_content_folder)

    if getattr(Settings, "SESSION", None) is None:
        processor("cuda")

    return render_template(
        template_name,
        host=Settings.FRONTEND_HOST,
        version=Settings.VERSION,
        changelog=json.dumps(version_app),
        user_file_tree=user_content_file_tree,
        user_content_id=user_folder_name,
        folder_name=content_folder_name,
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
    if request.method == "POST":
        if "logout" in request.form:
            session.pop(user_id, None)
            return redirect(url_for('logout'))
        elif "exit" in request.form:
            if user_id != Settings.ADMIN_ID:  # this will be id admin
                session.pop(user_id, None)
                return redirect(url_for('logout'))
            else:
                if not app.config['DEBUG'] and sys.platform != 'darwin' and ((sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32'):
                    close_application()
                else:
                    port = app.config['PORT']
                    kill_port(port)

    public_link = f"http://0.0.0.0:{app.config['PORT']}"

    return render_content_page(None, "profile.html", user_id=user_id, public_link=public_link, email="support@wunjo.online")


@app.route("/settings", methods=["GET"])
@cross_origin()
def settings_page():
    models_dict = {
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
    model_sam_media_path = RemoveBackground.is_model_sam()
    max_duration_sec = Settings.MAX_DURATION_SEC_REMOVE_BACKGROUND
    max_files_size = Settings.MAX_FILE_SIZE
    return render_content_page(CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, "remove_background.html", model_sam=model_sam_media_path, max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/remove-object", methods=["GET"])
@cross_origin()
def remove_object_page():
    model_sam_media_path = RemoveObject.is_model_sam()
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
    return render_content_page(CONTENT_LIP_SYNC_FOLDER_NAME, "lip_sync.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size, analysis_method=CONTENT_ANALYSIS_NAME)


@app.route("/separator", methods=["GET"])
@cross_origin()
def separator_page():
    max_duration_sec = Settings.MAX_DURATION_SEC_SEPARATOR
    max_files_size = 40 * 1024 * 1024  # 40 Mb
    return render_content_page(CONTENT_SEPARATOR_FOLDER_NAME, "separator.html", max_duration_sec=max_duration_sec, max_files_size=max_files_size)


@app.route("/generation", methods=["GET"])
@cross_origin()
def generation_page():
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
    try:  # if version gpu with diffusers
        sd_models = VideoGeneration.get_sd_models()
    except Exception as err:
        lprint(msg=err, user=user_valid(), level="warn")
        sd_models = []
    return render_content_page(CONTENT_GENERATION_FOLDER_NAME, "generation.html", max_files_size=max_files_size, sd_models=sd_models, inpaint_method=CONTENT_IMG2IMG_NAME, txt2img_method=CONTENT_TXT2IMG_NAME, outpaint_method=CONTENT_OUTPAINT_NAME)


"""FRONTEND METHODS"""

"""API LOGICAL"""

"""WORK WITH CONTENT FOLDER FILES"""


@app.route('/content-rename', methods=['POST'])
@cross_origin()
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
def generate_avatar():
    # Generate a unique ID for the request
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    data = request.get_json()
    gender = data.get("gender", "all")
    age = data.get("age", "all")
    ethnic = data.get("ethnic", "all")
    url = f"{Settings.FRONTEND_HOST}/generate-avatar"

    user_folder_name = user_id.replace(".", "_")
    user_avatar_folder = os.path.join(CONTENT_FOLDER, user_folder_name, CONTENT_AVATARS_FOLDER_NAME)

    if not os.path.exists(user_avatar_folder):
        os.makedirs(user_avatar_folder)

    try:
        # Send a POST request with the parameters
        image_response = requests.post(url, json={"gender": gender, "age": age, "ethnic": ethnic})

        # Check if the response contains image data
        if image_response.headers.get("Content-Type", "").startswith("image/jpeg"):
            # Save the image to the user's folder
            image_name = str(uuid.uuid4()) + '.jpg'
            image_file_path = os.path.join(user_avatar_folder, os.path.basename(image_name))

            # Write the image content to the file
            with open(image_file_path, "wb") as image_file:
                image_file.write(image_response.content)

            media_file_path = image_file_path.replace(MEDIA_FOLDER, "/media").replace("\\", "/")

            # Return the cropped image as bytes
            return jsonify({"status": 200, "media_file": media_file_path})
        else:
            # Return some default static image if the response does not contain image data
            return jsonify({"status": 300, "message": "Invalid image data received"})
    except Exception as err:
        return jsonify({"status": 400, "message": str(err)})


"""CPU TASKS"""

@app.route("/submit-cpu-task", methods=["POST"])
@cross_origin()
def submit_cpu_task():
    # Generate a unique ID for the request
    request_id = str(uuid.uuid4())
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    data = request.get_json()
    method = data.pop('method') if data.get('method') is not None else None
    filename = data.get('filename')
    # Keep datetime to clear old sessions
    queue_data = {user_id: {request_id: {"method": method, "data": data, "completed": False, "busy": None, "filename": filename, "response": {}, "date": datetime.now()}}}

    if method not in [CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_REMOVE_OBJECT_FOLDER_NAME, CONTENT_ANALYSIS_NAME, CONTENT_OUTPAINT_NAME, CONTENT_TXT2IMG_NAME, CONTENT_IMG2IMG_NAME]:
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
def query_cpu_task(request_id):
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    if app.config["QUEUE_CPU"].get(user_id) is not None:
        if app.config["QUEUE_CPU"][user_id].get(request_id) is not None:
            queue_task = app.config["QUEUE_CPU"][user_id][request_id]
            completed = queue_task.get("completed", False) if isinstance(queue_task, dict) else False
            if isinstance(queue_task, dict) and completed:
                response = queue_task.get("response")
                app.config["QUEUE_CPU"][user_id].pop(request_id)  # Clear
                return jsonify({"status": 200, "message": "Successfully!", "response": response })
            if completed == "cancel":
                # Canceled task
                app.config["QUEUE_CPU"][user_id].pop(request_id)  # Clear
                return jsonify({"status": 404, "message": "Task is canceled.", "response": None})
            return jsonify({"status": 100, "message": "Busy.", "response": None })
    return jsonify({"status": 300, "message": "Not found task.", "response": None})


def get_sam_embedding(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        filename = send_data.get("filename")
        file_path = os.path.join(TMP_FOLDER, filename)
        if check_tmp_file_uploaded(file_path) and SAMGetSegment.is_valid_image(file_path):
            image_data = SAMGetSegment.read(file_path)  # Read data from path
            lprint(msg="Start to get embedding by SAM.", user=user)
            response = SAMGetSegment.extract_embedding(image_data)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = response
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = True
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            SAMGetSegment.clear(file_path)  # Remove tpm file
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Get SAM {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": ""}
        else:
            lprint(msg=f"This is not image {request_id}.", level="warn", user=user)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = None
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Error with get SAM embedding on CPU but process busy still.", user=user, level="error")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with get SAM embedding on CPU."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = None
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg=f"Error with get SAM embedding on CPU but process busy still.", user=user, level="error")
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with get SAM embedding on CPU."}

    clear_cache()


def get_analysis(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    # Processing
    try:
        send_data = app.config["QUEUE_CPU"][user][request_id]
        filename = send_data.get("filename")
        file_path = os.path.join(TMP_FOLDER, filename)

        if check_tmp_file_uploaded(file_path) and Analyser.is_valid(file_path):
            lprint(msg="Start to get analysis content.", user=user)
            response = Analyser.analysis(user_name=user, source_file=file_path)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = response
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = True
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Get analysis {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": ""}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = None
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg="Error with get analysis because of content is not valid but process busy still.", user=user, level="error")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with get analysis because of content is not valid."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = None
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg="Error with get analysis on CPU but process busy still.", user=user, level="error")
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with get analysis on CPU."}

    clear_cache()


def get_inpaint(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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
            response = ImageGeneration.inpaint(user_name=user, source_file=file_path, sd_model_name=sd_model, params=diffusion_parameters, inpaint_file=inpaint_file_path, eta=eta)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = response
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = True
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Get inpaint {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": ""}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = None
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg="Error with inpaint because of content is not valid but process busy still.", user=user, level="error")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with inpaint because of content is not valid."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = None
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg="Error with inpaint on CPU or GPU but process busy still.", user=user, level="error")
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with inpaint on CPU or GPU."}

    clear_cache()


def get_txt2img(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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
        response = ImageGeneration.text_to_image(user_name=user, sd_model_name=sd_model, params=diffusion_parameters, width=width, height=height)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = response
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = True
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg=f"Get generate image {request_id} successfully finished but process busy still.", user=user)
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": ""}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = None
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg="Error with generate image on CPU or GPU but process busy still.", user=user, level="error")
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with generate image on CPU or GPU."}

    clear_cache()


def get_outpaint(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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
            response = ImageGeneration.outpaint(source_file=file_path, user_name=user, sd_model_name=sd_model, params=diffusion_parameters, width=width, height=height, blur_factor=blur_factor)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = response
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = True
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Get outpaint image {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": ""}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_CPU"][user].get(request_id):
                app.config["QUEUE_CPU"][user][request_id]["response"] = None
                app.config["QUEUE_CPU"][user][request_id]["busy"] = False
                app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
            # Get status of module task (not cpu task) to real right control status busy
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg="Error with outpaint because of content is not valid but process busy still.", user=user, level="error")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with inpaint because of content is not valid."}


    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_CPU"][user].get(request_id):
            app.config["QUEUE_CPU"][user][request_id]["response"] = None
            app.config["QUEUE_CPU"][user][request_id]["busy"] = False
            app.config["QUEUE_CPU"][user][request_id]["completed"] = "cancel"
            app.config["QUEUE_CPU"][user][request_id]["completed_date"] = datetime.now()
        # Get status of module task (not cpu task) to real right control status busy
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg="Error with outpaint image on CPU or GPU but process busy still.", user=user, level="error")
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Error with outpaint image on CPU or GPU."}

    clear_cache()

"""CPU TASKS"""
"""GPU TASKS"""

@app.route("/submit-module-task", methods=["POST"])
@cross_origin()
def submit_module_task():
    # Generate a unique ID for the request
    request_id = str(uuid.uuid4())
    user_id = user_valid()
    if user_id is None:
        lprint(msg="System couldn't identify user IP address.")
        return jsonify({"status": 404, "message": "System couldn't identify your IP address."})

    data = request.get_json()
    method = data.pop('method') if data.get('method') is not None else None
    # Keep datetime to clear old sessions
    queue_data = {user_id: {request_id: {"method": method, "data": data, "completed": False, "busy": None, "response": {}, "date": datetime.now()}}}

    module_methods = [CONTENT_REMOVE_BACKGROUND_FOLDER_NAME, CONTENT_REMOVE_OBJECT_FOLDER_NAME,
                      CONTENT_FACE_SWAP_FOLDER_NAME, CONTENT_ENHANCEMENT_FOLDER_NAME,
                      CONTENT_LIP_SYNC_FOLDER_NAME, CONTENT_SEPARATOR_FOLDER_NAME, CONTENT_GENERATION_FOLDER_NAME]

    if method not in module_methods:
        return jsonify({"status": 404, "message": "Method not found for module task."})

    # Message about community version
    current_queue_module = app.config["QUEUE_MODULE"].copy()
    for user in current_queue_module.keys():
        # Create a copy of the keys before iterating over them
        request_ids = list(current_queue_module[user].keys())
        for request_id in request_ids:
            busy = current_queue_module[user][request_id].get("busy", False)  # run only with None
            if busy:
                return jsonify({"status": 300, "message": "Community Edition version can run only one task at the same time."})

    for user in queue_data.keys():
        if app.config["QUEUE_MODULE"].get(user) is None:
            app.config["QUEUE_MODULE"][user] = {}
        for request_id in queue_data[user].keys():
            app.config["QUEUE_MODULE"][user][request_id] = queue_data[user][request_id]
            lprint(msg=f"Add in background module processing on {str(os.environ.get('WUNJO_TORCH_DEVICE', 'cpu')).upper()} {request_id}", user=user_id)

    return jsonify({"status": 200, "message": "Updated delay module task.", "id": request_id})


@app.route("/query-module-task/<method>", methods=["GET"])
@cross_origin()
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
            if method == user_method and completed:
                app.config["QUEUE_MODULE"][user_id].pop(request_id)  # Clear completed task
                return jsonify({"status": 201, "message": "Successfully!", "response": [{"request_id": request_id}]})
            if completed == "cancel":
                # Remove canceled task
                app.config["QUEUE_MODULE"][user_id].pop(request_id)  # Clear
            if method == user_method and not completed:
                created = user_queue_module[request_id].get("date")  # date created
                data = user_queue_module[request_id].get("data")  # parameters data
                module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
                user_save_name = module_parameters.get("nameFile")
                user_busy_task.append({"request_id": request_id, "name": user_save_name, "created": created})
        else:
            if len(user_busy_task) > 0:
                return jsonify({"status": 100, "message": "Busy.", "response": user_busy_task})
            return jsonify({"status": 200, "message": f"Not found module {method} tasks.", "response": None})
    return jsonify({"status": 200, "message": "Not found task for user.", "response": None})


def synthesize_remove_background(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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
        if check_tmp_file_uploaded(source_file_path) and RemoveBackground.is_valid(source_file_path):
            valid_point_list = RemoveBackground.is_valid_point_list(point_list, source_offset_width, source_offset_height)
            if len(valid_point_list.keys()) == 0:
                raise

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

            response = RemoveBackground.retries_synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, inverse_mask=inverse_mask,
                masks=valid_point_list, user_save_name=user_save_name,  segment_percentage=25, source_start=source_start_time,
                source_end=source_end_time, max_size=Settings.MAX_RESOLUTION_REMOVE_BACKGROUND, mask_color=background_color
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_REMOVE_BACKGROUND_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not image {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not image or video {request_id} but process busy still.", user=user, level="warn")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not image or video {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params remove background {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_remove_object(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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
        text_area = module_parameters.get("cropField", [])  # list of dict(x, y, width, height)

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_offset_width = source_file_parameters.get("offsetWidth")
        source_offset_height = source_file_parameters.get("offsetHeight")
        source_start_time = source_file_parameters.get("startTime")
        source_end_time = source_file_parameters.get("endTime")

        if source_start_time and source_end_time:  # set what time not more limits
            source_start_time = float(source_start_time)
            source_end_time = min(float(source_end_time), float(source_start_time) + Settings.MAX_DURATION_SEC_REMOVE_OBJECT)

        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))
        if check_tmp_file_uploaded(source_file_path) and RemoveObject.is_valid(source_file_path):
            valid_point_list = RemoveObject.is_valid_point_list(point_list, source_offset_width, source_offset_height)
            valid_text_area = RemoveObject.is_valid_text_area(text_area, source_offset_width, source_offset_height)
            if (len(valid_point_list.keys()) == 0 and not use_remove_text) and (len(valid_text_area.keys()) == 0 and use_remove_text):
                raise

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

            response = RemoveObject.retries_synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, use_improved_model=use_improved_model,
                masks=valid_point_list, user_save_name=user_save_name,  segment_percentage=25, source_start=source_start_time, inverse_mask=inverse_mask,
                source_end=source_end_time, max_size=Settings.MAX_RESOLUTION_REMOVE_OBJECT, text_area=valid_text_area, use_remove_text=use_remove_text
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_REMOVE_OBJECT_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not image or video {request_id} but process busy still.", user=user, level="warn")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not image or video {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params remove object {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_face_swap(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_FACE_SWAP_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_FACE_SWAP_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        face_discrimination_factor = float(module_parameters.get("faceDiscriminationFactor", 1))
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
        avatar_offset_width = int(avatar_file_parameters.get("offsetWidth"))
        avatar_offset_height = int(avatar_file_parameters.get("offsetHeight"))

        # valid data
        valid_source_area = FaceSwap.is_valid_source_area(changing_crop_face, source_offset_width, source_offset_height)
        valid_avatar_area = FaceSwap.is_valid_avatar_area(avatar_crop_face, avatar_offset_width, avatar_offset_height)
        resize_factor = FaceSwap.get_resize_factor(source_natural_width, source_natural_height, max_size=Settings.MAX_RESOLUTION_FACE_SWAP)
        is_valid_source = True if check_tmp_file_uploaded(source_file_path) and FaceSwap.is_valid(source_file_path) else False
        is_valid_avatar = True if check_tmp_file_uploaded(avatar_file_path) and FaceSwap.is_valid_avatar_image(avatar_file_path) else False

        if is_valid_source and is_valid_avatar:
            if len(valid_source_area) == 0 or len(valid_avatar_area) == 0:
                raise

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

            response = FaceSwap.retries_synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, avatar_file=avatar_file_path,
                user_save_name=user_save_name, source_crop_area=valid_source_area, avatar_crop_area=valid_avatar_area,
                source_start=source_start_time, source_end=source_end_time, is_gemini_faces=is_gemini_faces,
                replace_all_faces=replace_all_faces, face_discrimination_factor=face_discrimination_factor,
                resize_factor=resize_factor
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_FACE_SWAP_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not image or video {request_id} but process busy still.", user=user, level="warn")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not image or video {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params face swap {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_enhancement(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
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

        is_valid_source = True if check_tmp_file_uploaded(source_file_path) and Enhancement.is_valid(source_file_path) else False

        if is_valid_source:

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

            response = Enhancement.retries_synthesis(
                folder_name=folder_name, user_name=user, source_file=source_file_path,
                user_save_name=user_save_name, enhancer=enhancement_model
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200,"message": f"Successfully finished by task {request_id} in {CONTENT_ENHANCEMENT_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not image or video {request_id} but process busy still.", user=user, level="warn")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not image or video {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params face swap {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_lip_sync(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_LIP_SYNC_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_LIP_SYNC_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        face_discrimination_factor = float(module_parameters.get("faceDiscriminationFactor", 1))
        changing_crop_face = module_parameters.get("cropField", [])  # list of dict(x, y, width, height)
        overlay_audio = module_parameters.get("overlayAudio", False)

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
        valid_source_area = LipSync.is_valid_source_area(changing_crop_face, source_offset_width, source_offset_height)
        resize_factor = LipSync.get_resize_factor(source_natural_width, source_natural_height, max_size=Settings.MAX_RESOLUTION_FACE_SWAP)
        is_valid_source = True if check_tmp_file_uploaded(source_file_path) and LipSync.is_valid(source_file_path) else False
        is_valid_audio = True if check_tmp_file_uploaded(audio_file_path) and LipSync.is_valid_audio(audio_file_path) else False

        if is_valid_source and is_valid_audio:
            if len(valid_source_area) == 0:
                raise

            processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

            response = LipSync.retries_synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, audio_file=audio_file_path,
                user_save_name=user_save_name, source_crop_area=valid_source_area, source_start=source_start_time,
                source_end=source_end_time, audio_start=audio_start_time, audio_end=audio_end_time, overlay_audio=overlay_audio,
                face_discrimination_factor=face_discrimination_factor, resize_factor=resize_factor
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_LIP_SYNC_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not image/video and audio {request_id} but process busy still.", user=user, level="warn")
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not image/video and audio {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params lip sync {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_separator(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_SEPARATOR_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_SEPARATOR_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        extractMethod = module_parameters.get("extractMethod", "vocals")
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

            response = Separator.synthesis(
                folder_name=folder_name, user_name=user_folder, source_file=source_file_path, trim_silence=trim_silence,
                user_save_name=user_save_name, target=extractMethod
            )
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = response
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_SEPARATOR_FOLDER_NAME}"}
        else:
            lprint(msg=f"This is not right format {request_id}.", level="warn", user=user)
            if app.config["QUEUE_MODULE"][user].get(request_id):
                app.config["QUEUE_MODULE"][user][request_id]["response"] = None
                app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
                app.config["QUEUE_MODULE"][user][request_id]["completed"] = "cancel"
                app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
            is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
            if any(is_busy_list):
                lprint(msg=f"This is not audio {request_id} but process busy still.", user=user)
            else:
                app.config['SYNTHESIZE_STATUS'][user] = {"status": 300, "message": f"This is not audio {request_id}."}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params separator {request_id}."}

    # Clear cache
    clear_cache()


def synthesize_video_generation(user, request_id) -> None:
    # Check ffmpeg
    is_ffmpeg = is_ffmpeg_installed()
    if not is_ffmpeg:
        lprint(msg=f"Not found ffmpeg. Please download this.", user=user, level="error")
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": "Not found ffmpeg. Please download this."}
        return

    try:
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 100, "message": f"Busy by task {request_id} in {CONTENT_GENERATION_FOLDER_NAME}"}

        user_folder = str(user).replace(".", "_")
        folder_name = os.path.join(CONTENT_FOLDER, user_folder, CONTENT_GENERATION_FOLDER_NAME)
        send_data = app.config["QUEUE_MODULE"][user][request_id]
        data = send_data.get("data")

        module_parameters = data.get("moduleParameters") if isinstance(data, dict) else {}
        user_save_name = module_parameters.get("nameFile")  # can be none
        aspect_ratio = module_parameters.get("aspectRatio")
        fps = module_parameters.get("FPS")
        sd_model = module_parameters.get("sdModel")
        positive_prompt = module_parameters.get("positivePrompt")
        negative_prompt = module_parameters.get("negativePrompt")
        seed = module_parameters.get("seed")
        strength = module_parameters.get("strength")
        scale = module_parameters.get("scale")

        source_file_parameters = data.get("sourceFile") if isinstance(data, dict) else {}
        source_file_name = source_file_parameters.get("nameFile")
        source_file_path = os.path.join(TMP_FOLDER, str(source_file_name))

        processor(os.environ.get('WUNJO_TORCH_DEVICE', "cpu"))

        response = VideoGeneration.synthesis(
            folder_name=folder_name, user_name=user_folder, source_file=source_file_path, sd_model_name=sd_model,
            positive_prompt=positive_prompt, user_save_name=user_save_name, negative_prompt=negative_prompt,
            seed=seed, strength=strength, scale=scale, aspect_ratio=aspect_ratio, fps=fps
        )
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user][request_id]["response"] = response
            app.config["QUEUE_MODULE"][user][request_id]["busy"] = False
            app.config["QUEUE_MODULE"][user][request_id]["completed"] = True
            app.config["QUEUE_MODULE"][user][request_id]["completed_date"] = datetime.now()
        is_busy_list = [val.get("busy", False) for val in app.config["QUEUE_MODULE"].get(user, {}).values()]
        if any(is_busy_list):
            lprint(msg=f"Task {request_id} successfully finished but process busy still.", user=user)
        else:
            app.config['SYNTHESIZE_STATUS'][user] = {"status": 200, "message": f"Successfully finished by task {request_id} in {CONTENT_GENERATION_FOLDER_NAME}"}

    except Exception as err:
        lprint(msg=str(err), level="error", user=user)
        if app.config["QUEUE_MODULE"][user].get(request_id):
            app.config["QUEUE_MODULE"][user].pop(request_id)
        app.config['SYNTHESIZE_STATUS'][user] = {"status": 404, "message": f"Error with get params lip sync {request_id}."}

    # Clear cache
    clear_cache()

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

            if method == CONTENT_REMOVE_BACKGROUND_FOLDER_NAME:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_sam_embedding(user, request_id)
            elif method == CONTENT_REMOVE_OBJECT_FOLDER_NAME:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_sam_embedding(user, request_id)
            elif method == CONTENT_ANALYSIS_NAME:
                if available_ram_gb > Settings.MIN_RAM_CPU_TASK_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_analysis(user, request_id)
            elif method == CONTENT_IMG2IMG_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_inpaint(user, request_id)
            elif method == CONTENT_TXT2IMG_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_txt2img(user, request_id)
            elif method == CONTENT_OUTPAINT_NAME:
                if available_ram_gb > Settings.MIN_RAM_DIFFUSER_GB:
                    app.config["QUEUE_CPU"][user][request_id]["busy"] = True
                    get_outpaint(user, request_id)
            else:
                app.config["QUEUE_CPU"][user].pop(request_id)  # Remove task if not found method and enough RAM

            # Set delay to load models in RAM or VRAM and get real memory
            sleep(Settings.CPU_DELAY_TIME_SEC)


def background_task_module():
    """
    Can be on GPU or CPU
    """
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    max_queue_module_task = 1
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
            method = current_queue_module[user][request_id].get("method", None)
            if not busy and date is not None and date < cancel_delay_time:
                app.config["QUEUE_MODULE"][user].pop(request_id)
                continue
            if busy:
                max_queue_module_task -= 1
            elif busy is None:
                user_tasks.append((date, user, request_id, method))
        else:
            tasks.extend(user_tasks[:1])

    sorted_tasks = sorted(tasks, key=lambda x: x[0])

    # Check min RAM and VRAM from settings if max queue module task can be work a few task else run without check
    if max_queue_module_task > 0:
        for task in sorted_tasks[:1]:
            _, available_ram_gb, busy_ram_gb = get_ram_gb()
            _, available_vram_gb, busy_vram_gb = get_vram_gb()

            print(f"RAM for task module before {available_ram_gb} and busy {busy_ram_gb}.")
            print(f"VRAM for task module before {available_vram_gb} and busy {busy_vram_gb}.")

            date, user, request_id, method = task

            if app.config["QUEUE_MODULE"][user].get(request_id) is None:
                continue

            if method == CONTENT_REMOVE_BACKGROUND_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_REMOVE_BACKGROUND_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_background(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_VRAM_REMOVE_BACKGROUND_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_REMOVE_BACKGROUND_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_background(user, request_id)
            elif method == CONTENT_REMOVE_OBJECT_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_REMOVE_OBJECT_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_object(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_VRAM_REMOVE_OBJECT_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_REMOVE_OBJECT_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_remove_object(user, request_id)
            elif method == CONTENT_FACE_SWAP_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_FACE_SWAP_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_face_swap(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_VRAM_FACE_SWAP_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_FACE_SWAP_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_face_swap(user, request_id)
            elif method == CONTENT_ENHANCEMENT_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_ENHANCEMENT_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_enhancement(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_RAM_ENHANCEMENT_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_ENHANCEMENT_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_enhancement(user, request_id)
            elif method == CONTENT_LIP_SYNC_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_LIP_SYNC_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_lip_sync(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_VRAM_LIP_SYNC_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_LIP_SYNC_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_lip_sync(user, request_id)
            elif method == CONTENT_SEPARATOR_FOLDER_NAME:
                if use_cpu:  # if use cpu, then only ram using
                    if available_ram_gb > Settings.MIN_RAM_SEPARATOR_GB or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_separator(user, request_id)
                else:  # if use gpu, then vram using and less ram than on cpu
                    memory_threshold = max(Settings.MIN_VRAM_SEPARATOR_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_SEPARATOR_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_separator(user, request_id)
            elif method == CONTENT_GENERATION_FOLDER_NAME:
                if not use_cpu:  # diffusion only on gpu
                    memory_threshold = max(Settings.MIN_VRAM_VIDEO_DIFFUSER_GB, getattr(Settings, 'MEMORY', 99))
                    if (available_ram_gb > Settings.MIN_RAM_VIDEO_DIFFUSER_GB and available_vram_gb > memory_threshold) or max_queue_module_task == 1:
                        app.config["QUEUE_MODULE"][user][request_id]["busy"] = True
                        synthesize_video_generation(user, request_id)
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

    clear_cache()  # clear cache


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
    user_id = user_valid()
    if user_id is None or user_id != Settings.ADMIN_ID:
        lprint(msg="System couldn't identify user IP address.", is_server=False)
        return {"status": 404, "message": "System couldn't identify your IP address."}
    data = request.get_json()
    Settings.config_manager.update_config(data)
    for k, v in data.items():
        setattr(Settings, k, v)
    return {"status": 200}


@app.route('/get-init-config', methods=["GET"])
@cross_origin()
def get_init_config():
    all_parameters = Settings.config_manager.get_all_parameters().copy()
    all_parameters.pop("SECRET_KEY", None)
    all_parameters.pop("ADMIN_ID", None)
    return {"status": 200, "all_parameters": all_parameters}

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


@app.route("/media/<path:filename>", methods=["GET"])
@cross_origin()
def media_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=False)


def main():
    if not app.config['DEBUG']:
        background_clear_tmp()

    scheduler.add_job(background_task_cpu, 'interval', seconds=10, max_instances=2)
    scheduler.add_job(background_task_module, 'interval', seconds=30, max_instances=2)
    scheduler.add_job(background_clear_memory, 'interval', seconds=30, max_instances=1)
    scheduler.add_job(background_clear_tmp, 'interval', hours=4, max_instances=1)
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
