import os
import re
import sys
import zipfile
import ctypes

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # remove msg

import socket
import logging
import subprocess
from packaging import version
from flask import Flask, g, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from flask_babel import Babel
from flaskwebgui import FlaskUI, kill_port, close_application

from backend.folders import ALL_MODEL_FOLDER
from backend.config import Settings, WEBGUI_PATH
from backend.download import download_model


# pybabel extract -F babel.cfg -o messages.pot .
# pybabel update -i messages.pot -d translations -l ru (pybabel init -i messages.pot -d translations -l ru)
# pybabel compile -d translations

def get_locale():
    # if a user is logged in, use the locale from the user settings
    return Settings.LANGUAGE_INTERFACE_CODE if Settings.LANGUAGE_INTERFACE_CODE in ['en', 'ru'] else 'en'


def get_timezone():
    user = getattr(g, 'user', None)
    if user is not None:
        return user.timezone


app = Flask(__name__)
cors = CORS(app)
# Set translation
translation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'translations'))
babel = Babel(app, locale_selector=get_locale, timezone_selector=get_timezone, default_translation_directories=translation_path)


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
os.environ['DEBUG'] = 'False'  # str False or True
app.config['DEBUG'] = os.environ.get('DEBUG', 'False') == 'True'
app.config['PORT'] = set_available_port() if not app.config['DEBUG'] else Settings.STATIC_PORT

logging.getLogger('werkzeug').disabled = True


def is_admin():
    if sys.platform == 'win32':
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return os.getuid() == 0


def grant_permissions():
    if sys.platform == 'win32':
        try:
            app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root_path = os.path.dirname(app_path)
            app_packages_path = os.path.join(root_path, "app_packages")
            cmd = f'icacls "{app_packages_path}" /grant:r "Everyone:(R,W)" /T'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Permissions granted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to set permissions: {e}")


def unpack_whl(whl_file, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    # Open the .whl file as a zip archive
    with zipfile.ZipFile(whl_file, 'r') as zip_ref:
        # Extract all contents of the .whl file to the destination folder
        zip_ref.extractall(destination_folder)
    # Set permission if windows
    if sys.platform == 'win32':
        try:
            username = os.environ.get('USERNAME') or os.environ.get('USER')
            cmd = f'icacls "{destination_folder}" /grant:r "Everyone:(R,W)" /T' if is_admin() else f'icacls "{destination_folder}" /grant:r "{username}:(R,W)" /T'
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                os.system(cmd)
            else:
                # silence run
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(e)
    # Remove whl file
    os.remove(whl_file)


def download_torch():
    # Get CUDA version
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        nvcc_version = re.search(r"release (\d+\.\d+)", nvcc_version).group(1) if nvcc_version else None
        if not nvcc_version or version.parse(nvcc_version) < version.parse("11.8"):
            raise "Downloaded CUDA 11.8 or upper."
        driver = "gpu"
    except (subprocess.CalledProcessError, FileNotFoundError):
        driver = "cpu"

    # Download
    app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = os.path.dirname(app_path)
    app_packages_path = os.path.join(root_path, "app_packages")
    if driver == "gpu":
        torch_file = os.path.join(ALL_MODEL_FOLDER, "torch-2_2_2_cu118-cp310-cp310-win_amd64.whl")
        torch_link = "https://download.pytorch.org/whl/cu118/torch-2.2.2%2Bcu118-cp310-cp310-win_amd64.whl"
        torchaudio_file = os.path.join(ALL_MODEL_FOLDER, "torchaudio-2_2_2_cu118-cp310-cp310-win_amd64.whl")
        torchaudio_link = "https://download.pytorch.org/whl/cu118/torchaudio-2.2.2%2Bcu118-cp310-cp310-win_amd64.whl"
        torchvision_file = os.path.join(ALL_MODEL_FOLDER, "torchvision-0_17_2_cu118-cp310-cp310-win_amd64.whl")
        torchvision_link = "https://download.pytorch.org/whl/cu118/torchvision-0.17.2%2Bcu118-cp310-cp310-win_amd64.whl"
    else:
        torch_file = os.path.join(ALL_MODEL_FOLDER, "torch-2_2_2_cpu-cp310-cp310-win_amd64.whl")
        torch_link = "https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl"
        torchaudio_file = os.path.join(ALL_MODEL_FOLDER, "torchaudio-2_2_2_cpu-cp310-cp310-win_amd64.whl")
        torchaudio_link = "https://download.pytorch.org/whl/cpu/torchaudio-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl"
        torchvision_file = os.path.join(ALL_MODEL_FOLDER, "torchvision-0_17_2_cpu-cp310-cp310-win_amd64.whl")
        torchvision_link = "https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp310-cp310-win_amd64.whl"

    # Not need to check path, because of folders is not exist
    download_model(torch_file, torch_link)
    print("Torch is downloaded, start unpacking.")
    if not os.path.exists(torch_file):
        raise "Downloaded torch whl is not found"
    else:
        unpack_whl(torch_file, app_packages_path)
        print("Torch is unpacked.")

    download_model(torchaudio_file, torchaudio_link)
    print("Torchaudio is downloaded, start unpacking.")
    if not os.path.exists(torchaudio_file):
        raise "Downloaded torchaudio whl is not found"
    else:
        unpack_whl(torchaudio_file, app_packages_path)
        print("Torchaudio is unpacked.")

    download_model(torchvision_file, torchvision_link)
    print("Torchvision is downloaded, start unpacking.")
    if not os.path.exists(torchvision_file):
        raise "Downloaded torchvision whl is not found"
    else:
        unpack_whl(torchvision_file, app_packages_path)
        print("Torchvision is unpacked.")


@app.context_processor
def inject_locale():
    return dict(get_locale=get_locale)


if app.config['DEBUG']:
    @app.before_request
    def reload_templates():
        """Control reload page after HTML change if Debug"""
        app.jinja_env.cache.clear()


"""TRANSLATION"""
@app.route('/translations.json')
def translations():
    return jsonify({})
"""TRANSLATION"""


@app.route("/", methods=["GET"])
@cross_origin()
def index_page():
    try:
        import onnxruntime
        onnxruntime_bool = True
    except ImportError:
        onnxruntime_bool = False
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        nvcc_version = re.search(r"release (\d+\.\d+)", nvcc_version).group(1) if nvcc_version else None
        if version.parse(nvcc_version) < version.parse("11.8"):  # Min CUDA version
            nvcc_version = None
    except (subprocess.CalledProcessError, FileNotFoundError):
        nvcc_version = None
    return render_template("preload.html", host=Settings.FRONTEND_HOST, version=Settings.VERSION, nvcc_version=nvcc_version, platform=sys.platform, onnxruntime_bool=onnxruntime_bool, is_admin=is_admin())


@app.route('/open-app-packages', methods=['GET'])
def open_app_packages():
    import platform
    app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = os.path.dirname(app_path)
    app_packages_path = os.path.join(root_path, "app_packages")
    if not os.access(app_packages_path, os.R_OK):
        return jsonify({"status": "error", "message": "No permission to access the directory."}), 403
    try:
        if platform.system() == "Windows":
            # os.startfile(app_packages_path)  # Windows
            if os.path.exists(os.path.join(app_packages_path, "torch")):
                subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torch")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif os.path.exists(os.path.join(app_packages_path, "torchaudio")):
                subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torchaudio")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif os.path.exists(os.path.join(app_packages_path, "torchvision")):
                subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torchvision")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(f'start explorer /select,"{app_packages_path}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", app_packages_path])
        else:  # Linux
            subprocess.run(["xdg-open", app_packages_path])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/preload", methods=["POST"])
@cross_origin()
def preload():
    if sys.platform == 'win32':
        # Download folder
        app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_path = os.path.dirname(app_path)
        app_packages_path = os.path.join(root_path, "app_packages")
        # Exists
        if os.path.exists(os.path.join(app_packages_path, "torch")):
            # os.startfile(os.path.join(app_packages_path, "torch"))  # Windows
            subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torch")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Need to be removed old torch.")
            return jsonify({"status": 403})
        if os.path.exists(os.path.join(app_packages_path, "torchvision")):
            # os.startfile(os.path.join(app_packages_path, "torch"))  # Windows
            subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torchvision")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Need to be removed old torchvision.")
            return jsonify({"status": 403})
        if os.path.exists(os.path.join(app_packages_path, "torchaudio")):
            # os.startfile(os.path.join(app_packages_path, "torch"))  # Windows
            subprocess.run(f'start explorer /select,"{os.path.join(app_packages_path, "torchaudio")}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Need to be removed old torchaudio.")
            return jsonify({"status": 403})

    try:
        download_torch()
    except Exception as err:
        print(str(err))
        return jsonify({"status": 404})
    return jsonify({"status": 200})


@app.route("/exit", methods=["POST"])
@cross_origin()
def exit_app():
    if not app.config['DEBUG'] and sys.platform != 'darwin' and ((sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32'):
        close_application()
    else:
        port = app.config['PORT']
        kill_port(port)
    return jsonify({"status": 200})


if not app.config['DEBUG']:
    from time import time
    from io import StringIO

    print(f"http://127.0.0.1:{app.config['PORT']}")
    print(f"http://127.0.0.1:{app.config['PORT']}/console-log")

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

        # Covert bytes log
        combined_logs = [(log[0], log[1].decode('utf-8')) if isinstance(log[1], bytes) else log for log in combined_logs]

        # Filter out unwanted phrases and extract log messages and show user only user log if not admin
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


@app.route('/set-init-config', methods=["POST"])
@cross_origin()
def set_init_config():
    data = request.get_json()
    # Remove keys with None values
    filtered_data = {key: value for key, value in data.items() if value is not None}
    # Set settings
    Settings.config_manager.update_config(filtered_data)
    for k, v in filtered_data.items():
        setattr(Settings, k, v)
    return {"status": 200}


def main():
    if is_admin():
        grant_permissions()
    else:
        print("Please run the program as an administrator to set folder permissions.")

    def server(**server_kwargs):
        server_app = server_kwargs.pop("app", None)
        server_kwargs.pop("debug", None)
        server_app.run(host='0.0.0.0', **server_kwargs)

    # Init app by mode, OS and is display or not
    port = app.config['PORT']  # get set port
    if not app.config['DEBUG'] and sys.platform != 'darwin' and ((sys.platform == 'linux' and os.environ.get('DISPLAY')) or sys.platform == 'win32'):
        FlaskUI(server=server, browser_path=WEBGUI_PATH, server_kwargs={"app": app, "port": port}).run()
    else:
        print(f"http://127.0.0.1:{port}")
        app.run(host="0.0.0.0", port=port)
