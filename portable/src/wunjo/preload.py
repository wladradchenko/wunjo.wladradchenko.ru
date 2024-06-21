import os
import sys
import zipfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # remove msg

import socket
import logging
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from flaskwebgui import FlaskUI, kill_port, close_application

from backend.folders import ALL_MODEL_FOLDER
from backend.config import Settings, WEBGUI_PATH
from backend.download import download_model


app = Flask(__name__)
cors = CORS(app)


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


def unpack_whl(whl_file, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    # Open the .whl file as a zip archive
    with zipfile.ZipFile(whl_file, 'r') as zip_ref:
        # Extract all contents of the .whl file to the destination folder
        zip_ref.extractall(destination_folder)
    # Remove whl file
    os.remove(whl_file)


def download_torch(driver: str = None):
    app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = os.path.dirname(app_path)
    app_packages_path = os.path.join(root_path, "app_packages")
    if driver == "gpu":
        torch_file = os.path.join(ALL_MODEL_FOLDER, "torch-2_0_0_cu118-cp310-cp310-win_amd64.whl")
        torch_link = "https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl"
    else:
        torch_file = os.path.join(ALL_MODEL_FOLDER, "torch-2_0_0_cpu-cp310-cp310-win_amd64.whl")
        torch_link = "https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp310-cp310-win_amd64.whl"
    # Not need to check path, because of folders is not exist
    download_model(torch_file, torch_link)
    print("Torch is downloaded, start unpacking.")
    if not os.path.exists(torch_file):
        raise "Downloaded torch whl is not found"
    else:
        unpack_whl(torch_file, app_packages_path)
        print("Torch is unpacked.")


@app.route("/", methods=["GET"])
@cross_origin()
def index_page():
    return render_template("preload.html", host=Settings.FRONTEND_HOST, version=Settings.VERSION)


@app.route("/preload", methods=["POST"])
@cross_origin()
def preload():
    data = request.get_json()
    driver = data.get("driver")
    try:
        download_torch(driver)
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


def main():
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
