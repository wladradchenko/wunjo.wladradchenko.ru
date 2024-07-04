import os
import sys
import json
import psutil
import requests
from flaskwebgui import get_free_port

from backend.folders import ALL_MODEL_FOLDER, MEDIA_FOLDER, SETTING_FOLDER

# Params
ALL_MODELS_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/models.json"
HUGGINGFACE_REPO_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/repo.json"
DYNAMIC_PORT = get_free_port()


class ConfigurationManager:
    def __init__(self, config_file_path):
        mem = psutil.virtual_memory()

        self.config_file_path = config_file_path
        self.default_config = {
          "CPU_MAX_QUEUE_ALL_USER": 10,

          "CPU_MAX_QUEUE_ONE_USER": 2,

          "CPU_QUEUE_CANCEL_TIME_MIN": 5,
          "MODULE_QUEUE_CANCEL_TIME_MIN": 20,

          "CPU_DELAY_TIME_SEC": 3,
          "MODULE_DELAY_TIME_SEC": 10,

          "MAX_FILE_SIZE": 104857600,

          "MAX_RESOLUTION_REMOVE_BACKGROUND": 1920,
          "MAX_RESOLUTION_REMOVE_OBJECT": 1920,
          "MAX_RESOLUTION_FACE_SWAP": 1920,
          "MAX_RESOLUTION_ENHANCEMENT": 1920,
          "MAX_RESOLUTION_LIP_SYNC": 1920,
          "MAX_RESOLUTION_DIFFUSER": 1080,

          "MAX_DURATION_SEC_REMOVE_BACKGROUND": 1800,
          "MAX_DURATION_SEC_REMOVE_OBJECT": 1800,
          "MAX_DURATION_SEC_FACE_SWAP": 1800,
          "MAX_DURATION_SEC_ENHANCEMENT": 1800,
          "MAX_DURATION_SEC_LIP_SYNC": 1800,
          "MAX_DURATION_SEC_DIFFUSER": 300,
          "MAX_DURATION_SEC_SEPARATOR": 3000,
          "MAX_DURATION_SEC_CLONE_VOICE": 3000,

          "MAX_NUMBER_SYNBOLS_CLONE_VOICE": 5000,

          "CHECK_DOWNLOAD_SIZE_SAM_CPU": True,
          "CHECK_DOWNLOAD_SIZE_SAM_GPU": True,
          "CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT": True,
          "CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT": True,
          "CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT": True,
          "CHECK_DOWNLOAD_SIZE_FACE_SWAP": True,
          "CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN": True,
          "CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN": True,
          "CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN": True,
          "CHECK_DOWNLOAD_SIZE_LIP_SYNC": True,
          "CHECK_DOWNLOAD_SIZE_DIFFUSER": True,
          "CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER": True,
          "CHECK_DOWNLOAD_SIZE_CONTROLNET_HED": True,
          "CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY": True,
          "CHECK_DOWNLOAD_SIZE_SEPARATOR": True,
          "CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE": True,

          "STATIC_PORT": 8000,
          "SECRET_KEY": os.urandom(12).hex()
        }
        self.config_data = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file_path):
            # Create the config file with default parameters if it doesn't exist
            self.save_config(self.default_config)
        try:
            with open(self.config_file_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return self.default_config

    def save_config(self, config_data):
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)

    def update_config(self, update_config_data):
        for k, v in update_config_data.items():
            if self.config_data.get(k) is not None:
                if isinstance(self.config_data[k], int) and isinstance(update_config_data[k], int):
                    self.config_data[k] = v  # Use new value
                elif isinstance(self.config_data[k], bool) and isinstance(update_config_data[k], bool):
                    self.config_data[k] = v  # Use new value
        with open(self.config_file_path, 'w') as f:
            json.dump(self.config_data, f, indent=4)

    def get_parameter(self, parameter_name, default=None):
        if parameter_name in self.config_data:
            value = self.config_data[parameter_name]
            if isinstance(default, int) and not isinstance(value, int):
                value = default  # Use default value if type doesn't match
            elif isinstance(default, bool) and not isinstance(value, bool):
                value = default  # Use default value if type doesn't match
            return value
        else:
            return default

    def get_all_parameters(self):
        return self.config_data


class Settings:
    VERSION = "2.0.3ce"  # Version can be p (professional) and c (community edition)
    FRONTEND_HOST = "https://wunjo.online"

    config_manager = ConfigurationManager(os.path.join(SETTING_FOLDER, f"init.{VERSION}.json"))  # General config

    ADMIN_ID = "127.0.0.1"  # can be ip or id from db (relate to user_valid() in app.py)
    API_SERVER = ""
    SECRET_KEY = config_manager.get_parameter('SECRET_KEY', os.urandom(12).hex())

    SHARE = config_manager.get_parameter('SHARE', False)
    SHARE_LINK = None

    STATIC_PORT = config_manager.get_parameter('STATIC_PORT', 8000)
    DYNAMIC_PORT = DYNAMIC_PORT

    CPU_MAX_QUEUE_ALL_USER = config_manager.get_parameter('CPU_MAX_QUEUE_ALL_USER', 10)  # how mush cpu task run in sum of all user
    MODULE_MAX_QUEUE_ALL_USER = config_manager.get_parameter('MODULE_MAX_QUEUE_ALL_USER', 1)  # how mush module gpu/cpu task run in sum of all user

    CPU_MAX_QUEUE_ONE_USER = config_manager.get_parameter('CPU_MAX_QUEUE_ONE_USER', 2)  # how much cpu task run per user
    MODULE_MAX_QUEUE_ONE_USER = config_manager.get_parameter('MODULE_MAX_QUEUE_ONE_USER', 1)  # how much module gpu/cpu  task run per user

    CPU_QUEUE_CANCEL_TIME_MIN = config_manager.get_parameter('CPU_QUEUE_CANCEL_TIME_MIN', 5)  # task with datetime less than now() - set time will removed
    MODULE_QUEUE_CANCEL_TIME_MIN = config_manager.get_parameter('MODULE_QUEUE_CANCEL_TIME_MIN', 20)  # task with datetime less than now() - set time will removed

    CPU_DELAY_TIME_SEC = config_manager.get_parameter('CPU_DELAY_TIME_SEC', 3)  # time sleep after run one task
    MODULE_DELAY_TIME_SEC = config_manager.get_parameter('MODULE_DELAY_TIME_SEC', 10)  # time sleep after run one task

    MIN_RAM_CPU_TASK_GB = config_manager.get_parameter('MIN_RAM_CPU_TASK_GB', 7)  # min ram to run cpu task

    # limit ram and vram to run task
    MIN_RAM_REMOVE_BACKGROUND_GB = config_manager.get_parameter('MIN_RAM_REMOVE_BACKGROUND_GB', 5)
    MIN_VRAM_REMOVE_BACKGROUND_GB = config_manager.get_parameter('MIN_VRAM_REMOVE_BACKGROUND_GB', 5)
    MIN_RAM_REMOVE_OBJECT_GB = config_manager.get_parameter('MIN_RAM_REMOVE_OBJECT_GB', 5)
    MIN_VRAM_REMOVE_OBJECT_GB = config_manager.get_parameter('MIN_VRAM_REMOVE_OBJECT_GB', 5)
    MIN_RAM_FACE_SWAP_GB = config_manager.get_parameter('MIN_RAM_FACE_SWAP_GB', 5)
    MIN_VRAM_FACE_SWAP_GB = config_manager.get_parameter('MIN_VRAM_FACE_SWAP_GB', 2)
    MIN_RAM_ENHANCEMENT_GB = config_manager.get_parameter('MIN_RAM_ENHANCEMENT_GB', 5)
    MIN_VRAM_ENHANCEMENT_GB = config_manager.get_parameter('MIN_VRAM_ENHANCEMENT_GB', 5)
    MIN_RAM_LIP_SYNC_GB = config_manager.get_parameter('MIN_RAM_LIP_SYNC_GB', 10)
    MIN_VRAM_LIP_SYNC_GB = config_manager.get_parameter('MIN_VRAM_LIP_SYNC_GB', 4)
    MIN_RAM_DIFFUSER_GB = config_manager.get_parameter('MIN_RAM_DIFFUSER_GB', 10)
    MIN_VRAM_DIFFUSER_GB = config_manager.get_parameter('MIN_VRAM_DIFFUSER_GB', 7)
    MIN_RAM_SEPARATOR_GB = config_manager.get_parameter('MIN_RAM_SEPARATOR_GB', 5)
    MIN_VRAM_SEPARATOR_GB = config_manager.get_parameter('MIN_VRAM_SEPARATOR_GB', 4)
    MIN_VRAM_VIDEO_DIFFUSER_GB = config_manager.get_parameter('MIN_VRAM_VIDEO_DIFFUSER_GB', 7)
    MIN_RAM_VIDEO_DIFFUSER_GB = config_manager.get_parameter('MIN_RAM_VIDEO_DIFFUSER_GB', 7)

    MAX_FILE_SIZE = config_manager.get_parameter('MAX_FILE_SIZE', 100 * 1024 * 1024)  # 100 MB in bytes

    # max resolution if content with upper size then reduce size
    MAX_RESOLUTION_REMOVE_BACKGROUND = config_manager.get_parameter('MAX_RESOLUTION_REMOVE_BACKGROUND', 1920)
    MAX_RESOLUTION_REMOVE_OBJECT = config_manager.get_parameter('MAX_RESOLUTION_REMOVE_OBJECT', 1920)
    MAX_RESOLUTION_FACE_SWAP = config_manager.get_parameter('MAX_RESOLUTION_FACE_SWAP', 1920)
    MAX_RESOLUTION_ENHANCEMENT = config_manager.get_parameter('MAX_RESOLUTION_ENHANCEMENT', 1920)
    MAX_RESOLUTION_LIP_SYNC = config_manager.get_parameter('MAX_RESOLUTION_LIP_SYNC', 1920)
    MAX_RESOLUTION_DIFFUSER = config_manager.get_parameter('MAX_RESOLUTION_DIFFUSER', 1080)

    # max duration of content, if content time more, than cut end or skipp next frames
    MAX_DURATION_SEC_REMOVE_BACKGROUND = config_manager.get_parameter('MAX_DURATION_SEC_REMOVE_BACKGROUND', 1800)
    MAX_DURATION_SEC_REMOVE_OBJECT = config_manager.get_parameter('MAX_DURATION_SEC_REMOVE_OBJECT', 1800)
    MAX_DURATION_SEC_FACE_SWAP = config_manager.get_parameter('MAX_DURATION_SEC_FACE_SWAP', 1800)
    MAX_DURATION_SEC_ENHANCEMENT = config_manager.get_parameter('MAX_DURATION_SEC_ENHANCEMENT', 1800)
    MAX_DURATION_SEC_LIP_SYNC = config_manager.get_parameter('MAX_DURATION_SEC_LIP_SYNC', 1800)
    MAX_DURATION_SEC_DIFFUSER = config_manager.get_parameter('MAX_DURATION_SEC_DIFFUSER', 300)
    MAX_DURATION_SEC_SEPARATOR = config_manager.get_parameter('MAX_DURATION_SEC_SEPARATOR', 3000)
    MAX_DURATION_SEC_CLONE_VOICE = config_manager.get_parameter('MAX_DURATION_SEC_CLONE_VOICE', 3000)

    MAX_NUMBER_SYNBOLS_CLONE_VOICE = config_manager.get_parameter('MAX_NUMBER_SYNBOLS_CLONE_VOICE', 5000)
    # TODO добавить в html настройки этот параметр

    # use check download size for each module
    CHECK_DOWNLOAD_SIZE_SAM_CPU = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_SAM_CPU', True)
    CHECK_DOWNLOAD_SIZE_SAM_GPU = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_SAM_GPU', True)
    CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_DETECTOR_TEXT', True)
    CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_IMPROVE_REMOVE_OBJECT', True)
    CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_SIMPLE_REMOVE_OBJECT', True)
    CHECK_DOWNLOAD_SIZE_FACE_SWAP = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_FACE_SWAP', True)
    CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_ENHANCEMENT_ANIMESGAN', True)
    CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_ENHANCEMENT_REALSRGAN', True)
    CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_ENHANCEMENT_GFPGAN', True)
    CHECK_DOWNLOAD_SIZE_LIP_SYNC = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_LIP_SYNC', True)
    CHECK_DOWNLOAD_SIZE_SEPARATOR = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_SEPARATOR', True)
    CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_CONTROLNET_TILE', True)
    CHECK_DOWNLOAD_SIZE_DIFFUSER = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_DIFFUSER', True)
    CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_VIDEO_DIFFUSER', True)
    CHECK_DOWNLOAD_SIZE_CONTROLNET_HED = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_CONTROLNET_HED', True)
    CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY = config_manager.get_parameter('CHECK_DOWNLOAD_SIZE_CONTROLNET_CANNY', True)

    @staticmethod
    def get_max_resolution_diffuser(resolution=MAX_RESOLUTION_DIFFUSER, device="cuda"):
        import torch

        # get max limit target size
        gpu_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        # table of gpu memory limit
        gpu_table = {24: 1280, 18: 1024, 14: 768, 10: 640, 8: 576, 7: 512, 6: 448, 5: 320, 4: 192, 0: 0}
        # get user resize for gpu
        device_resolution = max(val for key, val in gpu_table.items() if key <= gpu_vram)
        if gpu_vram < 4:
            print("Small VRAM to use GPU. Will be use config resolution")
            return 0
        if resolution < device_resolution:
            # Video will not resize
            return resolution
        return device_resolution


# Get configs file with models
class ModelsConfig:
    CONFIG = {}

    def __call__(self):
        if len(self.CONFIG.keys()) == 0:
            print("Model config updated")
            self.get_config()
        return self.CONFIG

    def get_config(self):
        if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
            try:
                response = requests.get(ALL_MODELS_JSON_URL)
                with open(os.path.join(ALL_MODEL_FOLDER, 'all_models.json'), 'wb') as file:
                    file.write(response.content)
            except:
                print("Not internet connection to get actual versions of models")

        if not os.path.isfile(os.path.join(ALL_MODEL_FOLDER, 'all_models.json')):
            self.CONFIG = {}
        else:
            with open(os.path.join(ALL_MODEL_FOLDER, 'all_models.json'), 'r', encoding="utf8") as file:
                self.CONFIG = json.load(file)


class HuggingfaceRepoConfig:
    CONFIG = {}

    def __call__(self):
        if len(self.CONFIG.keys()) == 0:
            print("Model config updated")
            self.get_config()
        return self.CONFIG

    def get_config(self):
        if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'False':
            try:
                response = requests.get(HUGGINGFACE_REPO_JSON_URL)
                with open(os.path.join(ALL_MODEL_FOLDER, 'repo.json'), 'wb') as file:
                    file.write(response.content)
            except:
                print("Not internet connection to get actual versions of models")

        if not os.path.isfile(os.path.join(ALL_MODEL_FOLDER, 'repo.json')):
            self.CONFIG = {}
        else:
            with open(os.path.join(ALL_MODEL_FOLDER, 'repo.json'), 'r', encoding="utf8") as file:
                self.CONFIG = json.load(file)


# WebGUI Browser
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    WEBGUI_PATH = os.path.join(root_path, "win", "webgui.exe")
    if not os.path.exists(WEBGUI_PATH):
        WEBGUI_PATH = None
elif sys.platform == 'linux':
    WEBGUI_PATH = os.path.join(root_path, "linux", "webgui.AppImage")
    if not os.path.exists(WEBGUI_PATH):
        WEBGUI_PATH = None
else:
    WEBGUI_PATH = None