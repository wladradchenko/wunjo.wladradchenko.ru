import os
import sys
import json
import shutil
import requests
from time import gmtime, strftime

from backend.folders import SETTING_FOLDER
from backend.download import download_model, unzip, check_download_size


def is_ffmpeg_installed():
    if not shutil.which('ffmpeg'):
        print('Ffmpeg is not installed. You need to install it for comfortable use of the program.')
        print("You can visit documentation https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki to find out how install")
        download_ffmpeg()
        if not shutil.which('ffmpeg'):
            return False
    return True


def download_ffmpeg():
    """
    Install ffmpeg
    :return:
    """
    if sys.platform == 'win32':
        # Open folder for Windows
        print("Starting automation install ffmpeg on Windows")
        ffmpeg_path = os.path.join(SETTING_FOLDER, "ffmpeg")
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        if not os.path.exists(ffmpeg_path):
            download_model(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_url)
            unzip(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_path)
        else:
            check_download_size(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_url)
            if not os.listdir(ffmpeg_path):
                unzip(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_path)
        ffmpeg_bin_path = os.path.join(ffmpeg_path, "bin")
        os.environ["PATH"] += os.pathsep + ffmpeg_bin_path
        print(os.pathsep)
    elif sys.platform == 'darwin':
        # Ffmpeg for MacOS
        print("Please install ffmpeg using the following command")
        print("brew install ffmpeg")
    elif sys.platform == 'linux':
        # Ffmpeg for Linux
        print("Please install ffmpeg using the following command")
        print("sudo apt update && sudo apt install -y ffmpeg")


def download_ntlk():
    tokenizers_punkt_path = os.path.join(SETTING_FOLDER, "nltk_data", "tokenizers")

    punkt_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"

    try:
        if not os.path.exists(tokenizers_punkt_path):
            os.makedirs(tokenizers_punkt_path)
        if not os.path.exists(os.path.join(tokenizers_punkt_path, "punkt")):
            print(f"Punkt NTLK is not found. Download this {punkt_url}")
            download_model(os.path.join(tokenizers_punkt_path, "punkt.zip"), punkt_url)
            unzip(os.path.join(tokenizers_punkt_path, 'punkt.zip'), tokenizers_punkt_path)
        else:
            check_download_size(os.path.join(tokenizers_punkt_path, "punkt.zip"), punkt_url)
    except Exception as err:
        print(f"Error during download NLTK {err}")


def get_version_app():
    version_json_url = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/version_control.json"
    try:
        response = requests.get(version_json_url)
        return json.loads(response.text)
    except requests.RequestException:
        print("Error fetching the version information, when get information about updates of application")
        return {}
    except json.JSONDecodeError:
        print("Error decoding the JSON response, when get information about updates of application")
        return {}
    except:
        print("An unexpected error occurred, when get information about updates of application")
        return {}


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
            except Exception as err:
                print(f"Error ... {err}")
                return default_settings


def current_time(time_format: str = None):
    if time_format:
        return strftime(time_format, gmtime())
    return strftime("%Y%m%d%H%M%S", gmtime())


def _create_localization():
    localization_path = os.path.join(SETTING_FOLDER, "localization.json")
    with open(localization_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False)


# create localization file
_create_localization()
