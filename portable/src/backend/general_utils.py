import os
import re
import sys
import json
import time
import shutil
import requests
from time import strftime

from backend.folders import SETTING_FOLDER
from backend.download import download_model, unzip, check_download_size


def clean_text_by_language(text, lang, only_punct=False):
    """Clean text to use multilanguage synthesized in one text"""
    # Define patterns for each language
    patterns = {
        'en': re.compile(r'[^a-zA-Z\s]'),  # Keep only English letters and spaces
        'ru': re.compile(r'[^а-яА-Я\s]'),  # Keep only Russian letters and spaces
        'zh': re.compile(r'[^\u4e00-\u9fff\s]'),  # Keep only Chinese characters and spaces
        'punct': re.compile(r'[.!?;,:\s]+')  # Pattern to match punctuation and spaces
    }

    # Remove punctuation first
    text = patterns['punct'].sub('', text)

    if only_punct:
        return text

    # Remove all characters not belonging to the specified language
    text = patterns[lang].sub('', text)

    return text


def is_ffmpeg_installed():
    if sys.platform == 'win32':
        ffmpeg_path = os.path.join(SETTING_FOLDER, "ffmpeg")
        ffmpeg_bin_path = os.path.join(ffmpeg_path, "ffmpeg-master-latest-win64-gpl", "bin")
        os.environ["PATH"] += os.pathsep + ffmpeg_bin_path
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
        # I didn't not set check config url from sever because think what ffmpeg will not change
        # if changed, I set
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        if not os.path.exists(ffmpeg_path):
            download_model(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_url)
            unzip(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_path)
        else:
            check_download_size(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_url)
            if not os.listdir(ffmpeg_path):
                unzip(os.path.join(SETTING_FOLDER, 'ffmpeg.zip'), ffmpeg_path)
        print("Download ffmpeg is finished!")
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
    # I didn't not set check config url from sever because think what punkt will not change
    # if changed, I set
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
    version_json_url = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/CHANGELOG.json"
    try:
        response = requests.get(version_json_url, timeout=5)  # timeout in seconds
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request is not connected
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


def get_folder_size(folder_path):
    size_in_bytes = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                size_in_bytes += os.path.getsize(file_path)
    return size_in_bytes / (1024 * 1024)


def current_time():
    """Return the current time formatted as YYYYMMDD_HHMMSS."""
    return strftime("%Y_%m_%d_%H%M%S")

def format_dir_time(dir_time):
    """Convert a time string from YYYYMMDD_HHMMSS to DD.MM.YYYY HH:MM format."""
    # Extract components from the string
    year = dir_time[0:4]
    month = dir_time[5:7]
    day = dir_time[8:10]
    hour = dir_time[11:13]
    minute = dir_time[13:15]

    # Format the date and time
    formatted_date = f"{day}.{month}.{year} {hour}:{minute}"
    return formatted_date


def check_tmp_file_uploaded(file_path, retries=10, delay=30):
    """
    Check if a file exists, with a specified number of retries and delay between retries.

    :param file_path: Path to the file to check.
    :param retries: Number of times to retry.
    :param delay: Delay in seconds between retries.
    :return: True if file exists, False otherwise.
    """
    for _ in range(retries):
        if os.path.exists(file_path):
            return True
        time.sleep(delay)
    return False


def _create_localization():
    localization_path = os.path.join(SETTING_FOLDER, "localization.json")
    with open(localization_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False)


# create localization file
_create_localization()
