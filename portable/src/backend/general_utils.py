import os
import gc
import sys
import json
import time
import torch
import psutil
import shutil
import base64
import random
import requests
from time import strftime
from cryptography.fernet import Fernet

from backend.config import Settings
from backend.folders import SETTING_FOLDER
from backend.download import download_model, unzip, check_download_size


def get_vram_gb():
    if torch.cuda.is_available():
        device = "cuda"
        properties = torch.cuda.get_device_properties(device)
        total_vram_gb = properties.total_memory / (1024 ** 3)
        available_vram_gb = (properties.total_memory - torch.cuda.memory_allocated()) / (1024 ** 3)
        busy_vram_gb = total_vram_gb - available_vram_gb
        return total_vram_gb, available_vram_gb, busy_vram_gb
    return 0, 0, 0


def get_ram_gb():
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    available_ram_gb = mem.available / (1024 ** 3)
    busy_ram_gb = total_ram_gb - available_ram_gb
    return total_ram_gb, available_ram_gb, busy_ram_gb


def ensure_memory_capacity(limit_ram=0, limit_vram=0, device="cuda", retries=15, delay=30):
    """
    Check what memory will not be out, with a specified number of retries and delay between retries.

    :param limit_ram: How much need to use RAM for module.
    :param limit_vram: How much need to use VRAM for module.
    :param device: Whta is device using.
    :param retries: Number of times to retry.
    :param delay: Delay in seconds between retries.
    :return: True if file exists, False otherwise.
    """
    if Settings.MODULE_MAX_QUEUE_ALL_USER == 1 and Settings.MODULE_MAX_QUEUE_ONE_USER == 1:
        print("The parameters MODULE_MAX_QUEUE_ALL_USER and MODULE_MAX_QUEUE_ONE_USER are set to 1, which means that only one task is processed at a time. Go to the settings page to increase the number of MODULE_MAX_QUEUE_ALL_USER, MODULE_MAX_QUEUE_ONE_USER and adjust the RAM settings for the modules.")
        return True
    for _ in range(retries):
        random_delay = random.randint(1, delay)
        _, available_ram_gb, _ = get_ram_gb()
        _, available_vram_gb, _ = get_vram_gb()
        if device == "cuda":
            if available_ram_gb > limit_ram and available_vram_gb > limit_vram:
                return True
        else:
            if available_ram_gb > limit_ram:
                return True
        lprint(f"Waiting for free memory because of available RAM {available_ram_gb}, but need to have free RAM {limit_ram}")
        lprint(f"Waiting for free memory because of available VRAM {available_vram_gb}, but need to have free VRAM {limit_vram}")
        time.sleep(random_delay)
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    return False


def format_log_message(msg: str, is_server: bool = True, level: str = None, user: str = None) -> str:
    """Log formate message"""
    log_msg = "[SERVER]" if is_server else "[CLIENT]"

    if level == "error":
        log_msg += "[ERROR]"
    elif level == "warn":
        log_msg += "[WARNING]"
    else:
        log_msg += "[INFO]"

    if user is not None:
        log_msg += f"[{user}]"

    log_msg += f'[{strftime("%Y-%m-%d %H:%M:%S")}]'
    return f"{log_msg} {msg}"


def lprint(msg: str, is_server: bool = True, level: str = None, user: str = None) -> None:
    """Generate log print or maybe save print in file if needed"""
    print(format_log_message(msg, is_server, level, user))


def generate_file_tree(directory: str):
    """Generate files tress as list of dict"""
    file_tree = []
    # Iterate over the contents of the directory
    for item in sorted(os.listdir(directory), reverse=True, key=lambda x: os.path.getctime(os.path.join(directory, x)) if os.path.exists(os.path.join(directory, x)) else None):
        item_path = os.path.join(directory, item)
        item_info = {"name": item}
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Recursively generate the file tree for the directory
            item_info["isDirectory"] = True
            item_info["child"] = generate_file_tree(item_path)
            item_info["censure"] = False
            if not item_info["child"]:
                item_info["visible"] = False
            else:
                item_info["visible"] = any(child.get("visible", False) for child in item_info["child"])
        else:
            # If the item is a file, mark it as not a directory
            item_info["isDirectory"] = False
            item_info["visible"] = True
            item_info["censure"] = False
        # Get the creation time of the file or directory
        try:
            creation_time = os.path.getctime(item_path)
            item_info["created"] = creation_time if creation_time is not None else None
        except OSError:
            item_info["created"] = None
        file_tree.append(item_info)
    return file_tree


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


def decrypted_bytes(val_base64):
    return base64.b64decode(val_base64)


def decrypted_key(val_base64):
    return Fernet(decrypted_bytes(val_base64))


def decrypted_val(val_base64, key_byte, *args):
    encrypted_bytes = base64.b64decode(val_base64)
    decrypted_data = key_byte.decrypt(encrypted_bytes)
    decrypted_json = json.loads(decrypted_data)
    for a in args:
        if isinstance(decrypted_json, dict):
            decrypted_json = decrypted_json.get(a)
        else:
            return decrypted_json
    return decrypted_json
