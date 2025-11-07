import os
import sys
import time
import socket
import zipfile
import requests
import subprocess
import shutil
from tqdm import tqdm
import ctypes


def is_admin():
    if sys.platform == 'win32':
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return os.getuid() == 0


def is_connected(model_path: str, offline_mode: bool = False):
    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True' or offline_mode:
        # Offline mode
        return False
    try:
        with socket.create_connection(("www.google.com", 80)) as sock:
            return True
    except OSError:
        pass
    raise Exception(f"Model not found at {model_path}. The application cannot access the internet. Please allow Wunjo AI through your firewall or download the models manually. For more details, visit our documentation: https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki")


def get_nested_url(d, keys: list):
    if not isinstance(d, dict):
        return d
    if len(keys) == 0:
        return d
    key = keys[0]
    if key in d:
        return get_nested_url(d[key], keys[1:])
    else:
        raise Exception(f"This {key} is not found in config file. Update config file via internet! For more details, visit our documentation: https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki")


def unzip(zip_file_path, extract_dir, target_dir_name=None):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        if target_dir_name is None:
            zip_ref.extractall(extract_dir)
        else:
            # Extract the contents of the zip file to a temporary directory
            temp_dir = os.path.join(extract_dir, '_temp')
            zip_ref.extractall(path=temp_dir)

            # Get the extracted directory name (should be extensions.wunjo.wladradchenko.ru-1.0.0)
            extracted_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])

            # Rename the extracted directory to the target directory name
            target_dir = os.path.join(extract_dir, target_dir_name)
            os.rename(extracted_dir, target_dir)

            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

        # Set permission if windows
        if sys.platform == 'win32':
            try:
                username = os.environ.get('USERNAME') or os.environ.get('USER')
                if target_dir_name is None:
                    if is_admin:
                        cmd = f'icacls "{extract_dir}" /grant:r "Everyone:(R,W)" /T'
                    else:
                        cmd = f'icacls "{extract_dir}" /grant:r "{username}:(R,W)" /T'
                else:
                    if is_admin:
                        cmd = f'icacls "{target_dir}" /grant:r "Everyone:(R,W)" /T'
                    else:
                        cmd = f'icacls "{target_dir}" /grant:r "{username}:(R,W)" /T'
                if os.environ.get('DEBUG', 'False') == 'True':
                    # not silence run
                    os.system(cmd)
                else:
                    # silence run
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(e)


def download_model(download_path: str, download_link: str, retry_count: int = 2, retry_delay: int = 5, progress_callback=None) -> bool:
    # Create directory if it doesn't exist using exist_ok=True
    download_dir = os.path.dirname(download_path)
    os.makedirs(download_dir, exist_ok=True)

    print(f"Prepare download {download_link} in {download_path}")

    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
        # Offline mode
        print("Error... Internet connection is failed for offline mode!")
        # Updating progress
        if progress_callback:
            progress_callback(0, "Error... Internet connection is failed for offline mode!")
        return False

    for i in range(retry_count + 1):
        try:
            response = requests.get(download_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            total_size_gb = total_size / (1024 ** 3)
            chunk_len = 0
            response.raise_for_status()
            if total_size > 0:
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, file=sys.stdout)
            else:
                progress_bar = tqdm(unit='iB', unit_scale=True, file=sys.stdout)
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        progress_bar.update(len(chunk))
                        chunk_len += len(chunk)
                        f.write(chunk)
                        # Updating progress
                        if progress_callback:
                            if total_size != 0:
                                progress_callback(round(chunk_len / total_size * 100, 0), f"Download {total_size_gb:.2f} GB {download_link}")
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print(f'Download failed: network error. You can download the file yourself from the link {download_link}')
                continue
            else:
                print(f'Download finished in {download_path}')
                if sys.platform == 'win32':
                    try:
                        username = os.environ.get('USERNAME') or os.environ.get('USER')
                        if is_admin:
                            cmd = f'icacls "{download_path}" /grant:r "Everyone:(R,W)"'
                        else:
                            cmd = f'icacls "{download_path}" /grant:r "{username}:(R,W)"'
                        if os.environ.get('DEBUG', 'False') == 'True':
                            # not silence run
                            os.system(cmd)
                        else:
                            # silence run
                            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as e:
                        print(e)
                return True
        except requests.exceptions.RequestException:
            if i == retry_count:
                os.remove(download_path)  # delete not finished file
                raise RuntimeError("Error... Internet connection is failed!")
            time.sleep(retry_delay)


def huggingface_repo_files(repo_id: str, commit: str = None):
    # Use the Hugging Face API to get the list of files in the repo
    branch = "main" if commit is None else commit
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{branch}"
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    def traverse_files(file_obj, url):
        file_type = file_obj.get("type")
        file_path = file_obj.get("path")
        if file_type == "directory":
            update_api_url = f"{url}/{file_path}"
            response = requests.get(update_api_url)
            response.raise_for_status()
            sub_files = response.json()
            for sub_file in sub_files:
                yield from traverse_files(sub_file, api_url)
        else:
            yield file_path

    for file in files:
        yield from traverse_files(file, f"https://huggingface.co/api/models/{repo_id}/tree/{branch}")


def check_commit_exists(repo_id: str, commit: str) -> bool:
    """Check if a specific commit exists for a given Hugging Face repository."""
    try:
        api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{commit}"
        response = requests.get(api_url)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False
        raise


def download_huggingface_repo(repo_id: str, download_dir: str, compare_size: bool = False, progress_callback=None, commit=None, exclude_files=None):
    if not compare_size and not is_connected("https://huggingface.co", offline_mode=True):
        if os.path.isdir(download_dir) and os.listdir(download_dir):
            print("Compare size is false and not internet connection.")
            return

    exclude_files = exclude_files if isinstance(exclude_files, list) else []
    commit = commit if commit and check_commit_exists(repo_id, commit) else None

    files = huggingface_repo_files(repo_id, commit=commit)
    base_url = f"https://huggingface.co/{repo_id}/resolve/{'main' if commit is None else commit}/"

    for file in files:
        if file in exclude_files:
            continue
        file_url = base_url + file
        file_path = os.path.join(download_dir, *file.split("/"))
        if not os.path.exists(file_path):
            if is_connected(file_path):
                print("Model not found. Download size model.")
                # download pre-trained models from url
                download_model(file_path, file_url, progress_callback=progress_callback)
            else:
                raise Exception(f"Not internet connection to download model {file_url} in {file_path}.")
        else:
            if compare_size:
                check_download_size(file_path, file_url, progress_callback=progress_callback)


def check_download_size(download_path: str, download_link: str, progress_callback=None) -> bool:
    if os.environ.get('WUNJO_OFFLINE_MODE', 'False') == 'True':
        # Offline mode
        print("In offline mode, it is impossible to check compliance with the installed model. The model will be automatically returned as the size matches")
        return True

    try:
        # if is internet connection
        response = requests.get(download_link, stream=True)
        total_size = int(response.headers.get('content-length', 0))
    except:
        # if not internet connection
        total_size = os.path.getsize(download_path)

    if not os.path.exists(download_path):
        return download_model(download_path, download_link, progress_callback=progress_callback)

    downloaded_size = os.path.getsize(download_path)
    if downloaded_size == total_size:
        print(f'File verified {download_path}')
        return True
    print(f"File is not verified, re-download ones {download_link}")

    os.remove(download_path)  # delete not finished file

    return download_model(download_path, download_link, progress_callback=progress_callback)


def get_download_filename(download_link):
    response = requests.get(download_link, stream=True)
    filename = None

    if "content-disposition" in response.headers:
        _, params = response.headers["content-disposition"].split(";")
        for param in params.split(";"):
            key, value = param.strip().split("=")
            if key == "filename":
                filename = value.strip("\"'")

    return filename
