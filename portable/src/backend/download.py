import os
import sys
import time
import socket
import zipfile
import requests
import subprocess
import shutil
from tqdm import tqdm


def is_connected(model_path):
    try:
        socket.create_connection(("www.google.com", 80))
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
                    cmd = f'icacls "{extract_dir}" /grant:r "{username}:(R,W)" /T'
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


def download_model(download_path: str, download_link: str, retry_count: int = 2, retry_delay: int = 5) -> bool:
    for i in range(retry_count + 1):
        try:
            response = requests.get(download_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            response.raise_for_status()
            if total_size > 0:
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, file=sys.stdout)
            else:
                progress_bar = tqdm(unit='iB', unit_scale=True, file=sys.stdout)
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print(f'Download failed: network error. You can download the file yourself from the link {download_link}')
                continue
            else:
                print(f'Download finished in {download_path}')
                if sys.platform == 'win32':
                    try:
                        username = os.environ.get('USERNAME') or os.environ.get('USER')
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



def check_download_size(download_path: str, download_link: str) -> bool:
    try:
        # if is internet connection
        response = requests.get(download_link, stream=True)
        total_size = int(response.headers.get('content-length', 0))
    except:
        # if not internet connection
        total_size = os.path.getsize(download_path)

    if not os.path.exists(download_path):
        return download_model(download_path, download_link)

    downloaded_size = os.path.getsize(download_path)
    if downloaded_size == total_size:
        print(f'File verified {download_path}')
        return True
    print(f"File is not verified, re-download ones {download_link}")

    os.remove(download_path)  # delete not finished file

    return download_model(download_path, download_link)


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
