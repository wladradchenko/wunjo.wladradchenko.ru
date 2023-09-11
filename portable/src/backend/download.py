import os
import time
import zipfile
import requests
import shutil
from tqdm import tqdm


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


def download_model(download_path: str, download_link: str, retry_count: int = 2, retry_delay: int = 5) -> bool:
    headers = {
        'Accept': '"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
        'User-Agent': 'wunjo'
    }

    for i in range(retry_count + 1):
        try:
            response = requests.get(download_link, headers=headers, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            response.raise_for_status()
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
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
