import os
import time
import zipfile
import requests
from tqdm import tqdm


def unzip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def download_model(download_path: str, download_link: str, retry_count: int = 2, retry_delay: int = 5) -> bool:
    for i in range(retry_count + 1):
        try:
            response = requests.get(download_link, stream=True)
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
                print('Download failed: network error')
                continue
            else:
                print('Download finished')
                return True
        except requests.exceptions.RequestException:
            if i == retry_count:
                os.remove(download_path)  # delete not finished file
                raise "[Error] Internet connection is finished!"
            time.sleep(retry_delay)


def check_download_size(download_path: str, download_link: str) -> bool:
    response = requests.get(download_link, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if not os.path.exists(download_path):
        return download_model(download_path, download_link)

    downloaded_size = os.path.getsize(download_path)
    if downloaded_size == total_size:
        print(download_path, 'File is correct')
        return True
    print(download_path, "File is fall, re-download ones")
    os.remove(download_path)  # delete not finished file

    return download_model(download_path, download_link)