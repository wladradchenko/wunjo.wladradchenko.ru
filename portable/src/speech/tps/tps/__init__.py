import os
from backend.folders import RTVC_VOICE_FOLDER
from backend.download import download_model, unzip, check_download_size

os.environ["NLTK_DATA"] = os.path.join(RTVC_VOICE_FOLDER, "nltk_data")
tokenizers_punkt_path = os.path.join(RTVC_VOICE_FOLDER, "nltk_data", "tokenizers")

punkt_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"

try:
    if not os.path.exists(tokenizers_punkt_path):
        os.makedirs(tokenizers_punkt_path)
    if not os.path.exists(os.path.join(tokenizers_punkt_path, "punkt")):
        download_model(os.path.join(tokenizers_punkt_path, "punkt.zip"), punkt_url)
        unzip(os.path.join(tokenizers_punkt_path, 'punkt.zip'), tokenizers_punkt_path)
    else:
        check_download_size(os.path.join(tokenizers_punkt_path, "punkt.zip"), punkt_url)
except Exception as err:
    print(f"Error during download NLTK {err}")


from speech.tps.tps.handler import Handler, get_symbols_length
from speech.tps.tps.utils import cleaners, load_dict, save_dict, prob2bool, split_to_tokens
from speech.tps.tps.modules import ssml