import os
from backend.folders import SETTING_FOLDER

os.environ["NLTK_DATA"] = os.path.join(SETTING_FOLDER, "nltk_data")

from speech.tps.tps.handler import Handler, get_symbols_length
from speech.tps.tps.utils import cleaners, load_dict, save_dict, prob2bool, split_to_tokens
from speech.tps.tps.modules import ssml