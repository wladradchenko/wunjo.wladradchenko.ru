import os
import sys
import json
import requests
import librosa
import numpy as np
import soundfile as sf
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.encoder import inference as encoder
from speech.rtvc.encoder.params_model import model_embedding_size as speaker_embedding_size
from speech.rtvc.synthesizer.inference import Synthesizer
from speech.rtvc.vocoder import inference as vocoder
from backend.folders import MEDIA_FOLDER, RTVC_VOICE_FOLDER
from backend.download import download_model, check_download_size

sys.path.pop(0)


RTVC_MODELS_JSON_URL = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/rtvc.json"


def get_config_voice() -> dict:
    try:
        response = requests.get(RTVC_MODELS_JSON_URL)
        with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'wb') as file:
            file.write(response.content)
    except:
        print("Not internet connection")
    finally:
        if not os.path.isfile(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json')):
            rtvc_models = {}
        else:
            with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'r', encoding="utf8") as file:
                rtvc_models = json.load(file)
    return rtvc_models

def inspect_rtvc_model(rtvc_model: dict):
    """
    Inspect download model for rtvc
    :param rtvc_model:
    :return:
    """
    return

def load_rtvc_encoder(model_path: str):
    """
    Load Real Time Voice Clone encoder
    :param model_path: path to model
    :return: encoder
    """
    return encoder.load_model(model_path)


def load_rtvc_synthesizer(model_path: str):
    """
    Load Real Time Voice Clone synthesizer
    :param model_path: path to model
    :return: synthesizer
    """
    return Synthesizer(model_path)


def load_rtvc_vocoder(model_path: str):
    """
    Load Real Time Voice Clone vocoder
    :param model_path: path to model
    :return: vocoder
    """
    return vocoder.load_model(model_path)
