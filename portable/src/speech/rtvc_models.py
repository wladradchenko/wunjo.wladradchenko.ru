import os
import sys
import json
import requests
import numpy as np
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.encoder.inference import VoiceCloneEncoder
from speech.rtvc.encoder.audio import preprocess_wav   # We want to expose this function from here
from speech.rtvc.synthesizer.inference import Synthesizer
from speech.rtvc.vocoder.inference import VoiceCloneVocoder
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
        print("Not internet connection to get clone voice models")
    finally:
        if not os.path.isfile(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json')):
            rtvc_models = {}
        else:
            with open(os.path.join(RTVC_VOICE_FOLDER, 'rtvc.json'), 'r', encoding="utf8") as file:
                rtvc_models = json.load(file)
    return rtvc_models


# get rtvc models config
rtvc_models_config = get_config_voice()


def inspect_rtvc_model(rtvc_model: str, rtvc_model_url: str) -> str:
    """
    Inspect download model for rtvc
    :param rtvc_model:
    :param rtvc_model_url:
    :return:
    """
    if not os.path.exists(os.path.dirname(rtvc_model)):
        os.makedirs(os.path.dirname(rtvc_model))
    if not os.path.exists(rtvc_model):
        download_model(rtvc_model, rtvc_model_url)
    else:
        check_download_size(rtvc_model, rtvc_model_url)
    return rtvc_model


def load_rtvc_encoder(lang: str, device: str):
    """
    Load Real Time Voice Clone encoder
    :param lang: encode lang
    :param device: cuda or cpu
    :return: encoder
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        language_dict = rtvc_models_config.get("en")
    encoder_url = language_dict.get("encoder")
    encoder_path = os.path.join(RTVC_VOICE_FOLDER, lang, "encoder.pt")
    model_path = inspect_rtvc_model(encoder_path, encoder_url)
    if not os.path.exists(model_path):
        raise f"Model {encoder_path} not found. Check you internet connection to download"
    encoder = VoiceCloneEncoder()
    encoder.load_model(model_path, device)
    return encoder


def load_rtvc_synthesizer(lang: str, device: str):
    """
    Load Real Time Voice Clone synthesizer
    :param lang: encode lang
    :param device: cuda or cpu
    :return: encoder
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        language_dict = rtvc_models_config.get("en")
    synthesizer_url = language_dict.get("synthesizer")
    synthesizer_path = os.path.join(RTVC_VOICE_FOLDER, lang, "synthesizer.pt")
    model_path = inspect_rtvc_model(synthesizer_path, synthesizer_url)
    if not os.path.exists(model_path):
        raise f"Model {synthesizer_path} not found. Check you internet connection to download"
    return Synthesizer(model_fpath=model_path, device=device, charset=lang)


def load_rtvc_vocoder(lang: str, device: str):
    """
    Load Real Time Voice Clone vocoder
    :param lang: encode lang
    :param device: cuda or cpu
    :return: encoder
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        language_dict = rtvc_models_config.get("en")
    vocoder_url = language_dict.get("vocoder")
    vocoder_path = os.path.join(RTVC_VOICE_FOLDER, lang, "vocoder.pt")
    model_path = inspect_rtvc_model(vocoder_path, vocoder_url)
    if not os.path.exists(model_path):
        raise f"Model {vocoder_path} not found. Check you internet connection to download"
    vocoder = VoiceCloneVocoder()
    vocoder.load_model(model_path, device=device)
    return vocoder


def load_rtvc(lang: str):
    """

    :param lang:
    :return:
    """
    cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if torch.cuda.is_available() and not cpu:
        print("Processing will run on GPU")
        device = "cuda"
    else:
        print("Processing will run on CPU")
        device = "cpu"

    print("Load RTVC encoder")
    encoder = load_rtvc_encoder(lang, device)
    print("Load RTVC synthesizer")
    synthesizer = load_rtvc_synthesizer(lang, device)
    print("Load RTVC vocoder")
    vocoder = load_rtvc_vocoder(lang, device)

    return encoder, synthesizer, vocoder


def get_text_from_audio():
    pass

def translate_text_from_audio():
    pass


def clone_voice_rtvc(audio_file, text, encoder, synthesizer, vocoder, save_folder):
    """

    :param audio_file: audio file
    :param text: text to voice
    :param encoder: encoder
    :param synthesizer: synthesizer
    :param vocoder: vocoder
    :param save_folder: folder to save
    :return:
    """
    try:
        original_wav = synthesizer.load_preprocess_wav(str(audio_file))
        print("Loaded audio successfully")
    except Exception as e:
        print(f"Could not load audio file: {e}")
        return None

    try:
        embed = encoder.embed_utterance(original_wav, using_partials=False)
        print("Created the embedding for audio")
    except Exception as e:
        print(f"Could not create embedding for audio: {e}")
        return None

    # Generating the spectrogram and the waveform
    try:
        generated_wavs = synthesizer.synthesize(text=text, embeddings=embed, vocoder=vocoder)
        print("Created the mel spectrogram and waveform")
    except Exception as e:
        print(f"Could not create spectrogram or waveform: {e}\n")
        return None

    for i, generated_wav in enumerate(generated_wavs):
        audio = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        audio = preprocess_wav(fpath_or_wav=audio)
        # Save to disk
        synthesizer.save(audio=audio, path=save_folder, name="rtvc_output_part%02d.wav" % i)
        print(f"\nSaved output as tvc_output_part%02d.wav\n\n" % i)
