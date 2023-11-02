import os
import re
import sys
import json
import requests
import itertools
import numpy as np
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.encoder.inference import VoiceCloneEncoder
from speech.rtvc.encoder.hubert import HubertEmbedder
from speech.rtvc.encoder.audio import preprocess_wav   # We want to expose this function from here
from speech.rtvc.synthesizer.inference import Synthesizer
from speech.rtvc.synthesizer.utils.signature import DigitalSignature
from speech.rtvc.vocoder.inference import VoiceCloneVocoder
from backend.folders import MEDIA_FOLDER, RTVC_VOICE_FOLDER
from backend.download import download_model, check_download_size, get_nested_url, is_connected
from backend.translator import get_translate

sys.path.pop(0)


RTVC_MODELS_JSON_URL = "https://raw.githubusercontent.com/wladradchenko/wunjo.wladradchenko.ru/main/models/rtvc.json"


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
        # check what is internet access
        is_connected(rtvc_model)
        # download pre-trained models from url
        download_model(rtvc_model, rtvc_model_url)
    else:
        check_download_size(rtvc_model, rtvc_model_url)
    return rtvc_model


def load_rtvc_encoder(lang: str, device: str) -> tuple:
    """
    Load Real Time Voice Clone encoder
    :param lang: encode lang
    :param device: cuda or cpu
    :return: encoder
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        lang = "en"
    encoder_url = get_nested_url(rtvc_models_config, [lang, "encoder"])
    encoder_path = os.path.join(RTVC_VOICE_FOLDER, lang, "encoder.pt")
    encoder_model_path = inspect_rtvc_model(encoder_path, encoder_url)
    if not os.path.exists(encoder_model_path):
        raise Exception(f"Model {encoder_model_path} not found. Check you internet connection to download")
    encoder = VoiceCloneEncoder()
    encoder.load_model(encoder_model_path, device)
    # load hubert
    hubert_url = get_nested_url(rtvc_models_config, ["general", "hubert"])
    hubert_path = os.path.join(RTVC_VOICE_FOLDER, "general", "hubert.pt")
    hubert_model_path = inspect_rtvc_model(hubert_path, hubert_url)
    if not os.path.exists(hubert_model_path):
        raise Exception(f"Model {hubert_model_path} not found. Check you internet connection to download")
    hubert_encoder = HubertEmbedder(hubert_model_path, device)
    return encoder, hubert_encoder


def load_rtvc_synthesizer(lang: str, device: str):
    """
    Load Real Time Voice Clone synthesizer
    :param lang: synthesizer lang
    :param device: cuda or cpu
    :return: synthesizer
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        lang = "en"
    synthesizer_url = get_nested_url(rtvc_models_config, [lang, "synthesizer"])
    synthesizer_path = os.path.join(RTVC_VOICE_FOLDER, lang, "synthesizer.pt")
    model_path = inspect_rtvc_model(synthesizer_path, synthesizer_url)
    if not os.path.exists(model_path):
        raise Exception(f"Model {synthesizer_path} not found. Check you internet connection to download")
    return Synthesizer(model_fpath=model_path, device=device, charset=lang)


def load_rtvc_digital_signature(device: str):
    """
    Load Real Time Voice Clone digital signature
    :param lang: lang
    :param device: cuda or cpu
    :return: encoder
    """
    signature_url = get_nested_url(rtvc_models_config, ["general", "signature"])
    signature_path = os.path.join(RTVC_VOICE_FOLDER, "general", "signature.pt")
    model_path = inspect_rtvc_model(signature_path, signature_url)
    if not os.path.exists(model_path):
        raise Exception(f"Model {signature_path} not found. Check you internet connection to download")
    return DigitalSignature(model_path=model_path, device=device)


def load_rtvc_vocoder(lang: str, device: str):
    """
    Load Real Time Voice Clone vocoder
    :param lang: vocoder lang
    :param device: cuda or cpu
    :return: vocoder
    """
    language_dict = rtvc_models_config.get(lang)
    if language_dict is None:
        lang = "en"
    vocoder_url = get_nested_url(rtvc_models_config, [lang, "vocoder"])
    vocoder_path = os.path.join(RTVC_VOICE_FOLDER, lang, "vocoder.pt")
    model_path = inspect_rtvc_model(vocoder_path, vocoder_url)
    if not os.path.exists(model_path):
        raise Exception(f"Model {vocoder_path} not found. Check you internet connection to download")
    vocoder = VoiceCloneVocoder()
    vocoder.load_model(model_path, device=device)
    return vocoder


def load_rtvc(lang: str):
    """

    :param lang:
    :return:
    """
    use_cpu = False if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else True
    if torch.cuda.is_available() and not use_cpu:
        print("Processing will run on GPU")
        device = "cuda"
    else:
        print("Processing will run on CPU")
        device = "cpu"

    print("Load RTVC encoder")
    encoder = load_rtvc_encoder(lang, device)
    print("Load RTVC synthesizer")
    synthesizer = load_rtvc_synthesizer(lang, device)
    print("Load RTVC signature")
    signature = load_rtvc_digital_signature(device)
    print("Load RTVC vocoder")
    vocoder = load_rtvc_vocoder(lang, device)

    return encoder, synthesizer, signature, vocoder


def get_text_from_audio():
    pass

def translate_text_from_audio():
    pass


def clone_voice_rtvc(audio_file, text, encoders, synthesizer, vocoder, save_folder):
    """

    :param audio_file: audio file
    :param text: text to voice
    :param encoders: encoder and hubert encoder
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
        encoder, hubert_encoder = encoders  # get encoders
        # get embedding
        embed = encoder.embed_utterance(original_wav, using_partials=False)
        # get embedding max and min values by hubert encoder to improve quality
        hubert_embed = hubert_encoder.embed_audio(str(audio_file))
        hubert_embed_mean = np.mean(hubert_embed, axis=0)  # set equal shape with embed.shape
        # Get max and min values from hubert_embed
        hubert_max = np.max(hubert_embed_mean)
        hubert_min = np.min(hubert_embed_mean)
        # Get the indices of the max and min values in embed
        embed_max_index = np.argmax(embed)
        embed_min_index = np.argmin(embed)
        # Set the values in embed to the max and min values from hubert_embed
        embed[embed_max_index] = hubert_max
        embed[embed_min_index] = hubert_min
        # embed denoising zero threshold
        set_zero_thres = 0.03
        embed[embed < set_zero_thres] = 0
        print("Created the embedding for audio")
    except Exception as e:
        print(f"Could not create embedding for audio: {e}")
        return None

    # Generating the spectrogram and the waveform
    try:
        # Load synthesizer for each language to use multilanguage in one text
        text_list = detect_text_language(text)
        device = synthesizer.device
        synthesizer_dict = {synthesizer.text_handler.charset: synthesizer}
        generated_waves = []
        for text_param in text_list:
            lang, text = text_param
            if synthesizer_dict.get(lang) is None:
                synthesizer_dict[lang] = load_rtvc_synthesizer(lang, device)
            # generator of text waves
            generated_waves.append(synthesizer_dict[lang].synthesize(text=text, embeddings=embed, vocoder=vocoder))
        # (1) remove loop and uncommented down line if I will want to delete multilanguage
        # also change itertools.chain(*generated_waves) on generated_waves
        # generated_waves = synthesizer.synthesize(text=text, embeddings=embed, vocoder=vocoder)
        print("Created the mel spectrogram and waveform")
    except Exception as e:
        print(f"Could not create spectrogram or waveform: {e}\n")
        return None

    for i, generated_wav in enumerate(itertools.chain(*generated_waves)):
        audio = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        audio = preprocess_wav(fpath_or_wav=audio)
        # Save to disk
        synthesizer.save(audio=audio, path=save_folder, name="rtvc_output_part%02d.wav" % i)
        print(f"\nSaved output as tvc_output_part%02d.wav\n\n" % i)


def detect_text_language(text: str) -> list:
    """
    Detection text language with Regular Expressions as English, Chinese and Russian
    TODO if will be a more clone voice than improve this method, maybe use with langdetect
    :param text: text
    :return: list of list as lang and text
    """
    # Define patterns for different languages
    patterns = {
        'ru': re.compile(r'[\u0400-\u04FF]+'),
        'zh': re.compile(r'[\u4e00-\u9fff]+'),
        'punct': re.compile(r'[.!?;,:\s]')  # Pattern to match punctuation and spaces
    }

    # Function to determine the language of a word
    def get_language(word):
        for lang, pattern in patterns.items():
            if pattern.search(word):
                return lang
        return 'en'  # Default to English

    # Split the text into words
    words = re.split(r'(\W)', text)

    # List to hold tuples of language and text
    language_segments = []
    current_lang = None
    current_segment = ""

    # Build the language segments list
    for word in words:
        if not word.strip():  # Skip empty strings
            continue

        # Check if word is punctuation, if so, append it to the current segment
        if patterns['punct'].match(word) and current_segment:
            current_segment += word
        else:
            word_lang = get_language(word)
            # If the language changes, append the current segment and start a new one
            if current_lang and word_lang != current_lang:
                language_segments.append([current_lang, current_segment.strip()])
                current_segment = word
            else:
                current_segment += " " + word
            current_lang = word_lang

    # Add the last segment if it exists
    if current_segment:
        language_segments.append([current_lang, current_segment.strip()])

    # translate text on english if not Chinese or Russian
    language_segments = [[lang, get_translate(phrase, lang)] if lang == "en" else [lang, phrase] for lang, phrase in language_segments]

    return language_segments
