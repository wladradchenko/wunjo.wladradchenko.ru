import json
import os
import sys
import requests

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root_path}/backend")

from speech.tts.synthesizer import Synthesizer, set_logger
from backend.folders import MEDIA_FOLDER, AVATAR_FOLDER, VOICE_FOLDER
from backend.download import download_model, check_download_size

sys.path.pop(0)

VOICE_JSON_URL = "https://wladradchenko.ru/static/wunjo.wladradchenko.ru/voice.json"

def update_dict_paths(d, old_path, new_path):
    for k, v in d.items():
        if isinstance(v, dict):
            update_dict_paths(v, old_path, new_path)
        elif isinstance(v, str) and old_path in v:
            if not os.path.dirname(os.path.abspath(__file__)) in old_path:
                d[k] = v.replace(old_path, new_path)
            else:
                d[k] = v
    return d

def _create_config_voice(voices: dict) -> dict:
    general = {
        "general": {
            "device": "cuda",
            "pause_type": "silence",
            "sample_rate": 22050,

            "logging": {
                "log_level": "INFO",
                "log_file": None
            }
        }
    }
    return {**general, **voices}


def _get_avatars(voices: dict) -> None:
    if not os.path.exists(AVATAR_FOLDER):
        os.makedirs(AVATAR_FOLDER)

    for avatar in voices.keys():
        # Check if the avatar file exists
        if not os.path.exists(os.path.join(AVATAR_FOLDER, f"{avatar}.png")):
            # If it doesn't exist, download it from the URL
            request_png = requests.get(voices[avatar]["avatar_download"])
            with open(os.path.join(AVATAR_FOLDER, f"{avatar}.png"), 'wb') as file:
                file.write(request_png.content)


def get_config_voice() -> dict:
    try:
        response = requests.get(VOICE_JSON_URL)
        with open(os.path.join(VOICE_FOLDER, 'voice.json'), 'wb') as file:
            file.write(response.content)
    except:
        print("Not internet connection")
    finally:
        if not os.path.isfile(os.path.join(VOICE_FOLDER, 'voice.json')):
            voices={}
        else:
            with open(os.path.join(VOICE_FOLDER, 'voice.json'), 'r', encoding="utf8") as file:
                voices = json.load(file)

    _get_avatars(voices=voices)

    if voices.get("Unknown") is not None:
       voices.pop("Unknown")

    return _create_config_voice(voices=voices), list(voices.keys())


file_voice_config, voice_names = get_config_voice()
# print(config)

file_voice_config = update_dict_paths(file_voice_config, "voice/", f"{MEDIA_FOLDER}/voice/")
set_logger(**file_voice_config["general"].pop("logging"))


def inspect_model(voice_name: str):
    checkpoint_path = file_voice_config[voice_name]["engine"]["tacotron2"]["model_path"]
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        checkpoint_link = file_voice_config[voice_name]["checkpoint_download"]
        download_model(checkpoint_path, checkpoint_link)
    else:
        checkpoint_link = file_voice_config[voice_name]["checkpoint_download"]
        check_download_size(checkpoint_path, checkpoint_link)

    waveglow_path = file_voice_config[voice_name]["vocoder"]["waveglow"]["model_path"]
    if not os.path.exists(os.path.dirname(waveglow_path)):
        os.makedirs(os.path.dirname(waveglow_path))
    if not os.path.exists(waveglow_path):
        waveglow_link = file_voice_config[voice_name]["waveglow_download"]
        download_model(waveglow_path, waveglow_link)
    else:
        waveglow_link = file_voice_config[voice_name]["waveglow_download"]
        check_download_size(waveglow_path, waveglow_link)

def load_voice_models(user_voice_names: list, models: dict):
    local_config = file_voice_config.copy()
    prev_voice_names = list(models.keys())
    [models.pop(voice_name) for voice_name in prev_voice_names if voice_name not in user_voice_names]
    if sorted(user_voice_names) == sorted(list(models.keys())):
        # if models and user request was equal or after pop the model start to equal to user request
        # it means not need load need model
        print("Not need load new voice models")
        return models

    if models == {}:
        first_voice_name = user_voice_names[0]
        inspect_model(first_voice_name)  # inspect model to download
        voice_synthesizer_first = Synthesizer.from_config(local_config, name=first_voice_name)
        models[first_voice_name] = voice_synthesizer_first
    else:
        first_voice_name = list(models.keys())[0]
        voice_synthesizer_first = models[first_voice_name]

    for voice_name in user_voice_names:
        if models.get(voice_name) is None:
            inspect_model(voice_name)  # inspect model to download
            voice_config = local_config[voice_name]
            voice_synthesizer_other = Synthesizer(
                name=voice_name,
                text_handler=voice_synthesizer_first.text_handler,
                engine=Synthesizer.module_from_config(voice_config, "engine", "tacotron2", voice_synthesizer_first.device),
                vocoder=Synthesizer.module_from_config(voice_config, "vocoder", "waveglow", voice_synthesizer_first.device),
                sample_rate=voice_synthesizer_first.sample_rate,
                device=voice_synthesizer_first.device,
                pause_type=voice_synthesizer_first.pause_type,
                voice_control_cfg=voice_config["voice_control_cfg"],
                user_dict=voice_config["user_dict"]
            )
            models[voice_name] = voice_synthesizer_other

    print("New voice models added")
    return models
