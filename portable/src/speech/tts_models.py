import json
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.tts.synthesizer import Synthesizer, set_logger, _load_text_handler
from backend.folders import MEDIA_FOLDER, AVATAR_FOLDER, VOICE_FOLDER, CUSTOM_VOICE_FOLDER
from backend.download import download_model, check_download_size
from backend.config import get_tts_config, get_custom_tts_config, cover_windows_media_path_from_str, SUB_CUSTOM_VOICE

sys.path.pop(0)


def create_voice_instruction():
    # File path
    instruction_file = os.path.join(CUSTOM_VOICE_FOLDER, "INSTRUCTION.md")
    # Instruction content
    content = """
    ## Voice Instruction Guide

    To add a custom voice, follow the instructions below:

    1. **Setup Voice Directory**: 
        - Create a directory inside `user_trained_voice` with the desired voice name. For instance, if the voice is named "user", the directory should be `user_trained_voice/user`.

    2. **Add WaveGlow Model**:
        - Place the WaveGlow model in the directory `user_trained_voice/user/waveglow.pt`.
        - Update the settings with the path to WaveGlow as `user_trained_voice/custom_user.json`.

    3. **Add Tacotron2 Model**:
        - Place the Tacotron2 model in the directory `user_trained_voice/user/checkpoint_tacotron2`.
        - Update the settings with the path to Tacotron2 as `user_trained_voice/custom_user.json`.

    ### Important Notes:
    - Ensure the Tacotron2 checkpoint model contains the keys `hparams` and `state_dict`. 
    - If your model lacks these keys, consider downloading the advanced extension and training your Tacotron2 model using this extension.
    - If you're unsure about the parameter values of your models, you can reference the example settings provided below.

    ### Example Settings:

    ```
    {
    "User": {
        "avatar_download": "",
        "checkpoint_download":  "",
        "waveglow_download":  "",
        "voice_control_cfg": {
          "psola": {
            "max_hz": 1130,
            "min_hz": 60,
            "analysis_win_ms": 40,
            "max_change": 1.455,
            "min_change": 0.595
          },
          "phase": {
            "nfft": 256,
            "hop": 64
          }
        },
        "user_dict": null,
        "text_handler": {
          "config": "ru",
          "out_max_length": 200
        },
        "modules": {
          "engine": "tacotron2",
          "vocoder": "waveglow"
        },
        "engine": {
          "tacotron2": {
            "model_path": "user_trained_voice/user/checkpoint_tacotron",
            "hparams_path": null,
            "options": {
              "steps_per_symbol": 10,
              "gate_threshold": 0.5
            }
          }
        },
        "vocoder": {
          "waveglow": {
            "model_path": "user_trained_voice/user/waveglow.pt",
            "options": {
              "sigma": 0.666,
              "strength": 0.1
            }
          }
        }
      }
    }
    ```
    """

    # Check if file exists
    if not os.path.exists(instruction_file):
        # If the file doesn't exist, create and write to it
        with open(instruction_file, 'w') as f:
            f.write(content)
    else:
        print(f"{instruction_file} already exists!")

    # create default.json
    custom_voice_file = os.path.join(CUSTOM_VOICE_FOLDER, "custom_user.json")
    # Check if file exists
    if not os.path.exists(custom_voice_file):
        # If the file doesn't exist, create and write to it
        with open(custom_voice_file, 'w') as f:
            json.dump({}, f)
    else:
        print(f"{custom_voice_file} already exists!")


# Call the function
create_voice_instruction()
# app voices
file_voice_config, voice_names = get_tts_config()
set_logger(**file_voice_config["general"].pop("logging"))
# user voices
file_custom_voice_config, custom_voice_names = get_custom_tts_config()


def inspect_model(voice_name: str):
    if SUB_CUSTOM_VOICE not in voice_name:  # if not custom voice when check model
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
    local_config = {**file_voice_config.copy(), **file_custom_voice_config.copy()}
    prev_voice_names = list(models.keys())
    [models.pop(voice_name) for voice_name in prev_voice_names if voice_name not in user_voice_names]
    if sorted(user_voice_names) == sorted(list(models.keys())):
        # if models and user request was equal or after pop the model start to equal to user request
        # it means not need load need model
        print("Loaded TTS models already is loaded")
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
                text_handler=_load_text_handler(voice_config["text_handler"]),
                engine=Synthesizer.module_from_config(voice_config, "engine", "tacotron2", voice_synthesizer_first.device),
                vocoder=Synthesizer.module_from_config(voice_config, "vocoder", "waveglow", voice_synthesizer_first.device),
                sample_rate=voice_synthesizer_first.sample_rate,
                device=voice_synthesizer_first.device,
                pause_type=voice_synthesizer_first.pause_type,
                voice_control_cfg=voice_config["voice_control_cfg"],
                user_dict=voice_config["user_dict"]
            )
            models[voice_name] = voice_synthesizer_other

    print("Update loaded TTS models")
    return models
