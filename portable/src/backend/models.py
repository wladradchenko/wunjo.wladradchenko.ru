import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root_path}/backend")

from backend.tts.synthesizer import Synthesizer, set_logger

sys.path.pop(0)

config = Synthesizer.load_config("config.yaml")

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

config = update_dict_paths(config, "data/", f"{os.path.dirname(os.path.abspath(__file__))}/data/")

set_logger(**config["general"].pop("logging"))


keys = list(config.keys())
voice_synthesizer_first = None
models = {}
for voice_name in keys:
    if voice_name in ['general']:
        continue

    if voice_synthesizer_first is None:
        voice_synthesizer_first = Synthesizer.from_config(config, name=voice_name)
        models[voice_name] = voice_synthesizer_first
    else:
        voice_config = config[voice_name]
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
