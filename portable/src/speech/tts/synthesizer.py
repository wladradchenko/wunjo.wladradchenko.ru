import os
import sys
import time
import yaml

import numpy as np
import soundfile
from loguru import logger

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

import wrappers as bw
from utils.async_utils import BackgroundGenerator
from utils.voice_control import shift_pitch, stretch_wave
from tps import Handler, load_dict, save_dict, ssml
from tps.types import Charset

sys.path.pop(0)


def uniqid():
    from time import time
    return hex(int(time() * 1e7))[2:]


def set_logger(log_level="INFO", log_file=None):
    logger.remove()
    logger.add(sys.stdout, level=log_level)
    if log_file is not None:
        logger.add(log_file, encoding="utf8")


_modules_dict = {
    "tacotron2": bw.Tacotron2Wrapper,
    "waveglow": bw.WaveglowWrapper
}


class Synthesizer:
    def __init__(self, name, text_handler, engine, vocoder, sample_rate, device="cuda", pause_type="silence",
                voice_control_cfg=None, user_dict=None):
        self.name = name

        self.text_handler = text_handler
        self.engine = engine
        self.vocoder = vocoder

        self.sample_rate = sample_rate

        self.device = device

        self.pause_type = pause_type
        self.voice_control_cfg = self.load_config(voice_control_cfg)

        self.user_dict = None
        self._dict_source = None
        self.load_user_dict(user_dict)

        assert self.text_handler.charset == self.engine.charset

        logger.info("Synthesizer {} is ready".format(name))


    def synthesize(self, text, **kwargs):
        audio_list = list(self.text_to_audio_gen(text, **kwargs))
        audio = np.concatenate(audio_list)

        return audio


    def generate(self, text, **kwargs):
        return BackgroundGenerator(self.text_to_audio_gen(text, **kwargs))


    def text_to_audio_gen(self, text, **kwargs):
        logger.info(text)
        logger.debug("kwargs: {}".format(kwargs))

        mask_stress = kwargs.pop("mask_stress", False)
        mask_phonemes = kwargs.pop("mask_phonemes", False)
        pitch = kwargs.pop("pitch", 1.0)
        rate = kwargs.pop("rate", 1.0)
        volume = kwargs.pop("volume", 0)

        cleaners = kwargs.pop("cleaners", tuple())
        if "light_punctuation_cleaners" not in cleaners:
            cleaners = ("light_punctuation_cleaners",) + cleaners

        if text.startswith("<speak>") and text.endswith("</speak>"):
            logger.debug("SSML text is detected. Starting the parsing process...")
            sequence = ssml.parse_ssml_text(text)
            logger.debug("Done")
        else:
            sequence = self.text_handler.split_to_sentences(text, True, self.text_handler.language)
            sequence = [
                ssml.Text(elem, pitch, rate, volume) if not isinstance(elem, ssml.Pause) else elem for elem in sequence
            ]

        sequence_generator = BackgroundGenerator(
            self._sequence_to_sequence_gen(sequence, cleaners, mask_stress, mask_phonemes)
        )

        return self._sequence_to_audio_gen(sequence_generator, **kwargs)


    def _sequence_to_sequence_gen(self, sequence, cleaners, mask_stress, mask_phonemes):
        for element in sequence:
            if not isinstance(element, ssml.Pause):
                sentence = element.value

                sentence = self.text_handler.process(
                    string=sentence,
                    cleaners=cleaners,
                    user_dict=self.user_dict,
                    mask_stress=mask_stress,
                    mask_phonemes=mask_phonemes
                )

                if self.text_handler.out_max_length is not None:
                    _units = self.text_handler.split_to_units(sentence, self.text_handler.out_max_length, True)

                    for unit in _units:
                        yield ssml.Text(unit).inherit(element) if not isinstance(unit, ssml.Pause) else unit
                else:
                    element.update_value(sentence)
                    yield element
            else:
                yield element


    def _sequence_to_audio_gen(self, sequence, **kwargs):
        for unit in sequence:
            if isinstance(unit, ssml.Pause):
                audio = generate_pause(unit.samples(self.sample_rate), ptype=self.pause_type)
            else:
                logger.debug(unit)
                unit_value = self.text_handler.check_eos(unit.value)
                unit_value = self.text_handler.text2vec(unit_value)

                spectrogram = self.engine(unit_value, **kwargs)
                audio = self.vocoder(spectrogram)
                audio = self.vocoder.denoise(audio)

                audio = self.post_process(audio, unit.pitch, unit.rate, unit.volume)

            yield audio

        self.vocoder.clear_cache()


    def post_process(self, audio, pitch=1.0, rate=1.0, volume=0):
        audio = audio.squeeze()

        if pitch != 1.0 or rate != 1.0 or volume != 0:
            if pitch != 1.0:
                audio = self.change_pitch(audio, pitch)
            if rate != 1.0:
                audio = self.change_speed(audio, rate)
            if volume != 0:
                audio = self.change_volume(audio, volume)

            audio = self.vocoder.denoise(audio)
            audio = audio.squeeze()

        return audio


    def save(self, audio, path, prefix=None):
        os.makedirs(path, exist_ok=True)
        prefix = [prefix] if prefix is not None else []

        waves_format = ".wav"
        name = "_".join(prefix + [self.name, uniqid(), time.strftime("%Y-%m-%d_%H-%M")]) + waves_format

        file_path = os.path.join(path, name)
        soundfile.write(file_path, audio, self.sample_rate)

        logger.info("Audio was saved as {}".format(os.path.abspath(file_path)))

        return file_path


    def change_speed(self, audio, factor):
        if factor > 2 or factor < 0.5:
            print("ERROR: speed factor is out of range [0.5, 2.0] -- original signal returned")
            return audio

        params = self.voice_control_cfg["phase"]

        return stretch_wave(audio, factor, params)


    def change_pitch(self, audio, factor):
        if factor > 1.5 or factor < 0.75:
            print("ERROR: tone factor is out of range [0.75, 1.5] -- original signal returned")
            return audio

        params = self.voice_control_cfg["psola"]

        return shift_pitch(audio, self.sample_rate, factor, params)


    @staticmethod
    def change_volume(audio, dB):
        return audio * (10 ** (dB / 20))


    def load_user_dict(self, user_dict):
        data_dir = os.path.join(os.path.expanduser('~'), '.wunjo')
        if isinstance(user_dict, dict) or user_dict is None:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                logger.info("Data folder was created along the path {}".format(os.path.abspath(data_dir)))
            self._dict_source = os.path.join(data_dir, "{}_user_dict.json".format(self.name))
        else:
            self._dict_source = user_dict
        assert self._dict_source.endswith((".json", ".yaml"))

        self.user_dict = load_dict(user_dict)
        logger.info("User dictionary has been loaded")


    def get_user_dict(self):
        logger.info("Request for the user dictionary was received")
        return self.user_dict


    def update_user_dict(self, new_dict):
        self.user_dict.update(new_dict)
        logger.info("User dictionary has been updated")

        save_dict(self.user_dict, self._dict_source)
        logger.info("User dictionary has been saved")


    def replace_user_dict(self, new_dict):
        self.user_dict = new_dict
        logger.info("User dictionary has been replaced")

        save_dict(self.user_dict, self._dict_source)
        logger.info("User dictionary has been saved")


    @classmethod
    def from_config(cls, config, name):
        if isinstance(config, str):
            logger.debug("Loading synthesizer from config file {}".format(os.path.abspath(config)))

        config = cls.load_config(config)
        params = config["general"]

        if "logging" in params:
            set_logger(**params.pop("logging"))

        params["name"] = name
        device = params["device"]
        assert device is not None

        modules_config = config.pop(name)
        params["voice_control_cfg"] = modules_config["voice_control_cfg"]
        params["user_dict"] = modules_config["user_dict"]

        params["text_handler"] = _load_text_handler(modules_config["text_handler"])

        chosen = modules_config["modules"]

        for mtype, mname in chosen.items():
            params[mtype] = Synthesizer.module_from_config(modules_config, mtype, mname, device)

        return Synthesizer(**params)


    @staticmethod
    def module_from_config(modules_config, mtype, mname, device):
        logger.info("Loading {} module".format(mname))

        module_config = modules_config[mtype][mname]
        module_config["device"] = device

        for key, value in module_config.pop("options", {}).items():
            if value is not None:
                module_config[key] = value

        return _modules_dict[mname](**module_config)


    @staticmethod
    def load_config(config_source):
        if isinstance(config_source, dict):
            return config_source
        else:
            raise TypeError

def generate_pause(duration, eps=1e-4, ptype='white_noise'):
    if ptype == 'silence':
        pause = np.zeros((duration, ))
    elif ptype == 'white_noise':
        pause = np.random.random((duration, )) * eps
    else:
        raise TypeError

    return pause.astype(np.float32)


def _load_text_handler(config_dict):
    logger.info("Loading text handler")

    out_max_length = config_dict["out_max_length"]

    config = config_dict["config"]
    assert config is not None

    if config in Charset._member_names_:
        config_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/tps/data"
        handler = Handler.from_charset(config, data_dir=config_path, out_max_length=out_max_length, silent=True)
    else:
        handler_config = Synthesizer.load_config(config)
        handler_config["handler"]["out_max_length"] = out_max_length

        handler = Handler.from_config(handler_config)

    return handler