import math
from typing import Union


class Text:
    _valid_prosody = ["pitch", "rate", "volume"]

    def __init__(self, text, pitch=1.0, rate=1.0, volume=1.0):
        self.value = text

        self.pitch = parse_pitch(pitch)
        self.rate = parse_rate(rate)
        self.volume = parse_volume(volume)


    def __str__(self):
        return self.value


    def update_value(self, new_text):
        self.value = new_text


    def update_prosody(self, **kwargs):
        for key, value in kwargs.items():
            if key not in Text._valid_prosody:
                continue
            value = _prosodies_parsers[key](value)
            self.__setattr__(key, value)


    def __add__(self, other):
        text = self.value + other.value
        instance = Text(text).inherit(self)

        return instance


    def inherit(self, instance):
        self.pitch = instance.pitch
        self.rate = instance.rate
        self.volume = instance.volume

        return self


    @property
    def is_empty(self):
        return len(self.value.replace(" ", "")) == 0


class Pause:
    def __init__(self, time=None, strength=None, type_="custom"):
        assert not (time is None and strength is None)
        self.type = type_
        self.milliseconds = parse_duration(time if time is not None else strength)


    def __str__(self):
        return "<Pause.{}: {}ms>".format(self.type, self.milliseconds)


    @property
    def seconds(self):
        return self.milliseconds / 1000


    def samples(self, samplerate):
        return int(self.seconds * samplerate)


    @classmethod
    def paragraph(cls):
        return Pause(time=750, type_="paragraph")


    @classmethod
    def eos(cls):
        return Pause(time=500, type_="eos")


    @classmethod
    def semicolon(cls):
        return Pause(time=250, type_="semicolon")


    @classmethod
    def colon(cls):
        return Pause(time=150, type_="colon")


    @classmethod
    def comma(cls):
        return Pause(time=100, type_="comma")


    @classmethod
    def space(cls):
        return Pause(time=50, type_="space")


_pause_map = {
    "none": 0,
    "x-weak": 50,
    "weak": 100,
    "medium": 250,
    "strong": 500,
    "x-strong": 750
}


def parse_duration(value: Union[str, float, int]) -> int:
    if isinstance(value, str):
        if value.endswith("ms"):
            value = int(value.replace("ms", ""))
        elif value.endswith("s"):
            value = int(float(value.replace("s", "")) * 1000)
        elif value in _pause_map:
            value = _pause_map[value]
        else:
            raise ValueError
    elif isinstance(value, int):
        pass
    elif isinstance(value, float):
        value = int(value * 1000)
    else:
        raise TypeError

    return value


_pitch_map = {
    "default": 1.0,
    "x-weak": 0.75,
    "weak": 0.9,
    "medium": 1.0,
    "strong": 1.25,
    "x-strong": 1.5
}


def parse_pitch(value: Union[str, float, int]) -> float:
    if isinstance(value, str):
        if value.endswith("%"):
            factor = float(value.replace("%", "")) # -15.2%
            factor = 1 + factor / 100
        elif value.endswith("st"):
            factor = float(value.replace("st", "")) # 0.5st
            factor = (2 ** (factor / 12))
        elif value in _pitch_map:
            factor = _pitch_map[value]
        else:
            raise ValueError
    elif isinstance(value, (float, int)):
        factor = value
    else:
        raise TypeError

    return factor


_rate_map = {
    "default": 1.0,
    "x-slow": 0.5,
    "slow": 0.75,
    "medium": 1.0,
    "fast": 1.5,
    "x-fast": 2.0
}


def parse_rate(value: Union[str, float, int]) -> float:
    if isinstance(value, str):
        if value.endswith("%"):
            factor = float(value.replace("%", "")) # -15.2%
            factor = 1 + factor / 100
        elif value in _rate_map:
            factor = _rate_map[value]
        else:
            raise ValueError
    elif isinstance(value, (float, int)):
        factor = value
    else:
        raise TypeError

    return factor


_volume_map = {
    "silent": -math.inf,
    "x-soft": -12,
    "soft": 6,
    "medium": 0,
    "loud": 6,
    "x-loud": 12
}


def parse_volume(value: Union[str, int, float]) -> Union[int, float]:
    if isinstance(value, str):
        if value.endswith("dB"):
            factor = float(value.replace("dB", "")) # -6.0dB
        elif value in _volume_map:
            factor = _volume_map[value]
        else:
            raise ValueError
    elif isinstance(value, (float, int)):
        factor = value
    else:
        raise TypeError

    return factor


_prosodies_parsers = {
    "pitch": parse_pitch,
    "rate": parse_rate,
    "volume": parse_volume
}