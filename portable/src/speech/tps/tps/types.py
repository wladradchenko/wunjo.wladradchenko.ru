from enum import Enum


class BasedOn(str, Enum):
    rule = "rule_based"
    nn = "nn_based"
    hybrid = "hybrid"


class Charset(str, Enum):
    en = "en"
    en_cmu = "en_cmu"
    ru = "ru"


class Module(str, Enum):
    emphasizer = "emphasizer"
    phonetizer = "phonetizer"
    replacer = "replacer"


class SSMLTag(str, Enum):
    speak = "speak"
    break_ = "break"
    p = "p"
    s = "s"
    sub = "sub"
    prosody = "prosody"


    @classmethod
    def nested(cls, value):
        return value in [cls.speak, cls.p, cls.s, cls.prosody]