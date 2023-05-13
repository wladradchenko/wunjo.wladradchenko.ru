from string import punctuation

from tps.symbols import english as en
from tps.symbols import russian as ru
from tps.types import Charset


dot = '.'
intonation = '!?'
other = "():;"
comma = ','
dash = "—"
space = " "
accent = '+'
hyphen = "-"

separator = "_"
shields = ["{", "}"]

pad = "<pad>"
eos = "~"

tps_punctuation = dot + intonation + other + comma + dash

symbols_ = [pad] + [eos] + list(tps_punctuation + hyphen + space + accent)

symbols_en = symbols_ + en.EN_SET
symbols_en_cmu = symbols_ + en.EN_CMU_SET
symbols_ru = symbols_ + ru.RU_SET

symbols_map = {
    Charset.en: symbols_en,
    Charset.en_cmu: symbols_en_cmu,
    Charset.ru: symbols_ru
}

phoneme_map = {
    Charset.en: [],
    Charset.en_cmu: en.PHONEMES_EN_CMU,
    Charset.ru: []
}

for symb in [accent, hyphen, separator] + shields:
    punctuation = punctuation.replace(symb, "")
punctuation = list(punctuation) + [dash, space, '“', '”', '„', '«', '»']

_shielding = shields + [separator]
valid_symbols_map = {
    Charset.en: symbols_en + _shielding,
    Charset.en_cmu: symbols_en_cmu + _shielding,
    Charset.ru: symbols_ru + _shielding
}

language_map = {
    Charset.en: "english",
    Charset.en_cmu: "english",
    Charset.ru: "russian"
}