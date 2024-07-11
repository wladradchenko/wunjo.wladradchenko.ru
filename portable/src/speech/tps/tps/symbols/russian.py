_graphemes = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
GRAPHEMES_RU = list(_graphemes)

RU_SET = GRAPHEMES_RU

# Voice clone synthesis symbols
_characters_ru = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890!\'(),-.:;?+ '
_pad = "_"
_eos = "~"

RU_VOICE_CLONE_SYMBOLS = [_pad, _eos] + list(_characters_ru)