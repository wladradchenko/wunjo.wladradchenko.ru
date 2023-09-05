_graphemes = 'abcdefghijklmnopqrstuvwxyz12340'
GRAPHEMES_ZH = list(_graphemes)

ZH_SET = GRAPHEMES_ZH

# Voice clone synthesis symbols
_characters_zh = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz12340!\'(),-.:;? '
_pad = "_"
_eos = "~"

ZH_VOICE_CLONE_SYMBOLS = [_pad, _eos] + list(_characters_zh) #+ _arpabet