"""
Based on https://github.com/keithito/tacotron
"""

import re
from unidecode import unidecode

from .numbs import normalize_numbers


_punctuation_quote_1 = re.compile(r"(\b «)(?=[A-ZА-Я](?:\S*[^»] ){2,}(?:\S+»|\S+$))")  # поиск ёлочек, открывающих фразу
_punctuation_quote_2 = re.compile(r"(\b „)(?=[A-ZА-Я](?:\S*[^“] ){2,}(?:\S+“|\S+$))")  # аналогичное для кавычек

_hyphen = re.compile(r"\b-\b")
_star = re.compile(r"\b\#\b")

_punctuation_phrase_1 = re.compile(r"([?!.]+ - (?=[А-Я]))")  # конец реплики (строки вида '? - ')
_punctuation_phrase_2 = re.compile(", - ")  # строки вида ', - ', т.е. разрыв реплики
_punctuation_signs_1 = re.compile("(\|[?!.]|^[?!.])") # знаки препинания в начале строки
_punctuation_phrase_3 = re.compile("(\| |^ |\|- |^- )")  # пробелы или дефисы в начале строки

_punctuation_garbage = re.compile(r"(\.\b|\b-|-\b|“|”|„|«|»)")  # ниочёмные дефисы и всякий шлак вроде кавычек
_punctuation_bracket_1 = re.compile(r"\([^)]+$")  # поиск одинокой открывающей скобки
_punctuation_bracket_2 = re.compile(r"^[^(]+\)(?=.+$)") # поиск одинокой закрывающей скобки

_punctuation_colon = re.compile(r"\b: (?=\S+)") # двоеточие с последующим пояснением (не протестировано как следует)
_punctuation_signs_2 = re.compile("[?.!]{2,}") # двойные знаки препинания
_punctuation_signs_3 = re.compile("(?<=\W)[А-Я]{1}\.(?= [А-Я+])") # сокращение имён
_punctuation_signs_4 = re.compile("[, \n]+\.$") # неправильно стоящая точка в конце предложения
_punctuation_signs_5 = re.compile(r"\.(?=[А-Яа-я+])") # точка в начале слова

# Regular expression matching whitespace:
_whitespace_1 = re.compile(r"[ \t]+")
_whitespace_2 = re.compile(r"\n+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations_en = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def apply_regexp(regexp, line, replacement: callable):
    found = regexp.search(line)

    while found:
        elem = found.group()
        line = regexp.sub(replacement(elem), line, 1)
        found = regexp.search(line)

    return line


def expand_abbreviations_en(text):
    for regex, replacement in _abbreviations_en:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    text = _whitespace_1.sub(" ", text)
    text = _whitespace_2.sub("", text)
    return text


def convert_to_ascii(text):
    return unidecode(text)


def expand_numbers(text):
    return normalize_numbers(text)


def punctuation_cleaners(text):
    text = text.replace("–", "-")
    text = text.replace("‑", "-")
    text = text.replace("…", ".")

    text = apply_regexp(_hyphen, text, lambda elem: "#")  # экраниурем слова с дефисами

    text = apply_regexp(_punctuation_phrase_1, text, lambda elem: elem[0] + " ")
    text = apply_regexp(_punctuation_phrase_2, text, lambda elem: elem[1:])
    text = apply_regexp(_punctuation_phrase_3, text, lambda elem: "")

    text = apply_regexp(_punctuation_bracket_1, text, lambda elem: elem[1:])
    text = apply_regexp(_punctuation_bracket_2, text, lambda elem: elem[:-1])

    text = apply_regexp(_punctuation_signs_3, text, lambda elem: elem[0])
    text = apply_regexp(_punctuation_signs_5, text, lambda elem: "")

    text = apply_regexp(_star, text, lambda elem: "-")  # восстанавливаем дефисы
    text = text.replace(" - ", " — ")

    return text


def invalid_charset_cleaner(text, charset_re):
    return charset_re.sub("", text)


def light_punctuation_cleaners(text):
    text = text.strip()
    text = text.replace(" - ", " — ")
    for s in ['"', '“', '”', '„', '«', '»', '\'']:
        text = text.replace(s, "")
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations_en(text)
    text = collapse_whitespace(text)
    return text