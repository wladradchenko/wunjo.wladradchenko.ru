import re
import os
import sys
import inflect

from tps import modules as md
from tps.symbols import valid_symbols_map
from tps.utils import cleaners

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))
from backend.translator import get_translate
sys.path.pop(0)


class Lower(md.Processor):
    def process(self, string: str, **kwargs) -> str:
        return string.lower()


class Number(md.Processor):
    def __init__(self, charset):
        super().__init__()
        self.charset = charset

    def separate_and_convert_numbers(self, text):
        p = inflect.engine()
        separated_text = re.sub(r'(\D)(\d)', r'\1 \2', re.sub(r'(\d)(\D)', r'\1 \2', text))
        separated_text = re.sub(r'\s+\.\s+', '.', separated_text)
        numbers = re.findall(r'(\d+\.\d+|\d+)', separated_text)
        return {number: p.number_to_words(number) for number in numbers}

    def process(self, string: str, **kwargs) -> str:
        separated_numbers = self.separate_and_convert_numbers(string)
        for number_val, number_word in separated_numbers.items():
            translated_number = get_translate(number_word, self.charset)
            string = string.replace(number_val, translated_number)
        return string


class Cleaner(md.Processor):
    def __init__(self, charset):
        super().__init__()
        self.charset = charset
        self._invalid_charset = re.compile(
            "[^{}]".format(
                "".join(sorted(set(valid_symbols_map[self.charset])))
            )
        )

    def process(self, string: str, **kwargs) -> str:
        string = cleaners.invalid_charset_cleaner(string, self._invalid_charset)
        string = cleaners.collapse_whitespace(string)  # need to clean multiple white spaces that have appeared

        return string