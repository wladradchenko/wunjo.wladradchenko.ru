import re

from tps import modules as md
from tps.symbols import valid_symbols_map
from tps.utils import cleaners


class Lower(md.Processor):
    def process(self, string: str, **kwargs) -> str:
        return string.lower()


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