from typing import Union

from tps.modules import Replacer
from tps.utils import prob2bool
from tps.symbols import accent

"""
If you need to extend the Emphasizer functionality with
language-specific rules, just add a new descendant class.
"""

class Emphasizer(Replacer):
    def __init__(self, dict_source: Union[str, tuple, list, dict]=None, prefer_user: bool=True):
        """
        Base emphasizer with common functionality for all languages.

        :param dict_source: Union[str, tuple, list, dict]
            Source of dictionary that contains stress pairs such as {'hello': 'hell+o'}.
            Options:
                * str - path to file.
                    The file extension must explicitly show its format in case of json and yaml files.
                    In other cases, user must set the format himself (see below).
                * tuple, list - (path, format)
                    path - path to the dictionary file
                    format - format of the dictionary file (see tps.utils.load_dict function)
                * dict - just a dict
        :param prefer_user: bool
            If true, words with stress tokens set by user will be passed as is
        """
        super().__init__(dict_source, "Emphasizer")
        self.prefer_user = prefer_user


    def process(self, string: str, **kwargs) -> str:
        """
        Splits passed string to tokens and convert each to stressed one if it presents in dictionary.
        Keep it mind, that tokenization is simple here and it's better to pass normalized string.

        :param string: str
            Your text.
        :param kwargs:
            * mask_stress: Union[bool, float]
                Whether to mask each token.
                If float, then masking probability will be computed for each token independently.

        :return: str
        """
        mask = kwargs.get("mask_stress", False)
        return super().process(string, mask=mask)


    def _process_token(self, token, mask):
        if prob2bool(mask):
            return token.replace(accent, "")

        stress_exists = token.find(accent) != -1
        if stress_exists and self.prefer_user:
            return token

        token = self.entries.get(token, token)

        return token