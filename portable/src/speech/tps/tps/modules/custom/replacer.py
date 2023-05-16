from typing import Union

from tps.utils import load_dict, prob2bool
from tps.symbols import punctuation, accent
from tps.modules import Processor


class Replacer(Processor):
    def __init__(self, dict_source: Union[str, tuple, list, dict]=None,
                 name: str="Replacer"):
        """
        Base class for replacer-type processors.

        :param dict_source: Union[str, tuple, list, dict]
            Source of dictionary that contains replacement pairs.
            Options:
                * str - path to file.
                    The file extension must explicitly show its format in case of json and yaml files.
                    In other cases, user must set the format himself (see below).
                * tuple, list - (path, format)
                    path - path to the dictionary file
                    format - format of the dictionary file (see tps.utils.load_dict function)
                * dict - just a dict
        """
        super().__init__(None, name)

        fmt = None
        if isinstance(dict_source, (tuple, list)):
            dict_source, fmt = dict_source

        self.entries = load_dict(dict_source, fmt)


    def process(self, string: str, **kwargs) -> str:
        """
        Splits the passed string into tokens and replaces each one according to the dictionary (if exists).
        Keep it mind, that tokenization is simple here and it's better to pass normalized string.

        :param string: str
            Your text.
        :param kwargs:
            * mask: Union[bool, float]
                Whether to mask each token.
                If float, then masking probability will be computed for each token independently.

        :return: str
        """
        mask = kwargs.get("mask", False)
        tokens = self.split_to_tokens(string)

        for idx, token in enumerate(tokens):
            if token in punctuation:
                continue
            token = self._process_token(token, mask)
            tokens[idx] = token

        return self.join_tokens(tokens)


    def _process_token(self, token, mask):
        return token if prob2bool(mask) else self.entries.get(token, token)


class BlindReplacer(Replacer):
    def _process_token(self, token, mask):
        return token if prob2bool(mask) else self.entries.get(token.replace(accent, ""), token)