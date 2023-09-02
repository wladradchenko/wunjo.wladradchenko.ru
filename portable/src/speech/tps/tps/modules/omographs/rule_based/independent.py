from typing import Union

from tps.modules import Replacer
from tps.utils import prob2bool
from tps.symbols import accent

"""
If you need to extend the Omograph functionality with
language-specific rules, just add a new descendant class.
"""

class Omograph(Replacer):
    def __init__(self, dict_source: Union[str, tuple, list, dict]=None, prefer_user: bool=True):
        super().__init__(dict_source, "Omograph")
        self.prefer_user = prefer_user


    def process(self, string: str, **kwargs) -> str:
        mask = kwargs.get("mask_stress", False)
        return super().process(string, mask=mask, rules="Omograph")


    def _process_token(self, token, mask):
        if prob2bool(mask):
            return token.replace(accent, "")

        stress_exists = token.find(accent) != -1
        if stress_exists and self.prefer_user:
            return token

        token = self.entries.get(token, token)

        return token