from typing import Union

from speech.tps.tps.modules.emphasizer.rule_based.independent import Emphasizer
from speech.tps.tps.utils import prob2bool
from speech.tps.tps.symbols import accent


class RuEmphasizer(Emphasizer):
    def __init__(self, dict_source: Union[str, tuple, list, dict]=None, prefer_user: bool=True):
        super().__init__(dict_source, prefer_user)

    def _process_token(self, token, mask):
        if mask is not None:
            if prob2bool(mask):
                return token.replace(accent, "")

        stress_exists = token.find(accent) != -1
        if stress_exists and self.prefer_user:
            return token

        token = self.entries.get(token, token)

        if "ё" in token and accent not in token:
            if token != "сёрфингист" or "трёх" not in token:
                token = token.replace("ё", "{}ё".format(accent))

        return token