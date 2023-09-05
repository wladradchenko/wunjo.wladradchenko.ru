from pypinyin import lazy_pinyin, Style
from tps import modules as md


class ZhPolyphonic(md.Processor):
    def process(self, string: str, **kwargs) -> str:
        return " ".join(lazy_pinyin(string, style=Style.TONE3, neutral_tone_with_five=True))