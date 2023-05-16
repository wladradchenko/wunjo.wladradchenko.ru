import re
from collections import OrderedDict
from typing import Union, Pattern, Iterator

from nltk import sent_tokenize, word_tokenize

from tps.utils import split_to_tokens
from tps.modules.ssml.elements import Pause
from tps.symbols import separator, shields


char_map = OrderedDict({
    ". ": Pause.eos,
    "? ": Pause.eos,
    "! ": Pause.eos,
    ": ": Pause.colon,
    "; ": Pause.semicolon,
    ", ": Pause.comma,
    " ": Pause.space
})

_spaced_punctuation = re.compile(r" [{}]".format("".join([char for char in char_map if char != " "])))


class Processor:
    def __init__(self, max_unit_length: int=None, name="Processor"):
        """
        Base class for all text processors.

        :param max_unit_length: Optional[int]
            If not None, passed text will be split into units less than max_unit_length each.
            See Processor.__call__ and Processor.split_to_units
        :param name: str
        """
        self.max_unit_length = max_unit_length

        self.name = name


    def __call__(self, sentence: str, **kwargs) -> str:
        """
        Processes the passed sentence. Exactly one sentence should be passed for proper work.
        If the self.max_unit_length attribute is not None, then the sentence will be split into units,
        and each unit will be processed independently and joined at the end.

        :param sentence: str
        :param kwargs: dict
            See tps.Handler.generate_text

        :return: str
        """
        if self.max_unit_length is not None:
            parts = self.split_to_units(sentence, self.max_unit_length)
        else:
            parts = (sentence,)

        parts = [self.process(part, **kwargs) for part in parts]

        return " ".join(parts)


    def __str__(self):
        return "<{}: max unit length {}>".format(self.name, self.max_unit_length)


    def process(self, string: str, **kwargs) -> str:
        """
        Must be implemented in the descendant classes. Processes the passed string.

        :param string: str
        :param kwargs: dict
            See tps.Handler.generate_text

        :return: str
        """
        raise NotImplementedError


    def process_text(self, text: Union[str, list], keep_delimiters: bool=False, **kwargs) -> Union[str, list]:
        """
        Process any text: first of all splits it to sentences, if it's possible.
        The Processor.__call__ method is applied to each sentence after that.

        Wraps the Processor.generate_text method, converting iterator to a list of values.

        :param text: Union[str, list]
            See Processor.generate_text
        :param keep_delimiters: bool
            See Processor.generate_text
        :param kwargs:
            See tps.Handler.generate_text

        :return: Union[str, list]
            Returns text as a list of processed sentences (with Pause tokens, if keep_delimiters == True)
            or just a processed string.

            Cases:
                * list - if list was passed;
                * list - if string was passed and keep_delimiters == True;
                * str - if string was passed and keep_delimiters == False;
        """
        return_string = isinstance(text, str) and not keep_delimiters
        processed = list(self.generate_text(text, keep_delimiters, **kwargs))

        return " ".join(processed) if return_string else processed


    def generate_text(self, text: Union[str, list], keep_delimiters: bool=False,
                      **kwargs) -> Iterator[Union[str, Pause]]:
        """
        Produces a generator of processed sentences or units (with Pause tokens, if keep_delimiters == True).

        :param text: Union[str, list]
            Text that needs to be processed.

            Cases:
                * str - just an ordinary string;
                * list - it's assumed, that user submits text that has already been split into sentences
                (with or without delimiters), for example:
                    [
                        text_part_0,
                        <Pause.eos: 500ms>,
                        text_part_1
                    ]
        :param keep_delimiters: bool
            If True, final list will contain sentences and Pause tokens between them.
        :param kwargs:
            See tps.Handler.generate_text

        :return: Iterator[Union[str, tps.modules.ssml.Pause]]
        """
        if isinstance(text, str):
            sentences = self.split_to_sentences(text)
        elif isinstance(text, list):
            sentences = text
        else:
            raise TypeError

        for sentence in sentences:
            if not isinstance(sentence, Pause):
                processed = self(sentence, **kwargs)
                yield processed
            elif keep_delimiters:
                yield sentence
            else:
                continue


    def _calc_weight(self, text):
        """
        Calculates weight of the each unit. For example, we do not want to take the shield symbols into account,
        when calculating how many chars there are in the text.

        :param text: str

        :return: int
        """
        _text = text
        for symb in shields:
            _text = _text.replace(symb, "")

        _text = Processor.split_to_tokens(_text)

        weight = sum(len(s.split(separator)) if separator in s else len(s) for s in _text)

        return weight


    def _distribute_parts(self, parts, delimiter):
        """
        Auxiliary function for Processor.split_to_units.

        :param parts: list
        :param delimiter: str

        :return:
        """
        _delimiter = "" if delimiter == " " else delimiter.replace(" ", "")

        parts_grouped = [
            delimiter.join(parts[:len(parts) // 2]) + _delimiter,
            delimiter.join(parts[len(parts) // 2:])
        ]
        return parts_grouped


    def split_to_units(self, text: str, max_unit_length: int, keep_delimiter: bool=False) -> list:
        """
        Splits specified text into units, whose weight less than max_unit_length.

        :param text: str
        :param max_unit_length: int
        :param keep_delimiter: bool
            If True, final list will contain units and Pause tokens between them.

        :return: list
        """
        if self._calc_weight(text) <= max_unit_length:
            return [text]

        for delimiter in char_map:
            found = text.find(delimiter)
            if found != -1 and found != len(text) - 1:
                break

        if found != -1:
            parts = [p.strip() for p in text.split(delimiter)]
        else:
            parts = [text[:len(text) // 2], text[len(text) // 2:]]

        _parts_grouped = self._distribute_parts(parts, delimiter)
        if keep_delimiter and len(_parts_grouped) > 1:
            _parts_grouped.insert(1, char_map[delimiter]())

        parts_grouped = []
        for part in _parts_grouped:
            if isinstance(part, Pause) or self._calc_weight(part) <= max_unit_length:
                parts_grouped.append(part)
            else:
                parts_grouped.extend(self.split_to_units(part, max_unit_length, keep_delimiter))

        return parts_grouped


    @staticmethod
    def split_to_sentences(text: str, keep_delimiters: bool=False, language: str="russian") -> list:
        """
        Splits specified text into sentences using nltk library.

        :param text: str
        :param keep_delimiters: bool
            If True, final list will contain sentences and Pause tokens between them.
        :param language: str
            The model name in the nltk Punkt corpus

        :return: list
        """
        parts = sent_tokenize(text, language)

        if keep_delimiters:
            for i in range(1, len(parts)):
                parts.insert(i * 2 - 1, Pause.eos())

        return parts


    @staticmethod
    def split_to_words(text: str) -> list:
        """
        Splits specified text into words using nltk library.

        :param text: str

        :return: list
        """
        return word_tokenize(text)


    @staticmethod
    def join_words(words: list) -> str:
        """
        Reverses the self.split_to_words method.

        :param words: list
            List of words got from the self.split_to_words method.

        :return: str
        """
        words = " ".join(words)
        words = _spaced_punctuation.sub(lambda elem: elem.group(0)[-1], words)
        return words


    @staticmethod
    def split_to_tokens(text: str, punct_re: Pattern=None) -> list:
        """
        Splits specified text into words, treating whitespaces as independent elements.

        Unlike the self.split_to_words method can not recognize complex cases such as 'e.g.' and
        works a little faster.

        :param text: str
        :param punct_re: Pattern

        :return: str

        Example:
        --------
        >>> proc = Processor()
        >>> text = "splitting sentence, e.g. this one."
        >>> proc.split_to_words(text)
        ['splitting', 'sentence', ',', 'e.g', '.', 'this', 'one', '.']
        >>> proc.split_to_tokens(text)
        ['splitting', ' ', 'sentence', ',', ' ', 'e', '.', 'g', '.', ' ', 'this', ' ', 'one', '.']
        """
        return split_to_tokens(text, punct_re)


    @staticmethod
    def join_tokens(tokens: list) -> str:
        """
        Reverses the self.split_to_tokens method.

        :param tokens: list
            List of tokens got from the self.split_to_tokens method.

        :return: str
        """
        return "".join(tokens)