import os.path
import re
from collections import defaultdict
import typing
from typing import Union, Callable, Iterator, Tuple, List

from loguru import logger

import tps.symbols as smb
import tps.utils.cleaners as tps_cleaners
import tps.modules as md
import tps.types as _types
from tps.modules.ssml.elements import Pause


_curly = re.compile("({}.+?{})".format(*smb.shields))


class Handler(md.Processor):
    def __init__(self, charset: str, modules: list=None, out_max_length: int=None, save_state=False, name="Handler", use_cleaner=True):
        """
        This class stores a chain of passed modules and processes texts using this chain.

        :param charset: tps.types.Charset
            An element of the Charset class that has a corresponding symbol set (see tps.symbols).
        :param modules: Optional(list)
            A list of modules, that processes text in some way.
            If None, then the Handler object will have only basic functionality.
        :param out_max_length: Optional[int]
            If not None, text will be split into units less than out_max_length each.
        """
        super().__init__(name=name)
        self.charset = charset
        self.symbols = smb.symbols_map[charset]
        self.language = smb.language_map[charset]

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        self.modules = modules if modules is not None else []
        self._validate_modules(use_cleaner)

        self.out_max_length = out_max_length

        self._out_data = defaultdict(list)
        self.save_state = save_state


    @typing.overload
    def process(self, string: str, cleaners: str=None, user_dict: dict=None, **kwargs) -> str:
        ...


    @typing.overload
    def process(self, string: str, cleaners: Callable[[str], str]=None, user_dict: dict=None, **kwargs) -> str:
        ...


    @typing.overload
    def process(self, string: str, cleaners: Tuple[Union[str, Callable[[str], str]]]=None, user_dict: dict=None,
          **kwargs) -> str: ...


    @typing.overload
    def process(self, string: str, cleaners: List[Union[str, Callable[[str], str]]]=None, user_dict: dict=None,
          **kwargs) -> str: ...


    def process(self, string, cleaners=None, user_dict=None, **kwargs):
        """
        Apply the user_dict, the chain of modules and some cleaners to the passed sentence.

        :param string: str
            Sentence that needs to be processed.
        :param cleaners, user_dict, kwargs:
            See Handler.generate_text

        :return: str
            Returns processed string.
        """
        module: md.Processor
        origin_string = string

        cleaners = [] if cleaners is None else cleaners
        cleaners = [cleaners] if not isinstance(cleaners, (tuple, list)) else cleaners
        for _cleaner in cleaners:
            if isinstance(_cleaner, Callable):
                cleaner = _cleaner
            else:
                if hasattr(tps_cleaners, _cleaner):
                    cleaner = getattr(tps_cleaners, _cleaner)
                else:
                    print("Warning... There is no such cleaner {} in tps library.".format(_cleaner))
                    continue

            string = cleaner(string)

        if user_dict is not None:
            string = self.dict_check(string, user_dict)
            if self.save_state:
                self._out_data[origin_string].append(string)

        for module in self.modules:
            string = module(string, **kwargs)
            if self.save_state:
                self._out_data[origin_string].append(string)

        return string


    def process_text(self, text: Union[str, list], cleaners: Tuple[Union[str, Callable[[str], str]]]=None,
                     user_dict: dict=None, keep_delimiters: bool=True, **kwargs) -> Union[str, list]:
        """
        Process any text: first of all splits it to sentences, if it's possible.
        The Handler.process method is applied to each sentence after that.

        Wraps the Handler.generate_text method, converting iterator to a list of values.

        :param text, cleaners, user_dict, keep_delimiters, kwargs:
            See Handler.generate_text

        :return: Union[str, list]
            Returns text as a list of processed sentences (with Pause tokens, if keep_delimiters == True)
            or just a processed string.

            Cases:
                * list - if list was passed;
                * list - if string was passed and keep_delimiters == True;
                * str - if string was passed and keep_delimiters == False;

        Example:
        --------
        >>> text = "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?"
        >>> handler = Handler("en", [])
        >>> handler.process_text(text, keep_delimiters=False)
        'peter piper picked a peck of pickled peppers. how many pickled peppers did peter piper pick?'
        >>> handler(text, keep_delimiters=True)
        [
            'peter piper picked a peck of pickled peppers.',
            <Pause.eos: 500ms>,
            'how many pickled peppers did peter piper pick?'
        ]

        """
        return_string = isinstance(text, str) and not keep_delimiters
        processed = list(self.generate_text(text, cleaners, user_dict, keep_delimiters, **kwargs))

        return " ".join(processed) if return_string else processed


    def generate_text(self, text: Union[str, list], cleaners: Tuple[Union[str, Callable[[str], str]]]=None,
                      user_dict: dict=None, keep_delimiters: bool=True,
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
        :param cleaners: Optional[Tuple[Union[str, Callable[[str], str]]]]
            Tuple of cleaner functions (e.g. such that provided in tps.utils.cleaners).
        :param user_dict: dict
            See Handler.dict_check
        :param keep_delimiters: bool
            If True, final list will contain sentences and Pause tokens between them.
        :param kwargs:
            * mask_stress: Union[bool, float]
                Whether to mask each token in sentence.
                If float, then masking probability will be computed for each token independently.
            * mask_phonemes: Union[bool, float]
                Whether to mask each token in sentence.
                If float, then masking probability will be computed for each token independently.

        :return: Iterator[Union[str, tps.modules.ssml.Pause]]
        """
        self._clear_state()

        if isinstance(text, str):
            sentences = self.split_to_sentences(text, keep_delimiters, self.language)
        elif isinstance(text, list):
            sentences = text
        else:
            raise TypeError

        self._out_data = {sentence: [] for sentence in sentences if not isinstance(sentence, Pause)}

        for sentence in sentences:
            if not isinstance(sentence, Pause):
                sentence = self.process(sentence, cleaners, user_dict, **kwargs)

                if self.out_max_length is not None:
                    _units = self.split_to_units(sentence, self.out_max_length, keep_delimiters)

                    for unit in _units:
                        yield unit
                else:
                    yield sentence
            elif keep_delimiters:
                yield sentence
            else:
                continue


    def dict_check(self, string: str, user_dict: dict) -> str:
        """
        Checks the passed string using user_dict.

        :param string: str
            String that needs to be processed.
        :param user_dict: dict
            A dictionary containing specific cases that may occur in the text that
            needs to be resolved before main processing uses chain of modules.
            Example:
                {
                    "hello": "hell+o",
                    "compact": {
                        "a compact bag": "a c+ompact bag",
                        "to compact something": "to comp+act something"
                    },
                    "e.g": "for example"
                }

        :return: str
        """
        words = self.split_to_words(string)

        regexp_case = []
        for i, word in enumerate(words):
            key = word.lower()
            if key in user_dict:
                item = user_dict[key]

                if word.istitle():
                    item = item.capitalize()

                if isinstance(item, dict):
                    regexp_case.append(word)
                else:
                    words[i] = item

        regexp_case = set(regexp_case)
        string = self.join_words(words)

        for word in regexp_case:
            for case, value in user_dict[word].items():
                regexp = re.compile(case, re.IGNORECASE)
                string = regexp.sub(lambda elem: value, string)

        return string


    def text2vec(self, string: str) -> list:
        """
        Convert the passed string to the array of numbers.
        Each number is the corresponding index in the self.symbol_to_id dictionary
        for the characters in the string.

        :param string: str
            String that needs to be converted to the list of numbers.

        :return: list
            List of numbers.

        Example:
        --------
        >>> text = "Peter Piper picked a peck of pickled peppers."
        >>> handler = Handler("en", [])
        >>> processed = handler.process_text(text, keep_delimiters=False)
        >>> processed
        'peter piper picked a peck of pickled peppers.'
        >>> vector = handler.text2vec(processed)
        >>> vector
        [32, 16, 36, 16, 34, 12, 32, 17, 32, 16, 34, 12, 32, 17, 22, 28, 16, 23, 12, 15, 12, 32, 16, 22, 28,
        12, 18, 24, 12, 32, 17, 22, 28, 29, 16, 23, 12, 32, 16, 32, 32, 16, 34, 35, 2]
        """
        string = _curly.split(string)
        vector = []
        for elem in string:
            if elem.startswith(smb.shields[0]):
                elem = elem[1:-1].split(smb.separator)
            else:
                elem = list(elem)

            vector.extend(elem)

        return [self.symbol_to_id[s] for s in vector if self._should_keep_symbol(s)]


    def vec2text(self, vector: list) -> str:
        """
        Convert the passed array of numbers to the string.
        Each symbol of the string is the corresponding character in the self.id_to_symbol dictionary
        for the indices in the list.

        :param vector: list
            List of numbers, that needs to be converted to string.

        :return: str

        Example:
        --------
        >>> text = "Peter Piper picked a peck of pickled peppers."
        >>> handler = Handler("en", [])
        >>> processed = handler.process_text(text, keep_delimiters=False)
        >>> processed
        'peter piper picked a peck of pickled peppers.'
        >>> vector = handler.text2vec(processed)
        >>> vector
        [32, 16, 36, 16, 34, 12, 32, 17, 32, 16, 34, 12, 32, 17, 22, 28, 16, 23, 12, 15, 12, 32, 16, 22, 28,
        12, 18, 24, 12, 32, 17, 22, 28, 29, 16, 23, 12, 32, 16, 32, 32, 16, 34, 35, 2]
        >>> string = handler.vec2text(vector)
        >>> string
        'peter piper picked a peck of pickled peppers.'
        """
        text = []
        word = []
        prev = None
        _phonemes = smb.phoneme_map[self.charset]
        for elem_idx in vector:
            elem = self.id_to_symbol[elem_idx]
            if elem in _phonemes or (elem in [smb.hyphen, smb.accent] and prev in _phonemes):
                word.append(elem)
            else:
                if word:
                    text.append(smb.shields[0] + smb.separator.join(word) + smb.shields[1])
                text.append(elem)
                word = []
            prev = elem

        if word:
            text.append(smb.shields[0] + smb.separator.join(word) + smb.shields[1])

        return "".join(text)


    def check_eos(self, text: str):
        """
        Checks if there is an EOS token at the end of the text. If not, sets it.

        :param text: str

        :return: str
        """
        text = text if text.endswith(smb.eos) else text + smb.eos
        return text


    @classmethod
    def from_charset(cls, charset, out_max_length=None, data_dir=None, verify_checksum=True,
                     silent=False, use_cleaner=True):
        """
        Makes instance of the Handler class that is used by default for the passed charset.
        It's possible that some additional files need to be downloaded before -
        use tps.download to do that.

        :param charset: tps.types.Charset
            See Handler.__init__
        :param out_max_length: int
            See Handler.__init__
        :param inference: bool
            Whether the Handler should work at train or inference mode - the module list is set depending on this.
        :param data_dir: str
            See tps.download and tps.data.find
        :param verify_checksum: bool
            Whether verify or not the checksums of dictionaries and models
        :param silent: bool
            The dictionaries and models will be downloaded if ones don't exist or the checksums are invalid
            if silent == True, raises exceptions otherwise.

        :return: Handler
        """
        charset = _types.Charset(charset)
        modules = _get_default_modules(charset, data_dir, verify_checksum, silent, use_cleaner)

        return Handler(charset=charset, modules=modules, out_max_length=out_max_length, use_cleaner=use_cleaner)


    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id


    def _validate_modules(self, use_cleaner = True):
        omograph_exists = False
        emphasizer_exists = False
        number_exists = False
        lower_exists = False
        cleaner_exists = False
        phonetizer_type = None
        auxiliary_idx = 0

        for i, module in enumerate(self.modules):
            if isinstance(module, md.Lower):
                lower_exists = True
            if isinstance(module, md.Number):
                number_exists = True
            elif isinstance(module, md.Omograph):
                omograph_exists = True
            elif isinstance(module, md.Cleaner):
                cleaner_exists = True
            elif isinstance(module, md.Emphasizer):
                emphasizer_exists = True
            elif isinstance(module, md.Phonetizer):
                phonetizer_type = type(module)

                assert i + 1 == len(self.modules), "Phonetizer module must be the last one"
                if not omograph_exists:
                    print("Warning... There is no omographs in modules")
                if not emphasizer_exists:
                    print("Warning... There is no emphasizer in modules. "
                                   "Phonetizer will process words only with stress tokens set by user")

        if not lower_exists:
            self.modules.insert(auxiliary_idx, md.Lower())
            auxiliary_idx += 1
        if not number_exists:
            self.modules.insert(auxiliary_idx, md.Number(self.charset))
        if not cleaner_exists and use_cleaner:
            self.modules.insert(auxiliary_idx, md.Cleaner(self.charset))

        if self.charset == _types.Charset.ru:
            assert phonetizer_type is None


    def pop(self, item):
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str):
            for idx, module in enumerate(self.modules):
                if module.name == item:
                    break
        else:
            raise TypeError

        return self.modules.pop(idx)


    def _clear_state(self):
        del self._out_data
        self._out_data = {}


def get_symbols_length(charset: str):
    charset = _types.Charset[charset]
    return len(smb.symbols_map[charset])


def _get_file(name, data_dir):
    return os.path.join(data_dir, name)


def _get_default_modules(charset, data_dir=None, verify_checksum=True, silent=False, use_cleaner=True):
    modules = [
        md.Lower(),
        md.Number(charset)
    ]
    print(f"Using cleaner letters for language {charset} on text: {use_cleaner}")

    if use_cleaner:
        modules.extend([md.Cleaner(charset)])

    if charset == _types.Charset.ru:
        stress_a_dict = _get_file("stress.a.dict", data_dir)
        stress_b_dict = _get_file("stress.b.dict", data_dir)
        stress_c_dict = _get_file("stress.c.dict", data_dir)
        stress_d_dict = _get_file("stress.d.dict", data_dir)
        omographs_dict = _get_file("omographs.dict", data_dir)
        yo_dict = _get_file("yo.dict", data_dir)
        e_dict = _get_file("e.dict", data_dir)

        modules.extend([
            md.RuOmograph([omographs_dict, "plane"], True),
            md.BlindReplacer([e_dict, "plane"], name="Eficator"),
            md.BlindReplacer([yo_dict, "plane"], name="Yoficator"),
            md.RuEmphasizer([stress_a_dict, "plane"], True),
            md.RuEmphasizer([stress_b_dict, "plane"], True),
            md.RuEmphasizer([stress_c_dict, "plane"], True),
            md.RuEmphasizer([stress_d_dict, "plane"], True)
        ])
    elif charset == _types.Charset.en:
        pass
    elif charset == _types.Charset.en_cmu:
        raise NotImplementedError
    else:
        raise f"{ValueError}. This is language not support"

    return modules