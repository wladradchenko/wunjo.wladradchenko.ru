import os
import sys
import re

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.synthesizer.utils import cleaners

sys.path.pop(0)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, _symbol_to_id):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), _symbol_to_id)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), _symbol_to_id)
        sequence += _arpabet_to_sequence(m.group(2), _symbol_to_id)
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id["~"])
    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, _symbol_to_id):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _symbol_to_id)]


def _arpabet_to_sequence(text, _symbol_to_id):
    return _symbols_to_sequence(["@" + s for s in text.split()], _symbol_to_id)


def _should_keep_symbol(s, _symbol_to_id):
    return s in _symbol_to_id and s not in ("_", "~")
