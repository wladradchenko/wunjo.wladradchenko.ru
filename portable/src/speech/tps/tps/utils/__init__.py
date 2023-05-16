import os
import re
import json
import yaml
import numpy as np

from tps import symbols as smb


def prob2bool(prob):
    return prob if isinstance(prob, bool) else np.random.choice([True, False], p=[prob, 1 - prob])


_punct_re = re.compile("[{}]".format("".join(smb.punctuation)))
def split_to_tokens(text, punct_re=None):
    punct_re = _punct_re if punct_re is None else punct_re

    prepared = punct_re.sub(lambda elem: "⁑{}⁑".format(elem.group(0)), text)

    prepared = prepared.split("⁑")
    prepared = [t for t in prepared if t != ""]

    return prepared


def hide_stress(regexp, text):
    return regexp.sub(lambda elem: elem.group(0)[-1].upper(), text)


def reveal_stress(regexp, text):
    return regexp.sub(lambda elem: "+" + elem.group(0).lower(), text)


def load_dict(dict_source, fmt=None):
    _dict = {}

    if isinstance(dict_source, str):
        _, ext = os.path.splitext(dict_source)
        if ext in [".json", ".yaml"]:
            fmt = ext.replace(".", "")
        elif fmt is None:
            raise ValueError("File format must be specified ['json', 'yaml', 'plane']")

        assert os.path.exists(dict_source)

        with open(dict_source, "r", encoding="utf-8") as stream:
            if fmt == "json":
                _dict = json.load(stream)
            elif fmt == "yaml":
                _dict = yaml.safe_load(stream)
            elif fmt == "plane":
                _dict = stream.read().splitlines()
                _dict = tuple(line.split("|") for line in _dict)
                _dict = {elem[0]: elem[1] for elem in _dict}
            else:
                raise ValueError("File format must be specified ['json', 'yaml', 'plane']")

    elif isinstance(dict_source, dict):
        _dict = dict_source
    elif dict_source is None:
        pass
    else:
        raise TypeError

    return _dict


def save_dict(dict_obj, filepath, fmt=None):
    _dict = {}

    _, ext = os.path.splitext(filepath)
    if ext in [".json", ".yaml"]:
        fmt = ext.replace(".", "")
    elif fmt is None:
        raise ValueError("File format must be specified ['json', 'yaml', 'plane']")

    with open(filepath, "w", encoding="utf-8") as stream:
        if fmt == "json":
            json.dump(dict_obj, stream, indent=2, ensure_ascii=False)
        elif fmt == "yaml":
            yaml.dump(dict_obj, stream, indent=2, allow_unicode=True)
        else:
            raise ValueError("File format must be specified ['json', 'yaml']")

    return filepath