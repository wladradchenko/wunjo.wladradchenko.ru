from xml.etree import ElementTree as ET

from tps.modules.ssml.elements import Text, Pause
from tps.types import SSMLTag
from tps.utils.cleaners import collapse_whitespace


def parse_ssml_text(text):
    text = collapse_whitespace(text)
    root = ET.fromstring(text)

    sequence = _walk_ssml_elem(root)

    optimized_seq = [sequence.pop(0)]

    for elem in sequence:
        if isinstance(elem, Text):
            if isinstance(optimized_seq[-1], Text):
                optimized_seq[-1] += elem
            else:
                optimized_seq.append(elem)
        elif isinstance(elem, Pause):
            if isinstance(optimized_seq[-1], Pause):
                optimized_seq[-1] = max(optimized_seq[-1], elem, key=lambda pause: pause.milliseconds)
            else:
                optimized_seq.append(elem)

    return optimized_seq


def _wrap_text(text):
    if text is None:
        return Text("")
    else:
        return Text(collapse_whitespace(text))


def _walk_ssml_elem(elem, ancestor=None):
    body = _wrap_text(elem.text)
    tail = _wrap_text(elem.tail)

    if ancestor is not None:
        body.inherit(ancestor)
        tail.inherit(ancestor)

    sequence = []

    if SSMLTag.nested(elem.tag):
        if elem.tag == SSMLTag.p:
            pause = Pause.paragraph()
            sequence.append(pause)
        elif elem.tag == SSMLTag.s:
            pause = Pause.eos()
            sequence.append(pause)
        elif elem.tag == SSMLTag.prosody:
            body.update_prosody(**elem.attrib)
            ancestor = body

        if not body.is_empty:
            sequence.append(body)

        for child in elem:
            sequence.extend(_walk_ssml_elem(child, ancestor))

        if elem.tag == SSMLTag.p or elem.tag == SSMLTag.s:
            sequence.append(pause)
        else:
            sequence.append(Pause.space())

        if not tail.is_empty:
            sequence.append(tail)

        return sequence
    elif elem.tag == SSMLTag.break_:
        sequence.append(Pause(**elem.attrib))
    elif elem.tag == SSMLTag.sub:
        body.update_value(elem.attrib["alias"])
    elif elem.tag == SSMLTag.phoneme:
        pass
    else:
        raise KeyError

    if not body.is_empty:
        sequence.append(body)

    if not tail.is_empty:
        sequence.append(tail)

    return sequence