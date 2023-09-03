import os
import sys

wrappers_path = os.path.dirname(os.path.abspath(__file__))
tts_path = os.path.dirname(wrappers_path)
speech_path = os.path.dirname(tts_path)
src_path = os.path.dirname(speech_path)
root_path = os.path.dirname(src_path)
backend_path = os.path.join(tts_path, "backend")

import_path = os.path.join(root_path, "tacotron2")
sys.path.insert(0, import_path)
from .tacotron import Tacotron2Wrapper
sys.path.pop(0)

import_path = os.path.join(backend_path, "waveglow")
sys.path.insert(0, import_path)
from .waveglow import WaveglowWrapper
sys.path.pop(0)