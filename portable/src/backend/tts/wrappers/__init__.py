import os
import sys

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../backend")

import_path = os.path.join(backend_path, "tacotron2")
sys.path.insert(0, import_path)
from .tacotron import Tacotron2Wrapper
sys.path.pop(0)

import_path = os.path.join(backend_path, "waveglow")
sys.path.insert(0, import_path)
from .waveglow import WaveglowWrapper
sys.path.pop(0)