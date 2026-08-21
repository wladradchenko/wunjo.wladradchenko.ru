"""Voice cloning for Wunjo Make — the OpenVoice tone-colour converter.

The engine keeps *what* was said and replaces *who* says it, which is why the
plugin works on a clip that already has sound: the timing, the words and the
delivery are the ones on the timeline, and only the voice is the registered one.

    voice = CloneVoice(config, checkpoint, device)
    tone = voice.measure(reference)          # once, when a voice is registered
    spoken = voice.speak(clip_audio, tone)   # every time Generate is pressed
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from .api import DigitalSignature, ToneColorConverter
from .clone import chunk_seconds_for, convert, sign, speaker_embedding, voiced_audio

__all__ = ["CloneVoice", "DigitalSignature", "ToneColorConverter", "chunk_seconds_for", "voiced_audio"]


class CloneVoice:
    """The converter, loaded once and asked for either half of the job."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        self.converter = ToneColorConverter(config_path, device=device)
        self.converter.load_ckpt(checkpoint_path)
        self.device = device

    @property
    def sampling_rate(self) -> int:
        return self.converter.sampling_rate

    def measure(self, audio: np.ndarray) -> List[float]:
        """A recording reduced to the numbers that describe its voice.

        Returned as plain floats: this is what a voice preset stores, and a
        preset is a small json file that has to survive being copied between
        projects.
        """
        return [float(value) for value in speaker_embedding(self.converter, audio).flatten().cpu().numpy()]

    def tone(self, values: List[float]) -> torch.Tensor:
        """A stored measurement, back in the shape the converter conditions on."""
        return torch.FloatTensor(values).to(self.device).reshape(1, -1, 1)

    def speak(self, audio: np.ndarray, voice: List[float], tau: float = 0.3, chunk_seconds: float = 0.0,
              progress: Optional[Callable[[float], None]] = None, watermark: bool = True) -> np.ndarray:
        """@p audio, said the same way, by the voice @p voice was measured from."""
        source = speaker_embedding(self.converter, audio)
        spoken = convert(self.converter, audio, source, self.tone(voice), tau=tau,
                         chunk_seconds=chunk_seconds, progress=progress)
        if watermark:
            try:
                spoken = sign(spoken, self.device)
            except Exception as error:  # a missing mark must not cost the render
                print("could not sign the result:", error)
        return spoken
