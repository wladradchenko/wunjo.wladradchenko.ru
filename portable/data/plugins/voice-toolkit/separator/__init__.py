"""Pulling a voice out of what is behind it, or keeping what is behind it.

One model does both: MDX_Inst_HQ_3 was trained to find the instrumental, and the
voice is what is left of the mix once the instrumental is taken out. So the
checkbox in the effect costs nothing — the same separation answers either way,
and only the stem that is asked for is written.

    path = separate(model_path, "/abs/clip.wav", work, want="voice")
"""
from __future__ import annotations

import os
from typing import Callable, Optional

#: What this model pulls out, and what is left when it does.
MUSIC_STEM = "Instrumental"
VOICE_STEM = "Vocals"

#: MDX_Inst_HQ_3's own numbers. UVR looks these up in mdx_model_data.json by the
#: hash of the weights; there is one model here, so they are written down.
MODEL_DATA = {"compensate": 1.022, "mdx_dim_f_set": 3072, "mdx_dim_t_set": 8,
              "mdx_n_fft_scale_set": 6144, "primary_stem": MUSIC_STEM}
#: The rate the model was trained at. Anything else is resampled to it on the
#: way in, so the stem comes back at CD rate whatever the clip was.
SAMPLE_RATE = 44100


def providers_for(device: str) -> list:
    """The onnxruntime providers that exist on this machine, best first.

    Naming a provider that was not built in is an error, not a fallback: a CPU
    build asked for CUDA raises instead of quietly running on the processor.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in wanted if provider in available] or ["CPUExecutionProvider"]


def separate(model_path: str, audio_path: str, out_dir: str, want: str = "voice", device: str = "cpu",
             progress: Optional[Callable[[float, str], None]] = None) -> str:
    """Split @p audio_path and return the path of the stem @p want asks for."""
    import torch

    from .mdx import MDXSeparator

    stem = VOICE_STEM if want == "voice" else MUSIC_STEM
    separator = MDXSeparator(common_config={
        "torch_device": torch.device("cuda") if device != "cpu" else torch.device("cpu"),
        "torch_device_cpu": torch.device("cpu"),
        "torch_device_mps": None,
        "onnx_execution_provider": providers_for(device),
        "model_name": "MDX_Inst_HQ_3",
        "model_path": model_path,
        "model_data": MODEL_DATA,
        "output_format": "WAV",
        "output_dir": out_dir,
        "normalization_threshold": 0.9,
        # Only the stem that was asked for is written; the other one is still
        # computed, because one is the mix minus the other.
        "output_single_stem": stem,
        "invert_using_spec": False,
        "sample_rate": SAMPLE_RATE,
    })
    produced = separator.separate(audio_path, progress_callback=progress)
    for name in produced or []:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            return path
    # write_audio refuses to write a stem that came back silent, and a silent
    # stem is the honest answer to "take the singing out of a spoken clip".
    raise ValueError("nothing came out of the separation — the clip may hold no %s at all"
                     % ("voice" if want == "voice" else "music"))
