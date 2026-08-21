"""What the plugin actually does with the converter: measure a voice, wear it.

Two operations, and both work on arrays rather than on files:

  ``speaker_embedding``  reduces a recording to the 256 numbers that describe
                         *who* is speaking — the whole of what a voice preset
                         stores, and all the conversion needs of the voice it
                         copies.
  ``convert``            re-speaks a clip's sound with those numbers in place of
                         its own, in pieces short enough to fit in memory.

Finding the speech inside a recording is the Silero VAD's job here, as it was in
Wunjo v2 — the difference from the plain Clone Voice plugin, which reads the same
recording with an energy threshold. Loudness keeps whatever is loud, so over
music, a hum or a street the average ends up describing the room; a detector
trained on speech keeps the voice and nothing else.

What did change from v2 is the way that model is reached. v2 called
``whisper_timestamped.get_vad_segments``, which pulls snakers4/silero-vad off
GitHub through ``torch.hub`` the first time it runs — the download the donor's
own TODO complains about, and one more thing to fail on a machine with no
network. The model is published on PyPI with its weights inside the wheel
(2.3 MB of it), so here it arrives with the environment and nothing is fetched
while a job is running. The parameters below are the ones that wrapper used.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import librosa
import numpy as np
import torch

from .mel_processing import spectrogram_torch

#: A recording is averaged over pieces of about this length, as OpenVoice does.
SEGMENT_SECONDS = 10.0
#: Below this a piece is too short to say anything about a voice.
MIN_SEGMENT_SECONDS = 1.0
#: How far from an even cut the converter may look for a quiet moment to cut at.
CUT_SEARCH_SECONDS = 2.0


#: The rate the VAD hears at; everything is resampled to it and mapped back.
VAD_SAMPLE_RATE = 16000
#: Shorter than this is not a word (`min_speech_duration` in Wunjo v2).
VAD_MIN_SPEECH_SECONDS = 0.1
#: A gap shorter than this is a pause inside speech rather than the end of it —
#: a whole second, as v2 asked for: breath and thought belong to the voice.
VAD_MIN_SILENCE_SECONDS = 1.0
#: Every segment is grown by this at both ends and overlapping ones merged, as
#: the wrapper v2 went through did. It is what keeps the detector from clipping
#: the quiet start of a word off the sample the voice is measured from.
VAD_DILATATION_SECONDS = 0.5

_vad_model = None


def _silero():
    """The detector, loaded once from the weights inside the installed package."""
    global _vad_model
    if _vad_model is None:
        # Importing it pins torch to a single thread — the package does that at
        # import time, for its own tiny model. Left alone the setting stays for
        # the rest of the process, and the converter, which is the part that is
        # actually heavy, would then decode on one core.
        threads = torch.get_num_threads()
        from silero_vad import load_silero_vad

        torch.set_num_threads(threads)
        _vad_model = load_silero_vad()
    return _vad_model


def speech_segments(audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """Where somebody is speaking, as (start, end) in samples of @p sr."""
    from silero_vad import get_speech_timestamps

    if audio.size == 0:
        return []
    signal = audio if sr == VAD_SAMPLE_RATE else librosa.resample(audio, orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    heard = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
    # the cheap normalisation v2's wrapper did before asking: a quiet recording
    # is still speech, and the detector should not have to be told twice
    heard = heard / max(0.1, float(heard.abs().max()))
    stamps = get_speech_timestamps(heard, _silero(), sampling_rate=VAD_SAMPLE_RATE,
                                   min_speech_duration_ms=round(VAD_MIN_SPEECH_SECONDS * 1000),
                                   min_silence_duration_ms=round(VAD_MIN_SILENCE_SECONDS * 1000),
                                   return_seconds=False)
    grown = round(VAD_DILATATION_SECONDS * VAD_SAMPLE_RATE)
    merged: List[List[int]] = []
    for stamp in stamps:
        start = max(0, int(stamp["start"]) - grown)
        end = min(len(heard), int(stamp["end"]) + grown)
        if merged and merged[-1][1] >= start:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    scale = sr / float(VAD_SAMPLE_RATE)
    return [(int(round(start * scale)), min(len(audio), int(round(end * scale)))) for start, end in merged]


def loud_audio(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Everything above the noise floor, joined up — the fallback.

    Loudness cannot tell a voice from the music under it, which is the whole
    reason the detector is there. It is still the right answer when the detector
    has nothing to say: it keeps working on singing, on a whisper, and in
    languages Silero was not trained for.
    """
    if audio.size == 0:
        return audio
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return np.zeros(0, dtype=audio.dtype)
    return np.concatenate([audio[start:end] for start, end in intervals])


def voiced_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Everything a voice was heard in, joined up.

    Pauses carry no timbre, and a recording that is half silence would otherwise
    pull the average towards the room rather than the voice in it. The detector
    answers first, because what has to be kept out is not only silence — music
    under a voice, a fan, a street are all loud. When it hears nobody at all the
    loudness threshold answers instead: better a measurement taken from the loud
    parts of a song than a plugin that refuses to run on one.
    """
    segments = speech_segments(audio, sr)
    speech = np.concatenate([audio[start:end] for start, end in segments]) if segments else np.zeros(0, dtype=audio.dtype)
    if len(speech) >= MIN_SEGMENT_SECONDS * sr:
        return speech
    return loud_audio(audio, sr)


def _even_pieces(audio: np.ndarray, sr: int, seconds: float) -> List[np.ndarray]:
    """The recording in pieces of roughly @p seconds, none of them a sliver."""
    duration = len(audio) / float(sr)
    count = max(1, int(round(duration / seconds)))
    edges = np.linspace(0, len(audio), count + 1).astype(int)
    pieces = [audio[edges[i]:edges[i + 1]] for i in range(count)]
    return [piece for piece in pieces if len(piece) >= MIN_SEGMENT_SECONDS * sr] or [audio]


def speaker_embedding(converter, audio: np.ndarray) -> torch.Tensor:
    """The 256 numbers that describe who is speaking in @p audio.

    Measured over the speech alone and averaged, so that a long recording is
    described by its voice rather than by whichever sentence happened to be
    first.
    """
    hps = converter.hps
    sr = converter.sampling_rate
    speech = voiced_audio(audio, sr)
    if len(speech) < MIN_SEGMENT_SECONDS * sr:
        # Neither the detector nor the threshold found anything to keep: a very
        # quiet recording. Measure all of it rather than refuse — a clip on the
        # timeline still has to come back in the new voice.
        speech = audio
    if len(speech) < MIN_SEGMENT_SECONDS * sr:
        raise ValueError("there is not enough sound here to measure a voice")

    embeddings = []
    for piece in _even_pieces(speech, sr, SEGMENT_SECONDS):
        y = torch.FloatTensor(piece).to(converter.device).unsqueeze(0)
        spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                                 hps.data.hop_length, hps.data.win_length, center=False).to(converter.device)
        with torch.no_grad():
            embeddings.append(converter.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1).detach())
    return torch.stack(embeddings).mean(0)


def _quiet_cuts(audio: np.ndarray, sr: int, chunk_seconds: float) -> List[int]:
    """Where to cut a long clip, preferring the quiet moments.

    A cut in the middle of a vowel is audible: the two pieces are converted
    independently and the seam lands mid-sound. Nudging each cut to the quietest
    moment within a couple of seconds puts every seam in a pause instead, where
    there is nothing to interrupt.
    """
    chunk = int(chunk_seconds * sr)
    if chunk <= 0 or len(audio) <= chunk * 1.5:
        return []
    window = max(1, int(0.05 * sr))
    reach = int(CUT_SEARCH_SECONDS * sr)
    cuts: List[int] = []
    position = 0
    while len(audio) - position > chunk * 1.5:
        target = position + chunk
        low = max(position + chunk // 2, target - reach)
        high = min(len(audio) - window, target + reach)
        cut = target
        if high > low:
            frame = audio[low:high + window].astype(np.float64)
            # moving energy, summed once: the quietest window is the seam
            power = np.cumsum(np.concatenate(([0.0], frame * frame)))
            energy = power[window:] - power[:-window]
            cut = low + int(np.argmin(energy)) + window // 2
        cuts.append(cut)
        position = cut
    return cuts


def chunk_seconds_for(device: str, requested: float = 0.0) -> float:
    """How much sound to convert in one pass.

    Asked of the hardware rather than guessed: the decoder holds the whole piece
    at full sample rate across every upsampling stage, so this is what decides
    whether a long clip fits. The caller still halves it on an out-of-memory
    error — no estimate survives every machine.
    """
    if requested > 0:
        return requested
    if device == "cpu":
        return 30.0
    try:
        free, _total = torch.cuda.mem_get_info()
        return max(10.0, min(60.0, free / float(1024 ** 3) * 10.0))
    except Exception:
        return 15.0


def convert(converter, audio: np.ndarray, src_se: torch.Tensor, tgt_se: torch.Tensor, tau: float = 0.3,
            chunk_seconds: float = 0.0, progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """@p audio spoken by @p tgt_se instead of @p src_se, piece by piece."""
    hps = converter.hps
    sr = converter.sampling_rate
    pieces = np.split(audio, _quiet_cuts(audio, sr, chunk_seconds_for(str(converter.device), chunk_seconds)))
    done = 0
    out: List[np.ndarray] = []
    for piece in pieces:
        if len(piece) < hps.data.filter_length:
            # too short to hold a single spectrogram frame: keep it as it is
            out.append(piece)
            continue
        with torch.no_grad():
            y = torch.FloatTensor(piece).to(converter.device).unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                                     hps.data.hop_length, hps.data.win_length, center=False).to(converter.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(converter.device)
            converted = converter.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se,
                                                         tau=tau)[0][0, 0].data.cpu().float().numpy()
        # The decoder answers in whole hops, so a piece comes back a few samples
        # short of what went in. Left alone, those samples add up across a long
        # clip and the result drifts out of step with the picture it belongs to.
        out.append(np.pad(converted, (0, max(0, len(piece) - len(converted))))[:len(piece)])
        done += len(piece)
        if progress:
            progress(100.0 * done / max(1, len(audio)))
    return np.concatenate(out) if out else audio


def sign(audio: np.ndarray, device: str, message: str = "ai") -> np.ndarray:
    """Write the Wunjo mark into a converted track, if it is long enough."""
    from .api import DigitalSignature

    return DigitalSignature(device=device).set_encrypted(audio, message)
