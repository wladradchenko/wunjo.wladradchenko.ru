"""Who speaks when, from two ONNX models and nothing else.

The pyannote 3.1 recipe, rebuilt on onnxruntime: its own package brings a
training framework, three telemetry libraries and a cloud client along for two
files that weigh 31 MB between them.

The shape of it:

* a segmentation model reads the waveform in ten-second windows and says, for
  every 17 ms frame, which of *up to three* voices in that window are talking.
  It answers as one of seven classes — silence, one of three speakers, or one of
  three pairs talking over each other — so overlaps come out of the model
  directly rather than being guessed at afterwards.
* those speakers are local to their window: "the second voice here" is not "the
  second voice ten seconds later". So each one gets an embedding of its own
  voice, and the embeddings are clustered across the whole recording. A cluster
  is a person.
* the windows are then stitched back together through that mapping.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SECONDS = 10.0
STEP_SECONDS = 2.0
# The model's own frame rate: a ten-second window comes back as 589 frames.
FRAMES_PER_WINDOW = 589
# What the seven classes mean, read off the model's config: which of the three
# local voices each class says are speaking.
POWERSET = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
# The cosine distance two voices may be apart and still be one person.
#
# Measured twice, because one measurement was not enough. On clean studio speech
# — three known speakers, two halves of each — the same voice sits 0.12-0.27
# apart and different voices 0.87-1.04. On a real recording (two men in an
# exhibition hall, lapel mics, crowd behind them) the distances between the 187
# windows come out bimodal: a quarter below 0.39, the middle at 0.80.
#
# So the gap is wide, but it does not sit where pyannote's own 0.7045 expects.
# At 0.6 and above, centroid linkage chains: two clusters merge, the centre
# moves, and everything collapses into one person. 0.5 is inside the gap on both
# recordings and answers each of them correctly.
CLUSTER_THRESHOLD = 0.5


def load_audio(path: str, ffmpeg: str, start: float = 0.0, span: float = 0.0) -> np.ndarray:
    """The track as mono float32 at 16 kHz, which is all either model reads.

    Only @p span seconds from @p start: the clip on the timeline is often a
    minute taken out of an hour, and the hour is neither interesting nor cheap.
    """
    import subprocess

    command = [ffmpeg, "-v", "error"]
    if start > 0:
        command += ["-ss", "%.6f" % start]
    command += ["-i", path]
    if span > 0:
        command += ["-t", "%.6f" % span]
    command += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def fbank(wave: np.ndarray) -> np.ndarray:
    """80-bin log mel filterbank, exactly as Kaldi computes it.

    Not "close enough": the embedder was trained on these numbers, and every
    departure from them is a distortion it was never shown. Reading the same
    audio through librosa's defaults put the mel bands in different places
    (librosa's own scale, not HTK's), dropped the pre-emphasis and used the
    wrong window — and the embeddings came out just similar enough that two
    people in a room could not be told apart.

    The recipe, from Kaldi's `compute-fbank-feats` with WeSpeaker's settings:
    25 ms frames every 10 ms, the mean taken out of each frame, pre-emphasis at
    0.97, a Povey window, an FFT padded to a power of two, and 80 triangular
    bands laid out on the HTK mel scale between 20 Hz and 7600 Hz.
    """
    frame_length = int(0.025 * SAMPLE_RATE)   # 400
    frame_shift = int(0.010 * SAMPLE_RATE)    # 160
    if len(wave) < frame_length:
        return np.zeros((0, 80), dtype=np.float32)

    count = 1 + (len(wave) - frame_length) // frame_shift
    index = np.arange(frame_length)[None, :] + frame_shift * np.arange(count)[:, None]
    frames = wave[index].astype(np.float64)

    # Kaldi works in the int16 range; a constant scale would wash out under the
    # mean subtraction at the end, but the energy floor below is absolute.
    frames = frames * (1 << 15)
    frames -= frames.mean(axis=1, keepdims=True)
    # pre-emphasis, with the first sample of each frame standing in for the one
    # before it, exactly as Kaldi does at a frame boundary
    emphasised = np.empty_like(frames)
    emphasised[:, 0] = frames[:, 0] - 0.97 * frames[:, 0]
    emphasised[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]

    # Povey window: a Hann raised to 0.85
    n = np.arange(frame_length)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_length - 1))) ** 0.85
    spectrum = np.fft.rfft(emphasised * window, n=512)
    power = np.abs(spectrum) ** 2

    feats = np.log(np.maximum(power @ _mel_bank().T, np.finfo(np.float64).eps))
    # cepstral mean normalisation: the room and the microphone are not the voice
    feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


def _hz_to_mel(hz):
    """The HTK scale Kaldi uses — not librosa's default, which differs."""
    return 1127.0 * np.log(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


_MEL_BANK = None


def _mel_bank(bins: int = 80, low: float = 20.0, high: float = 7600.0) -> np.ndarray:
    """Kaldi's triangular bands: peaks evenly spaced in mel, edges at neighbours."""
    global _MEL_BANK
    if _MEL_BANK is not None:
        return _MEL_BANK
    points = _mel_to_hz(np.linspace(_hz_to_mel(low), _hz_to_mel(high), bins + 2))
    freqs = np.fft.rfftfreq(512, 1.0 / SAMPLE_RATE)
    bank = np.zeros((bins, len(freqs)))
    for i in range(bins):
        left, centre, right = points[i], points[i + 1], points[i + 2]
        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)
        bank[i] = np.maximum(0.0, np.minimum(rising, falling))
    _MEL_BANK = bank
    return bank


class Diarizer:
    def __init__(self, models_dir: str, device: str = "cpu"):
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.segmentation = onnxruntime.InferenceSession(os.path.join(models_dir, "segmentation.onnx"), providers=providers)
        self.embedder = onnxruntime.InferenceSession(os.path.join(models_dir, "embedding.onnx"), providers=providers)
        self.seg_input = self.segmentation.get_inputs()[0].name
        self.emb_input = self.embedder.get_inputs()[0].name

    def _windows(self, wave: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        size = int(WINDOW_SECONDS * SAMPLE_RATE)
        step = int(STEP_SECONDS * SAMPLE_RATE)
        if len(wave) <= size:
            return [(0, np.pad(wave, (0, size - len(wave))))]
        starts = list(range(0, len(wave) - size + 1, step))
        # never drop the tail: a last window flush against the end of the track
        if starts[-1] + size < len(wave):
            starts.append(len(wave) - size)
        return [(s, wave[s:s + size]) for s in starts]

    def _activity(self, window: np.ndarray) -> np.ndarray:
        """(frames, 3) — which of this window's three voices speaks per frame."""
        logits = self.segmentation.run(None, {self.seg_input: window[None, None, :].astype(np.float32)})[0][0]
        chosen = logits.argmax(axis=-1)
        active = np.zeros((logits.shape[0], 3), dtype=bool)
        for cls, speakers in enumerate(POWERSET):
            if not speakers:
                continue
            rows = chosen == cls
            for spk in speakers:
                active[rows, spk] = True
        return active

    def _embed(self, window: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """One voice's embedding, taken from the frames where only it speaks."""
        if mask.sum() < 20:  # under a third of a second is not a voice
            return None
        samples_per_frame = len(window) / float(len(mask))
        keep = np.zeros(len(window), dtype=bool)
        for index in np.flatnonzero(mask):
            keep[int(index * samples_per_frame):int((index + 1) * samples_per_frame)] = True
        speech = window[keep]
        if len(speech) < SAMPLE_RATE // 2:  # half a second, or the fbank is noise
            return None
        feats = fbank(speech)
        if feats.shape[0] < 25:
            return None
        vector = self.embedder.run(None, {self.emb_input: feats[None, ...]})[0][0]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None

    def run(self, wave: np.ndarray, speakers: int = 0, progress=None) -> Tuple[np.ndarray, float]:
        """Per-frame activity for the whole track, one column per person found."""
        windows = self._windows(wave)
        frame_seconds = WINDOW_SECONDS / FRAMES_PER_WINDOW
        total_frames = int(np.ceil(len(wave) / SAMPLE_RATE / frame_seconds)) + FRAMES_PER_WINDOW

        local: List[Tuple[int, int, np.ndarray]] = []  # window index, local speaker, activity
        vectors: List[np.ndarray] = []
        offsets: List[int] = []
        for index, (start, window) in enumerate(windows):
            active = self._activity(window)
            offsets.append(int(round(start / SAMPLE_RATE / frame_seconds)))
            for spk in range(active.shape[1]):
                if not active[:, spk].any():
                    continue
                # only frames where nobody else talks describe this voice cleanly
                alone = active[:, spk] & (active.sum(axis=1) == 1)
                vector = self._embed(window, alone if alone.sum() >= 20 else active[:, spk])
                if vector is None:
                    continue
                local.append((index, spk, active[:, spk]))
                vectors.append(vector)
            if progress:
                progress(int(80.0 * (index + 1) / len(windows)))

        if not vectors:
            return np.zeros((0, 0), dtype=bool), frame_seconds

        embeddings = np.stack(vectors)
        labels = cluster(embeddings, speakers)
        labels = settle(embeddings, labels)
        people = int(labels.max()) + 1
        timeline = np.zeros((total_frames, people), dtype=bool)
        for (index, _spk, activity), label in zip(local, labels):
            begin = offsets[index]
            end = min(begin + len(activity), total_frames)
            timeline[begin:end, label] |= activity[:end - begin]
        return timeline, frame_seconds


def cluster(vectors: np.ndarray, speakers: int = 0) -> np.ndarray:
    """Group voice embeddings into people, by distance or by a given count."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(vectors) == 1:
        return np.zeros(1, dtype=int)
    # Average linkage, on cosine distances.
    #
    # Centroid linkage was the obvious choice and the wrong one: on this
    # material it produced eleven inversions — merges happening at a smaller
    # distance than earlier ones — which makes a distance threshold meaningless.
    # Raising the threshold past 0.5 did not loosen the grouping, it collapsed
    # every voice into a single person. Average linkage is monotone, so the
    # threshold means what it says and the count falls off gently as it rises.
    tree = linkage(pdist(vectors, metric="cosine"), method="average")
    if speakers and speakers > 0:
        labels = fcluster(tree, t=speakers, criterion="maxclust")
    else:
        labels = fcluster(tree, t=CLUSTER_THRESHOLD, criterion="distance")
    return labels - 1


def settle(vectors: np.ndarray, labels: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Give every window a second hearing against the voices that were found.

    The tree that built the clusters judged each embedding against one other at
    a time, in the order the distances happened to fall. Once the voices are
    known, each window can be compared with all of them at once and moved to the
    one it actually sounds like — which is what stops a handful of borderline
    windows from dragging a stranger into somebody's turn.
    """
    for _ in range(rounds):
        centres = []
        for label in range(int(labels.max()) + 1):
            members = vectors[labels == label]
            if len(members) == 0:
                centres.append(np.zeros(vectors.shape[1], dtype=np.float32))
                continue
            centre = members.mean(axis=0)
            norm = np.linalg.norm(centre)
            centres.append(centre / norm if norm > 0 else centre)
        moved = np.argmax(vectors @ np.stack(centres).T, axis=1)
        if np.array_equal(moved, labels):
            break
        labels = moved
    return labels


def to_segments(timeline: np.ndarray, frame_seconds: float, min_duration: float = 0.5,
                fill_gap: float = 0.3) -> List[Dict]:
    """Turn per-frame activity into speech turns with a start and an end.

    Short holes inside a turn are filled and short turns are dropped: a person
    does not stop being the speaker because they drew breath, and a tenth of a
    second of somebody is a detection error, not a line of dialogue.
    """
    segments: List[Dict] = []
    gap_frames = int(round(fill_gap / frame_seconds))
    for person in range(timeline.shape[1]):
        active = timeline[:, person].copy()
        # close the small holes
        edges = np.flatnonzero(np.diff(active.astype(np.int8)))
        for i in range(0, len(edges) - 1):
            if not active[edges[i]] and (edges[i + 1] - edges[i]) <= gap_frames:
                active[edges[i] + 1:edges[i + 1] + 1] = True
        padded = np.concatenate(([False], active, [False]))
        changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for begin, end in zip(changes[0::2], changes[1::2]):
            start, stop = begin * frame_seconds, end * frame_seconds
            if stop - start >= min_duration:
                segments.append({"speaker": person, "start": round(start, 3), "end": round(stop, 3)})
    segments.sort(key=lambda s: (s["start"], s["speaker"]))
    return segments
