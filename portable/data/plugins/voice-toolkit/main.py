#!/usr/bin/env python3
"""Voice Toolkit — two ways to work on the voice in a clip.

The plugin brings two effects, and the right-click menu offers them separately:

  Clone Voice (effects/voice-toolkit-clone.xml)
      The clip says the same thing in a registered voice. The engine is the
      OpenVoice tone-colour converter: it converts rather than reads, so the
      words, the timing and the delivery stay the ones on the timeline and only
      who is speaking changes.
  Separator (effects/voice-toolkit-separator.xml)
      The voice on its own, or everything except the voice — one checkbox, one
      MDX model that answers both.

Neither runs while the timeline plays. Generate renders and the result lands in
the project bin and on an audio track of its own, leaving the clip it came from
untouched, so both effects are safe to try twice.

Three actions arrive over ``job.json``:
  input.action = "analyse"                        register a recording as a voice
  input.effect_id = "voice-toolkit.clone"         re-speak the clip in that voice
  input.effect_id = "voice-toolkit.separator"     split the clip

Both engines are ported from Wunjo v2 (portable/src/sound_processing).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

# stdout is reserved for the plugin protocol — send library chatter to stderr.
_real_print = print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)


import builtins

builtins.print = _stderr_print

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PLUGIN_DIR, "models")

#: A recording shorter than this describes a voice too poorly to be worth
#: keeping: the converter averages what it hears, and one word is one word said
#: one way. Ten to thirty seconds of ordinary speech is what it wants.
MIN_REFERENCE_SECONDS = 6.0
#: …and of that, this much has to be speech rather than room and pauses.
MIN_SPEECH_SECONDS = 3.0
#: The clip being converted needs enough sound to measure the voice it replaces.
MIN_CLIP_SECONDS = 1.0


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def flag(value: Any, default: bool = False) -> bool:
    """A settings-tab checkbox as a bool, however it survived the round trip."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no")
    return bool(value)


#: Which effect each model belongs to. Separating needs nothing from the
#: converter and cloning needs nothing from the separator, so a user who only
#: wants one half is not asked to download the other.
CLONE_MODELS = ("converter.pth", "config.json")
SEPARATOR_MODELS = ("MDX_Inst_HQ_3.onnx",)


def missing_model(effect_id: str = "", action: str = "") -> Optional[str]:
    """The first weight this job needs and does not have.

    Registering a voice measures it, so that needs the converter too.
    """
    wanted = SEPARATOR_MODELS if effect_id.endswith(".separator") and action != "analyse" else CLONE_MODELS
    for name in wanted:
        if not os.path.isfile(os.path.join(MODELS_DIR, name)):
            return name
    return None


def ffprobe_of(ffmpeg: str) -> str:
    return ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in os.path.basename(ffmpeg) else "ffprobe"


def audio_seconds(ffmpeg: str, path: str) -> float:
    """How long the track is, as the container reports it."""
    try:
        out = subprocess.check_output(
            [ffprobe_of(ffmpeg), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def decode_audio(ffmpeg: str, path: str, sample_rate: int, start: float = 0.0, seconds: Optional[float] = None):
    """The sound of @p path as one mono channel at the rate the model works at.

    Straight out of ffmpeg's pipe rather than through a temporary wav: the same
    decoder reads every container the editor can put on a track, and a clip is
    read once whatever it is.
    """
    import numpy as np

    command = [ffmpeg, "-v", "error"]
    if start > 0:
        command += ["-ss", "%.6f" % start]
    command += ["-i", path]
    if seconds is not None and seconds > 0:
        command += ["-t", "%.6f" % seconds]
    command += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    finished = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(finished.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError("there is no sound in %s" % os.path.basename(path))
    # Loud masters come out of the decoder above 1.0; the spectrogram expects a
    # signal that does not, and the mark is written into the samples themselves.
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio /= peak
    return audio


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple's GPU. torch on macOS ships with the Metal backend built in, so
        # asking only about CUDA left every Mac on its processor cores while the
        # graphics chip sat idle. Guarded because the attribute does not exist
        # in builds without it, and checked twice over: is_built() says the
        # backend was compiled in, is_available() that this machine has one.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_built() and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_with_smaller_chunks(work, seconds: float):
    """Run @p work, halving the piece each time the card says it is full.

    No estimate of free memory survives every machine — another program may take
    the card mid-render — so the fallback is part of the plan, not an accident.
    """
    while True:
        try:
            return work(seconds)
        except (RuntimeError, MemoryError) as error:
            if "out of memory" not in str(error).lower() or seconds <= 5:
                raise
            seconds = max(5.0, seconds / 2)
            emit("info:out of memory, retrying %.0f seconds at a time" % seconds)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


def load_engine(device: str):
    sys.path.insert(0, PLUGIN_DIR)
    from voice_clone import CloneVoice

    return CloneVoice(config_path=os.path.join(MODELS_DIR, "config.json"),
                      checkpoint_path=os.path.join(MODELS_DIR, "converter.pth"), device=device)


def preset_voice(set_path: Any) -> Dict[str, Any]:
    """The preset an effect points at, read before anything heavy loads."""
    path = str(set_path or "").strip()
    if not path or not os.path.isfile(path):
        raise ValueError("choose a voice first")
    with open(path, "r", encoding="utf-8") as handle:
        preset = json.load(handle)
    if not preset.get("embedding"):
        raise ValueError("this voice preset predates the current plugin — register the recording again")
    return preset


def run_analyse(job: dict) -> dict:
    """Register a recording as a voice.

    What is kept is the measurement, not the file: 256 numbers describing the
    timbre. The preset then works when the recording is moved or deleted, it
    travels between projects as a few kilobytes of json, and pressing Generate
    later costs nothing to look the voice up.
    """
    source = (job.get("input", {}).get("source") or "").strip()
    if not source or not os.path.isfile(source):
        raise ValueError("no recording to register")
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    settings = job.get("params", {})

    emit("info:reading the recording")
    emit("progress:10")
    seconds = audio_seconds(ffmpeg, source)
    if seconds <= 0:
        raise ValueError("this file has no sound")
    if seconds < MIN_REFERENCE_SECONDS:
        raise ValueError("a voice needs at least %d seconds to copy — this recording is %.1f s"
                         % (MIN_REFERENCE_SECONDS, seconds))

    device = pick_device(force_cpu=not flag(settings.get("use_gpu"), True))
    engine = load_engine(device)
    emit("progress:30")

    audio = decode_audio(ffmpeg, source, engine.sampling_rate)
    from voice_clone import voiced_audio

    speech = len(voiced_audio(audio, engine.sampling_rate)) / float(engine.sampling_rate)
    if speech < MIN_SPEECH_SECONDS:
        raise ValueError("only %.1f s of this recording is speech — at least %d s is needed, without music behind it"
                         % (speech, MIN_SPEECH_SECONDS))

    emit("info:measuring the voice")
    emit("progress:60")
    embedding = engine.measure(audio)

    out_path = os.path.join(job["output_dir"], "voice_preset.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"source": source, "count": int(round(seconds)), "embedding": embedding,
                   "sample_rate": engine.sampling_rate,
                   "data": {"seconds": round(seconds, 2), "speech": round(speech, 2)}}, handle)
    emit("progress:100")
    return {"outputs": [{"type": "data", "path": out_path}], "place": "none",
            "message": "Voice registered (%d:%02d, %.0f s of speech)." % (int(seconds) // 60, int(seconds) % 60, speech)}


def run_generate(job: dict) -> dict:
    """Re-speak the clip's sound in the voice the effect points at."""
    import numpy as np
    import soundfile

    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    asked = (job.get("input") or {}).get("effect_id", "")
    effect = next((e for e in effects if e.get("id") == asked), effects[0] if effects else {})
    params = effect.get("params", {})
    settings = job.get("params", {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    fps = float((job.get("project") or {}).get("fps") or 25.0)

    preset = preset_voice(params.get("cv_set"))
    start = int(clip.get("in") or 0)
    end = int(clip.get("out") or start)
    # The clip is a stretch of the file measured in timeline frames; the sound
    # under it is the same stretch in seconds.
    seconds = max(1, end - start + 1) / max(fps, 1e-6)
    if seconds < MIN_CLIP_SECONDS:
        raise ValueError("this clip is %.1f s — too short to hear a voice in, let alone replace one" % seconds)

    device = pick_device(force_cpu=not flag(settings.get("use_gpu"), True))
    emit("info:loading the converter on %s" % device)
    emit("progress:5")
    engine = load_engine(device)

    emit("info:reading the clip")
    emit("progress:15")
    audio = decode_audio(ffmpeg, clip["path"], engine.sampling_rate, start / max(fps, 1e-6), seconds)
    if float(np.abs(audio).max()) < 1e-4:
        raise ValueError("this clip is silent — there is nothing to say in another voice")

    emit("info:cloning the voice over %.1f s" % seconds)
    emit("progress:25")
    # Settle on the length of a pass here rather than inside the engine: what a
    # retry halves has to be a real number of seconds, and "choose by free
    # memory" halved is still "choose by free memory".
    from voice_clone import chunk_seconds_for

    piece_seconds = chunk_seconds_for(device, float(settings.get("chunk_seconds") or 0))
    spoken = run_with_smaller_chunks(
        lambda piece: engine.speak(audio, preset["embedding"], chunk_seconds=piece,
                                   progress=lambda percent: emit("progress:%d" % int(25 + 0.7 * max(0, min(100, percent))))),
        piece_seconds)

    out_path = os.path.join(job["output_dir"], "clone_voice.wav")
    soundfile.write(out_path, spoken, engine.sampling_rate)
    emit("progress:100")
    return {"outputs": [{"type": "audio", "path": out_path}],
            "message": "Voice cloned over %.1f s of sound." % (len(spoken) / float(engine.sampling_rate))}


def run_separate(job: dict) -> dict:
    """Split the clip's sound and keep the half the checkbox asks for."""
    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    asked = (job.get("input") or {}).get("effect_id", "")
    effect = next((e for e in effects if e.get("id") == asked), effects[0] if effects else {})
    params = effect.get("params", {})
    settings = job.get("params", {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    fps = float((job.get("project") or {}).get("fps") or 25.0)

    want = "voice" if flag(params.get("sp_voice"), True) else "music"
    start = int(clip.get("in") or 0)
    end = int(clip.get("out") or start)
    seconds = max(1, end - start + 1) / max(fps, 1e-6)

    sys.path.insert(0, PLUGIN_DIR)
    from separator import SAMPLE_RATE, separate

    with tempfile.TemporaryDirectory(prefix="wunjo-voice-toolkit-") as work:
        # The separator reads a file rather than an array, and the stretch under
        # the clip is not the whole file: cut it out first, at the rate the model
        # was trained for, so nothing downstream has to know about clip bounds.
        emit("info:reading the clip")
        emit("progress:5")
        source = os.path.join(work, "mix.wav")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "%.6f" % (start / max(fps, 1e-6)), "-i", clip["path"],
                        "-t", "%.6f" % seconds, "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), source],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        device = pick_device(force_cpu=not flag(settings.get("use_gpu"), True))
        emit("info:separating the %s on %s" % (want, device))
        emit("progress:10")
        produced = separate(os.path.join(MODELS_DIR, "MDX_Inst_HQ_3.onnx"), source, os.path.join(work, "stems"),
                            want=want, device=device,
                            progress=lambda percent, stage="": emit("progress:%d" % int(10 + 0.85 * max(0, min(100, percent)))))

        out_path = os.path.join(job["output_dir"], "separated_%s.wav" % want)
        shutil.move(produced, out_path)
    emit("progress:100")
    return {"outputs": [{"type": "audio", "path": out_path}],
            "message": "Separated the %s over %.1f s of sound." % (want, seconds)}


def process(job: dict) -> dict:
    """Which half of the toolkit was asked for.

    Cloning needs no scratch folder — sound arrives through a pipe and lives as
    an array until the one file it becomes. Separating does, because the model
    reads files; that folder is made and dropped inside run_separate.
    """
    action = (job.get("input") or {}).get("action", "")
    if action == "analyse":
        return run_analyse(job)
    if ((job.get("input") or {}).get("effect_id") or "").endswith(".separator"):
        return run_separate(job)
    return run_generate(job)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Path to job.json")
    args = parser.parse_args()

    try:
        with open(args.job, "r", encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, ValueError) as error:
        emit("info:failed to read job file: %s" % error)
        return 2

    action = job.get("input", {}).get("action")
    if action == "analyse":
        if not job.get("input", {}).get("source"):
            emit("info:missing a recording to register")
            return 2
    elif not job.get("input", {}).get("clips"):
        emit("info:missing input clip")
        return 2

    needed = missing_model(job.get("input", {}).get("effect_id") or "", action or "")
    if needed:
        emit("need:" + json.dumps({"kind": "model", "name": needed}))
        return 4

    sys.path.insert(0, PLUGIN_DIR)
    try:
        result = process(job)
    except MemoryError:
        emit("info:out of memory")
        return 5
    except ValueError as error:
        emit("info:%s" % error)
        return 2
    except Exception as error:
        _stderr_print("Voice Toolkit failed:", error)
        emit("info:Voice Toolkit failed: %s" % error)
        return 1

    emit("result:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
