#!/usr/bin/env python3
"""Cut Finder — where the cuts are, found rather than scrubbed for.

Two effects, and a clip only offers the ones that make sense on it: a silent
video can be cut by scene, an audio clip by who is talking, and a video with
sound offers both so the user picks the one they came for.

  Speaker Cuts      diarisation: which voice holds the floor, second by second.
                    Each person ends up on a track of their own.
  Scene Cuts        where the picture changes enough to be a different shot.
                    These cuts stay on the one track — a shot list, not layers.

Neither renders anything. They answer with timings, in a data file the editor
reads and turns into cuts; the audio and the picture are never re-encoded, so
nothing is lost and the work is undoable in one step.

Actions arrive over ``job.json → input.action``:
  generate  analyse the clip; the effect id says which question is being asked
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

# stdout is reserved for the plugin protocol — send library chatter to stderr.
_real_print = print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)


import builtins

builtins.print = _stderr_print

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PLUGIN_DIR, "models")


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def needed_models(job: dict) -> Optional[str]:
    """Only what the chosen question needs: scenes ask ffmpeg, not a model."""
    if not (job.get("input") or {}).get("effect_id", "").endswith(".speakers"):
        return None
    for name in ("segmentation.onnx", "embedding.onnx"):
        if not os.path.isfile(os.path.join(MODELS_DIR, name)):
            return name
    return None


def clip_range(clip: dict, fps: float) -> tuple:
    """The piece of the source this clip actually uses, in seconds.

    The editor gives in and out as timeline frames. Everything here answers in
    seconds counted from the start of that piece, because that is where the
    editor will place the cuts — a timing measured from the head of a long
    source file would land nowhere near the clip on screen.
    """
    if fps <= 0:
        return 0.0, 0.0
    start = max(0, int(clip.get("in") or 0)) / fps
    out = clip.get("out")
    span = ((int(out) + 1) / fps - start) if out is not None else 0.0
    return start, max(0.0, span)


def clip_seconds(ffmpeg: str, path: str) -> float:
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in os.path.basename(ffmpeg) else "ffprobe"
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def apply_to_timeline(job: dict, payload: dict) -> str:
    """Put the findings where the user can see them, and say what was done.

    Markers by default: a note on the footage that changes nothing and can be
    ignored. Cutting is the other choice, and it is a choice — a plugin that
    rearranged the timeline the moment it was asked a question would be a poor
    guest.
    """
    sys.path.insert(0, PLUGIN_DIR)
    from editor import Editor

    clip = job["input"]["clips"][0]
    fps = float((job.get("project") or {}).get("fps") or 25.0)
    editor = Editor()
    if not editor.available:
        return "the editor did not answer, so nothing was placed"

    bin_id = str(clip.get("bin_id") or "")
    clip_in = int(clip.get("in") or 0)
    marks = payload.get("cuts") or [s["start"] for s in payload.get("segments", [])]
    action = payload.get("action", "markers")

    if action != "cut":
        # Markers live on the footage, so they are placed in the source's own
        # frames — no need to find where the clip sits on the timeline.
        placed = 0
        for index, moment in enumerate(marks):
            label = payload.get("labels", {}).get(str(index), "")
            if editor.add_clip_marker(bin_id, int(round((clip_in / fps + moment) * fps)),
                                      label or "Cut %d" % (index + 1), 0):
                placed += 1
        return "%d markers placed on the clip" % placed

    found = editor.find_clip(bin_id, clip_in)
    if not found:
        return "could not find this clip on the timeline, so nothing was cut"
    if payload.get("kind") == "scenes":
        # Back to front. A cut leaves the left half holding the original id, so
        # cutting forwards walks off the end of it after the first one: every
        # later point falls in a piece this id no longer names. Going backwards,
        # the head keeps every point still to come.
        cut = 0
        for moment in sorted(marks, reverse=True):
            if editor.cut_clip(found["id"], int(round(found["position"] + moment * fps))):
                cut += 1
        return "%d cuts made" % cut

    # Speakers: a pair of tracks per voice — the picture belongs to whoever is
    # speaking as much as the sound does, and a turn with the video thrown away
    # is not something anyone can edit with. Video and audio go in as two
    # insertions because a whole A/V clip asks the timeline to hunt for a target
    # track; naming both tracks outright asks nothing.
    made = 0
    tracks = {}
    for segment in payload.get("segments", []):
        person = segment["speaker"]
        if person not in tracks:
            name = "SPEAKER %d" % (person + 1)
            tracks[person] = (editor.add_track(name, False), editor.add_track(name, True))
        video_track, audio_track = tracks[person]
        start = int(round(clip_in + segment["start"] * fps))
        stop = int(round(clip_in + segment["end"] * fps))
        where = int(round(found["position"] + segment["start"] * fps))
        placed = False
        if video_track > -1:
            placed |= editor.insert_clip("V%s/%d/%d" % (bin_id, start, stop), video_track, where) > -1
        if audio_track > -1:
            placed |= editor.insert_clip("A%s/%d/%d" % (bin_id, start, stop), audio_track, where) > -1
        made += 1 if placed else 0
    # the original is left in place but silenced, or every line is heard twice
    for track in editor.tracks():
        if track["audio"] and track["id"] == found.get("trackId"):
            editor.set_track_mute(track["id"], True)
    return "%d turns placed on %d speakers" % (made, len(tracks))


def write_result(job: dict, payload: dict, message: str) -> dict:
    """The answer as a data file. The editor does the cutting, not the plugin.

    A plugin that reached into the editor itself would be unreplayable — the
    whole job could no longer be re-run from its job.json — and every cut it
    made would be its own step in the undo history. Handing back the timings
    keeps this a function of its input, and lets the edit be one action.
    """
    out_path = os.path.join(job["output_dir"], "segments.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    emit("progress:100")
    return {"outputs": [{"type": "data", "path": out_path}], "place": "none", "message": message}


def find_scene_cuts(job: dict, work: str) -> dict:
    """Where the picture changes enough to call it another shot.

    ffmpeg's own scene score, so there is no model to download and nothing to
    run on a graphics card. The score is the fraction of the frame that changed;
    the threshold is what the user tunes when a dissolve reads as a cut, or a
    pan does not.
    """
    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    params = next((e.get("params", {}) for e in effects if e.get("id", "").endswith(".scenes")), {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    fps = float((job.get("project") or {}).get("fps") or 25.0)
    threshold = max(0.05, min(0.95, float(params.get("sc_threshold") or 0.30)))
    min_shot = max(0.0, float(params.get("sc_min_shot") or 1.0))
    # Only the piece the clip actually uses. A twenty-six minute source trimmed
    # down to a minute on the timeline would otherwise be decoded end to end —
    # minutes of work for a question about one minute of it — and the cuts would
    # come back numbered from the head of the file, where nothing is.
    start, span = clip_range(clip, fps)

    emit("info:looking for cuts")
    emit("progress:5")
    command = [ffmpeg, "-v", "error"]
    if start > 0:
        command += ["-ss", "%.6f" % start]
    command += ["-i", clip["path"]]
    if span > 0:
        command += ["-t", "%.6f" % span]
    command += ["-vf", "select='gt(scene,%f)',metadata=print:file=-" % threshold, "-an", "-f", "null", "-"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cuts: List[float] = []
    for match in re.finditer(r"pts_time:([0-9.]+)", proc.stdout):
        moment = float(match.group(1))
        # a cut a frame after the last one is the same cut seen twice
        if not cuts or moment - cuts[-1] >= max(min_shot, 1.0 / max(fps, 1e-6)):
            cuts.append(round(moment, 3))

    emit("progress:90")
    duration = span if span > 0 else clip_seconds(ffmpeg, clip["path"])
    payload = {
        "kind": "scenes",
        "source": clip["path"],
        "fps": fps,
        "duration": round(duration, 3),
        # Cuts, not ranges: these all stay on the one track, so the editor has
        # only to split the clip where they fall.
        "cuts": cuts,
    }
    payload["action"] = "cut" if str(params.get("sc_cut", "1")) not in ("0", "", "false", "False") else "markers"
    note = apply_to_timeline(job, payload)
    return write_result(job, payload, "Found %d cuts — %s." % (len(cuts), note))


def find_speaker_cuts(job: dict, work: str) -> dict:
    """Who holds the floor, second by second."""
    import numpy as np

    sys.path.insert(0, PLUGIN_DIR)
    from diarize import Diarizer, load_audio, to_segments

    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    params = next((e.get("params", {}) for e in effects if e.get("id", "").endswith(".speakers")), {})
    settings = job.get("params", {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    fps = float((job.get("project") or {}).get("fps") or 25.0)
    speakers = max(0, int(float(params.get("sp_count") or 0)))
    min_turn = max(0.1, float(params.get("sp_min_turn") or 0.5))

    emit("info:reading the track")
    emit("progress:3")
    start, span = clip_range(clip, fps)
    wave = load_audio(clip["path"], ffmpeg, start=start, span=span)
    if wave.size < 16000:
        raise ValueError("this clip is too short to tell voices apart")

    device = "cpu"
    if str(settings.get("use_gpu", "0")) not in ("0", "", "false", "False"):
        device = "cuda"
    emit("info:listening for voices")
    emit("progress:8")
    diarizer = Diarizer(MODELS_DIR, device=device)
    timeline, frame_seconds = diarizer.run(
        wave, speakers=speakers, progress=lambda pct: emit("progress:%d" % int(8 + 0.8 * pct)))
    if timeline.size == 0:
        raise ValueError("no speech found in this clip")

    segments = to_segments(timeline, frame_seconds, min_duration=min_turn)
    # A voice that holds the floor for a second across the whole recording is a
    # mistake, not a person: it drags stray windows into a name of their own and
    # makes the list unreadable. Fold those turns into silence rather than
    # inventing a speaker for them.
    speech = {}
    for segment in segments:
        speech[segment["speaker"]] = speech.get(segment["speaker"], 0.0) + segment["end"] - segment["start"]
    keep = {person for person, held in speech.items() if held >= max(2.0, 0.02 * sum(speech.values()))}
    if keep:
        segments = [s for s in segments if s["speaker"] in keep]
    people = sorted({s["speaker"] for s in segments})
    # number the people in the order they first speak, so SPEAKER 1 opens
    order = {person: index for index, person in enumerate(
        sorted(people, key=lambda p: min(s["start"] for s in segments if s["speaker"] == p)))}
    for segment in segments:
        segment["speaker"] = order[segment["speaker"]]
    segments.sort(key=lambda s: (s["start"], s["speaker"]))

    payload = {
        "kind": "speakers",
        "source": clip["path"],
        "fps": fps,
        "duration": round(len(wave) / 16000.0, 3),
        "speakers": len(order),
        # One track per person; the ranges say what to put on it and where.
        # Overlapping speech belongs to everyone talking, so the same moment can
        # appear under more than one speaker — that is the point of the tracks.
        "segments": segments,
    }
    payload["action"] = "cut" if str(params.get("sp_split", "1")) not in ("0", "", "false", "False") else "markers"
    payload["labels"] = {str(i): "SPEAKER %d" % (s["speaker"] + 1) for i, s in enumerate(segments)}
    note = apply_to_timeline(job, payload)
    return write_result(job, payload, "%d speakers, %d turns — %s." % (len(order), len(segments), note))


def process(job: dict) -> dict:
    asked = (job.get("input") or {}).get("effect_id", "")
    with tempfile.TemporaryDirectory(prefix="wunjo-cut-finder-") as work:
        if asked.endswith(".speakers"):
            return find_speaker_cuts(job, work)
        return find_scene_cuts(job, work)


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

    if not job.get("input", {}).get("clips"):
        emit("info:missing input clip")
        return 2

    needed = needed_models(job)
    if needed:
        emit("need:" + json.dumps({"kind": "model", "name": needed}))
        return 4

    try:
        result = process(job)
    except MemoryError:
        emit("info:out of memory")
        return 5
    except ValueError as error:
        emit("info:%s" % error)
        return 2
    except Exception as error:
        import traceback

        _stderr_print("Cut Finder failed:", error)
        _stderr_print(traceback.format_exc())
        emit("info:Cut Finder failed: %s" % error)
        return 1

    emit("result:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
