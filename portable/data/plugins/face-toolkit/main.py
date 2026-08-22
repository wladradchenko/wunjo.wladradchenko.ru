#!/usr/bin/env python3
"""Face Toolkit — two ways to work on one detected face.

Picking the plugin from a face box puts three effects on the clip: one Face
Region that follows the face, and two ways of using it — animate it from a
recorded expression, or replace it with the face from a photo. They share this
environment, this detection and this queue; the user tunes whichever one they
came for and presses its Generate.

Making a face speak used to live here too, on Wav2Lip. It is gone: the Lip Sync
(LatentSync) plugin does the same job far better, and keeping a worse second
answer to the same question only makes the choice harder.

Actions arrive over ``job.json → input.action``:
  analyse   register a preset; ``input.kind`` says which sort (expression / face)
  generate  render what one effect describes; the effect id says which engine
Everything heavy lives on disk in the job's temporary folder, which goes away
with the job.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# stdout is reserved for the plugin protocol — send library chatter to stderr.
_real_print = print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)


import builtins

builtins.print = _stderr_print

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PLUGIN_DIR, "models")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

WEIGHT_FILES = [
    "landmark.onnx",
    "appearance_feature_extractor.pth",
    "motion_extractor.pth",
    "spade_generator.pth",
    "warping_module.pth",
    "stitching_retargeting_module.pth",
]

PARAM_LIMITS = {
    "eye_ratio": (0.0, 0.8),
    "lip_ratio": (0.0, 0.8),
    "pitch": (-90.0, 90.0),
    "yaw": (-90.0, 90.0),
    "roll": (-90.0, 90.0),
    "x": (-0.19, 0.19),
    "y": (-0.19, 0.19),
    "z": (0.9, 1.2),
    "eyeball_direction_x": (-30.0, 30.0),
    "eyeball_direction_y": (-60.0, 60.0),
    "smile": (-0.3, 1.3),
    "wink": (0.0, 39.0),
    "eyebrow": (-30.0, 30.0),
    "grin": (0.0, 15.0),
    "pursing": (-20.0, 15.0),
    "pouting": (-0.09, 0.09),
    "lip_expression": (-90.0, 90.0),
}


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def missing_model() -> Optional[str]:
    for name in WEIGHT_FILES:
        if not os.path.isfile(os.path.join(MODELS_DIR, name)):
            return name
    return None


def face_rect_to_crop(rect: List[float], width: int, height: int) -> Dict[str, int]:
    """Convert normalized [x, y, w, h] (0..1) to LivePortrait crop pixels."""
    x, y, w, h = [float(v) for v in rect]
    return {
        "x": int(round(x * width)),
        "y": int(round(y * height)),
        "width": int(round(w * width)),
        "height": int(round(h * height)),
        "naturalWidth": int(width),
        "naturalHeight": int(height),
    }


def probe_size(ffmpeg: str, path: str) -> Tuple[int, int]:
    """Return (width, height) via ffprobe next to ffmpeg, or OpenCV fallback."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in os.path.basename(ffmpeg) else "ffprobe"
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        w_s, h_s = out.split("x")
        return int(w_s), int(h_s)
    except Exception:
        import cv2

        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w <= 0 or h <= 0:
            raise RuntimeError("could not probe media size")
        return w, h


def extract_frame(ffmpeg: str, path: str, frame_index: int, out_path: str, fps: float = 25.0) -> None:
    # Seek by timestamp derived from frame index (editor positions are frames).
    t = max(0.0, float(frame_index) / max(fps, 1e-6))
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{t:.6f}",
        "-i",
        path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def working_size(source: Tuple[int, int], profile: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """The size frames are worked at, or None to leave the source alone.

    Nothing is gained by carrying pixels the project will not render: a 4K clip
    in a 1080p sequence costs four times the decoding, blending and encoding per
    frame, and the render throws three quarters of it away. The project profile
    decides this, not the user — it is the resolution the clip ends up at either
    way. Never upscales: a clip smaller than the profile is as good as it gets.
    """
    sw, sh = source
    pw, ph = profile
    if sw <= 0 or sh <= 0 or pw <= 0 or ph <= 0:
        return None
    # Fit inside the profile frame, the way the render fits it: a 3276x4096
    # photo in a 1920x1080 project is shown as 864x1080 with bars either side,
    # so that is all of it that ever reaches the screen. Measuring the longest
    # side against the longest side instead keeps three times the pixels the
    # project can use, and hands every model here a needlessly large face.
    scale = min(float(pw) / sw, float(ph) / sh)
    if scale >= 1.0:
        return None
    # even dimensions: yuv420p has no half pixels
    return (max(2, int(round(sw * scale)) // 2 * 2), max(2, int(round(sh * scale)) // 2 * 2))


def extract_frames_range(
    ffmpeg: str,
    path: str,
    start_frame: int,
    end_frame: Optional[int],
    out_dir: str,
    fps: float,
    size: Optional[Tuple[int, int]] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%08d.png")
    cmd = [ffmpeg, "-y"]
    if start_frame and start_frame > 0:
        start_t = float(start_frame) / max(fps, 1e-6)
        cmd += ["-ss", f"{start_t:.6f}"]
    cmd += ["-i", path]
    if end_frame is not None and end_frame > start_frame:
        duration = float(max(1, end_frame - start_frame)) / max(fps, 1e-6)
        cmd += ["-t", f"{duration:.6f}"]
    # Resample to the project's rate first. Everything downstream counts in
    # timeline frames — the region's box is keyframed there — so a 23.976 fps
    # clip in a 25 fps project must yield one frame per timeline frame, or the
    # box, the mouth and the sound all drift apart by the end of the clip.
    filters = ["fps=%.6f" % fps] if fps > 1e-6 else []
    if size:
        filters.append("scale=%d:%d:flags=bicubic" % size)
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-vsync", "0", pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    files = sorted(
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        raise RuntimeError("no frames extracted from source clip")
    return files


def place_still(path: str, target: str, size: Optional[Tuple[int, int]] = None) -> str:
    """Put a still where the engines read their frames from, at the working size."""
    if not size:
        shutil.copy2(path, target)
        return target
    import cv2

    image = cv2.imread(path)
    if image is None:
        shutil.copy2(path, target)
        return target
    cv2.imwrite(target, cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    return target


def mux_audio(ffmpeg: str, video_path: str, audio_src: str, out_path: str) -> str:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_src,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return out_path
    except subprocess.CalledProcessError:
        shutil.copy2(video_path, out_path)
        return out_path


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


def load_live_portrait(device: str, source_max_dim: int = 1280):
    from portrait_animation import LivePortrait

    return LivePortrait(
        source_max_dim=source_max_dim,
        checkpoint_F=os.path.join(MODELS_DIR, "appearance_feature_extractor.pth"),
        checkpoint_M=os.path.join(MODELS_DIR, "motion_extractor.pth"),
        checkpoint_G=os.path.join(MODELS_DIR, "spade_generator.pth"),
        checkpoint_W=os.path.join(MODELS_DIR, "warping_module.pth"),
        checkpoint_S=os.path.join(MODELS_DIR, "stitching_retargeting_module.pth"),
        insightface_root=PLUGIN_DIR,  # → <plugin>/models/buffalo_l
        landmark_ckpt_path=os.path.join(MODELS_DIR, "landmark.onnx"),
        device=device,
    )


def read_params(params: Dict[str, Any]) -> Dict[str, float]:
    defaults = {
        "eye_ratio": 0.0,
        "lip_ratio": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "x": 0.0,
        "y": 0.0,
        "z": 1.0,
        "eyeball_direction_x": 0.0,
        "eyeball_direction_y": 0.0,
        "smile": 0.0,
        "wink": 0.0,
        "eyebrow": 0.0,
        "grin": 0.0,
        "pursing": 0.0,
        "pouting": 0.0,
        "lip_expression": 0.0,
    }
    out = {}
    for key, default in defaults.items():
        lo, hi = PARAM_LIMITS[key]
        out[key] = clamp(params.get(key, default), lo, hi)
    return out


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# What extract_portrait_parameters measures, under the names the effect uses.
ANALYSE_KEYS = ["lp_eye_ratio", "lp_lip_ratio", "lp_pitch", "lp_yaw", "lp_roll"]

# The model reads these five as absolute values (eye_ratio 0 means shut eyes,
# pitch 0 means facing straight), not as offsets. Whatever a set or the effect
# timeline says is therefore added to what the source face is already doing —
# leave that out and everyone talks with their eyes closed.
SOURCE_RELATIVE = {"eye_ratio", "lip_ratio", "pitch", "yaw", "roll"}


def probe_fps(path: str, fallback: float) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        probed = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        if probed > 1e-3:
            return probed
    except Exception:
        pass
    return fallback


def frames_per_batch(requested: int, device: str, per_frame_mb: int = 48) -> int:
    """How many frames to hold at once.

    Asked of the hardware rather than guessed: a card that is already busy with
    a browser has far less to give than its specification says. The caller still
    halves this on an out-of-memory error — no estimate survives every machine.
    @param per_frame_mb what one frame costs in this engine; lip sync carries a
    mel window and a parsing mask per frame, so it costs more than a swap.
    """
    if requested > 0:
        return requested
    # Apple's GPU works out of the machine's own memory, so there is no separate
    # figure to divide up and torch.cuda.mem_get_info does not exist to ask. The
    # modest fixed batch the processor gets is the right answer there too.
    if device in ("cpu", "mps"):
        return 16
    try:
        import torch

        free, _total = torch.cuda.mem_get_info()
        # keep well inside what is free — the model itself is already loaded
        return max(4, min(64, int(free * 0.6 / (max(1, per_frame_mb) * 1024 * 1024))))
    except Exception:
        return 8


def load_swapper(device: str, nsfw_filter: bool, progress_callback=None):
    from face_swap import FaceSwapProcessing

    swapper = FaceSwapProcessing(
        model_path=MODELS_DIR,
        face_swap_model_path=os.path.join(MODELS_DIR, "faceswap.onnx"),
        device=device,
        progress_callback=progress_callback,
    )
    if not nsfw_filter:
        # The setting lives in the plugin's own preferences, not in the effect:
        # it is a decision about the installation, not about one clip.
        class _Allow:
            @staticmethod
            def status(_frame, classes=None):
                return True

        swapper.filter_model = _Allow()
    return swapper




def analyse_expression(job: dict, work: str) -> dict:
    """Record what a performance does, frame by frame.

    The editor calls this from the set panel behind the pen: the result is a
    value per frame for the parameters the effect exposes, stored as a set the
    user then picks in "Expression source". Values are relative to the first
    frame — the effect parameters are offsets applied to the target face, not
    absolute poses of the driving actor.
    """
    params = job.get("params", {})
    source = (job.get("input", {}).get("source") or "").strip()
    if not source or not os.path.isfile(source):
        raise ValueError("no source to analyse")
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    fps = float(job.get("project", {}).get("fps") or 25.0)

    if source.lower().endswith(IMAGE_SUFFIXES):
        files = [source]
        source_fps = fps
    else:
        emit("info:extracting frames")
        emit("progress:5")
        files = extract_frames_range(ffmpeg, source, 0, None, os.path.join(work, "frames"), fps)
        source_fps = probe_fps(source, fps)

    device = pick_device(force_cpu=bool(params.get("lp_force_cpu", params.get("force_cpu", False))))
    emit(f"info:loading LivePortrait on {device}")
    emit("progress:15")
    # The frame the engine gets is already fitted to the project; it reads no
    # more than 1920 across, so that pair is the whole answer.
    max_dim = min(1920, max(int(float(job.get("project", {}).get("width") or 0)),
                            int(float(job.get("project", {}).get("height") or 0))) or 1920)
    lp = load_live_portrait(device=device, source_max_dim=max_dim)
    pipeline = lp.live_portrait_pipeline
    lp.reset_face_analysis(pipeline)

    emit("info:measuring the performance")
    measured: List[List[float]] = []
    total = len(files)
    for index, path in enumerate(files):
        measured.append([float(v) for v in lp.extract_portrait_parameters(img_src=path, pipeline=pipeline, crop=None)])
        if index % 5 == 0 or index == total - 1:
            emit("progress:%d" % int(15 + 80.0 * (index + 1) / max(1, total)))

    origin = measured[0]
    values = {key: [round(frame[i] - origin[i], 4) for frame in measured] for i, key in enumerate(ANALYSE_KEYS)}

    out_path = os.path.join(job["output_dir"], "expression_set.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": source,
                "fps": source_fps,
                "count": total,
                "origin": {key: round(origin[i], 4) for i, key in enumerate(ANALYSE_KEYS)},
                "values": values,
            },
            handle,
        )
    emit("progress:100")
    return {"outputs": [{"type": "data", "path": out_path}], "place": "none", "message": "Recorded %d frames." % total}


def sample_animation(spec: Any, count: int, default: float = 0.0) -> List[float]:
    """Turn an effect parameter into one value per frame.

    The editor hands parameters over exactly as the effect holds them: a plain
    number when it never moves, or ``frame=value;frame=value`` when the user
    keyframed it. Values in between are interpolated, as the timeline draws them.
    """
    if spec is None or spec == "":
        return [default] * count
    text = str(spec)
    if "=" not in text:
        try:
            return [float(text)] * count
        except ValueError:
            return [default] * count
    points: List[Tuple[int, float]] = []
    for chunk in text.split(";"):
        if "=" not in chunk:
            continue
        frame, _, value = chunk.partition("=")
        try:
            points.append((int(float(frame.strip())), float(value.strip().split()[0])))
        except ValueError:
            continue
    if not points:
        return [default] * count
    points.sort()
    out = []
    for i in range(count):
        if i <= points[0][0]:
            out.append(points[0][1])
            continue
        if i >= points[-1][0]:
            out.append(points[-1][1])
            continue
        after = next(idx for idx, (f, _) in enumerate(points) if f >= i)
        f0, v0 = points[after - 1]
        f1, v1 = points[after]
        out.append(v0 if f1 == f0 else v0 + (v1 - v0) * (i - f0) / (f1 - f0))
    return out


def sample_rect_animation(spec: Any, count: int) -> List[Optional[List[float]]]:
    """Same, for an animated rectangle (``frame=x y w h``)."""
    text = str(spec or "")
    if "=" not in text:
        return [None] * count
    points: List[Tuple[int, List[float]]] = []
    for chunk in text.split(";"):
        if "=" not in chunk:
            continue
        frame, _, value = chunk.partition("=")
        numbers = [float(v) for v in value.replace("%", "").split() if v.strip()]
        if len(numbers) >= 4:
            try:
                points.append((int(float(frame.strip())), numbers[:4]))
            except ValueError:
                continue
    if not points:
        return [None] * count
    points.sort()
    out: List[Optional[List[float]]] = []
    for i in range(count):
        if i <= points[0][0]:
            out.append(points[0][1]); continue
        if i >= points[-1][0]:
            out.append(points[-1][1]); continue
        after = next(idx for idx, (f, _) in enumerate(points) if f >= i)
        f0, r0 = points[after - 1]
        f1, r1 = points[after]
        t = 0.0 if f1 == f0 else (i - f0) / (f1 - f0)
        out.append([a + (b - a) * t for a, b in zip(r0, r1)])
    return out


def profile_rect_to_crop(rect: List[float], profile: Tuple[int, int], source: Tuple[int, int]) -> Dict[str, int]:
    """Map a rectangle given in project-profile pixels onto the source media.

    The editor measures faces on the frame it composites, so a clip whose aspect
    differs from the project sits inside black bars there. Undo that padding
    before the rectangle means anything on the original file.
    """
    pw, ph = profile
    sw, sh = source
    if pw <= 0 or ph <= 0 or sw <= 0 or sh <= 0:
        return {"x": int(rect[0]), "y": int(rect[1]), "width": int(rect[2]), "height": int(rect[3]),
                "naturalWidth": sw, "naturalHeight": sh}
    x, y, w, h = [float(v) for v in rect[:4]]
    # the part of the profile frame the media actually covers
    box_x, box_y, box_w, box_h = 0.0, 0.0, float(pw), float(ph)
    source_dar = sw / sh
    profile_dar = pw / ph
    if source_dar < profile_dar - 1e-3:
        box_w = ph * source_dar
        box_x = (pw - box_w) / 2
    elif source_dar > profile_dar + 1e-3:
        box_h = pw / source_dar
        box_y = (ph - box_h) / 2
    scale_x = sw / box_w
    scale_y = sh / box_h
    return {
        "x": int(round((x - box_x) * scale_x)),
        "y": int(round((y - box_y) * scale_y)),
        "width": int(round(w * scale_x)),
        "height": int(round(h * scale_y)),
        "naturalWidth": sw,
        "naturalHeight": sh,
    }


def generate_portrait(job: dict, work: str) -> dict:
    """Render the clip with the face animated as the effect describes.

    Everything is kept on disk: source frames are extracted into the job's
    temporary folder, each rendered frame is written next to them, and ffmpeg
    turns the folder into a video. The folder goes away with the job.
    """
    import cv2

    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    region = next((e for e in effects if e.get("id", "").endswith(".region")), {})
    asked = (job.get("input") or {}).get("effect_id", "")
    animation = next((e for e in effects if e.get("id") == asked),
                     next((e for e in effects if not e.get("id", "").endswith(".region")), {}))
    params = animation.get("params", {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    project = job.get("project", {})
    fps = float(project.get("fps") or 25.0)
    profile = (int(project.get("width") or 0), int(project.get("height") or 0))

    start = int(clip.get("in") or 0)
    end = int(clip.get("out") or start)
    count = max(1, end - start + 1)

    emit("info:extracting frames")
    emit("progress:2")
    source_dir = os.path.join(work, "source")
    size = working_size(probe_size(ffmpeg, clip["path"]), profile)
    if size:
        emit("info:working at %dx%d — the project renders no more" % size)
    if clip["path"].lower().endswith(IMAGE_SUFFIXES):
        # a still: the same picture carries the whole animation
        os.makedirs(source_dir, exist_ok=True)
        still = place_still(clip["path"], os.path.join(source_dir, "source.png"), size)
        source_files = [still] * count
    else:
        extracted = extract_frames_range(ffmpeg, clip["path"], start, end + 1, source_dir, fps, size)
        source_files = [extracted[min(i, len(extracted) - 1)] for i in range(count)]
    source_size = probe_size(ffmpeg, source_files[0])

    # what the head does: the recorded set first, then whatever was keyframed on
    # top of it — the timeline is an addition, not a replacement
    recorded: Dict[str, List[float]] = {}
    set_path = (params.get("lp_set") or "").strip()
    if set_path and os.path.isfile(set_path):
        with open(set_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle).get("values", {})
        for key, values in stored.items():
            if values:
                recorded[key] = [float(values[min(i, len(values) - 1)]) for i in range(count)]
    manual = {key: sample_animation(params.get("lp_" + key), count, 1.0 if key == "z" else 0.0) for key in PARAM_LIMITS}
    crops = sample_rect_animation(region.get("params", {}).get("fm_face"), count)

    device = pick_device(force_cpu=str(params.get("lp_force_cpu", "0")) not in ("0", "", "false"))
    emit(f"info:loading LivePortrait on {device}")
    emit("progress:8")
    lp = load_live_portrait(device=device, source_max_dim=min(1920, max(profile) or 1920))
    pipeline = lp.live_portrait_pipeline
    lp.reset_face_analysis(pipeline)

    out_dir = os.path.join(work, "render")
    os.makedirs(out_dir, exist_ok=True)
    emit("info:animating")
    still = clip["path"].lower().endswith(IMAGE_SUFFIXES)
    measured: Optional[Dict[str, float]] = None
    for i in range(count):
        crop = profile_rect_to_crop(crops[i], profile, source_size) if crops[i] else None
        # What the source face is already doing. A still picture is measured
        # once; a video changes every frame and has to be read every frame.
        if measured is None or not still:
            eye, lip, pitch, yaw, roll = lp.extract_portrait_parameters(
                img_src=source_files[i], pipeline=pipeline, crop=(dict(crop) if crop else None)
            )
            measured = {"eye_ratio": eye, "lip_ratio": lip, "pitch": pitch, "yaw": yaw, "roll": roll}
        values = {}
        for key, (lo, hi) in PARAM_LIMITS.items():
            total = manual[key][i] + recorded.get("lp_" + key, [0.0] * count)[i]
            if key in SOURCE_RELATIVE:
                total += measured.get(key, 0.0)
            values[key] = clamp(total, lo, hi)
        frame = lp.update_image_portrait_parameters(
            img_src=source_files[i],
            pipeline=pipeline,
            eye_ratio=values["eye_ratio"],
            lip_ratio=values["lip_ratio"],
            pitch=values["pitch"],
            yaw=values["yaw"],
            roll=values["roll"],
            x=values["x"],
            y=values["y"],
            z=values["z"],
            eyeball_direction_x=values["eyeball_direction_x"],
            eyeball_direction_y=values["eyeball_direction_y"],
            smile=values["smile"],
            wink=values["wink"],
            eyebrow=values["eyebrow"],
            grin=values["grin"],
            pursing=values["pursing"],
            pouting=values["pouting"],
            lip_expression=values["lip_expression"],
            crop=(dict(crop) if crop else None),
        )
        cv2.imwrite(os.path.join(out_dir, "%08d.png" % i), frame[..., ::-1])
        emit("progress:%d" % int(8 + 85.0 * (i + 1) / count))

    emit("info:encoding")
    silent = os.path.join(work, "render.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-framerate", "%.6f" % fps, "-i", os.path.join(out_dir, "%08d.png"),
         "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", silent],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    out_path = os.path.join(job["output_dir"], "liveportrait.mp4")
    if clip["path"].lower().endswith(IMAGE_SUFFIXES):
        shutil.move(silent, out_path)
    else:
        # keep the clip's own sound
        mux_audio(ffmpeg, silent, clip["path"], out_path)
    emit("progress:100")
    return {"outputs": [{"type": "video", "path": out_path}], "message": "Live Portrait rendered %d frames." % count}


def analyse_face(job: dict, work: str) -> dict:
    """Register a photo as the face to swap in.

    One measurement is taken and kept: the 512-d identity vector, which is the
    second input of the swap model. Doing it here means rendering runs nothing
    per frame but the swap itself, and the preset stays valid on its own.
    """
    import cv2

    source = (job.get("input", {}).get("source") or "").strip()
    if not source or not os.path.isfile(source):
        raise ValueError("no photo to register")
    if not source.lower().endswith(IMAGE_SUFFIXES):
        raise ValueError(
            "a face preset is made from a photo, not a video: pass a still image "
            "(jpg or png) as the source. A frame taken out of the clip works.")

    emit("info:reading the photo")
    emit("progress:40")
    frame = cv2.imread(source)
    if frame is None:
        raise ValueError("could not read the photo")

    poster = os.path.join(job["output_dir"], "poster.png")
    height, width = frame.shape[:2]
    scale = 320.0 / max(1, max(height, width))
    cv2.imwrite(poster, cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale)))) if scale < 1 else frame)

    emit("info:measuring the face")
    emit("progress:70")
    sys.path.insert(0, PLUGIN_DIR)
    from regionface import ArcFace, RegionFaces

    device = pick_device(force_cpu=False)
    # onnxruntime has no Metal provider. CoreML is the Apple one and is only
    # present in some builds, so the processor stays behind it rather than the
    # session failing to open — and asking for CUDA on a Mac, which is what
    # "anything but cpu" used to do, finds a provider that is not there at all.
    if device == "cuda":
        providers = ["CUDAExecutionProvider"]
    elif device == "mps":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    # No box comes with a registered photo, so the landmark model works from the
    # middle of the picture — where the face of a portrait chosen for a swap is.
    faces = RegionFaces(landmark_model=os.path.join(MODELS_DIR, "landmark.onnx"), device=device, refine=True)
    kps = faces.get(frame)[0].kps
    embedding = ArcFace(os.path.join(MODELS_DIR, "w600k_r50.onnx"), providers).embed(frame, kps)

    out_path = os.path.join(job["output_dir"], "face_preset.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"source": source, "count": 1, "embedding": [float(v) for v in embedding],
                   "data": {"width": width, "height": height}}, handle)
    emit("progress:100")
    return {"outputs": [{"type": "data", "path": out_path}, {"type": "image", "path": poster}],
            "place": "none", "message": "Face preset ready."}


def generate_swap(job: dict, work: str) -> dict:
    """Render the clip with the preset's face on the region's face."""
    import cv2
    import numpy as np

    clip = job["input"]["clips"][0]
    effects = job["input"].get("effects") or []
    region = next((e for e in effects if e.get("id", "").endswith(".region")), {})
    asked = (job.get("input") or {}).get("effect_id", "")
    main_effect = next((e for e in effects if e.get("id") == asked),
                       next((e for e in effects if not e.get("id", "").endswith(".region")), {}))
    params = main_effect.get("params", {})
    settings = job.get("params", {})
    ffmpeg = job.get("ffmpeg") or "ffmpeg"
    project = job.get("project", {})
    fps = float(project.get("fps") or 25.0)
    profile = (int(project.get("width") or 0), int(project.get("height") or 0))

    preset_path = (params.get("fs_set") or "").strip()
    if not preset_path or not os.path.isfile(preset_path):
        raise ValueError("choose a face preset first")
    with open(preset_path, "r", encoding="utf-8") as handle:
        preset = json.load(handle)
    photo = preset.get("source") or ""
    if not os.path.isfile(photo):
        raise ValueError("the photo this preset was made from is gone: %s" % photo)

    start = int(clip.get("in") or 0)
    end = int(clip.get("out") or start)
    count = max(1, end - start + 1)
    still = clip["path"].lower().endswith(IMAGE_SUFFIXES)

    device = pick_device(force_cpu=str(settings.get("force_cpu", "0")) not in ("0", "", "false"))
    emit("info:loading the face swap model on %s" % device)
    emit("progress:3")
    swapper = load_swapper(device, nsfw_filter=bool(settings.get("nsfw_filter", True)))
    emit("progress:8")

    # The identity vector was measured once, when the photo was registered.
    stored = preset.get("embedding")
    if not stored:
        raise ValueError("this face preset predates the current plugin — register the photo again")
    source_face = np.asarray(stored, dtype=np.float32)

    emit("info:extracting frames")
    emit("progress:10")
    frames_dir = os.path.join(work, "frames")
    size = working_size(probe_size(ffmpeg, clip["path"]), profile)
    if size:
        emit("info:working at %dx%d — the project renders no more" % size)
    if still:
        os.makedirs(frames_dir, exist_ok=True)
        frame_files = [place_still(clip["path"], os.path.join(frames_dir, "00000000.png"), size)]
    else:
        frame_files = extract_frames_range(ffmpeg, clip["path"], start, end + 1, frames_dir, fps, size)
    source_size = probe_size(ffmpeg, frame_files[0])
    rects = sample_rect_animation(region.get("params", {}).get("fm_face"), len(frame_files))
    # The region follows the face across the shot, so it answers "which face"
    # on every frame, not only on the first one.
    crops = [profile_rect_to_crop(rect, profile, source_size) if rect else None for rect in rects]
    crop = next((area for area in crops if area), None)
    multiface = False

    out_dir = os.path.join(work, "render")
    os.makedirs(out_dir, exist_ok=True)
    emit("info:swapping")
    if still:
        frame = cv2.imread(frame_files[0])
        swapper.swap_image(frame, source_face, crop, out_dir, multiface=multiface)
        produced = os.path.join(out_dir, "swapped_image.png")
        if not os.path.isfile(produced):
            raise RuntimeError("the swap produced no image — no face found, or the frame was refused")
        out_path = os.path.join(job["output_dir"], "face_swap.png")
        shutil.move(produced, out_path)
        emit("progress:100")
        return {"outputs": [{"type": "image", "path": out_path}], "message": "Face swapped."}

    batch = frames_per_batch(int(float(settings.get("batch_size") or 0)), device)
    emit("info:%d frames, %d per batch" % (len(frame_files), batch))
    # the donor reports (percent, stage); map its own scale into ours
    swapper.progress_callback = lambda percent, stage="": emit("progress:%d" % int(10 + 0.8 * max(0, min(100, percent))))
    # The swapper takes a folder and names the file itself, returning the name.
    produced_name = swapper.swap_video(
        target_frames_path=frames_dir,
        source_face=source_face,
        target_face_fields=crops,
        save_file=out_dir,
        multiface=multiface,
        fps=fps,
    )
    silent = os.path.join(out_dir, produced_name or "")
    if not produced_name or not os.path.isfile(silent) or os.path.getsize(silent) == 0:
        raise RuntimeError("the swap produced no video — no face found in the clip, or every frame was refused")
    out_path = os.path.join(job["output_dir"], "face_swap.mp4")
    mux_audio(ffmpeg, silent, clip["path"], out_path)
    emit("progress:100")
    return {"outputs": [{"type": "video", "path": out_path}], "message": "Face swapped over %d frames." % len(frame_files)}


def audio_seconds(ffmpeg: str, path: str) -> float:
    """Length of the track, which is all a preset needs to describe itself."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in os.path.basename(ffmpeg) else "ffprobe"
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return 0.0




def process(job: dict) -> dict:
    """Send the job to the engine the effect asks for."""
    action = (job.get("input") or {}).get("action", "")
    with tempfile.TemporaryDirectory(prefix="wunjo-face-manipulation-") as work:
        if action == "analyse":
            kind = (job.get("input") or {}).get("kind", "expression")
            if kind == "face":
                return analyse_face(job, work)
            return analyse_expression(job, work)
        # Both effects live on the clip together, so the job says which of them
        # pressed Generate; the list of effects is context, not the answer.
        asked = (job.get("input") or {}).get("effect_id", "")
        if asked.endswith(".swap"):
            return generate_swap(job, work)
        return generate_portrait(job, work)


def needed_models(job: dict) -> Optional[str]:
    """Only the weights the chosen engine actually opens."""
    action = (job.get("input") or {}).get("action", "")
    # The detector the swap is aimed by. Without it it falls back to guessing
    # points inside the region's rectangle, which is exactly the mush this
    # plugin used to produce — so ask for it rather than degrade.
    detector = os.path.join("buffalo_l", "det_10g.onnx")
    if action == "analyse":
        # registering a photo measures it once, and that needs three models
        if (job.get("input") or {}).get("kind") == "face":
            for name in (detector, "landmark.onnx", "w600k_r50.onnx"):
                if not os.path.isfile(os.path.join(MODELS_DIR, name)):
                    return name
        return None
    asked = (job.get("input") or {}).get("effect_id", "")
    if asked.endswith(".swap"):
        wanted = ["faceswap.onnx", "landmark.onnx", detector]
    else:
        wanted = WEIGHT_FILES
    for name in wanted:
        if not os.path.isfile(os.path.join(MODELS_DIR, name)):
            return name
    return None


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
            emit("info:missing source to register")
            return 2
    elif not job.get("input", {}).get("clips"):
        emit("info:missing input clip")
        return 2

    needed = needed_models(job)
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
        # The trace goes to stderr, where the app's log keeps it: "failed" alone
        # tells nobody which of the three engines gave up and where.
        import traceback

        _stderr_print("Face Manipulation failed:", error)
        _stderr_print(traceback.format_exc())
        emit("info:Face Manipulation failed: %s" % error)
        return 1

    emit("result:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
