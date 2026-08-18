"""Utility helpers for api."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def frames_to_timecode(frames: int, fps: float = 25.0) -> str:
    """Convert a frame count to HH:MM:SS:FF timecode."""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    remaining_frames = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remaining_frames:02d}"


def timecode_to_frames(timecode: str, fps: float = 25.0) -> int:
    """Convert HH:MM:SS:FF timecode to frame count."""
    parts = timecode.split(":")
    if len(parts) == 4:
        h, m, s, f = [int(p) for p in parts]
    elif len(parts) == 3:
        h, m, s = [int(p) for p in parts]
        f = 0
    else:
        raise ValueError(f"Invalid timecode format: {timecode}")
    return int((h * 3600 + m * 60 + s) * fps + f)


def seconds_to_frames(seconds: float, fps: float = 25.0) -> int:
    """Convert seconds to frame count."""
    return int(round(seconds * fps))


def frames_to_seconds(frames: int, fps: float = 25.0) -> float:
    """Convert frame count to seconds."""
    return frames / fps


def parse_script_scenes(script_path: str) -> list[dict]:
    """Parse script-scenes.md to extract scene metadata.

    Returns a list of dicts with keys:
        - number: int (scene number, 1-based)
        - title: str (scene title from ### header)
        - section: str (e.g. "INTRO", "VERSE 1", "DROP")
        - kadr: str (framing description)
        - poza: str (pose description)
        - nastroj: str (mood)
        - ambient: str (ambient motion description)
        - kamera: str (camera description)
    """
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    scenes = []
    current_section = ""

    # Split by scene headers
    lines = content.split("\n")
    scene_re = re.compile(r"^###\s+Scena\s+(\d+)\s*[—–-]\s*(.+)$")
    section_re = re.compile(r"^##\s+(.+?)(?:\s*[—–-]|$)")
    field_re = re.compile(r"^\*\*(.+?):\*\*\s*(.+)$")

    current_scene = None
    for line in lines:
        line = line.strip()

        # Section header
        m = section_re.match(line)
        if m:
            current_section = m.group(1).strip()
            continue

        # Scene header
        m = scene_re.match(line)
        if m:
            if current_scene:
                scenes.append(current_scene)
            current_scene = {
                "number": int(m.group(1)),
                "title": m.group(2).strip(),
                "section": current_section,
                "kadr": "",
                "poza": "",
                "nastroj": "",
                "ambient": "",
                "kamera": "",
            }
            continue

        # Field
        if current_scene:
            m = field_re.match(line)
            if m:
                key = m.group(1).lower().strip()
                value = m.group(2).strip()
                if key == "kadr":
                    current_scene["kadr"] = value
                elif key == "poza":
                    current_scene["poza"] = value
                elif key in ("nastrój", "nastroj"):
                    current_scene["nastroj"] = value
                elif key == "ambient motion":
                    current_scene["ambient"] = value
                elif key == "kamera":
                    current_scene["kamera"] = value

    if current_scene:
        scenes.append(current_scene)

    return scenes


def find_scene_video(output_dir: str, scene_number: int,
                     variant: str = "A") -> str | None:
    """Find the output video file for a given scene number and variant.

    Searches for files like: scene01-A.mp4, scene01-A-seed1.mp4, etc.
    Returns the first match or None.
    """
    pattern = f"scene{scene_number:02d}-{variant}*.mp4"
    import glob as globmod
    matches = sorted(globmod.glob(os.path.join(output_dir, pattern)))
    return matches[0] if matches else None


def collect_scene_videos(output_dir: str, num_scenes: int = 38,
                         variant: str = "A") -> list[str | None]:
    """Collect video file paths for all scenes.

    Returns a list of length num_scenes where each element is either
    a file path or None if not found.
    """
    return [find_scene_video(output_dir, i + 1, variant)
            for i in range(num_scenes)]
