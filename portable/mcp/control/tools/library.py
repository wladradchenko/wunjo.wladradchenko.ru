"""MCP tools: looking at a folder of footage before touching the project.

An edit starts with material somebody dropped in a folder, and the names tell
you nothing — `IMG_20241013_181303.jpg` could be anything. So there are three
tools here, and they are meant to be used in this order:

  list_media_folder   what is in the folder, and which of it has been looked at
  describe_media      what is actually in one file, in words
  find_media          the files whose description matches what you are after

The looking is done by a vision model, not by the assistant: the assistant reads
the descriptions. That keeps a folder of five hundred files from having to travel
through the conversation as five hundred pictures, and it means the same file is
only ever looked at once — descriptions are kept in a store keyed by the file's
identity, so a second project, or a second question about the same folder, is
answered from what was already learned.

Only media is listed. This is not a way to read the disk: a file that is not a
picture, a video or a sound is not reported, and nothing here returns the bytes
of a file — only what a model said it saw.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import Context

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_TYPES:
        return "image"
    if suffix in VIDEO_TYPES:
        return "video"
    if suffix in AUDIO_TYPES:
        return "audio"
    return ""


def _store() -> Path:
    """Where descriptions are kept — beside the app's data, not the project.

    The point of the store is that the work is not repeated, and footage is
    reused across projects, so it cannot live inside one of them.
    """
    root = os.environ.get("WUNJO_MEDIA_NOTES")
    if not root:
        root = str(Path.home() / ".local" / "share" / "wunjo" / "media-notes")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _identity(path: Path) -> str:
    """A name for the file's content that is cheap to compute.

    Size and modification time, not a hash of the bytes: hashing a nine-gigabyte
    video to decide whether it has been seen before costs more than looking at
    it again. Re-encode a file and it is a different file, which is right.
    """
    stat = path.stat()
    seed = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _note_path(identity: str) -> Path:
    return _store() / f"{identity}.json"


def _read_note(path: Path) -> dict:
    try:
        with open(_note_path(_identity(path)), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_note(path: Path, description: str) -> None:
    note = {"file": str(path.resolve()), "kind": _kind(path),
            "description": description, "seen": int(time.time())}
    try:
        with open(_note_path(_identity(path)), "w", encoding="utf-8") as handle:
            json.dump(note, handle, ensure_ascii=False)
    except OSError:
        pass


def _ffmpeg() -> str:
    return os.environ.get("WUNJO_FFMPEG") or shutil.which("ffmpeg") or "/app/bin/ffmpeg"


def _probe(path: Path) -> dict:
    """Length and size of a video, as far as ffprobe can tell."""
    probe = shutil.which("ffprobe") or str(Path(_ffmpeg()).with_name("ffprobe"))
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration:stream=width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=90).stdout
        data = json.loads(out or "{}")
        stream = (data.get("streams") or [{}])[0]
        return {"duration": float(data.get("format", {}).get("duration") or 0),
                "width": int(stream.get("width") or 0),
                "height": int(stream.get("height") or 0)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"duration": 0.0, "width": 0, "height": 0}


def _scene_scores(path: Path, samples_per_second: float = 4.0) -> list:
    """How much the picture changes, sampled along the whole file.

    ffmpeg's own scene score, which is the difference between one sample and
    the one before it. A sustained middling value is movement; a single spike is
    a cut. Measuring it needs no model and no extra dependency — the frames are
    scaled down first, so a long take costs seconds.
    """
    out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False,
                                      dir=str(Path.home() / ".cache"))
    out.close()
    try:
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path),
             "-vf", f"fps={samples_per_second},scale=256:-2,select='gte(scene,0)',"
                    f"metadata=print:key=lavfi.scene_score:file={out.name}",
             "-an", "-f", "null", "-"],
            capture_output=True, timeout=1800)
        series = []
        at = 0.0
        with open(out.name, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("frame:") and "pts_time:" in line:
                    try:
                        at = float(line.split("pts_time:")[1].split()[0])
                    except (IndexError, ValueError):
                        continue
                elif line.startswith("lavfi.scene_score="):
                    try:
                        series.append((at, float(line.split("=")[1])))
                    except (IndexError, ValueError):
                        continue
        # The first sample has nothing before it to differ from, so whatever
        # ffmpeg reports for it is not a measurement of anything. Left in, it
        # wins every time and sends the assistant to look at the opening frame
        # of every file it is ever given.
        return series[1:]
    except (OSError, subprocess.SubprocessError):
        return []
    finally:
        try:
            os.unlink(out.name)
        except OSError:
            pass


def _loudness(path: Path) -> dict:
    """Sound level per second, on a 0..1 scale. Empty when there is no audio."""
    out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False,
                                      dir=str(Path.home() / ".cache"))
    out.close()
    try:
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-af",
             "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:"
             f"file={out.name}", "-f", "null", "-"],
            capture_output=True, timeout=1800)
        levels = {}
        at = 0.0
        with open(out.name, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("frame:") and "pts_time:" in line:
                    try:
                        at = float(line.split("pts_time:")[1].split()[0])
                    except (IndexError, ValueError):
                        continue
                elif "RMS_level=" in line:
                    try:
                        levels[int(at)] = float(line.split("=")[-1])
                    except ValueError:
                        continue
        if not levels:
            return {}
        floor, ceiling = min(levels.values()), max(levels.values())
        span = (ceiling - floor) or 1.0
        return {second: (level - floor) / span for second, level in levels.items()}
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        try:
            os.unlink(out.name)
        except OSError:
            pass


def _rank_windows(change: list, loud: dict, total: float, want: int, window: float) -> str:
    """The stretches worth looking at, with what made each one stand out.

    Windows overlap so a moment is not sliced down the middle, and the ones that
    win exclude their neighbours — twenty candidates from the same ten seconds
    would be one candidate wearing twenty hats.
    """
    # What counts as "a lot of movement" is a property of the footage, not a
    # number that can be written down here: a locked-off camera in a barn and a
    # handheld chase differ by an order of magnitude, and a fixed threshold
    # would call everything in one of them remarkable and nothing in the other.
    # So each file is judged against itself.
    values = sorted(value for _when, value in change)
    def _at_rank(fraction: float) -> float:
        if not values:
            return 1.0
        return values[min(len(values) - 1, int(len(values) * fraction))]

    busy_here = _at_rank(0.80)
    cut_here = max(_at_rank(0.98), 0.10)

    # Some footage simply has no moments in it: a locked-off camera on an empty
    # yard changes by a thousandth from one second to the next for an hour.
    # Ranking that produces a tidy table of noise, which reads as if something
    # was found. Saying there is nothing is the more useful answer, and the
    # honest one.
    if _at_rank(0.98) < 0.02:
        return ("Nothing stands out: the picture barely changes from one second to the "
                "next, which is what a locked-off camera on a quiet scene looks like. "
                "There are no moments to choose between here — use the clip whole, cut "
                "it to length, or look for what you want some other way.")

    scored = []
    at = 0.0
    while at < max(total - window / 2, window):
        inside = [value for when, value in change if at <= when < at + window]
        if inside:
            movement = sum(inside) / len(inside)
            spike = max(inside)
            volume = (sum(loud.get(int(at) + offset, 0.0) for offset in range(max(1, int(window))))
                      / max(1, int(window))) if loud else 0.0
            score = 3.0 * movement + 1.8 * spike + 0.7 * volume
            reasons = []
            if spike >= cut_here:
                reasons.append("the shot changes")
            if movement >= busy_here:
                reasons.append("more happens here than elsewhere")
            if loud and volume > 0.7:
                reasons.append("the sound peaks")
            scored.append((score, at, min(total, at + window),
                           reasons or ["quietly among the busier stretches"]))
        at += window / 2.0

    scored.sort(key=lambda row: row[0], reverse=True)
    chosen = []
    for score, start, end, reasons in scored:
        if len(chosen) >= max(1, want):
            break
        if any(not (end <= taken[1] or start >= taken[2]) for taken in chosen):
            continue
        chosen.append((score, start, end, reasons))
    chosen.sort(key=lambda row: row[1])

    if not chosen:
        return "Nothing stood out; the picture barely changes."
    peak = chosen[0][0] or 1.0
    for row in chosen:
        peak = max(peak, row[0])
    lines = ["%6.2f – %6.2f  %.2f  %s" % (start, end, score / peak, ", ".join(reasons))
             for score, start, end, reasons in chosen]
    return ("start – end   score  what stood out\n" + "\n".join(lines)
            + "\n\nLook at a few of these (describe_media with `at`) before choosing.")


def _still_at(path: Path, at: float) -> str:
    """One frame from a chosen second, as a temporary JPEG."""
    target = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False,
                                         dir=str(Path.home() / ".cache")).name
    subprocess.run([_ffmpeg(), "-y", "-ss", f"{max(0.0, at):.2f}", "-i", str(path),
                    "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", target],
                   capture_output=True, timeout=180)
    return target if os.path.exists(target) and os.path.getsize(target) else ""


def _stills_from_video(path: Path, count: int = 4) -> list:
    """Frames spread across a video, as temporary JPEGs.

    One frame is not a video. A single still from the middle says nothing about
    what happens — a talking head and a chase scene can share it — so several
    are taken along the length and shown together, and the model is told they
    are in order. Four is enough to tell movement from a static shot without
    the cost of looking at a whole film.
    """
    duration = _probe(path).get("duration") or 0.0
    if duration <= 0:
        offsets = [1.0]
    else:
        # Skip the very edges: first and last frames are often black or a slate.
        span = duration * 0.9
        start = duration * 0.05
        offsets = [start + span * i / max(1, count - 1) for i in range(count)]

    frames = []
    for seek in offsets:
        target = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        subprocess.run([_ffmpeg(), "-y", "-ss", f"{seek:.2f}", "-i", str(path),
                        "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", target],
                       capture_output=True, timeout=180)
        if os.path.getsize(target) if os.path.exists(target) else 0:
            frames.append(target)
        else:
            try:
                os.unlink(target)
            except OSError:
                pass
    return frames


def _ask_vision(image_paths, prompt: str) -> str:
    """Show one or more pictures to the vision model and return what it says.

    Several at once is how a video is understood: the frames go in order in the
    same message, so the answer can be about what happens rather than about one
    moment of it.
    """
    base = os.environ.get("WUNJO_VISION_URL", "").rstrip("/")
    if not base:
        return "ERROR: no vision model is configured"
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    content = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    payload = {
        "model": os.environ.get("WUNJO_VISION_MODEL", "wunjo-local"),
        "max_tokens": 220,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(f"{base}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.load(response)
    return (answer["choices"][0]["message"].get("content") or "").strip()


def register(mcp, helpers):

    @mcp.tool()
    def list_media_folder(directory: str, pattern: str = "*") -> str:
        """List the media in a folder: what is there, and what is known about it.

        Only pictures, video and sound are reported. The "known" column says
        whether the file has already been looked at — call describe_media for
        the ones that have not, then decide what to import.

        Args:
            directory: Folder to look in.
            pattern: Optional glob, e.g. "*.mp4" or "IMG_*".
        """
        try:
            root = Path(directory).expanduser()
            if not root.is_dir():
                return f"ERROR: {directory} is not a folder"
            rows, skipped = [], 0
            for entry in sorted(root.glob(pattern)):
                if not entry.is_file():
                    continue
                kind = _kind(entry)
                if not kind:
                    skipped += 1
                    continue
                note = _read_note(entry)
                size = entry.stat().st_size
                rows.append((entry.name, kind, f"{size / 1_000_000:.1f} MB",
                             "yes" if note.get("description") else "no"))
            if not rows:
                return f"No media in {directory} matching {pattern}."
            table = ["| file | kind | size | described |", "|---|---|---|---|"]
            table += [f"| {n} | {k} | {s} | {d} |" for n, k, s, d in rows]
            tail = f"\n\n{len(rows)} media files"
            if skipped:
                tail += f"; {skipped} other files in the folder are not media and are not listed"
            return "\n".join(table) + tail
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def describe_media(path: str, question: str = "", at: float = -1.0) -> str:
        """Look at a picture or a video and say what is in it.

        The answer is remembered, so asking again about the same file is free.
        For a video a frame from the middle is used.

        Args:
            path: Absolute path to the file.
            question: Optional — what to pay attention to. Without it, a general
                description is produced and stored for reuse.
            at: A moment in seconds to look at, instead of the whole file.
                This is how a candidate from find_moments gets examined:
                measuring says where to look, looking says what is there.
                A description of one moment is not stored — it is an answer
                about a second of the film, not about the film.
        """
        try:
            target = Path(path).expanduser()
            if not target.is_file():
                return f"ERROR: {path} does not exist"
            kind = _kind(target)
            if kind == "audio":
                return "This is a sound file; there is nothing to look at."
            if not kind:
                return f"ERROR: {target.name} is not a picture or a video"

            if not question and at < 0:
                note = _read_note(target)
                if note.get("description"):
                    return note["description"]

            stills, temporary, facts = [str(target)], False, ""
            if kind == "video" and at >= 0:
                # One moment, asked for by name. Everything else about the file
                # is beside the point here.
                stills, temporary = [_still_at(target, at)], True
                if not stills[0]:
                    return f"ERROR: no frame at {at:.2f}s in {target.name}"
            elif kind == "video":
                stills, temporary = _stills_from_video(target), True
                if not stills:
                    return f"ERROR: no frame could be read from {target.name}"
                measured = _probe(target)
                if measured.get("duration"):
                    facts = (f" ({measured['duration']:.0f}s"
                             + (f", {measured['width']}x{measured['height']}"
                                if measured.get("width") else "") + ")")
            try:
                if question:
                    prompt = question
                elif kind == "video" and at >= 0:
                    prompt = ("Describe what is happening in this frame in two sentences: "
                              "who or what is in it, and what they are doing. Be concrete.")
                elif kind == "video":
                    prompt = (f"These {len(stills)} frames are taken in order from one video"
                              f"{facts}. Say what the video is about in two or three sentences: "
                              "the subject, the setting, and whether it is mostly static or "
                              "something happens. Be concrete; do not describe the frames "
                              "separately.")
                else:
                    prompt = ("Describe what is in this picture in two sentences: the subject, "
                              "the setting, and the mood. Be concrete.")
                description = _ask_vision(stills, prompt)
            finally:
                if temporary:
                    for still in stills:
                        try:
                            os.unlink(still)
                        except OSError:
                            pass

            if description and not description.startswith("ERROR"):
                description = description + facts if facts and facts not in description else description
                if not question and at < 0:
                    _write_note(target, description)
            return description or "The model returned nothing."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def describe_project_media(ctx: Context, bin_id: str = "", limit: int = 8) -> str:
        """Say what is in the clips the user has already put in the project bin.

        The other way round from list_media_folder: here the material was chosen
        by hand and is already in the project, and what is missing is knowing
        what it shows. Descriptions are remembered per file, so a clip described
        in one project is already known in the next.

        Args:
            bin_id: One clip to describe. Empty means every clip in the bin that
                has not been described yet.
            limit: How many to look at in one go when describing the whole bin.
        """
        try:
            app = helpers.get_resolve(ctx)._app
            pool = helpers.get_media_pool(ctx)
            wanted = [bin_id] if bin_id else [str(c.GetClipProperty("id") or c.GetName())
                                              for c in pool.GetAllClips()]
            if not wanted:
                return "The project bin is empty."

            lines, looked = [], 0
            for clip_id in wanted:
                props = app.get_clip_properties(clip_id) or {}
                name = props.get("name") or clip_id
                source = props.get("url") or props.get("resource") or ""
                if not source or not os.path.isfile(source):
                    lines.append(f"- {name} (bin {clip_id}): no file on disk to look at")
                    continue
                target = Path(source)
                note = _read_note(target)
                if note.get("description"):
                    lines.append(f"- {name} (bin {clip_id}): {note['description']}")
                    continue
                if looked >= max(1, limit):
                    lines.append(f"- {name} (bin {clip_id}): not looked at yet")
                    continue
                looked += 1
                lines.append(f"- {name} (bin {clip_id}): {describe_media(source)}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def find_moments(path: str, want: int = 20, window: float = 4.0) -> str:
        """When something happens in a file — the shortlist, not the whole film.

        The other half of transcribe_media. Where people talk, the transcript is
        the edit; where they do not — a wide shot, a camera left running — the
        picture has to be measured instead, and this measures it: how much
        changes from one moment to the next, where the shot cuts, where the
        sound peaks.

        It returns a couple of dozen candidates with a reason each, out of
        hundreds of samples, so you can then LOOK at a handful of them
        (describe_media with `at`, or render_bin_frame) and choose the ones that
        match what the user actually asked for. Deciding that is yours; finding
        where to look is this.

        Args:
            path: Absolute path to a video file.
            want: How many candidates to return.
            window: Length of each candidate, in seconds.
        """
        try:
            target = Path(path).expanduser()
            if not target.is_file():
                return f"ERROR: {path} does not exist"

            identity = _identity(target) + f".moments.{want}.{window}"
            cached = _store() / f"{identity}.json"
            if cached.exists():
                try:
                    with open(cached, encoding="utf-8") as handle:
                        return json.load(handle)["text"]
                except (OSError, ValueError, KeyError):
                    pass

            measured = _probe(target)
            total = measured.get("duration") or 0.0
            if total <= 0:
                return f"ERROR: {target.name} has no duration"

            change = _scene_scores(target)
            loud = _loudness(target)
            if not change:
                return f"ERROR: nothing could be measured in {target.name}"

            text = _rank_windows(change, loud, total, want, window)
            try:
                with open(cached, "w", encoding="utf-8") as handle:
                    json.dump({"file": str(target.resolve()), "text": text}, handle, ensure_ascii=False)
            except OSError:
                pass
            return text
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def transcribe_media(path: str, language: str = "", model: str = "tiny") -> str:
        """What is said in a file, with the timing of every line.

        This is where an edit of people talking starts: the sentence boundaries
        are the cut points. It reads the file directly — the clip does not have
        to be on the timeline and nothing is added to the subtitle track, so
        several sources can be transcribed one after another without their lines
        piling up on top of each other.

        The result is remembered per file, so asking twice is free.

        Args:
            path: Absolute path to a video or audio file.
            language: Language code ("ru", "en"); empty guesses.
            model: Whisper model — "tiny" is quick and enough for finding cuts,
                "large-v3-turbo" is slower and more accurate about words.
        """
        try:
            target = Path(path).expanduser()
            if not target.is_file():
                return f"ERROR: {path} does not exist"

            identity = _identity(target) + f".{model}.{language or 'auto'}"
            cached = _store() / f"{identity}.words.json"
            if cached.exists():
                try:
                    with open(cached, encoding="utf-8") as handle:
                        return json.load(handle)["text"]
                except (OSError, ValueError, KeyError):
                    pass

            # Hand whisper the sound, not the film. A half-hour 4K master is
            # gigabytes of picture wrapped around a few megabytes of audio, and
            # made to open the container itself the model spends its time moving
            # frames it will never look at. Stripping the audio out first costs
            # seconds and saves minutes.
            listen_to = str(target)
            stripped = ""
            if _kind(target) == "video":
                stripped = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                                       dir=str(Path.home() / ".cache")).name
                subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", str(target),
                                "-vn", "-ac", "1", "-ar", "16000", stripped],
                               capture_output=True, timeout=1800)
                if os.path.exists(stripped) and os.path.getsize(stripped) > 1024:
                    listen_to = stripped
                else:
                    stripped = ""

            python = os.environ.get("WUNJO_SPEECH_PYTHON")
            if not python or not os.path.isfile(python):
                return ("ERROR: the speech environment is not installed — open the Speech To Text "
                        "settings once and let it set itself up")

            script = (
                "import json,sys,whisper\n"
                "m=whisper.load_model(sys.argv[2], download_root=sys.argv[4])\n"
                "r=m.transcribe(sys.argv[1], language=(sys.argv[3] or None), verbose=False)\n"
                "print(json.dumps([{'start':round(s['start'],2),'end':round(s['end'],2),"
                "'text':s['text'].strip()} for s in r['segments']], ensure_ascii=False))\n"
            )
            cache_root = os.environ.get("WUNJO_SPEECH_MODELS", "")
            try:
                finished = subprocess.run(
                    [python, "-c", script, listen_to, model, language, cache_root],
                    capture_output=True, text=True, timeout=3600)
            finally:
                if stripped:
                    try:
                        os.unlink(stripped)
                    except OSError:
                        pass
            if finished.returncode != 0:
                tail = (finished.stderr or "").strip().splitlines()
                return f"ERROR: {tail[-1] if tail else 'transcription failed'}"

            segments = json.loads(finished.stdout.strip().splitlines()[-1])
            lines = [f"{s['start']:.2f} – {s['end']:.2f}  {s['text']}" for s in segments if s["text"]]
            text = "\n".join(lines) if lines else "Nothing was said in this file."
            try:
                with open(cached, "w", encoding="utf-8") as handle:
                    json.dump({"file": str(target.resolve()), "text": text}, handle, ensure_ascii=False)
            except OSError:
                pass
            return text
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def find_media(query: str, directory: str = "") -> str:
        """Find already-described files whose description mentions something.

        Searches what has been learned, so it costs nothing and works across
        projects. Describe a folder once, then ask it for "cat", "sunset",
        "person talking to camera".

        Args:
            query: Words to look for in the descriptions.
            directory: Optional — only files inside this folder.
        """
        try:
            words = [w for w in query.lower().split() if w]
            if not words:
                return "ERROR: nothing to look for"
            hits = []
            for note_file in _store().glob("*.json"):
                try:
                    with open(note_file, encoding="utf-8") as handle:
                        note = json.load(handle)
                except (OSError, ValueError):
                    continue
                where = note.get("file", "")
                if directory and not where.startswith(str(Path(directory).expanduser())):
                    continue
                if not os.path.exists(where):
                    continue
                text = note.get("description", "").lower()
                if all(word in text for word in words):
                    hits.append((where, note.get("description", "")))
            if not hits:
                return f"Nothing described so far matches '{query}'. Use describe_media first."
            return "\n\n".join(f"{path}\n{text}" for path, text in hits[:20])
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
