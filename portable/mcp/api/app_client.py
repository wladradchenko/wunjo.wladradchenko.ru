"""Low-level client for the Wunjo Make scripting interface.

The editor serves its Q_SCRIPTABLE methods on a local socket (a unix socket on
Linux and macOS, a named pipe on Windows) — see
`desktop/src/scripting/scriptingserver.cpp`. This used to be D-Bus, which meant
this file carried four backends (pydbus, dbus-next, jeepney, and shelling out
to dbus-send/qdbus/gdbus) and a hand-written parser for dbus-send's output. One
socket and `json` replace all of it, and the same code runs on every platform.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from api.constants import SOCKET_ENV_VAR, SOCKET_NAME


def _socket_candidates() -> list[str]:
    """Where the editor's socket might be, best first.

    The first instance takes the plain name; further ones append their pid. An
    explicit path in the environment wins over both, which is what makes it
    possible to talk to a particular copy while debugging.
    """
    override = os.environ.get(SOCKET_ENV_VAR)
    if override:
        return [override]

    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if os.name == "nt":
        # Named pipes are not filesystem paths and cannot be enumerated the
        # same way; the plain name is the only one that can be guessed.
        return [SOCKET_NAME]
    if not runtime:
        return [SOCKET_NAME]

    plain = os.path.join(runtime, SOCKET_NAME)
    found = [plain] if os.path.exists(plain) else []
    try:
        for entry in sorted(os.listdir(runtime)):
            if entry.startswith(SOCKET_NAME + "-"):
                found.append(os.path.join(runtime, entry))
    except OSError:
        pass
    return found or [plain]


class WunjoMakeClient:
    """Calls the editor's scriptable methods over its local socket.

    Every method below is a thin wrapper over :meth:`_call`, which sends one
    JSON object and reads one back. The connection is made on first use and
    remade once if it has gone away, so an editor that was restarted does not
    require restarting the MCP server.
    """

    #: How long to wait for one call. Some of them render a frame.
    TIMEOUT = 120.0

    def __init__(self):
        self._sock: socket.socket | None = None
        self._file = None
        self._next_id = 0
        self._path: str | None = None

    # ── connection ─────────────────────────────────────────────────────

    def _connect(self) -> None:
        errors = []
        for path in _socket_candidates():
            try:
                if os.name == "nt":
                    # Windows: QLocalServer is a named pipe under \\.\pipe\.
                    pipe = "\\\\.\\pipe\\" + os.path.basename(path)
                    sock = open(pipe, "r+b", buffering=0)
                    self._sock = None
                    self._file = sock
                else:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(self.TIMEOUT)
                    sock.connect(path)
                    self._sock = sock
                    self._file = sock.makefile("rwb")
                self._path = path
                return
            except OSError as error:
                errors.append(f"{path}: {error}")
        raise RuntimeError(
            "Wunjo Make is not running, or its scripting socket is not reachable "
            "(tried: " + "; ".join(errors) + ")"
        )

    def _close(self) -> None:
        for handle in (self._file, self._sock):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self._file = self._sock = self._path = None

    # ── calling ────────────────────────────────────────────────────────

    def _send(self, method: str, args: tuple) -> Any:
        self._next_id += 1
        request = {"id": self._next_id, "method": method, "args": list(args)}
        self._file.write((json.dumps(request) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ConnectionError("the editor closed the connection")
        reply = json.loads(line.decode("utf-8"))
        if not reply.get("ok"):
            raise RuntimeError(f"{method}: {reply.get('error', 'unknown error')}")
        return reply.get("result")

    def _call(self, method: str, *args) -> Any:
        """Call one scriptable method and return its result.

        Retries once on a dead connection and no more: the editor restarting is
        worth reconnecting for, an editor that is gone is worth an error rather
        than a hang.
        """
        if self._file is None:
            self._connect()
        try:
            return self._send(method, args)
        except (ConnectionError, OSError, json.JSONDecodeError):
            self._close()
            self._connect()
            return self._send(method, args)

    # ── introspection, useful when a call fails ────────────────────────

    def ping(self) -> bool:
        """True when the editor answers. Distinguishes "not running" from
        "running and does not have that method"."""
        try:
            return bool(self._call("__ping__"))
        except Exception:
            return False

    def methods(self) -> list[str]:
        """Every callable signature, as the editor reports it."""
        return self._call("__methods__")

    @staticmethod
    def _result_to_dict(result) -> dict:
        """Normalise a returned map.

        JSON already gives dicts, so this is nearly a no-op; it stays because
        the callers were written against the D-Bus client, where a map could
        also arrive as a list of pairs.
        """
        if isinstance(result, dict):
            return dict(result)
        if isinstance(result, list):
            merged = {}
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    merged[item[0]] = item[1]
                elif isinstance(item, dict):
                    merged.update(item)
            return merged
        return {}

    # ── Project Management ─────────────────────────────────────────────

    def new_project(self, name: str) -> str:
        return self._call("scriptNewProject", name)

    def open_project(self, file_path: str) -> bool:
        delay = 0.5
        for attempt in range(10):
            try:
                return bool(self._call("scriptOpenProject", file_path))
            except (subprocess.CalledProcessError, Exception):
                if attempt == 9:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)
        return False

    def save_project(self) -> bool:
        return bool(self._call("scriptSaveProject"))

    def save_project_as(self, file_path: str) -> bool:
        return bool(self._call("scriptSaveProjectAs", file_path))

    def get_project_name(self) -> str:
        return self._call("scriptGetProjectName")

    def get_project_path(self) -> str:
        return self._call("scriptGetProjectPath")

    def get_project_fps(self) -> float:
        result = self._call("scriptGetProjectFps")
        return float(result) if isinstance(result, str) else result

    def get_project_resolution_width(self) -> int:
        result = self._call("scriptGetProjectResolutionWidth")
        return int(result) if isinstance(result, str) else result

    def get_project_resolution_height(self) -> int:
        result = self._call("scriptGetProjectResolutionHeight")
        return int(result) if isinstance(result, str) else result

    def get_project_property(self, key: str) -> str:
        return self._call("scriptGetProjectProperty", key)

    def set_project_property(self, key: str, value: str) -> bool:
        return bool(self._call("scriptSetProjectProperty", key, value))

    def get_project_duration(self) -> int:
        """Get total timeline duration in frames."""
        result = self._call("scriptGetProjectDuration")
        return int(result) if isinstance(result, str) else result

    def get_project_color_space(self) -> str:
        """Get project color space (e.g. '709' for Rec.709, '2020' for Rec.2020)."""
        result = self._call("scriptGetProjectColorSpace")
        return result if isinstance(result, str) else ""

    def set_project_color_space(self, color_space: str) -> bool:
        """Set project color space."""
        result = self._call("scriptSetProjectColorSpace", color_space)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_project_audio_sample_rate(self) -> int:
        """Get project audio sample rate in Hz."""
        result = self._call("scriptGetProjectAudioSampleRate")
        return int(result) if isinstance(result, str) else result

    # ── Undo / Redo ──────────────────────────────────────────────────

    def undo(self, steps: int = 1) -> bool:
        """Undo the last N operations. Returns True if at least one was undone."""
        result = self._call("scriptUndo", steps)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def redo(self, steps: int = 1) -> bool:
        """Redo the last N undone operations. Returns True if at least one was redone."""
        result = self._call("scriptRedo", steps)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def undo_status(self) -> dict:
        """Get undo/redo status: can_undo, can_redo, undo_text, redo_text, index, count."""
        result = self._call("scriptUndoStatus")
        if not isinstance(result, str) or not result:
            return {}
        d = {}
        for pair in result.split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                d[k] = v
        return d

    # ── Media Pool (Bin) ───────────────────────────────────────────────

    def import_media(self, file_paths: list[str], folder_id: str = "-1") -> list[str]:
        """Import media files into the bin via addProjectClip (native Wunjo Make).

        Adds files one by one, tracking new IDs per file to preserve order.
        """
        result_ids = []
        for path in file_paths:
            ids_before = set(self.get_all_clip_ids())
            if folder_id and folder_id != "-1":
                self._call("addProjectClip", path, folder_id)
            else:
                self._call("addProjectClip", path)
            time.sleep(0.3)  # Let Wunjo Make register the clip
            ids_after = set(self.get_all_clip_ids())
            new = ids_after - ids_before
            if new:
                result_ids.append(sorted(new, key=int)[0])
        return result_ids

    def create_folder(self, name: str, parent_id: str = "-1") -> str:
        return self._call("scriptCreateFolder", name, parent_id)

    def get_all_clip_ids(self) -> list[str]:
        result = self._call("scriptGetAllClipIds")
        if isinstance(result, str):
            return [r for r in result.split("\n") if r] if result else []
        return list(result) if result else []

    def get_folder_clip_ids(self, folder_id: str) -> list[str]:
        result = self._call("scriptGetFolderClipIds", folder_id)
        if isinstance(result, str):
            return [r for r in result.split("\n") if r] if result else []
        return list(result) if result else []

    def get_clip_properties(self, bin_id: str) -> dict:
        """Get clip properties from the bin (name, duration, type, url)."""
        try:
            result = self._call("scriptGetClipProperties", bin_id)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                d = {}
                for item in result:
                    if isinstance(item, tuple) and len(item) == 2:
                        d[item[0]] = item[1]
                    elif isinstance(item, dict):
                        d.update(item)
                return d if d else {"id": bin_id}
            return {"id": bin_id}
        except Exception:
            return {"id": bin_id}

    def rename_bin_clip(self, bin_id: str, new_name: str) -> bool:
        """Rename a clip in the bin."""
        result = self._call("scriptRenameBinClip", bin_id, new_name)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def move_bin_clip(self, bin_id: str, target_folder_id: str) -> bool:
        """Move a bin clip to a different folder."""
        result = self._call("scriptMoveBinClip", bin_id, target_folder_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_clip_metadata(self, bin_id: str) -> dict:
        """Get extended metadata for a bin clip (codec, resolution, file size, etc.)."""
        result = self._call("scriptGetClipMetadata", bin_id)
        return self._result_to_dict(result)

    def delete_bin_clip(self, bin_id: str) -> bool:
        return bool(self._call("scriptDeleteBinClip", bin_id))

    def relink_bin_clip(self, bin_id: str, new_file_path: str) -> bool:
        """Relink a bin clip to a new file. Preserves all timeline instances."""
        result = self._call("scriptRelinkBinClip", bin_id, new_file_path)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def create_title_clip(self, title_xml: str, duration_frames: int,
                          clip_name: str = "Title clip",
                          folder_id: str = "-1") -> str:
        """Create a title clip in the bin. Returns bin ID or '-1'."""
        result = self._call("scriptCreateTitleClip",
                            title_xml, duration_frames, clip_name, folder_id)
        return str(result) if result else "-1"

    def get_title_xml(self, bin_id: str) -> str:
        """Get the XML content of a title clip. Returns empty string if not found."""
        result = self._call("scriptGetTitleXml", bin_id)
        return str(result) if result else ""

    def set_title_xml(self, bin_id: str, new_xml: str) -> bool:
        """Set the XML content of a title clip and reload. Returns True on success."""
        result = self._call("scriptSetTitleXml", bin_id, new_xml)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Timeline ───────────────────────────────────────────────────────

    def get_track_count(self, track_type: str) -> int:
        result = self._call("scriptGetTrackCount", track_type)
        return int(result) if isinstance(result, str) else result

    def get_track_info(self, track_index: int) -> dict:
        result = self._call("scriptGetTrackInfo", track_index)
        return dict(result) if result and not isinstance(result, str) else {}

    def get_all_tracks_info(self) -> list[dict]:
        result = self._call("scriptGetAllTracksInfo")
        if isinstance(result, list):
            out = []
            for t in result:
                if isinstance(t, dict):
                    out.append(t)
                elif isinstance(t, list):
                    # list of (key, val) tuples
                    d = {}
                    for item in t:
                        if isinstance(item, tuple) and len(item) == 2:
                            d[item[0]] = item[1]
                    if d:
                        out.append(d)
            return out
        return []

    def add_track(self, name: str, audio_track: bool) -> int:
        result = self._call("scriptAddTrack", name, audio_track)
        return int(result) if isinstance(result, str) else result

    def delete_track(self, track_id: int) -> bool:
        result = self._call("scriptDeleteTrack", track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def insert_space(self, track_id: int, position: int, duration: int,
                     all_tracks: bool = False) -> bool:
        """Insert blank space at position, pushing clips right."""
        result = self._call("scriptInsertSpace", track_id, position, duration, all_tracks)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def remove_space(self, track_id: int, position: int,
                     all_tracks: bool = False) -> bool:
        """Remove blank space at position, pulling clips left."""
        result = self._call("scriptRemoveSpace", track_id, position, all_tracks)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def insert_clip(self, bin_clip_id: str, track_id: int, position: int) -> int:
        valid = self._get_valid_track_ids()
        if valid and track_id not in valid:
            return -1  # Non-existent track
        result = self._call("scriptInsertClip", bin_clip_id, track_id, position)
        return int(result) if isinstance(result, str) else result

    def insert_clips_sequentially(self, bin_clip_ids: list[str], track_id: int,
                                   start_position: int) -> list[int]:
        """Insert clips back-to-back, one scriptInsertClip per clip.

        The batch D-Bus method (scriptInsertClipsSequentially) is broken —
        it returns -1 for every clip — so we insert individually and advance
        the position by each inserted clip's actual duration."""
        valid = self._get_valid_track_ids()
        if valid and track_id not in valid:
            raise ValueError(f"Invalid track_id {track_id}. Valid: {sorted(valid)}")
        clip_ids: list[int] = []
        position = start_position
        for bin_id in bin_clip_ids:
            result = self._call("scriptInsertClip", bin_id, track_id, position)
            clip_id = int(result) if isinstance(result, str) else int(result or -1)
            clip_ids.append(clip_id)
            if clip_id < 0:
                continue
            info = self.get_timeline_clip_info(clip_id)
            duration = int(info.get("duration", 0) or 0)
            position += max(duration, 1)
        return clip_ids

    def move_clip(self, clip_id: int, track_id: int, position: int) -> bool:
        valid = self._get_valid_track_ids()
        if valid and track_id not in valid:
            return False  # Non-existent track
        return bool(self._call("scriptMoveClip", clip_id, track_id, position))

    def resize_clip(self, clip_id: int, new_duration: int, from_right: bool) -> int:
        result = self._call("scriptResizeClip", clip_id, new_duration, from_right)
        return int(result) if isinstance(result, str) else result

    def delete_timeline_clip(self, clip_id: int) -> bool:
        return bool(self._call("scriptDeleteTimelineClip", clip_id))

    def _get_valid_track_ids(self) -> set[int]:
        """Return set of valid track IDs from current project."""
        tracks = self.get_all_tracks_info()
        ids = set()
        for t in tracks:
            tid = t.get("id")
            if tid is not None:
                ids.add(int(tid))
        return ids

    def get_clips_on_track(self, track_id: int) -> list[dict]:
        """Get all clips on a track. Validates track_id to prevent crash."""
        valid = self._get_valid_track_ids()
        if valid and track_id not in valid:
            return []  # Non-existent track — would crash Wunjo Make
        result = self._call("scriptGetClipsOnTrack", track_id)
        if isinstance(result, list):
            out = []
            for c in result:
                if isinstance(c, dict):
                    out.append(c)
                elif isinstance(c, list):
                    d = {}
                    for item in c:
                        if isinstance(item, tuple) and len(item) == 2:
                            d[item[0]] = item[1]
                    if d:
                        out.append(d)
            return out
        return []

    def get_timeline_clip_info(self, clip_id: int) -> dict:
        result = self._call("scriptGetTimelineClipInfo", clip_id)
        return self._result_to_dict(result)

    def cut_clip(self, clip_id: int, position: int) -> bool:
        return bool(self._call("scriptCutClip", clip_id, position))

    def slip_clip(self, clip_id: int, offset: int) -> bool:
        """Slip a clip's source in/out points by offset frames.

        Positive offset moves the source window forward (later in source).
        Negative offset moves it backward (earlier in source).
        Timeline position and duration remain unchanged.

        Args:
            clip_id: Timeline clip ID.
            offset: Number of frames to slip.
        """
        result = self._call("scriptSlipClip", clip_id, offset)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Ripple/Roll/Slide Editing ────────────────────────────────────

    def ripple_delete(self, clip_id: int) -> bool:
        """Delete a clip and close the gap (ripple delete)."""
        result = self._call("scriptRippleDelete", clip_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def ripple_trim(self, clip_id: int, delta: int, from_right: bool = True) -> bool:
        """Trim a clip and shift following clips. Positive delta = extend, negative = shrink."""
        result = self._call("scriptRippleTrim", clip_id, delta, from_right)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def roll_edit(self, clip_id: int, delta: int) -> bool:
        """Roll edit: move cut point between two adjacent clips."""
        result = self._call("scriptRollEdit", clip_id, delta)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def slide_edit(self, clip_id: int, delta: int) -> bool:
        """Slide edit: move clip while adjusting neighbors to fill gaps."""
        result = self._call("scriptSlideEdit", clip_id, delta)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Transitions & Mixes ────────────────────────────────────────────

    def add_mix(self, clip_id_a: int, clip_id_b: int, duration_frames: int) -> bool:
        return bool(self._call("scriptAddMix", clip_id_a, clip_id_b, duration_frames))

    def add_composition(self, transition_id: str, track_id: int,
                        position: int, duration: int) -> int:
        result = self._call("scriptAddComposition",
                            transition_id, track_id, position, duration)
        return int(result) if isinstance(result, str) else result

    def remove_mix(self, clip_id: int) -> bool:
        return bool(self._call("scriptRemoveMix", clip_id))

    def get_available_transitions(self) -> list[dict]:
        """Get all available transition/mix types with id and name."""
        result = self._call("scriptGetAvailableTransitions")
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def get_mix_params(self, clip_id: int) -> dict:
        """Get parameters of a mix/transition on a clip."""
        result = self._call("scriptGetMixParams", clip_id)
        return self._result_to_dict(result)

    def set_mix_duration(self, clip_id: int, new_duration: int) -> bool:
        """Change the duration of a mix on a clip."""
        result = self._call("scriptSetMixDuration", clip_id, new_duration)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Compositions ──────────────────────────────────────────────────

    def get_compositions(self) -> list[dict]:
        result = self._call("scriptGetCompositions")
        if isinstance(result, list):
            return [dict(c) for c in result] if result else []
        return []

    def get_composition_info(self, compo_id: int) -> dict:
        result = self._call("scriptGetCompositionInfo", compo_id)
        return self._result_to_dict(result)

    def move_composition(self, compo_id: int, track_id: int, position: int) -> bool:
        return bool(self._call("scriptMoveComposition", compo_id, track_id, position))

    def resize_composition(self, compo_id: int, new_duration: int, from_right: bool = True) -> int:
        result = self._call("scriptResizeComposition", compo_id, new_duration, from_right)
        return int(result) if isinstance(result, str) else result

    def delete_composition(self, compo_id: int) -> bool:
        return bool(self._call("scriptDeleteComposition", compo_id))

    def get_composition_types(self) -> list[dict]:
        result = self._call("scriptGetCompositionTypes")
        if isinstance(result, list):
            return [dict(t) for t in result] if result else []
        return []

    def set_composition_param(self, compo_id: int, param_name: str,
                              value: str) -> bool:
        """Set a single parameter on a composition."""
        return bool(self._call("scriptSetCompositionParam", compo_id,
                               param_name, value))

    def get_composition_param(self, compo_id: int, param_name: str) -> str:
        """Read a single parameter value from a composition."""
        result = self._call("scriptGetCompositionParam", compo_id, param_name)
        return result if isinstance(result, str) else ""

    # ── Effects ────────────────────────────────────────────────────────

    def get_available_effects(self) -> list[dict]:
        """Get all available effects with id, name, and type (audio/video)."""
        result = self._call("scriptGetAvailableEffects")
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def add_clip_effect(self, clip_id: int, effect_id: str,
                        params: dict[str, str] | None = None) -> bool:
        """Add an effect to a timeline clip with optional parameters."""
        keys = list(params.keys()) if params else []
        values = list(params.values()) if params else []
        return bool(self._call("scriptAddClipEffect", clip_id, effect_id,
                               keys, values))

    def remove_clip_effect(self, clip_id: int, effect_id: str) -> bool:
        """Remove an effect from a timeline clip by effect ID."""
        return bool(self._call("scriptRemoveClipEffect", clip_id, effect_id))

    def get_clip_effects(self, clip_id: int) -> str:
        """Get comma-separated list of effect names on a timeline clip."""
        result = self._call("scriptGetClipEffects", clip_id)
        return result if isinstance(result, str) else ""

    def set_effect_param(self, clip_id: int, effect_id: str,
                         param_name: str, value: str) -> bool:
        """Set a single parameter on an existing effect."""
        return bool(self._call("scriptSetEffectParam", clip_id, effect_id,
                               param_name, value))

    def get_effect_param(self, clip_id: int, effect_id: str,
                         param_name: str) -> str:
        """Read a single parameter value from an effect."""
        result = self._call("scriptGetEffectParam", clip_id, effect_id,
                            param_name)
        return result if isinstance(result, str) else ""

    def set_effect_expression(self, clip_id: int, effect_id: str,
                              param_name: str, expression: str,
                              base_value: float) -> bool:
        """Attach a JavaScript expression to an effect parameter."""
        return bool(self._call("scriptSetEffectExpression", clip_id,
                               effect_id, param_name, expression, base_value))

    def clear_effect_expression(self, clip_id: int, effect_id: str,
                                param_name: str) -> bool:
        """Remove a JavaScript expression from an effect parameter."""
        return bool(self._call("scriptClearEffectExpression", clip_id,
                               effect_id, param_name))

    # ── Effect Keyframes ──────────────────────────────────────────────

    def copy_clip_effects(self, clip_id: int) -> str:
        """Copy all effects from a clip as XML string."""
        result = self._call("scriptCopyClipEffects", clip_id)
        return result if isinstance(result, str) else ""

    def paste_clip_effects(self, target_clip_id: int, effects_xml: str) -> bool:
        """Paste effects XML onto a target clip."""
        result = self._call("scriptPasteClipEffects", target_clip_id, effects_xml)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_effect_keyframes(self, clip_id: int,
                             effect_index: int) -> list[dict]:
        """Get all keyframes for an effect on a timeline clip.

        Args:
            clip_id: Timeline clip ID.
            effect_index: 0-based index of the effect in the clip's effect stack.

        Returns list of dicts with keys: frame, type, value.
        """
        result = self._call("scriptGetEffectKeyframes", clip_id, effect_index)
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def add_effect_keyframe(self, clip_id: int, effect_index: int,
                            frame: int, value: float = 0.0,
                            keyframe_type: int = -1) -> bool:
        """Add a keyframe to an effect.

        Args:
            clip_id: Timeline clip ID.
            effect_index: 0-based effect index in the stack.
            frame: Frame position (relative to clip start).
            value: Normalized value 0.0–1.0.
            keyframe_type: KeyframeType enum (-1 = use value-based add,
                0=linear, 1=discrete, 2=smooth, 3=smooth_natural, ...).
        """
        result = self._call("scriptAddEffectKeyframe", clip_id, effect_index,
                            frame, value, keyframe_type)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def remove_effect_keyframe(self, clip_id: int, effect_index: int,
                               frame: int) -> bool:
        """Remove a keyframe from an effect.

        Args:
            clip_id: Timeline clip ID.
            effect_index: 0-based effect index in the stack.
            frame: Frame position of the keyframe to remove.
        """
        result = self._call("scriptRemoveEffectKeyframe", clip_id,
                            effect_index, frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def update_effect_keyframe(self, clip_id: int, effect_index: int,
                               old_frame: int, new_frame: int,
                               value: float = -1.0) -> bool:
        """Move and/or update a keyframe value.

        Args:
            clip_id: Timeline clip ID.
            effect_index: 0-based effect index in the stack.
            old_frame: Current frame position of the keyframe.
            new_frame: New frame position (same as old_frame to only change value).
            value: New normalized value 0.0–1.0 (-1 to keep existing value).
        """
        result = self._call("scriptUpdateEffectKeyframe", clip_id,
                            effect_index, old_frame, new_frame, value)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Effect Keyframes by parameter name ──────────────────────────

    def get_effect_keyframes_by_param(self, clip_id: int, effect_id: str,
                                       param_name: str = "") -> list[dict]:
        """Get keyframes for a specific effect parameter by name.

        Args:
            clip_id: Timeline clip ID.
            effect_id: Effect asset ID (e.g. "volume", "qtblend").
            param_name: Parameter name (e.g. "opacity"). Empty for primary param.
        """
        result = self._call("scriptGetEffectKeyframesByParam",
                            clip_id, effect_id, param_name)
        if isinstance(result, list):
            return [item if isinstance(item, dict) else {} for item in result]
        return []

    def add_effect_keyframe_by_param(self, clip_id: int, effect_id: str,
                                      param_name: str, frame: int,
                                      value: str = "", keyframe_type: int = -1) -> bool:
        """Add keyframe to a specific parameter by name.

        Args:
            clip_id: Timeline clip ID.
            effect_id: Effect asset ID.
            param_name: Parameter name.
            frame: Frame position.
            value: Value as string (effect-specific format).
            keyframe_type: 0=discrete, 1=linear, 2=smooth, -1=default(linear).
        """
        result = self._call("scriptAddEffectKeyframeByParam",
                            clip_id, effect_id, param_name,
                            frame, value, keyframe_type)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def remove_effect_keyframe_by_param(self, clip_id: int, effect_id: str,
                                         param_name: str, frame: int) -> bool:
        """Remove keyframe from a specific parameter by name.

        Args:
            clip_id: Timeline clip ID.
            effect_id: Effect asset ID.
            param_name: Parameter name.
            frame: Frame position.
        """
        result = self._call("scriptRemoveEffectKeyframeByParam",
                            clip_id, effect_id, param_name, frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Speed ─────────────────────────────────────────────────────────

    def set_clip_speed(self, clip_id: int, speed: float,
                       pitch_compensate: bool = False) -> bool:
        """Set clip speed. speed is percentage: 100=normal, 50=half, 200=double."""
        return bool(self._call("scriptSetClipSpeed", clip_id, speed, pitch_compensate))

    # ── Clip Transform Keyframes ─────────────────────────────────────

    def get_clip_transform_keyframes(self, clip_id: int) -> list[dict]:
        """Get transform keyframes (position, size, opacity) from qtblend effect."""
        result = self._call("scriptGetClipTransformKeyframes", clip_id)
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def set_clip_transform(self, clip_id: int, frame: int,
                           x: int, y: int, width: int, height: int,
                           opacity: float = 1.0) -> bool:
        """Set a transform keyframe on a clip (position + size + opacity)."""
        result = self._call("scriptSetClipTransform", clip_id,
                            frame, x, y, width, height, opacity)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def remove_clip_transform_keyframe(self, clip_id: int, frame: int) -> bool:
        """Remove a transform keyframe from a clip."""
        result = self._call("scriptRemoveClipTransformKeyframe", clip_id, frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Clip Properties ──────────────────────────────────────────────

    def get_clip_opacity(self, clip_id: int) -> float:
        """Get clip opacity (0.0-1.0). Returns -1 on error."""
        result = self._call("scriptGetClipOpacity", clip_id)
        return float(result) if isinstance(result, str) else result

    def set_clip_opacity(self, clip_id: int, opacity: float) -> bool:
        """Set clip opacity (0.0-1.0). Adds qtblend effect if needed."""
        result = self._call("scriptSetClipOpacity", clip_id, opacity)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def is_clip_enabled(self, clip_id: int) -> bool:
        """Check if a clip is enabled (not disabled/blind)."""
        result = self._call("scriptIsClipEnabled", clip_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_clip_enabled(self, clip_id: int, enabled: bool) -> bool:
        """Enable or disable a clip (blind eye toggle)."""
        result = self._call("scriptSetClipEnabled", clip_id, enabled)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_clip_color(self, clip_id: int) -> str:
        """Get clip color tags (semicolon-separated hex colors, e.g. '#ff0000;#00ff00'). Returns empty string if not found."""
        result = self._call("scriptGetClipColor", clip_id)
        return str(result) if result else ""

    def set_clip_color(self, clip_id: int, color_tag: str) -> bool:
        """Set clip color tags (semicolon-separated hex colors, e.g. '#ff0000;#00ff00')."""
        result = self._call("scriptSetClipColor", clip_id, color_tag)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Time Remap (speed ramping) ───────────────────────────────────

    def enable_time_remap(self, clip_id: int, enable: bool = True) -> bool:
        """Enable or disable time remap (variable speed) on a clip."""
        result = self._call("scriptEnableTimeRemap", clip_id, enable)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_time_remap(self, clip_id: int) -> dict:
        """Get time remap info for a clip.

        Returns dict with keys: enabled, time_map, pitch, image_mode.
        time_map format: "timecode=seconds;timecode=seconds;..."
        """
        result = self._call("scriptGetTimeRemap", clip_id)
        if isinstance(result, dict):
            return result
        return {}

    def set_time_remap(self, clip_id: int, time_map: str = "",
                       pitch: int = 0, image_mode: str = "nearest") -> bool:
        """Set time remap keyframes on a clip (must be enabled first).

        Args:
            clip_id: Timeline clip ID.
            time_map: Keyframe data "timecode=seconds;timecode=seconds;..."
            pitch: 1 for pitch compensation, 0 for normal.
            image_mode: "nearest" (sharp) or "blend" (interpolated).
        """
        result = self._call("scriptSetTimeRemap", clip_id, time_map, pitch, image_mode)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Audio ─────────────────────────────────────────────────────────

    def split_audio(self, clip_id: int) -> bool:
        """Separate audio from a video clip onto its own track."""
        result = self._call("scriptSplitAudio", clip_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_clip_volume(self, clip_id: int, dB: float) -> bool:
        """Set audio volume (gain) on a timeline clip in dB."""
        return bool(self._call("scriptSetClipVolume", clip_id, dB))

    def get_clip_volume(self, clip_id: int) -> float:
        """Get the current audio volume of a timeline clip in dB."""
        result = self._call("scriptGetClipVolume", clip_id)
        return float(result) if isinstance(result, str) else result

    def set_audio_fade(self, clip_id: int, fade_in_frames: int, fade_out_frames: int) -> bool:
        """Set audio fade in/out. Pass -1 to skip a fade."""
        return bool(self._call("scriptSetAudioFade", clip_id, fade_in_frames, fade_out_frames))

    def set_clip_pan(self, clip_id: int, pan: float) -> bool:
        """Set audio pan. -100 = full left, 0 = center, +100 = full right."""
        result = self._call("scriptSetClipPan", clip_id, pan)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_clip_pan(self, clip_id: int) -> float:
        """Get audio pan (-100 to +100). 0 = center."""
        result = self._call("scriptGetClipPan", clip_id)
        return float(result) if isinstance(result, str) else result

    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Mute or unmute a track."""
        return bool(self._call("scriptSetTrackMute", track_id, mute))

    def get_track_mute(self, track_id: int) -> bool:
        """Check if a track is muted."""
        result = self._call("scriptGetTrackMute", track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_track_locked(self, track_id: int, locked: bool) -> bool:
        """Lock or unlock a track."""
        return bool(self._call("scriptSetTrackLocked", track_id, locked))

    def get_track_locked(self, track_id: int) -> bool:
        """Check if a track is locked."""
        result = self._call("scriptGetTrackLocked", track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_track_hidden(self, track_id: int, hidden: bool) -> bool:
        """Hide or show a track."""
        return bool(self._call("scriptSetTrackHidden", track_id, hidden))

    def get_track_hidden(self, track_id: int) -> bool:
        """Check if a track is hidden."""
        result = self._call("scriptGetTrackHidden", track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_track_name(self, track_id: int) -> str:
        """Get track name/label."""
        result = self._call("scriptGetTrackName", track_id)
        return result if isinstance(result, str) else ""

    def set_track_name(self, track_id: int, name: str) -> bool:
        """Set track name/label."""
        result = self._call("scriptSetTrackName", track_id, name)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_track_color(self, track_id: int) -> int:
        """Get track color index. Returns -1 if not set."""
        result = self._call("scriptGetTrackColor", track_id)
        return int(result) if isinstance(result, str) and result else -1

    def set_track_color(self, track_id: int, color: int) -> bool:
        """Set track color index."""
        result = self._call("scriptSetTrackColor", track_id, color)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_track_solo(self, track_id: int) -> bool:
        """Check if track is solo'd."""
        result = self._call("scriptGetTrackSolo", track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Solo or unsolo a track."""
        result = self._call("scriptSetTrackSolo", track_id, solo)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_audio_levels(self, bin_id: str, stream: int = 0, downsample: int = 1,
                         mode: int = 0) -> list[float]:
        """Get normalized audio levels (0.0-1.0) from a media pool clip.

        mode: 0=peak (default), 1=RMS.
        """
        result = self._call("scriptGetAudioLevels", bin_id, stream, downsample, mode)
        if isinstance(result, list):
            try:
                return [float(v) for v in result]
            except (ValueError, TypeError):
                return []
        return []

    # ── Markers & Guides ───────────────────────────────────────────────

    def add_guide(self, frame: int, comment: str, category: int) -> bool:
        return bool(self._call("scriptAddGuide", frame, comment, category))

    def get_guides(self) -> list[dict]:
        result = self._call("scriptGetGuides")
        if isinstance(result, str):
            return []
        return [dict(g) for g in result] if result else []

    def delete_guide(self, frame: int) -> bool:
        return bool(self._call("scriptDeleteGuide", frame))

    def delete_guides_by_category(self, category: int) -> bool:
        return bool(self._call("scriptDeleteGuidesByCategory", category))

    # ── Clip Markers ──────────────────────────────────────────────────

    def add_clip_marker(self, bin_id: str, frame: int, comment: str, category: int) -> bool:
        return bool(self._call("scriptAddClipMarker", bin_id, frame, comment, category))

    def get_clip_markers(self, bin_id: str) -> list[dict]:
        result = self._call("scriptGetClipMarkers", bin_id)
        if isinstance(result, str):
            return []
        if isinstance(result, list):
            out = []
            for m in result:
                if isinstance(m, dict):
                    out.append(m)
                elif isinstance(m, list):
                    d = {}
                    for item in m:
                        if isinstance(item, tuple) and len(item) == 2:
                            d[item[0]] = item[1]
                    if d:
                        out.append(d)
            return out
        return []

    def delete_clip_marker(self, bin_id: str, frame: int) -> bool:
        return bool(self._call("scriptDeleteClipMarker", bin_id, frame))

    def delete_clip_markers_by_category(self, bin_id: str, category: int) -> bool:
        return bool(self._call("scriptDeleteClipMarkersByCategory", bin_id, category))

    # ── Playback & Monitor ─────────────────────────────────────────────

    def seek(self, frame: int) -> None:
        self._call("scriptSeek", frame)

    def get_position(self) -> int:
        result = self._call("scriptGetPosition")
        return int(result) if isinstance(result, str) else result

    def play(self) -> None:
        self._call("scriptPlay")

    def pause(self) -> None:
        self._call("scriptPause")

    def set_playback_speed(self, speed: float) -> bool:
        """Set preview playback speed. Positive=forward, negative=rewind."""
        result = self._call("scriptSetPlaybackSpeed", speed)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_playback_speed(self) -> float:
        """Get current preview playback speed."""
        result = self._call("scriptGetPlaybackSpeed")
        return float(result) if result else 0.0

    # ── Timeline Navigation ──────────────────────────────────────────

    def go_to_next_marker(self) -> int:
        """Seek to the next guide/marker. Returns frame or -1."""
        result = self._call("scriptGoToNextMarker")
        return int(result) if isinstance(result, str) else result

    def go_to_previous_marker(self) -> int:
        """Seek to the previous guide/marker. Returns frame or -1."""
        result = self._call("scriptGoToPreviousMarker")
        return int(result) if isinstance(result, str) else result

    def go_to_next_edit(self) -> int:
        """Seek to the next clip boundary. Returns frame or -1."""
        result = self._call("scriptGoToNextEdit")
        return int(result) if isinstance(result, str) else result

    def go_to_previous_edit(self) -> int:
        """Seek to the previous clip boundary. Returns frame or -1."""
        result = self._call("scriptGoToPreviousEdit")
        return int(result) if isinstance(result, str) else result

    # ── Scene Detection ───────────────────────────────────────────────

    def detect_scenes(self, bin_clip_id: str, threshold: float = 0.4,
                      min_duration: int = 0) -> list[float]:
        """Detect scene cuts in a clip using FFmpeg scene detection.

        Args:
            bin_clip_id: Clip ID in the bin/media pool.
            threshold: Sensitivity 0.0-1.0 (lower = more cuts). Default 0.4.
            min_duration: Minimum frames between cuts. Default 0 (no minimum).

        Returns:
            List of timestamps (seconds) where scene cuts were detected.
        """
        result = self._call("scriptDetectScenes", bin_clip_id, threshold, min_duration)
        if isinstance(result, list):
            return [float(t) for t in result]
        if isinstance(result, str) and result:
            return [float(t) for t in result.split("\n") if t]
        return []

    # ── Fill Frame ─────────────────────────────────────────────────────

    def fill_frame(self, clip_id: int) -> bool:
        """Scale-to-fill a timeline clip (remove black bars, center crop).

        Reads source resolution via MLT, calculates qtblend rect, applies effect.
        Returns True on success.
        """
        result = self._call("scriptFillFrame", clip_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Preview / Frame Rendering ─────────────────────────────────────

    def render_bin_frame(self, bin_id: str, frame: int, width: int,
                         height: int, output_path: str) -> str:
        """Render a single frame from a bin clip to a JPEG file.

        Returns the output path on success, empty string on failure.
        """
        result = self._call("scriptRenderBinFrame",
                            bin_id, frame, width, height, output_path)
        return result if isinstance(result, str) else ""

    def render_timeline_frame(self, frame: int, width: int,
                              height: int, output_path: str) -> str:
        """Render a composited timeline frame to a JPEG file.

        Returns the output path on success, empty string on failure.
        """
        result = self._call("scriptRenderTimelineFrame",
                            frame, width, height, output_path)
        return result if isinstance(result, str) else ""

    def capture_window(self, max_size: int, output_path: str) -> str:
        """Capture a screenshot of the Wunjo Make GUI window.

        Uses QWidget::grab() internally — works on Windows, Linux, macOS.

        Returns the output path on success, empty string on failure.
        """
        result = self._call("scriptCaptureWindow", max_size, output_path)
        return result if isinstance(result, str) else ""

    def get_panel_geometries(self) -> list[dict]:
        """Get bounding boxes of all dock panels in the Wunjo Make window.

        Returns a list of dicts with keys: name, title, x, y, width, height, visible.
        Coordinates are relative to the main window.
        """
        result = self._call("scriptGetPanelGeometries")
        if isinstance(result, str) and result:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return []
        return []

    # ── Subtitles ──────────────────────────────────────────────────────

    def get_subtitles(self) -> list[dict]:
        """Get all subtitles as a list of dicts with id, layer, startFrame, endFrame, text."""
        result = self._call("scriptGetSubtitles")
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def add_subtitle(self, start_frame: int, end_frame: int, text: str,
                     layer: int = 0) -> int:
        """Add a subtitle. Returns its ID or -1 on error."""
        result = self._call("scriptAddSubtitle", start_frame, end_frame, text, layer)
        return int(result) if isinstance(result, str) else result

    def edit_subtitle(self, subtitle_id: int, new_text: str) -> bool:
        """Edit subtitle text by ID."""
        result = self._call("scriptEditSubtitle", subtitle_id, new_text)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def move_subtitle(self, subtitle_id: int, new_start_frame: int) -> bool:
        """Move subtitle to a new start position (frames)."""
        result = self._call("scriptMoveSubtitle", subtitle_id, new_start_frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def resize_subtitle(self, subtitle_id: int, new_duration: int,
                        from_right: bool = True) -> bool:
        """Resize subtitle duration (frames). from_right=True extends end."""
        result = self._call("scriptResizeSubtitle", subtitle_id, new_duration, from_right)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def delete_subtitle(self, subtitle_id: int) -> bool:
        """Delete a subtitle by ID."""
        result = self._call("scriptDeleteSubtitle", subtitle_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def import_subtitle(self, file_path: str, offset: int = 0,
                        encoding: str = "UTF-8") -> bool:
        """Import a subtitle file (SRT/ASS/VTT/SBV)."""
        result = self._call("scriptImportSubtitle", file_path, offset, encoding)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def export_subtitles(self, file_path: str) -> bool:
        """Export subtitles to an .ass file. Returns True on success."""
        result = self._call("scriptExportSubtitles", file_path)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def speech_recognition(self, model: str = "turbo", language: str = "",
                           max_line_width: int = 42) -> bool:
        """Start headless Whisper transcription; SRT auto-imports when done."""
        result = self._call("scriptSpeechRecognition", model, language, max_line_width)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Subtitle Styles ───────────────────────────────────────────────

    def get_subtitle_styles(self, global_styles: bool = False) -> list[dict]:
        """Get all subtitle styles as a list of dicts.

        Args:
            global_styles: If True, return global styles; otherwise local (project) styles.
        """
        result = self._call("scriptGetSubtitleStyles", global_styles)
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def set_subtitle_style(self, name: str, params: dict,
                           global_style: bool = False) -> bool:
        """Create or update a subtitle style.

        Args:
            name: Style name (e.g. "Default", "Accent").
            params: Dict of style properties (camelCase keys matching C++ property names).
            global_style: If True, modify global styles; otherwise local.
        """
        keys = list(params.keys())
        values = [str(v) for v in params.values()]
        result = self._call("scriptSetSubtitleStyle", name, keys, values, global_style)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def delete_subtitle_style(self, name: str,
                              global_style: bool = False) -> bool:
        """Delete a subtitle style. Cannot delete 'Default'."""
        result = self._call("scriptDeleteSubtitleStyle", name, global_style)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_subtitle_style_name(self, subtitle_id: int,
                                style_name: str) -> bool:
        """Assign a named style to a subtitle event."""
        result = self._call("scriptSetSubtitleStyleName", subtitle_id, style_name)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Groups ─────────────────────────────────────────────────────────

    def group_clips(self, item_ids: list[int]) -> int:
        """Group timeline items (clips/compositions).

        Args:
            item_ids: List of timeline item IDs to group (minimum 2).

        Returns:
            Group ID on success, -1 on failure.
        """
        result = self._call("scriptGroupClips", item_ids)
        return int(result) if isinstance(result, str) else result

    def ungroup_clips(self, item_id: int) -> bool:
        """Ungroup the topmost group containing the given item.

        All members are released from the group.
        """
        result = self._call("scriptUngroupClips", item_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_group_info(self, item_id: int) -> dict:
        """Get group information for a timeline item.

        Returns dict with keys: isInGroup, isGroup, rootId, groupType,
        members (list of {id, type, trackId, position}).
        """
        result = self._call("scriptGetGroupInfo", item_id)
        return self._result_to_dict(result)

    def remove_from_group(self, item_id: int) -> bool:
        """Remove a single item from its group, keeping the rest grouped."""
        result = self._call("scriptRemoveFromGroup", item_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Zones (timeline in/out points) ────────────────────────────────

    def get_zone(self) -> dict:
        """Get the current timeline zone (in/out points).

        Returns dict with keys: zoneIn, zoneOut (frame numbers).
        """
        result = self._call("scriptGetZone")
        return self._result_to_dict(result)

    def set_zone(self, in_frame: int, out_frame: int) -> bool:
        """Set the timeline zone (in/out points)."""
        result = self._call("scriptSetZone", in_frame, out_frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_zone_in(self, in_frame: int) -> bool:
        """Set the timeline zone in-point only."""
        result = self._call("scriptSetZoneIn", in_frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def set_zone_out(self, out_frame: int) -> bool:
        """Set the timeline zone out-point only."""
        result = self._call("scriptSetZoneOut", out_frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def extract_zone(self, in_frame: int, out_frame: int,
                     lift_only: bool = False) -> bool:
        """Extract (remove) content in the given zone.

        Args:
            in_frame: Zone start frame.
            out_frame: Zone end frame.
            lift_only: If True, lifts without ripple (leaves gap).
                       If False, ripple deletes (closes gap).
        """
        result = self._call("scriptExtractZone", in_frame, out_frame, lift_only)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Sequences (multi-timeline) ────────────────────────────────────

    def create_sequence(self, name: str, audio_tracks: int = -1,
                        video_tracks: int = -1,
                        parent_folder: str = "-1") -> str:
        """Create a new sequence (multi-timeline).

        Args:
            name: Sequence name.
            audio_tracks: Number of audio tracks (-1 = project default).
            video_tracks: Number of video tracks (-1 = project default).
            parent_folder: Bin folder ID (-1 = root).

        Returns the bin clip ID of the new sequence, or "-1" on failure.
        """
        result = self._call("scriptCreateSequence", name,
                            audio_tracks, video_tracks, parent_folder)
        return str(result) if result else "-1"

    def get_sequences(self) -> list[dict]:
        """Get all sequences in the project.

        Returns list of dicts with keys: uuid, name, duration, tracks, active.
        """
        result = self._call("scriptGetSequences")
        if isinstance(result, list):
            out = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    d = {}
                    for pair in item:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            d[pair[0]] = pair[1]
                    if d:
                        out.append(d)
            return out
        return []

    def get_active_sequence(self) -> dict:
        """Get info about the currently active sequence.

        Returns dict with keys: uuid, name, duration, tracks.
        """
        result = self._call("scriptGetActiveSequence")
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            d = {}
            for pair in result:
                if isinstance(pair, tuple) and len(pair) == 2:
                    d[pair[0]] = pair[1]
            return d
        return {}

    def set_active_sequence(self, uuid: str) -> bool:
        """Switch to a sequence by its UUID.

        Args:
            uuid: Sequence UUID string (without braces).
        """
        result = self._call("scriptSetActiveSequence", uuid)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Proxy Clips ────────────────────────────────────────────────────

    def get_clip_proxy_status(self, bin_id: str) -> dict:
        """Get proxy status for a bin clip.

        Returns dict with keys: supportsProxy, hasProxy, proxyPath,
        originalUrl, isGenerating.
        """
        result = self._call("scriptGetClipProxyStatus", bin_id)
        return self._result_to_dict(result)

    def set_clip_proxy(self, bin_id: str, enabled: bool) -> bool:
        """Enable or disable proxy for a bin clip."""
        result = self._call("scriptSetClipProxy", bin_id, enabled)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def delete_clip_proxy(self, bin_id: str) -> bool:
        """Delete proxy file and disable proxy for a bin clip."""
        result = self._call("scriptDeleteClipProxy", bin_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def rebuild_clip_proxy(self, bin_id: str) -> bool:
        """Force regenerate proxy for a bin clip."""
        result = self._call("scriptRebuildClipProxy", bin_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Render ─────────────────────────────────────────────────────────

    def render(self, url: str) -> None:
        self._call("scriptRender", url)

    # ── Selection ─────────────────────────────────────────────────────

    def get_selection(self) -> list[int]:
        """Get currently selected timeline item IDs."""
        result = self._call("scriptGetSelection")
        if not result:
            return []
        # Result is a list of ID values (as strings from D-Bus)
        if isinstance(result, list):
            return [int(x) for x in result if isinstance(x, (int, str)) and str(x).lstrip('-').isdigit()]
        return []

    def set_selection(self, ids: list[int]) -> bool:
        """Set selection to the given timeline item IDs."""
        if not ids:
            return self.clear_selection()
        result = self._call("scriptSetSelection", ids)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def add_to_selection(self, item_id: int, clear: bool = False) -> bool:
        """Add an item to the selection. If clear=True, replaces current selection."""
        result = self._call("scriptAddToSelection", item_id, clear)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def clear_selection(self) -> bool:
        """Clear the current selection."""
        result = self._call("scriptClearSelection")
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def select_all(self) -> bool:
        """Select all clips, compositions, and subtitles on the timeline."""
        result = self._call("scriptSelectAll")
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def select_current_track(self) -> bool:
        """Select all items on the currently active track."""
        result = self._call("scriptSelectCurrentTrack")
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def select_items_in_range(self, track_ids: list[int], start_frame: int, end_frame: int) -> bool:
        """Select items within a frame range on specified tracks."""
        result = self._call("scriptSelectItems", track_ids, start_frame, end_frame)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Render ────────────────────────────────────────────────────────

    def render_with_params(self, output_file: str, preset_name: str = "",
                           in_frame: int = -1, out_frame: int = -1,
                           params: dict[str, str] | None = None) -> bool:
        """Render timeline with custom parameters.

        Args:
            output_file: Output file path.
            preset_name: Render preset name (e.g. "MP4-H264/MP4 - H.264").
            in_frame: Start frame (-1 = project start).
            out_frame: End frame (-1 = project end).
            params: Optional dict of FFmpeg/MLT parameter overrides
                    (e.g. {"crf": "19", "preset": "fast"}).
        """
        keys = list(params.keys()) if params else []
        values = list(params.values()) if params else []
        result = self._call("scriptRenderWithParams",
                            output_file, preset_name,
                            in_frame, out_frame, keys, values)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def get_render_presets(self) -> list[str]:
        """Get list of available render preset names."""
        result = self._call("scriptGetRenderPresets")
        if isinstance(result, list):
            return [str(p) for p in result]
        return []

    def get_render_jobs(self) -> list[dict]:
        """Get list of render jobs with status and progress.

        Returns list of dicts: {path, status, progress (0-100), frame}.
        Status values: waiting, starting, running, finished, failed, aborted.
        """
        result = self._call("scriptGetRenderJobs")
        if isinstance(result, list):
            jobs = []
            for item in result:
                if isinstance(item, dict):
                    jobs.append(item)
            return jobs
        return []

    def abort_render_job(self, output_path: str) -> bool:
        """Abort a running render job by its output file path."""
        result = self._call("scriptAbortRenderJob", output_path)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Project Profile ──────────────────────────────────────────────

    def set_project_profile(self, width: int, height: int,
                            fps_num: int, fps_den: int) -> bool:
        """Change project profile (resolution + FPS).

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps_num: FPS numerator (e.g. 25 for 25fps, 30000 for 29.97fps).
            fps_den: FPS denominator (e.g. 1 for 25fps, 1001 for 29.97fps).
        """
        result = self._call("scriptSetProjectProfile",
                            width, height, fps_num, fps_den)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    # ── Copy/Cut/Paste ───────────────────────────────────────────────

    def copy_clips(self) -> int:
        """Copy current selection to clipboard. Returns main clip ID or -1."""
        result = self._call("scriptCopyClips")
        return int(result) if isinstance(result, str) else (result if result else -1)

    def cut_clips(self) -> bool:
        """Cut current selection (copy + delete). Returns True on success."""
        result = self._call("scriptCutClips")
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

    def paste_clips(self, position: int = -1, track_id: int = -1) -> bool:
        """Paste clipboard contents at position on track.

        Args:
            position: Frame position (-1 = playhead).
            track_id: Target track (-1 = active track).
        """
        result = self._call("scriptPasteClips", position, track_id)
        if isinstance(result, str):
            return result.lower() == "true"
        return bool(result)

