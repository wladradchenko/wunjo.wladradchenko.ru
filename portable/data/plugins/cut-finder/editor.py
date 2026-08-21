"""Talking back to the editor that started this job.

The plugin is a separate process with no handle on the application, so it calls
the editor's scriptable methods over the local socket the editor listens on —
a unix socket on Linux and macOS, a named pipe on Windows, the same one the MCP
server and the assistant use. The editor hands the exact address over in
``WUNJO_SOCKET`` and the client itself in ``WUNJO_MCP_DIR``; nothing has to be
guessed, and nothing is added to this plugin's environment for the privilege.

This used to go over D-Bus with ``dbus-send``. D-Bus was removed from the
application — it exists on Linux only, and this editor is going to Windows and
macOS — and this file went on calling it, so the plugin found its cuts and
placed none of them: "the editor did not answer" was the whole of the report.
Every edit made here is still its own entry in the undo history: fifteen cuts
undo fifteen times, not once.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional


def _connect():
    """The editor's client, or None when there is nothing to talk to."""
    mcp_dir = os.environ.get("WUNJO_MCP_DIR", "")
    if mcp_dir and mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)
    try:
        from api.app_client import WunjoMakeClient

        client = WunjoMakeClient()
        # Ask before trusting it: every call below swallows what it cannot do,
        # so a client that reaches nobody would quietly place nothing.
        return client if client.ping() else None
    except Exception:  # noqa: BLE001
        return None


class Editor:
    """The running editor, or nothing at all if it is not there to answer."""

    def __init__(self) -> None:
        self._app = _connect()

    @property
    def available(self) -> bool:
        return self._app is not None

    def call(self, method: str, *args: Any) -> Any:
        """One scriptable method, or None when it could not be reached.

        The socket answers with the value itself — a list, a number, a bool —
        so nothing here parses text. The D-Bus version had to read `dbus-send`
        output back with regular expressions, which is where a bin id once came
        back as ")" and no clip was ever found.
        """
        if self._app is None:
            return None
        try:
            return self._app._call(method, *args)
        except Exception:  # noqa: BLE001
            return None

    # ---- reading ----

    def tracks(self) -> List[Dict]:
        """Every track as {id, audio}, in the order the timeline holds them."""
        out = []
        for entry in self.call("scriptGetAllTracksInfo") or []:
            try:
                out.append({"id": int(entry["id"]), "audio": bool(entry.get("audio", False))})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def clips_on_track(self, track_id: int) -> List[Dict]:
        """Clips on a track as {id, position, in, binId}."""
        out = []
        for entry in self.call("scriptGetClipsOnTrack", int(track_id)) or []:
            try:
                out.append({"id": int(entry["id"]), "position": int(entry["position"]),
                            "in": int(entry.get("in", 0)), "binId": str(entry.get("binId", ""))})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def find_clip(self, bin_id: str, clip_in: int) -> Optional[Dict]:
        """The timeline clip this job was started from.

        The job says which bin clip and which part of it, not which of the
        copies on the timeline — so it is matched by both, and the one whose in
        point agrees wins.
        """
        best = None
        for track in self.tracks():
            for clip in self.clips_on_track(track["id"]):
                if clip["binId"] != str(bin_id):
                    continue
                if clip["in"] == clip_in:
                    return dict(clip, audio=track["audio"])
                if best is None:
                    best = dict(clip, audio=track["audio"])
        return best

    # ---- writing ----

    def add_clip_marker(self, bin_id: str, frame: int, comment: str, category: int = 0) -> bool:
        return bool(self.call("scriptAddClipMarker", str(bin_id), int(frame), comment, int(category)))

    def cut_clip(self, clip_id: int, position: int) -> bool:
        return bool(self.call("scriptCutClip", int(clip_id), int(position)))

    def add_track(self, name: str, audio: bool) -> int:
        answer = self.call("scriptAddTrack", name, bool(audio))
        try:
            return int(answer)
        except (TypeError, ValueError):
            return -1

    def insert_clip(self, bin_clip_id: str, track_id: int, position: int) -> int:
        answer = self.call("scriptInsertClip", str(bin_clip_id), int(track_id), int(position))
        try:
            return int(answer)
        except (TypeError, ValueError):
            return -1

    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        return bool(self.call("scriptSetTrackMute", int(track_id), bool(mute)))
