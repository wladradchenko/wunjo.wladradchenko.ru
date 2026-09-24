"""Timeline and TimelineItem — timeline operations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.app_client import WunjoMakeClient
    from api.media_pool import MediaPoolItem

from api.constants import (
    TRACK_VIDEO, TRACK_AUDIO, TRACK_SUBTITLE,
    DEFAULT_MIX_DURATION, MARKER_COLOR_MAP, MARKER_CATEGORY_TO_COLOR,
)


class TimelineItem:
    """Represents a clip on the Wunjo Make timeline."""

    def __init__(self, app: WunjoMakeClient, clip_id: int, info: dict | None = None):
        self._app = app
        self._clip_id = clip_id
        self._info = info

    @property
    def clip_id(self) -> int:
        return self._clip_id

    def _load_info(self):
        if self._info is None:
            self._info = self._app.get_timeline_clip_info(self._clip_id)

    def GetName(self) -> str:
        self._load_info()
        return self._info.get("name", "")

    def GetDuration(self) -> int:
        self._load_info()
        return int(self._info.get("duration", 0))

    def GetStart(self) -> int:
        self._load_info()
        return int(self._info.get("position", 0))

    def GetEnd(self) -> int:
        return self.GetStart() + self.GetDuration()

    def GetTrackId(self) -> int:
        self._load_info()
        return int(self._info.get("trackId", -1))

    def GetMediaPoolItem(self):
        """Return the MediaPoolItem for this timeline clip."""
        self._load_info()
        bin_id = self._info.get("binId", "")
        if bin_id:
            from api.media_pool import MediaPoolItem
            return MediaPoolItem(self._app, bin_id)
        return None

    def SetDuration(self, frames: int, from_right: bool = True) -> int:
        """Resize this clip. Returns actual new duration."""
        result = self._app.resize_clip(self._clip_id, frames, from_right)
        self._info = None  # Invalidate cache
        return result

    def Move(self, track_id: int, position: int) -> bool:
        """Move this clip to a new position."""
        result = self._app.move_clip(self._clip_id, track_id, position)
        self._info = None
        return result

    def Delete(self) -> bool:
        """Remove this clip from the timeline."""
        return self._app.delete_timeline_clip(self._clip_id)

    def Cut(self, position: int) -> bool:
        """Cut this clip at the given frame position."""
        return self._app.cut_clip(self._clip_id, position)

    def SetClipColor(self, color: str) -> bool:
        """Set the clip color on the timeline. Stub."""
        return True

    def GetClipColor(self) -> str:
        """Get the clip color. Stub."""
        return ""

    # ── Fusion stubs (Resolve-only) ────────────────────────────────────

    def GetFusionCompCount(self) -> int:
        """Return fusion composition count. Always 0."""
        return 0

    def AddFusionComp(self) -> bool:
        """Add a fusion composition. No-op (Resolve-only)."""
        return False

    def GetFusionCompByIndex(self, index: int):
        return None

    def GetFusionCompNameList(self) -> list[str]:
        return []

    def GetLeftOffset(self) -> int:
        self._load_info()
        return int(self._info.get("in", 0))

    def GetRightOffset(self) -> int:
        return 0

    def __repr__(self):
        return f"TimelineItem(clip_id={self._clip_id})"


class Timeline:
    """Represents the active Wunjo Make timeline (sequence)."""

    def __init__(self, app: WunjoMakeClient):
        self._app = app

    def GetName(self) -> str:
        """Return the timeline name.

        In Wunjo Make, maps to the project name (single timeline model).
        """
        return self._app.get_project_name()

    def SetName(self, name: str) -> bool:
        """Set the timeline name. Stub."""
        return True

    def GetTrackCount(self, track_type: str = TRACK_VIDEO) -> int:
        """Return the number of tracks of the given type.

        Args:
            track_type: "video", "audio", or "subtitle".
                        "subtitle" always returns 0 in Wunjo Make.
        """
        if track_type == TRACK_SUBTITLE:
            return 0
        return self._app.get_track_count(track_type)

    def GetTrackInfo(self, track_index: int) -> dict:
        """Return info about a specific track."""
        return self._app.get_track_info(track_index)

    def GetAllTracksInfo(self) -> list[dict]:
        """Return info for all tracks."""
        return self._app.get_all_tracks_info()

    def AddTrack(self, name: str = "", audio: bool = False) -> int:
        """Add a new track. Returns track ID."""
        return self._app.add_track(name, audio)

    def DeleteTrack(self, track_id: int) -> bool:
        """Delete a track by ID. Returns True on success."""
        return self._app.delete_track(track_id)

    def InsertClip(self, bin_clip_id: str, track_id: int,
                   position: int) -> TimelineItem | None:
        """Insert a bin clip into the timeline at the given position."""
        clip_id = self._app.insert_clip(bin_clip_id, track_id, position)
        if clip_id >= 0:
            return TimelineItem(self._app, clip_id)
        return None

    def InsertClipAt(self, track_type: str, track_idx: int,
                     media_pool_item, position: int) -> TimelineItem | None:
        """Insert a MediaPoolItem at a position on a track by type/index.

        This is a Resolve-compatible convenience method.
        """
        tracks = self._app.get_all_tracks_info()
        target_tracks = [t for t in tracks
                         if t.get("audio", False) == (track_type == TRACK_AUDIO)]
        if track_idx >= len(target_tracks):
            return None
        track_id = target_tracks[track_idx].get("id", 0)
        return self.InsertClip(media_pool_item.bin_id, track_id, position)

    def InsertClipsSequentially(self, bin_clip_ids: list[str], track_id: int,
                                 start_position: int = 0) -> list[TimelineItem]:
        """Insert multiple clips sequentially on a track."""
        clip_ids = self._app.insert_clips_sequentially(
            bin_clip_ids, track_id, start_position
        )
        return [TimelineItem(self._app, cid) for cid in clip_ids if cid >= 0]

    def GetItemListInTrack(self, track_type: str,
                           track_idx: int) -> list[TimelineItem]:
        """Return all clips on a track identified by type and 1-based index.

        Resolve uses 1-based track indices. This method accepts both
        0-based and 1-based: if track_idx >= 1 and there are enough
        tracks, it treats it as 1-based (Resolve convention).

        For safety, also works with 0-based indices.
        """
        if track_type == TRACK_SUBTITLE:
            return []

        tracks = self._app.get_all_tracks_info()
        target = [t for t in tracks
                  if t.get("audio", False) == (track_type == TRACK_AUDIO)]

        if not target:
            return []

        # Resolve uses 1-based indexing
        # If index is in 1..len, treat as 1-based
        # If index is 0, treat as 0-based (first track)
        idx = track_idx
        if idx >= 1:
            idx = idx - 1  # Convert 1-based to 0-based

        if idx < 0 or idx >= len(target):
            return []

        track_id = target[idx].get("id", 0)
        clips = self._app.get_clips_on_track(track_id)
        return [TimelineItem(self._app, c["id"], c)
                for c in clips if "id" in c]

    def AddTransition(self, item_a: TimelineItem, item_b: TimelineItem,
                      duration: int = DEFAULT_MIX_DURATION) -> bool:
        """Add a same-track mix (transition) between two adjacent clips."""
        return self._app.add_mix(item_a.clip_id, item_b.clip_id, duration)

    def AddComposition(self, transition_id: str, track_id: int,
                       position: int, duration: int) -> int:
        """Add a cross-track composition/transition."""
        return self._app.add_composition(
            transition_id, track_id, position, duration
        )

    def RemoveMix(self, item: TimelineItem) -> bool:
        """Remove a mix from the given clip."""
        return self._app.remove_mix(item.clip_id)

    # ── Markers & Guides ───────────────────────────────────────────────

    def AddMarker(self, frame: int, color = 0,
                  name: str = "", note: str = "",
                  duration: int = 0) -> bool:
        """Add a guide/marker at the given frame.

        Accepts both Wunjo Make-style (int category) and Resolve-style
        (string color name) for the color parameter.
        """
        if isinstance(color, str):
            category = MARKER_COLOR_MAP.get(color, 0)
        else:
            category = color
        comment = name if not note else f"{name}: {note}" if name else note
        return self._app.add_guide(frame, comment, category)

    def GetMarkers(self) -> dict[int, dict]:
        """Return all guides as {frame: {color, duration, note, name, customData}}.

        Compatible with Resolve's marker dict format.
        """
        guides = self._app.get_guides()
        result = {}
        for g in guides:
            frame = g.get("frame", 0)
            cat = g.get("category", 0)
            comment = g.get("comment", "")
            result[frame] = {
                "color": MARKER_CATEGORY_TO_COLOR.get(cat, "Purple"),
                "duration": 1,
                "note": comment,
                "name": comment,
                "customData": "",
                # Also keep Wunjo Make-native fields
                "comment": comment,
                "category": cat,
            }
        return result

    def DeleteMarker(self, frame: int) -> bool:
        """Delete the guide at the given frame."""
        return self._app.delete_guide(frame)

    def DeleteMarkerAtFrame(self, frame: int) -> bool:
        """Alias for DeleteMarker (Resolve compat)."""
        return self._app.delete_guide(frame)

    def DeleteMarkersByColor(self, color) -> bool:
        """Delete all guides of a specific color.

        Accepts string color name (Resolve) or int category (Wunjo Make).
        """
        if isinstance(color, str):
            category = MARKER_COLOR_MAP.get(color, -1)
            if category < 0:
                return False
        else:
            category = color
        return self._app.delete_guides_by_category(category)

    # ── Playback ───────────────────────────────────────────────────────

    def Seek(self, frame: int) -> None:
        """Move playhead to a frame."""
        self._app.seek(frame)

    def GetPosition(self) -> int:
        """Return current playhead position."""
        return self._app.get_position()

    def Play(self) -> None:
        self._app.play()

    def Pause(self) -> None:
        self._app.pause()

    # ── Resolve stubs ──────────────────────────────────────────────────

    def ApplyGradeFromDRX(self, path: str, grade_mode: int = 0,
                          clips: list | None = None) -> bool:
        """Apply DRX grade. No-op (Resolve-only feature)."""
        warnings.warn("ApplyGradeFromDRX() is a Resolve-only feature",
                      stacklevel=2)
        return True

    def GetCurrentClipThumbnailImage(self) -> dict | None:
        """Get current clip thumbnail. Stub — returns None."""
        return None

    def Export(self, file_path: str, export_type: int,
              export_sub_type: int | None = None) -> bool:
        """Export the timeline to a file.

        Wunjo Make supports OTIO export natively. Other formats return
        True as a stub.
        """
        warnings.warn(f"Timeline.Export('{file_path}'): limited support "
                      "in Wunjo Make", stacklevel=2)
        return True

    def GetStartFrame(self) -> int:
        """Return the timeline start frame. Always 0."""
        return 0

    def GetEndFrame(self) -> int:
        """Return the last frame of the timeline."""
        return self.GetTotalDuration()

    # ── Info ───────────────────────────────────────────────────────────

    def GetTotalDuration(self) -> int:
        """Calculate the total duration of all clips on all tracks."""
        tracks = self.GetAllTracksInfo()
        max_end = 0
        for t in tracks:
            track_id = t.get("id", 0)
            clips = self._app.get_clips_on_track(track_id)
            for c in clips:
                end = int(c.get("position", 0)) + int(c.get("duration", 0))
                if end > max_end:
                    max_end = end
        return max_end

    def PrintSummary(self) -> None:
        """Print a human-readable summary of the timeline."""
        tracks = self.GetAllTracksInfo()
        print(f"Timeline: {len(tracks)} tracks")
        for t in tracks:
            track_id = t.get("id", 0)
            ttype = "Audio" if t.get("audio", False) else "Video"
            tname = t.get("name", f"Track {track_id}")
            clips = self._app.get_clips_on_track(track_id)
            print(f"  [{ttype}] {tname}: {len(clips)} clips")
            for c in clips:
                pos = c.get("position", 0)
                dur = c.get("duration", 0)
                name = c.get("name", "?")
                print(f"    {name} @ {pos} ({dur} frames)")
