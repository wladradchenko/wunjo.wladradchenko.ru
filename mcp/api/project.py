"""Project — wraps a Wunjo Make project (document)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.app_client import WunjoMakeClient

from api.media_pool import MediaPool
from api.timeline import Timeline


class Project:
    """Represents the currently open Wunjo Make project.

    Resolve has multiple timelines per project; Wunjo Make uses sequences.
    This wrapper treats the active sequence as the single timeline.
    """

    def __init__(self, app: WunjoMakeClient):
        self._app = app

    # ── Core ───────────────────────────────────────────────────────────

    def GetName(self) -> str:
        """Return the project name."""
        return self._app.get_project_name()

    def SetName(self, name: str) -> bool:
        """Set the project name. Maps to project property."""
        return self._app.set_project_property("projectName", name)

    def GetMediaPool(self) -> MediaPool:
        """Return the MediaPool (bin) for this project."""
        return MediaPool(self._app)

    def GetCurrentTimeline(self) -> Timeline:
        """Return the active timeline."""
        return Timeline(self._app)

    # ── Settings ───────────────────────────────────────────────────────

    def GetSetting(self, key: str) -> str:
        """Get a project property by key.

        Supports Resolve-style keys with mapping:
            timelineFrameRate → fps
            timelineResolutionWidth → width
            timelineResolutionHeight → height
        """
        # Map Resolve setting keys to Wunjo Make equivalents
        if key == "timelineFrameRate":
            return str(self._app.get_project_fps())
        elif key == "timelineResolutionWidth":
            return str(self._app.get_project_resolution_width())
        elif key == "timelineResolutionHeight":
            return str(self._app.get_project_resolution_height())
        return self._app.get_project_property(key)

    def SetSetting(self, key: str, value: str) -> bool:
        """Set a project property."""
        return self._app.set_project_property(key, value)

    def GetFps(self) -> float:
        """Return the project frame rate."""
        return self._app.get_project_fps()

    def GetResolution(self) -> tuple[int, int]:
        """Return (width, height) of the project."""
        w = self._app.get_project_resolution_width()
        h = self._app.get_project_resolution_height()
        return (w, h)

    def GetResolutionWidth(self) -> int:
        """Return the project width in pixels."""
        return self._app.get_project_resolution_width()

    def GetResolutionHeight(self) -> int:
        """Return the project height in pixels."""
        return self._app.get_project_resolution_height()

    # ── Save ───────────────────────────────────────────────────────────

    def Save(self) -> bool:
        """Save the project."""
        return self._app.save_project()

    def SaveAs(self, file_path: str) -> bool:
        """Save the project to a new path."""
        return self._app.save_project_as(file_path)

    def GetProjectPath(self) -> str:
        """Return the file path of the project."""
        return self._app.get_project_path()

    # ── Timeline management (Resolve compatibility) ────────────────────

    def GetTimelineCount(self) -> int:
        """Return the number of timelines.

        Wunjo Make has one active timeline (sequence). Returns 1 if a
        timeline exists, 0 otherwise.
        """
        try:
            count = self._app.get_track_count("video")
            return 1 if count > 0 else 0
        except Exception:
            return 1

    def GetTimelineByIndex(self, index: int) -> Timeline | None:
        """Return a timeline by 1-based index.

        Wunjo Make has one active timeline — returns it when index == 1.
        """
        if index == 1:
            return Timeline(self._app)
        return None

    def SetCurrentTimeline(self, timeline) -> bool:
        """Set the current timeline. No-op (single timeline)."""
        return True

    def GetCurrentVideoItem(self):
        """Get current video item. Returns None (Resolve-specific)."""
        return None

    # ── Render (Resolve compatibility stubs) ───────────────────────────

    def LoadRenderPreset(self, preset_name: str) -> bool:
        """Load a render preset. Stub — Wunjo Make uses render profiles."""
        warnings.warn(f"LoadRenderPreset('{preset_name}'): Wunjo Make uses render "
                      "profiles; ignoring preset name", stacklevel=2)
        return True

    def SetCurrentRenderFormatAndCodec(self, render_format: str,
                                        render_codec: str) -> bool:
        """Set render format and codec. Stub."""
        self._app.set_project_property("renderFormat", render_format)
        self._app.set_project_property("renderCodec", render_codec)
        return True

    def SetRenderSettings(self, settings: dict) -> bool:
        """Set render settings from a dict. Stub — stores as properties."""
        for key, value in settings.items():
            self._app.set_project_property(f"render_{key}", str(value))
        return True

    def GetRenderSettings(self) -> dict:
        """Return render settings. Stub."""
        return {}

    def AddRenderJob(self) -> str:
        """Add a render job. Stub — returns a fake job ID."""
        return "job_1"

    def StartRendering(self, *job_ids, is_interactive_mode: bool = False) -> bool:
        """Start rendering. Triggers Wunjo Make's scriptRender."""
        self._app.render("")
        return True

    def IsRenderingInProgress(self) -> bool:
        """Check if rendering is in progress. Stub — returns False."""
        return False

    def GetRenderJobList(self) -> list[dict]:
        """Return the render job list. Stub."""
        return []

    def GetRenderJobStatus(self, job_id) -> dict:
        """Return render job status. Stub."""
        return {"JobStatus": "Complete"}

    def DeleteAllRenderJobs(self) -> bool:
        """Delete all render jobs. Stub."""
        return True

    def DeleteRenderJob(self, job_id: str) -> bool:
        """Delete a render job. Stub."""
        return True
