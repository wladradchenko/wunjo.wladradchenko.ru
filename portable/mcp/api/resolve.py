"""Resolve — top-level entry point (DaVinci Resolve API compatibility)."""

from __future__ import annotations

import warnings

from api.app_client import WunjoMakeClient
from api.project_manager import ProjectManager
from api.media_storage import MediaStorage
from api import constants


class Resolve:
    """Entry point for Wunjo Make scripting, mirroring DaVinci Resolve's API.

    Usage::

        from api import Resolve
        resolve = Resolve()
        pm = resolve.GetProjectManager()
        project = pm.GetCurrentProject()
        timeline = project.GetCurrentTimeline()
    """

    # ── Export type constants (resolve.EXPORT_*) ───────────────────────
    EXPORT_AAF = constants.EXPORT_AAF
    EXPORT_DRT = constants.EXPORT_DRT
    EXPORT_EDL = constants.EXPORT_EDL
    EXPORT_FCP_7_XML = constants.EXPORT_FCP_7_XML
    EXPORT_FCPXML_1_3 = constants.EXPORT_FCPXML_1_3
    EXPORT_FCPXML_1_4 = constants.EXPORT_FCPXML_1_4
    EXPORT_FCPXML_1_5 = constants.EXPORT_FCPXML_1_5
    EXPORT_FCPXML_1_6 = constants.EXPORT_FCPXML_1_6
    EXPORT_FCPXML_1_7 = constants.EXPORT_FCPXML_1_7
    EXPORT_FCPXML_1_8 = constants.EXPORT_FCPXML_1_8
    EXPORT_FCPXML_1_9 = constants.EXPORT_FCPXML_1_9
    EXPORT_FCPXML_1_10 = constants.EXPORT_FCPXML_1_10
    EXPORT_HDL = constants.EXPORT_HDL
    EXPORT_TEXT_CSV = constants.EXPORT_TEXT_CSV
    EXPORT_TEXT_TAB = constants.EXPORT_TEXT_TAB
    EXPORT_DOLBY_VISION_VER_2_9 = constants.EXPORT_DOLBY_VISION_VER_2_9
    EXPORT_DOLBY_VISION_VER_4_0 = constants.EXPORT_DOLBY_VISION_VER_4_0
    EXPORT_OTIO = constants.EXPORT_OTIO
    EXPORT_AAF_NEW = constants.EXPORT_AAF_NEW
    EXPORT_AAF_EXISTING = constants.EXPORT_AAF_EXISTING
    EXPORT_CDL = constants.EXPORT_CDL
    EXPORT_SDL = constants.EXPORT_SDL
    EXPORT_MISSING_CLIPS = constants.EXPORT_MISSING_CLIPS

    def __init__(self):
        self._app = WunjoMakeClient()

    def GetProjectManager(self) -> ProjectManager:
        """Return the ProjectManager singleton."""
        return ProjectManager(self._app)

    def GetMediaStorage(self) -> MediaStorage:
        """Return the MediaStorage object.

        In Resolve this accesses OS-level media locations. In Wunjo Make
        it wraps import operations on the project bin.
        """
        return MediaStorage(self._app)

    def Fusion(self):
        """Return the Fusion scripting object.

        Wunjo Make has no Fusion equivalent — returns None with a warning.
        """
        warnings.warn("Fusion() is a Resolve-only feature; returning None",
                      stacklevel=2)
        return None

    def OpenPage(self, page_name: str) -> bool:
        """Switch to a page in the application.

        Wunjo Make is a single-page editor — this is a no-op that always
        returns True for compatibility.
        """
        return True

    def GetCurrentPage(self) -> str:
        """Return the current page name. Always 'edit' for Wunjo Make."""
        return "edit"

    def GetProductName(self) -> str:
        return "Wunjo Make"

    def GetVersion(self) -> list:
        return [25, 0, 0, 0, ""]

    def GetVersionString(self) -> str:
        return "25.0.0"

    def GetCurrentLayoutPreset(self) -> str:
        """Return the current layout preset name. Stub."""
        return ""

    def LoadLayoutPreset(self, preset_name: str) -> bool:
        return True

    def UpdateLayoutPreset(self, preset_name: str) -> bool:
        return True

    def ExportLayoutPreset(self, preset_name: str, path: str) -> bool:
        return True

    def DeleteLayoutPreset(self, preset_name: str) -> bool:
        return True

    def SaveLayoutPreset(self, preset_name: str) -> bool:
        return True

    def ImportLayoutPreset(self, path: str, preset_name: str = "") -> bool:
        return True

    def Quit(self) -> None:
        """Quit the application."""
        self._app._call("exitApp")
