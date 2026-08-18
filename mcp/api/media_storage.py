"""MediaStorage — wraps media import from OS-level paths.

In DaVinci Resolve, MediaStorage provides access to OS-mounted volumes
and can import media from paths. In Wunjo Make, we simply delegate to the
project bin (MediaPool) via D-Bus.
"""

from __future__ import annotations

import os
import glob as globmod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.app_client import WunjoMakeClient


class MediaStorage:
    """Resolve-compatible MediaStorage wrapper for Wunjo Make.

    Example (Resolve-style)::

        storage = resolve.GetMediaStorage()
        clips = storage.AddItemListToMediaPool("/path/to/media/")
        clips = storage.AddItemListToMediaPool(["/a.mp4", "/b.mp4"])
    """

    def __init__(self, app: WunjoMakeClient):
        self._app = app

    def GetMountedVolumeList(self) -> list[str]:
        """Return mounted volumes. Returns common paths on the current OS."""
        import sys
        if sys.platform.startswith("win"):
            # Return existing drive letters
            import string
            return [f"{d}:\\" for d in string.ascii_uppercase
                    if os.path.exists(f"{d}:\\")]
        elif sys.platform.startswith("darwin"):
            return ["/Volumes"]
        else:
            return ["/", "/media", "/mnt"]

    def GetSubFolderList(self, folder_path: str) -> list[str]:
        """Return subdirectories of a folder."""
        if not os.path.isdir(folder_path):
            return []
        return [os.path.join(folder_path, d)
                for d in sorted(os.listdir(folder_path))
                if os.path.isdir(os.path.join(folder_path, d))]

    def GetFileList(self, folder_path: str) -> list[str]:
        """Return files in a folder."""
        if not os.path.isdir(folder_path):
            return []
        return [os.path.join(folder_path, f)
                for f in sorted(os.listdir(folder_path))
                if os.path.isfile(os.path.join(folder_path, f))]

    def AddItemListToMediaPool(self, items) -> list:
        """Import media items into the Wunjo Make bin (media pool).

        Args:
            items: Can be:
                - A folder path (str) — imports all media files from that folder.
                - A list of file paths (list[str]).
                - A single file path (str ending with a media extension).
                - A list of subclip dicts: [{"media": path, "startFrame": int, "endFrame": int}].
                  (startFrame/endFrame are stored as metadata but Wunjo Make imports the full clip.)

        Returns:
            List of MediaPoolItem objects.
        """
        from api.media_pool import MediaPoolItem

        paths = []

        if isinstance(items, str):
            # Single path — could be folder or file
            if os.path.isdir(items):
                # Import all media files from folder
                media_exts = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm",
                              ".mp3", ".wav", ".flac", ".aac", ".ogg",
                              ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".exr"}
                for f in sorted(os.listdir(items)):
                    if os.path.splitext(f)[1].lower() in media_exts:
                        paths.append(os.path.join(items, f))
            else:
                paths.append(items)
        elif isinstance(items, (list, tuple)):
            for item in items:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict):
                    # Subclip dict: {"media": path} or {"mediaPoolItem": clip}
                    media_path = item.get("media", "")
                    if media_path:
                        paths.append(media_path)
                    # If it's a mediaPoolItem reference, skip import
                    # (already in bin)
        else:
            return []

        if not paths:
            return []

        # Normalize all paths
        paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
        if not paths:
            return []

        bin_ids = self._app.import_media(paths, "-1")
        return [MediaPoolItem(self._app, bid) for bid in bin_ids]

    def RevealInStorage(self, path: str) -> bool:
        """Reveal a path in the OS file manager. Stub."""
        return True
