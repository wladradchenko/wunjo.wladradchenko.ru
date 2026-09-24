"""Media pool tools: import, glob import, listing, bin folders."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_media_pool(ctx: Context, folder_id: str = "-1") -> str:
        """List all clips in the media pool (or a specific folder).

        Args:
            folder_id: Bin folder ID. "-1" for all clips.

        Returns a markdown table: bin_id | name | type | duration_frames | duration_tc
        """
        try:
            pool = helpers.get_media_pool(ctx)
            fps = helpers.get_fps(ctx)
            if folder_id == "-1":
                items = pool.GetAllClips()
            else:
                # Get clips from specific folder via D-Bus
                resolve = helpers.get_resolve(ctx)
                app = resolve._app
                clip_ids = app.get_folder_clip_ids(folder_id)
                items = [pool.GetClipById(cid) for cid in clip_ids]
            if not items:
                return "Media pool is empty."
            return helpers.media_table(items, fps)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def import_media(ctx: Context, file_paths: list[str], folder_name: str = "") -> str:
        """Import media files into the media pool. For full import + timeline assembly, use build_timeline instead.

        Args:
            file_paths: List of absolute file paths to import.
            folder_name: Optional folder name to import into (created if missing).

        Returns a markdown table of imported clips.
        """
        try:
            pool = helpers.get_media_pool(ctx)
            fps = helpers.get_fps(ctx)
            folder = None
            if folder_name:
                folder = pool.AddSubFolder(None, folder_name)
            items = pool.ImportMedia(file_paths, folder)
            if not items:
                return "ERROR: No clips imported."
            return f"Imported {len(items)} clip(s):\n\n{helpers.media_table(items, fps)}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def import_media_glob(ctx: Context, directory: str, pattern: str = "*.mp4", folder_name: str = "") -> str:
        """Import media files matching a glob pattern from a directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern (e.g. "*.mp4", "scene*-final.mp4").
            folder_name: Optional folder name to import into.

        Returns a markdown table of imported clips.
        """
        try:
            pool = helpers.get_media_pool(ctx)
            fps = helpers.get_fps(ctx)
            folder = None
            if folder_name:
                folder = pool.AddSubFolder(None, folder_name)
            items = pool.ImportMediaFromFolder(directory, pattern, folder)
            if not items:
                return "ERROR: No files matched or import failed."
            return f"Imported {len(items)} clip(s):\n\n{helpers.media_table(items, fps)}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def create_bin_folder(ctx: Context, name: str, parent_id: str = "-1") -> str:
        """Create a folder in the media pool bin.

        Args:
            name: Folder name.
            parent_id: Parent folder ID. "-1" for root.
        """
        try:
            pool = helpers.get_media_pool(ctx)
            parent = None
            if parent_id != "-1":
                # Use root folder as parent; sub-folder navigation limited by D-Bus
                parent = pool.GetRootFolder()
            folder = pool.AddSubFolder(parent, name)
            if folder is None:
                return f"ERROR: Could not create folder '{name}'"
            fid = folder.folder_id if hasattr(folder, "folder_id") else "?"
            return f"Created folder '{name}' (id: {fid})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def move_bin_clip(ctx: Context, bin_id: str, target_folder_id: str) -> str:
        """Move a media pool clip to a different bin folder.

        Args:
            bin_id: Media pool clip ID.
            target_folder_id: Target folder ID. "-1" for root.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.move_bin_clip(bin_id, target_folder_id)
            if ok:
                return f"Moved clip {bin_id} to folder {target_folder_id}"
            return f"ERROR: Could not move clip {bin_id} to folder {target_folder_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def rename_bin_clip(ctx: Context, bin_id: str, new_name: str) -> str:
        """Rename a clip in the media pool.

        Args:
            bin_id: Media pool clip ID.
            new_name: New clip name.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.rename_bin_clip(bin_id, new_name)
            if ok:
                return f"Renamed clip {bin_id} → '{new_name}'"
            return f"ERROR: Could not rename bin clip {bin_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_bin_clip_properties(ctx: Context, bin_id: str) -> str:
        """Get properties of a media pool (bin) clip.

        Args:
            bin_id: The clip's bin ID.

        Returns name, duration, type, url and timecode info as a markdown list.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            props = resolve._app.get_clip_properties(bin_id)
            if not props:
                return f"ERROR: No properties returned for bin clip {bin_id}"
            lines = []
            for key, value in props.items():
                if key == "duration":
                    dur = int(value)
                    tc = helpers.format_tc(dur, fps)
                    lines.append(f"- **{key}:** {dur} frames ({tc})")
                else:
                    lines.append(f"- **{key}:** {value}")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_all_clip_ids(ctx: Context) -> str:
        """Get a flat list of all clip IDs in the media pool.

        Returns all bin clip IDs as a comma-separated list with a total count.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ids = resolve._app.get_all_clip_ids()
            if not ids:
                return "Media pool is empty (0 clips)."
            return f"**{len(ids)} clip(s):** {', '.join(ids)}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clips_on_track(ctx: Context, track_id: int) -> str:
        """Get all clips on a specific timeline track.

        Args:
            track_id: The track ID (use get_track_list to find IDs).

        Returns a markdown table with clip_id, position, duration and name.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            clips = resolve._app.get_clips_on_track(track_id)
            if not clips:
                return f"No clips on track {track_id}."
            header = "| clip_id | position | duration | name |"
            sep = "|---------|----------|----------|------|"
            lines = [header, sep]
            for c in clips:
                cid = c.get("clip_id", c.get("id", ""))
                pos = int(c.get("position", 0))
                dur = int(c.get("duration", 0))
                name = c.get("name", "")
                pos_tc = helpers.format_tc(pos, fps)
                dur_tc = helpers.format_tc(dur, fps)
                lines.append(f"| {cid} | {pos_tc} | {dur_tc} | {name} |")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_metadata(ctx: Context, bin_id: str) -> str:
        """Get extended metadata for a media pool clip (codec, resolution, file size, etc.).

        Args:
            bin_id: The clip's bin ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            meta = resolve._app.get_clip_metadata(bin_id)
            if not meta or meta.get("id") is None:
                return f"ERROR: Clip {bin_id} not found"
            lines = []
            for key, val in meta.items():
                if val:
                    lines.append(f"- **{key}:** {val}")
            return "\n".join(lines) if lines else f"No metadata for clip {bin_id}"
        except Exception as e:
            return f"ERROR: {e}"
