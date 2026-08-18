"""Replace clip tool: compound delete + insert + duration match."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def replace_clip(ctx: Context, clip_id: int, new_bin_id: str, match_duration: bool = True) -> str:
        """Replace a timeline clip by clip_id with a different media pool clip. To replace by scene number (1-38) with auto-import, use replace_scene instead.

        Preserves the original position and (optionally) duration.

        Args:
            clip_id: Timeline clip ID to replace.
            new_bin_id: Media pool bin ID of the replacement clip.
            match_duration: If True, trim the new clip to match the old duration.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            tl = helpers.get_timeline(ctx)

            # Get old clip info
            info = resolve._app.get_timeline_clip_info(clip_id)
            if not info:
                return f"ERROR: Clip {clip_id} not found."

            old_start = int(info.get("start", 0))
            old_dur = int(info.get("duration", 0))
            old_track = int(info.get("track_id", 1))
            old_name = info.get("name", "?")

            # Delete old clip
            ok = resolve._app.delete_timeline_clip(clip_id)
            if not ok:
                return f"ERROR: Could not delete clip {clip_id}"

            # Insert new clip at same position
            new_item = tl.InsertClip(new_bin_id, old_track, old_start)
            if new_item is None:
                return f"ERROR: Deleted old clip but failed to insert new bin_id={new_bin_id}"

            # Match duration
            if match_duration and old_dur > 0:
                resolve._app.resize_clip(new_item.clip_id, old_dur, True)

            tc = helpers.format_tc(old_start, fps)
            return (
                f"Replaced '{old_name}' (clip {clip_id}) with bin_id={new_bin_id} "
                f"(new clip {new_item.clip_id}) at {tc}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def relink_clip(ctx: Context, bin_id: str, new_file_path: str) -> str:
        """Relink a media pool clip to a new file path.

        Changes the source file of a bin clip while preserving ALL timeline
        instances, their positions, durations, effects, and transitions.
        Use this instead of replace_clip when you want to keep everything intact.

        Args:
            bin_id: Media pool clip ID to relink.
            new_file_path: Absolute path to the new source file.
        """
        try:
            import os
            if not os.path.isabs(new_file_path):
                return "ERROR: new_file_path must be an absolute path."
            if not os.path.isfile(new_file_path):
                return f"ERROR: File not found: {new_file_path}"

            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.relink_bin_clip(bin_id, new_file_path)
            if ok:
                return f"Relinked bin clip {bin_id} → {new_file_path}"
            return f"ERROR: Relink failed for bin_id={bin_id}. Check bin ID and file path."
        except Exception as e:
            return f"ERROR: {e}"
