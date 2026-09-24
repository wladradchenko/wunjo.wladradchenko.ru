"""Clip selection tools: get, set, add, clear, select all/track/range."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_selection(ctx: Context) -> str:
        """Get the currently selected timeline item IDs.

        Returns a markdown table: clip_id | position (TC) | name
        For each selected item, shows its details using get_clip_info.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ids = resolve._app.get_selection()
            if not ids:
                return "No items selected."
            fps = helpers.get_fps(ctx)
            header = "| clip_id | position | name |"
            sep = "|---------|----------|------|"
            lines = [header, sep]
            for cid in ids:
                try:
                    info = resolve._app.get_timeline_clip_info(cid)
                    pos = info.get("position", "")
                    name = info.get("name", "")
                    tc = helpers.format_tc(int(pos), fps) if pos else ""
                    lines.append(f"| {cid} | {tc} | {name} |")
                except Exception:
                    lines.append(f"| {cid} | ? | ? |")
            return f"**{len(ids)} item(s) selected:**\n\n" + "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_selection(ctx: Context, ids: list[int]) -> str:
        """Set the timeline selection to specific item IDs.

        Args:
            ids: List of timeline clip/composition/subtitle IDs to select.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_selection(ids)
            if not ok:
                return f"ERROR: Could not set selection to {ids}"
            return f"Selected {len(ids)} item(s): {ids}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_to_selection(ctx: Context, item_id: int, clear: bool = False) -> str:
        """Add an item to the current selection.

        Args:
            item_id: Timeline clip/composition/subtitle ID to add.
            clear: If True, replaces the current selection instead of adding.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.add_to_selection(item_id, clear)
            if not ok:
                return f"ERROR: Could not add item {item_id} to selection"
            action = "Replaced selection with" if clear else "Added to selection:"
            return f"{action} item {item_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def clear_selection(ctx: Context) -> str:
        """Clear the current timeline selection."""
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.clear_selection()
            return "Selection cleared." if ok else "ERROR: Could not clear selection"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def select_all(ctx: Context) -> str:
        """Select all clips, compositions, and subtitles on the timeline."""
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.select_all()
            if not ok:
                return "ERROR: Could not select all"
            # Return count of selected items
            ids = resolve._app.get_selection()
            return f"Selected all items ({len(ids)} total)."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def select_current_track(ctx: Context) -> str:
        """Select all items on the currently active track."""
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.select_current_track()
            if not ok:
                return "ERROR: Could not select current track items"
            ids = resolve._app.get_selection()
            return f"Selected {len(ids)} item(s) on the active track."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def select_items_in_range(ctx: Context, track_ids: list[int],
                               start_frame: int, end_frame: int) -> str:
        """Select items within a frame range on specified tracks.

        Args:
            track_ids: List of track IDs to search.
            start_frame: Start frame of the range.
            end_frame: End frame of the range.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            ok = resolve._app.select_items_in_range(track_ids, start_frame, end_frame)
            if not ok:
                return "ERROR: Could not select items in range"
            ids = resolve._app.get_selection()
            tc_start = helpers.format_tc(start_frame, fps)
            tc_end = helpers.format_tc(end_frame, fps)
            return f"Selected {len(ids)} item(s) in range {tc_start}\u2013{tc_end} on tracks {track_ids}."
        except Exception as e:
            return f"ERROR: {e}"
