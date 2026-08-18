"""Marker tools: add, delete, list, delete by color (timeline + clip-level)."""

from __future__ import annotations

from mcp.server.fastmcp import Context

from api.constants import MARKER_COLOR_MAP, MARKER_CATEGORY_TO_COLOR


def register(mcp, helpers):

    # ── Timeline markers (guides) ─────────────────────────────────────

    @mcp.tool()
    def get_markers(ctx: Context) -> str:
        """List all timeline markers/guides.

        Returns a markdown table: frame | timecode | color | label
        """
        try:
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            markers = tl.GetMarkers()
            if not markers:
                return "No markers on timeline."
            return helpers.markers_table(markers, fps)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_marker(ctx: Context, frame: int, label: str, color: str = "Purple", note: str = "") -> str:
        """Add a marker/guide on the timeline.

        Args:
            frame: Frame position.
            label: Marker name/label.
            color: Color name (Purple, Blue, Cyan, Green, Yellow, Orange, Red).
            note: Optional note text.
        """
        try:
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            ok = tl.AddMarker(frame, color, label, note)
            if not ok:
                return f"ERROR: Could not add marker at frame {frame}"
            tc = helpers.format_tc(frame, fps)
            return f"Added marker '{label}' at {tc} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_marker(ctx: Context, frame: int) -> str:
        """Delete a marker at a specific frame.

        Args:
            frame: Frame position of the marker to delete.
        """
        try:
            tl = helpers.get_timeline(ctx)
            ok = tl.DeleteMarker(frame)
            if not ok:
                return f"ERROR: No marker at frame {frame}"
            return f"Deleted marker at frame {frame}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_markers_by_color(ctx: Context, color: str) -> str:
        """Delete all markers of a given color.

        Args:
            color: Color name (Purple, Blue, Cyan, Green, Yellow, Orange, Red).
        """
        try:
            tl = helpers.get_timeline(ctx)
            ok = tl.DeleteMarkersByColor(color)
            if not ok:
                return f"ERROR: Could not delete markers with color '{color}'"
            return f"Deleted all '{color}' markers"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Clip-level markers ────────────────────────────────────────────

    @mcp.tool()
    def get_clip_markers(ctx: Context, bin_clip_id: str) -> str:
        """List all markers on a media pool clip.

        Args:
            bin_clip_id: Media pool clip ID (from get_media_pool).

        Returns a markdown table: frame | timecode | color | label
        """
        try:
            fps = helpers.get_fps(ctx)
            mp = helpers.get_media_pool(ctx)
            item = mp.GetClipById(bin_clip_id)
            markers = item.GetMarkers()
            if not markers:
                return f"No markers on clip {bin_clip_id}."
            return helpers.markers_table(markers, fps)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_clip_marker(ctx: Context, bin_clip_id: str, frame: int, label: str,
                        color: str = "Purple", note: str = "") -> str:
        """Add a marker to a media pool clip.

        Args:
            bin_clip_id: Media pool clip ID (from get_media_pool).
            frame: Frame position (relative to clip start).
            label: Marker name/label.
            color: Color name (Purple, Blue, Cyan, Green, Yellow, Orange, Red).
            note: Optional note text.
        """
        try:
            fps = helpers.get_fps(ctx)
            mp = helpers.get_media_pool(ctx)
            item = mp.GetClipById(bin_clip_id)
            ok = item.AddMarker(frame, color, label, note)
            if not ok:
                return f"ERROR: Could not add marker at frame {frame} on clip {bin_clip_id}"
            tc = helpers.format_tc(frame, fps)
            return f"Added marker '{label}' at {tc} (frame {frame}) on clip {bin_clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_clip_marker(ctx: Context, bin_clip_id: str, frame: int) -> str:
        """Delete a marker from a media pool clip at a specific frame.

        Args:
            bin_clip_id: Media pool clip ID (from get_media_pool).
            frame: Frame position of the marker to delete.
        """
        try:
            mp = helpers.get_media_pool(ctx)
            item = mp.GetClipById(bin_clip_id)
            ok = item.DeleteMarkerAtFrame(frame)
            if not ok:
                return f"ERROR: No marker at frame {frame} on clip {bin_clip_id}"
            return f"Deleted marker at frame {frame} on clip {bin_clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_clip_markers_by_color(ctx: Context, bin_clip_id: str, color: str) -> str:
        """Delete all markers of a given color from a media pool clip.

        Args:
            bin_clip_id: Media pool clip ID (from get_media_pool).
            color: Color name (Purple, Blue, Cyan, Green, Yellow, Orange, Red).
        """
        try:
            mp = helpers.get_media_pool(ctx)
            item = mp.GetClipById(bin_clip_id)
            ok = item.DeleteMarkersByColor(color)
            if not ok:
                return f"ERROR: Could not delete '{color}' markers on clip {bin_clip_id}"
            return f"Deleted all '{color}' markers on clip {bin_clip_id}"
        except Exception as e:
            return f"ERROR: {e}"
