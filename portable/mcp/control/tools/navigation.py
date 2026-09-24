"""Timeline navigation tools: jump to markers, edit points."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def go_to_next_marker(ctx: Context) -> str:
        """Seek playhead to the next guide/marker on the timeline."""
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frame = int(resolve._app.go_to_next_marker())
            if frame < 0:
                return "No next marker found"
            return f"Seeked to next marker at {helpers.format_tc(int(frame), fps)} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def go_to_previous_marker(ctx: Context) -> str:
        """Seek playhead to the previous guide/marker on the timeline."""
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frame = int(resolve._app.go_to_previous_marker())
            if frame < 0:
                return "No previous marker found"
            return f"Seeked to previous marker at {helpers.format_tc(int(frame), fps)} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def go_to_next_edit(ctx: Context) -> str:
        """Seek playhead to the next clip boundary (cut point) on any track."""
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frame = int(resolve._app.go_to_next_edit())
            if frame < 0:
                return "No next edit point found"
            return f"Seeked to next edit at {helpers.format_tc(int(frame), fps)} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def go_to_previous_edit(ctx: Context) -> str:
        """Seek playhead to the previous clip boundary (cut point) on any track."""
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frame = int(resolve._app.go_to_previous_edit())
            if frame < 0:
                return "No previous edit point found"
            return f"Seeked to previous edit at {helpers.format_tc(int(frame), fps)} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"
