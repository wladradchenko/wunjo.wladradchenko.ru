"""Playback tools: seek, position, play, pause."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def seek_to(ctx: Context, frame: int) -> str:
        """Seek the playhead to a specific frame position.

        Args:
            frame: Target frame number to seek to.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            resolve._app.seek(frame)
            tc = helpers.format_tc(frame, fps)
            return f"Seeked to {tc} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_position(ctx: Context) -> str:
        """Get the current playhead position.

        Returns the current frame number and timecode.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frame = int(resolve._app.get_position())
            tc = helpers.format_tc(frame, fps)
            return f"Position: {tc} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def play(ctx: Context) -> str:
        """Start playback in the timeline monitor."""
        try:
            resolve = helpers.get_resolve(ctx)
            resolve._app.play()
            return "Playback started."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def pause(ctx: Context) -> str:
        """Pause playback in the timeline monitor."""
        try:
            resolve = helpers.get_resolve(ctx)
            resolve._app.pause()
            return "Playback paused."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_playback_speed(ctx: Context) -> str:
        """Get the current playback speed multiplier.

        Returns the speed as a float (e.g. 1.0 = normal, 2.0 = double).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            speed = resolve._app.get_playback_speed()
            return f"Playback speed: {speed}x"
        except Exception as e:
            return f"ERROR: {e}"
