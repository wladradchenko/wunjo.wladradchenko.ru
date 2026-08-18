"""Speed tools: constant speed, time remap (speed ramping)."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def set_clip_speed(ctx: Context, clip_id: int, speed_factor: float,
                       pitch_compensate: bool = False) -> str:
        """Set clip playback speed (constant).

        Args:
            clip_id: Timeline clip ID.
            speed_factor: Speed multiplier (0.5 = half speed, 2.0 = double).
            pitch_compensate: Keep original audio pitch.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            speed_pct = speed_factor * 100.0
            ok = resolve._app.set_clip_speed(clip_id, speed_pct, pitch_compensate)
            if not ok:
                return f"ERROR: Could not set speed {speed_factor}x on clip {clip_id}"
            return f"Set speed {speed_factor}x ({speed_pct:.0f}%) on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def enable_time_remap(ctx: Context, clip_id: int, enable: bool = True) -> str:
        """Enable or disable time remap (variable speed / speed ramping) on a clip.

        Time remap allows variable speed within a single clip using keyframes.
        Enable it first, then use set_time_remap to set the speed curve.

        Args:
            clip_id: Timeline clip ID.
            enable: True to enable, False to disable and restore original speed.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.enable_time_remap(clip_id, enable)
            if not ok:
                return f"ERROR: Could not {'enable' if enable else 'disable'} time remap on clip {clip_id}"
            return f"Time remap {'enabled' if enable else 'disabled'} on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_time_remap(ctx: Context, clip_id: int) -> str:
        """Get time remap info for a clip.

        Args:
            clip_id: Timeline clip ID.

        Returns enabled status, time_map keyframes, pitch, and image_mode.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            info = resolve._app.get_time_remap(clip_id)
            if not info:
                return f"ERROR: Could not get time remap for clip {clip_id}"
            enabled = info.get("enabled", False)
            if not enabled:
                return f"Clip {clip_id}: time remap NOT enabled"
            lines = [
                f"**Clip {clip_id} time remap:**",
                f"**time_map:** {info.get('time_map', '')}",
                f"**pitch:** {info.get('pitch', 0)}",
                f"**image_mode:** {info.get('image_mode', 'nearest')}",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_time_remap(ctx: Context, clip_id: int, time_map: str,
                       pitch: int = 0, image_mode: str = "nearest") -> str:
        """Set time remap keyframes on a clip (must enable time_remap first).

        The time_map format is "timecode=seconds;timecode=seconds;..."
        where timecode is the timeline position (HH:MM:SS.mmm) and
        seconds is the source time in seconds.

        Args:
            clip_id: Timeline clip ID (must have time remap enabled).
            time_map: Keyframe animation data.
            pitch: 1 for pitch compensation, 0 for normal.
            image_mode: "nearest" (sharp cuts) or "blend" (interpolated).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_time_remap(clip_id, time_map, pitch, image_mode)
            if not ok:
                return f"ERROR: Could not set time remap on clip {clip_id} (is it enabled?)"
            return f"Time remap updated on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"
