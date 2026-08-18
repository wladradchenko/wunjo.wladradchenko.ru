"""Zone tools: get/set timeline in/out points, extract zone."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_zone(ctx: Context) -> str:
        """Get the current timeline zone (in/out points).

        Returns the zone in/out as timecodes and frame numbers.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            zone = resolve._app.get_zone()
            if not zone:
                return "No zone set (or no timeline open)."
            zone_in = int(zone.get("zoneIn", 0))
            zone_out = int(zone.get("zoneOut", 0))
            in_tc = helpers.format_tc(zone_in, fps)
            out_tc = helpers.format_tc(zone_out, fps)
            duration = zone_out - zone_in
            dur_tc = helpers.format_tc(duration, fps)
            return (f"Zone: {in_tc} → {out_tc} "
                    f"(frames {zone_in}–{zone_out}, duration {dur_tc} = {duration} frames)")
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_zone(ctx: Context, in_frame: int, out_frame: int) -> str:
        """Set the timeline zone (in/out points).

        Args:
            in_frame: Zone start frame.
            out_frame: Zone end frame.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_zone(in_frame, out_frame)
            if not ok:
                return "ERROR: Could not set zone (check timeline)"
            fps = helpers.get_fps(ctx)
            in_tc = helpers.format_tc(in_frame, fps)
            out_tc = helpers.format_tc(out_frame, fps)
            return f"Set zone: {in_tc} → {out_tc} (frames {in_frame}–{out_frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_zone_in(ctx: Context, in_frame: int) -> str:
        """Set only the zone in-point (start), keeping the current out-point.

        Args:
            in_frame: New zone start frame.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_zone_in(in_frame)
            if not ok:
                return "ERROR: Could not set zone in-point"
            fps = helpers.get_fps(ctx)
            return f"Set zone in-point: {helpers.format_tc(in_frame, fps)} (frame {in_frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_zone_out(ctx: Context, out_frame: int) -> str:
        """Set only the zone out-point (end), keeping the current in-point.

        Args:
            out_frame: New zone end frame.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_zone_out(out_frame)
            if not ok:
                return "ERROR: Could not set zone out-point"
            fps = helpers.get_fps(ctx)
            return f"Set zone out-point: {helpers.format_tc(out_frame, fps)} (frame {out_frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def extract_zone(ctx: Context, in_frame: int, out_frame: int,
                     lift_only: bool = False) -> str:
        """Extract (remove) content in the specified zone from active tracks.

        Args:
            in_frame: Zone start frame.
            out_frame: Zone end frame.
            lift_only: If True, lift without closing the gap. If False, ripple delete (close gap).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.extract_zone(in_frame, out_frame, lift_only)
            if not ok:
                return "ERROR: Could not extract zone (check active tracks)"
            fps = helpers.get_fps(ctx)
            in_tc = helpers.format_tc(in_frame, fps)
            out_tc = helpers.format_tc(out_frame, fps)
            mode = "lift" if lift_only else "ripple delete"
            return f"Extracted zone {in_tc} → {out_tc} ({mode})"
        except Exception as e:
            return f"ERROR: {e}"
