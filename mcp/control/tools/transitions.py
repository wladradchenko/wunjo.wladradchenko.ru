"""Transition tools: add, batch, remove."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def add_transition(ctx: Context, clip_id_a: int, clip_id_b: int, duration: int = 13) -> str:
        """Add a dissolve transition between two adjacent clips. For batch transitions on all clips, use add_transitions_batch instead.

        Args:
            clip_id_a: First clip's timeline ID.
            clip_id_b: Second clip's timeline ID.
            duration: Transition duration in frames (default 13 ~ 0.5s at 25fps).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.add_mix(clip_id_a, clip_id_b, duration)
            if not ok:
                return f"ERROR: Could not add transition between {clip_id_a} and {clip_id_b}"
            return f"Added dissolve ({duration}f) between clips {clip_id_a} and {clip_id_b}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_transitions_batch(ctx: Context, track_id: int, duration: int = 13) -> str:
        """Add dissolve transitions between all adjacent clip pairs on a track.

        Args:
            track_id: Track ID to process.
            duration: Transition duration in frames (default 13).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            clips = resolve._app.get_clips_on_track(track_id)
            if not clips or len(clips) < 2:
                return "ERROR: Need at least 2 clips on the track."

            count = 0
            errors = []
            for i in range(len(clips) - 1):
                a_id = clips[i].get("clip_id", clips[i].get("id"))
                b_id = clips[i + 1].get("clip_id", clips[i + 1].get("id"))
                ok = resolve._app.add_mix(a_id, b_id, duration)
                if ok:
                    count += 1
                else:
                    errors.append(f"{a_id}->{b_id}")

            result = f"Added {count} transition(s) ({duration}f each)"
            if errors:
                result += f"\nFailed pairs: {', '.join(errors)}"
            return result
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def remove_transition(ctx: Context, clip_id: int) -> str:
        """Remove a transition (mix) from a clip.

        Args:
            clip_id: The clip's timeline ID that has the transition.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.remove_mix(clip_id)
            if not ok:
                return f"ERROR: Could not remove transition from clip {clip_id}"
            return f"Removed transition from clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_available_transitions(ctx: Context) -> str:
        """List all available transition types for mixes and compositions.

        Returns a markdown table: id | name
        """
        try:
            resolve = helpers.get_resolve(ctx)
            transitions = resolve._app.get_available_transitions()
            if not transitions:
                return "No transitions found."
            lines = ["| id | name |", "|----|------|"]
            for t in transitions:
                lines.append(f"| {t.get('id', '')} | {t.get('name', '')} |")
            return f"{len(transitions)} transitions:\n\n" + "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_mix_params(ctx: Context, clip_id: int) -> str:
        """Get parameters of a mix/transition on a clip.

        Args:
            clip_id: Timeline clip ID that has a mix on its edge.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            params = resolve._app.get_mix_params(clip_id)
            if not params:
                return f"No mix found on clip {clip_id}"
            lines = [f"- **{k}:** {v}" for k, v in params.items()]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_mix_duration(ctx: Context, clip_id: int, duration: int) -> str:
        """Change the duration of a mix/transition.

        Args:
            clip_id: Timeline clip ID with a mix.
            duration: New mix duration in frames.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_mix_duration(clip_id, duration)
            if not ok:
                return f"ERROR: Could not set mix duration for clip {clip_id}"
            return f"Mix duration on clip {clip_id} set to {duration} frames"
        except Exception as e:
            return f"ERROR: {e}"
