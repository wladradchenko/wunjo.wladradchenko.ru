"""Audio tools: volume, fades, mute, levels."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def set_clip_volume(ctx: Context, clip_id: int, dB: float = 0.0) -> str:
        """Set audio volume (gain) on a timeline clip.

        Args:
            clip_id: Timeline clip ID.
            dB: Volume in decibels. 0.0 = unity (no change), -6.0 = half, +6.0 = double.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_clip_volume(clip_id, dB)
            if not ok:
                return f"ERROR: Could not set volume {dB} dB on clip {clip_id}"
            return f"Set volume {dB} dB on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_volume(ctx: Context, clip_id: int) -> str:
        """Get the current audio volume of a timeline clip in dB.

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            vol = resolve._app.get_clip_volume(clip_id)
            return f"Clip {clip_id} volume: {vol} dB"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_audio_fade(ctx: Context, clip_id: int,
                       fade_in: int = -1, fade_out: int = -1) -> str:
        """Set audio fade in/out on a timeline clip.

        Args:
            clip_id: Timeline clip ID.
            fade_in: Fade-in duration in frames (-1 to skip, 0 to remove).
            fade_out: Fade-out duration in frames (-1 to skip, 0 to remove).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_audio_fade(clip_id, fade_in, fade_out)
            if not ok:
                return f"ERROR: Could not set audio fade on clip {clip_id}"
            parts = []
            if fade_in >= 0:
                parts.append(f"fade-in={fade_in}f")
            if fade_out >= 0:
                parts.append(f"fade-out={fade_out}f")
            return f"Set audio {', '.join(parts)} on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def split_audio(ctx: Context, clip_id: int) -> str:
        """Separate audio from a video clip onto its own audio track.

        The original clip becomes video-only and a new audio-only clip
        is created on an available audio track at the same position.

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.split_audio(clip_id)
            if not ok:
                return f"ERROR: Could not split audio from clip {clip_id}"
            return f"Split audio from clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_track_mute(ctx: Context, track_id: int, mute: bool) -> str:
        """Mute or unmute a timeline track.

        Args:
            track_id: Track ID.
            mute: True to mute, False to unmute.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_track_mute(track_id, mute)
            if not ok:
                return f"ERROR: Could not {'mute' if mute else 'unmute'} track {track_id}"
            return f"Track {track_id} {'muted' if mute else 'unmuted'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_track_mute(ctx: Context, track_id: int) -> str:
        """Check if a track is muted.

        Args:
            track_id: Track ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            muted = resolve._app.get_track_mute(track_id)
            return f"Track {track_id} is {'muted' if muted else 'not muted'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_track_locked(ctx: Context, track_id: int, locked: bool) -> str:
        """Lock or unlock a timeline track. Locked tracks prevent editing.

        Args:
            track_id: Track ID.
            locked: True to lock, False to unlock.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_track_locked(track_id, locked)
            if not ok:
                return f"ERROR: Could not {'lock' if locked else 'unlock'} track {track_id}"
            return f"Track {track_id} {'locked' if locked else 'unlocked'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_track_locked(ctx: Context, track_id: int) -> str:
        """Check if a track is locked.

        Args:
            track_id: Track ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            locked = resolve._app.get_track_locked(track_id)
            return f"Track {track_id} is {'locked' if locked else 'unlocked'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_track_hidden(ctx: Context, track_id: int, hidden: bool) -> str:
        """Hide or show a timeline track.

        Args:
            track_id: Track ID.
            hidden: True to hide, False to show.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_track_hidden(track_id, hidden)
            if not ok:
                return f"ERROR: Could not {'hide' if hidden else 'show'} track {track_id}"
            return f"Track {track_id} {'hidden' if hidden else 'visible'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_track_hidden(ctx: Context, track_id: int) -> str:
        """Check if a track is hidden.

        Args:
            track_id: Track ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            hidden = resolve._app.get_track_hidden(track_id)
            return f"Track {track_id} is {'hidden' if hidden else 'visible'}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_audio_levels(ctx: Context, bin_clip_id: str, stream: int = 0,
                         downsample: int = 5, mode: str = "peak") -> str:
        """Get audio peak levels from a media pool clip as a list of normalized values (0.0-1.0).

        Useful for beat detection, sync, or verifying audio presence.

        Args:
            bin_clip_id: Media pool clip ID.
            stream: Audio stream index (default 0 = first stream).
            downsample: Frames per sample (1 = per-frame, 5 = every 5 frames). Higher = less data.
            mode: "peak" (default) or "rms". RMS gives smoother energy envelope, better for procedural effects.
        """
        try:
            mode_int = 1 if mode == "rms" else 0
            resolve = helpers.get_resolve(ctx)
            levels = resolve._app.get_audio_levels(bin_clip_id, stream, downsample, mode_int)
            if not levels:
                return f"No audio levels for clip {bin_clip_id} stream {stream} (audio cache may not be ready)"
            # Summarize
            peak = max(levels)
            avg = sum(levels) / len(levels)
            mode_label = mode.upper()
            return (
                f"Audio levels [{mode_label}] for clip {bin_clip_id} (stream {stream}, downsample {downsample}):\n"
                f"  Samples: {len(levels)}\n"
                f"  Peak: {peak:.3f}\n"
                f"  Average: {avg:.3f}\n"
                f"  Values: [{', '.join(f'{v:.3f}' for v in levels[:50])}"
                f"{'...' if len(levels) > 50 else ''}]"
            )
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_clip_pan(ctx: Context, clip_id: int, pan: float) -> str:
        """Set audio pan (stereo balance).

        Args:
            clip_id: Timeline clip ID.
            pan: Pan value from -100 (full left) to +100 (full right). 0 = center.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_clip_pan(clip_id, pan)
            if not ok:
                return f"ERROR: Could not set pan for clip {clip_id}"
            side = "center" if pan == 0 else f"{'left' if pan < 0 else 'right'} ({abs(pan):.0f}%)"
            return f"Clip {clip_id} pan set to {side}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_pan(ctx: Context, clip_id: int) -> str:
        """Get audio pan (stereo balance) of a clip.

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            pan = resolve._app.get_clip_pan(clip_id)
            side = "center" if pan == 0 else f"{'left' if pan < 0 else 'right'} ({abs(pan):.0f}%)"
            return f"Clip {clip_id} pan: {side} ({pan:+.0f})"
        except Exception as e:
            return f"ERROR: {e}"
