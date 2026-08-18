"""Timeline tools: summary, tracks, insert, append, move, delete, trim, add_track."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_timeline_summary(ctx: Context, track_type: str = "all") -> str:
        """Get timeline contents as markdown tables — one per track.

        Args:
            track_type: Filter by "video", "audio", or "all".

        Returns markdown with clip tables per track (~20 tokens/row).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app
            fps = helpers.get_fps(ctx)
            tracks = app.get_all_tracks_info()
            if not tracks:
                return "Timeline is empty."

            sections: list[str] = []
            for t in tracks:
                is_audio = t.get("audio")
                ttype = "audio" if is_audio in (True, "true") else "video"
                if track_type != "all" and ttype != track_type:
                    continue

                tid = int(t.get("id", t.get("track_id", 0)))
                tname = t.get("name", "")
                label = ttype[0].upper()  # V or A

                # Get clips on this track via D-Bus directly
                clip_dicts = app.get_clips_on_track(tid)
                if not clip_dicts:
                    sections.append(f"## {label}{tid}{f' — {tname}' if tname else ''} — 0 clips")
                    continue

                # Use clip dicts directly (C++ now returns name, binId, url)
                clips_info = []
                for cd in clip_dicts:
                    cd["clip_id"] = int(cd.get("id", cd.get("clip_id", -1)))
                    clips_info.append(cd)

                total_frames = sum(int(c.get("duration", 0)) for c in clips_info)
                total_tc = helpers.format_tc(total_frames, fps)
                header = f"## {label}{tid}{f' — {tname}' if tname else ''} — {len(clips_info)} clips, {total_frames}f ({total_tc})"

                # Build table with transition info
                table_header = "| # | clip_id | start | end | dur | filename | transition |"
                table_sep = "|---|---------|-------|-----|-----|----------|------------|"
                rows = [header, "", table_header, table_sep]

                for i, c in enumerate(clips_info):
                    cid = c["clip_id"]
                    s = int(c.get("position", 0))
                    d = int(c.get("duration", 0))
                    e = s + d
                    name = c.get("name", f"clip-{cid}")

                    # Detect overlap with previous clip (= transition)
                    trans = "--"
                    if i > 0:
                        prev_s = int(clips_info[i - 1].get("position", 0))
                        prev_d = int(clips_info[i - 1].get("duration", 0))
                        prev_end = prev_s + prev_d
                        overlap = prev_end - s
                        if overlap > 0:
                            trans = f"dissolve {overlap}f"

                    rows.append(
                        f"| {i} | {cid} | {helpers.format_tc(s, fps)} | "
                        f"{helpers.format_tc(e, fps)} | {d} | {name} | {trans} |"
                    )

                sections.append("\n".join(rows))

            return "\n\n".join(sections) if sections else "No matching tracks."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_track_list(ctx: Context) -> str:
        """List all tracks in the timeline.

        Returns a markdown table: track_id | type | name | clips | total_frames
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app
            tracks = app.get_all_tracks_info()
            if not tracks:
                return "No tracks."

            # Enrich with clip counts (no per-clip D-Bus calls needed)
            enriched = []
            for t in tracks:
                tid = int(t.get("id", t.get("track_id", 0)))
                is_audio = t.get("audio")
                tname = t.get("name", "")
                clip_dicts = app.get_clips_on_track(tid)
                nclips = len(clip_dicts)
                total = sum(int(cd.get("duration", 0)) for cd in clip_dicts)
                enriched.append({
                    "id": tid, "audio": is_audio,
                    "name": tname, "clips": nclips, "total_frames": total,
                })
            return helpers.tracks_table(enriched)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_info(ctx: Context, clip_id: int) -> str:
        """Get detailed info about a specific timeline clip.

        Args:
            clip_id: The clip's integer ID on the timeline.
        """
        try:
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            # Use D-Bus to get clip info directly
            resolve = helpers.get_resolve(ctx)
            info = resolve._app.get_timeline_clip_info(clip_id)
            if not info:
                return f"ERROR: Clip {clip_id} not found."

            s = int(info.get("position", 0))
            d = int(info.get("duration", 0))
            e = s + d
            name = info.get("name", "")
            track = info.get("trackId", "")
            bin_id = info.get("binId", "")
            in_pt = int(info.get("in", 0))
            out_pt = int(info.get("out", 0))
            max_dur = int(info.get("maxDuration", 0))

            url = info.get("url", "")

            lines = [
                f"**Name:** {name}",
                f"**clip_id:** {clip_id}",
                f"**Track:** {track}",
                f"**Start:** {helpers.format_tc(s, fps)} (frame {s})",
                f"**End:** {helpers.format_tc(e, fps)} (frame {e})",
                f"**Duration:** {d} frames",
                f"**Source in:** {helpers.format_tc(in_pt, fps)} (frame {in_pt})",
                f"**Source out:** {helpers.format_tc(out_pt, fps)} (frame {out_pt})",
                f"**Max duration:** {helpers.format_tc(max_dur, fps)} ({max_dur} frames)",
                f"**Source range:** Using frames {in_pt}–{out_pt} of {max_dur} total",
                f"**bin_id:** {bin_id}",
                f"**Source:** {url}",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def insert_clip(ctx: Context, bin_id: str, track_id: int, position: int) -> str:
        """Insert a single clip at position. For assembling a full timeline from scene clips, use the build_timeline tool instead.

        Args:
            bin_id: Media pool clip ID (string).
            track_id: Target track ID.
            position: Position in frames on the timeline.
        """
        try:
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            item = tl.InsertClip(bin_id, track_id, position)
            if item is None:
                return f"ERROR: Could not insert bin_id={bin_id} at track {track_id}, frame {position}"
            tc = helpers.format_tc(position, fps)
            return f"Inserted clip {item.clip_id} at {tc} on track {track_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def append_clips(ctx: Context, bin_ids: list[str], track_id: int, start_position: int = 0) -> str:
        """Append multiple clips sequentially on a track. For full assembly with transitions and audio, use build_timeline instead.

        Args:
            bin_ids: List of media pool clip IDs to append in order.
            track_id: Target track ID.
            start_position: Starting frame position (default 0).

        Returns a table of inserted clips.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app
            fps = helpers.get_fps(ctx)
            clip_ids = app.insert_clips_sequentially(bin_ids, track_id, start_position)
            if not clip_ids:
                return "ERROR: No clips inserted. Check that bin_ids are valid."

            # Report per-clip results
            rows = ["| # | bin_id | clip_id | status |",
                    "|---|--------|---------|--------|"]
            ok_count = 0
            for i, (bid, cid) in enumerate(zip(bin_ids, clip_ids)):
                if cid == -1:
                    rows.append(f"| {i} | {bid} | -- | FAILED |")
                else:
                    rows.append(f"| {i} | {bid} | {cid} | ok |")
                    ok_count += 1
            # bin_ids longer than clip_ids means those were never attempted
            for i in range(len(clip_ids), len(bin_ids)):
                rows.append(f"| {i} | {bin_ids[i]} | -- | FAILED |")

            table = "\n".join(rows)
            return f"Appended {ok_count}/{len(bin_ids)} clip(s) on track {track_id}:\n\n{table}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def move_clip(ctx: Context, clip_id: int, track_id: int, position: int) -> str:
        """Move a timeline clip to a new track/position.

        Args:
            clip_id: The clip's timeline ID.
            track_id: Target track ID.
            position: New position in frames.
        """
        try:
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            # Get item by iterating tracks
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.move_clip(clip_id, track_id, position)
            if not ok:
                return f"ERROR: Could not move clip {clip_id}"
            tc = helpers.format_tc(position, fps)
            return f"Moved clip {clip_id} to track {track_id} at {tc}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_clip(ctx: Context, clip_id: int) -> str:
        """Delete a clip from the timeline.

        Args:
            clip_id: The clip's timeline ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.delete_timeline_clip(clip_id)
            if not ok:
                return f"ERROR: Could not delete clip {clip_id}"
            return f"Deleted clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_track(ctx: Context, name: str = "", audio: bool = False) -> str:
        """Add a new track to the timeline.

        Args:
            name: Track name (optional).
            audio: True for audio track, False for video track.
        """
        try:
            tl = helpers.get_timeline(ctx)
            tid = tl.AddTrack(name, audio)
            kind = "audio" if audio else "video"
            label = f"'{name}' " if name else ""
            return f"Added {kind} track {label}(id: {tid})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def insert_space(ctx: Context, track_id: int, position: int,
                     duration: int, all_tracks: bool = False) -> str:
        """Insert blank space at a position, pushing all clips to the right (ripple insert).

        Args:
            track_id: Track ID where to insert space.
            position: Frame position where space begins.
            duration: Duration of space to insert in frames.
            all_tracks: If True, insert on all tracks. If False, only on track_id.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.insert_space(track_id, position, duration, all_tracks)
            if not ok:
                return f"ERROR: Could not insert space at frame {position}"
            fps = helpers.get_fps(ctx)
            return f"Inserted {duration} frames of space at {helpers.format_tc(position, fps)}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def remove_space(ctx: Context, track_id: int, position: int,
                     all_tracks: bool = False) -> str:
        """Remove blank space at a position, pulling clips to the left (ripple delete).

        Args:
            track_id: Track ID where to remove space.
            position: Frame position within the blank to remove.
            all_tracks: If True, remove on all tracks. If False, only on track_id.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.remove_space(track_id, position, all_tracks)
            if not ok:
                return f"ERROR: Could not remove space at frame {position}"
            return f"Removed space at frame {position}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_track(ctx: Context, track_id: int) -> str:
        """Delete a track from the timeline.

        Args:
            track_id: The track ID to delete.
        """
        try:
            tl = helpers.get_timeline(ctx)
            ok = tl.DeleteTrack(track_id)
            if not ok:
                return f"ERROR: Could not delete track {track_id}"
            return f"Deleted track {track_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def split_clip(ctx: Context, clip_id: int, frame: int) -> str:
        """Split a clip into two parts at the given frame position.

        Args:
            clip_id: Timeline clip ID.
            frame: Absolute timeline frame position where to cut.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            ok = resolve._app.cut_clip(clip_id, frame)
            if not ok:
                return f"ERROR: Could not split clip {clip_id} at frame {frame}"
            tc = helpers.format_tc(frame, fps)
            return f"Split clip {clip_id} at {tc} (frame {frame})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def slip_clip(ctx: Context, clip_id: int, offset: int) -> str:
        """Slip a clip's source in/out points by offset frames.

        Moves the source media window without changing the clip's position
        or duration on the timeline. Positive offset = later in source,
        negative = earlier.

        Args:
            clip_id: Timeline clip ID.
            offset: Number of frames to slip (positive or negative).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.slip_clip(clip_id, offset)
            if not ok:
                return f"ERROR: Could not slip clip {clip_id} by {offset} frames"
            # Show updated in/out
            info = resolve._app.get_timeline_clip_info(clip_id)
            fps = helpers.get_fps(ctx)
            src_in = info.get("in", "?")
            src_out = info.get("out", "?")
            tc_in = helpers.format_tc(int(src_in), fps) if src_in != "?" else "?"
            tc_out = helpers.format_tc(int(src_out), fps) if src_out != "?" else "?"
            return f"Slipped clip {clip_id} by {offset} frames. New source range: {tc_in} – {tc_out}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def copy_clips(ctx: Context) -> str:
        """Copy current timeline selection to clipboard.

        Select clips first using set_selection or select_items_in_range,
        then call this to copy them.

        Returns the main copied clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            clip_id = resolve._app.copy_clips()
            if clip_id < 0:
                return "ERROR: Nothing selected to copy"
            return f"Copied selection (main clip: {clip_id})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def cut_clips(ctx: Context) -> str:
        """Cut current timeline selection (copy to clipboard + delete from timeline).

        Select clips first using set_selection or select_items_in_range.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.cut_clips()
            if not ok:
                return "ERROR: Nothing selected to cut"
            return "Cut selection to clipboard"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def paste_clips(ctx: Context, position: int = -1, track_id: int = -1) -> str:
        """Paste clipboard contents onto the timeline.

        Args:
            position: Frame position to paste at (-1 = playhead position).
            track_id: Target track ID (-1 = active track).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            ok = resolve._app.paste_clips(position, track_id)
            if not ok:
                return "ERROR: Paste failed (clipboard empty or invalid target)"
            pos_str = helpers.format_tc(position, fps) if position >= 0 else "playhead"
            trk_str = str(track_id) if track_id >= 0 else "active"
            return f"Pasted at {pos_str} on track {trk_str}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def trim_clip(ctx: Context, clip_id: int, new_duration: int, from_right: bool = True) -> str:
        """Trim a timeline clip to a new duration.

        Args:
            clip_id: The clip's timeline ID.
            new_duration: Desired duration in frames.
            from_right: If True, trims from the right edge. If False, from the left.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            info = resolve._app.get_timeline_clip_info(clip_id)
            old_dur = info.get("duration", 0) if info else 0
            actual = resolve._app.resize_clip(clip_id, new_duration, from_right)
            return f"Trimmed {clip_id}: {old_dur} -> {actual} frames"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_track_name(ctx: Context, track_id: int) -> str:
        """Get the name/label of a track.

        Args:
            track_id: Track ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            name = resolve._app.get_track_name(track_id)
            return f"Track {track_id} name: '{name}'" if name else f"Track {track_id} has no name set"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_track_name(ctx: Context, track_id: int, name: str) -> str:
        """Set the name/label of a track.

        Args:
            track_id: Track ID.
            name: New track name.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_track_name(track_id, name)
            if not ok:
                return f"ERROR: Could not set name for track {track_id}"
            return f"Track {track_id} renamed to '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def ripple_delete(ctx: Context, clip_id: int) -> str:
        """Delete a clip and close the gap (ripple delete).

        Args:
            clip_id: Timeline clip ID to delete.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.ripple_delete(clip_id)
            if not ok:
                return f"ERROR: Could not ripple delete clip {clip_id}"
            return f"Ripple deleted clip {clip_id} (gap closed)"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def ripple_trim(ctx: Context, clip_id: int, delta: int, from_right: bool = True) -> str:
        """Trim a clip and shift all following clips (ripple trim).

        Args:
            clip_id: Timeline clip ID.
            delta: Frames to add (positive) or remove (negative).
            from_right: If True, trim from right edge. If False, from left.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.ripple_trim(clip_id, delta, from_right)
            if not ok:
                return f"ERROR: Could not ripple trim clip {clip_id}"
            action = "extended" if delta > 0 else "shortened"
            return f"Clip {clip_id} {action} by {abs(delta)} frames (following clips shifted)"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def roll_edit(ctx: Context, clip_id: int, delta: int) -> str:
        """Roll edit: move cut point between two adjacent clips.

        Args:
            clip_id: Timeline clip ID (the right clip of the edit).
            delta: Frames to move the edit point (positive = right, negative = left).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.roll_edit(clip_id, delta)
            if not ok:
                return f"ERROR: Could not roll edit on clip {clip_id}"
            direction = "right" if delta > 0 else "left"
            return f"Roll edit on clip {clip_id}: cut point moved {abs(delta)} frames {direction}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def slide_edit(ctx: Context, clip_id: int, delta: int) -> str:
        """Slide edit: move clip while adjusting neighbors.

        Args:
            clip_id: Timeline clip ID.
            delta: Frames to slide (positive = right, negative = left).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.slide_edit(clip_id, delta)
            if not ok:
                return f"ERROR: Could not slide clip {clip_id}"
            direction = "right" if delta > 0 else "left"
            return f"Slid clip {clip_id} {abs(delta)} frames {direction}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_track_solo(ctx: Context, track_id: int, solo: bool) -> str:
        """Solo or unsolo a track (mutes all other tracks).

        Args:
            track_id: Track ID.
            solo: True to solo, False to unsolo.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_track_solo(track_id, solo)
            if not ok:
                return f"ERROR: Could not set solo for track {track_id}"
            state = "solo'd" if solo else "unsolo'd"
            return f"Track {track_id} {state}"
        except Exception as e:
            return f"ERROR: {e}"
