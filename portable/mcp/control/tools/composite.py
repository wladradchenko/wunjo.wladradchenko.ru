"""Composite tools — high-level workflows the model calls autonomously.

These orchestrate multiple atomic operations in a single tool call,
saving the agent 10-50 roundtrips per workflow.
"""

from __future__ import annotations

import glob
import os
import re

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def build_timeline(
        ctx: Context,
        video_dir: str,
        pattern: str = "*.mp4",
        audio_path: str = "",
        transition_frames: int = 13,
        folder_name: str = "scenes",
    ) -> str:
        """Build a complete timeline: import media, sequence clips, add transitions, optionally add audio.

        Two-phase workflow:
        1. Import — adds files to media pool, validates each clip
        2. Assembly — only successfully imported clips go on the timeline

        Args:
            video_dir: Directory containing scene video clips.
            pattern: Glob pattern for video files (default "*.mp4").
            audio_path: Optional path to audio/music file. Empty string to skip.
            transition_frames: Cross-dissolve duration in frames (default 13 ~ 0.5s at 25fps).
            folder_name: Media pool folder name for imported clips (default "scenes").
        """
        import time
        import glob as globmod

        try:
            resolve = helpers.get_resolve(ctx)
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            app = resolve._app
            log: list[str] = []

            # ── Phase 1: Import to media pool ──────────────────────────
            full_pattern = os.path.join(os.path.abspath(video_dir), pattern)
            files = sorted(globmod.glob(full_pattern))
            if not files:
                return f"ERROR: No files matched '{pattern}' in {video_dir}"

            # Import one by one, track which succeeded
            imported: list[tuple[str, str]] = []  # (filename, bin_id)
            already_present: list[str] = []
            for f in files:
                ids_before = set(app.get_all_clip_ids())
                app._call("addProjectClip", f)
                time.sleep(0.3)
                ids_after = set(app.get_all_clip_ids())
                new = ids_after - ids_before
                if new:
                    bid = sorted(new, key=int)[0]
                    imported.append((os.path.basename(f), bid))
                else:
                    already_present.append(os.path.basename(f))

            if imported:
                log.append(f"**Import:** {len(imported)} new clips added")
                for name, bid in imported:
                    log.append(f"  {name} → bin_id {bid}")
            if already_present:
                log.append(f"**Already in bin:** {', '.join(already_present)}")

            # If nothing new was imported, use existing clips from root folder
            if not imported:
                root_ids = app.get_folder_clip_ids("-1")
                bin_ids = sorted(root_ids, key=int)
                if not bin_ids:
                    return "ERROR: No media clips in bin."
                log.append(f"**Using existing bin clips:** {bin_ids}")
                imported = [(os.path.basename(files[i]) if i < len(files) else f"clip-{bid}",
                            bid) for i, bid in enumerate(bin_ids)]

            if not imported:
                return "\n".join(log) + "\nERROR: No clips available."

            # ── Phase 2: Assemble timeline ─────────────────────────────
            # Find the first (lowest-position) video track
            tracks = tl.GetAllTracksInfo()
            video_track_id = None
            video_tracks = []
            for t in tracks:
                is_audio = t.get("audio")
                if is_audio in (True, "true"):
                    continue
                tid = t.get("id", t.get("track_id"))
                pos = t.get("position", 0)
                if tid is not None:
                    video_tracks.append((int(pos), int(tid), t.get("name", "")))
            if video_tracks:
                video_tracks.sort()
                video_track_id = video_tracks[0][1]
            if video_track_id is None:
                video_track_id = app.add_track("V1", False)

            log.append(f"**Target track:** id={video_track_id}")

            # Insert clips one by one, get duration from timeline clip info
            tl_clip_ids = []
            position = 0
            for name, bid in imported:
                clip_id = app.insert_clip(bid, video_track_id, position)
                if clip_id < 0:
                    log.append(f"  SKIP {name}: insert failed")
                    continue
                tl_clip_ids.append(clip_id)
                time.sleep(0.2)
                info = app.get_timeline_clip_info(clip_id)
                dur = int(info.get("duration", 0)) if info else 0
                position += dur if dur > 0 else 125
                log.append(f"  {name} → clip_id {clip_id}, {dur}f at pos {position - (dur if dur > 0 else 125)}")

            if not tl_clip_ids:
                return "\n".join(log) + "\nERROR: No clips placed on timeline."
            log.append(f"**Sequenced:** {len(tl_clip_ids)} clips on track {video_track_id}")

            # Transitions
            if transition_frames > 0 and len(tl_clip_ids) >= 2:
                t_count = 0
                for i in range(len(tl_clip_ids) - 1):
                    ok = app.add_mix(tl_clip_ids[i], tl_clip_ids[i + 1],
                                      transition_frames)
                    if ok:
                        t_count += 1
                log.append(f"**Transitions:** {t_count} dissolves ({transition_frames}f each)")

            # Audio
            if audio_path:
                audio_ids_before = set(app.get_all_clip_ids())
                app._call("addProjectClip", audio_path)
                time.sleep(0.3)
                audio_ids_after = set(app.get_all_clip_ids())
                new_audio = sorted(audio_ids_after - audio_ids_before, key=int)
                if new_audio:
                    audio_track_id = None
                    for t in tracks:
                        if t.get("audio") in (True, "true"):
                            audio_track_id = int(t.get("id", t.get("track_id", 0)))
                            break
                    if audio_track_id is None:
                        audio_track_id = app.add_track("A1", True)
                    app.insert_clip(new_audio[0], audio_track_id, 0)
                    log.append(f"**Audio:** {os.path.basename(audio_path)}")
                else:
                    log.append(f"**Audio:** FAILED to import {audio_path}")

            return "\n".join(log)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def replace_scene(ctx: Context, scene_number: int, new_file: str) -> str:
        """Replace a single scene clip on the timeline by scene number.

        Finds the clip at position (scene_number - 1) on the first video track,
        imports the new file, and swaps it in — preserving position, duration,
        and neighboring transitions.

        Args:
            scene_number: Scene number (1-based, e.g. 1-38).
            new_file: Absolute path to the replacement video file.
        """
        import time

        try:
            resolve = helpers.get_resolve(ctx)
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            app = resolve._app

            # Find first video track (lowest position, non-audio)
            tracks = tl.GetAllTracksInfo()
            video_track_id = None
            video_tracks = []
            for t in tracks:
                if t.get("audio") in (True, "true"):
                    continue
                tid = t.get("id", t.get("track_id"))
                pos = t.get("position", 0)
                if tid is not None:
                    video_tracks.append((int(pos), int(tid)))
            if video_tracks:
                video_tracks.sort()
                video_track_id = video_tracks[0][1]
            if video_track_id is None:
                return "ERROR: No video track found."

            # Get clips on the track
            clips = app.get_clips_on_track(video_track_id)
            if not clips:
                return "ERROR: Video track is empty."

            idx = scene_number - 1
            if idx < 0 or idx >= len(clips):
                return f"ERROR: Scene {scene_number} out of range (1-{len(clips)})."

            clip_dict = clips[idx]
            old_clip_id = int(clip_dict.get("id", clip_dict.get("clip_id", -1)))
            old_info = app.get_timeline_clip_info(old_clip_id)
            old_start = int(old_info.get("position", 0)) if old_info else 0
            old_dur = int(old_info.get("duration", 0)) if old_info else 0
            old_name = old_info.get("name", f"clip-{old_clip_id}") if old_info else f"clip-{old_clip_id}"

            # Import new file
            ids_before = set(app.get_all_clip_ids())
            app._call("addProjectClip", new_file)
            time.sleep(0.3)
            ids_after = set(app.get_all_clip_ids())
            new_ids = ids_after - ids_before
            if not new_ids:
                return f"ERROR: Could not import {new_file}"
            new_bin_id = sorted(new_ids, key=int)[0]

            # Delete old clip
            app.delete_timeline_clip(old_clip_id)

            # Insert new clip at same position
            new_clip_id = app.insert_clip(new_bin_id, video_track_id, old_start)
            if new_clip_id < 0:
                return f"ERROR: Deleted scene {scene_number} but failed to insert replacement."

            # Match duration
            if old_dur > 0:
                app.resize_clip(new_clip_id, old_dur, True)

            tc = helpers.format_tc(old_start, fps)
            return (
                f"Replaced scene {scene_number} ('{old_name}', clip {old_clip_id}) "
                f"with '{os.path.basename(new_file)}' (clip {new_clip_id}) at {tc}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def detect_scenes(
        ctx: Context,
        bin_clip_id: str,
        threshold: float = 0.4,
        min_duration: int = 0,
    ) -> str:
        """Detect scene cuts in a media pool clip using FFmpeg scene detection.

        Returns timestamps of detected cuts. Combine with add_guide to mark cuts on the timeline.

        Args:
            bin_clip_id: Clip ID in the media pool/bin.
            threshold: Sensitivity 0.0-1.0 (lower = more cuts detected). Default 0.4.
            min_duration: Minimum frames between detected cuts. Default 0 (no minimum).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app
            fps = helpers.get_fps(ctx)

            timestamps = app.detect_scenes(bin_clip_id, threshold, min_duration)

            if not timestamps:
                return f"No scene cuts detected in clip {bin_clip_id} (threshold={threshold})."

            lines = [
                f"**Scene detection:** {len(timestamps)} cuts found in clip {bin_clip_id} (threshold={threshold})",
                "",
                "| # | Time (s) | Timecode |",
                "|---|----------|----------|",
            ]
            for i, t in enumerate(timestamps, 1):
                tc = helpers.format_tc(int(round(t * fps)), fps)
                lines.append(f"| {i} | {t:.3f} | {tc} |")

            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"
