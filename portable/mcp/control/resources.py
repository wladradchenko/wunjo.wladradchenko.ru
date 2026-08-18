"""MCP Resources — on-demand context the agent can read."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

COOKBOOK = """\
# Wunjo Make MCP Cookbook

## Workflow: Full Timeline Assembly

1. Call `build_timeline(video_dir, pattern, audio_path, transition_frames)`
   - Imports all matching clips, sequences them, adds transitions + markers + audio
   - Returns clip table with timecodes
2. Verify with `get_timeline_summary()`
3. **Visual check**: `render_frame(frame)` on a few key frames (first, middle, last scene)
4. Fix issues with atomic tools: `insert_clip`, `move_clip`, `trim_clip`, `delete_clip`
5. Save with `save_project()`

## Workflow: Replace a Scene

1. Call `replace_scene(scene_number, new_file)`
   - Imports new clip, swaps it in at the same position/duration
2. Verify with `get_timeline_summary()`
3. **Visual check**: `render_frame()` at the replaced scene's start frame to confirm it looks correct
4. Re-add transitions if needed: `add_transition(clip_a, clip_b, 13)`

## Workflow: Add Transitions

- Single: `add_transition(clip_id_a, clip_id_b, duration)`
- All clips on a track: `add_transitions_batch(track_id, duration)`
- Remove: `remove_transition(clip_id)`

## Workflow: Markers

- Add: `add_marker(frame, label, color, note)`
- List: `get_markers()`
- Delete one: `delete_marker(frame)`
- Delete by color: `delete_markers_by_color(color)`
- Colors: Purple, Blue, Cyan, Green, Yellow, Orange, Red

## Workflow: Checkpoints (Undo)

1. Before risky operations: `checkpoint_save(label)`
2. If something goes wrong: `checkpoint_restore(label)`
3. Checkpoints are saved as copies of the .wmproj project file

## Workflow: Visual Preview & QC

Preview tools return JPEG file paths. Read the file to display the image.

**When to use preview proactively (do this without being asked):**
- After `replace_scene` → `render_frame()` at that scene's start frame
- After `build_timeline` → `render_frame()` on 2-3 key frames
- When evaluating a new clip before inserting → `render_bin_frame()` or `render_contact_sheet()`
- When comparing two clips/scenes → `render_crop()` on the same region of both frames
- When user asks about visual quality, consistency, or artifacts → `render_crop()` on the area of interest

**Tools:**
- `render_frame(frame)` — composited timeline thumbnail (all tracks + effects + transitions)
- `render_bin_frame(bin_id, frame_position)` — single frame from a media pool clip ("first"/"middle"/"last"/int)
- `render_contact_sheet(bin_id, num_frames=8)` — grid overview of a clip's motion over time
- `render_crop(frame, region, crop_size=480)` — 1:1 pixel crop from timeline at native resolution
  - Regions: "center", "top-third", "bottom-third", "left-third", "right-third"
  - Or custom: `custom_x, custom_y, custom_w, custom_h` in pixels

**Typical QC checks:**
- Character identity consistency: `render_crop(region="center")` across scenes — compare face, outfit
- Transition quality: `render_frame()` at transition midpoint (scene_start - transition_frames/2)
- Edge matching: `render_crop()` on last frame of scene N and first frame of scene N+1

## ID System

- **bin_id** (str): identifies a clip in the media pool (bin)
- **clip_id** (int): identifies a clip instance on the timeline
- A single media pool clip can appear multiple times on the timeline with different clip_ids
- Use `get_media_pool()` to see bin_ids, `get_timeline_summary()` to see clip_ids

## Units

- **Input**: always frame numbers (integers)
- **Output**: timecodes (HH:MM:SS:FF) + frame numbers where useful
- Project FPS: typically 25fps (use `get_project_info()` to confirm)
- 13 frames @ 25fps = 0.52s (standard transition duration)
- 125 frames @ 25fps = 5.00s (standard scene duration)

## Track Indexing

- Track IDs are integers assigned by Wunjo Make
- Use `get_track_list()` to discover track IDs
- `GetItemListInTrack` uses 1-based indexing (like DaVinci Resolve)

## Error Handling

- All tools return "ERROR: ..." on failure (not exceptions)
- Check for "ERROR" prefix in tool output before proceeding
- Common errors: clip not found, track not found, file not importable

## Token Budget

- `get_timeline_summary` for 38 clips: ~750 tokens
- `get_media_pool` for 80 clips: ~1200 tokens
- `build_timeline` full assembly: ~500 tokens output
- Prefer composite tools (1 call) over atomic tools (N calls)
"""


def register(mcp: FastMCP):

    @mcp.resource("wunjo://cookbook")
    def cookbook() -> str:
        """Full workflow cookbook for Wunjo Make MCP — recipes, ID system, units, error handling."""
        return COOKBOOK
