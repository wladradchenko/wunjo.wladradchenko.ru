"""MCP server for Wunjo Make — stdio transport, FastMCP."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the vendored api (sibling package in mcp/) is importable
_api_dir = str(Path(__file__).resolve().parents[1])
if _api_dir not in sys.path:
    sys.path.insert(0, _api_dir)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Instructions — injected into agent context at MCP handshake (~200 tokens)
# ---------------------------------------------------------------------------
INSTRUCTIONS = """\
Wunjo Make MCP gives you full NLE control over a running Wunjo Make instance via D-Bus.

MENTAL MODEL — identical to DaVinci Resolve API:
  Resolve → ProjectManager → Project → MediaPool → Timeline → TimelineItem

CUTTING SPEECH — start here when people talk:
  speech_recognition transcribes the timeline with Whisper onto the subtitle
  track; get_subtitles then returns every line with its in and out. Those are
  the sentence boundaries: choose the lines that carry the story and cut on
  their timings instead of guessing from thumbnails.

COMPOSITE TOOLS (use these first):
  build_timeline    — full assembly from scene clips (import + sequence + transitions + audio + markers)
  replace_scene     — swap one scene clip by number, keep position and transitions
  detect_scenes     — FFmpeg scene detection on a bin clip, returns cut timestamps
  get_timeline_summary — text table of all clips on timeline (~20 tokens/row)
  add_transitions_batch — batch cross-dissolves between all clips on a track
  render_video      — export timeline to video file

PREVIEW TOOLS (visual inspection — returns JPEG file paths):
  render_frame        — composited timeline thumbnail at a given frame
  render_bin_frame    — single frame from a media pool clip
  render_contact_sheet — grid of evenly-spaced frames from a bin clip (requires Pillow)
  render_crop         — 1:1 pixel crop of a timeline frame for QC (requires Pillow)
  screenshot_window   — capture the Wunjo Make GUI window as JPEG + JSON panel map
  screenshot_panel    — crop a named panel from the GUI (e.g. "timeline", "effect_stack")

ATOMIC TOOLS (use when composite tools don't cover your case):
  import_media, import_media_glob, get_media_pool, create_bin_folder,
  insert_clip, append_clips, move_clip, trim_clip, split_clip, slip_clip, delete_clip, add_track,
  get_track_list, get_clip_info, get_project_info, new_project, open_project, save_project, load_project,
  add_transition, remove_transition,
  add_effect, remove_effect, get_clip_effects, set_clip_opacity,
  set_effect_param, get_effect_param, set_effect_expression, clear_effect_expression,
  get_effect_keyframes, add_effect_keyframe, remove_effect_keyframe, update_effect_keyframe,
  set_clip_speed,
  set_clip_volume, get_clip_volume, set_audio_fade, set_track_mute, get_track_mute, get_audio_levels,
  add_marker, delete_marker, delete_markers_by_color, get_markers,
  add_clip_marker, get_clip_markers, delete_clip_marker, delete_clip_markers_by_color,
  replace_clip, relink_clip,
  checkpoint_save, checkpoint_restore, undo, redo, undo_status,
  get_zone, set_zone, set_zone_in, set_zone_out, extract_zone,
  get_sequences, get_active_sequence, set_active_sequence,
  add_title,
  get_compositions, get_composition_info, move_composition, resize_composition,
  delete_composition, get_composition_types,
  get_clip_proxy_status, set_clip_proxy, delete_clip_proxy, rebuild_clip_proxy,
  group_clips, ungroup_clips, get_group_info, remove_from_group,
  get_subtitles, add_subtitle, edit_subtitle, delete_subtitle, export_subtitles,
  get_subtitle_styles, set_subtitle_style, delete_subtitle_style, set_subtitle_style_name,
  get_selection, set_selection, add_to_selection, clear_selection, select_all, select_current_track, select_items_in_range,
  seek_to, get_position, play, pause, get_playback_speed,

GUIDANCE (user-authored, per project):
  get_selected_skills — "how to work" notes chosen for this project
  get_selected_loop   — pipeline scenario (source material → finished video)
  AT SESSION START call both and follow them; empty = work unguided.
  Full management: list/get/save/delete/select for skills and loops.

AI PLUGINS (video/audio/face/generator — some bundled out of the box):
  list_plugins — ids + option schemas. ALWAYS plugin_status(id) BEFORE run_plugin:
  if not ready, tell the user IN THEIR LANGUAGE what's missing (API key on the
  plugin's settings tab / install deps via the banner / download a model).
  run_plugin(id, {...}) launches headless; produced media lands in the project bin
  (poll get_media_pool / render_bin_frame). Generator plugins take a prompt/params
  (no clip); you write the prompts. speech_recognition runs Whisper headless.

NARRATE INTO THE CHAT (do this on every task, in the USER'S LANGUAGE):
  The user does not see your reasoning — mirror it into the app's Chat dock so
  they know what is happening. Tools: chat_user (echo their request),
  chat_thinking(on/off), chat_assistant / chat_assistant_stream (your reply),
  chat_tool_start/progress/end (a live card for any long op — plugin, whisper,
  render: "running… / waiting…"). Always write these messages in the user's
  language. This is feedback only; it never blocks editing.

RULES:
  - Wunjo Make must be running (D-Bus runtime, not file-based)
  - State is text (get_timeline_summary) AND visual (preview tools)
  - Frames on input, timecodes on output
  - Prefer composite tools — they handle full workflows in one call
  - After replace_scene or build_timeline: ALWAYS render_frame to visually verify the result
  - When evaluating clips: use render_bin_frame or render_contact_sheet before deciding

WHEN IN DOUBT: call get_timeline_summary to see current state.
For detailed recipes and preview workflow: read the wunjo://cookbook resource.
"""


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Create the Resolve singleton once at startup."""
    from api import Resolve

    resolve = Resolve()
    yield {"resolve": resolve}


from control import profiles

# The instructions are part of the budget: the full page is written for a model
# that can afford it, the built-in assistant gets the short version.
_instructions = profiles.CORE_INSTRUCTIONS if profiles.selected() == "core" else INSTRUCTIONS

mcp = FastMCP("wunjo-make", instructions=_instructions, lifespan=lifespan)

# ---------------------------------------------------------------------------
# Register tool modules (atomic + composite)
# ---------------------------------------------------------------------------
from control import helpers
from control.tools import (
    library,
    guidance,
    plugins,
    face,
    chat,
    project,
    media,
    timeline,
    transitions,
    effects,
    markers,
    replace,
    checkpoints,
    composite,
    speed,
    audio,
    titles,
    preview,
    subtitles,
    keyframes,
    compositions,
    zones,
    sequences,
    proxy,
    groups,
    selection,
    playback,
    navigation,
)

for mod in [library, guidance, plugins, face, chat, project, media, timeline, transitions, effects, markers, replace, checkpoints, composite, speed, audio, titles, preview, subtitles, keyframes, compositions, zones, sequences, proxy, groups, selection, playback, navigation]:
    mod.register(mcp, helpers)

# ---------------------------------------------------------------------------
# Register prompts (user-facing slash commands)
# ---------------------------------------------------------------------------
from control import prompts

prompts.register(mcp)

# ---------------------------------------------------------------------------
# Register resources (on-demand context)
# ---------------------------------------------------------------------------
from control import resources

resources.register(mcp)

# Last, so it prunes the finished registry rather than racing the modules.
profiles.apply(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
