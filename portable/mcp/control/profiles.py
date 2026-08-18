"""How much of the editor a given assistant is handed.

The full server is written for a frontier model in a terminal: 177 tools and a
page of instructions, which is the right trade when the model can hold it. The
assistant that ships inside Wunjo is a few billion parameters running on the
user's own graphics card, and for it the same surface is the problem — the tool
schemas alone crowd out the conversation, and the more near-identical tools it
is offered the more often it picks the wrong one.

So the server has profiles. ``full`` is what an outside agent gets. ``core``
keeps the tools an edit actually passes through, and drops three kinds of thing:
what is dangerous (opening or replacing the user's project), what is fiddly
(keyframe-by-keyframe surgery), and what only pays off with a long context
(GUI screenshots, the effect catalogue). Anything left out is still reachable —
by the outside agent, or by the user.

Set ``WUNJO_TOOL_PROFILE=core`` in the server's environment to select it.
"""
from __future__ import annotations

import os
import sys

#: Tools the built-in assistant is offered. Order is grouping, not priority.
CORE_TOOLS = {
    # what is there
    "get_project_info",
    "get_project_duration",
    "get_timeline_summary",
    "get_media_pool",
    "get_track_list",
    "get_clip_info",
    "get_bin_clip_properties",
    "get_clip_metadata",
    # looking at what somebody dropped in a folder, before importing any of it:
    # the names say nothing, so the footage is described once by a vision model
    # and the descriptions are what the assistant reads
    "list_media_folder",
    "describe_media",
    "transcribe_media",
    "find_moments",
    "describe_project_media",
    "find_media",
    # bringing material in — a whole folder at a time, which is how an edit
    # actually starts
    "import_media",
    "import_media_glob",
    "create_bin_folder",
    "rename_bin_clip",
    # assembling
    "build_timeline",
    "replace_scene",
    "insert_clip",
    "append_clips",
    "move_clip",
    "trim_clip",
    "split_clip",
    "delete_clip",
    "ripple_delete",
    "add_track",
    "set_track_name",
    "detect_scenes",
    "group_clips",
    "ungroup_clips",
    "get_sequences",
    "create_sequence",
    "set_active_sequence",
    # joins
    "add_transition",
    "remove_transition",
    "add_transitions_batch",
    "get_available_transitions",
    # look
    "add_effect",
    "remove_effect",
    "get_clip_effects",
    "get_available_effects",
    "set_effect_param",
    "set_clip_opacity",
    "set_clip_speed",
    "set_clip_transform",
    "add_title",
    # sound
    "set_clip_volume",
    "set_audio_fade",
    "set_track_mute",
    "split_audio",
    "get_audio_levels",
    # notes and range
    "add_marker",
    "get_markers",
    "set_zone",
    "extract_zone",
    "seek_to",
    "get_position",
    # words
    "speech_recognition",
    "add_subtitle",
    "get_subtitles",
    "export_subtitles",
    # faces
    "get_faces_at_frame",
    "apply_face_effect",
    # seeing the result — and the material, before deciding what to do with it
    "render_frame",
    "render_bin_frame",
    "render_contact_sheet",
    "render_video",
    # the user's own instructions
    "get_selected_skills",
    "get_selected_loop",
    # the other plugins
    "list_plugins",
    "install_plugin",
    "plugin_status",
    "run_plugin",
    "list_plugin_sets",
    "generate_effect",
    "plugin_job_status",
    # taking it back
    "undo",
    "redo",
    "undo_status",
    "save_project",
    # telling the user about long work (the reply itself is not the model's to
    # post — the editor puts that in the chat when the turn ends)
    "chat_tool_start",
    "chat_tool_progress",
    "chat_tool_end",
}

CORE_INSTRUCTIONS = """\
You drive Wunjo Make, a video editor, for the person talking to you in its chat.

HOW TO WORK
  1. Call get_timeline_summary before changing anything, so you act on what is
     really there rather than what you remember.
  2. Make the change with one tool at a time and check the result.
  3. For anything slow (a plugin, speech recognition, a render) call
     chat_tool_start, then chat_tool_end when it finishes, so the user sees it.

KNOWING WHAT THE FOOTAGE IS
  File names tell you nothing, so look before you choose.
  - already in the project: describe_project_media says what the clips in the
    bin actually show. Start here — the user may have put them there for you.
  - still in a folder: list_media_folder says what is there and what has been
    looked at, describe_media says what one file shows, find_media searches
    those descriptions. Describe first, choose, then import only what you chose.
  Looking is remembered, so the same file is never examined twice.

CUTTING WHAT SOMEBODY SAYS
  For anything where people talk, do not guess the cuts from thumbnails.
  transcribe_media reads a file directly and returns every line with its
  timing — use it on each source before choosing anything. Where a file has
  little or no speech, find_moments measures the picture instead and hands
  back a shortlist of times worth looking at; describe_media(at=...) then
  shows you one of them. Measuring says where to look, looking says what is
  there, and the choice between them is yours. (speech_recognition
  is the other one: it puts subtitles on the timeline, which is what you want
  for burnt-in captions, not for deciding cuts.) Those timings are where sentences begin and end — cut there,
  keep the lines that carry the story, and drop the rest. A cut that lands mid
  word is what makes an edit look automatic.

USING A PLUGIN
  list_plugins gives each plugin's own manifest — read it, the user may have
  written the plugin themselves. A plugin that declares effects works through
  them (apply the effect, do not run it); one with a "sets" block needs
  something recorded first via run_plugin(action="analyse", source=..., kind=K)
  where K is the key under "sets" — "face" to register a face to swap in,
  "expression" to record how a face moves. list_plugin_sets then gives the
  FILE of each recorded set, and that path — not the name — is what goes
  into the effect's set parameter with set_effect_param.
  If plugin_status says it is not ready, call install_plugin and wait — the
  environment and the weights are fetched on first use, not shipped.
  Applying such an effect produces NOTHING by itself: it only records what to
  do. generate_effect is what renders it, and until that has finished and the
  new clip is in get_media_pool, nothing has been made. Never say a face was
  swapped, a voice cloned or a trailer rendered before you have seen it there.
  run_plugin waits for the plugin and answers with what it said, refusals
  included ("choose an audio preset first"); generate_effect hands back a job
  id to follow with plugin_job_status. Read those answers — they are how you
  learn that a step did not happen, and repeating it unchanged will fail the
  same way.

WHAT YOU CAN REACH
  Only these tools. You have no shell, no file editing and no internet. Media
  comes in with import_media by absolute path; everything you produce is written
  by the editor into the project. If a request needs something you cannot do,
  say so plainly instead of pretending.

FACTS
  Positions and lengths are in frames on input, timecodes on output.
  A clip must be in the media pool (get_media_pool) before it can go on a track.
  undo takes back the last change if you got it wrong.

Answer in the user's language. Keep replies short: they are read in a side panel.
"""


def selected() -> str:
    """The profile this process should serve."""
    return os.environ.get("WUNJO_TOOL_PROFILE", "full").strip().lower()


def apply(mcp) -> None:
    """Drop everything outside the profile from an already-built server.

    Filtering after registration rather than skipping whole modules keeps one
    list to read and lets the choice be made tool by tool — the modules do not
    divide along the lines that matter here.
    """
    if selected() != "core":
        return
    registry = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
    if not isinstance(registry, dict):
        # A FastMCP that keeps its tools somewhere else: better to serve them
        # all than to fail to start, and the mismatch is worth saying out loud.
        print("WUNJO_TOOL_PROFILE=core could not be applied to this MCP version", file=sys.stderr)
        return
    for name in [name for name in registry if name not in CORE_TOOLS]:
        del registry[name]
