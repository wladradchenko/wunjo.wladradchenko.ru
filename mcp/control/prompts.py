"""MCP Prompts — user-facing slash commands bundled with the server.

These appear as /mcp__wunjo-make__<name> in Claude Code.
Note: as of Jan 2026, the model cannot see or invoke Prompts autonomously
(bug #11054). These are user-facing only. The model uses composite tools instead.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


def register(mcp: FastMCP):

    @mcp.prompt()
    def build_timeline(
        video_dir: str,
        audio_path: str = "",
        variant: str = "A",
        transition_frames: str = "13",
    ) -> str:
        """Build complete timeline from scene clips with import, sequencing, transitions, markers and audio sync.

        Args:
            video_dir: Directory with scene video clips.
            audio_path: Path to audio/music file (optional).
            variant: Clip variant A or B (default A).
            transition_frames: Cross-dissolve length in frames (default 13).
        """
        parts = [
            "You are assembling a music video timeline in Wunjo Make.",
            "",
            "Use the `build_timeline` tool with these parameters:",
            f'- video_dir: "{video_dir}"',
            f'- pattern: "*{variant}*.mp4"' if variant else '- pattern: "*.mp4"',
            f'- audio_path: "{audio_path}"' if audio_path else "- No audio file specified",
            f"- transition_frames: {transition_frames}",
            "",
            "After the tool completes, verify the result by calling `get_timeline_summary`.",
            "Check that all scenes are present, transitions are applied, and audio is placed.",
            "",
            "If any scenes are missing, use `import_media` + `insert_clip` to add them manually.",
        ]
        return "\n".join(parts)

    @mcp.prompt()
    def replace_scene(scene_number: str, new_file: str) -> str:
        """Replace a single scene clip preserving position, duration and neighboring transitions.

        Args:
            scene_number: Scene number (1-38).
            new_file: Path to replacement video file.
        """
        return (
            f"Replace scene {scene_number} on the timeline with the new file:\n"
            f"  {new_file}\n\n"
            f"Use the `replace_scene` tool with scene_number={scene_number} and new_file=\"{new_file}\".\n\n"
            "After replacement, call `get_timeline_summary` to verify the clip is in place\n"
            "with correct duration and transitions preserved."
        )

    @mcp.prompt()
    def timeline_summary() -> str:
        """Get text table of all clips: scene, start TC, end TC, duration, filename, transition type."""
        return (
            "Call the `get_timeline_summary` tool to see the current state of the timeline.\n"
            "Review the output for:\n"
            "- All expected scenes present\n"
            "- Correct ordering\n"
            "- Transitions between adjacent clips\n"
            "- Total duration matches expected length"
        )

    @mcp.prompt()
    def render_final(output_path: str, format: str = "mp4") -> str:
        """Render timeline to final video file.

        Args:
            output_path: Output file path.
            format: Video format (mp4, webm, mov).
        """
        return (
            f"Render the timeline to: {output_path}\n\n"
            f"Use the `render_video` tool with output_path=\"{output_path}\".\n\n"
            "Before rendering, verify the timeline with `get_timeline_summary`.\n"
            "After starting the render, save the project with `save_project`."
        )
