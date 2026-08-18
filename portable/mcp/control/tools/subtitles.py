"""Subtitle tools: get, add, edit, delete, export subtitles + style management."""

from __future__ import annotations

from mcp.server.fastmcp import Context


# snake_case → camelCase mapping for style properties
_SNAKE_TO_CAMEL = {
    "font_name": "fontName",
    "font_size": "fontSize",
    "primary_colour": "primaryColour",
    "secondary_colour": "secondaryColour",
    "outline_colour": "outlineColour",
    "back_colour": "backColour",
    "strike_out": "strikeOut",
    "scale_x": "scaleX",
    "scale_y": "scaleY",
    "border_style": "borderStyle",
    "margin_l": "marginL",
    "margin_r": "marginR",
    "margin_v": "marginV",
}


def _to_camel(params: dict) -> dict:
    """Convert snake_case keys to camelCase for D-Bus transport."""
    out = {}
    for k, v in params.items():
        out[_SNAKE_TO_CAMEL.get(k, k)] = v
    return out


def register(mcp, helpers):

    @mcp.tool()
    def get_subtitles(ctx: Context) -> str:
        """List all subtitles in the timeline.

        Returns a markdown table: id | layer | start (TC) | end (TC) | style | text
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            subs = resolve._app.get_subtitles()
            if not subs:
                return "No subtitles found (subtitle track may be empty or not created)."
            header = "| id | layer | start | end | style | text |"
            sep = "|----|-------|-------|-----|-------|------|"
            lines = [header, sep]
            for s in subs:
                sid = s.get("id", "")
                layer = s.get("layer", "0")
                start_f = int(s.get("startFrame", 0))
                end_f = int(s.get("endFrame", 0))
                style = s.get("styleName", "Default")
                text = s.get("text", "").replace("\n", " ").replace("|", "\\|")
                start_tc = helpers.format_tc(start_f, fps)
                end_tc = helpers.format_tc(end_f, fps)
                lines.append(f"| {sid} | {layer} | {start_tc} | {end_tc} | {style} | {text} |")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_subtitle(ctx: Context, start_frame: int, end_frame: int,
                     text: str, layer: int = 0) -> str:
        """Add a subtitle to the timeline.

        Args:
            start_frame: Start position in frames.
            end_frame: End position in frames.
            text: Subtitle text.
            layer: Subtitle layer (default 0).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            sid = resolve._app.add_subtitle(start_frame, end_frame, text, layer)
            if sid == -1:
                return "ERROR: Could not add subtitle (check timeline and parameters)"
            fps = helpers.get_fps(ctx)
            start_tc = helpers.format_tc(start_frame, fps)
            end_tc = helpers.format_tc(end_frame, fps)
            return f"Added subtitle id={sid} [{start_tc} → {end_tc}]: {text}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def edit_subtitle(ctx: Context, subtitle_id: int, new_text: str) -> str:
        """Edit the text of an existing subtitle.

        Args:
            subtitle_id: The subtitle ID (from get_subtitles).
            new_text: New text for the subtitle.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.edit_subtitle(subtitle_id, new_text)
            if not ok:
                return f"ERROR: Could not edit subtitle {subtitle_id}"
            return f"Updated subtitle {subtitle_id}: {new_text}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def move_subtitle(ctx: Context, subtitle_id: int, new_start_frame: int) -> str:
        """Move a subtitle to a new start position on the timeline.

        Args:
            subtitle_id: The subtitle ID (from get_subtitles).
            new_start_frame: New start position in frames.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.move_subtitle(subtitle_id, new_start_frame)
            if not ok:
                return f"ERROR: Could not move subtitle {subtitle_id}"
            return f"Moved subtitle {subtitle_id} to frame {new_start_frame}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def resize_subtitle(ctx: Context, subtitle_id: int, new_duration: int,
                        from_right: bool = True) -> str:
        """Resize a subtitle's duration.

        Args:
            subtitle_id: The subtitle ID (from get_subtitles).
            new_duration: New duration in frames.
            from_right: If True, change end time. If False, change start time.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.resize_subtitle(subtitle_id, new_duration, from_right)
            if not ok:
                return f"ERROR: Could not resize subtitle {subtitle_id}"
            return f"Resized subtitle {subtitle_id} to {new_duration} frames"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_subtitle(ctx: Context, subtitle_id: int) -> str:
        """Delete a subtitle from the timeline.

        Args:
            subtitle_id: The subtitle ID (from get_subtitles).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.delete_subtitle(subtitle_id)
            if not ok:
                return f"ERROR: Could not delete subtitle {subtitle_id}"
            return f"Deleted subtitle {subtitle_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def import_subtitle(ctx: Context, file_path: str, offset: int = 0,
                        encoding: str = "UTF-8") -> str:
        """Import a subtitle file (SRT, ASS, VTT, or SBV) into the timeline.

        Args:
            file_path: Absolute path to the subtitle file.
            offset: Frame offset to apply to all imported subtitles (0 = no offset).
            encoding: Character encoding of the file (default: UTF-8).
        """
        try:
            import os
            if not os.path.isabs(file_path):
                return "ERROR: file_path must be an absolute path."
            if not os.path.isfile(file_path):
                return f"ERROR: File not found: {file_path}"
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.import_subtitle(file_path, offset, encoding)
            if not ok:
                return f"ERROR: Could not import subtitles from {file_path}"
            return f"Imported subtitles from {file_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def export_subtitles(ctx: Context, file_path: str) -> str:
        """Export subtitles to an .ass file.

        Args:
            file_path: Absolute path for the output .ass file.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.export_subtitles(file_path)
            if not ok:
                return f"ERROR: Could not export subtitles to {file_path}"
            return f"Exported subtitles to {file_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def speech_recognition(ctx: Context, model: str = "turbo", language: str = "",
                           max_line_width: int = 42) -> str:
        """Transcribe the timeline audio with Whisper and add subtitles (headless).

        Runs in the background: the SRT is imported automatically when the
        transcription finishes (typically 10s-2min). Poll get_subtitles to see
        the result.

        Args:
            model: Whisper model name (e.g. "turbo", "base", "small").
            language: Source language code (e.g. "en", "ru"); empty = autodetect.
            max_line_width: Max characters per subtitle line; 0 disables splitting.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app._call("scriptSpeechRecognition", model, language, max_line_width)
            if not ok:
                return "ERROR: Could not start speech recognition (empty timeline or speech environment not installed)"
            return "Speech recognition started in the background — poll get_subtitles for the result"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Subtitle Styles ───────────────────────────────────────────────

    @mcp.tool()
    def get_subtitle_styles(ctx: Context,
                            global_styles: bool = False) -> str:
        """List all subtitle styles (local or global).

        Args:
            global_styles: If True, list global styles shared across projects.

        Returns a markdown table: name | font | size | color | outline | shadow | alignment | bold | italic
        """
        try:
            resolve = helpers.get_resolve(ctx)
            styles = resolve._app.get_subtitle_styles(global_styles)
            if not styles:
                scope = "global" if global_styles else "local"
                return f"No {scope} subtitle styles found."
            header = "| name | font | size | color | outline | shadow | alignment | bold | italic |"
            sep = "|------|------|------|-------|---------|--------|-----------|------|--------|"
            lines = [header, sep]
            for s in styles:
                name = s.get("name", "")
                font = s.get("fontName", "")
                size = s.get("fontSize", "")
                # Show primary colour — strip alpha prefix if fully opaque (#ffRRGGBB → #RRGGBB)
                color = s.get("primaryColour", "")
                if isinstance(color, str) and len(color) == 9 and color.startswith("#ff"):
                    color = "#" + color[3:]
                outline_val = s.get("outline", "")
                shadow_val = s.get("shadow", "")
                alignment = s.get("alignment", "")
                bold = "yes" if str(s.get("bold", "")).lower() in ("true", "1") else "no"
                italic = "yes" if str(s.get("italic", "")).lower() in ("true", "1") else "no"
                lines.append(f"| {name} | {font} | {size} | {color} | {outline_val} | {shadow_val} | {alignment} | {bold} | {italic} |")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_subtitle_style(ctx: Context, name: str,
                           style: dict | None = None,
                           global_style: bool = False) -> str:
        """Create or update a subtitle style.

        Args:
            name: Style name (e.g. "Default", "Accent").
            style: Dict of style properties. Accepts snake_case keys:
                font_name, font_size, primary_colour, secondary_colour,
                outline_colour, back_colour, bold, italic, underline,
                strike_out, scale_x, scale_y, spacing, angle,
                border_style, outline, shadow, alignment,
                margin_l, margin_r, margin_v.
                Colors as "#RRGGBB" or "#AARRGGBB".
            global_style: If True, modify global styles.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            params = _to_camel(style) if style else {}
            ok = resolve._app.set_subtitle_style(name, params, global_style)
            if not ok:
                return f"ERROR: Could not set subtitle style '{name}'"
            action = "Updated" if style else "Created"
            scope = "global" if global_style else "local"
            return f"{action} {scope} subtitle style '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_subtitle_style(ctx: Context, name: str,
                              global_style: bool = False) -> str:
        """Delete a subtitle style. Cannot delete "Default".

        Args:
            name: Style name to delete.
            global_style: If True, delete from global styles.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.delete_subtitle_style(name, global_style)
            if not ok:
                return f"ERROR: Could not delete style '{name}' (is it 'Default'?)"
            scope = "global" if global_style else "local"
            return f"Deleted {scope} subtitle style '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_subtitle_style_name(ctx: Context, subtitle_id: int,
                                style_name: str) -> str:
        """Assign a named style to a specific subtitle.

        Args:
            subtitle_id: The subtitle ID (from get_subtitles).
            style_name: Name of the style to assign.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_subtitle_style_name(subtitle_id, style_name)
            if not ok:
                return f"ERROR: Could not assign style '{style_name}' to subtitle {subtitle_id}"
            return f"Assigned style '{style_name}' to subtitle {subtitle_id}"
        except Exception as e:
            return f"ERROR: {e}"
