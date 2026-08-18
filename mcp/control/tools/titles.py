"""MCP tools: add_title, edit_title, import_srt_as_titles."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

from mcp.server.fastmcp import Context


# ── XML attribute key mapping (style dict key → XML attribute name) ──────

_STYLE_TO_XML_ATTR = {
    "font": "font",
    "font_size": "font-pixel-size",
    "font_weight": "font-weight",
    "color": "font-color",
    "alignment": "alignment",
    "font_italic": "font-italic",
    "font_underline": "font-underline",
    "font_outline": "font-outline",
    "font_outline_color": "font-outline-color",
    "letter_spacing": "letter-spacing",
    "line_spacing": "line-spacing",
    "shadow": "shadow",
}


def _parse_srt(path: str, fps: float) -> list[tuple[int, int, str]]:
    """Parse an SRT file into a list of (start_frame, end_frame, text).

    SRT timecode format: HH:MM:SS,mmm --> HH:MM:SS,mmm
    """
    def tc_to_frames(tc: str) -> int:
        h, m, rest = tc.strip().split(":")
        s, ms = rest.split(",")
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        return round(total_seconds * fps)

    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    entries = []
    # Split on blank lines to get blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Line 0: index (ignored)
        # Line 1: timecodes
        tc_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not tc_match:
            continue
        start = tc_to_frames(tc_match.group(1))
        end = tc_to_frames(tc_match.group(2))
        text = "\n".join(lines[2:])
        entries.append((start, end, text))

    return entries


def _build_title_xml(w: int, h: int, x: int, y: int, text: str, style: dict) -> str:
    """Build kdenlivetitle XML string with conditional attributes."""
    # Always-present attributes
    attrs = [
        f'font="{style["font"]}"',
        f'font-pixel-size="{style["font_size"]}"',
        f'font-weight="{style["font_weight"]}"',
        f'font-color="{style["color"]}"',
        f'alignment="{style["alignment"]}"',
    ]

    # Optional attributes — only include when non-default
    if style.get("font_italic", "0") != "0":
        attrs.append(f'font-italic="{style["font_italic"]}"')
    if style.get("font_underline", "0") != "0":
        attrs.append(f'font-underline="{style["font_underline"]}"')
    if float(style.get("font_outline", "0")) > 0:
        attrs.append(f'font-outline="{style["font_outline"]}"')
        attrs.append(
            f'font-outline-color="{style.get("font_outline_color", "#000000")}"'
        )
    if style.get("letter_spacing", "0") != "0":
        attrs.append(f'letter-spacing="{style["letter_spacing"]}"')
    if style.get("line_spacing", "0") != "0":
        attrs.append(f'line-spacing="{style["line_spacing"]}"')
    if style.get("shadow", ""):
        attrs.append(f'shadow="{style["shadow"]}"')

    safe_text = html.escape(text)
    attr_str = "\n    ".join(attrs)

    xml = (
        f'<kdenlivetitle width="{w}" height="{h}" LC_NUMERIC="C">\n'
        f' <item type="QGraphicsTextItem" z-index="0">\n'
        f'  <position x="{x}" y="{y}"/>\n'
        f"  <content\n    {attr_str}\n"
        f'    text="{safe_text}"/>\n'
        f" </item>\n"
    )

    # Background rect (optional)
    bg_color = style.get("bg_color", "")
    if bg_color:
        bg_x, bg_y, bg_w, bg_h = _compute_bg_rect(w, h, y, style)
        xml += (
            f' <item type="QGraphicsRectItem" z-index="-1">'
            f'<content rect="{bg_x},{bg_y},{bg_w},{bg_h}" '
            f'brushcolor="{bg_color}"/>'
            f"</item>\n"
        )

    xml += "</kdenlivetitle>"
    return xml


def _compute_bg_rect(
    w: int, h: int, text_y: int, style: dict
) -> tuple[int, int, int, int]:
    """Compute background rect position and size.

    Returns (x, y, width, height) for the bg rect.

    Logic:
    - If explicit bg_rect in style → use it directly (format "x,y,w,h")
    - If position_y == "bottom" → bottom strip covering lower quarter
    - Otherwise → full frame (backwards compatible)
    """
    explicit = style.get("bg_rect", "")
    if explicit:
        parts = [int(v) for v in explicit.split(",")]
        return (parts[0], parts[1], parts[2], parts[3])

    pos_y = style.get("position_y", "center")
    padding = int(style.get("bg_padding", "20"))

    if pos_y == "bottom":
        # Bottom strip: starts at 3/4 height minus padding, extends to bottom
        rect_y = (h * 3) // 4 - padding
        rect_h = h - rect_y
        return (0, rect_y, w, rect_h)

    # Default: full frame
    return (0, 0, w, h)


def register(mcp, helpers):
    @mcp.tool()
    def add_title(
        ctx: Context,
        text: str,
        track_id: int,
        position: int,
        duration: int,
        style: dict | None = None,
    ) -> str:
        """Create a title card and place it on the timeline.

        Generates a kdenlivetitle clip (text overlay), adds it to the
        media pool, and inserts it on the specified track.

        Args:
            text: Title text (supports newlines via \\n).
            track_id: Target video track ID.
            position: Start position in frames.
            duration: Duration in frames.
            style: Optional dict with keys:
                font          — font family (default "Sans Serif")
                font_size     — pixel size (default "80")
                font_weight   — CSS weight (default "700")
                color         — text color hex (default "#ffffff")
                alignment     — 0=left, 1=center, 2=right (default "1")
                position_y    — "top", "center", or "bottom" (default "center")
                font_italic   — "0" or "1"
                font_underline — "0" or "1"
                font_outline  — outline width in pixels, e.g. "3" (default "0")
                font_outline_color — outline color hex (default "#000000")
                letter_spacing — extra spacing in pixels (default "0")
                line_spacing  — extra line spacing in pixels (default "0")
                shadow        — "r;g;b;a;xoff;yoff;blur" e.g. "0;0;0;255;2;2;3"
                bg_color      — background color hex, supports alpha "#aarrggbb"
                bg_rect       — explicit bg rect "x,y,w,h" (overrides auto)
                bg_padding    — padding for auto bg rect in pixels (default "20")

        Lower-third example:
            style={"position_y": "bottom", "bg_color": "#80000000",
                   "font_size": "48", "alignment": "0",
                   "font_outline": "2", "font_outline_color": "#000000"}
        """
        try:
            resolve = helpers.get_resolve(ctx)
            project = helpers.get_project(ctx)
            pool = helpers.get_media_pool(ctx)
            tl = helpers.get_timeline(ctx)
            fps = helpers.get_fps(ctx)
            app = resolve._app

            # Project resolution
            w = int(app.get_project_resolution_width())
            h = int(app.get_project_resolution_height())

            # Style defaults
            s = {
                "font": "Sans Serif",
                "font_size": "80",
                "font_weight": "700",
                "color": "#ffffff",
                "bg_color": "",
                "alignment": "1",
                "position_y": "center",
                "font_italic": "0",
                "font_underline": "0",
                "font_outline": "0",
                "font_outline_color": "#000000",
                "letter_spacing": "0",
                "line_spacing": "0",
                "shadow": "",
                "bg_rect": "",
                "bg_padding": "20",
            }
            if style:
                s.update({k: str(v) for k, v in style.items()})

            # Vertical position
            y_map = {
                "top": h // 6,
                "center": h // 2 - 40,
                "bottom": h * 3 // 4,
            }
            y = y_map.get(s["position_y"], y_map["center"])
            x = 0  # alignment handles horizontal

            title_xml = _build_title_xml(w, h, x, y, text, s)

            # Create title clip in bin
            clip_name = text[:30].replace("\n", " ")
            bin_id = app.create_title_clip(title_xml, duration, clip_name)
            if not bin_id or bin_id == "-1":
                return "ERROR: Failed to create title clip in bin"

            import time
            time.sleep(0.3)  # D-Bus async settle

            # Insert on timeline
            clip_id = app.insert_clip(bin_id, track_id, position)
            if clip_id < 0:
                return f"ERROR: Title created (bin {bin_id}) but insert failed"

            tc_start = helpers.format_tc(position, fps)
            tc_end = helpers.format_tc(position + duration, fps)
            return (
                f"Title '{clip_name}' created.\n"
                f"- Bin ID: {bin_id}\n"
                f"- Timeline clip ID: {clip_id}\n"
                f"- Position: {tc_start} → {tc_end} ({duration} frames)\n"
                f"- Track: {track_id}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def edit_title(
        ctx: Context,
        clip_id: int,
        new_text: str | None = None,
        style: dict | None = None,
    ) -> str:
        """Edit the text and/or style of an existing title clip on the timeline.

        Args:
            clip_id: Timeline clip ID of the title to edit.
            new_text: New text (None to keep current text).
            style: Dict of style keys to update (same keys as add_title).
        """
        if new_text is None and not style:
            return "ERROR: Provide new_text and/or style to update."
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            # Get bin_id from timeline clip info
            info = app.get_timeline_clip_info(clip_id)
            if not info:
                return f"ERROR: Clip {clip_id} not found on timeline."
            bin_id = str(info.get("binId", info.get("bin_id", "")))
            if not bin_id:
                return f"ERROR: Could not determine bin ID for clip {clip_id}."

            # Get current XML
            current_xml = app.get_title_xml(bin_id)
            if not current_xml:
                return f"ERROR: Clip {bin_id} has no title XML (not a title clip?)."

            # Parse XML
            root = ET.fromstring(current_xml)
            content_el = root.find(".//content")
            if content_el is None:
                return "ERROR: No <content> element found in title XML."

            # Update text
            if new_text is not None:
                content_el.set("text", html.escape(new_text))

            # Update style attributes
            if style:
                for key, value in style.items():
                    xml_attr = _STYLE_TO_XML_ATTR.get(key)
                    if xml_attr:
                        content_el.set(xml_attr, str(value))

            # Serialize back
            updated_xml = ET.tostring(root, encoding="unicode")

            # Apply
            ok = app.set_title_xml(bin_id, updated_xml)
            if not ok:
                return f"ERROR: set_title_xml failed for bin {bin_id}."

            parts = [f"Title clip {clip_id} (bin {bin_id}) updated."]
            if new_text is not None:
                parts.append(f"- Text: \"{new_text[:50]}\"")
            if style:
                parts.append(f"- Style keys: {', '.join(style.keys())}")
            return "\n".join(parts)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def import_srt_as_titles(
        ctx: Context,
        srt_path: str,
        track_id: int,
        style: dict | None = None,
        offset_frames: int = 0,
    ) -> str:
        """Import an SRT subtitle file as title clips on the timeline.

        Each SRT entry becomes a separate title clip. Useful for
        speech-to-text (Whisper) output or manual subtitle files.

        Args:
            srt_path: Absolute path to .srt file.
            track_id: Video track to place titles on.
            style: Style dict applied to all titles (same keys as add_title).
            offset_frames: Shift all timecodes by this many frames.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            project = helpers.get_project(ctx)
            fps = helpers.get_fps(ctx)
            app = resolve._app

            entries = _parse_srt(srt_path, fps)
            if not entries:
                return "ERROR: No subtitle entries found in SRT file."

            # Project resolution
            w = int(app.get_project_resolution_width())
            h = int(app.get_project_resolution_height())

            # Style defaults (subtitle-friendly: bottom position, semi-transparent bg)
            s = {
                "font": "Sans Serif",
                "font_size": "48",
                "font_weight": "400",
                "color": "#ffffff",
                "bg_color": "",
                "alignment": "1",
                "position_y": "bottom",
                "font_italic": "0",
                "font_underline": "0",
                "font_outline": "2",
                "font_outline_color": "#000000",
                "letter_spacing": "0",
                "line_spacing": "0",
                "shadow": "",
                "bg_rect": "",
                "bg_padding": "20",
            }
            if style:
                s.update({k: str(v) for k, v in style.items()})

            y_map = {
                "top": h // 6,
                "center": h // 2 - 40,
                "bottom": h * 3 // 4,
            }
            y = y_map.get(s["position_y"], y_map["center"])

            import time

            created = 0
            errors = 0
            for start, end, text in entries:
                start += offset_frames
                end += offset_frames
                duration = max(end - start, 1)

                title_xml = _build_title_xml(w, h, 0, y, text, s)
                clip_name = text[:30].replace("\n", " ")
                bin_id = app.create_title_clip(title_xml, duration, clip_name)
                if not bin_id or bin_id == "-1":
                    errors += 1
                    continue

                time.sleep(0.15)  # D-Bus settle

                cid = app.insert_clip(bin_id, track_id, start)
                if cid < 0:
                    errors += 1
                    continue
                created += 1

            lines = [f"Imported {created}/{len(entries)} subtitle entries."]
            if errors:
                lines.append(f"- {errors} entries failed.")
            lines.append(f"- Track: {track_id}")
            lines.append(f"- Offset: {offset_frames} frames")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"
