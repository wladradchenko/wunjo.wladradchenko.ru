"""MCP tools: visual preview — timeline thumbnails, bin clip thumbnails,
contact sheets, and QC crops."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import Context
from mcp.server.fastmcp import Image as McpImage  # PIL's Image is used below too


def _seen(path: str):
    """Hand a rendered frame back in the form the agent can actually use.

    An agent that can read files — Claude Code — is best served by the path: it
    opens the picture itself and nothing large travels through the tool result.
    A model running inside the editor has no such reach, so for it the picture
    itself is attached, and it can look at the footage instead of guessing from
    file names. WUNJO_VISION says which of the two is at the other end.
    """
    if not path or path.startswith("ERROR") or os.environ.get("WUNJO_VISION") != "1":
        return path
    try:
        return McpImage(path=path)
    except Exception:  # noqa: BLE001 — a picture that cannot be read is still a path
        return path

# ---------------------------------------------------------------------------
# Panel name aliases → Wunjo Make dock objectName
# ---------------------------------------------------------------------------

PANEL_ALIASES = {
    "effect_stack": "effect_stack",
    "effects": "effect_list",
    "timeline": "timeline",
    "project_bin": "project_bin",
    "bin": "project_bin",
    "clip_monitor": "clipmonitor",
    "project_monitor": "projectmonitor",
    "mixer": "mixer",
    "subtitles": "subtitles",
    "compositions": "transition_list",
    "library": "library",
    "markers": "markers",
    "timeremap": "timeremap",
    "textedit": "textedit",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREVIEW_DIR: Path | None = None


def _ensure_preview_dir() -> Path:
    """Return (and create) a temp directory for preview images.

    Cleans files older than 1 hour on each call.
    """
    global _PREVIEW_DIR
    if _PREVIEW_DIR is None:
        # ~/.cache, NOT /tmp: the editor runs inside a flatpak sandbox with a
        # private /tmp, so previews must live on a path both sides can see.
        _PREVIEW_DIR = Path(os.environ.get("WUNJO_PREVIEW_DIR",
                                           Path.home() / ".cache" / "wunjo-mcp-preview"))
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Cleanup stale files (>1 hour)
    cutoff = time.time() - 3600
    for f in _PREVIEW_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass

    return _PREVIEW_DIR


def _temp_path(suffix: str = ".jpg") -> str:
    """Return a unique temp file path."""
    d = _ensure_preview_dir()
    return str(d / f"{uuid.uuid4().hex}{suffix}")


def _calc_thumb_size(proj_w: int, proj_h: int, max_short_side: int = 480) -> tuple[int, int]:
    """Calculate thumbnail dimensions preserving aspect ratio.

    The shorter side is capped at max_short_side.
    """
    if proj_w <= 0 or proj_h <= 0:
        return max_short_side, max_short_side
    if proj_w >= proj_h:
        # Landscape — height is short side
        h = min(proj_h, max_short_side)
        w = int(proj_w * h / proj_h)
    else:
        # Portrait — width is short side
        w = min(proj_w, max_short_side)
        h = int(proj_h * w / proj_w)
    return w, h


# Region presets for QC crop (normalized 0-1 coordinates: x, y, w, h)
REGION_PRESETS = {
    "center": (0.25, 0.25, 0.5, 0.5),
    "top-third": (0.25, 0.0, 0.5, 0.333),
    "bottom-third": (0.25, 0.667, 0.5, 0.333),
    "left-third": (0.0, 0.25, 0.333, 0.5),
    "right-third": (0.667, 0.25, 0.333, 0.5),
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(mcp, helpers):

    # ── Tool 1: render_frame (timeline thumbnail) ─────────────────────

    @mcp.tool()
    def render_frame(
        ctx: Context,
        frame: int,
        max_short_side: int = 480,
    ) -> object:
        """Render a composited timeline frame as a JPEG thumbnail.

        Returns the file path to the rendered image (attach to response).

        Args:
            frame: Timeline frame number to render.
            max_short_side: Max pixel size of the shorter dimension (default 480).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            proj_w = int(app.get_project_resolution_width())
            proj_h = int(app.get_project_resolution_height())
            w, h = _calc_thumb_size(proj_w, proj_h, max_short_side)

            out = _temp_path()
            result = app.render_timeline_frame(frame, w, h, out)
            if not result:
                return "ERROR: Failed to render timeline frame (D-Bus returned empty)"
            return _seen(result)
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 2: render_bin_frame (bin clip thumbnail) ─────────────────

    @mcp.tool()
    def render_bin_frame(
        ctx: Context,
        bin_id: str,
        frame_position: str = "middle",
        max_short_side: int = 480,
    ) -> object:
        """Render a single frame from a media pool clip as a JPEG thumbnail.

        Args:
            bin_id: Media pool clip ID.
            frame_position: "first", "middle", "last", or an integer frame number.
            max_short_side: Max pixel size of the shorter dimension (default 480).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            # Resolve frame position
            props = app.get_clip_properties(bin_id)
            duration = int(props.get("duration", "0"))

            if frame_position == "first":
                frame = 0
            elif frame_position == "middle":
                frame = max(0, duration // 2)
            elif frame_position == "last":
                frame = max(0, duration - 1)
            else:
                try:
                    frame = int(frame_position)
                except ValueError:
                    return f"ERROR: Invalid frame_position '{frame_position}' — use 'first', 'middle', 'last', or an integer"

            proj_w = int(app.get_project_resolution_width())
            proj_h = int(app.get_project_resolution_height())
            w, h = _calc_thumb_size(proj_w, proj_h, max_short_side)

            out = _temp_path()
            result = app.render_bin_frame(bin_id, frame, w, h, out)
            if not result:
                return "ERROR: Failed to render bin frame (D-Bus returned empty)"
            return _seen(result)
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 3: render_contact_sheet (grid of frames) ─────────────────

    @mcp.tool()
    def render_contact_sheet(
        ctx: Context,
        bin_id: str,
        num_frames: int = 8,
        thumb_width: int = 320,
    ) -> object:
        """Render a contact sheet (grid of evenly-spaced frames) from a bin clip.

        Uses Pillow to assemble individual frames into a labeled grid image.
        Returns the file path to the contact sheet JPEG.

        Args:
            bin_id: Media pool clip ID.
            num_frames: Number of frames to sample (default 8, max 16).
            thumb_width: Width of each thumbnail in pixels (default 320).
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            resolve = helpers.get_resolve(ctx)
            app = resolve._app
            fps = helpers.get_fps(ctx)

            props = app.get_clip_properties(bin_id)
            duration = int(props.get("duration", "0"))
            clip_name = props.get("name", bin_id)

            num_frames = min(max(num_frames, 2), 16)
            if duration <= 0:
                return "ERROR: Clip has zero duration"

            # Calculate evenly-spaced frame positions
            step = max(1, (duration - 1) / (num_frames - 1))
            positions = [int(round(i * step)) for i in range(num_frames)]

            # Project aspect ratio for thumb height
            proj_w = int(app.get_project_resolution_width())
            proj_h = int(app.get_project_resolution_height())
            thumb_h = int(thumb_width * proj_h / proj_w) if proj_w > 0 else int(thumb_width * 9 / 16)

            # Render individual frames
            thumb_paths = []
            for pos in positions:
                p = _temp_path()
                result = app.render_bin_frame(bin_id, pos, thumb_width, thumb_h, p)
                if result:
                    thumb_paths.append((pos, result))

            if not thumb_paths:
                return "ERROR: Failed to render any frames"

            # Assemble grid (4 columns)
            cols = 4
            rows = (len(thumb_paths) + cols - 1) // cols
            label_h = 20
            cell_h = thumb_h + label_h
            grid_w = cols * thumb_width
            grid_h = rows * cell_h

            grid = Image.new("RGB", (grid_w, grid_h), (32, 32, 32))
            draw = ImageDraw.Draw(grid)

            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (IOError, OSError):
                font = ImageFont.load_default()

            for idx, (pos, path) in enumerate(thumb_paths):
                col = idx % cols
                row = idx // cols
                x = col * thumb_width
                y = row * cell_h

                try:
                    thumb = Image.open(path)
                    grid.paste(thumb, (x, y))
                except Exception:
                    pass

                # Label with timecode
                tc = helpers.format_tc(pos, fps)
                draw.text((x + 4, y + thumb_h + 2), f"#{pos} {tc}",
                          fill=(200, 200, 200), font=font)

            # Save contact sheet
            out = _temp_path()
            grid.save(out, "JPEG", quality=90)

            # Clean up individual thumbnails
            for _, path in thumb_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

            return _seen(out)
        except ImportError:
            return "ERROR: Pillow is required for contact sheets. Install with: pip install Pillow"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 4: render_crop (1:1 QC crop) ─────────────────────────────

    @mcp.tool()
    def render_crop(
        ctx: Context,
        frame: int,
        region: str = "center",
        crop_size: int = 480,
        custom_x: int | None = None,
        custom_y: int | None = None,
        custom_w: int | None = None,
        custom_h: int | None = None,
    ) -> object:
        """Render a timeline frame and crop a region for quality control.

        Renders at full project resolution, then crops a crop_size x crop_size
        square at 1:1 pixels (no scaling). Useful for inspecting detail
        (faces, text, artifacts, matching adjacent clips) at native resolution.

        Args:
            frame: Timeline frame number.
            region: Preset region name: "center", "top-third", "bottom-third",
                    "left-third", "right-third". Ignored if custom_x/y/w/h set.
            crop_size: Output crop size in pixels (default 480).
            custom_x: Custom crop X offset in pixels (overrides region preset).
            custom_y: Custom crop Y offset in pixels.
            custom_w: Custom crop width in pixels.
            custom_h: Custom crop height in pixels.
        """
        try:
            from PIL import Image

            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            proj_w = int(app.get_project_resolution_width())
            proj_h = int(app.get_project_resolution_height())

            # Render at full resolution
            full_path = _temp_path()
            result = app.render_timeline_frame(frame, proj_w, proj_h, full_path)
            if not result:
                return "ERROR: Failed to render timeline frame at full resolution"

            img = Image.open(full_path)
            actual_w, actual_h = img.size

            # Determine crop box (always crop_size x crop_size, 1:1 pixels)
            if custom_x is not None and custom_y is not None:
                cx = custom_x
                cy = custom_y
                cw = custom_w if custom_w is not None else crop_size
                ch = custom_h if custom_h is not None else crop_size
            else:
                preset = REGION_PRESETS.get(region)
                if not preset:
                    return f"ERROR: Unknown region '{region}'. Use: {', '.join(REGION_PRESETS.keys())}"
                # Preset gives normalized center point; we cut crop_size x crop_size around it
                rx, ry, rw, rh = preset
                center_x = int((rx + rw / 2) * actual_w)
                center_y = int((ry + rh / 2) * actual_h)
                cx = center_x - crop_size // 2
                cy = center_y - crop_size // 2
                cw = crop_size
                ch = crop_size

            # Clamp to image bounds
            cx = max(0, min(cx, actual_w - cw))
            cy = max(0, min(cy, actual_h - ch))
            cw = min(cw, actual_w - cx)
            ch = min(ch, actual_h - cy)

            cropped = img.crop((cx, cy, cx + cw, cy + ch))

            out = _temp_path()
            cropped.save(out, "JPEG", quality=95)

            # Clean up full-res temp
            try:
                os.unlink(full_path)
            except OSError:
                pass

            return out
        except ImportError:
            return "ERROR: Pillow is required for render_crop. Install with: pip install Pillow"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 5: screenshot_window (GUI capture) ────────────────────────

    @mcp.tool()
    def screenshot_window(
        ctx: Context,
        max_size: int = 600,
    ) -> str:
        """Capture a screenshot of the Wunjo Make GUI window with panel map.

        Uses Qt's QWidget::grab() — works on Windows, Linux, macOS.
        Returns the file path to the screenshot JPEG followed by a JSON
        panel map with bounding boxes of all visible dock panels.

        Args:
            max_size: Max dimension in pixels (default 600). Aspect ratio is preserved.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            out = _temp_path()
            result = app.capture_window(max_size, out)
            if not result:
                return "ERROR: Failed to capture window (D-Bus returned empty)"

            # Get panel geometries and scale to screenshot coordinates
            panels = app.get_panel_geometries()
            if panels:
                # Find main_window entry to compute scale factor
                main_win = next((p for p in panels if p.get("name") == "main_window"), None)
                if main_win and main_win.get("width", 0) > 0:
                    try:
                        from PIL import Image
                        img = Image.open(result)
                        img_w, img_h = img.size
                        scale_x = img_w / main_win["width"]
                        scale_y = img_h / main_win["height"]
                    except ImportError:
                        # Pillow not available — return unscaled panel map
                        visible = [p for p in panels
                                   if p.get("visible") and p.get("name") != "main_window"]
                        if visible:
                            return f"{result}\n\nPanel map (window coords, unscaled):\n{json.dumps(visible, indent=2)}"
                        return result

                    scaled_panels = []
                    for p in panels:
                        if p.get("name") == "main_window":
                            continue
                        if not p.get("visible", False):
                            continue
                        scaled_panels.append({
                            "name": p["name"],
                            "title": p.get("title", ""),
                            "x": int(p["x"] * scale_x),
                            "y": int(p["y"] * scale_y),
                            "width": int(p["width"] * scale_x),
                            "height": int(p["height"] * scale_y),
                        })
                    return f"{result}\n\nPanel map (screenshot coords):\n{json.dumps(scaled_panels, indent=2)}"

            return result
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 6: screenshot_crop (1:1 pixel crop of GUI region) ─────────

    @mcp.tool()
    def screenshot_crop(
        ctx: Context,
        x: int,
        y: int,
        width: int = 700,
        height: int = 700,
    ) -> str:
        """Crop a region of the Wunjo Make GUI window at 1:1 native pixels.

        Captures the full window at native resolution, then extracts the
        specified rectangle. Output is clamped to 700x700 px max.
        Useful for inspecting specific panels, buttons, or UI details.

        Args:
            x: Left edge of the crop region in pixels.
            y: Top edge of the crop region in pixels.
            width: Crop width in pixels (default/max 700).
            height: Crop height in pixels (default/max 700).
        """
        try:
            from PIL import Image

            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            # Capture full window at native resolution (maxSize=0 → no scaling)
            full_path = _temp_path()
            result = app.capture_window(0, full_path)
            if not result:
                return "ERROR: Failed to capture window (D-Bus returned empty)"

            img = Image.open(full_path)
            img_w, img_h = img.size

            # Clamp crop dimensions
            width = min(width, 700)
            height = min(height, 700)

            # Clamp origin to image bounds
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            x2 = min(x + width, img_w)
            y2 = min(y + height, img_h)

            cropped = img.crop((x, y, x2, y2))

            out = _temp_path()
            cropped.save(out, "JPEG", quality=95)

            # Clean up full-res capture
            try:
                os.unlink(full_path)
            except OSError:
                pass

            return out
        except ImportError:
            return "ERROR: Pillow is required for screenshot_crop. Install with: pip install Pillow"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Tool 7: screenshot_panel (crop a named panel) ─────────────────

    @mcp.tool()
    def screenshot_panel(
        ctx: Context,
        panel_name: str,
    ) -> str:
        """Capture a cropped screenshot of a specific Wunjo Make panel.

        Captures the full window at native resolution, resolves the panel's
        bounding box via D-Bus, and crops to that region. Upscales if the
        cropped area is smaller than 200px in any dimension.

        Args:
            panel_name: Panel name or alias. Common names:
                "timeline", "project_bin" (or "bin"), "effect_stack",
                "clip_monitor", "project_monitor", "mixer",
                "subtitles", "effects", "markers", "library".
        """
        try:
            from PIL import Image

            resolve = helpers.get_resolve(ctx)
            app = resolve._app

            # Resolve alias
            obj_name = PANEL_ALIASES.get(panel_name, panel_name)

            # Get panel geometries
            panels = app.get_panel_geometries()
            if not panels:
                return "ERROR: Could not retrieve panel geometries (D-Bus returned empty). Is Wunjo Make built with scriptGetPanelGeometries?"

            # Find main_window for coordinate reference
            main_win = next((p for p in panels if p.get("name") == "main_window"), None)
            if not main_win or main_win.get("width", 0) <= 0:
                return "ERROR: main_window geometry not found"

            # Find the requested panel
            panel = next((p for p in panels if p.get("name") == obj_name and p.get("visible")), None)
            if not panel:
                available = [p["name"] for p in panels if p.get("visible") and p.get("name") != "main_window"]
                return f"ERROR: Panel '{panel_name}' (objectName='{obj_name}') not found or not visible.\nAvailable panels: {', '.join(available)}"

            # Capture full window at native resolution
            full_path = _temp_path()
            result = app.capture_window(0, full_path)
            if not result:
                return "ERROR: Failed to capture window"

            img = Image.open(full_path)
            img_w, img_h = img.size

            # Scale panel coordinates from window coords to screenshot coords
            scale_x = img_w / main_win["width"]
            scale_y = img_h / main_win["height"]

            px = int(panel["x"] * scale_x)
            py = int(panel["y"] * scale_y)
            pw = int(panel["width"] * scale_x)
            ph = int(panel["height"] * scale_y)

            # Clamp to image bounds
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))
            pw = min(pw, img_w - px)
            ph = min(ph, img_h - py)

            cropped = img.crop((px, py, px + pw, py + ph))

            # Upscale if too small for useful inspection
            cw, ch = cropped.size
            if cw < 200 or ch < 200:
                factor = max(200 / cw, 200 / ch)
                cropped = cropped.resize(
                    (int(cw * factor), int(ch * factor)),
                    Image.LANCZOS,
                )

            out = _temp_path()
            cropped.save(out, "JPEG", quality=95)

            # Clean up full-res capture
            try:
                os.unlink(full_path)
            except OSError:
                pass

            return out
        except ImportError:
            return "ERROR: Pillow is required for screenshot_panel. Install with: pip install Pillow"
        except Exception as e:
            return f"ERROR: {e}"
