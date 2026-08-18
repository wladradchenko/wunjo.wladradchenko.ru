"""Effect tools: add, remove, list effects on timeline clips."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_available_effects(ctx: Context, effect_type: str = "all") -> str:
        """List all available effects in Wunjo Make.

        Args:
            effect_type: Filter by "video", "audio", or "all".

        Returns a markdown table: id | name | type
        """
        try:
            resolve = helpers.get_resolve(ctx)
            effects = resolve._app.get_available_effects()
            if not effects:
                return "No effects found."
            if effect_type != "all":
                effects = [e for e in effects if e.get("type") == effect_type]
            lines = ["| id | name | type |", "|-----|------|------|"]
            for e in effects:
                lines.append(f"| {e.get('id', '')} | {e.get('name', '')} | {e.get('type', '')} |")
            return f"{len(effects)} effects:\n\n" + "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def add_effect(ctx: Context, clip_id: int, effect_id: str,
                   params: dict[str, str] | None = None) -> str:
        """Add an effect/filter to a timeline clip with optional parameters.

        Common effects for fill-frame: "qtblend" with rect="0 0 W H 1", distort="1".

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID (e.g. "qtblend", "affine", "brightness").
            params: Optional parameter dict (key=value pairs).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.add_clip_effect(clip_id, effect_id, params)
            if not ok:
                return f"ERROR: Could not add effect '{effect_id}' to clip {clip_id}"
            param_str = ""
            if params:
                param_str = " with " + ", ".join(f"{k}={v}" for k, v in params.items())
            return f"Added effect '{effect_id}' to clip {clip_id}{param_str}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def remove_effect(ctx: Context, clip_id: int, effect_id: str) -> str:
        """Remove an effect from a timeline clip.

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID to remove (e.g. "qtblend").
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.remove_clip_effect(clip_id, effect_id)
            if not ok:
                return f"ERROR: Could not remove effect '{effect_id}' from clip {clip_id}"
            return f"Removed effect '{effect_id}' from clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_effects(ctx: Context, clip_id: int) -> str:
        """List all effects on a timeline clip.

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            effects = resolve._app.get_clip_effects(clip_id)
            if not effects:
                return f"Clip {clip_id}: no effects"
            return f"Clip {clip_id} effects: {effects}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_effect_param(ctx: Context, clip_id: int, effect_id: str,
                         param_name: str, value: str) -> str:
        """Set a single parameter on an existing effect.

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID (e.g. "placebo.shader", "brightness").
            param_name: Parameter name (e.g. "shader_text", "av.brightness").
            value: New value as string.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_effect_param(clip_id, effect_id,
                                                param_name, value)
            if not ok:
                return f"ERROR: Could not set param '{param_name}' on effect '{effect_id}' (clip {clip_id})"
            return f"Set {effect_id}.{param_name} = {value!r} on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_effect_param(ctx: Context, clip_id: int, effect_id: str,
                         param_name: str) -> str:
        """Read a single parameter value from an effect.

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID.
            param_name: Parameter name to read.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            val = resolve._app.get_effect_param(clip_id, effect_id,
                                                 param_name)
            if not val:
                return f"ERROR: No value for '{param_name}' on effect '{effect_id}' (clip {clip_id})"
            return f"{effect_id}.{param_name} = {val!r}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_effect_expression(ctx: Context, clip_id: int, effect_id: str,
                              param_name: str, expression: str,
                              base_value: float) -> str:
        """Attach a JavaScript expression to an effect parameter for animation.

        The expression can use variables: time, duration, value (= base_value),
        in, out, fps, width, height. Example: "linear(time, 0, duration, 0, value)".

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID.
            param_name: Parameter name to animate.
            expression: JavaScript expression string.
            base_value: The base "value" variable the expression can reference.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_effect_expression(clip_id, effect_id,
                                                     param_name, expression,
                                                     base_value)
            if not ok:
                return f"ERROR: Could not set expression on '{param_name}' (effect '{effect_id}', clip {clip_id})"
            return (f"Set expression on {effect_id}.{param_name}: "
                    f"{expression!r} (base={base_value})")
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def clear_effect_expression(ctx: Context, clip_id: int, effect_id: str,
                                param_name: str) -> str:
        """Remove a JavaScript expression from an effect parameter.

        Restores the parameter to its static base value.

        Args:
            clip_id: Timeline clip ID.
            effect_id: MLT effect ID.
            param_name: Parameter name to clear expression from.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.clear_effect_expression(clip_id, effect_id,
                                                       param_name)
            if not ok:
                return f"ERROR: Could not clear expression on '{param_name}' (effect '{effect_id}', clip {clip_id})"
            return f"Cleared expression on {effect_id}.{param_name} (clip {clip_id})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_clip_opacity(ctx: Context, clip_id: int, opacity: float = 1.0,
                         keyframes: dict[int, float] | None = None) -> str:
        """Set clip opacity (transparency). Uses qtblend effect internally.

        Args:
            clip_id: Timeline clip ID.
            opacity: Opacity 0.0 (transparent) to 1.0 (opaque). Ignored if keyframes provided.
            keyframes: Optional dict {frame: opacity} for animated opacity.
                       Frame is relative to clip start. Example: {0: 0.0, 25: 1.0} for 1s fade-in.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            params: dict[str, str] = {}
            if keyframes:
                # MLT keyframe syntax: "frame=value;frame=value"
                kf_str = ";".join(
                    f"{f}={int(v * 100)}" for f, v in sorted(keyframes.items())
                )
                params["opacity"] = kf_str
            else:
                params["opacity"] = str(int(opacity * 100))
            ok = resolve._app.add_clip_effect(clip_id, "qtblend", params)
            if not ok:
                return f"ERROR: Could not set opacity on clip {clip_id}"
            if keyframes:
                return f"Set animated opacity on clip {clip_id}: {params['opacity']}"
            return f"Set opacity {opacity:.0%} on clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def fill_frame(ctx: Context, clip_id: int) -> str:
        """Scale a timeline clip to fill the project frame (remove black bars).

        Auto-detects source resolution, calculates scale-to-fill with center crop,
        and applies a qtblend effect. No distortion — aspect ratio is preserved,
        excess is cropped equally from both sides.

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.fill_frame(clip_id)
            if not ok:
                return f"ERROR: Could not fill frame for clip {clip_id} (already matching or clip not found)"
            return f"Applied fill-frame to clip {clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def paste_effects(ctx: Context, source_clip_id: int,
                      target_clip_id: int) -> str:
        """Copy all effects from one clip and paste them onto another.

        Args:
            source_clip_id: Timeline clip ID to copy effects from.
            target_clip_id: Timeline clip ID to paste effects onto.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            xml = resolve._app.copy_clip_effects(source_clip_id)
            if not xml:
                return f"ERROR: No effects found on clip {source_clip_id}"
            ok = resolve._app.paste_clip_effects(target_clip_id, xml)
            if not ok:
                return f"ERROR: Could not paste effects onto clip {target_clip_id}"
            return f"Pasted effects from clip {source_clip_id} → clip {target_clip_id}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_opacity(ctx: Context, clip_id: int) -> str:
        """Get clip opacity (0.0 = transparent, 1.0 = fully opaque).

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            opacity = resolve._app.get_clip_opacity(clip_id)
            if opacity < 0:
                return f"ERROR: Clip {clip_id} not found"
            return f"Clip {clip_id} opacity: {opacity:.1%}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_clip_enabled(ctx: Context, clip_id: int, enabled: bool) -> str:
        """Enable or disable a clip (blind eye icon).

        Args:
            clip_id: Timeline clip ID.
            enabled: True to enable, False to disable.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_clip_enabled(clip_id, enabled)
            if not ok:
                return f"ERROR: Could not set enabled state for clip {clip_id}"
            state = "enabled" if enabled else "disabled"
            return f"Clip {clip_id} {state}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_clip_color(ctx: Context, clip_id: int) -> str:
        """Get clip color tags (semicolon-separated hex colors stored as bin clip tags).

        Args:
            clip_id: Timeline clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            color = resolve._app.get_clip_color(clip_id)
            if not color:
                return f"Clip {clip_id} has no color tags"
            return f"Clip {clip_id} color tags: {color}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_clip_color(ctx: Context, clip_id: int, color_tag: str) -> str:
        """Set clip color tags (stored on the bin clip).

        Args:
            clip_id: Timeline clip ID.
            color_tag: Semicolon-separated hex colors (e.g. '#ff0000;#00ff00'). Pass empty string to clear.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_clip_color(clip_id, color_tag)
            if not ok:
                return f"ERROR: Could not set color for clip {clip_id}"
            return f"Clip {clip_id} color tags set to: {color_tag}"
        except Exception as e:
            return f"ERROR: {e}"
