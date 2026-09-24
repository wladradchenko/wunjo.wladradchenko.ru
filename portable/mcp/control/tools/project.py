"""Project-level tools: info, save, load, render."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_project_info(ctx: Context) -> str:
        """Get current project settings: name, fps, resolution, path, track counts, duration."""
        try:
            proj = helpers.get_project(ctx)
            tl = proj.GetCurrentTimeline()
            fps = proj.GetFps()
            w, h = proj.GetResolution()
            name = proj.GetName()
            path = proj.GetProjectPath()

            video_tracks = tl.GetTrackCount("video") if tl else 0
            audio_tracks = tl.GetTrackCount("audio") if tl else 0
            duration = tl.GetTotalDuration() if tl else 0
            dur_tc = helpers.format_tc(duration, fps) if tl else "00:00:00:00"

            lines = [
                f"**Project:** {name}",
                f"**Path:** {path}",
                f"**FPS:** {fps}",
                f"**Resolution:** {w}x{h}",
                f"**Video tracks:** {video_tracks}",
                f"**Audio tracks:** {audio_tracks}",
                f"**Duration:** {dur_tc} ({duration} frames)",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def new_project(ctx: Context, name: str = "") -> str:
        """Create a new empty Wunjo Make project.

        Args:
            name: Project name (optional).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            resolve._app.new_project(name)
            # Creation is asynchronous (runs on the next event-loop tick to stay
            # out of the D-Bus dispatch); confirm with get_project_info.
            return "New project is being created — confirm with get_project_info."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def open_project(ctx: Context, file_path: str) -> str:
        """Open a Wunjo Make project file (with retry logic).

        Unlike load_project, this uses the lower-level D-Bus scriptOpenProject
        method which includes automatic retry for robustness.

        Args:
            file_path: Absolute path to .wmproj file.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.open_project(file_path)
            if not ok:
                return f"ERROR: Could not open {file_path}"
            return f"Opened project: {file_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_project_profile(ctx: Context, width: int, height: int,
                            fps_num: int, fps_den: int = 1) -> str:
        """Change project resolution and frame rate.

        Args:
            width: Frame width in pixels (e.g. 1920).
            height: Frame height in pixels (e.g. 1080).
            fps_num: FPS numerator (e.g. 25 for 25fps, 30000 for 29.97fps).
            fps_den: FPS denominator (default 1; use 1001 for 29.97/59.94fps).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_project_profile(width, height, fps_num, fps_den)
            if not ok:
                return f"ERROR: Could not set profile {width}x{height} {fps_num}/{fps_den}fps"
            fps = fps_num / fps_den
            return f"Profile set to {width}x{height} @ {fps:.2f}fps"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def save_project(ctx: Context, file_path: str = "") -> str:
        """Save the current project. If file_path is given, does Save As.

        Args:
            file_path: Optional path for Save As. Empty string saves in place.
        """
        try:
            proj = helpers.get_project(ctx)
            if file_path:
                ok = proj.SaveAs(file_path)
                return f"Saved to {file_path}" if ok else "ERROR: SaveAs failed"
            else:
                ok = proj.Save()
                return f"Saved to {proj.GetProjectPath()}" if ok else "ERROR: Save failed"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def render_video(ctx: Context, output_path: str = "",
                     preset_name: str = "", in_frame: int = -1,
                     out_frame: int = -1, params: dict[str, str] | None = None) -> str:
        """Start rendering the timeline to a video file.

        Args:
            output_path: Output file path. If empty, uses project default.
            preset_name: Render preset name (use get_render_presets to list).
                         If empty, uses the simple render path.
            in_frame: Start frame (-1 = project start).
            out_frame: End frame (-1 = project end).
            params: Optional dict of FFmpeg/MLT parameter overrides
                    (e.g. {"crf": "19", "preset": "fast", "b:v": "5000k"}).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            if preset_name or params or in_frame >= 0 or out_frame >= 0:
                # Advanced render with parameters
                if not output_path:
                    return "ERROR: output_path required for parametric render"
                ok = resolve._app.render_with_params(
                    output_path, preset_name, in_frame, out_frame, params
                )
                if not ok:
                    return "ERROR: Render failed (check preset name and parameters)"
                return f"Render started -> {output_path}"
            else:
                if output_path:
                    resolve._app.render(output_path)
                else:
                    proj = helpers.get_project(ctx)
                    proj.StartRendering()
                return "Render started" + (f" -> {output_path}" if output_path else "")
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_render_presets(ctx: Context) -> str:
        """List all available render presets (codec/format profiles).

        Returns a newline-separated list of preset names that can be
        used with render_video's preset_name parameter.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            presets = resolve._app.get_render_presets()
            if not presets:
                return "No render presets found."
            return f"{len(presets)} presets:\n" + "\n".join(f"- {p}" for p in presets)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_render_jobs(ctx: Context) -> str:
        """Get list of render jobs with status and progress.

        Returns a markdown table showing each job's output path, status,
        progress percentage, and current frame.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            jobs = resolve._app.get_render_jobs()
            if not jobs:
                return "No render jobs."
            lines = ["| path | status | progress | frame |",
                      "|------|--------|----------|-------|"]
            for j in jobs:
                lines.append(
                    f"| {j.get('path', '?')} | {j.get('status', '?')} "
                    f"| {j.get('progress', 0)}% | {j.get('frame', 0)} |"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def abort_render_job(ctx: Context, output_path: str) -> str:
        """Abort a running render job.

        Args:
            output_path: The output file path of the job to abort
                         (as shown in get_render_jobs).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.abort_render_job(output_path)
            if ok:
                return f"Abort signal sent for: {output_path}"
            return f"ERROR: Could not abort render job {output_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_playback_speed(ctx: Context, speed: float) -> str:
        """Set preview playback speed. Use to play at 2x, 0.5x, etc.

        Args:
            speed: Speed multiplier. 1.0 = normal, 2.0 = 2x forward,
                   -1.0 = 1x rewind, 0.5 = half speed. 0 = toggle play/pause.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_playback_speed(speed)
            if not ok:
                return f"ERROR: Could not set playback speed {speed}x"
            if speed == 0:
                return "Toggled play/pause"
            return f"Playback speed set to {speed}x"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def load_project(ctx: Context, file_path: str) -> str:
        """Load a Wunjo Make project file.

        Args:
            file_path: Absolute path to .wmproj file.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            pm = resolve.GetProjectManager()
            proj = pm.LoadProject(file_path)
            if proj is None:
                return f"ERROR: Could not load {file_path}"
            fps = proj.GetFps()
            w, h = proj.GetResolution()
            name = proj.GetName()
            return f"Loaded '{name}' ({fps}fps, {w}x{h})"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_project_duration(ctx: Context) -> str:
        """Get total timeline duration."""
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            frames = int(resolve._app.get_project_duration())
            tc = helpers.format_tc(frames, fps)
            return f"Timeline duration: {tc} ({frames} frames)"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_project_color_space(ctx: Context) -> str:
        """Get project color space setting."""
        try:
            resolve = helpers.get_resolve(ctx)
            cs = resolve._app.get_project_color_space()
            return f"Color space: {cs}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_project_color_space(ctx: Context, color_space: str) -> str:
        """Set project color space.

        Args:
            color_space: Color space identifier (e.g. "709", "2020", "smpte240m").
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_project_color_space(color_space)
            if not ok:
                return f"ERROR: Could not set color space to {color_space}"
            return f"Color space set to {color_space}"
        except Exception as e:
            return f"ERROR: {e}"
