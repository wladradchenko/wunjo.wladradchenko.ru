"""Assistant guidance: skills (how to work) and loops (pipeline scenarios).

Skills are reusable notes the user writes; several can be selected per
project. A loop is a start-to-finish scenario (script -> characters ->
locations -> video -> edit -> voiceover ...) that names which tools and
plugins to use at every step; at most one loop is selected per project.

The library and selection live in the editor (same storage the chat UI
uses), so everything here is a thin D-Bus wrapper.

AT SESSION START call get_selected_skills and get_selected_loop and follow
what they return; when both are empty, work without extra guidance.
"""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    def _app(ctx: Context):
        return helpers.get_resolve(ctx)._app

    # ── Skills ──────────────────────────────────────────────────────────

    @mcp.tool()
    def list_skills(ctx: Context) -> str:
        """List all skills in the library, marking the ones selected for this project."""
        try:
            app = _app(ctx)
            names = list(app._call("scriptListSkills") or [])
            selected = set(app._call("scriptGetSelectedSkills") or [])
            if not names:
                return "No skills in the library."
            lines = ["| skill | selected |", "|-------|----------|"]
            lines += [f"| {n} | {'yes' if n in selected else 'no'} |" for n in names]
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_skill(ctx: Context, name: str) -> str:
        """Read one skill's full text.

        Args:
            name: Skill name from list_skills.
        """
        try:
            content = _app(ctx)._call("scriptGetSkill", name)
            return content or f"ERROR: skill '{name}' not found or empty."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def save_skill(ctx: Context, name: str, content: str) -> str:
        """Create or overwrite a skill in the library.

        Args:
            name: Skill name (also the file name; no slashes).
            content: Full skill text (markdown).
        """
        try:
            ok = _app(ctx)._call("scriptSaveSkill", name, content)
            return f"Saved skill '{name}'." if ok else f"ERROR: could not save skill '{name}'."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_skill(ctx: Context, name: str) -> str:
        """Delete a skill from the library (deselects it everywhere).

        Args:
            name: Skill name.
        """
        try:
            ok = _app(ctx)._call("scriptDeleteSkill", name)
            return f"Deleted skill '{name}'." if ok else f"ERROR: skill '{name}' not found."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def select_skills(ctx: Context, names: list[str]) -> str:
        """Set which skills are active for the current project (replaces the set).

        Args:
            names: Skill names to activate; empty list deactivates all.
        """
        try:
            ok = _app(ctx)._call("scriptSetSelectedSkills", names)
            return f"Selected skills: {', '.join(names) if names else '(none)'}" if ok \
                else "ERROR: selection rejected (unknown skill name?)."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_selected_skills(ctx: Context) -> str:
        """Full text of every skill selected for this project (call at session start)."""
        try:
            app = _app(ctx)
            names = list(app._call("scriptGetSelectedSkills") or [])
            if not names:
                return "No skills selected — work without extra guidance."
            parts = [f"# Skill: {n}\n\n{app._call('scriptGetSkill', n)}" for n in names]
            return "\n\n---\n\n".join(parts)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    # ── Loops ───────────────────────────────────────────────────────────

    @mcp.tool()
    def list_loops(ctx: Context) -> str:
        """List all loops in the library, marking the one selected for this project."""
        try:
            app = _app(ctx)
            names = list(app._call("scriptListLoops") or [])
            selected = app._call("scriptGetSelectedLoop") or ""
            if not names:
                return "No loops in the library."
            lines = ["| loop | selected |", "|------|----------|"]
            lines += [f"| {n} | {'yes' if n == selected else 'no'} |" for n in names]
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_loop(ctx: Context, name: str) -> str:
        """Read one loop's full scenario text.

        Args:
            name: Loop name from list_loops.
        """
        try:
            content = _app(ctx)._call("scriptGetLoop", name)
            return content or f"ERROR: loop '{name}' not found or empty."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def save_loop(ctx: Context, name: str, content: str) -> str:
        """Create or overwrite a loop scenario in the library.

        Args:
            name: Loop name (also the file name; no slashes).
            content: The scenario: numbered steps from source material to the
                finished product, naming the tools/plugins for each step.
        """
        try:
            ok = _app(ctx)._call("scriptSaveLoop", name, content)
            return f"Saved loop '{name}'." if ok else f"ERROR: could not save loop '{name}'."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_loop(ctx: Context, name: str) -> str:
        """Delete a loop from the library (deselects it if selected).

        Args:
            name: Loop name.
        """
        try:
            ok = _app(ctx)._call("scriptDeleteLoop", name)
            return f"Deleted loop '{name}'." if ok else f"ERROR: loop '{name}' not found."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def select_loop(ctx: Context, name: str = "") -> str:
        """Select the loop for the current project (one at most).

        Args:
            name: Loop name; empty string clears the selection.
        """
        try:
            ok = _app(ctx)._call("scriptSelectLoop", name)
            return (f"Selected loop '{name}'." if name else "Loop selection cleared.") if ok \
                else f"ERROR: loop '{name}' not found."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_selected_loop(ctx: Context) -> str:
        """Full scenario of this project's selected loop (call at session start).

        Follow it step by step when present; report which step you are on."""
        try:
            app = _app(ctx)
            name = app._call("scriptGetSelectedLoop") or ""
            if not name:
                return "No loop selected — no pipeline scenario for this project."
            return f"# Loop: {name}\n\n{app._call('scriptGetLoop', name)}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
