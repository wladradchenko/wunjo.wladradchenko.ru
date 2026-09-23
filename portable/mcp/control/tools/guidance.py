"""Assistant guidance: skills (how to work) and loops (pipeline scenarios).

A skill is a short note on how to do one part of the job well — where to cut
spoken footage, how to keep subtitles readable, what makes a grade look
processed. The editor ships a library of them (editing craft, so that an agent
driving the timeline produces an edit rather than an assembly), and the user
adds their own; several can be active per project. A loop is a start-to-finish
scenario (script -> characters -> locations -> video -> edit -> voiceover ...)
naming which tools and plugins to use at every step; at most one per project.

The library and the selection live in the editor (same storage the chat UI
uses), reached over its local scripting socket, so everything here is a thin
wrapper.

AT SESSION START:
  1. call get_selected_skills and get_selected_loop — what the user pinned for
     this project outranks your own judgement;
  2. call list_skills and read (get_skill) the ones whose description fits the
     task in front of you. The library is there to be used, not only the pinned
     subset; reading three relevant skills before an edit is the difference
     between a competent edit and a mechanical one.
"""

from __future__ import annotations

from mcp.server.fastmcp import Context

_MAX_DESCRIPTION = 160


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into its metadata and its body.

    Documents may open with a small YAML-ish header:

        ---
        name: editing-cuts
        description: one line the agent judges relevance by
        ---

    A document without one is still valid — the user writes these by hand in
    the chat panel and must not be punished for a missing header.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            meta: dict[str, str] = {}
            for line in lines[1:index]:
                key, separator, value = line.partition(":")
                if separator:
                    meta[key.strip().lower()] = value.strip()
            return meta, "\n".join(lines[index + 1:]).lstrip("\n")
    return {}, text  # unterminated header: treat the whole thing as body


def _body(text: str) -> str:
    """The document without its metadata header — what the model should read."""
    return _split_front_matter(text)[1]


def _titled(label: str, name: str, text: str) -> str:
    """One document under a heading that names it as the library does.

    The document's own title goes: the agent needs the library name to talk
    about the skill or save it back, and two headings in a row is noise.
    """
    body = _body(text).lstrip()
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        body = "\n".join(lines[1:]).lstrip("\n")
    return f"# {label}: {name}\n\n{body}"


def _description(text: str) -> str:
    """One line saying what this document is for, for the library listing.

    Falls back to the first line of prose when there is no header, so a
    hand-written note still says something about itself in the table.
    """
    meta, body = _split_front_matter(text)
    description = meta.get("description", "")
    if not description:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped
                break
    description = " ".join(description.split()).replace("|", r"\|")
    if len(description) > _MAX_DESCRIPTION:
        description = description[: _MAX_DESCRIPTION - 1].rstrip() + "…"
    return description


def register(mcp, helpers):

    def _app(ctx: Context):
        return helpers.get_resolve(ctx)._app

    def _library(app, call_list: str, call_get: str) -> list[tuple[str, str]]:
        """Every document's name and description (one read per document)."""
        names = list(app._call(call_list) or [])
        return [(name, _description(app._call(call_get, name) or "")) for name in names]

    # ── Skills ──────────────────────────────────────────────────────────

    @mcp.tool()
    def list_skills(ctx: Context) -> str:
        """The skill library: what each one is for, and which are pinned to this project.

        Call this at session start. Then read (get_skill) the ones whose
        description matches the work — an edit, subtitles, a grade, narration.
        Pinned skills are the user's standing instructions; the rest are craft
        you should reach for yourself.
        """
        try:
            app = _app(ctx)
            selected = set(app._call("scriptGetSelectedSkills") or [])
            library = _library(app, "scriptListSkills", "scriptGetSkill")
            if not library:
                return "No skills in the library."
            lines = ["| skill | what it is for | pinned |", "|-------|----------------|--------|"]
            lines += [f"| {name} | {about} | {'yes' if name in selected else 'no'} |" for name, about in library]
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def get_skill(ctx: Context, name: str) -> str:
        """Read one skill's full text. Do this for every skill relevant to the task.

        Args:
            name: Skill name from list_skills.
        """
        try:
            content = _app(ctx)._call("scriptGetSkill", name)
            if not content:
                return f"ERROR: skill '{name}' not found or empty."
            return _titled("Skill", name, content)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def save_skill(ctx: Context, name: str, content: str) -> str:
        """Create or overwrite a skill in the library.

        Open with a header so the library stays browsable:

            ---
            name: <same as the name argument>
            description: <one line: when to use this>
            ---

        Editing a skill the editor ships keeps your version and leaves the
        original intact underneath.

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

        A skill the editor ships is hidden rather than erased; saving one under
        the same name brings it back.

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
        """Full text of every skill pinned to this project (call at session start)."""
        try:
            app = _app(ctx)
            names = list(app._call("scriptGetSelectedSkills") or [])
            if not names:
                return ("No skills pinned to this project. Call list_skills and read the ones "
                        "the task needs.")
            parts = [_titled("Skill", n, app._call("scriptGetSkill", n) or "") for n in names]
            return "\n\n---\n\n".join(parts)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    # ── Loops ───────────────────────────────────────────────────────────

    @mcp.tool()
    def list_loops(ctx: Context) -> str:
        """The loop library: what each scenario produces, and which one this project uses."""
        try:
            app = _app(ctx)
            selected = app._call("scriptGetSelectedLoop") or ""
            library = _library(app, "scriptListLoops", "scriptGetLoop")
            if not library:
                return "No loops in the library."
            lines = ["| loop | what it produces | selected |", "|------|------------------|----------|"]
            lines += [f"| {name} | {about} | {'yes' if name == selected else 'no'} |" for name, about in library]
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
            if not content:
                return f"ERROR: loop '{name}' not found or empty."
            return _titled("Loop", name, content)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def save_loop(ctx: Context, name: str, content: str) -> str:
        """Create or overwrite a loop scenario in the library.

        Open with the same header as a skill (`name`, `description`) so the
        library listing says what the loop produces.

        Args:
            name: Loop name (also the file name; no slashes).
            content: Full scenario text (markdown), numbered steps naming the
                tools and plugins to use.
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
            return _titled("Loop", name, app._call("scriptGetLoop", name) or "")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
