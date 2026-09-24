"""Checkpoint and undo/redo tools: state management for the project."""

from __future__ import annotations

import os
import time

from mcp.server.fastmcp import Context


# In-process checkpoint registry: {label: file_path}
_checkpoints: dict[str, str] = {}


def register(mcp, helpers):

    @mcp.tool()
    def checkpoint_save(ctx: Context, label: str = "") -> str:
        """Save a checkpoint of the current project state.

        Creates a timestamped copy of the project file that can be restored later.

        Args:
            label: Optional label for the checkpoint. Auto-generated if empty.
        """
        try:
            proj = helpers.get_project(ctx)
            orig_path = proj.GetProjectPath()
            if not orig_path:
                return "ERROR: Project has no path — save it first."

            if not label:
                label = f"ckpt-{int(time.time())}"

            base, ext = os.path.splitext(orig_path)
            ckpt_path = f"{base}__{label}{ext}"

            ok = proj.SaveAs(ckpt_path)
            if not ok:
                return f"ERROR: Could not save checkpoint to {ckpt_path}"

            # Re-save back to original path so we continue working on original
            proj.SaveAs(orig_path)

            _checkpoints[label] = ckpt_path
            return f"Checkpoint saved: {ckpt_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def checkpoint_restore(ctx: Context, label: str = "") -> str:
        """Restore a previously saved checkpoint.

        Args:
            label: Checkpoint label to restore. If empty, restores the most recent.
        """
        try:
            if not _checkpoints:
                return "ERROR: No checkpoints available."

            if label:
                ckpt_path = _checkpoints.get(label)
                if not ckpt_path:
                    available = ", ".join(_checkpoints.keys())
                    return f"ERROR: Checkpoint '{label}' not found. Available: {available}"
            else:
                # Most recent
                label = list(_checkpoints.keys())[-1]
                ckpt_path = _checkpoints[label]

            if not os.path.exists(ckpt_path):
                return f"ERROR: Checkpoint file missing: {ckpt_path}"

            resolve = helpers.get_resolve(ctx)
            pm = resolve.GetProjectManager()
            proj = pm.LoadProject(ckpt_path)
            if proj is None:
                return f"ERROR: Could not load checkpoint {ckpt_path}"

            return f"Restored checkpoint '{label}' from {ckpt_path}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def undo(ctx: Context, steps: int = 1) -> str:
        """Undo the last operation(s) in Wunjo Make.

        Args:
            steps: Number of operations to undo (default 1).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.undo(steps)
            if ok:
                status = resolve._app.undo_status()
                undo_text = status.get("undo_text", "")
                idx = status.get("index", "?")
                count = status.get("count", "?")
                msg = f"Undid {steps} operation(s). Stack position: {idx}/{count}."
                if undo_text:
                    msg += f" Next undo: \"{undo_text}\""
                return msg
            return "Nothing to undo."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def redo(ctx: Context, steps: int = 1) -> str:
        """Redo previously undone operation(s) in Wunjo Make.

        Args:
            steps: Number of operations to redo (default 1).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.redo(steps)
            if ok:
                status = resolve._app.undo_status()
                redo_text = status.get("redo_text", "")
                idx = status.get("index", "?")
                count = status.get("count", "?")
                msg = f"Redid {steps} operation(s). Stack position: {idx}/{count}."
                if redo_text:
                    msg += f" Next redo: \"{redo_text}\""
                return msg
            return "Nothing to redo."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def undo_status(ctx: Context) -> str:
        """Show current undo/redo status: what can be undone/redone and stack depth."""
        try:
            resolve = helpers.get_resolve(ctx)
            status = resolve._app.undo_status()
            if not status:
                return "No undo stack available."

            can_undo = status.get("can_undo", "false")
            can_redo = status.get("can_redo", "false")
            undo_text = status.get("undo_text", "")
            redo_text = status.get("redo_text", "")
            idx = status.get("index", "0")
            count = status.get("count", "0")

            lines = [f"Stack position: {idx}/{count}"]
            if can_undo == "true":
                lines.append(f"Can undo: \"{undo_text}\"")
            else:
                lines.append("Can undo: no")
            if can_redo == "true":
                lines.append(f"Can redo: \"{redo_text}\"")
            else:
                lines.append("Can redo: no")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"
