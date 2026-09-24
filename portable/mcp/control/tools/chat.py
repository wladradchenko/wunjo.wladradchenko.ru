"""Narrate into Wunjo Make's chat dock so the user sees what you are doing.

The user talks to you in their own editor/agent; these tools MIRROR that into
the app's Chat panel: the user's request, your replies, a "thinking" state and
live progress cards for long work (plugins, whisper, renders). This is purely
cosmetic feedback — it never blocks editing — but it is what makes the assistant
feel present in the app.

USE IT PROACTIVELY, in the USER'S LANGUAGE:
  - chat_user with the user's request when you start acting on it;
  - chat_thinking(True) while you reason, chat_thinking(False) when done;
  - chat_assistant for your reply (chat_assistant_stream to update it live);
  - chat_tool_start/progress/end around any long op (plugin, whisper, render)
    so a progress card shows "running… / waiting…".
"""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    def _app(ctx: Context):
        return helpers.get_resolve(ctx)._app

    @mcp.tool()
    def chat_user(ctx: Context, text: str) -> str:
        """Mirror the user's request into the chat as a user bubble.

        Args:
            text: The user's request, in their language.
        """
        try:
            _app(ctx)._call("scriptChatMessage", 0, text)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_assistant(ctx: Context, text: str) -> str:
        """Add your reply to the chat as an assistant bubble (user's language).

        Args:
            text: Your message to the user.
        """
        try:
            _app(ctx)._call("scriptChatMessage", 1, text)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_assistant_stream(ctx: Context, text: str) -> str:
        """Replace the last assistant bubble with @p text (live/streaming update).

        Call repeatedly with the growing text to update one bubble in place.

        Args:
            text: The full assistant text so far.
        """
        try:
            _app(ctx)._call("scriptChatStream", text)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_thinking(ctx: Context, on: bool = True) -> str:
        """Show/hide the "assistant is thinking…" indicator.

        Args:
            on: True while you reason, False when you start replying/acting.
        """
        try:
            _app(ctx)._call("scriptChatThinking", on)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_tool_start(ctx: Context, id: str, name: str) -> str:
        """Add a progress card for a long operation (plugin, whisper, render).

        Args:
            id: Your own id for this card (reuse it in progress/end).
            name: Short label shown on the card, in the user's language.
        """
        try:
            _app(ctx)._call("scriptChatToolStart", id, name)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_tool_progress(ctx: Context, id: str, percent: int = -1, message: str = "") -> str:
        """Update a progress card (percent and/or status text).

        Args:
            id: The card id from chat_tool_start.
            percent: 0..100, or -1 to keep the current value (indeterminate).
            message: Optional status line, in the user's language.
        """
        try:
            _app(ctx)._call("scriptChatToolProgress", id, percent, message)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def chat_tool_end(ctx: Context, id: str, is_error: bool = False, result: str = "") -> str:
        """Mark a progress card done (or failed) with a short result line.

        Args:
            id: The card id from chat_tool_start.
            is_error: True if the operation failed.
            result: Short outcome text, in the user's language.
        """
        try:
            _app(ctx)._call("scriptChatToolEnd", id, is_error, result)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
