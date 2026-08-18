"""Proxy clip tools: status, enable/disable, delete, rebuild."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def get_clip_proxy_status(ctx: Context, bin_id: str) -> str:
        """Get proxy status for a media pool clip.

        Args:
            bin_id: Media pool clip ID (from get_media_pool).

        Returns proxy info: supportsProxy, hasProxy, proxyPath, isGenerating.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            status = resolve._app.get_clip_proxy_status(bin_id)
            if not status:
                return f"ERROR: Clip {bin_id} not found."
            supports = status.get("supportsProxy", False)
            has = status.get("hasProxy", False)
            path = status.get("proxyPath", "")
            orig = status.get("originalUrl", "")
            generating = status.get("isGenerating", False)

            # Normalize D-Bus string booleans
            if isinstance(supports, str):
                supports = supports.lower() == "true"
            if isinstance(has, str):
                has = has.lower() == "true"
            if isinstance(generating, str):
                generating = generating.lower() == "true"

            lines = [
                f"**Proxy status for clip {bin_id}**",
                f"- supports proxy: {'yes' if supports else 'no'}",
                f"- has proxy: {'yes' if has else 'no'}",
                f"- proxy path: {path if path and path != '-' else '(none)'}",
                f"- original url: {orig if orig else '(none)'}",
                f"- generating: {'yes' if generating else 'no'}",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def set_clip_proxy(ctx: Context, bin_id: str, enabled: bool) -> str:
        """Enable or disable proxy for a media pool clip.

        Args:
            bin_id: Media pool clip ID.
            enabled: True to enable proxy, False to disable.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.set_clip_proxy(bin_id, enabled)
            if not ok:
                action = "enable" if enabled else "disable"
                return f"ERROR: Could not {action} proxy for clip {bin_id} (clip may not support proxies)."
            action = "Enabled" if enabled else "Disabled"
            return f"{action} proxy for clip {bin_id}."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def delete_clip_proxy(ctx: Context, bin_id: str) -> str:
        """Delete proxy file and disable proxy for a media pool clip.

        Args:
            bin_id: Media pool clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.delete_clip_proxy(bin_id)
            if not ok:
                return f"ERROR: Could not delete proxy for clip {bin_id}."
            return f"Deleted proxy for clip {bin_id}."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def rebuild_clip_proxy(ctx: Context, bin_id: str) -> str:
        """Force regenerate proxy for a media pool clip.

        Triggers proxy generation even if a proxy already exists.

        Args:
            bin_id: Media pool clip ID.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.rebuild_clip_proxy(bin_id)
            if not ok:
                return f"ERROR: Could not rebuild proxy for clip {bin_id} (clip may not support proxies)."
            return f"Rebuilding proxy for clip {bin_id} (generation runs in background)."
        except Exception as e:
            return f"ERROR: {e}"
