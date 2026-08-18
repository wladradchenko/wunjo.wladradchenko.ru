"""Group tools: create, ungroup, info, remove from group."""

from __future__ import annotations

from mcp.server.fastmcp import Context


def register(mcp, helpers):

    @mcp.tool()
    def group_clips(ctx: Context, item_ids: list[int]) -> str:
        """Group timeline items (clips and/or compositions) together.

        Grouped items move, copy, and delete as a unit.
        Requires at least 2 items.

        Args:
            item_ids: List of timeline item IDs to group.
        """
        try:
            if len(item_ids) < 2:
                return "ERROR: At least 2 items are required to create a group."
            resolve = helpers.get_resolve(ctx)
            gid = resolve._app.group_clips(item_ids)
            if gid == -1:
                return "ERROR: Could not create group (check that all IDs are valid timeline items)."
            return f"Created group {gid} with {len(item_ids)} items: {item_ids}"
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def ungroup_clips(ctx: Context, item_id: int) -> str:
        """Ungroup the topmost group containing the given item.

        All members are released. The group is destroyed.

        Args:
            item_id: Any timeline item ID that belongs to a group.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.ungroup_clips(item_id)
            if not ok:
                return f"ERROR: Could not ungroup item {item_id} (item may not be in a group)."
            return f"Ungrouped the group containing item {item_id}."
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def get_group_info(ctx: Context, item_id: int) -> str:
        """Get group information for a timeline item.

        Shows whether the item is grouped, the group type, root group ID,
        and a table of all member items with their tracks and positions.

        Args:
            item_id: Timeline item ID (clip, composition, or group).
        """
        try:
            resolve = helpers.get_resolve(ctx)
            fps = helpers.get_fps(ctx)
            info = resolve._app.get_group_info(item_id)
            if not info:
                return f"ERROR: Item {item_id} not found on timeline."

            in_group = info.get("isInGroup", False)
            is_group = info.get("isGroup", False)
            if isinstance(in_group, str):
                in_group = in_group.lower() == "true"
            if isinstance(is_group, str):
                is_group = is_group.lower() == "true"

            root_id = info.get("rootId", item_id)
            group_type = info.get("groupType", "Leaf")

            lines = [
                f"**Group info for item {item_id}**",
                f"- in group: {'yes' if in_group else 'no'}",
                f"- is group node: {'yes' if is_group else 'no'}",
                f"- root group ID: {root_id}",
                f"- group type: {group_type}",
            ]

            members = info.get("members", [])
            if members:
                lines.append(f"- members ({len(members)}):")
                lines.append("")
                lines.append("| id | type | track_id | position |")
                lines.append("|----|------|----------|----------|")
                for m in members:
                    # Normalize: D-Bus may return dicts, lists-of-tuples, or raw dicts
                    if isinstance(m, (list, tuple)):
                        m = dict(m) if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in m) else {}
                    if isinstance(m, dict):
                        mid = m.get("id", "?")
                        mtype = m.get("type", "?")
                        tid = m.get("trackId", "?")
                        pos = m.get("position", 0)
                        pos_tc = helpers.format_tc(int(pos), fps) if str(pos).lstrip('-').isdigit() else "?"
                        lines.append(f"| {mid} | {mtype} | {tid} | {pos_tc} (frame {pos}) |")

            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    @mcp.tool()
    def remove_from_group(ctx: Context, item_id: int) -> str:
        """Remove a single item from its group, keeping the remaining members grouped.

        If only one other member remains after removal, the group is dissolved.

        Args:
            item_id: Timeline item ID to remove from its group.
        """
        try:
            resolve = helpers.get_resolve(ctx)
            ok = resolve._app.remove_from_group(item_id)
            if not ok:
                return f"ERROR: Could not remove item {item_id} from group (item may not be in a group)."
            return f"Removed item {item_id} from its group."
        except Exception as e:
            return f"ERROR: {e}"
