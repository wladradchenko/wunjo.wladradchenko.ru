# MCP server

Gives an AI agent full NLE control over a running Wunjo Make instance over its
local scripting socket:
import media, build timelines, add transitions, markers, effects, and render.

Two packages live here: `api/` — the client, shaped like the DaVinci Resolve
API — and `control/`, the MCP server built on top of it.

**This project does not use D-Bus.** The editor serves its scriptable methods on
a `QLocalServer` (`online.wunjo.make.scripting` under `XDG_RUNTIME_DIR`; a named
pipe on Windows), one JSON object per line. `api/app_client.py` speaks it with
nothing but the standard library — no `pydbus`, no `dbus-send`, and the same
code path on Linux, Windows and macOS. Set `WUNJO_SOCKET` to an absolute path to
talk to one particular instance.

## Quick start

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "wunjo-make": {
      "command": "python",
      "args": ["-m", "control"]
    }
  }
}
```

The app writes this file for you: Chat panel ▸ hand the keys to an outside
agent. On a flatpak install it names `share/wunjo/mcp/start` instead, which
re-enters the sandbox where the environment actually lives.

## Requirements

- Python 3.10+
- [MCP SDK](https://pypi.org/project/mcp/) (`mcp>=1.0,<2`)
- A running Wunjo Make (it opens the scripting socket at startup)

```bash
pip install -r requirements.txt
```

## Tools

### Composite (use these first)

| Tool | Description |
|------|-------------|
| `build_timeline` | Full assembly from scene clips (import + sequence + transitions + audio + markers) |
| `replace_scene` | Swap one scene clip by number, keep position and transitions |
| `get_timeline_summary` | Text table of all clips on timeline |
| `add_transitions_batch` | Batch cross-dissolves between all clips on a track |
| `render_video` | Export timeline to video file |

### Atomic

| Domain | Tools |
|--------|-------|
| Project | `get_project_info`, `save_project`, `load_project` |
| Media | `get_media_pool`, `import_media`, `import_media_glob`, `create_bin_folder` |
| Timeline | `get_track_list`, `get_clip_info`, `insert_clip`, `append_clips`, `move_clip`, `delete_clip`, `add_track`, `trim_clip` |
| Transitions | `add_transition`, `remove_transition` |
| Markers | `get_markers`, `add_marker`, `delete_marker`, `delete_markers_by_color` |
| Replace | `replace_clip` |
| Checkpoints | `checkpoint_save`, `checkpoint_restore` |

