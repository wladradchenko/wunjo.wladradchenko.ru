# Plugin authoring guide

A Wunjo Make plugin adds one AI capability to the editor. It is a small Python
program plus a manifest that tells the editor what it operates on, how to run
it, and what it needs to be set up. This document is the complete contract.

## 1. Anatomy

```
<plugin-id>/
  plugin.json          # manifest (required)
  main.py              # entry point (required)
  requirements.txt     # pip deps, one per line (optional; empty ⇒ no venv)
  assets/              # optional bundled files (icons, small data)
  models/              # created on the user's machine; NOT shipped in the zip
```

`<plugin-id>` matches `^[a-z0-9][a-z0-9-]{1,63}$` and equals the manifest `id`.
Packed with `../pack.py` it becomes `<id>-<version>.wmplugin` — a plain zip with
this tree at its root, minus `models/`, `venv*`, and `__pycache__`.

Installed, the plugin lives at `<AppData>/plugins/<id>/`
(`~/.var/app/online.wunjo.make/data/wunjo/plugins/<id>/` in the flatpak). Its
private environment, when built, is a sibling `<AppData>/venv-<id>/`.

## 2. Manifest (`plugin.json`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `manifest_version` | int | yes | `1`. |
| `id` | string | yes | kebab-case; equals the folder name. |
| `name` | string | yes | Shown in menus and as the settings tab title. |
| `version` | string | yes | Semver. |
| `author` | string | no | Shown on the plugin's About window. |
| `author_email` | string | no | Adds a mail button beside the author. |
| `description` | string | no | One or two sentences. |
| `license` | string | no | SPDX id or free text; shown before install. Reserved hook for paid modules. |
| `kind` | enum | yes | `local` (models run here) or `api` (calls a paid HTTP provider). Drives key handling. |
| `venv` | enum | no | `private` (default; own `venv-<id>`) or `shared` (reuse the app's common `venv`). |
| `target` | enum | yes | `video` \| `audio` \| `face` \| `generator` — where it appears (§5). |
| `entry` | string | yes | Entry script relative to the root, usually `main.py`. |
| `python` | string | no | Minimum interpreter, e.g. `python3.10`. Informational. |
| `requirements` | string | no | Path to a pip requirements file. Empty/absent ⇒ runs on system Python, no venv. |
| `provider` | object | if `kind=api` | `{ "name", "key_setting", "signup_url" }`. `name` keys the stored key; `key_setting` is the env var the entry receives it in (defaults to `WUNJO_KEY_<NAME>`); `signup_url` is linked from the settings tab. |
| `models` | array | no | `[{ "name", "url", "sha256", "size_mb", "auto_download" }]`. Downloaded into `models/` on demand. |
| `hardware` | object | no | `{ "min_vram_gb", "cpu_ok" }`. Used by the feasibility check. |
| `os` | array | no | Subset of `["linux","windows","macos"]`. Install is refused on an OS not listed. |
| `params` | array | no | UI form spec (§6). |
| `input` | object | yes | `{ "clip": "video"\|"audio"\|"none", "multiple": bool, "zone": bool }`. |
| `result` | object | yes | `{ "type": "video"\|"audio"\|"image"\|"subtitle"\|"none", "place": "bin"\|"timeline"\|"replace-zone"\|"none" }`. |

The manifest rules are validated identically by `../pack.py` and by the editor's
importer; run `python ../pack.py --check <dir>` before shipping.

## 3. Entry-point contract

The editor launches:

```
<python> <entry> --job /abs/path/to/job.json
```

`<python>` is the plugin's venv interpreter when a venv is built, otherwise the
system `python3`. `job.json` is written by the editor:

```json
{
  "job_id": "3f9a1c",
  "plugin_id": "my-plugin",
  "input": {
    "clips": [ { "bin_id": "5", "path": "/abs/in.mp4", "in": 0, "out": 125 } ],
    "face":  { "rect": [0.41, 0.22, 0.18, 0.27], "position": 87 }
  },
  "params": { "strength": 0.5, "quality": "best" },
  "output_dir": "/abs/writable/work/3f9a1c",
  "project": { "fps": 25.0, "width": 1920, "height": 1080 },
  "ffmpeg": "/abs/ffmpeg"
}
```

- `input.clips` is a list (one element unless `input.multiple`); each entry has
  the source `path`, the bin `bin_id`, and the used `in`/`out` frames.
- `input.face` is present only for `target: "face"` — the normalized rectangle
  `[x, y, w, h]` in 0..1 and the source frame `position`.
- Write outputs into `output_dir`. Use `ffmpeg` for muxing if needed.

### stdout protocol (one directive per line)

```
progress:<0-100>          # optional, may repeat — drives the progress bar
info:<message>            # optional status line, surfaced to the user
need:<json>               # optional: abort asking for something, e.g.
                          #   need:{"kind":"api_key","provider":"runway"}
                          #   need:{"kind":"model","name":"weights.bin"}
result:<json>             # FINAL line on success (see below)
```

`result` payload:

```json
{ "outputs": [ { "type": "video", "path": "/abs/work/out.mp4" } ],
  "message": "optional human summary" }
```

Everything not matching a directive is ignored. **Use stderr for logs/tracebacks**
(captured to the job's `log.txt`). Flush stdout after each line.

Exit codes: `0` ok (must have printed `result:`), `2` bad input, `3` missing
key, `4` missing model, `5` out of memory, `1` any other error.

## 4. Environments (venv) and dependencies

- A `local` plugin with a non-empty `requirements.txt` gets a **private venv**
  by default (`venv-<id>`). The user creates it from the plugin's settings
  tab ("Install"), which runs the app's pip broker to install the requirements;
  if some are missing later the tab offers to add them. The tab shows the
  environment size and an "Uninstall plugin" button that removes just the venv.
- `venv: shared` reuses the app's common `venv` (the one Whisper/Vosk use). Only
  choose it for pure-python, conflict-free deps you have vetted — a heavy or
  version-pinned dependency must stay private.
- The first line of `requirements.txt` may pin acceptable interpreters:
  `#python3.10,python3.11,python3.12`. Other `#` lines are comments.
- Do **not** install anything from inside `main.py`. Dependencies come only
  through the declared `requirements.txt` and the app's installer.

## 5. Plugin types (where each appears, what it receives)

| `target` | Location in the UI | `input.clip` | Receives |
|---|---|---|---|
| `video` | Right-click a timeline clip ▸ **Artificial Intelligence** submenu (hidden on audio-only clips) | `video` | `input.clips` (all selected video clips if `multiple: true`). |
| `audio` | Same submenu, only on clips that carry sound | `audio` | `input.clips` (audio clips). |
| `face` | Menu that opens from a **detected-face box** in the monitor | `video` | one clip + `input.face` (rect + frame). |
| `generator` | **Media ▸ Generate with Artificial Intelligence** and the bin context menu | `none` | no clip; produces media from `params` alone. |

`result.place` tells the editor what to do with the produced file(s): `bin`
imports them into the project bin; `timeline` also inserts them at the zone;
`replace-zone` swaps the source zone; `none` means the plugin reports success
without returning media (the stubs use this).

## 6. Parameters

Each `params` entry renders one form row on the plugin's settings tab and is
passed to the plugin in `job.json → params`:

```json
{ "key": "quality", "type": "enum", "label": "Quality",
  "options": ["fast", "balanced", "best"], "default": "balanced" }
```

Types: `string` (QLineEdit), `number` (`min`/`max`/`step` → spin box), `bool`
(checkbox), `enum` (`options` → combo), `file` (`filter` → path picker). Values
persist per plugin and are read back into the tab.

## 7. Models

Declare weights in `models`; they are **not** packed. On the settings tab each
model shows a Download button (into `models/`), plus "Open" and "Delete all
models". At run time, resolve a model relative to your script:
`os.path.join(os.path.dirname(__file__), "models", "weights.bin")`. If a needed
model is absent, emit `need:{"kind":"model","name":"weights.bin"}` and exit `4`.

## 8. API keys (`kind: api`)

The key is stored by the editor, never by the plugin. It reaches `main.py` only
as the environment variable named by `provider.key_setting` (default
`WUNJO_KEY_<PROVIDER-UPPERCASED>`). Read it with `os.environ.get(...)`; if empty,
emit `need:{"kind":"api_key","provider":"<name>"}` and exit `3`. Never echo the
key, never write it to disk, never put it in `result`/`info`.
