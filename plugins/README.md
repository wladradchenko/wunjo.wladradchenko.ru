# Plugin format (`.wmplugin`)

A Wunjo Make plugin is a small Python program plus a manifest that tells the
editor **what it operates on**, **how to run it**, and **what it needs**. Plugins
live outside the free application and are loaded by the user (from a folder or a
`.wmplugin` archive) or, later, written by the built-in assistant.

This directory holds the reference plugins and the packer. Treat `plugin.json`
here as the source of truth for the format is the desktop software validates against
exactly these fields.

## Directory layout

```
<plugin-id>/
  plugin.json          # manifest (required, see below)
  main.py              # entry point (required)
  requirements.txt     # pip deps, one per line (optional; empty ⇒ no venv)
  effects/             # optional effect XMLs the plugin adds to the editor
  assets/              # optional bundled files
  models/              # created on the user machine; NEVER shipped in the archive
```

Packaged, that becomes `<plugin-id>-<version>.wmplugin` is a plain ZIP with the
same tree at its root. Model weights are **not** packed; they are downloaded on
install from the `models` URLs (keeps archives small and licensing clean).

## File `plugin.json`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `manifest_version` | int | yes | Format version. Currently `1`. |
| `id` | string | yes | `^[a-z0-9][a-z0-9-]{1,63}$`. Must equal the folder name. |
| `name` | string | yes | Human title shown in menus and the importer. |
| `version` | string | yes | Semver, e.g. `1.0.0`. |
| `author` | string | no | Displayed in the importer and on the plugin's About window. |
| `author_email` | string | no | Puts a "write to the author" button on that window. |
| `description` | string | no | One or two sentences. |
| `license` | string | no | SPDX id or free text (e.g. `GPL-3.0-only`, `Proprietary`). Shown before import. |
| `homepage` | string | no | URL shown in the importer. |
| `kind` | enum | yes | `api` (calls a paid HTTP provider) or `local` (runs models on this machine). Drives key handling; the environment is chosen by `venv` below. |
| `venv` | enum | no | `private` (**default**) ⇒ the plugin gets its own environment `venv-<id>`, isolated from everything else. `shared` ⇒ it reuses the global `venv`. Keep the default; pick `shared` only for light, conflict-free deps you have vetted — populating the global venv is a rare, manual act, not something a downloaded plugin should assume. |
| `target` | enum | yes | `video`, `audio`, `face`, `generator`, or `agent` — decides where the plugin appears (see below). |
| `entry` | string | yes | Entry script relative to the plugin root. |
| `python` | string | no | Minimum interpreter, e.g. `python3.10`. Informational for now. |
| `requirements` | string | no | Path to a pip requirements file. Empty/absent ⇒ no environment is built and the plugin runs on the system Python (regardless of `venv`). |
| `provider` | object | if `kind=api` | `{ "name", "key_setting", "signup_url" }`. `name` keys the stored API key; the key reaches the plugin only as env `WUNJO_KEY_<NAME>` (upper-cased), never on the command line, never in the model context. |
| `models` | array | no | `[{ "name", "url", "sha256", "size_mb", "auto_download" }]`. Downloaded into `models/` on install. A weight may also declare `unpack` (`zip` or `tar.gz` — the download is an archive, extracted into `models/<name>/` with the executable bit restored) and, when it comes in variants, which machine it is for: `platform` (`linux-x64`, `windows-x64`, `macos-arm64`), `backend` (`cuda`, `vulkan`, `cpu`) and `min_vram_gb` / `max_vram_gb`. Only the variants this machine can run are listed and downloaded, so one entry per name may appear several times. |
| `hardware` | object | no | `{ "min_vram_gb", "cpu_ok" }`. Used later by the feasibility check. |
| `os` | array | no | Subset of `["linux","windows","macos"]`. Import is refused on an unsupported OS. |
| `min_app_version` | string | no | Oldest Wunjo Make this plugin works with, e.g. `"3.1"` (see *Application version range*). |
| `max_app_version` | string | no | Newest Wunjo Make this plugin works with. |
| `params` | array | no | UI form spec (see *Parameters*). |
| `effects` | array | no | Paths of effect XMLs the plugin adds to the effect list (see *Effects*). |
| `input` | object | yes | What the plugin consumes (see *Targets*). |
| `result` | object | yes | `{ "type": video\|audio\|image\|subtitle\|none, "place": bin\|timeline\|replace-zone\|none }`. |

### Targets

| `target` | Appears in | `input.clip` | Notes |
|---|---|---|---|
| `video` | Right-click a timeline clip → **Artificial Intelligence** submenu | `video` | Set `input.multiple: true` to run over every selected video clip at once. |
| `audio` | Same submenu, but only on clips that carry sound | `audio` | |
| `face` | The little menu that pops from a **detected-face box** in the monitor | `video` | Receives the face rectangle + frame position in the job (see below). |
| `generator` | **Media ▸ Generate with Artificial Intelligence** (and the bin context menu) | `none` | Creates media from nothing (text→audio/image/video). |
| `agent` | The **Chat** panel, as a way of talking | `none` | Drives the whole editor instead of processing a clip. Cannot be combined with another target. |

### Assistant plugins (`target: agent`)

An assistant plugin is what answers in the Chat panel is the "MCP control" mark.
It is handed one message at a time through `input.action`:

```json
{ "input": { "action": "chat", "message": "cut this on the beats", "session": "…" } }
{ "input": { "action": "stop" } }
```

`stop` arrives when the user switches to another assistant: let go of whatever
is holding memory. It also gets three variables in its environment:
`WUNJO_MCP_DIR` (the MCP server that speaks to the running editor is the same one
an outside agent uses), `WUNJO_MODELS_DIR` (where its weights were downloaded —
a plugin that ships with the app runs from a read-only place and cannot assume
they sit beside it) and `WUNJO_GPU_BACKEND` (`cuda`, `vulkan` or `cpu`).

The reply does **not** come back through `result:`. An assistant writes into the
chat the way any agent does, over the editor's own tools — `chat_assistant`,
`chat_thinking`, `chat_tool_start/progress/end`, so the panel cannot tell an
assistant running here from Claude Code running in a terminal. `result:` only
says the turn is over.

Two rules the editor enforces rather than trusting: an assistant plugin is never
offered to a model through `list_plugins`/`run_plugin` (it would call itself),
and it never appears in a clip menu.

### Parameters (auto-generated dialog)

Each entry renders one form row before the plugin runs:

```json
{ "key": "quality", "type": "enum", "label": "Quality",
  "options": ["fast", "best"], "default": "fast" }
```

Types: `string`, `number` (`min`/`max`/`step`), `bool`, `enum` (`options`),
`file` (`filter`). Values are passed to the plugin in `job.json → params`.

Use `params` for install-wide settings (device, quality, an API model name).
Anything the user tunes **per clip** belongs in an effect instead — there it is
keyframable on the effect timeline.

### A plugin can bring its own

```json
"effects": ["effects/liveportrait.xml"]
```

Each entry is a normal Wunjo effect XML shipped inside the plugin folder. While
the plugin is installed the editor keeps a copy in its effects folder
(`<AppData>/effects/<effect-id>.xml`, stamped with the owning plugin id), so the
effect behaves like any other one — drag it on a clip, keyframe its parameters,
save it in the project. Uninstalling the plugin deletes those copies; the
effects the user saved themselves live in the same folder and are never touched.

Rules the packer and the importer enforce:

* the file must sit inside the plugin folder and hold a single `<effect>` root;
* `id` must be the plugin id or start with `<plugin-id>.` — a plugin cannot
  shadow a built-in effect, and a project that outlives the plugin still shows
  where the effect came from;
* `tag` names the MLT service the effect is built on, and that service must
  exist on the user machine or the effect is skipped.

Plugin effects are listed under **Plugins** in the effect list, and cannot be
edited or deleted from there — they belong to the plugin.

**A plugin that brings an effect works through it.** Its menu entry (the clip
▸ *Artificial Intelligence* submenu, or the detected-face box for `target: face`)
adds the effect instead of launching the entry script: the user sets it up and
keyframes it on the clip, and the plugin renders from those values later. Only a
plugin without effects runs its script straight from the menu.

For a `face` plugin, mark the parameter that holds the face with
`wunjo_fill="face"` and the editor fills it with that face's track (the same
animated rectangle Hide Face uses) when the effect is applied from a face box:

```xml
<parameter type="animatedrect" name="lp_face" default="50% 50% 25% 25%"
           opacity="false" wunjo_fill="face">
    <name>Face</name>
</parameter>
```

#### A region and what works inside it

Keep "where it happens" apart from "what happens". An effect that declares

```xml
<effect … id="liveportrait" wunjo_requires="liveportrait.region">
    <parameter type="readonly" name="lp_region" wunjo_fill="region"><name>Head region</name></parameter>
```

is applied together with `liveportrait.region`: the region effect gets the face
track, both get the same id in their `wunjo_fill="region"` parameter, and each
keeps its own keyframe timeline — the region can be corrected without touching
the animation. The required effect is never offered on its own in the menus (it
stays in the effect list for manual use), and it must be shipped by the same
plugin.

#### Recorded sets

A parameter declared as

```xml
<parameter type="urllist" paramlist="%pluginSets" name="lp_set">
    <name>Expression source</name>
</parameter>
```

behaves like Shape Alpha's resource: the list offers the sets recorded for this
plugin in the current project, and the pen next to it opens the panel where a
video or a photo is analysed into a new one. Sets are stored in
`<projectDataFolder>/plugin-sets/<plugin-id>/<name>.json` — they travel with the
project, and the panel imports and exports them to move one between projects
instead of analysing the same performance twice.

The editor asks the plugin for a set by running it with

```json
"input": { "action": "analyse", "source": "/abs/driving.mp4" }
```

and expects a json output whose `values` hold one value per frame, keyed by the
effect's parameter names:

```json
{"source": "/abs/driving.mp4", "fps": 25.0, "count": 137,
 "values": {"lp_pitch": [0.0, 0.4, …], "lp_yaw": [0.0, -0.2, …]}}
```

Picking that set fills those parameters with keyframes (points are kept only
where a value moves, so a long recording cannot choke the keyframe model), cut
to the clip's length after a warning. Every keyframe stays editable afterwards.

A model that cannot run while MLT plays (LivePortrait, diffusion, …) still gets
an effect: build it on a neutral service (e.g. `brightness` with a fixed
`level=1`, which leaves the image untouched) and use it to hold the parameters
and keyframes the plugin reads when it renders.

## Application version range

`manifest_version` pins the shape of `plugin.json`. `min_app_version` and
`max_app_version` pin the editor behind it — the tools, effect parameters and
job fields a plugin calls only exist from some release on, and change in later
ones. Each bound is optional and is simply not checked when absent:

| `min_app_version` | `max_app_version` | Runs on |
|---|---|---|
| — | — | any version (the normal case) |
| `3.1` | — | 3.1 and anything newer |
| — | `3.4` | 3.4 and anything older |
| `3.1` | `3.4` | 3.1 up to and including 3.4 |

Both are dotted numbers: `"3"`, `"3.1"`, `"3.1.2"`. Missing components count as
zero, so `3.1` and `3.1.0` are the same release. Anything else — `"3.x"`,
`"v3.1"`, `">=3.1"` — is rejected when the plugin is packed and when it is
imported, because a bound that parses as zero would silently block every
version.

Outside its range the plugin still appears in the list, but installing and
running it refuse with one sentence naming the version it wants. Prefer leaving
`max_app_version` out: set it only once a later release is *known* to break the
plugin, otherwise every editor update turns working plugins off.

## Install / Uninstall

Installing (from **Settings ▸ Plugins ▸ Load Plugins**) copies the plugin tree
into the user data folder (`<AppData>/plugins/<id>/`); each installed plugin then
gets its own settings tab where its environment, model downloads and API key are
managed (like the built-in Speech To Text / Object Detection tabs). For a
`local` plugin with a non-empty
`requirements.txt`, builds its environment on first run. Uninstalling from
**Settings ▸ Plugins** removes that folder **and** the plugin's private venv
(`venv-<id>`), its downloaded `models/` and the effects it brought, so
nothing is left behind; the shared venv is never touched. Bundled reference plugins can be disabled but not
deleted (they reappear from the read-only install location).

## Entry-point contract

The editor runs:

```
<python> <entry> --job /path/to/job.json
```

`job.json` the editor writes:

```json
{
  "job_id": "…",
  "plugin_id": "stub-video",
  "input": { "clips": [ { "bin_id": "3", "path": "/abs/in.mp4", "in": 0, "out": 125 } ],
             "face": { "rect": [0.4,0.2,0.2,0.3], "position": 87 } },
  "params": { "quality": "fast" },
  "output_dir": "/abs/work",
  "project": { "fps": 25.0, "width": 1920, "height": 1080 },
  "ffmpeg": "/abs/ffmpeg"
}
```

`input.clips` is a list (one element unless `input.multiple`); `input.face` is
present only for `face` plugins.

The plugin talks back over **stdout, one directive per line** (everything else is
ignored; use stderr for logs):

```
progress:<0-100>            # optional, may repeat
info:<message>              # optional status line, may repeat
need:{"kind":"api_key","provider":"runway"}   # optional; abort with an actionable request
result:{"outputs":[{"type":"video","path":"/abs/out.mp4"}]}   # final line on success
```

Exit codes: `0` ok (must have printed `result:`), `2` bad input, `3` missing key,
`4` missing model, `5` out of memory, `1` other error.

## Packing

```
python pack.py stub-video            # → dist/stub-video-1.0.0.wmplugin
python pack.py --all                 # pack every plugin folder here
python pack.py --check stub-video    # validate the manifest without packing
```
