# Limitations and rules

Hard constraints and gotchas when authoring a plugin. Breaking these means the
plugin is rejected at import, fails at run time, or is unsafe.

## Security

- **Secrets only via environment.** An API key reaches the plugin as the env var
  `WUNJO_KEY_<PROVIDER>` (or the manifest's `key_setting`). Never accept a key via
  argv or a file, never print it (`info:`/`result:`/stderr), never persist it.
- **No shell/exec tricks.** Do the work in Python (or a bundled binary you ship
  in `assets/`). Do not spawn shells to fetch code, and do not download and run
  arbitrary code at run time.
- **Write only inside `output_dir`** (and your own `models/` when downloading).
  Do not touch the user's project or other plugins.
- Agent-authored plugins are reviewed/enabled by the user; keep `main.py`
  readable and obvious so that review is meaningful.

## Environments

- `local` heavy or version-pinned deps ⇒ **private venv** (`venv-<id>`,
  the default). Only pure-python, conflict-free deps may use `venv: shared`.
- Building a venv costs time and disk (tens of MB to several GB with torch).
  Declare `hardware.min_vram_gb` / `cpu_ok` honestly; the feasibility check and
  the user rely on it.
- Never `pip install` from inside the plugin. Declare deps in `requirements.txt`
  only; the app's broker installs them.

## Models

- Weights are **never packed** into the `.wmplugin`. Declare them in `models`
  with a stable `url` (and `sha256` when possible); they download into `models/`
  on the user's machine. Large weights ≈ size × 2.5 free disk needed.
- Resolve models relative to `__file__`, not the CWD. If missing at run time,
  emit `need:{"kind":"model","name":"…"}` and exit `4` — do not download silently.

## Platform / sandbox

- The app is flatpak-only today (Linux). Declare `os`; installation is refused
  on an OS not in the list. Design cross-platform anyway (the app targets macOS
  and Windows later): no hard-coded `/tmp`, no Linux-only paths.
- Paths in `job.json` are absolute and host-visible; always use them verbatim.
  Do not assume a shared `$TMPDIR`.
- Network is available, but only assume it for `api` plugins and model
  downloads. A `local` inference plugin must run offline once its models exist.

## Runtime behaviour

- **Non-interactive and deterministic.** No GUI, no `input()`, no prompts. The
  only channel back to the editor is the stdout protocol.
- Emit `progress:` regularly on long jobs; the user can cancel, which kills the
  process — clean up partial files in `output_dir` on the next run, do not rely
  on shutdown hooks.
- Print exactly one final `result:` line on success, then exit `0`. If you exit
  `0` without a `result:`, the run is treated as failed.
- Keep stdout clean: logs, warnings and tracebacks go to **stderr**.

## Manifest

- `id` is immutable and equals the folder name; changing it makes a different
  plugin. Bump `version` on updates.
- `input`/`result` must match the `target` (e.g. a `generator` has
  `input.clip: none`; a `face` plugin receives `input.face`).
- Validate with `python ../pack.py --check <dir>` — the editor applies the same
  rules and will not show an Import button for an invalid manifest.
