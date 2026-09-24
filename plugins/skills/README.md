# Wunjo Make plugin skills

Authoring knowledge for building Wunjo Make plugins — written so the in-app
assistant (or a person) can produce a correct, installable plugin from scratch
without guessing. The reference plugins in the parent folder (`stub-video`,
`stub-audio`, `stub-face`) are the living examples these docs describe.

Read in this order:

1. [authoring-guide.md](authoring-guide.md) — the full contract: manifest fields,
   the entry-point protocol, environments (venv), models, API keys, parameters,
   and how each plugin *type* is wired into the editor.
2. [examples.md](examples.md) — four worked skeletons, one per archetype
   (local video, local audio + model, API generator, face), copy-paste ready.
3. [limitations.md](limitations.md) — the hard rules and gotchas: what a plugin
   may not do, how secrets and models are handled, sandbox paths, sizing.

## Golden rules (the short version)

- A plugin is a folder: `plugin.json` + `main.py` + `requirements.txt`. Packed
  it is a `.wmplugin` zip (see `../pack.py`); **model weights are never packed**,
  they download on install.
- The editor runs `python main.py --job job.json`. The plugin talks back over
  **stdout, one directive per line**: `progress:<0-100>`, `info:<text>`,
  `need:<json>`, and a final `result:<json>`. Everything else is ignored; logs
  go to stderr. Exit `0` on success (after printing `result:`).
- **Never** read secrets from arguments or files: an API key arrives only as the
  environment variable `WUNJO_KEY_<PROVIDER>`. Never print it.
- `kind` is `local` (runs models on the machine) or `api` (calls a paid HTTP
  provider). `venv` defaults to `private` (own `venv-<id>`); use `shared`
  only for light, vetted deps.
- `target` decides where the plugin appears: `video`/`audio` on timeline clips,
  `face` on the detected-face menu, `generator` under Media ▸ Generate.
- Declare `os`; the editor refuses to install a plugin on an unsupported OS.
- Keep `main.py` deterministic and non-interactive. No windows, no prompts.
