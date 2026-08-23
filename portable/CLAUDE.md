# Wunjo Make (desktop/) — build, run, and architecture rules

## Build & run (flatpak-only)

The host is Ubuntu 22.04 without KF6/Qt6.10/MLT — native cmake builds are impossible.
Always build via flatpak, **from this directory**, with a clean env (the dev session runs
inside the VS Code snap whose env breaks flatpak):

```
env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin LANG=C.UTF-8 \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always \
  /usr/bin/flatpak-builder --user --install --force-clean \
  --install-deps-from=flathub --default-branch=master build-flatpak .flatpak-manifest.json
```

- Run: `flatpak run online.wunjo.make` (from a normal terminal; in the agent session prefix the
  same `env -i … DISPLAY=:1 XAUTHORITY=… XDG_RUNTIME_DIR=…`).
- Several copies may run at once (they always could). Kill running `bwrap … wunjo` pids
  before relaunching if you want a clean one. Never `pkill -f 'app/online.wunjo.make'` — it
  kills the calling shell; use `ps | grep '[b]wrap.*wunjo' | awk … | kill`.
- A build killed mid-run leaves a stale rofiles-fuse mount → next build fails with
  "Build directory … not initialized": `fusermount -u` the rofiles mounts and delete
  `.flatpak-builder/rofiles/`, then rebuild.

## No D-Bus

**This project does not use D-Bus, and nothing new may introduce it.** It exists on
Linux only, and Wunjo Make is going to Windows and macOS; a Linux-only mechanism means
every feature built on it has to be written twice. `USE_DBUS`, `NODBUS`, `KDBusService`
and the generated `org.wunjo.MainWindow.xml` adaptor are all gone. What replaced each of
them:

- **Scripting / the assistant's hands** — `src/scripting/scriptingserver.cpp`: a
  `QLocalServer` (unix socket, named pipe on Windows) that dispatches every
  `Q_SCRIPTABLE` slot of MainWindow through `QMetaObject`. One JSON object per line;
  `__methods__` lists what is callable and `__ping__` says the editor is alive. Adding a
  `Q_SCRIPTABLE` slot is still all it takes to expose a method. The client is
  `mcp/api/app_client.py`.
- **Render progress** — `src/render/renderserver.cpp`, which already worked this way.
- **Keeping the machine awake** — `src/powermanagementinterface.cpp`: `systemd-inhibit`
  held as a child process on Linux, `SetThreadExecutionState` on Windows,
  `IOPMAssertion` on macOS.
- **Shutting the machine down after a render** — `systemctl poweroff` / `shutdown /s` /
  `osascript`.
- **Colour picker** — `QScreen::grabWindow`. Where the compositor refuses (Wayland
  without a portal) the button is disabled and says so, rather than doing nothing.

If something genuinely needs a desktop service that only speaks D-Bus, shell out to the
tool that desktop ships for it. Do not link QtDBus back in.

Background and the manual checks for each piece: `dev-docs/dbus-removal-checklist.md`.

## Dependency policy

- **MLT is pinned to a commit** in `packaging/flatpak/org.kde.kdenlive-dependencies.json`
  (plus `-DUSE_LV2=OFF -DUSE_VST2=OFF`; the KDE SDK has no lilv). Never track `master`:
  a silent bump once froze thumb-producer seeking and broke face detection. To bump MLT,
  change the commit deliberately and re-verify face detection produces *moving* rects
  before keeping it.

## UI architecture

- Global QSS lives in `src/assets/style.qss` (qrc `:/data/style.qss`), loaded in `main.cpp`.
  Its header lists prohibitions — keep them: no global `QPushButton` styling, no `*`/`QWidget`
  selectors, don't touch the QToolButton popupMode paddings.
- Icon theme "wunjo" (Lucide-based) lives in `data/icons-wunjo/`: `mapping.json` is the
  source of truth, `generate.py` regenerates `theme/` (committed — flatpak builds offline).
  SVG recolor contract: `<style id="current-color-scheme">` + `class="ColorScheme-Text"` +
  `stroke:currentColor`. Theme forcing must run **after** the QApplication constructor
  (`QIcon::setThemeName` + `KIconTheme::forceThemeForTests`) — KIconTheme's pre-routine
  re-applies breeze during construction.
- Workspace layouts (`data/layouts/*.json`) are KDDockWidgets LayoutSaver dumps. The five
  pills (Logging/Editing/Audio/Effects/Color) in the menu-bar corner are canonical
  workspaces — keep them. Startup applies the active pill's canonical layout (persisted as
  `activeLayout`), **not** a serialized session blob, so layout file changes take effect on
  restart. Never rename dock `uniqueName`s; create every dock (incl. `chat`) before the
  first layout restore. Editing layout composition: left column = hub tabs
  (Bin/History/Props/Effects) over mini Clip Monitor (tabbed with Effect/Composition Stack),
  center = Project Monitor over Timeline, right = full-height Chat.
- Tab-raise UX: clicking a bin clip raises the Clip Monitor tab; selecting a timeline clip
  raises the Effect/Composition Stack tab (stock raiseprops* settings drive the latter).
- `src/ui.rc`: any edit requires bumping its `version` attribute, or users keep the cached
  copy from `~/.local/share/kxmlgui5/`.

## AI features

- All AI actions live in the right-click clip menu under a top-level
  "Artificial Intelligence" submenu (spelled out, never "AI"), filtered by clip type
  (audio-only plugins on audio clips, video-only on video).
- Face detection is built-in (no venv): RetinaFace-MobileNet0.25 ONNX (~2 MB) via OpenCV
  DNN, model installed to `share/wunjo/ai/`, code in `src/ai/`. `facedetector.cpp` needs
  `-fexceptions` (set in `src/CMakeLists.txt`). It is a per-clip property: a background task
  (`FaceDetectTask`) analyses the clip and persists per-clip JSON to
  `<projectDataFolder>/faces/<binId>.json`; the **Project Monitor** draws clickable boxes
  from that data (never the Clip Monitor — the box acts on the timeline clip under it).
- **Approved face-pipeline shape — do not "optimize" it again**: analysis samples every
  `FaceDataStore::kAnalyseStep` (5) frames; the overlay interpolates between analysed
  frames; "Hide Face" applies ONE standard Motion Tracker (`opencv.tracker`) effect with
  pre-filled `results` keyframes emitted only when the face moved AND ≥15 frames passed
  since the last point (cap 2000). Per-frame keyframes freeze the app on long clips (MLT's
  animation parser and the KeyframeModel are quadratic); smoothing/Douglas-Peucker passes
  degraded tracking and were reverted. Tune the `spacing` constant only.
- Every Python environment is its own: a plugin gets `venv-<id>` (the manifest's `venv` field
  accepts only `private`), SAM uses `venv-sam`, the MCP server uses `venv-mcp`, and
  `<AppData>/venv` is speech-to-text alone. Sharing one was allowed once and only coupled
  unrelated stacks to each other's pins; uv builds a private one from cache instead. Small
  ONNX models (<10 MB) go directly into the build via OpenCV DNN instead of a venv plugin.
- The MCP server is installed on its own settings tab (Settings ▸ Plugins ▸ MCP,
  `McpPythonEnv`), never as part of a model plugin. Driving the editor from Claude Code or
  Cursor must not depend on weights that agent never loads.
