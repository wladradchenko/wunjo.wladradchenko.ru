# What D-Bus was responsible for, and how to check each part still works

> **D-Bus is gone.** `USE_DBUS`, `NODBUS`, `KDBusService`, `KF6::DBusAddons` and
> `org.wunjo.MainWindow.xml` no longer exist, and no source file includes a
> QtDBus header. `libQt6DBus.so` is still *loaded* at runtime because KDE
> Frameworks link it internally — the application binary has no direct
> dependency on it (`objdump -p wunjo | grep NEEDED` shows none).
>
> This document is kept as the record of what each piece used to do and how to
> check its replacement, because most of it can only be verified by hand.

Wunjo Make is dropping D-Bus so the same code paths run on Linux, Windows and
macOS. D-Bus did six separate jobs; they are being replaced one at a time and
they fail in different ways. This is the list of what to re-check, and how.

There is no functional test suite in this project (`mcp/tests/` holds only
`test_code_quality.py`, which reads source and never starts the application), so
these checks are the safety net. Run the whole list against a build before and
after each removal step and compare.

Status legend: **TODO** not started · **WIP** replacement written, not verified ·
**DONE** replaced and checked.

---

## 1. Scripting interface — 235 `Q_SCRIPTABLE` members · DONE (Linux)

**Was:** one call in `mainwindow.cpp`,
`QDBusConnection::sessionBus().registerObject("/MainWindow", this, ExportScriptableSlots | ExportScriptableSignals)`.
Everything the MCP server does went through it — the whole `mcp/api/` package
talks to those methods via `mcp/api/dbus_client.py`.

**Now:** `ScriptingServer`, a `QLocalServer` that dispatches by name through
`QMetaObject`. Same methods, same names, same arguments.

**Breaks like:** the Chat dock cannot do anything; every MCP tool returns an
error or times out; nothing in the GUI misbehaves. Silent from inside the app.

### How to check

Fastest signal — ask the running application what it exposes. Inside a flatpak
the socket is not at the host's `XDG_RUNTIME_DIR`; the sandbox maps it to
`/run/user/$UID/.flatpak/online.wunjo.make/xdg-run/`:

```
S=/run/user/$UID/.flatpak/online.wunjo.make/xdg-run/online.wunjo.make.scripting
echo '{"id":1,"method":"__methods__"}' | nc -U "$S"
```

Expect 235 entries. Then a read-only sweep, which cannot damage a project:

| Через MCP | Что должно вернуться |
|---|---|
| `get_project_info` | fps, resolution, path of the open project |
| `get_timeline_summary` | table of clips, one row per clip |
| `get_track_list` | video/audio tracks with names and ids |
| `get_media_pool` | bin contents |
| `list_plugins` | installed plugins |
| `undo_status` | what undo/redo would do next |

Then a scripted edit, which is what actually exercises argument marshalling —
run it on a scratch project, not on real work:

1. `new_project` → `import_media` (2–3 clips) → `build_timeline`
2. `add_transition`, `set_clip_volume`, `set_clip_transform` (7 arguments — the
   widest signature in the interface, and the one most likely to break first)
3. `add_effect` → `set_effect_param` → `add_effect_keyframe`
4. `speech_recognition` → `get_subtitles` (returns a list of maps)
5. `render_frame` → look at the JPEG
6. `undo` five times, `redo` five times, `save_project`

Every return type in the interface appears above: `bool` (121 methods),
`QVariantList` (23), `QString` (23), `int` (19), `QVariantMap` (14), `void`
(10), `QStringList` (7), `double` (5). If all six steps behave, the marshalling
is right.

---

## 2. Single instance and file opening — `KDBusService` · DONE

**Was:** `KDBusService programDBusService;` in `main.cpp` — and the KF6 default
is `Multiple`, not `Unique`. It claimed a bus name and nothing else: it never
prevented a second copy from starting, and no signal of it was connected, so a
second launch's arguments went nowhere. **There was no single-instance behaviour
to preserve.** (`desktop/CLAUDE.md` says otherwise; that note is wrong.)

**Now:** removed. `ScriptingServer` mirrors the same `Multiple` semantics — the
first instance takes the plain socket name, later ones get their pid appended.

**Breaks like:** nothing that worked before. The risk is the opposite one — the
socket refusing to open after a crash left its file behind, which would make the
application unscriptable until the file is deleted by hand.

### How to check — by hand only

1. Start the application, then start a second copy.
   → both must run, as they always did.
   → `ls /run/user/$UID/.flatpak/online.wunjo.make/xdg-run/` shows
     `online.wunjo.make.scripting` and `online.wunjo.make.scripting-<pid>`.
2. Kill the application with `-9`, then start it again.
   → must take the plain name back (the log says "reclaimed a stale socket").
   This is the case that a naive implementation gets wrong; test it twice.

---

## 3. Render progress — the 4-method XML adaptor · DONE (deleted, checks pending)

**Was:** `src/org.wunjo.MainWindow.xml` declares `setRenderingProgress`,
`setRenderingFinished`, `addProjectClip`, `addTimelineClip`, and
`qt_add_dbus_adaptor` generated an adaptor from it.

**Confirmed dead and deleted.** Nothing ever included the generated
`mainwindowadaptor.h`, so the adaptor was compiled and thrown away.
`src/org.wunjo.MainWindow.xml` and its `qt_add_dbus_adaptor` call are gone.
The checks below still have to be run, because they exercise the socket path
that now carries this alone. `src/render/renderserver.cpp` does the same
job over `QLocalServer` (`online.wunjo.make-<pid>`), it is compiled unconditionally
(the `if(NOT USE_DBUS)` guard in `src/render/CMakeLists.txt` is commented out),
and `renderer/renderjob.cpp` sends its progress as JSON over `QLocalSocket` on
every platform. **Verify before deleting** rather than assuming.

**Breaks like:** rendering works but the progress bar never moves, or a finished
render is never noticed and the job sits at 99% forever.

### How to check

1. Render a 30-second timeline to MP4.
2. The progress bar must move, and the job must go to "finished" on its own.
3. Start a render and press **Abort**. → the `melt` process must actually die
   (`ps aux | grep melt`), not keep writing the file.
4. Render two jobs queued together → both report progress separately.

---

## 4. Sleep inhibition — `powermanagementinterface.cpp` · DONE (checks pending)

**Was:** `org.freedesktop.ScreenSaver` / `org.freedesktop.PowerManagement` over
the session bus, ~15 `#ifndef NODBUS` blocks. Kept the machine awake during
playback and rendering.

**Now:** one `systemd-inhibit` child process per lock on Linux — logind hands
the lock to a process and takes it back when that process dies, so the handle is
the process. `SetThreadExecutionState` on Windows and `IOPMAssertion` on macOS,
both of which the file now really implements (the macOS branch used to be a
`// TODO?`).

**Breaks like:** the screen locks in the middle of a two-hour render. Nobody
notices until it happens to them.

### How to check

1. Set the screen lock to 1 minute in the system settings.
2. Start a render longer than 2 minutes. → screen must stay awake.
3. Play the timeline for longer than 2 minutes. → same.
4. Let the application sit idle for 2 minutes. → the screen **must** lock. (A
   broken inhibitor that never releases is the same bug in the other direction,
   and is the more annoying one.)

---

## 5. Colour picker — `colorpickerwidget.cpp` · DONE, with a loss

**Was:** `QScreen::grabWindow` where the display server allows it, and the
`org.freedesktop.portal.Screenshot` portal where it does not — which in practice
means Wayland.

**Now:** `QScreen::grabWindow` only. **This costs the eyedropper on Wayland**,
where the compositor will not hand over screen pixels and the portal — the only
way around that — is a D-Bus API by definition. The button is disabled there and
its tooltip says why, instead of being enabled and doing nothing (which is what
it did on Windows and macOS all along). If Wayland users need it back, the
portal has to come back with it.

**Breaks like:** the eyedropper returns black, or nothing happens when you click.

### How to check

1. Open any effect with a colour parameter (e.g. **Chroma Key**).
2. Click the eyedropper, pick a colour from another window.
3. The swatch must take that colour. Check on **both** X11 and Wayland —
   they are different code paths.

---

## 6. Build system · DONE

**Was:** `USE_DBUS` option in `CMakeLists.txt`, `NODBUS` compile definitions in
`src/CMakeLists.txt` and `renderer/CMakeLists.txt`, `find_package(Qt6 DBus)` and
`KF6DBusAddons`.

All of it is gone: the `USE_DBUS` option, both `NODBUS` definitions, the
`find_package(Qt6 DBus)` call, `KF6::DBusAddons`, and the renderer's conditional
link. Nothing in the tree asks for QtDBus any more.

**Breaks like:** builds fine on the machine you tested, fails on someone else's
because a dependency is only conditionally present.

### How to check

- Full clean flatpak build (the standard command in `desktop/CLAUDE.md`).
- `grep -rn "QDBus\|KDBusService\|NODBUS" src/ renderer/` → only the two
  desktop-service files should remain.

---

## Order of work

Items 1 and 2 share a socket, so they land together. Item 3 is a deletion once
verified. Items 4 and 5 are independent and can be done any time. Item 6 is the
clean-up pass at the end.

Nothing here can be checked on Windows or macOS until Craft produces a build;
until then, everything above is Linux-only evidence and the cross-platform claim
stays unproven.
