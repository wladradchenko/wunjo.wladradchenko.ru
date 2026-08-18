![](data/pics/wunjo-logo.png)

# Wunjo Make

Wunjo Make is a nonlinear video editor with Artificial Intelligence tools built in.
Multi-track timeline editing, transitions, effects and keyframes, colour correction,
audio post-production, subtitles and titling, and rendering to practically any format —
plus face detection, installable AI plugins, and a chat assistant that can drive the
editor itself.

- Website: [wunjo.online](https://wunjo.online)
- Source and issue tracker: [github.com/wladradchenko/wunjo.wladradchenko.ru](https://github.com/wladradchenko/wunjo.wladradchenko.ru)

## This is a modified version of Kdenlive

Wunjo Make is a fork of [Kdenlive](https://kdenlive.org), the video editor developed by
the KDE community, modified since 2025 by the Wunjo project. The rebranding, the
Artificial Intelligence features, the plugin platform, the scripting interface and the
interface changes are Wunjo's work; everything else is the work of the Kdenlive authors,
to whom this project owes its existence and offers its thanks.

**Wunjo Make is not endorsed by, affiliated with, or supported by KDE e.V. or the
Kdenlive project.** Please send support requests and bug reports to the Wunjo issue
tracker above — not to them.

## Technology

- **Video engine**: [MLT](https://mltframework.org) and [FFmpeg](https://ffmpeg.org)
- **Interface**: Qt 6 and KDE Frameworks 6
- **Plugins**: Python, in per-plugin environments built by `uv`
- **Scripting**: a local socket serving the editor's methods, driven by the MCP server
  in `../mcp` — see `dev-docs/dbus-removal-checklist.md` for why it is not D-Bus

## Building

The supported build is the flatpak one; see `CLAUDE.md` in this directory for the exact
command, and `dev-docs/build.md` for building against a system Qt/KF6.

## Licence

Wunjo Make is free software, released under the **GNU General Public License version 3**
(see `COPYING`). It inherits that licence from Kdenlive.

Per-file copyright and licensing is machine-readable: every file is covered either by an
SPDX header or by an entry in `REUSE.toml`, and `reuse lint` checks it. Files inherited
from Kdenlive keep their original copyright notices, as the GPL requires.

Third-party components keep their own licences — MLT (LGPL), FFmpeg (LGPL/GPL), Qt
(LGPL), KDE Frameworks (LGPL). The FFmpeg builds shipped with Wunjo Make are configured
with `--enable-gpl` and **without** `--enable-nonfree`.

## Contributing

See `../CONTRIBUTING.md`. Bug reports go to the GitHub issue tracker.
