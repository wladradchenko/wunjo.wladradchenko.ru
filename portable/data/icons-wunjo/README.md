# Wunjo icon theme

Monochrome outline icon theme (CapCut-like stroke style) for Wunjo Make.
Covers every icon name referenced in `src/` (see `check_coverage.py`);
anything unmapped falls back to the inherited **breeze-dark** theme from the
KDE flatpak runtime.

## Layout

- `mapping.json` — single source of truth: icon name → recipe
  (`lucide` | `compose` | `custom` | `alias` | `fallback`).
- `custom/` — hand-authored 24×24 stroke SVGs (Lucide drawing conventions;
  no color/stroke-width on elements — the generator's wrapper owns those).
- `theme/` — **generated and committed** output, installed verbatim to
  `share/icons/wunjo/` by CMake. Do not edit by hand; edit mapping/custom
  and regenerate. Committed because flatpak builds are network-isolated.
- `generate.py` — developer-time generator (see below).
- `check_coverage.py` — validates that all referenced names are covered;
  run after adding icons to the codebase.

## Icon sources

- **Lucide 1.24.0** (ISC license, `LICENSES/LICENSE.Lucide`) — primary set.
  Download: https://github.com/lucide-icons/lucide/releases/tag/1.24.0
  (asset `lucide-icons-1.24.0.zip`; never committed, never fetched at build).
- Custom icons in `custom/` (GPL-3.0-only, same as the app) — editor-specific
  glyphs with no Lucide equivalent (trim modes, timeline insert/lift/extract/
  overwrite, zones, keyframe-duplicate).

## Regenerating

```sh
unzip lucide-icons-1.24.0.zip -d /tmp/lucide
./generate.py --lucide /tmp/lucide/icons
./check_coverage.py
```

To bump the Lucide version: update the version above and in this README,
regenerate, and visually sweep the app (icon geometry occasionally changes
between releases).

## How recoloring works

Each generated SVG follows the Breeze/KIconLoader contract:
a `<style id="current-color-scheme">.ColorScheme-Text { color:#fcfcfc; }</style>`
block plus a `class="ColorScheme-Text"` group with `stroke:currentColor`.
KIconLoader rewrites the style's `color:` from the active KColorScheme
(`Wunjo.colors`), so the icons recolor for normal/selected/disabled states.
The baked `#fcfcfc` keeps icons white-on-dark even when loaded by plain
QIcon without KIconLoader.

The theme is forced app-side in `src/main.cpp` via `QIcon::setThemeName("wunjo")`
(with `breeze-dark` as the fallback theme), right after `KIconTheme::initTheme()`.
