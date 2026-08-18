#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the "wunjo" monochrome outline icon theme from mapping.json.

Developer-time tool — NOT run at build time. The emitted theme/ tree is
committed to the repository and installed verbatim by CMake (flatpak builds
are network-isolated, so icons must be committed).

Usage:
    ./generate.py --lucide /path/to/lucide-icons-<version>/icons

Source SVGs (Lucide and files in custom/) are 24x24, stroke-based
(stroke-width 2, round caps/joins), with no color or stroke-width attributes
on individual elements — colors and widths are applied by the wrapper this
script emits:

  - a <style id="current-color-scheme"> block + class="ColorScheme-Text" group
    (the standard KDE/Breeze contract: KIconLoader rewrites the style's color
    from the active color scheme, `stroke:currentColor` picks it up);
  - a baked default color #fcfcfc so icons render white-on-dark even when
    loaded by plain QIcon without KIconLoader recoloring;
  - actions/22: stroke-width 2 (the 24-grid renders crisply at 22/24/32px);
  - actions/16: same geometry, stroke-width 2.5 (2 would render as ~1.3px
    hairlines at 16px).

Custom SVGs in custom/ must therefore NOT set stroke-width/stroke/fill on
elements, or the 16px variant will not thicken.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WRAPPER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <defs><style type="text/css" id="current-color-scheme">.ColorScheme-Text {{ color:#fcfcfc; }}</style></defs>
  <g class="ColorScheme-Text" style="fill:none;stroke:currentColor;stroke-width:{stroke};stroke-linecap:round;stroke-linejoin:round">
{body}
  </g>
</svg>
"""

# Small overlay glyphs for `compose` recipes, drawn on the 24-grid around
# (19,19) so they sit in the bottom-right corner; `slash` is a full-icon
# "disabled" diagonal.
BADGES = {
    "plus": '<path d="M19 15.5v7"/><path d="M15.5 19h7"/>',
    "minus": '<path d="M15.5 19h7"/>',
    "x": '<path d="M16.5 16.5l5 5"/><path d="M21.5 16.5l-5 5"/>',
    "check": '<path d="M15.5 19.5l2.5 2.5 4.5-5"/>',
    "dot": '<circle cx="19" cy="19" r="2.75"/>',
    "chevron-right": '<path d="M17.5 15.5l3.5 3.5-3.5 3.5"/>',
    "chevron-left": '<path d="M20.5 15.5 17 19l3.5 3.5"/>',
    "pencil": '<path d="M15.5 22.5l.7-2.6 4.3-4.3a1.35 1.35 0 0 1 1.9 1.9l-4.3 4.3z"/>',
    "slash": '<path d="M4.5 19.5 19.5 4.5"/>',
}

SIZES = {"22": "2", "16": "2.5"}

SVG_TAG_RE = re.compile(r"<svg[^>]*>", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def inner_shapes(svg_text: str, origin: str) -> str:
    """Strip the outer <svg> element and comments, return inner markup."""
    text = COMMENT_RE.sub("", svg_text)
    m = SVG_TAG_RE.search(text)
    if not m:
        sys.exit(f"error: no <svg> tag in {origin}")
    text = text[m.end():]
    end = text.rfind("</svg>")
    if end < 0:
        sys.exit(f"error: no </svg> in {origin}")
    body = text[:end].strip()
    if not body:
        sys.exit(f"error: empty svg body in {origin}")
    for attr in ("stroke-width=", 'stroke="', 'fill="#', "style="):
        if attr in body:
            sys.exit(f"error: {origin} sets '{attr}' on elements; move it out — "
                     "the wrapper owns color and stroke width (see module docstring)")
    return body


def load_base(ref: str, lucide_dir: Path, origin: str) -> str:
    """Resolve 'lucide:<id>' or a custom/ file name to inner SVG markup."""
    if ref.startswith("lucide:"):
        path = lucide_dir / (ref.split(":", 1)[1] + ".svg")
        if not path.is_file():
            sys.exit(f"error: {origin}: lucide icon not found: {path}")
    else:
        path = HERE / "custom" / ref
        if not path.is_file():
            sys.exit(f"error: {origin}: custom icon not found: {path}")
    return inner_shapes(path.read_text(encoding="utf-8"), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lucide", required=True, type=Path,
                        help="path to the icons/ dir of an unpacked Lucide release")
    parser.add_argument("--mapping", type=Path, default=HERE / "mapping.json")
    parser.add_argument("--out", type=Path, default=HERE / "theme")
    args = parser.parse_args()

    if not args.lucide.is_dir():
        sys.exit(f"error: not a directory: {args.lucide}")
    mapping = {k: v for k, v in
               json.loads(args.mapping.read_text(encoding="utf-8")).items()
               if not k.startswith("_")}

    for size in SIZES:
        (args.out / "actions" / size).mkdir(parents=True, exist_ok=True)

    bodies = {}   # name -> inner markup
    aliases = {}  # name -> target name
    stats = {"lucide": 0, "custom": 0, "compose": 0, "alias": 0, "fallback": 0}

    for name, recipe in sorted(mapping.items()):
        source = recipe.get("source")
        if source == "fallback":
            stats["fallback"] += 1
            continue
        if source == "alias":
            aliases[name] = recipe["of"]
            continue
        if source == "lucide":
            bodies[name] = load_base("lucide:" + recipe["id"], args.lucide, name)
        elif source == "custom":
            bodies[name] = load_base(recipe["file"], args.lucide, name)
        elif source == "compose":
            badge = BADGES.get(recipe["badge"])
            if badge is None:
                sys.exit(f"error: {name}: unknown badge '{recipe['badge']}'")
            bodies[name] = load_base(recipe["base"], args.lucide, name) + "\n" + badge
        else:
            sys.exit(f"error: {name}: unknown source '{source}'")
        stats[source] += 1

    for name, target in aliases.items():
        if target not in bodies:
            sys.exit(f"error: alias {name} -> {target}, but {target} is not generated")
        bodies[name] = bodies[target]
        stats["alias"] += 1

    for name, body in bodies.items():
        indented = "\n".join("    " + line.strip() for line in body.splitlines() if line.strip())
        for size, stroke in SIZES.items():
            out = args.out / "actions" / size / f"{name}.svg"
            out.write_text(WRAPPER.format(stroke=stroke, body=indented), encoding="utf-8")

    (args.out / "index.theme").write_text(
        "[Icon Theme]\n"
        "Name=Wunjo\n"
        "Comment=Wunjo Make monochrome outline icons\n"
        "Inherits=breeze-dark\n"
        "Directories=actions/16,actions/22\n"
        "\n"
        "[actions/16]\n"
        "Size=16\n"
        "Context=Actions\n"
        "Type=Scalable\n"
        "MinSize=8\n"
        "MaxSize=16\n"
        "\n"
        "[actions/22]\n"
        "Size=22\n"
        "Context=Actions\n"
        "Type=Scalable\n"
        "MinSize=17\n"
        "MaxSize=512\n",
        encoding="utf-8")

    total = sum(stats.values())
    print(f"generated {len(bodies)} icons x {len(SIZES)} sizes into {args.out}")
    print(f"  recipes: {total} total — " + ", ".join(f"{k}: {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
