#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Validate that every icon name referenced in the source tree is covered.

Covered means: present in mapping.json (as a real recipe or an explicit
"fallback" to the inherited breeze-dark theme) — and, for real recipes,
present in the generated theme/ tree.

Run from anywhere:  ./check_coverage.py     (exit 1 on gaps)
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"

PATTERNS = [
    re.compile(r'QIcon::fromTheme\(\s*QStringLiteral\(\s*"([^"]+)"'),
    re.compile(r'QIcon::fromTheme\(\s*"([^"]+)"'),
    re.compile(r'image://icon/([A-Za-z0-9_-]+)'),
    re.compile(r'"(wunjo-[a-z][a-z0-9-]+)"'),
]
# QML icon.name lines may hold several names (ternaries) — grab every string
QML_LINE = re.compile(r'(?:icon\.name|iconName):([^\n]*)')
QML_STRING = re.compile(r'"([^"]+)"')
RC_PATTERN = re.compile(r'icon="([^"]+)"')

# Icon names loaded by KXmlGui/KStandardAction/KMessageBox internally, which a
# source grep cannot see. Keep in sync with mapping.json.
IMPLICIT_NAMES = {
    "application-exit", "configure-shortcuts", "configure-toolbars",
    "dialog-cancel", "dialog-error", "dialog-information", "dialog-ok",
    "dialog-warning", "document-close", "document-open-recent",
    "document-print", "document-print-preview", "edit-redo", "go-home",
    "help-whatsthis", "show-menu", "tools-report-bug", "window-close",
}

# Dynamic families built with QString::arg() etc.
DYNAMIC_PREFIXES = ("task-process-",)


def referenced_names() -> set:
    names = set(IMPLICIT_NAMES)
    for path in SRC.rglob("*"):
        if path.suffix in (".cpp", ".h", ".qml"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pat in PATTERNS:
                names.update(pat.findall(text))
            for line in QML_LINE.findall(text):
                names.update(QML_STRING.findall(line))
    rc = SRC / "wunjoui.rc"
    if rc.is_file():
        names.update(RC_PATTERN.findall(rc.read_text(encoding="utf-8")))
    # drop non-icon matches (paths, ternary fragments, the app icon template)
    return {n for n in names
            if re.fullmatch(r"[a-z][a-z0-9_-]+", n) and not n.startswith("wunjo-x")}


def main() -> int:
    mapping = {k: v for k, v in
               json.loads((HERE / "mapping.json").read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
    referenced = referenced_names()

    missing_from_mapping = sorted(
        n for n in referenced
        if n not in mapping and not n.startswith(DYNAMIC_PREFIXES))
    fallbacks = sorted(n for n, r in mapping.items() if r["source"] == "fallback")
    missing_files = sorted(
        n for n, r in mapping.items()
        if r["source"] != "fallback" and not (HERE / "theme/actions/22" / f"{n}.svg").is_file())
    stale = sorted(n for n in mapping
                   if n not in referenced and not n.startswith(DYNAMIC_PREFIXES))

    ok = True
    if missing_from_mapping:
        ok = False
        print(f"ERROR: {len(missing_from_mapping)} referenced names missing from mapping.json:")
        for n in missing_from_mapping:
            print(f"  {n}")
    if missing_files:
        ok = False
        print(f"ERROR: {len(missing_files)} mapped names have no generated file (run generate.py):")
        for n in missing_files:
            print(f"  {n}")
    if stale:
        print(f"note: {len(stale)} mapping entries no longer referenced in src/ (harmless):")
        print("  " + ", ".join(stale))
    if fallbacks:
        print(f"note: {len(fallbacks)} names intentionally fall back to breeze-dark:")
        print("  " + ", ".join(fallbacks))

    print(f"{'OK' if ok else 'FAILED'}: {len(referenced)} referenced, "
          f"{len(mapping)} mapped, {len(fallbacks)} fallback")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
