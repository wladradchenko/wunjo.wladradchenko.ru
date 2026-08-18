# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Point a Craft installation at the Wunjo blueprint in this checkout.

Craft looks for recipes in the directories listed under `[Blueprints]Locations`
in CraftSettings.ini, so registering ours is a matter of writing that one key.

`craft --add-blueprint-repository` is the documented way to add recipes and it
does not work here: it takes a URL, writes it into a generated recipe as a
gitUrl and clones it. Our recipe is a subdirectory of this repository, not a
repository of its own, so there is nothing for it to clone.

The edit is line-based rather than via ConfigParser because CraftSettings.ini is
a documented template — Craft's own bootstrap rewrites it the same way — and
round-tripping it through ConfigParser would drop every comment in it.

Usage: CRAFT_ROOT=<craft prefix> python3 register-blueprint.py
"""

import os
import re
import sys
from pathlib import Path

BLUEPRINTS = Path(__file__).resolve().parent / "blueprints"


def main() -> int:
    craft_root = os.environ.get("CRAFT_ROOT")
    if not craft_root:
        print("CRAFT_ROOT is not set", file=sys.stderr)
        return 1

    ini = Path(craft_root) / "etc" / "CraftSettings.ini"
    if not ini.is_file():
        print(f"No Craft settings at {ini}", file=sys.stderr)
        return 1

    lines = ini.read_text(encoding="utf-8").splitlines()
    section = None
    header = None
    for i, line in enumerate(lines):
        match = re.match(r"^\[(.+)\]\s*$", line)
        if match:
            section = match.group(1)
            if section == "Blueprints":
                header = i
        elif section == "Blueprints" and re.match(r"^[#;]?\s*Locations\s*=", line, re.I):
            lines[i] = f"Locations = {BLUEPRINTS}"
            break
    else:
        # No Locations key at all. Reuse the section if it is already there:
        # Craft reads this file with ConfigParser, which rejects a duplicate.
        if header is None:
            lines += ["", "[Blueprints]", f"Locations = {BLUEPRINTS}"]
        else:
            lines.insert(header + 1, f"Locations = {BLUEPRINTS}")

    ini.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Blueprints]Locations = {BLUEPRINTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
