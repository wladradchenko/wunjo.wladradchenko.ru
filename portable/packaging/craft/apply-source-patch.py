# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Apply a patch to a package's source after Craft has unpacked it.

Craft blueprints carry their own patches, but the blueprint for a KDE framework
belongs to KDE, not to us: overriding one means shadowing a recipe from another
repository and hoping the resolution order stays on our side of the argument.
Patching the unpacked tree between `--unpack` and `--compile` asks nothing of
Craft's internals except where it put the source, and that is discoverable.

Everything here is written to fail loudly. A patch that quietly does not apply
is worse than no patch at all: the build goes green, the .dmg is published, and
the failure surfaces as a crash on a user's Mac with nothing in the log
connecting it to a patch that was supposed to have been applied hours earlier.
So: the source must be found, the patch must apply, and the marker the patch was
written to remove must be gone afterwards. Any of the three failing stops the
build.

Usage:
    apply-source-patch.py <relative/path/inside/source> <patch file> [--gone TEXT]

The first argument identifies the package by a file only it has. --gone names
text that must no longer appear in that file once the patch is in.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_source(marker: str) -> list[Path]:
    """Every unpacked source tree that has this file, newest first.

    Craft keeps its working copies under CRAFT_ROOT/build/<category>/<package>/
    work/<version>/, and the version in that path changes with every release, so
    the tree is searched rather than spelled out. Directories Craft installs
    into are skipped: the same header can sit in both, and patching the
    installed copy changes nothing that will be compiled.
    """
    root = Path(os.environ.get("CRAFT_ROOT", "")) / "build"
    if not root.is_dir():
        sys.exit(f"no Craft build directory at {root} — is CRAFT_ROOT set?")
    found = [path.parent for path in root.glob(f"*/*/work/*/{marker}")]
    found += [path.parent for path in root.glob(f"*/*/*/work/*/{marker}")]
    # Strip the marker's own subdirectories back to the source root.
    depth = len(Path(marker).parts) - 1
    roots = {p.parents[depth - 1] if depth else p for p in found}
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("marker", help="a file only the package to patch has, e.g. src/foo.cpp")
    parser.add_argument("patch", help="the patch to apply, in -p1 form")
    parser.add_argument("--gone", default="", help="text that must be absent afterwards")
    arguments = parser.parse_args()

    patch = Path(arguments.patch).resolve()
    if not patch.is_file():
        sys.exit(f"no patch file at {patch}")

    candidates = find_source(arguments.marker)
    if not candidates:
        sys.exit(f"no unpacked source containing {arguments.marker} under $CRAFT_ROOT/build.\n"
                 f"Craft did not unpack it, or it came ready-built out of the binary cache — "
                 f"in which case this patch would never have been compiled in.")
    source = candidates[0]
    print(f"patching {source}")

    # --forward so a tree Craft has already patched is reported as such rather
    # than prompting; the check below decides whether that is acceptable.
    result = subprocess.run(["patch", "-p1", "--forward", "-i", str(patch)],
                            cwd=source, capture_output=True, text=True)
    print(result.stdout + result.stderr)

    target = source / arguments.marker
    text = target.read_text(encoding="utf-8", errors="replace")
    if arguments.gone and arguments.gone in text:
        sys.exit(f"{arguments.gone!r} is still in {target} after patching — the patch did not "
                 f"take, and a build from this source would ship the bug it was written to fix.")
    if result.returncode != 0 and not arguments.gone:
        sys.exit(f"patch exited {result.returncode}")
    print(f"applied; {arguments.gone!r} is gone" if arguments.gone else "applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
