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

    The tree is walked rather than matched against a pattern. Craft's layout is
    CRAFT_ROOT/build/<category path>/<package>/work/<version>/, and neither half
    of that is fixed: the category path is as deep as the blueprint tree happens
    to be — kconfigwidgets sits at kde/frameworks/tier3/kconfigwidgets, four
    levels — and the version directory changes with every release. A glob written
    to a guessed depth finds nothing the day the guess is wrong, and reports it
    as "Craft did not unpack it", which is the opposite of what happened.

    Directories Craft installs into are skipped. A source file has no business
    being in one, but patching an installed copy would change nothing that gets
    compiled, and silently doing nothing is the failure this script exists to
    prevent.
    """
    root = Path(os.environ.get("CRAFT_ROOT", "")) / "build"
    if not root.is_dir():
        sys.exit(f"no Craft build directory at {root} — is CRAFT_ROOT set?")

    wanted = Path(marker).parts
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith("image") and d != "install"]
        if files and wanted[-1] in files:
            here = Path(base) / wanted[-1]
            if here.parts[-len(wanted):] == wanted:
                found.append(Path(*here.parts[:-len(wanted)]))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def describe(root: Path) -> str:
    """What is actually under the build directory, for a failure worth reading.

    Without this the only way to learn why the source was not found is another
    push and another twenty-minute CI run.
    """
    lines = []
    for base, dirs, _files in os.walk(root):
        depth = len(Path(base).relative_to(root).parts)
        if depth > 5:
            dirs[:] = []
            continue
        lines.append("  " + str(Path(base).relative_to(root)))
        if len(lines) > 40:
            lines.append("  ...")
            break
    return "\n".join(lines) if lines else "  (empty)"


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
        root = Path(os.environ["CRAFT_ROOT"]) / "build"
        sys.exit(f"no unpacked source containing {arguments.marker} under {root}.\n"
                 f"Either Craft did not unpack it, or it came ready-built out of the binary "
                 f"cache — in which case this patch would never have been compiled in.\n"
                 f"What is there:\n{describe(root)}")
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
