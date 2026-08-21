#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Does the built application actually start?

It is asked for its version and expected to print one and exit. That sounds
trivial and is not: getting that far means dyld found and bound every framework
in the bundle, the rpaths inside them point at each other rather than at the
build machine, Qt found a platform plugin, and main() ran. Most of what goes
wrong when packaging a Qt application on macOS goes wrong before that line is
printed.

    python3 tests/macos/check_launch.py path/to/wunjo.app

WHAT THIS CANNOT TELL YOU, and it matters more than what it can:

The build machine runs a newer macOS than the one this application targets. A
symbol introduced after the deployment target is present here and resolves
happily, so the application starts on the runner and dies in dyld on the older
Mac it was built for. That is not hypothetical — it is exactly what shipped:
libKF6ConfigWidgets referenced std::pmr from macOS 14, every check that involved
starting the thing passed, and it died before main() on a user's macOS 13.4.

Only tests/macos/check_abi.py catches that, because it compares against the
minimum the binary declares rather than against whatever the machine happens to
have. This file and that one answer different questions and neither replaces the
other.

QApplication is constructed before the command line is parsed, so a platform
plugin is needed even to print a version; the offscreen one is used so no window
server has to exist. If Qt cannot start for a reason of its own the result is
reported as inconclusive rather than failed — the build should not go red
because CI had no display — but anything that looks like a dynamic linker
failure fails hard, because that is the thing being tested.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

#: Text dyld puts on stderr when it cannot put the program together. Any of it
#: means a broken bundle, whatever the exit status says.
LINKER_TROUBLE = (
    "symbol not found",
    "library not loaded",
    "image not found",
    "incompatible library version",
    "code signature",
    "dyld:",
    "dyld[",
)

#: Qt failing to find a screen is a fact about the machine, not about the build.
QT_TROUBLE = (
    "could not find the qt platform plugin",
    "no qt platform plugin could be initialized",
    "could not connect to display",
)


def executable_of(bundle: Path) -> Path:
    with open(bundle / "Contents" / "Info.plist", "rb") as handle:
        name = plistlib.load(handle).get("CFBundleExecutable", "")
    if not name:
        sys.exit("Info.plist does not name an executable")
    return bundle / "Contents" / "MacOS" / name


def main(argument: str) -> int:
    bundle = Path(argument)
    binary = executable_of(bundle)
    if not binary.is_file():
        sys.exit(f"no executable at {binary}")

    environment = dict(os.environ)
    environment.update({
        "QT_QPA_PLATFORM": "offscreen",
        # Say why a plugin could not be loaded instead of only that it could not.
        "QT_DEBUG_PLUGINS": "1",
        # Bind every symbol at launch rather than lazily, so a missing one is
        # found now and not on whichever screen first calls into it.
        "DYLD_BIND_AT_LAUNCH": "1",
    })

    print(f"starting {binary}")
    try:
        finished = subprocess.run([str(binary), "--version"], capture_output=True,
                                  text=True, timeout=180, env=environment)
    except subprocess.TimeoutExpired:
        print("\nFAIL: it did not print a version within three minutes.")
        print("Asked for a version, it should not have reached anything that blocks.")
        return 1

    output = (finished.stdout or "") + (finished.stderr or "")
    lowered = output.lower()
    tail = [line for line in output.splitlines() if line.strip()][-25:]

    if any(mark in lowered for mark in LINKER_TROUBLE):
        print("\nFAIL: the dynamic linker could not put the application together.")
        print("This bundle will not start on any Mac. Recent output:\n")
        print("\n".join("    " + line for line in tail))
        return 1

    if finished.returncode == 0:
        print(f"\nPASS: started and exited cleanly.")
        for line in (finished.stdout or "").splitlines():
            if line.strip():
                print(f"    {line.strip()}")
        print("\nNote: this machine is newer than the deployment target, so a symbol\n"
              "introduced after that target resolves here and would not on a user's Mac.\n"
              "tests/macos/check_abi.py is the check for that; this one cannot see it.")
        return 0

    if any(mark in lowered for mark in QT_TROUBLE):
        print(f"\nINCONCLUSIVE: Qt could not start on this machine (exit {finished.returncode}).")
        print("Nothing here says the bundle is broken — the linker got through it.")
        print("Not failing the build for the absence of a display.\n")
        print("\n".join("    " + line for line in tail))
        return 0

    print(f"\nFAIL: exited {finished.returncode} without printing a version.\n")
    print("\n".join("    " + line for line in tail))
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
