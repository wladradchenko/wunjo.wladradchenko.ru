#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Is the built .app the application it is supposed to be?

Not whether it runs — check-bundle-abi.py and the launch step cover that — but
whether it says who it is. A bundle whose Info.plist is subtly wrong installs
and starts perfectly and is wrong on the user's screen, which is the kind of
defect that survives every technical test and gets reported as "it says the
wrong name".

Three things go wrong here and all three have:

  * A name nobody set. CFBundleDisplayName was simply absent from the template,
    so Finder fell back to the bundle's filename and the application was called
    "wunjo" on a user's Mac while every other platform called it Wunjo Make.
  * A name CMake did not substitute. Only a fixed list of MACOSX_BUNDLE_*
    properties are replaced in a custom Info.plist; anything else is copied
    through as the literal text "${MACOSX_BUNDLE_DISPLAY_NAME}", which then
    appears on screen exactly like that.
  * Something that did not get packaged. The assistant is a directory of Python
    inside the bundle, and nothing in a compile or a link notices when it is not
    there — it goes missing quietly and the chat panel is empty on first run.

Usage:  check-bundle-identity.py path/to/wunjo.app
"""
from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

#: What Info.plist has to say. The display name is what Finder shows; the bundle
#: name is what the menu bar shows and Apple asks it to be 15 characters or less.
EXPECTED = {
    "CFBundleDisplayName": "Wunjo Make",
    "CFBundleName": "Wunjo Make",
    "CFBundleIdentifier": "online.wunjo.make",
}

#: Paths inside Contents/ that must exist, and why anyone cares.
REQUIRED = {
    # DATA_INSTALL_PREFIX is empty on macOS and "/wunjo" everywhere else, so
    # these sit one level higher inside the bundle than they do on Linux.
    "Resources/plugins/agent/plugin.json": "the assistant plugin",
    "Resources/plugins/agent/main.py": "the assistant's entry point",
    "Resources/mcp/run.py": "the MCP server the assistant speaks through",
    # main.cpp routes icons through this engine on macOS and falls back to Qt's
    # plain theme loader when it is absent — and that loader hands any name it
    # cannot resolve to the platform engine, which resolves names as SF Symbols
    # and aborts in AppKit. The exact directory main.cpp tests for.
    "PlugIns/kiconthemes6/iconengines": "KDE's icon engine, which keeps icons away from SF Symbols",
}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  PASS  {message}")


def check_plist(bundle: Path) -> None:
    path = bundle / "Contents" / "Info.plist"
    if not path.is_file():
        fail(f"no Info.plist at {path}")
        return
    with open(path, "rb") as handle:
        plist = plistlib.load(handle)

    for key, want in EXPECTED.items():
        got = plist.get(key)
        if got is None:
            fail(f"Info.plist has no {key} — it should be {want!r}")
        elif got != want:
            fail(f"Info.plist {key} is {got!r}, expected {want!r}")
        else:
            ok(f"{key} = {got!r}")

    # A value CMake was meant to substitute and did not. It reaches the screen
    # verbatim, and no other check would call it wrong.
    for key, value in plist.items():
        if isinstance(value, str) and "${" in value:
            fail(f"Info.plist {key} was never substituted: {value!r}")

    name = plist.get("CFBundleName", "")
    if len(name) > 15:
        fail(f"CFBundleName {name!r} is {len(name)} characters; Apple asks for 15 or fewer")


def check_payload(bundle: Path) -> None:
    for relative, why in REQUIRED.items():
        if (bundle / "Contents" / relative).exists():
            ok(f"{why} is in the bundle")
        else:
            # Craft's install layout has moved before; say what is actually
            # there so the fix does not need another twenty-minute run.
            found = list((bundle / "Contents").rglob(Path(relative).name))
            hint = f" (found instead at {found[0].relative_to(bundle)})" if found else ""
            fail(f"{why} is missing: Contents/{relative}{hint}")


def check_bundle_name(bundle: Path) -> None:
    """Is the bundle's own filename the product name?

    Finder, /Applications, the Dock and the Force Quit list all show the .app's
    filename. Not CFBundleName, not CFBundleDisplayName — the filename. Left as
    the CMake target name it read "wunjo" on a user's Mac while every other
    platform said Wunjo Make, and no plist key could have fixed it.
    """
    if bundle.name == "Wunjo Make.app":
        ok(f"the bundle is called {bundle.name!r}")
    else:
        fail(f"the bundle is called {bundle.name!r}, so Finder shows {bundle.stem!r} — "
             f"set OUTPUT_NAME in src/CMakeLists.txt")


def check_icons(bundle: Path) -> None:
    """Is the icon theme in the bundle at all?

    This is not a question about how the application looks. Craft's macOS
    blacklist opens with "share/icons/.*" — correct for a KDE application, whose
    icons are compiled into a library, and fatal for this one, whose icons are
    files. Packaged without them, the theme is simply not there: every
    QIcon::fromTheme falls through to the platform icon engine, that engine
    resolves names as SF Symbols, and AppKit aborts inside
    NSImageSymbolRepProvider the first time a toolbar is painted.

    That is not a hypothetical either. The application opened, and died on the
    first new project with "abort() called" and a stack ending in
    QAppleIconEngine::paint. Nothing about the crash mentioned icons.

    packaging/craft/blueprints/apps/wunjo/keep_macos.list is what carries the
    theme past the blacklist. This is the check that it still does.
    """
    icons = bundle / "Contents" / "Resources" / "icons"
    for theme in ("wunjo", "wunjo-light"):
        index = icons / theme / "index.theme"
        if not index.is_file():
            fail(f"the {theme} icon theme is not in the bundle ({index.relative_to(bundle)} "
                 f"is missing) — Craft's blacklist ate it, see keep_macos.list")
            continue
        drawings = list((icons / theme).rglob("*.svg"))
        if len(drawings) < 100:
            fail(f"the {theme} theme has only {len(drawings)} icons in it, which is too few "
                 f"to be the whole theme")
        else:
            ok(f"the {theme} theme is in the bundle with {len(drawings)} icons")


def check_executable(bundle: Path) -> None:
    path = bundle / "Contents" / "Info.plist"
    with open(path, "rb") as handle:
        name = plistlib.load(handle).get("CFBundleExecutable", "")
    binary = bundle / "Contents" / "MacOS" / name
    if not binary.is_file():
        fail(f"CFBundleExecutable names {name!r}, which is not in Contents/MacOS")
        return
    try:
        arch = subprocess.run(["lipo", "-archs", str(binary)],
                              capture_output=True, text=True, timeout=60).stdout.strip()
        ok(f"executable {name!r} is {arch or 'of an unreported architecture'}")
    except (OSError, subprocess.SubprocessError):
        ok(f"executable {name!r} is present (lipo unavailable to name its architecture)")


def main(argument: str) -> int:
    bundle = Path(argument)
    if not bundle.is_dir():
        sys.exit(f"no bundle at {bundle}")
    print(f"checking {bundle}")
    check_bundle_name(bundle)
    check_plist(bundle)
    check_executable(bundle)
    check_payload(bundle)
    check_icons(bundle)
    print(f"\n{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
