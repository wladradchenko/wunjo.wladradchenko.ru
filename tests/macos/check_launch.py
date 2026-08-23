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

import json
import os
import plistlib
import subprocess
import sys
import tempfile
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


def check_components(binary: Path, environment: dict) -> int:
    """Ask the application what it is built out of, and believe only itself.

    --setup-report writes the component list KAboutData was given, and the entry
    that matters is MLT: its version comes from a live mlt_version_get_string()
    call, so a version in that file is proof the media framework loaded inside
    the packaged bundle rather than proof that something linked at build time.

    An editor whose MLT did not come along starts, draws its whole interface and
    fails at the first clip — a failure that looks nothing like a packaging one
    and gets reported as "video does not work". Cheap to rule out here.

    FFmpeg is listed too but registered with no version string at all (see
    main.cpp), so only its presence can be asked about, not its version.
    """
    # It refuses to overwrite, so hand it a path that cannot exist yet.
    report = Path(tempfile.mkdtemp()) / "setup-report.json"
    finished = subprocess.run([str(binary), "--setup-report", str(report)],
                              capture_output=True, text=True, timeout=180, env=environment)
    if not report.is_file():
        print(f"\nFAIL: --setup-report wrote nothing (exit {finished.returncode}).")
        print("\n".join("    " + line for line in
                        ((finished.stdout or "") + (finished.stderr or "")).splitlines()[-15:]))
        return 1

    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except ValueError as error:
        print(f"\nFAIL: the report is not valid JSON: {error}")
        return 1

    components = {c.get("name", ""): c.get("version", "") for c in data.get("components", [])}
    print(f"\npackaged as {data.get('packageType', '(unstated)')!r}, built out of:")
    for name, version in components.items():
        print(f"    {name} {version}".rstrip())

    if "MLT" not in components:
        print("\nFAIL: MLT is not in the report at all.")
        return 1
    if not components["MLT"].strip():
        print("\nFAIL: MLT reports no version, so the framework did not load in the bundle.")
        print("Every clip would fail on a machine that has no MLT of its own.")
        return 1
    print(f"\nPASS: MLT {components['MLT']} answered from inside the bundle.")

    status = check_icons_are_reachable(data)
    return check_widget_style(data) or status


def check_widget_style(data: dict) -> int:
    """Is the brand stylesheet sitting on a style it was written for?

    The sheet in src/assets/style.qss is written against Breeze. The bundle
    ships no Breeze widget style — Contents/PlugIns/styles holds libqmacstyle
    and nothing else — so unless something says otherwise the application runs
    on the native macOS style and the sheet decorates a style with entirely
    different metrics. Seen on a user's screen: a combo box with its text in a
    corner and no padding, and one welcome-screen icon several times its proper
    size. A native style also draws through AppKit's NSCell path, which is where
    the application died — an assertion inside NSCrackRect with nothing of ours
    on the stack.

    None of that is visible from outside: the package is correct either way.
    """
    style = data.get("widgetStyle")
    if style is None:
        print("\nNote: the report does not name a widget style, so this bundle predates the check.")
        return 0

    native = {"macos", "macintosh", "mac"}
    if str(style).strip().lower() in native:
        print(f"\nFAIL: the interface is drawn by the native {style!r} style.")
        print("The brand stylesheet is written for Breeze and will not fit it — padding, icon")
        print("sizes and combo boxes all come out wrong, and AppKit's own geometry has crashed")
        print("the application from this path. main.cpp chooses Fusion on macOS for that reason.")
        return 1

    print(f"\nPASS: the interface is drawn by {style!r}, which the stylesheet is written for.")
    return 0


def check_icons_are_reachable(data: dict) -> int:
    """Can the application find its icon theme, not merely carry it?

    Having the files in the bundle is not the same as Qt looking at them, and the
    difference is invisible from outside. Qt searches only the directories its
    platform theme names and the Cocoa one names none, so the theme shipped
    correctly, every file present, and was never found: the search list held one
    entry, ":/icons", and the theme was on disk beside it.

    What follows is not a cosmetic problem. Each miss falls through to the
    platform icon engine, which resolves names as SF Symbols, and AppKit aborts
    inside NSImageSymbolRepProvider on the first symbol it cannot draw. Measured
    on the build that shipped: 158 icon lookups, 158 misses, 53 answered by SF
    Symbols, and the application dead on the first new project.

    tests/macos/check_identity.py asks whether the theme is in the bundle; this
    asks whether it is anywhere the application will look. Both are needed and
    the first one passed while this was broken.
    """
    icons = data.get("icons")
    if not isinstance(icons, dict):
        print("\nNote: the report has no icon section, so this bundle predates the check.")
        print("Nothing is asserted about icons here.")
        return 0

    print(f"\nicon theme {icons.get('theme')!r}, falling back to {icons.get('fallbackTheme')!r}")
    for path in icons.get("searchPaths", []):
        print(f"    looked for themes in {path}")

    # The healthy macOS arrangement: the Qt theme name holds KDE's icon engine,
    # which answers every name itself, so Qt never reaches the platform engine
    # that resolves names as SF Symbols and aborts in AppKit. The theme actually
    # in force is KIconTheme's, reported separately.
    if icons.get("theme") == "KIconEngine":
        kde = str(icons.get("kdeTheme", ""))
        if kde.startswith("wunjo"):
            print(f"\nPASS: icons go through KDE's engine with the {kde!r} theme in force,")
            print("so a name this application does not have cannot reach SF Symbols.")
            return 0
        print(f"\nFAIL: KDE's icon engine is in use but the theme in force is {kde!r},")
        print("not one of this application's. The icons would be whatever KDE defaults to.")
        return 1

    if not icons.get("themeFound"):
        print(f"\nFAIL: the application cannot find its icon theme {icons.get('theme')!r}.")
        print("It has no icons at all in this state, and dies in AppKit at the first one it")
        print("is asked to draw. The theme is very likely installed in the bundle and simply")
        print("not on any path listed above — QIcon::setThemeSearchPaths in main.cpp is what")
        print("puts it there.")
        return 1

    print(f"\nPASS: the icon theme {icons.get('theme')!r} is on a path the application searches.")
    return 0


def check_bundled_python(bundle: Path) -> int:
    """Does the interpreter the plugins are built from actually run?

    Existing is not enough, and that is the entire point of this check.
    Contents/MacOS holds a Craft shim named like the interpreter which redirects
    to ../lib/Python.framework — a directory the packager never creates, because
    it puts the framework in Contents/Frameworks instead. The shim is a real
    Mach-O of the right name and a plausible size that exits 255 with
    "KShimgen: Failed to locate".

    So every check that looked for a file passed, and every plugin needing Python
    failed on the user's Mac with "Cannot create the python virtual environment".
    Running it is the only question worth asking.
    """
    contents = bundle / "Contents"
    candidates = sorted((contents / "Frameworks" / "Python.framework" / "Versions").glob("*/bin/python3*"),
                        reverse=True)
    candidates += sorted(contents.glob("MacOS/python3*"))
    candidates = [p for p in candidates if not p.name.endswith("-config")]

    if not candidates:
        print("\nFAIL: the bundle carries no Python interpreter at all.")
        print("Every plugin that needs one is unusable.")
        return 1

    working = []
    for path in candidates:
        try:
            done = subprocess.run([str(path), "-c", ""], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"    {path.relative_to(bundle)} — could not be run ({error})")
            continue
        if done.returncode == 0:
            working.append(path)
            print(f"    {path.relative_to(bundle)} — runs")
        else:
            detail = ((done.stderr or done.stdout or "").strip().splitlines() or [""])[0]
            print(f"    {path.relative_to(bundle)} — exits {done.returncode}: {detail[:100]}")

    if not working:
        print("\nFAIL: the bundle carries Python but none of it runs.")
        print("The framework's own interpreter records its library as")
        print("@executable_path/../Frameworks/..., which resolves only when it is launched from")
        print("Contents/MacOS — and the shim that lives there looks for the framework under")
        print("lib/, where the packager does not put it. The two halves point past each other.")
        print("")
        print("Falling back to an interpreter on PATH is not an answer: macOS has shipped no")
        print("Python since 12.3, and /usr/bin/python3 is a stub that offers to install the")
        print("Command Line Tools. On a machine without them every plugin is dead.")
        print("")
        print("Package::internalCreatePackage in the wunjo blueprint puts the framework's real")
        print("binary where the shim was, which is what makes its own load command resolve.")
        print("It hangs off that method and not off preArchive for a reason: preArchive runs")
        print("before MacDylibBundler brings the framework in, so it found nothing and said so")
        print("in the package log. If this check fails again, read that log for 'replacing' —")
        print("its absence, and which warning came instead, says which half went wrong.")
        return 1

    print(f"\nPASS: {len(working)} of {len(candidates)} bundled interpreters run.")
    return 0


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

    python_status = check_bundled_python(bundle)

    print(f"\nstarting {binary}")
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
        # It starts; now ask it what it is made of.
        return check_components(binary, environment) or python_status

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
