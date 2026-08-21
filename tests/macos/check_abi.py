#!/usr/bin/env python3
"""Does the built bundle ask macOS for anything older macOS cannot give?

A .dmg that declares it runs on macOS 13.3 and then references a symbol Apple
only shipped in 14.0 does not fail at build time, does not fail at install time,
and does not fail in any way a developer on a new machine will ever see. It
fails on the user's Mac, in dyld, before one line of the application runs:

    Symbol not found: __ZNSt3__13pmr15memory_resourceD2Ev
    Referenced from: libKF6ConfigWidgets.6.29.0.dylib
    Expected in:     /usr/lib/libc++.1.dylib

That is a real crash from this project, and the deployment target was set
correctly when it happened — libc++'s availability annotations are disabled
somewhere in the toolchain, so the compiler stayed silent. This script is the
check that was missing. Point it at a built bundle:

    python3 tests/macos-abi/check_symbols.py path/to/wunjo.app

It reads each Mach-O's declared minimum macOS and its undefined symbols, and
fails when a binary references something introduced after that minimum.

A denylist is not a proof. It catches the families that have actually bitten
this project and any others added below, and it prints every unrecognised C++
standard-library symbol it saw so a new one can be spotted and added rather than
discovered by a user. Nothing here needs a Mac: it is all in the file format.
"""
from __future__ import annotations

import os
import struct
import sys

LC_SYMTAB, LC_VERSION_MIN_MACOSX, LC_BUILD_VERSION = 0x02, 0x24, 0x32
MACHO_MAGICS = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe")

#: Mangled-name fragments and the macOS release that first carried them in a
#: system dylib. Add to this whenever a new one is found — the comment matters
#: as much as the entry, because the next person needs to know why.
INTRODUCED = {
    # std::pmr lives in libc++.dylib, not in headers. Shipped in macOS 14.
    "NSt3__13pmr": (14, 0, "std::pmr"),
    # std::filesystem, out of line in libc++ since Catalina.
    "NSt3__14__fs10filesystem": (10, 15, "std::filesystem"),
    # std::barrier and std::latch arrived with Big Sur.
    "NSt3__17barrier": (11, 0, "std::barrier"),
    "NSt3__15latch": (11, 0, "std::latch"),
    # Floating-point std::to_chars — the reason this build already targets 13.3.
    "NSt3__18to_charsEPcS0_d": (13, 3, "std::to_chars (double)"),
    "NSt3__18to_charsEPcS0_f": (13, 3, "std::to_chars (float)"),
}


def _version(raw: int) -> tuple[int, int]:
    return raw >> 16, (raw >> 8) & 0xFF


def _read(path: str):
    """(minimum macOS, undefined symbols) for a Mach-O, or None if not one."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:4] not in MACHO_MAGICS:
        return None
    count = struct.unpack("<I", blob[16:20])[0]
    offset, minimum, symtab = 32, None, None
    for _ in range(count):
        command, size = struct.unpack("<II", blob[offset:offset + 8])
        if command == LC_BUILD_VERSION:
            minimum = _version(struct.unpack("<I", blob[offset + 12:offset + 16])[0])
        elif command == LC_VERSION_MIN_MACOSX:
            minimum = _version(struct.unpack("<I", blob[offset + 8:offset + 12])[0])
        elif command == LC_SYMTAB:
            symtab = struct.unpack("<IIII", blob[offset + 8:offset + 24])
        offset += size
    undefined = []
    if symtab:
        symoff, nsyms, stroff, _strsize = symtab
        for index in range(nsyms):
            entry = symoff + index * 16
            strx, kind = struct.unpack("<IB", blob[entry:entry + 5])
            if kind & 0x0E:  # anything but N_UNDF is defined here
                continue
            end = blob.index(b"\0", stroff + strx)
            undefined.append(blob[stroff + strx:end].decode("utf-8", "replace"))
    return minimum, undefined


def main(root: str) -> int:
    problems, unknown, examined = [], set(), 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            if os.path.islink(path):
                continue
            try:
                got = _read(path)
            except (OSError, ValueError, struct.error):
                continue
            if not got:
                continue
            minimum, undefined = got
            examined += 1
            for symbol in undefined:
                for fragment, (major, minor, label) in INTRODUCED.items():
                    if fragment in symbol:
                        if minimum and minimum < (major, minor):
                            problems.append((os.path.relpath(path, root), minimum,
                                             (major, minor), label, symbol))
                        break
                else:
                    if symbol.startswith("__ZNSt3__1"):
                        unknown.add(symbol)

    print(f"examined {examined} Mach-O binaries under {root}")
    if problems:
        print(f"\n{len(problems)} reference(s) to symbols newer than the binary's own minimum:\n")
        for where, minimum, need, label, symbol in problems:
            print(f"  {where}")
            print(f"    declares macOS {minimum[0]}.{minimum[1]}, but {label} "
                  f"needs macOS {need[0]}.{need[1]}")
            print(f"    {symbol}\n")
        print("This bundle will die in dyld at launch on a supported macOS version.")
    else:
        print("no binary references a symbol newer than the minimum it declares")

    if unknown:
        print(f"\n{len(unknown)} other C++ standard-library symbols were seen and not "
              f"recognised.\nNone is known to be a problem; they are listed so a new "
              f"one can be added to INTRODUCED\nrather than found by a user.")
        for symbol in sorted(unknown)[:15]:
            print(f"    {symbol}")
        if len(unknown) > 15:
            print(f"    ... and {len(unknown) - 15} more")

    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
