#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Does an agent outside the editor know where to find it?

The editor serves its scripting interface on a local socket whose path comes
from QStandardPaths::RuntimeLocation — see ScriptingServer::defaultSocketName().
That resolves somewhere different on every platform, and the client has to agree
with it exactly or nothing connects.

It did not agree. The client read XDG_RUNTIME_DIR, which is an XDG notion and
exists on Linux alone. On macOS the variable is unset, the client fell through
to the bare socket name, and connecting to a bare name means a path relative to
the working directory. The socket was in ~/Library/Application Support the whole
time. Every agent outside the editor — Claude Code, Cursor, anything driving it
over MCP — got "the editor is not answering on its scripting socket", with
nothing in that message to suggest the address was simply wrong. The bundled
assistant kept working and hid the problem, because the editor hands its plugins
the exact path in WUNJO_SOCKET.

This runs anywhere; each platform is reached by pretending to be it, which is
the only way one machine can check the arrangement for all of them.

    python3 tests/mcp/check_socket.py
"""
from __future__ import annotations

import os
import sys
import unittest.mock as mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "portable" / "mcp"))

import api.app_client as client  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if detail:
        print(f"          {detail}")


def candidates(platform: str, osname: str, environment: dict) -> list[str]:
    clean = {k: v for k, v in os.environ.items() if k not in ("WUNJO_SOCKET", "XDG_RUNTIME_DIR")}
    clean.update(environment)
    with mock.patch.object(sys, "platform", platform), \
         mock.patch.object(os, "name", osname), \
         mock.patch.dict(os.environ, clean, clear=True):
        return client._socket_candidates()


def main() -> int:
    print(f"checking socket discovery from {sys.platform}\n")

    # macOS maps RuntimeLocation to NSApplicationSupportDirectory, so this is
    # where the editor actually listens. The bug was that nothing looked here.
    mac = candidates("darwin", "posix", {})
    check("macOS looks in Application Support",
          all("Library/Application Support" in path for path in mac),
          f"got {mac}")
    check("macOS asks for an absolute path, not a bare name",
          all(os.path.isabs(path) for path in mac),
          "a bare name resolves against the working directory and never connects")

    linux = candidates("linux", "posix", {"XDG_RUNTIME_DIR": "/run/user/1000"})
    check("Linux still uses XDG_RUNTIME_DIR",
          linux == ["/run/user/1000/" + client.SOCKET_NAME], f"got {linux}")

    # With no RuntimeLocation the editor passes a bare name to QLocalServer,
    # which turns it into a file in QDir::tempPath(). The client has to follow.
    bare = candidates("linux", "posix", {})
    check("a Unix without XDG_RUNTIME_DIR falls back to the temp directory",
          all(os.path.isabs(path) for path in bare), f"got {bare}")

    # Named pipes are not files and cannot be enumerated; the plain name is
    # both correct and the only thing available there.
    windows = candidates("win32", "nt", {})
    check("Windows keeps the plain pipe name", windows == [client.SOCKET_NAME], f"got {windows}")

    explicit = candidates("darwin", "posix", {"WUNJO_SOCKET": "/tmp/chosen.sock"})
    check("an explicit WUNJO_SOCKET outranks every guess",
          explicit == ["/tmp/chosen.sock"], f"got {explicit}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
