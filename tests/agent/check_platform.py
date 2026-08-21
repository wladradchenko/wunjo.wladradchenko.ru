#!/usr/bin/env python3
"""Does the agent plugin's process handling work on the system it is run on?

Everything here exists because the plugin used to read /proc unconditionally.
On a system without one that raised, the error travelled up to the branch that
reports missing weights, and the first thing the assistant ever said on macOS
and on Windows was that the user should download a model they already had.

The point of this file is that it runs on the real thing. Each platform reaches
for a different way of listing processes — /proc, ``ps``, a CIM query — and the
one that matters is whichever the machine running this actually takes. Mocks are
used only to reach the *other* platforms' parsing, which is better than not
covering it at all but is not the same as having been there.

Run it on the machine in question:

    python tests/agent-platform/check_platform.py

Exit status is 0 when everything passed, 1 otherwise, so CI can go red.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest.mock as mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(REPO, "portable", "data", "plugins", "agent")

# A deliberately long models directory. On Windows the process listing comes
# back through PowerShell, which folds what it prints at a fixed width; a short
# path would fit on one line and the check would pass without ever testing the
# thing that breaks. This is comfortably past where folding happens.
MODELS = os.path.join(tempfile.mkdtemp(),
                      "wunjo-make-data", "plugins", "agent",
                      "a-directory-named-at-length-so-the-command-line-cannot-fit-on-one-line",
                      "models")
os.makedirs(MODELS)
os.environ["WUNJO_MODELS_DIR"] = MODELS

sys.path.insert(0, PLUGIN)
import serve  # noqa: E402  (the path above is what makes this importable)

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(("  PASS  " if condition else "  FAIL  ") + name + (f"\n          {detail}" if detail else ""))
    sys.stdout.flush()


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def native_name() -> str:
    if os.name == "nt":
        return "Windows (Get-CimInstance)"
    if sys.platform == "darwin":
        return "macOS (ps)"
    return "Linux (/proc)"


def main() -> int:
    print(f"python {sys.version.split()[0]} on {sys.platform} / os.name={os.name}")
    print(f"models dir: {MODELS}")

    model = serve._model_file()
    open(model, "w").close()

    # A stand-in for a model server left behind by an earlier run: a process
    # carrying both of the marks _is_ours looks for. Anything long-lived would
    # do; what matters is its command line.
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                              "llama-server", "--model", model])
    time.sleep(2)  # let the system's process table catch up before asking it

    try:
        section(f"1. The listing this machine actually uses — {native_name()}")
        started = time.time()
        pids = serve._server_pids(model)
        took = time.time() - started
        check("a leftover server is found", decoy.pid in pids,
              f"looked for pid {decoy.pid}, got {pids}")
        print(f"          (the query took {took:.1f}s)")
        if took > 10:
            check("the query is quick enough to run before every start", False,
                  f"{took:.1f}s is long enough to be felt when the model loads")

        check("a server on somebody else's weights is left alone",
              decoy.pid not in serve._server_pids(os.path.join(MODELS, "not-our-model.gguf")))

        # The whole reason the models directory above is so long: if the
        # system's listing truncates or folds the command line, the model path
        # falls off the end and no leftover is ever recognised.
        check("the full command line survives the listing",
              decoy.pid in pids and len(model) > 100,
              f"the model path alone is {len(model)} characters")

        section("2. The regression itself")
        with mock.patch.object(subprocess, "run", side_effect=FileNotFoundError("no such tool")), \
             mock.patch.object(os, "listdir", side_effect=FileNotFoundError("no /proc")):
            try:
                empty = serve._server_pids(model)
                check("a listing that cannot be had returns [] rather than raising", empty == [])
            except Exception as error:  # noqa: BLE001
                check("a listing that cannot be had returns [] rather than raising", False,
                      f"raised {error!r} — this is exactly the old bug")

        section("3. Finding the runtime under the name this platform gives it")
        binary_dir = os.path.join(MODELS, "llama-server", "build", "bin")
        os.makedirs(binary_dir, exist_ok=True)
        spelling = "llama-server.exe" if os.name == "nt" else "llama-server"
        open(os.path.join(binary_dir, spelling), "w").close()
        check(f"{spelling} is found where the archive put it",
              os.path.basename(serve._binary()) == spelling,
              f"got {serve._binary()!r}")

        section("4. Stopping, on a system with no process groups")
        with mock.patch.object(os, "kill") as killed:
            killpg = getattr(os, "killpg", None)
            if killpg is not None:
                del os.killpg
            try:
                with open(serve.RECORD, "w", encoding="utf-8") as handle:
                    handle.write('{"port": 1, "pid": %d}' % decoy.pid)
                serve.stop()
                check("stop() falls back to os.kill where killpg does not exist", killed.called)
            finally:
                if killpg is not None:
                    os.killpg = killpg

        section("5. The other platforms' parsing (mocked — not the real thing)")
        win_model = r"C:\Users\v\AppData\Local\wunjo\models\agent\model.gguf"
        listing = "\n".join([
            "4321 C:\\Windows\\explorer.exe",
            "9876 C:\\PROGRA~1\\llama-server.exe --model "
            "c:/users/v/appdata/local/wunjo/models/agent/MODEL.GGUF --port 51000",
            "1111 C:\\other\\llama-server.exe --model C:\\somebody\\else.gguf",
            "2222 C:\\Windows\\notepad.exe",
        ])
        with mock.patch.object(os, "name", "nt"):
            parsed = serve._pids_from_listing(listing, win_model)
            check("Windows: matched despite a different case and the other slash", parsed == [9876],
                  f"got {parsed}")
            check("Windows: another project's llama.cpp is not touched", 1111 not in parsed)

        unix_listing = "\n".join([
            "  501 /usr/sbin/somethingd",
            f"  777 /opt/llama-server --model {model} --port 51000",
        ])
        with mock.patch.object(os, "name", "posix"):
            parsed = serve._pids_from_listing(unix_listing, model)
            check("Unix: the leading spaces ps pads pids with are handled", parsed == [777],
                  f"got {parsed}")
    finally:
        decoy.kill()
        decoy.wait(timeout=10)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  failed: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
