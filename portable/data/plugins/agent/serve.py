"""The local model, kept warm between messages.

Loading a few gigabytes of weights takes long enough that doing it per message
would make the chat feel broken, so llama-server is started once and left
running. It is not a child of the plugin process — that one exits with every
answer — but a detached process the next answer finds again through a small
record in ``models/``. It shuts itself down after a stretch of silence, so a
chat nobody came back to does not hold the graphics card all evening.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
# The editor says where the weights are. A plugin that ships with the app runs
# from a read-only location, so its models cannot be beside it and guessing from
# __file__ would look in a folder that can never exist.
MODELS_DIR = os.environ.get("WUNJO_MODELS_DIR") or os.path.join(PLUGIN_DIR, "models")
RECORD = os.path.join(MODELS_DIR, "llama-server.json")
#: Where the model server's own output is kept, so a death has an explanation.
SERVER_LOG = os.path.join(MODELS_DIR, "llama-server.log")
#: Windows opens a console for every process a windowed application starts. The
#: model server, the watchdog and the device query would each flash one up over
#: the editor, and the server's stays for as long as the model is loaded. Zero
#: everywhere else, which is what the flag already is when nobody passes it.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _alive(port: int, pid: int) -> bool:
    """Whether the recorded server is there and answering.

    Judged by the port, not by the process id. The id is only a hint: the server
    outlives the run that started it, and the process asking may not be able to
    see it at all — a plugin started by the editor and one started by hand do
    not share a view of the process table. Trusting the id there means deciding
    a perfectly healthy server is gone and starting a second one, which then
    fails for want of memory the first one is holding.
    """
    del pid  # kept in the record for stop(), not consulted here
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _read_record() -> dict:
    try:
        with open(RECORD, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _binary() -> str:
    """The llama-server that came out of the downloaded archive.

    Release tarballs put it under a folder of their own and the name of that
    folder changes with every build, so it is searched for rather than spelled
    out — otherwise a routine version bump breaks the plugin.

    Windows publishes the same tool under a name of its own, so both spellings
    are looked for. Matching only the bare one there finds nothing, and "not
    found" is reported to the user as the runtime not being installed — about
    a file sitting in the folder it was downloaded to.
    """
    root = os.path.join(MODELS_DIR, "llama-server")
    for base, _dirs, files in os.walk(root):
        for name in ("llama-server", "llama-server.exe"):
            if name in files:
                return os.path.join(base, name)
    return ""


def _model_file() -> str:
    return os.path.join(MODELS_DIR, "model.gguf")


def _projector_file() -> str:
    """The vision projector, when it has been downloaded.

    With it the model can be shown a frame and say what is in it, which is what
    turns "put the clips in order" into a decision about the footage rather than
    about file names. Without it the same model still runs, text only.
    """
    path = os.path.join(MODELS_DIR, "mmproj.gguf")
    return path if os.path.isfile(path) else ""


def has_vision() -> bool:
    return bool(_projector_file())


def _gpu_layers() -> int:
    """All of them when there is a card, none when there is not.

    llama.cpp is happy to be told to offload more layers than exist, and the
    CPU build ignores the flag, so this only has to answer the coarse question.
    """
    return 99 if os.environ.get("WUNJO_GPU_BACKEND", "cpu") != "cpu" else 0


# Names that mean "this is the chip in the processor", not a graphics card.
_INTEGRATED = ("uhd graphics", "iris", "llvmpipe", "swiftshader", "vega", "radeon graphics", "apu")
# Names that mean a real card.
_DISCRETE = ("nvidia", "geforce", "radeon rx", "arc ", "quadro", "tesla")


def _pick_device(binary: str) -> str:
    """Which of the machine's GPUs to run on, by name.

    llama.cpp offloads to device 0, and on a laptop device 0 is usually the
    processor's own graphics — so the model quietly lands on the slow chip while
    a perfectly good card sits idle. That is not a small difference: measured
    here it was 4 tokens a second, which reads as a hung assistant.

    Free memory alone cannot decide it either, because integrated graphics
    report a share of system RAM and so claim *more* than the card has. So the
    name decides: a real card wins, and among equals the one with the most free
    memory.
    """
    try:
        listing = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=60,
                                 creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError) as error:
        log(f"could not list devices: {error}")
        return ""

    best, best_score = "", (-1, -1.0)
    for line in listing.splitlines():
        # "  Vulkan1: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MiB, 7211 MiB free)"
        match = re.match(r"\s+(\S+):\s+(.*?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", line)
        if not match:
            continue
        name, description, free = match.group(1), match.group(2).lower(), float(match.group(4))
        if any(mark in description for mark in _INTEGRATED) and not any(mark in description for mark in _DISCRETE):
            rank = 0
        elif any(mark in description for mark in _DISCRETE):
            rank = 2
        else:
            rank = 1
        if (rank, free) > best_score:
            best, best_score = name, (rank, free)
    if best:
        log(f"running on {best}")
    return best


def _is_ours(command: str, model: str) -> bool:
    """Whether a command line belongs to a model server this plugin started.

    Both marks have to be there: somebody else's llama.cpp is not ours to stop,
    and neither is a llama-server running somebody else's weights.
    """
    if "llama-server" not in command:
        return False
    if os.name == "nt":
        # The system reports back a path it has spelled its own way — a
        # different case, or the other slash — and compared literally that
        # never matches the one we passed, so no leftover is ever found.
        return model.replace("/", "\\").lower() in command.replace("/", "\\").lower()
    return model in command


def _pids_from_listing(listing: str, model: str) -> list:
    """Our servers' process ids out of a "<pid> <command line>" listing."""
    found = []
    for line in listing.splitlines():
        head, _, command = line.strip().partition(" ")
        if not head.isdigit() or int(head) == os.getpid():
            continue
        if _is_ours(command, model):
            found.append(int(head))
    return found


def _server_pids(model: str) -> list:
    """Every llama-server of ours that is running, however this system says so.

    Each platform is asked in its own way: /proc where there is one, ``ps`` on
    macOS, a process query on Windows. None of them may raise. This used to read
    /proc unconditionally, and on a system that has no /proc the FileNotFoundError
    travelled all the way up to the plugin's "the model is not installed" branch
    — so the first thing the assistant ever said on macOS and on Windows was
    that the user should download weights they had already downloaded.

    A listing that cannot be had means "no leftovers", which is the same answer
    a healthy machine gives and costs nothing worse than a second server failing
    to start for want of memory — a far better failure than never starting at all.
    """
    try:
        if os.name == "nt":
            listing = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 # No quotes anywhere in the command: it travels through Windows'
                 # own argument quoting on the way to the shell, and a filter with
                 # quotes in it arrives mangled. Cheaper to sift the lines here.
                 #
                 # Written straight to the console rather than returned as a
                 # value: PowerShell folds what it prints to a fixed width, and a
                 # command line is comfortably longer than that. Folded, the model
                 # path lands on a second line with no process id in front of it,
                 # and every line of the pair fails to be recognised — one for
                 # having no path, the other for having no id.
                 "Get-CimInstance Win32_Process | ForEach-Object { "
                 "[Console]::Out.WriteLine($_.ProcessId.ToString() + ' ' + $_.CommandLine) }"],
                capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW).stdout
            return _pids_from_listing(listing, model)
        if sys.platform == "darwin":
            # -ww, or macOS cuts every line to the width of a terminal that is
            # not even there — and what falls off the end is the tail of the
            # command line, which is exactly the model path being looked for.
            listing = subprocess.run(["ps", "-ax", "-ww", "-o", "pid=,command="],
                                     capture_output=True, text=True, timeout=30).stdout
            return _pids_from_listing(listing, model)
        found = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == os.getpid():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as handle:
                    command = handle.read().decode("utf-8", "replace")
            except OSError:
                continue
            if _is_ours(command, model):
                found.append(int(entry))
        return found
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        log(f"could not look for leftover model servers: {error}")
        return []


def _kill_orphans(model: str) -> None:
    """Stop any llama-server of ours that no record accounts for.

    Identified by the model path on its command line, so only servers this
    plugin started are touched — never somebody else's llama.cpp.
    """
    stopped = False
    for pid in _server_pids(model):
        log(f"stopping a leftover model server (pid {pid})")
        try:
            # On Windows this is a TerminateProcess rather than a signal, which
            # is all that platform offers and enough: the server holds nothing
            # that needs unwinding.
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except OSError:
            pass
    if stopped:
        time.sleep(2)  # let the graphics memory come back before asking for it


def ensure(context_tokens: int = 65536, idle_minutes: int = 10) -> str:
    """Return the base URL of a running llama-server, starting one if needed."""
    record = _read_record()
    port, pid = int(record.get("port", 0)), int(record.get("pid", 0))
    if port and pid and _alive(port, pid):
        os.utime(RECORD, None)  # the watchdog reads this to tell silence from work
        return f"http://127.0.0.1:{port}"

    binary, model = _binary(), _model_file()
    if not binary or not os.path.isfile(model):
        raise FileNotFoundError("the assistant's model or runtime is not installed")

    # Nothing usable is running, so nothing of ours should still be holding the
    # graphics card. One can be left behind — the record is deleted when the
    # model is unloaded, and a server that outlives its record becomes invisible
    # to us while still occupying the memory the next one needs, which then
    # fails to start for no reason the user can see.
    _kill_orphans(model)

    port = _free_port()
    command = [
        binary,
        "--model", model,
        "--port", str(port),
        "--host", "127.0.0.1",
        # Without --jinja llama.cpp does not parse tool calls out of the model's
        # reply at all, and an assistant that cannot call a tool is a chatbot.
        "--jinja",
        # Thinking off. With it on this model answers into `reasoning_content`
        # and leaves `content` empty, and the agent on the other end reports
        # "text part not found" and shows the user nothing. Reasoning also costs
        # tokens this size of context cannot spare.
        "--reasoning", "off",
        # Forced on everywhere it has always been, and left to llama.cpp on
        # Metal. Forced on there, the backend loaded the weights and answered
        # 83D0/%)59&#"=G?+H-:;+4G; — arithmetic gone wrong rather than a failure,
        # which no check that asked only whether a reply arrived would catch.
        # Whether the kernel exists for a given head size is llama.cpp's
        # judgement; where it does, auto takes it. Linux and Windows keep the
        # setting they were tested with.
        "--flash-attn", "auto" if sys.platform == "darwin" else "on",
        "--ctx-size", str(int(context_tokens)),
        "--n-gpu-layers", str(_gpu_layers()),
        "--alias", "wunjo-local",
    ]
    # Metal is asked for none of this. There is one GPU on a Mac, so naming a
    # device says nothing, and the memory it works out of is the machine's own,
    # so the cache does not need squeezing. Both flags exist for a laptop with a
    # discrete card and a fixed, small amount of video memory.
    #
    # They are not merely pointless there — with them the server died loading
    # the model on "ggml-metal-context.m: GGML_ASSERT(buf_dst) failed", which
    # reaches the user as the assistant refusing to start with a line of C in
    # place of a reason.
    metal = sys.platform == "darwin"
    device = _pick_device(binary) if (_gpu_layers() and not metal) else ""
    if device:
        command += ["--device", device]
    projector = _projector_file()
    if projector:
        command += ["--mmproj", projector]
    if device:
        # Keep the attention cache at eight bits. At a context this size it is
        # gigabytes at full width, and on a laptop card that is the difference
        # between running and dying part way through an answer — which the user
        # only ever sees as the reply breaking off. Not on Metal: see above.
        command += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    log(f"starting llama-server on port {port}")
    # Its output goes to a file, not to nowhere. When the model server dies —
    # out of memory, a bad flag, a broken download — this is the only place that
    # says why, and without it the failure reaches the user as an unexplained
    # decoding error.
    server_log = open(SERVER_LOG, "ab", buffering=0)
    process = subprocess.Popen(
        command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        # Outlives the plugin process that started it. On Windows a child
        # already survives its parent and this flag is ignored there, so the
        # detaching that matters is the same one line on both.
        start_new_session=True,
        creationflags=_NO_WINDOW,
    )
    with open(RECORD, "w", encoding="utf-8") as handle:
        json.dump({"port": port, "pid": process.pid, "idle_minutes": idle_minutes}, handle)
    _start_watchdog()

    deadline = time.time() + 180  # a cold read of several gigabytes off a slow disk
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the model could not be loaded — see {SERVER_LOG}: {_last_server_error()}")
        if _alive(port, process.pid):
            return f"http://127.0.0.1:{port}"
        time.sleep(1)
    stop()
    raise TimeoutError("the model did not start in time")


def stop() -> None:
    """Unload the model and forget where it was.

    The recorded pid is tried first, then every llama-server running on our own
    weights is swept up. The record can be stale — the editor may have been
    restarted since, or the number may belong to a process that has come and
    gone — and "stop" that leaves gigabytes on the graphics card is worse than
    no stop at all.
    """
    record = _read_record()
    pid = int(record.get("pid", 0))
    if pid:
        try:
            # The server runs in a session of its own, so on Unix the whole
            # group goes at once — llama.cpp starts helpers, and taking only
            # the leader leaves them holding the card. Windows has no process
            # group to signal and no killpg to call it with; there the process
            # itself is all there is to stop.
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    _kill_orphans(_model_file())
    try:
        os.remove(RECORD)
    except OSError:
        pass


def _last_server_error() -> str:
    """The line the model server died on, for a message the user can act on."""
    try:
        with open(SERVER_LOG, "r", encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle.readlines()[-40:] if line.strip()]
    except OSError:
        return "no log"
    for line in reversed(lines):
        if any(word in line.lower() for word in ("error", "failed", "out of memory", "cannot", "unable")):
            return line[:200]
    return lines[-1][:200] if lines else "no log"


def _busy_file() -> str:
    return RECORD + ".busy"


def busy(on: bool) -> None:
    """Say whether a turn is being worked on right now.

    The watchdog must not measure silence while the assistant is in the middle
    of answering. Looking at a folder of footage and then assembling it takes
    longer than any sensible idle timeout, and unloading the model half way
    through ends the turn with a decoding error the user cannot make sense of.
    """
    try:
        if on:
            with open(_busy_file(), "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        else:
            os.unlink(_busy_file())
            os.utime(RECORD, None)  # silence starts now, not when the server did
    except OSError:
        pass


def idle_seconds() -> float:
    """How long since the last message was handled.

    Zero while a turn is running. Otherwise measured from the record, which is
    touched as each turn ends — not from when the server started, which is a
    different thing entirely and would unload a model that is hard at work.
    """
    if os.path.exists(_busy_file()):
        return 0.0
    try:
        return time.time() - os.path.getmtime(RECORD)
    except OSError:
        return 0.0


def _start_watchdog() -> None:
    """A small detached process that unloads the model once nobody is asking.

    It cannot live in the plugin process, which exits with every answer, and it
    cannot live in llama-server, which has no such notion. So it is its own
    thing: it wakes now and then, looks at when the last message was handled,
    and shuts the server down when that was long enough ago.
    """
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--watch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        creationflags=_NO_WINDOW,
    )


def _watch() -> None:
    while True:
        time.sleep(30)
        record = _read_record()
        if not record:
            return  # somebody already stopped it; nothing left to watch
        limit = max(1.0, float(record.get("idle_minutes", 10))) * 60.0
        pid, port = int(record.get("pid", 0)), int(record.get("port", 0))
        if not pid or not _alive(port, pid):
            return
        if idle_seconds() > limit:
            log("unloading the model after a stretch of silence")
            stop()
            return


if __name__ == "__main__":
    if "--watch" in sys.argv:
        _watch()
