#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Does the chat work, on a machine with no graphics card?

    python3 tests/agent/check_chat.py              # logic only, seconds
    python3 tests/agent/check_chat.py --with-model # downloads a runtime and a
                                                   # small model and talks to it

The chat is four things in a row: llama-server holding the weights, Goose
deciding what to call, the MCP server, and the editor at the far end. Only the
first two can be reached from a build machine — the editor is not running there
— so the boundary is drawn deliberately and named in each check, rather than
pretending to an end-to-end that quietly tests nothing.

No GPU is needed and none is assumed. WUNJO_GPU_BACKEND is what decides whether
layers are offloaded, and with it unset serve.py asks for none: the same path a
user without a card takes, which is worth testing on its own account.

The default run touches no network. --with-model adds about 400 MB of downloads
and a minute of CPU inference; it uses a small stand-in model rather than the
2.5 GB one the plugin ships, because what is under test is our plumbing — that a
server starts, answers, and can be stopped — and not how well a model writes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "portable" / "data" / "plugins" / "agent"

#: Small, has a chat template — llama-server is started with --jinja and a model
#: without one will not load at all.
STAND_IN_MODEL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
                  "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")

# A caller may point this at somewhere that survives between runs — CI keeps the
# weights in the Actions cache rather than fetching gigabytes on every build.
# Left unset it is a temp directory, which is what a laptop wants.
MODELS = Path(os.environ.get("WUNJO_MODELS_DIR") or (Path(tempfile.mkdtemp()) / "models"))
MODELS.mkdir(parents=True, exist_ok=True)
os.environ["WUNJO_MODELS_DIR"] = str(MODELS)
os.environ.pop("WUNJO_GPU_BACKEND", None)  # no card, and none pretended

sys.path.insert(0, str(PLUGIN))
import serve  # noqa: E402
import main as agent  # noqa: E402  (rebinds print to stderr; harmless here)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    sys.stderr.write(("  PASS  " if condition else "  FAIL  ") + name + "\n")
    if detail and not condition:
        sys.stderr.write(f"          {detail}\n")
    sys.stderr.flush()


def section(title: str) -> None:
    sys.stderr.write(f"\n{title}\n{'-' * len(title)}\n")


# ---------------------------------------------------------------------------
# What the user actually sees: the reply, cut out of a terminal transcript
# ---------------------------------------------------------------------------

BANNER = "starting session | provider: openai model: wunjo-local\n    goose is ready\n"


def transcript(body: str) -> str:
    return BANNER + body


def test_reply_parsing() -> None:
    section("1. Pulling the reply out of Goose's transcript")

    check("a plain answer comes through whole",
          agent._reply_from(transcript("I have put the three clips on the timeline.")) ==
          "I have put the three clips on the timeline.")

    check("the banner is not part of the answer",
          "goose is ready" not in agent._reply_from(transcript("Done.")))

    # A real turn: the model says what it is about to do, calls a tool, then
    # reports. Only the report is wanted — the rest already showed as its own
    # card in the panel.
    with_tool = transcript(
        "Let me look at the folder first.\n"
        "─────────────────────────────\n"
        "▸ wunjo-make__list_media\n"
        "    path: /home/v/footage\n"
        "\n"
        "There are 12 clips; I added the four widest to the timeline.\n")
    check("only the prose after the last tool call is returned",
          agent._reply_from(with_tool) == "There are 12 clips; I added the four widest to the timeline.",
          f"got {agent._reply_from(with_tool)!r}")

    # A turn can end on a tool call. Silence is the one answer the panel must
    # never show, so the last thing that was said stands in.
    ends_on_tool = transcript(
        "I will render it now.\n"
        "─────────────────────────────\n"
        "▸ wunjo-make__render\n"
        "    preset: mp4\n")
    check("a turn ending on a tool call still says something",
          agent._reply_from(ends_on_tool) == "I will render it now.",
          f"got {agent._reply_from(ends_on_tool)!r}")

    check("terminal colour codes are stripped",
          agent._reply_from(transcript("\x1b[32mAll done.\x1b[0m")) == "All done.",
          f"got {agent._reply_from(transcript(chr(27) + '[32mAll done.' + chr(27) + '[0m'))!r}")

    check("an empty transcript yields nothing, not a crash",
          agent._reply_from("") == "")


# ---------------------------------------------------------------------------
# The recipe is the whole of what the assistant is allowed to touch
# ---------------------------------------------------------------------------

def test_recipe() -> None:
    section("2. The recipe that fences the assistant in")
    work = Path(tempfile.mkdtemp())
    path = agent.write_recipe(str(work), "put the clips in order", tool_set="focused")
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))

    check("the message is the prompt", recipe["prompt"] == "put the clips in order")
    check("exactly one extension is offered", len(recipe["extensions"]) == 1,
          f"got {[e.get('name') for e in recipe['extensions']]}")
    check("and it is the editor's own MCP server",
          recipe["extensions"][0]["name"] == "wunjo-make")

    envs = recipe["extensions"][0]["envs"]
    check("'focused' maps to the short tool profile", envs["WUNJO_TOOL_PROFILE"] == "core",
          f"got {envs['WUNJO_TOOL_PROFILE']!r}")

    full = json.loads(Path(agent.write_recipe(str(work), "x", tool_set="full")).read_text())
    check("'full' maps to the whole one",
          full["extensions"][0]["envs"]["WUNJO_TOOL_PROFILE"] == "full")

    check("vision is off when no projector was downloaded", envs["WUNJO_VISION"] == "0")

    # Whichever layout this machine has, the interpreter must not be a path that
    # simply does not exist on it — that was broken on Windows for a while.
    speech = agent._speech_python()
    check("the speech interpreter is a plausible path for this platform",
          ("Scripts" in speech) if os.name == "nt" else ("bin" in speech),
          f"got {speech!r}")


# ---------------------------------------------------------------------------
# Bookkeeping the watchdog depends on
# ---------------------------------------------------------------------------

def test_lifecycle() -> None:
    section("3. Knowing when the model is busy and when nobody is asking")
    Path(serve.RECORD).write_text('{"port": 1, "pid": 1, "idle_minutes": 10}', encoding="utf-8")

    serve.busy(True)
    check("a turn in progress reads as no idle time at all", serve.idle_seconds() == 0.0)
    serve.busy(False)
    time.sleep(1.2)
    check("silence is measured from the end of the turn, not the start of the server",
          0.5 < serve.idle_seconds() < 30, f"got {serve.idle_seconds():.1f}s")

    check("with no card, no layers are offloaded", serve._gpu_layers() == 0)
    os.environ["WUNJO_GPU_BACKEND"] = "metal"
    check("with one, they are", serve._gpu_layers() == 99)
    os.environ.pop("WUNJO_GPU_BACKEND")


# ---------------------------------------------------------------------------
# The real thing, on the CPU
# ---------------------------------------------------------------------------

def platform_key() -> str:
    machine = "arm64" if os.uname().machine in ("arm64", "aarch64") else "x64"
    system = {"darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)
    return f"{system}-{machine}"


class DownloadFailed(Exception):
    """The weights could not be fetched. Nothing about the build is wrong."""


def fetch(url: str, into: Path) -> Path:
    """Download @p url to @p into, with curl.

    These files run to gigabytes and come from a public mirror that throttles,
    so a stall in the middle is ordinary. Read straight through by hand, one
    stall raises TimeoutError and the whole download starts again from nothing —
    which is how a build came to fail on a socket having tested the application
    perfectly well.

    curl already solves this and is on every machine this runs on: -C - carries
    on from whatever arrived, --retry covers the transient failures, and it
    moves the bytes faster than a Python loop besides.
    """
    name = url.split("/")[-1].split("?")[0]
    sys.stderr.write(f"    downloading {name}\n")
    sys.stderr.flush()
    finished = subprocess.run(
        ["curl", "--location", "--fail", "--silent", "--show-error",
         "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
         "--connect-timeout", "30", "--continue-at", "-",
         "--output", str(into), url],
        capture_output=True, text=True, timeout=3600)
    if finished.returncode != 0:
        raise DownloadFailed(f"{name}: curl exited {finished.returncode} "
                             f"{(finished.stderr or '').strip()[:200]}")
    return into


def install_runtime() -> bool:
    """Fetch the llama-server this platform would get, straight from the manifest."""
    manifest = json.loads((PLUGIN / "plugin.json").read_text(encoding="utf-8"))
    key = platform_key()
    wanted = [m for m in manifest["models"]
              if m["name"] == "llama-server" and m.get("platform") == key]
    # Prefer the CPU build: the machine running this has no card by assumption,
    # and a Vulkan build on a runner with no driver fails for its own reasons.
    wanted.sort(key=lambda m: 0 if m.get("backend") == "cpu" else 1)
    if not wanted:
        check(f"the manifest offers a runtime for {key}", False)
        return False

    entry = wanted[0]
    archive = fetch(entry["url"], MODELS / "runtime.archive")
    target = MODELS / "llama-server"
    target.mkdir(exist_ok=True)
    if entry["unpack"] == "zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    else:
        with tarfile.open(archive) as bundle:
            # Python 3.14 filters extracted archives by default and warns until
            # then. These come from a release page, so "data" is the right rule
            # and saying so keeps the behaviour the same across versions.
            try:
                bundle.extractall(target, filter="data")
            except TypeError:  # older Python has no filter argument
                bundle.extractall(target)
    # Archives lose the executable bit; the application restores it after
    # unpacking and so must this, or the server cannot be started.
    for path in target.rglob("*"):
        if path.is_file() and not path.suffix:
            path.chmod(path.stat().st_mode | 0o755)

    found = serve._binary()
    check("the runtime is found after unpacking", bool(found), f"looked under {target}")
    return bool(found)


def test_real_server() -> None:
    section("4. A model server that actually answers (CPU only)")
    if not install_runtime():
        return
    fetch(STAND_IN_MODEL, MODELS / "model.gguf")

    try:
        base = serve.ensure(context_tokens=16384, idle_minutes=10)
    except Exception as error:  # noqa: BLE001
        check("llama-server starts", False, f"{error}")
        log = Path(serve.SERVER_LOG)
        if log.is_file():
            sys.stderr.write("\n".join("      " + l for l in
                                       log.read_text(errors="replace").splitlines()[-20:]) + "\n")
        return
    check("llama-server starts and reports itself healthy", True)

    try:
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"model": "wunjo-local",
                             "messages": [{"role": "user", "content": "Reply with the word ready."}],
                             "max_tokens": 24}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
        text = answer["choices"][0]["message"]["content"]
        check("it answers a chat request", bool(text and text.strip()), f"got {text!r}")
        sys.stderr.write(f"          it said: {text.strip()[:120]!r}\n")
        # The one that matters for the panel: llama.cpp must put the answer in
        # `content`. With reasoning left on it lands in reasoning_content, and
        # the panel shows the user nothing at all.
        check("the answer is in content, not tucked into reasoning_content",
              not (answer["choices"][0]["message"].get("reasoning_content") and not text))
    except Exception as error:  # noqa: BLE001
        check("it answers a chat request", False, f"{error}")

    serve.stop()
    time.sleep(2)
    check("stop() leaves nothing running", serve._server_pids(str(MODELS / "model.gguf")) == [],
          "a server still holding the weights would hold the memory too")


def looks_like_language(text: str) -> bool:
    """Is this a reply, or is it noise?

    A model whose arithmetic is wrong still answers — it answers rubbish. On
    Metal one produced 83D0/%)59&#"=G?+H-:;+4G; and every check that asked only
    whether something came back was satisfied. Letters and spaces against the
    total is a crude measure and a sufficient one: prose clears it comfortably,
    a broken sampler does not come close.
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    readable = sum(1 for c in stripped if c.isalpha() or c.isspace())
    return readable / len(stripped) >= 0.7


def ask(base: str, prompt: str) -> str:
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": "wunjo-local",
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 24}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def server_log_tail(lines: int = 25) -> str:
    log = Path(serve.SERVER_LOG)
    if not log.is_file():
        return "      (no server log)"
    return "\n".join("      " + l for l in
                      log.read_text(errors="replace").splitlines()[-lines:])


def install_manifest_model() -> bool:
    """Fetch the weights the plugin actually ships, projector and all.

    The stand-in used elsewhere is a plain text model. The one the plugin
    downloads comes with a vision projector, and --mmproj is passed whenever that
    file is present — so a run without it exercises a shorter command than any
    user ever gets. That gap matters: a model server can start perfectly on the
    stand-in and die on the real pair.

    The smaller of the two variants is taken. What is under test is that the
    weights load on this backend, and the larger one only makes the download
    longer.
    """
    manifest = json.loads((PLUGIN / "plugin.json").read_text(encoding="utf-8"))
    for name in ("model.gguf", "mmproj.gguf"):
        target = MODELS / name
        entries = [m for m in manifest["models"] if m["name"] == name]
        # max_vram_gb marks the variant meant for the smaller cards.
        entries.sort(key=lambda m: (0 if m.get("max_vram_gb") else 1, m.get("size_mb", 0)))
        if not entries:
            check(f"the manifest offers {name}", False)
            return False
        # Already the full size from a previous run, most likely out of the CI
        # cache. curl would work it out from a Range request anyway; skipping
        # saves the round trip.
        expected = int(entries[0].get("size_mb", 0)) * 1024 * 1024
        if target.exists() and expected and target.stat().st_size >= expected * 0.95:
            sys.stderr.write(f"    {name} is already here ({target.stat().st_size >> 20} MB)\n")
            continue
        try:
            fetch(entries[0]["url"], target)
        except DownloadFailed as error:
            check(f"{name} could be downloaded", False, str(error))
            return False
    return True


def test_metal_server(full_model: bool = False) -> None:
    """The same model server, this time on the GPU.

    Only reachable on a Mac, and only worth running there: the flags serve.py
    builds differ once layers are offloaded, and the ones that differ are
    exactly the ones that broke. With WUNJO_GPU_BACKEND unset the server runs on
    the CPU, none of that code is reached, and a run can pass while the real
    thing refuses to start.

    It did refuse. The server died loading the model on

        ggml-metal-context.m: GGML_ASSERT(buf_dst) failed

    which reached the user as the assistant declining to start with a line of C
    where the reason should be. The cause was flags meant for a laptop with a
    discrete card — naming a device, and squeezing the attention cache to eight
    bits — asked of a backend that wants neither.
    """
    section("5. The same server on Metal (macOS only)")
    if sys.platform != "darwin":
        sys.stderr.write("  skipped — not a Mac, there is no Metal to ask for\n")
        return

    serve.stop()
    time.sleep(2)
    if full_model:
        sys.stderr.write("  using the weights from plugin.json, projector included\n")
        if not install_manifest_model():
            return
    else:
        sys.stderr.write("  using the small stand-in; it carries no vision projector, so the\n"
                         "  --mmproj the plugin passes is not exercised here\n")
    os.environ["WUNJO_GPU_BACKEND"] = "metal"
    try:
        check("serve.py asks for the layers to be offloaded", serve._gpu_layers() == 99)
        try:
            base = serve.ensure(context_tokens=16384, idle_minutes=10)
        except Exception as error:  # noqa: BLE001
            check("llama-server starts with Metal", False, str(error))
            sys.stderr.write(server_log_tail() + "\n")
            return
        check("llama-server starts with Metal", True)
        try:
            text = ask(base, "Reply with the word ready.")
            sys.stderr.write(f"          it said: {text.strip()[:120]!r}\n")
            check("it answers with the model on the GPU", bool(text and text.strip()),
                  f"got {text!r}")
            # Loading is not the same as computing. See looks_like_language.
            check("and the answer is language rather than noise",
                  looks_like_language(text),
                  "the model loaded but its output is not text — the weights are being "
                  "run wrongly on this backend")
        except Exception as error:  # noqa: BLE001
            check("it answers with the model on the GPU", False, str(error))
            sys.stderr.write(server_log_tail() + "\n")
        serve.stop()
    finally:
        os.environ.pop("WUNJO_GPU_BACKEND", None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model", action="store_true",
                        help="also download a runtime and a small model and talk to it")
    parser.add_argument("--full-model", action="store_true",
                        help="use the weights named in plugin.json, projector and all (~3.6 GB)")
    arguments = parser.parse_args()

    sys.stderr.write(f"python {sys.version.split()[0]} on {sys.platform}, no GPU assumed\n")
    test_reply_parsing()
    test_recipe()
    test_lifecycle()
    if arguments.with_model or arguments.full_model:
        test_real_server()
        test_metal_server(full_model=arguments.full_model)
    else:
        section("4. The real model server")
        sys.stderr.write("  skipped — pass --with-model to download one and talk to it\n")

    sys.stderr.write(f"\n{len(PASSED)} passed, {len(FAILED)} failed\n")
    for name in FAILED:
        sys.stderr.write(f"  failed: {name}\n")
    sys.stderr.write("\nNot covered here: Goose's own loop and the editor at the far end.\n"
                     "Neither can be reached without the application running.\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
