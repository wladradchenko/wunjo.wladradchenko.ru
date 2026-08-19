#!/usr/bin/env python3
"""The assistant that lives in the chat panel.

It is the same arrangement an outside agent works in, with one part swapped.
Claude Code talks to the editor through the MCP server over its local socket;
so does this.
Claude Code decides what to call; here that decision is made by Goose driving a
model on the user's own machine. The editor never learns which of the two it is
talking to, and neither does the MCP server — that is the whole point, and the
reason a request typed in the panel and a request typed in a terminal do the
same work.

What the model can reach is fixed by the recipe: it lists exactly one extension,
the editor's MCP server on its short tool profile, and nothing else. Goose brings
no shell, no file editor and no web of its own into that run — asked to list a
directory or fetch a URL, the model answers that it cannot. That boundary is the
reason this engine was chosen over the alternative, which offered the model a
shell regardless of what its configuration said.

Actions arrive over ``job.json → input.action``:
  chat   answer one message from the user
  stop   unload the model and let go of the graphics card
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import serve

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
# See serve.py: the editor hands the weights' location over in the environment.
MODELS_DIR = os.environ.get("WUNJO_MODELS_DIR") or os.path.join(PLUGIN_DIR, "models")

# stdout carries the plugin protocol; anything a library prints goes to stderr.
_real_print = print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)


import builtins

builtins.print = _stderr_print


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Talking back to the editor
# ---------------------------------------------------------------------------

def _editor():
    """A handle on the running editor, or None.

    The client comes from the copy of the MCP server the app ships; loading it
    from there rather than vendoring a second one means the plugin and the
    outside agent always speak to the editor through the same code. That code
    talks over the editor's local socket — a unix socket on Linux and macOS, a
    named pipe on Windows — so this behaves the same on all three, and the
    editor puts the exact name in WUNJO_SOCKET so a second copy of the
    application is never answered to by mistake.
    """
    mcp_dir = os.environ.get("WUNJO_MCP_DIR", "")
    if mcp_dir and mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)
    try:
        from api.app_client import WunjoMakeClient

        client = WunjoMakeClient()
        # Ask once before trusting it. Every later call is made through _say,
        # which swallows what it cannot do, so a handle that cannot reach the
        # editor turns the whole conversation silent with the reason nowhere:
        # no reply in the panel, no progress card ever closed, and the user's
        # own skills missing from the model's instructions. That is exactly
        # what this plugin did after the editor stopped speaking D-Bus and this
        # function went on importing the client that went with it.
        if not client.ping():
            raise ConnectionError("the editor is not answering on its scripting socket")
        return client
    except Exception as error:  # noqa: BLE001
        _stderr_print(f"no connection to the editor: {error}")
        return None


def _say(editor, method: str, *args) -> None:
    """Narrate, and never let narration break the answer."""
    if editor is None:
        return
    try:
        editor._call(method, *args)
    except Exception as error:  # noqa: BLE001
        _stderr_print(f"{method} failed: {error}")


# ---------------------------------------------------------------------------
# The agent engine
# ---------------------------------------------------------------------------

def _goose_binary() -> str:
    """The executable out of the downloaded archive, wherever it landed."""
    root = os.path.join(MODELS_DIR, "goose")
    for base, _dirs, files in os.walk(root):
        for name in ("goose", "goose.exe"):
            if name in files:
                return os.path.join(base, name)
    return ""


def _python() -> str:
    """The interpreter the MCP server runs under — this plugin's own."""
    return sys.executable


def _speech_python() -> str:
    """The editor's shared interpreter, the one Whisper was installed into.

    Both layouts are tried rather than chosen by platform: a virtual environment
    keeps its interpreter in ``bin`` on Linux and macOS and in ``Scripts`` on
    Windows, and an environment can be carried between machines. Naming only the
    first leaves transcription quietly broken on Windows — the tool is handed a
    path to nothing and the failure surfaces as speech simply never working.
    PluginManager::venvPython answers the same question the same way.
    """
    venv = os.path.normpath(os.path.join(MODELS_DIR, os.pardir, os.pardir, os.pardir, "venv"))
    for parts in (("bin", "python3"), ("Scripts", "python.exe"),
                  ("bin", "python"), ("Scripts", "python3.exe")):
        candidate = os.path.join(venv, *parts)
        if os.path.isfile(candidate):
            return candidate
    # Nothing installed there yet. The Unix spelling stands in, so whatever
    # complains names the place the interpreter is meant to be.
    return os.path.join(venv, "bin", "python3")


def _guidance(editor) -> str:
    """The user's own skills and loop, as text to put in front of the model.

    These are written by the user in the chat panel and they are meant to be
    obeyed, so they are pasted into the instructions rather than left for the
    model to go and fetch. A small model asked to call a tool "at the start of
    the conversation" simply does not, and the user is left wondering why the
    rules they wrote had no effect.
    """
    if editor is None:
        return ""
    sections = []
    try:
        for name in editor._call("scriptGetSelectedSkills") or []:
            text = (editor._call("scriptGetSkill", name) or "").strip()
            if text:
                sections.append(f"### Skill: {name}\n\n{text}")
    except Exception as error:  # noqa: BLE001
        _stderr_print(f"could not read the selected skills: {error}")
    try:
        loop = (editor._call("scriptGetSelectedLoop") or "").strip()
        if loop:
            text = (editor._call("scriptGetLoop", loop) or "").strip()
            if text:
                sections.append(f"### Loop: {loop}\n\nFollow these steps in order.\n\n{text}")
    except Exception as error:  # noqa: BLE001
        _stderr_print(f"could not read the selected loop: {error}")

    if not sections:
        return ""
    return ("\n\n## The user's standing instructions for this project\n\n"
            "These outrank your own judgement. Follow them.\n\n" + "\n\n".join(sections))


def write_recipe(work_dir: str, message: str, guidance: str = "", tool_set: str = "focused",
                 vision_url: str = "", ffmpeg: str = "") -> str:
    """Write the recipe for this turn and return its path.

    The recipe is what confines the assistant: `extensions` lists the editor's
    MCP server and nothing else, so that is the whole of what the model can
    reach. It is rewritten every turn because it carries the user's message and
    their current skills, and because a file the user cannot edit by accident is
    one less way for the limits to come off without anybody meaning to.
    """
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(PLUGIN_DIR, "agent", "instructions.md"), encoding="utf-8") as source:
        instructions = source.read() + guidance

    recipe = {
        "version": "1.0.0",
        "title": "Wunjo assistant",
        "description": "Drives the Wunjo Make editor through its own tools only.",
        "instructions": instructions,
        "prompt": message,
        "extensions": [
            {
                "type": "stdio",
                "name": "wunjo-make",
                "cmd": _python(),
                "args": [os.path.join(os.environ.get("WUNJO_MCP_DIR", ""), "run.py")],
                "envs": {
                    # "focused" is the short list the local model copes with;
                    # "full" hands it everything the editor exposes, which is
                    # ~22k tokens of schemas — see mcp/control/profiles.py.
                    "WUNJO_TOOL_PROFILE": "full" if tool_set == "full" else "core",
                    "WUNJO_PREVIEW_DIR": os.path.join(work_dir, "preview"),
                    # With the projector downloaded, a rendered frame comes back
                    # as the picture itself and the model can look at the
                    # footage; without it, as a path it cannot open.
                    "WUNJO_VISION": "1" if serve.has_vision() else "0",
                    # Describing a folder of footage is done by the tool, not by
                    # the conversation: it shows each file to the model here and
                    # hands the assistant the words. Five hundred files must not
                    # become five hundred pictures in the chat history.
                    "WUNJO_VISION_URL": vision_url if serve.has_vision() else "",
                    # Kept beside the plugin, not inside a project: footage is
                    # reused, and looking at the same file twice is wasted time.
                    "WUNJO_MEDIA_NOTES": os.path.normpath(os.path.join(MODELS_DIR, os.pardir, "media-notes")),
                    "WUNJO_FFMPEG": ffmpeg,
                    # Whisper lives in the editor's shared environment, one copy
                    # for everything that needs speech.
                    "WUNJO_SPEECH_PYTHON": _speech_python(),
                    "WUNJO_SPEECH_MODELS": os.path.join(
                        os.path.expanduser("~"), ".cache", "whisper"),
                },
                "timeout": 600,
            }
        ],
    }
    path = os.path.join(work_dir, "recipe.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)  # YAML is a superset of JSON
    return path


def _environment(base_url: str) -> dict:
    """What Goose needs to find the model, and nothing it does not need."""
    env = dict(os.environ)
    env.update({
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": "wunjo-local",
        # llama-server ignores the key; the client insists on having one.
        "OPENAI_API_KEY": "local",
        "OPENAI_HOST": base_url,
        "OPENAI_BASE_PATH": "v1/chat/completions",
        # Headless: there is nobody at a terminal to approve a tool call.
        "GOOSE_MODE": "auto",
        # No desktop secret store inside the sandbox, and nothing to keep in one.
        "GOOSE_DISABLE_KEYRING": "1",
    })
    return env


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _sessions_file(work_dir: str) -> str:
    return os.path.join(work_dir, "sessions.json")


def _session_name(chat_session: str) -> str:
    """A chat session id, in the alphabet Goose names sessions with."""
    return "wunjo-" + re.sub(r"[^A-Za-z0-9_-]", "", chat_session)[:40]


def _seen_session(work_dir: str, name: str) -> bool:
    try:
        with open(_sessions_file(work_dir), encoding="utf-8") as handle:
            return name in json.load(handle)
    except (OSError, ValueError):
        return False


def _remember_session(work_dir: str, name: str) -> None:
    try:
        with open(_sessions_file(work_dir), encoding="utf-8") as handle:
            known = json.load(handle)
    except (OSError, ValueError):
        known = {}
    known[name] = True
    with open(_sessions_file(work_dir), "w", encoding="utf-8") as handle:
        json.dump(known, handle)


# ---------------------------------------------------------------------------
# Reading the answer back
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _reply_from(output: str) -> str:
    """The assistant's prose, out of a transcript meant for a terminal.

    Goose prints a banner, then a block per tool call, then the answer. Only the
    prose is wanted here: the tool calls already appear in the chat as their own
    cards, and the banner is noise. So the banner is cut at the line that ends
    it, and the tool blocks — a rule, a line naming the tool, and its indented
    arguments — are dropped.
    """
    text = _ANSI.sub("", output)
    lines = text.splitlines()

    # Everything up to and including "goose is ready" is the banner.
    for index, line in enumerate(lines):
        if "goose is ready" in line:
            lines = lines[index + 1:]
            break

    blocks, current, in_tool = [], [], False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_tool = False
            continue
        if set(stripped) <= {"─", "-", "—"} or stripped.startswith("▸"):
            # a tool call: whatever prose came before it was thinking out loud
            if current:
                blocks.append(current)
                current = []
            in_tool = True
            continue
        if in_tool and line.startswith(" "):  # the call's arguments
            continue
        in_tool = False
        current.append(stripped)
    if current:
        blocks.append(current)

    # What the user wants is the answer, not the working: the prose after the
    # last tool call is where the assistant says what it did. Everything before
    # it is a running commentary between steps — useful to watch, unreadable to
    # arrive at. Only when there were no tools at all is the whole reply meant.
    #
    # A turn can also end on a tool call with nothing said afterwards. Silence
    # is the one answer the panel must never show, so the last thing that was
    # actually said stands in.
    for block in reversed(blocks):
        text = "\n".join(block).strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def do_chat(job: dict) -> None:
    data = job.get("input") or {}
    message = (data.get("message") or "").strip()
    if not message:
        emit('result:{"outputs":[]}')
        return

    params = job.get("params") or {}
    project = job.get("project") or {}
    # One folder per project: the conversation, its sessions and the model's
    # scratch space belong to the project the user is editing, not to the run.
    work_dir = os.path.join(project.get("data_folder") or job.get("output_dir", ""), "assistant")

    editor = _editor()
    _say(editor, "scriptChatThinking", True)
    try:
        binary = _goose_binary()
        if not binary:
            raise FileNotFoundError("the assistant's engine is not installed")
        # The tools alone are some twelve thousand tokens before anybody has said
        # anything, and one round of looking at a folder, choosing clips and
        # running a plugin adds as much again: 32k was reached in a single turn
        # and the whole answer died on "request exceeds the available context
        # size". Whatever the setting says, do not start with less room than the
        # assistant is known to need.
        context_tokens = max(int(params.get("context_tokens", 65536) or 65536), 65536)
        base_url = serve.ensure(
            context_tokens=context_tokens,
            idle_minutes=int(params.get("idle_minutes", 10) or 10),
        )
        recipe = write_recipe(work_dir, message, _guidance(editor),
                              str(params.get("tool_set", "focused") or "focused"),
                              base_url, job.get("ffmpeg", ""))

        name = _session_name(data.get("session", "default"))
        command = [binary, "run", "--recipe", recipe, "--name", name,
                   # A small model that gets stuck repeats itself; this ends the
                   # turn instead of letting it spin until the timeout.
                   "--max-tool-repetitions", "3",
                   # Room for a real edit. Looking through a folder, choosing,
                   # assembling and then running a plugin on the result is
                   # easily fifty steps, and stopping half way through leaves
                   # the project in a state nobody asked for. Still bounded:
                   # a model that has genuinely lost the thread is caught by
                   # --max-tool-repetitions and by the timeout below.
                   "--max-turns", str(max(10, int(params.get("max_steps", 60) or 60)))]
        if _seen_session(work_dir, name):
            command.append("--resume")

        # Hold the model while this turn runs: a long answer must not be cut off
        # by the watchdog deciding nobody is asking.
        serve.busy(True)
        try:
            finished = subprocess.run(command, capture_output=True, text=True, timeout=1800,
                                      cwd=work_dir, env=_environment(base_url))
        finally:
            serve.busy(False)
        _remember_session(work_dir, name)
        # The engine's own transcript, kept beside the job. A turn that ends
        # without a word is otherwise unexplainable from the outside: the panel
        # shows a failed tool card and silence, and nothing on disk says why.
        try:
            with open(os.path.join(work_dir, "assistant.log"), "w", encoding="utf-8") as handle:
                handle.write(finished.stdout or "")
                handle.write("\n--- stderr ---\n")
                handle.write(finished.stderr or "")
        except OSError:
            pass
        reply = _reply_from(finished.stdout)
        _say(editor, "scriptChatThinking", False)

        if reply:
            # The reply is posted here rather than by the model: a small model
            # that forgets to call a tool would otherwise answer into the void.
            _say(editor, "scriptChatMessage", 1, reply)
        elif finished.returncode != 0:
            detail = (finished.stderr or "").strip().splitlines()
            raise RuntimeError(detail[-1] if detail else "the assistant stopped unexpectedly")
        else:
            # Nothing at all came back. Say so, and say what the last thing that
            # happened was: a turn that ran a plugin, was told "missing input
            # clip" and then stopped left the panel showing a failed card and
            # not one word about it, which reads as the assistant ignoring the
            # user. Silence is the one answer the panel must never show.
            spoken = [line.strip() for line in (finished.stdout or "").splitlines() if line.strip()]
            last = spoken[-1][:400] if spoken else ""
            _say(editor, "scriptChatMessage", 2,
                 f"The assistant stopped without answering. The last thing it did: {last}" if last
                 else "The assistant did not answer. Try again.")
        emit('result:{"outputs":[]}')
    except Exception as error:  # noqa: BLE001
        _say(editor, "scriptChatThinking", False)
        _say(editor, "scriptChatMessage", 2, str(error))
        emit(f"info:{error}")
        raise


def do_stop(_job: dict) -> None:
    serve.stop()
    emit('result:{"outputs":[]}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    arguments = parser.parse_args()
    with open(arguments.job, encoding="utf-8") as handle:
        job = json.load(handle)

    action = (job.get("input") or {}).get("action", "chat")
    try:
        if action == "stop":
            do_stop(job)
        else:
            do_chat(job)
    except FileNotFoundError as error:
        emit('need:{"kind":"model","name":"the assistant\'s model"}')
        _stderr_print(str(error))
        return 4
    except Exception as error:  # noqa: BLE001
        _stderr_print(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
