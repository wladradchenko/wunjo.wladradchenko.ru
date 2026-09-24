"""AI plugins (.wunjoplugin): discover option schemas and run headless.

Plugins are user-installed processors (video/audio/face/generator). Each
declares its options in its manifest; list_plugins exposes those schemas so
the assistant can fill parameters and launch jobs without any dialog.
"""

from __future__ import annotations

import json
import time

from mcp.server.fastmcp import Context

#: How long run_plugin waits for a plugin before handing the job back to be
#: followed. Long enough for the analyses that take minutes — registering a
#: face, measuring a voice — and short enough that a stuck one does not hold
#: the conversation.
WAIT_SECONDS = 600


def _await_job(app, job_id: str, timeout: int) -> dict:
    """Follow a job until it ends, and answer with what the editor knows.

    The editor keeps every job's state and its last sentence; this only asks.
    Polling rather than blocking the editor is deliberate: the waiting belongs
    in this process, so the person using the application keeps a live window.
    """
    deadline = time.monotonic() + timeout
    state = {"state": "running", "percent": 0, "message": ""}
    while True:
        raw = app._call("scriptPluginJobStatus", job_id)
        try:
            state = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            state = {"state": "unknown", "percent": 0, "message": ""}
        # Asked at least once, whatever the timeout: a status question with no
        # waiting in it is still a question.
        if state.get("state") in ("done", "failed", "unknown"):
            # "unknown" means the editor never started this job, or has
            # forgotten it — waiting would be waiting for nothing.
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    state.setdefault("state", "running")
    state.setdefault("percent", 0)
    state.setdefault("message", "")
    return state


def register(mcp, helpers):

    @mcp.tool()
    def plugin_job_status(ctx: Context, job_id: str) -> str:
        """How a plugin run or a render is doing, and what it said.

        Args:
            job_id: The id run_plugin or generate_effect answered with.

        Returns "running NN%", the plugin's own closing sentence, or the reason
        it stopped. A render takes minutes: ask again rather than assuming.
        """
        try:
            app = helpers.get_resolve(ctx)._app
            state = _await_job(app, job_id, timeout=0)
            if state["state"] == "unknown":
                return f"ERROR: no job '{job_id}' in this editor."
            if state["state"] == "running":
                return f"running {state['percent']}% — {state['message']}"
            if state["state"] == "failed":
                return f"ERROR: {state['message']}"
            return state["message"] or "finished"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def list_plugins(ctx: Context) -> str:
        """Every installed plugin's own manifest, as its author wrote it.

        Read this before touching any plugin — a user can write their own, and
        the manifest is what says what it is for, what it takes and what it
        makes. Fields vary by plugin; the ones always there are id, name,
        description, kind (local/api), target and params.

        How the editor drives them:
          - a plugin that declares "effects" works THROUGH them: apply the
            effect (apply_face_effect for a face, add_effect otherwise) instead
            of running the plugin. An effect with "wunjo_requires" needs that
            effect applied first.
          - a plugin with a "sets" block reads something recorded beforehand (a
            voice, a face to swap in). Record it with
            run_plugin(action="analyse", source="/abs/file"), then name the set
            in the effect's parameter with set_effect_param.
          - only a plugin without effects is launched with run_plugin.
        """
        try:
            plugins = helpers.get_resolve(ctx)._app._call("scriptListPlugins") or []
            if not plugins:
                return "No AI plugins installed."
            return json.dumps(list(plugins), indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def plugin_status(ctx: Context, plugin_id: str) -> str:
        """Preflight a plugin BEFORE running it: is it ready, and if not, what's missing?

        Returns JSON: installed, kind, target, provider, api_key_set, venv/venv_ready,
        models[], missing[] (e.g. "api_key", "deps", "models", "plugin"), ready.

        ALWAYS call this before using a plugin. If it is not ready, do not give
        up and do not send the user to the settings: call install_plugin and it
        will be made ready. Only an API key still has to come from the user.
        """
        try:
            status = helpers.get_resolve(ctx)._app._call("scriptPluginStatus", plugin_id)
            return json.dumps(dict(status), default=str) if status else "{}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def install_plugin(ctx: Context, plugin_id: str) -> str:
        """Make a plugin ready to use: build its environment, fetch its weights.

        A plugin arrives as a manifest and a script; what it needs to actually
        work is downloaded the first time somebody wants it. Call this when
        plugin_status reports "deps" or "models" missing, then poll
        plugin_status until ready is true — it takes minutes, not seconds, and
        several gigabytes may be on their way. Tell the user it is happening.

        Args:
            plugin_id: Plugin id from list_plugins.
        """
        try:
            ok = helpers.get_resolve(ctx)._app._call("scriptInstallPlugin", plugin_id)
            if not ok:
                return f"ERROR: plugin '{plugin_id}' is not installed."
            return (f"Preparing '{plugin_id}' — building its environment and downloading "
                    "what it needs. Poll plugin_status until ready is true.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def list_plugin_sets(ctx: Context, plugin_id: str, kind: str = "") -> str:
        """What a plugin has recorded so far — faces, voices, performances.

        An effect that reads a set wants the FILE, not the name: take "file"
        from here and write it into the effect's set parameter with
        set_effect_param. Record a new one with
        run_plugin(action="analyse", source=..., kind=...).

        Args:
            plugin_id: Plugin id from list_plugins.
            kind: Narrow to one sort, using a key from the manifest's "sets"
                (e.g. "face"). Empty lists everything the plugin has.
        """
        try:
            sets = helpers.get_resolve(ctx)._app._call("scriptListPluginSets", plugin_id, kind) or []
            if not sets:
                return (f"'{plugin_id}' has nothing recorded yet. Use "
                        "run_plugin(action=\"analyse\", source=..., kind=...) first.")
            return json.dumps([dict(s) for s in sets], indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def generate_effect(ctx: Context, clip_id: int, effect_id: str) -> str:
        """Render what a plugin effect describes — the step that produces media.

        A heavy plugin (a face swap, a lip sync) does not run while the timeline
        plays: its effect holds the settings, and nothing exists until this is
        called. Applying the effect and setting its parameters changes NOTHING
        on its own. Call this last, then wait and check get_media_pool — the new
        clip lands in the project bin, the clip on the timeline is left alone.

        Args:
            clip_id: Timeline clip the effect sits on.
            effect_id: The effect's id, e.g. "face-toolkit.swap".
        """
        try:
            app = helpers.get_resolve(ctx)._app
            job_id = app._call("scriptGenerateEffect", int(clip_id), effect_id)
            if not job_id:
                return (f"ERROR: '{effect_id}' is not on clip {clip_id}, or it is not an "
                        "effect that renders.")
            # A render can also refuse before it starts — a missing preset, a
            # weight that was never downloaded — and that answer comes back at
            # once. Only a render that is really running is reported as started.
            state = _await_job(app, job_id, timeout=5)
            if state["state"] == "failed":
                return f"ERROR: {state['message']}"
            if state["state"] == "done":
                return state["message"] or f"'{effect_id}' finished."
            return (f"Rendering '{effect_id}' on clip {clip_id} (job {job_id}). It takes "
                    f"minutes; follow it with plugin_job_status(\"{job_id}\") and tell the "
                    "user it has started.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def run_plugin(ctx: Context, plugin_id: str, clip: str = "", params: dict | None = None,
                   action: str = "", source: str = "", kind: str = "") -> str:
        """Run an installed AI plugin headless (no dialogs).

        Args:
            plugin_id: Plugin id from list_plugins.
            clip: Bin id of the clip to work on (from get_media_pool). Several
                ids separated by commas when the plugin takes more than one —
                its manifest says so with "input": {"multiple": true}, and
                that is how one reel is cut from several sources. Leave empty
                only for a plugin that makes media out of nothing.
            params: Option values, using the keys list_plugins reports for this
                plugin. Anything left out keeps the plugin's own default.
            action: Use "analyse" to record a file as a set the plugin's effects
                can then use — that is how a face is registered before a swap,
                or a voice before a clone. Leave empty for an ordinary run. Do
                not pass the wording from the manifest's "sets" block here; that
                is a button caption, not a job.
            source: The file "analyse" should read, as an absolute path.
            kind: Which sort of set to record, when the plugin keeps more
                than one — the keys of its manifest's "sets" block, e.g.
                "face" or "expression". Getting this wrong records the
                wrong thing and the effect will not find what it needs.

        The job runs in the background; results appear in the project bin.
        Verify with get_media_pool / render tools when it lands.
        """
        try:
            app = helpers.get_resolve(ctx)._app
            job: dict = {}
            # A plugin knows one job besides its ordinary run: reading a file
            # into a set. The manifest describes that with a button caption —
            # "Register face", "Add audio" — and a caller reading the manifest
            # naturally passes the caption. Take it either way: what was meant
            # is unambiguous once a source file is named.
            if action and action.strip().lower() != "analyse":
                action = "analyse" if source else ""
            if action:
                job["action"] = action
            if source:
                job["source"] = source
            if kind:
                job["kind"] = kind
            wanted = [one.strip() for one in str(clip).split(",") if one.strip()]
            if wanted:
                # The plugin contract wants each file, its bin id and the range —
                # see plugins/README.md. Working that out from bin ids is this
                # tool's job, not the caller's: a plugin handed the wrong shape
                # starts, finds no input and dies without saying why.
                job["clips"] = []
                for one in wanted:
                    properties = app.get_clip_properties(one) or {}
                    path = properties.get("url") or properties.get("resource") or ""
                    if not path:
                        return f"ERROR: clip '{one}' is not in the project bin."
                    try:
                        out = max(0, int(properties.get("duration", "0")) - 1)
                    except (TypeError, ValueError):
                        out = 0
                    job["clips"].append({"bin_id": str(one), "path": path, "in": 0, "out": out})
            if params:
                job["params"] = params

            job_id = app._call("scriptRunPlugin", plugin_id, json.dumps(job))
            if not job_id:
                return f"ERROR: plugin '{plugin_id}' not found or invalid input."
            # Wait here for the plugin's own last word. A plugin declines for
            # good reasons — "choose an audio preset first", "a face preset is
            # made from a photo" — and until this waited, the answer was always
            # "started", the refusal went to a message banner nobody was reading
            # and the assistant went on to report work that never happened. The
            # waiting is done in this process, so the editor stays responsive.
            state = _await_job(app, job_id, timeout=WAIT_SECONDS)
            if state["state"] == "failed":
                return f"ERROR: {state['message']}"
            if state["state"] == "running":
                return (f"'{plugin_id}' is still running ({state['percent']}%). "
                        f"Check it with plugin_job_status(\"{job_id}\").")
            return state["message"] or f"'{plugin_id}' finished."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
