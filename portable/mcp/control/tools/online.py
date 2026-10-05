"""Online Resources: stock libraries and the files plugins keep in their clouds.

The editor sends every request itself, as its Online Resources tab does, so a
plugin's key never reaches the assistant. Searching and importing take a while:
the editor answers with a request number at once and the outcome is asked for
until it is there — the waiting belongs in this process, not in the editor.
"""

from __future__ import annotations

import json
import time

from mcp.server.fastmcp import Context

#: A search is one request to one service.
SEARCH_SECONDS = 60
#: A stock video can be hundreds of megabytes; past this the import goes on in
#: the editor and calling again with the same arguments keeps waiting for it.
IMPORT_SECONDS = 600


def _json(raw, fallback):
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw if raw is not None else fallback)
    except (TypeError, ValueError):
        return fallback


def _wait(app, method: str, request: int, seconds: float, step: float) -> dict:
    deadline = time.monotonic() + seconds
    answer: dict = {}
    while True:
        answer = _json(app._call(method, request), {})
        if answer.get("done") or time.monotonic() >= deadline:
            return answer
        time.sleep(step)


def _cell(text, limit: int = 80) -> str:
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _length(seconds) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds else ""


def register(mcp, helpers):

    @mcp.tool()
    def online_services(ctx: Context) -> str:
        """Where files can be taken from the internet: stock libraries (Pexels,
        Pixabay, Freesound, Internet Archive) and the user's own files that a
        plugin made in its cloud (e.g. online-toolkit — generations, also those
        whose run the editor lost when it closed).

        Searching and importing cost nothing. Returns each service's id for
        online_search and online_import, whether it can be used now, and for a
        plugin's cloud the tools it can filter by.
        """
        try:
            app = helpers.get_resolve(ctx)._app
            services = _json(app._call("scriptOnlineServices"), [])
            if not services:
                return "No online services."
            lines = ["| id | name | kind | ready | note |", "|---|---|---|---|---|"]
            tools = []
            for s in services:
                lines.append(f"| {s.get('id')} | {_cell(s.get('name'), 40)} | {s.get('kind')} | "
                             f"{'yes' if s.get('ready') else 'no'} | {_cell(s.get('note'), 120)} |")
                if s.get("tools"):
                    names = ", ".join(f"{k} ({v})" for k, v in s["tools"].items())
                    tools.append(f"{s.get('id')} tools: {names}")
            return "\n".join(lines + ([""] + tools if tools else []))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def online_search(ctx: Context, service: str, query: str = "", page: int = 1,
                      date_from: str = "", date_to: str = "", tool: str = "") -> str:
        """Search a service from online_services.

        Stock library: query is required ("city night", "calm piano"); dates and
        tool do not apply. Plugin's cloud: newest first, 50 a page; date_from and
        date_to (yyyy-MM-dd) narrow it at the server, query (words of the
        prompt) and tool (a tool id) filter the page that came back. Page N of a
        cloud is reached through page N-1, with the same dates.

        Args:
            service: An id from online_services.
            query: What to look for.
            page: 1 for the first page.
            date_from: Plugin's cloud only, first day.
            date_to: Plugin's cloud only, last day.
            tool: Plugin's cloud only, e.g. "text-to-video".
        """
        try:
            app = helpers.get_resolve(ctx)._app
            request = app._call("scriptOnlineSearch", service, query, int(page or 1), date_from, date_to)
            if not request:
                return "ERROR: the editor did not take the search."
            answer = _wait(app, "scriptOnlineSearchAnswer", request, SEARCH_SECONDS, 0.5)
            if not answer.get("done"):
                return "ERROR: the service did not answer in time; try again."
            if not answer.get("ok"):
                return f"ERROR: {answer.get('message') or 'the search failed'}"
            items = answer.get("items") or []
            page_now, pages = answer.get("page", 1), answer.get("pages", 1)
            head = f"{service}, page {page_now} of {pages}" + (" or more" if answer.get("library") and pages > page_now else "")
            if answer.get("library"):
                words = query.lower().split()
                items = [i for i in items
                         if (not tool or i.get("tool") == tool)
                         and all(w in (i.get("name") or "").lower() for w in words)]
                lines = [head, "", "| id | made | tool | status | price | text |", "|---|---|---|---|---|---|"]
                for i in items:
                    status = i.get("status")
                    if i.get("downloaded"):
                        status += ", in project folder"
                    price = i.get("price", "")
                    lines.append(f"| {i.get('id')} | {i.get('made')} | {i.get('tool')} | {status} | {price} | {_cell(i.get('name'))} |")
            else:
                lines = [head, "", "| id | name | kind | length | size | author | license | versions |",
                         "|---|---|---|---|---|---|---|---|"]
                for i in items:
                    size = f"{i.get('width')}x{i.get('height')}" if i.get("width") else ""
                    versions = ", ".join(i.get("versions") or [])
                    lines.append(f"| {i.get('id')} | {_cell(i.get('name'), 50)} | {i.get('kind')} | {_length(i.get('duration'))} | "
                                 f"{size} | {_cell(i.get('author'), 30)} | {i.get('license') or ''} | {_cell(versions, 60)} |")
            if not items:
                lines.append("(nothing on this page)")
            lines.append("")
            lines.append("online_import(service, id) brings one into the project.")
            if pages > page_now:
                lines.append(f"Next page: page={page_now + 1}.")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @mcp.tool()
    def online_import(ctx: Context, service: str, item_id: str, version: str = "") -> str:
        """Download an item of an earlier online_search into the project and add
        it to the media pool. The file goes into the project's folder; a file
        downloaded before is not downloaded again. It is not put on the
        timeline: insert_clip does that with the returned bin id.

        For a stock item, tell the user its author and license; the credit is
        also added to the project notes.

        Args:
            service: The id it was found in.
            item_id: The id from online_search.
            version: Stock video only, one of its versions ("hd", "1920x1080");
                empty picks the largest that is not bigger than the project.
        """
        try:
            app = helpers.get_resolve(ctx)._app
            request = app._call("scriptOnlineImport", service, item_id, version)
            if not request:
                return "ERROR: the editor did not take the import."
            answer = _wait(app, "scriptOnlineImportAnswer", request, IMPORT_SECONDS, 1.0)
            if not answer.get("done"):
                return (f"Still downloading ({answer.get('percent', 0)}%). Call online_import again with "
                        "the same arguments to keep waiting.")
            if not answer.get("ok"):
                return f"ERROR: {answer.get('message') or 'the import failed'}"
            lines = [f"Imported \"{answer.get('name')}\" as bin clip {answer.get('bin_id')} in folder \"{answer.get('folder')}\"."]
            if answer.get("version"):
                lines.append(f"Version: {answer['version']}.")
            if answer.get("author") or answer.get("license"):
                lines.append(f"Author: {answer.get('author') or 'unknown'}; license: {answer.get('license') or 'unknown'}"
                             + (f"; {answer['page_url']}" if answer.get("page_url") else "")
                             + ". Credit added to the project notes.")
            if answer.get("preview"):
                lines.append("This is the service's mp3 preview; the full file needs the user's own login on the Online Resources tab.")
            lines.append(f"File: {answer.get('path')}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"
