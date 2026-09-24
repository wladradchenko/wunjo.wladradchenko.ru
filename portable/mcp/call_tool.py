#!/usr/bin/env python3
"""Dev helper: call a single MCP tool over stdio JSON-RPC.

Usage: call_tool.py <tool_name> ['{"arg": "value"}']
Prints the tool's text content; exits 1 if the tool reported an error.
State lives in the editor, so each invocation may spawn a fresh server.
"""

import json
import pathlib
import subprocess
import sys

srv = subprocess.Popen(
    [sys.executable, str(pathlib.Path(__file__).resolve().parent / "run.py")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
)


def send(obj):
    srv.stdin.write(json.dumps(obj) + "\n")
    srv.stdin.flush()


def recv(idv):
    while True:
        line = srv.stdout.readline()
        if not line:
            raise SystemExit("ERROR: MCP server exited unexpectedly")
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("id") == idv:
            return m


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "call_tool", "version": "0"}}})
recv(1)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

tool = sys.argv[1]
args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
      "params": {"name": tool, "arguments": args}})
result = recv(2).get("result", {})
for c in result.get("content", []):
    if c.get("type") == "text":
        print(c["text"])
srv.stdin.close()
sys.exit(1 if result.get("isError") else 0)
