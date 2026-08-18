"""Entry point for the MCP server — the api package is vendored next to control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from control.server import main

main()
