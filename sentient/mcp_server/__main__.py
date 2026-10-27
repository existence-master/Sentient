"""``python -m sentient.mcp_server``: the same as ``sentient mcp``."""

from __future__ import annotations

import sys

from sentient.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
