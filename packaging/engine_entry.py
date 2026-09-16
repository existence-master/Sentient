"""Entry point for the frozen engine executable (``sentient-engine.exe``).

The desktop app normally runs ``python -m sentient serve``. In a packaged build
there is no Python on the user's machine, so electron-builder ships this frozen
copy of the engine instead and the shell spawns::

    sentient-engine.exe serve --host 127.0.0.1 --port <port>

Running it with no arguments also serves, so a curious user who double-clicks
the executable gets the engine rather than a help screen.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys

# Commands understood by sentient.cli. Anything else on argv[1] is treated as an
# option for `serve` (so `sentient-engine.exe --port 7777` behaves sensibly).
_COMMANDS = {"serve", "node", "doctor", "config", "version"}


def main() -> None:
    # PyInstaller re-executes the bundle for child processes; without this a
    # stray multiprocessing/loky import would fork whole extra engines.
    multiprocessing.freeze_support()

    os.environ.setdefault("SENTIENT_FROZEN", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # A frozen console app inherits pipes from Electron; make them line friendly.
    for stream in (sys.stdout, sys.stderr):
        # A windowed build has no streams to reconfigure; that is fine.
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    args = sys.argv[1:]
    if not args or args[0] not in _COMMANDS:
        sys.argv = [sys.argv[0], "serve", *args]

    from sentient.cli import app as cli

    cli()


if __name__ == "__main__":
    main()
