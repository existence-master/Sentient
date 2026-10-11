"""Keep a seeded demo home quiet, so running the app on it can't reach anyone (issue #302).

    .venv/Scripts/python.exe desktop/scripts/seed_safety.py <SENTIENT_HOME>

Pauses every task (``enabled = 0``: the scheduler and triggers skip it), turns off every paired chat's delivery
switch and turns messaging apps off in config.yaml (``channels.enabled: false``), so no seeded task runs and nothing
is sent through Telegram, WhatsApp or Discord. The seed scripts call ``quiet()`` when they finish and
``scripts/smoke.mjs`` runs this file before it starts the app; both skip it with ``--keep-live``.

It refuses the default data folder (``~/.sentient``): that is the real user's, never a demo.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEP_LIVE_FLAG = "--keep-live"
KEEP_LIVE_HELP = "keep seeded tasks active and messaging deliveries on (by default they are paused, see seed_safety.py)"


def _table_has(db: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in db.execute(f"PRAGMA table_info({table})"))


def quiet(home: str | Path) -> dict:
    """Pause the tasks and turn off channel deliveries in ``home``. Returns what changed."""
    home = Path(home).expanduser().resolve()
    os.environ["SENTIENT_HOME"] = str(home)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from sentient import paths, secrets
    from sentient.config.loader import load_config, save_config

    if secrets.is_default_home(home):
        raise SystemExit(f"seed_safety: {home} is the default data folder (the real one); refusing to change it.")
    report = {"tasks_paused": 0, "chats_muted": 0, "channels_off": False}
    if paths.config_file().exists() or paths.db_file().exists():
        cfg = load_config()
        if cfg.channels.enabled:
            cfg.channels.enabled = False
            save_config(cfg)
            report["channels_off"] = True
    if paths.db_file().exists():
        db = sqlite3.connect(paths.db_file())
        try:
            with db:
                if _table_has(db, "tasks", "enabled"):
                    report["tasks_paused"] = db.execute("UPDATE tasks SET enabled = 0 WHERE enabled != 0").rowcount
                if _table_has(db, "channel_chats", "deliver"):
                    report["chats_muted"] = db.execute(
                        "UPDATE channel_chats SET deliver = 0 WHERE deliver != 0").rowcount
        finally:
            db.close()
    return report


def announce(report: dict) -> None:
    print(f"quiet demo home: {report['tasks_paused']} tasks paused, {report['chats_muted']} chat deliveries off, "
          f"messaging apps {'turned off' if report['channels_off'] else 'already off'} "
          f"(pass {KEEP_LIVE_FLAG} to keep them live)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: seed_safety.py <SENTIENT_HOME>")
    announce(quiet(sys.argv[1]))
