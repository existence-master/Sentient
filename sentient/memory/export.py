"""One-click export of what Sentient remembers (issue #156, docs/API.md section 7).

``export(app, fmt)`` writes everything Sentient remembers about the user to a new file in ``<home>/files/exports/``:

- ``markdown``: a readable page, facts grouped by topic, then what Sentient knows about the user (insights), then what
  still waits for review. Each line says where it came from and when.
- ``sqlite``: a small database another tool can read, with ``memories`` and ``about_you`` tables (the same columns as
  Sentient's own, without embeddings or review internals).

Nothing leaves the computer; the window shows the folder afterwards.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.memory.schema import ensure_memory_schema
from sentient.memory.topics import DEFAULT_TOPIC, TOPIC_NAMES

FORMATS = {"markdown": ".md", "sqlite": ".db"}
_SOURCE_WORDS = {
    "conversation": "a chat", "manual": "added by you", "onboarding": "onboarding", "user": "you",
    "inferred": "Sentient noticed it", "dreaming": "nightly tidying",
}
_APPS = {"chatgpt": "ChatGPT", "claude": "Claude", "openclaw": "OpenClaw", "hermes": "Hermes"}


def source_words(source: str) -> str:
    """``import:chatgpt`` -> "imported from ChatGPT", ``file:resume.pdf`` -> "resume.pdf"."""
    source = str(source or "")
    kind, _, rest = source.partition(":")
    if kind == "import" and rest:
        return f"imported from {_APPS.get(rest, rest)}"
    if kind == "file" and rest:
        return rest
    if kind == "task":
        return "a task"
    return _SOURCE_WORDS.get(source, source or "Sentient")


def _day(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).strftime("%d %b %Y").lstrip("0")
    except (TypeError, ValueError):
        return ""


async def _rows(app: Any) -> tuple[list[dict], list[dict]]:
    await ensure_memory_schema(app.store)
    facts = [dict(r) for r in await app.store.fetchall(
        "SELECT id, content, source, topics, memory_type, status, created_at, updated_at, expires_at, review"
        " FROM facts ORDER BY created_at, id"
    )]
    insights = [dict(r) for r in await app.store.fetchall(
        "SELECT id, dimension, statement, confidence, status, source, created_at, updated_at, review"
        " FROM user_insights WHERE status != 'retired' ORDER BY created_at, rid"
    )]
    return facts, insights


def _topics(raw: str) -> list[str]:
    try:
        found = [t for t in json.loads(raw or "[]") if t in TOPIC_NAMES]
    except (TypeError, ValueError):
        found = []
    return found or [DEFAULT_TOPIC]


def _line(text: str, source: str, at: str | None, review: str | None = None) -> str:
    where = source_words(source)
    try:
        note = json.loads(review) if review else None
    except (TypeError, ValueError):
        note = None
    if isinstance(note, dict) and note.get("title"):
        where += f', "{note["title"]}"'
    when = _day(at)
    return f"- {text} _({where}{', ' + when if when else ''})_"


def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def markdown(user: str, facts: list[dict], insights: list[dict], now: datetime) -> str:
    name = user.strip() or "you"
    active = [f for f in facts if f["status"] == "active"]
    waiting = [f for f in facts if f["status"] == "pending"] + [i for i in insights if i["status"] == "pending"]
    known = [i for i in insights if i["status"] != "pending"]
    out = [f"# What Sentient remembers about {name}", "",
           f"Exported on {now.strftime('%d %B %Y').lstrip('0')}: {_n(len(active), 'memory', 'memories')}, "
           f"{_n(len(known), 'thing', 'things')} about {name}, {len(waiting)} waiting for review.", ""]
    for topic in TOPIC_NAMES:
        rows = [f for f in active if _topics(f["topics"])[0] == topic]
        if rows:
            out += [f"## {topic}", "", *(_line(f["content"], f["source"], f["created_at"]) for f in rows), ""]
    if known:
        out += [f"## About {name}", ""]
        out += [_line(i["statement"], i["source"], i["created_at"]) for i in known] + [""]
    if waiting:
        out += ["## Waiting for your review", "", "Sentient doesn't use these until you approve them.", ""]
        out += [_line(w.get("content") or w.get("statement"), w["source"], w["created_at"], w.get("review"))
                for w in waiting] + [""]
    if not active and not known and not waiting:
        out += ["Sentient doesn't remember anything yet.", ""]
    return "\n".join(out)


def _sqlite(path: Path, facts: list[dict], insights: list[dict]) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL, source TEXT, topics TEXT,
                memory_type TEXT, status TEXT, created_at TEXT, updated_at TEXT, expires_at TEXT, came_from TEXT);
            CREATE TABLE about_you (id TEXT PRIMARY KEY, dimension TEXT, statement TEXT NOT NULL, confidence REAL,
                status TEXT, source TEXT, created_at TEXT, updated_at TEXT, came_from TEXT);
            """
        )
        con.executemany(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(f["id"], f["content"], f["source"], f["topics"], f["memory_type"], f["status"], f["created_at"],
              f["updated_at"], f["expires_at"], f["review"]) for f in facts],
        )
        con.executemany(
            "INSERT INTO about_you VALUES (?,?,?,?,?,?,?,?,?)",
            [(i["id"], i["dimension"], i["statement"], i["confidence"], i["status"], i["source"], i["created_at"],
              i["updated_at"], i["review"]) for i in insights],
        )
        con.commit()
    finally:
        con.close()


async def export(app: Any, fmt: str, now: datetime | None = None) -> dict:
    """Write the export and return ``{path, folder, name, format, memories, about_you, waiting}``."""
    if fmt not in FORMATS:
        raise ValueError("format must be markdown or sqlite")
    now = now or datetime.now().astimezone()
    facts, insights = await _rows(app)
    folder = paths.files_dir() / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sentient-memory-{now.strftime('%Y%m%d-%H%M%S')}{FORMATS[fmt]}"
    n = 1
    while path.exists():
        n += 1
        path = path.with_name(f"sentient-memory-{now.strftime('%Y%m%d-%H%M%S')}-{n}{FORMATS[fmt]}")
    if fmt == "markdown":
        text = markdown(app.config.assistant.user_name, facts, insights, now)
        await asyncio.to_thread(path.write_text, text, encoding="utf-8")
    else:
        await asyncio.to_thread(_sqlite, path, facts, insights)
    waiting = sum(1 for f in facts if f["status"] == "pending") + sum(1 for i in insights if i["status"] == "pending")
    return {
        "path": str(path), "folder": str(folder), "name": path.name, "format": fmt,
        "memories": sum(1 for f in facts if f["status"] == "active"),
        "about_you": sum(1 for i in insights if i["status"] != "pending"), "waiting": waiting,
    }
