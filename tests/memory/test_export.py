"""Exporting what Sentient remembers as Markdown and SQLite (issue #156)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient.gateway.app import create_app
from sentient.memory import export as memory_export
from sentient.memory import review


async def _seed(app) -> None:
    mem = app.memory
    await mem.remember("Maya lives in Bengaluru.", source="onboarding", use_llm=False)
    await mem.remember("Maya is allergic to cashews.", source="import:chatgpt", use_llm=False)
    await app.store.execute("UPDATE facts SET topics = ? WHERE content LIKE '%cashews%'", (json.dumps(["Health & Wellbeing"]),))
    await mem.remember("Maya runs the Wednesday design critique.", source="import:claude", use_llm=False,
                       review=review.note("Claude", "Notes for the Wednesday critique", title="Project Design critique"))
    await app.user_model.import_insight("Maya prefers short answers.", source="import:openclaw")
    item = next(i for i in (await review.inbox(app))["items"] if i["kind"] == "insight")
    await review.approve(app, "insight", str(item["id"]))


async def test_markdown_export(app):
    await _seed(app)
    out = await memory_export.export(app, "markdown", now=datetime(2026, 10, 12, 9, 30).astimezone())
    path = Path(out["path"])
    assert path.parent.name == "exports" and path.name == "sentient-memory-20261012-093000.md" == out["name"]
    assert out | {"path": None, "folder": None} == {
        "path": None, "folder": None, "name": path.name, "format": "markdown", "memories": 2, "about_you": 1, "waiting": 1}
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# What Sentient remembers about Maya\n\nExported on 12 October 2026: 2 memories, 1 thing about Maya, 1 waiting")
    assert "## Health & Wellbeing\n\n- Maya is allergic to cashews. _(imported from ChatGPT, " in text
    assert "- Maya lives in Bengaluru. _(onboarding," in text
    assert "## About Maya\n\n- Maya prefers short answers." in text
    assert '## Waiting for your review' in text
    assert '- Maya runs the Wednesday design critique. _(imported from Claude, "Project Design critique",' in text
    again = await memory_export.export(app, "markdown", now=datetime(2026, 10, 12, 9, 30).astimezone())
    assert again["name"] == "sentient-memory-20261012-093000-2.md"  # never overwrites an earlier export


async def test_sqlite_export(app):
    await _seed(app)
    out = await memory_export.export(app, "sqlite")
    con = sqlite3.connect(out["path"])
    try:
        rows = con.execute("SELECT content, source, status FROM memories ORDER BY id").fetchall()
        about = con.execute("SELECT statement, status FROM about_you").fetchall()
    finally:
        con.close()
    assert rows == [
        ("Maya lives in Bengaluru.", "onboarding", "active"),
        ("Maya is allergic to cashews.", "import:chatgpt", "active"),
        ("Maya runs the Wednesday design critique.", "import:claude", "pending"),
    ]
    assert about == [("Maya prefers short answers.", "active")]


async def test_empty_export(app):
    out = await memory_export.export(app, "markdown")
    assert "Sentient doesn't remember anything yet." in Path(out["path"]).read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        await memory_export.export(app, "pdf")


def test_export_route(app, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    with TestClient(create_app(app)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        out = c.post("/api/memories/export", json={"format": "sqlite"}).json()
        assert out["format"] == "sqlite" and Path(out["path"]).is_file()
        assert c.post("/api/memories/export", json={"format": "csv"}).status_code == 400
