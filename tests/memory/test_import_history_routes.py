from __future__ import annotations

import functools
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


def _make_docx(path):
    import docx

    d = docx.Document()
    d.add_paragraph("I am a backend engineer who knows Rust.")
    d.save(str(path))


def _make_pdf(path):
    # minimal single-page PDF with one text line (pypdf can extract it)
    text = "I am a backend engineer who knows Rust."
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


@pytest.mark.parametrize("suffix", [".txt", ".docx", ".pdf"])
async def test_import_document_adds_without_wiping(app, tmp_path, suffix):
    mem, llm = app.memory, app.fake
    keep = await mem.remember("Sarthak lives in Pune", use_llm=False)
    path = tmp_path / f"resume{suffix}"
    if suffix == ".txt":
        path.write_text("I am a backend engineer who knows Rust.", encoding="utf-8")
    elif suffix == ".docx":
        _make_docx(path)
    else:
        _make_pdf(path)
    llm.json_replies.append({"facts": ["Sarthak is a backend engineer", "Sarthak knows Rust"]})
    for content in ("Sarthak is a backend engineer", "Sarthak knows Rust"):
        llm.json_replies.append(
            {"action": "ADD", "fact_id": None, "content": content,
             "analysis": {"topics": ["Work & Learning"], "memory_type": "long-term", "duration": None}}
        )
    out = await mem.import_document(path, username="Sarthak")
    assert out == {"added": 2, "updated": 0, "skipped": 0, "source": f"file:resume{suffix}"}
    extraction_input = [c for c in llm.calls if c.get("json")][-3]["messages"][1]["content"]
    assert "backend engineer" in extraction_input
    assert await mem.get_fact(keep["id"]) is not None
    assert len(await mem.list_facts(source=f"file:resume{suffix}")) == 2


async def test_summaries_job_marks_messages(app):
    store, llm = app.store, app.fake
    app.config.memory.summary_chunk_messages = 6
    sid = await store.create_session()
    old = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    for i in range(4):
        await store.add_message(sid, "user", f"question {i} about the marketing plan")
        await store.add_message(sid, "assistant", f"answer {i}")
    await store.execute("UPDATE messages SET created_at = ?", (old,))
    await store.execute("UPDATE sessions SET updated_at = ?", (old,))
    llm.text_replies = ["Sarthak asked me about the marketing plan and I answered.", "We wrapped up the plan."]
    created = await app.evolution.summarize_tick()
    # 8 messages, chunks of 6: one full chunk + an idle partial chunk of 2
    assert len(created) == 2
    left = await store.fetchone("SELECT COUNT(*) AS n FROM messages WHERE summarized = 0")
    assert left["n"] == 0
    assert "Sarthak" in llm.text_calls[0]["messages"][0]["content"]
    hits = await app.memory.episodic.search("marketing plan")
    assert hits and hits[0]["session_id"] == sid
    log = await store.fetchall("SELECT kind FROM evolution_log")
    assert [r["kind"] for r in log] == ["summary_created", "summary_created"]
    # recent messages are not summarized
    await store.add_message(sid, "user", "fresh message")
    await store.add_message(sid, "assistant", "fresh reply")
    assert await app.evolution.summarize_tick() == []


async def test_history_tools(app):
    store = app.store
    reg = app.registry
    ctx = app.agent.tool_context(None, "desktop")
    sid = await store.create_session()
    await store.add_message(sid, "user", "what's the wifi password for the blue router?")
    await store.add_message(sid, "assistant", "It is on the sticker")
    kw = await reg.get("memory_search_history").call(ctx, {"query": "what's the blue router"})
    assert kw and "blue router" in kw[0]["text"]
    today = datetime.now(UTC).date().isoformat()
    ts = await reg.get("history_time_search").call(ctx, {"start_date": today, "end_date": today})
    assert ts["count"] == 2 and "sticker" in ts["conversation"]
    bad = await reg.get("history_time_search").call(ctx, {"start_date": "yesterday", "end_date": "today"})
    assert "error" in bad
    await app.memory.episodic.add_summary(sid, "I helped Sarthak find his router password.", [
        {"id": "x", "created_at": "2026-09-01T10:00:00+00:00"}, {"id": "y", "created_at": "2026-09-01T10:05:00+00:00"},
    ])
    sem = await reg.get("history_semantic_search").call(ctx, {"query": "router password"})
    assert sem["summaries"][0]["summary"].startswith("I helped")
    names = {t.name for t in reg.tools()}
    assert {"memory_recall", "memory_remember", "memory_forget", "memory_search_by_source",
            "memory_search_history", "history_semantic_search", "history_time_search"} <= names


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "r.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.llm = llm
        c.core = core
        yield c


def test_memory_routes(client):
    llm = client.llm
    llm.json_replies.append(
        {"action": "ADD", "fact_id": None, "content": "Sarthak's sister is Riya",
         "analysis": {"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": None}}
    )
    created = client.post("/api/memories", json={"content": "Sarthak's sister is Riya"}).json()
    assert created["action"] == "ADD"
    mid = created["id"]
    listed = client.get("/api/memories").json()
    assert listed[0]["topics"] == ["Relationships & Social Life"] and listed[0]["memory_type"] == "long-term"
    assert client.get("/api/memories", params={"topic": "Financial"}).json() == []
    topics = client.get("/api/memories/topics").json()
    assert len(topics) == 8 and next(t for t in topics if t["name"] == "Relationships & Social Life")["count"] == 1

    llm.json_replies.append({"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": None})
    upd = client.put(f"/api/memories/{mid}", json={"content": "Sarthak's younger sister is Riya"}).json()
    assert upd["id"] == mid and upd["content"] == "Sarthak's younger sister is Riya"
    assert client.put("/api/memories/9999", json={"content": "x"}).status_code == 404

    graph = client.get("/api/memories/graph").json()
    assert graph["nodes"][0]["id"] == mid and graph["links"] == []

    llm.json_replies.append({"facts": ["Sarthak enjoys sailing"]})
    llm.json_replies.append(
        {"action": "ADD", "fact_id": None, "content": "Sarthak enjoys sailing",
         "analysis": {"topics": ["Interests & Lifestyle"], "memory_type": "long-term", "duration": None}}
    )
    imp = client.post("/api/memories/import", files={"file": ("notes.md", b"I enjoy sailing.", "text/markdown")}).json()
    assert imp == {"added": 1, "updated": 0, "skipped": 0, "source": "file:notes.md"}
    assert client.post("/api/memories/import", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.delete("/api/memories/source/file:notes.md").json() == {"deleted": 1}

    assert client.get("/api/memories/summaries").json() == []
    ws = client.get("/api/memories/workspace").json()
    assert set(ws) == {"soul", "user", "memory", "today", "yesterday"}
    assert client.put("/api/memories/workspace/memory", json={"content": "# Long-term memory\n- x"}).json() == {"saved": True}
    personas = client.get("/api/memories/personas").json()
    assert {p["id"] for p in personas} >= {"friendly", "professional"} and "{name}" not in personas[0]["soul_md"]

    assert client.delete(f"/api/memories/{mid}").json() == {"deleted": True}
    assert client.delete(f"/api/memories/{mid}").status_code == 404
    core = client.core
    assert client.portal.call(functools.partial(core.memory.count)) == 0
