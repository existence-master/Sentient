"""A freed upload name is never handed out again (#458).

A chat shows its attachments by name. When a file goes (an incognito chat's capture is deleted with it, or the user
deletes a file) while an older chat still has its name, the next upload must not take that name, or the older chat
would show the new, unrelated file."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.channels.base import save_upload
from sentient.files import names
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider

FIRST = b"\x89PNG\r\n\x1a\n" + b"first capture" * 4
SECOND = b"\x89PNG\r\n\x1a\n" + b"second, unrelated capture" * 4


def _client(config, home, monkeypatch, llm: FakeProvider):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=llm, db_path=home / "names.db", enable_background=False))
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer test-token"})
    return client


def _share(client, source: str, data: bytes) -> str:
    filename, mime = ("Inbox - Mail.png", "image/png") if source == "screen" else ("Notepad.txt", "text/plain")
    up = client.post("/api/files", files={"file": (filename, data, mime)}, data={"source": source})
    assert up.status_code == 200
    return up.json()["name"]


def _chat(client, name: str, **extra) -> str:
    r = client.post("/api/chat", json={"text": "What is this?", "attachments": [name], **extra})
    return json.loads(r.text.splitlines()[0])["session_id"]


def _attached(client, sid: str) -> list[str]:
    rows = client.get(f"/api/sessions/{sid}/messages").json()
    return [a for m in rows if m["role"] == "user" for a in m.get("attachments") or []]


@pytest.mark.parametrize("source", ["screen", "selection"])
def test_an_older_chat_never_shows_a_newer_capture(config, isolated_home, monkeypatch, source):
    llm = FakeProvider(replies=["One.", "Two.", "Three."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        # a normal chat and an incognito chat share capture A
        a = _share(client, source, FIRST)
        old = _chat(client, a)
        secret = _chat(client, a, incognito=True)
        # the incognito chat is deleted, and A with it (#332 for selections; screens too once #452 lands, or the
        # user deletes it)
        client.delete(f"/api/sessions/{secret}")
        client.delete(f"/api/files/{a}")
        assert not (isolated_home / "files" / a).exists()

        # a new share of the same window never takes A's name
        b = _share(client, source, SECOND)
        assert b != a
        assert hashlib.sha256((isolated_home / "files" / b).read_bytes()).hexdigest() == hashlib.sha256(SECOND).hexdigest()
        new = _chat(client, b)

        # the old chat still has A, which is gone: it shows "no longer available", never the new capture
        assert _attached(client, old) == [a]
        assert client.get(f"/api/files/content/{a}").status_code == 404
        assert client.head(f"/api/files/content/{a}").status_code == 404
        assert client.head(f"/api/files/content/{b}").status_code == 200

        # the old chat doesn't keep the new capture: it goes with its own chat
        client.delete(f"/api/sessions/{new}")
        if source == "selection":
            assert not (isolated_home / "files" / b).exists()


def test_a_freed_upload_name_still_attached_is_not_reused(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["A report."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        first = client.post("/api/files", files={"file": ("report.png", FIRST, "image/png")}).json()["name"]
        assert first == "uploads/report.png"  # uploads keep their own name
        _chat(client, first)
        client.delete(f"/api/files/{first}")
        again = client.post("/api/files", files={"file": ("report.png", SECOND, "image/png")}).json()["name"]
        assert again == "uploads/report (1).png"
        # a name no message has is free again once its file is gone
        loose = client.post("/api/files", files={"file": ("loose.txt", b"x", "text/plain")}).json()["name"]
        client.delete(f"/api/files/{loose}")
        assert client.post("/api/files", files={"file": ("loose.txt", b"y", "text/plain")}).json()["name"] == loose


async def test_a_file_from_a_messaging_app_never_takes_a_name_a_chat_has(config, isolated_home):
    s = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "ch.db", enable_background=False)
    await s.store.open()
    try:
        sid = await s.store.create_session(title="Telegram")
        # stored with a backslash and other letter case: still the same file
        await s.store.add_message(sid, "user", "Look.", attachments=["uploads\\Photo.jpg"])
        assert await names.referenced(s.store, "uploads/photo.jpg")
        assert not await names.referenced(s.store, "uploads/photo (1).jpg")
        name = await save_upload("photo.jpg", SECOND, s.store)
        assert name == "uploads/photo (1).jpg"
        assert (isolated_home / "files" / name).read_bytes() == SECOND
    finally:
        await s.store.close()


async def test_captures_always_get_their_own_name(isolated_home):
    folder = isolated_home / "files" / "screens"
    first = await names.create(folder, "Screen region.png", unique=True)
    second = await names.create(folder, "Screen region.png", unique=True)
    assert first != second and first.is_file() and second.is_file()
    assert first.name.startswith("Screen region ") and first.suffix == ".png"
