"""Screen captures go with their chat, the way selected text does (#445, follow-up to #332)."""

from __future__ import annotations

import json
import os
import time

from sentient.agent import selections
from sentient.app import SentientApp
from tests.conftest import FakeProvider
from tests.test_ask_selection import _client

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _look(client, name: str = "Mail.png", **extra) -> tuple[str, str]:
    up = client.post("/api/files", files={"file": (name, PNG, "image/png")}, data={"source": "screen"}).json()
    assert up["name"].startswith("screens/")
    r = client.post("/api/chat", json={"text": "What is on my screen?", "attachments": [up["name"]], **extra})
    return up["name"], json.loads(r.text.splitlines()[0])["session_id"]


def test_a_screen_capture_is_deleted_with_its_chat(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["One.", "Two.", "Three."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        name, sid = _look(client)
        other = client.post("/api/files", files={"file": ("notes.txt", b"mine", "text/plain")}).json()["name"]
        path = isolated_home / "files" / name
        assert path.is_file()
        assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
        assert not path.exists()
        assert (isolated_home / "files" / other).is_file()  # only captures and selections go

        # still attached in another chat: kept until that one goes too
        name, first = _look(client)
        r = client.post("/api/chat", json={"text": "And now?", "attachments": [name]})
        second = json.loads(r.text.splitlines()[0])["session_id"]
        client.delete(f"/api/sessions/{first}")
        assert (isolated_home / "files" / name).is_file()
        client.delete(f"/api/sessions/{second}")
        assert not (isolated_home / "files" / name).exists()


def test_a_screen_capture_always_goes_with_an_incognito_chat(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["One.", "Two."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        name, sid = _look(client, incognito=True)
        client.post("/api/chat", json={"text": "Same screen.", "attachments": [name]})  # a normal chat has it too
        client.delete(f"/api/sessions/{sid}")
        assert not (isolated_home / "files" / name).exists()


async def test_screen_captures_no_chat_has_are_removed_at_start(config, isolated_home):
    folder = isolated_home / "files" / "screens"
    folder.mkdir(parents=True, exist_ok=True)
    old, fresh, used = folder / "old.png", folder / "fresh.png", folder / "used.png"
    for p in (old, fresh, used):
        p.write_bytes(PNG)
    hours_ago = time.time() - 2 * selections.STRAY_AFTER_S
    for p in (old, used):
        os.utime(p, (hours_ago, hours_ago))
    s = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "scr.db", enable_background=False)
    await s.store.open()
    sid = await s.store.create_session(title="Look")
    await s.store.add_message(sid, "user", "What is this?", attachments=["screens/used.png"])
    await s.store.close()
    await s.start()
    try:
        assert not old.exists()  # its chat is gone
        assert fresh.exists()  # may be waiting for its chat to start
        assert used.exists()  # a chat still has it
    finally:
        await s.stop()
