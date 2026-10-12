"""A chat keeps the model picked for it (#312): turns use it, the default answers when it can't, and says so once."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm import availability
from sentient.llm.events import Done, Notice, TextDelta
from sentient.llm.provider import ProviderError, ProviderFailed
from tests.conftest import FakeProvider, tool_call

PICKED = "lm_studio/big-model"  # a local server model: usable without a key or a call


@pytest.fixture(autouse=True)
def _fresh_tags():
    availability.forget_cache()
    yield
    availability.forget_cache()


async def start(config, isolated_home, llm, name: str) -> SentientApp:
    return await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()


async def turn(app, sid, text="hi", **kwargs):
    return [ev async for ev in app.agent.run_turn(sid, text, channel=kwargs.pop("channel", "desktop"), **kwargs)]


class FailingModel(FakeProvider):
    """The chat's model fails before it says anything; every other model answers from the script."""

    def __init__(self, failing: str, error: Exception, **kwargs):
        super().__init__(**kwargs)
        self.failing = failing
        self.error = error

    async def stream(self, role, messages, tools=None, *, model=None):
        if model == self.failing:
            self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
            raise self.error
            yield  # pragma: no cover
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


async def test_turns_use_the_chats_model_and_other_chats_follow_the_default(config, isolated_home):
    llm = FakeProvider(replies=["from the chat's model", "from the default", "from the explicit pick"])
    s = await start(config, isolated_home, llm, "use")
    try:
        picked = await s.store.create_session(channel="desktop", model=PICKED)
        default = await s.store.create_session(channel="desktop")
        events = await turn(s, picked)
        assert not any(isinstance(e, Notice) for e in events)
        await turn(s, default)
        await turn(s, picked, model="openai/explicit")  # one message's own pick wins over the chat's
        assert [c["model"] for c in llm.calls if not c.get("json")] == [PICKED, None, "openai/explicit"]
        assert (await s.store.get_session(picked))["model"] == PICKED  # the explicit pick didn't change the chat
    finally:
        await s.stop()


async def test_voice_turns_keep_the_voice_model(config, isolated_home):
    llm = FakeProvider(replies=["spoken"])
    s = await start(config, isolated_home, llm, "voice")
    try:
        sid = await s.store.create_session(channel="desktop", model=PICKED)
        await turn(s, sid, channel="voice")
        assert llm.calls[0]["role"] == "voice" and llm.calls[0]["model"] is None
    finally:
        await s.stop()


async def test_a_model_that_cant_answer_falls_back_with_one_notice(config, isolated_home):
    config.models.experimental_claude_code = False
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "Answered by the default."])
    s = await start(config, isolated_home, llm, "off")
    try:
        sid = await s.store.create_session(channel="desktop", model="claude-code/sonnet")
        events = await turn(s, sid)
        notices = [e for e in events if isinstance(e, Notice)]
        assert len(notices) == 1 and events[0] is notices[0]
        n = notices[0]
        assert n.code == "model_fallback" and n.model == "claude-code/sonnet" and n.used == "fake/primary"
        assert n.message.startswith("This chat uses Claude Sonnet, which can't answer right now, so primary is answering")
        assert "turned off" in n.message and n.session_id == sid
        assert all(c["model"] is None for c in llm.calls if not c.get("json"))
        assert isinstance(events[-1], Done) and events[-1].content == "Answered by the default."
        assert (await s.store.get_session(sid))["model"] == "claude-code/sonnet"  # kept: it comes back once usable
    finally:
        await s.stop()


async def test_a_model_that_fails_when_called_falls_back_once(config, isolated_home):
    llm = FailingModel(PICKED, ProviderFailed("Claude Code isn't signed in. Run claude and sign in."),
                       replies=[[tool_call("current_datetime")], "Done by the default."])
    s = await start(config, isolated_home, llm, "fails")
    try:
        sid = await s.store.create_session(channel="desktop", model=PICKED)
        events = await turn(s, sid)
        notices = [e for e in events if isinstance(e, Notice)]
        assert len(notices) == 1
        assert "big-model, which can't answer right now" in notices[0].message
        assert notices[0].message.endswith("Claude Code isn't signed in. Run claude and sign in.")
        assert [c["model"] for c in llm.calls if not c.get("json")] == [PICKED, None, None]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Done by the default."
    finally:
        await s.stop()


async def test_a_raw_failure_is_not_shown_and_an_explicit_pick_never_falls_back(config, isolated_home):
    llm = FailingModel(PICKED, ProviderError("litellm.APIError: {'raw': 'secret detail'}"), replies=["default"])
    s = await start(config, isolated_home, llm, "raw")
    try:
        sid = await s.store.create_session(channel="desktop", model=PICKED)
        notices = [e for e in await turn(s, sid) if isinstance(e, Notice)]
        assert notices[0].message.endswith("It didn't answer.") and "secret" not in notices[0].message
        other = await s.store.create_session(channel="desktop")
        events = await turn(s, other, model=PICKED)  # picked for this one message: strict, as before
        assert not any(isinstance(e, Notice) for e in events) and events[-1].type == "error"
    finally:
        await s.stop()


async def test_chat_model_same_as_default_is_just_the_default(config, isolated_home):
    llm = FakeProvider(replies=["ok"])
    s = await start(config, isolated_home, llm, "same")
    try:
        sid = await s.store.create_session(channel="desktop", model="fake/primary")
        await turn(s, sid)
        assert llm.calls[0]["model"] is None
    finally:
        await s.stop()


async def test_incognito_chats_keep_a_model_too(config, isolated_home):
    llm = FakeProvider(replies=["private"])
    s = await start(config, isolated_home, llm, "incog")
    try:
        sid = await s.store.create_session(channel="desktop", incognito=True, model=PICKED)
        await turn(s, sid)
        assert llm.calls[0]["model"] == PICKED
    finally:
        await s.stop()


async def test_helpers_use_the_chats_model_where_they_would_use_primary(config, isolated_home):
    config.subagents.role = "primary"
    config.chat.tool_selection = "all"  # a small local model is offered fewer tools; this test needs the helper's
    llm = FakeProvider(replies=[
        [tool_call("delegate_task", goal="Find out what time it is")],
        "It is noon.",
        "Your helper says it is noon.",
    ])
    s = await start(config, isolated_home, llm, "helpers")
    try:
        sid = await s.store.create_session(channel="desktop", model=PICKED)
        await turn(s, sid, "ask a helper")
        calls = [c for c in llm.calls if not c.get("json")]
        assert [c["model"] for c in calls] == [PICKED, PICKED, PICKED]
        assert await s.agent.helper_model(sid, "executor") is None  # a helper on its own role keeps it
        await s.store.set_session_model(sid, "claude-code/sonnet")
        assert await s.agent.helper_model(sid, "primary") is None  # Claude Code only answers chats
    finally:
        await s.stop()


# ---------------------------------------------------------------- availability
def test_labels():
    assert availability.label("claude-code/sonnet") == "Claude Sonnet"
    assert availability.label("ollama_chat/qwen3:8b") == "qwen3:8b"
    assert availability.label("openrouter/anthropic/claude-sonnet-5.5") == "anthropic/claude-sonnet-5.5"
    assert availability.label(None) == "the default model"


def test_model_ids():
    assert availability.model_id(" ollama_chat/qwen3:8b ") == "ollama_chat/qwen3:8b"
    assert availability.model_id("") is None and availability.model_id(None) is None
    for bad in ("qwen3", "ollama_chat/", "a b/c", "x/" + "y" * 201):
        with pytest.raises(ValueError):
            availability.model_id(bad)


async def test_unavailable_reasons(config, monkeypatch):
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: None)
    config.models.experimental_claude_code = False
    assert "turned off" in await availability.unavailable(config, "claude-code/sonnet")
    assert await availability.unavailable(config, "openai/gpt-5") == "No OpenAI key is set. Add it in Settings > Models."
    assert await availability.unavailable(config, "lm_studio/anything") is None
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        )
        assert await availability.unavailable(config, "ollama_chat/qwen3:8b") is None
        missing = await availability.unavailable(config, "ollama_chat/llama3.1:8b")
        assert missing.startswith("llama3.1:8b isn't downloaded on this computer.")
    availability.forget_cache()
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("down"))
        assert await availability.unavailable(config, "ollama_chat/llama3.1:8b") is None  # can't tell: try it


# ---------------------------------------------------------------- REST
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["one", "two", "three"])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.llm = llm
        yield c


def test_pick_and_clear_a_chats_model_over_rest(client):
    sid = client.post("/api/sessions", params={"model": PICKED}).json()["session_id"]
    row = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
    assert row["model"] == PICKED

    r = client.post("/api/chat", json={"text": "hello", "session_id": sid})
    assert json.loads(r.text.splitlines()[-1])["type"] == "done"
    assert client.llm.calls[-1]["model"] == PICKED

    assert client.patch(f"/api/sessions/{sid}", json={"model": "lm_studio/other"}).json() == {
        "ok": True, "model": "lm_studio/other"}
    assert client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"}).json() == {"ok": True, "title": "Renamed"}
    row = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
    assert row["model"] == "lm_studio/other" and row["title"] == "Renamed"  # a rename leaves the model alone

    assert client.patch(f"/api/sessions/{sid}", json={"model": None}).json() == {"ok": True, "model": None}
    client.post("/api/chat", json={"text": "again", "session_id": sid})
    assert client.llm.calls[-1]["model"] is None
    assert client.patch(f"/api/sessions/{sid}", json={"model": ""}).json()["model"] is None

    assert client.patch(f"/api/sessions/{sid}", json={"model": "qwen3"}).status_code == 422
    assert client.post("/api/sessions", params={"model": "not a model"}).status_code == 422
    assert client.patch("/api/sessions/nope", json={"model": PICKED}).status_code == 404


def test_new_incognito_chat_with_a_model(client):
    sid = client.post("/api/sessions", params={"model": PICKED, "incognito": True}).json()["session_id"]
    row = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
    assert row["incognito"] is True and row["model"] == PICKED


def test_the_model_change_is_published(client):
    sid = client.post("/api/sessions").json()["session_id"]
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        client.patch(f"/api/sessions/{sid}", json={"model": PICKED})
        ev = ws.receive_json()
        while ev.get("type") != "session.updated":
            ev = ws.receive_json()
        assert ev["data"] == {"session_id": sid, "model": PICKED}
