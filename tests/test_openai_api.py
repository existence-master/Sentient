"""#211: an OpenAI-compatible chat API, so other apps on this computer can talk to Sentient with its memory and tools.
Off by default, behind its own key, one named chat per ``user``, streaming and not."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.provider import ProviderFailed
from tests.conftest import FakeProvider, tool_call

TOKEN = "test-token"


@pytest.fixture
def keychain(monkeypatch):
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    return store


@pytest.fixture
def client(config, isolated_home, monkeypatch, keychain):
    """A gateway with the API on, its key made through the window's token, and a scripted model."""
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    config.openai_api.enabled = True
    llm = FakeProvider(replies=["Hello Maya.", "Second reply.", "Third reply."])
    with TestClient(create_app(SentientApp(config, llm=llm, db_path=isolated_home / "api.db",
                                           enable_background=False))) as c:
        made = c.post("/api/openai-api/key", headers={"Authorization": f"Bearer {TOKEN}"})
        assert made.status_code == 200
        c.key = made.json()["key"]
        c.llm = llm
        yield c


def _ask(c, *, key=None, **body):
    body.setdefault("model", "sentient")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return c.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key or c.key}"})


def _sessions(c) -> list[dict]:
    rows = c.get("/api/sessions", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    return [r for r in rows if r.get("channel") == "api"]


def test_off_by_default(config, isolated_home, monkeypatch, keychain):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    keychain["openai_api_key"] = "sk-sentient-x"
    with TestClient(create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "off.db",
                                           enable_background=False))) as c:
        r = c.post("/v1/chat/completions", json={"model": "sentient", "messages": [{"role": "user", "content": "hi"}]},
                   headers={"Authorization": "Bearer sk-sentient-x"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "api_off"


def test_the_key_comes_from_the_window_and_nothing_else_opens_it(client):
    assert client.key.startswith("sk-sentient-")
    assert client.post("/api/openai-api/key").status_code == 401  # making a key needs the window's token
    for key in ("wrong", TOKEN):  # the window's token is not the API key
        r = _ask(client, key=key)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_a_new_key_replaces_the_old_one(client):
    old = client.key
    new = client.post("/api/openai-api/key", headers={"Authorization": f"Bearer {TOKEN}"}).json()["key"]
    assert new != old
    assert _ask(client, key=old).status_code == 401
    assert _ask(client, key=new).status_code == 200


def test_models_lists_sentient(client):
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {client.key}"})
    assert r.status_code == 200 and [m["id"] for m in r.json()["data"]] == ["sentient"]


def test_a_reply_without_streaming(client):
    r = _ask(client, messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    assert r.status_code == 200
    out = r.json()
    assert out["object"] == "chat.completion" and out["id"].startswith("chatcmpl-")
    assert out["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Hello Maya."},
                               "finish_reason": "stop"}]
    assert set(out["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_a_streamed_reply(client):
    r = _ask(client, stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    lines = [x[len("data: "):] for x in r.text.split("\n\n") if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello Maya."
    assert chunks[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


def test_only_the_last_user_message_is_sent_into_one_named_chat_per_user(client):
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "Hello Maya."},
               {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "text", "text": "then?"}]}]
    assert _ask(client, user="Notes app").status_code == 200
    assert _ask(client, user="Notes app", messages=history).status_code == 200
    assert _ask(client).status_code == 200  # no user: the default "API" chat
    chats = {s["title"]: s["id"] for s in _sessions(client)}
    assert set(chats) == {"Notes app", "API"}
    rows = client.get(f"/api/sessions/{chats['Notes app']}/messages", headers={"Authorization": f"Bearer {TOKEN}"})
    assert [m["content"] for m in rows.json() if m["role"] == "user"] == ["hi", "and\nthen?"]


def test_a_request_without_a_user_message_is_refused(client):
    r = _ask(client, messages=[{"role": "system", "content": "Be brief."}])
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_user_message"


class Unreachable(FakeProvider):
    """The chat model can't be reached, so every turn ends in an error."""

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages})
        raise ProviderFailed("Sentient couldn't reach Ollama. Make sure it's running, then try again.")
        yield  # an async generator, like FakeProvider.stream


def test_a_failed_turn_is_shown_and_not_retried_by_the_openai_sdk(config, isolated_home, monkeypatch, keychain):
    # OpenAI's SDKs retry a 5xx twice by default, and each retry would run the turn and its tools again
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    config.openai_api.enabled = True
    llm = Unreachable()
    with TestClient(create_app(SentientApp(config, llm=llm, db_path=isolated_home / "fail.db",
                                           enable_background=False))) as c:
        key = c.post("/api/openai-api/key", headers={"Authorization": f"Bearer {TOKEN}"}).json()["key"]
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=c)
        with pytest.raises(openai.InternalServerError) as err:
            sdk.chat.completions.create(model="sentient", messages=[{"role": "user", "content": "hi"}])
        [chat] = _sessions(c)
        rows = c.get(f"/api/sessions/{chat['id']}/messages", headers={"Authorization": f"Bearer {TOKEN}"}).json()
        assert [m["content"] for m in rows if m["role"] == "user"] == ["hi"]  # one request, one turn
        assert "couldn't reach Ollama" in err.value.message
        assert err.value.response.headers["x-should-retry"] == "false"


# ---------------------------------------------------------------- approvals ask in the app (#211, option A)
WRITE = [tool_call("file_write", name="a.txt", content="one")]
WIN = {"Authorization": f"Bearer {TOKEN}"}


def _asking(client, reply: str = "Saved it.", **body):
    """Start a request whose turn writes a file, which the user is asked about first. Returns its future."""
    core = client.app.state.sentient
    core.config.tools.approvals.rules = {"files": "ask"}
    core.approvals.timeout_s = min(core.approvals.timeout_s, 20)  # a broken test fails in seconds, not 10 minutes
    client.llm.replies[:] = [WRITE, reply]
    return ThreadPoolExecutor(1).submit(_ask, client, **body)


def _approval(client) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        found = [n for n in client.get("/api/notifications", headers=WIN).json()["notifications"]
                 if n["kind"] == "approval"]
        if found:
            return found[0]
        time.sleep(0.05)
    raise AssertionError("no approval notification")


def _answer(client, note: dict, decision: str) -> None:
    r = client.post("/api/approvals", json={"approval_id": note["payload"]["approval_id"], "decision": decision},
                    headers=WIN)
    assert r.json() == {"resolved": True}


def _status(client, note: dict) -> str | None:
    [now] = [n for n in client.get("/api/notifications", headers=WIN).json()["notifications"] if n["id"] == note["id"]]
    return now["payload"].get("status")


def test_an_api_turn_asks_in_the_app_and_goes_on_when_allowed(client, isolated_home):
    pending = _asking(client, user="Notes app")
    note = _approval(client)  # the window's Allow and Deny card; the request is still waiting
    assert not pending.done()
    assert note["title"] == "Approval needed" and "Another app asked Sentient" in note["message"]
    p = note["payload"]
    assert (p["name"], p["source"], p["risk"]) == ("file_write", "api", "write") and p["arguments"]["name"] == "a.txt"
    assert p["session_id"] == {s["title"]: s["id"] for s in _sessions(client)}["Notes app"]
    _answer(client, note, "allow")
    r = pending.result(timeout=30)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Saved it."
    assert (isolated_home / "files" / "a.txt").read_text() == "one"
    assert _status(client, note) == "allow"
    items = client.get("/api/timeline", headers=WIN).json()["items"]
    assert any('From another app (API), in the chat "Notes app"' in (i.get("why") or "") for i in items)


def test_a_declined_call_is_never_claimed_done(client, isolated_home):
    pending = _asking(client, reply="I couldn't save it.")
    note = _approval(client)
    _answer(client, note, "deny")
    content = pending.result(timeout=30).json()["choices"][0]["message"]["content"]
    assert content.startswith("I couldn't save it.")
    assert content.endswith(" needed your OK in Sentient and didn't get it.") and "\n\nNot done: " in content
    assert not (isolated_home / "files" / "a.txt").exists()
    assert _status(client, note) == "deny"


def test_nobody_answering_is_a_no(client, isolated_home):
    client.app.state.sentient.approvals.timeout_s = 0.5
    pending = _asking(client, reply="I couldn't save it.")
    note = _approval(client)
    content = pending.result(timeout=30).json()["choices"][0]["message"]["content"]
    assert "\n\nNot done: " in content
    assert not (isolated_home / "files" / "a.txt").exists()
    assert _status(client, note) == "expired"


def test_stop_everything_ends_a_waiting_turn(client, isolated_home):
    pending = _asking(client)
    note = _approval(client)
    assert client.post("/api/stop-all", headers=WIN).status_code == 200
    r = pending.result(timeout=30)
    assert r.status_code == 409 and r.json()["error"] == {
        "message": "Stopped.", "type": "invalid_request_error", "param": None, "code": "stopped"}
    assert not (isolated_home / "files" / "a.txt").exists()
    assert _status(client, note) == "expired"


def test_a_streamed_turn_waits_for_the_answer_too(client, isolated_home):
    pending = _asking(client, stream=True)
    _answer(client, _approval(client), "allow")
    r = pending.result(timeout=30)
    chunks = [json.loads(x[len("data: "):]) for x in r.text.split("\n\n") if x.startswith("data: {")]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Saved it."
    assert (isolated_home / "files" / "a.txt").exists()
