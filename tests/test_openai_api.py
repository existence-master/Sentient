"""#211: an OpenAI-compatible chat API, so other apps on this computer can talk to Sentient with its memory and tools.
Off by default, behind its own key, one named chat per ``user``, streaming and not."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.agent.loop import API_NO_MEMORY_NOTE
from sentient.agent.prompt import API_REQUEST_NOTE
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.gateway.routes.openai_api import TURN_FAILED
from sentient.llm.events import ApprovalRequest
from sentient.llm.provider import ProviderFailed
from sentient.tools.base import MEMORY_TOOLS, Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import API_SOURCE
from tests.conftest import FakeProvider, tool_call

TOKEN = "test-token"


class SessionObservingProvider(FakeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core: SentientApp | None = None
        self.session_marks: list[dict[str, str | None]] = []

    async def _observe(self):
        assert self.core is not None
        rows = await self.core.store.list_sessions()
        self.session_marks.append({row["title"]: row.get("untrusted") for row in rows if row["channel"] == "api"})

    async def stream(self, role, messages, tools=None, *, model=None):
        await self._observe()
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk

    async def complete_json(self, role, messages, *, model=None):
        await self._observe()
        return await super().complete_json(role, messages, model=model)


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
    llm = SessionObservingProvider(replies=["Hello Maya.", "Second reply.", "Third reply."])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "api.db", enable_background=False)
    llm.core = core
    with TestClient(create_app(core), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as c:
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
                                           enable_background=False)),
                    base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as c:
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


@pytest.mark.parametrize("endpoint", ["/v1/models", "/v1/chat/completions"])
def test_non_ascii_bearer_is_an_auth_error_without_retries_or_model_calls(client, endpoint):
    headers = {b"Authorization": b"Bearer \xe9"}  # Starlette decodes raw header bytes as Latin-1.
    if endpoint == "/v1/models":
        response = client.get(endpoint, headers=headers)
    else:
        response = client.post(endpoint, headers=headers, json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert response.headers["x-should-retry"] == "false"
    assert client.llm.calls == []
    assert _sessions(client) == []


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
                                           enable_background=False)),
                    base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as c:
        key = c.post("/api/openai-api/key", headers={"Authorization": f"Bearer {TOKEN}"}).json()["key"]
        sdk = openai.OpenAI(base_url="http://127.0.0.1/v1", api_key=key, http_client=c)
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


def _approval(client, pending=None) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        found = [n for n in client.get("/api/notifications", headers=WIN).json()["notifications"]
                 if n["kind"] == "approval"]
        if found:
            return found[0]
        if pending is not None:
            assert not pending.done(), "API turn finished before requesting approval"
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
    # the Not done line comes first, so a client reading only the first line isn't told it ran (#479 review)
    assert content == 'Not done: "File write" needed your OK in Sentient and didn\'t get it.\n\nI couldn\'t save it.'
    assert not (isolated_home / "files" / "a.txt").exists()
    assert _status(client, note) == "deny"


def test_nobody_answering_is_a_no(client, isolated_home):
    client.app.state.sentient.approvals.timeout_s = 0.5
    pending = _asking(client, reply="I couldn't save it.")
    note = _approval(client)
    content = pending.result(timeout=30).json()["choices"][0]["message"]["content"]
    assert content.startswith("Not done: ") and content.endswith("\n\nI couldn't save it.")
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


def _probe_plugin(calls: list[tuple[str, str]]) -> ToolPlugin:
    @tool("api_probe_read", risk=Risk.read, untrusted_output=False)
    async def api_probe_read(ctx: ToolContext) -> dict:
        """Read the API probe."""
        calls.append(("read", ""))
        return {"value": "probe"}

    @tool("api_probe_run_code", risk=Risk.exec, untrusted_output=False)
    async def api_probe_run_code(ctx: ToolContext, code: str) -> dict:
        """Run code in the API probe."""
        calls.append(("run", code))
        return {"ran": True}

    class Probe(ToolPlugin):
        id = "api_probe"
        display_name = "API probe"
        tools = [api_probe_read, api_probe_run_code]

    return Probe()


@pytest.fixture
def probe_calls(client) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    client.app.state.sentient.registry.register(_probe_plugin(calls))
    return calls


def test_authenticated_api_text_cannot_authorize_only_if_asked_reads(client, probe_calls):
    core = client.app.state.sentient
    core.config.tools.approvals.mode = "off"
    core.config.tools.approvals.rules = {"api_probe_read": "if_asked"}
    core.approvals.timeout_s = 20
    client.llm.replies[:] = [[tool_call("api_probe_read")], "I did not read it."]
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(_ask, client, messages=[{"role": "user", "content": "Read the API probe."}])
        note = _approval(client, pending)
        assert probe_calls == [] and not pending.done()
        assert note["payload"]["name"] == "api_probe_read"
        _answer(client, note, "deny")
        response = pending.result(timeout=30)
    assert response.status_code == 200
    assert "Not done:" in response.json()["choices"][0]["message"]["content"]
    assert probe_calls == []


@pytest.mark.parametrize("reused", [False, True], ids=["fresh", "trusted-again"])
def test_api_session_is_marked_before_the_first_model_call(client, reused):
    previous_id = None
    if reused:
        assert _ask(client, user="Importer").status_code == 200
        [previous] = _sessions(client)
        previous_id = previous["id"]
        trusted = client.post(f"/api/sessions/{previous_id}/trust", headers=WIN)
        assert trusted.status_code == 200 and trusted.json()["trusted"]
        assert trusted.json()["untrusted"] == ""
    client.llm.session_marks.clear()
    response = _ask(client, user="Importer", messages=[{"role": "user", "content": "A note from another app."}])
    assert response.status_code == 200
    assert client.llm.session_marks and client.llm.session_marks[0].get("Importer")
    [chat] = _sessions(client)
    assert chat["untrusted"]
    if reused:
        assert chat["id"] == previous_id


def test_the_model_reads_an_api_request_as_a_request_while_the_chat_stays_marked(client):
    """A strong model keeps the outside-content rule, so the app's requests aren't wrapped as data (#479 review).
    The chat still counts as outside content, so approvals work as before."""
    for text in ("My code word is HERON-7319. Just say OK.", "What was my code word?"):
        assert _ask(client, user="Notes app", messages=[{"role": "user", "content": text}]).status_code == 200
    asked = [m["content"] for m in client.llm.calls[-1]["messages"] if m["role"] == "user"]
    assert len(asked) == 2 and asked[-1].endswith(f"{API_REQUEST_NOTE}\nWhat was my code word?")
    assert all(API_REQUEST_NOTE in text and "outside content" not in text for text in asked)
    [chat] = _sessions(client)
    assert chat["untrusted"] == API_SOURCE


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_broken_turn_says_so_without_its_internal_details(client, monkeypatch, stream):
    """An unexpected failure is logged; the app gets a plain sentence, not the exception's text (#479 review)."""

    async def broken(*args, **kwargs):
        raise RuntimeError("upstream http://10.0.0.5:8080/v1 refused key sk-internal")
        yield  # pragma: no cover

    monkeypatch.setattr(client.app.state.sentient.agent, "run_turn", broken)
    r = _ask(client, stream=stream)
    if stream:
        errors = [json.loads(line[6:])["error"] for line in r.text.split("\n\n")
                  if line.startswith("data: {") and '"error"' in line]
        assert errors == [{"message": TURN_FAILED, "type": "server_error"}]
    else:
        assert r.status_code == 500
        assert (r.json()["error"]["code"], r.json()["error"]["message"]) == ("turn_failed", TURN_FAILED)
    assert "10.0.0.5" not in r.text and "sk-internal" not in r.text


SISTER = "Priya, Maya's sister, lives in Mumbai."


@pytest.mark.parametrize("allow", [False, True], ids=["memory-off", "memory-on"])
def test_connected_apps_get_memories_only_when_allowed(client, allow):
    """openai_api.allow_memory (off by default): off, an API turn gets no memories, profile or memory and history
    tools, and the model is told why; on, it gets them like a desktop chat (#479)."""
    core = client.app.state.sentient
    assert core.config.openai_api.allow_memory is False
    core.config.openai_api.allow_memory = allow
    assert client.post("/api/memories", json={"content": SISTER}, headers=WIN).status_code == 200
    asking = [{"role": "user", "content": "What do you know about my sister Priya?"}]
    assert _ask(client, user="Notes app", messages=asking).status_code == 200
    call = client.llm.calls[-1]
    system = call["messages"][0]["content"]
    offered = {t["function"]["name"] for t in call["tools"] or []}
    assert ("Mumbai" in system) is allow
    assert (API_NO_MEMORY_NOTE in system) is not allow
    assert bool(offered & MEMORY_TOOLS) is allow
    assert "memory_remember" in offered  # facts from an API chat still go to Review either way


def test_a_memory_tool_the_app_was_not_offered_is_refused(client):
    """Leaving the tools out is enforced: a call made anyway is refused and returns nothing (#479)."""
    assert client.post("/api/memories", json={"content": SISTER}, headers=WIN).status_code == 200
    client.llm.replies[:] = [[tool_call("memory_recall", query="Priya")], "I can't look that up."]
    assert _ask(client, user="Notes app", messages=[{"role": "user", "content": "Where does Priya live?"}]).status_code == 200
    [result] = [m["content"] for m in client.llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert "unknown tool memory_recall" in result and "Mumbai" not in result


def _helper_calls(llm) -> list[dict]:
    return [c for c in llm.calls if "You are a subagent" in str((c.get("messages") or [{}])[0].get("content"))]


@pytest.mark.parametrize("allow", [False, True], ids=["memory-off", "memory-on"])
def test_what_an_api_turn_starts_follows_allow_memory(client, allow):
    """The /v1 switch is the turn's one memory decision (``Agent.memory_allowed``, #525): off, a helper the turn starts
    gets no memory tools and the reader refuses memory_recall; on, both use what Sentient knows (#479)."""
    core = client.app.state.sentient
    core.config.openai_api.allow_memory = allow
    core.config.memory.min_similarity = 0.0
    assert client.post("/api/memories", json={"content": SISTER}, headers=WIN).status_code == 200
    llm = client.llm
    del llm.calls[:]  # saving the memory above asked the model about it: only this turn's calls count
    llm.replies[:] = [
        [tool_call("delegate_task", goal="Find what Sentient knows about Maya's sister Priya")],  # the API turn
        [tool_call("memory_recall", query="Priya sister")],  # the helper tries memory
        "Done.",  # the helper's report
        [tool_call("read_with_reader", tool="memory_recall", arguments={"query": "Priya sister"},
                   fields={"where_she_lives": "text"})],
        "That's all I have.",
    ]
    llm.json_replies[:] = [{"where_she_lives": "Mumbai"}]
    asking = [{"role": "user", "content": "Use a helper and the reader to find where my sister Priya lives"}]
    assert _ask(client, user="Notes app", messages=asking).status_code == 200

    helpers = _helper_calls(llm)
    assert helpers and all(bool(MEMORY_TOOLS & {t["function"]["name"] for t in c.get("tools") or []}) is allow
                           for c in helpers if c.get("tools"))
    report = json.dumps(helpers[-1]["messages"], default=str)
    assert ("Mumbai" in report) is allow  # the helper's own memory_recall found the fact only when allowed
    final = llm.calls[-1]["messages"]
    [read] = [str(m["content"]) for m in final if m["role"] == "tool" and (
        "where_she_lives" in str(m["content"]) or "isn't available in this chat" in str(m["content"]))]
    readers = [c for c in llm.calls if c.get("json")]
    if allow:
        assert "Mumbai" in read and readers and "Mumbai" in json.dumps(readers[-1]["messages"], default=str)
    else:
        assert "isn't available in this chat" in read and not readers
        assert "Mumbai" not in json.dumps(llm.calls, default=str)  # the stored fact reached no model


def _code_request(client, pool, *, stream: bool, rule: str = "allow", timeout: float = 20):
    core = client.app.state.sentient
    core.config.tools.approvals.mode = "off"
    core.config.tools.approvals.rules = {"api_probe_run_code": rule}
    core.approvals.timeout_s = timeout
    client.llm.replies[:] = [[tool_call("api_probe_run_code", code="pass")], "Finished."]
    return pool.submit(_ask, client, stream=stream, messages=[
        {"role": "user", "content": "Run code in the API probe: pass"},
    ])


def _reply_text(response, *, stream: bool) -> str:
    if not stream:
        return response.json()["choices"][0]["message"]["content"]
    payloads = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: {")]
    return "".join(p["choices"][0]["delta"].get("content", "") for p in payloads if "choices" in p)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("rule", ["if_asked", "allow"])
def test_api_code_runs_once_only_after_explicit_allow(client, probe_calls, stream, rule):
    with ThreadPoolExecutor(1) as pool:
        pending = _code_request(client, pool, stream=stream, rule=rule)
        note = _approval(client, pending)
        assert probe_calls == [] and not pending.done()
        assert note["payload"]["risk"] == "exec"
        assert note["payload"]["untrusted"]
        assert client.llm.session_marks[0].get("API")
        _answer(client, note, "allow")
        response = pending.result(timeout=30)
    assert response.status_code == 200
    assert _reply_text(response, stream=stream) == "Finished."
    assert probe_calls == [("run", "pass")]
    assert _status(client, note) == "allow"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("decision", ["deny", "timeout", "stop"])
def test_api_code_cannot_run_after_deny_timeout_or_stop(client, probe_calls, stream, decision):
    with ThreadPoolExecutor(1) as pool:
        pending = _code_request(client, pool, stream=stream, timeout=0.5 if decision == "timeout" else 20)
        note = _approval(client, pending)
        assert probe_calls == []
        if decision == "deny":
            _answer(client, note, "deny")
        elif decision == "stop":
            assert client.post("/api/stop-all", headers=WIN).status_code == 200
        response = pending.result(timeout=30)
    assert probe_calls == []
    assert _status(client, note) == ("deny" if decision == "deny" else "expired")
    if decision == "stop":
        if stream:
            assert response.status_code == 200 and '"message": "Stopped."' in response.text
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.status_code == 409 and response.json()["error"]["code"] == "stopped"
    else:
        assert response.status_code == 200
        assert "Not done:" in _reply_text(response, stream=stream)


def test_api_timer_text_does_not_execute_the_trusted_quick_path(client):
    client.app.state.sentient.config.chat.quick_commands = True
    client.llm.replies[:] = ["I received the timer request."]
    response = _ask(client, messages=[{"role": "user", "content": "set a timer for 5 minutes"}])
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "I received the timer request."
    assert client.llm.session_marks and client.llm.session_marks[0].get("API")
    assert client.get("/api/reminders", headers=WIN).json()["reminders"] == []


@pytest.mark.parametrize(("name", "arguments", "words", "expected"), [
    ("api_probe_read", {}, "Read the API probe.", ("read", "")),
    ("api_probe_run_code", {"code": "pass"}, "Run code in the API probe: pass", ("run", "pass")),
])
async def test_desktop_words_still_authorize_only_if_asked_tools(
    config, isolated_home, name, arguments, words, expected,
):
    config.tools.approvals.mode = "always"
    config.tools.approvals.rules = {name: "if_asked"}
    calls: list[tuple[str, str]] = []
    llm = FakeProvider(replies=[[tool_call(name, **arguments)], "Finished."])
    core = await SentientApp(config, llm=llm, db_path=isolated_home / "desktop.db", enable_background=False).start()
    core.registry.register(_probe_plugin(calls))
    try:
        sid = await core.store.create_session(channel="desktop")
        asked = []
        async for event in core.agent.run_turn(sid, words, channel="desktop"):
            if isinstance(event, ApprovalRequest):
                asked.append(event.name)
                core.approvals.resolve(event.approval_id, "deny")
        assert asked == []
        assert calls == [expected]
        assert not (await core.store.get_session(sid))["untrusted"]
    finally:
        await core.stop()


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_api_execute_code_waits_for_approval_and_deny_never_invokes(client, monkeypatch, stream):
    core = client.app.state.sentient
    execute = core.registry.get("execute_code")
    assert execute is not None
    calls: list[tuple[str, str]] = []

    async def record_execution(ctx: ToolContext, code: str, purpose: str) -> dict:
        calls.append((code, purpose))
        return {"ok": True, "stdout": "stubbed"}

    monkeypatch.setattr(execute, "fn", record_execution)
    arguments = {"code": "print(1 + 1)", "purpose": "Check arithmetic."}
    core.config.tools.approvals.mode = "off"
    core.config.tools.approvals.rules = {"execute_code": "if_asked"}
    core.approvals.timeout_s = 20
    client.llm.replies[:] = [[tool_call("execute_code", **arguments)], "I did not run it."]
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(_ask, client, stream=stream, messages=[{
            "role": "user", "content": "Run this Python code with execute_code: print(1 + 1).",
        }])
        note = _approval(client, pending)
        assert calls == [] and not pending.done()
        assert note["payload"]["name"] == "execute_code"
        assert note["payload"]["risk"] == "exec"
        assert note["payload"]["arguments"] == arguments
        _answer(client, note, "deny")
        response = pending.result(timeout=30)
    assert response.status_code == 200
    assert _reply_text(response, stream=stream).startswith('Not done: "Execute code"')  # before the model's words
    assert _status(client, note) == "deny"
    assert calls == []
