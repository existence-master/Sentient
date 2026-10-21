"""The MCP server's engine client: loopback only, the token on every call, plain errors (ADR 0034)."""

from __future__ import annotations

import httpx
import pytest
import respx

from sentient.mcp_server.client import EngineClient, EngineError, discovered_engine_url

BASE = "http://127.0.0.1:7777"


def _client() -> EngineClient:
    return EngineClient(BASE, "test-token")


def test_the_readiness_probe_never_carries_the_token():
    # the probe answers "is anything there?" before the engine is identified, so it goes out
    # bare; the token is only for calls after that (status and everything the tools use)
    with respx.mock() as router:
        health = router.get(f"{BASE}/api/health").mock(
            return_value=httpx.Response(200, json={"ok": True, "name": "sentient", "version": "1.0"})
        )
        status = router.get(f"{BASE}/api/mcp/status").mock(
            return_value=httpx.Response(200, json={"enabled": True})
        )
        client = _client()
        assert client.health()["name"] == "sentient"
        assert "Authorization" not in health.calls.last.request.headers
        assert client.status()["enabled"] is True
        assert status.calls.last.request.headers["Authorization"] == "Bearer test-token"


def test_identity_is_proven_by_challenge_not_by_the_health_marker():
    # a listener on the discovered port can copy the health answer word for word; only the
    # holder of the token can answer a fresh nonce with its HMAC under that token
    import hashlib
    import hmac as hmac_mod

    def answer_challenge(request: httpx.Request) -> httpx.Response:
        nonce = request.url.params["nonce"]
        digest = hmac_mod.new(b"test-token", nonce.encode(), hashlib.sha256).hexdigest()
        return httpx.Response(200, json={"proof": digest})

    with respx.mock() as router:
        proof = router.get(f"{BASE}/api/mcp/proof").mock(side_effect=answer_challenge)
        assert _client().prove_identity() is True
        assert "Authorization" not in proof.calls.last.request.headers  # the challenge goes out bare

    with respx.mock() as router:
        router.get(f"{BASE}/api/mcp/proof").mock(
            return_value=httpx.Response(200, json={"proof": "0" * 64})
        )
        assert _client().prove_identity() is False  # a made-up proof is not the token holder

    with respx.mock() as router:
        router.get(f"{BASE}/api/mcp/proof").mock(return_value=httpx.Response(404))
        assert _client().prove_identity() is False  # no token issued: nothing to prove with


def test_a_non_local_engine_address_is_refused():
    with pytest.raises(EngineError, match="only works on this computer"):
        EngineClient("http://192.168.1.20:7777", "test-token")
    with pytest.raises(EngineError, match="only works on this computer"):
        EngineClient("https://example.com:7777", "test-token")


def test_the_engine_is_found_through_its_discovery_file(isolated_home):
    import json as _json

    from sentient import paths

    # no engine started: no file, and the error says what to do
    with pytest.raises(EngineError, match="Start Sentient first"):
        discovered_engine_url()
    found = paths.discovery_file()
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text(_json.dumps({"port": 8123}), encoding="utf-8")
    assert discovered_engine_url() == "http://127.0.0.1:8123"
    EngineClient(discovered_engine_url(), "test-token")  # passes the client's own loopback check


def test_calls_carry_the_token_and_hit_the_right_routes():
    with respx.mock() as router:
        search = router.get(f"{BASE}/api/memories").mock(
            return_value=httpx.Response(200, json=[{"id": 1, "content": "Maya drinks tea"}])
        )
        add = router.post(f"{BASE}/api/memories").mock(
            return_value=httpx.Response(200, json={"action": "ADD", "id": 7, "status": "pending"})
        )
        client = _client()
        assert client.search_memories("tea", 5) == [{"id": 1, "content": "Maya drinks tea"}]
        out = client.add_memory("Maya drinks tea", hold_for_review=True)
        assert out["status"] == "pending"
        assert search.calls[0].request.headers["authorization"] == "Bearer test-token"
        assert search.calls[0].request.url.params["q"] == "tea"
        import json as _json

        body = _json.loads(add.calls[0].request.content)
        assert body == {"content": "Maya drinks tea", "source": "mcp", "hold_for_review": True,
                        "review_from": "an MCP app"}


def test_task_and_notification_routes():
    with respx.mock() as router:
        # the real engine calls the field task_id; the client presents it as id
        create = router.post(f"{BASE}/api/tasks").mock(
            return_value=httpx.Response(200, json={"task_id": "t1", "status": "planning"})
        )
        note = router.post(f"{BASE}/api/notifications").mock(
            return_value=httpx.Response(200, json={"id": "n1", "kind": "info"})
        )
        client = _client()
        made = client.create_task("Book the dentist")
        assert made["status"] == "planning"
        assert made["id"] == "t1"
        assert client.notify("done")["id"] == "n1"
        import json as _json

        assert _json.loads(create.calls[0].request.content) == {"prompt": "Book the dentist"}
        assert _json.loads(note.calls[0].request.content) == {"message": "done", "source": "mcp"}


def test_a_refused_token_and_a_missing_engine_read_plainly():
    with respx.mock() as router:
        router.get(f"{BASE}/api/mcp/status").mock(return_value=httpx.Response(401, json={"detail": "no"}))
        with pytest.raises(EngineError, match="refused this server's token"):
            _client().status()
    with respx.mock() as router:
        router.get(f"{BASE}/api/memories").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(EngineError, match="not answering"):
            _client().search_memories("tea")


def test_a_slow_engine_is_not_reported_as_a_failed_change():
    # a save that times out may well have landed: the words must not invite a blind retry
    with respx.mock() as router:
        router.post(f"{BASE}/api/memories").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(EngineError, match="did not answer in time"):
            _client().add_memory("Maya drinks tea", hold_for_review=True)


def test_chat_turn_reads_the_ndjson_stream_until_done():
    ndjson = (
        '{"type": "session", "session_id": "s1"}\n'
        '{"type": "text_delta", "session_id": "s1", "text": "hi"}\n'
        '{"type": "done", "session_id": "s1", "content": "Tea is at five."}\n'
    )
    with respx.mock() as router:
        route = router.post(f"{BASE}/api/chat").mock(
            return_value=httpx.Response(200, text=ndjson, headers={"content-type": "application/x-ndjson"})
        )
        out = _client().chat_turn("when is tea?")
        assert out == {"content": "Tea is at five.", "session_id": "s1", "approval_asked": False}
        import json as _json

        assert _json.loads(route.calls[0].request.content) == {"text": "when is tea?", "channel": "mcp"}


def test_chat_turn_notes_when_the_user_was_asked_to_approve_a_step():
    ndjson = (
        '{"type": "session", "session_id": "s1"}\n'
        '{"type": "approval_request", "session_id": "s1", "approval_id": "a1", "name": "run_code"}\n'
        '{"type": "done", "session_id": "s1", "content": "Done."}\n'
    )
    with respx.mock() as router:
        router.post(f"{BASE}/api/chat").mock(
            return_value=httpx.Response(200, text=ndjson, headers={"content-type": "application/x-ndjson"})
        )
        out = _client().chat_turn("run it")
        assert out["approval_asked"] is True and out["content"] == "Done."


def test_the_app_name_rides_along_on_tasks_notes_and_held_memories():
    with respx.mock() as router:
        create = router.post(f"{BASE}/api/tasks").mock(
            return_value=httpx.Response(200, json={"task_id": "t1", "status": "planning"})
        )
        note = router.post(f"{BASE}/api/notifications").mock(
            return_value=httpx.Response(200, json={"id": "n1", "kind": "info"})
        )
        add = router.post(f"{BASE}/api/memories").mock(
            return_value=httpx.Response(200, json={"action": "ADD", "id": 7, "status": "pending"})
        )
        client = _client()
        client.create_task("Book the dentist", app="Claude Code")
        client.notify("done", app="Claude Code")
        client.add_memory("Maya drinks tea", hold_for_review=True, app="Claude Code")
        import json as _json

        assert _json.loads(create.calls[0].request.content) == {"prompt": "Book the dentist", "app": "Claude Code"}
        assert _json.loads(note.calls[0].request.content) == {
            "message": "done", "source": "mcp", "app": "Claude Code",
        }
        assert _json.loads(add.calls[0].request.content)["review_from"] == "Claude Code"
