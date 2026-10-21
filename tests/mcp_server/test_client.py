"""The MCP server's engine client: loopback only, the token on every call, plain errors (ADR 0034)."""

from __future__ import annotations

import httpx
import pytest
import respx

from sentient.mcp_server.client import EngineClient, EngineError

BASE = "http://127.0.0.1:7777"


def _client() -> EngineClient:
    return EngineClient(BASE, "test-token")


def test_a_non_local_engine_address_is_refused():
    with pytest.raises(EngineError, match="only works on this computer"):
        EngineClient("http://192.168.1.20:7777", "test-token")
    with pytest.raises(EngineError, match="only works on this computer"):
        EngineClient("https://example.com:7777", "test-token")


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
        router.get(f"{BASE}/api/config").mock(return_value=httpx.Response(401, json={"detail": "no"}))
        with pytest.raises(EngineError, match="refused this server's token"):
            _client().get_config()
    with respx.mock() as router:
        router.get(f"{BASE}/api/memories").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(EngineError, match="not answering"):
            _client().search_memories("tea")


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
        assert out == {"content": "Tea is at five.", "session_id": "s1"}
        import json as _json

        assert _json.loads(route.calls[0].request.content) == {"text": "when is tea?", "channel": "mcp"}
