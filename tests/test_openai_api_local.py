"""The public API remains local even when the gateway listens on the network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider
from tests.test_openai_api import TOKEN, WIN
from tests.test_openai_api import keychain as keychain


@pytest.mark.parametrize("stream", [None, False, True], ids=["models", "json", "stream"])
@pytest.mark.parametrize(("server", "peer", "allowed"), [
    ("127.0.0.1", "127.0.0.1", True),
    ("::1", "::1", True),
    ("::ffff:127.0.0.1", "::ffff:127.0.0.1", True),
    ("192.0.2.10", "192.0.2.20", False),
    ("127.0.0.1", "192.0.2.20", False),
    ("192.0.2.10", "127.0.0.1", False),
    ("::ffff:192.0.2.10", "::1", False),
    ("0.0.0.0", "127.0.0.1", False),
    (None, "127.0.0.1", False),
    ("127.0.0.1", None, False),
    ("localhost", "127.0.0.1", False),  # transport addresses are numeric, never DNS names
])
def test_only_loopback_connections_reach_api(config, isolated_home, monkeypatch, keychain, server, peer, allowed, stream):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    config.openai_api.enabled = True
    keychain["openai_api_key"] = "sk-sentient-test"
    llm = FakeProvider(replies=["Hello."])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "local.db", enable_background=False)
    gateway = create_app(core)

    async def connection(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "server": (server, 7777) if server else None,
                     "client": (peer, 50000) if peer else None}
        await gateway(scope, receive, send)

    with TestClient(connection) as client:
        headers = {"Authorization": "Bearer sk-sentient-test", "Host": "localhost:7777",
                   "X-Forwarded-For": "127.0.0.1", "Forwarded": "for=127.0.0.1;host=localhost"}
        if stream is None:
            result = client.get("/v1/models", headers=headers)
        else:
            result = client.post("/v1/chat/completions", headers=headers,
                                 json={"messages": [{"role": "user", "content": "hi"}], "stream": stream})
        assert result.status_code == (200 if allowed else 403)
        if not allowed:
            assert result.json()["error"]["code"] == "local_only"
            assert result.headers["x-should-retry"] == "false"
            assert llm.calls == []
            assert client.get("/api/sessions", headers=WIN).json() == []


def test_runtime_enable_and_trusted_proxy_headers_cannot_expose_network_socket(config, isolated_home, monkeypatch, keychain):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    keychain["openai_api_key"] = "sk-sentient-test"
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "proxy.db", enable_background=False)
    app = ProxyHeadersMiddleware(create_app(core), trusted_hosts="*")
    with TestClient(app, base_url="http://192.0.2.10", client=("192.0.2.20", 50000)) as client:
        headers = {"Authorization": "Bearer sk-sentient-test", "Host": "localhost", "X-Forwarded-For": "127.0.0.1"}
        assert client.get("/v1/models", headers=headers).json()["error"]["code"] == "api_off"
        assert client.patch("/api/config", headers=WIN, json={"openai_api": {"enabled": True}}).status_code == 200
        result = client.post("/v1/chat/completions", headers=headers,
                             json={"messages": [{"role": "user", "content": "hi"}]})
        assert result.status_code == 403
        assert result.json()["error"]["code"] == "local_only"
        assert llm.calls == []
        assert client.get("/api/sessions", headers=WIN).json() == []
