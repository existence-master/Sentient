"""Connecting plans people already pay for: OpenRouter sign-in, Claude and Nous Portal keys (issue #204)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm import connect
from sentient.llm.provider import LiteLLMProvider, litellm_model, provider_config
from tests.conftest import FakeProvider


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    """In-memory stand-in for the OS keychain so tests never touch the real one."""
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


@pytest.fixture
async def app(config, isolated_home):
    a = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "connect.db", enable_background=False)
    await a.start()
    try:
        yield a
    finally:
        await a.stop()


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    for env in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    api = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(api) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


# ---------------------------------------------------------------------------- PKCE
def test_pkce_pair_is_s256_and_random():
    verifier, challenge = connect.pkce_pair()
    assert 43 <= len(verifier) <= 128 and re.fullmatch(r"[A-Za-z0-9_-]+", verifier)
    assert challenge == _challenge(verifier) and "=" not in challenge
    assert connect.pkce_pair()[0] != verifier


def test_openrouter_auth_url_carries_challenge_and_state():
    url = connect.openrouter_auth_url("http://127.0.0.1:5000/oauth/callback", "chal", "st8")
    parts = urlsplit(url)
    q = parse_qs(parts.query)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "https://openrouter.ai/auth"
    assert q == {"callback_url": ["http://127.0.0.1:5000/oauth/callback"], "code_challenge": ["chal"],
                 "code_challenge_method": ["S256"], "state": ["st8"], "key_label": ["Sentient"]}


# ---------------------------------------------------------------------------- OpenRouter sign-in
async def test_openrouter_sign_in_stores_the_key(app, keychain):
    async with app.bus.subscribe() as q:
        started = await app.connections.start_openrouter()
        qs = parse_qs(urlsplit(started["auth_url"]).query)
        callback = qs["callback_url"][0]
        assert callback.startswith("http://127.0.0.1:") and qs["state"] == [started["state"]]
        assert app.connections.flow_status(started["state"]) == {"status": "waiting", "error": None}

        with respx.mock(assert_all_called=True) as router:
            router.route(host="127.0.0.1").pass_through()
            keys = router.post(connect.OPENROUTER_KEYS_URL).mock(return_value=httpx.Response(200, json={"key": "sk-or-v1-abc"}))
            async with httpx.AsyncClient() as http:
                ok = await http.get(callback, params={"code": "c0de", "state": started["state"]})
                again = await http.get(callback, params={"code": "c0de", "state": started["state"]})
        sent = json.loads(keys.calls.last.request.content)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
    assert ok.status_code == 200 and "OpenRouter is connected" in ok.text
    assert again.status_code == 400 and "already used" in again.text
    assert sent["code"] == "c0de" and sent["code_challenge_method"] == "S256"
    assert _challenge(sent["code_verifier"]) == qs["code_challenge"][0]
    assert keychain["openrouter"] == "sk-or-v1-abc"
    assert app.connections.flow_status(started["state"]) == {"status": "connected", "error": None}
    assert any(e["type"] == "config.updated" and e["data"]["sections"] == ["secrets"] for e in events)


async def test_forged_state_is_rejected_and_nothing_is_stored(app, keychain):
    started = await app.connections.start_openrouter()
    callback = parse_qs(urlsplit(started["auth_url"]).query)["callback_url"][0]
    with respx.mock(assert_all_called=False) as router:
        router.route(host="127.0.0.1").pass_through()
        keys = router.post(connect.OPENROUTER_KEYS_URL).mock(return_value=httpx.Response(200, json={"key": "stolen"}))
        async with httpx.AsyncClient() as http:
            forged = await http.get(callback, params={"code": "c0de", "state": "forged"})
            missing = await http.get(callback, params={"code": "c0de"})
    assert forged.status_code == 400 and "expired" in forged.text
    assert missing.status_code == 400
    assert not keys.called and "openrouter" not in keychain
    assert app.connections.flow_status(started["state"])["status"] == "waiting"  # the real sign-in can still finish


async def test_cancelled_or_refused_sign_in_reports_why(app, keychain):
    started = await app.connections.start_openrouter()
    ok, msg = await app.connections.oauth_callback({"state": started["state"], "error": "access_denied"})
    assert ok is False and "cancelled" in msg
    assert app.connections.flow_status(started["state"]) == {"status": "failed", "error": msg}

    started = await app.connections.start_openrouter()
    with respx.mock() as router:
        router.post(connect.OPENROUTER_KEYS_URL).mock(
            return_value=httpx.Response(403, json={"error": {"message": "Invalid code or code_verifier"}}))
        ok, msg = await app.connections.oauth_callback({"state": started["state"], "code": "bad"})
    assert ok is False and "Invalid code" in msg and "openrouter" not in keychain


# ---------------------------------------------------------------------------- key checks and model lists
async def test_claude_key_check(config, keychain):
    assert (await connect.check_key(config, "anthropic"))["ok"] is False  # no key yet
    keychain["anthropic"] = "sk-ant-api03-test"
    with respx.mock() as router:
        models = router.get("https://api.anthropic.com/v1/models").mock(return_value=httpx.Response(
            200, json={"data": [{"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5"}]}))
        good = await connect.check_key(config, "anthropic")
        sent = models.calls.last.request
    assert good == {"ok": True, "detail": "Your Anthropic key works. 1 model available."}
    assert sent.headers["x-api-key"] == "sk-ant-api03-test" and sent.headers["anthropic-version"]

    with respx.mock() as router:
        router.get("https://api.anthropic.com/v1/models").mock(return_value=httpx.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}))
        bad = await connect.check_key(config, "anthropic")
    assert bad["ok"] is False and "didn't accept this key" in bad["error"]


async def test_openrouter_key_check_and_catalog_marks_free_models(config, keychain):
    keychain["openrouter"] = "sk-or-v1-abc"
    with respx.mock() as router:
        router.get(connect.OPENROUTER_KEY_INFO_URL).mock(return_value=httpx.Response(
            200, json={"data": {"label": "Sentient", "limit_remaining": 4.5}}))
        router.get(connect.OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(200, json={"data": [
            {"id": "meta-llama/llama-4-maverick:free", "name": "Llama 4 Maverick (free)",
             "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"], "context_length": 128000},
            {"id": "anthropic/claude-sonnet-5", "name": "Claude Sonnet 5",
             "pricing": {"prompt": "0.000003", "completion": "0.000015"}, "supported_parameters": []},
        ]}))
        check = await connect.check_key(config, "openrouter")
        models = await connect.list_models(config, "openrouter")
    assert check["ok"] and "$4.50 left" in check["detail"]
    assert [(m["id"], m["free"], m["tools"]) for m in models] == [
        ("openrouter/meta-llama/llama-4-maverick:free", True, True),
        ("openrouter/anthropic/claude-sonnet-5", False, False),
    ]


async def test_nous_portal_key_uses_its_openai_compatible_endpoint(config, keychain):
    keychain["nous"] = "sk-nous-test"
    with respx.mock() as router:
        route = router.get("https://inference-api.nousresearch.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "Hermes-4-70B"}]}))
        models = await connect.list_models(config, "nous")
    assert models[0]["id"] == "nous/Hermes-4-70B"
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-nous-test"

    assert litellm_model("nous/Hermes-4-70B") == "openai/Hermes-4-70B"
    assert litellm_model("openrouter/anthropic/claude-sonnet-5") == "openrouter/anthropic/claude-sonnet-5"
    kwargs = LiteLLMProvider(config)._kwargs_for("nous/Hermes-4-70B", "primary")
    assert kwargs["api_base"] == "https://inference-api.nousresearch.com/v1" and kwargs["api_key"] == "sk-nous-test"
    config.models.providers["nous"] = type(provider_config(config, "nous"))(api_base="http://proxy.test/v1")
    assert provider_config(config, "nous").api_base == "http://proxy.test/v1"
    assert provider_config(config, "nous").api_key_env == "NOUS_API_KEY"


# ---------------------------------------------------------------------------- routes (settings and onboarding)
def test_connect_routes(client, keychain):
    provs = {p["id"]: p for p in client.get("/api/models/providers").json()}
    assert provs["nous"]["key_required"] and provs["nous"]["key_set"] is False

    started = client.post("/api/models/connect/openrouter").json()
    assert started["auth_url"].startswith("https://openrouter.ai/auth?")
    assert client.get(f"/api/models/connect/openrouter/{started['state']}").json()["status"] == "waiting"
    assert client.get("/api/models/connect/openrouter/nope").status_code == 404

    assert client.post("/api/models/connect/gemini/check").status_code == 404
    assert client.get("/api/models/catalog/gemini").status_code == 404
    assert client.post("/api/models/connect/anthropic/check").json()["ok"] is False
    assert client.get("/api/models/catalog/nous").status_code == 502  # no key yet

    with respx.mock() as router:
        catalog = router.get(connect.OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(
            200, json={"data": [{"id": "x/y:free", "name": "Y", "pricing": {"prompt": "0", "completion": "0"}}]}))
        first = client.get("/api/models/catalog/openrouter").json()
        client.get("/api/models/catalog/openrouter")
    assert first[0]["id"] == "openrouter/x/y:free" and first[0]["free"] is True
    assert catalog.call_count == 1  # cached

    # disconnect: removing the key takes it out of the keychain
    assert client.put("/api/secrets/openrouter", json={"value": "sk-or-v1-abc"}).json() == {"ok": True}
    assert {p["id"]: p for p in client.get("/api/models/providers").json()}["openrouter"]["key_set"] is True
    assert client.delete("/api/secrets/openrouter").json() == {"ok": True}
    assert "openrouter" not in keychain
    assert {p["id"]: p for p in client.get("/api/models/providers").json()}["openrouter"]["key_set"] is False
