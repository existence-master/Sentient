"""Sign in with ChatGPT and ChatGPT plan models through the Responses API shim (issue #205). Offline: respx only."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm import chatgpt, presets
from sentient.llm.provider import LiteLLMProvider, ProviderError
from tests.conftest import FakeProvider

RESPONSES = f"{chatgpt.API_BASE}/responses"
MODELS = f"{chatgpt.API_BASE}/models"
ISSUED = "oaiapp_test123"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


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
    a = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "chatgpt.db", enable_background=False)
    await a.start()
    try:
        yield a
    finally:
        await a.stop()


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    api = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(api) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _id_token(nonce: str, aud: str = ISSUED, key=KEY, **extra) -> str:
    head = _b64(json.dumps({"alg": "RS256", "kid": "k1", "typ": "JWT"}).encode())
    claims = {"iss": chatgpt.ISSUER, "aud": aud, "sub": "user-1", "email": "maya@example.com",
              "exp": int(time.time()) + 3600, "nonce": nonce, **extra}
    body = _b64(json.dumps(claims).encode())
    sig = key.sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{head}.{body}.{_b64(sig)}"


def _jwks() -> dict:
    n = KEY.public_key().public_numbers()
    return {"keys": [{"kty": "RSA", "kid": "k1", "alg": "RS256", "use": "sig",
                      "n": _b64(n.n.to_bytes((n.n.bit_length() + 7) // 8, "big")), "e": _b64(n.e.to_bytes(3, "big"))}]}


def _tokens(nonce: str, **extra) -> dict:
    return {"access_token": "at-1", "refresh_token": "rt-1", "id_token": _id_token(nonce), "token_type": "Bearer",
            "expires_in": 3600, "scope": chatgpt.SCOPES, **extra}


def _challenge(verifier: str) -> str:
    return _b64(hashlib.sha256(verifier.encode()).digest())


def _signed_in(keychain: dict, *, expires_in: float = 3600, access: str = "at-1", refresh: str = "rt-1") -> None:
    secrets.save_json(chatgpt.SECRET, {"client_id": ISSUED, "access_token": access, "refresh_token": refresh,
                                       "expires_at": time.time() + expires_in, "scope": chatgpt.SCOPES,
                                       "email": "maya@example.com"})


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


# ---------------------------------------------------------------------------- sign-in
async def test_first_sign_in_registers_and_stores_tokens(app, keychain):
    started = await app.connections.start_chatgpt()
    parts = urlsplit(started["auth_url"])
    q = {k: v[0] for k, v in parse_qs(parts.query).items()}
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == chatgpt.AUTHORIZE_URL
    assert q["client_id"] == "dynamic_agent_client" and q["agent_name_hint"] == "Sentient"
    assert q["response_type"] == "code" and q["code_challenge_method"] == "S256" and q["state"] == started["state"]
    assert q["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert q["resource"] == "https://api.openai.com/v1" and q["nonce"] and q["ext_agent_host_id"].startswith("urn:uuid:")
    assert q["redirect_uri"].startswith("http://127.0.0.1:") and q["redirect_uri"].endswith("/oauth/callback")

    async with app.bus.subscribe() as events:
        with respx.mock(assert_all_called=True) as router:
            router.route(host="127.0.0.1").pass_through()
            token = router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(200, json=_tokens(q["nonce"])))
            router.get(chatgpt.JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks()))
            async with httpx.AsyncClient() as http:
                ok = await http.get(q["redirect_uri"], params={"code": "c0de", "state": q["state"], "client_id": ISSUED,
                                                               "scope": chatgpt.SCOPES})
                again = await http.get(q["redirect_uri"], params={"code": "c0de", "state": q["state"]})
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
    sent = {k: v[0] for k, v in parse_qs(token.calls.last.request.content.decode()).items()}
    assert ok.status_code == 200 and "signed in with ChatGPT" in ok.text
    assert again.status_code == 400 and "already used" in again.text
    assert sent["grant_type"] == "authorization_code" and sent["client_id"] == ISSUED and sent["code"] == "c0de"
    assert _challenge(sent["code_verifier"]) == q["code_challenge"]
    assert sent["redirect_uri"] == q["redirect_uri"] and sent["resource"] == chatgpt.RESOURCE
    saved = chatgpt.load_tokens()
    assert saved["access_token"] == "at-1" and saved["refresh_token"] == "rt-1" and saved["client_id"] == ISSUED
    assert saved["email"] == "maya@example.com" and saved["expires_at"] > time.time() + 3000
    assert await app.store.get_meta(chatgpt.CLIENT_META) == ISSUED
    assert app.connections.flow_status(started["state"]) == {"status": "connected", "error": None}
    assert app.connections.chatgpt_status()["signed_in"] is True
    assert any(e["type"] == "config.updated" and e["data"]["sections"] == ["secrets"] for e in seen)

    # the next sign-in reuses the issued client id and doesn't register again
    again_q = {k: v[0] for k, v in parse_qs(urlsplit((await app.connections.start_chatgpt())["auth_url"]).query).items()}
    assert again_q["client_id"] == ISSUED and "agent_name_hint" not in again_q
    assert again_q["ext_agent_host_id"] == q["ext_agent_host_id"]
    ok2, msg = await app.connections.oauth_callback({"state": again_q["state"], "code": "x", "client_id": "oaiapp_other"})
    assert ok2 is False and "different app" in msg


async def test_forged_state_and_refusals_store_nothing(app, keychain):
    started = await app.connections.start_chatgpt()
    q = {k: v[0] for k, v in parse_qs(urlsplit(started["auth_url"]).query).items()}
    with respx.mock(assert_all_called=False) as router:
        token = router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(200, json=_tokens(q["nonce"])))
        forged = await app.connections.oauth_callback({"state": "forged", "code": "c", "client_id": ISSUED})
        assert forged[0] is False and "expired" in forged[1] and not token.called
        assert app.connections.flow_status(started["state"])["status"] == "waiting"  # the real one can still finish

        # registration that comes back without an issued client id is incomplete
        no_id = await app.connections.oauth_callback({"state": started["state"], "code": "c"})
        assert no_id[0] is False and "registering" in no_id[1] and not token.called

    started = await app.connections.start_chatgpt()
    cancelled = await app.connections.oauth_callback({"state": started["state"], "error": "access_denied"})
    assert cancelled == (False, "You cancelled the sign-in.")

    # plan usage not granted: nothing is kept
    started = await app.connections.start_chatgpt()
    q = {k: v[0] for k, v in parse_qs(urlsplit(started["auth_url"]).query).items()}
    with respx.mock() as router:
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(
            200, json=_tokens(q["nonce"], scope="openid profile email offline_access")))
        ok, msg = await app.connections.oauth_callback({"state": started["state"], "code": "c", "client_id": ISSUED})
    assert ok is False and "Plus or Pro" in msg and chatgpt.load_tokens() is None

    # an ID token for another sign-in attempt (wrong nonce) is refused
    started = await app.connections.start_chatgpt()
    with respx.mock() as router:
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(200, json=_tokens("someone-elses-nonce")))
        router.get(chatgpt.JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks()))
        ok, msg = await app.connections.oauth_callback({"state": started["state"], "code": "c", "client_id": ISSUED})
    assert ok is False and "couldn't be verified" in msg and chatgpt.load_tokens() is None

    # and so is one signed with a key that isn't OpenAI's
    started = await app.connections.start_chatgpt()
    q = {k: v[0] for k, v in parse_qs(urlsplit(started["auth_url"]).query).items()}
    forged_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with respx.mock() as router:
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(
            200, json={**_tokens(q["nonce"]), "id_token": _id_token(q["nonce"], key=forged_key)}))
        router.get(chatgpt.JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks()))
        ok, msg = await app.connections.oauth_callback({"state": started["state"], "code": "c", "client_id": ISSUED})
    assert ok is False and chatgpt.load_tokens() is None and "chatgpt" not in keychain

    # a reply without an ID token can't be checked, so it isn't kept either
    started = await app.connections.start_chatgpt()
    q = {k: v[0] for k, v in parse_qs(urlsplit(started["auth_url"]).query).items()}
    no_id_token = {k: v for k, v in _tokens(q["nonce"]).items() if k != "id_token"}
    with respx.mock() as router:
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(200, json=no_id_token))
        ok, msg = await app.connections.oauth_callback({"state": started["state"], "code": "c", "client_id": ISSUED})
    assert ok is False and "couldn't be verified" in msg and "chatgpt" not in keychain


async def test_sign_out_stops_sign_ins_in_flight(app, keychain):
    await app.store.set_meta(presets.PLAN_MODELS_META, json.dumps(["chatgpt/old-account-model"]))
    waiting = await app.connections.start_chatgpt()
    exchanging = await app.connections.start_chatgpt()
    q = {k: v[0] for k, v in parse_qs(urlsplit(exchanging["auth_url"]).query).items()}

    def sign_out_meanwhile(request: httpx.Request) -> httpx.Response:
        app.connections._flows[exchanging["state"]]["cancelled"] = True  # what a sign-out does to a flow mid-exchange
        return httpx.Response(200, json=_tokens(q["nonce"]))

    with respx.mock(assert_all_called=False) as router:  # nothing to revoke: no sign-in was kept yet
        router.post(chatgpt.REVOKE_URL).mock(return_value=httpx.Response(200))
        await app.connections.sign_out_chatgpt()
        ok, _msg = await app.connections.oauth_callback({"state": waiting["state"], "code": "c", "client_id": ISSUED})
        assert ok is False and app.connections.flow_status(waiting["state"])["status"] == "failed"

        flow = app.connections._flows[exchanging["state"]]
        flow.update(status="waiting", error=None)  # replay the second one as if the sign-out came mid-exchange
        flow.pop("cancelled", None)
        router.post(chatgpt.TOKEN_URL).mock(side_effect=sign_out_meanwhile)
        router.get(chatgpt.JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks()))
        ok, msg = await app.connections.oauth_callback({"state": exchanging["state"], "code": "c", "client_id": ISSUED})
    assert ok is False and "signed out" in msg and "chatgpt" not in keychain
    assert not await app.store.get_meta(presets.PLAN_MODELS_META)  # the old account's models are forgotten


def test_turned_off_without_a_client_id(client):
    status = client.get("/api/models/connect/chatgpt").json()
    assert status == {"available": True, "reason": None, "signed_in": False, "email": None,
                      "manage_usage_url": "https://chatgpt.com/settings/usage"}
    assert client.patch("/api/config", json={"models": {"chatgpt_client_id": ""}}).status_code == 200
    status = client.get("/api/models/connect/chatgpt").json()
    assert status["available"] is False and "turned off" in status["reason"]
    r = client.post("/api/models/connect/chatgpt")
    assert r.status_code == 409 and "turned off" in r.json()["detail"]


# ---------------------------------------------------------------------------- tokens
async def test_refresh_on_expiry_rotates_tokens_once(config, keychain):
    _signed_in(keychain, expires_in=10)  # inside the refresh margin
    with respx.mock() as router:
        refresh = router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(
            200, json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600, "scope": chatgpt.SCOPES}))
        tokens = await asyncio.gather(*(chatgpt.access_token(config) for _ in range(3)))
    assert tokens == ["at-2"] * 3 and refresh.call_count == 1  # serialized: a rotating token is used once
    sent = {k: v[0] for k, v in parse_qs(refresh.calls.last.request.content.decode()).items()}
    assert sent == {"grant_type": "refresh_token", "client_id": ISSUED, "refresh_token": "rt-1",
                    "resource": chatgpt.RESOURCE}
    saved = chatgpt.load_tokens()
    assert saved["refresh_token"] == "rt-2" and saved["expires_at"] > time.time() + 3000

    with respx.mock(assert_all_called=False) as router:  # a fresh token is used as it is
        again = router.post(chatgpt.TOKEN_URL)
        assert await chatgpt.access_token(config) == "at-2" and not again.called


async def test_dead_refresh_signs_out(config, keychain):
    _signed_in(keychain, expires_in=-5)
    with respx.mock() as router:  # not a final answer: the sign-in stays for the next try
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(401, json={"error": "invalid_client"}))
        with pytest.raises(chatgpt.ChatGPTError, match="right now"):
            await chatgpt.access_token(config)
    assert chatgpt.load_tokens() is not None
    with respx.mock() as router:
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(
            400, json={"error": "invalid_grant", "error_description": "refresh token reused"}))
        with pytest.raises(chatgpt.ChatGPTError, match="signed you out"):
            await chatgpt.access_token(config)
    assert chatgpt.load_tokens() is None and not keychain


def test_sign_out_revokes_and_removes_tokens(client, keychain):
    assert client.put("/api/secrets/chatgpt", json={"value": "pasted"}).status_code == 400
    _signed_in(keychain, access="a" * 2500)  # long enough to span several keychain entries
    assert len(keychain) > 1
    providers = {p["id"]: p for p in client.get("/api/models/providers").json()}
    assert providers["chatgpt"]["key_set"] is True and providers["chatgpt"]["sign_in"] is True
    assert client.get("/api/models/connect/chatgpt").json()["email"] == "maya@example.com"
    with respx.mock() as router:
        revoke = router.post(chatgpt.REVOKE_URL).mock(return_value=httpx.Response(200))
        assert client.delete("/api/models/connect/chatgpt").json() == {"ok": True}
    sent = {k: v[0] for k, v in parse_qs(revoke.calls.last.request.content.decode()).items()}
    assert sent == {"token": "rt-1", "token_type_hint": "refresh_token", "client_id": ISSUED}
    assert keychain == {} and client.get("/api/models/connect/chatgpt").json()["signed_in"] is False

    _signed_in(keychain)  # removing it like a key signs out too, even when OpenAI can't be reached
    with respx.mock() as router:
        router.post(chatgpt.REVOKE_URL).mock(side_effect=httpx.ConnectError("offline"))
        assert client.delete("/api/secrets/chatgpt").json() == {"ok": True}
    assert keychain == {}


# ---------------------------------------------------------------------------- the Responses API shim
TOOL = {"type": "function", "function": {"name": "find_city", "description": "Look up a city.",
                                         "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}


def _tool_call_stream() -> bytes:
    item = {"type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "find_city", "arguments": ""}
    return _sse(
        {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Looking it up."},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Let me "},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "check."},
        {"type": "response.output_item.added", "output_index": 2, "item": item},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 2, "delta": '{"name":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 2, "delta": '"Pune"}'},
        {"type": "response.output_item.done", "output_index": 2, "item": {**item, "arguments": '{"name":"Pune"}'}},
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "model": "gpt-test",
                                                    "usage": {"input_tokens": 42, "output_tokens": 7}}},
    )


async def test_streaming_tool_call_through_the_shim(config, keychain):
    _signed_in(keychain)
    config.models.temperature["primary"] = 0.3  # plan usage refuses temperature: never sent
    llm = LiteLLMProvider(config)
    with respx.mock() as router:
        route = router.post(RESPONSES).mock(return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_tool_call_stream()))
        chunks = [c async for c in llm.stream("primary", [{"role": "system", "content": "Be brief."},
                                                          {"role": "user", "content": "Weather in Pune?"}],
                                              [TOOL], model="chatgpt/gpt-test")]
    req = route.calls.last.request
    body = json.loads(req.content)
    assert req.headers["authorization"] == "Bearer at-1"
    assert body["model"] == "gpt-test" and body["stream"] is True and body["store"] is False
    assert body["instructions"] == "Be brief." and not any(i.get("role") == "system" for i in body["input"])
    assert body["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text",
                                                                              "text": "Weather in Pune?"}]}]
    assert body["tools"] == [{"type": "function", "name": "find_city", "description": "Look up a city.",
                              "parameters": TOOL["function"]["parameters"]}]
    for banned in ("temperature", "max_output_tokens", "previous_response_id", "metadata", "user", "top_p"):
        assert banned not in body
    assert body["reasoning"]["effort"] == "medium"
    assert "".join(c.text for c in chunks) == "Let me check." and "".join(c.thinking for c in chunks) == "Looking it up."
    done = chunks[-1]
    assert done.done and done.usage == {"prompt_tokens": 42, "completion_tokens": 7} and done.cost is None
    assert [(t.id, t.name, t.arguments) for t in done.tool_calls] == [("call_abc", "find_city", {"name": "Pune"})]

    # the next round sends the call and its result back with the same call_id
    history = [{"role": "user", "content": "Weather in Pune?"},
               {"role": "assistant", "content": "Let me check.", "tool_calls": [done.tool_calls[0].to_openai()]},
               {"role": "tool", "tool_call_id": "call_abc", "name": "find_city", "content": '{"city_id": "c-42"}'},
               {"role": "system", "content": "Note: answer in one line."}]
    with respx.mock() as router:
        route = router.post(RESPONSES).mock(return_value=httpx.Response(200, content=_sse(
            {"type": "response.output_text.delta", "delta": "Sunny."},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 1, "output_tokens": 1}}})))
        text = "".join([c.text async for c in llm.stream("primary", history, [TOOL], model="chatgpt/gpt-test")])
    items = json.loads(route.calls.last.request.content)["input"]
    assert text == "Sunny."
    assert items[1] == {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]}
    assert items[2] == {"type": "function_call", "call_id": "call_abc", "name": "find_city", "arguments": '{"name": "Pune"}'}
    assert items[3] == {"type": "function_call_output", "call_id": "call_abc", "output": '{"city_id": "c-42"}'}
    assert items[4]["role"] == "developer"


async def test_json_jobs_stream_and_errors_are_plain(config, keychain):
    _signed_in(keychain)
    llm = LiteLLMProvider(config)
    with respx.mock() as router:
        route = router.post(RESPONSES).mock(return_value=httpx.Response(200, content=_sse(
            {"type": "response.output_text.delta", "delta": '{"facts": '},
            {"type": "response.output_text.delta", "delta": '["likes tea"]}'},
            {"type": "response.completed", "response": {}})))
        assert await llm.complete_json("fast", [{"role": "user", "content": "Facts as JSON"}],
                                       model="chatgpt/gpt-test-mini") == {"facts": ["likes tea"]}
    body = json.loads(route.calls.last.request.content)
    assert body["stream"] is True and "reasoning" not in body  # the fast role thinks with "none"

    with respx.mock() as router:
        router.post(RESPONSES).mock(return_value=httpx.Response(429, json={"error": {
            "code": "subscription_sharing_usage_limit_exceeded", "message": "limit"}}))
        with pytest.raises(ProviderError, match="Usage limit reached"):
            await llm.complete_text("primary", [{"role": "user", "content": "hi"}], model="chatgpt/gpt-test")

    with respx.mock() as router:  # a limit reached mid-reply
        router.post(RESPONSES).mock(return_value=httpx.Response(200, content=_sse(
            {"type": "response.output_text.delta", "delta": "Hal"},
            {"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}})))
        with pytest.raises(ProviderError, match="Usage limit reached"):
            [c async for c in llm.stream("primary", [{"role": "user", "content": "hi"}], model="chatgpt/gpt-test")]

    with pytest.raises(ProviderError, match="embedding"):
        await llm.embed(["hello"], model="chatgpt/gpt-test")


async def test_a_refused_token_is_renewed_once(config, keychain):
    _signed_in(keychain)
    llm = LiteLLMProvider(config)
    with respx.mock() as router:
        route = router.post(RESPONSES).mock(side_effect=[
            httpx.Response(401, json={"error": {"code": "invalid_token", "message": "expired"}}),
            httpx.Response(200, content=_sse({"type": "response.output_text.delta", "delta": "ok"},
                                             {"type": "response.completed", "response": {}}))])
        router.post(chatgpt.TOKEN_URL).mock(return_value=httpx.Response(
            200, json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600}))
        text = await llm.complete_text("primary", [{"role": "user", "content": "hi"}], model="chatgpt/gpt-test")
    assert text == "ok" and route.calls.last.request.headers["authorization"] == "Bearer at-2"


async def test_not_signed_in_is_a_plain_error(config):
    with pytest.raises(ProviderError, match="Sign in with ChatGPT first"):
        await LiteLLMProvider(config).complete_text("primary", [{"role": "user", "content": "hi"}],
                                                    model="chatgpt/gpt-test")


# ---------------------------------------------------------------------------- models, presets and the checkup
async def test_plan_models_feed_the_catalog_and_cloud_preset(app, keychain):
    tags = {"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]}
    planned = app.config.model_copy(deep=True)
    planned.models.roles.primary = "chatgpt/gpt-test"
    with respx.mock() as router:  # never a real Ollama
        router.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json=tags))
        missing = await presets.missing(planned)
    assert [(m["kind"], m["roles"], m["action"]) for m in missing] == [("sign_in", ["primary"], None)]

    _signed_in(keychain)
    listing = {"models": [{"slug": "gpt-test", "display_name": "GPT Test", "visibility": "list"},
                          {"slug": "gpt-hidden", "display_name": "Hidden", "visibility": "hide"},
                          {"slug": "gpt-test-mini", "display_name": "GPT Test mini", "visibility": "list"}]}
    with respx.mock() as router:
        router.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json=tags))
        models = router.get(MODELS).mock(return_value=httpx.Response(200, json=listing))
        catalog = await app.connections.catalog("chatgpt")
        items = {p["name"]: p for p in (await presets.listing(app))["presets"]}
        result = await presets.apply(app, "Cloud")
    assert models.calls.last.request.headers["authorization"] == "Bearer at-1"
    assert [(m["id"], m["label"]) for m in catalog] == [("chatgpt/gpt-test", "GPT Test"),
                                                         ("chatgpt/gpt-test-mini", "GPT Test mini")]
    cloud = items["Cloud"]
    assert cloud["available"] and cloud["provider"] == "chatgpt"
    assert cloud["roles"]["primary"] == "chatgpt/gpt-test" and cloud["roles"]["fast"] == "chatgpt/gpt-test-mini"
    assert result["missing"] == [] and app.config.models.roles.primary == "chatgpt/gpt-test"

    app.connections.forget("chatgpt")  # offline later: the list from last time still names the plan's models
    with respx.mock() as router:
        router.get(MODELS).mock(side_effect=httpx.ConnectError("offline"))
        items = {p["name"]: p for p in (await presets.listing(app))["presets"]}
    assert items["Cloud"]["available"] and items["Cloud"]["roles"]["primary"] == "chatgpt/gpt-test"

    await app.store.set_meta(presets.PLAN_MODELS_META, json.dumps([1, None]))  # a damaged list is ignored
    with respx.mock() as router:
        router.get(MODELS).mock(side_effect=httpx.ConnectError("offline"))
        items = {p["name"]: p for p in (await presets.listing(app))["presets"]}
    assert items["Cloud"]["available"] is False and "Couldn't load" in items["Cloud"]["reason"]
