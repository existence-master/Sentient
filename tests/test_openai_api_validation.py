"""API refusal checks precede body parsing and all body errors use the OpenAI envelope."""

from __future__ import annotations

import pytest

from tests.test_openai_api import _sessions
from tests.test_openai_api import client as client
from tests.test_openai_api import keychain as keychain

INVALID_BODIES = [
    pytest.param(b"{", id="malformed-json"),
    pytest.param(b"\xff", id="invalid-utf8"),
    pytest.param(b"", id="missing-body"),
    pytest.param(b"{}", id="missing-messages"),
    pytest.param(b'{"messages":"not a list"}', id="invalid-schema"),
]


def _post(client, body, *, authenticated=False, key=None, url="/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = f"Bearer {client.key if key is None else key}"
    return client.post(url, content=body, headers=headers)


def _assert_error(response, status, code):
    assert response.status_code == status
    payload = response.json()
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == code
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] is None
    assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]
    assert response.headers["x-should-retry"] == "false"


def _assert_no_turn(client):
    assert client.llm.calls == []
    assert _sessions(client) == []


@pytest.mark.parametrize("body", INVALID_BODIES)
@pytest.mark.parametrize("refusal", ["off", "missing-key", "wrong-key"])
def test_api_refusal_precedes_invalid_request_body(client, body, refusal):
    if refusal == "off":
        client.app.state.sentient.config.openai_api.enabled = False
    response = _post(client, body, authenticated=refusal == "wrong-key", key="wrong-test-key")
    status, code = (404, "api_off") if refusal == "off" else (401, "invalid_api_key")
    _assert_error(response, status, code)
    _assert_no_turn(client)


@pytest.mark.parametrize("body", INVALID_BODIES)
def test_authenticated_invalid_body_uses_openai_error_without_starting_a_turn(client, body):
    _assert_error(_post(client, body, authenticated=True), 400, "invalid_request")
    _assert_no_turn(client)


@pytest.mark.parametrize("body", INVALID_BODIES)
def test_nonlocal_transport_is_refused_before_body_validation(client, body):
    # TestClient handles this URL in-process; it changes the ASGI server address without any network request.
    response = _post(client, body, authenticated=True, url="http://192.0.2.10/v1/chat/completions")
    _assert_error(response, 403, "local_only")
    _assert_no_turn(client)


def test_chat_completions_keeps_its_openapi_request_body_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/v1/chat/completions"]["post"]
    assert "4XX" in operation["responses"]
    assert "422" not in operation["responses"]
    assert "4XX" not in document["paths"]["/api/openai-api/key"]["post"]["responses"]
    body = operation["requestBody"]
    assert body["required"] is True
    schema = body["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/ChatCompletionBody")
    declared = document["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    assert "messages" in declared["required"]
    assert declared["properties"]["messages"]["type"] == "array"
