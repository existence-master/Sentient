from __future__ import annotations

import json

import httpx
import pytest
import respx

from sentient.integrations.base import IntegrationError
from sentient.tasks.scripts import ITEM_KEYS, RESULT_KEYS

KEYLESS = "https://api.you.com/v1/agents/search"
KEYED = "https://ydc-index.io/v1/search"
UA = "youdotcom-integration/existence-master-Sentient"
PAYLOAD = {"results": {
    "web": [
        {"url": "https://example.org/a", "title": "A", "description": "desc a", "snippets": ["one", "two", "three"]},
        {"url": "https://example.org/b", "title": "B", "description": "desc b"},
    ],
    "news": [{"url": "https://example.org/n", "title": "N", "description": "news"}],
}}


async def test_youcom_search_keyless(app, ctx):
    app.config.integrations.search_provider = "youcom"
    with respx.mock() as router:
        route = router.get(KEYLESS).mock(return_value=httpx.Response(200, json=PAYLOAD))
        res = await app.registry.get("web_search").call(ctx, {"query": "Pune monsoon", "max_results": 3})
    req = route.calls.last.request
    assert req.url.params["query"] == "Pune monsoon" and req.url.params["count"] == "3"
    assert req.headers["User-Agent"] == UA and "X-API-Key" not in req.headers
    assert res["provider"] == "youcom" and "note" not in res
    assert res["results"] == [
        {"title": "A", "url": "https://example.org/a", "snippet": "one two"},
        {"title": "B", "url": "https://example.org/b", "snippet": "desc b"},
        {"title": "N", "url": "https://example.org/n", "snippet": "news"},
    ]
    assert set(res) <= set(RESULT_KEYS["web_search"])
    assert set(res["results"][0]) == set(ITEM_KEYS[("web_search", "results")])


async def test_youcom_search_uses_connected_key(app, ctx):
    app.config.integrations.search_provider = "youcom"
    with respx.mock() as router:
        route = router.post(KEYED).mock(return_value=httpx.Response(200, json=PAYLOAD))
        await app.integrations.connect("youcom", {"api_key": "test-key"})
        res = await app.registry.get("web_search").call(ctx, {"query": "x", "max_results": 2})
    req = route.calls.last.request
    assert req.headers["X-API-Key"] == "test-key" and req.headers["User-Agent"] == UA
    assert json.loads(req.content) == {"query": "x", "count": 2}
    assert [r["url"] for r in res["results"]] == ["https://example.org/a", "https://example.org/b"]


async def test_youcom_keyless_limit_message(app, ctx):
    from sentient.integrations.plugins.search import youcom_search

    with respx.mock() as router:
        router.get(KEYLESS).mock(return_value=httpx.Response(429))
        with pytest.raises(IntegrationError, match="free search limit"):
            await youcom_search(None, "x", 5)


async def test_youcom_connect_validates_key(app):
    with respx.mock() as router:
        route = router.post(KEYED).mock(return_value=httpx.Response(200, json=PAYLOAD))
        await app.integrations.connect("youcom", {"api_key": "good"})
        assert route.calls.last.request.headers["X-API-Key"] == "good"
        router.post(KEYED).mock(return_value=httpx.Response(401))
        with pytest.raises(IntegrationError, match="didn't accept that key"):
            await app.integrations.connect("youcom", {"api_key": "bad"})
    assert (await app.integrations.get_credentials("youcom")) == {"api_key": "good"}
