"""Which address Google Custom Search calls: Google by default, the connected base_url when one is set."""

from __future__ import annotations

import httpx
import pytest
import respx

from sentient.integrations.base import IntegrationError
from sentient.integrations.plugins.alternatives import GoogleCSEPlugin
from sentient.integrations.plugins.search import GOOGLE_CSE_URL, google_cse_search

OTHER = "http://search.example.test/customsearch/v1"
ITEMS = {"items": [{"title": "T", "link": "https://example.com/a", "snippet": "S"}]}


def test_default_address_is_google():
    assert GOOGLE_CSE_URL == "https://www.googleapis.com/customsearch/v1"


@respx.mock
async def test_search_calls_google_without_base_url():
    route = respx.get(GOOGLE_CSE_URL).mock(return_value=httpx.Response(200, json=ITEMS))
    rows = await google_cse_search("k", "cx1", "sentient", 3)
    assert route.called
    assert route.calls.last.request.url.params["cx"] == "cx1"
    assert rows == [{"title": "T", "url": "https://example.com/a", "snippet": "S"}]


@respx.mock
async def test_search_calls_base_url_with_the_same_params():
    google = respx.get(GOOGLE_CSE_URL)
    route = respx.get(OTHER).mock(return_value=httpx.Response(200, json=ITEMS))
    rows = await google_cse_search("k", "cx1", "sentient", 3, OTHER)
    params = route.calls.last.request.url.params
    assert (params["key"], params["cx"], params["q"], params["num"]) == ("k", "cx1", "sentient", "3")
    assert not google.called
    assert rows[0]["url"] == "https://example.com/a"


@respx.mock
async def test_connect_without_base_url_checks_google_and_stores_no_address():
    route = respx.get(GOOGLE_CSE_URL).mock(return_value=httpx.Response(200, json=ITEMS))
    creds, _ = await GoogleCSEPlugin().validate({"api_key": "k", "cx": "cx1", "base_url": " "}, None)
    assert route.called
    assert creds == {"api_key": "k", "cx": "cx1"}


@respx.mock
async def test_connect_with_base_url_checks_it_and_stores_it():
    google = respx.get(GOOGLE_CSE_URL)
    route = respx.get(OTHER).mock(return_value=httpx.Response(200, json=ITEMS))
    creds, _ = await GoogleCSEPlugin().validate({"api_key": "k", "cx": "cx1", "base_url": OTHER}, None)
    assert route.called and not google.called
    assert creds == {"api_key": "k", "cx": "cx1", "base_url": OTHER}


async def test_connect_rejects_a_non_http_address():
    with pytest.raises(IntegrationError):
        await GoogleCSEPlugin().validate({"api_key": "k", "cx": "cx1", "base_url": "file:///etc/passwd"}, None)
