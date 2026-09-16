"""Optional keyed providers that replace a keyless builtin (no tools of their own)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField
from sentient.integrations.common import http_client

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager


def _require(fields: dict[str, str], *keys: str) -> list[str]:
    vals = [str(fields.get(k, "")).strip() for k in keys]
    if not all(vals):
        raise IntegrationError("Please paste the key first.")
    return vals


def _rejected(r: httpx.Response, service: str) -> None:
    if r.status_code in (401, 403):
        raise IntegrationError(f"{service} didn't accept that key. Check it was copied completely.")
    if r.status_code == 429:
        raise IntegrationError(f"{service} says the key is over its request limit. Try again later.")
    if r.status_code >= 400:
        raise IntegrationError(f"{service} returned an error (HTTP {r.status_code}).")


class AccuWeatherPlugin(IntegrationPlugin):
    id = "accuweather"
    display_name = "AccuWeather"
    description = ("Optional: use AccuWeather instead of the free Open-Meteo service for weather. "
                   "Needs a free AccuWeather developer key.")
    category = "utilities"
    icon = "weather"
    auth_type = "api_key"
    optional_alternative_for = "weather"
    selection_hint = ""
    tools = []
    setup_fields = [SetupField("api_key", "AccuWeather API key", secret=True, help="From developer.accuweather.com → My Apps.")]
    docs_url = "https://developer.accuweather.com/getting-started"
    instructions_md = (
        "1. Open https://developer.accuweather.com/ and click **Register** to create a free developer account.\n"
        "2. After signing in, open **My Apps** and click **Add a new App**.\n"
        "3. Give it any name (for example `Sentient`), choose the free **Core Weather** limited trial, and save.\n"
        "4. Open the app and copy its **API Key**.\n"
        "5. Paste it here and click **Connect**.\n"
        "6. In **Settings → Integrations**, set the weather provider to **AccuWeather**.\n"
    )

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        (key,) = _require(fields, "api_key")
        async with http_client() as http:
            r = await http.get("https://dataservice.accuweather.com/locations/v1/cities/search",
                               params={"apikey": key, "q": "London"})
        _rejected(r, "AccuWeather")
        return {"api_key": key}, None


class NewsAPIPlugin(IntegrationPlugin):
    id = "newsapi"
    display_name = "NewsAPI"
    description = "Optional: use NewsAPI.org instead of Google News for headlines and news search."
    category = "knowledge"
    icon = "news"
    auth_type = "api_key"
    optional_alternative_for = "news"
    tools = []
    setup_fields = [SetupField("api_key", "NewsAPI key", secret=True, help="From newsapi.org/account.")]
    docs_url = "https://newsapi.org/docs/get-started"
    instructions_md = (
        "1. Open https://newsapi.org/register and create a free account.\n"
        "2. After confirming your email, open https://newsapi.org/account.\n"
        "3. Copy the **API key** shown there.\n"
        "4. Paste it here and click **Connect**. Sentient will use NewsAPI for news from now on.\n"
    )

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        (key,) = _require(fields, "api_key")
        async with http_client(headers={"X-Api-Key": key}) as http:
            r = await http.get("https://newsapi.org/v2/top-headlines", params={"country": "us", "pageSize": 1})
        _rejected(r, "NewsAPI")
        return {"api_key": key}, None


class BravePlugin(IntegrationPlugin):
    id = "brave_search"
    display_name = "Brave Search"
    description = "Optional: search the web with the Brave Search API instead of DuckDuckGo."
    category = "knowledge"
    icon = "search"
    auth_type = "api_key"
    optional_alternative_for = "internet_search"
    tools = []
    setup_fields = [SetupField("api_key", "Brave Search API key", secret=True, help="From api-dashboard.search.brave.com.")]
    docs_url = "https://api-dashboard.search.brave.com/app/documentation/web-search/get-started"
    instructions_md = (
        "1. Open https://api-dashboard.search.brave.com/ and sign up.\n"
        "2. Open **Subscriptions** and choose the **Free** plan for Web Search (a card may be asked for "
        "verification; the free plan isn't charged).\n"
        "3. Open **API Keys**, click **Add API key**, and copy it.\n"
        "4. Paste it here and click **Connect**.\n"
        "5. In **Settings → Integrations**, set the search provider to **brave**.\n"
    )

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        (key,) = _require(fields, "api_key")
        async with http_client(headers={"X-Subscription-Token": key, "Accept": "application/json"}) as http:
            r = await http.get("https://api.search.brave.com/res/v1/web/search", params={"q": "sentient", "count": 1})
        _rejected(r, "Brave Search")
        return {"api_key": key}, None


class GoogleCSEPlugin(IntegrationPlugin):
    id = "google_cse"
    display_name = "Google Custom Search"
    description = "Optional: search the web with Google's Custom Search JSON API instead of DuckDuckGo."
    category = "knowledge"
    icon = "search"
    auth_type = "api_key"
    optional_alternative_for = "internet_search"
    tools = []
    setup_fields = [
        SetupField("api_key", "API key", secret=True, help="Google Cloud Console → APIs & Services → Credentials."),
        SetupField("cx", "Search engine ID", secret=False, help="From programmablesearchengine.google.com (the 'cx' value)."),
    ]
    docs_url = "https://developers.google.com/custom-search/v1/overview"
    instructions_md = (
        "1. Open https://programmablesearchengine.google.com/controlpanel/create and create a search engine. "
        "Choose **Search the entire web**.\n"
        "2. Open the engine's **Overview** and copy the **Search engine ID**.\n"
        "3. Open https://console.cloud.google.com/apis/library/customsearch.googleapis.com and click **Enable**.\n"
        "4. Open https://console.cloud.google.com/apis/credentials, click **Create credentials → API key**, and copy it.\n"
        "5. Paste both values here and click **Connect**.\n"
        "6. In **Settings → Integrations**, set the search provider to **google_cse**.\n"
    )

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        key, cx = _require(fields, "api_key", "cx")
        async with http_client() as http:
            r = await http.get("https://www.googleapis.com/customsearch/v1", params={"key": key, "cx": cx, "q": "test", "num": 1})
        if r.status_code == 400:
            raise IntegrationError("Google didn't accept the search engine ID. Check the 'cx' value.")
        _rejected(r, "Google Custom Search")
        return {"api_key": key, "cx": cx}, None


class GoogleMapsPlugin(IntegrationPlugin):
    id = "google_maps"
    display_name = "Google Maps"
    description = "Optional: use Google Maps for place search and directions (including public transit) instead of OpenStreetMap."
    category = "utilities"
    icon = "map"
    auth_type = "api_key"
    optional_alternative_for = "maps"
    tools = []
    setup_fields = [SetupField("api_key", "Google Maps API key", secret=True, help="Google Cloud Console → Credentials.")]
    docs_url = "https://developers.google.com/maps/get-started"
    instructions_md = (
        "1. Open https://console.cloud.google.com/ and select (or create) a project. Google Maps requires a "
        "billing account, but includes a monthly free allowance.\n"
        "2. Enable **Places API (New)**: https://console.cloud.google.com/apis/library/places.googleapis.com\n"
        "3. Enable **Directions API**: https://console.cloud.google.com/apis/library/directions-backend.googleapis.com\n"
        "4. Enable **Geocoding API**: https://console.cloud.google.com/apis/library/geocoding-backend.googleapis.com\n"
        "5. Open https://console.cloud.google.com/apis/credentials, click **Create credentials → API key**, and copy it.\n"
        "6. Paste it here and click **Connect**.\n"
    )

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        (key,) = _require(fields, "api_key")
        async with http_client() as http:
            r = await http.get("https://maps.googleapis.com/maps/api/geocode/json", params={"address": "London", "key": key})
        _rejected(r, "Google Maps")
        status = r.json().get("status")
        if status not in {"OK", "ZERO_RESULTS"}:
            raise IntegrationError(f"Google Maps didn't accept that key ({status}). Make sure the APIs are enabled.")
        return {"api_key": key}, None


PLUGINS = [AccuWeatherPlugin(), NewsAPIPlugin(), BravePlugin(), GoogleCSEPlugin(), GoogleMapsPlugin()]
