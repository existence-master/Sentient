"""Internet search. Keyless default: DuckDuckGo (ddgs). Alternatives: Brave, Google CSE, SearXNG."""

from __future__ import annotations

import asyncio
import logging

from sentient.integrations.base import IntegrationError, IntegrationPlugin, itool, manager_from
from sentient.integrations.common import http_client
from sentient.tools.base import ToolContext

log = logging.getLogger(__name__)


def _ddg_sync(query: str, n: int, region: str) -> list[dict]:
    from ddgs import DDGS

    try:
        rows = DDGS().text(query, region=region, safesearch="moderate", max_results=n) or []
    except Exception as exc:  # ddgs raises when nothing is found or a backend is rate limited
        if "no results" in str(exc).lower():
            return []
        raise IntegrationError(f"DuckDuckGo search failed: {exc}") from exc
    return [{"title": r.get("title"), "url": r.get("href") or r.get("url"), "snippet": r.get("body")} for r in rows]


async def ddg_search(query: str, n: int, region: str = "wt-wt") -> list[dict]:
    return await asyncio.to_thread(_ddg_sync, query, n, region)


async def brave_search(key: str, query: str, n: int) -> list[dict]:
    async with http_client(headers={"X-Subscription-Token": key, "Accept": "application/json"}) as http:
        r = await http.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": min(n, 20)})
    r.raise_for_status()
    rows = (r.json().get("web") or {}).get("results") or []
    return [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("description")} for x in rows]


async def google_cse_search(key: str, cx: str, query: str, n: int) -> list[dict]:
    async with http_client() as http:
        r = await http.get("https://www.googleapis.com/customsearch/v1",
                           params={"key": key, "cx": cx, "q": query, "num": min(n, 10)})
    r.raise_for_status()
    rows = r.json().get("items") or []
    return [{"title": x.get("title"), "url": x.get("link"), "snippet": x.get("snippet")} for x in rows]


async def searxng_search(base: str, query: str, n: int) -> list[dict]:
    async with http_client() as http:
        r = await http.get(base.rstrip("/") + "/search", params={"q": query, "format": "json"})
    r.raise_for_status()
    rows = r.json().get("results") or []
    return [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content")} for x in rows[:n]]


@itool("internet_search", "web_search")
async def web_search(ctx: ToolContext, query: str, max_results: int = 8) -> dict:
    """Search the internet for current, factual information (news, facts, prices, people, places, how-tos).
    Returns titles, URLs and snippets. Use web_fetch on a result URL to read the full page."""
    mgr = manager_from(ctx)
    cfg = mgr.app.config.integrations
    n = max(1, min(int(max_results or 8), 20))
    provider = cfg.search_provider
    note = None
    results: list[dict] | None = None
    if provider == "brave":
        c = await mgr.get_credentials("brave_search") if await mgr.is_connected("brave_search") else None
        if c and c.get("api_key"):
            results = await brave_search(c["api_key"], query, n)
        else:
            note = "Brave Search isn't connected; used DuckDuckGo instead."
    elif provider == "google_cse":
        c = await mgr.get_credentials("google_cse") if await mgr.is_connected("google_cse") else None
        if c and c.get("api_key") and c.get("cx"):
            results = await google_cse_search(c["api_key"], c["cx"], query, n)
        else:
            note = "Google Custom Search isn't connected; used DuckDuckGo instead."
    elif provider == "searxng":
        if cfg.searxng_url:
            results = await searxng_search(cfg.searxng_url, query, n)
        else:
            note = "No SearXNG URL is set; used DuckDuckGo instead."
    if results is None:
        provider = "duckduckgo"
        results = await ddg_search(query, n)
    out: dict = {"query": query, "provider": provider, "results": results}
    if note:
        out["note"] = note
    if not results:
        out["note"] = (note + " " if note else "") + "No results found. Try different words."
    return out


class SearchPlugin(IntegrationPlugin):
    id = "internet_search"
    display_name = "Internet Search"
    description = (
        "Searches the web for real-time, factual information on any topic. Works out of the box with "
        "DuckDuckGo and needs no account or key. You can switch to Brave Search, Google Custom Search or "
        "your own SearXNG server in Settings."
    )
    category = "knowledge"
    icon = "search"
    auth_type = "builtin"
    selection_hint = "Use to search the internet for current information, facts, prices, events or anything you don't know."
    tools = [web_search]


PLUGIN = SearchPlugin()
