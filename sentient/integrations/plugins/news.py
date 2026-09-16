"""News. Keyless default: Google News RSS. Alternative: NewsAPI (API key)."""

from __future__ import annotations

import asyncio
import re
from urllib.parse import quote_plus

from sentient.integrations.base import IntegrationPlugin, itool, manager_from
from sentient.integrations.common import http_client
from sentient.tools.base import ToolContext

GOOGLE_NEWS = "https://news.google.com/rss"
TOPICS = {"world": "WORLD", "nation": "NATION", "business": "BUSINESS", "technology": "TECHNOLOGY",
          "entertainment": "ENTERTAINMENT", "sports": "SPORTS", "science": "SCIENCE", "health": "HEALTH"}
COUNTRY_NAMES = {
    "india": "IN", "united states": "US", "usa": "US", "us": "US", "united kingdom": "GB", "uk": "GB",
    "england": "GB", "canada": "CA", "australia": "AU", "germany": "DE", "france": "FR", "japan": "JP",
    "singapore": "SG", "united arab emirates": "AE", "uae": "AE", "ireland": "IE", "new zealand": "NZ",
    "south africa": "ZA", "brazil": "BR", "spain": "ES", "italy": "IT", "netherlands": "NL", "mexico": "MX",
    "pakistan": "PK", "bangladesh": "BD", "sri lanka": "LK", "nepal": "NP", "philippines": "PH", "nigeria": "NG",
    "kenya": "KE", "indonesia": "ID", "malaysia": "MY", "china": "CN", "south korea": "KR", "russia": "RU",
}


def _country(ctx: ToolContext, country: str | None) -> str:
    cfg = manager_from(ctx).app.config
    raw = (country or cfg.integrations.news_country or "").strip()
    if not raw and cfg.assistant.location:
        raw = cfg.assistant.location.split(",")[-1].strip()
    if len(raw) == 2:
        return raw.upper()
    return COUNTRY_NAMES.get(raw.lower(), "US")


def _lang(ctx: ToolContext, language: str | None) -> str:
    raw = (language or manager_from(ctx).app.config.assistant.language or "en").strip()
    return (raw.split("-")[0] or "en").lower()


def parse_google_news(xml: str, limit: int = 10) -> list[dict]:
    import feedparser

    feed = feedparser.parse(xml)
    out = []
    for e in feed.entries[:limit]:
        title = e.get("title", "")
        source = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        summary = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
        out.append({"title": title.strip(), "source": source, "url": e.get("link"), "published_at": e.get("published"),
                    "summary": re.sub(r"\s+", " ", summary).strip()[:300] or None})
    return out


async def _google_news(url: str, limit: int) -> list[dict]:
    async with http_client() as http:
        r = await http.get(url)
    r.raise_for_status()
    return await asyncio.to_thread(parse_google_news, r.text, limit)


def _ceid(country: str, lang: str) -> str:
    return f"hl={lang}-{country}&gl={country}&ceid={country}:{lang}"


async def _newsapi_key(ctx: ToolContext) -> str | None:
    mgr = manager_from(ctx)
    if await mgr.is_connected("newsapi"):
        return (await mgr.get_credentials("newsapi") or {}).get("api_key")
    return None


async def _newsapi(key: str, endpoint: str, params: dict) -> list[dict]:
    async with http_client(headers={"X-Api-Key": key}) as http:
        r = await http.get(f"https://newsapi.org/v2/{endpoint}", params=params)
    r.raise_for_status()
    return [{"title": a.get("title"), "source": (a.get("source") or {}).get("name"), "url": a.get("url"),
             "published_at": a.get("publishedAt"), "summary": a.get("description")}
            for a in r.json().get("articles") or []]


@itool("news", "news_top_headlines")
async def news_top_headlines(ctx: ToolContext, country: str | None = None, category: str | None = None,
                             max_results: int = 10) -> dict:
    """Get today's top news headlines. `country` is a 2-letter code or name (default: the user's country);
    `category` is one of world, nation, business, technology, entertainment, sports, science, health."""
    cc = _country(ctx, country)
    lang = _lang(ctx, None)
    n = max(1, min(int(max_results or 10), 30))
    topic = TOPICS.get((category or "").lower().strip())
    key = await _newsapi_key(ctx)
    if key:
        params = {"country": cc.lower(), "pageSize": n}
        if category and category.lower() in {"business", "entertainment", "health", "science", "sports", "technology"}:
            params["category"] = category.lower()
        articles = await _newsapi(key, "top-headlines", params)
        if articles:
            return {"provider": "newsapi", "country": cc, "category": category, "articles": articles}
    url = f"{GOOGLE_NEWS}/headlines/section/topic/{topic}?{_ceid(cc, lang)}" if topic else f"{GOOGLE_NEWS}?{_ceid(cc, lang)}"
    return {"provider": "google_news", "country": cc, "category": category,
            "articles": await _google_news(url, n)}


@itool("news", "news_search")
async def news_search(ctx: ToolContext, query: str, country: str | None = None, max_results: int = 10) -> dict:
    """Search recent news articles about any topic, person or company. Returns headlines, sources, links and dates."""
    cc = _country(ctx, country)
    lang = _lang(ctx, None)
    n = max(1, min(int(max_results or 10), 30))
    key = await _newsapi_key(ctx)
    if key:
        articles = await _newsapi(key, "everything", {"q": query, "language": lang, "sortBy": "publishedAt",
                                                      "pageSize": n})
        return {"provider": "newsapi", "query": query, "articles": articles}
    url = f"{GOOGLE_NEWS}/search?q={quote_plus(query)}&{_ceid(cc, lang)}"
    return {"provider": "google_news", "query": query, "articles": await _google_news(url, n)}


class NewsPlugin(IntegrationPlugin):
    id = "news"
    display_name = "News"
    description = (
        "Top headlines for your country and news search on any topic, from Google News with no account or key "
        "needed. Connect NewsAPI if you prefer it."
    )
    category = "knowledge"
    icon = "news"
    auth_type = "builtin"
    selection_hint = "Use for current news, headlines, or recent articles about a topic."
    tools = [news_top_headlines, news_search]


PLUGIN = NewsPlugin()
