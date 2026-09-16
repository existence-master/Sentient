from __future__ import annotations

import httpx
import respx

from sentient import paths
from sentient.integrations.plugins import news as news_mod
from sentient.integrations.plugins import web as web_mod

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<title>Top stories - Google News</title>
<item><title>Monsoon brings heavy rain to Pune - The Times of India</title>
<link>https://news.google.com/rss/articles/abc</link><pubDate>Tue, 15 Sep 2026 06:00:00 GMT</pubDate>
<description>&lt;a href="x"&gt;Monsoon brings heavy rain&lt;/a&gt;</description>
<source url="https://timesofindia.indiatimes.com">The Times of India</source></item>
<item><title>Markets rally on rate cut hopes - Mint</title>
<link>https://news.google.com/rss/articles/def</link><pubDate>Tue, 15 Sep 2026 05:00:00 GMT</pubDate>
<source url="https://www.livemint.com">Mint</source></item>
</channel></rss>"""


async def test_duckduckgo_search(app, ctx, monkeypatch):
    import ddgs

    seen = {}

    class FakeDDGS:
        def text(self, query, region, safesearch, max_results):
            seen.update(query=query, max_results=max_results)
            return [{"title": "Pune monsoon", "href": "https://example.org/pune", "body": "Rain update"}]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    res = await app.registry.get("web_search").call(ctx, {"query": "Pune monsoon 2026", "max_results": 3})
    assert res["provider"] == "duckduckgo"
    assert res["results"] == [{"title": "Pune monsoon", "url": "https://example.org/pune", "snippet": "Rain update"}]
    assert seen == {"query": "Pune monsoon 2026", "max_results": 3}


async def test_keyed_search_provider_falls_back(app, ctx, monkeypatch):
    import ddgs

    class FakeDDGS:
        def text(self, *a, **k):
            return []

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    app.config.integrations.search_provider = "brave"
    res = await app.registry.get("web_search").call(ctx, {"query": "x"})
    assert res["provider"] == "duckduckgo" and "Brave Search isn't connected" in res["note"]


async def test_open_meteo_weather(app, ctx):
    app.config.assistant.location = "Pune, India"
    geo = {"results": [
        {"name": "Pune", "latitude": 1.0, "longitude": 2.0, "country": "Somewhere", "admin1": "X", "population": 10},
        {"name": "Pune", "latitude": 18.52, "longitude": 73.86, "country": "India", "country_code": "IN",
         "admin1": "Maharashtra", "population": 3000000, "timezone": "Asia/Kolkata"},
    ]}
    forecast = {"timezone": "Asia/Kolkata",
                "current": {"time": "2026-09-15T12:00", "temperature_2m": 26.5, "apparent_temperature": 29.0,
                            "relative_humidity_2m": 88, "precipitation": 1.2, "weather_code": 63,
                            "wind_speed_10m": 14.0, "is_day": 1},
                "daily": {"time": ["2026-09-15", "2026-09-16"], "weather_code": [63, 3],
                          "temperature_2m_max": [28.0, 29.0], "temperature_2m_min": [22.0, 23.0],
                          "precipitation_probability_max": [90, 40], "precipitation_sum": [12.0, 1.0],
                          "sunrise": ["06:20", "06:20"], "sunset": ["18:35", "18:34"]}}
    with respx.mock() as router:
        g = router.get("https://geocoding-api.open-meteo.com/v1/search").mock(return_value=httpx.Response(200, json=geo))
        f = router.get("https://api.open-meteo.com/v1/forecast").mock(return_value=httpx.Response(200, json=forecast))
        res = await app.registry.get("weather_current").call(ctx, {})
        assert g.calls.last.request.url.params["name"] == "Pune"
        assert f.calls.last.request.url.params["latitude"] == "18.52"
        week = await app.registry.get("weather_forecast").call(ctx, {"location": "Pune, India", "days": 2})
    assert res["location"] == "Pune, Maharashtra, India" and res["provider"] == "open-meteo"
    assert res["current"]["condition"] == "Rain" and res["current"]["temperature_c"] == 26.5
    assert res["today"]["rain_chance_percent"] == 90
    assert [d["condition"] for d in week["forecast"]] == ["Rain", "Overcast"]


async def test_weather_without_location_is_friendly(app, ctx):
    app.config.assistant.location = ""
    res = await app.registry.get("weather_current").call(ctx, {})
    assert "Which city" in res["error"]


def test_google_news_rss_parsing():
    items = news_mod.parse_google_news(RSS)
    assert items[0]["title"] == "Monsoon brings heavy rain to Pune"
    assert items[0]["source"] == "The Times of India"
    assert items[0]["url"] == "https://news.google.com/rss/articles/abc"
    assert items[1]["title"] == "Markets rally on rate cut hopes" and items[1]["source"] == "Mint"


async def test_news_top_headlines_uses_country_from_location(app, ctx):
    app.config.assistant.location = "Pune, India"
    with respx.mock() as router:
        route = router.get(url__startswith="https://news.google.com/rss").mock(return_value=httpx.Response(200, text=RSS))
        res = await app.registry.get("news_top_headlines").call(ctx, {})
        url = str(route.calls.last.request.url)
    assert "ceid=IN:en" in url and "gl=IN" in url
    assert res["country"] == "IN" and len(res["articles"]) == 2


async def test_web_fetch_readable_text(app, ctx, monkeypatch):
    monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.215.14", 0))])
    html = ("<html><head><title>Example</title><script>var x=1;</script></head><body><nav>Menu</nav>"
            "<h1>Hello</h1><p>World &amp; friends</p><footer>foot</footer></body></html>")
    with respx.mock() as router:
        router.get("https://example.com/").mock(return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"}))
        res = await app.registry.get("web_fetch").call(ctx, {"url": "https://example.com/"})
    assert res["title"] == "Example"
    assert "# Hello" in res["content"] and "World & friends" in res["content"]
    assert "var x" not in res["content"] and "Menu" not in res["content"] and "foot" not in res["content"]


async def test_web_fetch_blocks_localhost(app, ctx):
    res = await app.registry.get("web_fetch").call(ctx, {"url": "http://localhost:7777/api/config"})
    assert "isn't allowed" in res["error"]


async def test_charts_url_and_save(app, ctx):
    cfg = {"type": "bar", "data": {"labels": ["a", "b"], "datasets": [{"label": "n", "data": [1, 2]}]}}
    res = await app.registry.get("chart_create_url").call(ctx, {"chart_config": cfg})
    assert res["url"].startswith("https://quickchart.io/chart?")
    bad = await app.registry.get("chart_create_url").call(ctx, {"chart_config": {"type": "nope", "data": {}}})
    assert "Chart type" in bad["error"]
    with respx.mock() as router:
        router.post("https://quickchart.io/chart").mock(
            return_value=httpx.Response(200, content=b"\x89PNG\r\n", headers={"content-type": "image/png"}))
        saved = await app.registry.get("chart_save_image").call(ctx, {"chart_config": cfg, "filename": "steps"})
    assert saved["saved"] == "charts/steps.png"
    assert (paths.files_dir() / "charts" / "steps.png").read_bytes().startswith(b"\x89PNG")
