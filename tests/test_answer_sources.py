"""Sources under answers (#174): pages, emails, files and apps a reply used, recorded from tool results."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient.agent.sources import MAX_SOURCES, SEARCH_RESULTS_PER_CALL, AnswerSources
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.integrations.plugins import web as web_mod
from sentient.llm.events import Done
from tests.conftest import FakeProvider, tool_call

PAGE = (
    "<html><head><title>Greyhound care guide</title></head><body><h1>First week</h1>"
    "<p>Retired racers need a soft bed and quiet days.</p></body></html>"
)


def test_web_pages_search_hits_files_emails_and_apps():
    s = AnswerSources()
    s.add_tool_result("web_fetch", {"url": "https://example.org/dogs", "title": "Dogs", "content": "Soft  beds\nhelp."},
                      plugin="web", app_name="Web Pages", is_app=True)
    s.add_tool_result("web_search", {"results": [
        {"title": f"Result {i}", "url": f"https://example.org/r{i}", "snippet": "s"} for i in range(6)
    ]}, plugin="internet_search", app_name="Internet Search", is_app=True)
    s.add_tool_result("file_read", {"name": "uploads/plan.md", "content": "The plan"}, plugin="files")
    s.add_tool_result("gmail_search", {"messages": [
        {"id": "m1", "subject": "Adoption papers", "from": "Shelter <hi@shelter.org>", "snippet": "Signed copy",
         "url": "https://mail.google.com/mail/u/0/#all/m1"},
    ]}, plugin="gmail", app_name="Gmail", is_app=True)
    s.add_tool_result("gcal_list_events", {"events": [{"summary": "Vet visit"}]}, plugin="gcalendar",
                      app_name="Google Calendar", is_app=True)
    s.add_tool_result("gcal_get_event", {"summary": "Vet visit"}, plugin="gcalendar", app_name="Google Calendar",
                      is_app=True)
    items = s.items()
    kinds = [i["kind"] for i in items]
    assert kinds == ["web"] * (1 + SEARCH_RESULTS_PER_CALL) + ["file", "email", "app"]
    assert items[0] == {"kind": "web", "title": "Dogs", "url": "https://example.org/dogs", "snippet": "Soft beds help.",
                        "app": "web", "tool": "web_fetch"}
    assert items[4] == {"kind": "file", "title": "plan.md", "file": "uploads/plan.md", "snippet": "The plan",
                        "app": "files", "tool": "file_read"}
    assert items[5]["title"] == "Adoption papers" and items[5]["from"] == "Shelter <hi@shelter.org>"
    assert items[5]["url"].startswith("https://mail.google.com/")
    assert items[6] == {"kind": "app", "title": "Google Calendar", "app": "gcalendar", "tool": "gcal_list_events",
                        "tools": ["gcal_list_events", "gcal_get_event"]}


def test_errors_declines_duplicates_core_tools_and_the_cap():
    s = AnswerSources()
    s.add_tool_result("web_fetch", {"error": "That page could not be found."}, plugin="web", is_app=True)
    s.add_tool_result("gmail_send", {"declined": True, "error": "x"}, plugin="gmail", is_app=True)
    s.add_tool_result("memory_recall", [{"id": 1, "fact": "x"}], plugin="memory")  # memories live in memory_sources
    s.add_tool_result("web_fetch", {"url": "javascript:alert(1)", "title": "x", "content": "x"}, plugin="web")
    assert s.items() == []
    for _ in range(2):
        s.add_tool_result("web_fetch", {"url": "https://example.org/a", "title": "A", "content": "a"}, plugin="web")
    assert len(s) == 1
    for i in range(MAX_SOURCES + 5):
        s.add_tool_result("web_fetch", {"url": f"https://example.org/{i}", "title": "p", "content": "c"}, plugin="web")
    assert len(s) == MAX_SOURCES
    md = AnswerSources()
    md.add_tool_result("web_fetch", {"url": "https://example.org/m", "title": "M",
                                     "content": "# Checklist\n- Bring a collar\n1. Book a vet"}, plugin="web")
    assert md.items()[0]["snippet"] == "Checklist Bring a collar Book a vet"
    long = AnswerSources()
    long.add_tool_result("web_fetch", {"url": "https://example.org/l", "title": "L", "content": "word " * 400},
                         plugin="web")
    assert len(long.items()[0]["snippet"]) <= 280


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "src.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()


async def test_a_reply_that_read_a_page_carries_it_and_one_without_tools_has_none(app, monkeypatch):
    monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.215.14", 0))])
    sid = await app.store.create_session(channel="desktop")
    app.fake.replies.extend([[tool_call("web_fetch", url="https://example.org/greyhounds")], "Give him a soft bed."])
    with respx.mock() as router:
        router.get("https://example.org/greyhounds").mock(
            return_value=httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
        )
        events = [ev async for ev in app.agent.run_turn(sid, "tips for a retired greyhound?", channel="desktop")]
    [done] = [e for e in events if isinstance(e, Done)]
    assert [(s["kind"], s["title"], s["url"]) for s in done.sources] == [
        ("web", "Greyhound care guide", "https://example.org/greyhounds")
    ]
    assert "soft bed" in done.sources[0]["snippet"]
    rows = await app.store.recent_messages(sid, 20)
    final = [r for r in rows if r["role"] == "assistant"][-1]
    assert final["sources"] == done.sources
    assert all(r["sources"] == [] for r in rows if r is not final)

    app.fake.replies.append("Hello!")
    events = [ev async for ev in app.agent.run_turn(sid, "thanks", channel="desktop")]
    [done] = [e for e in events if isinstance(e, Done)]
    assert done.sources == []


def test_sources_are_in_the_transcript_route(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.215.14", 0))])
    llm = FakeProvider(replies=[[tool_call("web_fetch", url="https://example.org/greyhounds")], "Soft bed."])
    gw = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "g.db", enable_background=False))
    with TestClient(gw) as c, respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.get("https://example.org/greyhounds").mock(
            return_value=httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
        )
        c.headers.update({"Authorization": "Bearer test-token"})
        lines = [json.loads(x) for x in c.post("/api/chat", json={"text": "greyhound tips"}).text.splitlines()]
        sid = lines[0]["session_id"]
        [done] = [x for x in lines if x["type"] == "done"]
        assert done["sources"][0]["url"] == "https://example.org/greyhounds"
        rows = c.get(f"/api/sessions/{sid}/messages").json()
    assert rows[-1]["sources"] == done["sources"]
