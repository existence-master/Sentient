"""Real headless browser against a local web site (no internet). Skipped when no browser can launch."""

from __future__ import annotations

import asyncio
import re

from sentient import paths
from sentient.browser import tools as bt
from sentient.tools.base import Risk
from tests.browser.conftest import make_ctx


def ref_of(text: str, pattern: str) -> str:
    m = re.search(r"\[(e\d+)\] " + pattern, text)
    assert m, f"{pattern!r} not in snapshot:\n{text}"
    return m.group(1)


async def test_open_type_click_extract_screenshot(browser, site):
    app = browser.app
    frames: list = []
    ctx = make_ctx(app, frames)
    async with app.bus.subscribe() as q:
        snap = await bt.browser_open.call(ctx, {"url": f"{site}/index.html"})
        assert snap["title"] == "Test Shop"
        text = snap["text"]
        assert 'button "Search"' in text and 'textbox "Search" value=""' in text
        assert "Ghost button" not in text
        assert 'combobox "Country" value="India" options: India | Japan' in text
        assert "blue notebooks" in text

        search = ref_of(text, 'textbox "Search"')
        res = await bt.browser_type.call(ctx, {"ref": search, "text": "notebook", "submit": True})
        assert res["ok"] and "search.html?q=notebook" in res["url"]

        stale = await bt.browser_click.call(ctx, {"ref": search})
        assert "browser_snapshot" in stale["error"]

        back = await bt.browser_back.call(ctx, {})
        assert back["url"].endswith("/index.html")

        page = await bt.browser_extract.call(ctx, {})
        assert "Returns are accepted for thirty days" in page["content"]
        assert "Menu text that is navigation" not in page["content"]

        shot = await bt.browser_screenshot.call(ctx, {})
        assert shot["file"].startswith("outputs/browser/") and (paths.files_dir() / shot["file"]).stat().st_size > 1000

        events = []
        while not q.empty():
            events.append(q.get_nowait())
    assert frames and frames[0]["kind"] == "frame" and frames[0]["image"].startswith("data:image/jpeg;base64,")
    assert any(e["type"] == "browser.frame" for e in events)
    assert any(e["type"] == "browser.updated" for e in events)
    jpeg = await browser.capture_jpeg()
    assert jpeg and jpeg[:2] == b"\xff\xd8"


async def test_download_is_saved_and_reported(browser, site):
    ctx = make_ctx(browser.app)
    snap = await bt.browser_open.call(ctx, {"url": f"{site}/download.html"})
    link = ref_of(snap["text"], r'link "Download report"')
    result = await bt.browser_click.call(ctx, {"ref": link})
    download_paths = result.get("downloads", [])
    if not download_paths:
        assert result["downloads_in_progress"] == ["report.txt"]
        pending = [task for task in browser._download_tasks if not task.done()]
        assert pending
        await asyncio.wait_for(asyncio.gather(*pending), timeout=10)
        later_result = await bt.browser_snapshot.call(ctx, {})
        download_paths = later_result["downloads"]
    assert download_paths == ["downloads/report.txt"]
    saved = paths.files_dir() / download_paths[0]
    assert saved.read_text(encoding="utf-8") == "Sentient browser download test.\n"


async def test_open_direct_download_is_reported_as_normal_result(browser, site):
    ctx = make_ctx(browser.app)
    result = await bt.browser_open.call(ctx, {"url": f"{site}/direct-download"})

    download_paths = result.get("downloads", [])
    if not download_paths:
        assert result["downloads_in_progress"] == ["direct-report.txt"]
        pending = [task for task in browser._download_tasks if not task.done()]
        assert pending
        await asyncio.wait_for(asyncio.gather(*pending), timeout=10)
        later_result = await bt.browser_snapshot.call(ctx, {})
        download_paths = later_result["downloads"]
    assert download_paths == ["downloads/direct-report.txt"]
    saved = paths.files_dir() / download_paths[0]
    assert saved.read_text(encoding="utf-8") == "Direct browser-open download test.\n"


async def test_safety_in_real_pages(browser, site):
    app = browser.app
    app.config.tools.approvals.mode = "ask"
    ctx = make_ctx(app)
    text = (await bt.browser_open.call(ctx, {"url": f"{site}/index.html"}))["text"]

    pw = ref_of(text, r'textbox "Password"')
    refused = await bt.browser_type.call(ctx, {"ref": pw, "text": "hunter2"})
    assert "password" in refused["error"] and "Open browser" in refused["error"]
    card = ref_of(text, r'textbox "Card number"')
    assert "card" in (await bt.browser_type.call(ctx, {"ref": card, "text": "4111"}))["error"]
    coupon = ref_of(text, r'textbox "Coupon"')
    assert "card" in (await bt.browser_type.call(ctx, {"ref": coupon, "text": "4111 1111 1111 1111"}))["error"]

    place = ref_of(text, r'button "Place order"')
    assert bt.browser_click.risk_fn({"ref": place}, ctx) == Risk.send
    assert bt.browser_click.risk_fn({"ref": ref_of(text, r'button "Search"')}, ctx) == Risk.write

    delete = ref_of(text, r'button "Delete account"')
    # no approval check happened for this call (risk_fn not consulted): refuse
    assert "Open browser" in (await bt.browser_click.call(ctx, {"ref": delete}))["error"]
    # approvals saw it as send: the click goes through and the confirm() dialog is accepted
    assert bt.browser_click.risk_fn({"ref": delete}, ctx) == Risk.send
    res = await bt.browser_click.call(ctx, {"ref": delete})
    assert res["ok"] and res["dialogs"] == ["confirm: Really delete?"]
    assert "deleted-ok" in (await bt.browser_extract.call(ctx, {}))["content"]

    app.config.browser.block_domains = ["127.0.0.1"]
    blocked = await bt.browser_open.call(ctx, {"url": f"{site}/help.html"})
    assert "blocked" in blocked["error"]
    app.config.browser.block_domains = []
    app.config.browser.allow_domains = ["example.com"]
    assert "allowed" in (await bt.browser_open.call(ctx, {"url": f"{site}/help.html"}))["error"]
    assert "http" in (await bt.browser_open.call(ctx, {"url": "file:///C:/Windows/win.ini"}))["error"]


async def test_tabs_select_scroll_press(browser, site):
    ctx = make_ctx(browser.app)
    text = (await bt.browser_open.call(ctx, {"url": f"{site}/index.html"}))["text"]

    country = ref_of(text, r'combobox "Country"')
    assert (await bt.browser_select.call(ctx, {"ref": country, "option": "Japan"}))["ok"]
    assert 'combobox "Country" value="Japan"' in (await bt.browser_snapshot.call(ctx, {}))["text"]
    assert "isn't a dropdown" in (await bt.browser_select.call(ctx, {"ref": ref_of(text, r'button "Search"'), "option": "x"}))["error"]

    scrolled = await bt.browser_scroll.call(ctx, {"direction": "down"})
    assert scrolled["from_top"] > 0
    assert (await bt.browser_press.call(ctx, {"key": "home"}))["pressed"] == "Home"

    link = ref_of((await bt.browser_snapshot.call(ctx, {}))["text"], r'link "Open help in new tab"')
    res = await bt.browser_click.call(ctx, {"ref": link})
    assert res.get("new_tab") is True
    tabs = (await bt.browser_tabs.call(ctx, {}))["tabs"]
    assert len(tabs) == 2 and tabs[1]["active"]
    assert (await bt.browser_snapshot.call(ctx, {}))["title"] == "Help"
    switched = await bt.browser_switch_tab.call(ctx, {"index": 0})
    assert switched["title"] == "Test Shop"
    assert "no tab 5" in (await bt.browser_switch_tab.call(ctx, {"index": 5}))["error"]


async def test_relaunch_user_close_and_idle(browser, site):
    app = browser.app
    ctx = make_ctx(app)
    # same profile relaunched (headless here so no window pops up during tests)
    await browser._restart(headless=True, url=f"{site}/help.html")
    status = await browser.status()
    assert status["running"] and status["tabs"][-1]["url"].endswith("/help.html")

    # the user closes the window: state resets and the next tool call relaunches with the configured mode
    async with app.bus.subscribe() as q:
        await browser._context.close()

        async def closed_event() -> dict:
            while True:
                ev = await q.get()
                if ev["type"] == "browser.updated" and ev["data"]["running"] is False:
                    return ev

        event = await asyncio.wait_for(closed_event(), timeout=5)
    assert event["data"]["tabs"] == [] and not browser.running
    snap = await bt.browser_open.call(ctx, {"url": f"{site}/index.html"})
    assert snap["title"] == "Test Shop" and browser.running and browser._headless is True

    closed = await bt.browser_close.call(ctx, {})
    assert closed["ok"] and not browser.running

    app.config.browser.idle_minutes = 0.001
    browser.idle_check_s = 0.05
    await bt.browser_snapshot.call(ctx, {})
    assert browser.running
    for _ in range(100):
        if not browser.running:
            break
        await asyncio.sleep(0.05)
    assert not browser.running
