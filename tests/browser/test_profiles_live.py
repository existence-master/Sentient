"""Profiles and attaching with a real browser against local pages (issue #209). Skipped when no browser can start."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time

import httpx
import pytest

from sentient.browser import tools as bt
from sentient.browser.service import _engine_paths, profile_dir
from sentient.config.schema import BrowserProfileConfig
from tests.browser.conftest import make_ctx

REMEMBER = "() => { document.cookie = 'who=' + location.hash.slice(1) + '; max-age=3600; path=/';" \
    " localStorage.setItem('who', location.hash.slice(1)); return true; }"
RECALL = "() => [document.cookie, localStorage.getItem('who')]"


async def _remember(browser, ctx, site: str, who: str, profile: str = "") -> None:
    args = {"url": f"{site}/index.html#{who}"}
    if profile:
        args["profile"] = profile
    res = await bt.browser_open.call(ctx, args)
    assert res["title"] == "Test Shop" and res["profile"] == (profile or browser._profile)
    await browser._active.evaluate(REMEMBER)


async def _recall(browser, ctx, site: str, profile: str) -> list:
    await bt.browser_open.call(ctx, {"url": f"{site}/help.html", "profile": profile})
    return await browser._active.evaluate(RECALL)


async def test_profiles_keep_separate_sign_ins_and_switch(browser, site):
    app = browser.app
    app.config.browser.profiles["work"] = BrowserProfileConfig(notes="Work account")
    ctx = make_ctx(app)

    await _remember(browser, ctx, site, "home", profile="default")
    await _remember(browser, ctx, site, "office", profile="work")
    status = await browser.status()
    assert status["profile"] == "work" and status["running"] and not status["attached"]
    assert profile_dir("work").is_dir()

    # one browser at a time: switching back closes "work" and reopens "default" with its own cookies and storage
    assert await _recall(browser, ctx, site, "default") == ["who=home", "home"]
    assert (await browser.status())["profile"] == "default"
    assert await _recall(browser, ctx, site, "work") == ["who=office", "office"]

    # within the run the picked profile sticks; a new run that picks none gets "default", never "work"
    assert (await bt.browser_snapshot.call(ctx, {}))["title"] == "Help" and browser._profile == "work"
    tabs = await bt.browser_tabs.call(make_ctx(app), {})
    assert tabs["profile"] == "default" and browser._profile == "default"


async def test_task_default_profile_is_used(browser, site):
    app = browser.app
    app.config.browser.profiles["work"] = BrowserProfileConfig()
    ctx = make_ctx(app)
    ctx.extra["browser_profile"] = "work"  # what a task run with browser_profile "work" sets
    snap = await bt.browser_open.call(ctx, {"url": f"{site}/index.html"})
    assert snap["profile"] == "work" and (await browser.status())["profile"] == "work"
    # an explicit profile argument wins and sticks for the rest of the run
    await bt.browser_open.call(ctx, {"url": f"{site}/help.html", "profile": "default"})
    assert ctx.extra["browser_profile"] == "default"
    assert (await bt.browser_snapshot.call(ctx, {}))["title"] == "Help" and browser._profile == "default"


async def test_a_chat_after_a_task_on_another_profile_uses_default(browser, site):
    app = browser.app
    app.config.browser.profiles["x-posting"] = BrowserProfileConfig(notes="Signed in to the X account")
    await _remember(browser, make_ctx(app), site, "home", profile="default")
    task = make_ctx(app)
    task.extra["browser_profile"] = "x-posting"  # a task run on the X profile
    await _remember(browser, task, site, "poster")
    assert (await browser.status())["profile"] == "x-posting"

    chat = make_ctx(app)  # a later chat turn that names no profile
    snap = await bt.browser_open.call(chat, {"url": f"{site}/help.html"})
    assert snap["profile"] == "default" and (await browser.status())["profile"] == "default"
    assert await browser._active.evaluate(RECALL) == ["who=home", "home"]  # the default sign-ins, not the X ones


# ----------------------------------------------------------------------------- attach
def _executable() -> str | None:
    for channel in ("msedge", "chrome"):
        for path in _engine_paths(channel):
            if path.exists():
                return str(path)
    return None


def _answers(port: int) -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            return "webSocketDebuggerUrl" in client.get(f"http://127.0.0.1:{port}/json/version").json()
    except (httpx.HTTPError, ValueError):
        return False


async def _close_over_devtools(port: int) -> None:
    from playwright.async_api import async_playwright

    with contextlib.suppress(Exception):
        async with async_playwright() as pw:
            b = await asyncio.wait_for(pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}"), timeout=30)
            session = await b.new_browser_cdp_session()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(session.send("Browser.close"), timeout=10)


@pytest.fixture
async def own_browser(tmp_path):
    """A browser the 'user' started with a DevTools port, on a throwaway folder. Skipped when none can start.

    On Windows msedge.exe hands over to a detached browser process and exits, so the browser is identified (and
    closed afterwards) by its DevTools port, not by the process we started."""
    exe = _executable()
    if exe is None:
        pytest.skip("no Edge or Chrome installed")
    data = tmp_path / "user-browser"
    args = [exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={data}", "--no-first-run",
            "--no-default-browser-check", "about:blank"]
    if sys.platform.startswith("linux"):
        args.insert(1, "--no-sandbox")  # CI runners may not allow the sandbox; local test pages only
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = data / "DevToolsActivePort"
    deadline = time.monotonic() + 30
    port = None
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError, ValueError, IndexError):
            port = int(port_file.read_text(encoding="utf-8").splitlines()[0])
        if port and _answers(port):
            break
        port = None
        await asyncio.sleep(0.2)
    try:
        if port is None:
            pytest.skip("a browser with a DevTools port could not start")
        yield port
    finally:
        if port is not None:
            await _close_over_devtools(port)
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=15)


async def test_attach_to_a_running_browser_and_detach_leaves_it_running(browser, site, own_browser):
    port = own_browser
    app = browser.app
    await browser.create_profile("mine", kind="attach", endpoint=f"localhost:{port}")
    assert app.config.browser.profiles["mine"].endpoint == f"http://localhost:{port}"
    ctx = make_ctx(app)

    snap = await bt.browser_open.call(ctx, {"url": f"{site}/index.html", "profile": "mine"})
    assert snap["title"] == "Test Shop" and snap["profile"] == "mine"
    status = await browser.status()
    assert status["attached"] and status["profile"] == "mine" and status["engine"] is None
    assert len(status["tabs"]) >= 2  # the user's own tab plus the one Sentient opened for itself

    # the same safety applies inside the user's browser
    app.config.tools.approvals.mode = "ask"
    text = snap["text"]
    pw = next(line.split("]")[0][1:] for line in text.splitlines() if 'textbox "Password"' in line)
    assert "password" in (await bt.browser_type.call(ctx, {"ref": pw, "text": "hunter2"}))["error"]
    app.config.browser.block_domains = ["127.0.0.1"]
    assert "blocked" in (await bt.browser_open.call(ctx, {"url": f"{site}/help.html"}))["error"]
    app.config.browser.block_domains = []

    closed = await bt.browser_close.call(ctx, {})
    assert closed["ok"] and not browser.running and not (await browser.status())["attached"]
    await asyncio.sleep(0.5)
    assert _answers(port), "detaching must never close the user's browser"
    with httpx.Client(trust_env=False, timeout=5) as client:
        pages = [p.get("url", "") for p in client.get(f"http://127.0.0.1:{port}/json/list").json()]
    assert any(u.endswith("/index.html") for u in pages)  # Sentient's tab stays for the user

    # switching from the attached browser to a launched profile also only disconnects
    await bt.browser_open.call(ctx, {"url": f"{site}/index.html", "profile": "mine"})
    await bt.browser_open.call(ctx, {"url": f"{site}/help.html", "profile": "default"})
    assert browser._profile == "default" and not browser._attached and _answers(port)


async def test_writes_in_unload_handlers_survive_a_switch(browser, site):
    app = browser.app
    app.config.browser.profiles["work"] = BrowserProfileConfig()
    ctx = make_ctx(app)
    await bt.browser_open.call(ctx, {"url": f"{site}/index.html", "profile": "default"})
    await browser._active.evaluate(
        "() => { addEventListener('beforeunload', () => localStorage.setItem('bye', 'before'));"
        " addEventListener('pagehide', () => localStorage.setItem('hide', 'yes')); return true; }"
    )
    await bt.browser_open.call(ctx, {"url": f"{site}/help.html", "profile": "work"})  # closes "default"
    await bt.browser_open.call(ctx, {"url": f"{site}/help.html", "profile": "default"})
    stored = await browser._active.evaluate("() => [localStorage.getItem('bye'), localStorage.getItem('hide')]")
    assert stored == ["before", "yes"]
