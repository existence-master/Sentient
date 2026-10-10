"""Named browser profiles and attaching to a running browser, without launching one (issue #209)."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading

import pytest
from fastapi.testclient import TestClient

from sentient import paths
from sentient.app import SentientApp
from sentient.browser import safety
from sentient.browser import service as browser_service
from sentient.browser import tools as bt
from sentient.browser.service import (
    BrowserError,
    BrowserService,
    devtools_ws_url,
    profile_dir,
    profile_name,
)
from sentient.config.schema import BrowserProfileConfig, SentientConfig
from sentient.gateway.app import create_app
from sentient.skills.loader import SkillLibrary
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.builtin.skills_tool import skill_view
from tests.browser.conftest import make_app, make_ctx
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import RESULT


# ----------------------------------------------------------------------------- addresses
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9333", "http://127.0.0.1:9333"),
        ("127.0.0.1:9333", "http://127.0.0.1:9333"),
        ("http://localhost:9333/json/version", "http://localhost:9333"),
        ("http://[::1]:9222", "http://[::1]:9222"),
        ("ws://127.0.0.1:9222/devtools/browser/abc", "ws://127.0.0.1:9222/devtools/browser/abc"),
    ],
)
def test_devtools_endpoint_accepts_this_computer(raw, expected):
    assert safety.devtools_endpoint(raw) == (expected, None)


@pytest.mark.parametrize(
    "raw",
    ["http://192.168.1.20:9222", "10.0.0.5:9222", "http://example.com:9222", "http://127.0.0.1.nip.io:9222",
     "ws://0.0.0.0:9222/devtools/browser/x"],
)
def test_devtools_endpoint_refuses_other_computers(raw):
    assert safety.devtools_endpoint(raw) == ("", safety.NOT_LOCAL)


def test_devtools_endpoint_needs_a_port_and_a_web_scheme():
    assert "port" in safety.devtools_endpoint("http://127.0.0.1")[1]
    assert "isn't valid" in safety.devtools_endpoint("file://127.0.0.1:9222")[1]
    assert "Add the DevTools address" in safety.devtools_endpoint("")[1]


def _json_server(payload: dict):
    body = json.dumps(payload).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def test_devtools_port_pointing_elsewhere_is_refused():
    server = _json_server({"webSocketDebuggerUrl": "ws://10.0.0.5:9222/devtools/browser/x"})
    try:
        with pytest.raises(BrowserError, match="only attaches to a browser on this computer"):
            await devtools_ws_url(f"http://127.0.0.1:{server.server_address[1]}")
    finally:
        server.shutdown()
        server.server_close()


async def test_nothing_listening_is_explained():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.server_close()
    with pytest.raises(BrowserError, match="--remote-debugging-port"):
        await devtools_ws_url(f"http://127.0.0.1:{port}")


async def test_attach_profile_on_another_computer_is_refused_before_connecting():
    cfg = SentientConfig()
    cfg.browser.profiles["remote"] = BrowserProfileConfig(kind="attach", endpoint="http://192.168.1.20:9222")
    app = make_app(cfg)
    svc = BrowserService(app)
    app.browser = svc
    await svc.start()
    try:
        res = await bt.browser_open.call(make_ctx(app), {"url": "https://example.com", "profile": "remote"})
        assert res["error"] == safety.NOT_LOCAL and not svc.running
        missing = await bt.browser_open.call(make_ctx(app), {"url": "https://example.com", "profile": "nope"})
        assert "no browser profile named 'nope'" in missing["error"] and "default, remote" in missing["error"]
    finally:
        await svc.stop()


# ----------------------------------------------------------------------------- folders and config
def test_profile_names_and_folders():
    assert profile_name("  X Growth ") == "x-growth"
    with pytest.raises(BrowserError):
        profile_name("../..")
    with pytest.raises(BrowserError):
        profile_dir("../escape")
    assert profile_dir("work") == paths.home() / "browser" / "profiles" / "work"


def test_old_default_profile_folder_moves_once():
    old = paths.home() / "browser" / "profile"
    (old / "Default").mkdir(parents=True)
    (old / "Default" / "Cookies").write_text("kept", encoding="utf-8")
    new = profile_dir("default")
    assert new == paths.home() / "browser" / "profiles" / "default"
    assert (new / "Default" / "Cookies").read_text(encoding="utf-8") == "kept" and not old.exists()
    assert profile_dir("default") == new


def test_config_always_has_a_default_launch_profile():
    cfg = SentientConfig.model_validate({"browser": {"profiles": {"work": {"notes": "Work account"}}}})
    assert list(cfg.browser.profiles) == ["work", "default"]
    cfg = SentientConfig.model_validate(
        {"browser": {"profiles": {"default": {"kind": "attach", "endpoint": "9222", "notes": "mine"}}}}
    )
    assert cfg.browser.profiles["default"].kind == "launch" and cfg.browser.profiles["default"].notes == "mine"


# ----------------------------------------------------------------------------- routes
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    monkeypatch.setattr(browser_service, "installed_engines", lambda pref="auto": ["msedge"])
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "p.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_profile_routes(client, isolated_home):
    core = client.app.state.sentient
    listed = client.get("/api/browser/profiles").json()
    assert listed == {"active": None, "profiles": [
        {"name": "default", "kind": "launch", "engine": "", "endpoint": "", "notes": "", "running": False}]}

    r = client.post("/api/browser/profiles", json={"name": "X Growth", "notes": "Posting account"})
    assert r.status_code == 200 and [p["name"] for p in r.json()["profiles"]] == ["default", "x-growth"]
    assert core.config.browser.profiles["x-growth"].notes == "Posting account"
    assert "x-growth" in paths.config_file().read_text(encoding="utf-8")  # saved
    assert client.post("/api/browser/profiles", json={"name": "x-growth"}).status_code == 409

    far = client.post("/api/browser/profiles", json={"name": "lan", "kind": "attach", "endpoint": "192.168.1.9:9222"})
    assert far.status_code == 409 and far.json()["detail"] == safety.NOT_LOCAL
    near = client.post("/api/browser/profiles", json={"name": "brave", "kind": "attach", "endpoint": "9333"})
    brave = next(p for p in near.json()["profiles"] if p["name"] == "brave")
    assert brave["kind"] == "attach" and brave["endpoint"] == "http://127.0.0.1:9333"
    bad = client.patch("/api/browser/profiles/brave", json={"endpoint": "http://example.com:9333"})
    assert bad.status_code == 409 and bad.json()["detail"] == safety.NOT_LOCAL

    folder = profile_dir("x-growth")
    folder.mkdir(parents=True)
    (folder / "marker").write_text("signed in", encoding="utf-8")
    renamed = client.patch("/api/browser/profiles/x-growth", json={"name": "x-posts", "notes": "X account"})
    assert renamed.status_code == 200
    assert (profile_dir("x-posts") / "marker").exists() and not folder.exists()
    assert core.config.browser.profiles["x-posts"].notes == "X account" and "x-growth" not in core.config.browser.profiles

    assert client.patch("/api/browser/profiles/default", json={"name": "other"}).status_code == 409
    assert client.delete("/api/browser/profiles/default").status_code == 409
    assert client.delete("/api/browser/profiles/missing").status_code == 409

    gone = client.delete("/api/browser/profiles/x-posts").json()
    assert "x-posts" not in [p["name"] for p in gone["profiles"]] and not profile_dir("x-posts").exists()
    r = client.post("/api/browser/open", json={"profile": "missing"})
    assert r.status_code == 409 and "no browser profile named 'missing'" in r.json()["detail"]


def test_attach_profile_keeps_tools_visible_without_installed_browser(client, monkeypatch):
    core = client.app.state.sentient
    monkeypatch.setattr(browser_service, "installed_engines", lambda pref="auto": [])
    core.browser.sync_visibility()
    assert core.registry.is_hidden("browser")
    client.post("/api/browser/profiles", json={"name": "mine", "kind": "attach", "endpoint": "9333"})
    assert not core.registry.is_hidden("browser")


# ----------------------------------------------------------------------------- tasks and skills
def _recorder(seen: list) -> ToolPlugin:
    @tool("probe_check", risk=Risk.read)
    async def probe_check(ctx: ToolContext) -> dict:
        """Check something."""
        seen.append(ctx.extra.get("browser_profile"))
        return {"ok": True}

    class Probe(ToolPlugin):
        id = "probe"
        display_name = "Probe"
        tools = [probe_check]

    return Probe()


async def test_task_default_profile_reaches_the_tools(config, isolated_home):
    config.browser.profiles["work"] = BrowserProfileConfig(notes="Work account")
    seen: list = []
    llm = FakeProvider(replies=[[tool_call("probe_check")], "Checked."], json_replies=[dict(RESULT)])
    app = SentientApp(config, llm=llm, db_path=isolated_home / "t.db", enable_background=False)
    await app.start()
    try:
        app.registry.register(_recorder(seen))
        created = await app.tasks.create_task("Check the work dashboard", browser_profile="work")
        assert created["browser_profile"] == "work"
        with pytest.raises(ValueError, match="no browser profile named 'nope'"):
            await app.tasks.update(created["task_id"], {"browser_profile": "nope"})

        now = app.tasks.now_iso()
        task_id = await app.tasks.repo.insert_task({
            "name": "Check", "description": "Check the dashboard", "status": "approval_pending",
            "schedule": {"type": "once", "run_at": None}, "browser_profile": "work",
            "plan": [{"tool": "probe", "description": "Check it"}], "created_at": now, "updated_at": now,
        })
        await app.tasks.approve(task_id)
        await app.tasks.drain()
        assert seen == ["work"]

        updated = await app.tasks.update(task_id, {"browser_profile": "default"})
        assert updated["browser_profile"] is None
        await app.tasks.update(task_id, {"browser_profile": "work"})
        async with app.bus.subscribe() as q:
            await app.browser.update_profile("work", new_name="office")
            events = []
            while not q.empty():
                events.append(q.get_nowait())
        assert (await app.tasks.get(task_id))["browser_profile"] == "office"
        assert any(e["type"] == "task.updated" and e["data"]["task_id"] == task_id
                   and e["data"]["browser_profile"] == "office" for e in events)
    finally:
        await app.stop()


async def test_skill_can_name_a_profile(isolated_home):
    lib = SkillLibrary([isolated_home / "skills"])
    lib.root.mkdir(parents=True, exist_ok=True)
    folder = lib.root / "post-on-x"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: post-on-x\ndescription: Post the daily thread on X.\nbrowser_profile: x-growth\n---\n\n# Steps\n",
        encoding="utf-8",
    )
    lib.reload()
    skill = lib.get("post-on-x")
    assert skill.browser_profile == "x-growth" and skill.to_dict()["browser_profile"] == "x-growth"

    ctx = ToolContext(store=None, config=SentientConfig(), llm=None, extra={"skills": lib})
    out = await skill_view.call(ctx, {"name": "post-on-x"})
    assert out["browser_profile"] == "x-growth" and ctx.extra["browser_profile"] == "x-growth"

    edited = lib.edit("post-on-x", body="# New steps")
    assert edited.browser_profile == "x-growth"  # editing keeps it
    lib.write("post-on-x", "Post the daily thread on X.", "# Patched", author="assistant")
    lib.reload()
    assert lib.get("post-on-x").browser_profile == "x-growth"  # so does the assistant patching it


async def test_no_switch_while_the_user_signs_in():
    cfg = SentientConfig()
    cfg.browser.profiles["work"] = BrowserProfileConfig()
    app = make_app(cfg)
    svc = BrowserService(app)
    svc._context, svc._profile, svc._for_user = object(), "default", True  # a sign-in window is open
    ctx = make_ctx(app)
    ctx.extra["browser_profile"] = "work"
    with pytest.raises(BrowserError, match="close that window"):
        await svc._ensure(ctx)
    assert svc._context is not None and svc._profile == "default"
    assert await svc._ensure(make_ctx(app)) is svc._context  # the open profile keeps working


async def test_rename_keeps_folder_and_settings_in_step(monkeypatch):
    cfg = SentientConfig()
    cfg.browser.profiles["work"] = BrowserProfileConfig()
    app = make_app(cfg)
    svc = BrowserService(app)
    (profile_dir("work") / "Default").mkdir(parents=True)
    profile_dir("taken").mkdir(parents=True)
    with pytest.raises(BrowserError, match="already there"):
        await svc.update_profile("work", new_name="taken")

    def broken_save():
        raise OSError("disk full")

    app.save_config = broken_save
    with pytest.raises(BrowserError, match="nothing was changed"):
        await svc.update_profile("work", new_name="office")
    assert (profile_dir("work") / "Default").is_dir() and not profile_dir("office").exists()
    assert "work" in app.config.browser.profiles and "office" not in app.config.browser.profiles


async def test_notes_change_does_not_move_the_old_default_folder():
    app = make_app()
    svc = BrowserService(app)
    old = paths.home() / "browser" / "profile"
    old.mkdir(parents=True)
    await svc.update_profile("default", notes="Everyday browsing")
    assert old.is_dir() and not (paths.home() / "browser" / "profiles" / "default").exists()
    assert app.config.browser.profiles["default"].notes == "Everyday browsing"


def test_a_run_that_picks_no_profile_gets_default():
    cfg = SentientConfig()
    cfg.browser.profiles["x-posting"] = BrowserProfileConfig()
    app = make_app(cfg)
    svc = BrowserService(app)
    svc._context, svc._profile = object(), "x-posting"  # a task left the X profile open
    assert svc._wanted(make_ctx(app)) == "default"  # a later chat never keeps using it
    task = make_ctx(app)
    task.extra["browser_profile"] = "x-posting"
    assert svc._wanted(task) == "x-posting"
    assert svc._wanted(None) == "x-posting"  # the user's own Open window stays on what is open


# ----------------------------------------------------------------------------- site storage reaches disk
_REAL_SLEEP = asyncio.sleep


class _FakePage:
    def __init__(self, url: str, log: list):
        self.url, self._log, self._closed = url, log, False

    def is_closed(self) -> bool:
        return self._closed

    async def close(self, run_before_unload: bool = False) -> None:
        self._closed = True
        self._log.append(f"close page {self.url}" + (" with unload handlers" if run_before_unload else ""))

    async def wait_for_event(self, event: str, timeout: float = 0) -> None:
        while not self._closed:
            await _REAL_SLEEP(0)


class _FakeContext:
    def __init__(self, urls: list[str], log: list):
        self.pages, self._log = [_FakePage(u, log) for u in urls], log

    async def close(self) -> None:
        self._log.append("close context")


async def test_closing_a_launched_profile_lets_site_storage_reach_disk(monkeypatch):
    monkeypatch.setattr(browser_service, "STORAGE_FLUSH_S", 0.01)
    real_sleep = browser_service.asyncio.sleep
    log: list = []

    async def sleep(s):
        log.append("wait")
        await real_sleep(0)

    monkeypatch.setattr(browser_service.asyncio, "sleep", sleep)
    svc = BrowserService(make_app())
    svc._context = _FakeContext(["https://x.example/home", "about:blank"], log)
    await svc._close_context()
    assert log == ["close page https://x.example/home with unload handlers",
                   "close page about:blank with unload handlers", "wait", "close context"]
    assert svc._context is None

    log.clear()  # a website was shown earlier (say the user signed in, then closed that tab): still wait
    svc._context, svc._site_seen = _FakeContext(["about:blank"], log), True
    await svc._close_context()
    assert log == ["close page about:blank with unload handlers", "wait", "close context"]

    log.clear()  # no website was ever shown: close at once
    svc._context, svc._site_seen = _FakeContext(["about:blank"], log), False
    await svc._close_context()
    assert log == ["close context"]


async def test_launched_browsers_commit_site_storage_quickly(monkeypatch):
    seen: dict = {}

    class Chromium:
        async def launch_persistent_context(self, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

    monkeypatch.setattr(browser_service, "installed_engines", lambda pref="auto": ["msedge"])
    svc = BrowserService(make_app())
    svc._pw = type("PW", (), {"chromium": Chromium()})()
    with pytest.raises(BrowserError):
        await svc._launch_persistent(BrowserProfileConfig(), headless=True)
    assert browser_service.STORAGE_FLAG in seen["args"]
    assert seen["accept_downloads"] is True


async def test_finished_download_is_reported_on_the_next_result():
    svc = BrowserService(make_app())
    task = asyncio.create_task(asyncio.sleep(0, result="downloads/late-report.txt"))
    svc._download_tasks.add(task)
    await task

    result = await svc._include_downloads({"ok": True}, {task})

    assert result["downloads"] == ["downloads/late-report.txt"]
    assert task not in svc._download_tasks


async def test_unfinished_download_stays_tracked_until_a_later_result(monkeypatch):
    svc = BrowserService(make_app())
    completion = asyncio.get_running_loop().create_future()

    class SlowDownload:
        suggested_filename = "slow-report.pdf"

    async def save_download(download):
        return await completion

    monkeypatch.setattr(svc, "_save_download", save_download)
    svc._on_download(SlowDownload())
    task = next(iter(svc._download_tasks))

    result = await asyncio.wait_for(svc._include_downloads({"ok": True}, set()), timeout=0.5)

    assert result == {"ok": True, "downloads_in_progress": ["slow-report.pdf"]}
    assert task in svc._download_tasks
    assert not task.done()

    completion.set_result("downloads/slow-report.pdf")
    await task
    result = await svc._include_downloads({"ok": True}, set())

    assert result["downloads"] == ["downloads/slow-report.pdf"]
    assert task not in svc._download_tasks
    assert task not in svc._download_names
    if svc._download_cleanup_task:
        svc._download_cleanup_task.cancel()
        await asyncio.gather(svc._download_cleanup_task, return_exceptions=True)


async def test_failed_download_does_not_discard_action_or_successful_download():
    svc = BrowserService(make_app())

    async def fail_download():
        raise BrowserError("Couldn't save downloaded file 'broken.pdf'")

    successful = asyncio.create_task(asyncio.sleep(0, result="downloads/report.pdf"))
    failed = asyncio.create_task(fail_download())
    svc._download_tasks.update((successful, failed))
    await asyncio.gather(successful, failed, return_exceptions=True)

    result = await svc._include_downloads({"ok": True, "clicked": "Download"}, set())

    assert result == {
        "ok": True,
        "clicked": "Download",
        "downloads": ["downloads/report.pdf"],
        "download_errors": ["Couldn't save downloaded file 'broken.pdf'"],
    }
    assert successful not in svc._download_tasks
    assert failed not in svc._download_tasks


async def test_stalled_download_is_timed_out_and_reported(tmp_path, monkeypatch):
    svc = BrowserService(make_app())

    class StalledDownload:
        suggested_filename = "stalled.pdf"

        async def save_as(self, target):
            await asyncio.Future()

    monkeypatch.setattr(browser_service.paths, "files_dir", lambda: tmp_path)
    monkeypatch.setattr(browser_service, "DOWNLOAD_MAX_DURATION_S", 0.01)
    task = asyncio.create_task(svc._save_download(StalledDownload()))
    svc._download_tasks.add(task)

    with pytest.raises(BrowserError, match="did not finish within"):
        await task
    result = await svc._include_downloads({"ok": True}, set())

    assert result == {
        "ok": True,
        "download_errors": ["Download 'stalled.pdf' did not finish within 0.01 seconds."],
    }
    assert task not in svc._download_tasks


async def test_unreported_completed_download_expires_after_reporting_window(monkeypatch):
    svc = BrowserService(make_app())
    monkeypatch.setattr(browser_service, "DOWNLOAD_TASK_RETENTION_S", 0.01)
    monkeypatch.setattr(browser_service, "DOWNLOAD_TASK_CLEANUP_INTERVAL_S", 0.01)

    async def save_download(download):
        return f"downloads/{download}.pdf"

    monkeypatch.setattr(svc, "_save_download", save_download)
    await svc._lock.acquire()
    try:
        downloads = list(range(100))
        for download in downloads:
            svc._on_download(download)
        tasks = set(svc._download_tasks)
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        assert svc._download_cleanup_task is not None
        assert svc._download_cleanup_task.get_name() == "browser:download-cleanup"
        assert len(svc._download_tasks) == len(downloads)
        await asyncio.sleep(0.02)
        assert len(svc._download_tasks) == len(downloads)
    finally:
        svc._lock.release()

    for _ in range(20):
        if not svc._download_tasks:
            break
        await asyncio.sleep(0.01)
    assert not svc._download_tasks
    assert not svc._download_completed_at
    assert not svc._download_names
