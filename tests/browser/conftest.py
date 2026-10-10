"""Fixtures for browser tests: a tiny local web site and a BrowserService on a minimal app."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import threading
from types import SimpleNamespace

import pytest

from sentient.browser.service import BrowserError, BrowserService, installed_engines
from sentient.config.schema import SentientConfig
from sentient.events import EventBus
from sentient.tools.base import ToolContext
from sentient.tools.registry import ToolRegistry

INDEX = """<!doctype html><html><head><title>Test Shop</title></head><body>
<nav><a href="/nav-only.html">Navigation only link</a> Menu text that is navigation</nav>
<main>
  <h1>Welcome to the test shop</h1>
  <p>The shop sells blue notebooks and green pens. Returns are accepted for thirty days.</p>
  <form action="/search.html" method="get">
    <input name="q" placeholder="Search" aria-label="Search">
    <button type="submit">Search</button>
  </form>
  <a href="/help.html" target="_blank">Open help in new tab</a>
  <label for="country">Country</label>
  <select id="country"><option>India</option><option>Japan</option></select>
  <button id="ghost" style="display:none">Ghost button</button>
  <button id="del" onclick="if (confirm('Really delete?')) document.getElementById('out').textContent = 'deleted-ok'">Delete account</button>
  <div id="out">nothing yet</div>
  <form id="login"><label>Email <input type="email" name="email"></label>
    <label>Password <input type="password" name="password"></label><button>Sign in</button></form>
  <form id="checkout" onsubmit="return false">
    <label>Card number <input name="cardnumber" autocomplete="cc-number"></label>
    <label>Coupon <input name="coupon"></label>
    <button type="submit">Place order</button>
  </form>
  <div style="height:3000px">spacer</div>
  <p>Bottom of the page.</p>
</main></body></html>"""

PAGES = {
    "index.html": INDEX,
    "search.html": "<!doctype html><title>Results</title><main><h1>Results page</h1><p>Search results here.</p></main>",
    "help.html": "<!doctype html><title>Help</title><main><h1>Help center</h1></main>",
    "download.html": '<!doctype html><title>Downloads</title><a href="/report.txt" download>Download report</a>',
    "report.txt": "Sentient browser download test.\n",
}


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        return

    def do_GET(self):
        if self.path == "/direct-download":
            body = b"Direct browser-open download test.\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", 'attachment; filename="direct-report.txt"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    for name, html in PAGES.items():
        (root / name).write_text(html, encoding="utf-8")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def make_app(cfg: SentientConfig | None = None) -> SimpleNamespace:
    cfg = cfg or SentientConfig()
    cfg.tools.approvals.mode = "off"
    return SimpleNamespace(config=cfg, bus=EventBus(), registry=ToolRegistry(), enable_background=False)


def make_ctx(app, frames: list | None = None) -> ToolContext:
    ctx = ToolContext(store=None, config=app.config, llm=None, extra={"app": app})
    if frames is not None:
        ctx.progress = frames.append  # type: ignore[attr-defined]
    return ctx


@pytest.fixture
async def browser(isolated_home):
    """A started BrowserService with a launched headless browser; skipped when none can launch."""
    if not installed_engines("auto"):
        pytest.skip("no Edge or Chrome installed")
    app = make_app()
    svc = BrowserService(app)
    app.browser = svc
    await svc.start()
    try:
        await asyncio.wait_for(svc._ensure(), timeout=90)
    except (BrowserError, TimeoutError) as exc:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(svc.stop(), timeout=30)
        pytest.skip(f"browser could not launch: {exc!r}")
    try:
        yield svc
    finally:
        # always close what this test launched so no msedge processes are left behind
        with contextlib.suppress(Exception):
            await asyncio.wait_for(svc.stop(), timeout=30)
