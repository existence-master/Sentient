"""Browser control for sites without an integration (docs/API.md section 12).

One Playwright persistent context on ``~/.sentient/browser/profile`` driven through an
installed Edge or Chrome. It starts lazily on the first tool call, runs hidden (headless) by
default, closes itself after ``browser.idle_minutes`` without use and can be reopened as a
visible window (``open_for_user``) so the user signs in themselves; the cookies stay in the
profile for the assistant to use afterwards.

Every acting tool call captures a small JPEG for the live view: ``ctx.progress`` gets it
immediately and ``browser.frame`` is published on the bus at most once per second.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import glob
import inspect
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from sentient import paths
from sentient.browser import safety
from sentient.browser.snapshot import (
    ELEMENT_INFO_JS,
    EXTRACT_JS,
    FOCUSED_INFO_JS,
    REF_ATTR,
    SNAPSHOT_JS,
    focus_on_question,
    format_element,
    format_snapshot,
)
from sentient.services import Service
from sentient.tools.base import Risk

log = logging.getLogger(__name__)

PLUGIN_ID = "browser"
FRAME_INTERVAL_S = 1.0
FRAME_WIDTH = 1024
FRAME_QUALITY = 55
HEADLESS_VIEWPORT = {"width": 1280, "height": 800}
LAUNCH_TIMEOUT_S = 60.0
CLOSE_TIMEOUT_S = 20.0
# models often pass the whole snapshot line ("[e4] button \"Place order\"") instead of just "e4"
_REF_RE = re.compile(r"\b(e\d+)\b", re.IGNORECASE)

# Set by BrowserService.start() so tools called without an app in ctx still find the service.
_CURRENT: BrowserService | None = None


class BrowserError(Exception):
    """A failure whose message is safe and useful to show the user and the model."""


# ----------------------------------------------------------------------------- engine discovery
def _engine_paths(channel: str) -> list[Path]:
    env = os.environ
    out: list[Path] = []
    if sys.platform == "win32":
        roots = [env.get("PROGRAMFILES(X86)"), env.get("PROGRAMFILES"), env.get("LOCALAPPDATA")]
        rel = {"msedge": "Microsoft/Edge/Application/msedge.exe", "chrome": "Google/Chrome/Application/chrome.exe"}
        out = [Path(r) / rel[channel] for r in roots if r and channel in rel]
    elif sys.platform == "darwin":
        app = {"msedge": "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
               "chrome": "Google Chrome.app/Contents/MacOS/Google Chrome"}.get(channel)
        if app:
            out = [Path("/Applications") / app, Path.home() / "Applications" / app]
    else:
        names = {"msedge": ["microsoft-edge", "microsoft-edge-stable"],
                 "chrome": ["google-chrome", "google-chrome-stable"]}.get(channel, [])
        out = [Path(p) for p in (shutil.which(n) for n in names) if p]
        out += [Path("/opt/microsoft/msedge/msedge")] if channel == "msedge" else []
        out += [Path("/opt/google/chrome/chrome")] if channel == "chrome" else []
    return out


def _bundled_chromium_installed() -> bool:
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not base:
        if sys.platform == "win32":
            base = str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ms-playwright")
        elif sys.platform == "darwin":
            base = str(Path.home() / "Library" / "Caches" / "ms-playwright")
        else:
            base = str(Path.home() / ".cache" / "ms-playwright")
    return bool(glob.glob(str(Path(base) / "chromium-*")))


def installed_engines(preference: str = "auto") -> list[str]:
    """Channels to try, in order. ``auto`` prefers Edge, then Chrome."""
    order = ["msedge", "chrome"] if preference == "auto" else [preference]
    found: list[str] = []
    for ch in order:
        if ch == "chromium":
            if _bundled_chromium_installed():
                found.append(ch)
        elif any(p.exists() for p in _engine_paths(ch)):
            found.append(ch)
    return found


ENGINE_NAMES = {"msedge": "Microsoft Edge", "chrome": "Google Chrome", "chromium": "Chromium"}


def _friendly_playwright_error(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    if "has been closed" in low or "target closed" in low or "browser has disconnected" in low:
        return "The browser window was closed. Call the tool again and it will reopen."
    if "timeout" in low:
        first = msg.splitlines()[0] if msg else ""
        return f"The page took too long to respond ({first[:160]}). Try again, scroll, or take a new snapshot."
    if "net::err_name_not_resolved" in low:
        return "That website couldn't be found. Check the address."
    if "net::err_internet_disconnected" in low:
        return "This computer seems to be offline."
    if "net::err_" in low:
        m = re.search(r"net::(ERR_[A-Z_]+)", msg)
        return f"The page couldn't be loaded ({m.group(1) if m else 'network error'})."
    return msg.splitlines()[0][:300] if msg else type(exc).__name__


class BrowserService(Service):
    name = "browser"

    def __init__(self, app):
        super().__init__(app)
        self._pw: Any = None
        self._context: Any = None
        self._engine: str | None = None
        self._headless = True
        self._active: Any = None
        self._snap: dict | None = None  # {"page", "url", "refs": {ref: element descriptor}}
        self._focused: dict | None = None
        self._risk_evals: dict[tuple[str, str], tuple[float, Risk]] = {}
        self._dialogs: list[str] = []
        self._accept_confirm = False
        self._lock = asyncio.Lock()       # one browser action at a time
        self._life_lock = asyncio.Lock()  # launch / close
        self._closing = False
        self._last_used = time.monotonic()
        self._last_error: str | None = None
        self._idle_task: asyncio.Task | None = None
        self._frame_task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._last_frame_at = 0.0
        self._last_tabs_sig: Any = None
        self.idle_check_s = 30.0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        global _CURRENT
        _CURRENT = self
        from sentient.browser.tools import BrowserPlugin

        reg = self.app.registry
        if reg.plugin(PLUGIN_ID) is None:
            reg.register(BrowserPlugin())
        self.sync_visibility()
        self._loops.append(asyncio.create_task(self._watch_config(), name="browser:config"))

    async def stop(self) -> None:
        await super().stop()
        for t in (self._idle_task, self._frame_task):
            if t is not None and not t.done():
                t.cancel()
        await self._shutdown(publish=False)
        for t in list(self._bg):
            t.cancel()
        if self._pw is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._pw.stop(), timeout=15)
            self._pw = None
        global _CURRENT
        if _CURRENT is self:
            _CURRENT = None

    def availability(self) -> tuple[bool, str | None]:
        cfg = self.app.config.browser
        if not cfg.enabled:
            return False, "The browser is turned off in Settings."
        if not installed_engines(cfg.engine):
            if cfg.engine == "auto":
                return False, ("No supported browser was found. Install Microsoft Edge or Google Chrome so "
                               "Sentient can use websites for you.")
            return False, f"{ENGINE_NAMES.get(cfg.engine, cfg.engine)} isn't installed on this computer."
        return True, None

    def sync_visibility(self) -> None:
        ok, _ = self.availability()
        reg = self.app.registry
        if reg.plugin(PLUGIN_ID) is not None:
            reg.set_hidden(PLUGIN_ID, not ok)

    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                ev = await q.get()
                if ev.get("type") != "config.updated":
                    continue
                self.sync_visibility()
                if not self.app.config.browser.enabled and self._context is not None:
                    await self.close()

    @property
    def running(self) -> bool:
        return self._context is not None

    # ------------------------------------------------------------------ status and events
    async def _tabs(self) -> list[dict]:
        ctx = self._context
        if ctx is None:
            return []
        out = []
        for i, page in enumerate(list(ctx.pages)):
            title = ""
            with contextlib.suppress(Exception):
                title = await asyncio.wait_for(page.title(), timeout=2)
            out.append({"index": i, "url": page.url, "title": title, "active": page is self._active})
        return out

    async def status(self) -> dict:
        ok, why = self.availability()
        cfg = self.app.config.browser
        engines = installed_engines(cfg.engine) if ok else []
        running = self._context is not None
        return {
            "available": ok,
            "running": running,
            "engine": self._engine if running else (engines[0] if engines else None),
            "headless": self._headless if running else cfg.headless,
            "tabs": await self._tabs(),
            "error": why or self._last_error,
        }

    async def _publish_status(self) -> None:
        st = await self.status()
        self._last_tabs_sig = [(t["url"], t["title"], t["active"]) for t in st["tabs"]]
        self.app.bus.publish("browser.updated", st)

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ------------------------------------------------------------------ launch / close
    async def _launch(self, headless: bool) -> None:
        ok, why = self.availability()
        if not ok:
            raise BrowserError(why)
        cfg = self.app.config.browser
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - dependency is part of the app
            raise BrowserError("The browser component (Playwright) is missing from this installation.") from exc
        if self._pw is None:
            try:
                self._pw = await asyncio.wait_for(async_playwright().start(), timeout=LAUNCH_TIMEOUT_S)
            except NotImplementedError as exc:
                raise BrowserError("The browser can't start in this mode of the app (unsupported event loop).") from exc
        profile = paths.home() / "browser" / "profile"
        profile.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        context = None
        for channel in installed_engines(cfg.engine):
            kwargs: dict[str, Any] = {
                "user_data_dir": str(profile),
                "headless": headless,
                "timeout": 45_000,
                "args": ["--no-first-run", "--no-default-browser-check", "--hide-crash-restore-bubble"],
            }
            if channel != "chromium":
                kwargs["channel"] = channel
            if headless:
                kwargs["viewport"] = HEADLESS_VIEWPORT
            else:
                kwargs["no_viewport"] = True
            try:
                context = await asyncio.wait_for(
                    self._pw.chromium.launch_persistent_context(**kwargs), timeout=LAUNCH_TIMEOUT_S
                )
                self._engine = channel
                break
            except Exception as exc:
                log.warning("browser: launching %s failed: %s", channel, exc)
                errors.append(f"{ENGINE_NAMES.get(channel, channel)}: {str(exc).splitlines()[0][:200]}")
        if context is None:
            self._last_error = (
                "The browser couldn't start. If a Sentient browser window is still open, close it and try again. "
                + " ".join(errors)
            ).strip()
            raise BrowserError(self._last_error)
        self._last_error = None
        self._context = context
        self._headless = headless
        self._snap = None
        context.set_default_timeout(15_000)
        context.set_default_navigation_timeout(30_000)
        context.on("close", lambda: self._on_context_closed(context))
        context.on("page", self._on_page)
        for page in context.pages:
            self._wire_page(page)
        self._active = context.pages[0] if context.pages else await context.new_page()
        self._last_used = time.monotonic()
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watch(), name="browser:idle")
        await self._publish_status()

    async def _ensure(self) -> Any:
        """The context, launching it (with the configured headless mode) when needed."""
        async with self._life_lock:
            if self._context is None:
                await self._launch(headless=self.app.config.browser.headless)
            return self._context

    async def _page(self) -> Any:
        ctx = await self._ensure()
        if self._active is None or self._active.is_closed():
            pages = [p for p in ctx.pages if not p.is_closed()]
            self._active = pages[-1] if pages else await ctx.new_page()
        return self._active

    async def _shutdown(self, publish: bool = True) -> None:
        async with self._life_lock:
            ctx = self._context
            if ctx is None:
                return
            self._closing = True
            try:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ctx.close(), timeout=CLOSE_TIMEOUT_S)
            finally:
                self._closing = False
                self._context = None
                self._active = None
                self._snap = None
                self._focused = None
        if self._frame_task and not self._frame_task.done():
            self._frame_task.cancel()
        idle = self._idle_task
        if idle is not None and not idle.done() and idle is not asyncio.current_task():
            idle.cancel()
        self._idle_task = None
        if publish:
            await self._publish_status()

    async def close(self) -> dict:
        await self._shutdown()
        return await self.status()

    async def open_for_user(self, url: str | None = None) -> dict:
        """Relaunch the same profile in a visible window so the user can sign in or finish a step."""
        await self._restart(headless=False, url=url)
        return await self.status()

    async def _restart(self, headless: bool, url: str | None = None) -> None:
        ok, why = self.availability()
        if not ok:
            raise BrowserError(why)
        target = safety.normalize_url(url or "")
        if target:
            problem = self._url_problem(target)
            if problem:
                raise BrowserError(problem)
        async with self._lock:
            if not target and self._active is not None and not self._active.is_closed():
                current = self._active.url
                if current.startswith(("http://", "https://")):
                    target = current
            if self._context is not None and self._headless == headless:
                page = await self._context.new_page() if target else await self._page()
            else:
                await self._shutdown(publish=False)
                async with self._life_lock:
                    await self._launch(headless=headless)
                page = await self._page()
            self._active = page
            if target:
                with contextlib.suppress(Exception):
                    await page.goto(target, wait_until="domcontentloaded")
            with contextlib.suppress(Exception):
                await page.bring_to_front()
            self._last_used = time.monotonic()
        await self._publish_status()

    def _on_context_closed(self, context: Any) -> None:
        if self._context is not context:
            return
        self._context = None
        self._active = None
        self._snap = None
        self._focused = None
        if not self._closing:  # the user closed the visible window
            self._spawn(self._publish_status())

    def _wire_page(self, page: Any) -> None:
        page.on("close", lambda: self._on_page_closed(page))
        page.on("dialog", self._on_dialog)

    def _on_page(self, page: Any) -> None:
        self._wire_page(page)
        self._active = page  # a link opened a new tab: keep working in it
        self._spawn(self._publish_status())

    def _on_page_closed(self, page: Any) -> None:
        if self._snap and self._snap.get("page") is page:
            self._snap = None
        ctx = self._context
        if self._active is page:
            remaining = [p for p in (ctx.pages if ctx else []) if p is not page and not p.is_closed()]
            self._active = remaining[-1] if remaining else None
        if ctx is not None and not self._closing:
            self._spawn(self._publish_status())

    async def _on_dialog(self, dialog: Any) -> None:
        self._dialogs.append(f"{dialog.type}: {dialog.message[:200]}")
        with contextlib.suppress(Exception):
            if dialog.type in {"alert", "beforeunload"} or (dialog.type == "confirm" and self._accept_confirm):
                await dialog.accept()
            else:
                await dialog.dismiss()

    async def _idle_watch(self) -> None:
        while self._context is not None:
            await asyncio.sleep(self.idle_check_s)
            minutes = self.app.config.browser.idle_minutes
            if (
                minutes
                and self._context is not None
                and self._headless
                and not self._lock.locked()
                and time.monotonic() - self._last_used > minutes * 60
            ):
                log.info("browser: closing after %.1f idle minutes", minutes)
                await self._shutdown()
                return

    # ------------------------------------------------------------------ safety helpers
    def _url_problem(self, url: str) -> str | None:
        cfg = self.app.config.browser
        return safety.url_problem(url, cfg.allow_domains, cfg.block_domains)

    async def _enforce_domains(self, page: Any) -> None:
        url = page.url or ""
        if not url.startswith(("http://", "https://")):
            return
        problem = self._url_problem(url)
        if problem:
            with contextlib.suppress(Exception):
                await page.goto("about:blank")
            raise BrowserError(problem + " The browser went back to a blank page.")

    def risk_for(self, kind: str, arguments: dict | None) -> Risk:
        """Effective risk of a click/type/press call from the last snapshot (called before approvals)."""
        args = arguments or {}
        cfg = self.app.config.browser
        if kind == "click":
            ref = _clean_ref(args.get("ref"))
            el = (self._snap or {}).get("refs", {}).get(ref) if ref else None
            risk, key = safety.click_risk(el), ref or ""
        elif kind == "type":
            ref = _clean_ref(args.get("ref"))
            el = (self._snap or {}).get("refs", {}).get(ref) if ref else None
            risk, key = safety.type_risk(el, bool(args.get("submit"))), ref or ""
        else:
            risk, key = safety.press_risk(str(args.get("key", "")), self._focused), str(args.get("key", ""))
        if not cfg.confirm_purchases:
            risk = Risk.write
        now = time.monotonic()
        window = self._eval_window()
        self._risk_evals = {k: v for k, v in self._risk_evals.items() if now - v[0] <= window}
        self._risk_evals[(kind, key)] = (now, risk)
        return risk

    def link_for(self, arguments: dict | None) -> str:
        """Where clicking this element (from the last snapshot) goes: its link as a full address, else ""."""
        ref = _clean_ref((arguments or {}).get("ref"))
        el = (self._snap or {}).get("refs", {}).get(ref) if ref else None
        href = str((el or {}).get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "#")):
            return ""
        return urljoin(str((self._snap or {}).get("url") or ""), href)

    def describe_for(self, kind: str, arguments: dict | None) -> dict:
        """Approval wording for a click/type/press from the last snapshot: ``{risk_label, target}``."""
        args = arguments or {}
        if kind == "press":
            key = str(args.get("key", ""))
            focused = safety.approval_target(self._focused)
            return {"risk_label": safety.risk_label_for(focused, "press"), "target": f"{key} in {focused}" if focused else key}
        ref = _clean_ref(args.get("ref"))
        el = (self._snap or {}).get("refs", {}).get(ref) if ref else None
        target = safety.approval_target(el)
        context = " ".join([target, str((el or {}).get("form_submit_text") or "")]) if kind == "type" else target
        return {"risk_label": safety.risk_label_for(context, kind), "target": target}

    def _eval_window(self) -> float:
        return float(self.app.config.tools.approvals.timeout_s) + 120.0

    def _guard(self, kind: str, key: str, live_risk: Risk) -> str | None:
        """Defensive check at call time: refuse a send-looking action that approvals did not see as send."""
        cfg = self.app.config
        ev = self._risk_evals.pop((kind, key), None)
        if live_risk < Risk.send or not cfg.browser.confirm_purchases or cfg.tools.approvals.mode == "off":
            return None
        if ev is None or time.monotonic() - ev[0] > self._eval_window():
            return (
                "This looks like it would buy, pay, send, post or delete something, and the user could not be "
                "asked to approve it from here. Tell the user exactly what you were about to do and ask them to "
                "do this step themselves (they can click 'Open browser' in Sentient)."
            )
        if ev[1] < Risk.send:
            return (
                "The page changed since your last snapshot and this now looks like it buys, pays, sends, posts or "
                "deletes something. Call browser_snapshot, then call the tool again so the user can approve it."
            )
        return None

    # ------------------------------------------------------------------ element lookup
    async def _locate(self, ref: Any) -> tuple[Any, Any, dict]:
        clean = _clean_ref(ref)
        if not clean:
            raise BrowserError(f"'{ref}' isn't a valid ref. Refs look like e12; take them from browser_snapshot.")
        page = await self._page()
        loc = page.locator(f'[{REF_ATTR}="{clean}"]')
        count = 0
        with contextlib.suppress(Exception):
            count = await loc.count()
        if count == 0:
            if self._snap is None or self._snap.get("page") is not page:
                raise BrowserError("There is no snapshot of this page yet. Call browser_snapshot to get refs, then use them.")
            raise BrowserError(
                f"Element {clean} is no longer on the page (the page changed or navigated). "
                "Call browser_snapshot again to get fresh refs."
            )
        loc = loc.first
        snap_el = dict((self._snap or {}).get("refs", {}).get(clean) or {})
        live: dict = {}
        with contextlib.suppress(Exception):
            live = await loc.evaluate(ELEMENT_INFO_JS)
        # the snapshot's role/name read better; security-relevant attributes come from the live element
        info = {**live, **{k: v for k, v in snap_el.items() if v not in ("", None)}}
        for key in ("disabled", "type", "autocomplete", "form_payment", "is_submit", "form_submit_text", "target"):
            if key in live:
                info[key] = live[key]
        info["ref"] = clean
        info["live"] = live
        return page, loc, info

    # ------------------------------------------------------------------ live view
    async def capture_jpeg(self, quality: int = FRAME_QUALITY, width: int = FRAME_WIDTH) -> bytes | None:
        page = self._active
        if self._context is None or page is None or page.is_closed():
            return None
        try:
            return await self._jpeg(page, quality, width)
        except Exception as exc:
            log.debug("browser: frame capture failed: %s", exc)
            return None

    async def _jpeg(self, page: Any, quality: int, width: int) -> bytes:
        try:
            vw, vh, sx, sy = await page.evaluate(
                "() => [window.innerWidth, window.innerHeight, window.visualViewport ? visualViewport.pageLeft : scrollX,"
                " window.visualViewport ? visualViewport.pageTop : scrollY]"
            )
            scale = min(1.0, width / max(1, vw))
            session = await self._context.new_cdp_session(page)
            try:
                res = await session.send(
                    "Page.captureScreenshot",
                    {"format": "jpeg", "quality": quality,
                     "clip": {"x": sx, "y": sy, "width": vw, "height": vh, "scale": scale}},
                )
            finally:
                with contextlib.suppress(Exception):
                    await session.detach()
            return base64.b64decode(res["data"])
        except Exception:
            return await page.screenshot(type="jpeg", quality=quality, timeout=10_000)

    async def _after_action(self, ctx: Any, page: Any) -> None:
        self._last_used = time.monotonic()
        if self.app.config.browser.live_view and page is not None and not page.is_closed():
            image = await self.capture_jpeg()
            if image is not None:
                data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
                progress = getattr(ctx, "progress", None)
                if callable(progress):
                    with contextlib.suppress(Exception):
                        r = progress({"kind": "frame", "image": data_url})
                        if inspect.isawaitable(r):
                            await r
                now = time.monotonic()
                if now - self._last_frame_at >= FRAME_INTERVAL_S:
                    await self._publish_frame(page, data_url)
                elif self._frame_task is None or self._frame_task.done():
                    delay = FRAME_INTERVAL_S - (now - self._last_frame_at)
                    self._frame_task = asyncio.create_task(self._trailing_frame(delay), name="browser:frame")
        sig = [(t["url"], t["title"], t["active"]) for t in await self._tabs()]
        if sig != self._last_tabs_sig:
            await self._publish_status()

    async def _publish_frame(self, page: Any, data_url: str) -> None:
        title = ""
        with contextlib.suppress(Exception):
            title = await asyncio.wait_for(page.title(), timeout=2)
        self._last_frame_at = time.monotonic()
        self.app.bus.publish("browser.frame", {"url": page.url, "title": title, "image": data_url})

    async def _trailing_frame(self, delay: float) -> None:
        await asyncio.sleep(max(0.0, delay))
        page = self._active
        image = await self.capture_jpeg()
        if image is not None and page is not None:
            await self._publish_frame(page, "data:image/jpeg;base64," + base64.b64encode(image).decode())

    async def _settle(self, page: Any, timeout_ms: int = 2_000) -> None:
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms * 2)
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)

    def _take_dialogs(self, out: dict) -> dict:
        if self._dialogs:
            out["dialogs"] = list(self._dialogs)
            self._dialogs.clear()
        return out

    # ------------------------------------------------------------------ tool actions
    async def open(self, ctx: Any, url: str) -> dict:
        target = safety.normalize_url(url)
        if not target:
            raise BrowserError("Give a web address to open, for example https://example.com.")
        problem = self._url_problem(target)
        if problem:
            raise BrowserError(problem)
        async with self._lock:
            page = await self._page()
            await page.goto(target, wait_until="domcontentloaded")
            await self._settle(page)
            await self._enforce_domains(page)
            result = await self._snapshot_locked(page)
            await self._after_action(ctx, page)
            return self._take_dialogs(result)

    async def snapshot(self, ctx: Any) -> dict:
        async with self._lock:
            page = await self._page()
            self._last_used = time.monotonic()
            return self._take_dialogs(await self._snapshot_locked(page))

    async def _snapshot_locked(self, page: Any) -> dict:
        cfg = self.app.config.browser
        data = await page.evaluate(SNAPSHOT_JS, {"maxElements": 400, "maxText": cfg.max_snapshot_chars * 2})
        elements = data.get("elements") or []
        self._snap = {"page": page, "url": data.get("url"), "refs": {e["ref"]: e for e in elements}}
        focused = next((e for e in elements if e.get("focused")), None)
        if focused:
            self._focused = focused
        text, truncated = format_snapshot(
            data.get("title", ""), data.get("url", ""), elements, data.get("text", ""), cfg.max_snapshot_chars,
            data.get("scroll"),
        )
        out = {"url": data.get("url", page.url), "title": data.get("title", ""), "text": text}
        if truncated:
            out["truncated"] = True
        return out

    async def click(self, ctx: Any, ref: str) -> dict:
        async with self._lock:
            page, loc, info = await self._locate(ref)
            if info.get("disabled"):
                raise BrowserError(f"{format_element(info)} is disabled. Something else on the page may need to be done first.")
            live_risk = safety.click_risk(info) if self.app.config.browser.confirm_purchases else Risk.write
            refusal = self._guard("click", info["ref"], live_risk)
            if refusal:
                return {"error": refusal}
            pages_before = len(self._context.pages)
            self._accept_confirm = live_risk >= Risk.send or self.app.config.tools.approvals.mode == "off"
            try:
                await loc.click(timeout=10_000)
            finally:
                self._accept_confirm = False
            await self._settle(page)
            # a new tab arrives as a separate event; links with target=_blank get longer to show up
            limit = 3.0 if str(info.get("target", "")).lower() == "_blank" else 0.3
            waited = 0.0
            while self._context is not None and len(self._context.pages) <= pages_before and waited < limit:
                await asyncio.sleep(0.05)
                waited += 0.05
            if self._context is None:
                raise BrowserError("The browser window was closed. Call the tool again and it will reopen.")
            active = self._active or page
            await self._settle(active)
            await self._enforce_domains(active)
            out = {
                "ok": True,
                "clicked": format_element(info),
                "url": active.url,
                "message": "Clicked. Call browser_snapshot to see the page now.",
            }
            if len(self._context.pages) > pages_before:
                out["new_tab"] = True
                out["message"] = "Clicked; it opened a new tab, which is now active. Call browser_snapshot to see it."
            await self._after_action(ctx, active)
            return self._take_dialogs(out)

    async def type(self, ctx: Any, ref: str, text: str, submit: bool = False) -> dict:
        async with self._lock:
            page, loc, info = await self._locate(ref)
            kind = safety.sensitive_field(info) or safety.sensitive_field(info.get("live"))
            if kind:
                return safety.needs_user(kind)
            if safety.looks_like_card_number(text):
                return safety.needs_user("card number")
            live_risk = safety.type_risk(info, submit) if self.app.config.browser.confirm_purchases else Risk.write
            refusal = self._guard("type", info["ref"], live_risk)
            if refusal:
                return {"error": refusal}
            try:
                await loc.fill(text, timeout=8_000)
            except Exception:
                try:
                    await loc.click(timeout=5_000)
                    await page.keyboard.type(text)
                except Exception as exc:
                    raise BrowserError(
                        f"Couldn't type into {format_element(info)}: it doesn't look like a text field. "
                        "Take a new snapshot and pick a textbox."
                    ) from exc
            self._focused = info
            if submit:
                await loc.press("Enter")
                await self._settle(page)
            active = self._active or page
            await self._enforce_domains(active)
            described = format_element({k: v for k, v in info.items() if k != "value"})
            out = {"ok": True, "typed_into": described, "submitted": bool(submit), "url": active.url}
            await self._after_action(ctx, active)
            return self._take_dialogs(out)

    async def select(self, ctx: Any, ref: str, option: str) -> dict:
        async with self._lock:
            page, loc, info = await self._locate(ref)
            if str(info.get("tag", "")).lower() != "select":
                raise BrowserError(
                    f"{format_element(info)} isn't a dropdown list. Click it with browser_click, take a snapshot "
                    "and click the option instead."
                )
            chosen = None
            attempts: list[dict] = [{"label": option}, {"value": option}]
            if option.strip().isdigit():
                attempts.append({"index": int(option.strip())})
            for attempt in attempts:
                try:
                    chosen = await loc.select_option(**attempt, timeout=3_000)
                    break
                except Exception:
                    continue
            if chosen is None:
                opts = ", ".join(info.get("options") or [])
                raise BrowserError(f"No option '{option}' in {format_element(info)}. Options: {opts}")
            await self._settle(page, 1_000)
            out = {"ok": True, "selected": option, "in": format_element(info), "url": page.url}
            await self._after_action(ctx, page)
            return self._take_dialogs(out)

    async def press(self, ctx: Any, key: str) -> dict:
        name = normalize_key(key)
        async with self._lock:
            page = await self._page()
            focused = None
            with contextlib.suppress(Exception):
                focused = await page.evaluate(FOCUSED_INFO_JS)
            if focused:
                self._focused = focused
                kind = safety.sensitive_field(focused)
                if kind and len(name) == 1:
                    return safety.needs_user(kind)
            live_risk = safety.press_risk(name, focused) if self.app.config.browser.confirm_purchases else Risk.write
            refusal = self._guard("press", str(key), live_risk)
            if refusal:
                return {"error": refusal}
            try:
                await page.keyboard.press(name)
            except Exception as exc:
                raise BrowserError(f"'{key}' isn't a key the browser knows. Try Enter, Tab, Escape, ArrowDown or PageDown.") from exc
            await self._settle(page, 1_000)
            active = self._active or page
            await self._enforce_domains(active)
            out = {"ok": True, "pressed": name, "url": active.url}
            await self._after_action(ctx, active)
            return self._take_dialogs(out)

    async def scroll(self, ctx: Any, direction: str = "down") -> dict:
        d = (direction or "down").strip().lower()
        scripts = {
            "down": "window.scrollBy(0, window.innerHeight * 0.85)",
            "up": "window.scrollBy(0, -window.innerHeight * 0.85)",
            "top": "window.scrollTo(0, 0)",
            "bottom": "window.scrollTo(0, document.documentElement.scrollHeight)",
            "right": "window.scrollBy(window.innerWidth * 0.85, 0)",
            "left": "window.scrollBy(-window.innerWidth * 0.85, 0)",
        }
        if d not in scripts:
            raise BrowserError("direction must be one of: down, up, top, bottom, left, right.")
        async with self._lock:
            page = await self._page()
            pos = await page.evaluate(
                "() => { " + scripts[d] + "; return [Math.round(scrollY), Math.round(document.documentElement.scrollHeight"
                " - scrollY - innerHeight)]; }"
            )
            await asyncio.sleep(0.15)
            out = {"ok": True, "scrolled": d, "from_top": pos[0], "more_below": max(0, pos[1]),
                   "message": "Call browser_snapshot to see what is visible now."}
            await self._after_action(ctx, page)
            return out

    async def back(self, ctx: Any) -> dict:
        async with self._lock:
            page = await self._page()
            resp = await page.go_back(wait_until="domcontentloaded")
            if resp is None and page.url in {"about:blank", ""}:
                raise BrowserError("There is no earlier page in this tab.")
            await self._settle(page)
            await self._enforce_domains(page)
            title = ""
            with contextlib.suppress(Exception):
                title = await page.title()
            out = {"ok": True, "url": page.url, "title": title}
            await self._after_action(ctx, page)
            return out

    async def tabs(self, ctx: Any) -> dict:
        async with self._lock:
            await self._ensure()
            self._last_used = time.monotonic()
            return {"tabs": await self._tabs()}

    async def switch_tab(self, ctx: Any, index: int) -> dict:
        async with self._lock:
            c = await self._ensure()
            pages = list(c.pages)
            if not 0 <= index < len(pages):
                raise BrowserError(f"There is no tab {index}. Open tabs are numbered 0 to {len(pages) - 1}.")
            self._active = pages[index]
            with contextlib.suppress(Exception):
                await self._active.bring_to_front()
            title = ""
            with contextlib.suppress(Exception):
                title = await self._active.title()
            out = {"ok": True, "index": index, "url": self._active.url, "title": title,
                   "message": "Switched. Call browser_snapshot to see this tab."}
            await self._after_action(ctx, self._active)
            return out

    async def extract(self, ctx: Any, question: str = "") -> dict:
        cfg = self.app.config.browser
        async with self._lock:
            page = await self._page()
            self._last_used = time.monotonic()
            data = await page.evaluate(EXTRACT_JS, {"maxText": cfg.max_extract_chars * 5})
            content, truncated = focus_on_question(data.get("text", ""), question, cfg.max_extract_chars)
            out = {"url": data.get("url", page.url), "title": data.get("title", ""), "content": content or "(no text)"}
            if question:
                out["question"] = question
            if truncated:
                out["truncated"] = True
            return out

    async def screenshot(self, ctx: Any) -> dict:
        async with self._lock:
            page = await self._page()
            self._last_used = time.monotonic()
            folder = paths.files_dir() / "outputs" / "browser"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"screenshot-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')[:-3]}.png"
            await page.screenshot(path=str(folder / name), type="png")
            title = ""
            with contextlib.suppress(Exception):
                title = await page.title()
            return {"file": f"outputs/browser/{name}", "url": page.url, "title": title}

    async def close_tool(self, ctx: Any) -> dict:
        was_running = self._context is not None
        async with self._lock:
            await self._shutdown()
        return {"ok": True, "message": "Browser closed." if was_running else "The browser wasn't open."}


def _clean_ref(ref: Any) -> str | None:
    m = _REF_RE.search(str(ref or "").strip())
    return m.group(1).lower() if m else None


_KEY_ALIASES = {
    "enter": "Enter", "return": "Enter", "tab": "Tab", "esc": "Escape", "escape": "Escape", "space": "Space",
    "backspace": "Backspace", "delete": "Delete", "del": "Delete", "up": "ArrowUp", "down": "ArrowDown",
    "left": "ArrowLeft", "right": "ArrowRight", "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "pagedown": "PageDown", "page down": "PageDown",
    "pageup": "PageUp", "page up": "PageUp", "home": "Home", "end": "End",
}


def normalize_key(key: str) -> str:
    """Friendly key names (enter, esc, page down, ctrl+a) to Playwright names (Enter, Escape, PageDown, Control+a)."""
    raw = str(key or "").strip()
    if not raw:
        return raw
    parts = [p.strip() for p in raw.split("+")] if len(raw) > 1 else [raw]
    mods = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "meta": "Meta", "alt": "Alt", "shift": "Shift"}
    out = []
    for i, p in enumerate(parts):
        low = p.lower()
        if i < len(parts) - 1 and low in mods:
            out.append(mods[low])
        elif low in _KEY_ALIASES:
            out.append(_KEY_ALIASES[low])
        elif len(p) > 1 and re.fullmatch(r"f\d{1,2}", low):
            out.append(p.upper())
        else:
            out.append(p)
    return "+".join(out)


def service_from(ctx: Any) -> BrowserService:
    app = (getattr(ctx, "extra", None) or {}).get("app")
    svc = getattr(app, "browser", None)
    if isinstance(svc, BrowserService):
        return svc
    if _CURRENT is not None:
        return _CURRENT
    raise BrowserError("The browser isn't available right now.")
