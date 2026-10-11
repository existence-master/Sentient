"""Browser control for sites without an integration (docs/API.md section 12).

One Playwright persistent context at a time, on a named profile (``browser.profiles``). A
``launch`` profile has its own folder, ``~/.sentient/browser/profiles/<name>``, driven through an
installed Edge or Chrome. It starts lazily on the first tool call, runs hidden (headless) by
default, closes itself after ``browser.idle_minutes`` without use and can be reopened as a
visible window (``open_for_user``) so the user signs in themselves; the cookies stay in the
profile for the assistant to use afterwards. An ``attach`` profile connects over the DevTools
protocol to a browser the user started on this computer; closing only disconnects from it.
Switching profiles closes the current one first.

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
from urllib.parse import urljoin, urlsplit

import httpx

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
from sentient.config.schema import BrowserProfileConfig
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
# Site storage (localStorage, where many sites keep a sign-in) reaches disk on a timer, not at once. Launched browsers
# use Chromium's short 1 s commit delay (STORAGE_FLAG); before closing one, its tabs are closed and this long is waited
# so a sign-in made just before a profile switch, idle close or quit isn't lost.
STORAGE_FLAG = "--enable-aggressive-domstorage-flushing"
STORAGE_FLUSH_S = 1.5
PAGE_CLOSE_TIMEOUT_S = 5.0
DOWNLOAD_APPEAR_GRACE_S = 0.3
DOWNLOAD_MAX_DURATION_S = 120.0
DOWNLOAD_CANCEL_TIMEOUT_S = 5.0
DOWNLOAD_TASK_RETENTION_S = 300.0
DOWNLOAD_TASK_CLEANUP_INTERVAL_S = 30.0
# models often pass the whole snapshot line ("[e4] button \"Place order\"") instead of just "e4"
_REF_RE = re.compile(r"\b(e\d+)\b", re.IGNORECASE)

DEFAULT_PROFILE = "default"
PROFILE_KEY = "browser_profile"  # ctx.extra: the profile this run uses (task default, skill, or a tool argument)
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

# Set by BrowserService.start() so tools called without an app in ctx still find the service.
_CURRENT: BrowserService | None = None


class BrowserError(Exception):
    """A failure whose message is safe and useful to show the user and the model."""


# ----------------------------------------------------------------------------- profiles
def profile_name(raw: Any) -> str:
    """A profile name from what the user typed: lowercase letters, numbers and dashes ("X growth" -> "x-growth")."""
    name = re.sub(r"[^a-z0-9]+", "-", str(raw or "").strip().lower()).strip("-")[:40].strip("-")
    if not _PROFILE_NAME_RE.match(name):
        raise BrowserError("Give the profile a name made of letters, numbers or dashes.")
    return name


def profile_dir(name: str) -> Path:
    """The browser folder of a launched profile. The first use of ``default`` moves the folder older versions used
    (``~/.sentient/browser/profile``); if it can't be moved (a window still open), the old folder keeps working."""
    if not _PROFILE_NAME_RE.match(name or ""):
        raise BrowserError(f"'{name}' isn't a valid profile name.")
    base = paths.home() / "browser"
    folder = base / "profiles" / name
    if name == DEFAULT_PROFILE:
        old = base / "profile"
        if old.is_dir() and not folder.exists():
            try:
                folder.parent.mkdir(parents=True, exist_ok=True)
                old.rename(folder)
            except OSError as exc:
                log.warning("browser: could not move the old profile folder: %s", exc)
                return old
    return folder


async def devtools_ws_url(endpoint: str) -> str:
    """The browser's DevTools websocket address behind ``endpoint`` (checked to be on this computer too)."""
    if endpoint.startswith(("ws://", "wss://")):
        return endpoint
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            data = (await client.get(endpoint + "/json/version")).json()
    except Exception as exc:
        raise BrowserError(
            f"Nothing answered at {endpoint}. Start the browser with --remote-debugging-port set to that port, "
            "then try again."
        ) from exc
    ws = str((data or {}).get("webSocketDebuggerUrl") or "") if isinstance(data, dict) else ""
    if not ws.startswith(("ws://", "wss://")):
        raise BrowserError(f"The program at {endpoint} doesn't look like a browser's DevTools port.")
    if not safety.is_loopback_host(urlsplit(ws).hostname or ""):
        raise BrowserError(safety.NOT_LOCAL)
    return ws


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
        self._browser: Any = None  # attach profiles: the user's browser we are connected to
        self._attached = False
        self._download_pages: set[Any] = set()
        self._internal_pages: set[Any] = set()
        self._profile = DEFAULT_PROFILE  # the running profile, or the next one to start
        self._for_user = False  # the open window was shown for the user (Open to sign in): don't switch under them
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
        self._site_seen = False  # a tab of the open browser showed a website (its storage may need writing out)
        self._last_tabs_sig: Any = None
        self.idle_check_s = 30.0
        self._download_tasks: set[asyncio.Task[str]] = set()
        self._download_names: dict[asyncio.Task[str], str] = {}
        self._download_completed_at: dict[asyncio.Task[str], float] = {}
        self._download_cleanup_task: asyncio.Task | None = None
        self._download_lock = asyncio.Lock()

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

    def availability(self, profile: str | None = None) -> tuple[bool, str | None]:
        cfg = self.app.config.browser
        if not cfg.enabled:
            return False, "The browser is turned off in Settings."
        prof = cfg.profiles.get(profile or DEFAULT_PROFILE)
        if prof is not None and prof.kind == "attach":
            return True, None
        engine = (prof.engine if prof is not None else "") or cfg.engine
        if not installed_engines(engine):
            if engine == "auto":
                return False, ("No supported browser was found. Install Microsoft Edge or Google Chrome so "
                               "Sentient can use websites for you.")
            return False, f"{ENGINE_NAMES.get(engine, engine)} isn't installed on this computer."
        return True, None

    def sync_visibility(self) -> None:
        cfg = self.app.config.browser
        ok = self.availability()[0] or (cfg.enabled and any(p.kind == "attach" for p in cfg.profiles.values()))
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
                cfg = self.app.config.browser
                if self._context is not None and (not cfg.enabled or self._profile not in cfg.profiles):
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
        for i, page in enumerate(self._usable_pages(ctx)):
            title = ""
            with contextlib.suppress(Exception):
                title = await asyncio.wait_for(page.title(), timeout=2)
            out.append({"index": i, "url": page.url, "title": title, "active": page is self._active})
        return out

    def _usable_pages(self, context: Any) -> list[Any]:
        return [
            page for page in list(context.pages)
            if not page.is_closed() and page not in self._internal_pages
        ]

    async def status(self) -> dict:
        running = self._context is not None
        name = self._profile if running else DEFAULT_PROFILE
        ok, why = self.availability(name)
        cfg = self.app.config.browser
        prof = cfg.profiles.get(name)
        attach = prof is not None and prof.kind == "attach"
        engines = installed_engines((prof.engine if prof else "") or cfg.engine) if ok and not attach else []
        return {
            "available": ok,
            "running": running,
            "engine": self._engine if running else (engines[0] if engines else None),
            "headless": self._headless if running else cfg.headless,
            "tabs": await self._tabs(),
            "error": why or self._last_error,
            "profile": name,
            "attached": self._attached,
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

    # ------------------------------------------------------------------ profiles
    def _profile_cfg(self, name: str) -> Any:
        prof = self.app.config.browser.profiles.get(name)
        if prof is None:
            names = ", ".join(self.app.config.browser.profiles)
            raise BrowserError(f"There is no browser profile named '{name}'. Profiles: {names}.")
        return prof

    def _wanted(self, ctx: Any = None) -> str:
        """The profile a tool call uses: the run's own (task default, skill or tool argument), else ``default``.

        Profiles are signed in as different people, so a run that picked none never keeps using another run's
        profile. Internal calls without a ctx (the user's Open window) stay on the open profile."""
        if ctx is None:
            return self._profile if self._context is not None else DEFAULT_PROFILE
        name = str(((getattr(ctx, "extra", None) or {}).get(PROFILE_KEY)) or "").strip() or DEFAULT_PROFILE
        self._profile_cfg(name)
        return name

    def use_profile(self, ctx: Any, name: str) -> None:
        """Pin ``name`` for the rest of this run (a tool's ``profile`` argument)."""
        name = str(name or "").strip()
        if not name:
            return
        self._profile_cfg(name)
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            extra[PROFILE_KEY] = name

    def profiles(self) -> dict:
        running = self._profile if self._context is not None else None
        return {
            "active": running,
            "profiles": [
                {"name": name, "kind": p.kind, "engine": p.engine, "endpoint": p.endpoint, "notes": p.notes,
                 "running": name == running}
                for name, p in self.app.config.browser.profiles.items()
            ],
        }

    def _save_profiles(self, profiles: dict) -> None:
        self.app.config.browser.profiles = profiles
        save = getattr(self.app, "save_config", None)
        if callable(save):
            save()

    def _profile_fields(self, kind: str, engine: str, endpoint: str, notes: str) -> Any:
        if kind not in {"launch", "attach"}:
            raise BrowserError("A profile is either one Sentient starts (launch) or one it attaches to (attach).")
        fields = {"kind": kind, "engine": engine or "", "endpoint": "", "notes": str(notes or "").strip()[:500]}
        if kind == "attach":
            fields["engine"] = ""
            fields["endpoint"], problem = safety.devtools_endpoint(endpoint)
            if problem:
                raise BrowserError(problem)
        try:
            return BrowserProfileConfig(**fields)
        except ValueError as exc:
            raise BrowserError("That browser choice isn't one Sentient knows.") from exc

    async def create_profile(self, name: str, kind: str = "launch", engine: str = "", endpoint: str = "",
                             notes: str = "") -> dict:
        clean = profile_name(name)
        if clean in self.app.config.browser.profiles:
            raise BrowserError(f"There is already a profile named '{clean}'.")
        prof = self._profile_fields(kind, engine, endpoint, notes)
        self._save_profiles({**self.app.config.browser.profiles, clean: prof})
        self.sync_visibility()
        return self.profiles()

    async def update_profile(self, name: str, *, new_name: str | None = None, engine: str | None = None,
                             endpoint: str | None = None, notes: str | None = None) -> dict:
        """Rename a profile (its folder moves with it) or change its browser, address or notes."""
        prof = self._profile_cfg(name)
        target = profile_name(new_name) if new_name is not None else name
        if target != name:
            if name == DEFAULT_PROFILE:
                raise BrowserError("The default profile can't be renamed.")
            if target in self.app.config.browser.profiles:
                raise BrowserError(f"There is already a profile named '{target}'.")
        updated = self._profile_fields(
            prof.kind,
            prof.engine if engine is None else engine,
            prof.endpoint if endpoint is None else endpoint,
            prof.notes if notes is None else notes,
        )
        restart = target != name or updated.engine != prof.engine or updated.endpoint != prof.endpoint
        async with self._lock:
            if restart and self._context is not None and self._profile == name:
                await self._shutdown()
            moved = False
            # only a launch profile's rename touches folders (never the default one, which can't be renamed)
            old, new = (profile_dir(name), profile_dir(target)) if target != name and prof.kind == "launch" else (None, None)
            if old is not None and new is not None and old.exists():
                if new.exists():
                    raise BrowserError(f"A browser folder named '{target}' is already there. Pick another name.")
                try:
                    old.rename(new)
                    moved = True
                except OSError as exc:
                    raise BrowserError(
                        "Couldn't rename the profile's folder. Close any browser window using it and try again."
                    ) from exc
            previous = self.app.config.browser.profiles
            try:
                self._save_profiles({
                    (target if k == name else k): (updated if k == name else v) for k, v in previous.items()
                })
            except Exception as exc:  # keep the folder and the settings in step
                self.app.config.browser.profiles = previous
                if moved and old is not None and new is not None:
                    try:
                        new.rename(old)
                    except OSError as undo_exc:
                        log.error("browser: could not move profile folder %s back to %s: %s", new, old, undo_exc)
                        raise BrowserError(
                            f"Couldn't save the profile change, and its folder is now named '{target}'. Rename the "
                            f"folder {new} back to '{name}' to keep its sign-ins."
                        ) from exc
                raise BrowserError("Couldn't save the profile change, so nothing was changed.") from exc
            if target != name:
                rename = getattr(getattr(self.app, "tasks", None), "rename_browser_profile", None)
                if callable(rename):
                    await rename(name, target)
        return self.profiles()

    async def delete_profile(self, name: str) -> dict:
        """Remove a profile. A launched profile's folder (its sign-ins) is deleted; an attached browser is untouched."""
        prof = self._profile_cfg(name)
        if name == DEFAULT_PROFILE:
            raise BrowserError("The default profile can't be deleted.")
        async with self._lock:
            if self._context is not None and self._profile == name:
                await self._shutdown()
            if prof.kind == "launch":
                folder = profile_dir(name)
                if folder.exists():
                    try:
                        shutil.rmtree(folder)
                    except OSError as exc:
                        raise BrowserError(
                            "Couldn't delete the profile's folder. Close any browser window using it and try again."
                        ) from exc
            self._save_profiles({k: v for k, v in self.app.config.browser.profiles.items() if k != name})
        self.sync_visibility()
        return self.profiles()

    # ------------------------------------------------------------------ launch / close
    async def _start_playwright(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - dependency is part of the app
            raise BrowserError("The browser component (Playwright) is missing from this installation.") from exc
        if self._pw is None:
            try:
                self._pw = await asyncio.wait_for(async_playwright().start(), timeout=LAUNCH_TIMEOUT_S)
            except NotImplementedError as exc:
                raise BrowserError("The browser can't start in this mode of the app (unsupported event loop).") from exc

    async def _launch(self, headless: bool) -> None:
        """Start (or attach to) the browser of ``self._profile``. Hold ``_life_lock``."""
        prof = self._profile_cfg(self._profile)
        ok, why = self.availability(self._profile)
        if not ok:
            raise BrowserError(why)
        await self._start_playwright()
        if prof.kind == "attach":
            context = await self._attach(prof)
            headless = False
        else:
            context = await self._launch_persistent(prof, headless)
        self._last_error = None
        self._context = context
        self._site_seen = False
        self._headless = headless
        self._snap = None
        context.set_default_timeout(15_000)
        context.set_default_navigation_timeout(30_000)
        context.on("close", lambda: self._on_context_closed(context))
        context.on("page", self._on_page)
        for page in context.pages:
            self._wire_page(page)
            if self._is_downloads_hub(page.url):
                self._internal_pages.add(page)
        if self._attached:  # work in a tab of our own, never in one the user is using
            self._active = await self._new_page(context)
        else:
            pages = self._usable_pages(context)
            self._active = pages[0] if pages else await context.new_page()
        self._last_used = time.monotonic()
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watch(), name="browser:idle")
        await self._publish_status()

    async def _launch_persistent(self, prof: Any, headless: bool) -> Any:
        cfg = self.app.config.browser
        folder = profile_dir(self._profile)
        folder.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for channel in installed_engines(prof.engine or cfg.engine):
            kwargs: dict[str, Any] = {
                "user_data_dir": str(folder),
                "headless": headless,
                "accept_downloads": True,
                "timeout": 45_000,
                "args": ["--no-first-run", "--no-default-browser-check", "--hide-crash-restore-bubble", STORAGE_FLAG],
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
                return context
            except Exception as exc:
                log.warning("browser: launching %s failed: %s", channel, exc)
                errors.append(f"{ENGINE_NAMES.get(channel, channel)}: {str(exc).splitlines()[0][:200]}")
        self._last_error = (
            "The browser couldn't start. If a Sentient browser window is still open, close it and try again. "
            + " ".join(errors)
        ).strip()
        raise BrowserError(self._last_error)

    async def _attach(self, prof: Any) -> Any:
        """Connect to a browser the user started with a DevTools port on this computer (loopback only)."""
        endpoint, problem = safety.devtools_endpoint(prof.endpoint)
        if problem:
            raise BrowserError(problem)
        ws = await devtools_ws_url(endpoint)
        try:
            browser = await asyncio.wait_for(
                self._pw.chromium.connect_over_cdp(ws, timeout=30_000), timeout=LAUNCH_TIMEOUT_S
            )
        except Exception as exc:
            self._last_error = f"Couldn't attach to the browser at {endpoint}: {str(exc).splitlines()[0][:200]}"
            raise BrowserError(self._last_error) from exc
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        browser.on("disconnected", lambda *_: self._on_context_closed(context))
        self._browser = browser
        self._attached = True
        self._engine = None
        return context

    async def _ensure(self, ctx: Any = None) -> Any:
        """The context of the profile this call uses, switching profiles or launching (configured mode) when needed."""
        want = self._wanted(ctx)
        async with self._life_lock:
            if self._context is not None and self._profile != want:
                if self._for_user:
                    raise BrowserError(
                        f"The browser is open in a window on the '{self._profile}' profile, maybe so the user can "
                        f"sign in. Ask the user to close that window, then try again to use '{want}'."
                    )
                await self._close_context()
            if self._context is None:
                self._profile = want
                await self._launch(headless=self.app.config.browser.headless)
            return self._context

    async def _page(self, ctx: Any = None) -> Any:
        c = await self._ensure(ctx)
        if self._active is None or self._active.is_closed():
            pages = self._usable_pages(c)
            self._active = pages[-1] if pages and not self._attached else await self._new_page(c)
        return self._active

    async def _new_page(self, context: Any) -> Any:
        page = await context.new_page()
        if self._attached:
            self._enable_attached_page_downloads(page)
        return page

    async def _close_context(self) -> None:
        """Close the running profile. An attached browser is only disconnected, never closed. Hold ``_life_lock``."""
        ctx, browser, attached = self._context, self._browser, self._attached
        if ctx is None and browser is None:
            return
        self._closing = True
        try:
            with contextlib.suppress(Exception):
                if attached:
                    await asyncio.wait_for(browser.close(), timeout=CLOSE_TIMEOUT_S)
                else:
                    await self._flush_storage(ctx)
                    await asyncio.wait_for(ctx.close(), timeout=CLOSE_TIMEOUT_S)
        finally:
            self._closing = False
            self._context = None
            self._browser = None
            self._attached = False
            self._for_user = False
            self._active = None
            self._snap = None
            self._focused = None
            self._download_pages.clear()
            self._internal_pages.clear()

    async def _flush_storage(self, ctx: Any) -> None:
        """Let a launched browser write site storage to disk before it closes: close its tabs with their unload
        handlers (so their last writes run and reach the browser), then wait out the storage commit delay. Skipped
        when no website was ever shown in this browser."""
        pages = [p for p in list(ctx.pages) if not p.is_closed()]
        if not (self._site_seen or any(str(p.url or "").startswith(("http://", "https://")) for p in pages)):
            return
        for page in pages:
            await self._close_page(page)
        await asyncio.sleep(STORAGE_FLUSH_S)

    async def _close_page(self, page: Any) -> None:
        """Close a tab running its beforeunload/unload handlers (a beforeunload prompt is accepted by _on_dialog);
        a tab that doesn't close in time is closed without them."""
        closed = asyncio.ensure_future(page.wait_for_event("close", timeout=PAGE_CLOSE_TIMEOUT_S * 1000))
        closed.add_done_callback(lambda f: f.cancelled() or f.exception())  # never "exception was never retrieved"
        try:
            await page.close(run_before_unload=True)
            await asyncio.wait_for(asyncio.shield(closed), timeout=PAGE_CLOSE_TIMEOUT_S)
            return
        except Exception:
            closed.cancel()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(page.close(), timeout=PAGE_CLOSE_TIMEOUT_S)

    async def _shutdown(self, publish: bool = True) -> None:
        async with self._life_lock:
            await self._close_context()
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

    async def open_for_user(self, url: str | None = None, profile: str | None = None) -> dict:
        """Show a profile in a visible window so the user can sign in or finish a step (attached: a new tab)."""
        await self._restart(headless=False, url=url, profile=profile)
        return await self.status()

    async def _restart(self, headless: bool, url: str | None = None, profile: str | None = None) -> None:
        name = (profile or "").strip() or (self._profile if self._context is not None else DEFAULT_PROFILE)
        self._profile_cfg(name)
        ok, why = self.availability(name)
        if not ok:
            raise BrowserError(why)
        target = safety.normalize_url(url or "")
        if target:
            problem = self._url_problem(target)
            if problem:
                raise BrowserError(problem)
        async with self._lock:
            same = self._context is not None and self._profile == name
            if not target and same and self._active is not None and not self._active.is_closed():
                current = self._active.url
                if current.startswith(("http://", "https://")):
                    target = current
            if same and (self._headless == headless or self._attached):
                page = await self._context.new_page() if target and not self._attached else await self._page()
            else:
                await self._shutdown(publish=False)
                async with self._life_lock:
                    self._profile = name
                    await self._launch(headless=headless)
                page = await self._page()
            self._active = page
            self._for_user = not headless and not self._attached
            if target and page.url != target:
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
        self._browser = None
        self._attached = False
        self._for_user = False
        self._active = None
        self._snap = None
        self._focused = None
        self._download_pages.clear()
        if not self._closing:  # the user closed the visible window (or their attached browser)
            self._spawn(self._publish_status())

    def _wire_page(self, page: Any) -> None:
        page.on("close", lambda: self._on_page_closed(page))
        page.on("dialog", self._on_dialog)
        page.on("popup", lambda popup: self._on_popup(page, popup))
        page.on("framenavigated", lambda frame: self._on_navigated(page, frame))
        if not self._attached:
            page.on("download", self._on_download)

    def _enable_attached_page_downloads(self, page: Any) -> None:
        if page not in self._download_pages:
            self._download_pages.add(page)
            page.on("download", self._on_download)

    def _on_navigated(self, page: Any, frame: Any) -> None:
        if frame is page.main_frame and self._is_downloads_hub(frame.url):
            self._mark_internal_page(page)
            return
        # any website shown, by the assistant or by the user in a visible window
        if frame is page.main_frame and str(frame.url or "").startswith(("http://", "https://")):
            self._site_seen = True

    @staticmethod
    def _is_downloads_hub(url: Any) -> bool:
        return str(url or "").lower().startswith("edge://downloads-hub")

    def _mark_internal_page(self, page: Any) -> None:
        self._internal_pages.add(page)
        if self._active is page:
            context = self._context
            pages = [
                candidate for candidate in (self._usable_pages(context) if context is not None else [])
                if not self._attached or candidate in self._download_pages
            ]
            self._active = pages[-1] if pages else None

    def _on_page(self, page: Any) -> None:
        self._wire_page(page)
        if self._is_downloads_hub(page.url):
            self._mark_internal_page(page)
        elif not self._attached:  # a link opened a new tab: keep working in it
            self._active = page
        self._spawn(self._publish_status())

    def _on_popup(self, opener: Any, popup: Any) -> None:
        if self._is_downloads_hub(popup.url):
            self._mark_internal_page(popup)
            return
        if self._attached and opener in self._download_pages:
            self._enable_attached_page_downloads(popup)
            if self._active is opener:
                self._active = popup

    def _on_page_closed(self, page: Any) -> None:
        self._download_pages.discard(page)
        self._internal_pages.discard(page)
        if self._snap and self._snap.get("page") is page:
            self._snap = None
        ctx = self._context
        if self._active is page:
            remaining = [
                p for p in (self._usable_pages(ctx) if ctx else [])
                if p is not page and (not self._attached or p in self._download_pages)
            ]
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

    def _on_download(self, download: Any) -> None:
        task = asyncio.create_task(self._save_download(download), name="browser:download")
        self._download_tasks.add(task)
        self._download_names[task] = self._download_filename(download)
        task.add_done_callback(self._on_download_task_done)

    def _on_download_task_done(self, task: asyncio.Task[str]) -> None:
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.exception()
        if task in self._download_tasks:
            self._download_completed_at[task] = time.monotonic()
            if self._download_cleanup_task is None or self._download_cleanup_task.done():
                self._download_cleanup_task = asyncio.create_task(
                    self._cleanup_download_tasks(), name="browser:download-cleanup"
                )
                self._bg.add(self._download_cleanup_task)
                self._download_cleanup_task.add_done_callback(self._bg.discard)

    async def _cleanup_download_tasks(self) -> None:
        try:
            while self._download_completed_at:
                await asyncio.sleep(DOWNLOAD_TASK_CLEANUP_INTERVAL_S)
                expired_before = time.monotonic() - DOWNLOAD_TASK_RETENTION_S
                async with self._lock:
                    expired = [
                        task
                        for task, completed_at in self._download_completed_at.items()
                        if completed_at <= expired_before and task.done()
                    ]
                    for task in expired:
                        self._download_tasks.discard(task)
                        self._download_names.pop(task, None)
                        self._download_completed_at.pop(task, None)
        finally:
            self._download_cleanup_task = None

    @staticmethod
    def _download_filename(download: Any) -> str:
        raw_name = str(getattr(download, "suggested_filename", None) or "download")
        name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
        return re.sub(r"[^\w.\- ()]+", "_", name).strip(" .")[:160] or "download"

    async def _save_download(self, download: Any) -> str:
        name = self._download_filename(download)

        def reject_symlinked_folder(folder: Path) -> None:
            if folder.is_symlink():
                raise BrowserError("The downloads folder must not be a symbolic link.")

        async def reserve_target() -> Path:
            async with self._download_lock:
                folder = paths.files_dir() / "downloads"
                reject_symlinked_folder(folder)
                folder.mkdir(parents=True, exist_ok=True)
                stem = Path(name).stem
                suffix = Path(name).suffix
                index = 0
                while True:
                    candidate_name = f"{stem}({index}){suffix}" if index else name
                    candidate = folder / candidate_name
                    try:
                        with candidate.open("xb"):
                            pass
                    except FileExistsError:
                        index += 1
                        continue
                    return candidate

        async def request_download_cancel() -> str | None:
            cancel_task = asyncio.create_task(download.cancel())
            done, _ = await asyncio.wait({cancel_task}, timeout=DOWNLOAD_CANCEL_TIMEOUT_S)
            if not done:
                cancel_task.cancel()
                cancel_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                return f"timed out after {DOWNLOAD_CANCEL_TIMEOUT_S:g} seconds"
            try:
                cancel_task.result()
            except asyncio.CancelledError:
                return "the cancellation request was cancelled"
            except Exception as cancel_exc:
                return str(cancel_exc).strip() or type(cancel_exc).__name__
            return None

        def save_failure(exc: BaseException) -> str:
            reason = str(exc).strip().splitlines()[0][:300] or type(exc).__name__
            return f"Couldn't save downloaded file '{name}': {reason}"

        timeout = asyncio.timeout(DOWNLOAD_MAX_DURATION_S)
        target: Path | None = None
        try:
            async with timeout:
                target = await reserve_target()
                await download.save_as(str(target))
        except asyncio.CancelledError as exc:
            try:
                cancel_error = await request_download_cancel()
            finally:
                cleanup_error = self._remove_download_reservation(target)
            if cancel_error:
                exc.add_note(f"Browser cancellation failed: {cancel_error}")
            if cleanup_error:
                exc.add_note(f"Couldn't remove the reserved file: {cleanup_error}")
            raise
        except TimeoutError as exc:
            if timeout.expired():
                cancel_error = await request_download_cancel()
                cleanup_error = self._remove_download_reservation(target)
                message = f"Download '{name}' did not finish within {DOWNLOAD_MAX_DURATION_S:g} seconds."
                if cancel_error:
                    message += f" Browser cancellation failed: {cancel_error}"
                if cleanup_error:
                    message += f" Couldn't remove the reserved file: {cleanup_error}"
                raise BrowserError(message) from exc
            cleanup_error = self._remove_download_reservation(target)
            message = save_failure(exc)
            if cleanup_error:
                message += f"; couldn't remove the reserved file: {cleanup_error}"
            raise BrowserError(message) from exc
        except Exception as exc:
            cleanup_error = self._remove_download_reservation(target)
            if isinstance(exc, BrowserError):
                if cleanup_error:
                    raise BrowserError(f"{exc}; couldn't remove the reserved file: {cleanup_error}") from exc
                raise
            message = save_failure(exc)
            if cleanup_error:
                message += f"; couldn't remove the reserved file: {cleanup_error}"
            raise BrowserError(message) from exc
        assert target is not None
        return target.relative_to(paths.files_dir()).as_posix()

    @staticmethod
    def _remove_download_reservation(target: Path | None) -> str | None:
        if target is None:
            return None
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            return str(exc).strip() or type(exc).__name__
        return None

    async def _include_downloads(
        self,
        result: dict,
        before: set[asyncio.Task[str]],
        wait_for_download_start: bool = False,
    ) -> dict:
        if wait_for_download_start:
            deadline = time.monotonic() + DOWNLOAD_APPEAR_GRACE_S
            while time.monotonic() < deadline:
                if self._download_tasks - before or any(task.done() for task in self._download_tasks):
                    break
                await asyncio.sleep(0.05)

        # Collect finished downloads, but never wait for a save to complete inside a browser action.
        tasks = (self._download_tasks - before) | {task for task in self._download_tasks if task.done()}
        done = {task for task in tasks if task.done()}
        if done:
            try:
                outcomes = await asyncio.gather(*done, return_exceptions=True)
            finally:
                self._download_tasks.difference_update(done)
                for task in done:
                    self._download_names.pop(task, None)
                    self._download_completed_at.pop(task, None)
            names = []
            errors = []
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    message = str(outcome).strip() or f"{type(outcome).__name__} while saving download"
                    errors.append(message)
                else:
                    names.append(outcome)
            if names:
                result["downloads"] = names
            if errors:
                result["download_errors"] = errors

        pending = [task for task in self._download_tasks if not task.done()]
        if pending:
            result["downloads_in_progress"] = [
                self._download_names.get(task, "download") for task in pending
            ]
        return result

    async def _idle_watch(self) -> None:
        while self._context is not None:
            await asyncio.sleep(self.idle_check_s)
            minutes = self.app.config.browser.idle_minutes
            if (
                minutes
                and self._context is not None
                and (self._headless or self._attached)
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
    async def _locate(self, ctx: Any, ref: Any) -> tuple[Any, Any, dict]:
        clean = _clean_ref(ref)
        if not clean:
            raise BrowserError(f"'{ref}' isn't a valid ref. Refs look like e12; take them from browser_snapshot.")
        page = await self._page(ctx)
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
    async def open(self, ctx: Any, url: str, profile: str = "") -> dict:
        target = safety.normalize_url(url)
        if not target:
            raise BrowserError("Give a web address to open, for example https://example.com.")
        problem = self._url_problem(target)
        if problem:
            raise BrowserError(problem)
        self.use_profile(ctx, profile)
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
            download_started = False
            try:
                await page.goto(target, wait_until="domcontentloaded")
            except Exception as exc:
                if "Download is starting" not in str(exc):
                    raise
                download_started = True
            if not download_started:
                await self._settle(page)
            await self._enforce_domains(page)
            result = await self._snapshot_locked(page)
            result["profile"] = self._profile
            await self._after_action(ctx, page)
            result = await self._include_downloads(
                result, downloads_before, wait_for_download_start=True
            )
            return self._take_dialogs(result)

    async def snapshot(self, ctx: Any) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
            self._last_used = time.monotonic()
            result = await self._snapshot_locked(page)
            result = await self._include_downloads(result, downloads_before)
            return self._take_dialogs(result)

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
            downloads_before = self._download_tasks.copy()
            page, loc, info = await self._locate(ctx, ref)
            if info.get("disabled"):
                raise BrowserError(f"{format_element(info)} is disabled. Something else on the page may need to be done first.")
            live_risk = safety.click_risk(info) if self.app.config.browser.confirm_purchases else Risk.write
            refusal = self._guard("click", info["ref"], live_risk)
            if refusal:
                return {"error": refusal}
            pages_before = len(self._usable_pages(self._context))
            self._accept_confirm = live_risk >= Risk.send or self.app.config.tools.approvals.mode == "off"
            try:
                await loc.click(timeout=10_000)
            finally:
                self._accept_confirm = False
            await self._settle(page)
            # a new tab arrives as a separate event; links with target=_blank get longer to show up
            limit = 3.0 if str(info.get("target", "")).lower() == "_blank" else 0.3
            waited = 0.0
            while (
                self._context is not None
                and len(self._usable_pages(self._context)) <= pages_before
                and waited < limit
            ):
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
            if len(self._usable_pages(self._context)) > pages_before:
                out["new_tab"] = True
                out["message"] = "Clicked; it opened a new tab, which is now active. Call browser_snapshot to see it."
            await self._after_action(ctx, active)
            out = await self._include_downloads(
                out, downloads_before, wait_for_download_start=True
            )
            return self._take_dialogs(out)

    async def type(self, ctx: Any, ref: str, text: str, submit: bool = False) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page, loc, info = await self._locate(ctx, ref)
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
            out = await self._include_downloads(
                out, downloads_before, wait_for_download_start=True
            )
            return self._take_dialogs(out)

    async def select(self, ctx: Any, ref: str, option: str) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page, loc, info = await self._locate(ctx, ref)
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
            out = await self._include_downloads(
                out, downloads_before, wait_for_download_start=True
            )
            return self._take_dialogs(out)

    async def press(self, ctx: Any, key: str) -> dict:
        name = normalize_key(key)
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
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
            out = await self._include_downloads(
                out, downloads_before, wait_for_download_start=True
            )
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
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
            pos = await page.evaluate(
                "() => { " + scripts[d] + "; return [Math.round(scrollY), Math.round(document.documentElement.scrollHeight"
                " - scrollY - innerHeight)]; }"
            )
            await asyncio.sleep(0.15)
            out = {"ok": True, "scrolled": d, "from_top": pos[0], "more_below": max(0, pos[1]),
                   "message": "Call browser_snapshot to see what is visible now."}
            await self._after_action(ctx, page)
            return await self._include_downloads(out, downloads_before)

    async def back(self, ctx: Any) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
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
            return await self._include_downloads(
                out, downloads_before, wait_for_download_start=True
            )

    async def tabs(self, ctx: Any, profile: str = "") -> dict:
        self.use_profile(ctx, profile)
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            await self._ensure(ctx)
            self._last_used = time.monotonic()
            result = {"profile": self._profile, "tabs": await self._tabs()}
            return await self._include_downloads(result, downloads_before)

    async def switch_tab(self, ctx: Any, index: int) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            c = await self._ensure(ctx)
            pages = self._usable_pages(c)
            if not 0 <= index < len(pages):
                raise BrowserError(f"There is no tab {index}. Open tabs are numbered 0 to {len(pages) - 1}.")
            url = pages[index].url or ""
            problem = self._url_problem(url) if url.startswith(("http://", "https://")) else None
            if problem:  # an attached browser holds the user's own tabs: leave them as they are
                raise BrowserError(problem)
            self._active = pages[index]
            with contextlib.suppress(Exception):
                await self._active.bring_to_front()
            title = ""
            with contextlib.suppress(Exception):
                title = await self._active.title()
            out = {"ok": True, "index": index, "url": self._active.url, "title": title,
                   "message": "Switched. Call browser_snapshot to see this tab."}
            await self._after_action(ctx, self._active)
            return await self._include_downloads(out, downloads_before)

    async def extract(self, ctx: Any, question: str = "") -> dict:
        cfg = self.app.config.browser
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
            self._last_used = time.monotonic()
            data = await page.evaluate(EXTRACT_JS, {"maxText": cfg.max_extract_chars * 5})
            content, truncated = focus_on_question(data.get("text", ""), question, cfg.max_extract_chars)
            out = {"url": data.get("url", page.url), "title": data.get("title", ""), "content": content or "(no text)"}
            if question:
                out["question"] = question
            if truncated:
                out["truncated"] = True
            return await self._include_downloads(out, downloads_before)

    async def screenshot(self, ctx: Any) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            page = await self._page(ctx)
            self._last_used = time.monotonic()
            folder = paths.files_dir() / "outputs" / "browser"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"screenshot-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')[:-3]}.png"
            await page.screenshot(path=str(folder / name), type="png")
            title = ""
            with contextlib.suppress(Exception):
                title = await page.title()
            result = {"file": f"outputs/browser/{name}", "url": page.url, "title": title}
            return await self._include_downloads(result, downloads_before)

    async def close_tool(self, ctx: Any) -> dict:
        async with self._lock:
            downloads_before = self._download_tasks.copy()
            was_running = self._context is not None
            shutdown_errors = []
            download_deadline = time.monotonic() + DOWNLOAD_MAX_DURATION_S
            shutdown_deadline = download_deadline + DOWNLOAD_CANCEL_TIMEOUT_S
            while pending := {task for task in self._download_tasks if not task.done()}:
                _, pending = await asyncio.wait(
                    pending,
                    timeout=max(0.0, download_deadline - time.monotonic()),
                )
                if not pending:
                    continue

                interrupted_names = {
                    task: self._download_names.get(task, "download") for task in pending
                }
                for task in pending:
                    task.cancel()
                _, still_pending = await asyncio.wait(
                    pending, timeout=max(0.0, shutdown_deadline - time.monotonic())
                )
                for task, name in interrupted_names.items():
                    if task.cancelled():
                        self._download_tasks.discard(task)
                        self._download_names.pop(task, None)
                        self._download_completed_at.pop(task, None)
                        shutdown_errors.append(
                            f"Download '{name}' was cancelled because the browser was closing."
                        )
                    elif task in still_pending:
                        shutdown_errors.append(
                            f"Download '{name}' did not stop before browser shutdown."
                        )
                break
            await self._shutdown()
        result = {"ok": True, "message": "Browser closed." if was_running else "The browser wasn't open."}
        result = await self._include_downloads(result, downloads_before)
        if shutdown_errors:
            result.setdefault("download_errors", []).extend(shutdown_errors)
        return result


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
