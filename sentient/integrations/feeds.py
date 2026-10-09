"""Change feeds, push watchers, and the shared "already emitted" record.

Every new item from any origin goes through :meth:`FeedEngine.emit`:

- ``poll``     items returned by ``IntegrationManager.poll_source`` (proactivity's timer)
- ``feed``     items found by a change feed (Gmail history, Calendar sync token) or a push watcher (IMAP IDLE)
- ``webhook``  requests to ``POST /hooks/{id}``

``emit`` claims each item key once per source in ``integration_seen`` (persisted), applies the
source's privacy filters and publishes domain event ``source.items``
``{source, event, origin, items}``. A feed and a poll therefore never emit the same item twice.

Change feeds run every ``integrations.fast_sync_seconds`` for connected sources only, cost no model
calls, and back off exponentially on failure with a readable ``last_error``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from sentient.integrations.base import IntegrationError, _http_error_message
from sentient.store.db import now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

ORIGINS = ("poll", "feed", "webhook")
INITIAL_DELAY_S = 5.0
MIN_INTERVAL_S = 5.0
BACKOFF_MAX_S = 30 * 60
ACTIVE_MAX_FAILURES = 3  # after this many consecutive failures, proactivity polls again
SEEN_RETENTION = timedelta(days=30)
PRUNE_EVERY_S = 6 * 3600

_STATE_COLUMNS = ("cursor", "status", "failures", "last_sync_at", "last_success_at", "last_error", "note",
                  "next_attempt_at", "emitted")


def backoff_seconds(failures: int, base: float) -> float:
    """base, 2*base, 4*base ... capped at 30 minutes."""
    return float(min(BACKOFF_MAX_S, max(float(base), MIN_INTERVAL_S) * 2 ** max(0, failures - 1)))


def readable_error(display: str, exc: BaseException) -> str:
    if isinstance(exc, IntegrationError):
        return str(exc) or f"{display} couldn't be checked."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{display} couldn't be checked ({_http_error_message(exc)})."
    if isinstance(exc, httpx.RequestError):
        return f"Couldn't reach {display} ({type(exc).__name__}). Check the internet connection."
    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return f"{display} couldn't be checked ({detail})."


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class FeedEngine:
    def __init__(self, mgr: IntegrationManager):
        self.mgr = mgr
        self._loop_task: asyncio.Task | None = None
        self._watchers: dict[str, asyncio.Task] = {}
        self._failures: dict[str, int] = {}
        self._last_prune = 0.0

    @property
    def app(self) -> Any:
        return self.mgr.app

    # ------------------------------------------------------------------ lifecycle
    async def load(self) -> None:
        for row in await self.app.store.fetchall("SELECT source, failures FROM integration_feed_state"):
            self._failures[row["source"]] = int(row["failures"] or 0)

    def start_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._loop(), name="integrations:change-feeds")

    async def stop(self) -> None:
        tasks = [t for t in [self._loop_task, *self._watchers.values()] if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._loop_task = None
        self._watchers.clear()

    @property
    def loop_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    async def _loop(self) -> None:
        await asyncio.sleep(INITIAL_DELAY_S)
        while True:
            seconds = self.app.config.integrations.fast_sync_seconds
            if seconds > 0 and not self.app.stopped:  # Stop everything: cursors wait, items arrive after resume
                try:
                    await self.sync_all()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("change feed sweep failed")
            await asyncio.sleep(max(float(seconds), MIN_INTERVAL_S) if seconds > 0 else 30.0)

    # ------------------------------------------------------------------ push watchers (IMAP IDLE)
    def start_watch(self, plugin_id: str) -> None:
        p = self.mgr.plugin(plugin_id)
        watch = getattr(p, "watch", None)
        if watch is None or not self.mgr._connected_sync(plugin_id):
            return
        current = self._watchers.get(plugin_id)
        if current is not None and not current.done():
            return
        self._watchers[plugin_id] = asyncio.create_task(watch(self.mgr), name=f"integrations:watch:{plugin_id}")

    async def stop_watch(self, plugin_id: str) -> None:
        t = self._watchers.pop(plugin_id, None)
        if t is not None:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    def watching(self, plugin_id: str) -> bool:
        t = self._watchers.get(plugin_id)
        return t is not None and not t.done()

    # ------------------------------------------------------------------ public status
    def active(self, source: str) -> bool:
        """True when a change feed or push watcher is keeping ``source`` up to date (so timer polling can skip it)."""
        p = self.mgr.plugin(source)
        if p is None or not self.mgr._connected_sync(source):
            return False
        if self._failures.get(source, 0) >= ACTIVE_MAX_FAILURES:
            return False
        if getattr(p, "change_feed", None) is not None:
            return self.loop_running and self.app.config.integrations.fast_sync_seconds > 0
        if getattr(p, "watch", None) is not None:
            return self.watching(source)
        return False

    async def status(self) -> list[dict]:
        out = []
        for p in self.mgr.plugins():
            has_feed = getattr(p, "change_feed", None) is not None
            if not has_feed and getattr(p, "watch", None) is None:
                continue
            st = await self.state(p.id)
            connected = self.mgr._connected_sync(p.id)
            active = self.active(p.id)
            if not connected:
                status = "disconnected"
            elif has_feed and not (self.loop_running and self.app.config.integrations.fast_sync_seconds > 0):
                status = "off"
            elif st["status"] == "error":
                status = "error"
            elif st["cursor"]:
                status = "ok"
            else:
                status = "starting"
            out.append({
                "source": p.id, "display_name": p.display_name, "kind": p.feed_kind, "connected": connected,
                "active": active, "status": status, "last_sync_at": st["last_sync_at"],
                "last_success_at": st["last_success_at"], "last_error": st["last_error"], "note": st["note"],
                "failures": st["failures"], "next_attempt_at": st["next_attempt_at"], "emitted": st["emitted"],
            })
        return out

    # ------------------------------------------------------------------ state
    async def state(self, source: str) -> dict:
        row = await self.app.store.fetchone("SELECT * FROM integration_feed_state WHERE source = ?", (source,))
        if row is None:
            return {"source": source, "cursor": None, "status": None, "failures": 0, "last_sync_at": None,
                    "last_success_at": None, "last_error": None, "note": None, "next_attempt_at": None,
                    "emitted": 0, "updated_at": None}
        d = dict(row)
        d["failures"] = int(d.get("failures") or 0)
        d["emitted"] = int(d.get("emitted") or 0)
        return d

    async def save(self, source: str, **changes: Any) -> dict:
        st = {**(await self.state(source)), **changes}
        st["failures"] = int(st.get("failures") or 0)
        cols = ", ".join(_STATE_COLUMNS)
        await self.app.store.execute(
            f"INSERT INTO integration_feed_state(source, {cols}, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(source) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in (*_STATE_COLUMNS, "updated_at")),
            (source, *[st.get(c) for c in _STATE_COLUMNS], now_iso()),
        )
        self._failures[source] = st["failures"]
        return st

    async def reset(self, source: str) -> None:
        """Forget the cursor (next sync re-baselines from now). Used on connect/disconnect."""
        await self.app.store.execute("DELETE FROM integration_feed_state WHERE source = ?", (source,))
        self._failures.pop(source, None)

    async def record_success(self, source: str, *, cursor: str | None, emitted: int = 0, note: str | None = None,
                             started_at: str | None = None) -> dict:
        st = await self.state(source)
        ts = started_at or now_iso()
        return await self.save(source, cursor=cursor, status="ok", failures=0, last_error=None, next_attempt_at=None,
                               last_sync_at=ts, last_success_at=ts, emitted=st["emitted"] + emitted,
                               note=note if note is not None else st["note"])

    async def record_failure(self, source: str, exc: BaseException, *, base: float | None = None) -> dict:
        p = self.mgr.plugin(source)
        display = p.display_name if p else source
        msg = readable_error(display, exc)
        st = await self.state(source)
        failures = st["failures"] + 1
        delay = backoff_seconds(failures, base if base is not None else self.app.config.integrations.fast_sync_seconds)
        next_at = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
        await self.save(source, status="error", failures=failures, last_error=msg, last_sync_at=now_iso(),
                        next_attempt_at=next_at)
        log.warning("%s change feed failed (%d in a row, retry in %.0fs): %s", source, failures, delay, msg)
        return {"ok": False, "error": msg, "failures": failures, "retry_in_s": delay, "emitted": 0}

    # ------------------------------------------------------------------ change feeds
    async def sync(self, source: str) -> dict:
        """Run one change-feed sync for ``source`` now (ignores backoff). Never raises for service errors."""
        p = self.mgr.plugin(source)
        feed = getattr(p, "change_feed", None)
        if p is None or feed is None:
            raise ValueError(f"{source} has no change feed")
        if not self.mgr._connected_sync(source):
            return {"ok": False, "skipped": "not connected", "emitted": 0}
        async with self.mgr.lock(f"feed:{source}"):
            st = await self.state(source)
            started = now_iso()
            try:
                batch = await feed(self.mgr, st["cursor"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return await self.record_failure(source, exc)
            kept = await self.emit(source, "feed", batch.items) if batch.items else []
            await self.record_success(source, cursor=batch.cursor, emitted=len(kept), note=batch.note,
                                      started_at=started)
            return {"ok": True, "emitted": len(kept), "rebaselined": batch.rebaselined}

    async def sync_all(self) -> int:
        """Sync every connected source that has a change feed and isn't waiting out a backoff."""
        total = 0
        now = datetime.now(UTC)
        for p in self.mgr.plugins():
            if getattr(p, "change_feed", None) is None or not self.mgr._connected_sync(p.id):
                continue
            nxt = _parse((await self.state(p.id))["next_attempt_at"])
            if nxt is not None and nxt > now:
                continue
            total += int((await self.sync(p.id)).get("emitted", 0))
        await self._maybe_prune()
        return total

    async def _maybe_prune(self) -> None:
        if time.monotonic() - self._last_prune < PRUNE_EVERY_S:
            return
        self._last_prune = time.monotonic()
        cutoff = (datetime.now(UTC) - SEEN_RETENTION).isoformat()
        await self.app.store.execute("DELETE FROM integration_seen WHERE seen_at < ?", (cutoff,))

    # ------------------------------------------------------------------ shared seen record + source.items
    async def claim(self, source: str, keys: list[str], origin: str) -> set[str]:
        """Record keys as emitted; returns the ones that were not seen before."""
        db = self.app.store.db
        ts = now_iso()
        fresh: set[str] = set()
        for key in dict.fromkeys(keys):
            cur = await db.execute(
                "INSERT OR IGNORE INTO integration_seen(source, item_key, origin, seen_at) VALUES(?,?,?,?)",
                (source, key, origin, ts),
            )
            if cur.rowcount > 0:
                fresh.add(key)
        await db.commit()
        return fresh

    async def emit(self, source: str, origin: str, items: list[dict], *, event: str | None = None) -> list[dict]:
        """Claim, privacy-filter and publish new items. Returns the items that were published (in order).

        While Sentient is stopped (Stop everything) nothing is claimed or published."""
        if origin not in ORIGINS:
            raise ValueError(f"unknown origin {origin}")
        if self.app.stopped:
            return []
        candidates: list[tuple[str, dict]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            key = item.get("_key", item.get("id"))
            if key is None or key == "":
                continue
            candidates.append((str(key), item))
        if not candidates:
            return []
        fresh_keys = await self.claim(source, [k for k, _ in candidates], origin)
        fresh: list[dict] = []
        used: set[str] = set()
        for key, item in candidates:
            if key in fresh_keys and key not in used:
                used.add(key)
                fresh.append(item)
        p = self.mgr.plugin(source)
        kind = getattr(p, "privacy_kind", None)
        kept = await self.mgr.filter_items(source, fresh, kind) if kind and fresh else fresh
        default_event = event or ((p.triggers[0]["event"]) if p is not None and p.triggers else "new_item")
        groups: dict[str, list[dict]] = {}
        out: list[dict] = []
        for item in kept:
            ev = item.pop("_event", None) or default_event
            item.pop("_key", None)
            groups.setdefault(ev, []).append(item)
            out.append(item)
        for ev, group in groups.items():
            self.app.bus.publish("source.items", {"source": source, "event": ev, "origin": origin, "items": group})
        return out
