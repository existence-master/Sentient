"""Proactivity: watches connected apps and runs the v2 proactive pipeline, learning from
approve/dismiss feedback.

OWNER: EVOLUTION AGENT (proactivity, evolution, skills).

New items arrive two ways (docs/API.md section 16):
    - timer polling: ``app.integrations.poll_source`` for every watched source whose change feed
      is not active; the items it returns are processed here (their ``source.items`` bus event with
      ``origin: "poll"`` is ignored so nothing is processed twice)
    - push: ``source.items`` bus events with ``origin`` ``feed`` (Gmail history, Calendar sync
      tokens, IMAP IDLE) or ``webhook``
Triggered tasks are no longer fired from here: the tasks package consumes ``source.items`` itself.

Per new item (gmail ``new_email`` / gcalendar ``new_event`` / webhook call):
    1. skip items that already produced a suggestion, and webhooks a triggered task handles
    2. v2 ``event_pre_filter`` heuristics
    3. formulate search queries (fast)
    4. gather context, the "cognitive scratchpad": memory recall, past-conversation summaries,
       related tasks, what the user model knows, and a short read-only agent step over connected
       apps (v2 unified search)
    5. proactive reasoner (v2 prompt + JSON schema)
    6. drop near-duplicates of suggestions still open; standardize the type (v2)
    7. learned threshold ``base - 0.05 * score`` clamped to [0.40, 0.95] (v2)
    8. quiet hours (deferred, delivered afterwards) -> ``app.notify("proactive", ...)``
Suggestions nobody acted on expire (``suggestion_ttl_hours``; calendar ones when the event starts).

Integration contract used (guarded so this works before it exists):
``await app.integrations.is_connected(source) -> bool``,
``await app.integrations.poll_source(source, since: str) -> list[dict]`` and
``app.integrations.feed_active(source) -> bool`` (sync or async).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
from datetime import UTC, datetime, time, timedelta
from typing import Any

from sentient.agent.loop import LoopResult

try:  # raised by the integrations package when a service call fails (after recording poll_state.last_error)
    from sentient.integrations.base import IntegrationError
except ImportError:  # pragma: no cover

    class IntegrationError(Exception):  # type: ignore[no-redef]
        pass

from sentient.integrations import redact
from sentient.llm.provider import parse_json_loose
from sentient.proactivity import followups, prompts
from sentient.proactivity.prefilter import (
    event_pre_filter,
    event_start,
    event_summary,
    extract_query_text,
    in_quiet_hours,
)
from sentient.services import Service
from sentient.store.db import new_id
from sentient.tools.base import Risk
from sentient.tools.builtin.time_tool import resolve_tz

log = logging.getLogger(__name__)

SOURCE_EVENTS = {"gmail": "new_email", "gcalendar": "new_event", "email_imap": "new_email"}
EMAIL_SOURCES = {"gmail", "email_imap"}
PUSH_ORIGINS = {"feed", "webhook"}
SOURCE_LABELS = {"gmail": "Gmail", "gcalendar": "Calendar", "email_imap": "Email", "heartbeat": "Check-in", "webhook": "Webhook"}
DEFAULT_TYPE = "custom_proactive_action"
CONTEXT_EXCLUDED_PLUGINS = {"memory", "skills", "files", "core"}
ATTENTION_STATUSES = {"approval_pending", "clarification_pending", "waiting_for_user", "error", "completed_with_errors"}
INACTIVE_TASK_STATUSES = {"archived", "cancelled", "declined", "completed"}
OPEN_STATUSES = ("pending", "deferred")
HEARTBEAT_TTL_HOURS = 12
DUPLICATE_SIMILARITY = 0.8
FOLLOWUP_SOURCES = ("gmail", "email_imap")  # email accounts follow-ups can read; followups.sources picks among them
FOLLOWUP_LOCAL_TIME = time(8, 0)  # the daily follow-up check runs from this local time on
FOLLOWUP_MAX_CANDIDATES = 10      # threads per check that reach the model
FOLLOWUP_SCAN_THREADS = 40        # recent threads read per mail account per check
FOLLOWUP_META = "proactivity.followups_last_run"
FOLLOWUP_SEEN_SOURCE = "followups"


class SuggestionError(Exception):
    """Raised by ``act_on_suggestion``; ``status`` is the HTTP status the route returns."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _to_utc_iso(value: Any) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return _iso(dt)


def _trim_item(item: dict, max_body: int = 3000) -> dict:
    out = {}
    for k, v in item.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, str) and len(v) > max_body:
            v = v[:max_body] + " [...]"
        out[k] = v
    return out


def source_kind(source: str) -> str:
    """Item shape of a source: IMAP email uses the gmail item shape and filters."""
    return "gmail" if source in EMAIL_SOURCES else source


def similar(a: str, b: str) -> bool:
    """Word-set overlap: 'Draft a reply to Jane' vs 'Draft reply to Jane' counts as the same suggestion."""
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= DUPLICATE_SIMILARITY


def part_of_day(local: datetime) -> str:
    h = local.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


class ProactiveEngine(Service):
    name = "proactivity"

    def __init__(self, app):
        super().__init__(app)
        self._poll_lock = asyncio.Lock()  # one pipeline at a time: polls and pushed items share the model
        self._followup_lock = asyncio.Lock()
        self._acting: set[str] = set()  # suggestions being approved or dismissed right now
        self._background: set[asyncio.Task] = set()

    @property
    def cfg(self):
        return self.app.config.proactivity

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if not self.app.enable_background:
            return
        self._loops.append(asyncio.create_task(self._listen(), name="proactivity:bus"))
        # one light ticker so interval/heartbeat/enabled changes in Settings apply without a restart
        self.run_every(60, self._tick, name="tick", initial_delay=45)

    async def stop(self) -> None:
        for t in list(self._background):
            t.cancel()
        await super().stop()

    async def _listen(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                data = event.get("data")
                if event.get("type") != "source.items" or not isinstance(data, dict):
                    continue
                if str(data.get("origin") or "").lower() not in PUSH_ORIGINS:
                    continue
                task = asyncio.create_task(self.on_source_items(data), name="proactivity:items")
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    async def _tick(self) -> None:
        now = datetime.now(UTC)
        # polling also feeds triggered tasks (poll_source publishes source.items), so it runs while either consumer exists
        consumers = self.cfg.enabled or hasattr(self.app.tasks, "handle_event")
        if consumers and await self._due("proactivity.poll_last_run", timedelta(minutes=self.cfg.poll_interval_minutes), now):
            await self.poll_all()
        if self.cfg.enabled and self.cfg.heartbeat_minutes > 0 and await self._due(
            "proactivity.heartbeat_last_run", timedelta(minutes=self.cfg.heartbeat_minutes), now
        ):
            await self.heartbeat()
        if self.cfg.enabled and self.cfg.followups.enabled and await self._followups_due(now):
            await self.run_followups(now=now)
        if self.cfg.enabled:
            await self.flush_deferred()
        await self.expire_stale(now)

    async def _due(self, key: str, every: timedelta, now: datetime) -> bool:
        raw = await self.app.store.get_meta(key)
        last = datetime.fromisoformat(raw) if raw else None
        if last is not None and now - last < every:
            return False
        await self.app.store.set_meta(key, now.isoformat())
        return True

    # ------------------------------------------------------------------ integrations (guarded)
    async def _is_connected(self, source: str) -> bool:
        integ = self.app.integrations
        if not hasattr(integ, "is_connected"):
            return False
        try:
            return bool(await _maybe_await(integ.is_connected(source)))
        except Exception as exc:
            log.debug("is_connected(%s) failed: %s", source, exc)
            return False

    async def _feed_active(self, source: str) -> bool:
        # TODO(integrations): expose `feed_active(source: str) -> bool` (sync or async) on IntegrationManager:
        # True while a change feed (Gmail history, Calendar sync token, IMAP IDLE) delivers source.items for
        # `source`. Until it exists every watched source is timer-polled.
        fn = getattr(self.app.integrations, "feed_active", None)
        if not callable(fn):
            return False
        try:
            return bool(await _maybe_await(fn(source)))
        except Exception as exc:
            log.debug("feed_active(%s) failed: %s", source, exc)
            return False

    async def _user_email(self, source: str) -> str | None:
        integ = self.app.integrations
        for getter in ("integration", "get_integration", "get"):
            if not hasattr(integ, getter):
                continue
            with contextlib.suppress(Exception):
                info = await _maybe_await(getattr(integ, getter)(source))
                label = info.get("account_label") if isinstance(info, dict) else getattr(info, "account_label", None)
                if label and "@" in str(label):
                    return str(label)
        return None

    async def _set_source(self, source: str, **fields: Any) -> None:
        store = self.app.store
        await store.execute(
            "INSERT INTO proactive_sources(source, updated_at) VALUES(?, ?) ON CONFLICT(source) DO NOTHING",
            (source, _iso(datetime.now(UTC))),
        )
        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            await store.execute(
                f"UPDATE proactive_sources SET {cols}, updated_at = ? WHERE source = ?",
                [*fields.values(), _iso(datetime.now(UTC)), source],
            )

    # ------------------------------------------------------------------ polling
    async def poll_all(self, *, force: bool = False) -> int:
        """Poll every watched, connected source once. Returns the number of new items.

        Sources whose change feed is active are skipped (their items arrive as ``source.items``)
        unless ``force`` (the "check now" button)."""
        async with self._poll_lock:
            events = 0
            integ = self.app.integrations
            for source in self.cfg.sources:
                started = datetime.now(UTC)
                connected = await self._is_connected(source)
                await self._set_source(source, connected=int(connected))
                if not connected or not hasattr(integ, "poll_source"):
                    continue
                if not force and await self._feed_active(source):
                    continue
                row = await self.app.store.fetchone(
                    "SELECT last_success_at FROM proactive_sources WHERE source = ?", (source,)
                )
                since = (
                    datetime.fromisoformat(row["last_success_at"])
                    if row and row["last_success_at"]
                    else started - timedelta(hours=self.cfg.first_poll_lookback_hours)
                )
                try:
                    items = await _maybe_await(integ.poll_source(source, _iso(since))) or []
                except IntegrationError as exc:
                    error = await self._integration_error(source, exc)
                    log.warning("polling %s failed: %s", source, error)
                    await self._set_source(source, last_poll_at=_iso(started), last_error=error[:500])
                    continue
                except Exception as exc:  # never let one source take the poll loop down
                    log.exception("polling %s crashed", source)
                    await self._set_source(
                        source, last_poll_at=_iso(started), last_error=f"{type(exc).__name__}: {exc}"[:500]
                    )
                    continue
                await self._set_source(source, last_poll_at=_iso(started), last_success_at=_iso(started), last_error=None)
                for item in list(items)[: self.cfg.max_items_per_poll]:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    if not await self._mark_seen(source, item):
                        continue
                    events += 1
                    await self.handle_item(source, item)
            return events

    async def on_source_items(self, data: dict) -> int:
        """Consume one ``source.items`` event. Only ``feed`` and ``webhook`` origins are processed here:
        ``poll`` items come back from ``poll_source`` inside ``poll_all``. Returns the number of new items."""
        origin = str(data.get("origin") or "").strip().lower()
        if origin not in PUSH_ORIGINS:
            return 0
        source = str(data.get("source") or "").strip().lower()
        items = data.get("items")
        if not source or not isinstance(items, list):
            return 0
        if source == "webhook" or origin == "webhook":
            if not self.cfg.webhook_suggestions:
                return 0
        elif source not in self.cfg.sources:
            return 0
        event_type = str(data.get("event") or SOURCE_EVENTS.get(source, "new_item"))
        count = 0
        async with self._poll_lock:
            for item in items[: self.cfg.max_items_per_poll]:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                if not await self._mark_seen(source, item, event_type):
                    continue
                count += 1
                await self.handle_item(source, item, event_type=event_type)
        return count

    async def _integration_error(self, source: str, exc: Exception) -> str:
        """Prefer the message the integrations package recorded in its poll state."""
        integ = self.app.integrations
        if hasattr(integ, "poll_state"):
            with contextlib.suppress(Exception):
                state = await _maybe_await(integ.poll_state(source))
                if isinstance(state, dict) and state.get("last_error"):
                    return str(state["last_error"])
        return str(exc) or type(exc).__name__

    async def _mark_seen(self, source: str, item: dict, event_type: str | None = None) -> bool:
        cur = await self.app.store.execute(
            "INSERT OR IGNORE INTO proactive_seen(source, item_id, event_type, item, starts_at, seen_at) VALUES(?,?,?,?,?,?)",
            (
                source, str(item["id"]), event_type or SOURCE_EVENTS.get(source, "new_item"),
                json.dumps(_trim_item(item, 1500), default=str),
                _to_utc_iso(event_start(item)) if source == "gcalendar" else None,
                _iso(datetime.now(UTC)),
            ),
        )
        if cur.rowcount > 0:
            await self.app.store.execute(
                "UPDATE proactive_sources SET items_seen = items_seen + 1 WHERE source = ?", (source,)
            )
            return True
        return False

    async def handle_item(self, source: str, item: dict, *, event_type: str | None = None) -> dict | None:
        """Run the suggestion pipeline for one new item (triggered tasks are the tasks package's job)."""
        if not self.cfg.enabled:
            return None
        event_type = event_type or SOURCE_EVENTS.get(source, "new_item")
        try:
            if source == "webhook" and await self._hook_has_task(event_type):
                log.info("webhook %s is handled by a triggered task: no suggestion", event_type)
                return None
            return await self.process_event(source, event_type, item)
        except Exception:
            log.exception("proactive pipeline failed for %s item %s", source, item.get("id"))
            return None

    async def poll_now(self) -> dict:
        """The "Check now" button: poll every source, and start a follow-up check in the background."""
        events = await self.poll_all(force=True)
        if self.cfg.enabled and self.cfg.followups.enabled and not self._followup_lock.locked():
            task = asyncio.create_task(self.run_followups(), name="proactivity:followups")
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return {"ok": True, "events": events}

    async def _hook_has_task(self, hook_id: str) -> bool:
        for t in await self._task_list():
            schedule = t.get("schedule") or {}
            if not isinstance(schedule, dict) or schedule.get("type") != "triggered":
                continue
            if t.get("enabled") is False or t.get("status") in INACTIVE_TASK_STATUSES:
                continue
            if str(schedule.get("source") or "").lower() == "webhook" and str(schedule.get("event") or "") in {"", hook_id}:
                return True
        return False

    # ------------------------------------------------------------------ pipeline
    @staticmethod
    def _event_label(source: str, event_type: str, item: dict) -> str:
        if source == "webhook":
            return f"call to the webhook '{item.get('name') or event_type}'"
        return f"{source} {event_type}"

    async def _already_suggested(self, source: str, item_id: str) -> bool:
        row = await self.app.store.fetchone(
            "SELECT 1 AS x FROM proactive_suggestions WHERE source = ? AND item_id = ? LIMIT 1", (source, item_id)
        )
        return row is not None

    async def process_event(self, source: str, event_type: str, item: dict, *, now: datetime | None = None) -> dict | None:
        """Run the whole proactive pipeline for one item. Returns the suggestion record, or None."""
        now = now or datetime.now(UTC)
        kind = source_kind(source)
        if source in EMAIL_SOURCES and redact.enabled(self.app):
            # the mail plugins already hide one-time codes and sign-in links; this keeps any other path safe too
            item = redact.hide_email_secrets(dict(item))
        if item.get("id") and await self._already_suggested(source, str(item["id"])):
            log.info("%s item %s already produced a suggestion", source, item.get("id"))
            return None
        if not event_pre_filter(item, kind, await self._user_email(source)):
            log.info("proactive pre-filter discarded %s item %s", source, item.get("id"))
            return None
        queries = await self.formulate_queries(source, event_type, item)
        scratchpad = await self.gather_context(source, event_type, item, queries)
        about = await self._user_model_context(f"{event_summary(item, kind)} {' '.join(queries.values())}")
        if about:
            scratchpad["about_the_user"] = about
        scratchpad["user_preferences"] = await self.preference_scores()
        scratchpad["suggestions_already_made"] = await self._recent_descriptions(now)
        trigger = {"event_type": source, "event": event_type, "event_data": _trim_item(item)}
        if source == "webhook":
            trigger["note"] = prompts.WEBHOOK_NOTE.format(name=item.get("name") or event_type)
        scratchpad["trigger_event"] = trigger
        scratchpad["current_time_utc"] = _iso(now)
        scratchpad["user"] = self._user_block(now)
        result = await self.run_reasoner(scratchpad)
        return await self.handle_reasoner_result(
            result, source=source, event_type=event_type, item=item, context=scratchpad, now=now
        )

    def _user_block(self, now: datetime) -> dict:
        a = self.app.config.assistant
        tz = resolve_tz(a.timezone)
        local = now.astimezone(tz)
        return {
            "name": a.user_name or None,
            "timezone": str(tz),
            "local_time": local.strftime("%A %Y-%m-%d %H:%M"),
            "time_of_day": part_of_day(local),
            "location": a.location or None,
        }

    async def _user_model_context(self, text: str) -> str | None:
        """What the user model knows that matters here (docs/API.md section 15); None when unavailable."""
        fn = getattr(getattr(self.app, "user_model", None), "context_for", None)
        if not callable(fn):
            return None
        try:
            out = await _maybe_await(fn(text[:1000]))
        except Exception as exc:
            log.debug("user model context failed: %s", exc)
            return None
        return str(out).strip()[:1500] if isinstance(out, str) and out.strip() else None

    async def _recent_descriptions(self, now: datetime, *, statuses: tuple[str, ...] | None = None, limit: int = 10) -> list[str]:
        sql = "SELECT description FROM proactive_suggestions WHERE created_at >= ?"
        params: list[Any] = [_iso(now - timedelta(hours=24))]
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params += list(statuses)
        rows = await self.app.store.fetchall(sql + " ORDER BY created_at DESC LIMIT ?", [*params, limit])
        return [r["description"] for r in rows]

    async def formulate_queries(self, source: str, event_type: str, item: dict) -> dict[str, str]:
        kind = source_kind(source)
        try:
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.QUERY_FORMULATION_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Trigger event: a {self._event_label(source, event_type, item)}.\n\nEvent data:\n"
                        + json.dumps(_trim_item(item, 2000), indent=2, default=str)
                        + "\n\nRespond with ONLY the JSON object mapping snake_case purposes to natural language search questions.",
                    },
                ],
            )
        except Exception as exc:
            log.warning("query formulation failed: %s", exc)
            raw = None
        queries: dict[str, str] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                # small models sometimes echo the input ({"event_type": "gmail"}) instead of writing questions
                if isinstance(v, str) and len(v.split()) >= 2 and v.strip().lower() not in {source, event_type}:
                    queries[re.sub(r"[^a-z0-9_]+", "_", str(k).lower()).strip("_") or f"query_{len(queries)}"] = v.strip()
                if len(queries) >= 4:
                    break
        if not queries:
            queries = {"event_context": (extract_query_text(item, kind) or event_summary(item, kind))[:500]}
        return queries

    async def gather_context(self, source: str, event_type: str, item: dict, queries: dict[str, str]) -> dict:
        results: dict[str, Any] = {}
        mem = self.app.memory
        for key, question in queries.items():
            entry: dict[str, Any] = {"query": question}
            if mem is not None:
                with contextlib.suppress(Exception):
                    entry["memories"] = [f["content"] for f in await mem.recall(question, top_k=5)]
                with contextlib.suppress(Exception):
                    hits = await mem.episodic.search(question, limit=3)
                    entry["past_conversations"] = [
                        h["content"][:600] for h in hits if h.get("similarity", 1.0) >= 0.3
                    ]
            results[key] = entry
        related = await self.related_tasks(f"{extract_query_text(item, source_kind(source))} {' '.join(queries.values())}")
        if related:
            results["related_tasks"] = related
        live = await self.live_search(source, event_type, item, queries)
        if live:
            results["connected_apps_search"] = live
        return {"universal_search_results": results}

    async def _task_list(self) -> list[dict]:
        tasks = self.app.tasks
        for getter in ("list", "list_tasks"):
            if hasattr(tasks, getter):
                with contextlib.suppress(Exception):
                    items = await _maybe_await(getattr(tasks, getter)())
                    return [t for t in items or [] if isinstance(t, dict)]
        return []

    async def related_tasks(self, text: str, limit: int = 5) -> list[dict]:
        words = {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())}
        scored = []
        for t in await self._task_list():
            if t.get("status") in INACTIVE_TASK_STATUSES:
                continue
            blob = f"{t.get('name', '')} {t.get('description', '')}".lower()
            overlap = sum(1 for w in words if w in blob)
            if overlap:
                scored.append((overlap, t))
        scored.sort(key=lambda x: -x[0])
        return [
            {
                "name": t.get("name"), "status": t.get("status"),
                "description": (t.get("description") or "")[:200],
                "next_execution_at": t.get("next_execution_at"),
            }
            for _, t in scored[:limit]
        ]

    async def _read_tool_names(self) -> list[str]:
        """Read-only tools of connected apps. ``registry.tools()`` already hides disconnected integrations and
        "never" tools; tools with an "ask" rule are left out because nobody is there to ask."""
        if self.app.agent is None:
            return []
        return [
            t.name for t in self.app.registry.tools()
            if t.risk == Risk.read and t.plugin not in CONTEXT_EXCLUDED_PLUGINS and self.app.approvals.rule(t) != "ask"
        ]

    async def live_search(self, source: str, event_type: str, item: dict, queries: dict[str, str]) -> str | None:
        rounds = self.cfg.context_agent_rounds
        if rounds <= 0 or self.app.agent is None:
            return None
        names = await self._read_tool_names()
        if not names:
            return None
        questions = "\n".join(f"- {q}" for q in queries.values())
        messages = [
            {"role": "system", "content": prompts.UNIFIED_SEARCH_SYSTEM},
            {
                "role": "user",
                "content": f"Trigger event ({self._event_label(source, event_type, item)}): "
                f"{event_summary(item, source_kind(source))}\n\nQuestions:\n{questions}",
            },
        ]
        result = LoopResult()
        ctx = self.app.agent.tool_context(None, "proactive")
        try:
            async for _event in self.app.agent.run_loop(
                messages, ctx, result=result, role="fast", tool_names=names, max_rounds=rounds,
                use_approvals=False, source="proactive",
            ):
                pass
        except Exception as exc:
            log.warning("proactive live search failed: %s", exc)
            return None
        text = re.sub(r"</?(answer|think)>", "", result.text or "").strip()
        if result.tool_calls and (not text or text.startswith("I reached the step limit")):
            tool_outputs = [m["content"][:800] for m in result.messages if m.get("role") == "tool"][-5:]
            text = "\n".join(tool_outputs)
        return text[:4000] or None

    async def run_reasoner(self, scratchpad: dict) -> dict:
        body = json.dumps(scratchpad, indent=1, default=str, ensure_ascii=False)
        if len(body) > 24_000:
            body = body[:24_000]
        messages = [
            {"role": "system", "content": prompts.PROACTIVE_REASONER_SYSTEM},
            {"role": "user", "content": prompts.REASONER_USER.format(scratchpad=body)},
        ]
        for attempt in (1, 2):
            try:
                raw = await self.app.llm.complete_json(self.cfg.reasoner_role, messages)
            except Exception as exc:
                log.warning("proactive reasoner failed: %s", exc)
                return {"actionable": False, "error": str(exc)}
            if isinstance(raw, dict) and "actionable" in raw:
                return raw
            # small models sometimes echo the scratchpad back instead of deciding
            log.warning("proactive reasoner returned no decision (attempt %d)", attempt)
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(raw, default=str)[:400]},
                {"role": "user", "content": prompts.REASONER_RETRY},
            ]
        return {"actionable": False, "error": "reasoner returned no decision"}

    async def known_types(self) -> list[dict]:
        rows = await self.app.store.fetchall("SELECT type_name, description FROM proactive_suggestion_types ORDER BY builtin DESC, type_name")
        return [dict(r) for r in rows]

    async def standardize_type(self, description: str) -> str:
        types = await self.known_types()
        names = [t["type_name"] for t in types]
        try:
            text = await self.app.llm.complete_text(
                "fast",
                [
                    {"role": "system", "content": prompts.SUGGESTION_TYPE_STANDARDIZER_SYSTEM},
                    {
                        "role": "user",
                        "content": f'Action Description:\n"{description}"\n\nAvailable Canonical Types:\n{json.dumps(types, indent=2)}',
                    },
                ],
            )
        except Exception as exc:
            log.warning("suggestion type standardization failed: %s", exc)
            return DEFAULT_TYPE
        cleaned = (text or "").strip().strip("`'\"").lower()
        found = next((n for n in names if re.search(rf"\b{re.escape(n)}\b", cleaned)), None)
        if found:
            return found
        m = re.search(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", cleaned)
        stype = m.group(0) if m else re.sub(r"[^a-z0-9]+", "_", " ".join(cleaned.split()[:5])).strip("_")
        stype = stype[:64] or DEFAULT_TYPE
        await self.app.store.execute(
            "INSERT OR IGNORE INTO proactive_suggestion_types(type_name, description, builtin, created_at) VALUES(?,?,0,?)",
            (stype, description[:300], _iso(datetime.now(UTC))),
        )
        return stype

    def threshold_for(self, score: int) -> float:
        return round(max(0.40, min(0.95, self.cfg.base_confidence_threshold - 0.05 * score)), 4)

    async def handle_reasoner_result(
        self, result: dict, *, source: str, event_type: str, item: dict, context: dict | None = None,
        now: datetime | None = None,
    ) -> dict | None:
        now = now or datetime.now(UTC)
        if not isinstance(result, dict) or not result.get("actionable"):
            return None
        description = " ".join(str(result.get("suggestion_description") or "").split())
        if not description:
            return None
        if any(similar(description, d) for d in await self._recent_descriptions(now, statuses=OPEN_STATUSES, limit=50)):
            log.info("suggestion %r duplicates one still open", description)
            return None
        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence_score") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        stype = await self.standardize_type(str(result.get("suggestion_type_description") or description))
        score = (await self.preference_scores()).get(stype, 0)
        threshold = self.threshold_for(score)
        if confidence < threshold:
            log.info("suggestion %s suppressed: confidence %.2f < threshold %.2f", stype, confidence, threshold)
            return None
        details = result.get("suggestion_action_details")
        kind = source_kind(source)
        source_event = {
            "source": source, "event_type": event_type, "summary": event_summary(item, kind),
            "item_id": str(item.get("id")) if item.get("id") else None,
        }
        if item.get("url"):
            source_event["url"] = item["url"]
        if source == "webhook" and item.get("name"):
            source_event["hook_name"] = str(item["name"])
        payload = {
            "suggestion": {
                "suggestion_type": stype,
                "description": description,
                "action_details": details if isinstance(details, dict) else {"action_type": stype},
                "reasoning": str(result.get("reasoning") or ""),
                "confidence": round(confidence, 3),
                "source_event": source_event,
            },
            "status": "pending",
            "task_id": None,
        }
        pruned = {
            "universal_search_results": (context or {}).get("universal_search_results"),
            "trigger_event": {"event_type": source, "event": event_type, "summary": source_event["summary"], "url": item.get("url")},
        }
        return await self._deliver(payload, pruned, threshold=threshold, now=now)

    def _quiet(self, now: datetime) -> bool:
        spec = self.cfg.quiet_hours
        return bool(spec) and in_quiet_hours(spec, now.astimezone(resolve_tz(self.app.config.assistant.timezone)))

    async def _deliver(self, payload: dict, context: dict, *, threshold: float, now: datetime) -> dict:
        s = payload["suggestion"]
        sid = new_id()
        record = {
            "id": sid, "notification_id": None, "status": "deferred", "threshold": threshold,
            "suggestion": s,
        }
        await self.app.store.execute(
            "INSERT INTO proactive_suggestions(id, suggestion_type, description, status, confidence, threshold, source,"
            " event_type, item_id, payload, context, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid, s["suggestion_type"], s["description"], "deferred", s["confidence"], threshold,
                s["source_event"]["source"], s["source_event"]["event_type"], s["source_event"].get("item_id"),
                json.dumps(payload, default=str), json.dumps(context, default=str)[:20_000], _iso(now),
            ),
        )
        if self._quiet(now):
            log.info("quiet hours: suggestion %s deferred", sid)
            return record
        return await self._notify(sid, payload, record)

    @staticmethod
    def notification_title(suggestion: dict) -> str:
        """'Gmail: Jane Doe: Meeting about Project Phoenix' says what the suggestion is about at a glance."""
        ev = suggestion.get("source_event") or {}
        source = str(ev.get("source") or "")
        label = SOURCE_LABELS.get(source, "Sentient")
        about = " ".join(str(ev.get("hook_name") or ev.get("summary") or "").split())
        if source == "heartbeat" or not about:
            return f"Suggestion from {label}"
        title = f"{label}: {about}"
        return title if len(title) <= 90 else title[:89].rstrip() + "…"

    async def _notify(self, sid: str, payload: dict, record: dict) -> dict:
        s = payload["suggestion"]
        message = followups.notification_message(s) if s.get("follow_up") else s["description"]
        note = await self.app.notify("proactive", message, title=self.notification_title(s), payload=payload)
        await self.app.store.execute(
            "UPDATE proactive_suggestions SET notification_id = ?, status = 'pending' WHERE id = ?", (note["id"], sid)
        )
        return {**record, "notification_id": note["id"], "status": "pending"}

    async def flush_deferred(self, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        if self._quiet(now):
            return 0
        rows = await self.app.store.fetchall(
            "SELECT id, payload, threshold, created_at FROM proactive_suggestions WHERE status = 'deferred' ORDER BY created_at"
        )
        sent = 0
        for r in rows:
            if r["created_at"] < _iso(now - timedelta(hours=24)):
                await self.app.store.execute("UPDATE proactive_suggestions SET status = 'expired' WHERE id = ?", (r["id"],))
                continue
            payload = json.loads(r["payload"])
            await self._notify(r["id"], payload, {"id": r["id"], "threshold": r["threshold"], "suggestion": payload["suggestion"]})
            sent += 1
        return sent

    async def expire_stale(self, now: datetime | None = None) -> int:
        """Expire pending suggestions nobody acted on: after ``suggestion_ttl_hours`` (check-ins after
        12 h at most), and calendar suggestions once their event has started."""
        now = now or datetime.now(UTC)
        store = self.app.store
        rows = await store.fetchall(
            "SELECT s.id, s.notification_id, s.source, s.created_at, seen.starts_at FROM proactive_suggestions s"
            " LEFT JOIN proactive_seen seen ON seen.source = s.source AND seen.item_id = s.item_id"
            " WHERE s.status = 'pending'"
        )
        ttl = self.cfg.suggestion_ttl_hours
        expired = 0
        for r in rows:
            hours = min(ttl, HEARTBEAT_TTL_HOURS) if r["source"] == "heartbeat" else ttl
            stale = r["created_at"] < _iso(now - timedelta(hours=hours))
            started = r["source"] == "gcalendar" and r["starts_at"] and r["starts_at"] <= _iso(now)
            if not (stale or started):
                continue
            await store.execute(
                "UPDATE proactive_suggestions SET status = 'expired', actioned_at = ? WHERE id = ?", (_iso(now), r["id"])
            )
            if r["notification_id"]:
                with contextlib.suppress(Exception):
                    note = await self.app.notifications.get(r["notification_id"])
                    if note is not None and note["payload"].get("status", "pending") == "pending":
                        await self.app.notifications.update_payload(r["notification_id"], {**note["payload"], "status": "expired"})
                        await self.app.notifications.mark_read(r["notification_id"])
            expired += 1
        return expired

    # ------------------------------------------------------------------ follow-ups (dropped email threads)
    async def _followups_due(self, now: datetime) -> bool:
        """Once a day, from FOLLOWUP_LOCAL_TIME in the user's timezone."""
        tz = resolve_tz(self.app.config.assistant.timezone)
        local = now.astimezone(tz)
        if local.time() < FOLLOWUP_LOCAL_TIME:
            return False
        raw = await self.app.store.get_meta(FOLLOWUP_META)
        last = followups.parse_when(raw)
        return last is None or last.astimezone(tz).date() < local.date()

    async def _followup_known(self, source: str, key: str) -> bool:
        """Already suggested (pending, approved, dismissed or expired) or already judged by the model."""
        if await self._already_suggested(source, key):
            return True
        row = await self.app.store.fetchone(
            "SELECT 1 AS x FROM proactive_seen WHERE source = ? AND item_id = ?", (FOLLOWUP_SEEN_SOURCE, f"{source}:{key}")
        )
        return row is not None

    async def _mark_followup_checked(self, c: followups.Candidate, now: datetime) -> None:
        await self.app.store.execute(
            "INSERT OR IGNORE INTO proactive_seen(source, item_id, event_type, item, starts_at, seen_at) VALUES(?,?,?,?,?,?)",
            (FOLLOWUP_SEEN_SOURCE, f"{c.source}:{c.key}", followups.EVENT_TYPE, None, None, _iso(now)),
        )

    async def followup_candidates(self, now: datetime | None = None) -> list[followups.Candidate]:
        """Threads from the email accounts in ``followups.sources`` that are connected, pass the deterministic
        filters and are new."""
        now = now or datetime.now(UTC)
        fu = self.cfg.followups
        fetch = getattr(self.app.integrations, "recent_threads", None)
        if not callable(fetch):
            return []
        out: list[followups.Candidate] = []
        wanted = {str(s).strip().lower() for s in fu.sources}
        for source in FOLLOWUP_SOURCES:
            if source not in wanted or not await self._is_connected(source):
                continue
            try:
                data = await _maybe_await(fetch(
                    source, newer_than_days=fu.max_age_days,
                    idle_days=min(fu.waiting_on_you_days, fu.waiting_on_them_days), limit=FOLLOWUP_SCAN_THREADS,
                ))
            except Exception as exc:  # one account failing never stops the other
                log.warning("follow-ups: reading %s failed: %s", source, exc)
                continue
            me = {str(a).lower() for a in (data or {}).get("addresses") or []}
            if redact.enabled(self.app):
                for thread in (data or {}).get("threads") or []:
                    thread["messages"] = [redact.hide_email_secrets(dict(m)) for m in thread.get("messages") or []]
            user = await self._user_email(source)
            if user:
                me.add(user.lower())
            for thread in (data or {}).get("threads") or []:
                c, why = followups.classify(source, thread, me, fu, now)
                if c is None:
                    log.debug("follow-ups: skipped %s thread %s: %s", source, thread.get("thread_id"), why)
                    continue
                if await self._followup_known(source, c.key):
                    continue
                out.append(c)
        out.sort(key=lambda c: (c.kind != followups.WAITING_ON_YOU, -followups.parse_when(c.last["date"]).timestamp()))  # type: ignore[union-attr]
        return out

    async def run_followups(self, *, now: datetime | None = None) -> list[dict]:
        """One follow-up check: deterministic filters, then the fast model per candidate (at most
        FOLLOWUP_MAX_CANDIDATES), at most ``followups.max_suggestions`` suggestions. Never sends anything."""
        now = now or datetime.now(UTC)
        fu = self.cfg.followups
        if not (self.cfg.enabled and fu.enabled) or self._followup_lock.locked():
            return []
        async with self._followup_lock:
            await self.app.store.set_meta(FOLLOWUP_META, now.isoformat())
            delivered: list[dict] = []
            for c in (await self.followup_candidates(now))[:FOLLOWUP_MAX_CANDIDATES]:
                if len(delivered) >= fu.max_suggestions:
                    break
                try:
                    record = await self._follow_up(c, now)
                except Exception:
                    log.exception("follow-up for %s thread %s failed", c.source, c.key)
                    continue
                if record is not None:
                    delivered.append(record)
            return delivered

    async def _follow_up(self, c: followups.Candidate, now: datetime) -> dict | None:
        messages = followups.build_messages(
            c, self.app.config.assistant.user_name, prompts.FOLLOW_UP_SYSTEM,
            prompts.FOLLOW_UP_WAITING_ON_YOU, prompts.FOLLOW_UP_WAITING_ON_THEM,
        )
        try:
            text = await self.app.llm.complete_text("fast", messages)
        except Exception as exc:
            log.warning("follow-up model call failed: %s", exc)
            return None
        decision = followups.parse_decision(text)
        if decision is None:
            log.info("follow-up for %s %s: no usable answer from the model, trying again next check", c.source, c.key)
            return None
        if decision.needed and followups.has_placeholder(decision.draft):
            # the draft would be sent exactly as written: drop it now, ask again next check
            log.info("follow-up for %s %s: the draft has a placeholder, trying again next check", c.source, c.key)
            return None
        await self._mark_followup_checked(c, now)
        if not decision.needed:
            log.info("follow-up for %s %s: the model says no follow-up is needed", c.source, c.key)
            return None
        stype = followups.TYPE_FOR[c.kind]
        threshold = self.threshold_for((await self.preference_scores()).get(stype, 0))
        if decision.confidence < threshold:
            log.info("follow-up %s suppressed: confidence %.2f < threshold %.2f", stype, decision.confidence, threshold)
            return None
        payload = {
            "suggestion": followups.suggestion_for(c, decision, confidence=decision.confidence),
            "status": "pending",
            "task_id": None,
        }
        context = {"follow_up": {"kind": c.kind, "thread_id": c.thread.get("thread_id"), "days_waiting": c.days,
                                 "last_message_at": c.last.get("date")}}
        return await self._deliver(payload, context, threshold=threshold, now=now)

    # ------------------------------------------------------------------ feedback & preferences
    async def preference_scores(self) -> dict[str, int]:
        rows = await self.app.store.fetchall("SELECT suggestion_type, score FROM proactive_preferences")
        return {r["suggestion_type"]: int(r["score"]) for r in rows}

    async def record_feedback(self, suggestion_type: str, positive: bool) -> None:
        """v2 learning: +1 on approve, -1 on dismiss."""
        await self.app.store.execute(
            "INSERT INTO proactive_preferences(suggestion_type, score, approvals, dismissals, updated_at) VALUES(?,?,?,?,?)"
            " ON CONFLICT(suggestion_type) DO UPDATE SET score = score + excluded.score,"
            " approvals = approvals + excluded.approvals, dismissals = dismissals + excluded.dismissals,"
            " updated_at = excluded.updated_at",
            (suggestion_type, 1 if positive else -1, int(positive), int(not positive), _iso(datetime.now(UTC))),
        )

    async def preferences(self) -> list[dict]:
        rows = await self.app.store.fetchall(
            "SELECT suggestion_type, score, approvals, dismissals FROM proactive_preferences ORDER BY suggestion_type"
        )
        return [
            {
                "suggestion_type": r["suggestion_type"], "score": int(r["score"]),
                "threshold": self.threshold_for(int(r["score"])),
                "approvals": int(r["approvals"]), "dismissals": int(r["dismissals"]),
            }
            for r in rows
        ]

    async def reset_preference(self, suggestion_type: str) -> bool:
        cur = await self.app.store.execute(
            "DELETE FROM proactive_preferences WHERE suggestion_type = ?", (suggestion_type,)
        )
        return cur.rowcount > 0

    @staticmethod
    def task_prompt(suggestion: dict) -> str:
        """v2 create_task_from_suggestion: name from the description, body from details + reasoning + trigger."""
        ev =suggestion.get("source_event") or {}
        parts = [
            str(suggestion.get("description") or "Proactive task"),
            "Action details:\n" + json.dumps(suggestion.get("action_details") or {}, indent=2, default=str),
        ]
        if suggestion.get("reasoning"):
            parts.append(f"Why this was suggested: {suggestion['reasoning']}")
        if ev.get("summary"):
            parts.append(f"Triggered by {ev.get('source')} {ev.get('event_type')}: {ev['summary']}" + (f" ({ev['url']})" if ev.get("url") else ""))
        return "\n\n".join(parts)

    async def act_on_suggestion(self, notification_id: str, action: str) -> dict:
        action = action.strip().lower()
        action = {"approved": "approve", "dismissed": "dismiss"}.get(action, action)  # v2 spelling
        if action not in {"approve", "dismiss"}:
            raise SuggestionError(400, "action must be approve or dismiss")
        note = await self.app.notifications.get(notification_id)
        if note is None or note["kind"] != "proactive" or not isinstance(note["payload"].get("suggestion"), dict):
            raise SuggestionError(404, "suggestion not found")
        payload = note["payload"]
        if payload.get("status", "pending") != "pending":
            raise SuggestionError(409, f"suggestion already {payload.get('status')}")
        # claim it before the first await: two clicks at once (window and Telegram) must not send twice
        if notification_id in self._acting:
            raise SuggestionError(409, "suggestion already being handled")
        self._acting.add(notification_id)
        try:
            return await self._act_on_suggestion(notification_id, note, payload, action)
        finally:
            self._acting.discard(notification_id)

    async def _act_on_suggestion(self, notification_id: str, note: dict, payload: dict, action: str) -> dict:
        suggestion = payload["suggestion"]
        stype = suggestion.get("suggestion_type") or DEFAULT_TYPE
        task_id = None
        if action == "approve":
            tasks = self.app.tasks
            context = {
                "source": "proactive", "suggestion_type": stype, "notification_id": notification_id,
                "trigger_event": suggestion.get("source_event"),
            }
            if isinstance(suggestion.get("follow_up"), dict):
                # "Send reply" showed the exact draft: that click approves this one exact call, so the task runs
                # now with the arguments fixed (no plan to approve, no model in between)
                if not hasattr(tasks, "create_approved_call"):
                    raise SuggestionError(503, "tasks are not available yet")
                call = followups.send_call(suggestion)
                try:
                    created = await _maybe_await(tasks.create_approved_call(
                        call["name"], call["tool"], call["arguments"], step=call["step"],
                        description=call["description"], source="proactive", original_context=context,
                        done_text=call["done_text"],
                    ))
                except ValueError as exc:
                    raise SuggestionError(503, f"Sending isn't available right now ({exc}).") from exc
            elif not hasattr(tasks, "create_task"):
                raise SuggestionError(503, "tasks are not available yet")
            else:
                created = await _maybe_await(
                    tasks.create_task(self.task_prompt(suggestion), source="proactive", original_context=context)
                )
            if isinstance(created, dict):
                task_id = created.get("task_id") or created.get("id")
            elif isinstance(created, str):
                task_id = created
            else:
                task_id = getattr(created, "task_id", None)
        payload = {**payload, "status": "approved" if action == "approve" else "dismissed", "task_id": task_id}
        await self.app.notifications.update_payload(notification_id, payload)
        await self.app.notifications.mark_read(notification_id)
        await self.app.store.execute(
            "UPDATE proactive_suggestions SET status = ?, task_id = ?, actioned_at = ? WHERE notification_id = ?",
            (payload["status"], task_id, _iso(datetime.now(UTC)), notification_id),
        )
        await self.record_feedback(stype, action == "approve")
        return {"ok": True, "task_id": task_id} if action == "approve" else {"ok": True}

    # ------------------------------------------------------------------ status
    async def status(self) -> dict:
        now = datetime.now(UTC)
        rows = {
            r["source"]: dict(r)
            for r in await self.app.store.fetchall("SELECT * FROM proactive_sources")
        }
        sources = []
        for source in self.cfg.sources:
            r = rows.get(source) or {}
            sources.append(
                {
                    "source": source, "connected": await self._is_connected(source),
                    "last_poll_at": r.get("last_poll_at"), "last_error": r.get("last_error"),
                    "feed_active": await self._feed_active(source),
                }
            )
        n = await self.app.store.fetchone(
            "SELECT COUNT(*) AS n FROM proactive_suggestions WHERE created_at >= ? AND status != 'expired'",
            (self._local_midnight_iso(now),),
        )
        return {
            "enabled": self.cfg.enabled,
            "last_poll_at": {s["source"]: s["last_poll_at"] for s in sources},
            "sources": sources,
            "suggestions_today": int(n["n"]) if n else 0,
            "quiet_now": self._quiet(now),
            "heartbeat_minutes": self.cfg.heartbeat_minutes,
            "followups": {
                "enabled": bool(self.cfg.enabled and self.cfg.followups.enabled),
                "last_run_at": await self.app.store.get_meta(FOLLOWUP_META),
            },
        }

    def _local_midnight_iso(self, now: datetime) -> str:
        tz = resolve_tz(self.app.config.assistant.timezone)
        return _iso(now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0))

    # ------------------------------------------------------------------ heartbeat
    async def heartbeats_today(self, now: datetime) -> int:
        row = await self.app.store.fetchone(
            "SELECT COUNT(*) AS n FROM proactive_suggestions WHERE source = 'heartbeat' AND created_at >= ?",
            (self._local_midnight_iso(now),),
        )
        return int(row["n"]) if row else 0

    async def _upcoming_events(self, now: datetime, hours: int = 12, limit: int = 8) -> list[dict]:
        rows = await self.app.store.fetchall(
            "SELECT item FROM proactive_seen WHERE source = 'gcalendar' AND starts_at >= ? AND starts_at <= ? ORDER BY starts_at LIMIT ?",
            (_iso(now), _iso(now + timedelta(hours=hours)), limit),
        )
        upcoming = []
        for r in rows:
            with contextlib.suppress(Exception):
                it = json.loads(r["item"])
                if str(it.get("status", "")).lower() != "cancelled":
                    keys = ("summary", "start", "end", "all_day", "location", "attendees", "meet_link")
                    ev = {k: it.get(k) for k in keys if it.get(k)}
                    if isinstance(ev.get("attendees"), list):
                        ev["attendees"] = [
                            (a.get("email") if isinstance(a, dict) else str(a)) for a in ev["attendees"][:6]
                        ]
                    upcoming.append(ev)
        return upcoming

    async def heartbeat(self, now: datetime | None = None) -> dict | None:
        """Periodic check-in with no trigger (OpenClaw-style). Gathers a compact picture of the user's day;
        the model answers NO_REPLY unless one nudge is worth an interruption. Skipped during quiet hours
        (nudges are about now, so they are not held) and after ``heartbeat_daily_cap`` check-in suggestions."""
        now = now or datetime.now(UTC)
        if self._quiet(now):
            return None
        if await self.heartbeats_today(now) >= self.cfg.heartbeat_daily_cap:
            return None
        upcoming = await self._upcoming_events(now)
        attention = [
            {"name": t.get("name"), "status": t.get("status"), "description": (t.get("description") or "")[:200]}
            for t in await self._task_list()
            if t.get("status") in ATTENTION_STATUSES
        ][:8]
        expiring = []
        if self.app.memory is not None:
            with contextlib.suppress(Exception):
                expiring = [
                    {"fact": f["content"], "expires_at": f["expires_at"]}
                    for f in await self.app.memory.expiring_soon(timedelta(days=1))
                ][:8]
        if not (upcoming or attention or expiring):
            return None
        recent_desc = await self._recent_descriptions(now)
        situation: dict[str, Any] = {
            "user": self._user_block(now),
            "upcoming_events_next_12h": upcoming,
            "tasks_needing_attention": attention,
            "short_term_memories_expiring_within_24h": expiring,
            "suggestions_already_made_last_24h": recent_desc,
            "user_preferences": await self.preference_scores(),
        }
        focus = " ".join(
            [str(e.get("summary") or "") for e in upcoming] + [str(t.get("name") or "") for t in attention]
        )
        about = await self._user_model_context(f"What matters to the user this {part_of_day(now.astimezone(resolve_tz(self.app.config.assistant.timezone)))}: {focus}")
        if about:
            situation["about_the_user"] = about
        try:
            text = await self.app.llm.complete_text(
                "fast",
                [
                    {"role": "system", "content": prompts.HEARTBEAT_SYSTEM},
                    {"role": "user", "content": json.dumps(situation, default=str, ensure_ascii=False)},
                ],
            )
        except Exception as exc:
            log.warning("heartbeat failed: %s", exc)
            return None
        text = (text or "").strip()
        if not text or text.strip("`\"' .").upper().startswith("NO_REPLY") or (len(text) < 40 and "NO_REPLY" in text.upper()):
            return None
        try:
            parsed = parse_json_loose(text)
        except ValueError:
            return None
        if not isinstance(parsed, dict) or parsed.get("actionable") is False:
            return None
        parsed.setdefault("actionable", True)
        desc = " ".join(str(parsed.get("suggestion_description") or "").split())
        if not desc or any(similar(desc, d) for d in recent_desc):
            return None
        return await self.handle_reasoner_result(
            parsed, source="heartbeat", event_type="heartbeat",
            item={"id": f"heartbeat:{now.strftime('%Y%m%d%H%M')}", "summary": "Periodic check-in"},
            context={"universal_search_results": situation}, now=now,
        )
