"""The Daily Brief (docs/API.md section 6): a short, capped morning digest built from read-only look-ups, and an
optional Evening Brief that wraps up the day.

Each is an ordinary recurring task the user can see, edit, pause or delete: setting one up creates an already approved
task whose run is one fixed call to ``daily_brief_build`` (``app.tasks.create_approved_call`` with a recurring
schedule). Each run:

    1. reads, for the morning: today's calendar, emails that need you (pending email suggestions and follow-ups, then
       unread important Gmail), tasks due today or waiting for you, the weather at the user's location and news on
       chosen topics; for the evening: tasks finished or failed today, emails sent today, files made today, what is
       still waiting for you and tomorrow's first events.
       Only ``read`` tools are called, never ones behind an Ask or Never rule; nothing is sent or changed.
    2. ranks sections by the learned per-type scores (``daily_brief_<section>``) and caps the brief at
       ``proactivity.brief.max_items`` lines, taking one line from each section in turn.
    3. lets the ``fast`` model shorten unread emails to one line each (tolerant parsing; the sender and subject are
       used when the answer is unusable). The model never adds or removes items.
    4. delivers it as a ``brief`` notification (paired chats get it too, ``channels.deliver_briefs``) that expires at
       the end of the user's day.

Thumbs up and down on an item or a section feed ``ProactiveEngine.record_feedback`` with the section's type, so later
briefs give that section more or fewer lines and move it up or down.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.builtin.time_tool import resolve_tz

if TYPE_CHECKING:  # pragma: no cover
    from sentient.proactivity.service import ProactiveEngine

log = logging.getLogger(__name__)

MORNING, EVENING = "morning", "evening"
KINDS = (MORNING, EVENING)
SECTIONS = ("calendar", "email", "tasks", "weather", "news")
EVENING_SECTIONS = ("done", "sent", "files", "waiting", "tomorrow")
KIND_SECTIONS = {MORNING: SECTIONS, EVENING: EVENING_SECTIONS}
SECTION_LABELS = {"calendar": "Calendar", "email": "Email", "tasks": "Tasks", "weather": "Weather", "news": "News",
                  "done": "Done today", "sent": "Sent today", "files": "New files", "waiting": "Still waiting for you",
                  "tomorrow": "Tomorrow"}
BASE_LIMITS = {"calendar": 3, "email": 3, "tasks": 3, "weather": 1, "news": 2,
               "done": 3, "sent": 2, "files": 2, "waiting": 3, "tomorrow": 3}
TYPE_PREFIX = "daily_brief_"  # learned preference type per section: daily_brief_calendar, ...
BUILD_TOOL = "daily_brief_build"
READ_TOOL = "daily_brief_today"
TASK_META = {MORNING: "brief.task_id", EVENING: "brief.evening_task_id"}
TASK_NAMES = {MORNING: "Daily Brief", EVENING: "Evening Brief"}
TASK_DESCRIPTIONS = {
    MORNING: "Gathers today's calendar, emails that need you, tasks due or waiting for you and the weather into one "
             "short brief. It only reads: it never sends or changes anything.",
    EVENING: "Wraps up your day: tasks finished or failed, emails sent, files made, what is still waiting for you and "
             "tomorrow's first events. It only reads: it never sends or changes anything.",
}
TITLES = {MORNING: "Your Daily Brief for {day}", EVENING: "Your Evening Brief for {day}"}
EMPTY_TEXT = {MORNING: "Nothing needs you this morning. Enjoy your day.",
              EVENING: "A quiet day. Nothing is waiting for you tonight."}
DEFAULT_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ALL_DAYS = [*DEFAULT_DAYS, "Saturday", "Sunday"]
DEFAULT_SCHEDULES: dict[str, tuple[str, Any]] = {MORNING: ("07:30", DEFAULT_DAYS), EVENING: ("21:00", "daily")}
DEFAULT_TIME = DEFAULT_SCHEDULES[MORNING][0]
TIME_WORDS = {"early": "06:30", "morning": "07:30", "midday": "12:00", "noon": "12:00", "afternoon": "14:00",
              "evening": "18:00", "night": "21:00"}
SEND_TOOLS = {"gmail_send", "gmail_reply", "email_imap_send"}
FINISHED_OK = {"completed", "completed_with_errors"}
TOMORROW_EVENTS = 3
FILES_SCAN_LIMIT = 500
WAITING_LABELS = {  # most urgent first
    "waiting_for_user": "asked you a question",
    "approval_pending": "plan waiting for your approval",
    "clarification_pending": "waiting for your answers",
    "error": "failed last time",
}
DUE_STATUSES = {"active", "pending"}
EMAIL_SOURCES = ("gmail", "email_imap")
UNREAD_QUERY = "is:unread is:important newer_than:2d -category:promotions -category:social"
MAX_LINE = 140
SUMMARY_SYSTEM = (
    "You write one short line per email for a morning brief. Use only what the email says. "
    "No greetings, no advice, at most 14 words per line."
)
_NUMBERED = re.compile(r"^\s*(\d{1,2})\s*[.):-]\s*(.+?)\s*$")
_CLOCK = re.compile(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$")


class BriefError(Exception):
    """Raised by the brief API; ``status`` is the HTTP status the route returns."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def preference_type(section: str) -> str:
    return f"{TYPE_PREFIX}{section}"


def section_limit(section: str, score: int) -> int:
    """Lines a section may take: liked sections get one more, disliked ones one fewer (never below one)."""
    base = BASE_LIMITS.get(section, 2)
    if score >= 3:
        return base + 1
    if score <= -3:
        return max(1, base - 1)
    return base


def order_sections(sections: list[str], scores: dict[str, int], known: tuple[str, ...] = SECTIONS) -> list[str]:
    """Wanted sections out of ``known``, best liked first; ties keep the usual order."""
    wanted = [s for s in known if s in {str(x).strip().lower() for x in sections}]
    return sorted(wanted, key=lambda s: (-scores.get(preference_type(s), 0), known.index(s)))


def check_kind(kind: Any) -> str:
    value = str(kind or MORNING).strip().lower()
    if value not in KINDS:
        raise BriefError(400, "kind must be morning or evening")
    return value


def cap_items(found: dict[str, list[dict]], order: list[str], scores: dict[str, int], max_items: int) -> list[dict]:
    """At most ``max_items`` lines: first one from every section, then the rest in ``order`` (best liked first),
    each section up to its limit. The result is grouped by section in ``order``."""
    room = {s: min(len(found.get(s) or []), section_limit(s, scores.get(preference_type(s), 0))) for s in order}
    picked: dict[str, int] = dict.fromkeys(order, 0)
    total = 0
    for s in order:
        if total < max_items and room[s]:
            picked[s], total = 1, total + 1
    for s in order:
        extra = min(room[s] - picked[s], max_items - total)
        if extra > 0:
            picked[s] += extra
            total += extra
    return [item for s in order for item in (found.get(s) or [])[: picked[s]]]


def item_id(section: str, key: str) -> str:
    return f"{section}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def one_line(text: Any, limit: int = MAX_LINE) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def parse_summaries(text: str, count: int) -> dict[int, str]:
    """``{index: line}`` from a numbered reply ('1. ...'). Lines that are empty, too long or look like code are left out."""
    out: dict[int, str] = {}
    for raw in str(text or "").splitlines():
        m = _NUMBERED.match(raw.strip().strip("*`"))
        if not m:
            continue
        n = int(m.group(1))
        line = m.group(2).strip().strip("\"'*` ").strip()
        if not 1 <= n <= count or n in out or not 3 <= len(line) <= MAX_LINE or line.startswith(("{", "[")):
            continue
        out[n] = line
    return out


def sender_name(value: Any) -> str:
    text = str(value or "").strip()
    m = re.match(r'^\s*"?([^"<]+?)"?\s*<[^>]+>\s*$', text)
    if m:
        return m.group(1).strip()
    return text.split("@")[0] if "@" in text else text


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def schedule_for(time_value: Any = None, days: Any = None, tz_name: str | None = None, kind: str = MORNING) -> dict:
    """A recurring task schedule. ``time_value`` is 'HH:MM' or a word like 'morning'; ``days`` a list of day names,
    'weekdays' or 'daily'. Left out, they default to the brief kind's (weekdays at 07:30, or daily at 21:00)."""
    default_time, default_days = DEFAULT_SCHEDULES[kind]
    when = str(time_value or default_time).strip().lower()
    when = TIME_WORDS.get(when, when)
    if not _CLOCK.match(when):
        raise ValueError("Pick a time like 07:30, or a word: early, morning, midday, afternoon, evening or night.")
    if days is None or days == []:
        days = default_days
    if isinstance(days, str) and days.strip().lower() in {"weekday", "weekdays"}:
        day_list: Any = list(DEFAULT_DAYS)
    elif isinstance(days, str) and days.strip().lower() in {"daily", "every day", "everyday"}:
        day_list = None
    else:
        day_list = list(days) if isinstance(days, list) else days
    out: dict[str, Any] = {"type": "recurring", "time": when}
    if day_list is None:
        out["frequency"] = "daily"
    else:
        out.update(frequency="weekly", days=day_list)
    if tz_name:
        out["timezone"] = tz_name
    return out


class DailyBrief:
    """Builds, delivers, steers and expires Daily Briefs. One per app, owned by ``ProactiveEngine`` (``.brief``)."""

    def __init__(self, engine: ProactiveEngine):
        self.engine = engine
        self.app = engine.app

    @property
    def cfg(self):
        return self.app.config.proactivity.brief

    def _tz(self):
        return resolve_tz(self.app.config.assistant.timezone) or UTC

    # ------------------------------------------------------------------ the task
    async def task(self, kind: str = MORNING) -> dict | None:
        """The brief's task (morning or evening), or None when it was never set up or was deleted."""
        tasks = self.app.tasks
        task_id = await self.app.store.get_meta(TASK_META[kind])
        if task_id and hasattr(tasks, "get"):
            with contextlib.suppress(Exception):
                return await tasks.get(task_id)
        return None

    async def _brief_task_ids(self) -> set[str]:
        return {t for t in [await self.app.store.get_meta(TASK_META[k]) for k in KINDS] if t}

    async def setup(self, body: dict | None = None) -> dict:
        """Create a brief's task (once) or change it: ``{kind, time, days, sections, news_topics, max_items}``, all
        optional; ``kind`` is morning (default) or evening."""
        body = dict(body or {})
        kind = check_kind(body.get("kind"))
        self._apply_config(body, kind)
        tasks = self.app.tasks
        existing = await self.task(kind)
        if existing is None:
            if not hasattr(tasks, "create_approved_call"):
                raise BriefError(503, "Tasks are not available yet.")
            step = ("Build today's brief from your calendar, email, tasks and weather" if kind == MORNING
                    else "Wrap up today: finished tasks, sent emails, new files, what is waiting and tomorrow's events")
            created = await tasks.create_approved_call(
                TASK_NAMES[kind], BUILD_TOOL, {"kind": kind}, step=step,
                description=TASK_DESCRIPTIONS[kind], source="brief",
                original_context={"source": "brief", "brief": kind},
                done_text=f"Your {TASK_NAMES[kind]} is ready.",
                schedule=schedule_for(body.get("time"), body.get("days"), kind=kind), quiet=True,
            )
            await self.app.store.set_meta(TASK_META[kind], created["task_id"])
        elif "time" in body or "days" in body:
            old = existing.get("schedule") or {}
            schedule = schedule_for(
                body.get("time") or old.get("time"),
                body.get("days") if "days" in body else ("daily" if old.get("frequency") == "daily" else old.get("days")),
                old.get("timezone"), kind=kind,
            )
            await tasks.update(existing["task_id"], {"schedule": schedule})
        return await self.state()

    def _apply_config(self, body: dict, kind: str = MORNING) -> None:
        keys = ("sections", "news_topics", "max_items")
        if not any(k in body for k in keys):
            return
        cfg = self.app.config.model_copy(deep=True)
        brief = cfg.proactivity.brief
        if "sections" in body:
            raw = body["sections"] if isinstance(body["sections"], list) else []
            picked = [s for s in KIND_SECTIONS[kind] if s in {str(x).strip().lower() for x in raw}]
            if kind == EVENING:
                brief.evening_sections = picked
            else:
                brief.sections = picked
        if "news_topics" in body:
            raw = body["news_topics"] if isinstance(body["news_topics"], list) else []
            brief.news_topics = [one_line(t, 60) for t in raw if str(t).strip()][:5]
        if "max_items" in body:
            try:
                brief.max_items = max(1, min(20, int(body["max_items"])))
            except (TypeError, ValueError) as exc:
                raise BriefError(400, "max_items must be a number from 1 to 20.") from exc
        self.app.save_config(cfg)

    def _sections(self, kind: str) -> list[str]:
        return list(self.cfg.evening_sections if kind == EVENING else self.cfg.sections)

    async def run_now(self, kind: str = MORNING) -> dict:
        kind = check_kind(kind)
        task = await self.task(kind)
        if task is None:
            raise BriefError(409, f"Set up your {TASK_NAMES[kind]} first.")
        try:
            await self.app.tasks.run_now(task["task_id"])
        except ValueError as exc:  # TaskConflict: the task can't run right now
            raise BriefError(409, str(exc)) from exc
        return {"ok": True, "task_id": task["task_id"]}

    async def _kind_state(self, kind: str) -> dict:
        task = await self.task(kind)
        schedule = (task or {}).get("schedule") or {}
        return {
            "set_up": task is not None,
            "task_id": (task or {}).get("task_id"),
            "enabled": bool(task and task.get("enabled")),
            "time": schedule.get("time"),
            "days": ALL_DAYS if schedule.get("frequency") == "daily" else schedule.get("days"),
            "next_at": (task or {}).get("next_execution_at"),
            "sections": self._sections(kind),
        }

    async def state(self, now: datetime | None = None) -> dict:
        """The morning brief's fields at the top level, the evening brief's under ``evening``."""
        return {
            **await self._kind_state(MORNING),
            "news_topics": list(self.cfg.news_topics),
            "max_items": self.cfg.max_items,
            "available": await self.available(),
            "today": await self.today(now),
            "evening": await self._kind_state(EVENING),
        }

    async def available(self) -> dict[str, bool]:
        """Which sections can find anything right now (an app connected, a city or topics set)."""
        calendar = await self._connected("gcalendar")
        return {
            "calendar": calendar,
            "email": any([await self._connected(s) for s in EMAIL_SOURCES]),
            "tasks": True,
            "weather": bool(self.app.config.assistant.location.strip()),
            "news": bool(self.cfg.news_topics),
            "done": True,
            "sent": True,
            "files": True,
            "waiting": True,
            "tomorrow": calendar,
        }

    # ------------------------------------------------------------------ reading (read tools only)
    async def _connected(self, plugin_id: str) -> bool:
        return await self.engine._is_connected(plugin_id)

    def _ctx(self) -> ToolContext | None:
        agent = self.app.agent
        return agent.tool_context(None, "proactive", origin="proactive") if agent is not None else None

    async def read(self, name: str, arguments: dict) -> dict | None:
        """Call one read-only tool. Anything that is not ``read``, or is behind an Ask or Never rule, is never called."""
        t = self.app.registry.get(name)
        if t is None or t.risk != Risk.read:
            return None
        with contextlib.suppress(Exception):
            if self.app.approvals.rule(t) in {"ask", "never"}:
                return None
        ctx = self._ctx()
        if ctx is None:
            return None
        try:
            out = await t.call(ctx, arguments)
        except Exception as exc:
            log.warning("brief: %s failed: %s", name, exc)
            return None
        if not isinstance(out, dict) or out.get("error"):
            log.info("brief: %s gave nothing usable: %s", name, (out or {}).get("error") if isinstance(out, dict) else out)
            return None
        return out

    async def _events(self, local: datetime, section: str, *, tomorrow: bool = False) -> tuple[list[dict], str | None]:
        """Today's events that are not over yet, or tomorrow's (``section`` names the brief section)."""
        when_word = "tomorrow's first" if tomorrow else "today's"
        if not await self._connected("gcalendar"):
            return [], f"Connect Google Calendar to see {when_word} events."
        day = (local.date() + timedelta(days=1 if tomorrow else 0)).isoformat()
        out = await self.read("gcal_list_events", {"time_min": day, "time_max": day, "max_results": 20})
        if out is None:
            return [], "Couldn't read your calendar this time."
        items = []
        for ev in out.get("events") or []:
            if not isinstance(ev, dict) or str(ev.get("status") or "").lower() == "cancelled":
                continue
            start, end = parse_dt(ev.get("start")), parse_dt(ev.get("end"))
            if not tomorrow and not ev.get("all_day") and end is not None and end <= local:
                continue  # already over
            when = "All day" if ev.get("all_day") or start is None else start.astimezone(local.tzinfo).strftime("%H:%M")
            text = f"{when} {one_line(ev.get('summary') or 'Untitled event', 100)}"
            if ev.get("location") and len(str(ev["location"])) <= 40:
                text += f" ({one_line(ev['location'], 40)})"
            items.append({
                "id": item_id(section, str(ev.get("id") or text)), "section": section, "text": text,
                "link": ev.get("url") or ev.get("meet_link"),
                "why": "First on your calendar tomorrow" if tomorrow else "On your calendar today",
            })
        return (items[:TOMORROW_EVENTS] if tomorrow else items), None

    async def _calendar(self, local: datetime) -> tuple[list[dict], str | None]:
        return await self._events(local, "calendar")

    async def _tomorrow(self, local: datetime) -> tuple[list[dict], str | None]:
        return await self._events(local, "tomorrow", tomorrow=True)

    async def _pending_email_suggestions(self, now: datetime) -> list[dict]:
        rows = await self.app.store.fetchall(
            "SELECT notification_id, payload, item_id FROM proactive_suggestions WHERE status = 'pending'"
            f" AND source IN ({','.join('?' * len(EMAIL_SOURCES))}) ORDER BY created_at DESC LIMIT 10",
            EMAIL_SOURCES,
        )
        items = []
        for r in rows:
            with contextlib.suppress(Exception):
                s = json.loads(r["payload"])["suggestion"]
                ev = s.get("source_event") or {}
                fu = s.get("follow_up") if isinstance(s.get("follow_up"), dict) else None
                if fu:
                    why = f"No reply for {fu.get('days_waiting')} days" if fu.get("days_waiting") else "Waiting on a reply"
                else:
                    why = "Sentient has a suggestion for this email"
                items.append({
                    "id": item_id("email", f"s:{r['notification_id']}"), "section": "email",
                    "text": one_line(s.get("description")), "link": ev.get("url"), "why": why,
                    "notification_id": r["notification_id"],
                    "_keys": {str(r["item_id"] or ""), str(fu.get("thread_id") if fu else ""), str(fu.get("message_id") if fu else "")},
                })
        return items

    async def _email(self, local: datetime) -> tuple[list[dict], str | None]:
        connected = [s for s in EMAIL_SOURCES if await self._connected(s)]
        if not connected:
            return [], "Connect Gmail or email to see messages that need you."
        items = await self._pending_email_suggestions(local)
        known = set().union(*(i.pop("_keys") for i in items)) if items else set()
        if "gmail" in connected:
            out = await self.read("gmail_search", {"query": UNREAD_QUERY, "max_results": 5})
            for m in (out or {}).get("messages") or []:
                if not isinstance(m, dict) or {str(m.get("id")), str(m.get("thread_id"))} & known:
                    continue
                name = sender_name(m.get("from")) or "Someone"
                items.append({
                    "id": item_id("email", f"m:{m.get('thread_id') or m.get('id')}"), "section": "email",
                    "text": one_line(f"{name}: {m.get('subject') or '(no subject)'}"), "link": m.get("url"),
                    "why": "Unread and marked important in Gmail",
                    "_mail": {"from": name, "subject": m.get("subject") or "", "snippet": one_line(m.get("snippet"), 300)},
                })
        return items, None

    async def _task_list(self) -> list[dict] | None:
        """Every task except the briefs' own, or None when tasks can't be read."""
        lister = getattr(self.app.tasks, "list", None)
        if not callable(lister):
            return []
        try:
            tasks = await lister()
        except Exception as exc:
            log.warning("brief: listing tasks failed: %s", exc)
            return None
        own = await self._brief_task_ids()
        return [t for t in tasks or [] if isinstance(t, dict) and t.get("task_id") not in own]

    @staticmethod
    def _waiting_tasks(tasks: list[dict], section: str) -> list[dict]:
        """Tasks waiting for the user, most urgent first."""
        urgency = list(WAITING_LABELS)
        waiting = [t for t in tasks if t.get("status") in WAITING_LABELS]
        waiting.sort(key=lambda t: urgency.index(t["status"]))
        return [
            {"id": item_id(section, f"w:{t.get('task_id')}:{t['status']}"), "section": section,
             "text": f"{one_line(t.get('name') or 'Untitled task', 90)}: {WAITING_LABELS[t['status']]}",
             "link": f"/tasks/{t.get('task_id')}", "why": "Waiting for you in Tasks"}
            for t in waiting
        ]

    async def _tasks(self, local: datetime) -> tuple[list[dict], str | None]:
        tasks = await self._task_list()
        if tasks is None:
            return [], "Couldn't read your tasks this time."
        end_of_day = datetime.combine(local.date() + timedelta(days=1), time.min, local.tzinfo)
        due = []
        for t in tasks:
            nxt = parse_dt(t.get("next_execution_at"))
            if t.get("enabled") and t.get("status") in DUE_STATUSES and nxt is not None and local <= nxt < end_of_day:
                name = one_line(t.get("name") or "Untitled task", 90)
                due.append((nxt, {"id": item_id("tasks", f"d:{t.get('task_id')}"), "section": "tasks",
                                  "text": f"{name} runs at {nxt.astimezone(local.tzinfo).strftime('%H:%M')}",
                                  "link": f"/tasks/{t.get('task_id')}", "why": "Due today"}))
        due.sort(key=lambda x: x[0])
        return self._waiting_tasks(tasks, "tasks") + [d for _, d in due], None

    # ------------------------------------------------------------------ evening sections
    @staticmethod
    def _runs_today(task: dict, local: datetime) -> list[dict]:
        """The task's runs that finished today (user's day), newest first."""
        out = []
        for run in task.get("runs") or []:
            done = parse_dt(run.get("finished_at"))
            if done is not None and done.astimezone(local.tzinfo).date() == local.date():
                out.append(run)
        return sorted(out, key=lambda r: str(r.get("finished_at") or ""), reverse=True)

    @staticmethod
    def _sends(task: dict, run: dict) -> bool:
        """True when the run sent an email (an approved follow-up, or a send tool it called)."""
        fixed = (task.get("original_context") or {}).get("fixed_call") or {}
        if fixed.get("tool") in SEND_TOOLS:
            return True
        return any(
            (u.get("message") or {}).get("type") == "tool_call" and (u.get("message") or {}).get("tool_name") in SEND_TOOLS
            for u in run.get("progress_updates") or [] if isinstance(u, dict)
        )

    async def _done(self, local: datetime) -> tuple[list[dict], str | None]:
        tasks = await self._task_list()
        if tasks is None:
            return [], "Couldn't read your tasks this time."
        failed, finished = [], []
        for t in tasks:
            runs = self._runs_today(t, local)
            if not runs or (runs[0].get("status") in FINISHED_OK and self._sends(t, runs[0])):
                continue  # nothing today, or it is a sent email (the "sent" section)
            name = one_line(t.get("name") or "Untitled task", 100)
            run, link = runs[0], f"/tasks/{t.get('task_id')}"
            if run.get("status") == "error":
                failed.append({"id": item_id("done", f"f:{run.get('run_id')}"), "section": "done",
                               "text": f"{name}: failed", "link": link, "why": "Failed today. Open it to retry."})
            elif run.get("status") in FINISHED_OK:
                finished.append({"id": item_id("done", f"d:{run.get('run_id')}"), "section": "done",
                                 "text": f"{name}: done", "link": link, "why": "Finished today"})
        return failed + finished, None

    async def _sent(self, local: datetime) -> tuple[list[dict], str | None]:
        tasks = await self._task_list()
        if tasks is None:
            return [], "Couldn't read your tasks this time."
        items = []
        for t in tasks:
            for run in self._runs_today(t, local):
                if run.get("status") not in FINISHED_OK or not self._sends(t, run):
                    continue
                name = one_line(t.get("name") or "an email", 110)
                text = f"Sent {name[5:]}" if name.lower().startswith("send ") else f"{name}: email sent"
                fixed = (t.get("original_context") or {}).get("fixed_call") or {}
                why = "Sent today after you approved it" if fixed.get("tool") in SEND_TOOLS else "Sent by a task today"
                items.append({"id": item_id("sent", str(run.get("run_id"))), "section": "sent", "text": text,
                              "link": f"/tasks/{t.get('task_id')}", "why": why})
        return items, None

    def _saved_files(self, local: datetime) -> list[tuple[float, str]]:
        """Files changed today in Sentient's files folder, newest first (your uploads and tool dumps left out)."""
        from sentient import paths

        root = paths.files_dir()
        if not root.is_dir():
            return []
        found: list[tuple[float, str]] = []
        seen = 0
        for p in root.rglob("*"):
            seen += 1
            if seen > FILES_SCAN_LIMIT:
                break
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or rel.startswith("uploads/") or (rel.startswith("outputs/") and p.name.startswith("tool-")):
                continue
            mtime = p.stat().st_mtime
            if datetime.fromtimestamp(mtime, local.tzinfo).date() == local.date():
                found.append((mtime, rel))
        return sorted(found, reverse=True)

    async def _files(self, local: datetime) -> tuple[list[dict], str | None]:
        items, names = [], set()
        for t in await self._task_list() or []:
            for run in self._runs_today(t, local):
                for f in ((run.get("result") or {}).get("files_created") or []):
                    name = str((f or {}).get("filename") or "").strip()
                    if not name or name.lower() in names:
                        continue
                    names.add(name.lower())
                    items.append({"id": item_id("files", f"{run.get('run_id')}:{name}"), "section": "files",
                                  "text": one_line(name, 100), "link": f"/tasks/{t.get('task_id')}",
                                  "why": f"Made by '{one_line(t.get('name') or 'a task', 50)}' today"})
        for _mtime, rel in self._saved_files(local):
            base = rel.rsplit("/", 1)[-1]
            if base.lower() in names or rel.lower() in names:
                continue
            names.add(base.lower())
            items.append({"id": item_id("files", f"saved:{rel}"), "section": "files", "text": one_line(rel, 100),
                          "link": None, "why": "Saved in your Sentient files today"})
        return items, None

    async def _waiting(self, local: datetime) -> tuple[list[dict], str | None]:
        tasks = [  # a task that failed today is already under "Done today"
            t for t in await self._task_list() or [] if not (t.get("status") == "error" and self._runs_today(t, local))
        ]
        items = self._waiting_tasks(tasks, "waiting")
        rows = await self.app.store.fetchall(
            "SELECT notification_id, payload FROM proactive_suggestions WHERE status = 'pending'"
            " ORDER BY created_at DESC LIMIT 10"
        )
        for r in rows:
            with contextlib.suppress(Exception):
                s = json.loads(r["payload"])["suggestion"]
                items.append({"id": item_id("waiting", f"s:{r['notification_id']}"), "section": "waiting",
                              "text": one_line(s.get("description")),
                              "link": (s.get("source_event") or {}).get("url") or "/notifications",
                              "why": "A suggestion waiting for your answer", "notification_id": r["notification_id"]})
        return items, None

    async def _weather(self, local: datetime) -> tuple[list[dict], str | None]:
        place = self.app.config.assistant.location.strip()
        if not place:
            return [], "Add your city in Settings to see the weather."
        out = await self.read("weather_current", {})
        if out is None:
            return [], "Couldn't get the weather this time."
        cur = out.get("current") or {}
        today = out.get("today") or {}
        parts = [str(cur.get("condition") or "").strip().lower() or None]
        if cur.get("temperature_c") is not None:
            parts.append(f"{round(float(cur['temperature_c']))}°C now")
        if today.get("max_c") is not None:
            parts.append(f"high {round(float(today['max_c']))}°C")
        if today.get("rain_chance_percent"):
            parts.append(f"{today['rain_chance_percent']}% chance of rain")
        shown = [p for p in parts if p]
        if not shown:
            return [], None
        where = str(out.get("location") or place).split(",")[0]
        return [{"id": item_id("weather", local.date().isoformat()), "section": "weather",
                 "text": f"{where}: {', '.join(shown)}", "link": None,
                 "why": f"Weather for {place}, your city in Settings"}], None

    async def _news(self, local: datetime) -> tuple[list[dict], str | None]:
        topics = [t for t in self.cfg.news_topics if str(t).strip()][:3]
        if not topics:
            return [], "Pick a few news topics to see headlines."
        items, seen = [], set()
        for topic in topics:
            out = await self.read("news_search", {"query": topic, "max_results": 3})
            for a in (out or {}).get("articles") or []:
                title = one_line((a or {}).get("title"), 110)
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                source = f" ({a['source']})" if a.get("source") else ""
                items.append({"id": item_id("news", str(a.get("url") or title)), "section": "news",
                              "text": f"{title}{source}", "link": a.get("url"), "why": f"You follow '{topic}'"})
                break
        return items, None

    async def _summarize(self, items: list[dict]) -> None:
        """One short line per unread email from the fast model; the sender and subject stay when it fails."""
        mails = [i for i in items if i.get("_mail")]
        if not mails or not self.cfg.summarize_emails:
            return
        listing = "\n".join(
            f"{n}. From: {m['_mail']['from']} | Subject: {m['_mail']['subject']} | Text: {m['_mail']['snippet']}"
            for n, m in enumerate(mails, 1)
        )
        ask = f"{listing}\n\nReply with exactly {len(mails)} lines, each starting with its number, like '1. ...'."
        try:
            text = await self.app.llm.complete_text(
                "fast", [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": ask}]
            )
        except Exception as exc:
            log.info("brief: email summaries skipped: %s", exc)
            return
        for n, line in parse_summaries(text, len(mails)).items():
            m = mails[n - 1]
            name = m["_mail"]["from"]
            first = (name.split() or [name])[0].lower()
            m["text"] = one_line(line if first and first in line.lower() else f"{name}: {line}")

    # ------------------------------------------------------------------ building and delivering
    async def build(self, now: datetime | None = None, kind: str = MORNING) -> dict:
        """Today's brief of ``kind``, not delivered: ``{kind, day, title, sections, items, skipped}``."""
        now = now or datetime.now(UTC)
        local = now.astimezone(self._tz())
        scores = await self.engine.preference_scores()
        order = order_sections(self._sections(kind), scores, KIND_SECTIONS[kind])
        readers = {"calendar": self._calendar, "email": self._email, "tasks": self._tasks,
                   "weather": self._weather, "news": self._news, "done": self._done, "sent": self._sent,
                   "files": self._files, "waiting": self._waiting, "tomorrow": self._tomorrow}
        found: dict[str, list[dict]] = {}
        skipped: list[dict] = []
        for s in order:
            try:
                items, reason = await readers[s](local)
            except Exception:
                log.exception("brief: the %s section failed", s)
                items, reason = [], "Couldn't read this one this time."
            found[s] = items
            if not items and reason:
                skipped.append({"section": s, "label": SECTION_LABELS[s], "reason": reason})
        items = cap_items(found, order, scores, self.cfg.max_items)
        await self._summarize(items)
        for i in items:
            i.pop("_mail", None)
            i.setdefault("link", None)
            i["feedback"] = None
        shown = [s for s in order if any(i["section"] == s for i in items)]
        return {
            "kind": kind,
            "day": local.date().isoformat(),
            "title": TITLES[kind].format(day=local.strftime("%A")),
            "sections": [{"id": s, "label": SECTION_LABELS[s], "feedback": None} for s in shown],
            "items": items,
            "skipped": skipped,
        }

    @staticmethod
    def as_text(brief: dict, *, links: bool = False) -> str:
        """The brief as short lines (notification message, chats, voice)."""
        if not brief.get("items"):
            return EMPTY_TEXT.get(str(brief.get("kind") or MORNING), EMPTY_TEXT[MORNING])
        lines = []
        for s in brief.get("sections") or []:
            lines.append(f"**{s['label']}**")
            for i in brief["items"]:
                if i["section"] != s["id"]:
                    continue
                link = i.get("link")
                if links and isinstance(link, str) and link.startswith(("http://", "https://")):
                    lines.append(f"- [{i['text']}]({link})")
                else:
                    lines.append(f"- {i['text']}")
        return "\n".join(lines)

    def _expires_at(self, now: datetime) -> str:
        local = now.astimezone(self._tz())
        end = datetime.combine(local.date() + timedelta(days=1), time.min, local.tzinfo)
        return end.astimezone(UTC).isoformat()

    async def deliver(self, now: datetime | None = None, kind: str = MORNING) -> dict:
        """Build today's brief of ``kind`` and post it as a ``brief`` notification. One brief shows at a time, so
        earlier briefs still showing (of either kind) expire."""
        now = now or datetime.now(UTC)
        brief = await self.build(now, kind)
        brief["expires_at"] = self._expires_at(now)
        task = await self.task(kind)
        task_id = (task or {}).get("task_id")
        await self.expire(now, everything=True)
        payload = {"brief": brief, "status": "active", "task_id": task_id}
        note = await self.app.notify("brief", self.as_text(brief), title=brief["title"], payload=payload)
        return self._public(note)

    @staticmethod
    def _public(note: dict) -> dict:
        p = note.get("payload") or {}
        return {"kind": MORNING, **(p.get("brief") or {}), "id": note["id"], "status": p.get("status") or "active",
                "task_id": p.get("task_id"), "created_at": note.get("created_at")}

    async def _notes(self, limit: int = 50) -> list[dict]:
        rows = await self.app.store.fetchall(
            "SELECT id FROM notifications WHERE kind = 'brief' ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        out = []
        for r in rows:
            note = await self.app.notifications.get(r["id"])
            if note is not None and isinstance(note["payload"].get("brief"), dict):
                out.append(note)
        return out

    async def today(self, now: datetime | None = None, kind: str | None = None) -> dict | None:
        """The brief still showing (delivered today and not expired; of ``kind`` when given), or None."""
        now = now or datetime.now(UTC)
        for note in await self._notes(5):
            p = note["payload"]
            if kind and str(p["brief"].get("kind") or MORNING) != kind:
                continue
            if p.get("status") == "active" and str(p["brief"].get("expires_at") or "") > now.astimezone(UTC).isoformat():
                return self._public(note)
        return None

    async def expire(self, now: datetime | None = None, *, everything: bool = False) -> int:
        """Expire briefs whose day is over (``everything``: every brief still showing). They are marked read."""
        now = now or datetime.now(UTC)
        count = 0
        for note in await self._notes():
            p = note["payload"]
            if p.get("status") != "active":
                continue
            if not everything and str(p["brief"].get("expires_at") or "") > now.astimezone(UTC).isoformat():
                continue
            await self.app.notifications.update_payload(note["id"], {**p, "status": "expired"})
            if not note.get("read"):
                await self.app.notifications.mark_read(note["id"])
            count += 1
        return count

    async def feedback(self, brief_id: str, value: str, *, item: str | None = None, section: str | None = None) -> dict:
        """Thumbs up or down on one item or one section; it moves that section's learned score. Changing a rating
        replaces it (the latest wins: the earlier one is taken back first); the same rating again changes nothing."""
        value = str(value or "").strip().lower()
        if value not in {"up", "down"}:
            raise BriefError(400, "value must be up or down")
        if bool(item) == bool(section):
            raise BriefError(400, "Give either item_id or section.")
        note = await self.app.notifications.get(brief_id)
        if note is None or note["kind"] != "brief" or not isinstance(note["payload"].get("brief"), dict):
            raise BriefError(404, "Brief not found.")
        payload = note["payload"]
        if payload.get("status", "active") != "active":
            raise BriefError(409, "This brief has expired.")
        brief = dict(payload["brief"])
        if item:
            targets = brief.get("items") or []
            match = next((i for i in targets if i.get("id") == item), None)
        else:
            targets = brief.get("sections") or []
            match = next((s for s in targets if s.get("id") == section), None)
        if match is None:
            raise BriefError(404, "That line isn't in this brief.")
        previous = match.get("feedback")
        if previous == value:
            return self._public(note)
        match["feedback"] = value
        stype = preference_type(match.get("section") or match.get("id"))
        if previous in {"up", "down"}:
            await self.engine.undo_feedback(stype, previous == "up")
        await self.engine.record_feedback(stype, value == "up")
        await self.app.notifications.update_payload(brief_id, {**payload, "brief": brief})
        updated = await self.app.notifications.get(brief_id)
        return self._public(updated or note)

    async def text_for_today(self, now: datetime | None = None, kind: str | None = None) -> dict:
        """The brief for reading out: the one showing (of ``kind`` when given), else a fresh one that is not
        delivered (the morning one unless ``kind`` says evening)."""
        brief = await self.today(now, kind)
        if brief is None:
            brief = await self.build(now, kind or MORNING)
        return {"kind": brief.get("kind") or MORNING, "day": brief["day"], "title": brief["title"],
                "text": self.as_text(brief).replace("**", ""), "items": len(brief.get("items") or [])}


# ---------------------------------------------------------------------------- tools
def _brief(ctx: ToolContext) -> DailyBrief | None:
    app = (ctx.extra or {}).get("app") if ctx is not None else None
    return getattr(getattr(app, "proactivity", None), "brief", None)


@tool(BUILD_TOOL, risk=Risk.write, internal=True)
async def daily_brief_build(ctx: ToolContext, kind: str = MORNING) -> dict:
    """Make the user's Daily Brief now and show it in Notifications (and paired chats). `kind` is "morning" (the
    day ahead) or "evening" (a wrap-up of the day). It only reads; it never sends or changes anything."""
    brief = _brief(ctx)
    if brief is None:
        return {"error": "The Daily Brief is not available."}
    try:
        kind = check_kind(kind)
    except BriefError as exc:
        return {"error": exc.detail}
    out = await brief.deliver(kind=kind)
    return {"ok": True, "brief_id": out["id"], "items": len(out.get("items") or []), "text": DailyBrief.as_text(out)}


@tool(READ_TOOL)
async def daily_brief_today(ctx: ToolContext, kind: str = "") -> dict:
    """Read the user's brief ("read my brief", "what's on today?", "how did today go?") as short lines: the morning
    one has today's meetings, emails that need them, tasks and the weather; the evening one wraps up the day.
    Leave `kind` empty for the one showing now, or say "morning" or "evening". Read the lines out as they are."""
    brief = _brief(ctx)
    if brief is None:
        return {"error": "The Daily Brief is not available."}
    try:
        wanted = check_kind(kind) if str(kind or "").strip() else None
    except BriefError as exc:
        return {"error": exc.detail}
    return await brief.text_for_today(kind=wanted)


class BriefPlugin(ToolPlugin):
    id = "daily_brief"
    display_name = "Daily Brief"
    description = "Your Daily Brief: today's calendar, emails that need you, tasks and the weather in a few lines."
    category = "core"
    icon = "IconSunrise"
    selection_hint = "Use for the user's daily or evening brief, morning summary, day wrap-up or 'what's on today'."
    tools = [daily_brief_build, daily_brief_today]
