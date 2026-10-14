"""The activity timeline (docs/API.md section 21, issue #147).

One list of what Sentient did, newest first, and of what it will do next, built only from what the other packages
already keep: chat tool calls (``messages``), task runs, script checks, notifications, memories, dreams, the skills
log, backup files, the "Always allow" notes in config and model spending. The one thing added for it is
``messages.approval``: how the user answered a chat's approval request, which was kept nowhere before.

**Paging.** Every past item has a key ``(at, id)``; ``id`` is ``<source>:<row id>``. A page asks each source for its
newest items strictly before the cursor, merges them and keeps ``limit``. A source that filters rows in Python (an
app filter on task runs, the kinds of notifications) reads in batches and may stop early; it then reports how far it
read (its *floor*), and the page never goes below the highest floor, so no item is skipped or shown twice. Every SQL
source reads newest first along an index on its time column (``idx_messages_tool_time``, ``idx_runs_created``,
``idx_facts_created``, ``idx_notifications_created``, ``idx_evolution_log_ts``, ``idx_dreams_started``).

Incognito chats are never read. Nothing here calls a model or changes anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentient.activity import words
from sentient.agent import grants as grant_rules
from sentient.agent import reversible
from sentient.llm.provider import ToolCall
from sentient.store.db import new_id
from sentient.tasks.schedule import calculate_next_run, get_tz, iso, parse_iso, user_timezone_name
from sentient.tools.rules import tool_title

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

# What the window filters by. Order is the order of the filter chips.
KINDS = ("chat", "tasks", "briefs", "monitors", "approvals", "channels", "notifications", "memory", "skills",
         "backups", "cost")
CHANNELS = ("telegram", "discord", "whatsapp")
MAX_LIMIT = 200
MAX_BATCHES = 8              # a Python-filtered source reads at most this many batches per page
UPCOMING_DAYS = 7
MAX_PER_TASK = 7             # occurrences of one recurring task listed in the upcoming week

Cursor = tuple[str, str]
# the app a file snapshot counts under (#142): what ran the code that changed the files
SNAPSHOT_APPS = {"terminal": "terminal", "code": "code", "file_write": "files", "restore": "files"}


@dataclass
class Item:
    id: str
    kind: str
    at: str | None
    title: str
    why: str | None = None
    detail: str | None = None
    status: str | None = None
    app: str | None = None
    app_label: str | None = None
    ref: dict | None = None
    upcoming: bool = False
    # a way back the window can offer (#142): {"action": "restore", "label": "Restore"} for something moved to the trash
    undo: dict | None = None
    _args: tuple | None = field(default=None, repr=False)  # (session_id, call id, before) for the arguments look-up
    _apps: frozenset[str] = field(default_factory=frozenset, repr=False)  # other apps it counts under in a filter

    @property
    def key(self) -> Cursor:
        return (self.at or "", self.id)

    def text(self) -> str:
        return " ".join(p for p in (self.title, self.why, self.detail, self.app_label) if p).lower()

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "at": self.at, "title": self.title, "why": self.why,
                "detail": self.detail, "status": self.status, "app": self.app, "app_label": self.app_label,
                "ref": self.ref, "upcoming": self.upcoming, "undo": self.undo}


@dataclass
class Query:
    kinds: frozenset[str]
    app: str | None = None
    q: str | None = None

    def wants(self, *kinds: str) -> bool:
        return any(k in self.kinds for k in kinds)

    def like(self) -> str:
        escaped = (self.q or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    def matches(self, item: Item, *, text: bool = True) -> bool:
        """Whether ``item`` passes the filters. ``text=False`` leaves out the search words, for sources that
        searched their own columns in SQL already."""
        if item.kind not in self.kinds:
            return False
        if self.app and item.app != self.app and self.app not in item._apps:
            return False
        return not text or not self.q or self.q in item.text()


def encode_cursor(key: Cursor | None) -> str | None:
    return f"{key[0]}|{key[1]}" if key else None


def decode_cursor(raw: str | None) -> Cursor | None:
    if not raw or "|" not in raw:
        return None
    at, _, iid = raw.partition("|")
    return (at, iid) if at and iid else None


def _before(ts: str, key: str, cur: Cursor | None) -> tuple[str, list]:
    if cur is None:
        return "1", []
    return f"({ts} < ? OR ({ts} = ? AND {key} < ?))", [cur[0], cur[0], cur[1]]


def _json(column: str, path: str) -> str:
    """``json_extract`` that reads unreadable JSON as NULL instead of failing the whole query."""
    return f"(CASE WHEN json_valid({column}) THEN json_extract({column}, '{path}') END)"


def _loads(raw: Any) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


Source = Callable[[Cursor | None, int, Query], Awaitable[tuple[list[Item], Cursor | None]]]


class Timeline:
    def __init__(self, app: SentientApp):
        self.app = app

    # ------------------------------------------------------------------ helpers
    @property
    def store(self):
        return self.app.store

    def tz(self) -> tzinfo:
        return get_tz(user_timezone_name(self.app.config.assistant.timezone))

    def query(self, kinds: Iterable[str] | None, app: str | None, q: str | None) -> Query:
        wanted = frozenset(k for k in (kinds or ()) if k in KINDS) or frozenset(KINDS)
        text = " ".join(str(q or "").split()).lower()[:200] or None
        return Query(wanted, (app or "").strip().lower() or None, text)

    def _app(self, app: str | None) -> tuple[str | None, str | None]:
        return app, words.app_label(app, self.app.registry)

    def _tool_app(self, name: str) -> str | None:
        """The app a tool belongs to; for a tool that is gone (an app disconnected since), its name's first word
        when that names a known app ("gmail_send" -> "gmail")."""
        tool = self.app.registry.get(name)
        if tool is not None:
            return tool.plugin
        head = name.split("_", 1)[0]
        return head if head in words.APP_LABELS or self.app.registry.plugin(head) is not None else None

    def _tool_names(self, app: str) -> list[str]:
        return [t.name for t in self.app.registry.tools(include_hidden=True) if t.plugin == app]

    async def _sql(
        self, *, select: str, where: list[str], params: list, ts: str, key: str, build: Callable[[Any], Item | None],
        cur: Cursor | None, n: int, query: Query, filtered: bool,
    ) -> tuple[list[Item], Cursor | None]:
        """Newest rows strictly before ``cur``, built into items that pass the kind and app filters (the search
        words are matched in SQL). ``filtered``: rows may be dropped here, so read in batches until ``n`` items or
        the end; stopping early returns the floor (the last row read)."""
        batch = max(n * 2, 100) if filtered else n + 1
        items: list[Item] = []
        floor = cur
        for _ in range(MAX_BATCHES):
            cond, cp = _before(ts, key, floor)
            rows = await self.store.fetchall(
                f"{select} WHERE {' AND '.join([*where, cond])} ORDER BY {ts} DESC, {key} DESC LIMIT ?",
                [*params, *cp, batch],
            )
            for row in rows:
                try:
                    item = build(row)
                except Exception:  # one odd row never breaks the timeline
                    log.debug("activity: could not read a row", exc_info=True)
                    item = None
                if item is not None and query.matches(item, text=False):
                    items.append(item)
            if len(rows) < batch:
                return items, None
            floor = (rows[-1]["_at"], rows[-1]["_key"])
            if len(items) >= n:
                break
        return items, floor

    # ------------------------------------------------------------------ the past
    async def past(
        self, *, before: str | None = None, limit: int = 50, kinds: Iterable[str] | None = None,
        app: str | None = None, q: str | None = None,
    ) -> dict:
        """``{items, next}``: up to ``limit`` items newest first, before the cursor ``before``; ``next`` is the cursor
        of the next page, or None at the end."""
        query = self.query(kinds, app, q)
        limit = max(1, min(int(limit or 50), MAX_LIMIT))
        cur = decode_cursor(before)
        sources: list[Source] = [
            self._chat_tools, self._channel_replies, self._channel_deliveries, self._task_runs, self._script_checks, self._notifications,
            self._facts, self._dreams, self._skills, self._backups_past, self._always_allowed, self._cost,
            self._file_snapshots,
        ]
        results = await asyncio.gather(*(src(cur, limit, query) for src in sources))
        candidates = [i for items, _ in results for i in items if cur is None or i.key < cur]
        floors = [f for _, f in results if f is not None]
        bound = max(floors) if floors else None
        if bound is not None:
            candidates = [i for i in candidates if i.key >= bound]
        candidates.sort(key=lambda i: i.key, reverse=True)
        page = candidates[:limit]
        if len(candidates) > limit:
            nxt: Cursor | None = page[-1].key
        else:
            nxt = bound
        await self._fill_arguments(page)
        return {"items": [i.as_dict() for i in page], "next": encode_cursor(nxt)}

    # -- chats: tool calls (and approvals answered in a chat)
    async def _chat_tools(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("chat", "approvals"):
            return [], None
        where = ["m.role = 'tool'", "s.incognito = 0"]
        params: list = []
        if not query.wants("chat"):
            where.append("m.approval IS NOT NULL")
        elif not query.wants("approvals"):
            where.append("m.approval IS NULL")
        if query.app:
            names = self._tool_names(query.app)
            marks = ",".join("?" * len(names)) or "NULL"
            where.append(f"(m.name IN ({marks}) OR s.channel = ?)")
            params += [*names, query.app]
        if query.q:
            where.append("(m.name LIKE ? ESCAPE '\\' OR s.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'"
                         " OR m.approval LIKE ? ESCAPE '\\')")
            params += [query.like()] * 4
        return await self._sql(
            select="SELECT m.created_at AS _at, ('chat:' || m.id) AS _key, m.id, m.session_id, m.name, m.content,"
                   " m.approval, m.tool_call_id, s.title, s.channel FROM messages m JOIN sessions s ON s.id = m.session_id",
            where=where, params=params, ts="m.created_at", key="('chat:' || m.id)", build=self._chat_item,
            cur=cur, n=n, query=query, filtered=False,
        )

    def _where_chat(self, channel: str | None, title: str | None) -> str:
        if channel in CHANNELS:
            return f"You asked on {words.app_label(channel)}"
        if channel == "voice":
            return "You asked by voice"
        name = (title or "").strip()
        return f'You asked in the chat "{words.short(name, 60)}"' if name else "You asked in a chat"

    def _chat_item(self, row: Any) -> Item:
        name = str(row["name"] or "")
        tool = self.app.registry.get(name)
        app = tool.plugin if tool is not None else None
        app, label = self._app(app)
        title = tool_title(tool) if tool is not None else name.replace("_", " ").capitalize()
        approval = _loads(row["approval"])
        result = _loads(row["content"])
        error = result.get("error") if isinstance(result, dict) else None
        if approval is None and isinstance(result, dict) and (result.get("declined") or "user declined" in str(error)):
            approval = {"decision": grant_rules.DENY}  # said no before answers were kept (#147): still a "no"
        ref = {"type": "chat", "id": row["session_id"], "message_id": row["id"]}
        where = self._where_chat(row["channel"], row["title"])
        args = (row["session_id"], row["tool_call_id"], row["_at"])
        also = frozenset({row["channel"]}) if row["channel"] in CHANNELS else frozenset()
        undone = isinstance(result, dict) and result.get("undone") is True  # Undo pressed in its window (#142)
        # something moved to the trash can come back when its app has a way to restore it (#142)
        back = reversible.restore_twin(tool, self.app.registry) if tool is not None and not error else None
        undo = {"action": "restore", "label": "Restore"} if back is not None else None
        if isinstance(approval, dict):
            decision = str(approval.get("decision") or "")
            target = str(approval.get("target") or "").strip()
            what = f'"{title}" in {label}' if label else f'"{title}"'
            asked = f"Asked you before {what}" + (f" ({words.short(target, 60)})" if target else "")
            if decision == grant_rules.DENY:
                said, status = "You said no, so it was not done.", "declined"
            elif undone:
                said, status = "You allowed it, then pressed Undo, so nothing was sent.", "stopped"
            else:
                said, status = f"You chose {grant_rules.decision_label(decision)}.", "done"
                if error:
                    said, status = f"You chose {grant_rules.decision_label(decision)}, but it didn't work: " \
                                   f"{words.short(error, 120)}", "failed"
            if approval.get("untrusted"):  # the loop's own sentence: "Sentient read content from Gmail in this chat..."
                why = str(approval["untrusted"]).rstrip(". ")
            elif approval.get("rule") == "ask":
                why = f"You set it to ask first. {where}"
            else:
                risk = str(approval.get("risk_label") or "It changes something")
                why = f"{risk}, so Sentient asks first. {where}"
            return Item(id=row["_key"], kind="approvals", at=row["_at"], title=asked, why=why, detail=said,
                        status=status, app=app, app_label=label, ref=ref, undo=undo, _args=args, _apps=also)
        verb = f"Used {label}: {title}" if label else f"Used {title}"
        if undone:
            return Item(id=row["_key"], kind="chat", at=row["_at"], title=f"Held back: {title}" + (f" in {label}" if label else ""),
                        why=where, detail="You pressed Undo, so nothing was sent.", status="stopped", app=app,
                        app_label=label, ref=ref, _apps=also)
        if error:
            return Item(id=row["_key"], kind="chat", at=row["_at"],
                        title=f"Couldn't {title[:1].lower() + title[1:]}" + (f" in {label}" if label else ""),
                        why=where, detail=words.short(error, 160), status="failed", app=app, app_label=label,
                        ref=ref, _args=args, _apps=also)
        return Item(id=row["_key"], kind="chat", at=row["_at"], title=verb, why=where, status="done", app=app,
                    app_label=label, ref=ref, undo=undo, _args=args, _apps=also)

    async def _call_arguments(self, session_id: str, call_id: str, at: str) -> dict | None:
        """The arguments a chat tool call was given, read from the assistant message that made it."""
        row = await self.store.fetchone(
            "SELECT tool_calls FROM messages WHERE session_id = ? AND role = 'assistant' AND created_at <= ?"
            " AND tool_calls IS NOT NULL ORDER BY created_at DESC LIMIT 1", (session_id, at),
        )
        calls = _loads(row["tool_calls"]) if row else None
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, dict) or call.get("id") != call_id:
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            args = fn.get("arguments")
            args = _loads(args) if isinstance(args, str) else args
            return args if isinstance(args, dict) else None
        return None

    async def _fill_arguments(self, page: list[Item]) -> None:
        """A short line of what each chat tool call was given ("query: weather in Pune"), read only for the page."""
        for item in page:
            if item._args is None or item.detail is not None:
                continue
            args = await self._call_arguments(*item._args)
            if args:
                item.detail = _brief(args) or None

    async def restore(self, item_id: str) -> dict:
        """Bring back what a chat tool call moved to the trash, with its app's restore tool (#142), or put back the
        files code changed (``snap:<id>``, ``sentient.snapshots``). Returns ``{"ok": True, "result": ...}`` or
        ``{"ok": False, "error": "..."}`` in plain words. Never rules and Read only connections still apply
        (``Agent.run_tool``)."""
        kind, _, message_id = str(item_id or "").partition(":")
        if kind == "snap" and message_id:
            snaps = getattr(self.app, "snapshots", None)
            if snaps is None:
                return {"ok": False, "error": "Putting files back isn't available right now."}
            return await snaps.restore(message_id)
        if kind != "chat" or not message_id:
            return {"ok": False, "error": "That item can't be undone."}
        row = await self.store.fetchone(
            "SELECT m.session_id, m.name, m.content, m.tool_call_id, m.created_at FROM messages m"
            " JOIN sessions s ON s.id = m.session_id WHERE m.id = ? AND m.role = 'tool' AND s.incognito = 0",
            (message_id,),
        )
        tool = self.app.registry.get(str(row["name"] or "")) if row else None
        back = reversible.restore_twin(tool, self.app.registry) if tool is not None else None
        result = _loads(row["content"]) if row else None
        if row is None or tool is None or back is None or (isinstance(result, dict) and result.get("error")):
            return {"ok": False, "error": "There is nothing to restore for this item."}
        args = await self._call_arguments(row["session_id"], row["tool_call_id"], row["created_at"])
        fitted = reversible.fit_arguments(back, args)
        if fitted is None or self.app.agent is None:
            return {"ok": False, "error": "Sentient can't tell what to restore for this item."}
        ctx = self.app.agent.tool_context(row["session_id"], "desktop")
        res, is_error, _ = await self.app.agent.run_tool(ToolCall(id=new_id(), name=back.name, arguments=fitted), ctx)
        error = res.get("error") if isinstance(res, dict) else None
        if is_error or error:
            return {"ok": False, "error": str(error or res)[:300]}
        log.info("activity: restored %s with %s", item_id, back.name)
        return {"ok": True, "result": res}

    # -- files code changed, with the copies kept first (#142)
    async def _file_snapshots(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("chat", "tasks"):
            return [], None
        where = ["(session_id IS NULL OR session_id NOT IN (SELECT id FROM sessions WHERE incognito = 1))"]
        params: list = []
        if not query.wants("chat"):
            where.append("task_id IS NOT NULL")
        elif not query.wants("tasks"):
            where.append("task_id IS NULL")
        if query.app:
            sources = [s for s, app in SNAPSHOT_APPS.items() if app == query.app]
            if not sources:
                return [], None
            where.append(f"source IN ({','.join('?' * len(sources))})")
            params += sources
        if query.q:
            where.append("(what LIKE ? ESCAPE '\\' OR files LIKE ? ESCAPE '\\')")
            params += [query.like()] * 2
        snaps = getattr(self.app, "snapshots", None)
        try:
            return await self._sql(
                select="SELECT created_at AS _at, ('snap:' || id) AS _key, id, source, what, session_id, task_id,"
                       " run_id, files, created, notes, restored_at FROM file_snapshots",
                where=where, params=params, ts="created_at", key="('snap:' || id)",
                build=lambda row: self._snapshot_item(row, snaps), cur=cur, n=n, query=query, filtered=False,
            )
        except sqlite3.OperationalError:  # an older database without the table yet
            return [], None

    def _snapshot_item(self, row: Any, snaps: Any) -> Item:
        files = [f for f in (_loads(row["files"]) or []) if isinstance(f, dict)]
        notes = [str(x) for x in (_loads(row["notes"]) or []) if x]
        source = str(row["source"] or "")
        count = len(files)
        plural = f"{count} file{'s' if count != 1 else ''}"
        name = Path(str(files[0].get("path") or "")).name if files else ""
        title = {
            "terminal": f"A command changed {plural}", "code": f"Code changed {plural}",
            "file_write": f"Wrote over {name}" if count == 1 else f"Wrote over {plural}",
            "restore": f"Put back {plural}",
        }.get(source, f"Changed {plural}")
        kept = [f for f in files if f.get("copy")]
        if source == "restore":
            why = "You pressed Restore. Sentient kept copies of what it replaced, so you can undo it."
        elif kept:
            why = "Sentient kept copies first, so you can put them back."
        else:
            why = "Sentient couldn't keep copies: " + (str(files[0].get("note") or "") if files else "")
        names = [Path(str(f.get("path") or "")).name + (" (deleted)" if f.get("change") == "deleted" else "")
                 for f in files[:5]]
        detail = ", ".join(names) + (f" and {count - 5} more" if count > 5 else "")
        what = str(row["what"] or "").strip()
        if what and source in {"terminal", "code"}:
            detail = f"{words.short(what, 80)}: {detail}"
        if notes:
            detail += f". {notes[0]}"
        snap = {"id": row["id"], "files": files, "restored_at": row["restored_at"]}
        undo = ({"action": "restore", "label": "Undo" if source == "restore" else "Restore"}
                if snaps is not None and snaps.restorable(snap) else None)
        status = "restored" if row["restored_at"] else "done"
        if row["task_id"]:
            ref = {"type": "task", "id": row["task_id"], "run_id": row["run_id"]}
        elif row["session_id"]:
            ref = {"type": "chat", "id": row["session_id"]}
        else:
            ref = None
        app, label = self._app(SNAPSHOT_APPS.get(source))
        return Item(id=row["_key"], kind="tasks" if row["task_id"] else "chat", at=row["_at"], title=title, why=why,
                    detail=words.short(detail, 240) or None, status=status, app=app, app_label=label, ref=ref,
                    undo=undo)

    # -- messaging apps: Sentient's replies in Telegram, Discord and WhatsApp chats
    async def _channel_replies(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("channels"):
            return [], None
        if query.app and query.app not in CHANNELS:
            return [], None
        channels = [query.app] if query.app else list(CHANNELS)
        marks = ",".join("?" * len(channels))
        where = [f"m.session_id IN (SELECT id FROM sessions WHERE channel IN ({marks}) AND incognito = 0)",
                 "m.role = 'assistant'", "m.tool_calls IS NULL", "COALESCE(m.content, '') != ''"]
        params: list = list(channels)
        if query.q:
            where.append("m.content LIKE ? ESCAPE '\\'")
            params.append(query.like())

        def build(row: Any) -> Item:
            app, label = self._app(row["channel"])
            chat = (row["title"] or "").strip()
            return Item(id=row["_key"], kind="channels", at=row["_at"], title=f"Replied to you on {label}",
                        why=f'You wrote to Sentient in "{words.short(chat, 60)}"' if chat else "You wrote to Sentient",
                        detail=words.short(row["content"], 200), status="done", app=app, app_label=label,
                        ref={"type": "chat", "id": row["session_id"], "message_id": row["id"]})

        return await self._sql(
            select="SELECT m.created_at AS _at, ('reply:' || m.id) AS _key, m.id, m.session_id, m.content, s.title,"
                   " s.channel FROM messages m JOIN sessions s ON s.id = m.session_id",
            where=where, params=params, ts="m.created_at", key="('reply:' || m.id)", build=build,
            cur=cur, n=n, query=query, filtered=False,
        )

    # -- messaging apps: notifications sent out to paired chats (#345)
    async def _channel_deliveries(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("channels") or (query.app and query.app not in CHANNELS):
            return [], None
        where: list[str] = []
        params: list = []
        if query.app:
            where.append("channel = ?")
            params.append(query.app)
        if query.q:
            where.append("(title LIKE ? ESCAPE '\\' OR chat_label LIKE ? ESCAPE '\\' OR error LIKE ? ESCAPE '\\'"
                         " OR channel LIKE ? ESCAPE '\\')")  # "telegram" finds "Sent to Telegram: ..."
            params += [query.like()] * 4

        def build(row: Any) -> Item:
            app, label = self._app(row["channel"])
            title = words.short(row["title"], 100)
            chat = words.short((row["chat_label"] or "").strip(), 60)
            to = f'To your chat "{chat}"' if chat else "To your paired chat"
            why = f"{to}. {words.DELIVERY_REASONS.get(row['reason'], words.DELIVERY_REASONS['default'])}"
            ref = _loads(row["ref"])
            ref = ref if isinstance(ref, dict) and ref.get("type") else None
            if row["status"] == "failed":
                return Item(id=row["_key"], kind="channels", at=row["_at"], title=f"Couldn't send to {label}: {title}",
                            why=why, detail=words.short(row["error"] or "", 200) or None, status="failed", app=app,
                            app_label=label, ref=ref)
            return Item(id=row["_key"], kind="channels", at=row["_at"], title=f"Sent to {label}: {title}", why=why,
                        detail=words.DELIVERY_WHAT.get(row["what"]), status="done", app=app, app_label=label, ref=ref)

        try:
            return await self._sql(
                select="SELECT created_at AS _at, ('sent:' || id) AS _key, channel, chat_label, title, what, reason,"
                       " status, error, ref FROM channel_deliveries",
                where=where or ["1"], params=params, ts="created_at", key="('sent:' || id)", build=build,
                cur=cur, n=n, query=query, filtered=False,
            )
        except sqlite3.OperationalError:  # the channels package hasn't made its tables yet
            return [], None

    # -- task runs (tasks, briefs, and monitors that start a run)
    async def _task_runs(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("tasks", "briefs", "monitors"):
            return [], None
        brief = f"{_json('t.original_context', '$.source')} = 'brief'"
        kinds: list[str] = []
        if query.wants("briefs"):
            kinds.append(f"COALESCE({brief}, 0)")
        if query.wants("monitors"):
            kinds.append("t.task_type = 'script'")
        if query.wants("tasks"):
            kinds.append(f"(t.task_type != 'script' AND NOT COALESCE({brief}, 0))")
        where = [f"({' OR '.join(kinds)})"]
        params: list = []
        if query.q:
            where.append("(t.name LIKE ? ESCAPE '\\' OR t.description LIKE ? ESCAPE '\\' OR r.result LIKE ? ESCAPE"
                         " '\\' OR r.error LIKE ? ESCAPE '\\')")
            params += [query.like()] * 4
        tz = self.tz()

        return await self._sql(
            select="SELECT r.created_at AS _at, ('run:' || r.id) AS _key, r.id, r.task_id, r.status, r.trigger_data,"
                   " r.result, r.error, r.retry_of, r.pending_question, r.started_at, r.finished_at, t.name,"
                   " COALESCE(r.dry_run, 0) AS dry_run,"
                   " t.task_type, t.schedule, t.original_context, t.source AS task_source"
                   " FROM task_runs r JOIN tasks t ON t.id = r.task_id",
            where=where, params=params, ts="r.created_at", key="('run:' || r.id)",
            build=lambda row: self._run_item(row, tz), cur=cur, n=n, query=query, filtered=bool(query.app),
        )

    def _run_item(self, row: Any, tz: tzinfo) -> Item:
        name = row["name"] or "Untitled task"
        context = _loads(row["original_context"]) or {}
        schedule = _loads(row["schedule"]) or {}
        result = _loads(row["result"]) or {}
        is_brief = isinstance(context, dict) and context.get("source") == "brief"
        is_script = row["task_type"] == "script"
        kind = "briefs" if is_brief else "monitors" if is_script else "tasks"
        status = str(row["status"] or "")
        apps = [str(a) for a in (result.get("tools_used") or []) if isinstance(result, dict) and isinstance(a, str)]
        trigger = schedule.get("source") if isinstance(schedule, dict) and schedule.get("type") == "triggered" else None
        app = trigger or next((a for a in apps if a not in {"memory", "files", "tasks", "time", "skills"}), None)
        app, label = self._app(app)
        tried = bool(row["dry_run"])  # a try run (#142): nothing in it was sent, changed or run
        if tried:
            done, fail, wait, run = (f"Tried {name}", f"The try run of {name} didn't finish",
                                     f"The try run of {name} is waiting for you", f"Trying {name}")
        elif is_brief:
            done, fail, wait, run = (f"Made your {name}", f"Couldn't make your {name}", f"Your {name} is waiting for you",
                                     f"Making your {name}")
        else:
            done, fail, wait, run = (f"Ran {name}", f"{name} didn't finish", f"{name} is waiting for you",
                                     f"Running {name}")
        title, state = {
            "completed": (done, "done"), "completed_with_errors": (f"{done}, with some errors", "done"),
            "error": (fail, "failed"), "waiting_for_user": (wait, "waiting"), "processing": (run, "running"),
            "cancelled": (f"Stopped {name}", "stopped"),
        }.get(status, (f"{name}: {status.replace('_', ' ')}", None))
        # why it ran
        trigger_data = _loads(row["trigger_data"])
        if tried:
            why = "You chose Try it first, so it only noted what it would send, change or run"
        elif row["retry_of"]:
            why = "You chose Retry"
        elif isinstance(trigger_data, dict) and "script_result" in trigger_data:
            why = "Its check found something: " + words.short(trigger_data.get("message") or "a change", 100)
        elif isinstance(schedule, dict) and schedule.get("type") in {"recurring", "triggered"}:
            why = words.schedule_why(schedule, tz, registry=self.app.registry)
        elif row["task_source"] == "chat":
            why = "You asked for it in a chat"
        elif row["task_source"] == "proactive" or (isinstance(context, dict) and context.get("source") == "proactive"):
            why = "You approved a suggestion"
        else:
            why = words.schedule_why(schedule, tz, registry=self.app.registry)
        detail = None
        if status == "error" and row["error"]:
            detail = words.short(row["error"], 200)
        elif status == "waiting_for_user":
            question = _loads(row["pending_question"]) or {}
            detail = words.short(question.get("question"), 200) if isinstance(question, dict) else None
        elif isinstance(result, dict) and result.get("summary"):
            detail = words.short(result["summary"], 200)
        return Item(id=row["_key"], kind=kind, at=row["_at"], title=title, why=why, detail=detail, status=state,
                    app=app, app_label=label,
                    ref={"type": "task", "id": row["task_id"], "run_id": row["id"]},
                    _apps=frozenset([*apps, *([trigger] if trigger else [])]))

    # -- monitors: each script check
    async def _script_tasks(self) -> list[Any]:
        return await self.store.fetchall(
            "SELECT id, name, schedule, script, enabled, status, next_execution_at, original_context, task_type"
            " FROM tasks WHERE task_type = 'script'"
        )

    async def _script_checks(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("monitors"):
            return [], None
        tz = self.tz()
        items: list[Item] = []
        for row in await self._script_tasks():
            script = _loads(row["script"]) or {}
            schedule = _loads(row["schedule"]) or {}
            checks = script.get("checks") if isinstance(script, dict) else None
            source = schedule.get("source") if isinstance(schedule, dict) and schedule.get("type") == "triggered" else None
            app, label = self._app(source)
            why = words.schedule_why(schedule, tz, verb="checks", registry=self.app.registry)
            name = row["name"] or "Untitled check"
            for index, check in enumerate(checks if isinstance(checks, list) else []):
                if not isinstance(check, dict):
                    continue
                outcome = str(check.get("outcome") or "")
                count = int(check.get("count") or 1)
                note = str(check.get("note") or "")
                title, status = {
                    "baseline": (f"Checked {name} for the first time", "done"),
                    "nothing_new": (f"Checked {name}: nothing new", "done"),
                    "quiet": (f"Checked {name}: nothing to report", "done"),
                    "changed": (f"{name} changed. You were told", "done"),
                    "minor": (f"{name} changed a little. Not worth telling you", "done"),
                    "alert": (f"{name} found something. You were told", "done"),
                    "reported": (f"Checked {name} and told you the result", "done"),
                    "failed": (f"Couldn't check {name}", "failed"),
                }.get(outcome, (f"Checked {name}", "done"))
                if count > 1:
                    title += f" ({count} checks)"
                at = str(check.get("last_at") or check.get("at") or "")
                items.append(Item(
                    id=f"check:{row['id']}:{index:02d}", kind="monitors", at=at, title=title, why=why,
                    detail=words.short(note, 200) or None, status=status, app=app, app_label=label,
                    ref={"type": "task", "id": row["id"]},
                ))
        return [i for i in items if (cur is None or i.key < cur) and query.matches(i)], None

    # -- notifications Sentient sent (except ones another source already shows)
    async def _notifications(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("notifications", "approvals", "tasks", "monitors", "memory", "backups", "cost"):
            return [], None
        where = [
            "kind NOT IN ('brief', 'skill')",  # a brief is its run; a skill change is in the skills log
            # a finished run or a check is shown from the run or the check itself
            f"COALESCE({_json('payload', '$.event')}, '') NOT IN ('run_completed', 'run_failed', 'script_alert')",
            f"{_json('payload', '$.dream_id')} IS NULL",  # the dream itself
        ]
        params: list = []
        if query.q:
            where.append("(title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\')")
            params += [query.like()] * 2
        return await self._sql(
            select="SELECT created_at AS _at, ('note:' || id) AS _key, id, kind, title, body, payload, read"
                   " FROM notifications",
            where=where, params=params, ts="created_at", key="('note:' || id)",
            build=self._notification_item, cur=cur, n=n, query=query, filtered=True,
        )

    def _notification_item(self, row: Any) -> Item | None:
        payload = _loads(row["payload"]) or {}
        payload = payload if isinstance(payload, dict) else {}
        kind = row["kind"]
        title = (row["title"] or "").strip()
        body = row["body"] or ""
        event = str(payload.get("event") or "")
        status = str(payload.get("status") or "")
        ref: dict = {"type": "notification", "id": row["id"]}
        task_id = payload.get("task_id")
        if task_id:
            ref = {"type": "task", "id": task_id, "run_id": payload.get("run_id"), "notification_id": row["id"]}
        app: str | None = None
        item_kind, item_title, why, detail, state = "notifications", f"Told you: {title or words.short(body, 80)}", \
            None, words.short(body, 200), "done"
        if kind == "task" and event == "question":
            task = words.short(title.removesuffix(" needs your answer").removesuffix(" is stuck"), 80)
            answer = str(payload.get("answer") or "")
            said = {"answered": f"You answered: {words.short(answer, 80)}" if answer else "You answered",
                    "expired": "Nobody answered in time, so it stopped", "cancelled": "The run was stopped"}
            approval = payload.get("approval") if isinstance(payload.get("approval"), dict) else None
            if approval is not None:
                label = approval.get("label") or approval.get("tool") or "a tool"
                item_kind, item_title = "approvals", f"Asked you before {task} used {label}"
                why = ("The task had read outside content, so it checks first" if approval.get("untrusted")
                       else "You set it to ask first")
                tool_app = self._tool_app(str(approval.get("tool") or ""))
                app = tool_app
            else:
                item_kind, item_title = "tasks", f"{task} asked you a question"
                why = "It needed your answer to carry on"
            detail = said.get(status) or words.short(payload.get("question") or body, 200)
            declined = status == "answered" and grant_rules.parse_answer(answer, [grant_rules.DENY]) == grant_rules.DENY
            state = "declined" if declined else "waiting" if not status else "done"
        elif kind == "task" and event in {"approval_needed", "clarification_needed"}:
            item_kind = "approvals" if event == "approval_needed" else "tasks"
            item_title = "Asked you to approve a plan" if event == "approval_needed" else "Asked you about a new task"
            why, detail = "New tasks wait for your OK before they run", words.short(body, 200)
        elif kind == "task" and event == "planning_failed":
            item_kind, item_title, state = "tasks", "Couldn't make a plan", "failed"
            why = "A new task needed a plan"
        elif kind == "task" and event in {"script_failed", "script_recovered"}:
            item_kind = "monitors"
            item_title = "A check stopped working" if event == "script_failed" else "A check is working again"
            state = "failed" if event == "script_failed" else "done"
            why = "A watch couldn't run its check" if event == "script_failed" else "It had stopped working before"
        elif kind == "task" and event == "away_check":
            item_kind, item_title = "tasks", "Asked whether to keep your routines running"
            why = "You hadn't used Sentient for a while" if payload.get("reason") != "results" else \
                "Results were piling up while you were away"
            detail = {"kept": "You kept them running", "paused": "Nobody answered, so they paused"}.get(status, detail)
        elif kind == "task" and event == "away_paused":
            count = len(payload.get("task_ids") or [])
            item_kind = "tasks"
            item_title = "Paused your routines while you were away" + (f" ({count})" if count else "")
            why = "Nobody answered the question about keeping them running"
            detail = "You resumed them" if status == "resumed" else "They wait until you choose Resume routines"
            state = "paused" if status != "resumed" else "done"
        elif kind == "task" and event == "caught_up":
            item_kind, item_title = "tasks", title or "Caught up on missed runs"
            why = "The computer was off or asleep when they were due"
        elif kind == "task":
            item_kind, why = "tasks", "An update about one of your tasks"
        elif kind == "proactive":
            suggestion = payload.get("suggestion") if isinstance(payload.get("suggestion"), dict) else {}
            source = suggestion.get("source_event") if isinstance(suggestion.get("source_event"), dict) else {}
            app = str(source.get("source") or "") or None
            if app == "heartbeat":
                app = None
            item_title = "Suggested: " + words.short(suggestion.get("description") or body, 100)
            what = words.short(source.get("summary") or "", 100)
            why = (f"Something in {words.app_label(app, self.app.registry)}" + (f": {what}" if what else "")) if app else \
                "It noticed something you might want done"
            detail = {"approved": "You said yes", "dismissed": "You dismissed it", "expired": "It expired"}.get(status)
            state = {"approved": "done", "dismissed": "declined", "expired": "stopped"}.get(status, "waiting")
        elif kind == "backup":
            item_kind, item_title, state = "backups", "A backup didn't work", "failed"
            why = "Automatic backups are on"
            ref = {"type": "backups"}
        elif kind == "budget":
            item_kind, item_title = "cost", title or "Model budget"
            why = "You set a monthly budget for cloud models"
            ref = {"type": "usage"}
        elif kind == "error":
            state, why = "failed", "Something stopped working and needs you"
        elif payload.get("subagent_id"):
            why = "A helper you started in a chat finished its work"
        elif kind == "info":
            why = "Sentient let you know"
        if item_kind == "notifications" and kind == "info" and "memories" in body.lower():
            item_kind = "memory"
        app_id, label = self._app(app)
        return Item(id=row["_key"], kind=item_kind, at=row["_at"], title=item_title, why=why, detail=detail,
                    status=state, app=app_id, app_label=label, ref=ref)

    # -- memories saved
    async def _facts(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("memory") or query.app:
            return [], None
        where = ["1"]
        params: list = []
        if query.q:
            where.append("content LIKE ? ESCAPE '\\'")
            params.append(query.like())

        def build(row: Any) -> Item:
            source = str(row["source"] or "")
            if source == "conversation":
                why = "You mentioned it in a chat"
            elif source == "manual":
                why = "You added it"
            elif source == "onboarding":
                why = "From what you told Sentient when you set it up"
            elif source.startswith("file:"):
                why = f"From the file {source[5:]}"
            elif source.startswith("task:"):
                why = "A task found it"
            elif source.startswith("import"):
                why = "Imported from another assistant"
            else:
                why = "Sentient learned it"
            pending = row["status"] == "pending"
            return Item(id=row["_key"], kind="memory", at=row["_at"],
                        title="Saved a memory for your review" if pending else "Remembered something about you",
                        why=why, detail=words.short(row["content"], 200), status="waiting" if pending else "done",
                        ref={"type": "memory", "id": str(row["id"]), "review": pending})

        return await self._sql(
            select="SELECT created_at AS _at, ('fact:' || id) AS _key, id, content, source, status FROM facts",
            where=where, params=params, ts="created_at", key="('fact:' || id)", build=build, cur=cur, n=n,
            query=query, filtered=False,
        )

    # -- dreams (nightly memory tidy-up)
    async def _dreams(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("memory") or query.app:
            return [], None
        where = ["1"]
        params: list = []
        if query.q:
            where.append("journal_md LIKE ? ESCAPE '\\'")
            params.append(query.like())
        time_text = words.clock_text(self.app.config.dreaming.time)

        def build(row: Any) -> Item:
            status = row["status"]
            title = {"completed": "Tidied up your memories", "error": "Couldn't finish tidying up your memories",
                     "running": "Tidying up your memories"}.get(status, "Tidied up your memories")
            why = "You asked for it" if row["trigger"] == "manual" else \
                f"It runs each night around {time_text} while you're not chatting"
            journal = str(row["journal_md"] or "").strip()
            detail = words.short(row["error"], 200) if status == "error" and row["error"] else words.short(journal, 200)
            return Item(id=row["_key"], kind="memory", at=row["_at"], title=title, why=why, detail=detail or None,
                        status={"completed": "done", "error": "failed", "running": "running"}.get(status),
                        ref={"type": "dream", "id": row["id"]})

        return await self._sql(
            select="SELECT started_at AS _at, ('dream:' || id) AS _key, id, status, trigger, journal_md, error FROM dreams",
            where=where, params=params, ts="started_at", key="('dream:' || id)", build=build, cur=cur, n=n,
            query=query, filtered=False,
        )

    # -- skills changed (and the profile Sentient keeps about you)
    async def _skills(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("skills", "memory") or query.app:
            return [], None
        where = ["kind NOT IN ('curator_run', 'summary_created')"]
        params: list = []
        if query.q:
            where.append("detail LIKE ? ESCAPE '\\'")
            params.append(query.like())

        def build(row: Any) -> Item | None:
            detail = _loads(row["detail"]) or {}
            detail = detail if isinstance(detail, dict) else {}
            kind = row["kind"]
            name = str(detail.get("name") or "a skill")
            pending = bool(detail.get("pending"))
            by_user = detail.get("author") == "user" or detail.get("by") == "user"
            approved = bool(detail.get("approved"))
            if kind == "profile_updated":
                return Item(id=row["_key"], kind="memory", at=row["_at"], title="Updated what it knows about you",
                            why="It reads your recent chats now and then", status="done", ref={"type": "about"})
            if kind == "skill_created":
                title = (f"You added the skill {name}" if by_user else f"You approved the new skill {name}" if approved
                         else f"Drafted a new skill for your review: {name}" if pending else f"Learned a new skill: {name}")
                why = "You did it" if by_user or approved else "It noticed a repeatable way of doing something"
            elif kind == "skill_patched":
                title = (f"You approved an update to {name}" if approved else f"Drafted an update to {name} for your review"
                         if pending else f"Improved the skill {name}")
                why = "You did it" if approved else "It found a better way to do it"
                if detail.get("user_edited"):
                    why = "You had edited it yourself, so your version stays until you approve"
            elif kind == "skill_repair_proposed":
                title, why = f"Proposed a fix for {name}", "The skill didn't work as written"
            elif kind == "skill_archived":
                title = f"You archived {name}" if by_user else f"Put away {name}"
                idle = detail.get("idle_days")
                why = "You did it" if by_user else (f"It wasn't used for {int(idle)} days" if idle else "It wasn't used")
            elif kind == "skill_reset":
                title, why = f"You reset {name} to its original", "You did it"
            else:
                title, why = f"Changed the skill {name}", None
            reason = str(detail.get("reason") or "")
            return Item(id=row["_key"], kind="skills", at=row["_at"], title=title, why=why,
                        detail=words.short(reason, 200) or None, status="waiting" if pending else "done",
                        ref={"type": "skill", "id": name})

        return await self._sql(
            select="SELECT ts AS _at, ('skill:' || id) AS _key, id, kind, detail FROM evolution_log",
            where=where, params=params, ts="ts", key="('skill:' || id)", build=build, cur=cur, n=n, query=query,
            filtered=True,
        )

    # -- backups and snapshots (files in the backups folders)
    async def _backup_list(self) -> list[dict]:
        """Backups in the backups folder and the safety snapshots (a few files: each one's manifest is read)."""
        from sentient.backup import archive

        def read() -> list[dict]:
            backups = [b for b in archive.list_archives(self.app.backups.folder()) if b["kind"] != "snapshot"]
            snapshots = [s for s in archive.list_archives(archive.snapshots_dir()) if s["kind"] == "snapshot"]
            return [*backups, *snapshots]

        try:
            return await asyncio.to_thread(read)
        except Exception:
            log.debug("activity: could not list backups", exc_info=True)
            return []

    async def _backups_past(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("backups") or query.app:
            return [], None
        schedule = self.app.config.backups.schedule
        items = []
        for entry in await self._backup_list():
            at = str(entry.get("created_at") or "")
            kind = entry.get("kind")
            counts = entry.get("counts") or {}
            detail = ", ".join(f"{v} {k}" for k, v in counts.items() if isinstance(v, int)) or None
            if kind == "snapshot":
                title = "Took a safety copy of your data"
                why = str(entry.get("reason") or "Before a big change").capitalize()
            elif kind == "scheduled":
                title, why = "Backed up everything", f"{'Daily' if schedule == 'daily' else 'Weekly'} backups are on"
            else:
                title, why = "Backed up everything", "You asked for it"
            items.append(Item(id=f"backup:{entry.get('name')}", kind="backups", at=at, title=title, why=why,
                              detail=detail, status="done", ref={"type": "backups", "id": entry.get("name")}))
        return [i for i in items if (cur is None or i.key < cur) and query.matches(i)], None

    # -- "Always allow" answers (kept as notes next to the rule in config)
    async def _always_allowed(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        if not query.wants("approvals"):
            return [], None
        origins = getattr(self.app.config.tools.approvals, "rule_origins", None) or {}
        rules = getattr(self.app.config.tools.approvals, "rules", None) or {}
        items = []
        for key, origin in origins.items():
            data = origin.model_dump() if hasattr(origin, "model_dump") else origin
            if not isinstance(data, dict) or data.get("rule") != "allow" or not data.get("at") or rules.get(key) != "allow":
                continue
            tool = self.app.registry.get(key)
            app, label = self._app(tool.plugin if tool is not None else key)
            what = f'"{tool_title(tool)}" in {label}' if tool is not None else (label or key)
            via = data.get("via") or "desktop"
            where = "in the app" if via == "desktop" else f"on {words.app_label(via)}"
            source = "an approval request" if data.get("source") == "approval" else "a chat"
            items.append(Item(id=f"rule:{key}", kind="approvals", at=str(data["at"]), title=f"You chose Always allow for {what}",
                              why=f"You answered {source} {where}", status="done", app=app, app_label=label,
                              ref={"type": "approvals"}))
        return [i for i in items if (cur is None or i.key < cur) and query.matches(i)], None

    # -- what paid cloud models cost, one line per day with spending
    async def _cost(self, cur: Cursor | None, n: int, query: Query) -> tuple[list[Item], Cursor | None]:
        """One line per local day with paid calls, at that day's last paid call. Reads the paid calls newest first
        along ``idx_usage_paid``, from the end of the cursor's day (that day's line sits at its newest call), and
        stops once ``n`` days are complete."""
        if not query.wants("cost") or query.app or query.q:
            return [], None
        tz = self.tz()
        upper = "9999"
        if cur is not None and (mark := parse_iso(cur[0])) is not None:
            local = mark.astimezone(tz)
            upper = iso((local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(UTC)) or upper
        items: list[Item] = []
        day: dict[str, Any] | None = None

        def flush() -> None:
            if day is None:
                return
            calls = day["calls"]
            top = max(day["models"].items(), key=lambda kv: kv[1])[0]
            item = Item(id=f"cost:{day['day']}", kind="cost", at=day["at"],
                        title=f"Spent {words.money(day['cost'])} on cloud models",
                        why="Paid models were used for chats or tasks that day",
                        detail=f"{calls} paid call{'s' if calls != 1 else ''}, most on {top}", status="done",
                        ref={"type": "usage"})
            if cur is None or item.key < cur:
                items.append(item)

        async with self.store.db.execute(
            "SELECT created_at, model, cost_usd FROM usage WHERE cost_usd > 0 AND created_at < ?"
            " ORDER BY created_at DESC", (upper,),
        ) as rows:
            while len(items) <= n:
                batch = await rows.fetchmany(500)
                if not batch:
                    flush()
                    return items, None
                for row in batch:
                    stamp = parse_iso(row["created_at"])
                    if stamp is None:
                        continue
                    key = stamp.astimezone(tz).date().isoformat()
                    if day is None or day["day"] != key:
                        flush()
                        if len(items) > n:
                            break
                        day = {"day": key, "at": row["created_at"], "cost": 0.0, "calls": 0, "models": {}}
                    day["cost"] += float(row["cost_usd"])
                    day["calls"] += 1
                    day["models"][row["model"]] = day["models"].get(row["model"], 0.0) + float(row["cost_usd"])
        return items, items[-1].key

    # ------------------------------------------------------------------ the future
    async def upcoming(
        self, *, days: int = UPCOMING_DAYS, kinds: Iterable[str] | None = None, app: str | None = None,
        q: str | None = None,
    ) -> dict:
        """``{items, paused}``: what needs the user now and what is scheduled in the next ``days`` days, soonest
        first; ``paused`` lists routines that would run but are paused."""
        query = self.query(kinds, app, q)
        days = max(1, min(int(days or UPCOMING_DAYS), 31))
        now = datetime.now(UTC)
        until = now + timedelta(days=days)
        tz = self.tz()
        items: list[Item] = []
        paused: list[Item] = []
        items += await self._waiting(now)
        scheduled, stopped = await self._scheduled(now, until, tz)
        items += scheduled
        paused += stopped
        items += await self._next_backup(now, until)
        items += await self._next_dream(now, until, tz)
        items += await self._grants_ending(now, until)
        items = [i for i in items if query.matches(i)]
        paused = [i for i in paused if query.matches(i)]
        items.sort(key=lambda i: (0 if i.status == "waiting" else 1, i.at or "", i.id))
        paused.sort(key=lambda i: (i.title.lower(), i.id))
        return {"items": [i.as_dict() for i in items], "paused": [i.as_dict() for i in paused]}

    async def _waiting(self, now: datetime) -> list[Item]:
        out: list[Item] = []
        try:
            questions = await self.app.tasks.waiting_questions()
        except Exception:
            questions = []
        for q in questions:
            approval = q.get("kind") == "approval"
            out.append(Item(
                id=f"wait:{q['run_id']}", kind="approvals" if approval else "tasks", at=q.get("asked_at") or now.isoformat(),
                title=(f"{q['task_name']} is waiting for your OK" if approval else f"{q['task_name']} is waiting for your answer"),
                why="It asked before going on" + (" and stops if nobody answers in time" if q.get("expires_at") else ""),
                detail=words.short(q.get("question"), 200), status="waiting",
                ref={"type": "task", "id": q["task_id"], "run_id": q["run_id"]}, upcoming=True,
            ))
        rows = await self.store.fetchall(
            "SELECT id, name, updated_at FROM tasks WHERE status = 'approval_pending' ORDER BY updated_at"
        )
        for row in rows:
            out.append(Item(id=f"plan:{row['id']}", kind="approvals", at=row["updated_at"],
                            title=f"The plan for {row['name'] or 'a task'} is waiting for your OK",
                            why="New tasks wait for your OK before they run", status="waiting",
                            ref={"type": "task", "id": row["id"]}, upcoming=True))
        return out

    async def _scheduled(self, now: datetime, until: datetime, tz: tzinfo) -> tuple[list[Item], list[Item]]:
        rows = await self.store.fetchall(
            "SELECT id, name, status, enabled, schedule, next_execution_at, task_type, original_context, script"
            " FROM tasks WHERE status IN ('active', 'pending', 'processing') AND schedule IS NOT NULL"
        )
        try:
            away = await self.app.tasks.away.status()
            away_ids = set(away.get("task_ids") or []) if away.get("state") == "paused" else set()
        except Exception:
            away_ids = set()
        items: list[Item] = []
        paused: list[Item] = []
        for row in rows:
            schedule = _loads(row["schedule"]) or {}
            if not isinstance(schedule, dict):
                continue
            context = _loads(row["original_context"]) or {}
            is_brief = isinstance(context, dict) and context.get("source") == "brief"
            is_script = row["task_type"] == "script"
            kind = "briefs" if is_brief else "monitors" if is_script else "tasks"
            name = row["name"] or "Untitled task"
            title = f"Will make your {name}" if is_brief else f"Will check {name}" if is_script else f"Will run {name}"
            verb = "checks" if is_script else "runs"
            why = words.schedule_why(schedule, tz, verb=verb, registry=self.app.registry)
            trigger = schedule.get("source") if schedule.get("type") == "triggered" else None
            app, label = self._app(trigger)
            ref = {"type": "task", "id": row["id"]}
            if not row["enabled"]:
                if schedule.get("type") in {"recurring", "triggered"}:
                    reason = "Paused while you were away" if row["id"] in away_ids else "You paused it"
                    paused.append(Item(id=f"paused:{row['id']}", kind=kind, at=None, title=f"{name} is paused",
                                       why=reason, detail=why, status="paused", app=app, app_label=label, ref=ref,
                                       upcoming=True))
                continue
            if schedule.get("type") == "triggered":
                continue  # nothing to put on a clock
            first = parse_iso(row["next_execution_at"])
            if first is None or first > until:
                continue
            times = [max(first, now)]
            interval = str(schedule.get("frequency") or "").lower() == "interval"
            if schedule.get("type") == "recurring" and not interval:
                while len(times) < MAX_PER_TASK:
                    nxt = calculate_next_run(schedule, times[-1])
                    if nxt is None or nxt > until:
                        break
                    times.append(nxt)
            for index, at in enumerate(times):
                items.append(Item(id=f"next:{row['id']}:{index}", kind=kind, at=iso(at), title=title, why=why,
                                  status="scheduled", app=app, app_label=label, ref=ref, upcoming=True,
                                  detail="The exact minute is picked inside that time" if index and schedule.get("window")
                                  else None))
        return items, paused

    async def _next_backup(self, now: datetime, until: datetime) -> list[Item]:
        from sentient.backup.service import SCHEDULE_EVERY, SCHEDULE_META

        schedule = self.app.config.backups.schedule
        every = SCHEDULE_EVERY.get(schedule)
        if every is None:
            return []
        last = parse_iso(await self.store.get_meta(SCHEDULE_META))
        at = max(now, last + every) if last else now + timedelta(minutes=5)
        if at > until:
            return []
        return [Item(id="next:backup", kind="backups", at=iso(at), title="Will back up everything",
                     why=f"{'Daily' if schedule == 'daily' else 'Weekly'} backups are on", status="scheduled",
                     ref={"type": "backups"}, upcoming=True)]

    async def _next_dream(self, now: datetime, until: datetime, tz: tzinfo) -> list[Item]:
        cfg = self.app.config.dreaming
        if not cfg.enabled or self.app.memory is None:
            return []
        hour, minute = (int(p) for p in (cfg.time or "03:00").split(":")[:2])
        local = now.astimezone(tz)
        at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        today = local.date().isoformat()
        row = await self.store.fetchone(
            "SELECT started_at FROM dreams WHERE trigger = 'schedule' ORDER BY started_at DESC LIMIT 1"
        )
        ran_today = row is not None and (parse_iso(row["started_at"]) or now).astimezone(tz).date().isoformat() == today
        if at <= local or ran_today:
            at += timedelta(days=1)
        if at.astimezone(UTC) > until:
            return []
        return [Item(id="next:dream", kind="memory", at=iso(at.astimezone(UTC)), title="Will tidy up your memories",
                     why=f"It runs each night around {words.clock(hour, minute)} while you're not chatting",
                     status="scheduled", ref={"type": "dream"}, upcoming=True)]

    async def _grants_ending(self, now: datetime, until: datetime) -> list[Item]:
        grants = self.app.approvals.grants
        if grants is None:
            return []
        try:
            live = await grants.list()
        except Exception:
            return []
        out: list[Item] = []
        for g in live:
            if g.get("scope") != "timed" or not g.get("expires_at"):
                continue
            ends = parse_iso(g["expires_at"])
            if ends is None or ends > until:
                continue
            tool = self.app.registry.get(g["tool"])
            app, label = self._app(tool.plugin if tool is not None else None)
            out.append(Item(id=f"grant:{g['id']}", kind="approvals", at=iso(ends),
                            title=f"Your OK for {g['label']} ends", why="You allowed it for a while",
                            detail="After this Sentient asks again", status="scheduled", app=app, app_label=label,
                            ref={"type": "approvals"}, upcoming=True))
        return out

    # ------------------------------------------------------------------ filter choices
    async def apps(self) -> list[dict]:
        """The apps the timeline can be filtered by: tools used in chats, messaging apps and task triggers."""
        found: set[str] = set()
        rows = await self.store.fetchall(
            "SELECT DISTINCT m.name FROM messages m INDEXED BY idx_messages_tool_time JOIN sessions s ON s.id = m.session_id"
            " WHERE m.role = 'tool' AND s.incognito = 0 AND m.created_at > ?",
            ((datetime.now(UTC) - timedelta(days=90)).isoformat(),),
        )
        for row in rows:
            plugin = self._tool_app(str(row["name"] or ""))
            if plugin:
                found.add(plugin)
        rows = await self.store.fetchall("SELECT DISTINCT channel FROM sessions WHERE incognito = 0")
        found |= {r["channel"] for r in rows if r["channel"] in CHANNELS}
        try:
            rows = await self.store.fetchall("SELECT DISTINCT channel FROM channel_deliveries")
            found |= {r["channel"] for r in rows if r["channel"] in CHANNELS}
        except sqlite3.OperationalError:
            pass
        rows = await self.store.fetchall(
            f"SELECT DISTINCT {_json('schedule', '$.source')} AS source FROM tasks"
            f" WHERE {_json('schedule', '$.type')} = 'triggered'"
        )
        found |= {r["source"] for r in rows if r["source"]}
        found -= {"memory", "files", "time", "tasks", "skills"}
        return sorted(({"id": a, "label": words.app_label(a, self.app.registry)} for a in found),
                      key=lambda a: str(a["label"]).lower())


def _brief(arguments: Any) -> str:
    """Up to two short ``key: value`` pairs of a call's arguments."""
    if not isinstance(arguments, dict):
        return ""
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str | int | float) and not isinstance(value, bool) and str(value).strip():
            parts.append(f"{str(key).replace('_', ' ')}: {words.short(value, 60)}")
        if len(parts) == 2:
            break
    return ", ".join(parts)
