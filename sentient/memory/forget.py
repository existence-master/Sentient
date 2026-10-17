"""Disconnect and forget (#153): "Also forget what Sentient learned from it".

Disconnecting Gmail, IMAP, Home Assistant or another app, an MCP server, a messaging app, a device or a folder of notes
can also forget what Sentient learned from it. :func:`preview` says how much that is before the user decides, and
:func:`forget` does it. Without the box ticked nothing here runs and nothing is deleted.

What goes, by source (``memory.origins``):

- every kind: facts only that source backs, with their vectors and merged copies. A fact another source backs too
  loses this source and keeps the others (``FactMemory.purge_source``). Insights about the user that Sentient inferred
  only from what goes (its evidence: those facts, chats and summaries) go too; one with other evidence keeps that;
- a messaging app (``channel:<id>``) or a device (``device:<node id>``): its chats with all their messages, and the
  summaries and summary vectors made of them. Paired chats stay paired, and a new chat starts on the next message;
- an app (``app:<plugin id>``): what its tools returned in any chat (the tool result is replaced by a short note and
  it leaves the replies' sources), summaries of chats that read it, and what proactivity and change feeds kept from it
  (cached emails and events, suggestions made from them, cursors and seen ids);
- a folder (``folder:<id>``): removing the folder already drops its index of notes; the box adds the facts.

Every forget is kept in ``memory_forgets`` for the Activity timeline. Nothing here calls a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentient.memory import origins
from sentient.memory.schema import ensure_memory_schema
from sentient.store.db import new_id, now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

EMAIL_APPS = frozenset({"gmail", "email_imap"})
FOLLOWUP_SEEN_SOURCE = "followups"  # proactivity.service: follow-ups keep "<app>:<key>" ids there
FORGOTTEN = "This result was forgotten when you disconnected {label}."


@dataclass
class Plan:
    source: str
    sessions: list[str] = field(default_factory=list)
    messages: int = 0
    message_ids: list[str] = field(default_factory=list)
    summaries: list[tuple[int, str]] = field(default_factory=list)  # (rowid, id)
    results: list[int] = field(default_factory=list)  # message rowids of tool results to forget
    answers: list[int] = field(default_factory=list)  # message rowids of replies whose sources name the app
    items: int = 0


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _tool_names(app: SentientApp, plugin: str) -> set[str]:
    return {t.name for t in app.registry.tools(include_hidden=True) if t.plugin == plugin}


def _names_app(raw: Any, plugin: str) -> bool:
    try:
        items = json.loads(raw or "[]")
    except ValueError:
        return False
    return any(isinstance(s, dict) and s.get("app") == plugin for s in items if isinstance(items, list))


async def _count(app: SentientApp, sql: str, params: tuple = ()) -> int:
    try:
        row = await app.store.fetchone(sql, params)
    except Exception:  # a package's table isn't there (an older test store)
        return 0
    return int(row[0] or 0) if row else 0


async def _plan(app: SentientApp, source: str) -> Plan:
    kind, ident = origins.split(source)
    store = app.store
    plan = Plan(source)
    if kind in {"channel", "device"}:
        column = "channel" if kind == "channel" else "node_id"
        rows = await store.fetchall(f"SELECT id FROM sessions WHERE {column} = ? ORDER BY created_at", (ident,))
        plan.sessions = [str(r["id"]) for r in rows]
        if plan.sessions:
            marks = _marks(len(plan.sessions))
            rows = await store.fetchall(f"SELECT id FROM messages WHERE session_id IN ({marks})", tuple(plan.sessions))
            plan.message_ids = [str(r["id"]) for r in rows]
            plan.messages = len(plan.message_ids)
            rows = await store.fetchall(
                f"SELECT rowid, id FROM summaries WHERE session_id IN ({marks})", tuple(plan.sessions)
            )
            plan.summaries = [(int(r["rowid"]), str(r["id"])) for r in rows]
    elif kind == "app":
        names = _tool_names(app, ident)
        prefix = f"{ident}_"
        rows = await store.fetchall(
            "SELECT rowid, session_id, name FROM messages WHERE role = 'tool' AND name IS NOT NULL"
        )
        read: set[str] = set()
        for r in rows:
            name = str(r["name"] or "")
            if name in names or name.startswith(prefix):
                plan.results.append(int(r["rowid"]))
                read.add(str(r["session_id"]))
        rows = await store.fetchall(
            "SELECT rowid, session_id, sources FROM messages WHERE role = 'assistant' AND sources LIKE ?",
            (f'%"{ident}"%',),
        )
        for r in rows:
            if _names_app(r["sources"], ident):
                plan.answers.append(int(r["rowid"]))
                read.add(str(r["session_id"]))
        label = _display_name(app, ident)
        params: list[Any] = [label]
        where = "untrusted = ?"
        if read:
            where += f" OR session_id IN ({_marks(len(read))})"
            params += sorted(read)
        rows = await store.fetchall(f"SELECT rowid, id FROM summaries WHERE {where}", tuple(params))
        plan.summaries = [(int(r["rowid"]), str(r["id"])) for r in rows]
        plan.items = await _count(app, "SELECT COUNT(*) FROM proactive_seen WHERE source = ?", (ident,))
    return plan


async def _insights(app: SentientApp, plan: Plan, fact_ids: list[int], *, apply: bool) -> int:
    """User-model insights Sentient inferred only from what goes (#153): their evidence is the forgotten chats'
    messages, summaries and facts. Such an insight is deleted; one with other evidence too keeps that evidence. An
    insight the user confirmed or wrote stays. Returns how many are (or would be) deleted."""
    gone = {("message", m) for m in plan.message_ids} | {("summary", s) for _, s in plan.summaries}
    gone |= {("fact", str(f)) for f in fact_ids}
    if not gone:
        return 0
    try:
        rows = await app.store.fetchall("SELECT id, evidence FROM user_insights WHERE source = 'inferred'")
    except Exception:  # no user model tables yet
        return 0
    deleted = 0
    for r in rows:
        try:
            evidence = json.loads(r["evidence"] or "[]")
        except ValueError:
            continue
        if not isinstance(evidence, list) or not evidence:
            continue
        kept = [e for e in evidence if not (isinstance(e, dict) and (e.get("kind"), str(e.get("ref"))) in gone)]
        if len(kept) == len(evidence):
            continue
        if not kept:
            deleted += 1
            if apply:
                await app.user_model.delete_insight(str(r["id"]))
        elif apply:
            await app.store.execute("UPDATE user_insights SET evidence = ? WHERE id = ?", (json.dumps(kept), r["id"]))
    return deleted


def _display_name(app: SentientApp, plugin: str) -> str:
    p = app.registry.plugin(plugin)
    if p is None and getattr(app, "integrations", None) is not None:
        p = app.integrations.plugin(plugin)
    return str(getattr(p, "display_name", "") or plugin)


async def preview(app: SentientApp, source: str) -> dict[str, Any]:
    """How much forgetting ``source`` would delete, before the user ticks the box. Nothing changes."""
    kind, ident = origins.split(source)
    if not kind:
        raise ValueError(f"not a source you can disconnect: {source}")
    plan = await _plan(app, source)
    doomed: list[int] = []
    facts = await app.memory.source_counts(source, doomed) if app.memory is not None else {"memories": 0, "shared": 0}
    insights = await _insights(app, plan, doomed, apply=False)
    return {
        "source": source,
        "memories": facts["memories"],
        "shared": facts["shared"],
        "chats": len(plan.sessions),
        "messages": plan.messages,
        "summaries": len(plan.summaries),
        "results": len(plan.results),
        "items": plan.items,
        "insights": insights,
        "item_noun": "emails" if ident in EMAIL_APPS else "items",
    }


async def _delete_summaries(app: SentientApp, summaries: list[tuple[int, str]]) -> None:
    if not summaries:
        return
    await app.store.execute(
        f"DELETE FROM summaries WHERE id IN ({_marks(len(summaries))})", tuple(s[1] for s in summaries)
    )
    if app.memory is not None:
        for rowid, _ in summaries:
            await app.memory.episodic.vec.delete(rowid)


async def _forget_results(app: SentientApp, plan: Plan, plugin: str, label: str) -> None:
    """Replace what the app's tools returned with a short note, keeping the keyword index in step (it has no update
    trigger), and take the app out of the replies' sources."""
    store = app.store
    note = json.dumps({"forgotten": FORGOTTEN.format(label=label)})
    for rowid in plan.results:
        row = await store.fetchone("SELECT content FROM messages WHERE rowid = ?", (rowid,))
        if row is None:
            continue
        await store.execute(
            "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', ?, ?)", (rowid, row["content"])
        )
        await store.execute("UPDATE messages SET content = ? WHERE rowid = ?", (note, rowid))
        await store.execute("INSERT INTO messages_fts(rowid, content) VALUES(?, ?)", (rowid, note))
    for rowid in plan.answers:
        row = await store.fetchone("SELECT sources FROM messages WHERE rowid = ?", (rowid,))
        try:
            items = json.loads(row["sources"] or "[]") if row else []
        except ValueError:
            continue
        kept = [s for s in items if not (isinstance(s, dict) and s.get("app") == plugin)]
        await store.execute("UPDATE messages SET sources = ? WHERE rowid = ?", (json.dumps(kept), rowid))


async def _forget_app_cache(app: SentientApp, plugin: str) -> None:
    store = app.store
    for sql, params in (
        ("DELETE FROM proactive_seen WHERE source = ?", (plugin,)),
        ("DELETE FROM proactive_seen WHERE source = ? AND item_id LIKE ?", (FOLLOWUP_SEEN_SOURCE, f"{plugin}:%")),
        # a suggestion still waiting for the user stays, without the notes it was made from
        ("UPDATE proactive_suggestions SET context = NULL WHERE source = ? AND status = 'pending'", (plugin,)),
        ("DELETE FROM proactive_suggestions WHERE source = ? AND status != 'pending'", (plugin,)),
        ("DELETE FROM proactive_sources WHERE source = ?", (plugin,)),
        ("DELETE FROM integration_seen WHERE source = ?", (plugin,)),
        ("DELETE FROM integration_poll_state WHERE source = ?", (plugin,)),
        ("DELETE FROM integration_feed_state WHERE source = ?", (plugin,)),
    ):
        try:
            await store.execute(sql, params)
        except Exception as exc:  # a package's table isn't there (an older test store)
            log.debug("forgetting %s: %s skipped: %s", plugin, sql.split(" WHERE")[0], exc)


async def forget(app: SentientApp, source: str, label: str) -> dict[str, Any]:
    """Forget everything learned from ``source`` (see the module docstring). ``label`` is its plain name ("Telegram",
    "Gmail", the device's or folder's name) for the Activity timeline. Returns what was deleted, in
    :func:`preview`'s words."""
    kind, ident = origins.split(source)
    if not kind:
        raise ValueError(f"not a source you can disconnect: {source}")
    await ensure_memory_schema(app.store)
    plan = await _plan(app, source)
    facts = {"deleted": 0, "unlinked": 0}
    gone: list[int] = []
    if app.memory is not None:
        facts = await app.memory.purge_source(source, gone)
        if facts["deleted"] or facts["unlinked"]:
            app.memory.publish("DELETE", None, None, source=source, count=facts["deleted"], unlinked=facts["unlinked"],
                               reason="forgotten")
    insights = await _insights(app, plan, gone, apply=True)  # before the evidence it reads goes
    await _delete_summaries(app, plan.summaries)
    for sid in plan.sessions:
        await app.delete_chat(sid, reason="forgotten")
    if kind == "app":
        await _forget_results(app, plan, ident, label)
        await _forget_app_cache(app, ident)
    counts = {
        "memories": facts["deleted"],
        "shared": facts["unlinked"],
        "chats": len(plan.sessions),
        "messages": plan.messages,
        "summaries": len(plan.summaries),
        "results": len(plan.results),
        "items": plan.items,
        "insights": insights,
        "item_noun": "emails" if ident in EMAIL_APPS else "items",
    }
    entry = {"id": new_id(), "source": source, "label": label, "counts": counts, "at": now_iso()}
    await app.store.execute(
        "INSERT INTO memory_forgets(id, source, label, counts, at) VALUES(?,?,?,?,?)",
        (entry["id"], source, label, json.dumps(counts), entry["at"]),
    )
    app.bus.publish("memory.forgotten", entry)
    log.info("forgot what %s taught: %s", source, counts)
    return {"source": source, **counts}
