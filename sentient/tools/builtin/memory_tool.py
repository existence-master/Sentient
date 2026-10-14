"""Memory tools: the assistant's explicit handle on what it knows about the user.

Ports the v2 memory MCP (search, cud, search by source) and the v2 history MCP
(semantic search over conversation summaries, time-based search over messages).
"""

from __future__ import annotations

import json
import re

from sentient.memory import review
from sentient.memory.episodic import EpisodicMemory
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import brings_untrusted

INCOGNITO_REFUSAL = (
    "Not saved: this is an incognito chat, so nothing from it is remembered. Don't tell the user you will remember it."
)
INSTRUCTION_NOTE = (
    "Saved for the user to review first, because it reads like an instruction for how Sentient should act. It is not "
    "used until they approve it. Tell the user it is waiting in their memory review."
)
HELD_NOTE = (
    "Saved for the user to review first, because this run read outside content or nobody asked for it. "
    "It is not used until they approve it."
)


async def _memory_closed(ctx: ToolContext) -> bool:
    """An incognito chat with ``chat.incognito_uses_memory`` off reads nothing Sentient knows (#154)."""
    return not ctx.config.chat.incognito_uses_memory and await ctx.is_incognito()


def _episodic(ctx: ToolContext) -> EpisodicMemory:
    if ctx.memory is not None and getattr(ctx.memory, "episodic", None) is not None:
        return ctx.memory.episodic
    return EpisodicMemory(ctx.store, ctx.llm, ctx.config)


def _fts_query(text: str, joiner: str) -> str:
    tokens = re.findall(r"\w+", text)
    return joiner.join(f'"{t}"' for t in tokens[:12])


@tool("memory_recall", risk=Risk.read)
async def memory_recall(ctx: ToolContext, query: str, limit: int = 8) -> list[dict]:
    """Search what you remember about the user. Use when a question depends on personal
    context (their preferences, people, projects, routines) that is not already in your prompt."""
    if ctx.memory is None or await _memory_closed(ctx):
        return []
    facts = await ctx.memory.recall(query, top_k=limit)
    return [
        {"id": f["id"], "fact": f["content"], "topics": f["topics"], "similarity": f["similarity"]} for f in facts
    ]


@tool("memory_remember", risk=Risk.write, internal=True)
async def memory_remember(ctx: ToolContext, fact: str) -> dict:
    """Save one atomic fact about the user in third person (e.g. "Maya's sister lives in Pune").
    Call this whenever you learn something lasting about the user. One fact per call.
    Existing facts on the same subject are updated instead of duplicated."""
    if ctx.memory is None:
        return {"error": "memory unavailable"}
    if await ctx.is_incognito():  # nothing is kept from an incognito chat (#154)
        return {"error": INCOGNITO_REFUSAL}
    held = review.for_context(ctx)  # outside content or unprompted work: the user reviews it first (ADR 0021)
    if held is not None:
        held["snippet"] = review.note("", await _last_outside_result(ctx))["snippet"]
    # in a chat, the user's own words this turn: a fact they back is kept even after outside content, and an
    # instruction's review card shows what the user said (#234)
    own = None if review.is_unprompted_note(held) else await _own_words(ctx)
    out = await ctx.memory.remember(
        fact, source="conversation", notify=True, review=held, own_words=own, session_id=ctx.session_id
    )
    if out.get("status") == "pending":
        return {**out, "note": INSTRUCTION_NOTE if out.get("reason") == review.INSTRUCTION else HELD_NOTE}
    return out


OWN_WORDS_ROWS = 40  # recent rows read to find this turn's user messages


async def _own_words(ctx: ToolContext) -> str | None:
    """What the user typed or said in this chat turn: the user messages (and steer messages) since the last reply.
    Read from the stored chat, never from the model, so nothing it read can pass for the user's words."""
    if not ctx.session_id or getattr(ctx, "origin", "user") != "user":
        return None
    rows = await ctx.store.recent_messages(ctx.session_id, OWN_WORDS_ROWS)
    words: list[str] = []
    for row in reversed(rows):
        if row.get("role") == "assistant" and not row.get("tool_calls"):
            break  # the previous turn's reply
        if row.get("role") == "user":
            if str(row.get("content") or "").strip():
                words.append(str(row["content"]))
            if not row.get("interjection"):
                break  # the message that started this turn: nothing before it is this turn's
    return "\n".join(reversed(words)) or None


async def _mark_if_outside(ctx: ToolContext, session_ids: list) -> None:
    """Messages from a chat that read outside content bring that content into this run too (ADR 0018, 0021): mark it,
    so sends ask and what it learns waits for review. Summaries of such chats are not searchable here at all."""
    ids = sorted({str(s) for s in session_ids if s and s != ctx.session_id})
    if ctx.untrusted or not ids:
        return
    row = await ctx.store.fetchone(
        f"SELECT untrusted FROM sessions WHERE id IN ({','.join('?' * len(ids))}) AND COALESCE(untrusted, '') != ''"
        " LIMIT 1",
        ids,
    )
    if row is not None:
        ctx.untrusted = str(row["untrusted"])


async def _last_outside_result(ctx: ToolContext) -> str | None:
    """The latest successful tool result in this chat that brought in outside content (the review card shows it).
    Errors and declined calls are skipped: they say nothing about where a fact came from.

    SQL does only the cheap, index-backed part (session, tool role, the ``name IN`` allow-list and the newest-first
    ordering); it carries no fixed window, so any number of failed calls after the real read never buries it (#270).
    The success/failure decision stays in Python (``_failed``), preserving the exact truthiness semantics the old
    code had: ``{"error": false}`` / ``{"error": 0}`` / ``{"error": {}}`` are successes, and ``{"declined": "yes"}``
    is a decline. Rows are read in pages so a long chat never loads at once."""
    registry = ctx.extra.get("registry")
    if not ctx.session_id or registry is None or not ctx.untrusted:
        return None
    names = [t.name for t in registry.tools(include_hidden=True) if brings_untrusted(t)]
    if not names:
        return None
    placeholders = ",".join("?" * len(names))
    sql = (
        f"SELECT content FROM messages WHERE session_id = ? AND role = 'tool' AND name IN ({placeholders})"
        " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
    )
    offset = 0
    while True:
        try:
            rows = await ctx.store.fetchall(sql, (ctx.session_id, *names, 50, offset))
        except Exception:
            return None
        if not rows:
            return None
        for r in rows:
            if not _failed(r["content"]):
                return r["content"]
        offset += len(rows)


def _failed(content: str | None) -> bool:
    """A stored tool result that is an error or a declined call (``{"error": ...}`` or ``{"declined": true}``)."""
    try:
        data = json.loads(content or "")
    except ValueError:
        return False
    return isinstance(data, dict) and bool(data.get("error") or data.get("declined"))


@tool("memory_forget", risk=Risk.send)
async def memory_forget(ctx: ToolContext, fact_id: int) -> dict:
    """Permanently delete a remembered fact by id (from memory_recall). Use when the user asks you to forget something."""
    if ctx.memory is None:
        return {"error": "memory unavailable"}
    ok = await ctx.memory.forget(fact_id, notify=True)
    return {"deleted": ok, "id": fact_id}


@tool("memory_search_by_source", risk=Risk.read)
async def memory_search_by_source(ctx: ToolContext, query: str, source: str, limit: int = 5) -> list[dict]:
    """Search remembered facts that came from one source, e.g. "file:resume.pdf", "onboarding",
    "conversation" or "manual". Use when the user asks what you learned from a specific document."""
    if ctx.memory is None or await _memory_closed(ctx):
        return []
    facts = await ctx.memory.search_by_source(query, source, top_k=limit)
    return [{"id": f["id"], "fact": f["content"], "similarity": f["similarity"]} for f in facts]


@tool("memory_search_history", risk=Risk.read)
async def memory_search_history(ctx: ToolContext, query: str, limit: int = 8) -> list[dict]:
    """Keyword search over past conversations. Use when the user refers to something
    discussed earlier ("that link you sent", "what did we decide about X")."""
    if await _memory_closed(ctx):
        return []
    rows = []
    for joiner in (" ", " OR "):
        q = _fts_query(query, joiner)
        if not q:
            return []
        rows = await ctx.store.search_messages(q, limit=limit)
        if rows:
            break
    await _mark_if_outside(ctx, [r.get("session_id") for r in rows])
    return [
        {"when": r["created_at"], "role": r["role"], "session_id": r["session_id"], "text": (r["content"] or "")[:500]}
        for r in rows
    ]


@tool("history_semantic_search", risk=Risk.read)
async def history_semantic_search(ctx: ToolContext, query: str, limit: int = 5) -> dict:
    """Meaning-based search of summaries of older conversations. Use for topics, ideas or past
    decisions when you do not know when they were discussed ("what did we decide about the marketing plan?")."""
    if await _memory_closed(ctx):
        return {"result": "No relevant conversation summaries found."}
    hits = await _episodic(ctx).search(query, limit=limit)
    if not hits:
        return {"result": "No relevant conversation summaries found."}
    return {
        "summaries": [
            {"summary": h["content"], "from": h["start_at"], "to": h["end_at"], "session_id": h["session_id"]}
            for h in hits
        ]
    }


@tool("history_time_search", risk=Risk.read)
async def history_time_search(ctx: ToolContext, start_date: str, end_date: str, limit: int = 200) -> dict:
    """Every message exchanged in a date range. Use when the user asks about a specific day or period
    ("what did we talk about yesterday afternoon?"). Dates are ISO 8601, e.g. '2026-09-14T00:00:00Z'
    or '2026-09-14' (a bare end date includes that whole day)."""
    if await _memory_closed(ctx):
        return {"result": "No messages found in that time period."}
    try:
        rows = await _episodic(ctx).messages_between(start_date, end_date, limit=limit)
    except ValueError:
        return {"error": "Invalid date format. Use ISO 8601, e.g. 2026-09-14T00:00:00Z."}
    if not rows:
        return {"result": "No messages found in that time period."}
    await _mark_if_outside(ctx, [r.get("session_id") for r in rows])
    log = "\n".join(f"[{r['created_at'][:16].replace('T', ' ')}] {r['role']}: {(r['content'] or '')[:600]}" for r in rows)
    return {"conversation": log, "count": len(rows)}


@tool("user_model_ask", risk=Risk.read)
async def user_model_ask(ctx: ToolContext, question: str) -> dict:
    """Predict what the user would want or prefer, from the insights and facts you have learned about them.
    Use before choosing on their behalf ("which option would they pick?", "how do they like reports written?")."""
    if await _memory_closed(ctx):
        return {"error": "This incognito chat doesn't use what Sentient knows about the user."}
    app = ctx.extra.get("app")
    um = getattr(app, "user_model", None)
    if um is None or not hasattr(um, "ask"):
        return {"error": "user model unavailable"}
    return await um.ask(question)


class MemoryPlugin(ToolPlugin):
    id = "memory"
    display_name = "Memory"
    description = "Long-term memory about you: facts, preferences, documents you shared, and past conversations."
    category = "core"
    icon = "IconBrain"
    selection_hint = "personal facts, preferences, people, documents, past conversations"
    tools = [
        memory_recall,
        memory_remember,
        memory_forget,
        memory_search_by_source,
        memory_search_history,
        history_semantic_search,
        history_time_search,
        user_model_ask,
    ]


PLUGIN = MemoryPlugin()
