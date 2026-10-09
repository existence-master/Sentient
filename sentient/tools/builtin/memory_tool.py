"""Memory tools: the assistant's explicit handle on what it knows about the user.

Ports the v2 memory MCP (search, cud, search by source) and the v2 history MCP
(semantic search over conversation summaries, time-based search over messages).
"""

from __future__ import annotations

import re

from sentient.memory.episodic import EpisodicMemory
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool


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
    if ctx.memory is None:
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
    return await ctx.memory.remember(fact, source="conversation", notify=True)


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
    if ctx.memory is None:
        return []
    facts = await ctx.memory.search_by_source(query, source, top_k=limit)
    return [{"id": f["id"], "fact": f["content"], "similarity": f["similarity"]} for f in facts]


@tool("memory_search_history", risk=Risk.read)
async def memory_search_history(ctx: ToolContext, query: str, limit: int = 8) -> list[dict]:
    """Keyword search over past conversations. Use when the user refers to something
    discussed earlier ("that link you sent", "what did we decide about X")."""
    rows = []
    for joiner in (" ", " OR "):
        q = _fts_query(query, joiner)
        if not q:
            return []
        rows = await ctx.store.search_messages(q, limit=limit)
        if rows:
            break
    return [
        {"when": r["created_at"], "role": r["role"], "session_id": r["session_id"], "text": (r["content"] or "")[:500]}
        for r in rows
    ]


@tool("history_semantic_search", risk=Risk.read)
async def history_semantic_search(ctx: ToolContext, query: str, limit: int = 5) -> dict:
    """Meaning-based search of summaries of older conversations. Use for topics, ideas or past
    decisions when you do not know when they were discussed ("what did we decide about the marketing plan?")."""
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
    try:
        rows = await _episodic(ctx).messages_between(start_date, end_date, limit=limit)
    except ValueError:
        return {"error": "Invalid date format. Use ISO 8601, e.g. 2026-09-14T00:00:00Z."}
    if not rows:
        return {"result": "No messages found in that time period."}
    log = "\n".join(f"[{r['created_at'][:16].replace('T', ' ')}] {r['role']}: {(r['content'] or '')[:600]}" for r in rows)
    return {"conversation": log, "count": len(rows)}


@tool("user_model_ask", risk=Risk.read)
async def user_model_ask(ctx: ToolContext, question: str) -> dict:
    """Predict what the user would want or prefer, from the insights and facts you have learned about them.
    Use before choosing on their behalf ("which option would they pick?", "how do they like reports written?")."""
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
