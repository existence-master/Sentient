"""Columns added before the steps were numbered. A database from before #287 says version 0 and runs every step."""

from __future__ import annotations

from typing import Any

COLUMNS = [
    ("sessions", "context_summary", "TEXT"),
    ("sessions", "context_upto", "TEXT"),
    ("sessions", "untrusted", "TEXT"),
    ("sessions", "visited_hosts", "TEXT"),
    ("messages", "attachments", "TEXT"),
    ("messages", "interjection", "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "memory_sources", "TEXT"),
    ("facts", "status", "TEXT NOT NULL DEFAULT 'active'"),
    ("facts", "review", "TEXT"),
    ("facts", "recall_count", "INTEGER NOT NULL DEFAULT 0"),
    ("facts", "last_recalled_at", "TEXT"),
    ("summaries", "untrusted", "TEXT"),
    ("user_insights", "review", "TEXT"),
    ("integrations", "status", "TEXT"),
    ("integrations", "error", "TEXT"),
    ("skill_stats", "created_at", "TEXT"),
    ("skill_stats", "success_count", "INTEGER NOT NULL DEFAULT 0"),
    ("skill_stats", "failure_count", "INTEGER NOT NULL DEFAULT 0"),
    ("skill_stats", "last_failure_at", "TEXT"),
    ("tasks", "assignee", "TEXT NOT NULL DEFAULT 'ai'"),
    ("tasks", "model", "TEXT"),
    ("tasks", "chat_history", "TEXT"),
    ("tasks", "clarifying_questions", "TEXT"),
    ("tasks", "swarm_details", "TEXT"),
    ("tasks", "original_context", "TEXT"),
    ("tasks", "error", "TEXT"),
    ("tasks", "next_execution_at", "TEXT"),
    ("tasks", "last_execution_at", "TEXT"),
    ("tasks", "script", "TEXT"),
    ("tasks", "browser_profile", "TEXT"),
    ("tasks", "deliver_to", "TEXT"),
    ("task_runs", "plan", "TEXT"),
    ("task_runs", "resume_count", "INTEGER NOT NULL DEFAULT 0"),
    ("task_runs", "retry_of", "TEXT"),
    ("task_runs", "pending_question", "TEXT"),
    ("task_runs", "limits", "TEXT"),
    ("task_runs", "last_activity_at", "TEXT"),
    ("task_runs", "memory_sources", "TEXT"),
]


async def migrate(store: Any) -> None:
    for table, column, decl in COLUMNS:
        await store.ensure_column(table, column, decl)
    # summaries written before the mark existed take it from their chat (ADR 0021)
    if {"untrusted", "session_id"} <= await store.columns("summaries") and "untrusted" in await store.columns("sessions"):
        await store.db.execute(
            "UPDATE summaries SET untrusted = (SELECT s.untrusted FROM sessions s WHERE s.id = summaries.session_id)"
            " WHERE untrusted IS NULL AND session_id IS NOT NULL"
        )
