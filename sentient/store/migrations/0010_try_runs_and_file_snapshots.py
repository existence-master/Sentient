"""Try runs and file snapshots (#142), plus columns that arrived at version 9 without a step of their own."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("task_runs", "dry_run", "INTEGER NOT NULL DEFAULT 0")
    await store.ensure_column("task_runs", "would_do", "TEXT")
    await store.ensure_column("file_snapshots", "restoring_at", "TEXT")
    # approvals answered from paired chats name the request (#337)
    await store.ensure_column("channel_questions", "request", "TEXT")
    # a memory the nightly tidy-up merged into another one, kept for undo (#353)
    await store.ensure_column("facts", "merged_into", "INTEGER")
    await store.ensure_column("facts", "merged_at", "TEXT")
    await store.ensure_column("facts", "merged_by", "TEXT")
