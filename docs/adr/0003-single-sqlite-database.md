# 0003. Keep all state in one SQLite file

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

v2 needed MongoDB, Postgres with pgvector, Chroma and Redis. A personal assistant on a laptop needs durable
tasks, chat history, full-text search and vector search, but cannot ask people to run database servers.

## Decision

All state lives in one SQLite database (`~/.sentient/sentient.db`) in WAL mode, with FTS5 for text search
and the `sqlite-vec` extension for embeddings. Each feature package owns an idempotent `schema.sql` and adds
columns with `store.ensure_column`. Background work runs in-process (asyncio services), not in a queue.

## Consequences

Backup is copying one file. Tests open a fresh database per test with no setup. We accept SQLite's single
writer: long work must not hold transactions, and services share one connection. If `sqlite-vec` cannot load,
semantic memory turns off with a clear message instead of crashing.

## Alternatives considered

Postgres + pgvector: excellent, but a server to install and keep running. A separate vector store (Chroma,
LanceDB): a second source of truth to keep in sync. Plain files: no transactions or queries.
