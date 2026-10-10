-- Sentient v3 storage. One SQLite file replaces Mongo + Postgres/pgvector + Chroma + Redis.
-- Vectors: sqlite-vec virtual tables (created lazily once the embedding dimension is known).
-- Keyword search: FTS5.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------- conversations
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    channel     TEXT NOT NULL DEFAULT 'web',      -- web | cli | telegram | glasses | ...
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0,
    context_summary TEXT,                         -- running summary of turns older than the history window
    context_upto    TEXT,                         -- created_at of the last message folded into the summary
    untrusted       TEXT,                         -- app whose content this chat read ("Gmail"); sends then ask (ADR 0018)
    visited_hosts   TEXT                          -- JSON list of web hosts this chat loaded (ADR 0018)
);

CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,                  -- system | user | assistant | tool
    content       TEXT,
    tool_calls    TEXT,                           -- JSON list (assistant messages)
    tool_call_id  TEXT,                           -- (tool messages)
    name          TEXT,                           -- tool name (tool messages)
    thinking      TEXT,                           -- model reasoning, never re-sent to the model
    attachments   TEXT,                           -- JSON list of file names (user messages)
    interjection  INTEGER NOT NULL DEFAULT 0,     -- 1 for user messages sent while a reply was running (steering)
    memory_sources TEXT,                          -- JSON list: memories a final assistant reply had in mind
    created_at    TEXT NOT NULL,
    summarized    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

-- Episodic tier: first-person summaries of older conversation chunks.
CREATE TABLE IF NOT EXISTS summaries (
    id            TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    content       TEXT NOT NULL,
    start_at      TEXT NOT NULL,
    end_at        TEXT NOT NULL,
    message_ids   TEXT NOT NULL,                  -- JSON list
    created_at    TEXT NOT NULL,
    untrusted     TEXT                            -- app whose content the chat had read (ADR 0018/0021): kept out of
                                                  -- other chats, proactivity and MEMORY.md; "" or NULL = clean
);

-- ---------------------------------------------------------------- semantic memory (atomic facts)
CREATE TABLE IF NOT EXISTS facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'conversation',   -- conversation | manual | onboarding | file:<name> | task:<id>
    topics          TEXT NOT NULL DEFAULT '[]',             -- JSON list of topic names
    memory_type     TEXT NOT NULL DEFAULT 'long-term',      -- long-term | short-term
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    expires_at      TEXT,
    embedding_model TEXT,
    previous_content TEXT,                                  -- kept on UPDATE so edits are auditable
    status          TEXT NOT NULL DEFAULT 'active',         -- active | pending (held for the user's review, ADR 0021)
    review          TEXT                                    -- JSON {from, snippet, session_id}: where a held memory came from
);
CREATE INDEX IF NOT EXISTS idx_facts_expires ON facts(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source);

-- ---------------------------------------------------------------- notifications / approvals
CREATE TABLE IF NOT EXISTS notifications (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,                    -- info | task | approval | proactive | skill | error
    title       TEXT,
    body        TEXT NOT NULL,                    -- markdown message shown to the user
    payload     TEXT,                             -- JSON (task_id, suggestion, skill name, ...)
    read        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at);

-- ---------------------------------------------------------------- usage (tokens per model, Hermes-style insights)
CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model             TEXT NOT NULL,
    role              TEXT,
    source            TEXT NOT NULL DEFAULT 'chat',     -- chat | task | memory | proactive | evolution | voice
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);

-- ---------------------------------------------------------------- integrations / credentials metadata
-- Actual OAuth tokens go to the OS keychain under sentient/integration:<id>;
-- this table only records connection state so the UI can render it.
CREATE TABLE IF NOT EXISTS integrations (
    id            TEXT PRIMARY KEY,               -- plugin id, e.g. gmail
    connected     INTEGER NOT NULL DEFAULT 0,
    auth_type     TEXT,
    account_label TEXT,                           -- e.g. the email address
    settings      TEXT,                           -- JSON (privacy filters etc.)
    connected_at  TEXT,
    updated_at    TEXT
);

-- ---------------------------------------------------------------- skills (self-evolution ledger)
CREATE TABLE IF NOT EXISTS skill_stats (
    name         TEXT PRIMARY KEY,
    use_count    INTEGER NOT NULL DEFAULT 0,
    view_count   INTEGER NOT NULL DEFAULT 0,
    patch_count  INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    state        TEXT NOT NULL DEFAULT 'active'   -- active | pending_review | stale | archived
);

-- ---------------------------------------------------------------- chat subagents (owner: core, docs/API.md section 10)
CREATE TABLE IF NOT EXISTS subagents (
    id             TEXT PRIMARY KEY,
    session_id     TEXT,                          -- chat that spawned it (NULL outside chats)
    parent_call_id TEXT,                          -- tool call id of delegate_task / delegate_tasks
    goal           TEXT NOT NULL,
    context        TEXT,
    tools          TEXT,                          -- JSON list of requested tool names, or NULL
    status         TEXT NOT NULL DEFAULT 'running',   -- running | completed | error | cancelled
    background     INTEGER NOT NULL DEFAULT 0,
    summary        TEXT,
    error          TEXT,
    tool_calls     INTEGER NOT NULL DEFAULT 0,
    files_created  TEXT,                          -- JSON list of names under files/
    events         TEXT,                          -- JSON list of ProgressUpdate
    started_at     TEXT NOT NULL,
    finished_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_subagents_session ON subagents(session_id, started_at);
