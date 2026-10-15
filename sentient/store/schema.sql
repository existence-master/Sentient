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
    context_upto    TEXT,                         -- watermark of the messages folded into the summary (store.db.next_watermark)
    untrusted       TEXT,                         -- app whose content this chat read ("Gmail"); sends then ask (ADR 0018)
    visited_hosts   TEXT,                         -- JSON list of web hosts this chat loaded (ADR 0018)
    incognito       INTEGER NOT NULL DEFAULT 0,   -- 1: nothing is learned from it and it is deleted when closed (#154)
    model           TEXT                          -- the model picked for this chat; NULL follows the default (#312)
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
    sources       TEXT,                           -- JSON list: pages, emails, files and apps a final reply used (#174)
    approval      TEXT,                           -- JSON (tool messages): how the user answered the call's approval
                                                  -- request {decision, asked_at, answered_at, target, risk_label,
                                                  -- untrusted, rule}; NULL when it did not ask (activity timeline, #147)
    created_at    TEXT NOT NULL,
    summarized    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
-- the activity timeline (#147) reads tool calls newest first across every chat
CREATE INDEX IF NOT EXISTS idx_messages_tool_time ON messages(created_at) WHERE role = 'tool';

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
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);   -- the activity timeline (#147)

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
    cost_usd          REAL,                             -- the call's price when known (#148); NULL: free, plan or unknown
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

-- ---------------------------------------------------------------- rules from chat (owner: core, docs/API.md section 2)
-- "never delete my emails" said in a chat becomes a proposed lasting rule; only the user's click makes it a rule.
-- While pending it makes the chat ask before the matched tools (a session rule that only tightens).
CREATE TABLE IF NOT EXISTS rule_proposals (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id  TEXT,                             -- the user message (or steer message) it came from
    said        TEXT NOT NULL,                    -- the user's words that matched
    rule        TEXT NOT NULL,                    -- never | ask
    keys        TEXT NOT NULL,                    -- JSON list of rule keys (tool names or app ids)
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | declined
    created_at  TEXT NOT NULL,
    decided_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_rule_proposals_session ON rule_proposals(session_id, status);

-- ---------------------------------------------------------------- approval grants (owner: core, docs/API.md section 2)
-- How wide a "yes" to an approval request was (#139): later calls of one tool in one task, or anywhere until
-- expires_at. "Allow once" and "Allow for this chat" are not stored here; "Always allow" becomes an Allow rule.
CREATE TABLE IF NOT EXISTS approval_grants (
    id          TEXT PRIMARY KEY,
    tool        TEXT NOT NULL,                    -- tool name
    scope       TEXT NOT NULL,                    -- task | timed
    task_id     TEXT,                             -- the task a "task" grant belongs to
    risk        INTEGER NOT NULL,                 -- highest effective risk it covers (tools.base.Risk)
    via         TEXT NOT NULL DEFAULT 'desktop',  -- where it was answered: desktop | telegram | discord | whatsapp
    created_at  TEXT NOT NULL,
    expires_at  TEXT                              -- set for "timed"
);

-- ---------------------------------------------------------------- skills from outside (#146, docs/API.md section 8)
-- Where an imported skill came from, the hash of its folder and the safety check that saw it: a lock record that is
-- checked again whenever the skill is shown or approved.
CREATE TABLE IF NOT EXISTS skill_imports (
    name          TEXT PRIMARY KEY,                -- the skill's name in Sentient
    source        TEXT NOT NULL,                   -- link | hermes
    origin        TEXT,                            -- the web address or the Hermes folder it came from
    hash          TEXT NOT NULL,                   -- SHA-256 of its folder when it was imported
    scanner       TEXT NOT NULL,                   -- version of the safety check
    verdict       TEXT NOT NULL,                   -- high | check | clear
    imported_at   TEXT NOT NULL,
    decision      TEXT,                            -- approved (declined skills are deleted)
    decided_at    TEXT,
    approved_hash TEXT                             -- folder hash the user approved
);

-- ---------------------------------------------------------------- MCP tools the user reviewed (#146, section 5)
-- One row per tool of an MCP server: the hash of what it said about itself when the user decided. A tool whose hash
-- changes is held back until the user looks again.
CREATE TABLE IF NOT EXISTS mcp_tool_reviews (
    server       TEXT NOT NULL,
    tool         TEXT NOT NULL,                    -- the tool's name on the server
    hash         TEXT NOT NULL,                    -- SHA-256 of name, title, description, input schema and hints
    decision     TEXT NOT NULL,                    -- approved | declined
    description  TEXT,                             -- the description that was approved, to show what changed
    decided_at   TEXT NOT NULL,
    alerted_hash TEXT,                             -- a changed hash the user was already told about
    PRIMARY KEY (server, tool)
);
