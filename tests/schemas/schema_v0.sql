-- The schema of a database made by 4a5cb2d9 (the first v3 build), before the steps were numbered: version 0.
-- Made by starting that build's app on a new home and dumping sqlite_master (sqlite-vec tables left out).

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    channel     TEXT NOT NULL DEFAULT 'web',      -- web | cli | telegram | glasses | ...
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0,
    context_summary TEXT,                         -- running summary of turns older than the history window
    context_upto    TEXT                          -- created_at of the last message folded into the summary
);
CREATE TABLE messages (
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
    created_at    TEXT NOT NULL,
    summarized    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content, content='messages', content_rowid='rowid'
);
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TABLE summaries (
    id            TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    content       TEXT NOT NULL,
    start_at      TEXT NOT NULL,
    end_at        TEXT NOT NULL,
    message_ids   TEXT NOT NULL,                  -- JSON list
    created_at    TEXT NOT NULL
);
CREATE TABLE facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'conversation',   -- conversation | manual | onboarding | file:<name> | task:<id>
    topics          TEXT NOT NULL DEFAULT '[]',             -- JSON list of topic names
    memory_type     TEXT NOT NULL DEFAULT 'long-term',      -- long-term | short-term
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    expires_at      TEXT,
    embedding_model TEXT,
    previous_content TEXT                                   -- kept on UPDATE so edits are auditable
, recall_count INTEGER NOT NULL DEFAULT 0, last_recalled_at TEXT);
CREATE INDEX idx_facts_expires ON facts(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_facts_source ON facts(source);
CREATE TABLE notifications (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,                    -- info | task | approval | proactive | skill | error
    title       TEXT,
    body        TEXT NOT NULL,                    -- markdown message shown to the user
    payload     TEXT,                             -- JSON (task_id, suggestion, skill name, ...)
    read        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX idx_notifications_created ON notifications(created_at);
CREATE TABLE usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model             TEXT NOT NULL,
    role              TEXT,
    source            TEXT NOT NULL DEFAULT 'chat',     -- chat | task | memory | proactive | evolution | voice
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE INDEX idx_usage_created ON usage(created_at);
CREATE TABLE integrations (
    id            TEXT PRIMARY KEY,               -- plugin id, e.g. gmail
    connected     INTEGER NOT NULL DEFAULT 0,
    auth_type     TEXT,
    account_label TEXT,                           -- e.g. the email address
    settings      TEXT,                           -- JSON (privacy filters etc.)
    connected_at  TEXT,
    updated_at    TEXT
, status TEXT, error TEXT);
CREATE TABLE skill_stats (
    name         TEXT PRIMARY KEY,
    use_count    INTEGER NOT NULL DEFAULT 0,
    view_count   INTEGER NOT NULL DEFAULT 0,
    patch_count  INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    state        TEXT NOT NULL DEFAULT 'active'   -- active | pending_review | stale | archived
, created_at TEXT, success_count INTEGER NOT NULL DEFAULT 0, failure_count INTEGER NOT NULL DEFAULT 0, last_failure_at TEXT);
CREATE TABLE subagents (
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
CREATE INDEX idx_subagents_session ON subagents(session_id, started_at);
CREATE TABLE tasks (
    id                    TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    description           TEXT,
    status                TEXT NOT NULL,              -- planning | clarification_pending | approval_pending | pending | active | processing | completed | completed_with_errors | error | declined | cancelled | archived
    priority              INTEGER NOT NULL DEFAULT 1, -- 0 high, 1 medium, 2 low
    task_type             TEXT NOT NULL DEFAULT 'single',   -- single | swarm | script
    schedule              TEXT,                       -- JSON: {type: once|recurring|triggered, ...}
    plan                  TEXT,                       -- JSON list of {tool, description}
    original_prompt       TEXT,
    source                TEXT NOT NULL DEFAULT 'user',      -- user | chat | proactive | trigger
    enabled               INTEGER NOT NULL DEFAULT 1,
    assignee              TEXT NOT NULL DEFAULT 'ai',
    model                 TEXT,                       -- explicit executor model override
    chat_history          TEXT,                       -- JSON list of {role, content, timestamp}
    clarifying_questions  TEXT,                       -- JSON list of {question_id, text, answer}
    swarm_details         TEXT,                       -- JSON (swarm tasks only)
    original_context      TEXT,                       -- JSON
    script                TEXT,                       -- JSON (script jobs only): {code, condition, then, last_result, last_run_at, last_error}
    error                 TEXT,
    next_execution_at     TEXT,                       -- UTC ISO-8601, seconds precision (lexicographically comparable)
    last_execution_at     TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);
CREATE TABLE task_runs (
    id            TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    status        TEXT NOT NULL,                  -- processing | completed | completed_with_errors | error | cancelled
    plan          TEXT,                           -- JSON plan snapshot (swarm: worker configurations)
    trigger_data  TEXT,                           -- JSON event that started this run (triggered tasks)
    messages      TEXT,                           -- JSON checkpoint of the agent transcript (resumable)
    result        TEXT,                           -- JSON {summary, links_created, links_found, files_created, tools_used}
    error         TEXT,
    resume_count  INTEGER NOT NULL DEFAULT 0,
    retry_of      TEXT,                           -- run id this run retries (continues from its transcript)
    started_at    TEXT,
    finished_at   TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_runs_task ON task_runs(task_id, created_at);
CREATE TABLE run_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                    -- info | thought | tool_call | tool_result | final_answer | error
    payload     TEXT NOT NULL,                    -- JSON ProgressUpdate {timestamp, message}
    created_at  TEXT NOT NULL
);
CREATE INDEX idx_run_events_run ON run_events(run_id, id);
CREATE TABLE seen_events (
    source      TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, event_id)
);
CREATE TABLE task_trigger_seen (
    task_id     TEXT NOT NULL,
    item_key    TEXT NOT NULL,                    -- "<source>:<item id>"
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (task_id, item_key)
);
CREATE TABLE integration_poll_state (
    source      TEXT PRIMARY KEY,
    cursor      TEXT,                 -- ISO timestamp of the last successful poll
    seen        TEXT,                 -- JSON list of recently returned item keys
    last_error  TEXT,
    updated_at  TEXT
);
CREATE TABLE integration_seen (
    source      TEXT NOT NULL,
    item_key    TEXT NOT NULL,
    origin      TEXT,                 -- poll | feed | webhook (who saw it first)
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, item_key)
);
CREATE INDEX idx_integration_seen_at ON integration_seen(seen_at);
CREATE TABLE integration_feed_state (
    source           TEXT PRIMARY KEY,
    cursor           TEXT,
    status           TEXT,            -- ok | error
    failures         INTEGER NOT NULL DEFAULT 0,
    last_sync_at     TEXT,
    last_success_at  TEXT,
    last_error       TEXT,
    note             TEXT,
    next_attempt_at  TEXT,
    emitted          INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT
);
CREATE TABLE hooks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    secret_hash     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_called_at  TEXT,
    calls           INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE proactive_sources (
    source          TEXT PRIMARY KEY,
    connected       INTEGER NOT NULL DEFAULT 0,
    last_poll_at    TEXT,
    last_success_at TEXT,
    last_error      TEXT,
    items_seen      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT
);
CREATE TABLE proactive_seen (
    source      TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    event_type  TEXT,
    item        TEXT,
    starts_at   TEXT,                 -- calendar events: start time (UTC ISO) for the heartbeat
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, item_id)
);
CREATE INDEX idx_proactive_seen_start ON proactive_seen(source, starts_at);
CREATE TABLE proactive_preferences (
    suggestion_type TEXT PRIMARY KEY,
    score           INTEGER NOT NULL DEFAULT 0,
    approvals       INTEGER NOT NULL DEFAULT 0,
    dismissals      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL
);
CREATE TABLE proactive_suggestion_types (
    type_name   TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    builtin     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT
);
CREATE TABLE proactive_suggestions (
    id              TEXT PRIMARY KEY,
    notification_id TEXT UNIQUE,
    suggestion_type TEXT NOT NULL,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL,        -- deferred | pending | approved | dismissed
    confidence      REAL,
    threshold       REAL,
    source          TEXT,
    event_type      TEXT,
    item_id         TEXT,
    payload         TEXT NOT NULL,        -- JSON notification payload
    context         TEXT,                 -- JSON pruned cognitive scratchpad
    task_id         TEXT,
    created_at      TEXT NOT NULL,
    actioned_at     TEXT
);
CREATE INDEX idx_proactive_suggestions_created ON proactive_suggestions(created_at);
CREATE TABLE evolution_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,          -- skill_created | skill_patched | skill_repair_proposed | skill_archived | profile_updated | curator_run | summary_created
    detail  TEXT NOT NULL           -- JSON
);
CREATE INDEX idx_evolution_log_ts ON evolution_log(ts);
CREATE TABLE evolution_reviews (
    target        TEXT PRIMARY KEY,   -- session:<id> | run:<task_id>:<run_id>
    reviewed_at   TEXT NOT NULL,
    activity_at   TEXT,               -- last message / run finish covered by the review
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    decision      TEXT,               -- none | create | patch | skipped
    skill         TEXT
);
CREATE VIRTUAL TABLE facts_fts USING fts5(
    content, content='facts', content_rowid='id'
);
CREATE TRIGGER facts_fts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER facts_fts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER facts_fts_au AFTER UPDATE OF content ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TABLE user_insights (
    rid         INTEGER PRIMARY KEY AUTOINCREMENT,     -- rowid mirrored by user_insights_vec
    id          TEXT NOT NULL UNIQUE,
    dimension   TEXT NOT NULL DEFAULT 'context',
    statement   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    status      TEXT NOT NULL DEFAULT 'active',       -- active | confirmed | disputed | retired
    source      TEXT NOT NULL DEFAULT 'inferred',     -- inferred | user
    evidence    TEXT NOT NULL DEFAULT '[]',           -- JSON [{kind, ref, quote, at}]
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX idx_user_insights_status ON user_insights(status);
CREATE TABLE user_questions (
    id          TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    insight_id  TEXT REFERENCES user_insights(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'open',         -- open | answered | dismissed
    answer      TEXT,
    created_at  TEXT NOT NULL,
    answered_at TEXT
);
CREATE INDEX idx_user_questions_status ON user_questions(status);
CREATE TABLE dreams (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',      -- running | completed | error
    trigger     TEXT NOT NULL DEFAULT 'manual',       -- schedule | manual
    stats       TEXT NOT NULL DEFAULT '{}',
    journal_md  TEXT NOT NULL DEFAULT '',
    error       TEXT
);
CREATE INDEX idx_dreams_started ON dreams(started_at);
CREATE TABLE nodes (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'custom',
    platform      TEXT NOT NULL DEFAULT '',
    app_version   TEXT NOT NULL DEFAULT '',
    capabilities  TEXT NOT NULL DEFAULT '[]',
    token_hash    TEXT,
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT,
    battery       INTEGER,
    revoked       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_nodes_token ON nodes(token_hash);
CREATE TABLE channel_state (
    channel       TEXT PRIMARY KEY,           -- "telegram" | "discord"
    enabled       INTEGER NOT NULL DEFAULT 0, -- connected by the user (reconnect on start)
    status        TEXT NOT NULL DEFAULT 'disconnected',
    account_label TEXT,
    error         TEXT,
    cursor        TEXT,                       -- telegram update offset / discord resume info
    updated_at    TEXT
);
CREATE TABLE channel_chats (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    label      TEXT,
    paired_at  TEXT NOT NULL,
    deliver    INTEGER NOT NULL DEFAULT 1,
    session_id TEXT,
    PRIMARY KEY (channel, chat_id)
);
CREATE TABLE channel_pairing_codes (
    channel    TEXT PRIMARY KEY,              -- one active code per channel
    code_hash  TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE channel_refusals (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    refused_at TEXT NOT NULL,
    PRIMARY KEY (channel, chat_id)
);
CREATE INDEX idx_tasks_next_exec ON tasks(status, enabled, next_execution_at);
PRAGMA user_version = 0;
