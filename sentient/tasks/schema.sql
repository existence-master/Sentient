-- Owned by sentient.tasks. Loaded by Store.open() after the core schema.
-- Must stay idempotent (CREATE ... IF NOT EXISTS). Columns added after the first
-- stub are also ensured by TaskRepo.ensure_schema() (store.ensure_column), and
-- indexes on those columns are created there, so databases created by older
-- builds still open.

CREATE TABLE IF NOT EXISTS tasks (
    id                    TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    description           TEXT,
    status                TEXT NOT NULL,              -- planning | clarification_pending | approval_pending | pending | active | processing | waiting_for_user | completed | completed_with_errors | error | declined | cancelled | archived
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
    browser_profile       TEXT,                       -- named browser profile the task's browser calls use (NULL: default)
    deliver_to            TEXT,                       -- JSON (tasks/delivery.py): "desktop" or [{channel, chat_id}]; NULL: default
    error                 TEXT,
    next_execution_at     TEXT,                       -- UTC ISO-8601, seconds precision (lexicographically comparable)
    last_execution_at     TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
    id            TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    status        TEXT NOT NULL,                  -- processing | waiting_for_user | completed | completed_with_errors | error | cancelled
    plan          TEXT,                           -- JSON plan snapshot (swarm: worker configurations)
    trigger_data  TEXT,                           -- JSON event that started this run (triggered tasks)
    messages      TEXT,                           -- JSON checkpoint of the agent transcript (resumable)
    result        TEXT,                           -- JSON {summary, links_created, links_found, files_created, tools_used}
    error         TEXT,
    resume_count  INTEGER NOT NULL DEFAULT 0,
    retry_of      TEXT,                           -- run id this run retries (continues from its transcript)
    pending_question TEXT,                        -- JSON {question, options, tool_call_id, asked_at, limit?, stop_error?} while waiting_for_user
    limits        TEXT,                           -- JSON {base, max, used} of steps, seconds, tokens, cost_usd (tasks/limits.py)
    started_at    TEXT,
    finished_at   TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON task_runs(task_id, created_at);

CREATE TABLE IF NOT EXISTS run_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                    -- info | thought | tool_call | tool_result | final_answer | error
    payload     TEXT NOT NULL,                    -- JSON ProgressUpdate {timestamp, message}
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);

-- Dedupe of external events that fire triggered tasks (e.g. gmail message ids).
CREATE TABLE IF NOT EXISTS seen_events (
    source      TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, event_id)
);

-- Per-task dedupe of items that fired a triggered task (source.items consumer).
CREATE TABLE IF NOT EXISTS task_trigger_seen (
    task_id     TEXT NOT NULL,
    item_key    TEXT NOT NULL,                    -- "<source>:<item id>"
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (task_id, item_key)
);
