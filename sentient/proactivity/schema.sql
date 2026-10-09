-- Owned by sentient.proactivity. Loaded by Store.open() after the core schema. Idempotent.

-- Poll state per watched source (gmail, gcalendar, ...).
CREATE TABLE IF NOT EXISTS proactive_sources (
    source          TEXT PRIMARY KEY,
    connected       INTEGER NOT NULL DEFAULT 0,
    last_poll_at    TEXT,
    last_success_at TEXT,
    last_error      TEXT,
    items_seen      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT
);

-- Dedupe of polled items (v2 processed_items). The item JSON is kept for the heartbeat.
CREATE TABLE IF NOT EXISTS proactive_seen (
    source      TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    event_type  TEXT,
    item        TEXT,
    starts_at   TEXT,                 -- calendar events: start time (UTC ISO) for the heartbeat
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, item_id)
);
CREATE INDEX IF NOT EXISTS idx_proactive_seen_start ON proactive_seen(source, starts_at);

-- Learned feedback per suggestion type (v2 user_proactive_preferences).
CREATE TABLE IF NOT EXISTS proactive_preferences (
    suggestion_type TEXT PRIMARY KEY,
    score           INTEGER NOT NULL DEFAULT 0,
    approvals       INTEGER NOT NULL DEFAULT 0,
    dismissals      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL
);

-- Canonical suggestion types (v2 proactive_suggestion_templates + types learned since).
CREATE TABLE IF NOT EXISTS proactive_suggestion_types (
    type_name   TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    builtin     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT
);
INSERT OR IGNORE INTO proactive_suggestion_types(type_name, description, builtin, created_at) VALUES
    ('draft_meeting_confirmation_email', 'Drafts an email to confirm a meeting, check availability, or ask for an agenda.', 1, '2026-09-15T00:00:00+00:00'),
    ('schedule_calendar_event', 'Creates a new event on the user''s calendar based on details from a message.', 1, '2026-09-15T00:00:00+00:00'),
    ('create_follow_up_task', 'Creates a new task in the user''s task list to follow up on a specific item or conversation.', 1, '2026-09-15T00:00:00+00:00'),
    ('summarize_document_or_thread', 'Summarizes a long document, email thread, or message chain for the user.', 1, '2026-09-15T00:00:00+00:00'),
    ('follow_up_reply', 'Drafts a reply to an email that has been waiting on the user for days.', 1, '2026-10-09T00:00:00+00:00'),
    ('follow_up_nudge', 'Drafts a polite nudge when someone has not answered the user''s question for days.', 1, '2026-10-09T00:00:00+00:00');

-- Every suggestion produced (delivered, deferred by quiet hours, approved, dismissed).
CREATE TABLE IF NOT EXISTS proactive_suggestions (
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
CREATE INDEX IF NOT EXISTS idx_proactive_suggestions_created ON proactive_suggestions(created_at);
