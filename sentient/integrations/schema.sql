-- Integrations package tables (idempotent). Connection state itself lives in the core
-- `integrations` table; credentials live in the OS keychain, never here.

-- Cursor + recently returned ids per polled source (gmail, gcalendar) so poll_source
-- only hands proactivity / triggered tasks items they have not seen yet.
CREATE TABLE IF NOT EXISTS integration_poll_state (
    source      TEXT PRIMARY KEY,
    cursor      TEXT,                 -- ISO timestamp of the last successful poll
    seen        TEXT,                 -- JSON list of recently returned item keys
    last_error  TEXT,
    updated_at  TEXT
);

-- Shared "already emitted" record across every origin (poll, feed, webhook). An item key
-- (gmail message id, calendar "<event id>:<updated>", IMAP Message-ID) is emitted once per source.
CREATE TABLE IF NOT EXISTS integration_seen (
    source      TEXT NOT NULL,
    item_key    TEXT NOT NULL,
    origin      TEXT,                 -- poll | feed | webhook (who saw it first)
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (source, item_key)
);
CREATE INDEX IF NOT EXISTS idx_integration_seen_at ON integration_seen(seen_at);

-- Change feed / push watcher state per source (gmail history id, calendar sync token,
-- IMAP {uidvalidity, last_uid}) with failure counts for exponential backoff.
CREATE TABLE IF NOT EXISTS integration_feed_state (
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

-- Inbound webhooks. Only a SHA-256 hash of the secret is stored; the secret is shown once.
CREATE TABLE IF NOT EXISTS hooks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    secret_hash     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_called_at  TEXT,
    calls           INTEGER NOT NULL DEFAULT 0
);
