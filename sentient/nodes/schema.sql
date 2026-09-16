-- Devices on the node protocol (owner: nodes). Idempotent.
CREATE TABLE IF NOT EXISTS nodes (
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
CREATE INDEX IF NOT EXISTS idx_nodes_token ON nodes(token_hash);
