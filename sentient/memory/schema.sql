-- Memory package tables (idempotent). Executed by sentient.memory.schema.ensure_memory_schema.
-- The core `facts` and `summaries` tables live in sentient/store/schema.sql.

-- Keyword index over facts for hybrid (FTS5 + vector) recall.
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content, content='facts', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE OF content ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Dialectic user model: evolving insights about the user.
CREATE TABLE IF NOT EXISTS user_insights (
    rid         INTEGER PRIMARY KEY AUTOINCREMENT,     -- rowid mirrored by user_insights_vec
    id          TEXT NOT NULL UNIQUE,
    dimension   TEXT NOT NULL DEFAULT 'context',
    statement   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    status      TEXT NOT NULL DEFAULT 'active',       -- active | confirmed | disputed | retired | pending (held for review)
    source      TEXT NOT NULL DEFAULT 'inferred',     -- inferred | user
    evidence    TEXT NOT NULL DEFAULT '[]',           -- JSON [{kind, ref, quote, at}]
    review      TEXT,                                 -- JSON {from, snippet, session_id} while held for review
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_insights_status ON user_insights(status);

CREATE TABLE IF NOT EXISTS user_questions (
    id          TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    insight_id  TEXT REFERENCES user_insights(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'open',         -- open | answered | dismissed
    answer      TEXT,
    created_at  TEXT NOT NULL,
    answered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_questions_status ON user_questions(status);

-- Nightly consolidation runs.
CREATE TABLE IF NOT EXISTS dreams (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',      -- running | completed | error
    trigger     TEXT NOT NULL DEFAULT 'manual',       -- schedule | manual
    stats       TEXT NOT NULL DEFAULT '{}',
    journal_md  TEXT NOT NULL DEFAULT '',
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_dreams_started ON dreams(started_at);
