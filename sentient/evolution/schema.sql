-- Owned by sentient.evolution. Loaded by Store.open() after the core schema. Idempotent.

-- Everything self-evolution does, for the Skills > Evolution log screen.
CREATE TABLE IF NOT EXISTS evolution_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,          -- skill_created | skill_patched | skill_repair_proposed | skill_archived | profile_updated | curator_run | summary_created
    detail  TEXT NOT NULL           -- JSON
);
CREATE INDEX IF NOT EXISTS idx_evolution_log_ts ON evolution_log(ts);

-- Background reviewer bookkeeping: what has been reviewed, up to which activity.
CREATE TABLE IF NOT EXISTS evolution_reviews (
    target        TEXT PRIMARY KEY,   -- session:<id> | run:<task_id>:<run_id>
    reviewed_at   TEXT NOT NULL,
    activity_at   TEXT,               -- last message / run finish covered by the review
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    decision      TEXT,               -- none | create | patch | skipped
    skill         TEXT
);
