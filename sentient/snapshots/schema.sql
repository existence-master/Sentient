-- Owned by sentient.snapshots. Loaded by Store.open() after the core schema. Must stay idempotent.
-- Copies of files kept before code changed them (#142, ADR 0026). A row exists only when something changed; the
-- copies themselves are under <home>/snapshots/<id>/.

CREATE TABLE IF NOT EXISTS file_snapshots (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,                -- terminal | code | file_write | restore
    what         TEXT,                         -- the command, the script's purpose or the file name
    session_id   TEXT,                         -- the chat it ran in, if any
    task_id      TEXT,                         -- the task run it ran in, if any
    run_id       TEXT,
    call_id      TEXT,                         -- the tool call that ran it
    files        TEXT NOT NULL,                -- JSON [{path, change: changed|deleted, bytes, copy, note}]
    created      TEXT,                         -- JSON list of paths it made (not put back)
    notes        TEXT,                         -- JSON list of plain sentences (files left out and why)
    bytes        INTEGER NOT NULL DEFAULT 0,   -- size of the copies kept
    restored_at  TEXT,                         -- when Restore put the files back
    restored_by  TEXT,                         -- the snapshot taken of the files Restore replaced (undo of a Restore)
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_file_snapshots_created ON file_snapshots(created_at);
