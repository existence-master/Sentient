-- Folders and Obsidian vaults the user added as searchable notes (#176). Idempotent.
-- Sentient only reads these folders; these tables are its index of them.

CREATE TABLE IF NOT EXISTS folders (
    id            TEXT PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,              -- absolute, as the user picked it
    name          TEXT NOT NULL,                     -- the folder's own name (an Obsidian vault's name)
    kind          TEXT NOT NULL DEFAULT 'folder',    -- folder | obsidian
    ignore        TEXT NOT NULL DEFAULT '[]',        -- JSON list of this folder's own ignore patterns
    added_at      TEXT NOT NULL,
    last_scan_at  TEXT,
    last_scan_ms  INTEGER,
    skipped       INTEGER NOT NULL DEFAULT 0,        -- files left out at the last look (too big, unreadable, over the cap)
    error         TEXT,                              -- plain sentence, or NULL
    scope         TEXT NOT NULL DEFAULT 'always'     -- the chats its notes are recalled in: always, personal, work, a project (#392)
);

CREATE TABLE IF NOT EXISTS folder_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id   TEXT NOT NULL,
    rel         TEXT NOT NULL,                       -- path inside the folder, with forward slashes
    size        INTEGER NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    digest      TEXT,                                -- hash of the extracted text: unchanged text is not embedded again
    title       TEXT,
    tags        TEXT NOT NULL DEFAULT '[]',
    clipped     TEXT,                                -- the web address a clipped page came from, or "web" (outside content)
    error       TEXT,                                -- why it could not be read
    indexed_at  TEXT,
    UNIQUE(folder_id, rel)
);
CREATE INDEX IF NOT EXISTS idx_folder_files_folder ON folder_files(folder_id);

CREATE TABLE IF NOT EXISTS folder_chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,    -- rowid mirrored by folder_vec and folder_fts
    file_id    INTEGER NOT NULL,
    folder_id  TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    heading    TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL,
    embedded   INTEGER NOT NULL DEFAULT 0            -- 1 once its vector is in folder_vec
);
CREATE INDEX IF NOT EXISTS idx_folder_chunks_file ON folder_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_folder_chunks_pending ON folder_chunks(folder_id) WHERE embedded = 0;

-- keyword search over the same pieces; rowid = folder_chunks.id
CREATE VIRTUAL TABLE IF NOT EXISTS folder_fts USING fts5(title, heading, text, tokenize='unicode61 remove_diacritics 2');
