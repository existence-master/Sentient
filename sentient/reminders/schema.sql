-- Timers and one-off reminders (owner: reminders, #150). Idempotent; applied by Store.open().

CREATE TABLE IF NOT EXISTS reminders (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,                      -- "timer" | "reminder" | "alarm" (#359)
    message       TEXT NOT NULL DEFAULT '',           -- what to remind about ("call Ravi"); '' for a timer
    connector     TEXT NOT NULL DEFAULT '',           -- how it was asked: "to" | "about" | "that" | ''
    label         TEXT NOT NULL DEFAULT '',           -- a timer's or an alarm's name ("pasta", "gym")
    duration_s    REAL,                               -- a timer's length, or a reminder set "in 20 minutes"
    fire_at       TEXT NOT NULL,                      -- UTC ISO time it is due
    status        TEXT NOT NULL DEFAULT 'scheduled',  -- "scheduled" | "fired" | "cancelled"
    session_id    TEXT,                               -- the chat it was set in (none for an incognito chat)
    channel       TEXT,                               -- where it was set: "desktop" | "voice" | "telegram" ...
    language      TEXT NOT NULL DEFAULT 'en',
    source        TEXT NOT NULL DEFAULT 'chat',       -- "quick" (no model), "chat" (the model), "api"
    created_at    TEXT NOT NULL,
    fired_at      TEXT,
    cancelled_at  TEXT,
    notification_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, fire_at);
