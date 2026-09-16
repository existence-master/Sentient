-- Messaging channels (owner: channels). Idempotent; applied by ChannelService.start().
-- Bot tokens are never stored here: they live in the OS keychain (sentient.secrets).

CREATE TABLE IF NOT EXISTS channel_state (
    channel       TEXT PRIMARY KEY,           -- "telegram" | "discord"
    enabled       INTEGER NOT NULL DEFAULT 0, -- connected by the user (reconnect on start)
    status        TEXT NOT NULL DEFAULT 'disconnected',
    account_label TEXT,
    error         TEXT,
    cursor        TEXT,                       -- telegram update offset / discord resume info
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS channel_chats (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    label      TEXT,
    paired_at  TEXT NOT NULL,
    deliver    INTEGER NOT NULL DEFAULT 1,
    session_id TEXT,
    PRIMARY KEY (channel, chat_id)
);

CREATE TABLE IF NOT EXISTS channel_pairing_codes (
    channel    TEXT PRIMARY KEY,              -- one active code per channel
    code_hash  TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_refusals (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    refused_at TEXT NOT NULL,
    PRIMARY KEY (channel, chat_id)
);
