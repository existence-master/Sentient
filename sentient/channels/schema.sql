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

-- Messages that delivered a task's question (ask_user): a reply to one of them is the answer.
CREATE TABLE IF NOT EXISTS channel_questions (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    message_id TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    request    TEXT,                          -- which request of the run (#337): its expires_at, else request_id
    PRIMARY KEY (channel, chat_id, message_id)
);

-- Notifications sent out to paired chats (#345): what went where, and whether it arrived. The Activity timeline
-- shows them under "Messaging apps". ``ref`` is JSON naming the source item (a task, a chat or a notification).
CREATE TABLE IF NOT EXISTS channel_deliveries (
    id         TEXT PRIMARY KEY,
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    chat_label TEXT,
    title      TEXT NOT NULL,                 -- the message's first line, short
    what       TEXT NOT NULL,                 -- task | question | approval | plan | brief | suggestion | subagent | other
    reason     TEXT NOT NULL,                 -- chosen (the task picked this chat) | default (the chat's delivery is on) | chat | reminder | reminder_copy
    status     TEXT NOT NULL,                 -- delivered | failed
    error      TEXT,
    ref        TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_channel_deliveries_created ON channel_deliveries(created_at);

CREATE TABLE IF NOT EXISTS channel_refusals (
    channel    TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    refused_at TEXT NOT NULL,
    PRIMARY KEY (channel, chat_id)
);
