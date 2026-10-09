# Changelog

All notable changes to Sentient are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Tasks can pause to ask you a question ("Which of these two flights?") and carry on with your answer. The question
  shows on the task, as a notification with quick answers, and in paired Telegram or Discord chats with buttons;
  replying to the question message there works too. A waiting task keeps waiting across restarts.
- Proper documentation: getting started, privacy, architecture decision records, agent guides, verification report.
- README screenshots and a new banner.
- Lasting rules for apps and tools in Settings > Approvals & safety: **Allow** (go ahead without asking),
  **Ask** (always ask first) or **Never** (Sentient can't use it). Purchases always ask.
- **Follow-ups:** once a day Sentient notices emails waiting on your reply, and your own questions nobody answered,
  in Gmail and IMAP email, and suggests a ready draft. Newsletters, no-reply senders and answered threads are skipped,
  and nothing is sent until you approve it. Settings under Proactivity.
- **Stop everything:** one control that stops every running reply, task, helper, script and browser action at once
  and pauses scheduled tasks, triggers and suggestions until you press Resume, even after a restart. It is in the
  title bar, the tray menu, the shortcut Ctrl+Alt+Shift+S, `/stopall` in paired Telegram and Discord chats, and on
  paired phones. It never asks the model, so nothing can talk its way past it.

### Security
- Purchases now ask even when approvals are switched off, and code Sentient writes can never make one. A tool set
  to Never while a request waits for your yes, or while a script runs, is blocked right away.

## [3.0.0-alpha.0] - 2026-10-08

Sentient v3: a rewrite as a desktop app that runs on your own computer. No account, no servers, no `.env`.

### Added
- **Desktop app** (Electron + React) that starts a local engine, with onboarding, chat, tasks, memory, integrations,
  skills, notifications, voice mode, devices and settings.
- **Any model for any job:** separate models for chat, background work, planning, running tasks, voice, vision and
  embeddings, local (Ollama, LM Studio) or cloud, with fallbacks and a one-click test.
- **Long-running tasks** carried over from v2 (one-off, recurring, triggered, swarm), plus **watcher tasks** that run
  small scripts without a model, **retry** of failed runs, and changing tasks from chat.
- **Memory** with topics and expiry, an **About you** page (an evolving picture you can correct) and nightly
  **consolidation** that merges duplicates and settles contradictions.
- **Proactivity** that reacts to change feeds (Gmail, Calendar), IMAP push and **webhooks**, plus a quieter heartbeat.
- **Self-improving skills** that Sentient proposes and repairs, always waiting for your review.
- **Devices:** pair phones and smart glasses with a code over an encrypted home-network connection; the desktop is a
  device too. Protocol in `docs/NODES.md`, reference ESP32-S3 firmware in `firmware/`.
- **Messaging:** Telegram and Discord bots with pairing codes, streaming replies and approvals as buttons.
- **Browser control** on your installed Edge or Chrome, and **code execution** in a sandbox that can call tools.
- **Helper agents** that work beside the chat, and **steering** a reply while it is still being written.
- **Wake word** ("Hey Sentient") and talk mode.
- **Windows installer** with the engine bundled; release workflow for Windows, macOS and Linux.

### Changed
- One SQLite file replaces MongoDB, Postgres, Chroma and Redis. Keys live in the OS keychain.
- Contribution model: GitHub flow on `main`, CI without secrets, issue forms and a starter backlog.

### Removed
- The v2 cloud web app and its deployment. It is preserved on the `v2` branch and the `v2-final` tag.

## v2

The cloud web app. Its history is on the [`v2` branch](https://github.com/existence-master/Sentient/tree/v2).

[Unreleased]: https://github.com/existence-master/Sentient/compare/v2-final...main
[3.0.0-alpha.0]: https://github.com/existence-master/Sentient/tree/main
