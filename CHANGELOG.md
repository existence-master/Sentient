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
- **Memory sources:** under a chat reply, "Used 3 memories" shows what Sentient had in mind when it answered: the
  memories and things it has learned about you that were in front of it, and any it looked up. Each one has
  **This is wrong** to fix or forget it on the spot, and a link to it on the Memory or About you page.

### Fixed
- Local Ollama models now read 8192 tokens by default instead of Ollama's 4096, so long tasks, big inboxes
  and many tools are no longer cut off without warning. Change it with `models.context_length`, per role with
  `models.context_length_per_role`; it never goes above the model's own maximum, and `sentient doctor` shows it.

### Security
- Work Sentient does on its own (proactive checks, the heartbeat, follow-up scans, dreaming and any subagent they
  start) can only look things up and change Sentient's own things. Sending, deleting, buying, running code, changing
  anything outside Sentient or creating tasks is refused in code, even with an Allow rule. Sentient may offer the
  refused action to you as a suggestion to approve. Chats, your tasks and suggestions you approve work as before.
- One-time codes, verification codes, magic sign-in links and password reset links in Gmail and IMAP email are
  replaced with a short placeholder before the AI reads them, in tool results, proactive suggestions and follow-ups.
  Booking, order, ticket and reference codes, dates, prices and ordinary links are left alone, and the email itself
  is unchanged in your mail app.
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
