# Sentient v3 roadmap

Status legend: done · in progress · next · later.

## M0 · Core engine (done 2026-09-12)
Config model, keychain secrets, SQLite + sqlite-vec store, LiteLLM provider with
roles, tool plugin system with risk levels, fact memory, workspace markdown,
skills library, agent loop with approvals, gateway, tests.

## M1 · Desktop revamp foundations (done 2026-09-15)
- Desktop-only product direction; v2 USPs preserved (tasks, memory, proactivity,
  integrations, voice); no auth.
- Contract-first renderer ↔ engine API (`docs/API.md`), ownership rules (`CLAUDE.md`).
- Composition root with feature services, EventBus forwarded over `/ws`,
  package-owned schemas, router per package.
- Reusable `Agent.run_loop`; chat attachments, per-message model override,
  cancel, token usage, auto titles, context compression, channel-aware prompts.
- Model configuration API: providers, local models, test with tool-support
  probe, roles, fallbacks, Ollama pull, keychain secrets. Onboarding, files,
  sessions, notifications, usage endpoints.

## M2 · Feature parity (done 2026-09-15)
Built in parallel by package owners; 200+ engine tests, lint clean:
- Tasks: state machine, scheduler, planner/executor/result generator,
  triggered tasks with filter DSL, swarm, restart resume, tasks tools.
- Integrations: Google (native OAuth), GitHub, Slack, Notion, Discord, Trello,
  WhatsApp, keyless search/weather/maps/news/charts/web fetch, MCP client,
  privacy filters, poll sources.
- Memory parity + proactivity + self-evolution: v2 topics, graph, import,
  episodic summaries, proactive pipeline with learning, heartbeat, skill
  reviewer, curator, profile upkeep.
- Voice: faster-whisper, system/Kokoro/cloud TTS, VAD, interruptible `/ws/voice`.
- Desktop foundation: Electron shell, design system, onboarding, chat, settings
  with model configuration.

## M3 · Remaining screens (done 2026-09-15)
Tasks (list, calendar, day, detail with plan/log/results/run history, schedule
editor, approvals), Memory (graph + list + topics + import + personality files),
Integrations (connect flows, privacy filters, MCP servers), Skills (active,
pending review with diff, archived, evolution log), Notifications panel with
proactive suggestion approve/dismiss, Voice mode.

## M4 · Integration and real-machine verification (in progress 2026-09-15)
End-to-end runs on the founder's PC with local models: onboarding → chat with
tools → memory → task lifecycle → triggered task → proactive suggestion → skill
proposal → voice round trip. Screenshot review of every screen. Fix everything
found.

Done so far: every screen captured from seeded data and reviewed; engine gaps found by the
UI builders fixed (live notification updates, OAuth cancel, readable poll errors, skill
proposal editing, tool selection and a no-tools fallback for small local models).

## M4.5 · V3 leap (built 2026-09-15, real-model verification in progress)
Approved by the founder on 2026-09-15. Contract: `docs/API.md` sections 10-16.
- Biggest leaps: device nodes (desktop, phone web app, reference glasses node, `docs/NODES.md`),
  messaging channels with pairing (Telegram, Discord), browser control, code execution that calls tools,
  chat subagents, steering a running reply.
- Knowing the user: dialectic user model ("About you"), nightly memory consolidation (dreams), memory flush
  before compression, skills that propose their own repairs.
- Doing more on its own: change feeds and webhooks instead of timer polling, IMAP push, script jobs with no
  model calls, wake word and talk mode.
- Agent improvements: tool progress streaming, dynamic risk, parallel read tools, large-result guard, prompt caching.
- Not doing: multiple named assistants, open skill marketplace, network-exposed gateway by default.

## M5 · Packaging (later)
One installer per OS: bundled Python runtime (uv-managed) or frozen engine,
electron-builder NSIS/DMG/AppImage, auto-update, first-run model download helper.

## M6 · Devices and channels (moved into M4.5)
Device node protocol over the same engine (`node.hello` with capabilities:
audio in/out, camera, notify) for phone and the smart glasses; messaging
channels (Telegram first) with pairing codes.
