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

## M4 · Integration and real-machine verification (done 2026-09-16)
End-to-end runs on the founder's PC with local models: onboarding → chat with
tools → memory → task lifecycle → triggered task → proactive suggestion → skill
proposal → voice round trip. Screenshot review of every screen. Fix everything
found.

Done so far: every screen captured from seeded data and reviewed; engine gaps found by the
UI builders fixed (live notification updates, OAuth cancel, readable poll errors, skill
proposal editing, tool selection and a no-tools fallback for small local models).

## M4.5 · V3 leap (done 2026-09-16: verified on local qwen3:8b and on Claude Sonnet 5)
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

## M5 · Packaging (Windows done 2026-09-16)
A non-technical person double-clicks one installer and gets a working assistant
with no Python and no terminal.
- Engine frozen with PyInstaller (`packaging/sentient-engine.spec`, onedir) into
  `sentient-engine.exe`, shipped as `resources/engine` by electron-builder.
  `electron/main/paths.ts` prefers it in a packaged build; `npm run dev` still runs
  `.venv/Scripts/python.exe -m sentient serve`. `~/.sentient` in both modes.
- `npm run package` → `desktop/dist/Sentient-Setup-<version>.exe`: per-user NSIS,
  no administrator rights, ~216 MiB. Installed, launched, screenshotted and
  uninstalled on the founder's PC; uninstall leaves `~/.sentient` intact.
- Heavy optional extras (faster-whisper, CTranslate2, onnxruntime, Kokoro,
  openWakeWord, OpenCV, PyTorch) and browser binaries are left out on purpose;
  a runtime hook explains what to use instead.
- Still to do: macOS DMG and Linux AppImage are configured but unbuilt and
  unsigned, no code signing or notarization, no auto-update feed.

## M6 · Devices and channels (moved into M4.5)
Device node protocol over the same engine (`node.hello` with capabilities:
audio in/out, camera, notify) for phone and the smart glasses; messaging
channels (Telegram first) with pairing codes.

## Open source and contributors (2026-10-08)
- v3 became the default branch (`main`); v2 moved to the `v2` branch and `v2-final` tag.
- GitHub flow: forks, small pull requests to `main`, one maintainer approval, squash merge.
- CI on every pull request with no secrets needed; release workflow builds installers from version tags.
- Issue forms, area labels applied automatically, `good first issue` and `help wanted` backlog.

## Next
- Signed installers and auto-update; first public v3 release.
- macOS and Linux installers verified on real machines (the release workflow builds them).
- Real-account testing of integrations and messaging channels; real phones and the glasses hardware.
