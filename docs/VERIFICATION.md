# Verified on a real PC

What has actually been tested end to end, on what hardware, and what has not. Updated with each round of real-model testing. Automated tests (CI) are described in [DEVELOPING.md](DEVELOPING.md).

Tested on 2026-09-15 on a Windows 11 laptop (RTX 4060 with 8 GB, 15 GB RAM) with local Ollama models only:
`qwen3:8b` for every chat role and `nomic-embed-text` for embeddings. Scripts drove the real engine through
its public REST and WebSocket interface, and the desktop screens were captured from the same data.

**Core flows**

| Flow | Result |
|---|---|
| Onboarding, chat with a tool call, recurring task, triggered task | Pass |
| Memory: learn a fact in chat and recall it later | Pass |
| One-off task: plan, approve, run, file created, report, notification | Pass |
| Proactive suggestion from a calendar event | Pass |
| Skill proposed for review after multi-step work | Pass |
| Voice turn with system text-to-speech | Pass. First token in about 6 s |

**New in this release**

| Flow | Result |
|---|---|
| Code execution: the model writes and runs Python that returns a number | Pass |
| Steering: a message sent mid-reply changes the answer | Pass |
| Subagents: a helper runs with its own tools and reports back | Pass |
| Browser: open a local shop page and read a price | Pass |
| Browser: "Place order" asks for approval as a purchase; declining stops it | Pass |
| Devices: pair simulated glasses, show text on them, read their location | Pass |
| User model refresh and nightly consolidation (Pune to Bengaluru contradiction settled) | Pass |
| Skill repair: a failed step produces a fix proposal with the reason | Pass |
| Webhook: wrong secret refused, right secret starts a triggered task | Pass |
| Watcher script job alerts with no model call | Pass |
| Wake word: synthesized "Hey Sentient, what is two plus two?" answered by voice | Pass. About 10 s end to end |
| Sandbox status and a direct run | Pass |

**Also verified on 2026-09-16**

| Check | Result |
|---|---|
| The same 12 flows on cloud models: Claude Sonnet 5 for chat, Claude Haiku 4.5 for background work | Pass, and much faster: most flows finish in 5 to 15 s |
| Smart glasses over the home network: mDNS discovery, TLS with certificate pinning, pairing code, text and notification shown on the device | Pass with the reference device program |
| "What can you see?" through a device camera, using a real webcam frame and a local vision model | Pass with `ollama_chat/qwen2.5vl:3b` |
| Telegram with a real bot and a real phone, on `qwen3:8b`: pairing code, chat, approval buttons before running code (then the Docker sandbox), weather, `/help`, and two voice notes understood by faster-whisper | Pass. Replies to voice notes are text unless `channels.telegram.voice_replies` is on |

The automated suite (522 engine tests, desktop typecheck and build) passes.

Real-model runs found and fixed these problems, each now covered by a test: a memory update dropped a
city; a task run stopped after announcing its next step; the model passed a whole snapshot line as a browser
element reference, which hid a purchase from the approval check; empty replies after a tool call; scripts
that forgot to import `result`; the tiny wake-word model hearing "Hey" as "He"; and a model claiming a declined
action had happened (declined results now say plainly that nothing was done).

**Not yet verified on real hardware or accounts:**

- Integrations with real Google, GitHub, Slack, Notion, Discord, Trello or WhatsApp accounts, and a paired
  Discord chat. Spoken replies to Telegram voice notes.
- A real phone and real glasses hardware. The device protocol was exercised over the network with the
  reference device program, and the camera path with this laptop's webcam. The firmware in `firmware/` has
  never been compiled or flashed.
- The Docker code-execution backend (Docker was not running) and IMAP push against a real mail server.
- A live microphone in voice mode. Wake word and speech were driven with synthesized audio.
- The macOS and Linux installers. The Windows installer is built and verified: a per-user setup that needs no
  admin rights, installs in about a minute, starts the app with its own bundled engine (no Python needed)
  and uninstalls without touching your data. It is not code signed yet, so Windows will warn on first run.

**Known limits:** older `qwen3:4b` pulls cannot call tools in Ollama, so use `qwen3:8b` or larger. Photo
and screen questions need a vision-capable model. With qwen3:8b on this laptop, chat replies with tools take
about 15 to 45 seconds; cloud models are much faster. Running several engines at once on 8 GB of VRAM pushes
Ollama onto the CPU.
