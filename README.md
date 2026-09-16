<div align="center">

![Sentient](./.github/assets/banner.png)

<h1>Sentient</h1>
<p><b>Your personal assistant, running on your own computer.</b></p>

</div>

Sentient is a desktop app that learns who you are, runs long tasks for you in the background,
notices things in your apps before you ask, and talks with you by voice. It works with a model on
your own machine (Ollama, LM Studio) or any cloud model you choose. There is no account and no login:
your data lives in one folder on your computer and your keys stay in your system keychain.

> **Status: v3 alpha.** The app runs from source today. An installer is next. See
> [Verified on a real PC](#verified-on-a-real-pc) for exactly what has been tested.

## What it does

- **Long-running tasks.** Describe work in plain language. Sentient refines it, drafts a plan you
  approve, and runs it: once, on a schedule ("every weekday at 9"), when something happens ("when an
  email arrives from…"), or as a swarm of parallel agents. Every run keeps a live log and a structured
  result, and you can chat with a task to change its plan.
- **Memory.** Atomic facts about you, grouped into topics, with short-term facts that expire, a memory
  graph, document import, conversation summaries, and editable profile files. What it remembers is
  used in every conversation, and you can correct or forget anything.
- **Proactivity.** Sentient watches connected apps such as Gmail and Calendar, reasons over your
  context, and suggests actions. Approve and it becomes a task; dismiss and it learns what you don't want.
- **Integrations.** Gmail, Google Calendar, Drive, Docs, Sheets, Slides and Contacts, GitHub, Slack,
  Notion, Discord, Trello and WhatsApp, plus web search, weather, maps, news, web pages and charts that
  need no setup. Add any MCP server for more.
- **Voice mode.** Hands-free or push-to-talk conversation with interruptions, local speech-to-text
  (faster-whisper) and your choice of system, local neural or cloud voices.
- **Self-evolution.** After substantial work Sentient proposes reusable skills (inspired by Hermes
  Agent). You review each proposal with a diff before it becomes active; unused skills are retired.
- **Your devices.** Pair a phone or smart glasses with a six-digit code. Sentient can show text on them,
  speak through them, check where you are, or look through the camera when you ask. The desktop app is a
  device too (screen, camera, clipboard), and a documented protocol lets anyone build their own hardware.
- **Message it from anywhere.** Connect your own Telegram or Discord bot and pair your chat. Replies stream
  in, voice notes are transcribed, approvals arrive as buttons, and task results and suggestions follow you.
- **Hands on the web and on data.** Sentient can drive a real browser for sites without an integration (you
  sign in yourself; purchases and posts always ask first) and write short Python scripts that call its tools,
  so a dozen lookups become one step.
- **Helpers and steering.** It can hand work to helper agents that run beside your chat, and you can add a
  message while it is still replying to change course.
- **It gets to know you.** Beyond facts, Sentient keeps an evolving picture of your preferences, goals and
  style that you can confirm or correct on the About you page. Overnight it consolidates memory: merging
  duplicates, settling contradictions and writing a short journal of what changed.
- **Acts on its own, cheaply.** Gmail and Calendar changes arrive within a minute with no model calls, IMAP
  mail is pushed, webhooks let any app start a task, and watcher jobs run small scripts on a schedule and only
  wake the model when something happens. Say "Hey Sentient" for hands-free voice.
- **Any model, any job.** Pick different models for chat, background work, planning, running tasks,
  voice, vision and embeddings, with fallbacks and a one-click test that checks tool support.
- **Safe by default.** Sending, deleting or running anything asks for approval first. The engine only
  listens on your own machine.

## Install it

Windows: build the installer with `cd desktop && npm run package`, then run `desktop/dist/Sentient-Setup-*.exe`.
It installs for your user only, brings its own engine, and keeps your data in `~/.sentient`. Prebuilt
downloads and signed builds come next; macOS and Linux packaging is configured but not built yet.

## Run it from source

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 22, and either
[Ollama](https://ollama.com/download) or an API key for a cloud provider.

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[voice]"
cd desktop && npm install && npm run dev
```

On macOS or Linux use `.venv/bin/python`. The window starts the engine for you and walks you
through setup. Full developer notes are in [docs/DEVELOPING.md](docs/DEVELOPING.md).

## How it is built

- `desktop/` Electron window with a React and TypeScript interface.
- `sentient/` the local engine the window starts: agent loop with subagents and steering, tasks,
  memory and the user model, proactivity, self-evolution, integrations, browser, code execution,
  devices, messaging channels and voice, backed by a single SQLite database.
- `docs/NODES.md` the device protocol for phones, glasses and your own hardware.
- `docs/` [architecture](docs/ARCHITECTURE.md), the [window-to-engine contract](docs/API.md) and the
  [roadmap](docs/ROADMAP.md).
- `src/` the previous cloud version (v2), kept as a reference while the last pieces are ported.

## Verified on a real PC

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
| Telegram bot connects and polls with a real bot token | Pass. Pairing a chat needs a person on Telegram |

The automated suite (522 engine tests, desktop typecheck and build) passes.

Real-model runs found and fixed these problems, each now covered by a test: a memory update dropped a
city; a task run stopped after announcing its next step; the model passed a whole snapshot line as a browser
element reference, which hid a purchase from the approval check; empty replies after a tool call; scripts
that forgot to import `result`; the tiny wake-word model hearing "Hey" as "He"; and a model claiming a declined
action had happened (declined results now say plainly that nothing was done).

**Not yet verified on real hardware or accounts:**

- Integrations with real Google, GitHub, Slack, Notion, Discord, Trello or WhatsApp accounts, and a paired
  Telegram or Discord chat. The Telegram bot itself was connected live; pairing needs a person to send the code.
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

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the ownership notes in
[CLAUDE.md](CLAUDE.md).

## License

Distributed under the GNU AGPL. See [LICENSE.txt](LICENSE.txt).

## Team

<table>
  <tr>
     <td align="center">
       <a href="https://github.com/itsskofficial">
         <img src="https://avatars.githubusercontent.com/u/65887545?v=4?s=100" width="100px;" alt=""/>
         <br />
         <sub><b>itsskofficial (Sarthak)</b></sub>
       </a>
     </td>
     <td align="center">
       <a href="https://github.com/kabeer2004">
         <img src="https://avatars.githubusercontent.com/u/59280736?v=4" width="100px;" alt=""/>
         <br />
         <sub><b>kabeer2004</b></sub>
       </a>
     </td>
  </tr>
</table>
