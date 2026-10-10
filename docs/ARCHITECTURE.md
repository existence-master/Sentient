# Sentient v3 architecture

*Living document. Direction set 2026-09-12, refined 2026-09-15. The reasons behind each major choice are recorded as
[architecture decision records](adr/README.md).*

## What Sentient is

Sentient is a **desktop personal assistant** that runs on your own computer. You
install one app, answer a few onboarding questions, pick a local or cloud model,
and it starts working for you: chatting, remembering, running long tasks on a
schedule or when something happens in your apps, suggesting things before you
ask, and talking with you by voice. There is no account, no login and no plan.

The shape follows the Hermes Agent desktop app: a native window over a local
agent engine. Sentient's own USPs from v2 are preserved exactly:

- **Long-running tasks**: one-off, scheduled once, recurring, triggered by app
  events, and swarm (parallel sub-agents). Refine → plan → approve → execute →
  structured result, with full run history and task chat for replanning.
- **Memory**: atomic facts about you with topics, short/long term decay, CUD
  against similar memories, a memory graph, episodic conversation summaries and
  history search.
- **Proactivity**: watches connected apps, reasons over your context and
  suggests actions, learning per suggestion type from approve/dismiss.
- **Integrations**: Google Workspace, GitHub, Slack, Notion, Discord, Trello,
  WhatsApp, web search, weather, maps, news, charts, and any MCP server.
- **Voice mode**: hands-free conversation, and the path to the smart glasses.

What changed from v2 is everything around them: model configuration is fully
user-controlled, the assistant evolves itself (Hermes-style skills, reviewer,
curator, profile upkeep), the UI is rebuilt, and the stack runs as one local
process with one database file.

## Process model

```
┌────────────────────────── Sentient.app (Electron) ──────────────────────────┐
│ main process                                                                │
│  • picks a free port + random token, spawns the engine, restarts on crash   │
│  • window, tray, global shortcuts, screen sharing, notifications, autostart │
│  • preload bridge window.sentient {getConnection, openExternal, ...}        │
│                                                                             │
│ renderer (React + TypeScript)                                               │
│  Chat · Tasks · Memory · Integrations · Skills · Notifications · Voice ·    │
│  Settings (models, personality, …) · Onboarding                             │
│        │ REST (Bearer token)          │ WebSocket /ws  (chat + domain events)│
│        │                              │ WebSocket /ws/voice (audio)          │
└────────┼──────────────────────────────┼─────────────────────────────────────┘
         ▼                              ▼
┌────────────────────── engine: python -m sentient serve ─────────────────────┐
│ gateway (FastAPI, 127.0.0.1 only)  routes/{core,models,notifications,tasks, │
│                                     integrations,memory,skills,proactivity, │
│                                     voice}.py                               │
│ SentientApp (composition root)                                              │
│   Agent.run_loop ── LLM roles (LiteLLM) ── Ollama / OpenAI / Anthropic / …  │
│   ToolRegistry ── builtin tools · integration plugins · MCP servers         │
│   services: notifications · integrations · subagents · sandbox · terminal · │
│             browser · nodes · tasks · proactivity · evolution · user model ·│
│             dreaming · voice · channels          EventBus → /ws             │
│   FactMemory (sqlite-vec) · Workspace markdown · SkillLibrary               │
│ Store: ~/.sentient/sentient.db (SQLite WAL, FTS5, sqlite-vec)               │
│ Secrets: OS keychain                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Outside the window, three optional doors lead to the same engine:

- **Devices LAN listener** (`nodes.lan_enabled`, off by default): TLS on your home network, pairing code
  first, then a per-device token. It serves only `/ws/node`, `/ws/voice` for device tokens, the web device
  app at `/node/`, device uploads and `POST /hooks/{id}`. The main `/api` never reaches the network.
  mDNS announces `_sentient._tcp.local.` so glasses can find the computer.
- **Messaging channels**: outbound long polling (Telegram) or gateway connections (Discord) from the engine,
  so nothing listens for them; only paired chats are answered.
- **Webhooks**: `POST /hooks/{id}` with a per-hook secret on loopback (and the LAN listener when on).

The engine is a separate process so heavy work (models, speech, long tasks)
never freezes the window, and so phones and glasses talk to the
same engine. The token between window and engine is generated per launch and is
never shown to the user.

## Backend packages

| Package | Responsibility | v2 origin |
|---|---|---|
| `sentient/app.py` | composition root, start/stop order, `notify`, `save_config` | `main/app.py`, supervisord |
| `sentient/services.py` | Service base with resilient periodic jobs | Celery beat |
| `sentient/events.py` | in-process EventBus forwarded to the window | websocket manager |
| `sentient/config` | Pydantic config (JSON schema drives Settings) | 350 lines of env vars |
| `sentient/secrets.py` | OS keychain | `.env`, static-IV AES |
| `sentient/store` | SQLite + FTS5 + sqlite-vec; packages add `schema.sql` | Mongo, Postgres, Chroma, Redis |
| `sentient/llm` | roles, fallbacks, reasoning effort, streaming tool calls, embeddings | qwen-agent, LiteLLM proxy |
| `sentient/agent` | prompt assembly, `run_loop` engine, `run_turn` chat, steering, approvals (effective risk), subagents, context compression, auto titles | `main/chat/utils.py` two-stage pipeline |
| `sentient/files` | attachment and document text extraction | textract in file MCP |
| `sentient/notifications` | persisted notifications + live push | `main/notifications` |
| `sentient/tasks` | task state machine, scheduler, planner/executor/result, triggers, swarm | `main/tasks`, `workers/*` |
| `sentient/integrations` | plugins with native OAuth/API keys, privacy filters, pollers' data source, MCP client | 22 `mcp_hub` processes, Composio |
| `sentient/memory` | facts with topics/CUD/decay, graph, import, summaries, workspace, personas | `mcp_hub/memory`, `main/memories`, history MCP |
| `sentient/proactivity` | pollers, pre-filter, context gathering, reasoner, learned thresholds, heartbeat | `workers/proactive` (recovered from git history) |
| `sentient/evolution`, `sentient/skills` | skill reviewer, curator, profile upkeep, SKILL.md library | new (Hermes Agent) |
| `sentient/voice` | STT/TTS providers, VAD, wake word and talk mode, device audio (pcm16), `/ws/voice` loop | `main/voice` (FastRTC) |
| `sentient/sandbox` | `execute_code`: Python scripts calling tools over a one-time loopback bridge; process or Docker backend | new (Hermes programmatic tool calling) |
| `sentient/terminal` | `terminal_run`: approved commands on the host in allowed folders, built-in blocklist, secrets kept out, off by default (ADR 0019) | new (Hermes local terminal) |
| `sentient/browser` | Playwright on the installed Edge/Chrome, snapshot refs, risk heuristics, headed sign-in, live frames | new (OpenClaw, Hermes) |
| `sentient/nodes` | device pairing, invoke/result protocol, LAN TLS listener, mDNS, device tools, web device app, reference node | new (OpenClaw nodes) |
| `sentient/channels` | Telegram and Discord bots with pairing codes, streaming replies, approvals as buttons, delivery | new (OpenClaw, Hermes gateway) |
| `sentient/memory/usermodel.py`, `dreaming.py` | dialectic user model (insights, evidence, questions) and nightly consolidation | new (Honcho-style, OpenClaw dreaming) |

Ownership and the renderer contract are in `CLAUDE.md` and `docs/API.md`.

## Key decisions

### Models are roles the user controls
`models.roles` maps jobs to `provider/model` strings: `primary` (chat),
`fast` (background JSON work), `planner`, `executor`, `embedding`, `vision`, and
`voice` (spoken turns, reasoning off by default).
Each role has an optional fallback chain, reasoning effort and temperature, and
Ollama roles get an explicit context length (`num_ctx`). An
explicit pick (per chat message, per task) is strict: no silent fallback.
Settings → Models lists installed Ollama models, provider suggestions, key status
and a one-click test that also reports tool-calling support. Keys go to the OS
keychain.

Local-model note: Ollama's `think` flag must be passed (as `reasoning_effort`)
or qwen3's reasoning leaks into visible text through LiteLLM streaming; the
`fast` role runs with thinking off, which is several times faster.

### One reusable agent engine
`Agent.run_loop` streams typed events (`thinking_delta`, `text_delta`,
`tool_call`, `tool_result`, `approval_request`, `usage`, `error`), runs tools,
asks for approval when configured, records token usage and fills a
`LoopResult`. Chat turns, task runs, swarm workers, subagents, proactive context
gathering and voice all use it, so a fix to the loop fixes every surface.

Each tool call runs in its own asyncio task. A context variable routes the tool's
`ctx.progress(...)` to a queue the loop drains while the tool works, so code
output, browser frames and subagent steps reach the window as `tool_progress`
before the result. Before running, every call's *effective* risk is computed
(`Tool.risk_fn`), an optional `policy` may refuse it, and approvals are decided;
consecutive look-ups that need no approval then run concurrently while results
keep the model's order. Oversized results are cut for the model and saved in
full under `files/outputs/`.

Steering: `run_turn` registers a `SteerQueue` per session. `agent.steer()` (the
`chat.steer` message, `chat.send` during a reply, channels, voice) appends to
it; the loop applies queued text at the next round boundary as a persisted user
message, and text that arrives after the last round starts a new turn.

Subagents (`agent/subagents.py`, tools `delegate_task`/`delegate_tasks`): a
`SubagentManager` runs isolated `run_loop`s with a focused prompt on the
`subagents.role` model, bounded by rounds, a timeout and a concurrency limit,
recorded in the `subagents` table and published as `subagent.updated`. Its
policy refuses delegation, `send`/`exec` calls and anything needing approval.
Foreground subagents stream steps to the parent call; background ones post their
summary into the chat and notify.

Hooks: the system prompt includes `app.user_model.context_for(text)` (2 s
budget), context compression first calls `memory.flush_conversation`, and
`chat.turn_completed` carries tool errors and viewed skills for skill repair.

### Memory is pushed, not pulled
Each turn the system prompt carries SOUL.md (persona), USER.md, MEMORY.md,
recent daily notes, recalled facts, the skill index, the clock and a running
summary of long conversations. v2 relied on the model deciding to call a memory
tool before it knew anything about you. Tools remain for explicit recall,
remembering, forgetting and history search. Memories learned from outside
content (an email, a web page), from work nobody asked for, or from an import
wait in a review inbox and reach no prompt until you approve them (ADR 0021).

### Tasks keep v2 semantics, lose Celery
Same statuses, JSON field names and stages as v2 (so the task UI ports
directly). The scheduler is an in-process loop that atomically claims due
tasks; runs are bounded by a semaphore and a timeout, checkpoint their
transcript for resume after restart, and stream progress to the window.

### Self-evolution with a human in the loop
After substantial chats and task runs a background reviewer proposes a skill
(SKILL.md with When to use / Procedure / Pitfalls / Verification). With
`skills.write_approval` on, proposals wait in Skills → Pending. A curator
retires unused skills; profile upkeep refreshes MEMORY.md and appends learned
facts to USER.md without overwriting your own text.

### Doing more with small local models
Local models are slow per round and weak at long tool chains, so the engine helps them: tools are offered
by relevance (with trigger phrases for browser, code, devices and helpers), read-only tool calls in one
round run in parallel, big results are cut and saved to a file, change feeds and watcher scripts avoid
model calls entirely, and a task run that only announces its next step is nudged to actually do it. A round that ends with an empty
reply gets one request to answer, and scripts can call `result` and `tools` without importing them.
A model update of a memory may not silently drop a name, place or number.

### Safety defaults
Loopback bind with a per-launch token; approvals before tools that change things
outside Sentient, send, delete or execute (off / ask / always, "allow for this chat"),
while the assistant's own memory, skill proposals, files folder and task list never
interrupt; secrets only in the
keychain; file tools confined to `~/.sentient/files`; external links open in the
system browser; strict renderer CSP with context isolation and sandboxing.

Approvals use each call's effective risk: a browser click on "Place order" or a device photo is a `send`
that asks every time, even after "allow for this chat". Lasting rules per app or tool (Allow, Ask, Never;
[ADR 0016](adr/0016-lasting-approval-rules.md)) are applied in code before the mode, and purchases ask even under
Allow. The browser never types into password, card or
one-time-code fields. Scripts from `execute_code` get no API keys and may only call read and internal tools.
Subagents cannot send, execute or spawn more subagents. Messaging bots answer paired chats only; device and
hook secrets are stored as hashes; bot tokens live in the keychain and are scrubbed from logs.

**Stop everything** (`app.stop_all`, API section 17) is code, never a model call: it saves a stopped flag, cancels
every running reply, task run, helper and background job (each service's `halt()`), and pauses schedules,
triggers, feeds, webhooks, proactivity and dreaming (`Service.pause_on_stop`) until Resume, across restarts.

## On disk

```
~/.sentient/
  config.yaml      everything configurable
  sentient.db      all state
  workspace/       SOUL.md USER.md MEMORY.md notes/
  skills/          <name>/SKILL.md · pending/ · archived/
  files/           assistant outputs · uploads/ · charts/
  models/          downloaded voice and wake-word models
  browser/         Sentient's own browser profile (you sign in here yourself)
  sandbox/         short-lived working folders for code runs
  tls/             self-signed certificate for the devices LAN listener
  logs/            backend.log
```
