# Sentient - notes for AI coding assistants

Sentient v3 is a **desktop application** (Electron shell + React renderer + a local
Python backend). It is a personal assistant whose USPs are long-running tasks,
memory, proactivity, integrations and voice. Read `docs/ARCHITECTURE.md` and the
contract in `docs/API.md` before changing anything.

## Where things are
- `desktop/` - Electron main/preload + React/TypeScript renderer (the product UI).
- `sentient/` - Python backend the desktop app launches (`python -m sentient serve`).
- `src/` - v2 (FastAPI + 22 MCP servers + Celery + Next.js). **Reference only.** Port
  behaviour, field names, statuses and prompts out of it. Never add to it.
- `tests/` - pytest, `asyncio_mode=auto`, scripted `FakeProvider`; no network.
- `docs/` - architecture, API contract (sections 10-16 are the V3 leap features), roadmap, developing guide,
  `NODES.md` device protocol.

## Non-negotiables
- v2 behaviour of tasks, memory, proactivity, integrations and voice is preserved.
  When in doubt, read the v2 code and match it.
- No login, accounts, plans or feature gating. The gateway token is invisible to the user.
- One process, one SQLite file. No Redis, Celery, Mongo, Postgres, Docker requirements.
- LLM access only through `app.llm` with a *role* (`primary`, `fast`, `planner`,
  `executor`, `embedding`, `vision`) and optional explicit `model`. Never hard-code a
  model outside `config/schema.py` defaults.
- Secrets only via `sentient.secrets` (OS keychain). Never write keys to config or logs.
- Every tool declares a `Risk`. Sending, deleting, spending or executing is `send`/`exec`.
  Mark `internal=True` when the effect stays inside Sentient (its memory, skills, files folder,
  task list): approvals mode "ask" does not interrupt for internal `write` tools.
- New config keys go in your package's section of `config/schema.py` with a `description`.
- The UI consumes typed events and JSON from `docs/API.md`. If a shape changes, update
  `docs/API.md` in the same edit.

## Ownership (parallel development)
| Area | Owner | Files |
|---|---|---|
| core | core agent | `sentient/{app,events,services,paths,secrets}.py`, `agent/**` (incl. `subagents.py`), `llm/`, `store/`, `tools/`, `config/` (core sections + `SubagentsConfig`), `files/`, `notifications/`, `gateway/app.py`, `gateway/routes/{core,models,notifications,subagents}.py`, `cli.py` (except the `node` command), `tests/conftest.py`, `tests/test_*.py` |
| sandbox | sandbox agent | `sentient/sandbox/**`, `gateway/routes/sandbox.py`, `tests/sandbox/**`, `SandboxConfig` |
| browser | browser agent | `sentient/browser/**`, `gateway/routes/browser.py`, `tests/browser/**`, `BrowserConfig` |
| devices | nodes agent | `sentient/nodes/**`, `gateway/routes/nodes.py`, the `node` command in `cli.py`, `docs/NODES.md`, `tests/nodes/**`, `NodesConfig` |
| channels | channels agent | `sentient/channels/**`, `gateway/routes/channels.py`, `tests/channels/**`, `ChannelsConfig` |
| tasks | tasks agent | `sentient/tasks/**`, `gateway/routes/tasks.py`, `tests/tasks/**`, `TasksConfig` |
| integrations | integrations agent | `sentient/integrations/**`, `gateway/routes/{integrations,hooks}.py`, `tests/integrations/**`, `IntegrationsConfig` |
| memory | memory agent | `sentient/memory/**` (incl. `usermodel.py`, `dreaming.py`), `sentient/tools/builtin/memory_tool.py`, `gateway/routes/{memory,user_model}.py`, `tests/memory/**`, `MemoryConfig`, `UserModelConfig`, `DreamingConfig` |
| proactivity, evolution, skills | evolution agent | `sentient/proactivity/**`, `sentient/evolution/**`, `sentient/skills/**`, `sentient/tools/builtin/skills_tool.py`, `gateway/routes/{skills,proactivity}.py`, `tests/{proactivity,evolution}/**`, `ProactivityConfig`, `SkillsConfig`, `EvolutionConfig` |
| voice | voice agent | `sentient/voice/**`, `gateway/routes/voice.py`, `tests/voice/**`, `VoiceConfig` |
| desktop: conversation and devices | desktop agent A | `desktop/electron/**`, `desktop/src/features/{chat,devices,channels,browser}/**`, `desktop/src/pages/{devices,channels}/**`, shared shell files below |
| desktop: knowing you and automation | desktop agent B | `desktop/src/features/{memory,usermodel,tasks,automations,voice,skills,notifications,settings,integrations}/**`, `desktop/src/pages/{memory,tasks,skills,about,integrations}/**`, `desktop/src/lib/leap/**` |

Shared desktop files (`src/App.tsx`, `src/components/shell/**`, `src/lib/{api,types,events}.ts`) are owned by desktop
agent A. Desktop agent B puts new types and API calls in `src/lib/leap/{types,api}-b.ts` and registers its screens
through `src/lib/leap/routes-b.tsx` and its domain events through `src/lib/leap/events-b.ts`, which the lead has wired
into `App.tsx` and the sidebar. `pyproject.toml` and `package.json` belong to the lead: list new dependencies in your report.

Rules: edit only files you own. Need something from another area? Use its public
service methods (`app.tasks`, `app.integrations`, `app.memory`, `app.notify`, `app.bus`),
or leave a `TODO(owner):` note and mention it in your report. Put package-specific
test fixtures in `tests/<pkg>/conftest.py`; do not edit `tests/conftest.py`.
Package tables go in `sentient/<pkg>/schema.sql` (idempotent) and additive columns via
`store.ensure_column`.
Engine builders never call real models: tests use the scripted `FakeProvider`. The lead runs real-model checks
serially because parallel engines push Ollama onto the CPU. Desktop builds take `desktop/.build.lock` first.
Run long commands in the foreground; do not wait on background jobs or monitors.

## Commands
```
.venv/Scripts/python -m pytest -q
.venv/Scripts/ruff check sentient tests
.venv/Scripts/python -m sentient serve --port 7777      # backend only
cd desktop && npm run dev                                # desktop app (spawns backend)
```
