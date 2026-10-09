# AGENTS.md

Guidance for AI coding agents working on Sentient. Humans are welcome to read it too; the contribution workflow
is in [CONTRIBUTING.md](CONTRIBUTING.md).

Sentient is a **desktop personal assistant**: an Electron + React app that starts a local Python engine. Its core
is long-running tasks, memory, proactivity, integrations and voice, plus devices (phones, smart glasses),
messaging channels, browser control, code execution and self-improving skills.

Area guides with deeper conventions: [`sentient/AGENTS.md`](sentient/AGENTS.md) for the engine and
[`desktop/AGENTS.md`](desktop/AGENTS.md) for the app.

## Before you change anything

- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the shape of the system.
- Anything that crosses the window ↔ engine boundary is defined in [docs/API.md](docs/API.md). Change the contract
  in the same change as the code.
- The reasons behind the big rules are in [docs/adr/](docs/adr/README.md). If your change reverses one of them,
  add a new ADR instead of quietly working around it.

## Repository map

| Path | What it is |
|---|---|
| `sentient/` | The engine (`python -m sentient serve`). One package per area: `agent`, `llm`, `store`, `tools`, `gateway`, `tasks`, `memory`, `proactivity`, `evolution`, `skills`, `integrations`, `channels`, `nodes`, `browser`, `sandbox`, `terminal`, `voice` |
| `desktop/` | Electron main and preload (`desktop/electron`) and the React renderer (`desktop/src`) |
| `tests/` | pytest, `asyncio_mode=auto`. One folder per area; core tests are `tests/test_*.py` |
| `docs/` | Architecture, API contract, ADRs, devices protocol, guides |
| `firmware/` | Reference firmware for ESP32-S3 smart glasses (never compiled in CI) |
| `packaging/` | Freezing the engine for installers |

Sentient v2 (the old cloud web app) is on the `v2` branch only. Do not port new code from it.

## Rules every change keeps

- **No accounts.** No login, plans or feature gating. The window ↔ engine token is invisible to users.
- **One process, one SQLite file.** No Redis, Celery, Postgres or required Docker.
- **No shared keys.** Never add a `.env` requirement. Secrets go through `sentient.secrets` (the OS keychain) and
  never into config files, logs, tests or fixtures.
- **Models through roles.** Use `app.llm` with a role (`primary`, `fast`, `planner`, `executor`, `embedding`,
  `vision`, `voice`) and an optional explicit model. Never hard-code a model outside `config/schema.py`.
- **Every tool declares a `Risk`.** Sending, deleting, buying or running code is `send` or `exec` and asks the user.
  Use `risk_fn` when risk depends on the arguments; mark `internal=True` when the effect stays inside Sentient.
- **Safety is deterministic.** A model may raise a risk, never lower it.
- **Config is self-describing.** New keys go in the right section of `sentient/config/schema.py` with a
  `description`; the Settings screen is generated from it.
- **Small local models are first-class.** Short, structured prompts; tolerant JSON parsing; never assume a frontier
  model.
- **Plain language in the UI.** Users are not engineers: "device", not "node"; no jargon; no em-dashes.

## How to work

1. Find the area guide and the tests for what you are changing. Read the existing code before adding new code.
2. Make the smallest change that solves the problem. Match the surrounding style.
3. Add or update tests. Tests never touch the network or a real model: use the scripted `FakeProvider` from
   `tests/conftest.py` and `respx` for HTTP.
4. Run the checks for what you touched, then the full suite before you open a pull request.
5. Update `docs/API.md` for contract changes, an ADR for decision changes, and `CHANGELOG.md` for user-visible
   changes.

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                 # engine tests (macOS/Linux: .venv/bin/python)
.venv/Scripts/python.exe -m pytest -q tests/tasks     # one area
.venv/Scripts/ruff.exe check sentient tests packaging # lint
.venv/Scripts/python.exe -m sentient serve            # engine only
cd desktop && npm run dev                             # the app (starts the engine)
cd desktop && npm run typecheck && npm run build      # desktop checks
cd desktop && npm run package                         # installer for this OS
```

CI runs the same checks on every pull request, on Linux and Windows, with no secrets.
