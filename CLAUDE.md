# Sentient: notes for AI coding assistants (and curious humans)

Sentient is a **desktop personal assistant**: an Electron + React app that starts a local Python engine. Its core
features are long-running tasks, memory, proactivity, integrations and voice, plus devices (phones, smart glasses),
messaging channels, browser control, code execution and self-improving skills.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the contract in [docs/API.md](docs/API.md) before changing
anything that crosses the desktop ↔ engine boundary. How to contribute is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Where things are
- `sentient/`: the engine the desktop app launches (`python -m sentient serve`). One package per area.
- `desktop/`: Electron main and preload (`desktop/electron`) and the React renderer (`desktop/src`).
- `tests/`: pytest with `asyncio_mode=auto`. Each area has a folder; core tests are `tests/test_*.py`.
- `docs/`: architecture, API contract, roadmap, developing guide, `NODES.md` (device protocol).
- `firmware/`: reference firmware for ESP32-S3 smart glasses. `packaging/`: the installer build.
- Sentient v2 (the old cloud web app) is on the `v2` branch only. Do not port anything new from it.

## Rules that keep Sentient Sentient
- No login, accounts, plans or feature gating. The gateway token between window and engine is invisible.
- One process, one SQLite file. No Redis, Celery, Postgres or required Docker.
- Nobody needs project keys: never add a `.env` requirement. Secrets go through `sentient.secrets` (the OS
  keychain) and never into config files, logs or tests.
- Models are used only through `app.llm` with a role (`primary`, `fast`, `planner`, `executor`, `embedding`,
  `vision`, `voice`) and an optional explicit model. Never hard-code a model outside `config/schema.py`.
- Every tool declares a `Risk`. Sending, deleting, spending or running code is `send` or `exec` and asks the user.
  Use `risk_fn` when risk depends on the arguments (a browser click on "Place order"). Mark `internal=True` when
  the effect stays inside Sentient (its memory, skills, files folder or task list).
- Safety checks stay deterministic: a model may raise a risk, never lower it.
- New config keys go in the right section of `sentient/config/schema.py` with a `description` (Settings is
  generated from it).
- The desktop app consumes typed events and JSON defined in `docs/API.md`. If a shape changes, update the
  contract in the same change.
- Small local models are first-class users. Keep prompts short and structured and parse model JSON tolerantly.

## Conventions
- Tests never touch the network or a real model: use the scripted `FakeProvider` in `tests/conftest.py` and
  `respx` for HTTP. Put area fixtures in `tests/<area>/conftest.py`.
- Package tables live in `sentient/<area>/schema.sql` (idempotent); add columns with `store.ensure_column`.
- Services subclass `sentient.services.Service`, start from `SentientApp` and talk to each other through their
  public methods (`app.tasks`, `app.memory`, `app.notify`, `app.bus`).
- Desktop: shared types and API calls live in `desktop/src/lib/{types,api,events}.ts`; features under
  `desktop/src/features/<area>`. Reuse `desktop/src/components/ui`.
- User-facing text is plain language for non-technical people: say "device", not "node"; no jargon.

## Commands
```
.venv/Scripts/python.exe -m pytest -q          # engine tests (macOS/Linux: .venv/bin/python)
.venv/Scripts/ruff.exe check sentient tests    # lint
.venv/Scripts/python.exe -m sentient serve     # engine only
cd desktop && npm run dev                      # desktop app (starts the engine)
cd desktop && npm run typecheck && npm run build
cd desktop && npm run package                  # installer for this OS
```
