# Developing Sentient v3

Sentient is a desktop app: an Electron window (`desktop/`) over a local Python
engine (`sentient/`). The window starts the engine for you.

## Prerequisites
- Windows 11, macOS or Linux
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node 22 and npm
- [Ollama](https://ollama.com/download) with a tool-capable chat model and an embedding model, or an API key for a cloud provider

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev,voice]"   # Windows
# uv pip install --python .venv/bin/python -e ".[dev,voice]"          # macOS / Linux
cd desktop && npm install
```

Optional GPU speech-to-text on NVIDIA without a system CUDA install:

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[voice-gpu]"
```

## Run the desktop app

```bash
cd desktop && npm run dev
```

The Electron main process picks a free port, generates an invisible token, starts
`.venv/Scripts/python.exe -m sentient serve`, waits for it to be healthy, then
shows onboarding (first run) or chat. Engine logs: `~/.sentient/logs/backend.log`.

Use an isolated profile while developing:

```bash
SENTIENT_HOME=/tmp/sentient-dev npm run dev
```

Override the Python interpreter with `SENTIENT_PYTHON=/path/to/python`.

### Test homes and the keychain

API keys, tokens and sign-ins live in the OS keychain, never in the data folder. Each data folder has its own
keychain namespace (issue #302): the default `~/.sentient` uses the service `sentient`, and any other
`SENTIENT_HOME` uses `sentient-<hash of the folder>`, so a test or demo home starts with no keys and can't read,
overwrite or delete yours. The engine logs which service it uses when it starts (`keychain service '...'`).

To test against your real sign-ins on purpose (for example your Composio sign-in), start the engine with
`SENTIENT_KEYCHAIN_NAMESPACE=default`; it then shares the default folder's entries, so anything it changes or removes
changes yours. Any other value (`SENTIENT_KEYCHAIN_NAMESPACE=e2e`) gives a named namespace (`sentient-e2e`) that
several test homes can share. Remove test entries when you are done (Windows: Credential Manager, entries ending in
`@sentient-<namespace>`).

### Renderer in a normal browser

```bash
.venv/Scripts/python.exe -m sentient serve --port 7777     # prints a dev URL with a token
cd desktop && npm run dev:web
```

Open the Vite URL with `?api=http://127.0.0.1:7777&token=<token>`.

## Desktop scripts

| Command | What it does |
|---|---|
| `npm run dev` | Electron + hot reload, engine started automatically |
| `npm run dev:web` | Renderer only, in a browser |
| `npm run typecheck` | TypeScript, zero errors expected |
| `npm run build` | Production build into `desktop/out/{main,preload,renderer}` |
| `npm run package:engine` | Freeze the Python engine into `desktop/build/engine` |
| `npm run package` | Frozen engine + installer for this OS (see Packaging) |
| `node scripts/smoke.mjs <route> <out.png> [WxH]` | Launch the built app, capture a route to PNG, quit |

In Git Bash write smoke routes without a leading slash (`settings/models`), or MSYS
turns them into Windows paths. Set `SENTIENT_HOME` to a seeded profile to capture
realistic screens. Demo data seeders live in `desktop/scripts/seed-*.py` (`seed-chats-devices-channels.py` for
chat tool cards, devices, helpers and messaging channels; `seed-automations-usermodel.py` for About you, dreams,
watcher jobs and webhooks). When they finish, the seeders pause every task in the home, decline any task still
being planned, cancel any run in progress (so resuming a seeded task never carries seeded work on with the model) and turn
off messaging apps and every paired chat's delivery (`desktop/scripts/seed_safety.py`), and `smoke.mjs` does the
same before it starts the app on a `SENTIENT_HOME`, so a demo never calls a model, runs a task or messages anyone. Pass `--keep-live` to either to keep
them as seeded. `seed_safety.py` refuses the default `~/.sentient`.

When several people build and capture screenshots at once, take `desktop/.build.lock`
(write your name, delete it when done) because `desktop/out` is shared.

## Engine

```bash
.venv/Scripts/python.exe -m sentient serve --port 7777   # engine only
.venv/Scripts/python.exe -m sentient doctor              # checks models, database, sqlite-vec
.venv/Scripts/python.exe -m sentient config show --defaults
.venv/Scripts/python.exe -m sentient node --url ws://127.0.0.1:7777/ws/node --code 123456   # simulated glasses
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check sentient tests
```

To check how background work behaves on battery on a computer that is plugged in, start the engine with
`SENTIENT_DEBUG_BATTERY` set: `ac`, `battery`, or a percent such as `35` or `10` (tests and debugging only; the
engine logs a warning while it is set). The engine log then shows `model job held: ... (on battery ...)`,
`upkeep burst on battery` and `model job released: ...` lines.

To try the Telegram channel without a real bot, point the engine at a Bot API server on this computer with
`SENTIENT_TELEGRAM_API=http://127.0.0.1:<port>` (a test double, or Telegram's own local Bot API server). Every Bot API
call, file downloads included, then goes there instead of `https://api.telegram.org`. Use it with a test
`SENTIENT_HOME`, whose keychain namespace keeps a made-up token away from your real one.

No test touches the network; the LLM is a scripted, deterministic fake
(`tests/conftest.py`). Browser tests drive a hidden local Edge or Chrome against local pages and skip
when none can start; Docker sandbox tests skip when Docker is not running. The suite is large: on a busy
machine run it per folder (`tests/test_*.py`, `tests/tasks`, `tests/memory` ...).

The web device app is served at `http://127.0.0.1:<port>/node/` so it can be tried on the same computer
(camera and microphone work on localhost). For a phone, turn on Devices → Allow devices on my Wi-Fi.

### Changing the database

Everything lives in one SQLite file ([ADR 0003](adr/0003-single-sqlite-database.md)). Two things describe its shape
([ADR 0030](adr/0030-numbered-migration-files.md)):

- the `schema.sql` files (`sentient/store/schema.sql` and one per package in `PACKAGE_SCHEMAS`): idempotent
  `CREATE ... IF NOT EXISTS` statements for the **whole current shape**. A new database is made from them alone.
- numbered steps in `sentient/store/migrations/`, one file per step, that bring a database made by an older build
  up to date. The highest number is the schema version (`SCHEMA_VERSION`, stored in `PRAGMA user_version`).

When Sentient opens an existing database with a lower version, it takes a snapshot (issue #151), runs every step
above that version in order (stamping the version after each), and then runs the `schema.sql` files, which add any
missing tables and indexes.

**A new table or index.** Add it to your package's `schema.sql`. That is all: the file runs on every start, so older
databases get it too. An index on a new column also goes there: the step below adds the column first.

**A new column** (or a change to existing rows):

1. Add the column to the `CREATE TABLE` in the `schema.sql`, so new databases have it.
2. Add the next step: copy the newest file in `sentient/store/migrations/` to the next number with a short name, for
   example `0014_task_colour.py`:

   ```python
   """Tasks remember their colour (#123)."""

   from __future__ import annotations

   from typing import Any


   async def migrate(store: Any) -> None:
       await store.ensure_column("tasks", "colour", "TEXT")
   ```

   Use the same type and default as in `schema.sql`. A step may also fix rows (`await store.db.execute("UPDATE ...")`).
   A `.sql` file works too when plain SQL is enough, but only for statements that are safe to run twice
   (`ALTER TABLE ... ADD COLUMN` is not, so columns go in a `.py` step).
3. Run `pytest -q tests/test_migrations.py`. It fails when two steps share a number, a number is skipped, or a
   database migrated from an old build differs from a new one (tables, columns, indexes and triggers).

Never type the version anywhere, and never edit a step that is already on `main`: databases that ran it will not
run it again. A step must work twice (a crash before its version is stamped runs it again) and on a table that is
missing (a very old database), which `store.ensure_column` already handles.

**When two pull requests both add a step.** Both pick the next free number, say `0014`, and git sees no conflict
because the file names differ. Whoever merges second rebases on `main`, sees `tests/test_migrations.py` fail with
"two migrations share a number", and renames their file to `0015_...`.

## Packaging

The shipped app carries its own engine, so the person installing it needs neither
Python nor a virtualenv. PyInstaller freezes `python -m sentient serve` into
`sentient-engine.exe`, electron-builder ships that folder as `resources/engine`,
and `electron/main/paths.ts` picks it in a packaged build while `npm run dev`
keeps using `.venv/Scripts/python.exe -m sentient serve`. `~/.sentient` is the
data folder in both modes.

```bash
uv pip install --python .venv/Scripts/python.exe pyinstaller   # once
cd desktop
npm run package            # frozen engine + installer for this OS  (~5 min cold)
npm run package:engine     # just the engine -> desktop/build/engine
npm run package:win        # just the installer (reuses desktop/build/engine)
```

Take `desktop/.build.lock` first: packaging rebuilds the shared `desktop/out`.
The Windows result is `desktop/dist/Sentient-Setup-<version>.exe` (~216 MiB), a
per-user NSIS installer that needs no administrator rights and installs into
`%LOCALAPPDATA%\Programs\Sentient`. `/S` installs and uninstalls silently.
Uninstalling leaves `~/.sentient` alone: that is the user's memories and tasks.

| File | What it is |
|---|---|
| `packaging/sentient-engine.spec` | PyInstaller spec: hidden imports, data files, exclusions |
| `packaging/engine_entry.py` | Entry point; defaults to `serve` when run with no command |
| `packaging/build_engine.py` | Freezes, smoke-tests and stages the engine |
| `packaging/runtime_hooks/excluded_extras.py` | Friendly message when a left-out extra is imported |
| `desktop/electron-builder.yml` | NSIS / DMG / AppImage targets and `extraResources` |

A frozen build cannot see packages the user installs later, so the heavy optional
extras (faster-whisper, CTranslate2, onnxruntime, Kokoro, openWakeWord, OpenCV,
PyTorch) are deliberately left out and the runtime hook turns an attempt to import
them into a sentence that points at Settings → Voice. Voice models still download
on demand for source installs, and the browser drives the user's own Edge or
Chrome, so no browser binaries are bundled either.

To test the packaged shell without installing, build the engine and run the app
with `SENTIENT_ENGINE=/path/to/sentient-engine.exe`; `SENTIENT_PYTHON` still
forces the interpreter path.

Only the Windows installer is built and verified on the founder's PC. The macOS
DMG and Linux AppImage blocks in `electron-builder.yml` are written but untested,
and each needs `npm run package:engine` run on that OS first (PyInstaller does not
cross-compile). macOS builds are unsigned and unnotarized so far.

## Checks, branches and releases

- Every pull request runs [CI](../.github/workflows/ci.yaml): ruff, the engine tests on Linux and Windows, the
  desktop typecheck and build, and a secret scan. It needs no secrets, so it runs the same on forks.
- `main` is the only long-lived branch. Contributors fork, branch from `main` and open pull requests back to it;
  maintainers squash-merge after one approval and green checks. See [CONTRIBUTING.md](../CONTRIBUTING.md).
- Releases are tags on `main`. Pushing a tag such as `v3.0.0-alpha.1` runs the
  [release workflow](../.github/workflows/release.yaml), which builds the Windows, macOS and Linux installers and
  attaches them to a draft GitHub release for a maintainer to review and publish.
- Sentient v2 lives on the `v2` branch and the `v2-final` tag. It is not developed any more.

## Where to change things

| You want to… | Go to |
|---|---|
| Change what the window and engine exchange | `docs/API.md` first, then both sides |
| Add a table or a column | your package's `schema.sql`, plus a numbered step for a column ([Changing the database](#changing-the-database)) |
| Add a config option | your package's section in `sentient/config/schema.py` (with a `description`; Settings renders it) |
| Add a tool | a plugin in `sentient/integrations/plugins/` (integrations) or `sentient/tools/builtin/` (core); set `risk`, and `internal=True` if it only touches Sentient's own data |
| Change how turns run | `sentient/agent/loop.py` (`run_loop`, `run_turn`) and `agent/toolselect.py` |
| Change prompts | `sentient/agent/prompt.py`, `sentient/tasks/prompts.py`, `sentient/memory/prompts.py`, `sentient/proactivity/prompts.py`, `sentient/evolution/prompts.py` |
| Add a device capability | `docs/NODES.md`, `sentient/nodes/tools.py`, and the device (web app in `sentient/nodes/web/`, `sentient/nodes/reference.py`, or your firmware) |
| Add a messaging channel | a `Channel` subclass in `sentient/channels/` (see `telegram.py`) |
| Add a screen | `desktop/src/features/<area>/`, reusing `components/ui`, `lib/api.ts`, `hooks/` |

Ownership rules for parallel work are in `CLAUDE.md`.

## Local model notes
- Prefer `qwen3:8b` (or larger) for the primary role on local hardware. An older
  `qwen3:4b` download could not tool-call through Ollama 0.9; Sentient then answers
  without tools and says so. Settings → Models → Test checks tool support.
- The `fast` and `voice` roles run with reasoning off by default; that is several
  times faster for background jobs and spoken replies.
- Do not time anything while other engines or smoke runs share the GPU: Ollama can
  fall back to CPU when RAM runs out.
- Photo and screen questions need a vision-capable model in the vision role. With Ollama use the
  `ollama_chat/` prefix (`ollama_chat/qwen2.5vl:3b` works well and is about 3 GB); the `ollama/` prefix also
  works now that Pillow is a dependency. Without a vision model Sentient saves the image and says so.
- The Docker code-execution backend needs Docker Desktop running. If it fails to start with an inference
  socket error, turn off Docker Model Runner in its settings or reboot; Sentient falls back to the process
  backend, which is the default.
- The wake word uses the base local Whisper model on the CPU by default (tiny often hears "Hey" as "He"),
  downloaded on first use.
