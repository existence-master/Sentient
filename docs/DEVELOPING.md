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
| `npm run package:win` | NSIS installer (Python is not bundled yet) |
| `node scripts/smoke.mjs <route> <out.png> [WxH]` | Launch the built app, capture a route to PNG, quit |

In Git Bash write smoke routes without a leading slash (`settings/models`), or MSYS
turns them into Windows paths. Set `SENTIENT_HOME` to a seeded profile to capture
realistic screens. Demo data seeders live in `desktop/scripts/seed-*.py` (`seed-leap-a.py` for chat tool
cards, devices and helpers; `seed-leap-b.py` for About you, dreams, watcher jobs and webhooks).

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

No test touches the network; the LLM is a scripted, deterministic fake
(`tests/conftest.py`). Browser tests drive a hidden local Edge or Chrome against local pages and skip
when none can start; Docker sandbox tests skip when Docker is not running. The suite is large: on a busy
machine run it per folder (`tests/test_*.py`, `tests/tasks`, `tests/memory` ...).

The web device app is served at `http://127.0.0.1:<port>/node/` so it can be tried on the same computer
(camera and microphone work on localhost). For a phone, turn on Devices → Allow devices on my Wi-Fi.

## Where to change things

| You want to… | Go to |
|---|---|
| Change what the window and engine exchange | `docs/API.md` first, then both sides |
| Add a config option | your package's section in `sentient/config/schema.py` (with a `description`; Settings renders it) |
| Add a tool | a plugin in `sentient/integrations/plugins/` (integrations) or `sentient/tools/builtin/` (core); set `risk`, and `internal=True` if it only touches Sentient's own data |
| Change how turns run | `sentient/agent/loop.py` (`run_loop`, `run_turn`) and `agent/toolselect.py` |
| Change prompts | `sentient/agent/prompt.py`, `sentient/tasks/prompts.py`, `sentient/memory/prompts.py`, `sentient/proactivity/prompts.py`, `sentient/evolution/prompts.py` |
| Add a device capability | `docs/NODES.md`, `sentient/nodes/tools.py`, and the device (web app in `sentient/nodes/web/`, `sentient/nodes/reference.py`, or your firmware) |
| Add a messaging channel | a `Channel` subclass in `sentient/channels/` (see `telegram.py`) |
| Add a screen | `desktop/src/pages/` or `desktop/src/features/`, reusing `components/ui`, `lib/api.ts`, `hooks/` |

Ownership rules for parallel work are in `CLAUDE.md`.

## Local model notes
- Prefer `qwen3:8b` (or larger) for the primary role on local hardware. An older
  `qwen3:4b` download could not tool-call through Ollama 0.9; Sentient then answers
  without tools and says so. Settings → Models → Test checks tool support.
- The `fast` and `voice` roles run with reasoning off by default; that is several
  times faster for background jobs and spoken replies.
- Do not time anything while other engines or smoke runs share the GPU: Ollama can
  fall back to CPU when RAM runs out.
- Photo and screen questions need a vision-capable model in the vision role (for example a
  `qwen2.5vl` or `llava` model in Ollama, or a cloud model); without one Sentient saves the image and says so.
- The wake word uses the base local Whisper model on the CPU by default (tiny often hears "Hey" as "He"),
  downloaded on first use.
