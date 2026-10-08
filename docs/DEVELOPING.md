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
| `npm run package:engine` | Freeze the Python engine into `desktop/build/engine` |
| `npm run package` | Frozen engine + installer for this OS (see Packaging) |
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
- Photo and screen questions need a vision-capable model in the vision role. With Ollama use the
  `ollama_chat/` prefix (`ollama_chat/qwen2.5vl:3b` works well and is about 3 GB); the `ollama/` prefix also
  works now that Pillow is a dependency. Without a vision model Sentient saves the image and says so.
- The Docker code-execution backend needs Docker Desktop running. If it fails to start with an inference
  socket error, turn off Docker Model Runner in its settings or reboot; Sentient falls back to the process
  backend, which is the default.
- The wake word uses the base local Whisper model on the CPU by default (tiny often hears "Hey" as "He"),
  downloaded on first use.
