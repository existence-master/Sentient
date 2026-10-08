# Contributing to Sentient

Thanks for helping. Sentient is a personal assistant that runs on your own computer, and it is built in the
open. This guide gets you from zero to a merged pull request.

## You never need our keys

Sentient v3 has no `.env` file and no shared API keys. Please do not ask for them: there is nothing to share.

- **To run the app**, use a free local model through [Ollama](https://ollama.com/download), or paste your own
  provider key (Anthropic, OpenAI, Gemini and others) in **Settings → Models**. Keys stay in your own system
  keychain.
- **To run the tests**, you need nothing at all. Every test uses a scripted fake model and no network.
- **Integrations** (Gmail, Slack, Notion, Telegram...) connect with your own accounts from inside the app, and
  each one has step-by-step setup instructions. Their tests use mocked APIs.

If something only works with a key you don't have, that is a bug in our tests or docs: open an issue.

## Get it running

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 22, and optionally Ollama.

```bash
git clone https://github.com/<you>/Sentient.git && cd Sentient
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev,voice]"   # macOS/Linux: .venv/bin/python
cd desktop && npm install && npm run dev
```

The window starts the engine for you and walks you through setup. More detail, including how to run the
engine on its own and how to capture screenshots, is in [docs/DEVELOPING.md](docs/DEVELOPING.md).

## How we work

We use **GitHub flow**:

1. **Find or open an issue.** Look for [`good first issue`](../../labels/good%20first%20issue) and
   [`help wanted`](../../labels/help%20wanted). For anything bigger than a small fix, comment on the issue first
   so we can agree on the approach and avoid duplicate work. Small fixes and docs need no issue.
2. **Fork, then branch from `main`.** Name the branch after what it does, for example
   `fix/telegram-long-messages` or `feat/outlook-integration`.
3. **Keep the pull request small and focused.** One change per PR is much faster to review.
4. **Open the PR against `main`.** The checks below run automatically, also on forks.
5. **A maintainer reviews it.** One approval and green checks are enough. We squash-merge, so the PR title
   becomes the commit message.

There are no long-lived `development` or `staging` branches. `main` is always the latest working version, and
releases are tags on `main`.

### PR titles

Use [Conventional Commits](https://www.conventionalcommits.org/) with an area, because the title becomes the
commit message:

```
feat(channels): send Telegram replies as voice notes
fix(browser): keep refs stable after a scroll
docs(nodes): document the battery event
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`. Areas match the folders below.

### What the checks run

Run the same things locally before you push:

```bash
.venv/Scripts/ruff.exe check sentient tests            # lint (macOS/Linux: .venv/bin/ruff)
.venv/Scripts/python.exe -m pytest -q                  # engine tests, or one folder: pytest tests/tasks
cd desktop && npm run typecheck && npm run build       # desktop
```

The full engine suite takes a few minutes. While you work, run just the folder you touched.

### A good pull request

- Says what changed and why, and links the issue (`Closes #123`).
- Adds or updates tests. Bug fixes come with a test that failed before the fix.
- Updates [docs/API.md](docs/API.md) in the same PR when it changes anything the desktop app and the engine
  exchange. The contract comes first.
- Includes a screenshot or short clip for visible UI changes.
- Keeps the product rules: no login or accounts, keys only in the keychain, every tool declares a risk, and
  sending, deleting, buying or running code asks the user first.

## Where things live

| Area | Folder | Notes |
|---|---|---|
| Engine core | `sentient/agent`, `sentient/llm`, `sentient/store`, `sentient/tools`, `sentient/gateway` | Agent loop, model roles, approvals, SQLite |
| Tasks | `sentient/tasks` | Long-running, scheduled, triggered and watcher tasks |
| Memory | `sentient/memory` | Facts, user model, nightly consolidation |
| Proactivity and skills | `sentient/proactivity`, `sentient/evolution`, `sentient/skills` | Suggestions, self-improving skills |
| Integrations | `sentient/integrations` | One plugin per app; start from an existing one |
| Channels | `sentient/channels` | Telegram, Discord |
| Devices | `sentient/nodes`, `firmware/`, `docs/NODES.md` | Phones, glasses, the device protocol |
| Browser and code | `sentient/browser`, `sentient/sandbox` | Browser control, sandboxed scripts |
| Voice | `sentient/voice` | Speech in and out, wake word |
| Desktop app | `desktop/` | Electron + React + TypeScript |
| Packaging | `packaging/`, `desktop/electron-builder.yml` | Installers |

Each area's tests live in the matching `tests/<area>` folder. [CLAUDE.md](CLAUDE.md) has the deeper
conventions; it is written for AI coding assistants but is a good read for people too.

### Adding an integration

Copy a small plugin such as `sentient/integrations/plugins/trello.py`: declare the setup fields, write each
tool as a typed async function with a docstring and a `Risk`, and add tests with mocked HTTP (`respx`). The
desktop app renders the connect screen from your plugin, so most integrations need no UI code.

## Using AI coding assistants

Welcome. Point your assistant at [CLAUDE.md](CLAUDE.md) (also linked from `AGENTS.md`). You are still the
author: read the diff, run the checks, and make sure the PR description is yours and accurate.

## Contributor License Agreement

The first time you open a PR, a bot asks you to agree to our [CLA](CLA.md) by posting one comment. It lets us
keep offering Sentient under the AGPL and fund its development. You only do this once.

## Questions, bugs and security

- **Questions and setup help:** [GitHub Discussions](../../discussions/categories/q-a).
- **Bugs and feature requests:** [open an issue](../../issues/new/choose).
- **Security problems:** please report privately, see [SECURITY.md](SECURITY.md).

## The previous version

Sentient v2 was a cloud web app with its own servers. It now lives on the [`v2` branch](../../tree/v2) and the
`v2-final` tag, and it is no longer developed. New work goes into v3 on `main`.

Everyone taking part follows our [Code of Conduct](CODE_OF_CONDUCT.md).
