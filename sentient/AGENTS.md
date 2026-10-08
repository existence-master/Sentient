# Engine guide (`sentient/`)

Read the root [AGENTS.md](../AGENTS.md) first.

## Shape

- `app.py` is the composition root. `SentientApp` builds every subsystem once and starts services in dependency
  order: store, memory, notifications, integrations, built-in tools, skills, agent, then the feature services.
- Feature services subclass `services.Service`, use `run_every(...)` for periodic jobs and reach each other only
  through public methods on the app (`app.tasks`, `app.memory`, `app.integrations`, `app.notify(...)`, `app.bus`).
- `events.EventBus` carries domain events (`task.updated`, `notification.new`...). The gateway forwards every event
  to the window over `/ws`. Event names and payloads are part of [docs/API.md](../docs/API.md).
- `agent/loop.py` holds `Agent.run_loop`, the one tool-calling loop every surface uses, and `run_turn` for chat.
  See [ADR 0007](../docs/adr/0007-one-agent-loop.md).

## Adding things

**A tool.** Write an async function with type hints and a docstring (the model reads both), wrap it with
`@tool(risk=...)`, and add it to a `ToolPlugin`. Return small JSON-friendly dicts; return `{"error": "..."}` with a
plain sentence instead of raising for expected failures. Call `ctx.progress({...})` to stream progress.

**An integration.** Copy a small plugin from `integrations/plugins/` (Trello is a good model). Declare setup fields
and plain-language instructions; the desktop renders the connect screen from them. Mock HTTP with `respx`.

**A config option.** Add it to your area's section in `config/schema.py` with a default and a `description`.

**A table or column.** Put idempotent DDL in your package's `schema.sql`; add columns with `store.ensure_column`.

**A REST route.** Add it to your package's router in `gateway/routes/`, document it in `docs/API.md`.

## Testing

- `asyncio_mode=auto`: write `async def test_...` directly.
- Build an app with `SentientApp(config, llm=FakeProvider(...), enable_background=False)`; the `config` fixture in
  `tests/conftest.py` gives a clean configuration and every test gets its own `SENTIENT_HOME`.
- `FakeProvider(replies=[...], json_replies=[...])` scripts the model: a string is a text reply, a list is tool calls.
- Area fixtures go in `tests/<area>/conftest.py`. Never edit `tests/conftest.py` for one area's needs.
- Tests must be quick and offline. Browser tests use local pages and skip when no browser can start; Docker tests
  skip when Docker cannot run Linux containers.

## Small-model robustness

Real runs on 8 GB laptops with `qwen3:8b` shaped this code. Keep prompts short and structured, parse JSON with the
tolerant helpers (`parse_json_loose`, `tasks/jsonio.py`), guard model decisions with deterministic checks, and
prefer one more nudge over a failed run.
