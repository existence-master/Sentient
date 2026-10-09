# Sentient desktop ↔ backend contract

This is the contract between the Electron renderer and the Python backend
(`sentient serve`). Backend packages implement it; the UI is built against it.
If you need to change a shape, change this file in the same edit.

- Base URL: `http://127.0.0.1:<port>` (port chosen by the desktop shell).
- Every `/api/*` request carries `Authorization: Bearer <token>`. `GET /api/health` is open.
- WebSockets pass the token as `?token=<token>`.
- Timestamps are ISO-8601 UTC strings. IDs are strings unless stated (fact ids are integers).
- Errors: HTTP status + `{"detail": "human readable"}`.

---

## 1. Live channel `WS /ws`

### Client → server
| type | fields | meaning |
|---|---|---|
| `chat.send` | `session_id?`, `text`, `attachments?: string[]` (file names from `/api/files`), `model?` (override primary for this turn), `client_id?` | start a turn; omit `session_id` for a new chat |
| `chat.steer` | `session_id`, `text`, `client_id?` | add to the reply in progress (section 10); without a running reply same as `chat.send` |
| `chat.cancel` | `session_id` | stop the reply in progress |
| `approval.respond` | `approval_id`, `decision: "allow"\|"allow_session"\|"deny"` | answer an approval request |
| `ping` | | keepalive |

### Server → client: chat/agent events (no dot in `type`)
All carry `session_id` and `turn_id`.

| type | fields |
|---|---|
| `hello` | `version`, `assistant` |
| `session` | `session_id`, `client_id` (echo) |
| `thinking_delta` | `text` |
| `text_delta` | `text` |
| `tool_call` | `call_id`, `name`, `arguments` |
| `tool_progress` | `call_id`, `name`, `kind: stdout\|stderr\|status\|frame\|subagent`, `text?`, `image?`, `data?` (section 10) |
| `tool_result` | `call_id`, `name`, `result`, `is_error`, `duration_ms` (a call the user declined has `result: {error, declined: true}`) |
| `approval_request` | `approval_id`, `call_id`, `name`, `arguments`, `risk: read\|write\|send\|exec` (effective risk), `reason`, `risk_label?`, `target?` (section 10) |
| `user_interjection` | `text` (a steer message the model just received, section 10) |
| `steer_ack` | `session_id`, `queued`, `client_id` (echo; no `turn_id`) |
| `usage` | `model`, `prompt_tokens`, `completion_tokens` |
| `error` | `message`, `recoverable` |
| `done` | `content` (final text), `message_id`, `cancelled?`, `memory_sources: [MemorySource]` (what this reply had in mind, section 2; `[]` when none), `dropped?: string[]` (section 17: messages queued behind a stopped reply, never sent) |
| `approval.ack` | `approval_id`, `resolved` |

### Server → client: domain events (dotted `type`, payload in `data`)
Envelope: `{"type": "task.updated", "data": {...}, "ts": "..."}`

| type | data |
|---|---|
| `task.updated` | full **Task** (§4) |
| `task.deleted` | `{task_id}` |
| `task.run_progress` | `{task_id, run_id, update: ProgressUpdate}` (also moves the run's `last_activity_at` to `update.timestamp`) |
| `task.run_activity` | `{task_id, run_id, last_activity_at}`: a working run is alive (the model is writing), at most every 10 s |
| `notification.new` | **Notification** (§6) |
| `notification.updated` | full **Notification** after its payload changed (suggestion approved/dismissed, approval answered, task plan approved/declined: a `task` notification with `payload.event = "approval_needed"` gains `payload.status = "approved"\|"declined"`; a task question (`payload.event = "question"`) gains `payload.status = "answered"` with `payload.answer`, or `"cancelled"`) |
| `notification.read` / `notification.deleted` | `{id}` (`null` = all) |
| `integration.updated` | **Integration** (§5) |
| `memory.updated` | `{action: "ADD"\|"UPDATE"\|"DELETE", id, content?}`; bulk changes (import, delete by source, expiry purge) send `id: null` plus `source?`/`reason?` and `count` |
| `skill.updated` | `{name, state: "active"\|"pending_review"\|"stale"\|"archived"\|"rejected"\|"deleted"}` |
| `session.updated` | `{session_id, title}` |
| `config.updated` | `{sections: string[]}` |
| `voice.state` | `{state, session_id}` (mirrors voice socket for other windows) |
| `stop.updated` | **StopState** (section 17): Stop everything was turned on or off |

---

## 2. Core

### `GET /api/health` → `{ok, version, name}` (no token)

### `GET /api/bootstrap`
Everything the renderer needs on launch.
```json
{
  "version": "3.0.0a0",
  "home": "C:\\Users\\me\\.sentient",
  "assistant": {"name": "Sentient", "user_name": "Sarthak", "timezone": "auto", "location": "", "language": "en", "onboarding_complete": true},
  "models": {"primary": "ollama_chat/qwen3:8b", "fast": "...", "planner": null, "executor": null, "embedding": "...", "vision": null, "voice": null},
  "memory_enabled": true,
  "unread_notifications": 3,
  "ui": {"theme": "dark", "accent": "sentient", "launch_at_login": false, "minimize_to_tray": true},
  "features": {"voice": true, "proactivity": true},
  "stop": {"stopped": false, "stopped_at": null, "source": null}
}
```
`stop` is the **StopState** of section 17.

### Onboarding
`POST /api/onboarding`
```json
{"user_name": "Sarthak", "assistant_name": "Sentient", "timezone": "Asia/Kolkata", "location": "Pune, India",
 "professional_context": "...", "personal_context": "...", "persona": "friendly|professional|concise|custom"}
```
Saves config, writes USER.md, seeds memory facts (source `onboarding`) in the background, sets
`assistant.onboarding_complete = true`. → `{ok: true}`

### Config
- `GET /api/config` → full config object (see `sentient/config/schema.py`).
- `GET /api/config/schema` → JSON schema (every field has `description`; Settings forms are generated from it).
- `PUT /api/config` body: full config → `{saved: true}`. Hot-applied.
- `PATCH /api/config` body: partial nested object, deep-merged → `{saved: true, config}`. Inside free-form maps (`models.fallbacks`, `models.reasoning`, `models.temperature`, `models.context_length_per_role`, `models.providers`, `integrations.mcp_servers`, `tools.approvals.rules`) a `null` value removes that entry.
- Validation failures on PUT/PATCH return 422 with `detail: [{loc: string[], msg, type}]`.

### Lasting approval rules (`tools.approvals.rules`, ADR 0016)
`{"<key>": "allow" | "ask" | "never"}`, default `{}`. A key is a tool name (`gmail_send_email`, `execute_code`,
`mcp_<server>_<tool>`) or a plugin id from `GET /api/tools` (`gmail`, `slack`, `mcp_<server>`) meaning every tool of
that plugin; a tool's own rule beats its plugin's rule. Keys are trimmed and values lower-cased on save; anything else
is a 422. Remove a rule with `PATCH /api/config {"tools": {"approvals": {"rules": {"<key>": null}}}}`. Rules are
applied in code before every call and take effect at once:
- `never`: the tool is not offered to the model (chat, voice, channels, tasks, subagents, proactivity, scripts) and is
  left out of planners' tool lists. A call made anyway does not run; its `tool_result` is
  `{error: "You've set Sentient to never use <name>. Change this in Settings > Approvals & safety."}` (`<name>` is the
  app's display name for a plugin rule, e.g. `Slack`, else the tool's plain name and app, e.g. `"Post message" in Slack`).
- `ask`: an `approval_request` every time, even with approvals mode `off`, after "Allow for this chat", and for `read`
  tools. Where nobody can be asked the call does not run: a task run or swarm worker stops there and fails with
  `"<name> is set to Ask, and tasks can't ask yet. Change it in Settings > Approvals & safety."` (the run's `error`,
  and the usual `run_failed` notification; `LoopResult.stopped_by_rule` and `LoopResult.error` carry the same text).
  Subagents and scripts refuse the call, and proactive look-ups leave the tool out.
- `allow`: runs without asking in modes `ask` and `always`, except a purchase (effective risk `send` or higher whose
  approval wording `risk_label` is "Purchase", such as a browser click on "Place order"), which asks every time.
  "Allow for this chat" never covers a purchase. `allow` does not widen what scripts may call
  (section 11): they still only read.
- Purchases ask in every approvals mode, `off` included (only `browser.confirm_purchases: false` turns that off),
  and scripts refuse them in every mode. Rules are read again right before a tool runs and on every script tool call,
  so a rule changed while a call waits for approval, or while a script runs, applies to it.
Engine helpers: `app.approvals.rule(tool)`, `app.approvals.is_never(tool)`,
`await app.approvals.decide(tool, session_id, risk, arguments, ctx) -> bool`, pure helpers in `sentient.tools.rules`.

### Sessions (chats)
- `GET /api/sessions?limit=100` → `[{id, title, channel, created_at, updated_at}]` newest first
- `POST /api/sessions` → `{session_id}`
- `PATCH /api/sessions/{id}` `{title}` → `{ok}`
- `DELETE /api/sessions/{id}` → `{ok}`
- `GET /api/sessions/{id}/messages?limit=500` → raw transcript rows, oldest first:
  ```json
  {"id": "...", "role": "user|assistant|tool", "content": "...", "thinking": "...|null",
   "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "memory_recall", "arguments": "{\"query\":\"sister\"}"}}],
   "tool_call_id": "call_1", "name": "memory_recall", "attachments": ["report.pdf"], "interjection": false,
   "memory_sources": [], "created_at": "..."}
  ```
  The UI folds each `assistant` message with `tool_calls` plus its following `tool` messages into one assistant turn.
- **Memory sources.** The final assistant message of a chat turn (also a stopped one) carries `memory_sources`, the
  memories that reply had in mind, also sent on its `done` event. Every other row has `[]`. A **MemorySource** is
  `{kind: "fact"|"insight", id, text, source, via: "prompt"|"tool"}`: `id` is the Memory id (int, section 7) or the
  Insight id (string, section 15); `text` is the memory as it was during that turn; `source` is the fact's source
  (`conversation`, `manual`, `file:<name>`...) or the insight's (`user` | `inferred`); `via: "prompt"` means it was in
  the system prompt (recalled facts, user-model insights), `"tool"` that `memory_recall` or `memory_search_by_source`
  returned it during the turn. Recorded deterministically (the model is never asked which it used); deduplicated,
  first mention wins, at most 40.
- `GET /api/sessions/search?q=` → `[{session_id, message_id, role, snippet, created_at}]`
- `POST /api/chat` NDJSON fallback of the WebSocket turn: body `{text, session_id?, attachments?, model?}`; lines are the chat events above, first line `{type: "session", session_id}`.
- `POST /api/approvals` `{approval_id, decision}` → `{resolved}`

### Files (attachments + assistant outputs)
- `POST /api/files` multipart `file` → `{name, size, mime}` (stored under `~/.sentient/files/uploads/`)
- `GET /api/files` → `[{name, size, mime, modified_at}]`
- `GET /api/files/content/{name}` → the file bytes
- `DELETE /api/files/{name}` → `{ok}`

### Tools
- `GET /api/tools` → `[{id, display_name, description, category, icon, auth, selection_hint, tools: [{name, description, risk}]}]`
  (connected apps and built-in tools; tools behind a `never` rule are still listed so Settings can show and change the rule)

### Usage (insights)
- `GET /api/usage?days=30` → `{totals: {prompt_tokens, completion_tokens}, by_model: [{model, prompt_tokens, completion_tokens, calls}], by_source: [...], by_day: [{day, prompt_tokens, completion_tokens}]}`

---

## 3. Models & secrets (fully changeable model configuration)

- `GET /api/models/providers` →
  ```json
  [{"id": "anthropic", "label": "Anthropic", "kind": "cloud|local", "key_required": true, "key_set": false,
    "api_base": null, "docs_url": "https://console.anthropic.com/", "suggested": ["anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5"]}]
  ```
- `GET /api/models/local` → `{ollama: {reachable, models: [{name, size, family, parameter_size, is_embedding, capabilities: string[]}]}, lm_studio: {reachable, models: [...]}}` (`capabilities` from Ollama, e.g. completion/tools/thinking/vision/embedding — a hint; `POST /api/models/test` is the authoritative tool-support check)
- `POST /api/models/test` `{model, role?}` → `{ok, latency_ms, reply?, error?, supports_tools?}`
- `POST /api/models/test-embedding` `{model}` → `{ok, dim?, error?}`
- `POST /api/models/checkup` `{roles?: {role: model | null}}` → streams NDJSON while it checks each role's model,
  one role at a time (local models are never loaded side by side). Without `roles` it checks every role in the saved
  config; with `roles` it checks only those, with those models (onboarding checks its picks before saving). It is
  informational only and never changes config. Each step has a short timeout (60 s). Roles with the same model
  and the same settings the tests depend on (provider address, reasoning effort, context length, temperature) run
  each model test (`reply`, `tools`, `chain`, `json`) once; the later role reuses it with a detail starting
  "Same as primary." (the first role's name). Lines:
  - `{type: "start", roles: [{role, model}]}` (`model` null = an optional role that uses the main model)
  - `{type: "step", role, label}` progress, e.g. "Trying a tool call"
  - `{type: "role", role, model, provider, local, inherits, status, checks: [Check]}` when a role is finished.
    `inherits: "primary"` with `status: "skip"` and no checks for an optional role with no model of its own.
  - `{type: "done", status, roles: [role results]}` (`status` = the worst role)

  `Check` = `{id, label, status: "pass"|"warn"|"fail"|"skip", detail, fix?, action?}`. `detail` and `fix` are
  plain sentences for the UI. Checks, in order, skipping those that do not apply:
  `connection` (Ollama running and model downloaded, or a cloud key set; a failure stops the rest), `reply` (one
  short answer), `tools` (one scripted `find_city` call; roles that use tools: primary, fast, executor, vision,
  voice), `chain` (a second `get_weather` call using the first result; primary and executor), `json` (a JSON reply;
  fast and planner), `thinking` (Ollama models that can think: thinking matches the role's reasoning setting),
  `context` (tokens in use vs the model's maximum from `/api/show`), `gpu` (from Ollama `/api/ps`: `size_vram` vs
  `size`, warns when part of the model runs on the processor), `embedding` (embedding role only). Cloud and
  LM Studio models get no Ollama checks. `action` is an optional one-click fix the window may offer:
  `{kind: "use_model", role, model, label}` (`PUT /api/models/roles`), `{kind: "pull_model", name, label}`
  (`POST /api/models/ollama/pull`), `{kind: "set_reasoning", role, value, label}` (`models.reasoning`),
  `{kind: "set_context_length", value, role: string|null, label}` (`models.context_length`, or
  `models.context_length_per_role[role]` when `role` is set). Unknown role → 400. `sentient doctor --models` prints
  the same check-up as a table.
- `PUT /api/models/roles` `{primary?, fast?, planner?, executor?, embedding?, vision?, voice?}` → updated roles (null = use primary). The `voice` role is used for `channel` voice/glasses turns and defaults to reasoning `none`.
- `PUT /api/models/fallbacks` `{role: [model, ...]}` → `{ok}`
- `POST /api/models/ollama/pull` `{name}` → streams NDJSON `{status, completed?, total?}`
- `GET /api/secrets` → `[{name, set: bool, source: "keychain"|"env"|null, kind: "provider"|"integration"}]` for every provider + integration secret name
- `PUT /api/secrets/{name}` `{value}` → `{ok}` (stored in OS keychain; never echoed back)
- `DELETE /api/secrets/{name}` → `{ok}`

---

## 4. Tasks (long-running work) — v2 semantics preserved

### Task object (same field names as v2 so the task UI ports directly)
```json
{
  "task_id": "hex", "name": "Weekly inbox digest", "description": "...",
  "status": "planning|clarification_pending|approval_pending|pending|active|processing|waiting_for_user|completed|completed_with_errors|error|declined|cancelled|archived",
  "priority": 0, "assignee": "ai", "task_type": "single|swarm|script",
  "schedule": {"type": "once", "run_at": "2026-09-16T09:00|null", "timezone": "Asia/Kolkata", "catch_up?": "run|skip"}
            | {"type": "recurring", "frequency": "daily|weekly", "days": ["Monday"], "time": "09:00", "timezone": "...", "catch_up?": "run|skip"}
            | {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "..."}
            | {"type": "triggered", "source": "gmail|gcalendar|webhook|...", "event": "new_email|new_event|<hook id>|...", "filter": {}, "timezone": "..."},
  "script": {"code": "...", "condition": "alert|changed", "then": "notify|run", "last_result": null, "last_run_at": "...|null", "last_error": "...|null"} | null,
  "plan": [{"tool": "gmail", "description": "Fetch unread emails from the last 7 days"}],
  "runs": [Run], "chat_history": [{"role": "user|assistant", "content": "...", "timestamp": "..."}],
  "clarifying_questions": [{"question_id": "q1", "text": "...", "answer": null}],
  "swarm_details": {"goal": "...", "items": [], "total_agents": 0, "completed_agents": 0,
                    "progress_updates": [{"worker_id": "agent-1|aggregator", "timestamp": "...", "status": "processing|completed|error|aggregating", "message": "..."}],
                    "aggregated_results": []} | null,
  "enabled": true, "model": null, "original_context": {"source": "manual_creation|chat|proactive|trigger", "...": "..."},
  "error": "...|null",
  "next_execution_at": "...|null", "last_execution_at": "...|null", "created_at": "...", "updated_at": "..."
}
```
`swarm_details` is `null` for single tasks. `script` is `null` unless `task_type` is `script` (section 16).
`error` holds the last planning/run failure message (v2 `task.error`).
Each run embeds its most recent 200 progress updates; the full log is at the `/events` endpoint.
`interval` schedules (every `interval_minutes`, minimum 5; `frequency: "hourly"` is normalized to 60) are additive; the
task-creation prompt still only produces daily/weekly (v2), the planner uses `interval` for script jobs.

**Run**
```json
{"run_id": "hex", "status": "processing|waiting_for_user|completed|completed_with_errors|error|cancelled", "created_at": "...", "execution_start_time": "...", "finished_at": "...",
 "plan": [...], "trigger_event_data": {}|null, "progress_updates": [ProgressUpdate],
 "result": {"summary": "markdown", "links_created": [{"url": "", "description": ""}], "links_found": [...], "files_created": [{"filename": "", "description": ""}], "tools_used": ["gmail"]},
 "error": null, "retry_of": "run id this run retries|null",
 "pending_question": {"question": "Which flight should I book?", "options": ["IndiGo 07:10", "Air India 09:40"], "asked_at": "...",
                      "kind": "question|limit|stuck", "reason": "...|null"} | null,
 "last_activity_at": "...|null"}
```
`pending_question` is set only while the run is `waiting_for_user` (see "Tasks that ask you a question", "Limits on a run"
and "Stuck runs" below). `kind` says why it waits; `reason` is set for `stuck` only. `last_activity_at` is when the run
last showed any sign of work (a progress update, or streamed model output; falls back to `execution_start_time`).
Already approved one-call tasks (a follow-up's Send, section 6) carry `original_context.fixed_call = {tool, arguments,
done_text}` and a one-step `plan`. They are created `pending`, start a run at once with no planner and no executor model,
and the run makes exactly that call with exactly those arguments (`tool_call`, `tool_result`, `final_answer` updates). A
lasting Never rule on the tool or its app fails the run with the rule's message; a tool error fails it with that error. A
run interrupted by a restart after the call started is not repeated: it fails and asks the user to check. Backend API:
`await app.tasks.create_approved_call(name, tool, arguments, *, step, description=None, source, original_context, done_text)`
→ Task.
**ProgressUpdate** `{"timestamp": "...", "message": {"type": "info|thought|tool_call|tool_result|final_answer|error", "content": "...", "tool_name": "...", "parameters": {}, "result": "...", "is_error": false}}`

### Endpoints
- `GET /api/tasks` → `[Task]`
- `GET /api/tasks/{id}` → `Task`
- `POST /api/tasks` `{prompt, is_swarm?: bool, assignee?: "ai", model?}` → `Task` (status `planning`; refinement + planning continue in the background and arrive as `task.updated`)
- `POST /api/tasks/preview` `{prompt}` → `{name, description, priority, schedule}` (v2 generate-plan)
- `PATCH /api/tasks/{id}` any of `{name, description, priority, schedule, plan, enabled, status, model, script}` → `Task`
  (`script`: partial `{code?, condition?, then?}` merged into the current script and validated, 400 when the code does not
  compile; changing `code` or `condition` resets `last_result`/`last_run_at`/`last_error`; `script: null` turns a script job back
  into a `single` task; a `script` on a single task makes it a script job)
- `DELETE /api/tasks/{id}` → `{ok}`
- `POST /api/tasks/{id}/approve` → `Task`
- `POST /api/tasks/{id}/decline` → `Task`
- `POST /api/tasks/{id}/rerun` → `Task` (new copy, v2 behaviour) 
- `POST /api/tasks/{id}/run-now` → `Task` (recurring/triggered: start a run immediately)
- `POST /api/tasks/{id}/archive` → `Task`
- `POST /api/tasks/{id}/chat` `{message}` → `Task` (change request → replanning)
- `POST /api/tasks/{id}/clarifications` `{answers: [{question_id, answer_text}]}` → `Task`
- `POST /api/tasks/{id}/runs/{run_id}/cancel` → `Task` (a `processing` or `waiting_for_user` run; cancelling a waiting run
  withdraws its question)
- `POST /api/tasks/{id}/runs/{run_id}/answer` `{answer}` → `Task` (answers the question of a `waiting_for_user` run; the run
  goes back to `processing` and continues from where it stopped, with the answer as the result of its `ask_user` call.
  400 empty answer, 404 unknown task/run, 409 the run is not waiting, for example already answered or cancelled)
- `POST /api/tasks/{id}/runs/{run_id}/retry` → `Task` (failed or cancelled run of a non-swarm task: a new run with `retry_of`
  continues from the old run's transcript, telling the model what went wrong, so finished steps are not repeated; without a
  transcript it starts fresh with the same plan and trigger data. 409 for other statuses, swarm tasks, or a one-off task that is already running)
- `POST /api/tasks/{id}/script/test` optional body `{code}` → SandboxResult (section 11; runs the stored script, or `code`, once
  without touching `last_result`; 409 when the task has no script and no `code` is given, 400 when the code does not compile;
  when code execution is unavailable the result has `ok: false` and an `error`)
- `GET /api/tasks/{id}/runs/{run_id}/events` → `[ProgressUpdate]` (full log, for long runs)

Errors: `404` unknown task/run, `409` not allowed in the current state (approve without a plan,
chat while processing or waiting for an answer, cancel a finished run, clarifications on a task without questions,
answer a run that is not waiting),
`400` invalid input (empty prompt, unknown status), `503` model unavailable (preview), `502` model returned the wrong shape.

Behaviour notes:
- Swarm tasks (`is_swarm: true`) skip approval (v2): `planning → processing → completed|completed_with_errors|error`.
- `PATCH` with a new `schedule`, a status of `active`/`pending`, or `enabled: true` recomputes `next_execution_at`.
- `run-now` on a one-off task creates a new run immediately; on a swarm task it re-orchestrates.
- Triggered tasks accept events while a previous run is still `processing` (each event gets its own run).
- Triggered tasks fire from the `source.items` domain event for every origin (section 16). Each task handles an item id at
  most once, whichever path delivers it. For `source: "webhook"` the `filter` can name fields of the JSON body directly
  (`{"status": "failed"}`) as well as item fields (`name`, `body.status`).
- Task notification `payload.event` values: `approval_needed`, `clarification_needed`, `planning_failed`, `run_completed`,
  `run_failed` (with `run_id`), `disabled`, `script_alert` (with `result`), `script_failed`, `script_recovered`,
  `question` (with `run_id`, `question`, `options`; a stuck run adds `stuck: true` and `reason`; later
  `status: "answered"|"cancelled"` and `answer`), `caught_up` (with `reason`, `ran`, `skipped`; see "Missed runs").
- Every way a run ends sends a notification or leaves a visible status: a run that crashes inside Sentient ends
  `error` with `Something went wrong inside Sentient while running this task.` and the usual `run_failed` notification.
- Chat tools (plugin `tasks`): `create_task_from_prompt`, `search_tasks`, `get_task_status` (results include `script`),
  `update_task(task_id, name?, description?, enabled?, schedule?, script_code?, script_condition?, script_then?)` (write,
  internal; changed script code goes back to `approval_pending` when `tasks.require_plan_approval`), and
  `request_task_change(task_id, message)` (write, internal; same as `POST /chat`).
- On restart, runs left `processing` resume from their transcript checkpoint (`tasks.resume_interrupted_runs`), else they end with error `Interrupted by restart`.
  Runs `waiting_for_user` are left alone: they keep waiting, with their question, until answered or cancelled.
- Disconnecting an integration sets `enabled: false` on tasks whose plan or trigger uses it (instead of v2's delete) and sends a notification.

### Tasks that ask you a question
- Inside a task run (never in chat, subagents, swarm workers or the planner) the executor can call
  `ask_user(question: str, options: list[str] | None = None)` (plugin `task_questions`, risk `write`, internal). It is meant
  only for choices the run cannot make on its own; confirming risky actions stays with approvals. `options` are cleaned
  to at most 6 short, distinct choices. One question per round (a second call in the same round gets an error) and at
  most 5 per run.
- The loop stops after that round without another model call. The run's transcript and
  `{question, options, tool_call_id, asked_at}` are stored on the run in SQLite and the run becomes `waiting_for_user`.
  The task becomes `waiting_for_user` too, unless another run of it is still `processing` (it switches once that run
  ends). The progress log gets `Waiting for your answer: ...`; `task.updated` carries the run's `pending_question`.
- A notification is created: `kind: "task"`, title `"<task name> needs your answer"`, `message` = the question,
  `payload: {task_id, event: "question", run_id, question, options}`.
- Answering (`POST .../answer`, a button, or a reply to the question message in a paired chat) puts the answer in place of the `ask_user` tool
  result, logs `You answered: ...`, sets the notification's `payload.status = "answered"` (and marks it read), moves run and
  task back to `processing` and continues the run without the restart note.
- While a task waits: `approve`, `chat`, `run-now` and `retry` on a one-off task return 409; recurring tasks are not
  started by the scheduler until the answer arrives; triggered tasks keep accepting events (each in its own run) and
  return to `waiting_for_user` when those runs end.

### Limits on a run
Every task run has limits, checked deterministically (no model decides). A single run that reaches its step, time,
token or cost limit **pauses and asks**, using the same `waiting_for_user` machinery as `ask_user` above: the run's
`pending_question` is the question below with `options: ["Keep going", "Stop here"]`, and the same
`"<task name> needs your answer"` notification (`payload.event: "question"`) is sent. The run's transcript is kept.

| Limit | Config (`tasks.*`) | Default | Question | Error after "Stop here" |
|---|---|---|---|---|
| Steps (model rounds) | `max_tool_rounds` | 40 | `This task has used 40 steps and isn't finished yet. Keep going for another 40 steps, or stop here?` | `Stopped after 40 steps without finishing. The limits for one run are in Settings > Tasks.` |
| Active time | `run_timeout_minutes` | 30 | `This task has been working for 30 minutes and isn't finished yet. Keep going for another 30 minutes, or stop here?` | `Stopped after 30 minutes without finishing. ...` |
| Tokens on cloud models | `max_tokens_per_run` | 2000000 | `This task has used 2,000,400 tokens of your 2,000,000 token limit and isn't finished yet. Keep going for another 2,000,000 tokens, or stop here?` | `Stopped after using 2,000,400 tokens without finishing. The limit for one run is 2,000,000 tokens. ...` |
| Spend on cloud models | `max_cost_per_run_usd` | 5.0 | `This task has used $5.02 of your $5.00 limit and isn't finished yet. Keep going for another $5.00, or stop here?` | `Stopped after spending about $5.02 on the model without finishing. The limit for one run is $5.00. ...` |

- Answering `Keep going` (any case, surrounding spaces and a final `.`/`!` ignored) raises that limit by its original
  amount for this run only and continues from the transcript (no restart note). Any other answer, including
  `Stop here`, fails the run: status `error` with the message above, an `error` progress update and the usual
  `Task failed` notification (`payload.event: "run_failed"`). Cancel works as for any waiting run.
- The run keeps `limits: {base, max, used}` (each `{steps, seconds, tokens, cost_usd}`) in `task_runs.limits`, saved
  whenever the run pauses, ends or is interrupted (quit or Cancel), and together with the resume when a limit is
  raised, so used amounts and raised limits carry across questions and restarts.
  `pending_question` also stores `limit` (`steps`|`seconds`|`tokens`|`cost_usd`) and `stop_error`; the API shows only
  `{question, options, asked_at}`.
- Time counts only while the run is working, never while it waits for an answer. Time, tokens and cost are checked
  before each model call, so a run goes at most one call past its limit. A tool or model call still running 120 s
  after the time limit (`executor.HARD_DEADLINE_GRACE_S`) is cancelled and the run asks the time question; the
  transcript keeps every finished step and the cancelled call is dropped, so "Keep going" simply tries it again.
  Steps include the executor's "carry on" nudges. Tokens and cost of models with a local prefix (`ollama`,
  `ollama_chat`, `lm_studio`, `llamafile`, `vllm`, `hosted_vllm`) are not counted. Streamed calls to every model except
  Ollama ask for usage (`stream_options.include_usage`). Cost uses LiteLLM's price list and only adds up for models
  whose price it knows. `0` turns a token or cost limit off. A retry starts with fresh limits.
- Repeated calls (`tools.repeated_call_limit`, default 3) that keep getting the same **error** make the run stuck (below).
  Ones that keep getting the same working result never ask: the run fails at once with
  `Stopped because the same step kept repeating: file_read ran 3 times with the same details and got the same result each time. Edit the task to add what it needs, or retry it.`
- Swarm runs and fixed-call runs do not ask. A swarm's workers share one token and cost limit and stop with an error
  when it runs out (the swarm finishes `completed_with_errors`); `run_timeout_minutes` stops the whole swarm or
  fixed call with `Stopped after 30 minutes without finishing. ...`.

### Stuck runs
A single run that stops getting anywhere **pauses and says why** instead of failing silently (`sentient/tasks/stuck.py`).
It reuses the `waiting_for_user` pause (not a new status), so restarts, Cancel, notifications and answering from a paired
chat work as for any question. Checked deterministically; no model decides.

| Signal | Config (`tasks.*`) | Default | `reason` example |
|---|---|---|---|
| No activity: no streamed model output (text or thinking), tool progress or tool result | `stuck_after_minutes` (0 = off) | 10 | `the AI model hasn't answered for 10 minutes`, `Browser hasn't responded for 10 minutes` |
| One tool fails with the same error that many times in a row (any details) | `stuck_after_repeated_errors` (0 = off) | 5 | `Restaurant keeps failing with the same error: The booking server said no` |
| The same call gets the same error `tools.repeated_call_limit` times (loop breaker) | `tools.repeated_call_limit` | 3 | as above |
| A step only the user can do: a tool result with `needs_user` (the browser refusing a password, PIN, card or one-time code field), or a browser page asking the visitor to prove they are a person | | | `the page asks for your password`, `the site wants proof that you're a person (a CAPTCHA)` |

- On no activity the call in flight is cancelled; the transcript keeps every finished step and drops that call.
- The run's `pending_question` is `{question: "I'm stuck: <reason>. What should I do?", options: ["Try again",
  "Skip this step", "Cancel"], kind: "stuck", reason}` (stored with `stuck`: `stalled|errors|blocked`). The progress log
  gets `Waiting for your answer: ...` and the notification is `kind: "task"`, title `"<task name> is stuck"`, message
  `Sentient is stuck on '<task name>': <reason>. Open it to help or cancel.`, `payload: {task_id, event: "question",
  run_id, question, options, stuck: true, reason}`.
- Answering (`POST .../answer`, a button, or a reply in a paired chat; case, spaces and a final `.`/`!` ignored):
  `Cancel` cancels the run (progress `Cancelled after getting stuck.`). `Try again`, `Skip this step` or any other text
  adds a note for the model (try again and differently if it fails the same way / skip that step and say so in the
  final answer / the user's own words) and the run carries on from its transcript.
- Tools can say a step needs the user by returning `{"error": "...", "needs_user": "<plain reason>"}`.
- The executor is told to call `ask_user` when only the user can get past a step (a login, a CAPTCHA, a code).
- Swarm workers and fixed-call runs are not watched; they have their own time limit.

### Missed runs (catch-up)
Scheduled runs (`active` recurring and `pending` one-off tasks) whose time passed while the computer was off or asleep,
Sentient was not running, or Stop everything was on, are handled on the next scheduler tick (`sentient/tasks/catchup.py`).
A run counts as missed when it is more than `max(300, 3 × tasks.tick_seconds)` seconds late.

- Each missed task follows its schedule's `catch_up`: absent (auto) runs it **once** now if it is less than
  `tasks.catch_up_window_hours` (default 12; 0 always skips) late, else skips it; `"run"` always runs it once;
  `"skip"` never catches up. Other values are dropped when the schedule is saved; triggered schedules have none.
- Never a backlog: a recurring task's next time is computed from now, so missed occurrences are not replayed.
  A skipped recurring task moves to its next time and stays `active`. A skipped one-off task becomes `error` with
  `Skipped: it was due Sep 14, 18:30, while the computer was off or asleep. Choose Run now if you still want it.`
- `interval` schedules (every N minutes) run once and are not reported, since their next check is due anyway; with
  `catch_up: "skip"` the missed check is skipped (and reported) instead.
- One notification per catch-up: `kind: "task"`, title `Caught up after sleep: ran 1, skipped 2` (or `after Sentient was
  off` on startup, `after resuming` after Stop everything), message `Ran once now: 'A'.` and/or `Skipped: 'B', 'C'. Open
  one and choose Run now if you still want it.`, `payload: {event: "caught_up", reason: "start"|"sleep"|"resume",
  ran: [{task_id, name, due_at}], skipped: [...], task_id?}` (`task_id` when only one task is listed).
- Waking from sleep is a jump of the wall clock between scheduler ticks of more than `tick_seconds` + 120 s.
- While Stop everything is on nothing is caught up; it happens on the first tick after Resume.

---

## 5. Integrations

**Integration object**
```json
{"id": "gmail", "display_name": "Gmail", "description": "...", "category": "productivity|communication|knowledge|development|utilities|core",
 "icon": "gmail", "auth_type": "builtin|oauth|api_key|manual|mcp",
 "connected": false, "account_label": "me@example.com", "status": "disconnected|connecting|connected|error", "error": null,
 "setup": {"fields": [{"key": "client_id", "label": "OAuth Client ID", "secret": false, "required": true, "help": "..."}],
           "instructions_md": "1. Open ... 2. ...", "docs_url": "https://..."},
 "privacy_filters": {"supported": true, "fields": ["keywords", "emails", "labels"]},
 "triggers": [{"event": "new_email", "label": "New email"}],
 "alternative_for": null,
 "tools": [{"name": "gmail_search", "description": "...", "risk": "read"}]}
```
- `alternative_for`: for optional keyed providers that replace a keyless builtin (`accuweather` → `weather`,
  `newsapi` → `news`, `brave_search`/`google_cse` → `internet_search`, `google_maps` → `maps`); these have no tools
  of their own. `null` otherwise.
- Builtins (`internet_search`, `web`, `weather`, `maps`, `news`, `charts`) are always `connected: true`.
- Google integrations share one OAuth Desktop client stored once; after it is saved, their `setup.fields` come back
  with `required: false` and `connect` may be called with `{fields: {}}`.
- Tools of disconnected integrations are hidden from the model (`integrations.hide_disconnected_tools`), but are
  always listed here.
- `GET /api/integrations` → `[Integration]`
- `GET /api/integrations/{id}` → `Integration`
- `POST /api/integrations/{id}/connect` `{fields: {...}}` →
  - api_key/manual: validates → `Integration`
  - oauth: `{auth_url, state}`; the desktop opens `auth_url` in the system browser; a loopback listener finishes the flow and emits `integration.updated`
  - github with `integrations.github_oauth_client_id` set and no `token` field: device flow, `{auth_url, state, user_code}`; show `user_code` for the user to type at `auth_url`; completion arrives as `integration.updated`
  - validation failures → HTTP 400 `{detail: "friendly message"}`; the integration's `status`/`error` also update
- `POST /api/integrations/{id}/disconnect` → `Integration` (also disables tasks that depend on it, v2 behaviour)
- `POST /api/integrations/{id}/test` → `{ok, detail}`
- `GET /api/integrations/{id}/privacy-filters` → `{keywords: [], emails: [], labels: []}`
- `PUT /api/integrations/{id}/privacy-filters` same shape → `{ok}`
- `GET /api/integrations/mcp` → `[{name, transport: "stdio|http", command, args, url, env_keys, enabled, status: "connecting|connected|error|disconnected|disabled", tools: [{name, mcp_name, description, risk}], error}]`
  (`name` is the Sentient tool name `mcp_<server>_<tool>`; `env` values are kept in the keychain, only `env_keys` are returned)
- `POST /api/integrations/mcp` `{name, transport, command?, args?, url?, env?, enabled?}` → server object (waits up to 15 s for the first connection; replaces a server with the same name; 400 on invalid input)
- `DELETE /api/integrations/mcp/{name}` → `{ok}`
- `POST /api/integrations/mcp/{name}/test` → `{ok, tools: [mcp tool names], error?}`
- `PUT /api/integrations/{id}/privacy-filters` → 400 when the integration has `privacy_filters.supported: false`

- `GET /api/integrations/feeds` → `[{source, display_name, kind: "gmail_history"|"calendar_sync_token"|"imap_idle", connected, active,
  status: "disconnected"|"off"|"starting"|"ok"|"error", last_sync_at, last_success_at, last_error, note, failures, next_attempt_at, emitted}]`
  (change feeds and push watchers, section 16; `note` explains a re-baseline such as expired Gmail history)
- `POST /api/integrations/feeds/{source}/sync` → `{ok, emitted, rebaselined}` or `{ok: false, error, failures, retry_in_s, emitted: 0}`
  or `{ok: false, skipped: "not connected", emitted: 0}`; 404 when the source has no change feed.
- `email_imap` ("Email (IMAP)", `auth_type: "manual"`): setup fields `host`, `port` (993), `username`, `password` (app password,
  secret), optional `smtp_host`, `smtp_port`; connect signs in to IMAP (and SMTP when given) before saving. Tools
  `email_imap_search` (read), `email_imap_read` (read), `email_imap_send` (send). Privacy filters like Gmail. Trigger `new_email`.
- `webhook` ("Webhooks", builtin, no tools): its `triggers` are computed from the user's hooks,
  `[{event: <hook id>, label: "When \"<name>\" is called"}]`; `integration.updated` is sent when hooks are created or deleted.
- `gcalendar.triggers` are `new_event` ("New event") and `updated_event` ("Changed event").

Backend-only API used by tasks/proactivity (not HTTP):
- `app.integrations.poll_source("gmail"|"gcalendar", since_iso|None)` returns new, privacy-filtered items once each and also
  publishes them as `source.items` with `origin: "poll"` (section 16).
- `app.integrations.feed_active(source) -> bool` (sync): a change feed (gmail, gcalendar) or IMAP push watcher (email_imap) is
  keeping the source current, so timer polling can skip it. False when not connected, feeds are off, or after 3 failed syncs in a row.
- `await app.integrations.feed_status()` → the list returned by `GET /api/integrations/feeds`.
- `await app.integrations.emit_items(source, origin, items, event=None)` → published items (shared seen record, privacy filters).
- `await app.integrations.recent_threads("gmail"|"email_imap", newer_than_days=, idle_days=, limit=40)` → `{addresses, threads, note?}`
  for follow-ups (section 6). `threads`: `[{source, thread_id, url, messages: [...]}]`, conversations active in the last
  `newer_than_days` days whose newest message is at least `idle_days` old. Messages use the gmail item shape plus `cc`,
  `message_id`, `headers` (only `list-unsubscribe`, `list-id`, `precedence`, `auto-submitted`, `content-type`) and
  `from_me`; IMAP messages also carry `mailbox` and only the newest message has its text. `addresses` are the user's own
  addresses seen (account address and senders of sent mail). A thread is dropped whole when the privacy filters hide any
  of its messages or people. IMAP finds the Sent mailbox by SPECIAL-USE `\Sent`, else the names `Sent`, `Sent Items`,
  `[Gmail]/Sent Mail`, `Sent Messages`, `Sent Mail`, `INBOX.Sent`; without one it returns no threads and
  `note: "no_sent_mailbox"`. Read-only; raises `IntegrationError` when the service call fails.
  Gmail pages through the thread listing (at most 5 pages of 50) and drops threads whose newest message is newer than
  the idle cutoff before they count toward `limit`, so busy threads can't crowd out quiet ones. IMAP runs two capped
  searches per mailbox (INBOX and Sent): quiet messages from the max age up to the idle cutoff (newest 150) and recent
  messages after it (newest 400, to see which conversations are still active), and reads the headers of both.

Item shapes. gmail: `{id, thread_id, from, sender_email, to, subject, snippet, body, date, labels, url}`;
email_imap: the gmail shape plus `message_id` (`id` is the IMAP UID, `url` is null, `labels` are `INBOX` plus `UNREAD`/`STARRED`);
gcalendar: `{id, summary, description, start, end, all_day, location, attendees, organizer_email, url, status, created, updated, meet_link}`.

One-time codes and sign-in links in email. While `integrations.hide_one_time_codes` is on (the default), every gmail
and email_imap item (tool results, polls, change feeds, IMAP push, follow-up threads) has one-time codes, verification
and 2FA codes, magic sign-in links and password reset links in its `subject`, `snippet` and `body` replaced with
`[one-time code hidden]`, `[sign-in link hidden]` or `[password reset link hidden]` before it leaves the plugin
(`sentient/integrations/redact.py`). Detection is deterministic: a code needs a sign-in or verification cue nearby
(one-time or OTP, verification, security, sign-in or login, 2FA, two-factor, authentication, passcode, "use this code
to", "enter this code", "is your ... code", or a code alone on the line after such a cue) or a sign-in or reset subject
or sender; a link must carry a token-like value and look like sign-in or reset (its path, the words before it, or the
email's subject). Booking, order, ticket, reservation, PNR and reference codes are always kept, even when numeric, and
an email whose subject is a booking or order (with no sign-in cue) keeps all its codes. Order numbers, dates, prices,
phone numbers and ordinary links are left alone. The proactive pipeline and follow-ups
run the same masking again before their prompts. The original stays in the user's mail app (`url` for Gmail).

---

## 6. Notifications & proactivity

**Notification** `{id, kind: "info|task|approval|proactive|skill|error", title, message (markdown), payload: {}, task_id, read, created_at}`

Proactive suggestion payload:
```json
{"suggestion": {"suggestion_type": "draft_email_reply", "description": "Draft a reply to Jane confirming Tuesday",
  "action_details": {"action_type": "draft_email", "...": "..."}, "reasoning": "...", "confidence": 0.82,
  "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Jane: Meeting Tuesday?"}},
 "status": "pending|approved|dismissed|expired", "task_id": null}
```
`source_event` also carries `item_id` and `url` (when the item has one). `source` is the watched app (`gmail`, `gcalendar`,
`email_imap`), `webhook` (then `event_type` is the hook id and `hook_name` is set) or `heartbeat`.
Notification `title` names what the suggestion is about: `"Gmail: Jane Doe: Meeting Tuesday?"`, `"Webhook: Deploy alerts"`,
`"Suggestion from Check-in"`; `message` is the suggestion description.
Suggestions produced during `proactivity.quiet_hours` are held and delivered as notifications when quiet hours end (dropped after 24 h).
One item produces at most one suggestion, and a suggestion that repeats one still pending is dropped.
Pending suggestions nobody acted on expire after `proactivity.suggestion_ttl_hours` (check-ins after 12 h at most; calendar
suggestions as soon as the event starts): `payload.status` becomes `expired` (`notification.updated`) and the notification is marked read.
The heartbeat (`heartbeat_minutes > 0`) looks at upcoming calendar events, tasks needing attention, expiring short-term
memories, the time of day and `app.user_model.context_for`; the model answers `NO_REPLY` unless one nudge is worth it.
It does not run during quiet hours and makes at most `proactivity.heartbeat_daily_cap` suggestions per day.

**Follow-ups** (`proactivity.followups`, on unless proactivity is off) notice dropped email threads in the email accounts
listed in `followups.sources` (default `["gmail", "email_imap"]`; only accounts that are both listed and connected are read,
independent of `proactivity.sources`) through `app.integrations.recent_threads` (section 5). The check runs once a day from 08:00 in the user's timezone,
and in the background after `POST /api/proactivity/poll-now`. Two kinds:
- `waiting_on_you`: the newest message is from someone else, has you in `To` (not only Cc/Bcc), and has had no answer for
  `waiting_on_you_days` (3).
- `waiting_on_them`: your own newest message asked something or asked for something and has had no answer for
  `waiting_on_them_days` (4).

Deterministic filters run first and skip: no-reply and notification senders, mailing lists and bulk mail (`List-Unsubscribe`,
`List-Id`, `Precedence: bulk|list|junk`, `Auto-Submitted`, Gmail Promotions/Social/Updates/Forums), calendar invites and
auto-replies, mail from your own addresses, notes to yourself, short thank-you notes, threads quiet for more than
`max_age_days` (21), and any thread already suggested, dismissed or judged by the model in the same state (dedupe on thread id
plus newest message id, so a new message makes it eligible again). At most 10 threads per check reach the `fast` model, which
answers `{needs_follow_up, about, draft, confidence}` (parsed loosely; an unusable answer, or a draft with a placeholder
such as `[day]`, `{name}`, `<date>`, `XX`, `TBD` or `(your name)`, is dropped and retried at the next check, because the
draft is sent exactly as written);
at most `max_suggestions` (3) suggestions per check, and the learned per-type threshold applies. Nothing is ever sent by the
check itself.

Follow-up suggestion: `suggestion_type` `follow_up_reply` or `follow_up_nudge`, `source_event.event_type: "follow_up"`,
`source_event.item_id` is `<thread id>:<newest message id>`, `description` is a plain title
(`"Priya is waiting for your reply about the invoice"`, `"No reply from Rohan about Saturday yet"`), the notification
`message` is the description followed by the draft as a quote, and the suggestion carries
```json
"follow_up": {"kind": "waiting_on_you|waiting_on_them", "person": "Priya Shah", "person_email": "priya@acme.example",
  "to": "priya@acme.example", "subject": "Re: Invoice for September", "draft": "Hi Priya, ...", "days_waiting": 4,
  "thread_id": "...", "message_id": "<gmail id or IMAP uid>", "mailbox": "INBOX (IMAP only)"}
```
Approving it (the card's **Send reply** / **Send nudge**, which shows the exact draft) is the approval of that one send:
it creates an already approved task (`app.tasks.create_approved_call`, section 4) that runs right away, whatever
`tasks.require_plan_approval` says, and makes exactly one call with the draft unchanged: `gmail_reply {message_id, body}` in
the same thread, or `email_imap_send {to, subject, body, reply_to_message_id?}`. No model is involved in the send. A lasting
Never rule on that tool or its app (`tools.approvals.rules`) still applies: the run fails with the rule's message and
nothing is sent.
- `GET /api/notifications?limit=&unread_only=` → `{notifications: [Notification], unread}`
- `POST /api/notifications/{id}/read`, `POST /api/notifications/read-all`
- `DELETE /api/notifications/{id}`, `DELETE /api/notifications`
- `POST /api/proactivity/suggestions/{notification_id}` `{action: "approve"|"dismiss"}` → `{ok, task_id?}` (approve creates a task from `action_details` with `original_context.source = "proactive"`; for a follow-up it creates the already approved send task described above; both update the learned per-type threshold, mark the notification read and set `payload.status`/`payload.task_id`). Errors: 400 bad action, 404 not a suggestion, 409 already actioned or expired, 503 tasks unavailable. v2 spellings `approved`/`dismissed` are accepted.
- `GET /api/proactivity/status` → `{enabled, last_poll_at: {gmail, gcalendar}, sources: [{source, connected, last_poll_at, last_error, feed_active}], suggestions_today, quiet_now, heartbeat_minutes, followups: {enabled, last_run_at}}`. `sources[].last_error` is the message the integrations package recorded when `poll_source` raised `IntegrationError` (other sources keep polling); `null` after the next successful poll. `feed_active: true` means a change feed delivers that source's items, so it is not timer-polled.
- `POST /api/proactivity/poll-now` → `{ok, events}` (`events` = new items seen; polls every connected source, including ones with an active change feed, then starts a follow-up check in the background when follow-ups are on. Triggered tasks are fired by the tasks package from `source.items`, section 16)
- `GET /api/proactivity/preferences` → `[{suggestion_type, score, threshold, approvals, dismissals}]`
- `DELETE /api/proactivity/preferences/{suggestion_type}` → `{ok}` (`ok: false` when there was nothing to reset)

`threshold = clamp(base_confidence_threshold - 0.05 * score, 0.40, 0.95)` (v2).

---

## 7. Memory

**Memory** `{id: int, content, topics: string[], source, memory_type: "long-term|short-term", created_at, updated_at, expires_at, previous_content: string|null}` (`previous_content` is the text before the last UPDATE)

Topics are the v2 set: Personal Identity, Interests & Lifestyle, Work & Learning, Health & Wellbeing,
Relationships & Social Life, Financial, Goals & Challenges, Miscellaneous.

- `GET /api/memories?topic=&q=&source=&limit=&offset=` → `[Memory]` newest first, expired short-term facts excluded (`q` = hybrid search when embeddings are available: vector neighbours plus FTS5 keyword matches, ordered by `score` = `similarity` + `memory.keyword_weight` × share of query words present, each result carrying `similarity` and `score`; falls back to a keyword match)
- `GET /api/memories/topics` → `[{name, description, count}]`
- `GET /api/memories/graph` → `{nodes: [{id, label, title, content, topics, memory_type, source, created_at}], links: [{source, target, value}]}` (`label` = content truncated to 25 chars, `title` = full content, as in v2; a link means cosine similarity ≥ `memory.graph_link_similarity`, `value` is that similarity)
- `POST /api/memories` `{content, source?}` → `{action: "ADD"|"UPDATE"|"DELETE"|"SKIP", id, content}` (runs the CUD decision, so a duplicate returns `SKIP` with the existing id; `source` defaults to `manual`. The decision also sees up to 3 facts about the same person and attribute found by keyword (where they live, job, relationship, diet, health, ownership, routine), and a new current residence ("moved to Bengaluru") always UPDATEs the old one ("lives in Pune") rather than adding a second home; past-tense facts are left alone)
- `PUT /api/memories/{id}` `{content}` → `Memory` (id kept; topics, long/short-term and expiry re-analyzed; embedding refreshed; 404 if missing)
- `DELETE /api/memories/{id}` → `{deleted: true}` (404 if missing)
- `DELETE /api/memories/source/{source}` → `{deleted: n}`
- `POST /api/memories/import` multipart `file` (pdf/txt/md/docx) → `{added, updated, skipped, source}` (`source` = `file:<name>`; existing memories are kept; 400 for other types)
- `GET /api/memories/summaries?limit=` → `[{id, content, start_at, end_at, session_id}]`
- `GET /api/memories/workspace` → `{soul, user, memory, today, yesterday}` (full file contents, not the prompt-budgeted snapshot)
- `PUT /api/memories/workspace/{soul|user|memory}` `{content}` → `{saved}`
- `GET /api/memories/personas` → `[{id, name, description, soul_md}]` (SOUL.md presets rendered with the current assistant and user names)

- `GET /api/memories/dreams?limit=`, `POST /api/memories/dreams/run`, `GET /api/memories/dreams/{id}`: see section 15.

When sqlite-vec cannot load, list/topics/graph return empty data and write routes return 503.

Engine notes: recall used by the system prompt and `memory_recall` is hybrid (same ranking as `q` above) and
counts each recalled fact (`facts.recall_count`, `last_recalled_at`; dreaming promotes often-recalled short-term
facts). Bulk and consolidation changes publish `memory.updated` with `reason` (`merged` | `contradicted` |
`promoted` | `expired`) and, for merges/contradictions, `merged_into` / `superseded_by`.

---

## 8. Skills & self-evolution

**Skill** `{name, description, author: "user|assistant|community", state: "active|pending_review|stale|archived", tags, requires_tools, version, use_count, view_count, patch_count, last_used_at, success_count, failure_count, last_failure_at, created_by_review: bool}`
(`success_count`/`failure_count` count chat turns and task runs that viewed the skill and went well or failed, see section 15)

- `GET /api/skills` → `{active: [Skill], pending: [Skill & {reason, origin, proposed_at}], archived: [Skill]}` (`active` includes skills whose `state` is `stale`; a pending entry whose name also appears in `active` is a proposed update, see `/diff`; `origin` is `{session_id?, task_id?, run_id?, curator?, merged_from?, repair?: true}` or `null`)
- `GET /api/skills/{name}` → `Skill & {body}` (active, else pending, else archived copy; 404 if none)
- `POST /api/skills` `{name, description, body, tags?, requires_tools?}` → `Skill` (author `user`, active immediately; 400 invalid name, 409 exists)
- `PUT /api/skills/{name}` `{description?, body?, tags?, requires_tools?}` → `Skill` (edits the active skill, else the pending proposal; bumps `version` of active skills)
- `POST /api/skills/{name}/approve` | `/archive` | `/restore` → `Skill`; `/reject` → `{ok}` (404 when there is nothing to act on; restore 409 if an active skill has that name)
- `DELETE /api/skills/{name}` → `{ok}`
- `GET /api/skills/{name}/diff` (pending update of an existing skill) → `{current, proposed}`
- `POST /api/skills/review-now` `{session_id?}` → `{reviewed, proposed: [name]}` (with `session_id`: review that chat now; without: every chat with enough tool calls not yet reviewed, plus completed task runs seen on the bus)
- `GET /api/skills/evolution-log?limit=` → `[{ts, kind: "skill_created|skill_patched|skill_repair_proposed|skill_archived|profile_updated|curator_run|summary_created", detail: {}}]` newest first. `detail` examples: `skill_created {name, pending, reason?, session_id?|task_id?}`, `skill_repair_proposed {name, pending: true, origin: "repair", failure: "tool_errors"|"user_correction"|"run_failed", reason, session_id?, turn_id?, task_id?, run_id?}`, `curator_run {staled: [], archived: [], merge_proposals: []}`, `profile_updated {learned_appended, facts_considered, summaries_considered, memory_chars}`, `summary_created {id, session_id, start_at, end_at}`.

Skills proposed by the background reviewer or curator always land in pending with a `skill` notification `payload: {skill, action: "create"|"patch", origin}`.
Repair proposals (section 15) send `payload: {skill, action: "patch", origin: "repair", reason, session_id?, turn_id?, task_id?, run_id?}` with title "Skill fix to review".
When the curator proposes merging near-duplicates it keeps the more reliable skill (successes vs failures), then the more used one.

---

- `PUT /api/skills/{name}` also accepts `target: active|pending|archived` (`pending` edits the proposal only). Pending skills in `GET /api/skills` carry `reason`, `origin` and `proposed_at`.

---

## 9. Voice

### `WS /ws/voice?token=` (desktop) or `WS /ws/voice?node_token=` (devices)
Authentication: the gateway token (query or bearer header), or a device token checked with
`await app.nodes.verify_token(token)` (returns the Node). A bad token, or a nodes service without `verify_token`,
closes the socket with code 4401.

Client → server:
- JSON `{"type": "start", "session_id?": "...", "sample_rate": 16000, "channel?": "voice"|"glasses"|"phone", "mode?": "conversation"|"wake", "audio_format?": "wav"|"pcm16", "output_sample_rate?": 16000, "max_frame_bytes?": 0}` → `ready`, then `state: listening` (`state: standby` in `wake` mode, section 16). Omit `session_id` for a new chat; an unknown id gets a recoverable `error` and a new session. Sending `start` again resets the conversation state.
  - `channel` is stored on a new session and makes the agent use the voice role. Default `voice`; device sessions default to the node's `kind` when it is `glasses` or `phone`. (`phone` sessions currently run the agent as channel `voice`.)
  - `audio_format: "pcm16"` (microcontrollers): replies come as raw PCM16 little-endian mono at `output_sample_rate` (default `sample_rate`, clamped to 8000-48000, resampled on the server), in binary frames of at most `max_frame_bytes` (0 = one frame per chunk). Default `wav`.
- binary frames: PCM16 little-endian mono at `sample_rate` (any chunk size; 20-100 ms is typical). Frames sent before `start` get one `error`.
- JSON `{"type": "end_utterance"}` (push-to-talk release; otherwise server VAD decides). In standby it processes the speech without the wake word.
- JSON `{"type": "wake"}` in standby: wake without the phrase (device button, UI tap) → `wake` with `source: "client"`
- JSON `{"type": "interrupt"}` (barge-in: stop speaking)
- JSON `{"type": "text", "text"}` (typed input into the voice conversation; skips STT)
- JSON `{"type": "approval.respond", "approval_id", "decision": "allow"|"allow_session"|"deny"}` → `approval.ack`
- JSON `{"type": "ping"}` → `{"type": "pong"}`
- JSON `{"type": "stop"}` → cancels any turn in progress, sends `state: idle`, closes the socket

Server → client:
- `{"type": "ready", "session_id", "stt": "...", "tts": "...", "sample_rate", "channel", "mode", "audio_format", "output_sample_rate?" (pcm16), "node_id?" (devices), "wake?": {engine, phrase, ready, model?, error?} (wake mode)}`
- `{"type": "state", "state": "standby|listening|transcribing|thinking|speaking|idle", "session_id"}`
- `{"type": "wake", "phrase", "source": "voice"|"client", "score?", "session_id"}` (section 16)
- `{"type": "transcript", "text", "final": true, "session_id", "stt_ms?"}`
- chat/agent events exactly as on `/ws` (`text_delta`, `thinking_delta`, `tool_call`, `tool_result`, `approval_request`, `usage`, `done` ...)
- `{"type": "audio", "format": "wav", "sentence_index": 0, "text", "session_id"}` followed by one binary frame (a complete PCM16 mono WAV for that sentence or first clause; `text` is what is spoken, markdown/URLs/code removed)
- with `audio_format: "pcm16"`: `{"type": "audio", "format": "pcm16", "sample_rate", "bytes", "frames", "sentence_index", "text", "session_id"}` followed by exactly `frames` binary frames of raw PCM16 mono (`bytes` in total)
- the wake chime uses the same messages with `"earcon": true, "sentence_index": -1, "text": ""`
- `{"type": "audio_end", "session_id", "sentences", "metrics": {audio_ms?, stt_ms?, first_token_ms?, first_sentence_ms?, first_tts_ms?, first_audio_ms?, tts_ms?, total_ms}}` after every turn, or once, early, as `{"type": "audio_end", "session_id", "interrupted": true, "reason": "client"|"barge_in"|"cancelled"}` when speech is cut short (at most one `audio_end` per turn; metrics are milliseconds from the end of the utterance)
- `{"type": "approval.ack", "approval_id", "resolved"}`
- `{"type": "error", "message", "recoverable", "session_id?"}`

Behaviour:
- Server VAD: RMS energy with an adaptive noise floor; `voice.vad_silence_ms` of silence ends an utterance, sounds shorter than `voice.vad_min_speech_ms` are ignored, utterances are cut at `voice.vad_max_utterance_s`.
- Reply text is split into sentences as it streams; each sentence is synthesized as soon as it is complete, so the first audio arrives while the model is still writing. The first chunk of a reply may end at a clause boundary (comma, semicolon, colon, dash) once it is `voice.tts_first_clause_chars` long (default 40, 0 = sentences only).
- Opening a session warms up STT and TTS in the background; `voice.preload_on_start` loads them when Sentient starts. Loaded models stay in memory.
- `interrupt`, or ~300 ms of the user speaking while `state` is `speaking` (when `voice.barge_in` is on), stops synthesis and sending immediately; the model's text keeps streaming and is saved.
- A new utterance (or `text`) while a turn is still running cancels it (`done` with `cancelled: true`, then the new `transcript`). An utterance that ends while the previous one is still being transcribed is merged with it.
- On `approval_request` the assistant also says "I need your approval to use <tool>. Say yes or no."; a spoken yes/no (or `approval.respond`) answers it. Other speech cancels the turn and starts a new one.
- If the socket drops mid-turn, speech stops but the reply finishes so the transcript is persisted.

### Engine API used by other packages
- `await app.voice.transcribe_bytes(data: bytes, filename: str = "") -> str` transcribes a complete audio file (ogg/opus,
  webm, m4a/mp4/aac, mp3, wav, flac; the type comes from `filename`'s extension or is sniffed from the bytes) with the
  configured STT; raises `VoiceError` with a friendly message (empty, over 50 MB, unsupported type, undecodable,
  provider failure). Channels use it for voice notes.
- `/ws/voice?node_token=` authenticates a device with `await app.nodes.verify_token(token) -> dict | None` (the Node).
- Reusable socket for the nodes LAN listener, `sentient.voice.socket`: `voice_socket_endpoint(ws)` authorizes (gateway token
  when `app.state.token` exists, else `node_token`) and serves; mount it with
  `lan_app.add_api_websocket_route("/ws/voice", voice_socket_endpoint)` after setting `lan_app.state.sentient`.
  Lower level: `authorize_voice_socket(ws, core) -> (allowed, node)` and `serve_voice_socket(ws, core, node=None)`
  (accepts the socket and runs the session until it closes). When a middleware has already validated the device and put
  the Node in `scope["state"]["node"]` (the LAN listener does), that node is used without a second lookup.

### REST
- `GET /api/voice/status` → `{sessions, stt: {provider, model, ready, device, compute_type?, note?, error?}, tts: {provider, voice, ready, voices: [{id, name, language}], backend?, downloaded?, variant?, error?}, wake: {engine, phrase, ready, model?, error?}}`. Nothing is loaded by this call; `ready` means the model is in memory (local) or a key is set (cloud). `note` explains a CPU fallback.
- `POST /api/voice/transcribe` multipart `file` (wav/webm/ogg/m4a/mp3/flac) → `{text}` (dictation button). 415 unsupported type, 413 over 50 MB, 503 provider error.
- `POST /api/voice/speak` `{text, voice?}` → `audio/wav` (markdown stripped first). 400 when nothing is speakable, 503 provider error.
- `POST /api/voice/prepare` `{target?: "all"|"stt"|"tts"|"wake"}` (`all` = stt and tts) → NDJSON progress for local models, one object per line: `{stage: "download"|"loading"|"ready"|"error", component: "stt"|"tts"|"wake", progress: 0..1|null, file?, bytes?, total?, device?, note?, message?}`, ending with `{stage: "done", progress: 1, ok}`. Cloud providers report `ready` immediately. First use downloads and loads on demand too; `prepare` just lets the UI show progress.

---

# V3 leap features (contract added 2026-09-15)

Sections 10 to 16 describe the new capabilities. Owners are listed in `CLAUDE.md`.
Every new tool declares a `Risk`; approvals behave as in section 1.

## 10. Agent upgrades (owner: core)

### Steering a running reply
- `WS /ws` client message `chat.steer` `{session_id, text}`. If a reply is running for that session, the text is
  queued and applied at the next model round: it is persisted as a user message and the model sees it before
  continuing. If no reply is running it behaves exactly like `chat.send`.
- Server events: `steer_ack` `{session_id, queued: bool, client_id}` right away, then `user_interjection` `{text}` (with
  `session_id`, `turn_id`) when the model receives it. A steer that arrives after the final round starts a new turn.
- `chat.send` while a reply is running is treated as `chat.steer` (no more "A reply is already in progress" error).
  Exception: a `chat.send` with `attachments` never steers; it starts a new turn once the running one finishes.
- A steer that arrives while the final answer is still streaming is applied too: that answer is kept (persisted as an
  assistant message), the interjection follows, and the model continues in the same turn (one `done` at the end).
  When no rounds are left, or the reply already finished, the text becomes a new turn (`done`, then a new `turn_id`).
- Steer messages are persisted with `interjection: true` (message rows from `GET /api/sessions/{id}/messages`;
  every other row has `false`). A steer that became a new turn is an ordinary user message.
- `POST /api/chat` with the `session_id` of a running reply (and no attachments) steers it and returns a single NDJSON
  line `{type: "steer_ack", session_id, queued: true}`.
- Engine API (channels, voice): `app.agent.steer(session_id, text) -> bool` (False when no reply is running: start a
  normal turn) and `app.agent.is_running(session_id) -> bool`.

### Tool progress, dynamic risk, parallel reads, large results
- Chat event `tool_progress` `{call_id, name, kind: "stdout"|"stderr"|"status"|"frame"|"subagent", text?, image?, data?}`
  streams output of long tools (code runs, browser, subagents) before their `tool_result`.
  Tools emit it through `ctx.progress(payload: dict)`; outside a chat turn it is a no-op. It may be called with or
  without `await`, and from worker threads started with `asyncio.to_thread`. Unknown `kind` becomes `status`; payload
  keys other than `kind`, `text`, `image`, `data` are merged into `data`. `ctx.call_id` is the running call's id.
  Task runs and subagents also receive these events from `run_loop` (they may ignore them).
- `Tool.risk_fn(arguments, ctx) -> Risk | None` optional (sync or async; also `@tool(..., risk_fn=...)`): overrides
  `risk` per call (a browser click on "Place order" becomes `send`). A `risk_fn` that raises counts as `exec`. Approvals
  use the effective risk; `approval_request.risk` reports it. `internal` keeps its meaning: an internal tool whose
  effective risk is `write` does not ask in mode "ask"; `send`/`exec` always ask. "Allow for this chat" covers that tool
  up to the risk level that was approved, but never a call whose `risk_fn` raised it to `send`/`exec` (those ask every time).
  Lasting rules (`tools.approvals.rules`, section 2) are checked before the mode: `ask` and `never` win over everything,
  `allow` skips the question except for purchases.
  Outside `run_loop`, use `await app.approvals.requires_approval(tool, arguments, ctx) -> (bool, Risk)`;
  `needs_approval(tool, session_id, risk=None, *, arguments=None, ctx=None)` evaluates a synchronous `risk_fn` when
  given `arguments` (an async one counts as at least `send` there).
- `approval_request` also carries `risk_label` (e.g. "Purchase", "Deletes", "Posts publicly", "Sends", "Runs code",
  "Changes something") and `target` (short label of what is acted on, e.g. "Place order", or null). They come from the
  optional `Tool.describe_fn(arguments, ctx) -> {risk_label?, target?}` (sync or async, also `@tool(..., describe_fn=...)`);
  without it `risk_label` is "Looks something up" / "Changes something" / "Sends" / "Runs code" by effective risk.
- Chats on the `telegram` and `discord` channels get phone-friendly reply guidance in the system prompt; the `phone`
  channel (device voice) is spoken and uses the voice role like `voice` and `glasses`. Helper: `sentient.tools.base.effective_risk(tool, arguments, ctx)`.
- Consecutive tool calls of effective risk `read` that need no approval run concurrently (`chat.parallel_read_tools`,
  default on); others run in order. Their `tool_call` events come first, `tool_progress` may interleave, and
  `tool_result` events, tool messages and persisted rows keep the order the model asked for.
- A tool result longer than `chat.tool_result_max_chars` (default 16000) is cut, the full text is saved under
  `files/outputs/tool-<call_id>.txt`, and the model is told where it is. The `tool_result` event still carries the full result.
- Anthropic models (`anthropic/*`) get prompt caching (`cache_control`) on the system prompt and the tool list.
- Tool arguments that fail validation return `{error: "Invalid arguments for <tool>: ...", schema}` so the model can retry.

### Work nobody asked for (ADR 0017)
- `ToolContext.origin`: `"user"` (default) for work someone asked for (chats, tasks the user created or approved,
  suggestions the user accepted), or one of `sentient.tools.base.UNPROMPTED_ORIGINS` (`"proactive"`, `"heartbeat"`,
  `"followups"`, `"dreaming"`, `"background"`) for work nobody asked for. `app.agent.tool_context(session_id, channel,
  origin=None)` uses the channel when it is one of those names, else `"user"`.
- `run_loop` treats a run as unprompted when `ctx.origin` or its `source` is in `UNPROMPTED_ORIGINS`. When only the
  `source` says so, `ctx.origin` is set to it, so tools the run starts (a subagent) inherit it. Such a run only
  runs calls whose effective risk is `read`, or `internal` tools at `write` outside the `tasks` app
  (`sentient.tools.rules.unprompted_allows`). Anything else is refused before approval modes and lasting rules are
  looked at, so an Allow rule never lifts it; the tool result is `{error: "Nobody asked for this work, so Sentient can
  only look things up. <tool> was not done. If it would help, suggest it to the user instead."}` and the call is
  listed in `LoopResult.held` as `{tool, arguments}`. Broker helper: `app.approvals.unprompted_refusal(tool, risk)`.
- Proactive look-ups run with origin `"proactive"`; held calls are passed to the reasoner, which can offer them as a
  suggestion card. A subagent started from an unprompted run inherits its origin (`delegate(..., origin=)`).
  Heartbeat, follow-ups and dreaming make no tool calls.

### Subagents
- Tools (plugin `subagents`):
  - `delegate_task(goal, context="", tools: list[str] | None = None, background: bool = False)` (risk `write`, internal).
    Foreground: waits and returns `{subagent_id, status, summary, files_created}`. Background: returns
    `{subagent_id, status: "running"}` at once; when it finishes, a summary is added to the chat as an assistant message,
    `subagent.updated` fires and a notification is created.
  - `delegate_tasks(tasks: list[{goal, context?}])` runs up to `subagents.max_concurrent` in parallel and returns every summary.
- **Subagent** `{subagent_id, session_id, parent_call_id, goal, status: "running"|"completed"|"error"|"cancelled", background, summary, error, tool_calls, started_at, finished_at, events: [ProgressUpdate]}`
- `GET /api/sessions/{id}/subagents` → `[Subagent]` newest first; `GET /api/subagents/{id}` → `Subagent`;
  `POST /api/subagents/{id}/cancel` → `Subagent`. Domain event `subagent.updated` → `Subagent` (without `events`).
- While a foreground subagent runs, the parent emits `tool_progress` with `kind: "subagent"` and `data: {subagent_id, message}`.
- Subagents cannot spawn subagents, use the voice role, or run tools of effective risk `send` or `exec`; those calls are
  denied with an explanation the parent can relay.
- Limits: `subagents.max_rounds` steps (default 24), `subagents.timeout_minutes` (20), and on cloud models
  `subagents.max_tokens` (1000000) and `subagents.max_cost_usd` (2.0, when the price is known; `0` = no limit). The
  repeated-call breaker applies too. Reaching one ends the subagent with `status: "error"` and a plain `error`
  (`Stopped after 24 steps without finishing.`, `Stopped after using 1,000,600 tokens without finishing. ...`).

### Hooks other packages rely on
- Before compressing old turns, `run_turn` calls the memory package's
  `FactMemory.flush_conversation(transcript: str, user_name: str) -> list[dict]` so durable facts survive compression.
- The system prompt includes `await app.user_model.context_for(user_text)` (a short markdown block, may be empty).
- `chat.turn_completed` data: `{session_id, turn_id, tool_calls, tool_errors, skills_viewed: [name], user_text, reply}`.
  Task runs publish `task.run_finished` `{task_id, run_id, status, tool_errors, skills_viewed}`.
- `Agent.run_loop(..., stop=fn)`: `fn()` is checked after each round of tool results; when it returns True the loop ends
  without another model call and sets `LoopResult.paused` (task runs use it for `ask_user`).
- `Agent.run_loop(..., budget=Budget(max_tokens, max_cost_usd))` (`sentient.agent.loop.Budget`): checked before each
  model call; when it has run out the loop ends with `LoopResult.stopped_by_budget` (also in `error`). Usage from
  local models is not counted; `StreamChunk.cost` (US dollars, or None when the price is unknown) feeds the cost.
- Loop breaker, on every surface: when the same tool with the same arguments returns the same result
  `tools.repeated_call_limit` times (default 3; a different result starts the count again; `0` = off), the loop ends
  with `LoopResult.stopped_by_repeat`. Unattended loops (tasks, swarm workers, subagents, proactivity) also set `error`.
  With `repeat_nudge=True` (chat turns, including voice and channels) the model is first told once to try something
  else; a further repeat ends the turn with a plain assistant reply (streamed as `text_delta`, then `done`) saying it
  stopped because it kept repeating the same step.
- A `ToolPlugin` with `scoped = True` is never offered by default: `registry.tools()`, `registry.catalog()` and
  `registry.openai_schemas()` leave it out, and `openai_schemas(names)` includes its tools only when named.

## 11. Code execution (owner: sandbox)

> Scripts also get `tools` and `result` without importing them: small local models often forget the import.

- Tool `execute_code(code: str, purpose: str)` (risk `exec`, plugin `code`). The script runs in Python 3 and can call Sentient tools:
  `from sentient_tools import tools, result` then `tools.internet_search(query="...")` returns the tool's JSON result;
  `result(value)` sets the value returned to the model (otherwise stdout is returned). Arguments are keyword-only.
  `tools.available()` lists the callable tool names. A refused call raises `ToolRefused`; a tool that raises, or returns a
  dict with a truthy `error`, raises `ToolError` (both importable from `sentient_tools`; `ToolRefused` subclasses `ToolError`).
- Tool calls from a script: effective risk `read` and internal `write` tools run; other `write`, `send` and `exec` tools
  are refused with a message telling the model to call that tool directly (where approvals can ask the user), unless
  approvals mode is `off`. Lasting rules apply (section 2): `never` tools are not listed and are refused, `ask` tools
  are refused, and `allow` changes nothing here (scripts still only read). Subagent, voice and code tools are not available inside scripts. At most
  `sandbox.max_tool_calls` calls run per script; `tool_calls` counts calls that ran (refused ones are not counted).
- Returns **SandboxResult** `{ok, backend: "process"|"docker", stdout, stderr, result, files_created: [name], tool_calls, duration_ms, error}`.
  stdout and stderr stream as `tool_progress` (`kind: "stdout"|"stderr"`, newlines normalized to `\n`) and are each capped at
  `sandbox.max_output_chars` (a `[output cut after N characters]` note is appended). Files the script writes to its working
  folder are copied to `files/outputs/<run_id>/`; `files_created` holds names relative to the files folder
  (e.g. `outputs/<run_id>/chart.png`, usable with `/api/files`). `error` is a plain-language sentence when `ok` is false:
  syntax errors (checked before running, with the line number), runtime errors with the failing line, timeouts, code
  execution turned off, Docker not running when the backend is `docker`.
- Python API: `await app.sandbox.run(code, *, session_id=None, channel="system", timeout_s=None, allowed_tools=None, on_output=None, read_only=False) -> dict` (SandboxResult).
  `allowed_tools`: tool names or plugin ids the script may call (`None` = every tool the policy allows).
  `on_output(kind, text)`: sync or async callback per output chunk. `read_only`: only effective risk `read` tool calls run.
- Backends: `process` runs the engine's Python (`-I`, one process per run, killed as a whole tree on timeout) in an
  isolated working folder `~/.sentient/sandbox/<run_id>` (removed afterwards) with a scrubbed environment; tools are
  reached over a loopback-only HTTP endpoint with a one-time token. `docker` runs `sandbox.docker_image` (default
  `python:3.12-slim`) with the folder mounted, memory/CPU/pids limits and no capabilities; with
  `allow_network_in_docker` off (default) the container has `--network none` and reaches tools through request files
  in the mounted folder, otherwise over `host.docker.internal`. `auto` uses Docker when `docker info` succeeds, else
  `process`. Neither receives API keys.
- `GET /api/sandbox/status` → `{enabled, backend, docker_available, python_version}` (`backend` is the one a run would use now).
- `POST /api/sandbox/run` `{code}` → SandboxResult (tool calls limited to risk `read` regardless of approvals mode; used by "Run again" and script-job tests).

## 12. Browser (owner: browser)

- One persistent browser profile under `~/.sentient/browser/profile`, driven with Playwright through an installed Edge
  or Chrome (`browser.engine` auto: Edge, then Chrome). Starts on the first tool call, hidden unless `browser.headless`
  is false, one window with tabs, closes itself after `browser.idle_minutes` without use (not while visible).
- The `browser` plugin is hidden from the model when `browser.enabled` is false or no supported browser is installed;
  status `error` says why in plain words. Config: `enabled, engine, headless, idle_minutes, allow_domains,
  block_domains, max_snapshot_chars, max_extract_chars, confirm_purchases, live_view`.
- Tools (plugin `browser`). Failures return `{error}` with a message the model can act on.
  - `browser_open(url)` read → same as `browser_snapshot` (http/https only; allow/block lists apply, also after redirects).
  - `browser_snapshot()` read → `{url, title, text, truncated?}`. `text` is `Page:`/`URL:`/`Scroll:` header, interactive
    elements one per line (`[e12] button "Sign in"`, `[e4] textbox "Search" value=""`, `[e7] combobox "Country"
    value="India" options: India | Japan`, `[e3] link "Docs" -> /docs`, flags `checked`, `disabled`, `focused`) and the
    readable page text, capped at `browser.max_snapshot_chars`. Refs are valid until the next snapshot of that page or a
    navigation; a stale ref returns an error telling the model to snapshot again.
  - `browser_click(ref)` write; effective risk `send` (via `risk_fn`) when the element's text/aria/value looks like buy,
    pay, place order, checkout, send, post, publish, delete, remove, confirm, subscribe, transfer, or it submits a form
    with payment fields → `{ok, clicked, url, message, new_tab?, dialogs?}`. A link opening a new tab makes it active.
    `confirm()` dialogs are accepted only for clicks approved as `send`; alerts are accepted; `dialogs` lists them.
    Defensive check at call time: a send-looking click that approvals did not rate `send` (page changed, or no approval
    step ran and approvals mode is not `off`) is refused with an explanation.
  - `browser_type(ref, text, submit=False)` write (`send` when submit presses Enter in a payment form, a form whose button
    sends/posts/buys, or a message box) → `{ok, typed_into, submitted, url}`. Refuses password, card, CVV, one-time-code,
    PIN, bank/ID-number and secret-key fields (type, autocomplete, name/id/label heuristics) and text containing a card
    number, telling the model to ask the user to use Open browser.
  - `browser_select(ref, option)` write (label, then value, then index) → `{ok, selected, in, url}`.
  - `browser_press(key)` write (`send` for Enter under the same rules as submit; friendly names like `enter`, `esc`,
    `page down`, `ctrl+a`) → `{ok, pressed, url}`.
  - `browser_scroll(direction="down")` read (`down|up|top|bottom|left|right`) → `{ok, scrolled, from_top, more_below}`.
  - `browser_back()` read → `{ok, url, title}`; `browser_tabs()` read → `{tabs: [{index, url, title, active}]}`;
    `browser_switch_tab(index)` read → `{ok, index, url, title}`.
  - `browser_extract(question="")` read → `{url, title, content, question?, truncated?}`: readable main text without
    nav/header/footer/scripts, capped at `browser.max_extract_chars`; with a question, the most related paragraphs are kept.
  - `browser_screenshot()` read → `{file: "outputs/browser/screenshot-....png", url, title}` (name usable with `/api/files`).
  - `browser_close()` read → `{ok, message}`: closes the whole browser (it reopens on the next tool call).
- Signing in is always done by the user: `POST /api/browser/open` `{url?}` closes the hidden browser and relaunches the
  same profile as a visible window at `url` (default: the page the assistant was on). When the user closes that window,
  the next tool call relaunches hidden.
- `GET /api/browser/status` → `{available, running, engine: "msedge"|"chrome"|"chromium"|null, headless, tabs: [{index, url, title, active}], error}`
- `POST /api/browser/open` `{url?}` → status (409 `{detail}` when the browser is off, missing, the URL is blocked or it
  cannot start); `POST /api/browser/close` → status; `GET /api/browser/screenshot` → `image/jpeg` (409 when not running).
- Domain events: `browser.updated` → status (start, stop, window closed by the user, tab list/url/title changes);
  `browser.frame` `{url, title, image: "data:image/jpeg;base64,..."}` (~1024px wide, quality 55) at most once per second
  after acting tools, with a trailing frame so the last state is shown. The same image also goes to the running chat as
  `tool_progress` `{kind: "frame", image}` when `browser.live_view` is on.

## 13. Devices (owner: nodes)

A device ("node") is a phone, a pair of smart glasses, a watch, or the desktop app itself. Full protocol in `docs/NODES.md`.

- `WS /ws/node` on the gateway and, when `nodes.lan_enabled`, on the LAN listener. On the gateway, `?token=<gateway token>`
  authenticates the desktop app itself as the built-in node `desktop` (kind `desktop`, no pairing); paired devices and pairing
  codes also work on the gateway (local testing). On the LAN listener (`wss://<lan-ip>:<nodes.lan_port>/ws/node`, self-signed
  certificate, SHA-256 fingerprint pinned by devices) devices authenticate with `token` or `pair_code` in `hello`; the gateway
  token is refused there. The LAN listener serves only `/ws/node`, `/ws/voice`, `/node/` and `POST /api/nodes/upload`, and is
  announced over mDNS as `_sentient._tcp.local.` (TXT `version`, `protocol`, `fingerprint`, `port`, `path`, `tls`) when `nodes.mdns_enabled`.
- node → server `hello` `{protocol: 1, name, kind: "phone"|"glasses"|"desktop"|"watch"|"custom", platform, app_version, capabilities: [string], token?, pair_code?, state?}`;
  server → node `welcome` `{protocol: 1, node_id, name, assistant, server_version, keepalive_s, idle_timeout_s, token?}` (`token` only on first
  pairing; the node stores it; `stopped` is true while Stop everything is on, section 17) or `error` `{code, message}` then close. Codes (close code): `pairing_required`, `bad_token`, `bad_code`,
  `revoked` (4401), `rate_limited` (4429, 5 wrong codes per minute per address), `disabled` (4403), `protocol` (4400), `replaced` (4409, the
  same node connected again). Later non-fatal errors: `protocol`, `unknown_type`.
- Capabilities: `camera.photo`, `screen.capture`, `location.get`, `notify.show`, `display.text`, `display.card`,
  `audio.play`, `audio.pcm`, `speak`, `mic.stream`, `clipboard.read`, `clipboard.write`, `button.events`, `battery`. Params per
  capability in `docs/NODES.md`. `audio.pcm` is `audio.play` for microcontrollers: the engine sends raw PCM16 as a binary frame
  instead of base64 (`params` `{format: "pcm16", sample_rate, channels, text, binary: true, bytes}`).
- server → node `invoke` `{id, capability, params, timeout_ms}`, optionally followed by one binary frame when
  `params.binary` is true (`params.bytes` is its length); node → server `result` `{id, ok, data?, error?: string | {code, message}}`.
  `camera.photo` and `screen.capture` data `{mime, base64}`; `location.get` `{lat, lon, accuracy_m, label?}`. Large payloads may instead
  arrive as `data: {mime, binary: true}` followed by one binary frame, or as `data: {mime, upload_id}` after `POST /api/nodes/upload`
  (`Authorization: Bearer <node token>`, raw body or multipart `file`, max 20 MB → `{upload_id, mime, size}`); the engine converts both to `base64`.
- node → server `event` `{event: "button"|"wake"|"gesture"|"battery"|"presence"|"notification_action", data}`;
  `state` `{battery?, charging?, worn?}`; `ping` → `pong`. The engine never pings devices; a device silent for `nodes.idle_timeout_s` is disconnected.
- node → server `stop_all` and `resume` (no fields): Stop everything and Resume from a paired device (section 17).
  server → node `stop_state` `{stopped, stopped_at, source}` goes to every connected device whenever the state changes
  (and to the sender when it did not change). Devices that do not show it can ignore it.
- Voice from a device: `WS /ws/voice?node_token=<token>` with `start` `{channel: "glasses"|"phone", sample_rate, audio_format?: "wav"|"pcm16"}`.
  On the LAN listener the node token is verified before the voice socket runs.
- Pairing: `POST /api/nodes/pairing` → `{code, expires_at, lan_enabled, urls: [string], web_url, fingerprint, qr: "sentient://pair?url=...&code=...&fp=...", qr_svg}`
  (code valid 10 minutes, single use). `urls` are the LAN `wss://` node URLs when the listener runs, else the gateway's `ws://` URL;
  `web_url` opens the web device app with the code (`https://<lan-ip>:<port>/node/#code=...`, or the gateway's `/node/`);
  `qr_svg` is an inline SVG QR code of `web_url`; `fingerprint` is null when the LAN listener is not running.
- **Node** `{node_id, name, kind, platform, app_version, capabilities, online, connection: "lan"|"local"|null, last_seen_at, battery, charging, worn, created_at}`
- `GET /api/nodes` → `[Node]` (online first); `GET /api/nodes/{id}` → `Node`; `PATCH /api/nodes/{id}` `{name}` → `Node`;
  `DELETE /api/nodes/{id}` → `{ok}` (revokes the token and disconnects; 400 for `desktop`);
  `POST /api/nodes/{id}/invoke` `{capability, params, timeout_ms?}` → `{ok, data?, error?, code?}` with `code` `offline`, `unsupported`,
  `timeout`, `not_allowed` (desktop screen capture off), `bad_upload` or the device's own code;
  `GET /api/nodes/lan` → `{enabled, running, port, urls, web_urls, fingerprint, mdns, error}`.
- Domain events `node.updated` → `Node`, `node.deleted` `{node_id}`, `node.event` `{node_id, event, data}`.
- Tools (plugin `devices`, hidden unless an online device offers a capability the tools use; the desktop node counts only for
  `screen.capture`): `device_list()` read; `device_take_photo(device="", question="")` read (effective risk `send` while
  `nodes.camera_requires_approval`), returns `{file, device, description, note?}` (`file` under `files/outputs/devices/`; description from
  the vision role when a question is given; `note` explains a model that cannot read images); `device_capture_screen(device="", question="")`
  same shape and risk; `device_get_location(device="")` read → `{device, lat, lon, accuracy_m, label?, map}`; `device_notify(text, device="")`,
  `device_display(text, device="")`, `device_speak(text, device="")` write internal → `{ok, device}` (speak falls back to engine TTS + `audio.play`).
- Engine API: `await app.nodes.verify_token(token) -> Node | None`, `await app.nodes.invoke(node_id, capability, params, timeout_ms) -> {ok, ...}`,
  `app.nodes.online() -> [{node_id, name, kind, capabilities}]`.
- `GET /node/` on the LAN listener and on the gateway serves the web device app (pair, camera, location, notifications, display, push-to-talk).
- `sentient node --url ... --code ... [--fingerprint FP] [--insecure] [--camera N] [--no-speech]` runs a reference node from a PC
  (webcam as camera, speaker, keyboard as button); `--url` also accepts the `sentient://pair` link.

## 14. Messaging channels (owner: channels)

- **Channel** `{id: "telegram"|"discord", display_name, status: "disconnected"|"connecting"|"connected"|"error", account_label, error, paired: [PairedChat], setup: {fields, instructions_md}}`
  - `setup.fields`: `[{key: "bot_token", label, secret: true, required: true, help, placeholder}]`; `instructions_md` is a
    step-by-step guide for non-technical people (@BotFather for Telegram, the Developer Portal for Discord).
  - `account_label`: `@botname` (Telegram) or the bot's username (Discord). `status` is `connecting` while Sentient
    reconnects at startup, `error` (with a friendly `error`) after 3 failed polls in a row or when the token was revoked.
- **PairedChat** `{chat_id, label, paired_at, deliver: bool, session_id}` (`deliver` defaults to `channels.<id>.deliver_default`).
- `GET /api/channels` → `[Channel]` (always both ids).
- `POST /api/channels/{id}/connect` `{fields: {bot_token}}` → `Channel`. The token is checked with the service (Telegram
  `getMe`, Discord `GET /users/@me`), stored in the OS keychain as `channel_<id>_token` (never in config, the database or
  logs) and polling starts. 400 invalid token or channel turned off in Settings; 404 unknown channel; 500 keychain unavailable.
- `POST /api/channels/{id}/disconnect` → `Channel` (stops receiving, deletes the token, cancels the pairing code;
  paired chats are kept so reconnecting the same bot needs no re-pairing; remove them with DELETE).
- `POST /api/channels/{id}/pairing` → `{code, expires_at, instructions}`. 6 digits, valid `channels.pairing_code_minutes`
  (10), single use, one active code per channel (a new code replaces the old one). 409 when the channel is not connected.
- `PATCH /api/channels/{id}/paired/{chat_id}` `{deliver: bool}` → `Channel` (400 when `deliver` is not a boolean, 404 unknown chat).
- `DELETE /api/channels/{id}/paired/{chat_id}` → `Channel` (stops a running reply for that chat; 404 unknown chat).
- `POST /api/channels/{id}/test` `{chat_id?}` → `{ok, error?}`: with `chat_id` sends a test message to that paired chat;
  without it re-checks the token. `ok: false` when not connected or the chat is not paired.
- Pairing: the user sends `/pair <code>` to the bot. `channels.pairing_max_attempts` (5) wrong codes cancel the active code,
  and a chat that sent that many wrong codes is ignored for the code lifetime. Messages from chats that are not paired get
  one polite refusal (once per chat, remembered across restarts) and are otherwise ignored. Telegram: private chats only
  (groups are ignored). Discord: direct messages only (server messages are ignored).
- A paired chat is a normal Sentient chat (`channel` = channel id, so the desktop shows a badge). Commands: `/new` starts a
  fresh chat (new session), `/stop` cancels the reply (the partial text is kept with "(stopped)"), `/stopall` (or
  `/stop all`) is Stop everything and `/resume` undoes it (section 17; source `telegram` or `discord`), `/help`; other
  `/commands` get a hint. Unpaired chats cannot use any command except `/pair`. Replies stream by editing the message at most once per `channels.<id>.edit_interval_s` (1 s) when
  `stream_edits` is on; long replies are split (Telegram 4096, Discord 2000 characters, code blocks kept balanced).
  Telegram replies use HTML parse mode (bold, italics, strikethrough, code, code blocks, links, lists, quotes; everything
  else escaped; plain text fallback). While tools run, a short status message ("Searching the web...") is shown and deleted
  when the answer continues (`show_tool_activity`). A typing indicator is sent while the reply runs.
- A message sent while a reply runs calls `app.agent.steer(session_id, text)` (section 10); when that returns false (or the
  message has attachments) it is queued and answered as the next turn.
- Voice notes and audio files are downloaded (20 MB max) and transcribed with `app.voice.transcribe_bytes`; the chat sees
  "Heard: ..." first. With `voice_replies` on, a voice note is also answered with `app.voice.speak` audio. Photos and
  documents are saved to `files/uploads/` and passed as attachments.
- Approvals arrive as messages with **Allow**, **Allow for this chat** and **Deny** buttons (resolving through
  `app.approvals`); the message is edited to show the answer, also when it was answered on the desktop.
- Delivery (chats with `deliver: true`, from `notification.new`): task results and failures (`payload.event` in
  `run_completed`, `run_failed`, `planning_failed`, `clarification_needed`, `disabled`; a completed run adds its result
  summary), questions from running tasks (`payload.event = "question"`, with `channels.deliver_task_results`; each
  option is a quick-reply button, callback `tq:<option index>:<run_id>`, that answers through
  `app.tasks.answer_question`), plans awaiting approval (**Approve plan** / **Decline** → `app.tasks.approve/decline`), pending proactive
  suggestions (**Approve** / **Dismiss** → same as `POST /api/proactivity/suggestions/{id}`), and notifications with
  `payload.subagent_id`. Text is a short, link-free summary. Buttons are replaced by the outcome when pressed, or when
  `notification.updated` shows the plan/suggestion was handled elsewhere. A background subagent (`subagent.updated`,
  `background: true`, `completed`/`error`) whose session belongs to a paired chat sends its summary to that chat once.
  Toggles: `channels.deliver_task_results`, `deliver_plans`, `deliver_suggestions`, `deliver_subagents`.
- Answering a task's question by replying: the delivered question ends with "Tap an option, or reply to this message
  with your answer." (without options: "Reply to this message with your answer."). The ids of the messages that carried
  the question are stored per chat in SQLite (`channel_questions`, kept 90 days), so this survives restarts. A text
  message or voice note (its transcript) without attachments that replies to one of them (Telegram
  `reply_to_message.message_id`, Discord `message_reference.message_id`) is the answer to that question: it goes to
  `app.tasks.answer_question`, the chat gets "Thanks! I passed your answer to '<task>'. It's carrying on now." and no chat
  turn starts. A reply to a question that was already answered or cancelled gets "That question has already been
  handled, so I didn't pass this on." Every other message, including replies to other messages, is normal chat, however
  many questions are waiting. Question buttons settle to "Answered: <answer>" or "Cancelled" when the question is
  handled anywhere. `Incoming.reply_to` carries the replied-to message id for every channel.
- Domain events `channel.updated` → `Channel` (connect, disconnect, status changes, pairing, deliver changes);
  `channel.message` `{channel, chat_id, session_id, direction: "in"|"out", text}`: `in` is the user's text (the transcript
  for voice notes); `out` is the final reply text, or a delivered notification with `session_id: null`.

## 15. Knowing the user (owner: memory)

### User model
- **Insight** `{id, dimension: "preferences"|"communication"|"goals"|"routines"|"relationships"|"values"|"work_style"|"dislikes"|"context", statement, confidence, status: "active"|"confirmed"|"disputed"|"retired", source: "inferred"|"user", evidence: [{kind: "fact"|"message"|"summary"|"feedback", ref, quote, at}], created_at, updated_at}`
- `GET /api/user-model` → `{summary, updated_at, insights: [Insight], questions: [{id, question, insight_id, created_at}]}`
  (`summary` is a markdown portrait of at most 180 words, `""` until the first refresh; `updated_at` is `null` until
  something changes; insights of every status, ordered confirmed, active, disputed, retired, then by confidence;
  `questions` are the open ones only)
- `POST /api/user-model/insights` `{statement, dimension}` → `Insight` (source `user`, status `confirmed`, confidence 1; 400 without statement; unknown dimensions become `context`)
- `PATCH /api/user-model/insights/{id}` `{statement?, status?}` → `Insight` (a new statement makes it source `user`,
  status `confirmed`; confirming, retiring or rewording closes its open question; 400 bad status, 404 missing);
  `DELETE /api/user-model/insights/{id}` → `{ok}` (404 missing; its questions go too)
- `POST /api/user-model/refresh` → `{added, updated, disputed, questions}` (runs now, ignoring the daily limit; no model call when there is no new evidence)
- `POST /api/user-model/questions/{id}` `{answer}` → `{ok, verdict: "confirm"|"retire"|"rewrite", insight: Insight|null}`
  (the insight is confirmed, retired or reworded as source `user`, and a fact with source `user_model` is remembered;
  400 empty answer, 404 when not open); `DELETE /api/user-model/questions/{id}` → `{ok}` (dismiss; 404 when not open)
- Refresh triggers: every `user_model.refresh_after_turns` `chat.turn_completed` events (at most every
  `user_model.min_refresh_hours`), each dream, and the REST call. Evidence is what arrived since the last refresh:
  user messages, new or changed facts and conversation summaries. Operations: `add`, `support` (+`support_step`),
  `contradict` (−`contradict_step`; below `dispute_below` → `disputed` plus a question), `retire`.
  Confirmed or user-written insights are never changed automatically: contradict/retire only queue a question.
- Engine: `await app.user_model.context_for(text) -> str` (no model call; ≤ `user_model.context_max_chars`, default 600;
  `## What I have learned about <name>` then confirmed insights, then active insights relevant to `text` by embedding
  similarity ≥ `context_min_similarity`, then active insights with confidence ≥ 0.6; low-confidence lines end in
  "(likely)"; `""` when disabled or empty). `await app.user_model.context_with_sources(text) -> (block, [Insight])` is
  the same plus the insights in the block (chat turns use it for memory sources, section 2). Tool `user_model_ask(question)` (risk read, plugin `memory`) →
  `{answer, insights: [statement], facts: [content]}` from one model call.
- Domain event `user_model.updated` `{summary_changed: bool, insights: int (insights changed), questions: int (open questions now)}`.

### Dreams (nightly consolidation)
- **Dream** `{id, started_at, finished_at, status: "running"|"completed"|"error", trigger: "schedule"|"manual", stats: {facts_reviewed, merged, contradictions_resolved, promoted, expired, insights_updated}, journal_md, error}`
  (`merged` = facts folded into another; `error` lists steps that failed while the rest completed, else `null`;
  dreams still running when the backend restarts become `error`)
- `GET /api/memories/dreams?limit=` → `[Dream]` newest first; `POST /api/memories/dreams/run` → `Dream` (status `running`;
  returns the running dream if one is already in progress); `GET /api/memories/dreams/{id}` → `Dream` (404 missing)
- Schedule: once per local day (assistant timezone) at or after `dreaming.time`, only when there was no chat turn or user
  message for `dreaming.require_idle_minutes`. Steps: merge near-duplicates (similarity ≥ `merge_similarity` and word
  overlap ≥ `merge_min_overlap`, never facts whose names/places/numbers conflict; the kept fact keeps its id), settle
  contradictions: facts are paired when they are about the same subject (name, possessive relation like "Sarthak's sister
  Riya", or the user as "I"/"the user") and share an attribute family or differ in names/places/numbers or negation;
  place, job, relationship and diet pairs are always checked, broader families (routine, ownership, health) need
  similarity ≥ `contradiction_similarity`; the model confirms and names the current fact (the newer one wins unless the
  model picks the older and only the older has explicit recent wording such as "now" or "this month"); the loser is
  deleted and its text kept as the winner's `previous_content`; pairs the model cleared are not asked again until either
  fact changes; promote short-term
  facts recalled ≥ `promote_min_recalls` times, purge expired, refresh the user model. Model calls for merging and
  contradictions are capped by `dreaming.max_model_calls`. `journal_md` is first person ("Tonight I merged 3 duplicate
  memories into 1 ..."). A notification (kind `info`, `payload.dream_id`) is sent only when something other than expiry changed.
- Domain event `dream.updated` → `Dream` (when it starts and when it finishes).
- Memory flush: `FactMemory.flush_conversation(transcript, user_name) -> [{action, id, content}]` stores durable facts from
  turns about to be compressed: one extraction call over user/assistant lines (tool and system lines dropped, bounded by
  `memory.flush_max_chars`), at most `memory.flush_max_facts` facts through the normal CUD path, `memory.updated` per change.

### Skills that repair themselves
- After a chat turn or task run that viewed a skill and then hit tool errors or a user correction, the evolution
  reviewer proposes a patch to that skill. It lands in pending review with `origin: "repair"` and a `reason` naming the failure.
  Evolution log kind `skill_repair_proposed` `{name, session_id?|task_id?, reason}`.
- Owner: evolution. Inputs: `chat.turn_completed` (`skills_viewed`, `tool_errors` as a count or a list, `user_text`, `reply`)
  and `task.run_finished` (`status`, `tool_errors`, `skills_viewed`). Failures: tool errors in the turn or run; a run with
  status `error`, `failed` or `completed_with_errors`; or the next user message in that chat correcting the reply (a phrase
  check such as "that's wrong", "didn't work", "not what I asked", "you forgot", confirmed by one `fast` call).
- Each viewed skill's use is counted in `Skill.success_count` / `failure_count` (a correction turns the earlier success into a failure).
- The proposal is the complete updated SKILL.md in `skills/pending/<name>` (`GET /api/skills/{name}/diff` compares it with the
  active skill), a `skill` notification (section 8), `skill.updated {state: "pending_review"}`, and the log entry above with
  `origin: "repair"`, `failure` and `pending: true`. In `GET /api/skills` the pending entry has `reason` and `origin.repair = true`.
- Never activated automatically, whatever `skills.write_approval` says. At most one open proposal per skill (no repair while
  any proposal for it is pending) and at most one repair per `evolution.repair_cooldown_hours`; `evolution.skill_repair: false` turns it off.
  Only skills in the writable skills folder are repaired.

## 16. Doing more on its own

### Change feeds and webhooks (owner: integrations)
- Connected Gmail and Google Calendar are watched with incremental change feeds (Gmail history, Calendar sync tokens)
  every `integrations.fast_sync_seconds` (default 60; 0 turns feeds off); this costs no model calls. IMAP email
  (`email_imap`, app password) uses IDLE push, or a NOOP check every `fast_sync_seconds` when the server lacks IDLE.
  - Gmail: the first sync takes the mailbox `historyId` (nothing emitted); later syncs emit messages added to INBOX
    (not SENT/DRAFT/SPAM/TRASH, at most 50 per sync). Expired history (HTTP 404) re-baselines from now without emitting.
  - Calendar: the first sync is a full sync for a sync token (nothing emitted); later syncs emit `new_event` (created
    within 2 minutes of its last update) or `updated_event`, skipping cancelled and already-ended events. HTTP 410
    triggers a full resync without re-emitting.
  - IMAP: the first check records the highest UID; a changed UIDVALIDITY re-baselines. Wrong app password marks the
    integration `status: "error"`.
  - Failures back off exponentially (base interval doubling, max 30 minutes) with a readable `last_error`
    (see `GET /api/integrations/feeds`); a failure never moves the cursor. Disconnecting (or connecting a different
    account) forgets the cursor.
- Every new item is published exactly once as domain event `source.items`
  `{source, event, origin: "poll"|"feed"|"webhook", items: [item]}` with the item shapes from section 5, whether it was
  found by a change feed, by `poll_source`, or arrived on a webhook. The integrations package keeps the shared
  "already seen" record so a feed and a poll never emit the same item twice.
- Consumers: the tasks package fires triggered tasks for every origin (proactivity no longer calls tasks for polled
  items). Proactivity keeps processing the items its own poll returns and also processes `feed` and `webhook` origins
  from the bus. When change feeds are active for a source, proactivity skips timer polling for it.
- Proactivity details (owner: evolution): `poll` origins on the bus are ignored (those items come back from `poll_source`).
  `feed` items are processed for sources in `proactivity.sources`; `webhook` items when `proactivity.webhook_suggestions`
  is on, unless an enabled triggered task already handles that hook, and not when the body is empty. The reasoner is told
  the payload came from the user's webhook named `<name>`. Feed state is read from `app.integrations.feed_active(source) -> bool`
  (sync or async); while it is missing every watched source is timer-polled. `POST /api/proactivity/poll-now` polls regardless.
- Deduplication key per source: gmail message id; calendar `<event id>:<updated>` (so a later change is new again);
  IMAP `Message-ID` header (else `<uidvalidity>:<uid>`); webhook calls are always new. Keys are kept 30 days.
  Items hidden by privacy filters are recorded as seen and never published.
- Webhooks: `GET /api/hooks` → `[{id, name, url, created_at, last_called_at, calls}]`;
  `POST /api/hooks` `{name}` (1-80 characters, else 400) → hook plus `secret` (shown once; only its SHA-256 is stored);
  `DELETE /api/hooks/{id}` → `{ok}` (404 unknown). `url` is the loopback gateway address the request came in on.
  `POST /hooks/{id}` (no bearer token; header `X-Sentient-Secret: <secret>` or `?secret=`, header preferred because
  query strings can end up in logs) accepts a JSON, form (urlencoded or multipart), text or empty body up to
  `integrations.webhook_max_body_kb` (default 256) → `{ok: true, item_id}`. Each hook accepts at most
  `integrations.webhook_rate_limit_per_minute` calls a minute (default 30, a token bucket per hook; 0 turns it off).
  Errors: 404 unknown hook, 401 wrong or missing secret, 429 too many calls (header `Retry-After` in whole seconds;
  checked after the secret, so wrong-secret calls never use up a hook's budget, and a refused call is not published
  or counted), 413 body too large, 400 invalid JSON with a JSON content type, 503 while Stop everything is on (header
  `Retry-After: 60`; the call is not published or counted, so the caller can send it again after Resume). It publishes `source.items`
  `{source: "webhook", event: <hook id>, origin: "webhook", items: [{id, name, body, received_at, content_type, query}]}`
  (`query` excludes `secret`; multipart files appear as `{filename, content_type, size}`).
  Triggered tasks use `schedule: {type: "triggered", source: "webhook", event: <hook id>, filter}`; the `webhook`
  integration lists one trigger per hook (section 5).

### Script jobs (owner: tasks)
- New `task_type: "script"`. Task field `script` `{code, condition: "changed"|"alert", then: "notify"|"run", last_result, last_run_at, last_error}`.
  Runs on the task's schedule (recurring, `interval`, triggered, or a one-off `run_at`) through
  `app.sandbox.run(code, channel="task", allowed_tools=[read-risk tool names])`, with no model calls. A check does not
  create a Run: it updates `script` and the task goes `active → processing → active` (next run computed as usual).
  `alert`: acts when the script calls `result({"alert": true, "message": "..."})`. `changed`: acts when the result differs
  from `last_result`; the first successful result is only the baseline, and an empty (`null`) result is never a change
  and keeps the baseline.
  The value is the script's `result(...)`, else its trimmed stdout.
  `then: "notify"` sends a `task` notification (title = task name, message = the alert `message` or a short description of the
  value, `payload: {task_id, event: "script_alert", result}`); `then: "run"` starts a normal model run whose
  `trigger_event_data` is `{script_result, message, trigger_event?}`.
- Triggered script jobs receive the event as a `trigger_event` variable (a prelude is prepended to the code).
- Failures (`ok: false`, timeouts, code execution unavailable) set `last_error` and keep `last_result`; a `script_failed`
  notification is sent on the first failure only, and a `script_recovered` notification when a later check succeeds.
- Approval: script jobs wait in `approval_pending` like plans (`tasks.require_plan_approval`); the UI shows `script.code`.
  `approve` returns 409 when the code does not compile.
- The planner may propose a script job for watch-style requests ("tell me when...", "check every hour whether...") by adding
  `script` (and a `schedule`, e.g. `interval`) to its reply. The code must use `from sentient_tools import tools, result` and
  read-only tools; it is checked with `ast.parse`, the planner gets one chance to fix code that does not compile, then
  planning fails with a readable `error`. A script job the planner leaves on an immediate one-off schedule becomes `interval` 60.
- `POST /api/tasks/{id}/script/test` → SandboxResult (does not change `last_result`); see section 4.

### Wake word and talk mode (owner: voice)
- `/ws/voice` `start` accepts `mode: "conversation"|"wake"` (default `conversation`). In `wake` mode the server sends
  `state: standby` and listens only for `voice.wake_word`; on detection it sends `{type: "wake", phrase, source: "voice", score?}`,
  the chime when `voice.wake_earcon` is on (an `audio` message with `earcon: true`), and switches to `listening`.
  After a reply, follow-ups are accepted without the wake word for `voice.follow_up_seconds` (default 8), counted from the
  estimated end of playback of the audio sent; then it returns to `standby`. The same window applies after a wake with no request.
- Speech after the phrase in the same breath is the request: "hey sentient what's the weather" wakes and asks at once
  (the phrase is removed from the `transcript`).
- `{type: "wake"}` from the client wakes without the phrase; `end_utterance` in standby processes speech without it.
- Engines (`voice.wake_engine`):
  - `whisper` (default): speech segments from the VAD are checked by transcribing their first 3 s with
    `voice.wake_whisper_model` (tiny/base, always local on the CPU) and fuzzy-matching the phrase, tolerating greeting swaps
    ("hi sentient"), split words and common mishearings ("hey sentence"). A leading greeting must be heard as a greeting.
  - `openwakeword`: streaming detector for pretrained phrases (`hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`) chosen by
    `voice.wake_model` or derived from `voice.wake_word`, or a custom `.onnx` model path in `voice.wake_model`. Model files
    download once to `~/.sentient/models/openwakeword` on first use.
  - `voice.wake_sensitivity` (0..1, default 0.5): higher wakes more easily.
  If the engine cannot be used (e.g. no pretrained model for the phrase) `ready.wake.error` is set, a recoverable
  `error` follows `state: standby`, and the session stays in standby (client `wake` still works).
- `GET /api/voice/status` adds `wake: {engine, phrase, ready, model?, error?}` (nothing is loaded by the call).
  `POST /api/voice/prepare {target: "wake"}` loads (and downloads) the wake engine.

### Domain events added
| type | data |
|---|---|
| `subagent.updated` | Subagent |
| `browser.updated`, `browser.frame` | browser status, frame |
| `node.updated`, `node.deleted`, `node.event` | see section 13 |
| `channel.updated`, `channel.message` | see section 14 |
| `user_model.updated` | see section 15 |
| `dream.updated` | Dream |
| `source.items` | see section 16 |
| `task.run_finished` | see section 10 |
| `stop.updated` | see section 17 |

## 17. Stop everything (owner: core)

One deterministic control that stops all of Sentient's work at once. It never goes through the model, so no prompt
can ignore it.

- **StopState** `{stopped: bool, stopped_at: ISODate | null, source: string | null}`. `source` says where it was last
  turned on or off: `desktop`, `tray`, `hotkey`, `telegram`, `discord` or `device`.
- `GET /api/stop` → StopState.
- `POST /api/stop-all` `{source?}` (body optional, default `desktop`) → StopState plus `cancelled` (how many running
  jobs were cancelled). Calling it again while stopped keeps `stopped_at` and `source` and cancels anything started since.
- `POST /api/resume` `{source?}` → StopState (`stopped: false`, `stopped_at: null`).
- Every change publishes domain event `stop.updated` (StopState) and `stop_state` to connected devices (section 13).
  Devices hear about a stop after the work is cancelled; each device gets at most 2 s, side by side, so a slow
  device never holds up the stop, the resume or the other devices.
  `GET /api/bootstrap` includes it as `stop`.
- Stop everything, in this order: the flag is set and saved (so nothing new starts), then every running chat reply
  (desktop, channels, voice; the partial reply is kept with "(stopped)"), task run (status `cancelled`, progress
  "Run stopped by Stop everything.", retryable from its checkpoint), task planning and check scripts, helper
  (subagent, status `cancelled`), running dream (status `error`, "Stopped by Stop everything.") and background job
  (memory notes, reviews, suggestions) is cancelled. Code runs and browser actions stop with the reply or run they
  belong to. Runs waiting for the user's answer keep waiting.
- Messages queued before the stop are dropped, never sent: steer messages the running reply had not picked up yet,
  `chat.send` turns waiting behind it on `/ws` (they end with `error` `{message: "Stopped. Your queued message wasn't
  sent.", dropped: true, client_id}` and `done` `{cancelled: true, dropped: [text], client_id}`, echoing the
  `client_id` the message was sent with so the window can mark that message, not the newest turn), and messages queued in Telegram or Discord chats (the chat gets the
  same note). The window shows steers of a cancelled reply as not sent. Messages sent after the stop work as usual.
- While stopped: the scheduler claims nothing (due tasks start on the first tick after Resume, following the missed-run
  rules in section 4), triggered tasks
  do not fire, change feeds, polls, push watchers and webhooks publish nothing (`source.items`), and proactivity,
  heartbeats, follow-ups, self-improvement reviews, the user model and dreaming do not run. Things the user starts
  directly still work: chat, Run now, answering a task's question, Dream now.
- The state is stored in the database (`meta` key `stop.state`), so a restart keeps Sentient stopped; interrupted runs
  are not resumed until Resume.
- Resume clears the flag, then resumes work a restart would resume (runs left processing, tasks left planning).
- Engine API: `await app.stop_all(source) -> dict`, `await app.resume(source) -> dict`, `app.stopped -> bool`,
  `app.stop_state`. Services implement `async halt() -> int` (cancel in-flight work) and set `pause_on_stop = True`
  so their `run_every` jobs are skipped while stopped.
- Surfaces: the desktop title bar button and banner, the tray menu, the global shortcut `Ctrl+Alt+Shift+S`
  (`Cmd+Alt+Shift+S` on macOS; stop only), `/stopall` and `/resume` in paired Telegram and Discord chats, and the
  `stop_all` / `resume` device messages (the web device app has a button).
