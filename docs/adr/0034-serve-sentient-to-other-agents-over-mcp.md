# 0034. Serve Sentient itself to other agents over MCP, off by default and read-mostly

- **Status:** Accepted
- **Date:** 2026-10-21
- **Builds on:** [ADR 0016](0016-lasting-approval-rules.md) (lasting approval rules),
  [ADR 0018](0018-untrusted-content-gates-sending.md) (outside content gates sending),
  [ADR 0021](0021-hold-untrusted-memories-for-review.md) (untrusted memories wait for review),
  [ADR 0027](0027-vet-skills-and-mcp-tools-from-outside.md) (vet MCP tools from outside)

## Context

Sentient can use MCP servers, but other agents cannot use Sentient. Someone who works in Claude
Code, Cursor or another MCP client wants their assistant's memory and tasks there too, the way
Hermes and OpenClaw expose theirs. Exposing Sentient the other way round is the dangerous
direction: an MCP client is software the user did not write, driven by a model, and anything it
asks for is outside content in the sense of ADR 0018. A careless version of this feature would
hand that client Sentient's whole tool set, including sending, deleting and running commands,
with the user's lasting Allow rules and approvals mode chosen for their own requests, not for a
stranger's client.

## Decision

- **A separate, thin process.** `sentient mcp` (package `sentient/mcp_server/`) is an MCP server
  that holds no state and never touches the database. Every tool call goes to the running engine's
  REST API with the engine token, so there is one implementation of every safety rule, in the
  engine, and this process cannot bypass it. It only talks to a loopback address and refuses any
  other at startup.
- **A whitelist of eight tools, nothing else.** `memory_search`, `memory_get`, `memory_add`,
  `task_create`, `task_list`, `task_status`, `notify_me` and `ask_sentient`. There is no tool
  that sends a message, deletes anything or runs a command, so a client cannot do those things
  through Sentient at all, whatever it asks. The list is fixed in code and a test fails if a
  send, delete or exec-shaped tool ever appears in it.
- **Off until the user turns it on, and checked live.** The server refuses to start while
  `integrations.mcp_server_enabled` is off, and every tool call reads the switch from the running
  engine again, so turning it off in Settings stops a server that is already running. The
  Integrations screen shows the switch and the config snippet to paste into a client
  (`claude mcp add sentient -- sentient mcp`).
- **Writes land where the user can see them.** A memory an MCP app adds waits in Review like an
  import's does (ADR 0021), held with a note saying it came from an MCP app, unless the user
  turns on `integrations.mcp_server_memories_skip_review`. A task is created the normal way, so
  its plan waits for the user's approval in Sentient before anything runs. A notification is a
  plain info note inside Sentient; the endpoint that creates it ignores any kind a caller asks
  for, so a client cannot dress a note up as an approval or a result.
- **Asking Sentient is a normal chat turn.** `ask_sentient` runs one turn through `POST /api/chat`
  on its own channel. The turn's tools run under Sentient's own approvals, rules (ADR 0016) and
  outside-content checks (ADR 0018): anything risky asks the user in the Sentient window, and
  nothing the MCP client says can answer that question, because answers only come from the user
  in the app or a paired chat, never from a model or a tool result.
- **stdio first, HTTP only on this computer.** stdio is the default transport. Streamable HTTP
  (`--transport http`) binds 127.0.0.1 only, for clients that cannot start a local command.

## Consequences

Other agents can finally read and add to the same memory and task list the user sees, without a
second copy of anything. The price is two small engine additions (a hold-for-review flag on
`POST /api/memories` and a create route for notifications) and one more surface to keep honest:
every future tool proposed for this server has to argue why it belongs on the whitelist, and the
default answer for anything that sends, deletes or runs is no. Because the MCP process reads the
switch from the engine on every call, a running server adds one small local request per tool
call, which is fine at this scale.

## Alternatives considered

Letting the MCP process open the database directly: faster, but it would copy the memory and
task logic and quietly bypass approvals, review and the plan gate, which is the whole point of
going through the engine. Exposing Sentient's full internal tool registry: the registry is built
for a model the user is talking to, with the user's own approvals behind it; most of it (sending,
deleting, the terminal, the browser) must never be offered to an outside client. Serving over
HTTP only, with its own password: a second credential to manage and a network-shaped surface,
when stdio needs neither and the engine token already exists. Saving MCP memories directly:
rejected for the reason in ADR 0021, memory is how an injection becomes permanent.
