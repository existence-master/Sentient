# 0034. Serve Sentient itself to other agents over MCP, off by default and read-mostly

- **Status:** Accepted
- **Date:** 2026-10-21
- **Builds on:** [ADR 0016](0016-lasting-approval-rules.md) (lasting approval rules),
  [ADR 0018](0018-untrusted-content-gates-sending.md) (outside content gates sending),
  [ADR 0021](0021-hold-untrusted-memories-for-review.md) (untrusted memories wait for review),
  [ADR 0027](0027-vet-skills-and-mcp-tools-from-outside.md) (vet MCP tools from outside),
  [ADR 0033](0033-tasks-by-email.md) (tasks by email: outside content that always waits)

## Context

Sentient can use MCP servers, but other agents cannot use Sentient. Someone who works in Claude
Code, Cursor or another MCP client wants their assistant's memory and tasks there too, the way
Hermes and OpenClaw expose theirs. Exposing Sentient the other way round is the dangerous
direction: an MCP client is software the user did not write, driven by a model, and anything it
asks for is outside content in the sense of ADR 0018. A careless version of this feature would
hand that client Sentient's whole tool set, including sending, deleting and running commands,
with the user's lasting Allow rules and approvals mode chosen for their own requests, not for a
stranger's client.

Review of the first version of this feature settled three things it had got wrong or left open.
It signed in with the gateway token, the all-powerful per-launch token: any app holding it could
drive the whole engine, and a hand-started `sentient mcp` created a token file as a side effect.
Its chat turn let the client's text pass as the user's own words, so a client could get code
executed and memories saved directly simply by asking. And an approval a turn raised travelled
only on the chat event stream, which an MCP client discards, so the user never saw the question
it was waiting on.

## Decision

- **A separate, thin process.** `sentient mcp` (package `sentient/mcp_server/`) is an MCP server
  that holds no state and never touches the database. Every tool call goes to the running engine's
  REST API, so there is one implementation of every safety rule, in the engine, and this process
  cannot bypass it. It only talks to a loopback address and refuses any other at startup. It
  finds the engine through a discovery file the engine writes at startup (`engine-discovery.json`,
  the address it bound): the desktop starts the engine on a fresh port every launch, so the
  configured port is not where it listens. Because the file holds only an address, the server
  proves the engine's identity before it sends the token anywhere: the health probe goes out
  without credentials, and the engine must then answer a nonce challenge (`GET /api/mcp/proof`,
  the HMAC of `nonce:host:port` under the MCP token), which a process that merely copied the
  health marker cannot forge. Keying the proof to the address the request arrived on stops a
  relay: a squatter can forward the challenge to the real engine, but the proof it gets back
  is keyed to the real engine's address and fails the check (the host half matters because a
  squatter can share the port on the other loopback address, ::1 against 127.0.0.1; for the
  same reason the server only talks to literal loopback addresses and pins "localhost" to
  127.0.0.1). The proof is not
  a startup-only step: the server re-reads the discovery file and proves again before every
  call, so a running server follows the engine to its new port after a restart instead of
  sending the token to whatever took over the old one.
- **Its own token, with permissions only for these tools.** When the user turns the feature on,
  the engine issues an MCP-only token and keeps it in the keychain (`sentient.secrets`, name
  `mcp-server-token`). `sentient mcp` reads it from there itself; no client settings file ever
  holds a secret. The gateway combines both tokens on the routes these tools use: the gateway
  token passes everywhere, the MCP token only on the (method, route) pairs the eight tools call,
  and never on config, approvals, sessions, deletes or the memory review inbox. Handlers see such
  a caller as `caller == "mcp"`, distinct from the user's own surfaces. The user can replace the
  token from the Integrations card (`POST /api/mcp/token/rotate`); apps holding the old one stop
  working at once. `sentient mcp` never reads or writes the gateway token. On the same machine
  the real boundary is this narrow permission set, since a process running as the user could
  read whatever this process can read.
- **A whitelist of eight tools, nothing else.** `memory_search`, `memory_get`, `memory_add`,
  `task_create`, `task_list`, `task_status`, `notify_me` and `ask_sentient`. There is no tool
  that sends a message, deletes anything or runs a command, so a client cannot do those things
  through Sentient at all, whatever it asks. The list is fixed in code and a test fails if a
  send, delete or exec-shaped tool ever appears in it.
- **Off until the user turns it on, and checked live.** The server refuses to start while
  `integrations.mcp_server_enabled` is off, and every tool call reads `GET /api/mcp/status` from
  the running engine again, so turning it off in Settings stops a server that is already running.
  The Integrations screen shows the switch, the config snippet to paste into a client and the
  Replace key button.
- **Writes land where the user can see them, in the app's name.** A memory an MCP app adds waits
  in Review like an import's does (ADR 0021), held with the app's name, unless the user turns on
  `integrations.mcp_server_memories_skip_review`. A task is created the way a task by email is
  (ADR 0033): in the app's name, never refined, its plan always waiting for the user's approval
  even when plans normally approve themselves, and every run marked as started by that app. A
  notification is a plain info note inside Sentient; the endpoint ignores any kind a caller asks
  for, and for an MCP caller the engine puts the app's name in the note's title, so a note can
  never pass itself off as a message from Sentient itself. The name is the app's own text, so
  it is cleaned before it is shown anywhere, and where a note is rendered as Markdown it is
  put in an inline code span as well, so nothing inside it is parsed into a link or emphasis
  above an approval decision.
- **Asking Sentient is outside content from the start.** `ask_sentient` runs one turn through
  `POST /api/chat`, and an MCP caller is forced onto the `mcp` channel whatever it claims. The
  turn follows ADR 0033's email pattern: the session is marked untrusted before anything runs,
  the client's text never becomes the user's words (so it cannot clear a memory into storage or
  ground a fact as the user saying it, and turn-end extraction is told the words are not theirs),
  sends and code runs ask through the outside-content gate, and deletes ask too, by an explicit
  rule since they sit below send level. Nothing the client says can answer a question: answers
  only come from the user.
- **Its approvals reach the user.** An approval raised in such a turn is also posted as a
  `kind: "approval"` notification carrying the same `approval_id`; the notification center and
  the paired messaging chats render Allow and Deny for it, and any answer settles the one broker
  the waiting turn is blocked on. When the answer lands, the loop fills in the decision and marks
  the note read, so it never waits forever.
- **stdio only.** No HTTP transport ships with this feature. stdio needs no listener and no
  address to keep current, and it is what the clients this feature targets already use. HTTP may
  return later as its own change, protected by the MCP-only token rather than the gateway token.

## Consequences

Other agents can finally read and add to the same memory and task list the user sees, without a
second copy of anything. The price is a few small engine additions (a hold-for-review flag on
`POST /api/memories`, a create route for notifications, the `mcp` token and status endpoints, the
loop marking and the approval notification) and one more surface to keep honest: every future
tool proposed for this server has to argue why it belongs on the whitelist, and the default
answer for anything that sends, deletes or runs is no. Because the MCP process reads the status
from the engine on every call, a running server adds one small local request per tool call,
which is fine at this scale.

## Alternatives considered

Letting the MCP process open the database directly: faster, but it would copy the memory and
task logic and quietly bypass approvals, review and the plan gate, which is the whole point of
going through the engine. Exposing Sentient's full internal tool registry: the registry is built
for a model the user is talking to, with the user's own approvals behind it; most of it (sending,
deleting, the terminal, the browser) must never be offered to an outside client. Signing in with
the gateway token: it exists to keep other pages and processes out, not to be handed to apps, and
it opens everything; a dedicated token with an allowlist states exactly what this feature may
touch. Saving MCP memories directly: rejected for the reason in ADR 0021, memory is how an
injection becomes permanent. An HTTP transport guarded by the gateway token, reached through a
launcher that injects it: workable, and an earlier version of this change shipped it, but it
spreads the all-powerful token into a second process for no user-visible gain over stdio;
removed before the feature landed, to return only behind the MCP-only token.
