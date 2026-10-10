# 0022. Let chats use Claude through the user's own Claude Code, off by default

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

People with a Claude Pro or Max plan ask to use it in Sentient. Hermes and OpenClaw offer a backend that runs the
user's installed `claude` program (Claude Code) instead of calling the API. Anthropic's rules, as of October 2026:

- Apps may not offer Claude.ai sign-in or route requests through a user's plan, and may not "collect, store, or
  intermediate" Claude.ai credentials or tokens. Developers building on the Agent SDK should use API keys
  ([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance),
  [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)).
- This does not stop an end user from signing in to the unmodified Claude Code binary with their own plan.
- `claude -p`, the Agent SDK and third-party apps still draw on plan limits. A change to Agent SDK usage was announced
  and then paused on 2026-06-15, and on 2026-10-07 Max and Team plans gained monthly API credits
  ([Use the Claude Agent SDK with your Claude plan](https://support.claude.com/en/articles/15036540), checked
  2026-10-10). Those credits cover `claude -p` only when it runs with an API key; signed in with a plan, it draws on
  the plan's limits and never on the credits
  ([Monthly API credits for Max and Team plans](https://support.claude.com/en/articles/17154008)).
- Plan usage "is designed to support ordinary use of native Anthropic applications". For third-party software the
  preferred route is an API key; Anthropic may allow certain third-party tools for subscribers who turned on usage
  credits and "reserves the right" to charge their use to those credits instead of plan limits; and tools that route
  third-party traffic against plan limits are not allowed
  ([Log in to your Claude account](https://support.claude.com/en/articles/13189465), 2026-05-19).
- In April 2026 Anthropic emailed subscribers that third-party harnesses using plan logins (OpenClaw first) would
  draw on paid extra usage instead of plan limits. That email was reported in the press; no current help article
  repeats it, and the login article above is the standing rule.

So the one possible route is the user running their own Claude Code, and how Anthropic counts or allows that may
change. Claude Code is also an agent with its own tools (shell, file edits, web fetch) that would bypass Sentient's
approvals, rules ([ADR 0016](0016-lasting-approval-rules.md)), outside-content checks
([ADR 0018](0018-untrusted-content-gates-sending.md)) and Stop everything.

## Decision

Add an experimental model provider, `claude-code/<model>` (`claude-code/sonnet`, `claude-code/opus`), outside
LiteLLM ([ADR 0004](0004-model-roles-through-litellm.md)), behind `models.experimental_claude_code` (default off) with
a Settings switch and the warning "Experimental. Uses your own Claude Code install and login. Anthropic may change how
this is counted or allowed."

- **The user's own install and login.** Sentient finds `claude` on PATH and starts it per reply in print mode with
  streaming JSON. It never reads, copies or stores Claude credentials or files under `~/.claude`, never offers a
  Claude sign-in, and never names Sentient "Claude Code". Claude Code prefers other credentials over the plan login
  ([authentication precedence](https://code.claude.com/docs/en/authentication)), so Sentient leaves every one of them
  out of its environment: every `ANTHROPIC_*` variable (API key, bearer token, base URL, profile, federation),
  `CLAUDE_CODE_OAUTH_*` (a `setup-token`), the `CLAUDE_CODE_USE_*` cloud switches (Bedrock, Vertex, Foundry),
  `CLAUDE_CODE_SIMPLE` (bare mode, which never reads the login) and a parent Claude Code session's
  `CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH`. `CLAUDE_CONFIG_DIR`, which picks which of the user's
  own logins to use, is kept. The user's Claude Code settings are not loaded either, so an `apiKeyHelper` there is not
  used. Settings shows `claude --version` while the switch is on;
  whether it is signed in is only checked when the user presses Test, which is one small reply.
- **Claude Code only writes the reply.** It starts with `--tools ""` (none of its own tools), the dangerous built-ins
  also denied by name, `--permission-mode dontAsk`, no user settings, hooks, plugins, slash commands or other MCP
  servers, no saved session, one turn, Sentient's system prompt in place of its own, and a scratch folder under
  Sentient's home. Sentient's tools are offered through a local stdio MCP server that lists them and never runs one.
  A tool Claude asks for comes back as an ordinary tool call, and the one agent loop ([ADR 0007](0007-one-agent-loop.md))
  runs it with every check Sentient has. Before reading anything else, Sentient checks the tool list Claude Code
  reports; if any of its own tools are still there (other than EndConversation and ToolSearch, which can't act), or it
  doesn't report one, Sentient kills it and says why.
- **Chats only.** Only a chat reply (`run_loop` with source `chat`, on any surface) or the Test button may use it.
  Tasks, subagents, proactivity, heartbeats, follow-ups, dreaming, briefs, memory notes, titles and summaries are
  refused with a plain message, so the role's fallback model takes over. It makes no embeddings. The check-up never
  calls it.
- **Stop everything** kills every running Claude Code process tree (a Windows job object or a POSIX process group).

## Consequences

People can chat with Claude on their own plan without Sentient touching their login, and every action still goes
through Sentient's safety rules. In exchange: Claude Code adds start-up time to each reply, the conversation is sent
as a transcript each time (Claude Code caches it), tool calls arrive after the model's turn ends, there is no
`tool_choice` or token limit, and background roles need another model. A change in Claude Code's flags or output can
break it; the tool check makes such a change fail closed. If Anthropic changes how this is counted or allowed, the
switch and this ADR are where to change course.

## Alternatives considered

Reading `~/.claude/.credentials.json` or a `setup-token` and calling the API with Claude Code headers (Hermes's native
path): forbidden by Anthropic's terms, and since April 2026 billed as extra usage anyway. The Claude Agent SDK for Python: it drives the
same binary, but adds a dependency and its in-process MCP tools would run inside Claude Code's loop instead of
Sentient's. Letting Claude Code run Sentient's tools through an MCP server that calls back into the engine: a second
tool loop to keep in step with approvals, rules and outside-content checks. Allowing it for background work: it would
go beyond the "ordinary use" plans are meant for and run without anyone watching. A Claude Max plan's monthly API credits through
an ordinary API key remain the fully supported way to use a plan.
