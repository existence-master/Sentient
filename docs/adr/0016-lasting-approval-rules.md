# 0016. Let people set lasting Allow, Ask or Never rules per app and tool

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

[ADR 0008](0008-effective-risk-and-approvals.md) decides approvals by each call's effective risk, with one global
mode and "Allow for this chat". People want rules that last: "always draft Gmail replies without asking", "always
ask before posting to Slack", "never delete files". A per-chat answer has to be given again in every chat, and the
global mode is too coarse. Any such rule must stay deterministic: a model, a web page or an email must never be able
to loosen it.

## Decision

Add `tools.approvals.rules`, a map from a key to `allow`, `ask` or `never`, saved in the config file. A key is a tool
name or a plugin id (every tool of that app); a tool's own rule beats its app's rule. The broker, the agent loop and
the sandbox bridge apply rules in code before a call runs; no model sees or decides them. This extends ADR 0008:

- **Never**: the registry stops offering the tool to the model on every surface (chat, voice, channels, tasks,
  subagents, proactivity, scripts) and planners do not list it. A call made anyway is refused with a plain sentence
  that points to Settings, and the tool does not run.
- **Ask**: always ask, even when the global mode is off, after "Allow for this chat" and for look-ups. Where nobody
  can be asked the call never runs unasked: a task run stops and fails with a plain reason ("Slack is set to Ask, and
  tasks can't ask yet. Change it in Settings > Approvals & safety.") and the usual "Task failed" notification;
  subagents and scripts refuse the call; proactive look-ups leave the tool out. Making tasks pause and ask instead
  is a planned follow-up.
- **Allow**: run without asking, whatever the mode. A purchase (effective risk `send` or higher whose approval
  wording is "Purchase", such as a browser click on "Place order") still asks whenever approvals are on, and
  "Allow for this chat" never covers it. With approvals off nothing asks, as before. Allow never widens what
  scripts from code execution may call: they keep [ADR 0012](0012-code-execution-sandbox.md)'s read-only rule,
  as docs/PRIVACY.md promises.

Rules only ever add questions or remove tools, except for Allow, which the person sets for a named tool or app
themselves, and which still never reaches purchases or scripts.

## Consequences

People can tune Sentient once instead of answering the same question in every chat. A "never" rule also shrinks the
tool list, which helps small local models. Every new place that runs tools must go through `run_loop` or apply
`sentient.tools.rules` itself, and every new money-moving tool must label its approval "Purchase" so Allow cannot
skip it. A task run that reaches an "ask" tool now stops with a clear message until the rule changes or tasks
learn to pause and ask.

## Alternatives considered

Pausing background tasks until someone answers: more machinery than this change needs, so it is left for a
follow-up. Letting Allow open scripts to sending tools: convenient for bulk work, but it would break the promise
that code Sentient writes can only read. Letting the model pick rules from chat ("stop asking me about Gmail"): convenient, but it would
let prompt injection loosen safety. Treating every `send` as un-allowable: safe, but then "always send my Gmail
drafts without asking" could not be expressed.
