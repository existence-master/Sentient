# 0027. Check skills and MCP tools from outside before the model sees them

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

A skill's steps and an MCP tool's description go straight into the model's prompt as if Sentient wrote them. Once a
skill is active, `skill_view` is Sentient's own content ([ADR 0018](0018-untrusted-content-gates-sending.md)), so it
is neither marked as outside content ([ADR 0023](0023-mark-outside-content-in-prompts.md)) nor does reading it make
sends ask. Public skill hubs were found full of malicious skills that exfiltrate data, run downloaded code or hide
instructions in invisible Unicode, and MCP servers can poison their tool descriptions or change them after the user
trusted them. A model can't be trusted to notice either, and safety is deterministic.

## Decision

- Skills from outside (Hermes imports, skills added from a link) wait in pending with a deterministic safety report:
  the linter's `danger`, `hidden` and `outbound` rules on the SKILL.md and on every text file and script in its
  folder, after removing invisible characters. A lock record (`skill_imports`) keeps the source, the folder hash, the
  scanner version, the verdict and the hash the user approved; the report is computed again from the files whenever it
  is shown or approved. A `high` verdict makes declining the default: approving needs an explicit `accept_risk`.
  Once approved, the skill counts as the user's own, which is why the check comes before approval.
- Every MCP tool is checked and hashed (name, title, description, input schema, hints) each time its server connects.
  A tool is registered only when the user approved that exact hash. New tools and changed tools are held back and
  listed with their findings; a change raises a notification with the old and new description. Findings at `error`
  level (orders about which tools to call, hidden text, instructions to ignore rules, a name that shadows one of
  Sentient's own tools) make "decline" the suggested answer. Servers configured before this change are trusted as they
  are on their first connection after it.
- Tool results stay outside content and every send gate of ADR 0018 and [ADR 0024](0024-second-look-at-outgoing-actions.md)
  still applies to approved tools. Nothing in the check lowers a risk.

## Consequences

Adding an MCP server takes one more step before its tools work, and a server that changes its descriptions on every
release asks again each time. The rules are pattern based: a careful attacker can still write instructions they miss,
so the report is a help to the user's judgement, not a guarantee. A stdio server's program has already run when its
tools are listed; the check protects the model, not the computer.

## Alternatives considered

Asking a model to judge whether a skill or description is malicious: open to the same injection. Blocking high-risk
skills outright, as Hermes does: safer, but it stops users who wrote or trust the skill, and the issue asked for the
user to decide. Letting new MCP tools in under "Ask" instead of holding them: the description is already in the
prompt before any call is asked about, which is where tool poisoning works.
