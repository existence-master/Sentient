# 0018. Ask before sending anything once untrusted content is in play

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

Private data, untrusted content and a way to send data out together make the "lethal trifecta": an email or web
page can carry instructions that make the assistant forward the inbox somewhere. An OpenClaw user showed this by
emailing himself injected instructions. Sentient has all three: it reads mail, the web and messages, it knows the
user's data, and it can send. Lasting Allow rules ([ADR 0016](0016-lasting-approval-rules.md)) and approvals mode
"off" were chosen by people thinking of their own requests, not of a stranger's email steering a chat. A model cannot
be trusted to notice an injection, so the defence has to be in code.

## Decision

Tools carry two tags next to their `Risk` ([ADR 0008](0008-effective-risk-and-approvals.md)):

- `untrusted_output`: the result can carry content someone else wrote. Default: look-ups (base risk `read`) of every
  app except Sentient's own (`memory`, `files`, `skills`, `time`, `tasks`, `subagents`, `devices`) and plain data
  services (`weather`, `charts`). Browser tools, code runs, MCP server tools, subagent results, device photos and
  screenshots are marked explicitly. Writes and sends return confirmations, so they do not count.
- `exfiltrates`: the call can move data out even below `send`: typing into a web page, editing a GitHub issue, any
  change on an MCP server.

A run is marked (`ToolContext.untrusted`, the app's name) when a tool with untrusted output actually ran. The
event that started a triggered task (an email, a webhook body) marks its run from the start, and a resumed task run
is marked again from its transcript. From then on every call whose effective risk is `send` or `exec`, or that
`exfiltrates`, asks the user, even under an Allow rule, with approvals mode "off" and after "Allow for this chat".
Calls the model chose in the same round as the read are not affected: they were decided before the content
arrived. The approval card says why ("Sentient read content from Gmail in this chat, so it checks with you before
sending anything.") and offers no "Allow for this chat". "Never" rules and the unprompted-work limit
([ADR 0017](0017-unprompted-work-reads-only.md)) still come first.

How long the mark lasts:

- **Chat:** the rest of the chat. It is saved on the session (`sessions.untrusted`), because the content stays in
  the conversation the model reads in later turns, also after a restart. A new chat starts clean.
- **Task run:** the rest of the run. A run cannot ask in-line, so the call is held and the run pauses with the
  existing ask-the-user machinery (#121): "This task read content from Gmail, so it checks with you before anything
  leaves Sentient. OK to use "Send" in Gmail (to: ...)?" with "Yes, go ahead" and "No, stop the task". Yes runs
  exactly that held call once and the run carries on; any other answer fails the run.
- **Subagents** inherit their parent's mark; swarm workers and subagents cannot ask, so held calls are refused
  with a plain message.

Nothing a model says can clear the mark or lower a call below asking. The user's own yes to one call is the only way
through.

## Consequences

A chat or task that read outside content can no longer send, post, submit, share or run code without the user
seeing exactly what will happen. Chats and tasks that never touched outside content behave as before. People who
rely on Allow rules see more questions after reading mail or the web. Every new tool must think about whether its
result is outside content and whether it can move data out; new apps get the safe default.

## Alternatives considered

Letting the model judge whether content looks malicious: unreliable and itself open to injection. Clearing the mark
at the end of each chat turn: simpler, but the email is still in the conversation the next turn reads. Gating every
outward request, including plain page loads whose address could carry data: closes more channels but would ask on
nearly every browse; left out for now. Failing task runs instead of asking: safe, but it breaks useful triggered
tasks such as "reply to new mail from my team".
