# 0017. Work nobody asked for can only read

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

Sentient does some work on its own: proactive checks of new mail and events, the heartbeat, the daily follow-up
scan, nightly dreaming, and any subagent those start. Unprompted work is where assistants surprise people: an email
nobody asked for, a booking with a cancellation fee. Until now the proactive look-ups were read-only only because
their caller offered read tools; nothing in the loop guaranteed it, and lasting Allow rules
([ADR 0016](0016-lasting-approval-rules.md)) or a tool whose `risk_fn` raises the risk could slip through. A subagent
started from such work did not know where it came from.

## Decision

Every run has an origin on its `ToolContext` (`origin`): `"user"` for work someone asked for (a chat, a task the user
created or approved, a suggestion the user accepted) and `"proactive"`, `"heartbeat"`, `"followups"`, `"dreaming"` or
`"background"` for work nobody asked for. `Agent.tool_context(..., "proactive")` sets it from the channel, and a
`run_loop` whose `source` is one of those names counts as unprompted too, so either signal is enough.

In an unprompted run the agent loop allows a call only when its effective risk is `read`, or it is an internal
`write` (Sentient's own memory, notes, files folder, skills for review, a subagent). Everything else (sending,
deleting, buying, running code, changing anything outside Sentient, and creating or changing tasks, which later act
on their own) is refused in code before approvals, modes or Allow rules are looked at, with a plain message telling
the model to suggest it instead. The refused calls are kept on `LoopResult.held`; the proactive pipeline passes them
to the reasoner, so they can come back as a suggestion card the user approves. Subagents inherit the origin of the
run that started them.

## Consequences

The rule holds whatever the settings say, and new background jobs get it by naming their channel or source. A
proactive check can still read mail, calendars and the web. Anything else it wants to do is not done; the reasoner
may offer it as a suggestion, and approving one starts a user-origin task, so approved work is unaffected. A future background job that needs to
act must go through a suggestion or a task the user approves.

## Alternatives considered

Relying on each caller to offer only read tools: that is what we had, and one wrong list or one `risk_fn` breaks it.
Letting an Allow rule lift the limit: convenient, but people set Allow while thinking of their own requests, not of
work Sentient starts by itself. Letting triggered tasks opt in to one declared action: useful later, but triggered
tasks are already created and approved by the user, so they keep today's behaviour.
