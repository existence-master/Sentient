# 0032. Let people opt in to using their own Claude Code for everything

- **Status:** Accepted (amends [ADR 0022](0022-claude-through-your-own-claude-code.md))
- **Date:** 2026-10-17

## Context

[ADR 0022](0022-claude-through-your-own-claude-code.md) lets chats use Claude through the user's own installed and
signed-in Claude Code, and refuses everything else: tasks, subagents, proactivity, heartbeats, follow-ups, dreaming,
briefs, memory notes, titles and summaries. It gave two reasons: plan usage "is designed to support ordinary use of
native Anthropic applications", and background work would run without anyone watching.

People who pay for a Claude plan keep asking to use it for all of Sentient, not only for chats. On 2026-10-17 the
owner decided (#444) that Sentient offers two ways to use Claude:

1. An Anthropic API key, for everything. A Max or Team plan's monthly API credits come through such a key, from the
   Claude Console organization the plan is linked to. This is the supported route.
2. The user's own Claude Code: chats by default, as in ADR 0022, plus an opt-in to use it for everything, behind a
   stronger warning.

Anthropic's rules have not changed since ADR 0022. Its help article
([Log in to your Claude account](https://support.claude.com/en/articles/13189465)) prefers an API key for third-party
software, may charge such use to extra usage credits, and does not allow tools that route third-party traffic against
plan limits. There is still no "Sign in with Claude" for third-party apps unless Anthropic approved the app, and
Sentient still never reads or stores Claude credentials.

## Decision

Add `models.claude_code_everywhere` (default off). It only counts while `models.experimental_claude_code` is on.

- **The user accepts the risk explicitly.** Settings turns it on only after a separate dialog that says plainly that
  Anthropic says Claude plans are for its own apps, that other apps running work on the plan may be limited, charged as
  extra usage or not allowed, that Sentient will then run tasks, memory and background checks through Claude Code
  while the user is away, and that this uses plan limits much faster. It links Anthropic's article and needs a
  checkbox before the Turn on button works. Settings shows "Chats only" or "Everything" next to the switch.
- **Every role but embeddings.** When on, the Claude Code provider accepts unattended work: planner, executor,
  fast/background, subagents, proactivity, heartbeats, follow-ups, dreaming, briefs, titles, summaries and the
  rules-from-chat check. Text and JSON jobs collect a Claude Code reply without tools. Claude Code still makes no
  embeddings. `GET /api/models/claude-code` reports the roles that accept it (`roles`), which the window uses to decide
  where to offer Claude Code (#442).
- **A cap on processes.** At most `models.claude_code_max_concurrent` (default 2) Claude Code processes run at once.
  The rest wait in a queue, a reply someone is waiting for first, then tasks, then Sentient's own upkeep. The log says
  when a call is queued and when it starts.
- **Turning it off moves the roles back.** Each role still on `claude-code/*` that no longer accepts it moves to its
  first other model (its fallbacks, the main model's fallbacks, the main model). A role with no other model stays and
  is refused as before. One notification says what moved.
- **Usage by role.** The cost meter lists Claude Code calls by role, so people can see what used their plan.
- **Nothing else changes from ADR 0022.** Claude Code only writes the reply; every tool call goes through Sentient's
  one agent loop, approvals, rules and outside-content checks. Stop everything kills every Claude Code process and now
  also drops the calls waiting in the queue.

## Consequences

People can run all of Sentient on their own Claude plan if they choose to, with the risk stated where they decide.
In exchange:

- Their plan limits run out much faster, and Anthropic may limit, charge or block this use of their account. That
  risk is the user's, and Sentient says so; the API key and the plan's API credits remain the recommended routes.
- Background work through Claude Code starts a process per call, which is slower than an API call; the cap keeps it
  from flooding the computer or the plan, and chats go first.
- Claude Code has no JSON mode, so JSON jobs rely on the tolerant parser.
- If Anthropic changes how this is counted or allowed, the switch, its warning and this record are where to change
  course. Turning it off needs no other change.

## Alternatives considered

Keeping chats only (ADR 0022): safest for the user's account, but the owner decided people may take the risk with a
clear warning. Turning it on by default, or with the same light warning as chats: the risk to the account is bigger
and should be a separate, explicit choice. Claude Code for background work with no cap: a burst of upkeep could start
many processes and use up the plan in minutes. Reading Claude Code's login to call the API directly: still forbidden
by Anthropic's terms, as in ADR 0022.
