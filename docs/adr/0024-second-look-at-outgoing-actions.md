# 0024. Take a second look at outgoing actions before they run

- **Status:** Accepted
- **Date:** 2026-10-13
- **Amended by:** [ADR 0029](0029-group-chats-and-trusting-a-chat-again.md) (marked text inside a user turn is outside content)

## Context

[ADR 0018](0018-untrusted-content-gates-sending.md) makes every send ask once a run has read outside content, but
the card only says that the chat read a web page; it does not say that this email goes to an address the page
suggested. Outside that case, a send runs without a question under approvals mode "off", an Allow rule or a grant
([ADR 0016](0016-lasting-approval-rules.md)), and a small model can still make up a recipient or an amount (in a
qwen3:8b check of follow-ups, one draft agreed to an invoice split on the user's behalf). OpenAI's agent runs an
auto-review on such actions that cannot be turned off. Sentient's rule is that safety is deterministic and a model
may raise a risk, never lower it ([ADR 0008](0008-effective-risk-and-approvals.md)).

## Decision

Before a call that can send data out (`rules.sends_out`) runs, the agent loop takes a second look
(`sentient.agent.second_look`), after "never" rules, Read only connections and the unprompted-work limit:

- **Deterministic, always:** every recipient (email address or phone number in a recipient field) must be in the
  user's own words, in trusted records (Sentient's own tool results, Google Contacts, saved memories) or belong to a
  name the user gave (marked outside content inside a user turn, such as a group member's message, is not the user's
  words, [ADR 0029](0029-group-chats-and-trusting-a-chat-again.md)); every amount must be one the user gave; after outside content, a link must not come from that
  content. Not run on code (`exec`).
- **Model, only for `send` and `exec`:** when the call would otherwise run without a question, or the run read
  outside content, one short, tool-free `fast`-role prompt sees only the user's words and the action's arguments,
  never the outside content, and answers whether they match. No usable answer fails closed. It can be turned off
  (`tools.approvals.second_look_model`); the deterministic part cannot.

A flagged call is raised to at least `send` and asks, with the reason on the approval card ("You asked to email
Arjun, but this goes to x@evil.example."), whatever the approvals mode, Allow rules, "Allow for this chat" or grants
say, and offers only "Allow once". Task runs pause and ask; subagents and swarm workers refuse. This adds one item
to ADR 0016's list of what still asks with approvals "off".

## Consequences

A steered or made-up recipient, amount or link is shown to the user before anything leaves Sentient, with a reason
they can act on. Sends that match the request run as before; the model check adds one short fast-model call only to
sends that would otherwise not ask, or that follow outside content, and none when the user is about to be asked in
a clean chat. A send to someone the user didn't name and Sentient doesn't know (the user's own address when it is in
no memory, a reply to a sender whose address doesn't match their name) now asks once. A slow or missing fast model
makes such sends ask instead of running. With Claude Code as the fast model, background task sends use the primary
model for the check, or ask.

## Alternatives considered

Showing the outside content to the checking model: it is exactly what can steer it. Letting the check approve calls
that would otherwise ask: a model would lower a risk. Checking every write: drafts and notes stay inside the user's
accounts, so only calls that can send data out are checked. Matching any word of the request against an address:
"email Arjun the notes" would let `notes@evil.example` through, so only names count.
