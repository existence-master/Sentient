# 0008. Decide approvals by each call's effective risk

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

An assistant that sends mail, clicks buttons and runs code must ask before doing irreversible things,
without nagging about harmless ones. Risk depends on arguments: clicking "Next" is harmless, clicking
"Place order" is a purchase.

## Decision

Every tool declares a `Risk` (`read`, `write`, `send`, `exec`) and may add a `risk_fn(arguments)` that
raises it per call. Approvals use the effective risk: modes off, ask, always, with "allow for this chat" never
covering a call raised to `send` or `exec`. Tools whose effect stays inside Sentient are `internal` and do not
interrupt. Safety floors are deterministic code: a model may raise a risk, never lower it. The browser refuses
password, card and one-time-code fields outright.

## Consequences

Purchases, posts and photos always ask; reading and note-taking never do. Approval cards can say what is
about to happen ("Purchase: Place order"). Every new tool must think about its risk, which reviewers check.

## Alternatives considered

Approve every tool call: safe but unusable. Let the model judge risk: unpredictable and open to prompt
injection from web pages and emails.
