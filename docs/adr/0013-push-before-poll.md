# 0013. Prefer change feeds and scripts over model calls

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Proactivity and triggered tasks used to poll apps on a timer and ask a model about every item. On a laptop
that wastes battery and model time, and reacts slowly.

## Decision

Gmail and Calendar are watched with incremental change feeds, IMAP mail with IDLE push, and any app can call
a webhook. Every new item is published once as `source.items`, whatever its origin. Watcher tasks run small
scripts on a schedule and wake the model only when the script reports a change or an alert.

## Consequences

Sentient reacts within a minute with no model calls for routine checks. Integrations must maintain cursors
and deduplicate items across feeds, polls and webhooks, which the integrations package owns.

## Alternatives considered

Faster polling: more load, still slow. Model calls for every item: accurate but expensive on local
hardware.
