# 0001. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

Sentient v3 was a rewrite with many deliberate breaks from v2, built quickly by several people and coding
agents in parallel. Without a record, the reasons behind choices like "one SQLite file" or "no `.env`" get lost,
and contributors either re-argue them or quietly undo them.

## Decision

We keep short architecture decision records in `docs/adr/`, numbered in order, one decision per file,
using [the template](0000-template.md). A decision is recorded when it is hard to reverse, crosses areas, or
explains a rule in `AGENTS.md`. Records are never deleted: a changed decision gets a new record that supersedes
the old one.

## Consequences

New contributors can find out *why* without asking. Pull requests that change one of these decisions must
add a new ADR, which makes the trade-off explicit in review.

## Alternatives considered

Design notes inside `ARCHITECTURE.md` only: good for the current shape, poor at keeping history and
reasons. Wiki pages: drift from the code and are not reviewed in pull requests.
