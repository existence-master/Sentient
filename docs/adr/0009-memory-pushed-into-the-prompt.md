# 0009. Push memory into every turn and consolidate it nightly

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

v2 relied on the model deciding to call a memory tool before it knew anything about the user, so it often
answered as a stranger. Memory also accumulates duplicates and contradictions ("lives in Pune" and later
"moved to Bengaluru").

## Decision

Each turn's system prompt carries the persona, the user profile, recalled facts, the user model's relevant
insights and a running summary of long chats. Facts go through an add, update, delete or skip decision against
similar facts, with code-level guards (an update may not drop a name, place or number). A nightly consolidation
merges duplicates, settles contradictions and writes a journal the user can read. The user can see, correct and
delete everything on the Memory and About you pages.

## Consequences

Sentient feels like it knows you from the first message. Prompts get longer, so recall is budgeted and
ranked. Model decisions about memory are guarded by deterministic checks because small models make mistakes.

## Alternatives considered

Pure tool-based recall: cheaper prompts, but the model forgets to look. Unlimited raw history: does not fit
small context windows.
