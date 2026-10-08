# 0010. Let Sentient write its own skills, with human review

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

Inspired by Hermes Agent, Sentient can learn reusable procedures from successful work. Unreviewed
self-modification is risky: a bad skill silently changes future behaviour.

## Decision

After substantial chats and task runs, a reviewer proposes skills (`SKILL.md` with When to use, Procedure,
Pitfalls and Verification). When a skill was followed by errors or a user correction, it proposes a repair.
Proposals wait in Skills → Pending review with a diff and never activate on their own. A curator retires unused
skills.

## Consequences

Sentient improves with use while the user stays in control. Review adds a step, kept light by good diffs
and plain-language reasons.

## Alternatives considered

Auto-activate skills: faster learning, unsafe. No self-evolution: a static assistant.
