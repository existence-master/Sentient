# 0007. Use one agent loop for every surface

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Chat, task runs, swarm workers, helper agents, proactive context gathering and voice all need the same
thing: stream a model, run tools, ask for approval, record usage. v2 had several slightly different loops, and
fixes landed in one but not the others.

## Decision

`Agent.run_loop` is the single tool-calling engine. It streams typed events, runs each tool in its own task
so `ctx.progress` can stream output, computes each call's effective risk
([ADR 0008](0008-effective-risk-and-approvals.md)), runs approval-free read calls concurrently, cuts oversized
results, and applies steering messages at round boundaries. Surfaces differ only in the messages, tools, role
and policy they pass in.

## Consequences

A fix or improvement to the loop reaches every surface at once, and the robustness work for small models
(empty-answer nudges, malformed-argument recovery) lives in one file. The loop is a hot spot that needs strong
tests; it has them in `tests/test_agent_*.py`.

## Alternatives considered

A framework such as LangGraph: powerful, but adds abstraction exactly where we need tight control over
streaming, approvals and local-model quirks.
