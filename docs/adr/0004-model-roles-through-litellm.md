# 0004. Let the user pick a model for every job

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

People run Sentient on very different hardware and budgets: an 8 GB laptop GPU with Ollama, a Mac with LM
Studio, or a cloud key. Different jobs need different models: chat needs tool calling, background extraction
needs speed, embeddings need a dedicated model.

## Decision

Model access goes through LiteLLM behind *roles*: `primary`, `fast`, `planner`, `executor`, `embedding`,
`vision` and `voice`. Each role maps to a `provider/model` string with an optional fallback chain, reasoning
effort and temperature, all editable in Settings → Models with a one-click test that also checks tool calling.
Code never names a model outside the defaults in `config/schema.py`.

## Consequences

Any provider LiteLLM supports works, local and cloud can be mixed, and a weak local model can be paired with
a fast one for background work. Small local models are first-class users, which shapes prompts and parsing
across the codebase (short prompts, tolerant JSON, nudges). Provider quirks (for example Ollama's thinking flag)
are handled in one place.

## Alternatives considered

One global model: simplest, but wastes a big model on background jobs or starves chat with a small one.
Provider SDKs directly: more control, many more code paths.
