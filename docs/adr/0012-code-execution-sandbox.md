# 0012. Run model-written code in a sandbox that calls tools

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Small local models are slow per round and weak at long tool chains. Letting the model write one short
script that calls many tools turns a dozen rounds into one, but running model-written code is dangerous.

## Decision

`execute_code` runs Python in an isolated working folder with a scrubbed environment (no API keys), a
timeout that kills the whole process tree, and output caps. The script calls Sentient tools through a one-time
local bridge that allows only read and internal tools. The default backend is a separate local process; Docker
is used when it is running Linux containers. Running code always asks for approval.

## Consequences

Data work and repetitive lookups become fast and cheap. The sandbox is not a security boundary against a
determined attacker on the same machine, which is why approval is required and keys are withheld.

## Alternatives considered

No code execution: loses a large capability for local models. Docker only: most people do not run it.
