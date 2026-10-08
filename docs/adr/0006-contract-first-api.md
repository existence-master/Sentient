# 0006. Define the desktop and engine contract first

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

The window and the engine were built in parallel by different people and agents. Without a shared
contract, field names and event shapes drift and bugs appear only at runtime.

## Decision

[`docs/API.md`](../API.md) is the contract for every REST endpoint, WebSocket message and domain event. A
change to a shape updates the contract in the same pull request. Domain events use a dotted `type` with the
payload in `data`; chat events stream on the same socket.

## Consequences

Both sides can be built and tested independently against the document. Reviewers check the contract diff
instead of reverse-engineering behaviour. The cost is discipline: the document must be kept true.

## Alternatives considered

Generate TypeScript from OpenAPI: covers REST but not WebSocket events, which carry most of the product.
Shared schema package: adds a build step across two languages.
