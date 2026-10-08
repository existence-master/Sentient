# 0002. Ship a desktop app that runs a local engine

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

v2 was a cloud web app: Next.js, FastAPI, 22 MCP servers, Celery, Redis, MongoDB, Postgres and Chroma,
configured through hundreds of environment variables. Self-hosting it was hard enough that people asked
maintainers for their keys. Sentient's value is personal: it reads your mail, remembers your life and acts for
you, which is exactly the data people do not want on someone else's server.

## Decision

Sentient is a desktop application. An Electron window (React + TypeScript) starts a local Python engine
(`python -m sentient serve`, or a frozen `sentient-engine` in installed builds) on a random loopback port with a
per-launch token, and talks to it over REST and WebSockets. There is no account, no login and no Sentient
server.

## Consequences

Installing Sentient is one installer. All data lives in `~/.sentient`. The engine stays a separate process,
so heavy work never freezes the window and phones or glasses can talk to the same engine
([ADR 0011](0011-device-node-protocol.md)). We lose zero-install access from any browser and must build and test
installers for three operating systems.

## Alternatives considered

Keep the web app and add a desktop wrapper: keeps every server dependency. A single native process with the
UI inside Python (Qt, Tauri + sidecar): smaller, but the web UI stack is where contributors and component
libraries are. Pure Electron with a Node engine: would drop the mature Python ecosystem for speech, models and
integrations.
