# 0005. Keep secrets in the OS keychain and never require a .env

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

v2 read secrets from a large `.env` file, and contributors could not run it without the maintainers' keys.
Keys in config files also leak through backups, screenshots and support logs.

## Decision

API keys, OAuth tokens and bot tokens are stored only in the operating system keychain through
`sentient.secrets` (environment variables are a fallback for developers). Config files reference secrets by
name only. Logs scrub tokens. Device and webhook secrets are stored as hashes. Nothing in the project requires a
shared key: the app runs on a local model or the user's own key, and tests use a scripted fake model.

## Consequences

Contributors never need anything from us to run or test Sentient. CI needs no secrets, so it runs on forks
([ADR 0014](0014-github-flow-and-no-secrets-ci.md)). On headless Linux the keychain may be missing; CI uses a
file-backed keyring, and the app explains what to install.

## Alternatives considered

Encrypted config file with a master password: another password for non-technical users. `.env` files:
the problem we were solving.
