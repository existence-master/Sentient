# Architecture decision records

Short records of the decisions that shape Sentient: what we decided, why, and what it costs. Read these before
proposing a change to one of them.

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-09-12 |
| [0002](0002-desktop-app-with-local-engine.md) | Ship a desktop app that runs a local engine | Accepted | 2026-09-15 |
| [0003](0003-single-sqlite-database.md) | Keep all state in one SQLite file | Accepted | 2026-09-12 |
| [0004](0004-model-roles-through-litellm.md) | Let the user pick a model for every job | Accepted | 2026-09-12 |
| [0005](0005-secrets-in-os-keychain.md) | Keep secrets in the OS keychain and never require a .env | Accepted | 2026-09-12 |
| [0006](0006-contract-first-api.md) | Define the desktop and engine contract first | Accepted | 2026-09-15 |
| [0007](0007-one-agent-loop.md) | Use one agent loop for every surface | Accepted | 2026-09-15 |
| [0008](0008-effective-risk-and-approvals.md) | Decide approvals by each call's effective risk | Accepted | 2026-09-15 |
| [0009](0009-memory-pushed-into-the-prompt.md) | Push memory into every turn and consolidate it nightly | Accepted | 2026-09-15 |
| [0010](0010-self-evolution-with-human-review.md) | Let Sentient write its own skills, with human review | Accepted | 2026-09-12 |
| [0011](0011-device-node-protocol.md) | Connect devices with a small JSON protocol over WebSockets | Accepted | 2026-09-15 |
| [0012](0012-code-execution-sandbox.md) | Run model-written code in a sandbox that calls tools | Accepted | 2026-09-15 |
| [0013](0013-push-before-poll.md) | Prefer change feeds and scripts over model calls | Accepted | 2026-09-15 |
| [0014](0014-github-flow-and-no-secrets-ci.md) | Use GitHub flow with CI that needs no secrets | Accepted | 2026-10-08 |
| [0015](0015-archive-v2-on-a-branch.md) | Archive v2 on a branch instead of keeping it in main | Accepted | 2026-10-08 |

## Adding a record

1. Copy [0000-template.md](0000-template.md) to the next number, for example `0016-short-title.md`.
2. Fill in context, decision, consequences and alternatives. Keep it to one page.
3. Add it to the table above and open a pull request. Discussion happens in review.

Records are never deleted. To change a decision, add a new record and mark the old one
"Superseded by ADR NNNN".
