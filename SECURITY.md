# Security policy

Sentient runs on people's own computers with access to their messages, files and accounts, so we take security
reports seriously.

## Reporting a vulnerability

Please **do not** open a public issue. Report it privately through
[GitHub security advisories](https://github.com/existence-master/Sentient/security/advisories/new).

Include what you found, how to reproduce it, and what an attacker could do with it. We aim to acknowledge reports
within three working days and to agree on a fix and disclosure date with you.

## What is in scope

- The engine (`sentient/`): the local gateway, the devices listener, webhooks, messaging channels, approvals,
  the code sandbox and the browser tools.
- The desktop app (`desktop/`) and the installers.
- The device protocol and reference firmware (`docs/NODES.md`, `firmware/`).

Especially interesting: anything that lets a web page, email, message or device make Sentient act without the
user's approval, read secrets from the keychain, or reach the engine from the network when it should not.

## Supported versions

Only the latest release and `main` receive security fixes. Sentient v2 (the `v2` branch) is no longer
maintained.
