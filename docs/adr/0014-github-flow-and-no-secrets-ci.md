# 0014. Use GitHub flow with CI that needs no secrets

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

v2 used `development`, `staging` and `master` branches, accepted pull requests only into `development`,
required two approvals on every branch, and needed secrets to run. Outside contributors could not run the app,
their CI checks failed (including an expired CLA token), and maintainers were asked for environment
variables.

## Decision

One long-lived branch, `main`. Contributors fork and open small pull requests to `main`; one maintainer
approval and green checks merge them by squash or rebase. CI runs lint, engine tests on Linux and Windows, the
desktop typecheck and build, and a secret scan, all without secrets, so it behaves the same on forks. Releases are
version tags that build installers. The CLA bot stores signatures in this repository with the built-in token.

## Consequences

Contributing needs only a fork. History stays linear. Maintainers must keep `main` releasable, because
there is no staging branch to catch mistakes; CI is that safety net.

## Alternatives considered

Git flow with develop and release branches: more process than a small team needs. Trunk-based development
with direct pushes: unsafe for an open project.
