# 0015. Archive v2 on a branch instead of keeping it in main

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

After v3 replaced it, v2's code (`src/`) stayed in the repository as a reference. Two products in one tree
confused contributors, kept v2's heavy setup instructions in view, and doubled what CI would have to cover.

## Decision

v2 lives on the read-only `v2` branch and the `v2-final` tag, with its full history. `main` contains only
v3. Issues about v2 are closed with a pointer to v3.

## Consequences

The repository describes one product. Nothing from v2 is lost, and anyone can still check it out. Porting
from v2 now means reading another branch, which is rare now that v3 has parity.

## Alternatives considered

Keep `src/` in `main`: the confusion we wanted to end. Delete v2: loses history and the maintainers' work.
