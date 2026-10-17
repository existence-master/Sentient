<!-- Title: conventional commit with an area, e.g. "fix(tasks): keep the schedule when a plan is edited" -->

## What and why

<!-- What does this change, and why? Link the issue: Closes #123 -->

## How I tested it

<!-- The tests and checks you ran (pytest, ruff, typecheck) and their results. -->

## Real test

<!-- Unit tests are not enough: what did you run for real, end to end? See docs/AGENT_WORKFLOW.md.
     - Setup: engine from this branch with its own SENTIENT_HOME, or the built app; for repo/CI changes, the run on
       GitHub; for docs only, what you checked the text against.
     - Model: which one, by id (for example ollama_chat/qwen3:8b), or "no model involved".
     - What you did and what happened: steps, plus trimmed log lines, API responses or a run link.
     - Attach screenshots or traces to this PR (drag them in). A path on your computer is not evidence.
     - Anything you could not test, and why. -->

## Screenshots

<!-- For visible UI changes. Delete if not applicable. -->

## Checklist

- [ ] Tests added or updated, and `pytest` / `npm run typecheck` pass locally for what I touched
- [ ] `ruff check sentient tests` is clean
- [ ] `docs/API.md` updated if the desktop ↔ engine contract changed
- [ ] A changelog entry in `changes/` (for example `changes/123.fixed.md`, see `changes/README.md`) if users will
      notice this change. Don't edit `CHANGELOG.md`.
- [ ] No keys, tokens or personal data in the code, tests or screenshots
