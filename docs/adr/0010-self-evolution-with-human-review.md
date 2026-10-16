# 0010. Let Sentient write its own skills, with human review

- **Status:** Accepted
- **Date:** 2026-09-12

## Context

Inspired by Hermes Agent, Sentient can learn reusable procedures from successful work. Unreviewed
self-modification is risky: a bad skill silently changes future behaviour.

## Decision

After substantial chats and task runs, a reviewer proposes skills (`SKILL.md` with When to use, Procedure,
Pitfalls and Verification). When a skill was followed by errors or a user correction, it proposes a repair.
Proposals wait in Skills → Pending review with a diff and never activate on their own. A curator retires unused
skills.

## Consequences

Sentient improves with use while the user stays in control. Review adds a step, kept light by good diffs
and plain-language reasons.

## Alternatives considered

Auto-activate skills: faster learning, unsafe. No self-evolution: a static assistant.

## Amendment (2026-10-16): a skill can say how to check it works (#386)

**Context.** A drafted skill was only read by a person before it was used. Nothing checked that its steps work, so a
broken skill could still be proposed, and a repair could make a skill worse. The six-field template planned in #186
has a "how to validate" field, but as prose a model can't run.

**Decision.** A skill can declare checks in its header under `validate`: one check or a list of up to five. A check is
one of two kinds, each with an optional `expect`:

```yaml
validate:
  - file: digests/latest.md      # this file must exist: in the skill's folder, else Sentient's files folder
    expect: "## Important"       # optional: text the file must contain
  - run: |                       # a few lines of Python, run in the code sandbox (ADR 0012)
      print(sum([120, 45, 35]))
    expect: "200"                # optional: text the output must contain
    timeout_s: 30                # optional, 5 to 120 seconds
```

Sentient runs the checks in these places:
- **Before it proposes a self-evolved skill:** a reviewer draft, a repair or a curator merge. The reviewer may add
  `validate` to its answer, and a repair or merge keeps the skill's checks. A draft whose check fails is not proposed
  (evolution log `skill_check_failed {name, reason, ...origin}`).
- **Before `skill_save` changes a skill that has checks:** a failing check leaves the skill unsaved and returns a plain
  error.
- **Before it saves an approved skill:** `POST /api/skills/{name}/approve` runs the checks again. A failure returns
  409 and the skill stays waiting for review.

The last report is kept with the skill and shown on its card. A report is passed, failed, waiting or not run.

The rules for code are the same as for any other code:
- File checks only read, so they always run.
- Proposing is work Sentient started itself, so it never runs code (ADR 0017). At that point a code check waits.
- The user's Approve click, under a card that shows the code and says "Approve and run check", is the yes ADR 0012
  asks for. The check then runs once in the sandbox: no keys, a timeout, read-only tools.
- A "Never" rule on `execute_code`, a Read only setting, or code execution turned off means the check doesn't run.
  The report says why.
- Skills from outside (ADR 0027) never have their code checks run. The linter's danger rules read a check's code too.

`validate` is the machine-checkable form of #186's "how to validate" field, so the six-field template can carry
`validate` in the header next to its prose section.

**Consequences.** A skill that says how to check itself can't be proposed broken, and a repair can't break a check
that used to pass. A wrong check written by a small model costs a good draft, which we accept: the reviewer is told to
leave `validate` out unless it is sure. Code checks add a step to approval rather than running unseen.

**Alternatives considered.**
- Shell commands: they need the host terminal and its every-call approval (ADR 0019), and they are hard to keep the
  same across operating systems. Python in the sandbox is already scrubbed and limited.
- Running code checks during the background review: faster feedback, but it would break ADR 0017.
- A model judging whether the skill works: not a check, and small models can't be trusted with it.
