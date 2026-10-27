# 0035. Install skills from the hub into review, check them first, and pin the exact version

- **Status:** Accepted
- **Date:** 2026-10-23

## Context

Sentient writes and improves its own skills, but people also want skills others wrote. Hermes has a skills hub with
several sources, trust levels and a lockfile (`~/.hermes/skills/.hub/lock.json`), and OpenClaw installs from ClawHub
with `.clawhub/lock.json`. All three read the same Agent Skills format (https://agentskills.io/specification): a
folder with a `SKILL.md` and optional `scripts/`, `references/` and `assets/`. Anthropic publishes one such repository
(`anthropics/skills`) and OpenAI another (`openai/skills`). Malicious skills on public marketplaces were a real problem
in 2026 (ClawHub's "ClawHavoc" skills), and a skill's steps go into the prompt as if Sentient wrote them
([ADR 0027](0027-vet-skills-and-mcp-tools-from-outside.md)). A version that changes after the user trusted it is
the same risk as a tool description that changes after approval.

## Decision

- **Sources.** Two kinds: a public GitHub repository of skill folders, read at an exact commit through GitHub's API
  without an account, and a site that publishes the Hermes `/.well-known/skills/index.json` index. Sentient lists a
  small default set, Anthropic's and OpenAI's repositories, as `trusted`. Users can add any other source; it is
  `community`, its trust level is shown wherever the source or its skills appear, and installing from it needs an
  explicit extra yes.
- **Outside content, checked before install.** Everything fetched is outside content
  ([ADR 0018](0018-untrusted-content-gates-sending.md)) and never reaches the model before the user approves it. A
  skill is downloaded into a quarantine folder (GitHub files must match the tree's blob hashes), and the deterministic
  safety check of ADR 0027 runs on the whole folder. A `high` verdict blocks the install unless the user says "install
  anyway", and approving it still needs "approve anyway".
- **Never straight to active.** Every install and every update lands in pending review with its safety report,
  where it came from, who made it and what it asks for (tools, programs, keys, platforms and the scripts it brings,
  read from the Agent Skills, Hermes and OpenClaw header fields). Its code checks never run, as for any skill from
  outside.
- **A lockfile pins the version.** `skills/.hub/lock.json` records, per skill, the source, the exact commit (or the
  index's version), a fingerprint over every file's git blob hash, the files themselves, and both the newest version
  fetched and the version the user approved. An update is fetched and checked the same way, shows which files it adds,
  removes and changes plus the SKILL.md diff, and waits in review while the approved version stays in use. Declining it
  keeps the pin. Nothing updates on its own.

## Consequences

Users can find and try skills others wrote without giving up review, and a source that changes a skill later can't
change what Sentient runs without the user seeing it. Each install costs an extra click, two for a source the user
added, and three for a high-risk skill. GitHub allows about 60 unauthenticated requests an hour, so listings are kept
for 15 minutes and an update check costs two requests per source; heavy use may need an optional GitHub token later.
A skill whose name is already taken is installed with the source's id added, so `pdf` from two sources can coexist.
The safety check is pattern based (ADR 0027), so `trusted` means "a known publisher", not "safe".

## Alternatives considered

Copying the Hermes source list (skills.sh, ClawHub, LobeHub and others) as defaults: more skills, but most are
community uploads, which is where the malicious ones were found. Blocking high-risk skills outright, as Hermes does
for `dangerous` verdicts: safer, but ADR 0027 chose to let the user decide, and the extra confirmations keep that
choice deliberate. Keeping the pin only in the database (`skill_imports`): it already holds the safety record, but a
lockfile next to the skills is what Hermes and OpenClaw users expect, can be read and backed up with the skills, and
records the exact upstream commit, which the database never had. Updating automatically when a source changes: that
is exactly the "changed after it was trusted" attack.
