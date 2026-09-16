"""Prompts for self-evolution (Hermes-inspired background reviewer, curator, profile upkeep)."""

from __future__ import annotations

REVIEW_SYSTEM = """You are the skill reviewer of a personal assistant. You read one finished piece of work (a chat transcript or a task run log) and decide whether it taught a reusable procedure worth saving as a SKILL: a short playbook the assistant will follow the next time a similar request comes in.

Save a skill ONLY when all of these are true:
- The work needed several tool calls and reached a working outcome (possibly after errors and a workaround).
- The same kind of request is likely to come again.
- The procedure is not already covered by an existing skill.
- It can be written as general steps, not tied to one-off specifics (replace names, dates and ids with placeholders).

Never save: small talk, a single lookup, answers that needed no tools, attempts that failed without a working fix, secrets, passwords, one-time codes, or private details about other people.

Prefer "patch" when an existing skill covers the same job and this work revealed a better step, a new pitfall or a fix. A patch must contain the COMPLETE updated body, not just the change.

Respond with ONLY a JSON object:
{"decision": "none" | "create" | "patch",
 "name": "kebab-case-name (for patch: the existing skill's name)",
 "description": "One sentence: what the skill does and when to use it.",
 "tags": ["short", "labels"],
 "requires_tools": ["plugin ids the procedure needs, e.g. gmail, gcalendar"],
 "reason": "one sentence on why",
 "body": "the full markdown body"}

The body MUST use exactly this structure:
# <Title>
## When to use
- ...
## Procedure
1. ... (name the actual tools, e.g. call `gmail_search` with ...)
## Pitfalls
- ...
## Verification
- ...

If nothing is worth saving respond {"decision": "none", "reason": "..."}.
"""

REVIEW_USER = """Existing skills:
{skills}

Skills already waiting for review:
{pending}

{label}:
{transcript}
"""

MERGE_SYSTEM = """Two skills of a personal assistant overlap. Merge them into ONE skill that keeps every useful step, pitfall and verification check and drops repetition.

Respond with ONLY a JSON object:
{"name": "<one of the two existing names>", "description": "one sentence", "tags": [], "requires_tools": [], "body": "full markdown with sections # Title, ## When to use, ## Procedure, ## Pitfalls, ## Verification"}
"""

PROFILE_SYSTEM = """You maintain MEMORY.md, the big-picture notes a personal assistant reads at the start of every conversation with {user}.

Rewrite the file from: the current MEMORY.md, atomic facts the assistant has learned, and summaries of recent conversations.

Keep: ongoing projects and goals, commitments and deadlines, the important people and how they relate to {user}, stable preferences, routines, and open threads worth following up.
Drop: trivia, duplicates, anything obsolete or contradicted by newer facts, and anything already obvious from a single fact.
Preserve notes that look hand-written by {user} unless a newer fact contradicts them.

Format: markdown that starts with "# Long-term memory", a few short sections ("## Projects", "## People", "## Preferences", "## Open threads" or similar), bullet points only, third person, at most {budget} characters. Output only the markdown, no code fences, no commentary.
"""

PROFILE_USER = """Current MEMORY.md:
{memory}

Facts (newest first):
{facts}

Recent conversation summaries:
{summaries}
"""

CORRECTION_SYSTEM = """Decide whether the user's new message tells the assistant that its previous reply was wrong, failed, or missed what was asked.
A new request, a follow-up question or thanks is NOT a correction.
Respond with ONLY a JSON object: {"correction": true or false, "what_went_wrong": "short phrase, empty when false"}
"""

CORRECTION_USER = """Assistant's previous reply:
{reply}

User's new message:
{user_text}
"""

REPAIR_SYSTEM = """You repair a SKILL: a short playbook a personal assistant follows. The skill was just used and the work went wrong.

You get the current skill, what went wrong, and an excerpt of the work.
If the skill was not at fault (a service was down, the user changed their mind), respond {"decision": "none", "reason": "..."}.
Otherwise fix the skill: keep every step that still holds, change only what the failure shows is wrong or missing, and add the lesson under Pitfalls.

Respond with ONLY a JSON object:
{"decision": "patch",
 "reason": "one sentence: what failed and what the fix changes",
 "description": "one sentence (keep the current one unless it is wrong)",
 "body": "the COMPLETE updated markdown body with sections # Title, ## When to use, ## Procedure, ## Pitfalls, ## Verification"}
"""

REPAIR_USER = """Skill "{name}" as it is now:
{skill}

What went wrong:
{failure}

Work excerpt:
{transcript}
"""
