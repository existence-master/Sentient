# 0026. Hold sends in an undo window and prefer the trash

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

Approvals ([ADR 0008](0008-effective-risk-and-approvals.md), [ADR 0016](0016-lasting-approval-rules.md)), grants and
the second look ([ADR 0024](0024-second-look-at-outgoing-actions.md)) decide whether a call may run. Once a send is
allowed it goes at once, and an approval card showed raw JSON arguments, which people skim. Mail apps learned that a
short undo window catches the mistakes a confirmation misses (a wrong recipient noticed a second after clicking). A
model asked to "delete the note" also picks whatever delete tool it sees first, even when the app offers a trash it
could come back from.

## Decision

- **Previews.** Approval requests, undo windows and task approval questions carry a preview built in code from the
  call's tool name and arguments (`sentient.agent.preview`): who it goes to, the subject, the full text, the files,
  the amount. Never written by a model, so it shows exactly what will be sent.
- **Undo window, after every other check.** In a chat someone can press Undo in (the desktop, or a messaging app with
  buttons), an email, message or post waits `tools.approvals.undo_seconds` (default 10) after Never rules, Read only,
  approvals, grants and the second look said it may run. It runs only when the window ends; Undo means it never runs,
  Send now ends the window early, and stopping the reply stops it too. Which calls wait is decided in code from the
  tool name (a send verb), never by a model.
- **Not delayed:** purchases and running code (the user's yes was deliberate and holding them back helps nobody), and
  work nobody is watching: task runs, briefs, script jobs, subagents and other unprompted work. A window nobody sees
  only delays the send. For tasks the approval itself is the user's look at the exact call (it now shows the
  preview), and the Activity timeline lists what ran.
- **Trash instead of delete.** With `tools.approvals.prefer_trash` (default on), a call to a delete-for-good tool is
  swapped, before it is saved or checked, for the same app's trash tool for the same thing when one exists and takes
  the call's arguments, unless the user's own words in the run ask for a permanent delete. Twins are found from tool
  names alone. A chat call that went to the trash shows **Restore** on the Activity timeline when the app has a
  restore tool.

## Consequences

A send in a watched chat reaches its recipient about ten seconds later than before, and the chat reply waits that
long; users who don't want it set the window to 0. A wrong email or message can be taken back before it leaves, from
the desktop or the messaging app it was asked in. Deletes in apps with a trash become reversible by default, and the
model, the transcript and the approval card all see the trash call. Restore from Activity is offered only for chat
calls, because task runs are listed per run, not per call. An app whose tools don't follow the delete/trash naming
gets no swap; a permanent delete then asks as before.

## Alternatives considered

Returning to the model at once and sending in the background: the reply would claim a send that might still be
undone, and a send could be lost on shutdown. A window for task runs with a notification: nobody is looking at most
runs, and an unanswered window would just delay them. Hiding permanent-delete tools when a trash exists: the user
could then never delete for good. Asking a model which tool is reversible: safety decisions stay in code.

## Amendment (2026-10-14): try runs and file snapshots

The two remaining parts of #142.

**Try runs.** "Try it first" (`POST /api/tasks/{id}/dry-run`) runs a task's plan once with every call above a
look-up noted instead of made: sends, posts, purchases, deletes, code, and changes inside Sentient too (a memory, a
file, the task's notepad), since any of them would make the try run leave a trace. Look-ups run for real, so the
preview is of what the task would do with today's real data, not a guess. Which calls are noted is decided in code
from the call's effective risk, in `Agent._plan_call` after Never rules and Read only connections and before
approvals, Ask rules and the second look, so a try run never stops for an approval but says which calls a real run
would ask about. The only call above a look-up that still runs is the run's own `ask_user`: the answer shapes the rest
of the run, and nothing leaves Sentient. The report is written in code from the noted calls and their previews, never
by a model, because a model reading "TRY RUN" results may still say "I sent the email". A try run never changes the
task (status, schedule, last result, a recurring task's memory of earlier runs, the self-improvement reviewer), so it
is safe on a plan that is still waiting for approval, which is when it is most useful.

**File snapshots before code writes.** Code can change files nobody can take back: a command in an allowed folder, a
script run as a local process (it can reach any path), `file_write` over an existing file. Before each runs, Sentient
copies what it could change (the allowed folders and Sentient's files folder; the one file for `file_write`) and,
when it ends, keeps only the copies of files that changed, with a row the Activity timeline shows with **Restore**,
the same button #352 added for the trash. Copying before is the only way to have the old content afterwards, since a
script or command doesn't say which files it will touch. To keep that cheap and bounded: text files only, each under
`tools.snapshots.max_file_mb`, never `.git`, `node_modules` and the like, kept `keep_days` and at most `max_kept_mb`
in all; anything left out is named with a plain reason. Whether to copy is decided from file sizes and names alone
(`stat`, no file opened), so a big folder doesn't hold the command up. When the allowed folders hold more text than
`max_files` (500) or `max_total_mb` (25), nothing is copied for that command at all and Activity says "Too many files
to keep a copy": a partial set would look like a safety net and not be one. Commands on the never-ask list only
read, and a Docker script can only touch its own folder, so neither is copied. Restore keeps copies of what it
replaces first, so it can be undone too, and counts as done only once every file is written; one cut short by a
crash is cleared at the next start and can be done again. Files the code made are listed but not removed: deleting
is the one thing Restore should never do on its own.

Not done: try runs of swarm and script tasks (a check script only looks things up already; a swarm's workers would
need the same notes merged), and snapshots for folders outside the allowed ones, which code is not supposed to touch.
