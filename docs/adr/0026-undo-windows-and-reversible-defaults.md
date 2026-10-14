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
