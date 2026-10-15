# 0028. Let a quick light command count as its own yes, and make opening the home always ask

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

Quick commands (#150, #356) do a plain timer or reminder at once, with no model, and only when the tool would run
without asking. Home Assistant (#175) adds lights and switches: "turn off the living room lights" should feel like a
light switch, on every surface, including voice. But a light is outside Sentient, so its tools are a change (`write`),
and approvals mode "ask" (the default, [ADR 0008](0008-effective-risk-and-approvals.md)) asks before every change. Read
literally, the quick path would never run for a light, and the user would be asked to confirm the exact words they had
just said.

## Decision

- A home quick command is the user's own message naming exactly one change: which devices (an exact name, the lights
  of a room, or "all the lights") and what (on, off, a brightness). In approvals mode "ask", those words count as the
  yes for that call, like an "Only if I asked" rule ([ADR 0016](0016-lasting-approval-rules.md), #140).
  `ApprovalBroker.quick_runs` decides it in code; `QuickPlan.asked` marks such a plan.
- Only for an effective risk of `write`. Never with approvals mode "always", an "Ask", "Hand it to me" or "Never" rule
  on the tool or on `home_assistant`, a Read only connection (#141), or a device the
  entity picker leaves out. Then the model answers and the usual approval card or refusal follows.
- Locks, alarms, garage doors, gates and anything that sounds like one never go through a quick command: the parser
  skips "lock", "unlock" and "alarm", the matcher refuses security devices, and their tools are `send` with
  `allow_for_chat: false`, so they ask every time with only "Allow once".
- **Opening the home is a hard stop, like a purchase.** Unlocking a door, turning off the alarm, opening a garage
  door, gate or door, and switching a switch, script or button that sounds like one ask in every approvals mode,
  `off` included. An Allow rule, "Allow for this chat" and grants never cover them; a task run pauses and asks,
  subagents stop, and scripts refuse them. Code decides it from the tool's approval label
  (`rules.OPENS_HOME_LABELS`, `rules.always_asks`), never a model; an entity Sentient can't look up yet counts as one.
  Locking and arming follow the normal rules: getting them wrong leaves the home safer, not open.
- Anything not exactly clear ("the bedroom light" when there are two, "the lights" with no room) goes to the model
  unchanged.

## Consequences

- "Turn off the living room lights" takes milliseconds and no model, also from Telegram or voice.
- A lasting rule still wins: "Always ask" for Home Assistant makes every light ask again, through the model.
- The exception is narrow and deterministic. Timers and reminders keep the old rule (they are Sentient-internal and
  never ask in mode "ask" anyway).
- "Approvals off" no longer means "never asks" for the home: the list of what mode `off` still asks for (docs/API.md
  section 2, the Settings text) names it.

## Alternatives considered

- **Ask anyway.** Correct by the letter, but a confirmation for words the user just said trains people to click yes
  without reading, which weakens every other question.
- **Make light tools `internal` or `read`.** Wrong: a light is outside Sentient, and the model's own calls to the same
  tools should still ask in mode "ask".
- **Mark every Home Assistant change as Allow by default.** Would also let the model change things nobody asked for.
