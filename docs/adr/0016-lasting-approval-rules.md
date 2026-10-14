# 0016. Let people set lasting Allow, Ask or Never rules per app and tool

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

[ADR 0008](0008-effective-risk-and-approvals.md) decides approvals by each call's effective risk, with one global
mode and "Allow for this chat". People want rules that last: "always draft Gmail replies without asking", "always
ask before posting to Slack", "never delete files". A per-chat answer has to be given again in every chat, and the
global mode is too coarse. Any such rule must stay deterministic: a model, a web page or an email must never be able
to loosen it.

## Decision

Add `tools.approvals.rules`, a map from a key to `allow`, `ask` or `never`, saved in the config file. A key is a tool
name or a plugin id (every tool of that app); a tool's own rule beats its app's rule. The broker, the agent loop and
the sandbox bridge apply rules in code before a call runs; no model sees or decides them. This extends ADR 0008:

- **Never**: the registry stops offering the tool to the model on every surface (chat, voice, channels, tasks,
  subagents, proactivity, scripts) and planners do not list it. A call made anyway is refused with a plain sentence
  that points to Settings, and the tool does not run.
- **Ask**: always ask, even when the global mode is off, after "Allow for this chat" and for look-ups. Where nobody
  can be asked the call never runs unasked: a task run stops and fails with a plain reason ("Slack is set to Ask, and
  tasks can't ask yet. Change it in Settings > Approvals & safety.") and the usual "Task failed" notification;
  subagents and scripts refuse the call; proactive look-ups leave the tool out. Making tasks pause and ask instead
  is a planned follow-up.
- **Allow**: run without asking, whatever the mode. A purchase (effective risk `send` or higher whose approval
  wording is "Purchase", such as a browser click on "Place order") still asks, even with approvals mode off,
  and "Allow for this chat" never covers it; `browser.confirm_purchases` is the only switch for that. Allow never widens what
  scripts from code execution may call: they keep [ADR 0012](0012-code-execution-sandbox.md)'s read-only rule,
  as docs/PRIVACY.md promises.

Rules only ever add questions or remove tools, except for Allow, which the person sets for a named tool or app
themselves, and which still never reaches purchases or scripts.

## Consequences

People can tune Sentient once instead of answering the same question in every chat. A "never" rule also shrinks the
tool list, which helps small local models. Every new place that runs tools must go through `run_loop` or apply
`sentient.tools.rules` itself, and every new money-moving tool must label its approval "Purchase" so Allow cannot
skip it. A task run that reaches an "ask" tool now stops with a clear message until the rule changes or tasks
learn to pause and ask.

## Alternatives considered

Pausing background tasks until someone answers: more machinery than this change needs, so it is left for a
follow-up. Letting Allow open scripts to sending tools: convenient for bulk work, but it would break the promise
that code Sentient writes can only read. Letting the model pick rules from chat ("stop asking me about Gmail"): convenient, but it would
let prompt injection loosen safety. Treating every `send` as un-allowable: safe, but then "always send my Gmail
drafts without asking" could not be expressed.

## Amendment (2026-10-09)

Review found three gaps, now closed: purchases asked only while approvals were on (they now always ask, and
scripts refuse them in every mode); a "never" rule set while a call waited for approval did not stop that call
(the rule is now checked again right before any tool runs); and a running script kept the rules it started with
(scripts now read the current rules on every call).

## Amendment (2026-10-10): rules from chat (#130)

An OpenClaw user's "don't delete my emails" was lost when the agent compacted its context, and it deleted real email.
Something said once in a chat can be lost; a rule in code cannot. When a chat message reads like a standing "never"
or "ask me first" about an action, Sentient now proposes the matching Never or Ask rule as a card in the chat ("Make
this a rule? Never: Gmail > Trash"). This keeps the decision above: a model never picks rules. A deterministic
pre-filter and a word match against the tool registry come first, the fast model only maps the words to tool names or
app ids, those are checked in code, and only the user's click on "Make it a rule" saves the rule (with a note in
`tools.approvals.rule_origins` so Settings says where it came from). Until the user answers, the proposal makes that
chat ask before the matched tools, saved with the chat so a summary or a restart cannot drop it; like every rule, it
only adds questions. Declining saves nothing. Instructions that ask for fewer questions ("don't ask me before
archiving") are never proposed: loosening stays a choice made in Settings. A whole app is proposed only when the
words name no specific action ("never use Slack"); when they name one, code narrows the proposal to the matching
tools. A proposal made in a messaging app is sent back there with the same two choices.

## Amendment (2026-10-10): Read only connections (#141)

People want Sentient to read their mail or repos without ever being able to change them, and a rule per write tool
is a lot to set up. Each connection (an integration or an MCP server) now has an access level, **Read only** or
**Read and write** (the default), chosen when connecting and changed any time in Integrations. It is stored in
`integrations.read_only`, a list of app ids, and read on every check, so a switch takes effect without a restart.

A Read only connection works exactly like a Never rule on every one of its tools whose effective risk is above
`read`: those tools are not offered on any surface and are left out of planners' lists, and a call made anyway is
refused in code before modes, rules, "Allow for this chat" or a script's approvals mode are consulted, and again
right before the tool runs. Like a Never rule it only takes away: an Allow rule never reopens a write tool of a Read
only connection. A tool whose risk depends on its arguments (`risk_fn`) stays offered and each call is judged on its
effective risk. MCP tools that run other tools named by a slug with their arguments (Composio's
`COMPOSIO_MULTI_EXECUTE_TOOL`) now get such a `risk_fn`, whatever the server's own hints say: the verb in each slug decides (`GMAIL_FETCH_EMAILS` reads,
`GMAIL_SEND_EMAIL` sends), a slug with no known verb counts as `write`, and a call is `read` only when every slug it
runs is a plain look-up. This also means those look-ups no longer ask in approvals mode "ask"; they still ask once
the chat has read outside content (ADR 0018), because the tool stays marked as one that can send data out.


## Amendment (2026-10-12): approvals off and undecided proposals (#306)

With approvals mode "off", a test chat whose message said "Do not call browser_open." still asked before a read-only
`browser_snapshot`, and the request waited ten minutes and was declined; another that said "do not call browser_open
again in part A" asked before `browser_open`. The rules-from-chat check had turned those words into proposals (the
first for the whole Web Browser app), and an undecided proposal asked before every matched tool, look-ups included,
whatever the mode said. A proposal is a guess from the chat that the user has not confirmed, so it
no longer outweighs "off" for reading: with approvals off, an undecided proposal (and a check that is still running)
asks only before calls whose effective risk is above `read`. A call that changes, sends, deletes or runs something
still asks until the user answers the card, because that is the harm the feature guards against ("never delete my
emails" said a moment ago). Modes "ask" and "always" keep asking before every matched tool. The detection also got
tighter: words that name a tool exactly give that tool, never its whole app, and a sentence limited to the request at
hand ("this time", "just this once", "for now", "in part A", "in step 2") is not a standing instruction.

"Off" now means no questions apart from this complete list: purchases, tools or apps set to Ask, sending after outside
content was read ([ADR 0018](0018-untrusted-content-gates-sending.md)), and changes covered by an undecided proposal.
Since 2026-10-13 also calls a second look flagged ([ADR 0024](0024-second-look-at-outgoing-actions.md)).
Never rules and Read only connections refuse instead of asking.

## Amendment (2026-10-12): tasks ask, scoped answers and approvals outside the app (#139)

A task run that reached an Ask tool failed ("tasks can't ask yet"), so a scheduled task could not use a tool its owner
only wanted to be asked about, and every approval was a one-call yes or "for this chat". Now:

- **Tasks pause and ask.** A single task run that reaches an Ask tool holds the call and pauses with the same machinery
  as the outside-content check (ADR 0018); the answer runs exactly that call once and the run carries on. Swarm
  workers and subagents still cannot ask, so they keep refusing.
- **Scoped answers.** The answer says how wide the yes is: once, this chat, this task, for a while
  (`tools.approvals.grant_minutes`, "Allow for 1 hour"), or always (an Allow rule with an origin note). A task or timed
  grant covers later calls of that one tool up to the risk approved, and it does cover the tool's Ask rule: the person
  was asked about that tool and chose the width themselves, as they could by setting Allow in Settings. Purchases,
  calls after outside content, tools that ask every time, raised risks and running code are only ever offered "once";
  never rules, Read only connections, undecided proposals and the unprompted-work limit come before any grant. A yes
  wider than a request offers counts as "once", so no client can widen it. Grants are listed in Settings >
  Approvals with Revoke; a task's grants end with the task.
- **Outside the app.** A waiting task's approval request goes to the paired messaging chats as buttons (numbered on
  WhatsApp) or reply words, only while nobody is looking at the window (the window reports focus over `/ws`). Only the
  chat a request was delivered to can answer it, and only through that message; outside content can never answer
  one because answers come only from button presses and replies in paired chats, never from a model or a tool. Each
  request expires (`tools.approvals.task_request_minutes`): the run stops and a late answer is refused.

The alternative of time-limited "Allow for this chat" only would have kept tasks unable to ask. Letting grants cover
purchases or code would make a phone tap a standing permission for spending or running code, which ADR 0008 and ADR
0019 rule out.

## Amendment (2026-10-14): "Only if I asked" and "Hand it to me" (#140)

Allow, Ask and Never left out two common wishes: "go ahead without asking, but only when I asked for it", and "prepare
it, then let me press the button". Two more levels now sit next to them, applied in code like the others:

- **Only if I asked** (`if_asked`): Allow for a call the user's own words in this chat turn asked for, Ask everywhere
  else. The words are the chat message and its steer messages as typed or spoken, never the assistant's text, an
  attachment, a tool result or outside content, so nothing a model reads or writes can make a call "asked for". Only
  a chat turn has them: tasks pause and ask (a task's brief or an approved plan is not the user asking just now),
  swarm workers stop, and subagents, scripts, briefs and proactive look-ups treat it as Ask. A model never decides
  whether the words asked: `rule_levels.asked_for` matches, per clause, a verb for the action and a word for what it
  acts on or its app (a bare "yes, send it" counts while the clause names no other connected app and no email address
  for a tool that isn't email), skips clauses that say not to do it or to ask first, and counts a look-up as asked for
  when the clause names what it reads. A missed match costs one question; a wrong one would be an action nobody asked for, so
  the matching stays narrow. Like Allow it never reaches purchases, and everything on the list of what still asks with
  approvals off (outside content, a second look, an undecided proposal) still asks.
- **Hand it to me** (`hand_off`): a call that would change, send, delete or run something never runs, whatever the
  mode, other rules, "Allow for this chat" or grants say; look-ups run. Its result tells the model it was not done and
  carries a preview built in code from the call's arguments ([ADR 0026](0026-undo-windows-and-reversible-defaults.md)),
  which the window shows with Copy and, for an email, Open in your email app. It is checked before the outside-content
  check and the second look (nothing leaves Sentient), and again right before a tool runs, and a delete for good it
  covers is never swapped for the trash version. Task runs hand calls over the same way and notify "Ready for you";
  scripts refuse such calls, since they can't hand anything over.

Rules from chat (#130) can now propose these levels, which revisits "loosening stays a choice made in Settings" for
one case. The level is chosen in code from the words, never by the model ("I'll send them myself", "just draft it"
give Hand it to me; "unless I ask", "only when I tell you to" give Only if I asked). A proposal is still only made
where it tightens, with strictness Allow, no rule, Only if I asked, Ask, Hand it to me, Never: Only if I asked is
proposed over no rule or an Allow, never over Ask, because "don't send emails unless I ask" is a restriction the
user states, and it still only becomes a rule when the user clicks "Make it a rule". Only if I asked can mean fewer
questions than no rule in a chat whose words ask for the action; that is exactly what the user's words and click
asked for, and it asks more than no rule in tasks.

Alternatives considered: asking the fast model whether the message asked for the action (it would let a model loosen
a rule, and outside content in the conversation could steer it); treating every action word in the message as asking
for every tool that uses it (summarizing "my Slack messages" would then post to Slack); and making Hand it to me a
variant of Never that still lists the tool (the model then can't prepare anything, which is the point).
