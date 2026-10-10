# 0021. Hold memories from untrusted sources for the user's review

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

[ADR 0009](0009-memory-pushed-into-the-prompt.md) puts recalled facts and user-model insights into every prompt
and saves what Sentient learns without asking. That makes memory a way to make an injection last: an email or web
page that gets one line into memory ("Sarthak wants invoices forwarded to ...") steers every later chat, also after
the chat that read it is gone. Memory poisoning is OWASP ASI06; attacks such as MINJA and ZombieAgent work this way,
and one 2026 preprint planted target memories in 24 of 25 trials. [ADR 0018](0018-untrusted-content-gates-sending.md)
already marks runs that read outside content, and [ADR 0017](0017-unprompted-work-reads-only.md) marks work nobody
asked for, but both only gate actions, not what is remembered.

## Decision

Facts and user-model insights get a **pending** state. A memory is saved as pending, with a review note
`{from, snippet, session_id}`, when it comes from:

- a chat turn or task run marked untrusted (`ToolContext.untrusted`): facts extracted after the turn, facts saved
  when an untrusted chat is compressed, and `memory_remember` calls in that run;
- work nobody asked for (`ToolContext.origin` in `UNPROMPTED_ORIGINS`) calling `memory_remember`;
- an import: Hermes (`import:hermes`) and documents (`file:<name>`);
- a user-model refresh whose new insight cites only outside material (messages or summaries of a chat with
  `sessions.untrusted` set), or cites nothing while such material was offered. Support, contradict and retire
  operations that cite only outside material are ignored.

A pending memory is never put into a prompt, returned by `memory_recall` or `memory_search_by_source`, listed among
memories, used by proactivity, read by a user-model refresh or touched by dreaming. Pending facts get no vector, so
vector recall cannot find them, and every other read filters on `status = 'active'`. A held fact only ever adds: it
never updates or deletes a remembered one.

Only the user moves a memory out of pending, on the Memory page's Review tab: Approve, Edit (approve in their own
words), Discard, or Approve all from one source. Nothing a model says can approve one. Memories nobody reviewed are
deleted after `memory.review_expire_days` (default 30) with a plain notification. The user's own words in a clean
chat, memories they add themselves, onboarding and answers to the user model's questions stay active as before.

This narrows ADR 0009: what is pushed into the prompt is only what the user said in a clean chat or approved.

## Consequences

An email, web page, MCP result or background job can no longer plant a lasting memory on its own. People who read
mail or browse in chat will see a Review badge in the sidebar and need to approve what they want kept; a fact learned
in such a chat is not used until they do, even when it was the user who said it. Imports land in the inbox instead
of memory. Every new way of writing memory has to decide whether its source is clean and pass a review note when it
is not.

## Alternatives considered

Asking the model whether a memory looks like an injection: unreliable and itself open to injection. Saving
everything and showing an undo toast: the memory is used until the user notices. Never saving anything from
untrusted chats: loses useful facts the user would gladly keep. Holding only instruction-shaped memories ("from now
on ..."): a plain-looking false fact ("my new bank account is ...") is just as harmful.
