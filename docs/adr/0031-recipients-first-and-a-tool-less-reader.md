# 0031. Fix where a message goes before reading what it says, and read outside documents with a tool-less reader

- **Status:** Accepted
- **Date:** 2026-10-17
- **Amends:** [ADR 0024](0024-second-look-at-outgoing-actions.md) (which recipients a name may vouch for; reply
  destinations), [ADR 0018](0018-untrusted-content-gates-sending.md) (a reader result made only of dates, numbers,
  yes or no, ids and choices does not mark a run)

## Context

[ADR 0018](0018-untrusted-content-gates-sending.md) makes every send ask once a run read outside content, and the
second look ([ADR 0024](0024-second-look-at-outgoing-actions.md)) explains on the card when a recipient isn't one the
user gave. Two gaps were left (#144). First, the second look let an address through when it matched a name the user
gave: "reply to Arjun" vouched for `arjun@evil.example` written in the email's own text, and a reply took its
destination from a Reply-To header nobody looked at. Second, the model that plans and acts still read every email in
full. The plan-then-execute and dual-LLM patterns (Willison, "Design patterns for securing LLM agents against prompt
injections", 2025; CaMeL, 2025) fix the destinations before untrusted text is read and give that text only to a model
that has no tools, whose answer is cut down to typed fields. A plain dual-LLM split still lets injected text swap a
recipient argument, so destinations have to be fixed in code.

## Decision

**Recipients first.** Who a send flow reaches is decided from the user's words, trusted records (contacts, memories,
Sentient's own results) and the thread's headers, never from message text:

- Replies take their destination from the original message's headers in code (`integrations/mail.reply_targets`):
  From, or Reply-To, or the recipients of the user's own message; Cc with reply-all. `gmail_reply` and the new
  `email_imap_reply` take no recipient argument. `Tool.destinations_fn` reads those headers before the call runs, so
  the second look checks them and the approval card and undo window show "To". A Reply-To that sends the reply to
  another address than the sender, on another domain, that the user never gave, is flagged ("This reply goes to
  x@evil.example, the address that email asked replies to go to, not to the person who sent it (arjun@example.com).").
- Forwards (`gmail_forward`, `email_imap_forward`) take `to` like any send; the second look checks it.
- Follow-ups already fixed their recipient from the headers before the model saw the thread. The thread now reaches
  the model marked as outside content, and a draft that copies an address or a link only the other person's messages
  have is dropped, like a draft with a blank.
- The second look reads outside tool results as JSON: the keys come from the tool, so an email item's `from` is its
  sender and its `to`, `cc`, `reply_to` are header lines. An address that outside content names anywhere other than
  as a message's sender (its text, its To or Cc line, a web page) passes only when the user's words or trusted records
  have it. A name match no longer vouches for it. The card says where it was: "That address came from the text of an
  email from Arjun Mehta <arjun@example.com> (Gmail), not from you."

**The reader.** `read_with_reader(tool, arguments, fields)` runs one look-up tool in code and hands its result to a
reader: one model call with no tools at all (on the `fast` role, then the primary; Claude Code only inside a chat
reply, [ADR 0022](0022-claude-through-your-own-claude-code.md)), with the document marked as outside content. Code
keeps only the requested fields, each checked against its kind: `text` (one short sentence), `email` and `url` (must be
written in the document), `date`, `number`, `yes_no`, `id` (no spaces, a digit, written in the document) or a list of
choices. Everything else is dropped. The reader is a subagent row with `kind = "reader"`, `tool_calls = 0` and a trace
of the fields asked for and returned. It only takes look-up tools, never one that sends or could carry data to a new
site in a marked run.

A reader result with only dates, numbers, yes or no, ids and choices cannot carry an instruction or a destination, so
it does not mark the run. Text, addresses and links do: the result says `outside_content` and the run is marked as if
the email itself were read, and an address in it is outside content to the second look. The tool is offered with any
tool that reads outside content, and the chat prompt says to decide destinations from the user, contacts or the
sender.

## Consequences

An email that asks to be forwarded to a new address, or names a look-alike of a real contact, can't change where
anything goes without a card that names where the address came from. "Forward Arjun's email to Priya" can run
without reading the email at all, so with Allow rules it needs no question. Reply-To redirections, a common phishing
trick, now ask. A send to a contact whose address also appears on a Cc line Sentient read asks once more than before,
in a run that already asks. The reader costs one short model call; a small model may fill fewer fields, which come
back null. A sender who spoofs From as a contact's look-alike is still treated as that sender when the user replies to
them, as every mail client does; the card shows the address.

## Alternatives considered

- **Route every email through the reader.** Safest, but the model then can't summarize or answer long threads well,
  and small local models lose detail. The reader is offered next to the full-text tools instead; the code gates stay
  the defence either way.
- **Let the reader return free text without marking the run.** The free text is exactly what an injection writes.
- **Trust any header address.** Every header of an email is written by its sender, so a Cc line is as open to
  injection as the text; only the From of a message the user is answering is treated as its sender.
- **Drop Reply-To.** Mailing lists and support desks rely on it; asking once with a reason keeps them working.
