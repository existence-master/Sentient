# 0025. Folders the user adds are their own content; web pages clipped into them are not

- **Status:** Accepted
- **Date:** 2026-10-14
- **Amends:** [ADR 0018](0018-untrusted-content-gates-sending.md) and [ADR 0023](0023-mark-outside-content-in-prompts.md)

## Context

Issue #176 lets the user add folders and Obsidian vaults that Sentient searches (`folder_search`, `folder_read`).
ADR 0018 says a look-up tool's result counts as outside content unless the app is Sentient's own, and once a run has
read outside content, anything that can send data out asks the user first. ADR 0023 wraps such results in markers
that tell the model they are data.

A folder the user picked is mostly their own writing: journals, meeting notes, plans. Treating every search of it
like a web page would make a normal "what did I note about Ravi? Email him a summary" ask every time, and would hold
everything learned from the user's own notes for review (ADR 0021). But vaults also hold text written by others.
Obsidian Web Clipper and similar tools save whole web pages into the vault, with the page address in the note's front
matter, a `clippings` tag, or a `Clippings` folder. A clipped page can carry an injection just like the page itself.

## Decision

- Folder tools are the user's own content: `folder_search` and `folder_read` set `untrusted_output=False`.
- A note is a **clipping** when its front matter has a web address under `source`, `url`, `clipped_from`,
  `source_url`, `original_url` or `link`, when a tag is `clippings` (or `clipping`, `clipped`, `webclip`,
  `web-clip`, `web-clipping`, `readlater`), or when it sits in a folder named `Clippings`, `Web clippings` or
  `Webclips`. The scan records it (`folder_files.clipped`); Settings shows how many each folder has.
- With `folders.clippings_are_outside` (default on), a result that includes a clipping starts with
  `"outside_content": "a web page clipped into your notes"`. The key comes first so it survives a cut result.
- A general rule, not one for folders only: any tool result whose first key is `outside_content` counts as outside
  content (`rules.result_outside_source`), for marking the run (ADR 0018), wrapping it for the model (ADR 0023) and
  classifying a stored transcript (`rules.outside_source`). It can only add a mark, never remove one, so a tool
  can't use it to look safer.
- To keep clippings from coming along for nothing, a piece found only by meaning must be within 0.15 similarity of
  the best match.

## Consequences

Searching the user's own notes doesn't add approvals or hold memories. A search that brings in a clipped page marks
the chat like a web page would: later sends ask first and the clipped text reaches the model marked as data. Clipped
pages that don't follow these conventions (no source, no tag, a different folder) are treated as the user's own; the
setting description says what is detected. Users who clip a lot and trust their vault can turn the setting off.

## Alternatives considered

Treating whole vaults as outside content: safe, but it would make the most common use (answer from my notes, then act)
ask every time and send everything learned from notes to review. Asking the user to mark folders as trusted: an extra
decision most people can't judge. Leaving clippings out of the index: safer, but clipped pages are often exactly what
the user wants to search. Detecting clippings with a model: unreliable and itself open to injection.
