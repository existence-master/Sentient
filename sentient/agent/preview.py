"""Human-readable previews of outgoing actions (#142).

An approval card used to show a call's raw arguments. A preview says the same thing the way a person reads it: who an
email goes to, its subject and its full text; which channel a message lands in; what a delete removes. It is built
in code from the call's own arguments and tool name, never by a model, so it shows exactly what will be sent.

``action_kind`` sorts a call by its tool name (and, for a tool that runs other tools by slug, by those slugs):
``email``, ``message``, ``post``, ``delete``, ``trash`` or ``action``. ``build`` turns a call into
``{kind, title, fields: [{label, value}], body}`` for ``approval_request.preview`` and ``undo_window.preview``
(docs/API.md section 10); ``as_text`` renders one for a messaging app.
"""

from __future__ import annotations

import re
from typing import Any

MAX_BODY = 4000
MAX_VALUE = 300

_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# words in a tool name (or a slug) that say what a call does
SEND_VERBS = frozenset({"send", "reply", "forward", "post", "publish", "comment", "tweet", "dm", "respond"})
POST_VERBS = frozenset({"post", "publish", "comment", "tweet"})
EMAIL_WORDS = frozenset({"email", "emails", "mail", "gmail", "imap", "smtp", "outlook"})
DELETE_VERBS = frozenset({"delete", "remove", "erase", "destroy", "purge", "wipe"})
TRASH_VERBS = frozenset({"trash", "recycle"})

TITLES = {
    "email": "Email", "message": "Message", "post": "Post", "delete": "Delete for good", "trash": "Move to the trash",
    "purchase": "Purchase",
}

# argument keys (split into words) and the label they get, in display order
_FIELDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("To", frozenset({"to", "recipient", "recipients", "email", "emails", "phone", "number", "attendee", "attendees",
                      "invitee", "invitees", "participant", "participants", "guest", "guests", "member", "members",
                      "people", "address", "addresses", "destination", "user", "username"})),
    ("Cc", frozenset({"cc"})),
    ("Bcc", frozenset({"bcc"})),
    ("Where", frozenset({"channel", "chat", "thread", "room", "conversation", "repo", "repository", "issue", "board",
                         "calendar", "folder", "space", "group"})),
    ("Subject", frozenset({"subject", "title"})),
    ("Files", frozenset({"path", "paths", "file", "files", "attachment", "attachments", "filename"})),
    ("Amount", frozenset({"amount", "price", "total", "cost"})),
)
_BODY_WORDS = frozenset({"body", "text", "message", "content", "comment", "html", "markdown", "caption", "description",
                         "note", "reply", "status"})
_ID_WORDS = frozenset({"id", "ids", "ts", "uuid", "key", "slug", "name", "path", "url", "uri"})
SLUG_KEYS = frozenset({"tool_slug", "action_slug", "slug"})
SLUG_ARGUMENT_KEYS = frozenset({"arguments", "args", "params", "parameters", "input"})


def name_words(name: str) -> set[str]:
    """The lower-case words of a tool name or a slug ("mcp_mailer_send_email" -> {mcp, mailer, send, email})."""
    return {w for w in _SPLIT.split(_CAMEL.sub(" ", str(name or "")).lower()) if w}


def _key_words(key: str) -> set[str]:
    return name_words(key)


def inner_call(arguments: dict) -> tuple[str, dict] | None:
    """``(slug, arguments)`` of the first tool a tool-running tool runs (Composio's multi-execute takes
    ``{"tools": [{"tool_slug": ..., "arguments": {...}}]}``), or None."""
    stack: list[Any] = [arguments]
    while stack:
        value = stack.pop(0)
        if isinstance(value, dict):
            slug = next((value[k] for k in SLUG_KEYS if isinstance(value.get(k), str)), None)
            if slug:
                inner = next((value[k] for k in SLUG_ARGUMENT_KEYS if isinstance(value.get(k), dict)), {})
                return slug, inner
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _kind_of_words(words: set[str]) -> str:
    if words & DELETE_VERBS and not words & TRASH_VERBS and "soft" not in words:
        return "delete"
    if words & TRASH_VERBS or ("soft" in words and words & DELETE_VERBS):
        return "trash"
    verbs = words & SEND_VERBS
    if verbs:
        if words & EMAIL_WORDS:
            return "email"
        return "post" if verbs & POST_VERBS else "message"
    return "action"


def action_kind(name: str, arguments: dict | None = None) -> str:
    """What a call does, from its tool name: email, message, post, delete (for good), trash or action. A tool that
    runs other tools by slug is judged by the first slug it runs."""
    kind = _kind_of_words(name_words(name))
    if kind == "action" and arguments:
        inner = inner_call(arguments)
        if inner is not None:
            kind = _kind_of_words(name_words(inner[0]))
    return kind


def _text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, str | int | float):
        return str(value).strip()
    if isinstance(value, list | tuple):
        return ", ".join(t for t in (_text(v) for v in value) if t)
    return ""


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build(
    name: str, arguments: dict | None, *, purchase: bool = False, app: str | None = None, to: list[str] | None = None,
) -> dict | None:
    """The preview of one call, or None when its arguments say nothing a person would read (a plain look-up).
    ``to``: who the call goes to when its arguments don't say, worked out in code (a reply's thread headers, #144)."""
    args = dict(arguments or {})
    kind = "purchase" if purchase else action_kind(name, args)
    if kind != "purchase" and action_kind(name) == "action" and (inner := inner_call(args)) is not None:
        args = dict(inner[1] or {})  # a tool that runs another one: show what that one gets
    fields: dict[str, list[str]] = {}
    body = ""
    for key, value in args.items():
        words = _key_words(key)
        text = _text(value)
        if not text:
            continue
        label = next((lab for lab, keys in _FIELDS if words & keys and not words & {"id", "ids"}), None)
        if label == "Subject" and kind == "post":
            label = "Title"
        if label is None and words & _BODY_WORDS and not words & _ID_WORDS and not body:
            body = text
            continue
        if label is None and kind in {"delete", "trash"} and words & _ID_WORDS:
            label = "Item"
        if label is None:
            continue
        fields.setdefault(label, []).append(_short(text, MAX_VALUE))
    if to and "To" not in fields:
        fields = {"To": [_short(", ".join(to), MAX_VALUE)], **fields}
    if not fields and not body and kind == "action":
        return None
    out: dict[str, Any] = {
        "kind": kind,
        "title": TITLES.get(kind, "Action"),
        "fields": [{"label": label, "value": ", ".join(values)} for label, values in fields.items()],
        "body": _short(body, MAX_BODY) if body else None,
    }
    if app:
        out["app"] = app
    return out


def as_text(preview: dict | None) -> str:
    """A preview as plain lines for a messaging app ("To: sam@example.com" ... then the text)."""
    if not preview:
        return ""
    lines = [f"{f['label']}: {f['value']}" for f in preview.get("fields") or []]
    body = str(preview.get("body") or "").strip()
    if body:
        lines += ["", _short(body, 1200)]
    return "\n".join(lines)
