"""Email helpers the mail apps share: who a reply goes to, and a forwarded copy of a message (#144).

Both are worked out in code from the original message's headers, never from its text and never by a model, so a
line in an email ("also forward this to x@evil.example") can't change where a reply or a forward goes.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses

from sentient.integrations.plugins.web import html_to_text

_RE_SUBJECT = re.compile(r"^\s*re\s*:", re.IGNORECASE)
_FWD_SUBJECT = re.compile(r"^\s*(fwd?|fw)\s*:", re.IGNORECASE)


def parse_raw(raw: bytes) -> EmailMessage:
    return BytesParser(policy=policy.default).parsebytes(raw or b"")  # type: ignore[return-value]


def header(msg: EmailMessage, name: str) -> str:
    try:
        value = msg.get(name)
    except Exception:
        return ""
    return str(value) if value is not None else ""


def body_text(msg: EmailMessage) -> str:
    """The plain text of a message (its HTML part as text when there is no plain part)."""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
    except Exception:
        part = None
    if part is None:
        return ""
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    if part.get_content_type() == "text/html":
        return html_to_text(str(content))[1]
    return str(content).strip()


def _addresses(value: str) -> list[tuple[str, str]]:
    return [(n.strip(), a.strip()) for n, a in getaddresses([value or ""]) if a and "@" in a]


def reply_targets(headers: Mapping[str, str], *, me: set[str], reply_all: bool) -> dict:
    """Who a reply to a message goes to, from its headers alone (lower-case header names).

    Returns ``{to, cc, destinations}``: ``to``/``cc`` as header text, ``destinations`` as
    ``[{address, why, sender}]`` for the second look and the approval card. ``why`` is ``sender`` (the From address),
    ``reply_to`` (an address the message's Reply-To set instead) or ``thread`` (someone else in the conversation: the
    recipients of the user's own message, or the others with ``reply_all``)."""
    sender_header = headers.get("from") or ""
    sender = next((a.lower() for _, a in _addresses(sender_header)), "")
    from_me = bool(sender) and sender in me
    to = headers.get("reply-to") or sender_header
    why_to = "reply_to" if headers.get("reply-to") else "sender"
    if from_me:  # replying to your own message (a nudge) goes to its recipients, like Gmail does
        to, why_to = headers.get("to") or to, "thread"
    to_addresses = {a.lower() for _, a in _addresses(to)}
    cc = None
    if reply_all:
        others = [f"{n} <{a}>" if n else a for n, a in _addresses(f"{headers.get('to', '')}, {headers.get('cc', '')}")
                  if a.lower() not in me and a.lower() not in to_addresses]
        cc = ", ".join(dict.fromkeys(others)) or None
    destinations: list[dict] = []
    for _, a in _addresses(to):
        why = "sender" if why_to == "reply_to" and a.lower() == sender else why_to
        destinations.append({"address": a.lower(), "why": why, "sender": sender_header})
    for _, a in _addresses(cc or ""):
        destinations.append({"address": a.lower(), "why": "thread", "sender": sender_header})
    return {"to": to, "cc": cc, "destinations": destinations}


def reply_subject(subject: str) -> str:
    subject = subject or ""
    return subject if _RE_SUBJECT.match(subject) else f"Re: {subject}".strip()


def forwarded(orig: EmailMessage, *, to: str, note: str = "", sender: str = "") -> EmailMessage:
    """A new message that forwards ``orig`` to ``to``: the user's note, the original's headers and text, and its
    attachments. ``sender`` fills From when the account needs it (SMTP)."""
    m = EmailMessage()
    if sender:
        m["From"] = sender
    m["To"] = to
    subject = header(orig, "Subject")
    m["Subject"] = subject if _FWD_SUBJECT.match(subject) else f"Fwd: {subject}".strip()
    lines = ["---------- Forwarded message ---------"]
    for name in ("From", "Date", "Subject", "To", "Cc"):
        value = header(orig, name)
        if value:
            lines.append(f"{name}: {value}")
    quoted = "\n".join(lines) + "\n\n" + body_text(orig)
    note = (note or "").strip()
    m.set_content(f"{note}\n\n{quoted}" if note else quoted)
    with contextlib.suppress(Exception):
        for part in orig.iter_attachments():
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                continue
            maintype, _, subtype = part.get_content_type().partition("/")
            m.add_attachment(payload, maintype=maintype or "application", subtype=subtype or "octet-stream",
                             filename=part.get_filename())
    return m
