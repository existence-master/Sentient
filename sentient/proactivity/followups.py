"""Follow-ups: notice email conversations that were dropped and offer a ready draft (docs/API.md section 6).

Two kinds:
    - waiting on you: the newest message is from someone else, sent to you directly (you are in To), and has had no
      answer for ``followups.waiting_on_you_days``
    - waiting on them: your own newest message asked something or asked for something, and has had no answer for
      ``followups.waiting_on_them_days``

Everything here is deterministic. The service (``ProactiveEngine.run_followups``) asks the ``fast`` model only about
the threads that survive ``classify``, and the draft is sent only when the user clicks Send on the suggestion: that
creates an already approved task that makes exactly one tool call with the draft unchanged (``send_call``). This
module never sends anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import getaddresses, parseaddr
from typing import Any

from sentient.llm.provider import parse_json_loose
from sentient.proactivity.prefilter import _BULK_LABELS, _CAL_SUBJECTS, _FLUFF

WAITING_ON_YOU = "waiting_on_you"
WAITING_ON_THEM = "waiting_on_them"
TYPE_FOR = {WAITING_ON_YOU: "follow_up_reply", WAITING_ON_THEM: "follow_up_nudge"}
EVENT_TYPE = "follow_up"
DEFAULT_CONFIDENCE = 0.8

_NOREPLY_RE = re.compile(
    r"(no-?reply|do-?not-?reply|donotreply|mailer-daemon|postmaster|bounces?[+@.-]|notifications?@|notify@|alerts?@"
    r"|newsletters?@|digest@|automated@)",
    re.IGNORECASE,
)
_AUTO_SUBJECTS = ("automatic reply", "auto reply", "auto-reply", "autoreply", "out of office", "delivery status",
                  "undeliverable", "read:", "accepted:", "declined:")
_ASK_RE = re.compile(
    r"\?|\b(could you|can you|would you|will you|please|let me know|lmk|any update|any news|get back to me|"
    r"send me|waiting for|waiting to hear|hear back|looking forward to hearing)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"\[[^\]\n]{0,40}\]"                       # [day], [name], [ ]
    r"|\{[^}\n]{0,40}\}"                       # {date}
    r"|<\s*[A-Za-z][A-Za-z _-]{0,30}\s*>"      # <name> (not <a@b.com>)
    r"|\bX{2,}\b"                              # XX, XXX
    r"|\b(?:TBD|TBC)\b"
    r"|\((?:your|insert|add|enter)\s[^)\n]{1,30}\)",  # (your name)
    re.IGNORECASE,
)
_SUBJECT_PREFIX_RE =re.compile(r"^\s*((re|fwd?|aw|wg|sv)\s*:\s*)+", re.IGNORECASE)


@dataclass
class Candidate:
    source: str
    kind: str
    thread: dict
    last: dict
    person: str          # display name for titles ("Priya")
    person_full: str     # full name or address ("Priya Shah")
    person_email: str
    days: int
    key: str             # dedupe key: thread id + newest message id

    @property
    def subject(self) -> str:
        return clean_subject(self.last.get("subject"))


def parse_when(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def clean_subject(subject: Any) -> str:
    return _SUBJECT_PREFIX_RE.sub("", " ".join(str(subject or "").split())).strip()


def addresses(value: Any) -> list[str]:
    return [a.strip().lower() for _, a in getaddresses([str(value or "")]) if a and "@" in a]


def is_noreply(address: str) -> bool:
    return bool(_NOREPLY_RE.search(address or ""))


def is_bulk(m: dict) -> bool:
    """Mailing lists, bulk and automated mail (headers, Gmail categories, auto-reply subjects)."""
    headers = {str(k).lower(): str(v).strip().lower() for k, v in (m.get("headers") or {}).items()}
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if headers.get("precedence") in {"bulk", "junk", "list"}:
        return True
    auto = headers.get("auto-submitted", "")
    if auto and auto != "no":
        return True
    if {str(label).upper() for label in m.get("labels") or []} & _BULK_LABELS:
        return True
    return str(m.get("subject") or "").strip().lower().startswith(_AUTO_SUBJECTS)


def is_calendar(m: dict) -> bool:
    headers = {str(k).lower(): str(v).lower() for k, v in (m.get("headers") or {}).items()}
    if "text/calendar" in headers.get("content-type", ""):
        return True
    subject = clean_subject(m.get("subject")).lower()
    return subject.startswith(_CAL_SUBJECTS) or str(m.get("subject") or "").strip().lower().startswith(_CAL_SUBJECTS)


def _is_fluff(m: dict) -> bool:
    text = " ".join(str(m.get("body") or m.get("snippet") or "").split()).lower()
    return len(text.split()) < 6 and any(p in text for p in _FLUFF)


def _name_of(header: Any, address: str) -> tuple[str, str]:
    """('Priya', 'Priya Shah') from 'Priya Shah <priya@x.com>'; the address when there is no display name."""
    name = parseaddr(str(header or ""))[0].strip().strip('"')
    if not name or "@" in name:
        return address, address
    return name.split()[0], name


def classify(source: str, thread: dict, me: set[str], cfg: Any, now: datetime) -> tuple[Candidate | None, str]:
    """Deterministic filters. Returns (candidate, "") or (None, reason the thread was skipped)."""
    msgs = [m for m in thread.get("messages") or [] if parse_when(m.get("date"))]
    if not msgs:
        return None, "no dated messages"
    msgs.sort(key=lambda m: parse_when(m.get("date")))  # type: ignore[arg-type,return-value]
    last = msgs[-1]
    age = now - parse_when(last.get("date"))  # type: ignore[operator]
    days = int(age.total_seconds() // 86400)
    if age.total_seconds() > cfg.max_age_days * 86400:
        return None, "older than the cap"
    others = [m for m in msgs if not m.get("from_me")]
    if any(is_bulk(m) for m in others):
        return None, "mailing list or bulk mail"
    if any(is_noreply(str(m.get("sender_email") or "")) for m in others):
        return None, "automated sender"
    if any(is_calendar(m) for m in msgs):
        return None, "calendar invite"
    thread_id = str(thread.get("thread_id") or last.get("thread_id") or "")
    key = f"{thread_id}:{last.get('message_id') or last.get('id')}"

    if last.get("from_me"):
        if days < cfg.waiting_on_them_days:
            return None, "not waiting long enough"
        recipients = [a for a in addresses(last.get("to")) if a not in me]
        if not recipients:
            return None, "a note to yourself"
        if any(is_noreply(a) for a in recipients):
            return None, "automated recipient"
        if not _ASK_RE.search(f"{last.get('subject') or ''}\n{last.get('body') or last.get('snippet') or ''}"):
            return None, "no question asked"
        # the display name of the first recipient, when the To header has one
        header = next((f"{n} <{a}>" for n, a in getaddresses([str(last.get("to") or "")])
                       if a and a.strip().lower() == recipients[0]), recipients[0])
        first, full = _name_of(header, recipients[0])
        return Candidate(source, WAITING_ON_THEM, thread, last, first, full, recipients[0], days, key), ""

    if days < cfg.waiting_on_you_days:
        return None, "not waiting long enough"
    sender = str(last.get("sender_email") or "").lower()
    if not sender or sender in me:
        return None, "sent from your own address"
    if not set(addresses(last.get("to"))) & me:
        return None, "not sent to you directly"
    if _is_fluff(last):
        return None, "nothing to answer"
    first, full = _name_of(last.get("from"), sender)
    return Candidate(source, WAITING_ON_YOU, thread, last, first, full, sender, days, key), ""


# ---------------------------------------------------------------------------- the model step
def _days(n: int) -> str:
    return "1 day" if n == 1 else f"{n} days"


def thread_text(c: Candidate, *, max_messages: int = 3) -> str:
    """The newest few messages, newest last, trimmed for a small model."""
    msgs = [m for m in c.thread.get("messages") or [] if parse_when(m.get("date"))]
    msgs.sort(key=lambda m: parse_when(m.get("date")))  # type: ignore[arg-type,return-value]
    parts = []
    shown = msgs[-max_messages:]
    for i, m in enumerate(shown):
        limit = 1500 if i == len(shown) - 1 else 400
        body = " ".join(str(m.get("body") or m.get("snippet") or "").split())
        if len(body) > limit:
            body = body[: limit - 1].rstrip() + "…"
        when = parse_when(m.get("date"))
        who = "You" if m.get("from_me") else str(m.get("from") or m.get("sender_email") or "")
        parts.append(
            f"From: {who}\nTo: {m.get('to') or ''}\nDate: {when.date().isoformat() if when else ''}\n"
            f"Subject: {m.get('subject') or ''}\n\n{body or '(no text)'}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(c: Candidate, user_name: str, system: str, you_tpl: str, them_tpl: str) -> list[dict]:
    user = (user_name or "").strip() or "the user"
    first = user.split()[0] if user != "the user" else "me"
    tpl = you_tpl if c.kind == WAITING_ON_YOU else them_tpl
    return [
        {"role": "system", "content": system.format(user=user, first=first)},
        {"role": "user", "content": tpl.format(user=user, person=c.person_full, days=_days(c.days), thread=thread_text(c))},
    ]


@dataclass
class Decision:
    needed: bool
    about: str = ""
    draft: str = ""
    confidence: float = DEFAULT_CONFIDENCE


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def clean_draft(text: Any) -> str:
    lines = str(text or "").replace("\r\n", "\n").strip().split("\n")
    if lines and re.match(r"^\s*subject\s*:", lines[0], re.IGNORECASE):
        lines = lines[1:]
    draft = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip().strip('"').strip()
    return draft[:1500]


def has_placeholder(draft: str) -> bool:
    """True when a draft has a blank to fill in ([day], {name}, <date>, XX, TBD...). Drafts are sent exactly as
    written, so such a draft must never reach the user as a suggestion."""
    return bool(_PLACEHOLDER_RE.search(draft or ""))


def clean_about(text: Any) -> str:
    about = " ".join(str(text or "").split()).strip().strip("\"'“”.").strip()
    about = re.sub(r"^(about|re|regarding)\s*:?\s+", "", about, flags=re.IGNORECASE)
    about = about.replace(chr(0x2014), ", ").replace(chr(0x2013), "-")  # no em or en dashes in titles
    return about[:60].rstrip() if len(about) >= 2 else ""


def parse_decision(text: str) -> Decision | None:
    """Tolerant parse of the model's answer. None when it gave no usable decision (try again next run)."""
    try:
        data = parse_json_loose(text or "", expect_keys=("needs_follow_up", "draft"))
    except ValueError:
        return None
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        return None
    needed = _truthy(data.get("needs_follow_up", data.get("needs_reply")))
    if needed is None:
        return None
    if not needed:
        return Decision(False)
    draft = clean_draft(data.get("draft"))
    if not draft:
        return None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return Decision(True, clean_about(data.get("about")), draft, max(0.0, min(1.0, confidence)))


# ---------------------------------------------------------------------------- the suggestion
def title_for(c: Candidate, about: str) -> str:
    topic = about or (f'"{c.subject}"' if c.subject else ("their email" if c.kind == WAITING_ON_YOU else "your email"))
    if c.kind == WAITING_ON_YOU:
        return f"{c.person} is waiting for your reply about {topic}"
    return f"No reply from {c.person} about {topic} yet"


def reasoning_for(c: Candidate) -> str:
    if c.kind == WAITING_ON_YOU:
        return f"{c.person_full} wrote to you directly {_days(c.days)} ago and you have not replied yet."
    return f"You wrote to {c.person_full} {_days(c.days)} ago and asked something. There has been no reply since."


def reply_subject(c: Candidate) -> str:
    subject = str(c.last.get("subject") or "").strip()
    return subject if re.match(r"^\s*re:", subject, re.IGNORECASE) else f"Re: {subject or c.subject}".strip()


def suggestion_for(c: Candidate, d: Decision, *, confidence: float) -> dict:
    """The ``suggestion`` object of the notification payload (docs/API.md section 6)."""
    stype = TYPE_FOR[c.kind]
    follow_up = {
        "kind": c.kind,
        "person": c.person_full,
        "person_email": c.person_email,
        "to": c.person_email if c.kind == WAITING_ON_YOU else str(c.last.get("to") or c.person_email),
        "subject": reply_subject(c),
        "draft": d.draft,
        "days_waiting": c.days,
        "thread_id": c.thread.get("thread_id"),
        "message_id": str(c.last.get("id")),
    }
    if c.last.get("mailbox"):
        follow_up["mailbox"] = c.last["mailbox"]
    who = c.person_full
    summary = f"{who}: {c.subject or '(no subject)'}" if c.kind == WAITING_ON_YOU else f"You to {who}: {c.subject or '(no subject)'}"
    source_event: dict[str, Any] = {"source": c.source, "event_type": EVENT_TYPE, "summary": summary, "item_id": c.key}
    url = c.thread.get("url") or c.last.get("url")
    if url:
        source_event["url"] = url
    return {
        "suggestion_type": stype,
        "description": title_for(c, d.about),
        "action_details": {"action_type": "send_email_reply" if c.kind == WAITING_ON_YOU else "send_follow_up_nudge",
                           "to": follow_up["to"], "subject": follow_up["subject"], "body": d.draft},
        "reasoning": reasoning_for(c),
        "confidence": round(confidence, 3),
        "source_event": source_event,
        "follow_up": follow_up,
    }


def notification_message(suggestion: dict) -> str:
    """Description plus the draft as a quote, so a phone or chat app shows what approving would send."""
    draft = str((suggestion.get("follow_up") or {}).get("draft") or "").strip()
    if not draft:
        return str(suggestion.get("description") or "")
    quoted = "\n".join(f"> {line}" if line else ">" for line in draft.split("\n"))
    return f"{suggestion.get('description')}\n\n{quoted}"


def send_call(suggestion: dict) -> dict:
    """The one exact tool call that approving a follow-up runs: the draft, unchanged, in the same conversation
    when possible. Returns ``{tool, arguments, name, description, step, done_text}``."""
    f = suggestion.get("follow_up") or {}
    source = (suggestion.get("source_event") or {}).get("source")
    draft = str(f.get("draft") or "")
    kind = "reply" if f.get("kind") == WAITING_ON_YOU else "nudge"
    who = str(f.get("person") or f.get("to") or "")
    if source == "gmail":
        tool, arguments = "gmail_reply", {"message_id": str(f.get("message_id")), "body": draft}
    else:
        tool = "email_imap_send"
        arguments = {"to": str(f.get("to") or ""), "subject": str(f.get("subject") or ""), "body": draft}
        if f.get("kind") == WAITING_ON_YOU and str(f.get("mailbox") or "INBOX").upper() == "INBOX":
            arguments["reply_to_message_id"] = str(f.get("message_id"))
    subject = clean_subject(f.get("subject"))
    return {
        "tool": tool,
        "arguments": arguments,
        "name": f"Send {kind} to {who}" + (f": {subject}" if subject else ""),
        "description": f"Sends this {kind} to {who} exactly as you approved it:\n\n{draft}",
        "step": f"Send the {kind} to {f.get('to') or who} exactly as drafted",
        "done_text": f"Sent your {kind} to {who}.",
    }
