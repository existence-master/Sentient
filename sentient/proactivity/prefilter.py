"""Cheap, LLM-free filters ahead of the proactive pipeline.

``event_pre_filter`` ports v2 ``workers/proactive/utils.py`` onto the normalized items the
integrations package returns (gmail ``{id, thread_id, from, sender_email, to, subject,
snippet, body, date, labels, url, headers?}``; gcalendar ``{id, summary, description,
start, end, location, attendees, organizer_email, url, status}``). Keyword checks use word
boundaries (v2 matched substrings, so "sale" also discarded "Salesforce" threads).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time
from typing import Any

log = logging.getLogger(__name__)

_FLUFF = ["thanks", "thank you", "got it", "ok", "okay", "received", "sounds good"]
_GMAIL_KEYWORDS = [
    "unsubscribe", "promotional", "newsletter", "sale", "discount", "special offer",
    "limited time", "no-reply", "noreply", "order confirmation", "shipping update",
    "your receipt for", "verify your email",
]
_KEYWORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _GMAIL_KEYWORDS) + r")\b", re.IGNORECASE)
_BULK_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES", "CATEGORY_FORUMS", "SPAM", "TRASH"}
_NOREPLY_RE = re.compile(r"(no-?reply|do-?not-?reply|mailer-daemon|notifications?@)", re.IGNORECASE)
_CAL_SUBJECTS = ("invitation:", "accepted:", "declined:", "updated invitation:", "tentatively accepted:", "canceled event:", "cancelled event:")
_AUTO_SUBJECTS = ("automatic reply", "auto reply", "auto-reply", "autoreply", "out of office")
_BLOCKING = ["busy", "hold for", "blocked", "focus time", "ooo", "out of office"]
_BLOCKING_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _BLOCKING) + r")\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"<([^>]+)>")


def sender_address(item: dict) -> str:
    raw = str(item.get("sender_email") or item.get("from") or "")
    m = _EMAIL_RE.search(raw)
    return (m.group(1) if m else raw).strip().lower()


def event_pre_filter(item: dict[str, Any], source: str, user_email: str | None = None) -> bool:
    """True if the item should go through the proactive pipeline."""
    user_email = (user_email or "").strip().lower() or None
    if source == "gmail":
        headers = {str(k).lower(): str(v) for k, v in (item.get("headers") or {}).items()}
        subject = str(item.get("subject") or "")
        subject_l = subject.strip().lower()
        snippet = str(item.get("snippet") or "")
        body = str(item.get("body") or "")
        labels = {str(label).upper() for label in item.get("labels") or []}
        sender = sender_address(item)

        if (user_email and sender == user_email) or ("SENT" in labels and "INBOX" not in labels):
            log.debug("gmail pre-filter: sent by the user: %s", subject)
            return False
        auto = headers.get("auto-submitted", "")
        if (auto and auto != "no") or subject_l.startswith(_AUTO_SUBJECTS):
            log.debug("gmail pre-filter: auto-reply: %s", subject)
            return False
        if "list-unsubscribe" in headers or headers.get("precedence") in {"bulk", "junk", "list"} or labels & _BULK_LABELS:
            log.debug("gmail pre-filter: bulk/list mail: %s", subject)
            return False
        if "text/calendar" in headers.get("content-type", "") or subject_l.startswith(_CAL_SUBJECTS):
            log.debug("gmail pre-filter: calendar mail: %s", subject)
            return False
        text = (body or snippet).strip().lower()
        if len(text.split()) < 5 and any(p in text for p in _FLUFF):
            log.debug("gmail pre-filter: short fluff: %r", text)
            return False
        if _NOREPLY_RE.search(sender) or _KEYWORD_RE.search(f"{subject} {snippet}"):
            log.debug("gmail pre-filter: transactional/promotional: %s", subject)
            return False

    elif source == "gcalendar":
        summary = str(item.get("summary") or "")
        if str(item.get("status") or "").lower() == "cancelled":
            return False
        organizer = str(item.get("organizer_email") or "").lower()
        if user_email and organizer == user_email:
            log.debug("gcal pre-filter: created by the user: %s", summary)
            return False
        if user_email:
            for att in item.get("attendees") or []:
                if isinstance(att, dict) and str(att.get("email", "")).lower() == user_email and att.get("responseStatus") == "declined":
                    return False
        if _BLOCKING_RE.search(summary):
            log.debug("gcal pre-filter: blocking event: %s", summary)
            return False

    elif source == "webhook":
        body = item.get("body")
        if body in (None, "", {}, []) or (isinstance(body, str) and not body.strip()):
            log.debug("webhook pre-filter: empty body for %s", item.get("name"))
            return False
    return True


def extract_query_text(item: dict[str, Any], source: str) -> str:
    """v2 fallback query for universal search."""
    if source == "gmail":
        return f"{item.get('subject', '')} {item.get('snippet', '')}".strip()
    if source == "gcalendar":
        return f"{item.get('summary', '')} {item.get('description', '')}".strip()
    if source == "webhook":
        return f"{item.get('name') or ''} {_body_text(item.get('body'), 400)}".strip()
    return json.dumps(item, default=str)[:500]


def _body_text(body: Any, limit: int) -> str:
    text = body if isinstance(body, str) else json.dumps(body, default=str, ensure_ascii=False)
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def event_summary(item: dict[str, Any], source: str) -> str:
    if source == "gmail":
        who = str(item.get("from") or item.get("sender_email") or "").split("<")[0].strip().strip('"')
        return f"{who}: {item.get('subject') or '(no subject)'}" if who else str(item.get("subject") or "(no subject)")
    if source == "gcalendar":
        start = event_start(item)
        return f"{item.get('summary') or '(untitled event)'}" + (f" ({start})" if start else "")
    if source == "webhook":
        body = _body_text(item.get("body"), 80)
        name = str(item.get("name") or "webhook")
        return f"{name}: {body}" if body else name
    return str(item.get("summary") or item.get("subject") or item.get("title") or source)


def event_start(item: dict[str, Any]) -> str | None:
    start = item.get("start")
    if isinstance(start, dict):
        start = start.get("dateTime") or start.get("date")
    return str(start) if start else None


def parse_quiet_hours(spec: str) -> tuple[time, time] | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec or "")
    if not m:
        return None
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        return None
    return time(h1, m1), time(h2, m2)


def in_quiet_hours(spec: str, local_now: datetime) -> bool:
    """``spec`` like '22:00-07:00' (wraps midnight) or '13:00-14:00'."""
    parsed = parse_quiet_hours(spec)
    if parsed is None:
        return False
    start, end = parsed
    now_t = local_now.time().replace(tzinfo=None)
    if start == end:
        return False
    if start < end:
        return start <= now_t < end
    return now_t >= start or now_t < end
