"""Trigger filter DSL for triggered tasks (v2 ``_event_matches_filter``).

Supports MongoDB-like conditions: ``$or``, ``$and``, ``$not`` and per-field
``$eq``, ``$ne``, ``$in``, ``$nin``, ``$contains``, ``$regex`` plus plain equality.

Normalized event shapes sent by the pollers:
- gmail ``new_email``: {id, thread_id, from, sender_email, to, subject, snippet, body, date, labels, url}
- gcalendar ``new_event``: {id, summary, description, start, end, location, attendees, organizer_email, url, status}

v2 compatibility: for gmail, ``from`` compares against the sender's email address
(case-insensitive); ``$contains``/``$regex`` on ``from`` also see the display name.
Additions: logical operators may be combined with field conditions in one object,
dotted field paths (``organizer.email``) work, and list-valued fields (``labels``,
``attendees``) match when any element matches.
"""

from __future__ import annotations

import re
from typing import Any


def extract_email(header: Any) -> str:
    if not isinstance(header, str):
        return ""
    m = re.search(r"<(.+?)>", header)
    return (m.group(1) if m else header).lower().strip()


def _lookup(data: dict, field: str) -> Any:
    if field in data:
        return data[field]
    cur: Any = data
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _equals(value: Any, expected: Any) -> bool:
    if isinstance(value, list) and not isinstance(expected, list):
        return expected in value
    return value == expected


def _contains(value: Any, needle: Any) -> bool:
    if not isinstance(needle, str):
        return False
    if isinstance(value, list):
        return any(_contains(v, needle) for v in value)
    if isinstance(value, dict):
        return needle.lower() in str(value).lower()
    return isinstance(value, str) and needle.lower() in value.lower()


def _regex(value: Any, pattern: Any) -> bool:
    if isinstance(value, list):
        return any(_regex(v, pattern) for v in value)
    if not isinstance(value, str) or not isinstance(pattern, str):
        return False
    try:
        return re.search(pattern, value, re.IGNORECASE) is not None
    except re.error:
        return False


def _op_ok(op: str, op_val: Any, value: Any, text_value: Any, email_field: bool) -> bool:
    def norm(v: Any) -> Any:
        return extract_email(v) if email_field and isinstance(v, str) else v

    if op == "$eq":
        if email_field and not isinstance(op_val, str):
            return False
        return _equals(value, norm(op_val))
    if op == "$ne":
        return not _equals(value, norm(op_val))
    if op == "$in":
        return isinstance(op_val, list) and any(_equals(value, norm(v)) for v in op_val)
    if op == "$nin":
        return isinstance(op_val, list) and not any(_equals(value, norm(v)) for v in op_val)
    if op == "$contains":
        return _contains(text_value, op_val)
    if op == "$regex":
        return _regex(text_value, op_val)
    return False  # unknown operator never matches


def event_matches_filter(event_data: dict[str, Any], task_filter: dict[str, Any] | None, source: str) -> bool:
    if not task_filter:
        return True  # an empty filter matches everything
    if not isinstance(event_data, dict):
        return False

    def field_ok(field: str, query: Any) -> bool:
        email_field = field == "from" and source == "gmail"
        if email_field:
            raw = event_data.get("from") or event_data.get("sender") or ""
            value: Any = (event_data.get("sender_email") or extract_email(raw) or "").lower().strip()
            text_value: Any = f"{raw} {value}".strip()
        else:
            value = _lookup(event_data, field)
            text_value = value
        if isinstance(query, dict):
            return all(_op_ok(op, op_val, value, text_value, email_field) for op, op_val in query.items())
        if email_field:
            return isinstance(query, str) and value == extract_email(query)
        return _equals(value, query)

    def evaluate(condition: Any) -> bool:
        if not isinstance(condition, dict):
            return False
        for key, sub in condition.items():
            if key == "$or":
                ok = isinstance(sub, list) and any(evaluate(c) for c in sub)
            elif key == "$and":
                ok = isinstance(sub, list) and all(evaluate(c) for c in sub)
            elif key == "$not":
                ok = not evaluate(sub)
            else:
                ok = field_ok(key, sub)
            if not ok:
                return False
        return True

    return evaluate(task_filter)
