"""Shared helpers: HTTP client, keychain JSON credentials, privacy filters."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from sentient import secrets as _secrets
from sentient.integrations.base import USER_AGENT

log = logging.getLogger(__name__)

PRIVACY_KEYS = ("keywords", "emails", "labels")


def http_client(timeout: float = 30.0, **kwargs: Any) -> httpx.AsyncClient:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers, **kwargs)


# ---------------------------------------------------------------------------- keychain
# Looked up through the module attribute so tests can monkeypatch sentient.secrets.
def load_secret_json(name: str) -> dict | None:
    raw = _secrets.get_secret(name)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def store_secret_json(name: str, data: dict) -> bool:
    return _secrets.set_secret(name, json.dumps(data))


def delete_secret(name: str) -> bool:
    return _secrets.delete_secret(name)


# ---------------------------------------------------------------------------- privacy filters
def normalize_filters(raw: dict | None) -> dict[str, list[str]]:
    raw = raw or {}
    out: dict[str, list[str]] = {}
    for k in PRIVACY_KEYS:
        vals = raw.get(k) or []
        if isinstance(vals, str):
            vals = [vals]
        seen: list[str] = []
        for v in vals:
            s = str(v).strip()
            if s and s.lower() not in {x.lower() for x in seen}:
                seen.append(s)
        out[k] = seen
    return out


def email_of(value: str | None) -> str:
    if not value:
        return ""
    m = re.search(r"<([^>]+)>", value)
    return (m.group(1) if m else value).strip().strip('"').lower()


def email_blocked(msg: dict, filters: dict) -> bool:
    """v2 gmail rule: keyword in subject/body, sender email match, or label match."""
    keywords = [k.lower() for k in filters.get("keywords", [])]
    emails = [e.lower() for e in filters.get("emails", [])]
    labels = [lab.lower() for lab in filters.get("labels", [])]
    text = f"{msg.get('subject') or ''} {msg.get('body') or msg.get('snippet') or ''}".lower()
    if any(k in text for k in keywords):
        return True
    sender = msg.get("sender_email") or email_of(msg.get("from"))
    if any(e in sender for e in emails):
        return True
    msg_labels = [str(lab).lower() for lab in msg.get("labels") or []]
    return any(lab in msg_labels for lab in labels)


def event_blocked(event: dict, filters: dict) -> bool:
    """v2 gcal rule: keyword in summary/description, or a blocked attendee email."""
    keywords = [k.lower() for k in filters.get("keywords", [])]
    emails = {e.lower() for e in filters.get("emails", [])}
    text = f"{event.get('summary') or ''} {event.get('description') or ''}".lower()
    if any(k in text for k in keywords):
        return True
    attendees = event.get("attendees") or []
    addrs = {(a.get("email") if isinstance(a, dict) else str(a)).lower() for a in attendees if a}
    organizer = event.get("organizer_email") or (event.get("organizer") or {}).get("email") or ""
    if organizer:
        addrs.add(organizer.lower())
    return bool(emails & addrs)


def truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n[... truncated ...]", True
