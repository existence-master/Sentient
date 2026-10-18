"""Which new emails are meant for Sentient, and whether their sender may give it tasks (#159).

Pure functions over the mail apps' item shape (docs/API.md section 5, with ``cc`` and ``headers`` from the push
feeds). ``sentient/tasks/email_inbox.py`` turns an email that passes into a task. Nothing here calls a model: who
sent an email is decided from its From line in code, by the real address alone. The display name never counts, so
``"Maya Rao" <attacker@evil.example>`` is attacker@evil.example (Hermes once let quoted display names through).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sentient.integrations.mail import address_set, real_address

MAIL_APPS = ("gmail", "email_imap")

# what the provider found when it checked the sender (RFC 8601 Authentication-Results)
_DMARC_FAIL = re.compile(r"\bdmarc\s*=\s*fail\b", re.IGNORECASE)
_SPF_FAIL = re.compile(r"\bspf\s*=\s*fail\b", re.IGNORECASE)
_DKIM_PASS = re.compile(r"\bdkim\s*=\s*pass\b", re.IGNORECASE)
_DMARC_ANY = re.compile(r"\bdmarc\s*=", re.IGNORECASE)
_BULK_PRECEDENCE = {"bulk", "junk", "list"}


def settings(app: object) -> Any:
    """``tasks.email`` when tasks by email are on, else None."""
    cfg = getattr(getattr(getattr(app, "config", None), "tasks", None), "email", None)
    return cfg if cfg is not None and cfg.enabled else None


def sentient_address(app: object) -> str:
    """The user's Sentient address when tasks by email are on, else ""."""
    cfg = settings(app)
    return cfg.address.strip().lower() if cfg is not None else ""


def headers_of(item: dict) -> dict[str, str]:
    raw = item.get("headers")
    return {str(k).lower(): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def recipients(item: dict) -> set[str]:
    """Every address an email went to: its To and Cc lines and the addresses it was delivered to."""
    h = headers_of(item)
    return address_set(item.get("to"), item.get("cc"), h.get("delivered-to"), h.get("x-original-to"))


def addressed(item: dict, address: str, label: str = "") -> bool:
    """True when an email was sent or copied to the Sentient ``address``, or landed in its ``label`` or folder."""
    address = (address or "").strip().lower()
    if address and address in recipients(item):
        return True
    label = (label or "").strip().lower()
    if not label:
        return False
    names = [*(item.get("labels") or []), *(item.get("label_names") or []), item.get("mailbox") or ""]
    return any(str(n).strip().lower() == label for n in names)


def base_address(address: str) -> str:
    """``you+sentient@example.com`` -> ``you@example.com``."""
    local, at, domain = (address or "").strip().lower().partition("@")
    if not at:
        return ""
    return f"{local.split('+', 1)[0]}@{domain}"


@dataclass
class Sender:
    """Who an email is from and whether it may give Sentient a task. ``reason`` is "" when it may."""

    address: str  # the real address its From line names, "" when it names none clearly
    reason: str = ""  # why it is ignored, one plain sentence
    code: str = ""  # not_allowed | unclear_sender | failed_check | automated


def check_sender(item: dict, allowed: Iterable[str]) -> Sender:
    """Decide in code whether an email's sender may give Sentient a task.

    - the From line must name exactly one address, clearly (``mail.real_address``), and only once;
    - the email service must not have found that the address was forged (DMARC fail, or SPF fail with no DKIM pass);
    - automatic mail (auto-replies, mailing lists, bulk mail) never starts a task, so two systems can't loop;
    - the real address must be on the allowed list. The display name is never compared with anything."""
    h = headers_of(item)
    raw_from = h.get("from") or str(item.get("from") or "")
    address = real_address(raw_from)
    if not address:
        return Sender("", "Its sender address couldn't be read clearly, so Sentient can't tell who sent it.",
                      "unclear_sender")
    auth = h.get("authentication-results", "")
    if _DMARC_FAIL.search(auth) or (not _DMARC_ANY.search(auth) and _SPF_FAIL.search(auth)
                                    and not _DKIM_PASS.search(auth)):
        return Sender(address, f"Its email service couldn't confirm it really came from {address}.", "failed_check")
    auto = h.get("auto-submitted", "").strip().lower()
    if (auto and auto != "no") or h.get("precedence", "").strip().lower() in _BULK_PRECEDENCE or h.get("list-id") \
            or h.get("list-unsubscribe"):
        return Sender(address, "It looks like an automatic or mailing-list email.", "automated")
    allowed_set = {a.strip().lower() for a in allowed if a and a.strip()}
    if address not in allowed_set:
        return Sender(address, f"{address} isn't on your list of people who can give Sentient tasks by email.",
                      "not_allowed")
    return Sender(address)
