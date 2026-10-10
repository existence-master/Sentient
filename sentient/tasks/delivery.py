"""Where a task's notifications go besides the app (docs/API.md section 4, "Where results go").

A task's ``deliver_to`` is one of:

- ``"default"``: paired chats with delivery on, as the ``channels.deliver_*`` toggles say (stored as NULL);
- ``"desktop"``: the app only, never a messaging app;
- ``[{channel, chat_id}]``: only these paired chats. ``{"channel": "whatsapp", "chat_id": "self"}`` is the
  WhatsApp "Message yourself" chat, whatever number is linked.

The app always gets the notification; this only decides the messaging apps.
"""

from __future__ import annotations

from typing import Any

CHANNELS = ("telegram", "discord", "whatsapp")
WHATSAPP_SELF = "self"
MAX_CHATS = 10


def normalize_deliver_to(value: Any) -> str | list[dict]:
    """``"default"``, ``"desktop"`` or a deduplicated list of ``{channel, chat_id}``; ValueError otherwise."""
    if value is None or value == "" or value == "default":
        return "default"
    if value == "desktop":
        return "desktop"
    items = value if isinstance(value, list) else None
    if items is None:
        raise ValueError("deliver_to must be 'default', 'desktop' or a list of chats.")
    out: list[dict] = []
    for item in items:
        fields = item if isinstance(item, dict) else {}
        channel = str(fields.get("channel") or "").strip().lower()
        chat_id = str(fields.get("chat_id") or "").strip()
        if channel not in CHANNELS or not chat_id:
            raise ValueError("Each chat in deliver_to needs a channel (telegram, discord or whatsapp) and a chat_id.")
        chat = {"channel": channel, "chat_id": chat_id}
        if chat not in out:
            out.append(chat)
    if not out:
        return "desktop"  # no chats picked: nothing goes to messaging apps
    if len(out) > MAX_CHATS:
        raise ValueError(f"Pick at most {MAX_CHATS} chats.")
    return out


def stored(value: Any) -> str | list[dict] | None:
    """The column value: NULL for the default."""
    normalized = normalize_deliver_to(value)
    return None if normalized == "default" else normalized


def from_stored(value: Any) -> str | list[dict]:
    """Read back tolerantly: anything unreadable is the default."""
    try:
        return normalize_deliver_to(value)
    except ValueError:
        return "default"
