"""Undo window for emails, messages and posts (#142, ADR 0025).

A preview catches mistakes before a send; an undo window catches the ones a preview missed. In a chat someone is
watching, a call that sends an email, a message or a post waits ``tools.approvals.undo_seconds`` after every other
check said it may run (Never rules, Read only, the second look, approvals and grants all come first). The agent loop
emits ``undo_window`` (docs/API.md section 10) and the call runs only when the window ends. Pressing Undo in that time
means it never runs; "Send now" ends the window early. Stopping the reply also stops the send.

Which calls wait is decided in code (``delayable``): an email, message or post by its tool name
(``preview.action_kind``), never a purchase and never running code, both of which are deliberate yeses the user just
gave and can't be held back in a way that helps. Only chats where someone can press Undo get a window: the desktop
and messaging apps with buttons (``attended``). Task runs, briefs and other work nobody is watching never wait: the
user approved the task or the call itself, and a window nobody sees only delays it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sentient.agent.preview import action_kind
from sentient.tools.base import Risk, Tool

log = logging.getLogger(__name__)

DELAYED_KINDS = frozenset({"email", "message", "post"})
# where someone can press Undo: the desktop window and messaging apps that show buttons
ATTENDED_CHANNELS = frozenset({"desktop", "telegram", "discord", "whatsapp"})

UNDO, SEND_NOW = "undo", "send_now"
ACTIONS = (UNDO, SEND_NOW)
# how a window ended
SENT, UNDONE, SENT_NOW = "sent", "undone", "sent_now"

UNDONE_MESSAGE = ("NOT SENT. The user pressed Undo before it went out, so nothing was sent. Don't send it again "
                  "unless the user asks.")


def delayable(tool: Tool, name: str, arguments: dict | None, risk: Risk, *, purchase: bool = False) -> bool:
    """True when this call is an email, message or post that may wait in an undo window. Never a purchase, never
    code, never a change that stays inside Sentient."""
    if purchase or Risk(risk) >= Risk.exec or Risk(risk) < Risk.write or tool.internal:
        return False
    return action_kind(name, arguments) in DELAYED_KINDS


def attended(channel: str | None, source: str, unprompted: bool) -> bool:
    """True for a chat reply someone can press Undo in (the desktop or a messaging app with buttons)."""
    return source == "chat" and not unprompted and str(channel or "") in ATTENDED_CHANNELS


@dataclass
class Hold:
    undo_id: str
    call_id: str
    name: str
    session_id: str | None
    channel: str
    seconds: int
    until: str
    future: asyncio.Future = field(repr=False)


class UndoBroker:
    """Waiting sends, by undo id. ``wait`` holds one; ``respond`` ends it with Undo or Send now."""

    def __init__(self) -> None:
        self._holds: dict[str, Hold] = {}

    def open(self, undo_id: str, *, call_id: str, name: str, session_id: str | None, channel: str,
             seconds: int) -> Hold:
        until = (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
        hold = Hold(undo_id=undo_id, call_id=call_id, name=name, session_id=session_id, channel=channel,
                    seconds=seconds, until=until, future=asyncio.get_running_loop().create_future())
        self._holds[undo_id] = hold
        return hold

    async def wait(self, undo_id: str) -> str:
        """Hold the send until its window ends: ``sent`` when time ran out, ``undone`` or ``sent_now`` when the user
        pressed a button. Cancelling the waiting task (Stop) cancels the send too."""
        hold = self._holds.get(undo_id)
        if hold is None:
            return SENT
        try:
            action = await asyncio.wait_for(asyncio.shield(hold.future), timeout=hold.seconds)
        except TimeoutError:
            return SENT
        finally:
            self._holds.pop(undo_id, None)
            if not hold.future.done():
                hold.future.cancel()
        return UNDONE if action == UNDO else SENT_NOW

    def respond(self, undo_id: str, action: str) -> bool:
        """Undo or send now. False when the window already ended or the action is unknown."""
        hold = self._holds.get(undo_id)
        if hold is None or hold.future.done() or action not in ACTIONS:
            return False
        hold.future.set_result(action)
        return True

    def get(self, undo_id: str) -> Hold | None:
        return self._holds.get(undo_id)

    def pending_ids(self) -> list[str]:
        return list(self._holds)
