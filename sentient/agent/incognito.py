"""Incognito chats (#154): deleted when they are closed.

The window deletes an incognito chat when the user leaves it. This service covers the rest: chats left over when
Sentient was closed are deleted at the next start (and at shutdown), and a chat with no new message for
``chat.incognito_delete_after_minutes`` is deleted by a sweep. What the engine refuses to learn from an incognito chat
is enforced where it happens (``Agent.run_turn``, the memory and skill tools, summaries, the user model, skill review).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sentient.services import Service

log = logging.getLogger(__name__)

SWEEP_EVERY_S = 60


class IncognitoSweeper(Service):
    name = "incognito"

    async def start(self) -> None:
        await self.purge()  # chats from an earlier run were closed with it
        if self.app.enable_background:
            self.run_every(SWEEP_EVERY_S, self.sweep, name="sweep", initial_delay=SWEEP_EVERY_S)

    async def stop(self) -> None:
        await super().stop()  # the final purge runs in SentientApp.stop, after replies and background work end

    async def sweep(self, now: datetime | None = None) -> list[str]:
        """Delete incognito chats with no new message for ``chat.incognito_delete_after_minutes``."""
        minutes = self.app.config.chat.incognito_delete_after_minutes
        cutoff = (now or datetime.now(UTC)) - timedelta(minutes=minutes)
        return await self.purge(cutoff.isoformat())

    async def purge(self, idle_before: str | None = None, *, skip_running: bool = True) -> list[str]:
        """Delete incognito chats (only those idle since ``idle_before`` when given). A chat with a reply that hasn't
        finished writing is left for the next sweep unless ``skip_running`` is off. Returns the deleted ids."""
        agent = self.app.agent
        deleted: list[str] = []
        for sid in await self.app.store.incognito_sessions(idle_before):
            if skip_running and agent is not None and agent.is_busy(sid):
                continue
            await self.app.delete_chat(sid, reason="incognito")
            deleted.append(sid)
        if deleted:
            log.info("deleted %d incognito chat(s)", len(deleted))
        return deleted
