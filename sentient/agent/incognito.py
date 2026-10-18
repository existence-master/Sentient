"""Incognito chats (#154): deleted when they are closed.

The window deletes an incognito chat when the user leaves it. This service covers the rest: chats left over when
Sentient was closed are deleted at the next start (and at shutdown), and a chat with no new message for
``chat.incognito_delete_after_minutes`` is deleted by a sweep. What the engine refuses to learn from an incognito chat
is enforced where it happens (``Agent.run_turn``, the memory and skill tools, summaries, the user model, skill review).

Long tool results an incognito chat (or a helper it started) read are saved in its own folder,
``files/outputs/incognito-<chat id>/`` (``outputs_folder``), and that folder goes with the chat (#291): those files are
incidental, nobody asked for them. Files it wrote on request and tasks it created stay. A folder whose chat is already
gone (Sentient stopped in between) is removed at the next start.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta

from sentient import paths
from sentient.agent import selections
from sentient.services import Service

log = logging.getLogger(__name__)

SWEEP_EVERY_S = 60
OUTPUTS_PREFIX = "incognito-"


def outputs_folder(session_id: str) -> str:
    """Where an incognito chat's long tool results go, relative to the files folder."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:80] or "chat"
    return f"outputs/{OUTPUTS_PREFIX}{safe}"


def delete_outputs(session_id: str) -> bool:
    """Remove the long tool results an incognito chat saved. True when there was a folder."""
    folder = paths.files_dir() / outputs_folder(session_id)
    if not folder.is_dir():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def stray_outputs(live: set[str]) -> list[str]:
    """Incognito output folders whose chat is gone, as names under ``files/outputs``."""
    root = paths.files_dir() / "outputs"
    if not root.is_dir():
        return []
    keep = {outputs_folder(sid).rsplit("/", 1)[-1] for sid in live}
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(OUTPUTS_PREFIX) and p.name not in keep)


class IncognitoSweeper(Service):
    name = "incognito"

    async def start(self) -> None:
        await self.purge()  # chats from an earlier run were closed with it
        await self.remove_stray_outputs()
        try:  # selected text whose chat is gone (#332)
            await selections.remove_strays(self.app.store)
        except OSError:
            log.exception("removing selected text no chat has failed")
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

    async def remove_stray_outputs(self) -> list[str]:
        """Remove output folders of incognito chats that no longer exist. Returns the removed folder names."""
        live = set(await self.app.store.incognito_sessions())
        stray = await asyncio.to_thread(stray_outputs, live)
        root = paths.files_dir() / "outputs"
        for name in stray:
            await asyncio.to_thread(shutil.rmtree, root / name, True)
        if stray:
            log.info("removed %d leftover incognito output folder(s)", len(stray))
        return stray
