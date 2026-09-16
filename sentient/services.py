"""Base class for long-lived feature services owned by the composition root.

Every feature package (tasks, integrations, proactivity, evolution, voice,
notifications) exposes one Service subclass. ``SentientApp`` constructs them,
calls ``start()`` in dependency order after the agent is ready, and ``stop()``
in reverse order on shutdown.

Services reach shared infrastructure through ``self.app``:
``app.config``, ``app.store``, ``app.llm``, ``app.bus``, ``app.registry``,
``app.memory``, ``app.agent``, ``app.skills``, ``app.workspace``,
``app.notify(...)``, and each other (``app.tasks``, ``app.integrations`` ...).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)


class Service:
    name: str = "service"

    def __init__(self, app: SentientApp):
        self.app = app
        self._loops: list[asyncio.Task] = []

    async def start(self) -> None:
        """Open resources, register tools, start background loops."""

    async def stop(self) -> None:
        for t in self._loops:
            t.cancel()
        for t in self._loops:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._loops.clear()

    def run_every(self, seconds: float, fn: Callable[[], Awaitable[None]], *, name: str, initial_delay: float = 0) -> None:
        """Start a background loop that survives exceptions in ``fn``."""

        async def loop() -> None:
            if initial_delay:
                await asyncio.sleep(initial_delay)
            while True:
                try:
                    await fn()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("%s: periodic job %s failed", self.name, name)
                await asyncio.sleep(seconds)

        self._loops.append(asyncio.create_task(loop(), name=f"{self.name}:{name}"))
