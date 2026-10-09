"""Base class for long-lived feature services owned by the composition root.

Every feature package (tasks, integrations, proactivity, evolution, voice,
notifications) exposes one Service subclass. ``SentientApp`` constructs them,
calls ``start()`` in dependency order after the agent is ready, and ``stop()``
in reverse order on shutdown.

Services reach shared infrastructure through ``self.app``:
``app.config``, ``app.store``, ``app.llm``, ``app.bus``, ``app.registry``,
``app.memory``, ``app.agent``, ``app.skills``, ``app.workspace``,
``app.notify(...)``, and each other (``app.tasks``, ``app.integrations`` ...).

Stop everything (``app.stop_all``) calls ``halt()`` on every service: cancel in-flight work and
leave the service ready for ``app.resume``. A service with ``pause_on_stop`` also skips its
``run_every`` jobs while Sentient is stopped, and ``halt()`` cancels the job that is running.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)


class Service:
    name: str = "service"
    pause_on_stop: bool = False  # skip run_every jobs while Sentient is stopped (app.stop_all)

    def __init__(self, app: SentientApp):
        self.app = app
        self._loops: list[asyncio.Task] = []
        self._jobs: set[asyncio.Task] = set()  # run_every jobs running right now

    async def start(self) -> None:
        """Open resources, register tools, start background loops."""

    async def stop(self) -> None:
        for t in self._loops:
            t.cancel()
        for t in self._loops:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._loops.clear()

    def paused(self) -> bool:
        """True while Sentient is stopped (Stop everything) and this service pauses with it."""
        return self.pause_on_stop and getattr(self.app, "stopped", False) is True

    async def halt(self) -> int:
        """Stop everything: cancel in-flight work. Returns how many jobs were cancelled."""
        if not self.pause_on_stop:
            return 0
        return await cancel_tasks(self._jobs)

    async def _run_job(self, fn: Callable[[], Awaitable[None]], name: str) -> None:
        """One run_every job of a pausable service: skipped while stopped, in its own task so halt() can cancel it."""
        if self.paused():
            return
        job = asyncio.create_task(fn(), name=f"{self.name}:{name}:job")  # type: ignore[arg-type]
        self._jobs.add(job)
        try:
            await asyncio.wait({job})
        except asyncio.CancelledError:
            job.cancel()
            await asyncio.wait({job})
            raise
        finally:
            self._jobs.discard(job)
        if not job.cancelled() and job.exception() is not None:
            log.error("%s: periodic job %s failed", self.name, name, exc_info=job.exception())

    def run_every(self, seconds: float, fn: Callable[[], Awaitable[None]], *, name: str, initial_delay: float = 0) -> None:
        """Start a background loop that survives exceptions in ``fn``."""

        async def loop() -> None:
            if initial_delay:
                await asyncio.sleep(initial_delay)
            while True:
                if self.pause_on_stop:
                    await self._run_job(fn, name)
                else:
                    try:
                        await fn()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("%s: periodic job %s failed", self.name, name)
                await asyncio.sleep(seconds)

        self._loops.append(asyncio.create_task(loop(), name=f"{self.name}:{name}"))


async def cancel_tasks(tasks: Iterable[asyncio.Task], timeout: float = 5.0) -> int:
    """Cancel the running ones among ``tasks`` (never the caller) and wait up to ``timeout`` s for them."""
    me = asyncio.current_task()
    running = [t for t in list(tasks) if not t.done() and t is not me]
    for t in running:
        t.cancel()
    if running:
        await asyncio.wait(running, timeout=timeout)
    return len(running)
