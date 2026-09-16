"""In-process event bus.

Services publish domain events (task updated, run progress, notification,
integration connected, skill pending...) and the gateway forwards every event
to connected desktop windows over the WebSocket. Nothing here knows about
HTTP; tests can subscribe directly.

Event envelope sent to clients::

    {"type": "task.updated", "data": {...}, "ts": "2026-09-15T10:00:00+00:00"}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, max_queue: int = 1000):
        self._subscribers: set[asyncio.Queue] = set()
        self._max_queue = max_queue

    def publish(self, type_: str, data: Any = None) -> None:
        """Fire-and-forget. Safe to call from any coroutine on the main loop."""
        event = {"type": type_, "data": data, "ts": datetime.now(UTC).isoformat()}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event subscriber queue full; dropping %s", type_)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
