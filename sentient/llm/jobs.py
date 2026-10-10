"""One local model job at a time, your chats first (#149).

Two calls to a local Ollama model at once push it off the graphics card and make both crawl, so every local call
(``ollama/`` and ``ollama_chat/`` models: chat, text, JSON and embeddings) takes the one *slot* in
``ModelJobs.slot``. Cloud models never queue.

**Who goes first.** Each call carries a *kind*, read from a context variable, so callers never pass it around:

- ``chat``: a chat reply, everything it runs (tool calls, foreground subagents, the rule check), set by
  ``ModelJobs.chat_turn`` in ``Agent.run_turn``;
- ``interactive`` (the default): anything else a person asked for right now (the Test button, dictation clean-up);
- ``task``: task planning and runs, background subagents;
- ``suggestions``, ``memory``, ``skills``, ``titles``, ``background``: proactivity, memory upkeep and dreaming,
  skill review, chat titles, other service work.

Waiting calls are served by kind, then first come first served. Services set their kind once for everything they
start (``Service.model_kind``); work spawned from a chat that must not count as the chat runs in
``detached(kind)``.

**Background work yields.** Kinds from ``task`` down wait while you chat (a reply is running, or ended less than
``models.background_quiet_s`` ago) and while the computer is on battery (``models.background_on_battery`` off).
Calls already running finish; nothing is cut off.

**Streams hold the slot until they end.** The provider holds it across a whole stream and lets it go just before
the final chunk, so the caller can run tools that call the model again. Cancelling (Stop everything) releases it.

**It can't deadlock.**

- A call made while the same context holds the slot (between the chunks of a stream) does not wait for it.
- Work a chat waits on is part of the chat: it carries the chat's kind and is never deferred while that chat runs.
- Safety valve: when a chat reply is running but has not used the model for ``STALL_S`` (a slow tool, or it waits on
  something that waits on the model), deferred work may go again, still after any chat call.

The scheduler publishes ``model.busy`` (``status()``) whenever what it is doing changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import itertools
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from sentient.llm import power

log = logging.getLogger(__name__)

LOCAL_PREFIXES = frozenset({"ollama", "ollama_chat"})
RANKS = {"chat": 0, "interactive": 1, "task": 2, "suggestions": 3, "memory": 3, "skills": 3, "titles": 3,
         "background": 3}
KINDS = tuple(RANKS)
DEFAULT_KIND = "interactive"
GATED_RANK = 2  # kinds from here down wait while you chat or on battery
STALL_S = 60.0  # a running reply that has not used the model this long stops holding background work back
RECHECK_S = 15.0  # how often deferred work looks at the battery again
NOTIFY_DELAY_S = 0.2  # quick back-to-back changes are published once

_kind: contextvars.ContextVar[str | None] = contextvars.ContextVar("sentient_model_kind", default=None)
_turn: contextvars.ContextVar[_Turn | None] = contextvars.ContextVar("sentient_model_turn", default=None)
_held: contextvars.ContextVar[Slot | None] = contextvars.ContextVar("sentient_model_slot", default=None)


def is_local(model: str | None) -> bool:
    return bool(model) and "/" in model and model.split("/", 1)[0] in LOCAL_PREFIXES  # type: ignore[union-attr]


def current_kind() -> str:
    return _kind.get() or DEFAULT_KIND


@contextlib.contextmanager
def as_kind(kind: str | None) -> Iterator[None]:
    """Model calls inside count as ``kind`` (None: leave it as it is)."""
    if kind is None:
        yield
        return
    token = _kind.set(kind)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):  # reset from another context: an async generator moved tasks
            _kind.reset(token)


def detached(kind: str) -> contextvars.Context:
    """A context for ``asyncio.create_task(..., context=detached(kind))``: work started from here that nobody waits
    on (a memory note after a reply, a task run). Its calls count as ``kind``, not as the chat that started it."""
    ctx = contextvars.copy_context()
    ctx.run(_kind.set, kind)
    ctx.run(_turn.set, None)
    ctx.run(_held.set, None)
    return ctx


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class _Turn:
    """A chat reply in progress; work that carries it is never deferred while it runs."""

    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True


class Slot:
    """What a caller holds while it uses the model. ``release()`` is safe to call more than once."""

    def __init__(self, jobs: ModelJobs | None, kind: str = DEFAULT_KIND, model: str = "", waited: float = 0.0):
        self._jobs = jobs
        self.kind = kind
        self.model = model
        self.waited = waited
        self.started = time.monotonic()
        self.started_at = _now_iso()

    def release(self) -> None:
        jobs, self._jobs = self._jobs, None
        if jobs is not None:
            jobs._release(self)


class _Waiter:
    __slots__ = ("fut", "kind", "model", "rank", "seq", "since", "turn")

    def __init__(self, kind: str, seq: int, model: str, turn: _Turn | None, since: float, fut: asyncio.Future):
        self.kind = kind
        self.rank = RANKS.get(kind, RANKS[DEFAULT_KIND])
        self.seq = seq
        self.model = model
        self.turn = turn
        self.since = since
        self.fut = fut


class ModelJobs:
    """The scheduler for local model calls. One per provider; the app publishes its changes as ``model.busy``."""

    def __init__(
        self,
        settings: Callable[[], Any],
        *,
        on_battery: Callable[[], bool] = power.on_battery,
        clock: Callable[[], float] = time.monotonic,
        on_change: Callable[[dict], None] | None = None,
    ):
        self._settings = settings  # () -> the current ModelsConfig (config can be replaced while running)
        self.on_change = on_change
        self._on_battery = on_battery
        self._clock = clock
        self._waiters: list[_Waiter] = []
        self._current: Slot | None = None
        self._seq = itertools.count()
        self._turns = 0
        self._last_chat_end = float("-inf")
        self._last_chat_use = float("-inf")
        self._timer: asyncio.TimerHandle | None = None
        self._notify: asyncio.TimerHandle | None = None
        self._published: dict | None = None

    # ------------------------------------------------------------------ settings
    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings(), "local_queue", True))

    def _quiet_s(self) -> float:
        return float(getattr(self._settings(), "background_quiet_s", 30))

    def _battery_ok(self) -> bool:
        return bool(getattr(self._settings(), "background_on_battery", False))

    # ------------------------------------------------------------------ the slot
    @contextlib.asynccontextmanager
    async def slot(self, model: str) -> AsyncIterator[Slot]:
        """Hold the local model while the body runs. Cloud models (and the queue turned off) pass straight through."""
        if not self.enabled or not is_local(model):
            yield Slot(None, current_kind(), model)
            return
        held = _held.get()
        if held is not None and held is self._current:
            # this context already holds the slot (a call between a stream's chunks): waiting would wait on itself
            yield Slot(None, current_kind(), model)
            return
        slot = await self._acquire(model)
        token = _held.set(slot)
        try:
            yield slot
        finally:
            slot.release()
            with contextlib.suppress(ValueError):
                _held.reset(token)

    async def _acquire(self, model: str) -> Slot:
        loop = asyncio.get_running_loop()
        w = _Waiter(current_kind(), next(self._seq), model, _turn.get(), self._clock(), loop.create_future())
        self._waiters.append(w)
        self._dispatch()
        try:
            return await w.fut
        except asyncio.CancelledError:
            if w.fut.done() and not w.fut.cancelled():
                w.fut.result().release()  # granted just as it was cancelled: hand it on
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(w)
                self._dispatch()
            raise

    def _release(self, slot: Slot) -> None:
        if self._current is not slot:
            return
        now = self._clock()
        self._current = None
        if slot.kind == "chat":
            self._last_chat_use = now
        log.info("model job done: %s %s after %.1fs", slot.kind, _short(slot.model), now - slot.started)
        self._dispatch()

    def _deferral(self, w: _Waiter, now: float) -> str | None:
        """Why ``w`` must wait although the slot is free: ``"chat"``, ``"battery"`` or None (it may go)."""
        if w.rank < GATED_RANK or (w.turn is not None and w.turn.active):
            return None
        if self._turns and now - self._last_chat_use >= STALL_S:
            return None  # the safety valve: the running reply is not using the model, maybe waiting on this
        if not self._battery_ok() and self._on_battery():
            return "battery"
        if self._turns or now - self._last_chat_end < self._quiet_s():
            return "chat"
        return None

    def _dispatch(self) -> None:
        now = self._clock()
        self._waiters = [w for w in self._waiters if not w.fut.done()]
        if self._current is None and self._waiters:
            ready = [w for w in self._waiters if self._deferral(w, now) is None]
            if ready:
                w = min(ready, key=lambda x: (x.rank, x.seq))
                self._waiters.remove(w)
                slot = Slot(self, w.kind, w.model, now - w.since)
                self._current = slot
                if w.kind == "chat":
                    self._last_chat_use = now
                w.fut.set_result(slot)
                deferred = sum(1 for x in self._waiters if self._deferral(x, now))
                log.info("model job start: %s %s (waited %.1fs; %d more waiting, %d held back)", w.kind,
                         _short(w.model), slot.waited, len(self._waiters) - deferred, deferred)
        self._schedule_recheck(now)
        self._changed()

    def _schedule_recheck(self, now: float) -> None:
        """Wake deferred work when the quiet time ends, the valve opens, or to look at the battery again."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._current is not None or not any(self._deferral(w, now) for w in self._waiters):
            return
        waits = [RECHECK_S]
        if self._turns:
            waits.append(self._last_chat_use + STALL_S - now)
        else:
            waits.append(self._last_chat_end + self._quiet_s() - now)
        with contextlib.suppress(RuntimeError):  # no running loop
            self._timer = asyncio.get_running_loop().call_later(max(0.05, min(waits)), self._dispatch)

    # ------------------------------------------------------------------ chat activity
    @contextlib.contextmanager
    def chat_turn(self) -> Iterator[None]:
        """A chat reply runs inside: its calls go first and background work waits until it has been quiet."""
        turn = _Turn()
        turn_token = _turn.set(turn)
        kind_token = _kind.set("chat")
        self._turns += 1
        self._last_chat_use = self._clock()
        try:
            yield
        finally:
            turn.active = False
            self._turns = max(0, self._turns - 1)
            self._last_chat_end = self._clock()
            for var, token in ((_kind, kind_token), (_turn, turn_token)):
                with contextlib.suppress(ValueError):
                    var.reset(token)  # type: ignore[arg-type]
            with contextlib.suppress(RuntimeError):
                self._dispatch()

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        """``model.busy``: what the local model is doing and what waits for it."""
        now = self._clock()
        cur = self._current
        reasons = [self._deferral(w, now) for w in self._waiters if not w.fut.done()]
        held = [r for r in reasons if r]
        return {
            "busy": cur is not None,
            "job": cur.kind if cur else None,
            "model": cur.model if cur else None,
            "since": cur.started_at if cur else None,
            "waiting": len(reasons) - len(held),
            "deferred": len(held),
            "deferred_reason": ("battery" if "battery" in held else "chat") if held else None,
        }

    def _changed(self) -> None:
        if self.on_change is None or self._notify is not None:
            return
        with contextlib.suppress(RuntimeError):  # no running loop
            self._notify = asyncio.get_running_loop().call_later(NOTIFY_DELAY_S, self._publish)

    def _publish(self) -> None:
        self._notify = None
        state = self.status()
        key = {k: v for k, v in state.items() if k != "since"}
        if key == self._published or self.on_change is None:
            return
        self._published = key
        try:
            self.on_change(state)
        except Exception:
            log.exception("model.busy listener failed")


def _short(model: str) -> str:
    return model.split("/", 1)[-1] if model else ""
