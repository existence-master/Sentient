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
``models.background_quiet_s`` ago). Calls already running finish; nothing is cut off.

**On battery (#296) only Sentient's own upkeep goes easy.** Tasks you scheduled run on time at any battery level.
Upkeep (``suggestions``, ``memory``, ``skills``, ``titles``, ``background``) on battery goes only while:

- the battery is above ``models.battery_min_percent`` (unless ``models.background_on_battery`` lifts the limits),
  and never at or below ``models.battery_critical_percent``;
- you have not chatted for ``models.battery_idle_s``;
- in short bursts at most every ``models.battery_upkeep_every_s``: a burst stays open while upkeep keeps using the
  model (a gap under ``BURST_GRACE_S``, for at most ``BURST_MAX_S``), so a pass of several calls finishes together.

None of these holds a call longer than ``models.battery_max_wait_s``: then it goes like it would plugged in, and opens
a burst, so the rest of its pass follows instead of waiting the whole time again.

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
GATED_RANK = 2  # kinds from here down wait while you chat
UPKEEP_RANK = 3  # kinds from here down are Sentient's own upkeep: they go easy on battery (#296)
STALL_S = 60.0  # a running reply that has not used the model this long stops holding background work back
RECHECK_S = 15.0  # how often deferred work looks at the battery again
BURST_GRACE_S = 60.0  # on battery, an upkeep burst stays open while upkeep used the model less than this ago
BURST_MAX_S = 300.0  # ... but no longer than this
BATTERY_REASONS = ("battery_low", "battery")  # most telling first
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
        self.burst = False  # upkeep that went on battery: it keeps the burst open until it ends

    def release(self) -> None:
        jobs, self._jobs = self._jobs, None
        if jobs is not None:
            jobs._release(self)


class _Waiter:
    __slots__ = ("fut", "held", "kind", "model", "rank", "seq", "since", "turn", "why")

    def __init__(self, kind: str, seq: int, model: str, turn: _Turn | None, since: float, fut: asyncio.Future):
        self.kind = kind
        self.rank = RANKS.get(kind, RANKS[DEFAULT_KIND])
        self.seq = seq
        self.model = model
        self.turn = turn
        self.since = since
        self.fut = fut
        self.held: str | None = None  # the last reason it was held back for
        self.why: str | None = None  # which battery limit held it last (logged when it changes)


class ModelJobs:
    """The scheduler for local model calls. One per provider; the app publishes its changes as ``model.busy``."""

    def __init__(
        self,
        settings: Callable[[], Any],
        *,
        on_battery: Callable[[], bool] = power.on_battery,
        battery_percent: Callable[[], int | None] = power.percent,
        clock: Callable[[], float] = time.monotonic,
        on_change: Callable[[dict], None] | None = None,
    ):
        self._settings = settings  # () -> the current ModelsConfig (config can be replaced while running)
        self.on_change = on_change
        self._on_battery = on_battery
        self._battery_percent = battery_percent
        self._clock = clock
        self._waiters: list[_Waiter] = []
        self._current: Slot | None = None
        self._seq = itertools.count()
        self._turns = 0
        self._last_chat_end = float("-inf")
        self._last_chat_use = float("-inf")
        self._last_upkeep_use = float("-inf")  # on battery: when upkeep last used the model (a burst is open)
        self._burst_start = float("-inf")  # on battery: when the last upkeep burst began
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

    def _setting(self, name: str, default: float) -> float:
        return float(getattr(self._settings(), name, default))

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
        if slot.burst:
            self._last_upkeep_use = now
        log.info("model job done: %s %s after %.1fs", slot.kind, _short(slot.model), now - slot.started)
        self._dispatch()

    def _deferral(self, w: _Waiter, now: float) -> str | None:
        """Why ``w`` must wait although the slot is free: ``"chat"``, ``"battery"``, ``"battery_low"`` or None."""
        if w.rank < GATED_RANK or (w.turn is not None and w.turn.active):
            return None
        if self._turns and now - self._last_chat_use >= STALL_S:
            # the safety valve: the running reply is not using the model, maybe waiting on this. It comes before the
            # battery holds on purpose: a reply stuck behind held upkeep would otherwise hang until the cap (#296)
            return None
        if w.rank >= UPKEEP_RANK and (hold := self._battery_hold(w, now)):
            return hold
        if self._turns or now - self._last_chat_end < self._quiet_s():
            return "chat"
        return None

    def _battery_hold(self, w: _Waiter, now: float) -> str | None:
        """Why upkeep ``w`` waits on battery (#296): ``"battery_low"``, ``"battery"`` or None. Tasks never get here."""
        hold = self._battery_check(w, now)
        return hold[0] if hold else None

    def _battery_check(self, w: _Waiter, now: float) -> tuple[str, str] | None:
        """``(reason, which limit holds it)`` for upkeep on battery, or None when it may go."""
        if not self._on_battery():
            return None
        if now - w.since >= self._setting("battery_max_wait_s", 7200):
            return None  # the cap: never held longer than this, whatever the battery
        if self._burst_open(now):
            return None  # a burst is under way (maybe one the cap let go): the rest of its pass goes with it
        pct = self._battery_percent()
        if pct is not None and pct <= self._setting("battery_critical_percent", 15):
            return "battery_low", "at or below models.battery_critical_percent"
        if self._battery_ok():
            return None
        if pct is not None and pct <= self._setting("battery_min_percent", 40):
            return "battery", "not above models.battery_min_percent"
        if self._turns or now - self._last_chat_end < self._setting("battery_idle_s", 300):
            return "battery", "chatted less than models.battery_idle_s ago"
        if now - self._burst_start < self._setting("battery_upkeep_every_s", 1800):
            return "battery", "last burst less than models.battery_upkeep_every_s ago"
        return None

    def _burst_open(self, now: float) -> bool:
        return now - self._last_upkeep_use < BURST_GRACE_S and now - self._burst_start < BURST_MAX_S

    def _battery_detail(self) -> str:
        pct = self._battery_percent()
        return f"on battery at {pct}%" if pct is not None else "on battery"

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
                if w.rank >= UPKEEP_RANK and self._on_battery():
                    self._start_burst(w, slot, now)
                w.fut.set_result(slot)
                deferred = sum(1 for x in self._waiters if self._deferral(x, now))
                log.info("model job start: %s %s (waited %.1fs; %d more waiting, %d held back)", w.kind,
                         _short(w.model), slot.waited, len(self._waiters) - deferred, deferred)
        self._log_holds(now)
        self._schedule_recheck(now)
        self._changed()

    def _start_burst(self, w: _Waiter, slot: Slot, now: float) -> None:
        """Upkeep goes on battery: open a burst (or keep the open one going) and say why it went."""
        slot.burst = True
        if w.held in BATTERY_REASONS and slot.waited >= self._setting("battery_max_wait_s", 7200):
            log.info("model job released: %s waited %.0fs %s, the most it may wait (models.battery_max_wait_s)",
                     w.kind, slot.waited, self._battery_detail())
        if not self._burst_open(now):
            self._burst_start = now
            log.info("upkeep burst on battery: %s goes (%s)", w.kind, self._battery_detail())
        self._last_upkeep_use = now

    def _log_holds(self, now: float) -> None:
        """Log when a waiting call starts being held back for the battery, once per change."""
        for w in self._waiters:
            w.held = self._deferral(w, now)
            hold = self._battery_check(w, now) if w.held in BATTERY_REASONS else None
            why = hold[1] if hold else None
            if why and why != w.why:
                log.info("model job held: %s %s (%s, %s: %s; goes within %.0fs at the latest)", w.kind,
                         _short(w.model), w.held, self._battery_detail(), why,
                         max(0.0, w.since + self._setting("battery_max_wait_s", 7200) - now))
            w.why = why

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
            waits.append(self._last_chat_end + self._setting("battery_idle_s", 300) - now)
        cap = self._setting("battery_max_wait_s", 7200)
        waits.extend(w.since + cap - now for w in self._waiters if w.held in BATTERY_REASONS)
        waits.append(self._burst_start + self._setting("battery_upkeep_every_s", 1800) - now)
        soonest = min((x for x in waits if x > 0), default=0.05)
        with contextlib.suppress(RuntimeError):  # no running loop
            self._timer = asyncio.get_running_loop().call_later(max(0.05, soonest), self._dispatch)

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
            "deferred_reason": next((r for r in (*BATTERY_REASONS, "chat") if r in held), None),
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
