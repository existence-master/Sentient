"""What a language pack for quick commands gives (#150).

A language pack turns one message into a ``Command`` (or None when it is not a plain quick command) and says the
short confirmations and notification texts in its language. Packs never touch the database or the model: the
planner in ``sentient.agent.fastpath`` resolves a command against the reminders that exist and runs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# what a command is about: "timer", "reminder", or "any" (both) for listing and cancelling
TIMER, REMINDER, ANY = "timer", "reminder", "any"


@dataclass
class Command:
    action: str                 # "timer" | "reminder" | "cancel" | "list"
    seconds: float | None = None  # a timer's length, or a reminder "in 20 minutes"
    at: datetime | None = None  # a reminder at a clock time (aware, in the user's timezone)
    message: str = ""           # what to remind about ("call Ravi")
    connector: str = ""         # how the message was asked for: "to", "about", "that" or "" (none)
    label: str = ""             # a timer's name ("pasta")
    target: str = ANY           # cancel / list: TIMER, REMINDER or ANY
    which: str = ""             # cancel: words naming which one ("pasta", "10 minute", "stretch")
    all: bool = False           # cancel: every one of them


class Language(Protocol):
    code: str

    def parse(self, text: str, now: datetime, names: Sequence[str] = ()) -> Command | None:
        """The quick command in ``text`` (``now`` is aware, in the user's timezone), or None. ``names``: what the
        assistant is called, so "hey Sentient, ..." is understood."""
        ...

    def duration(self, seconds: float) -> str: ...

    def matches_duration(self, words: str, seconds: float) -> bool:
        """True when ``words`` ("10 minute") name a length of ``seconds``."""
        ...

    def timer_set(self, seconds: float, label: str) -> str: ...

    def reminder_set(self, reminder: dict, now: datetime) -> str: ...

    def cancelled(self, reminders: list[dict]) -> str: ...

    def nothing(self, target: str) -> str: ...

    def listing(self, reminders: list[dict], now: datetime) -> str: ...

    def failed(self, error: str) -> str: ...

    def timer_done(self, reminder: dict) -> tuple[str, str]:
        """(title, body) of the notification when a timer ends."""
        ...

    def reminder_due(self, reminder: dict, late_at: datetime | None) -> tuple[str, str]:
        """(title, body) of a reminder's notification; ``late_at`` is when it was due, when it fires late."""
        ...
