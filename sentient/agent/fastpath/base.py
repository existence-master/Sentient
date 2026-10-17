"""What a language pack for quick commands gives (#150).

A language pack turns one message into a ``Command`` (or None when it is not a plain quick command) and says the
short confirmations and notification texts in its language. Packs never touch the database or the model: the
planner in ``sentient.agent.fastpath`` resolves a command against the reminders (or the Home Assistant devices, #175)
that exist and runs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# what a command is about: "timer", "reminder", "alarm", or "any" (all of them) for listing and cancelling
TIMER, REMINDER, ALARM, ANY = "timer", "reminder", "alarm", "any"
# the kinds of home device a quick command can name (#175): lights (and lamps, bulbs) or switches (and plugs)
LIGHT, SWITCH = "light", "switch"


@dataclass
class Command:
    action: str                 # "timer" | "reminder" | "alarm" | "cancel" | "list" | "home" | "calendar_next" |
                                # "alarm_stop" | "alarm_snooze" (only while an alarm rings, #390)
    seconds: float | None = None  # a timer's length, a reminder or alarm "in 20 minutes", or a snooze's length
    at: datetime | None = None  # a reminder or alarm at a clock time (aware, in the user's timezone)
    message: str = ""           # what to remind about ("call Ravi")
    connector: str = ""         # how the message was asked for: "to", "about", "that" or "" (none)
    label: str = ""             # a timer's or an alarm's name ("pasta", "gym")
    target: str = ANY           # cancel / list: TIMER, REMINDER or ANY
    which: str = ""             # cancel, alarm_stop, alarm_snooze: words naming which one ("pasta", "10 minute",
                                # "stretch", "gym"); home: the device
                                # as the user named it ("living room lights", "bedroom lamp")
    all: bool = False           # cancel: every one of them; home: "all the lights"
    power: str = ""             # home: "on" or "off"
    level: int | None = None    # home: a light's brightness, 1 to 100 percent ("dim the lamp to 30%")
    device: str = ""            # home: LIGHT, SWITCH or "" when no kind was named ("turn off the coffee maker")
    plural: bool = False        # home: several were asked for ("lights")


class Language(Protocol):
    code: str

    def parse(self, text: str, now: datetime, names: Sequence[str] = ()) -> Command | None:
        """The quick command in ``text`` (``now`` is aware, in the user's timezone), or None. ``names``: what the
        assistant is called, so "hey Sentient, ..." is understood."""
        ...

    def ringing(self, text: str, names: Sequence[str] = ()) -> Command | None:
        """While an alarm rings (#390): "stop", "turn it off" (``alarm_stop``) or "snooze", "snooze for 5 minutes"
        (``alarm_snooze``, ``seconds`` when a length was said), or None. Never used when nothing rings."""
        ...

    def alarm_stopped(self, reminders: list[dict]) -> str: ...

    def alarm_snoozed(self, reminder: dict, now: datetime) -> str:
        """What to say after a snooze; ``reminder`` is the new alarm."""
        ...

    def duration(self, seconds: float) -> str: ...

    def matches_duration(self, words: str, seconds: float) -> bool:
        """True when ``words`` ("10 minute") name a length of ``seconds``."""
        ...

    def timer_set(self, seconds: float, label: str) -> str: ...

    def reminder_set(self, reminder: dict, now: datetime) -> str: ...

    def alarm_set(self, reminder: dict, now: datetime) -> str: ...

    def calendar_next(self, events: list[dict], now: datetime, days: int) -> str:
        """What's next on the calendar, from ``gcal_list_events`` events (soonest first, ``now`` aware, in the user's
        timezone); ``days`` is how far ahead was looked."""
        ...

    def cancelled(self, reminders: list[dict]) -> str: ...

    def nothing(self, target: str) -> str: ...

    def listing(self, reminders: list[dict], now: datetime) -> str: ...

    def failed(self, error: str) -> str: ...

    def unconfirmed(self, detail: str) -> str:
        """What to say when the call said it worked but reading it back shows it didn't (#386): ``detail`` is
        "Hallway light still reports off"."""
        ...

    def home_done(self, command: Command, count: int) -> str:
        """What to say after a home command switched ``count`` devices ("Turned off the living room lights.")."""
        ...

    def timer_done(self, reminder: dict) -> tuple[str, str]:
        """(title, body) of the notification when a timer ends."""
        ...

    def reminder_due(self, reminder: dict, late_at: datetime | None) -> tuple[str, str]:
        """(title, body) of a reminder's notification; ``late_at`` is when it was due, when it fires late."""
        ...

    def alarm_due(self, reminder: dict, at: datetime, late: bool) -> tuple[str, str]:
        """(title, body) of an alarm's notification; ``at`` is when it was set for, in the user's timezone."""
        ...
