"""Quick commands (#150, #359): timers, reminders, alarms, "what's next on my calendar" and, with Home Assistant
connected, lights (#175), done at once without the model.

``Agent.run_turn`` asks ``plan`` first. When a message is *only* a plain timer or reminder command ("set a timer for
10 minutes", "remind me at 6 pm to call Ravi", "cancel the pasta timer", "what timers do I have"), the plan holds the
tool calls to make (the same ``reminders`` tools the model uses) and the confirmation to say, and the turn runs them
with no model call: it takes milliseconds on every surface (chat, voice, Telegram, Discord, WhatsApp). Anything else,
or anything that could mean two things, returns None and goes to the model unchanged. While a Sentient alarm rings,
"stop", "turn it off" and "snooze" (#390) stop or snooze it (``stop_alarm``, ``snooze_alarm``); with nothing ringing
they are not quick commands.

A quick command never sends, buys or deletes anything: it only starts, lists or cancels Sentient's own timers,
reminders and alarms, reads the connected calendar (``gcal_list_events``; with no calendar connected the model
answers), or turns lights and switches on and off ("turn off the living room lights", "dim the bedroom lamp to 30%")
when the words name exactly which ones (``IntegrationManager.home_devices``). Locks, alarms, garage doors and anything
that sounds like one never go through a home command. It runs only when its tool would run without asking
(approvals mode, "Ask" and "Never" rules); a home command is the user's own words for exactly that change, so in
approvals mode "ask" those words are the yes (``ApprovalBroker.quick_runs``, ``QuickPlan.asked``). Otherwise the model
handles the message and the usual approval card appears.

Languages: one module per language (``en.py``), see ``base.Language``. The packs for ``assistant.language`` and
English are tried, in that order. Turned off with ``chat.quick_commands``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sentient.agent.fastpath import en
from sentient.agent.fastpath.base import ALARM, ANY, REMINDER, TIMER, Command, Language
from sentient.llm.provider import ToolCall
from sentient.store.db import new_id

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

__all__ = ["ALARM", "ANY", "CALENDAR_DAYS", "LANGUAGES", "REMINDER", "TIMER", "Command", "Language", "QuickPlan", "language", "parse", "plan"]

LANGUAGES: dict[str, Language] = {"en": en}  # type: ignore[dict-item]
CALENDAR_DAYS = 7  # "what's next on my calendar" looks this far ahead
CALENDAR_PLUGIN, CALENDAR_TOOL = "gcalendar", "gcal_list_events"


def language(code: str | None) -> Language:
    """The pack for a BCP-47 code ("en-GB" -> English), English when there is none."""
    return LANGUAGES.get(str(code or "en").split("-")[0].lower()) or LANGUAGES["en"]


def _packs(code: str | None) -> list[Language]:
    first = language(code)
    return [first] if first is LANGUAGES["en"] else [first, LANGUAGES["en"]]


def parse(text: str, now: datetime, *, lang: str | None = "en", names: Sequence[str] = ()) -> tuple[Command, Language] | None:
    """The quick command in ``text`` and the language it was said in, or None."""
    for pack in _packs(lang):
        cmd = pack.parse(text, now, names)
        if cmd is not None:
            return cmd, pack
    return None


@dataclass
class QuickPlan:
    command: Command
    language: Language
    now: datetime
    calls: list[ToolCall] = field(default_factory=list)
    # what to say, from each call's (result, is_error); with no calls it says ``said``
    reply: Callable[[list[tuple[Any, bool]]], str] | None = None
    said: str = ""
    # the user's own words asked for exactly these calls (a home command): approvals mode "ask" counts them as the yes
    # for a change (``ApprovalBroker.quick_runs``)
    asked: bool = False

    def answer(self, outcomes: list[tuple[Any, bool]]) -> str:
        for res, is_error in outcomes:
            error = res.get("error") if isinstance(res, dict) else None
            if is_error or error:
                return self.language.failed(str(error or res))
        return self.reply(outcomes) if self.reply is not None else self.said


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"quick_{new_id()[:12]}", name=name, arguments=arguments)


def _reminder_of(res: Any) -> dict:
    return res.get("reminder") if isinstance(res, dict) and isinstance(res.get("reminder"), dict) else {}


def _matches(reminder: dict, which: str, pack: Language) -> bool:
    which = " ".join(which.lower().split())
    for text in (reminder.get("label"), reminder.get("message")):
        if text and which in str(text).lower():
            return True
    seconds = reminder.get("duration_s")
    return bool(seconds) and pack.matches_duration(which, float(seconds))


async def plan(app: SentientApp, text: str, *, now: datetime | None = None) -> QuickPlan | None:
    """What to do for ``text`` without the model, or None to let the model answer."""
    from sentient.tasks.schedule import get_tz, user_timezone_name

    config = app.config
    reminders = getattr(app, "reminders", None)
    if not config.chat.quick_commands or not (text or "").strip():
        return None
    tz = get_tz(user_timezone_name(config.assistant.timezone))
    local = (now or datetime.now(UTC)).astimezone(tz)
    if reminders is not None:
        ring = await _ringing_plan(app, text, local)
        if ring is not None:
            return ring
    found = parse(text, local, lang=config.assistant.language, names=[config.assistant.name])
    if found is None:
        return None
    cmd, pack = found
    if cmd.action == "home":
        return await _home(app, cmd, pack, local)
    out = QuickPlan(cmd, pack, local)
    if cmd.action == "calendar_next":
        return await _calendar_plan(app, out)
    if reminders is None:
        return None
    if cmd.action == "timer":
        out.calls = [_call("set_timer", seconds=int(cmd.seconds or 0), label=cmd.label)]
        out.reply = lambda results: pack.timer_set(float(_reminder_of(results[0][0]).get("duration_s") or cmd.seconds or 0),
                                                   cmd.label)
    elif cmd.action == "reminder":
        args: dict[str, Any] = {"message": cmd.message, "connector": cmd.connector}
        if cmd.seconds is not None:
            args["in_minutes"] = cmd.seconds / 60
        elif cmd.at is not None:
            args["at"] = cmd.at.isoformat()
        out.calls = [_call("set_reminder", **args)]
        out.reply = lambda results: pack.reminder_set(_reminder_of(results[0][0]), local)
    elif cmd.action == "alarm":
        alarm: dict[str, Any] = {"label": cmd.label}
        if cmd.seconds is not None:
            alarm["in_minutes"] = cmd.seconds / 60
        elif cmd.at is not None:
            alarm["at"] = cmd.at.isoformat()
        out.calls = [_call("set_alarm", **alarm)]
        out.reply = lambda results: pack.alarm_set(_reminder_of(results[0][0]), local)
    elif cmd.action == "list":
        out.calls = [_call("list_reminders", kind=cmd.target)]
        out.reply = lambda results: _listing(pack, results[0][0], cmd.target, local)
    elif cmd.action == "cancel":
        active = [r for r in await reminders.list() if cmd.target == ANY or r["kind"] == cmd.target]
        if not active:
            if cmd.target == ALARM:
                return None  # "turn off the alarm" with none set may mean a home alarm: the model asks
            out.said = pack.nothing(cmd.target)
            return out
        chosen = [r for r in active if _matches(r, cmd.which, pack)] if cmd.which else active
        if not chosen or (len(chosen) > 1 and not cmd.all):
            return None  # which one? the model asks (it has list_reminders and cancel_reminder)
        out.calls = [_call("cancel_reminder", reminder_id=r["id"]) for r in chosen]
        out.reply = lambda results: pack.cancelled([_reminder_of(res) for res, _ in results])
    else:
        return None
    return out


async def _ringing_plan(app: SentientApp, text: str, now: datetime) -> QuickPlan | None:
    """ "stop" or "snooze" while a Sentient alarm rings (#390). With nothing ringing these words are not a quick
    command: "stop the alarm" may mean a house alarm (#175), so the usual parse (and then the model) has them."""
    config = app.config
    for pack in _packs(config.assistant.language):
        cmd = pack.ringing(text, [config.assistant.name])
        if cmd is not None:
            break
    else:
        return None
    ringing = await app.reminders.ringing(now.astimezone(UTC))
    if not ringing or (cmd.which and not any(_matches(r, cmd.which, pack) for r in ringing)):
        return None
    out = QuickPlan(cmd, pack, now)
    if cmd.action == "alarm_stop":
        out.calls = [_call("stop_alarm", label=cmd.which) if cmd.which else _call("stop_alarm")]
        out.reply = lambda results: pack.alarm_stopped(_list_of(results[0][0], "stopped"))
        return out
    minutes = cmd.seconds / 60 if cmd.seconds else config.chat.alarm_snooze_minutes
    out.calls = [_call("snooze_alarm", minutes=int(minutes) if float(minutes).is_integer() else round(minutes, 2))]
    out.reply = lambda results: pack.alarm_snoozed(_reminder_of(results[0][0]), now)
    return out


def _list_of(res: Any, key: str) -> list[dict]:
    items = res.get(key) if isinstance(res, dict) else None
    return [i for i in items or [] if isinstance(i, dict)]


async def _home(app: SentientApp, cmd: Command, pack: Language, now: datetime) -> QuickPlan | None:
    """Lights and switches through Home Assistant (#175), when it is connected and the words name exactly which."""
    integrations = getattr(app, "integrations", None)
    find = getattr(integrations, "home_devices", None)
    if find is None:
        return None
    found = await find(cmd.which, device=cmd.device, plural=cmd.plural, everywhere=cmd.all)
    if not found or (cmd.level is not None and any(e.get("domain") != "light" for e in found)):
        return None
    ids = ",".join(str(e["entity_id"]) for e in found)
    if cmd.level is not None:
        call = _call("home_assistant_turn_on", entity_id=ids, brightness_pct=cmd.level)
    else:
        call = _call(f"home_assistant_turn_{'on' if cmd.power == 'on' else 'off'}", entity_id=ids)
    out = QuickPlan(cmd, pack, now, calls=[call], asked=True)
    out.reply = lambda results: pack.home_done(cmd, len(found))
    return out


async def _calendar_plan(app: SentientApp, out: QuickPlan) -> QuickPlan | None:
    """Read what's next from the connected calendar; None (the model answers) when no calendar is connected."""
    integrations = getattr(app, "integrations", None)
    if integrations is None or app.registry.get(CALENDAR_TOOL) is None:
        return None
    try:
        if not await integrations.is_connected(CALENDAR_PLUGIN):
            return None
    except Exception:
        return None
    pack, now = out.language, out.now
    out.calls = [_call(CALENDAR_TOOL, max_results=10)]  # from now to CALENDAR_DAYS ahead, soonest first
    out.reply = lambda results: pack.calendar_next(_events(results[0][0]), now, CALENDAR_DAYS)
    return out


def _events(res: Any) -> list[dict]:
    items = res.get("events") if isinstance(res, dict) else None
    return [e for e in items or [] if isinstance(e, dict)]


def _listing(pack: Language, res: Any, target: str, now: datetime) -> str:
    items = res.get("reminders") if isinstance(res, dict) else None
    items = [r for r in items or [] if isinstance(r, dict)]
    return pack.listing(items, now) if items else pack.nothing(target)
