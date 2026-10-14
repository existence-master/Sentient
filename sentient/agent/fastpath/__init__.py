"""Quick commands (#150): timers and reminders done at once, without the model.

``Agent.run_turn`` asks ``plan`` first. When a message is *only* a plain timer or reminder command ("set a timer for
10 minutes", "remind me at 6 pm to call Ravi", "cancel the pasta timer", "what timers do I have"), the plan holds the
tool calls to make (the same ``reminders`` tools the model uses) and the confirmation to say, and the turn runs them
with no model call: it takes milliseconds on every surface (chat, voice, Telegram, Discord, WhatsApp). Anything else,
or anything that could mean two things, returns None and goes to the model unchanged.

A quick command never sends, buys or deletes anything: it only starts, lists or cancels Sentient's own timers and
reminders. It runs only when its tool would run without asking (approvals mode, "Ask" and "Never" rules); otherwise
the model handles the message and the usual approval card appears.

Languages: one module per language (``en.py``), see ``base.Language``. The packs for ``assistant.language`` and
English are tried, in that order. Turned off with ``chat.quick_commands``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sentient.agent.fastpath import en
from sentient.agent.fastpath.base import ANY, REMINDER, TIMER, Command, Language
from sentient.llm.provider import ToolCall
from sentient.store.db import new_id

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

__all__ = ["ANY", "LANGUAGES", "REMINDER", "TIMER", "Command", "Language", "QuickPlan", "language", "parse", "plan"]

LANGUAGES: dict[str, Language] = {"en": en}  # type: ignore[dict-item]


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
    if not config.chat.quick_commands or reminders is None or not (text or "").strip():
        return None
    tz = get_tz(user_timezone_name(config.assistant.timezone))
    local = (now or datetime.now(UTC)).astimezone(tz)
    found = parse(text, local, lang=config.assistant.language, names=[config.assistant.name])
    if found is None:
        return None
    cmd, pack = found
    out = QuickPlan(cmd, pack, local)
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
    elif cmd.action == "list":
        out.calls = [_call("list_reminders", kind=cmd.target)]
        out.reply = lambda results: _listing(pack, results[0][0], cmd.target, local)
    elif cmd.action == "cancel":
        active = [r for r in await reminders.list() if cmd.target == ANY or r["kind"] == cmd.target]
        if not active:
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


def _listing(pack: Language, res: Any, target: str, now: datetime) -> str:
    items = res.get("reminders") if isinstance(res, dict) else None
    items = [r for r in items or [] if isinstance(r, dict)]
    return pack.listing(items, now) if items else pack.nothing(target)
