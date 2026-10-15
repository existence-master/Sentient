"""English quick commands (#150): timers, one-off reminders, and listing or cancelling them; with Home Assistant
connected, also lights and switches (#175): "turn off the living room lights", "dim the bedroom lamp to 30%".

The grammar is deliberately narrow. A message must be *only* the command (with "please", "hey Sentient" and the
like around it); anything else, or anything that could mean two things, is left to the model:

- a clock time without AM or PM ("at 6") is ambiguous, unless it is written as 24-hour time ("18:00", "06:30");
- a repeating wish ("every day", "each Monday") is a task, not a one-off reminder;
- "today at 9 AM" when 9 AM has passed, a day name, "tonight" without a time, "in a bit" all go to the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, time, timedelta

from sentient.agent.fastpath.base import ANY, LIGHT, REMINDER, SWITCH, TIMER, Command

code = "en"

MAX_TIMER_S = 24 * 3600
MAX_AHEAD_S = 366 * 24 * 3600

_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day": 86400, "days": 86400,
}
_PHRASES = [  # spoken lengths that are not "<number> <unit>"
    (r"\b(?:an|one) hour and a half\b", "90 minutes"),
    (r"\b(?:a|one) minute and a half\b", "90 seconds"),
    (r"\b(?:a |one )?half (?:an |a )?hour\b", "30 minutes"),
    (r"\b(?:a )?quarter (?:of an |an )?hour\b", "15 minutes"),
]
# a repeating wish is a task (create_task_from_prompt), never a one-off reminder
_REPEATING = re.compile(
    r"\b(?:every|each|daily|weekly|monthly|yearly|hourly|weekdays?|weekends?|always|repeat(?:ing|edly)?)\b", re.I
)
_POLITE_HEAD = [
    r"(?:hey|hi|hello|ok|okay|yo)\s+{names}\b\s*",
    r"{names}\b\s+",
    r"(?:please|kindly)\s+",
    r"(?:can|could|would|will)\s+you(?:\s+please)?\s+",
    r"(?:i\s+(?:want|need|would\s+like)\s+you\s+to|i'd\s+like\s+you\s+to)\s+",
]
_POLITE_TAIL = r"\s+(?:please|thanks|thank\s+you|for\s+me)$"

_VERB = r"(?:set|start|create|make|put\s+on|run|add|begin|give\s+me)"
_TIMER_A = re.compile(
    rf"^(?:{_VERB}(?:\s+me)?\s+)?(?:(?:a|an|the|another|one\s+more)\s+)?(?:(?P<label>[a-z']+)\s+)?timer"
    r"(?:\s+(?:for|of|on|at))?\s+(?P<rest>.+)$", re.I,
)
_TIMER_B = re.compile(
    rf"^(?:{_VERB}(?:\s+me)?\s+)?(?:(?:a|an|another)\s+)?(?P<dur>.+?)[\s-]+timer"
    r"(?:\s+(?:called|named|labell?ed|for)\s+(?P<label>.+))?$", re.I,
)
_REMIND = r"(?:remind\s+me|(?:set|create|add|make)\s+(?:a\s+|me\s+a\s+)?reminder)"
_REMIND_RE = re.compile(rf"^{_REMIND}(?:\s+(?P<rest>.+))?$", re.I)
_WHEN_WORD = r"(?:in|after|at|for|by|on|today|tomorrow|tonight)"
_LINK = re.compile(r"\s+(to|that|about)\s+", re.I)
_CANCEL_VERB = r"(?:cancel|stop|end|turn\s+off|switch\s+off|dismiss|kill|clear|silence)"
_CANCEL_A = re.compile(
    rf"^{_CANCEL_VERB}(?P<all>\s+all(?:\s+of)?)?(?:\s+(?:the|my|that|this|those|these))?(?:\s+(?P<which>.+?))?"
    r"\s+(?P<kind>timers?|reminders?)$", re.I,
)
_CANCEL_B = re.compile(
    rf"^{_CANCEL_VERB}(?:\s+(?:the|my|that|this))?\s+(?P<kind>timer|reminder)\s+(?:to|about|for|called|named|that)\s+"
    r"(?P<which>.+)$", re.I,
)
_LIST = [
    re.compile(
        r"^(?:what|which|show(?:\s+me)?|list|tell\s+me|read(?:\s+me)?)(?:\s+(?:are|is))?(?:\s+all)?\s+(?:my|the)"
        r"(?:\s+(?:active|current|running|upcoming))?\s+(?P<kind>timers?|reminders?)"
        r"(?:\s+(?:(?:are|do\s+i\s+have|i\s+have)\s+)?(?:set|running|left|active|on))?$", re.I,
    ),
    re.compile(
        r"^(?:do\s+i\s+have|have\s+i\s+got|are\s+there|is\s+there|any)(?:\s+(?:any|a))?\s+(?P<kind>timers?|reminders?)"
        r"(?:\s+(?:set|running|on|left))?$", re.I,
    ),
    re.compile(r"^(?:what|which)\s+(?P<kind>timers?|reminders?)\s+(?:do\s+i\s+have|are\s+(?:set|running|on))$", re.I),
    re.compile(r"^how\s+(?:much\s+time|long)(?:\s+(?:is|do\s+i\s+have|have\s+i\s+got))?\s+left"
               r"(?:\s+on\s+(?:my|the)\s+(?P<kind>timer))?$", re.I),
    re.compile(r"^(?:what's|what\s+is)\s+(?:left|the\s+time\s+left|remaining)\s+on\s+(?:my|the)\s+(?P<kind>timer)$",
               re.I),
]
_CLOCK = re.compile(
    r"^(?P<h>\d{1,2})(?:(?:[:.]|\s+)(?P<m>\d{2}))?(?:\s*o'?\s*clock)?\s*(?P<ap>[ap])?(?:\.?\s*m\b\.?)?"
    r"(?:\s+(?P<part>in\s+the\s+morning|in\s+the\s+afternoon|in\s+the\s+evening|at\s+night|tonight))?$", re.I,
)


# ---------------------------------------------------------------------------- text helpers
def _normalize(text: str) -> str:
    s = (text or "").replace(chr(0x2019), "'").replace(chr(0x2018), "'")
    s = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", s)  # "10min" -> "10 min", "6pm" -> "6 pm"
    s = re.sub(r"\s*,\s*", " ", s)
    s = " ".join(s.split())
    return s.strip(" .!?;:")


def _strip_polite(text: str, names: Sequence[str]) -> str:
    alts = "|".join(re.escape(n) for n in dict.fromkeys(n.strip() for n in [*names, "sentient"] if n and n.strip()))
    s = text
    changed = True
    while changed and s:
        changed = False
        for pat in _POLITE_HEAD:
            m = re.match(pat.format(names=f"(?:{alts})"), s, re.I)
            if m and m.end() < len(s):
                s, changed = s[m.end():], True
        m = re.search(_POLITE_TAIL, s, re.I)
        if m and m.start() > 0:
            s, changed = s[: m.start()], True
        s = s.strip(" .!?;:")
    return s


def _number(tokens: list[str], i: int) -> tuple[float | None, int]:
    if i >= len(tokens):
        return None, i
    tok = tokens[i].lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return float(tok), i + 1
    if tok in {"a", "an"}:
        return 1.0, i + 1
    if tok in _SMALL:
        return float(_SMALL[tok]), i + 1
    if "-" in tok:  # "twenty-five"
        head, _, tail = tok.partition("-")
        if head in _TENS and tail in _SMALL and 0 < _SMALL[tail] < 10:
            return float(_TENS[head] + _SMALL[tail]), i + 1
        return None, i
    if tok in _TENS:
        value = _TENS[tok]
        if i + 1 < len(tokens) and tokens[i + 1].lower() in _SMALL and 0 < _SMALL[tokens[i + 1].lower()] < 10:
            return float(value + _SMALL[tokens[i + 1].lower()]), i + 2
        return float(value), i + 1
    return None, i


def _half(m: re.Match) -> str:
    num, _ = _number([m.group(1)], 0)
    return f"{num + 0.5} {m.group(2)}s" if num is not None else m.group(0)


def parse_duration(text: str, *, days: bool = True) -> float | None:
    """Seconds in "10 minutes", "1 hour and 30 minutes", "half an hour", "ninety seconds", "ten-minute"; None when the
    words are anything else. ``days``: whether "2 days" is allowed."""
    s = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", text or "").lower()
    s = re.sub(r"-(?=(?:second|sec|minute|min|hour|hr|day)s?\b)", " ", s)  # "ten-minute"
    s = " " + " ".join(s.split()) + " "
    for pat, repl in _PHRASES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\b(\w+) and a half (second|minute|hour|day)s?\b", _half, s)
    tokens = s.split()
    total, parts, i = 0.0, 0, 0
    while i < len(tokens):
        if tokens[i] in {"and", "plus"} and parts:
            i += 1
            continue
        num, nxt = _number(tokens, i)
        if num is None or nxt >= len(tokens):
            return None
        mult = _UNITS.get(tokens[nxt])
        if mult is None or (mult == 86400 and not days):
            return None
        total += num * mult
        parts += 1
        i = nxt + 1
    return total if parts and total > 0 else None


def _clock(text: str) -> tuple[int, int] | None:
    """(hour, minute) of "6 pm", "6:30 p.m.", "18:00", "06:30", "noon", "seven in the evening"; None when it is not
    a time or the half of the day is unclear ("at 6", "6:30")."""
    s = " ".join((text or "").lower().replace("a.m.", "am").replace("p.m.", "pm").split())
    if s in {"noon", "midday", "12 noon"}:
        return 12, 0
    if s in {"midnight", "12 midnight"}:
        return 0, 0
    words = s.split()
    if words and not words[0][0].isdigit():  # "six pm", "six thirty pm", "seven in the evening"
        hour, nxt = _number(words, 0)
        if hour is None or hour != int(hour):
            return None
        rest = words[nxt:]
        minute: float | None = None
        if rest and not rest[0].startswith(("a", "p", "i", "o")) and rest[0] not in {"at", "tonight"}:
            minute, used = _number(rest, 0)
            if minute is None or minute != int(minute):
                return None
            rest = rest[used:]
        s = f"{int(hour)}" + (f":{int(minute):02d}" if minute is not None else "") + (" " + " ".join(rest) if rest else "")
    m = _CLOCK.match(s)
    if m is None:
        return None
    h, minute_s, ap, part = int(m["h"]), m["m"], (m["ap"] or "").lower(), (m["part"] or "").lower()
    mi = int(minute_s) if minute_s else 0
    if mi > 59 or h > 23:
        return None
    if ap or part:
        if h < 1 or h > 12:
            return None
        pm = ap == "p" or (not ap and part not in {"in the morning"})
        if ap == "a" and part and part != "in the morning":
            return None
        if ap == "p" and part == "in the morning":
            return None
        return (h % 12) + (12 if pm else 0), mi
    # no AM or PM: only 24-hour times are clear
    if h >= 13 or h == 0 or (m["h"].startswith("0") and minute_s is not None):
        return h, mi
    return None


def parse_when(text: str, now: datetime) -> tuple[datetime, float | None] | None:
    """When a reminder is due: ``(at, None)`` for a clock time, ``(at, seconds)`` for "in 20 minutes". ``now`` is aware,
    in the user's timezone. None when it is not clear."""
    s = " ".join((text or "").lower().split())
    m = re.match(r"^(?:in|after)\s+(.+)$", s)
    if m:
        seconds = parse_duration(m.group(1), days=True)
        if seconds is None or seconds > MAX_AHEAD_S:
            return None
        return now + timedelta(seconds=seconds), seconds
    day = None
    m = re.match(r"^(?:on\s+|for\s+)?(today|tomorrow|tonight)\s+(.+)$", s)
    if m:
        day, s = m.group(1), m.group(2)
    m = re.match(r"^(.+?)\s+(today|tomorrow|tonight)$", s)
    if m and day is None:
        s, day = m.group(1), m.group(2)
    s = re.sub(r"^(?:at|by|for|around)\s+", "", s)
    if day == "tonight" and re.fullmatch(r"\d{1,2}(?::\d{2})?", s):
        s += " pm"
    clock = _clock(s)
    if clock is None:
        return None
    hour, minute = clock
    date = now.date() + timedelta(days=1 if day == "tomorrow" else 0)
    at = datetime.combine(date, time(hour, minute), tzinfo=now.tzinfo)
    if day == "tonight" and hour < 12:
        return None
    if at <= now:
        if day in {"today", "tonight", "tomorrow"}:
            return None  # "today at 9 AM" after 9 AM: unclear what they meant
        at += timedelta(days=1)
    return at, None


def _clean_message(text: str) -> str:
    return " ".join((text or "").split()).strip(" .!?;:")


# ---------------------------------------------------------------------------- the commands
def _timer(s: str) -> Command | None:
    m = _TIMER_A.match(s)
    if m:
        label = (m["label"] or "").lower()
        if label in {"new", "quick", "count", "countdown", "kitchen"}:
            label = ""
        rest = m["rest"]
        seconds = parse_duration(rest, days=False)
        if seconds is None:  # "for 10 minutes called pasta" / "for 8 minutes for the pasta"
            for sep in re.finditer(r"\s+(?:called|named|labell?ed|for)\s+", rest, re.I):
                seconds = parse_duration(rest[: sep.start()], days=False)
                if seconds is not None:
                    label = label or _label(rest[sep.end():])
                    break
        if seconds is not None:
            return _timer_command(seconds, label)
    m = _TIMER_B.match(s)
    if m:
        dur = m["dur"]
        dur = re.sub(r"^(?:a|an)\s+", "", dur, flags=re.I) if not re.match(r"^(?:a|an)\s+(?:minute|hour|second)\b", dur, re.I) else dur
        label = _label(m["label"] or "")
        seconds = parse_duration(dur, days=False)
        if seconds is None and " " in dur:  # "five minute pasta timer"
            head, _, last = dur.rpartition(" ")
            seconds = parse_duration(head, days=False)
            label = label or last.lower()
        if seconds is not None:
            return _timer_command(seconds, label)
    return None


def _label(text: str) -> str:
    words = re.sub(r"^(?:the|my|a|an)\s+", "", _clean_message(text), flags=re.I)
    return words[:40] if words and len(words.split()) <= 4 else ""


def _timer_command(seconds: float, label: str) -> Command | None:
    if seconds < 1 or seconds > MAX_TIMER_S:
        return None
    return Command("timer", seconds=round(seconds), label=label)


def _reminder(s: str, now: datetime) -> Command | None:
    m = _REMIND_RE.match(s)
    if m is None:
        return None
    rest = (m["rest"] or "").strip()
    if not rest or _REPEATING.search(rest):
        return None
    # "remind me in 20 minutes [to stretch]"
    if re.match(rf"^{_WHEN_WORD}\b", rest, re.I):
        whole = parse_when(rest, now)
        if whole is not None:
            return _reminder_command(whole, "", "")
        for link in _LINK.finditer(rest):
            when = parse_when(rest[: link.start()], now)
            if when is not None:
                return _reminder_command(when, rest[link.end():], link.group(1).lower())
        return None
    # "remind me to call Ravi at 6 pm"
    m = re.match(r"^(to|that|about)\s+(.+)$", rest, re.I)
    if m is None:
        return None
    connector, body = m.group(1).lower(), m.group(2)
    for split in re.finditer(rf"\s+(?={_WHEN_WORD}\b)", body, re.I):
        when = parse_when(body[split.end():], now)
        if when is not None:
            return _reminder_command(when, body[: split.start()], connector)
    return None


def _reminder_command(when: tuple[datetime, float | None], message: str, connector: str) -> Command | None:
    at, seconds = when
    message = _clean_message(message)
    if connector and not message:
        return None
    return Command("reminder", at=at, seconds=seconds, message=message, connector=connector if message else "")


def _kind(word: str | None) -> str:
    w = (word or "").lower()
    return TIMER if w.startswith("timer") else REMINDER if w.startswith("reminder") else ANY


def _cancel(s: str) -> Command | None:
    m = _CANCEL_B.match(s)
    if m:
        return Command("cancel", target=_kind(m["kind"]), which=_clean_message(m["which"]).lower())
    m = _CANCEL_A.match(s)
    if m is None:
        return None
    plural = m["kind"].lower().endswith("s")
    everything = bool(m["all"])
    if plural and not everything:
        return None  # "turn off reminders" may mean a setting: the model asks
    which = _clean_message(m["which"] or "").lower()
    return Command("cancel", target=_kind(m["kind"]), which=which, all=everything)


def _list(s: str) -> Command | None:
    for pat in _LIST:
        m = pat.match(s)
        if m:
            return Command("list", target=_kind(m.groupdict().get("kind")) if m.groupdict().get("kind") else TIMER)
    return None


# ---------------------------------------------------------------------------- lights and switches (#175)
_ON_OFF_A = re.compile(r"^(?:turn|switch|shut|put)\s+(?P<power>on|off)\s+(?P<target>.+)$", re.I)
_ON_OFF_B = re.compile(r"^(?:turn|switch|shut|put)\s+(?P<target>.+?)\s+(?P<power>on|off)$", re.I)
_ON_OFF_C = re.compile(  # "living room lights off", never a question ("is the porch light on")
    r"^(?!(?:is|are|was|were|what|which|why|how|do|does|did|can|could|should|would|will|when|who|where|leave|keep)\b)"
    r"(?P<target>.+?)\s+(?P<power>on|off)$", re.I,
)
_LEVEL = re.compile(
    r"^(?:set|dim|brighten|turn(?:\s+(?:up|down))?|put)\s+(?P<target>.+?)(?:\s+brightness)?\s+(?:to|at)\s+"
    r"(?P<level>\d{1,3})\s*(?:%|percent|per\s+cent)(?:\s+brightness)?$", re.I,
)
_DEVICES = {
    "light": (LIGHT, False), "lights": (LIGHT, True), "lamp": (LIGHT, False), "lamps": (LIGHT, True),
    "bulb": (LIGHT, False), "bulbs": (LIGHT, True), "switch": (SWITCH, False), "switches": (SWITCH, True),
    "plug": (SWITCH, False), "plugs": (SWITCH, True), "socket": (SWITCH, False), "sockets": (SWITCH, True),
}
# never a quick home command: settings, Sentient's own things, and locks and alarms (they always ask) go to the model
_NOT_HOME = re.compile(r"\b(?:timers?|reminders?|notifications?|alerts?|approvals?|sentient|wi-?fi|bluetooth|"
                       r"computer|laptop|phone|everything|all of it|alarms?|locks?|unlock(?:ed)?)\b", re.I)


def _target(raw: str) -> tuple[str, str, bool, bool] | None:
    """``(which, device, plural, all)`` for a device as named: "the living room lights", "all the lights", "the lights
    in the kitchen", "the bedroom lamp". None when it is not a plain name of a device."""
    t = " ".join(raw.lower().split())
    everything = False
    m = re.match(r"^all(?:\s+of)?(?:\s+(?:the|my|our))?\s+(.+)$", t)
    if m:
        t, everything = m.group(1), True
    t = re.sub(r"^(?:the|my|our)\s+", "", t)
    m = re.match(r"^(?P<dev>.+?)\s+in\s+(?:the\s+|my\s+|our\s+)?(?P<room>.+)$", t)
    if m and m["dev"].split()[-1] in _DEVICES:  # "the lights in the kitchen" -> "kitchen lights"
        t = f"{m['room']} {m['dev']}"
    words = t.split()
    if not words or len(words) > 6 or _NOT_HOME.search(t) or not re.fullmatch(r"[a-z0-9' -]+", t):
        return None
    device, plural = _DEVICES.get(words[-1], ("", False))
    if everything and not plural:
        return None  # "all the lamp"
    if not device and len(words) > 4:
        return None
    return t, device, plural, everything


def _home(s: str) -> Command | None:
    m = _LEVEL.match(s)
    if m:
        found = _target(m["target"])
        level = int(m["level"])
        if found is None or found[1] != LIGHT or not 1 <= level <= 100:
            return None
        which, device, plural, everything = found
        return Command("home", which=which, device=device, plural=plural, all=everything, power="on", level=level)
    for pat in (_ON_OFF_A, _ON_OFF_B, _ON_OFF_C):
        m = pat.match(s)
        if m is None:
            continue
        found = _target(m["target"])
        if found is None or (pat is _ON_OFF_C and not found[1]):
            return None
        which, device, plural, everything = found
        return Command("home", which=which, device=device, plural=plural, all=everything, power=m["power"].lower())
    return None


def parse(text: str, now: datetime, names: Sequence[str] = ()) -> Command | None:
    s = _strip_polite(_normalize(text), names)
    if not s or len(s) > 300 or "\n" in (text or "").strip():
        return None
    for read in (_timer, _cancel, _list):
        cmd = read(s)
        if cmd is not None:
            return cmd
    return _reminder(s, now) or _home(s)


# ---------------------------------------------------------------------------- what Sentient says
def duration(seconds: float) -> str:
    total = round(seconds)
    parts = []
    for size, name in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        n, total = divmod(total, size)
        if n:
            parts.append(f"{n} {name}{'' if n == 1 else 's'}")
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def matches_duration(words: str, seconds: float) -> bool:
    w = re.sub(r"\b(minute|hour|second|day)\b", r"\1s", (words or "").lower())
    found = parse_duration(w, days=True)
    return found is not None and abs(found - seconds) < 1


def _clock_text(at: datetime) -> str:
    return f"{at.hour % 12 or 12}:{at.minute:02d} {'AM' if at.hour < 12 else 'PM'}"


def _day_text(at: datetime, now: datetime) -> str:
    days = (at.date() - now.date()).days
    if days == 0:
        return f"at {_clock_text(at)}"
    if days == 1:
        return f"tomorrow at {_clock_text(at)}"
    if days < 7:
        return f"on {at:%A} at {_clock_text(at)}"
    return f"on {at:%A} {at.day} {at:%B} at {_clock_text(at)}"


def _yours(text: str) -> str:
    """The user's words said back to them: "call my mom" -> "call your mom"."""
    return re.sub(r"\bmy\b", "your", re.sub(r"\bme\b", "you", text, flags=re.I), flags=re.I)


def _what(reminder: dict) -> str:
    message = str(reminder.get("message") or "")
    if not message:
        return ""
    connector = str(reminder.get("connector") or "to")
    return f"{connector} {_yours(message)}"


def _local(reminder: dict, now: datetime) -> datetime:
    at = reminder.get("fire_at")
    if isinstance(at, str):
        at = datetime.fromisoformat(at)
    return at.astimezone(now.tzinfo) if isinstance(at, datetime) else now


def timer_set(seconds: float, label: str) -> str:
    if label:
        return f"{label[:1].upper() + label[1:]} timer set for {duration(seconds)}."
    return f"Timer set for {duration(seconds)}."


def reminder_set(reminder: dict, now: datetime) -> str:
    what = _what(reminder)
    if reminder.get("duration_s"):
        when = f"in {duration(float(reminder['duration_s']))}"
    else:
        when = _day_text(_local(reminder, now), now)
    return f"OK, I'll remind you {what + ' ' if what else ''}{when}."


def length(seconds: float) -> str:
    """A length said before a noun: "10 minute", "1 hour 30 minute"."""
    total = round(seconds)
    parts = []
    for size, name in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        n, total = divmod(total, size)
        if n:
            parts.append(f"{n} {name}")
    return " ".join(parts) or "0 second"


def name(reminder: dict) -> str:
    """ "pasta timer", "10 minute timer", "reminder to call Ravi"."""
    if reminder.get("kind") == TIMER:
        label = str(reminder.get("label") or "")
        return f"{label} timer" if label else f"{length(float(reminder.get('duration_s') or 0))} timer"
    what = _what(reminder)
    return f"reminder {what}" if what else "reminder"


def cancelled(reminders: list[dict]) -> str:
    if len(reminders) == 1:
        return f"Cancelled your {name(reminders[0])}."
    timers = sum(1 for r in reminders if r.get("kind") == TIMER)
    others = len(reminders) - timers
    if not others:
        return f"Cancelled your {timers} timers."
    if not timers:
        return f"Cancelled your {others} reminders."
    return f"Cancelled {timers} timer{'s' if timers > 1 else ''} and {others} reminder{'s' if others > 1 else ''}."


def nothing(target: str) -> str:
    if target == TIMER:
        return "You don't have any timers running."
    if target == REMINDER:
        return "You don't have any reminders set."
    return "You don't have any timers or reminders set."


def listing(reminders: list[dict], now: datetime) -> str:
    timers = [r for r in reminders if r.get("kind") == TIMER]
    others = [r for r in reminders if r.get("kind") != TIMER]
    said = []
    if timers:
        items = []
        for r in timers[:5]:
            left = max(0.0, (_local(r, now) - now).total_seconds())
            label = str(r.get("label") or "") or duration(float(r.get("duration_s") or 0))
            items.append(f"{label}, {duration(left if left >= 60 else round(left)) if left >= 1 else 'ending now'}"
                         f"{' left' if left >= 1 else ''}")
        said.append(f"You have {len(timers)} timer{'s' if len(timers) > 1 else ''}: " + _join(items) + ".")
    if others:
        items = [f"{_what(r) or 'a reminder'} {_day_text(_local(r, now), now)}".strip() for r in others[:5]]
        head = "You have" if not said else "And"
        said.append(f"{head} {len(others)} reminder{'s' if len(others) > 1 else ''}: " + _join(items) + ".")
    return " ".join(said)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else "; ".join(items[:-1]) + f"; and {items[-1]}"


def home_done(command: Command, count: int) -> str:
    if command.all and command.which in {"lights", "lamps", "bulbs", "switches", "plugs", "sockets"}:
        what = f"all the {command.which}"
    else:
        what = f"the {_yours(command.which)}"
    if command.level is not None:
        return f"Set {what} to {command.level}%."
    return f"Turned {command.power} {what}."


def failed(error: str) -> str:
    return f"I couldn't do that: {error}" if error else "I couldn't do that."


def timer_done(reminder: dict) -> tuple[str, str]:
    label = str(reminder.get("label") or "")
    if label:
        return "Timer done", f"Your {label} timer ({duration(float(reminder.get('duration_s') or 0))}) is done."
    return "Timer done", f"Your {name(reminder)} is done."


def reminder_due(reminder: dict, late_at: datetime | None) -> tuple[str, str]:
    message = str(reminder.get("message") or "").strip()
    body = (message[:1].upper() + message[1:]) if message else "You asked me to remind you now."
    if late_at is not None:
        body += f" (It was due at {_clock_text(late_at)}, while Sentient wasn't running.)"
    return "Reminder", body
