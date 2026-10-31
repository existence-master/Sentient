"""Check the outcome, not the model's opinion of it (#165).

After a call that should change something in the world (write a file, send an email or a message, create a calendar
event) has run, code checks what really happened, never a model:

1. the tool's result says it worked: no ``error``, no ``ok: false``, no failed status, and no list of recipients or
   items that failed (``result_problem``), also inside an MCP server's text or structured result;
2. when the tool declares ``Tool.verify(arguments, result, ctx)``, the effect is read back: the file holds the
   content, the email is in the Sent folder, the event exists.

Each check is ``{tool, app, kind, action, ok, detail, how, targets, missed}``. ``finalize`` decides which failures
still stand at the end (a later call that did the same thing again, or covered what was missed, settles one) and
writes the plain sentence the user sees: "Sentient said it sent the email, but mailer reported a problem: ...".
A task run with a failure that still stands ends ``completed_with_errors`` (shown as Needs attention), never as done.

Which calls are checked: tools that declare ``Tool.outcome`` (a kind below) or ``Tool.verify``, and every MCP tool
that is not a look-up. A call the user declined or undid, a try run's noted call and a call handed to the user are
never checked: nothing was meant to happen.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

VERIFY_TIMEOUT_S = 15.0
MAX_DETAIL = 300

# kind: (what it tries to do, what it says it did, short label once checked)
KINDS: dict[str, tuple[str, str, str]] = {
    "email": ("send the email", "sent the email", "email sent"),
    "message": ("send the message", "sent the message", "message sent"),
    "file": ("write the file", "saved the file", "file written"),
    "event": ("create the event", "created the event", "event created"),
    "home": ("run the home action", "ran the home action", "home action done"),
}

# the verbs a final answer uses to say an action happened, by kind ("sent the email", "turned on the lights")
_CLAIM_VERBS: dict[str, str] = {
    "email": r"sent|emailed|delivered|mailed",
    "message": r"sent|posted|messaged|delivered",
    "file": r"saved|written|wrote|created|stored",
    "event": r"created|added|scheduled|booked",
    "home": r"turned|switched|set|done|ran",
}

# words in a final answer that claim the action happened, by kind
_CLAIMS: dict[str, re.Pattern] = {
    kind: re.compile(rf"\b(?:{verbs})\b", re.I) for kind, verbs in _CLAIM_VERBS.items()
}
_CLAIM_ANY = re.compile(r"\b(done|completed|finished|success(?:fully)?|sent|saved|created|updated)\b", re.I)
_BROWSER_CLICK_CLAIM = re.compile(r"\b(?:clicked|was clicked)\b", re.I)
_NEGATED_BROWSER_CLICK = re.compile(
    r"\b(?:not|didn't|hasn't|wasn't|haven't|never|nothing)\b[^.!?]{0,50}\bclicked\b", re.I
)
_OTHER_PERSON_BROWSER_CLICK = re.compile(
    r"\b(?:you|he|she|they|someone|anyone|(?:my|your|his|her|their)\s+\w+)\s+"
    r"(?:have\s+|had\s+|just\s+)?clicked\b", re.I
)
_ASSISTANT_BROWSER_CLICK = re.compile(
    r"\b(?:i(?:'ve)?|we|sentient)\s+(?:(?:have|had|just|only|also|then|already|recently|finally|even|really|\w+ly)\s+){0,2}clicked\b", re.I
)
_HYPOTHETICAL_BROWSER_CLICK = re.compile(
    r"\b(?:if|would|could|might|may)\b[^.!?]{0,50}\b(?:clicked|was clicked|were clicked)\b", re.I
)

_ACTION_SUBJECT = r"(?:i(?:'ve)?|we|sentient)"
_ACTION_MODIFIERS = r"(?:have|had|just|only|also|then|already|recently|finally|even|really|\w+ly)"
# continuations that never describe the tool's own action: "ran into an issue", "set out to help"
_NON_ACTION_TAIL = r"(?!\s+(?:into|out of|across|through|over|apart|aside)\b)"
# a self-attributed completed claim, by kind ("I've sent the email", "I turned on the lights")
_ASSISTANT_ACTION_CLAIM: dict[str, re.Pattern] = {
    kind: re.compile(
        rf"\b{_ACTION_SUBJECT}\s+(?:{_ACTION_MODIFIERS}\s+){{0,2}}(?:{verbs})\b{_NON_ACTION_TAIL}",
        re.I,
    )
    for kind, verbs in _CLAIM_VERBS.items()
}
_NEGATED_ACTION_CLAIM: dict[str, re.Pattern] = {
    kind: re.compile(
        # also U+2019 (the curly apostrophe): a model writing one must not read as a completed claim (#492 review)
        rf"\b(?:not|didn['\u2019]t|hasn['\u2019]t|wasn['\u2019]t|haven['\u2019]t|never|nothing)\b[^.!?]{{0,50}}\b(?:{verbs})\b",
        re.I,
    )
    for kind, verbs in _CLAIM_VERBS.items()
}
_OTHER_PERSON_ACTION_CLAIM: dict[str, re.Pattern] = {
    kind: re.compile(
        rf"\b(?:you|he|she|they|someone|anyone|(?:my|your|his|her|their)\s+\w+)\s+"
        rf"(?:have\s+|had\s+|just\s+)?(?:{verbs})\b",
        re.I,
    )
    for kind, verbs in _CLAIM_VERBS.items()
}
_HYPOTHETICAL_ACTION_CLAIM: dict[str, re.Pattern] = {
    kind: re.compile(rf"\b(?:if|would|could|might|may)\b[^.!?]{{0,50}}\b(?:{verbs})\b", re.I)
    for kind, verbs in _CLAIM_VERBS.items()
}
# nouns that name the kind in a claim ("sent the email"): shared verbs ("sent" is email and message, "created"
# is file and event) stay ambiguous without one, and an ambiguous never-made claim is left alone
_KIND_NOUNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\be-?mails?\b", re.I),
    "message": re.compile(r"\b(messages?|texts?|sms|dms?|posts?|chats?)\b", re.I),
    "file": re.compile(r"\b(files?|notes?|drafts?|spreadsheets?)\b", re.I),
    "event": re.compile(r"\b(events?|meetings?|appointments?|calendar)\b", re.I),
    "home": re.compile(
        r"\b(lights?|lamps?|thermostat|alarm|doors?|locks?|blinds?|covers?|scenes?|heating|cooling)\b",
        re.I,
    ),
}
# verbs that also describe ordinary chat text ("wrote a summary", "added context"): a never-made claim with one
# is only corrected when it names the kind ("saved the file", "created the meeting")
_CHAT_VERBS = {"wrote", "written", "created", "added", "stored"}
# what names the action in a claim, by kind: an email or message noun, a file name, a calendar word, a device.
# Verbs like "saved", "scheduled", "set" or "ran" alone also describe actions this check doesn't track (memory,
# tasks, reminders), so a never-made claim with only such a verb is left alone (#492 review).
_ACTION_NAMERS: dict[str, re.Pattern] = {
    "email": re.compile(r"\be-?mails?\b", re.I),
    "message": re.compile(r"\b(messages?|texts?|sms|dms?|posts?|chats?)\b", re.I),
    "file": re.compile(r"\b[\w~.-]+\.[a-z0-9]{1,6}\b", re.I),
    "event": re.compile(r"\b(events?|meetings?|appointments?|calendar)\b", re.I),
    "home": re.compile(
        r"\b(lights?|lamps?|thermostat|alarm|doors?|locks?|blinds?|covers?|scenes?|heating|cooling)\b",
        re.I,
    ),
}
# a message claim names the action only with a recipient or an app name: "text", "chat" and "post" alone also
# describe chat replies, which are not message-tool calls (#495 review)
_MESSAGE_APPS = re.compile(r"\b(telegram|slack|discord|whatsapp|signal|teams|imessage|messenger)\b", re.I)
_MESSAGE_RECIPIENT = re.compile(r"\bto\s+(?:@\S+|[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)?|the\s+\w+)")
# a file name names the action only with a known extension (a lowercase stem, so "Node.js" does not count), or
# with the word "file" / "saved to": dotted words like "a.m.", "e.g.", "example.com" and "3.5" do not (#495 review)
_FILE_EXTENSIONS = (
    r"md|txt|json|yaml|yml|toml|csv|pdf|doc|docx|xls|xlsx|ppt|pptx|py|js|ts|jsx|tsx|html|htm|css|sh|bash|"
    r"go|rs|java|c|cpp|h|hpp|png|jpg|jpeg|gif|svg|mp3|mp4|zip|log|ini|cfg|sql|rb|php|swift|kt|lock"
)
_FILE_NAME = re.compile(rf"\b[a-z0-9][a-zA-Z0-9_~.-]*\.(?:{_FILE_EXTENSIONS})\b")
_FILE_WORD = re.compile(r"\bfile\b", re.I)
_SAVED_TO = re.compile(r"\b(?:saved|wrote|written|stored)\s+to\b", re.I)


def _names_action(kind: str, claim_text: str) -> bool:
    """True when the sentence names the action for ``kind``: an email or message noun, a file name, a calendar
    word, or a device. A never-made claim is only corrected when it does, or when this turn actually tried that
    action: otherwise the verb alone may describe an untracked action ("I saved that..." after memory_remember)."""
    if kind == "message":
        # "I only sent those two exact text replies" is a chat reply, not a message-tool call: it needs a
        # recipient ("to Alice", "@maya") or an app name ("on Telegram") to count as naming a message action.
        return bool(
            _MESSAGE_APPS.search(claim_text)
            or _MESSAGE_RECIPIENT.search(claim_text)
            or re.search(r"[@#]\w+", claim_text)
        )
    if kind == "file":
        return bool(
            _FILE_WORD.search(claim_text) or _SAVED_TO.search(claim_text) or _FILE_NAME.search(claim_text)
        )
    if kind == "event":
        # "meeting"/"appointment" alone also describe a task reminder and a browser booking, neither of which
        # is a calendar event: a task/reminder context, or a booked appointment with a person, does not name one.
        if re.search(r"\btask\b|\bremind\w*\b", claim_text, re.I):
            return False
        if re.search(r"\bbook\w*\b[^.!?]*\bappointment\b[^.!?]*\bwith\b", claim_text, re.I):
            return False
    return bool(_ACTION_NAMERS[kind].search(claim_text))


def _owns_claim(kind: str, claim_text: str) -> bool:
    """True when ``kind`` unambiguously owns a self-attributed claim: its verb is unique to the kind and not an
    ordinary chat verb, or the claim names the kind ("sent the email", "created the meeting"). A shared verb
    without a naming noun ("I sent it") stays ambiguous: saying which action didn't happen would risk
    contradicting a true reply about the other kind."""
    kinds = [k for k, pattern in _ASSISTANT_ACTION_CLAIM.items() if pattern.search(claim_text)]
    if kind not in kinds:
        return False
    verb_match = _CLAIMS[kind].search(claim_text)
    verb = verb_match.group(0).lower() if verb_match else ""
    if verb in _CHAT_VERBS and not _KIND_NOUNS[kind].search(claim_text):
        return False
    if len(kinds) == 1:
        return True
    return bool(_KIND_NOUNS[kind].search(claim_text))


def _owns_passive_claim(kind: str, claim: str) -> bool:
    """True when ``kind`` unambiguously owns a passive claim ("the email was sent")."""
    kinds = [k for k in _CLAIM_VERBS if _CLAIMS[k].search(claim)]
    if kind not in kinds:
        return False
    if len(kinds) == 1:
        return True
    return bool(_KIND_NOUNS[kind].search(claim))

# an MCP tool's kind from its name (servers declare nothing like it)
_NAME_KINDS: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("email", re.compile(r"mail|smtp"), re.compile(r"send|reply|forward")),
    ("message", re.compile(r"message|msg|chat|slack|discord|telegram|whatsapp|sms"), re.compile(r"send|post|reply")),
    ("event", re.compile(r"event|calendar|meeting|appointment"), re.compile(r"create|add|schedule|book|insert")),
    ("file", re.compile(r"file|document|note"), re.compile(r"write|save|create|put|upload")),
]

FAILED_STATUSES = {
    "error", "failed", "failure", "fail", "rejected", "bounced", "undeliverable", "not_sent", "unsent", "declined",
    "refused", "invalid",
}
_FAILED_LIST_KEYS = ("failed", "failures", "errors", "rejected", "bounced", "not_sent", "undelivered", "unsent")
_SUCCESS_FLAGS = ("ok", "success", "succeeded", "sent", "delivered", "created", "saved")
_WHO_KEYS = ("to", "recipient", "email", "address", "name", "id", "path", "item")
_WHY_KEYS = ("error", "reason", "message", "detail", "description", "status")
_ERROR_TEXT = re.compile(r"^\s*(error|failed|failure|could ?n[o']t|unable to)\b", re.I)
_ADDRESS = re.compile(r"[^\s,;<>\"']+@[^\s,;<>\"']+")


def _straight_apostrophes(text: str) -> str:
    """Fold curly apostrophes to straight ones so negation patterns match either."""
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def _clip(text: Any, limit: int = MAX_DETAIL) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def _who_why(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return item, ""
    if not isinstance(item, dict):
        return str(item), ""
    who = next((str(item[k]) for k in _WHO_KEYS if isinstance(item.get(k), str | int) and str(item[k]).strip()), "")
    why = next((_clip(item[k], 120) for k in _WHY_KEYS if isinstance(item.get(k), str) and item[k].strip()), "")
    return who, why


def _failed_items(items: Any) -> tuple[str, list[str]]:
    """A plain description of a list (or mapping) of failed items, and who they were."""
    if isinstance(items, dict):
        items = [{"to": k, "error": v if isinstance(v, str) else json.dumps(v, default=str)} for k, v in items.items()]
    if isinstance(items, str):
        items = [items]
    parts: list[str] = []
    missed: list[str] = []
    for item in items if isinstance(items, list) else []:
        who, why = _who_why(item)
        if who:
            missed.append(who)
        parts.append(f"{who} ({why})" if who and why else who or why)
    parts = [p for p in parts if p]
    if not parts:
        return "", missed
    shown = ", ".join(parts[:3]) + (f" and {len(parts) - 3} more" if len(parts) > 3 else "")
    return shown, missed


def _parse_text(text: str) -> Any:
    s = text.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


def result_problem(res: Any, _depth: int = 0) -> tuple[str, list[str]] | None:
    """What in a tool result says the action did not (fully) work, as ``(plain detail, missed items)``; None when
    nothing does. Deterministic: an ``error``, a false success flag, a failed status, a non-empty list of failed
    items (one level of nesting, MCP text that is JSON included). Never raises."""
    if _depth > 4:
        return None
    try:
        if isinstance(res, str):
            parsed = _parse_text(res)
            return result_problem(parsed, _depth + 1) if parsed is not None else None
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict | list) and (found := result_problem(item, _depth + 1)):
                    return found
            return None
        if not isinstance(res, dict):
            return None
        err = res.get("error")
        if isinstance(err, str) and err.strip():
            who, _why = _who_why({k: v for k, v in res.items() if k != "error"})
            return (f"{_clip(err)} ({who})", [who]) if who else (_clip(err), [])
        if isinstance(err, dict) and err:
            return _clip(err.get("message") or json.dumps(err, default=str)), []
        for key in _FAILED_LIST_KEYS:
            items = res.get(key)
            if items and isinstance(items, list | dict | str) and not isinstance(items, bool):
                shown, missed = _failed_items(items)
                if shown:
                    return f"it couldn't do it for {shown}", missed
        for key in _SUCCESS_FLAGS:
            if res.get(key) is False:
                _who, why = _who_why(res)
                return (why or f"it reported {key}: false"), []
        for key in ("status", "state", "result"):
            value = res.get(key)
            if isinstance(value, str) and value.strip().lower().replace(" ", "_") in FAILED_STATUSES:
                who, why = _who_why({k: v for k, v in res.items() if k not in ("status", "state", "result")})
                text = why or f"its status was {value.strip()}"
                return (f"{text} ({who})" if who and why else text), [who] if who else []
        content = res.get("content")
        if isinstance(content, str):
            parsed = _parse_text(content)
            if parsed is not None:
                if found := result_problem(parsed, _depth + 1):
                    return found
            elif _ERROR_TEXT.match(_straight_apostrophes(content)):
                return _clip(content), []
        for key, value in res.items():
            if key in ("content", "error") or not isinstance(value, dict | list):
                continue
            if found := result_problem(value, _depth + 1):
                return found
    except Exception as exc:  # a check must never break the call it checks
        log.debug("result check skipped: %s", exc)
    return None


def kind_of(tool: Any) -> str | None:
    """The kind of effect a tool has: its declared ``outcome``, or for an MCP tool a guess from its name."""
    declared = getattr(tool, "outcome", None)
    if declared:
        return str(declared)
    name = str(getattr(tool, "name", "")).lower()
    for kind, noun, verb in _NAME_KINDS:
        if noun.search(name) and verb.search(name):
            return kind
    return None


def checked(tool: Any, *, mcp: bool) -> bool:
    """True when calls of ``tool`` get an outcome check: it declares one, or it is an MCP tool that changes things."""
    if getattr(tool, "outcome", None) or getattr(tool, "verify", None):
        return True
    return mcp and int(getattr(tool, "risk", 0)) >= 1


def action_call(
    tool: Any,
    arguments: dict,
    res: Any,
    *,
    ran: bool,
    declined: bool,
    is_error: bool,
    err: str | None,
    check: dict | None,
) -> dict | None:
    """The code-recorded evidence for one call of an action tool with a clear outcome (#492): its kind, whether
    the user declined it, whether it worked, and what it acted on. None when the tool has no clear outcome.
    Recorded for calls that ran and for declined ones, mirroring ``browser_clicks`` for #407."""
    kind = kind_of(tool)
    if kind not in KINDS or not (ran or declined):
        return None
    targets = check.get("targets") if isinstance(check, dict) else None
    return {
        "kind": kind,
        "declined": declined,
        "ok": ran and not is_error and not err and result_problem(res) is None
        and (not isinstance(check, dict) or check.get("ok") is not False),
        "uncertain": ran and is_error,
        "targets": list(targets or []) or targets_of(kind, arguments),
    }


def _short_name(tool: Any) -> str:
    name = str(getattr(tool, "name", "") or "")
    plugin = str(getattr(tool, "plugin", "") or "")
    if plugin and name.startswith(plugin + "_"):
        name = name[len(plugin) + 1:]
    return name.replace("_", " ").strip() or "the action"


def labels(check: dict) -> tuple[str, str, str]:
    """(what it tries to do, what it says it did, short label) for a check."""
    kind = check.get("kind")
    if kind in KINDS:
        return KINDS[kind]
    what = str(check.get("action") or "the action")
    return f"do {what}", f"did {what}", what


def targets_of(kind: str | None, arguments: dict) -> list[str]:
    """What a call acted on, for telling whether a later call covered a failure: recipients, a file name, a title."""
    args = arguments or {}
    if kind in {"email", "message"}:
        found: list[str] = []
        for key in ("to", "cc", "bcc", "recipient", "recipients", "channel", "chat_id", "email"):
            value = args.get(key)
            values = value if isinstance(value, list) else [value] if value else []
            for v in values:
                text = str(v)
                found += _ADDRESS.findall(text) or [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
        return list(dict.fromkeys(t.lower() for t in found))
    if kind == "home" and args.get("entity_id"):  # one id, several separated by commas, or a list
        value = args["entity_id"]
        ids = value if isinstance(value, list) else re.split(r"[,\s]+", str(value))
        return list(dict.fromkeys(str(i).strip().lower() for i in ids if str(i).strip()))
    for key in ("name", "path", "filename", "summary", "title", "text"):
        if isinstance(args.get(key), str) and args[key].strip():
            return [args[key].strip().lower()]
    return []


async def _verify(tool: Any, arguments: dict, res: Any, ctx: Any) -> dict | None:
    fn = getattr(tool, "verify", None)
    if fn is None:
        return None
    try:
        value = fn(arguments or {}, res, ctx)
        if inspect.isawaitable(value):
            value = await asyncio.wait_for(value, VERIFY_TIMEOUT_S)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # couldn't read it back: the tool's own result stands
        log.info("could not read back the result of %s: %s", getattr(tool, "name", "?"), exc)
        return None
    return value if isinstance(value, dict) and "ok" in value else None


async def check(tool: Any, arguments: dict, res: Any, is_error: bool, ctx: Any, *, app: str, mcp: bool) -> dict | None:
    """The outcome check of one call that ran, or None when ``tool`` isn't checked or nothing was meant to happen.
    Never raises (a failing read-back leaves the tool's own result standing)."""
    if not checked(tool, mcp=mcp):
        return None
    if isinstance(res, dict) and (res.get("declined") or res.get("undone") or res.get("try_run") or res.get("handed_off")):
        return None
    kind = kind_of(tool)
    out: dict[str, Any] = {
        "tool": getattr(tool, "name", ""), "app": app, "kind": kind,
        "action": KINDS[kind][2] if kind in KINDS else _short_name(tool),
        "ok": True, "detail": "", "how": "result", "targets": targets_of(kind, arguments), "missed": [],
    }
    problem = result_problem(res)
    if is_error and problem is None:
        problem = (_clip(res) if res else "the call failed"), []
    if problem is not None:
        detail, missed = problem
        out.update(ok=False, detail=f"{app} reported a problem: {detail}", missed=[m.lower() for m in missed])
        return out
    verified = await _verify(tool, arguments, res, ctx)
    if verified is not None:
        out["how"] = "read_back"
        if not verified.get("ok"):
            out.update(ok=False, detail=_clip(verified.get("detail") or f"{app} doesn't show it"))
            if isinstance(verified.get("missed"), list):  # which of the things it acted on didn't change
                out["missed"] = [str(m).lower() for m in verified["missed"]]
        elif verified.get("detail"):  # fine, with a note ("Garage door is still opening")
            out["detail"] = _clip(verified["detail"])
    return out


def model_note(check: dict | None) -> str:
    """A line for the model after a call whose result said it worked but whose read-back shows it didn't (#386), so
    the reply never calls it done; "" otherwise (a failed result is already in the result the model reads)."""
    if not check or check.get("ok") is not False or check.get("how") != "read_back":
        return ""
    detail = str(check.get("detail") or "it did not happen").rstrip(" .")
    return f"[Sentient checked afterwards and it did not work: {detail}. Tell the user plainly; don't say it worked.]"


def claimed(text: str, kind: str | None) -> bool:
    """True when a final answer says the action happened ("I've sent the email")."""
    return bool((_CLAIMS.get(kind or "") or _CLAIM_ANY).search(text or ""))


def _claimed_click_target(text: str) -> str:
    """The named target in an assistant or passive click claim, or empty for ``clicked it``."""
    match = _ASSISTANT_BROWSER_CLICK.search(text)
    if match is not None:
        tail = re.split(r"\b(?:on|and|but)\b|[,;.!?]", text[match.end():], maxsplit=1, flags=re.I)[0]
    else:
        passive = re.search(r"(?P<target>.+?)\s+was clicked\b", text, re.I)
        if passive is None:
            return ""
        tail = passive.group("target")
    target = re.sub(r"^\s*(?:only|just|the|a|an)\s+", "", tail, flags=re.I).strip()
    return "" if target.lower() in {"", "it", "that", "this", "button", "link"} else target


def _target_words(value: str) -> set[str]:
    """Meaningful target words, without generic element labels."""
    return {
        word for word in re.findall(r"[a-z0-9]+", value.lower())
        if word not in {"a", "an", "the", "button", "link", "only", "just"} and not re.fullmatch(r"e\d+", word)
    }


def _matches_click_target(claim: str, click: dict) -> bool:
    """Whether a named claim plausibly refers to this click's recorded browser element."""
    if not claim:
        return True

    claim_words = _target_words(claim)
    target_words = _target_words(str(click.get("target") or ""))

    def same_word(left: str, right: str) -> bool:
        """Accept the small morphological variations common in a natural-language reply."""
        return left == right or (
            len(left) >= 6 and len(right) >= 6 and left[:6] == right[:6]
        )

    # The recorded element name is the dependable evidence.  It need only appear
    # in the claim: a reply naturally adds context such as "for the newsletter"
    # or an element ref after naming the clicked control.
    return bool(target_words) and all(
        any(same_word(target_word, claim_word) for claim_word in claim_words)
        for target_word in target_words
    )


def _click_note(clicks: list[dict], target: str) -> str:
    """The correction for a click claim after selecting evidence for its named target."""
    matches = [click for click in clicks if not target or _matches_click_target(target, click)]
    has_success = any(click.get("ok") for click in clicks)
    if not target and has_success:
        # A targetless claim could refer to any completed action. A later
        # declined click is not enough evidence to contradict it.
        return ""
    if target and not matches and not has_success:
        # A declined call may retain only its transient ref; it is still evidence that the named action was not made.
        matches = [
            click for click in clicks
            if not click.get("ok") and not _target_words(str(click.get("target") or ""))
        ]
    if target and not matches:
        if has_success:
            # A successful click of another named control makes an unmatched
            # claim ambiguous.  Do not turn an honest report into a correction.
            return ""
        return "I didn't click it, so nothing changed on the page."
    if not matches:
        return "I didn't click it, so nothing changed on the page."
    if any(click.get("ok") for click in matches):
        # the claimed click did happen; a later declined or failed try of the same control doesn't undo it
        return ""
    click = matches[-1]
    if click.get("declined"):
        return "I didn't click it: you declined, so nothing changed on the page."
    if click.get("uncertain"):
        return "I couldn't confirm whether it was clicked."
    return "I couldn't click it, so nothing changed on the page."


def unmade_browser_click_note(text: str, calls: list[dict], history: list[dict] | None = None) -> str:
    """A short correction when a reply says it clicked although this turn did not.

    Small local models occasionally state that a browser click happened without making
    the call at all, or after the user declined its approval.  This is deliberately
    narrow: it only corrects the completed form of ``click`` and only looks at calls
    recorded by the loop, never at text supplied by a page or the user.
    """
    current_clicks = [call for call in calls if call.get("name") == "browser_click"]
    previous_clicks = [call for call in history or [] if call.get("name") == "browser_click"]
    claims = re.split(r"(?<=[.!?])\s+", text or "")
    # A reply can summarize a successful click from an earlier turn alongside a
    # declined click from this one, so consider the complete chronological record.
    clicks = previous_clicks + current_clicks
    for claim in claims:
        if _HYPOTHETICAL_BROWSER_CLICK.search(claim):
            continue
        explicit = list(_ASSISTANT_BROWSER_CLICK.finditer(claim))
        # Each self-attributed claim needs its own evidence.  In particular, a
        # successful "I clicked Subscribe" must not hide a later false
        # "I clicked Delete account" in the same sentence.
        explicit_parts = [
            claim[match.start(): explicit[i + 1].start() if i + 1 < len(explicit) else None]
            for i, match in enumerate(explicit)
        ]
        for part in explicit_parts:
            # The reply already tells the user that this attempted action was
            # declined, so a second correction would add no information and
            # can contradict an earlier successful click in the same reply.
            if re.search(r"\b(?:declined|denied)\b", part, re.I):
                continue
            if note := _click_note(clicks, _claimed_click_target(part)):
                return note
        passive = (
            not explicit
            and clicks
            and _BROWSER_CLICK_CLAIM.search(claim)
            and not _NEGATED_BROWSER_CLICK.search(_straight_apostrophes(claim))
            and not _OTHER_PERSON_BROWSER_CLICK.search(claim)
        )
        if passive and (note := _click_note(clicks, _claimed_click_target(claim))):
            return note
    return ""


def _claimed_action_target(text: str, kind: str) -> str:
    """The named target after an action claim ("alice" in "I sent the email to Alice"), or "" for "it" and for a
    bare kind noun ("the file" in "I saved the file"): the claim could refer to any recorded call of the kind."""
    match = _CLAIMS[kind].search(text)
    tail = text[match.end():] if match is not None else ""
    # A "." only ends the target at a sentence boundary: "arjun@example.com" and "plan.md" name the full
    # address or file name (#492 review).
    tail = re.split(r"\b(?:on|and|but|because|so)\b|[,;!?]|\.(?=\s|$)", tail, maxsplit=1, flags=re.I)[0]
    target = re.sub(r"^\s*(?:only|just|the|a|an|to|for|my|your)\s+", "", tail, flags=re.I).strip()
    if target.lower() in {"", "it", "that", "this", "them", "everything"}:
        return ""
    return "" if _KIND_NOUNS[kind].fullmatch(target) else target


def _call_target_words(call: dict) -> set[str]:
    """Meaningful words in what a recorded action call acted on."""
    return _target_words(" ".join(str(target) for target in call.get("targets") or []))


def _matches_action_target(claim: str, call: dict) -> bool:
    """Whether a named claim plausibly refers to what this recorded action call acted on."""
    if not claim:
        return True
    claim_words = _target_words(claim)
    target_words = _call_target_words(call)

    def same_word(left: str, right: str) -> bool:
        """Accept the small morphological variations common in a natural-language reply."""
        return left == right or (
            len(left) >= 6 and len(right) >= 6 and left[:6] == right[:6]
        )

    # The recorded target is the dependable evidence.  It need only appear in
    # the claim: a reply naturally adds context such as "for the newsletter"
    # after naming what the action acted on.
    return bool(target_words) and all(
        any(same_word(target_word, claim_word) for claim_word in claim_words)
        for target_word in target_words
    )


def _turn_names(kind: str, claim_text: str, target: str, turn_calls: list[dict]) -> bool:
    """True when this turn actually tried ``kind`` for what the sentence names, not just the kind: a declined
    file write this turn must not make "I've saved your preference" (after a real memory save) get corrected
    (#495 review). A targetless claim counts, a matching target counts, and for email/message a person-like
    target counts (a name may map to an address the record holds); anything else does not."""
    same = [call for call in turn_calls or [] if call.get("kind") == kind]
    if not same:
        return False
    if not target:
        return True
    if any(_matches_action_target(target, call) for call in same):
        return True
    return bool(
        kind in {"email", "message"}
        and re.search(
            r"(?:^|[\s,])(?:@\w+|[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)?)(?:[\s,.]|$)", target
        )
    )


def _action_note(
    kind: str,
    target: str,
    calls: list[dict],
    claim_text: str,
    turn_calls: list[dict],
    history_complete: bool,
) -> str:
    """The correction for an action claim after selecting evidence for its kind and named target.

    ``turn_calls`` are the action calls this turn actually tried; ``history_complete`` is False when the loaded
    history doesn't cover the whole conversation, so a missing record is "unsure", never "never made"."""
    same = [call for call in calls if call.get("kind") == kind]
    has_success = any(call.get("ok") for call in same)
    if not target and has_success:
        # A targetless claim could refer to any completed action. A later
        # declined call is not enough evidence to contradict it.
        return ""
    matches = [call for call in same if _matches_action_target(target, call)]
    if target and not matches and not has_success:
        # A declined call may name only its transient target; it is still evidence the named action was not made.
        matches = [call for call in same if not call.get("ok") and not _call_target_words(call)]
    doing, did, _label = labels({"kind": kind})
    if target and not matches:
        if has_success:
            # A successful action on another target makes an unmatched claim
            # ambiguous. Do not turn an honest report into a correction.
            return ""
        # Only correct when the kind is unambiguous, never on a shared verb ("sent" is email and message) that
        # could contradict a true reply about the other kind; and only when the sentence names the action or
        # this turn actually tried it for what it names: a bare verb may describe an untracked action
        # ("I saved that..." after memory_remember, "I've scheduled a task...").
        if _owns_claim(kind, claim_text) and (
            _names_action(kind, claim_text) or _turn_names(kind, claim_text, target, turn_calls)
        ):
            if any(call.get("uncertain") for call in same) or not history_complete:
                # An uncertain call may have done it, and calls outside the loaded history window are invisible
                # here: neither is proof it never happened.
                return f"I couldn't confirm whether I {did}."
            return f"I didn't {doing}, so nothing changed."
        return ""
    if not matches:
        if _owns_claim(kind, claim_text) and (
            _names_action(kind, claim_text) or _turn_names(kind, claim_text, target, turn_calls)
        ):
            if not history_complete:
                return f"I couldn't confirm whether I {did}."
            return f"I didn't {doing}, so nothing changed."
        return ""
    if any(call.get("ok") for call in matches):
        # the claimed action did happen; a later declined or failed try of the same target doesn't undo it (#488)
        return ""
    call = matches[-1]
    if call.get("declined"):
        return f"I didn't {doing}: you declined, so nothing changed."
    if call.get("uncertain"):
        return f"I couldn't confirm whether I {did}."
    return f"I couldn't {doing}, so nothing changed."


def unmade_action_note(
    text: str,
    action_calls: list[dict],
    history: list[dict] | None = None,
    *,
    history_complete: bool = True,
) -> str:
    """A short correction when a reply says an action happened although this turn did not (#492).

    The #407 approach for the other action tools with a clear outcome: send email/message, file write,
    calendar create, Home Assistant calls. Reuses the outcome checks (#165/#380) through the code-recorded
    ``action_calls`` of this turn and ``history`` of earlier turns. Deliberately narrow: it only corrects
    completed claims the run's own records contradict, never a true reply, and leaves the text alone when
    unsure. ``history_complete`` is False when the loaded history doesn't cover the whole conversation: a
    missing record then reads as "unsure", never "never made". Voice and channel replies are covered too:
    they speak and post this same final text.
    """
    calls = [*(history or []), *(action_calls or [])]
    for kind in _CLAIM_VERBS:
        if note := _unmade_action_claim_note(text, kind, calls, action_calls or [], history_complete):
            return note
    return ""


def _unmade_action_claim_note(
    text: str, kind: str, calls: list[dict], turn_calls: list[dict], history_complete: bool
) -> str:
    # fold curly apostrophes first (#500's helper): a claim with a curly apostrophe must read as a claim,
    # not slip past the subject pattern that only knows the straight apostrophe
    claims = re.split(r"(?<=[.!?])\s+", _straight_apostrophes(text or ""))
    explicit = _ASSISTANT_ACTION_CLAIM[kind]
    for claim in claims:
        if _HYPOTHETICAL_ACTION_CLAIM[kind].search(claim):
            continue
        found = list(explicit.finditer(claim))
        # Each self-attributed claim needs its own evidence.  In particular, a
        # successful "I sent the report" must not hide a later false
        # "I also booked the room" in the same sentence.
        parts = [
            claim[match.start(): found[i + 1].start() if i + 1 < len(found) else None]
            for i, match in enumerate(found)
        ]
        for part in parts:
            # The reply already tells the user that this attempted action was
            # declined, so a second correction would add no information.
            if re.search(r"\b(?:declined|denied)\b", part, re.I):
                continue
            if note := _action_note(
                kind, _claimed_action_target(part, kind), calls, part, turn_calls, history_complete
            ):
                return note
        passive = (
            not found
            and any(call.get("kind") == kind for call in calls)
            and _owns_passive_claim(kind, claim)
            and _CLAIMS[kind].search(claim)
            and not _NEGATED_ACTION_CLAIM[kind].search(claim)
            and not _OTHER_PERSON_ACTION_CLAIM[kind].search(claim)
        )
        if passive and (
            note := _action_note(
                kind, _claimed_action_target(claim, kind), calls, claim, turn_calls, history_complete
            )
        ):
            return note
    return ""


def _settled(failed: dict, later: list[dict]) -> bool:
    """A failed check counts as settled when a later successful call of the same tool covered what it missed. A
    failure that names nothing it acted on (a call with arguments the tool couldn't read) is settled only by the very
    next call of that tool, if it worked: the model's retry, never an unrelated later call."""
    want = set(failed.get("missed") or []) or set(failed.get("targets") or [])
    same = [c for c in later if c.get("tool") == failed.get("tool")]
    if not want:
        return bool(same) and bool(same[0].get("ok"))
    return any(c.get("ok") and want <= set(c.get("targets") or []) for c in same)


def finalize(checks: list[dict], final_text: str = "") -> list[dict]:
    """The checks of a whole run or reply, each failure marked ``settled`` (a later call fixed it) or given the plain
    ``message`` the user sees, worded from what the final answer claimed."""
    out: list[dict] = []
    items = [c for c in checks or [] if isinstance(c, dict)]
    for i, c in enumerate(items):
        c = {k: v for k, v in c.items() if k not in ("message", "settled")}
        if c.get("ok") is False:
            if _settled(c, items[i + 1:]):
                c["settled"] = True
            else:
                doing, did, _label = labels(c)
                lead = f"Sentient said it {did}" if claimed(final_text, c.get("kind")) else f"Sentient tried to {doing}"
                c["message"] = f"{lead}, but {str(c.get('detail') or 'it did not happen').rstrip(' .')}."
        out.append(c)
    return out


def failures(checks: list[dict]) -> list[dict]:
    """Finalized checks whose failure still stands."""
    return [c for c in checks or [] if c.get("ok") is False and not c.get("settled")]


def summary(checks: list[dict]) -> str:
    """One plain line: "Checked: file written, email sent." or the failures, in order."""
    bad = failures(checks)
    if bad:
        return " ".join(str(c.get("message") or "") for c in bad).strip()
    done = list(dict.fromkeys(labels(c)[2] for c in checks or [] if c.get("ok")))
    return f"Checked: {', '.join(done)}." if done else ""


# A reply that says it kept something in memory although no memory_remember call worked this turn (#526). Real
# qwen3:8b created a task instead and still said "I've saved that you wake up at 7 a.m.". Deliberately narrow, like
# the click check: only a claim that is Sentient's own, about this turn and about its memory, never a file save
# (#492), a task or reminder, a negation ("I didn't save that", "Nothing was saved"), a question or an offer.
MEMORY_NOT_SAVED = "I haven't saved that as a memory yet. Want me to?"
MEMORY_NOT_SAVED_INCOGNITO = "This is an incognito chat, so I didn't save that as a memory."
_MEMORY_VERBS = r"saved|stored|noted|recorded|remembered|memori[sz]ed|logged|kept|added|put|written|jotted"
_MEMORY_CLAIMS = [
    # "I've saved that you wake up at 7", "I noted your wake-up time", "I've jotted that down"
    re.compile(
        r"\bi(?:'ve| have)?\s+(?:(?:also|now|just|gone ahead and|went ahead and|successfully)\s+){0,2}"
        r"(?:saved|stored|noted|recorded|remembered|memori[sz]ed|logged|jotted)\s+"
        r"(?:that\s+(?:you|your)\b|your\b|the fact\b|(?:it|that|this)\s+down\b|down\b)",
        re.I,
    ),
    # "I've made a note that you ...", "I made a note of it"
    re.compile(r"\bi(?:'ve| have)?\s+(?:just\s+)?made a (?:mental )?note\b", re.I),
    # "I'll remember that", "I will remember your wake-up time", "I'll note your wake-up time" (real qwen3:8b, no call)
    re.compile(
        r"\bi(?:'ll| will|'m going to| am going to)\s+(?:(?:always|definitely|now|make sure to|be sure to)\s+)?"
        r"(?:(?:remember|memori[sz]e)\b|(?:note|jot)\s+(?:that|this|it|your|down)\b"
        # "save" and "store" only with a memory or the user named: "I'll save it once you confirm" is an offer
        r"|(?:save|store)\s+(?:that\s+you\b|your\b|(?:that|this|it)\s+(?:as|in|into|to)\s+(?:(?:my|your|the|a)\s+)?"
        r"memor(?:y|ies)\b))",
        re.I,
    ),
    # "I've saved that to my memory", "Saved to memory", "That's now in your memories"
    re.compile(
        rf"\b(?:(?:{_MEMORY_VERBS})\b[^.!?\n]{{0,60}}?\b(?:to|in|into|as|under)|(?:it|that|this)(?:'s| is)\s+(?:now\s+)?in)"
        r"\s+(?:(?:my|your|the|a|our)\s+)?(?:long-term\s+)?memor(?:y|ies)\b",
        re.I,
    ),
]
# before the claim: "I didn't...", "Nothing was saved...", "I haven't..."
_MEMORY_NEGATION = re.compile(r"n't\b|\b(?:not|never|nothing|none|no)\b", re.I)
_MEMORY_HEDGE = re.compile(
    r"\b(?:if|would|could|should|can|might|may|shall|want me to|do you want|let me know|whether)\b", re.I
)
# someone else's save: "you saved it in your memory"
_MEMORY_OTHER_SUBJECT = re.compile(
    rf"\b(?:you|he|she|they|someone)\s+(?:have\s+|had\s+|just\s+)?(?:{_MEMORY_VERBS})\b", re.I
)
# an earlier save, not this turn's: "I already have that", "as I noted earlier"
_MEMORY_EARLIER = re.compile(r"\b(?:already|earlier|previously|last time)\b", re.I)
# a claim about something else that keeps things: a file (#492), a task, a reminder, a calendar or an app
_MEMORY_OTHER_PLACE = re.compile(
    r"\b(?:files?|folders?|documents?|docs?|pdfs?|notes|notebooks?|drafts?|spreadsheets?|sheets?|pages?|bookmarks?"
    r"|desktop|downloads|drive|disk|tasks?|remind\w*|alarms?|calendar|events?|schedule|lists?|contacts?|emails?"
    r"|messages?|projects?|boards?|cards?)\b"
    r"|\b[a-z0-9][\w~.-]*\.(?:md|txt|json|csv|pdf|docx?|xlsx?|html?|py)\b",
    re.I,
)


def memory_claim(text: str) -> str:
    """The first sentence of ``text`` in which Sentient says it kept something in its memory, or "". Folds curly
    apostrophes, and skips questions, offers, negations, earlier saves and saves to anything other than memory."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", _straight_apostrophes(text)):
        s = sentence.strip()
        if not s or s.endswith("?") or _MEMORY_HEDGE.search(s) or _MEMORY_EARLIER.search(s):
            continue
        if _MEMORY_OTHER_PLACE.search(s):
            continue
        cleaned = re.sub(r"\bno (?:problem|worries)\b", "", s, flags=re.I)  # "No problem, I've saved that..."
        for pattern in _MEMORY_CLAIMS:
            match = pattern.search(cleaned)
            before = cleaned[: match.end()] if match else ""
            if match and not _MEMORY_NEGATION.search(before) and not _MEMORY_OTHER_SUBJECT.search(before):
                return s
    return ""


def unmade_memory_claim(text: str, saves: list[dict]) -> str:
    """The sentence of a reply that claims a memory save when no ``memory_remember`` call worked this turn, or "".
    Any call that worked keeps the reply as it is: a new fact, an update, a near-duplicate it skipped ("I already
    had that") and a fact sent to Review all count."""
    if any(save.get("ok") for save in saves or []):
        return ""
    return memory_claim(text)
