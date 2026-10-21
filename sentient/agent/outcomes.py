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

# words in a final answer that claim the action happened, by kind
_CLAIMS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b(sent|emailed|delivered|mailed)\b", re.I),
    "message": re.compile(r"\b(sent|posted|messaged|delivered)\b", re.I),
    "file": re.compile(r"\b(saved|written|wrote|created|stored)\b", re.I),
    "event": re.compile(r"\b(created|added|scheduled|booked)\b", re.I),
    "home": re.compile(r"\b(turned|switched|set|done|ran)\b", re.I),
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
            elif _ERROR_TEXT.match(content):
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


def _targets(kind: str | None, arguments: dict) -> list[str]:
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
        "ok": True, "detail": "", "how": "result", "targets": _targets(kind, arguments), "missed": [],
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
    click = matches[-1]
    if click.get("ok"):
        return ""
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
            and not _NEGATED_BROWSER_CLICK.search(claim)
            and not _OTHER_PERSON_BROWSER_CLICK.search(claim)
        )
        if passive and (note := _click_note(clicks, _claimed_click_target(claim))):
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
