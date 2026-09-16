"""Script jobs: watch-style tasks that run a short Python check without model calls (docs/API.md section 16).

A script job is a task with ``task_type: "script"`` and a ``script`` object::

    {code, condition: "changed"|"alert", then: "notify"|"run", last_result, last_run_at, last_error}

The code runs through ``app.sandbox.run`` and may call read-only Sentient tools with
``from sentient_tools import tools, result``. This module holds the pure helpers:
validation, normalization, reading the sandbox outcome and deciding whether to act.
"""

from __future__ import annotations

import ast
import json
from typing import Any

CONDITIONS = ("alert", "changed")
THEN = ("notify", "run")
MAX_CODE_CHARS = 20000
MAX_RESULT_CHARS = 20000
SCRIPT_STATE_FIELDS = ("last_result", "last_run_at", "last_error")


class ScriptInvalid(ValueError):
    """The script code or settings cannot be used."""


def validate_code(code: Any) -> str:
    """Return the code when it is non-empty Python that compiles; raise ScriptInvalid otherwise."""
    if not isinstance(code, str) or not code.strip():
        raise ScriptInvalid("The check script is empty.")
    if len(code) > MAX_CODE_CHARS:
        raise ScriptInvalid(f"The check script is too long (over {MAX_CODE_CHARS} characters).")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        raise ScriptInvalid(f"The check script is not valid Python{where}: {exc.msg}") from exc
    return code


def normalize_script(raw: Any, previous: dict | None = None) -> dict:
    """Merge user/planner input over ``previous``. Changing the code resets the stored state."""
    if not isinstance(raw, dict):
        raise ScriptInvalid("script must be an object with code, condition and then.")
    prev = dict(previous or {})
    code = raw.get("code", prev.get("code"))
    code = validate_code(code)
    condition = str(raw.get("condition") or prev.get("condition") or "alert").strip().lower()
    if condition not in CONDITIONS:
        raise ScriptInvalid("script.condition must be 'alert' or 'changed'.")
    then = str(raw.get("then") or prev.get("then") or "notify").strip().lower()
    if then not in THEN:
        raise ScriptInvalid("script.then must be 'notify' or 'run'.")
    out = {"code": code, "condition": condition, "then": then}
    keep = previous is not None and prev.get("code") == code and prev.get("condition") == condition
    for key in SCRIPT_STATE_FIELDS:
        out[key] = prev.get(key) if keep else None
    return out


def trigger_prelude(trigger_data: dict | None) -> str:
    """Lines prepended to the code of a triggered script so it can read the event as ``trigger_event``."""
    payload = json.dumps(trigger_data if isinstance(trigger_data, dict) else None, ensure_ascii=False, default=str)
    return f"import json as _sentient_json\ntrigger_event = _sentient_json.loads({payload!r})\n"


def outcome_value(outcome: dict) -> Any:
    """The value a script produced: ``result(...)`` when set, else its trimmed stdout."""
    value = outcome.get("result")
    if value is None:
        stdout = outcome.get("stdout")
        value = stdout.strip() if isinstance(stdout, str) and stdout.strip() else None
    return cap(value)


def cap(value: Any) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)[:MAX_RESULT_CHARS]
    if len(text) <= MAX_RESULT_CHARS:
        return json.loads(text)  # plain JSON types only
    return text[:MAX_RESULT_CHARS] + " ... [truncated]"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def outcome_error(outcome: dict) -> str | None:
    """A short, user-readable failure message, or None when the script succeeded."""
    if outcome.get("ok"):
        return None
    detail = str(outcome.get("error") or "").strip()
    if not detail:
        stderr = str(outcome.get("stderr") or "").strip()
        detail = stderr.splitlines()[-1] if stderr else "it stopped without saying why"
    return f"The check script failed: {detail[:500]}"


def should_act(condition: str, value: Any, previous: Any) -> bool:
    """``alert``: the script returned ``{"alert": true, ...}``. ``changed``: the value differs from the
    previous successful result. The first successful result of a ``changed`` job is only the baseline, and an
    empty (``None``) result is never a change (a page that briefly returns nothing should not alert)."""
    if condition == "changed":
        return value is not None and previous is not None and canonical(value) != canonical(previous)
    return isinstance(value, dict) and bool(value.get("alert"))


def alert_message(task_name: str, condition: str, value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("message"), str) and value["message"].strip():
        return value["message"].strip()
    shown = value if isinstance(value, str) else canonical(value)
    shown = shown if len(shown) <= 500 else shown[:497] + "..."
    if condition == "changed":
        return f"'{task_name}' noticed a change: {shown}"
    return f"'{task_name}' raised an alert: {shown}"


def describe_for_approval(script: dict) -> str:
    """Plain-language summary shown next to the code in a plan card."""
    when = "the script raises an alert" if script.get("condition") == "alert" else "the result changes"
    action = "send you a notification" if script.get("then") == "notify" else "start a full task run"
    return f"Runs a small check script with no AI calls; when {when}, it will {action}."
