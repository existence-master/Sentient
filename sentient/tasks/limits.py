"""Limits on one task run: steps, active time, tokens and cost (issue #133).

When a run reaches a limit it pauses with a question ("Keep going" or "Stop here") through the same
``waiting_for_user`` machinery as ``ask_user`` (``tasks/ask.py``). "Keep going" raises that limit by its
original amount for this run only; any other answer fails the run with a plain message. The state lives
on the run (``task_runs.limits``) so used amounts and raised limits survive pauses and restarts. Time
counts only while the run is working, never while it waits for an answer. Deterministic: no model decides.
"""

from __future__ import annotations

from typing import Any

KEEP_GOING = "Keep going"
STOP_HERE = "Stop here"
KINDS = ("steps", "seconds", "tokens", "cost_usd")
LIMITS_HINT = "The limits for one run are in Settings > Tasks."


def new_state(config: Any) -> dict:
    """Fresh limits for a run, from ``tasks.*`` config. 0 means no limit (steps and time always have one)."""
    t = config.tasks
    base = {
        "steps": int(t.max_tool_rounds),
        "seconds": float(t.run_timeout_minutes) * 60,
        "tokens": int(t.max_tokens_per_run),
        "cost_usd": float(t.max_cost_per_run_usd),
    }
    return {"base": base, "max": dict(base), "used": dict.fromkeys(KINDS, 0)}


def load(run: dict, config: Any) -> dict:
    """The run's stored limits, or fresh ones (missing keys are filled from config)."""
    fresh = new_state(config)
    stored = run.get("limits") if isinstance(run.get("limits"), dict) else {}
    out: dict[str, dict] = {}
    for part in ("base", "max", "used"):
        given = stored.get(part) if isinstance(stored.get(part), dict) else {}
        out[part] = {k: _num(given.get(k), fresh[part][k]) for k in KINDS}
    return out


def _num(value: Any, default: float) -> Any:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else default


def raise_limit(state: dict, kind: str) -> dict:
    """'Keep going': the limit grows by its original amount, for this run only."""
    state["max"][kind] = state["max"][kind] + state["base"][kind]
    return state


def _minutes(seconds: float) -> str:
    m = seconds / 60
    text = str(round(m)) if m >= 1 else f"{m:.2f}".rstrip("0").rstrip(".")
    return f"{text} minute" + ("" if text == "1" else "s")


def _amount(kind: str, value: float) -> str:
    if kind == "steps":
        return f"{int(value)} steps"
    if kind == "seconds":
        return _minutes(value)
    if kind == "tokens":
        return f"{int(value):,} tokens"
    return f"${value:.2f}"


def question(state: dict, kind: str) -> str:
    used, limit, more = state["used"][kind], state["max"][kind], state["base"][kind]
    if kind == "steps":
        done = f"has used {_amount(kind, used)}"
    elif kind == "seconds":
        done = f"has been working for {_amount(kind, used)}"
    elif kind == "tokens":
        done = f"has used {_amount(kind, used)} of your {int(limit):,} token limit"
    else:
        done = f"has used {_amount(kind, used)} of your {_amount(kind, limit)} limit"
    return f"This task {done} and isn't finished yet. Keep going for another {_amount(kind, more)}, or stop here?"


def stop_message(state: dict, kind: str) -> str:
    """The plain failure message when the user stops a run at a limit."""
    used, limit = state["used"][kind], state["max"][kind]
    if kind in {"steps", "seconds"}:
        text = f"Stopped after {_amount(kind, used)} without finishing."
    elif kind == "tokens":
        text = f"Stopped after using {_amount(kind, used)} without finishing. The limit for one run is {_amount(kind, limit)}."
    else:
        text = (
            f"Stopped after spending about {_amount(kind, used)} on the model without finishing. "
            f"The limit for one run is {_amount(kind, limit)}."
        )
    return f"{text} {LIMITS_HINT}"


def pending(state: dict, kind: str) -> dict:
    """The ``pending_question`` a run stores while it waits at a limit."""
    return {
        "question": question(state, kind),
        "options": [KEEP_GOING, STOP_HERE],
        "tool_call_id": "",
        "limit": kind,
        "stop_error": stop_message(state, kind),
    }


def keeps_going(answer: str) -> bool:
    return " ".join(str(answer or "").split()).strip(" .!").lower() == KEEP_GOING.lower()
