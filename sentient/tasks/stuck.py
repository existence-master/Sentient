"""Stuck runs (issue #134): a run that stops getting anywhere pauses and tells the user why, instead of failing silently.

The executor feeds every streamed agent event of a single run to a ``Watch``. Deterministic signals, no model decides:

- **no activity**: no model output, tool progress or tool result for ``tasks.stuck_after_minutes`` (the call in
  flight is cancelled and dropped from the transcript, so "Try again" simply makes it again);
- **the same error again and again**: one tool fails with the same error ``tasks.stuck_after_repeated_errors`` times
  in a row, or the loop breaker (``tools.repeated_call_limit``) stops a call that kept getting the same error;
- **only the user can do it**: a tool result with ``needs_user`` (the browser refusing a password or card field) or a
  browser page asking the visitor to prove they are a person.

A stuck run pauses through the ``waiting_for_user`` machinery (``tasks/ask.py``, as ``tasks/limits.py`` does) instead
of a separate status, so restarts, Cancel, notifications and answering from a paired chat all work unchanged. Its
``pending_question`` keeps ``stuck`` (the signal) and ``reason``; the options are "Try again", "Skip this step" and
"Cancel". Any other answer goes to the model as the user's advice.
"""

from __future__ import annotations

import re
from typing import Any

from sentient.llm.events import ToolCallEvent, ToolResultEvent

TRY_AGAIN = "Try again"
SKIP = "Skip this step"
CANCEL = "Cancel"
OPTIONS = [TRY_AGAIN, SKIP, CANCEL]
KINDS = ("stalled", "errors", "blocked")

_HUMAN_CHECK_RE = re.compile(
    r"verify (?:that )?you(?:'re| are) (?:a )?human|are you a robot|i'?m not a robot|confirm you(?:'re| are) not a robot"
    r"|unusual traffic from your computer|complete the security check|press (?:and|&) hold",
    re.IGNORECASE,
)


def _minutes(seconds: float) -> str:
    m = seconds / 60
    text = str(round(m)) if m >= 1 else f"{m:.2f}".rstrip("0").rstrip(".")
    return f"{text} minute" + ("" if text == "1" else "s")


def tool_label(registry: Any, name: str) -> str:
    """The app a tool belongs to, as the user knows it ("Browser"), else the tool's name."""
    tool = registry.get(name) if registry is not None else None
    if tool is not None:
        for plugin in registry.plugins():
            if plugin.id == tool.plugin and plugin.display_name:
                return str(plugin.display_name)
    return name


def error_text(result: Any, is_error: bool) -> str | None:
    """The error a tool result reports, else None."""
    if isinstance(result, dict) and result.get("error"):
        return " ".join(str(result["error"]).split())[:200]
    return " ".join(str(result).split())[:200] if is_error else None


def blocked_reason(name: str, result: Any) -> str | None:
    """Why only the user can get past this step, from the tool's own ``needs_user`` or a person check on a page."""
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("needs_user"), str) and result["needs_user"].strip():
        return result["needs_user"].strip().rstrip(".")
    if name.startswith("browser_"):
        page = " ".join(str(result.get(k) or "")[:2000] for k in ("title", "text", "content"))
        if _HUMAN_CHECK_RE.search(page):
            return "the site wants proof that you're a person (a CAPTCHA)"
    return None


class Watch:
    """What one run segment is doing, for the stuck checks. ``reason`` is set once the run is stuck."""

    def __init__(self, registry: Any, *, stall_s: float, error_limit: int):
        self.registry = registry
        self.stall_s = stall_s
        self.error_limit = error_limit
        self.kind: str | None = None
        self.reason: str | None = None
        self.running: dict[str, str] = {}  # call id -> tool name, for calls still waiting for their result
        self.last_error: tuple[str, str] | None = None  # (tool, error) when the latest result was an error
        self._count = 0

    def stuck(self, kind: str, reason: str) -> None:
        if self.reason is None:
            self.kind, self.reason = kind, reason

    def see(self, event: Any) -> None:
        if isinstance(event, ToolCallEvent):
            self.running[event.call_id] = event.name
        elif isinstance(event, ToolResultEvent):
            self.running.pop(event.call_id, None)
            self._result(event.name, event.result, event.is_error)

    def _result(self, name: str, result: Any, is_error: bool) -> None:
        blocked = blocked_reason(name, result)
        if blocked:
            self.stuck("blocked", blocked)
        error = error_text(result, is_error)
        if error is None:
            self.last_error, self._count = None, 0
            return
        key = (name, error)
        self._count = self._count + 1 if key == self.last_error else 1
        self.last_error = key
        if self.error_limit and self._count >= self.error_limit:
            self.stuck("errors", self._errors_reason())

    def _errors_reason(self) -> str:
        assert self.last_error is not None
        name, error = self.last_error
        return f"{tool_label(self.registry, name)} keeps failing with the same error: {error.rstrip('.')}"

    def stalled(self) -> None:
        """No activity for ``stall_s``: name what it was waiting for."""
        waited = _minutes(self.stall_s)
        if self.running:
            name = next(iter(self.running.values()))
            self.stuck("stalled", f"{tool_label(self.registry, name)} hasn't responded for {waited}")
        else:
            self.stuck("stalled", f"the AI model hasn't answered for {waited}")

    def repeated(self) -> bool:
        """The loop breaker stopped the run: stuck when the repeated call kept failing (True), else a plain failure."""
        if self.last_error is None:
            return False
        self.stuck("errors", self._errors_reason())
        return True


def question(reason: str) -> str:
    return f"I'm stuck: {reason}. What should I do?"


def notice(name: str, reason: str) -> str:
    """The notification text for a stuck run."""
    return f"Sentient is stuck on '{name}': {reason}. Open it to help or cancel."


def pending(kind: str, reason: str) -> dict:
    """The ``pending_question`` a stuck run stores while it waits."""
    return {"question": question(reason), "options": list(OPTIONS), "tool_call_id": "", "stuck": kind, "reason": reason}


def _norm(answer: str) -> str:
    return " ".join(str(answer or "").split()).strip(" .!").lower()


def choice(answer: str) -> str | None:
    """``retry``, ``skip`` or ``cancel`` for one of the offered options, else None (the user's own words)."""
    return {TRY_AGAIN.lower(): "retry", SKIP.lower(): "skip", CANCEL.lower(): "cancel"}.get(_norm(answer))


def note(reason: str, answer: str) -> str:
    """The message the model reads when the run carries on after being stuck."""
    picked = choice(answer)
    if picked == "retry":
        return f"You got stuck ({reason}). The user asked you to try again. If it fails the same way, try a different way."
    if picked == "skip":
        return (
            f"You got stuck ({reason}). The user said to skip that step. Do not try it again: carry on with the rest of "
            "the plan and say in your final answer that this step was skipped."
        )
    return f"You got stuck ({reason}). The user replied: {answer}\nCarry on with the task using this."
