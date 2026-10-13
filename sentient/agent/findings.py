"""A plain answer built from tool results when the model gives none (#322).

Real qwen3:8b sometimes ends a run with an empty answer after its tool calls, even after the loop asks once more.
The loop then answers with what the tools returned instead of nothing: deterministic, short, and no model call.
"""

from __future__ import annotations

import json
import re
from typing import Any

INTRO = "I couldn't put together a full answer, so here is what I found:"
NOTHING = "I looked this up, but the results had nothing to report."
MAX_FINDINGS = 6
MAX_CHARS = 300  # per finding
MAX_ROWS = 5  # rows of a list result
# keys whose value is the readable part of a result, in order of preference
TEXT_KEYS = ("answer", "summary", "message", "content", "text", "fact", "result", "output", "title", "name")


def summarize(found: list[tuple[str, str, Any]]) -> str:
    """``found`` is ``[(tool name, label, result)]`` of the calls that worked, in order; ``label`` is the plain name
    shown ("Memory", "Web"). One line per tool whose latest result says something, at most ``MAX_FINDINGS``."""
    latest: dict[str, tuple[str, str]] = {}
    for name, label, result in found:
        brief = _brief(result)
        if brief:
            latest.pop(name, None)  # keep the order of the latest result
            latest[name] = (label, brief)
    if not latest:
        return NOTHING
    lines = [f"- {label}: {brief}" for label, brief in list(latest.values())[-MAX_FINDINGS:]]
    return "\n".join([INTRO, *lines])


def _brief(value: Any) -> str:
    return _cut(_text(value))


def _text(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        rows = [r for r in (_text(item) for item in value[:MAX_ROWS]) if r]
        more = len(value) - MAX_ROWS
        return "; ".join(rows) + (f" (and {more} more)" if more > 0 and rows else "")
    if isinstance(value, dict):
        if value.get("error"):
            return ""
        for key in TEXT_KEYS:
            if key in value:
                text = _text(value[key])
                if text:
                    return text
        for key in ("results", "items", "memories", "facts", "rows", "data"):
            if isinstance(value.get(key), list):
                return _text(value[key])
        pairs = [
            f"{k}: {v}" for k, v in value.items()
            if isinstance(v, str | int | float) and not isinstance(v, bool) and not str(k).endswith("id") and str(v).strip()
        ]
        return ", ".join(pairs)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _cut(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= MAX_CHARS:
        return text
    return text[: MAX_CHARS - 3].rstrip() + "..."
