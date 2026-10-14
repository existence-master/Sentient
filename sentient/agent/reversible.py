"""Reversible defaults: trash instead of delete (#142, ADR 0025).

Many apps offer both a delete that is final and one that moves the item to a trash or archive it can come back from
("delete_note" and "trash_note"). A model asked to "delete the note" picks the first one. With
``tools.approvals.prefer_trash`` on, the agent loop swaps such a call for its trash twin before anything else looks
at it (``prefer_trash``), so the approval card, the second look and the transcript all see the reversible call. The
swap is skipped when the user's own words in this run ask for a permanent delete ("delete it for good",
"permanently"), when the twin can't take the call's arguments, or when the twin is hidden by a rule.

A twin is found in code from tool names alone (``preview.action_kind``): the same app, a trash verb instead of a
delete verb, and the same object words ("note"). ``restore_twin`` finds the way back ("restore_note",
"gmail_restore") so the Activity timeline can offer Restore for something that went to the trash.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from sentient.agent.preview import DELETE_VERBS, TRASH_VERBS, action_kind, name_words
from sentient.tools.base import Tool

RESTORE_VERBS = frozenset({"restore", "untrash", "recover", "undelete", "unarchive"})
_FILLER = frozenset({"to", "the", "a", "an", "move", "moves", "item", "items", "for", "good", "permanently",
                     "permanent", "hard", "soft", "from", "into", "in", "out", "of", "bin", "back"})
FOR_GOOD = re.compile(
    r"\b(?:permanent(?:ly)?|for good|forever|hard[- ]?delete|irreversibl[ey]|(?:skip|bypass)(?:ing)? the (?:trash|bin)"
    r"|empty(?:ing)? the (?:trash|bin)|can(?:no|')?t be (?:undone|restored|recovered)|not (?:to|in|into) the trash)\b",
    re.IGNORECASE,
)
TRASH_NOTE = "Moved to the trash instead of deleting it for good, so it can be restored."


def _object_words(name: str) -> set[str]:
    return name_words(name) - DELETE_VERBS - TRASH_VERBS - RESTORE_VERBS - _FILLER


def _twin(tool: Tool, registry: Any, is_twin: Callable[[str], bool]) -> Tool | None:
    """The tool in the same app that ``is_twin`` accepts by name and that acts on the same object words, or None."""
    want = _object_words(tool.name)
    best: tuple[int, Tool] | None = None
    for other in registry.tools(include_hidden=True):
        if other is tool or other.plugin != tool.plugin or not is_twin(other.name):
            continue
        have = _object_words(other.name)
        if have != want and not (have < want):
            continue
        score = len(have & want)
        if best is None or score > best[0]:
            best = (score, other)
    return best[1] if best else None


def trash_twin(tool: Tool, registry: Any) -> Tool | None:
    """The trash version of a delete-for-good tool in the same app ("trash_note" for "delete_note")."""
    if action_kind(tool.name) != "delete":
        return None
    return _twin(tool, registry, lambda name: action_kind(name) == "trash")


def restore_twin(tool: Tool, registry: Any) -> Tool | None:
    """The tool that brings back what ``tool`` moved to the trash ("restore_note" for "trash_note")."""
    if action_kind(tool.name) != "trash":
        return None
    return _twin(tool, registry, lambda name: bool(name_words(name) & RESTORE_VERBS))


def fit_arguments(tool: Tool, arguments: dict | None) -> dict | None:
    """``arguments`` cut down to what ``tool`` takes, or None when one of its required fields is missing."""
    try:
        schema = tool.openai_schema()["function"]["parameters"] or {}
    except Exception:
        return None
    props = schema.get("properties") or {}
    required = [str(r) for r in schema.get("required") or []]
    args = dict(arguments or {})
    if any(r not in args for r in required):
        return None
    return {k: v for k, v in args.items() if k in props} if props else {}


def asks_for_good(text: str) -> bool:
    """True when the user's own words ask for a permanent delete."""
    return bool(FOR_GOOD.search(text or ""))
