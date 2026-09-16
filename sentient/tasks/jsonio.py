"""Tolerant JSON completions for the task pipeline.

Observed with Ollama + qwen3 (8b and 4b):
- JSON mode corrupts the reply prefix (``{"{"name": ...``, ``{\\n{\\n "name": ...``), which the
  core ``parse_json_loose`` rejects because it slices from the first ``{`` to the last ``}``;
- JSON mode returns valid JSON of the wrong shape (a bare list of strings);
- plain-text replies with small syntax slips, where the only decodable fragment is an inner
  object (one plan step, the schedule) instead of the whole reply;
- synonyms for requested keys (``steps`` instead of ``plan``), handled by the callers.

Strategy:
- Ollama-backed roles ask for plain text first; other providers use JSON mode first.
- Parsing collects every decodable candidate (whole text, ``json_repair`` if installed, and
  ``raw_decode`` from each ``{``/``[``), prefers candidates containing the caller's expected
  keys, then the one covering the most text.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from sentient.llm.provider import ProviderError

log = logging.getLogger(__name__)


def _clean(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()


def _decoded_candidates(cleaned: str, expect: tuple[type, ...]) -> list[tuple[int, Any]]:
    """``(span_length, value)`` for every well-formed JSON value of an expected type in the text."""
    found: list[tuple[int, Any]] = []
    try:
        whole = json.loads(cleaned)
        if isinstance(whole, expect) and whole:
            found.append((len(cleaned) + 1, whole))  # an exact parse always wins ties
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, expect) and obj:
            found.append((end - i, obj))
    return found


def _repaired_candidate(cleaned: str, expect: tuple[type, ...]) -> Any:
    """Last resort: ``json_repair`` (optional dependency) on the text from the first bracket."""
    try:
        import json_repair  # type: ignore[import-not-found]
    except ImportError:
        return None
    start = min((p for p in (cleaned.find("{"), cleaned.find("[")) if p >= 0), default=-1)
    if start < 0:
        return None
    try:
        repaired = json_repair.loads(cleaned[start:])
    except Exception:
        return None
    return repaired if isinstance(repaired, expect) and repaired else None


def _best_with_keys(found: list[tuple[int, Any]], wanted: set[str]) -> Any:
    with_keys = [c for c in found if isinstance(c[1], dict) and wanted & c[1].keys()]
    if not with_keys:
        return None
    # most expected keys first, then the widest span
    return max(with_keys, key=lambda c: (len(wanted & c[1].keys()), c[0]))[1]


def parse_json_tolerant(
    text: str, expect: tuple[type, ...] = (dict, list), keys: Iterable[str] = ()
) -> Any:
    """Exact parse, then well-formed fragments, then ``json_repair`` only when those fail
    (no fragment at all, or none carrying the expected keys)."""
    cleaned = _clean(text)
    wanted = set(keys)
    found = _decoded_candidates(cleaned, expect)
    if wanted:
        best = _best_with_keys(found, wanted)
        if best is not None:
            return best
    elif found:
        return max(found, key=lambda c: c[0])[1]
    repaired = _repaired_candidate(cleaned, expect)
    if repaired is not None and (not wanted or _best_with_keys([(0, repaired)], wanted) is not None):
        return repaired
    if found:
        log.warning("model JSON has none of the expected keys %s: %r", sorted(wanted), (text or "")[:500])
        return max(found, key=lambda c: c[0])[1]
    if repaired is not None:
        return repaired
    raise ValueError(f"Model did not return JSON: {(text or '')[:200]!r}")


def _prefers_text(llm: Any, role: str, model: str | None) -> bool:
    try:
        name = model or llm.model_for(role)
    except Exception:
        return False
    return isinstance(name, str) and name.startswith(("ollama/", "ollama_chat/"))


def _has_keys(data: Any, keys: Iterable[str]) -> bool:
    wanted = set(keys)
    return not wanted or (isinstance(data, dict) and bool(wanted & data.keys()))


async def _via_text(
    llm: Any, role: str, messages: list[dict], model: str | None, expect: tuple[type, ...], keys: Iterable[str]
) -> Any:
    text = await llm.complete_text(role, messages, model=model)
    return parse_json_tolerant(text, expect, keys)


async def _via_json_mode(llm: Any, role: str, messages: list[dict], model: str | None) -> Any:
    try:
        return await llm.complete_json(role, messages, model=model)
    except ProviderError as exc:
        if "did not return JSON" not in str(exc):
            raise
        log.info("JSON mode reply for role %s was malformed", role)
        return None


async def complete_json(
    llm: Any,
    role: str,
    messages: list[dict],
    *,
    model: str | None = None,
    expect: tuple[type, ...] = (dict, list),
    keys: Iterable[str] = (),
) -> Any:
    keys = tuple(keys)
    if _prefers_text(llm, role, model):
        try:
            return await _via_text(llm, role, messages, model, expect, keys)
        except ValueError:
            log.info("plain-text JSON reply for role %s was unusable; trying JSON mode", role)
        data = await _via_json_mode(llm, role, messages, model)
        if isinstance(data, expect):
            return data
        raise ValueError(f"Model did not return the expected JSON for role '{role}': {str(data)[:200]}")
    data = await _via_json_mode(llm, role, messages, model)
    if isinstance(data, expect) and _has_keys(data, keys):
        return data
    return await _via_text(llm, role, messages, model, expect, keys)


async def complete_json_object(
    llm: Any, role: str, messages: list[dict], *, model: str | None = None, keys: Iterable[str] = ()
) -> dict:
    return await complete_json(llm, role, messages, model=model, expect=(dict,), keys=keys)
