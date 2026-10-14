"""The same memory saved in two languages (#353): pairs for the nightly tidy-up to check.

Since #351 a fact in Hindi is never skipped as a copy of an English one (embedding models can't be trusted across
scripts), so "माया शाकाहारी है" and "Maya is vegetarian" can both be saved. Dreaming finds such pairs here and asks
the fast model whether they say exactly the same thing; only then is one merged into the other, reversibly.

A pair is one fact led by Devanagari and one with no Devanagari at all, clearly about the same person (both have a
subject and the names match across scripts, :func:`facts.same_subject`), with the same numbers, and where every name or
place the English fact gives has a match in the Hindi one (else the English one adds a detail). With a multilingual
embedding model (bge-m3) the two must also be close in meaning; with an English-first one (nomic-embed-text) the
similarity only orders the candidates, since it can't compare languages.
"""

from __future__ import annotations

import re

from sentient.memory.facts import keyword_tokens, same_subject, subject_aliases
from sentient.memory.scripts import has_devanagari, leads_with_devanagari, name_key, script_words
from sentient.memory.vectors import cosine

PER_FACT = 3  # English candidates checked for each Hindi fact, best first
MULTILINGUAL_MODELS = ("bge-m3",)
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_LATIN_WORD = re.compile(r"[A-Za-z][\w'-]*")


def multilingual(model: str) -> bool:
    """Whether the embedding model compares meaning across languages."""
    low = (model or "").lower()
    return any(m in low for m in MULTILINGUAL_MODELS)


def numbers(text: str) -> set[str]:
    """The numbers in ``text``, Devanagari digits read as 0 to 9."""
    return set(_NUMBER.findall((text or "").translate(_DIGITS)))


def details_covered(english: str, hindi: str) -> bool:
    """Every name or place the English fact gives (a capitalised word after the first) is in the Hindi fact too,
    in either script."""
    keys = {name_key(w) for w in script_words(hindi)}
    latin = {w.lower() for w in _LATIN_WORD.findall(hindi)}
    for i, w in enumerate(_LATIN_WORD.findall(english)):
        base = w.removesuffix("'s")
        if i == 0 or not base[:1].isupper():
            continue
        if base.lower() not in latin and name_key(base) not in keys:
            return False
    return True


def _stem(word: str) -> str:
    for end in ("ing", "ed", "es", "s"):
        if len(word) > len(end) + 3 and word.endswith(end):
            return word[: -len(end)]
    return word


def same_words(translation: str, english: str) -> bool:
    """Whether the model's English for the Hindi fact has the same meaningful words as the English fact (endings
    aside): "Maya does yoga every morning" is not "Maya does yoga". A different word for the same thing counts as
    different, so a merge only happens when there is no doubt."""
    a = {_stem(w) for w in keyword_tokens(translation, None)}
    b = {_stem(w) for w in keyword_tokens(english, None)}
    return bool(a) and a == b


def language_pairs(facts: list[dict], user_name: str, min_similarity: float | None) -> list[tuple[float, int, int]]:
    """``(similarity, i, j)`` with ``facts[i]`` in Devanagari and ``facts[j]`` not, best first: up to ``PER_FACT``
    per Devanagari fact. ``min_similarity`` (multilingual embeddings only) drops pairs below it."""
    hindi = [n for n, f in enumerate(facts) if leads_with_devanagari(f["content"])]
    latin = [n for n, f in enumerate(facts) if not has_devanagari(f["content"])]
    if not hindi or not latin:
        return []
    named = {n: bool(subject_aliases(facts[n]["content"], user_name)) for n in hindi + latin}
    out: list[tuple[float, int, int]] = []
    for i in hindi:
        a = facts[i]["content"]
        if not named[i]:
            continue
        found: list[tuple[float, int, int]] = []
        for j in latin:
            b = facts[j]["content"]
            if not named[j] or not same_subject(a, b, user_name) or numbers(a) != numbers(b):
                continue
            if not details_covered(b, a):
                continue
            va, vb = facts[i].get("vector"), facts[j].get("vector")
            sim = cosine(va, vb) if va and vb and len(va) == len(vb) else 0.0
            if min_similarity is not None and sim < min_similarity:
                continue
            found.append((sim, i, j))
        found.sort(key=lambda p: -p[0])
        out.extend(found[:PER_FACT])
    out.sort(key=lambda p: -p[0])
    return out
