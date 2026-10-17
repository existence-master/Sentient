"""Semantic memory: atomic facts with vector recall, dedup, update and decay.

Ported from the v2 memory MCP (``mcp_hub/memory/utils.py``: pgvector + Gemini) onto
SQLite + sqlite-vec with a pluggable embedder. Kept from v2: the eight topics, the
extraction -> CUD pipeline with a full analysis (topics, long/short-term, duration),
short-term expiry, update-by-id with re-analysis, delete/search by source, building
memory from documents, and the similarity graph. Changes:

- Recall returns facts with scores; callers inject them into the system prompt
  (v2 summarized facts into a paragraph with six extra LLM calls per lookup).
- Exact-match and near-duplicate short circuits happen before any LLM call.
- UPDATE keeps the row id and records ``previous_content`` (v2 deleted + re-inserted).
- Expired facts are filtered at query time as well as purged hourly.
- Importing a document adds to memory instead of wiping it (v2 build_initial_memory).
- Search by source works (v2 formatted a prompt with a missing key and crashed).
- Every fact has a scope (#155, ``memory.scopes``): personal, work, always or a project. Recall reads the scopes it
  is given (a chat's own plus always); saving puts the fact in one scope and only compares it with facts in that scope
  and always, so a save never changes a memory in another scope.
- Memories from outside content, unprompted work or imports are held for review (``status = 'pending'``, ADR
  0021): they have no vector, and every read here except ``pending_facts`` skips them, so recall, prompts,
  proactivity, the user model and dreaming never see one until the user approves it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sentient.config.schema import SentientConfig
from sentient.llm.provider import LLMProvider
from sentient.memory import origins, prompts
from sentient.memory import review as reviews
from sentient.memory import scopes as memory_scopes
from sentient.memory.episodic import EpisodicMemory
from sentient.memory.schema import ensure_memory_schema
from sentient.memory.scripts import (
    devanagari_subjects,
    leads_with_devanagari,
    names_match,
    script_words,
)
from sentient.memory.topics import DEFAULT_TOPIC, TOPICS, normalize_topics
from sentient.memory.vectors import VecTable, cosine, pack
from sentient.store.db import Store, now_iso

log = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?", re.IGNORECASE)
_WORD_DURATION_RE = re.compile(r"\b(a|an|one)\s+(minute|hour|day|week|month|year)\b", re.IGNORECASE)
_REQUEST_RE = re.compile(
    r"\b(asked|asks|is asking|wants to know|requested|requests|would like to know|needs help with|is looking for help)\b",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(r"\b(today's date|the current (date|time)|current weather|the weather (is|was))\b", re.IGNORECASE)
# "Maya's sister's job is unknown": a gap, not a fact
_UNKNOWN_RE = re.compile(
    r"\b(is|are|was|remains?) (still )?(unknown|not known|unspecified|not specified|not mentioned|not stated|"
    r"not provided|unclear)\b",
    re.IGNORECASE,
)
_USER_REF_RE = re.compile(r"\b(the user|USERNAME)\b", re.IGNORECASE)
# self-evident facts small models like to emit ("Maya's name is Maya.")
_TAUTOLOGY_RE = re.compile(r"^\s*(?P<n>[\w .'-]+?)['\u2019]s name is (?P=n)\s*\.?\s*$", re.IGNORECASE)

MEMORY_COLUMNS = (
    "id, content, source, topics, memory_type, created_at, updated_at, expires_at, previous_content, status, review,"
    " scope"
)
KEPT_APART_META = "memory.kept_apart"  # pairs the user split again after the tidy-up merged them (#353)
ACTIVE = "status = 'active'"  # held memories (``pending``) are left out of every read but ``pending_facts``

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is", "are", "was", "were", "be",
        "been", "am", "my", "me", "i", "you", "your", "it", "its", "this", "that", "these", "those", "what",
        "whats", "how", "can", "could", "please", "with", "at", "by", "from", "about", "as", "has", "have", "had",
        "do", "does", "did", "not", "no", "so", "than", "then", "there", "their", "they", "them", "he", "she",
        "his", "her", "him", "we", "our", "us", "who", "whom", "which", "when", "where", "why", "will", "would",
        "should", "also", "very", "just", "now", "into", "over", "under", "more", "most", "some", "any", "all",
        "each", "every", "user", "users",
    }
)
_ROLE_LINE_RE = re.compile(r"^\s*(user|assistant|tool|system)\s*:", re.IGNORECASE)


def keyword_tokens(text: str, limit: int | None = 10) -> list[str]:
    """Meaningful lowercase words of ``text`` in order, without stopwords or duplicates."""
    out: list[str] = []
    for w in _TOKEN_RE.findall((text or "").lower()):
        if len(w) > 2 and w not in STOPWORDS and w not in out:
            out.append(w)
            if limit is not None and len(out) >= limit:
                break
    return out


def word_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the meaningful words of two texts (0..1)."""
    wa, wb = set(keyword_tokens(a, None)), set(keyword_tokens(b, None))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_DETAIL_WORD_RE = re.compile(r"[A-Za-z][\w'-]*|\d[\d.,:/-]*")


def new_details(new_fact: str, existing: str) -> set[str]:
    """Names, places and numbers in ``new_fact`` that ``existing`` does not mention.

    Used to overrule a SKIP from the CUD model when the new fact clearly adds detail.
    The first word is ignored (sentence case), as are possessive forms of known words.
    """
    have = {w.lower().removesuffix("'s") for w in _DETAIL_WORD_RE.findall(existing)}
    words = _DETAIL_WORD_RE.findall(new_fact)
    out: set[str] = set()
    for i, w in enumerate(words):
        base = w.removesuffix("'s")
        if base.lower() in have:
            continue
        if base[:1].isdigit() or (i > 0 and base[:1].isupper()):
            out.add(base)
    return out


# ---------------------------------------------------------------------- subjects and attribute families
_RSQUO = chr(0x2019)  # curly apostrophe some models write in possessives
_NAME_WORD_RE = re.compile(r"[A-Za-z][\w'-]*")
_LEAD_SKIP = STOPWORDS | {
    "every", "last", "next", "since", "today", "tonight", "yesterday", "tomorrow", "recently", "currently",
    "lately", "usually", "always", "sometimes", "often", "earlier", "later", "after", "before", "during", "once",
    "still", "morning", "evening", "night", "week", "weekend", "month", "year", "day", "until", "yes", "okay",
    "actually", "update", "correction", "oh", "well",
}
_RELATION_WORDS = {
    "sister", "brother", "mother", "mom", "mum", "father", "dad", "wife", "husband", "partner", "girlfriend",
    "boyfriend", "son", "daughter", "friend", "manager", "boss", "colleague", "cousin", "uncle", "aunt", "grandmother",
    "grandfather", "grandma", "grandpa", "roommate", "cofounder", "co-founder", "fiance", "fiancee", "parents",
    "company", "startup", "team", "dog", "cat",
}
ATTRIBUTE_FAMILIES: dict[str, re.Pattern[str]] = {
    "residence": re.compile(
        r"\b(lives?|living|lived|moved|moves|moving|relocat\w*|based in|resides?|residing|shifted to|settled in|"
        r"stays? in|staying in|home is in|hometown)\b"
    ),
    "job": re.compile(
        r"\b(works?|working|worked|job|employ\w*|hired|joined|quit|resign\w*|promot\w*|intern\w*|position at|"
        r"role at|career|founder|founded|ceo|cto|salary|laid off|fired)\b"
    ),
    "relationship": re.compile(
        r"\b(married|marriage|engaged|dating|girlfriend|boyfriend|wife|husband|divorc\w*|single|broke up|"
        r"relationship|fiance\w*)\b"
    ),
    "diet": re.compile(
        r"\b(vegetarian|vegan|pescatarian|eats?|eating|ate|meat|fish|chicken|beef|pork|seafood|eggs|diet|keto|"
        r"alcohol|sober|teetotal\w*|drinks? (?:alcohol|beer|wine))\b"
    ),
    "health": re.compile(
        r"\b(allerg\w*|medication|medicine|injur\w*|illness|sick|diabet\w*|asthma|surgery|therapy|pregnan\w*)\b"
    ),
    "ownership": re.compile(r"\b(owns?|owned|bought|buys|sold|sells|drives|car|bike|motorcycle|house|pet)\b"),
    "schedule": re.compile(
        r"\b(every (?:day|morning|evening|night|week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"wakes? up|sleeps?|bedtime|routine|schedule|gym|workout|daily|weekly|mondays|tuesdays|wednesdays|thursdays|"
        r"fridays|saturdays|sundays)\b"
    ),
}
# families where two differing facts about one subject usually cannot both hold (checked without a similarity floor)
TIGHT_FAMILIES = frozenset({"residence", "job", "relationship", "diet"})
# FTS terms used by the live CUD path to find the old fact of the same family ("lives in Pune" for "moved to ...")
FAMILY_TERMS: dict[str, list[str]] = {
    "residence": ["lives", "live", "living", "moved", "based", "relocated", "resides"],
    "job": ["works", "work", "job", "joined", "employed", "founder"],
    "relationship": ["married", "engaged", "dating", "girlfriend", "boyfriend", "wife", "husband", "single"],
    "diet": ["vegetarian", "vegan", "eats", "eating", "meat", "fish", "diet"],
    "health": ["allergic", "allergy", "medication"],
    "ownership": ["owns", "bought", "sold", "drives"],
    "schedule": ["every", "daily", "weekly", "routine", "gym"],
}
NEGATION_RE = re.compile(
    r"\b(not|no longer|never|stopped|quit|doesn't|does not|don't|isn't|wasn't|no more|anymore|used to|formerly|former)\b",
    re.IGNORECASE,
)
TIME_WORDING_RE = re.compile(
    r"\b(now|currently|recently|just|lately|these days|nowadays|this (?:week|month|year)|last (?:week|month|year)|"
    r"since|started|starting|moved|switched|new|anymore|no longer)\b",
    re.IGNORECASE,
)


def fact_subjects(text: str, user_name: str = "") -> set[str]:
    """Who a fact is about, as lowercase aliases: {"maya"}, {"maya's sister", "riya"} for
    "Maya's sister Riya ...". "I", "my" and "the user" mean ``user_name``. Empty when unclear."""
    user = user_name.strip().lower() or "user"
    text = (text or "").replace(_RSQUO, "'").strip()
    text = re.sub(r"^(the user's|my)\b", f"{user}'s", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the user|i)\b", user, text, flags=re.IGNORECASE)
    words = _NAME_WORD_RE.findall(text)
    for i, w in enumerate(words[:6]):
        low = w.lower()
        poss = low.endswith("'s")
        root = low[:-2] if poss else low
        is_name = (w[0].isupper() and root not in _LEAD_SKIP) or root == user
        if not is_name:
            if w[0].isupper() or root in _LEAD_SKIP:
                continue  # sentence starters like "Every", "Since", "Last"
            return set()
        nxt = words[i + 1].lower() if i + 1 < len(words) else ""
        if poss and nxt in _RELATION_WORDS:
            aliases = {f"{root}'s {nxt}"}
            after = words[i + 2] if i + 2 < len(words) else ""
            if after[:1].isupper() and after.lower() not in _LEAD_SKIP:
                aliases.add(after.lower().removesuffix("'s"))
            return aliases
        return {root}
    return set()


def attribute_families(text: str) -> set[str]:
    low = (text or "").lower().replace(_RSQUO, "'")
    return {name for name, rx in ATTRIBUTE_FAMILIES.items() if rx.search(low)}


PAST_RE = re.compile(
    r"\b(lived|used to|grew up|as a (?:child|kid|teenager)|born|formerly|previously|childhood|in (?:19|20)\d\d)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FactProfile:
    """Everything ``conflict_strength`` needs, computed once per fact (dreaming compares many pairs)."""

    subjects: frozenset[str]
    families: frozenset[str]
    negation: bool
    details: frozenset[str]  # capitalised words (not the first) and numbers
    words: frozenset[str]  # every word, lowercase, possessive stripped
    tokens: frozenset[str]  # meaningful words for overlap


def profile_fact(text: str, user_name: str = "") -> FactProfile:
    found = _DETAIL_WORD_RE.findall(text or "")
    details = set()
    for i, w in enumerate(found):
        base = w.removesuffix("'s")
        if base[:1].isdigit() or (i > 0 and base[:1].isupper()):
            details.add(base)
    return FactProfile(
        subjects=frozenset(fact_subjects(text, user_name)),
        families=frozenset(attribute_families(text)),
        negation=bool(NEGATION_RE.search(text or "")),
        details=frozenset(details),
        words=frozenset(w.lower().removesuffix("'s") for w in found),
        tokens=frozenset(keyword_tokens(text, None)),
    )


# ---------------------------------------------------------------------- is a new fact about the same thing?
# verbs that name what a preference or habit is about ("does not want files deleted" vs "... written")
_ACTION_VERBS = frozenset({
    "delete", "remove", "erase", "trash", "write", "edit", "change", "modify", "overwrite", "send", "email",
    "message", "text", "call", "phone", "post", "tweet", "share", "publish", "read", "open", "buy", "purchase",
    "order", "pay", "spend", "book", "schedule", "cancel", "move", "rename", "upload", "download", "install",
    "uninstall", "run", "execute", "archive", "forward", "reply", "print", "save", "create", "contact", "disturb",
    "wake", "remind", "notify", "ping", "record", "sign", "submit", "invite", "unsubscribe", "copy", "translate",
})
_IRREGULAR_VERBS = {
    "wrote": "write", "written": "write", "sent": "send", "paid": "pay", "bought": "buy", "spent": "spend",
    "ran": "run", "woke": "wake", "woken": "wake", "rewritten": "write", "rewrote": "write",
}
# words a vaguer fact uses where a fuller one names the thing ("lives in a city" vs "lives in Lisbon")
_PLACEHOLDERS = frozenset({
    "city", "town", "country", "place", "somewhere", "someone", "somebody", "something", "company", "job",
    "person", "area", "region", "state", "thing", "things", "stuff",
})


def fact_actions(text: str) -> set[str]:
    """The actions a fact names, as base verbs: "does not want files to be written" -> {"write"}."""
    out: set[str] = set()
    for w in _TOKEN_RE.findall((text or "").lower().replace(_RSQUO, "'")):
        if w in _IRREGULAR_VERBS:
            out.add(_IRREGULAR_VERBS[w])
            continue
        forms = [w, w[:-1], w[:-2], w[:-3], w[:-3] + "e", w[:-4]]  # deletes, deleted, sending, writing, shipping
        if w.endswith("ing") or w.endswith("ed") or w.endswith("s") or w in _ACTION_VERBS:
            hit = next((f for f in forms if len(f) >= 3 and f in _ACTION_VERBS), None)
            if hit:
                out.add(hit)
    return out


def subject_aliases(text: str, user_name: str = "") -> set[str]:
    """:func:`fact_subjects`, or for a fact written in Devanagari (Hindi...) its subject from
    :func:`scripts.devanagari_subjects`, which finds names without capital letters (#339)."""
    if leads_with_devanagari(text):
        return devanagari_subjects(text, user_name)
    return fact_subjects(text, user_name)


def _subject_keys(subjects: set[str], users: set[str]) -> set[tuple[str, str]]:
    """(owner, relation) per subject, with every name for the user as "user": "maya's sister" -> ("user", "sister")."""
    out = set()
    for s in subjects:
        owner, _, rel = s.partition("'s ")
        out.add(("user" if any(names_match(owner, u) for u in users) else owner, rel))
    return out


def same_subject(a: str, b: str, user_name: str = "") -> bool:
    """False only when both facts clearly name different people or things ("Maya's sister" vs "Maya's brother",
    "प्रिया" vs "Riya"). Names match across scripts ("माया" is "Maya"). Unclear subjects count as the same, and so
    does "the user" next to a name while the user's name isn't set."""
    sa, sb = subject_aliases(a, user_name), subject_aliases(b, user_name)
    if not sa or not sb or sa & sb:
        return True
    users = {"user", "the user", (user_name or "").strip().lower()} - {""}
    unnamed = not (user_name or "").strip()
    ka, kb = _subject_keys(sa, users), _subject_keys(sb, users)
    for oa, ra in ka:
        for ob, rb in kb:
            if ra == rb and (names_match(oa, ob) or (unnamed and "user" in (oa, ob))):
                return True
    return False


def same_thing(new_fact: str, existing: str, user_name: str = "") -> bool:
    """True when ``new_fact`` may replace ``existing``: the same subject and, where both name one, the same action.
    "does not want their files deleted" and "does not want files to be written" are two preferences, and so are
    "does not want her files deleted" and "does not want her emails deleted". A replacement must also keep every
    action ``existing`` names: "doesn't want files deleted" never replaces "doesn't want files deleted or sent"."""
    if not same_subject(new_fact, existing, user_name):
        return False
    acts_new, acts_old = fact_actions(new_fact), fact_actions(existing)
    if not acts_new or not acts_old:
        return True
    if not acts_new & acts_old or acts_old - acts_new:
        return False
    if NEGATION_RE.search(new_fact) and NEGATION_RE.search(existing):
        # two "does not want X deleted" rules about different things ("files", "emails") both hold
        shared = acts_new & acts_old
        a, b = _object_words(new_fact, shared), _object_words(existing, shared)
        return not (a - b and b - a)
    return True


# ---------------------------------------------------------------------- is a new fact a correction of an old one?
# words a correction adds that say nothing about the thing itself ("Mira's school NOW starts in July")
_CORRECTION_FILLER = frozenset({
    "now", "actually", "instead", "anymore", "change", "changed", "changes", "update", "updated", "correction",
    "new", "longer", "turn", "turned", "out", "really", "rather", "fact", "since",
})
_NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty",
    "thirty", "forty", "fifty", "hundred", "dozen",
})
# the old value a correction names as replaced: "from Pune to Delhi", "in July, not June", "instead of 5"
_REPLACED_RE = re.compile(
    r"\b(?:from|instead of|rather than|not|no longer)\s+(?:the\s+|a\s+|an\s+)?([A-Z][\w'-]*|\d[\d.,:/-]*\w*)"
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _values(text: str) -> dict[str, str]:
    """The names, places, dates and numbers a fact states, lowercase -> as written: capitalised words (not the
    first), numbers and number words."""
    out: dict[str, str] = {}
    for i, w in enumerate(_DETAIL_WORD_RE.findall((text or "").replace(_RSQUO, "'"))):
        base = w.removesuffix("'s").rstrip(".,")
        if base[:1].isdigit() or (i > 0 and base[:1].isupper()) or base.lower() in _NUMBER_WORDS:
            out.setdefault(base.lower(), base)
    return out


def _frame(text: str, drop: set[str]) -> set[str]:
    """What a fact says about its subject: its plain words, stemmed, without the subject, values and fillers."""
    return {
        _stem(w) for w in keyword_tokens(text, None)
        if w not in drop and w not in _CORRECTION_FILLER and not any(ch.isdigit() for ch in w)
    }


def correction_of(new_fact: str, existing: str, user_name: str = "") -> str | None:
    """When ``new_fact`` corrects ``existing``, the fact to store in its place, else None (#364).

    A correction is about the same subject and action (:func:`same_thing`), says the same thing about it, and
    changes a value: a date, place, number or name ``existing`` states is gone and a new one is there ("Mira's
    school now starts in July" for "Maya's daughter Mira starts school in June"). With one value swapped for another
    the old wording is kept with the new value ("... starts school in July"); otherwise the new fact replaces it."""
    if not new_fact.strip() or not existing.strip():
        return None
    new_fact, existing = new_fact.replace(_RSQUO, "'"), existing.replace(_RSQUO, "'")
    core = _REPLACED_RE.sub(" ", new_fact)  # "moved from Pune to Delhi" states Delhi; Pune is the old value
    if bool(NEGATION_RE.search(core)) != bool(NEGATION_RE.search(existing)):
        return None
    if bool(PAST_RE.search(core)) != bool(PAST_RE.search(existing)):
        return None
    if not same_thing(core, existing, user_name):
        return None
    new_vals, old_vals = _values(core), _values(existing)
    core_words = {w.lower().removesuffix("'s") for w in _DETAIL_WORD_RE.findall(core)}
    old_words = {w.lower().removesuffix("'s") for w in _DETAIL_WORD_RE.findall(existing)}
    gained = {k: v for k, v in new_vals.items() if k not in old_words}
    lost = {k: v for k, v in old_vals.items() if k not in core_words}
    if not gained or not lost:
        return None
    subjects = subject_aliases(core, user_name) | subject_aliases(existing, user_name)
    drop = {w for s in subjects for w in _TOKEN_RE.findall(s)} | set(new_vals) | set(old_vals)
    drop |= {"user", (user_name or "").strip().lower()}
    frame_new, frame_old = _frame(core, drop), _frame(existing, drop)
    tight = attribute_families(core) & attribute_families(existing) & TIGHT_FAMILIES
    union = frame_new | frame_old
    alike = bool(frame_old) and (frame_old <= frame_new or len(frame_new & frame_old) / len(union) >= 0.6)
    if not (alike or tight):
        return None
    if len(gained) == 1 and len(lost) == 1 and frame_new <= frame_old:
        [(old_key, _)] = lost.items()
        [new_value] = gained.values()
        swapped, n = re.subn(rf"\b{re.escape(old_key)}\b", new_value, existing, count=1, flags=re.IGNORECASE)
        if n:
            return swapped.strip()
    if len(lost) > len(gained):
        # a value would go with nothing in its place ("works at Infosys in Pune" -> "got a job at Google"): the model
        # decides, and its update may not drop a name, place or number either
        return None
    return new_fact.strip()


def says_something_else(new_fact: str, existing: str, user_name: str = "") -> bool:
    """True when two facts about the same person say unrelated things: no attribute in common (where they live, job,
    diet...) and no word of what they say about it. "Mira has a piano recital in July" never replaces "Mira starts
    school in July", although both are about Mira in July (#364)."""
    if attribute_families(new_fact) & attribute_families(existing):
        return False
    subjects = subject_aliases(new_fact, user_name) | subject_aliases(existing, user_name)
    drop = {w for s in subjects for w in _TOKEN_RE.findall(s)} | set(_values(new_fact)) | set(_values(existing))
    drop |= {"user", (user_name or "").strip().lower()}
    a, b = _frame(new_fact, drop), _frame(existing, drop)
    return bool(a) and bool(b) and not a & b


def changes_value(new_fact: str, existing: str, user_name: str = "") -> bool:
    """A looser reading for the model's hint: about the same thing, and each fact states a value the other lacks."""
    if not same_thing(new_fact, existing, user_name):
        return False
    return bool(new_details(new_fact, existing)) and bool(new_details(existing, new_fact))


# ways a fact names the assistant itself ("doesn't want Sentient to ..." = "doesn't want the assistant to ...")
_ASSISTANT_WORDS = frozenset({"assistant", "sentient", "ai", "bot"})


def _plain_words(text: str) -> set[str]:
    return {w.removesuffix("s") for w in keyword_tokens(text, None) if w not in _ASSISTANT_WORDS}


def _object_words(text: str, actions: set[str]) -> set[str]:
    """The plain words of a fact without the given action verbs and without numbers ("9am"), which are values.
    "emails" stays an object when the action is "delete"."""
    return {
        w for w in _plain_words(text)
        if not (fact_actions(w) and fact_actions(w) <= actions) and not any(ch.isdigit() for ch in w)
    }


def _only_names_relation(new_fact: str, existing: str, user_name: str) -> bool:
    """True when ``new_fact`` only says a relation exists ("Maya has a sister") that ``existing`` is about
    ("Maya's sister Meera lives in Lisbon")."""
    owners = {s for s in fact_subjects(new_fact, user_name) if "'s " not in s}
    rels = set()
    for s in fact_subjects(existing, user_name):
        owner, _, rel = s.partition("'s ")
        if rel and owner in owners:
            rels.add(rel.removesuffix("s"))
    words = _plain_words(new_fact) - owners - {"user", (user_name or "").strip().lower()}
    return bool(rels) and bool(words) and words <= rels | _PLACEHOLDERS


def covered_by(new_fact: str, existing: str, user_name: str = "") -> bool:
    """True when ``existing`` already says everything ``new_fact`` says: same subject, same negation and tense, no
    new name, place or number, and every other word is in it (or a placeholder like "a city" that it names). "Maya
    has a sister" is covered by any fact about Maya's sister."""
    if not new_fact.strip() or not existing.strip():
        return False
    if bool(NEGATION_RE.search(new_fact)) != bool(NEGATION_RE.search(existing)):
        return False
    if bool(PAST_RE.search(new_fact)) != bool(PAST_RE.search(existing)):
        return False
    if new_details(new_fact, existing) or script_words(new_fact) - script_words(existing):
        # a new name, place or number, or words in another script (Hindi) that the English word checks below miss
        return False
    if not same_subject(new_fact, existing, user_name):
        return _only_names_relation(new_fact, existing, user_name)
    have = _plain_words(existing)
    user = (user_name or "").strip().lower()
    have.update(w for w in (user, "user") if w)
    missing = _plain_words(new_fact) - have
    if missing - _PLACEHOLDERS:
        return False
    if missing:
        # a placeholder ("lives in a city") is only covered by a fact that names it ("lives in Lisbon"): the same
        # kind of attribute plus a name, place or number the new fact lacks. "has a job" is not covered by "has a dog"
        shared = attribute_families(new_fact) & attribute_families(existing)
        return bool(shared) and bool(new_details(existing, new_fact))
    return True


def conflict_strength(a: str | FactProfile, b: str | FactProfile, user_name: str = "") -> int:
    """0: different subjects, or duplicates. 1: same subject and only a broad shared attribute (needs a
    similarity check). 2: same subject and a tight attribute (residence, job, relationship, diet), or
    differing names/places/numbers, or one side negated."""
    pa = a if isinstance(a, FactProfile) else profile_fact(a, user_name)
    pb = b if isinstance(b, FactProfile) else profile_fact(b, user_name)
    if not pa.subjects & pb.subjects:
        return 0
    da = any(d.lower() not in pb.words for d in pa.details)
    db = any(d.lower() not in pa.words for d in pb.details)
    union = pa.tokens | pb.tokens
    overlap = len(pa.tokens & pb.tokens) / len(union) if union else 0.0
    if overlap >= 0.85 and not da and not db:
        return 0
    shared = pa.families & pb.families
    differing = da and db
    negation = pa.negation != pb.negation
    if shared & TIGHT_FAMILIES or (shared and (differing or negation)):
        return 2
    if overlap >= 0.4 and (differing or negation):
        return 2
    return 1 if shared else 0


def parse_duration(text: str | None) -> timedelta | None:
    if not text:
        return None
    m = _DURATION_RE.search(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
    else:
        m = _WORD_DURATION_RE.search(text)
        if not m:
            return None
        n, unit = 1, m.group(2).lower()
    return {
        "minute": timedelta(minutes=n),
        "hour": timedelta(hours=n),
        "day": timedelta(days=n),
        "week": timedelta(weeks=n),
        "month": timedelta(days=30 * n),
        "year": timedelta(days=365 * n),
    }[unit]


def expires_for(memory_type: str, duration: str | None, now: datetime | None = None) -> str | None:
    if memory_type != "short-term":
        return None
    delta = parse_duration(duration)
    if delta is None:
        delta = timedelta(days=7)
    return ((now or datetime.now(UTC)) + delta).isoformat()


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into chunks of at most ``max_chars``."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        while len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def _row_to_memory(r: Any, similarity: float | None = None) -> dict:
    d = {
        "id": int(r["id"]),
        "content": r["content"],
        "topics": json.loads(r["topics"] or "[]"),
        "source": r["source"],
        "memory_type": r["memory_type"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "expires_at": r["expires_at"],
        "previous_content": r["previous_content"] if "previous_content" in r.keys() else None,  # noqa: SIM118 - sqlite Row, where `in row` checks values
        "status": r["status"] if "status" in r.keys() else "active",  # noqa: SIM118
        "review": reviews.load(r["review"]) if "review" in r.keys() else None,  # noqa: SIM118
        "scope": (r["scope"] if "scope" in r.keys() else None) or memory_scopes.PERSONAL,  # noqa: SIM118
    }
    if similarity is not None:
        d["similarity"] = round(similarity, 4)
    return d


def _in_scopes(scopes: list[str] | None, col: str = "scope") -> tuple[str, list[str]]:
    """SQL condition (with its parameters) that keeps facts in ``scopes``; ``("1", [])`` for every scope."""
    if scopes is None:
        return "1", []
    return f"{col} IN ({','.join('?' * len(scopes))})", list(scopes)


def _analysis_from(raw: Any) -> tuple[list[str] | None, str, str | None]:
    """Accept both the v2 nested ``analysis`` object and flat keys."""
    if not isinstance(raw, dict):
        return None, "long-term", None
    src = raw.get("analysis") if isinstance(raw.get("analysis"), dict) else raw
    topics = normalize_topics(src["topics"]) if src.get("topics") else None
    memory_type = str(src.get("memory_type") or "long-term").strip().lower()
    if memory_type not in {"long-term", "short-term"}:
        memory_type = "short-term" if "short" in memory_type else "long-term"
    duration = src.get("duration")
    return topics, memory_type, (str(duration) if duration else None)


class FactMemory:
    def __init__(self, store: Store, llm: LLMProvider, config: SentientConfig, bus: Any = None):
        self.store = store
        self.llm = llm
        self.config = config
        self.bus = bus  # EventBus; set by the evolution service so changes reach the UI
        self.vec = VecTable(store, "facts_vec")
        self.episodic = EpisodicMemory(store, llm, config)
        self._qcache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    async def ensure_schema(self) -> None:
        await ensure_memory_schema(self.store)

    # ------------------------------------------------------------------ events
    def publish(self, action: str, fid: int | None, content: str | None = None, **extra: Any) -> None:
        if self.bus is not None:
            self.bus.publish("memory.updated", {"action": action, "id": fid, "content": content, **extra})

    # ------------------------------------------------------------------ vectors
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = await self.llm.embed(texts)
        if vecs:
            await self.vec.ensure(len(vecs[0]), self.reindex, self.llm.model_for("embedding"))
        return vecs

    async def reindex(self) -> int:
        """Re-embed every fact (embedding model or metric changed)."""
        rows = await self.store.fetchall(f"SELECT id, content FROM facts WHERE {ACTIVE} ORDER BY id")
        done = 0
        for i in range(0, len(rows), 32):
            batch = rows[i : i + 32]
            vecs = await self.llm.embed([r["content"] for r in batch])
            for r, v in zip(batch, vecs, strict=False):
                await self.vec.upsert(int(r["id"]), v)
                done += 1
        await self.store.execute("UPDATE facts SET embedding_model = ?", (self.llm.model_for("embedding"),))
        return done

    async def embed_query(self, text: str) -> list[float]:
        """Embedding of a query, cached briefly: the system prompt, the user model and the
        recall tool often embed the same user message within one turn."""
        key = (self.llm.model_for("embedding"), text)
        cached = self._qcache.get(key)
        if cached is not None and self.vec.ready:
            self._qcache.move_to_end(key)
            return cached
        [vec] = await self._embed([text])
        self._qcache[key] = vec
        while len(self._qcache) > 64:
            self._qcache.popitem(last=False)
        return vec

    # ------------------------------------------------------------------ recall
    async def _keyword_ids(
        self, tokens: list[str], limit: int, source: str | None, scopes: list[str] | None = None
    ) -> list[int]:
        if not tokens:
            return []
        in_scope, scope_params = _in_scopes(scopes, "f.scope")
        sql = (
            "SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
            f" WHERE facts_fts MATCH ? AND f.status = 'active' AND {in_scope}"
        )
        params: list[Any] = [" OR ".join(f'"{t}"' for t in tokens), *scope_params]
        if source is not None:
            sql += " AND f.source = ?"
            params.append(source)
        sql += " ORDER BY bm25(facts_fts) LIMIT ?"
        params.append(limit)
        try:
            return [int(r["id"]) for r in await self.store.fetchall(sql, params)]
        except Exception as exc:
            log.debug("keyword recall unavailable: %s", exc)
            return []

    async def _search(
        self, query: str, k: int, min_sim: float, source: str | None = None, scopes: list[str] | None = None
    ) -> list[dict]:
        """Hybrid recall: vector neighbours plus FTS5 keyword matches, ranked by
        ``similarity + keyword_weight * share of query words present``. Embedding models
        compress similarities (unrelated text still scores ~0.65), so exact words help ordering.
        ``scopes`` keeps facts in those scopes only (#155); None reads every scope."""
        await self.ensure_schema()
        qvec = await self.embed_query(query)
        now = now_iso()
        pool = max(k * 3, 10)
        in_scope, scope_params = _in_scopes(scopes, "f.scope")
        if source is None and scopes is None:
            sims = dict(await self.vec.knn(qvec, pool))
        else:
            by_source = "f.source = ?" if source is not None else "1"
            rows = await self.store.fetchall(
                "SELECT f.id, vec_distance_cosine(v.embedding, ?) AS distance"
                f" FROM facts f JOIN facts_vec v ON v.rowid = f.id WHERE {by_source} AND {in_scope}"
                " AND f.status = 'active' ORDER BY distance LIMIT ?",
                (pack(qvec), *([source] if source is not None else []), *scope_params, pool),
            )
            sims = {int(r["id"]): 1.0 - float(r["distance"]) for r in rows}
        weight = self.config.memory.keyword_weight
        tokens = keyword_tokens(query)
        kw_ids = await self._keyword_ids(tokens, pool, source, scopes) if weight > 0 else []
        missing = [i for i in kw_ids if i not in sims]
        if missing:
            marks = ",".join("?" * len(missing))
            rows = await self.store.fetchall(
                "SELECT f.id, vec_distance_cosine(v.embedding, ?) AS distance"
                f" FROM facts f JOIN facts_vec v ON v.rowid = f.id WHERE f.id IN ({marks})",
                [pack(qvec), *missing],
            )
            sims.update({int(r["id"]): 1.0 - float(r["distance"]) for r in rows})
        if not sims:
            return []
        marks = ",".join("?" * len(sims))
        in_scope, scope_params = _in_scopes(scopes)
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE id IN ({marks}) AND {ACTIVE} AND {in_scope}",
            [*sims, *scope_params],
        )
        qset = set(tokens)
        scored = []
        for r in rows:
            if r["expires_at"] and r["expires_at"] < now:
                continue
            sim = sims[int(r["id"])]
            share = len(qset & set(keyword_tokens(r["content"], None))) / len(qset) if qset and weight > 0 else 0.0
            if sim < min_sim and share < 0.5:
                continue
            m = _row_to_memory(r, sim)
            m["score"] = round(sim + weight * share, 4)
            scored.append(m)
        scored.sort(key=lambda m: (-m["score"], -m["similarity"]))
        return scored[:k]

    async def recall(
        self,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        *,
        source: str | None = None,
        track: bool = True,
        scopes: list[str] | None = None,
    ) -> list[dict]:
        """Relevant facts for ``query``. ``track`` counts the recall (dreaming promotes
        short-term facts that keep being recalled); internal lookups pass False. ``scopes``: only facts in these
        scopes (``memory.scopes.recall_scopes``: a chat's scope plus always); None reads every scope."""
        top_k = top_k if top_k is not None else self.config.memory.facts_top_k
        min_sim = min_similarity if min_similarity is not None else self.config.memory.min_similarity
        if top_k <= 0 or not query.strip():
            return []
        try:
            out = await self._search(query, top_k, min_sim, source, scopes)
        except Exception as exc:
            log.warning("memory recall skipped: %s", exc)
            return []
        if track and out:
            ids = [m["id"] for m in out]
            with contextlib.suppress(Exception):
                await self.store.execute(
                    "UPDATE facts SET recall_count = COALESCE(recall_count, 0) + 1, last_recalled_at = ?"
                    f" WHERE id IN ({','.join('?' * len(ids))})",
                    [now_iso(), *ids],
                )
        return out

    async def search_by_source(
        self, query: str, source: str, top_k: int = 5, scopes: list[str] | None = None
    ) -> list[dict]:
        return await self.recall(query, top_k=top_k, min_similarity=0.0, source=source, track=False, scopes=scopes)

    # ------------------------------------------------------------------ write paths
    async def _insert(
        self,
        content: str,
        vec: list[float],
        *,
        source: str,
        topics: list[str],
        memory_type: str,
        duration: str | None,
        review: dict | None = None,
        scope: str = memory_scopes.PERSONAL,
    ) -> int:
        """A held fact (``review`` set) is saved as pending without a vector, so recall cannot find it."""
        await self.ensure_schema()
        ts = now_iso()
        cur = await self.store.execute(
            "INSERT INTO facts(content, source, topics, memory_type, created_at, updated_at, expires_at, embedding_model,"
            " status, review, scope) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                content, source, json.dumps(topics), memory_type, ts, ts,
                expires_for(memory_type, duration), self.llm.model_for("embedding"),
                "pending" if review is not None else "active", reviews.dump(review), scope,
            ),
        )
        fid = int(cur.lastrowid)
        if review is None:
            await self.vec.upsert(fid, vec)
        return fid

    async def _update(
        self, fid: int, content: str, vec: list[float], topics: list[str], memory_type: str, duration: str | None
    ) -> None:
        row = await self.store.fetchone("SELECT content, status FROM facts WHERE id = ?", (fid,))
        prev = row["content"] if row else None
        await self.store.execute(
            "UPDATE facts SET content = ?, topics = ?, memory_type = ?, expires_at = ?, updated_at = ?,"
            " previous_content = ?, embedding_model = ? WHERE id = ?",
            (
                content, json.dumps(topics), memory_type, expires_for(memory_type, duration), now_iso(),
                prev, self.llm.model_for("embedding"), fid,
            ),
        )
        if row is None or row["status"] == "active":  # a held fact stays out of the vector index until approved
            await self.vec.upsert(fid, vec)

    async def analyze(self, text: str) -> tuple[list[str], str, str | None]:
        """v2 fact analysis: topics, long/short-term and duration for one fact."""
        try:
            raw = await self.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.FACT_ANALYSIS_SYSTEM},
                    {"role": "user", "content": prompts.FACT_ANALYSIS_USER.format(text=text)},
                ],
            )
            topics, memory_type, duration = _analysis_from(raw)
            return topics or [DEFAULT_TOPIC], memory_type, duration
        except Exception as exc:
            log.warning("fact analysis failed: %s", exc)
            return [DEFAULT_TOPIC], "long-term", None

    async def forget(self, fid: int, *, notify: bool = False) -> bool:
        """Delete a fact, with the copies in other languages the tidy-up merged into it (#353)."""
        await self.ensure_schema()
        row = await self.store.fetchone("SELECT content FROM facts WHERE id = ?", (fid,))
        await self.store.execute(
            "DELETE FROM fact_sources WHERE fact_id = ?"
            " OR fact_id IN (SELECT id FROM facts WHERE merged_into = ? AND status = 'merged')",
            (fid, fid),
        )
        await self.store.execute("DELETE FROM facts WHERE merged_into = ? AND status = 'merged'", (fid,))
        cur = await self.store.execute("DELETE FROM facts WHERE id = ?", (fid,))
        await self.vec.delete(fid)
        deleted = cur.rowcount > 0
        if deleted and notify:
            self.publish("DELETE", fid, row["content"] if row else None)
        return deleted

    # ------------------------------------------------------------------ sources (#153)
    async def link_sources(self, fid: int, sources: list[str]) -> None:
        """Record that fact ``fid`` was also learned from ``sources`` (its own ``source`` is never linked twice)."""
        await self.ensure_schema()
        row = await self.store.fetchone("SELECT source FROM facts WHERE id = ?", (fid,))
        if row is None:
            return
        ts = now_iso()
        for src in dict.fromkeys(s for s in sources if s and s != row["source"]):
            await self.store.execute(
                "INSERT OR IGNORE INTO fact_sources(fact_id, source, linked_at) VALUES(?,?,?)", (fid, src, ts)
            )

    async def _linked(self, fid: int) -> list[str]:
        rows = await self.store.fetchall(
            "SELECT source FROM fact_sources WHERE fact_id = ? ORDER BY rowid", (fid,)
        )
        return [str(r["source"]) for r in rows]

    async def sources_of(self, fid: int) -> list[str]:
        """Every source of fact ``fid``: its own first, then the others it was learned from, then those of copies
        the tidy-up merged into it (#353). Empty when it doesn't exist."""
        await self.ensure_schema()
        row = await self.store.fetchone("SELECT source FROM facts WHERE id = ?", (fid,))
        if row is None:
            return []
        out = [str(row["source"]), *await self._linked(fid)]
        merged = await self.store.fetchall(
            "SELECT id, source FROM facts WHERE merged_into = ? AND status = 'merged' ORDER BY id", (fid,)
        )
        for m in merged:
            out += [str(m["source"]), *await self._linked(int(m["id"]))]
        return list(dict.fromkeys(out))

    async def _from_source(self, source: str) -> list[int]:
        await self.ensure_schema()
        rows = await self.store.fetchall(
            "SELECT id FROM facts WHERE source = ? UNION SELECT fact_id AS id FROM fact_sources WHERE source = ?"
            " ORDER BY id",
            (source, source),
        )
        return [int(r["id"]) for r in rows]

    async def source_counts(self, source: str) -> dict[str, int]:
        """What forgetting ``source`` would do: ``memories`` deleted (no other source backs them) and ``shared``
        kept because another source still does. Copies merged into another memory are counted with it."""
        deleted = shared = 0
        for fid in await self._from_source(source):
            row = await self.store.fetchone("SELECT status FROM facts WHERE id = ?", (fid,))
            if row is None or row["status"] == "merged":
                continue
            if [s for s in await self.sources_of(fid) if s != source]:
                shared += 1
            else:
                deleted += 1
        return {"memories": deleted, "shared": shared}

    async def purge_source(self, source: str) -> dict[str, int]:
        """Forget what ``source`` taught: a fact only it backs is deleted with its vector and merged copies; a fact
        another source backs too loses this source and keeps the others. ``{deleted, unlinked}``."""
        deleted = unlinked = 0
        for fid in await self._from_source(source):
            row = await self.store.fetchone("SELECT source, status FROM facts WHERE id = ?", (fid,))
            if row is None:  # gone with the fact it was merged into
                continue
            merged = row["status"] == "merged"
            others = [s for s in (await self._linked(fid) if merged else await self.sources_of(fid)) if s != source]
            if merged and row["source"] != source:
                others.insert(0, str(row["source"]))
            if not others:
                deleted += 0 if merged else 1
                await self.forget(fid)
                continue
            await self.store.execute("DELETE FROM fact_sources WHERE fact_id = ? AND source = ?", (fid, source))
            if row["source"] == source:  # its next source becomes its own
                await self.store.execute("UPDATE facts SET source = ? WHERE id = ?", (others[0], fid))
                await self.store.execute("DELETE FROM fact_sources WHERE fact_id = ? AND source = ?", (fid, others[0]))
            unlinked += 0 if merged else 1
        return {"deleted": deleted, "unlinked": unlinked}

    async def forget_source(self, source: str) -> int:
        """v2 delete_memory_by_source: forget what ``source`` taught (:meth:`purge_source`); the number deleted."""
        done = await self.purge_source(source)
        if done["deleted"] or done["unlinked"]:
            self.publish("DELETE", None, None, source=source, count=done["deleted"], unlinked=done["unlinked"])
        return done["deleted"]

    async def set_scope(self, fid: int, scope: str) -> dict:
        """Move a memory to another scope (#155). KeyError when it doesn't exist, ValueError for an empty scope."""
        target = memory_scopes.normalize(scope)
        if not target:
            raise ValueError("scope is empty")
        row = await self.store.fetchone("SELECT content, scope FROM facts WHERE id = ?", (fid,))
        if row is None:
            raise KeyError(fid)
        if row["scope"] != target:
            await self.store.execute(
                "UPDATE facts SET scope = ?, updated_at = ? WHERE id = ?", (target, now_iso(), fid)
            )
            self.publish("UPDATE", fid, row["content"], scope=target, previous_scope=row["scope"], reason="moved")
        fact = await self.get_fact(fid)
        assert fact is not None
        return fact

    async def forget_scope(self, scope: str) -> int:
        """Reset one scope: delete every memory in it (also ones waiting for review), and nothing else."""
        target = memory_scopes.normalize(scope)
        if not target:
            return 0
        rows = await self.store.fetchall(
            "SELECT id FROM facts WHERE scope = ? AND status IN ('active', 'pending')", (target,)
        )
        for r in rows:
            await self.forget(int(r["id"]))
        if rows:
            self.publish("DELETE", None, None, scope=target, reason="scope_reset", count=len(rows))
        return len(rows)

    async def scope_of(self, fid: int) -> str | None:
        row = await self.store.fetchone("SELECT scope FROM facts WHERE id = ?", (fid,))
        return (row["scope"] or memory_scopes.PERSONAL) if row else None

    async def _related_facts(
        self, fact: str, exclude: set[int], scopes: list[str] | None = None
    ) -> list[dict]:
        """Up to 3 facts about the same subject and attribute family as ``fact``, found by keyword rather
        than embedding ("moved to Bengaluru last month" scores low against "lives in Pune")."""
        user = self.config.assistant.user_name
        prof = profile_fact(fact, user)
        if not prof.subjects or not prof.families:
            return []
        terms = [t for s in sorted(prof.subjects) for t in keyword_tokens(s)]
        for fam in sorted(prof.families):
            terms.extend(FAMILY_TERMS.get(fam, []))
        ids = [
            i for i in await self._keyword_ids(list(dict.fromkeys(terms))[:24], 60, None, scopes) if i not in exclude
        ]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE id IN ({marks}) AND {ACTIVE}"
            " AND (expires_at IS NULL OR expires_at > ?)",
            [*ids, now_iso()],
        )
        out = []
        for r in rows:
            other = profile_fact(r["content"], user)
            if prof.families & other.families and conflict_strength(prof, other) > 0:
                out.append(_row_to_memory(r))
        out.sort(key=lambda m: m["updated_at"] or "", reverse=True)
        return out[:3]

    def _duplicate(self, fact: str, neighbour: dict) -> bool:
        """A near-identical fact, skipped without asking the model: above ``duplicate_similarity``, about the same
        person or thing, and with no word in Hindi (or another script without capitals) that the other lacks.
        Embedding models score short texts in one script very close together (#339)."""
        other = neighbour["content"]
        return (
            neighbour["similarity"] >= self.config.memory.duplicate_similarity
            and same_subject(fact, other, self.config.assistant.user_name)
            and not script_words(fact) - script_words(other)
        )

    def _residence_peer(self, fact: str, candidates: list[dict]) -> dict | None:
        """The current residence fact a new residence fact replaces ("moved to X" vs "lives in Y"), if any.
        Past-tense facts ("lived in Nagpur as a child") are history and never replaced."""
        user = self.config.assistant.user_name
        if PAST_RE.search(fact) or "residence" not in attribute_families(fact):
            return None
        subjects = fact_subjects(fact, user)
        if not subjects:
            return None
        peers = [
            n for n in candidates
            if "residence" in attribute_families(n["content"])
            and subjects & fact_subjects(n["content"], user)
            and not PAST_RE.search(n["content"])
            and new_details(fact, n["content"])
        ]
        return max(peers, key=lambda n: n.get("updated_at") or "") if peers else None

    async def remember(
        self,
        fact: str,
        *,
        source: str = "conversation",
        use_llm: bool = True,
        notify: bool = False,
        review: dict | None = None,
        own_words: str | None = None,
        session_id: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Store one fact with dedup / CUD logic in ``scope`` (#155; personal when None). Returns {action, id, content}
        (an UPDATE adds ``previous``, the words it replaced, so the save can be undone; an ADD adds ``scope``).

        With ``review`` (``memory.review.note``) the fact is held for the user's review instead: it is only ever
        added (never updates or deletes another fact), and the result carries ``status: "pending"``. ``own_words`` is
        what the user typed or said in the chat it came from: a fact those words fully back is saved even in a chat
        that read outside content (#234, ``review.grounded``). A fact from a chat that reads like an instruction for
        how Sentient should act is always held (``review.instruction_shaped``). ``session_id`` goes on the
        ``memory.updated`` event, so that chat can show "Remembered: ... Undo". Duplicates, updates and corrections
        only look at facts in ``scope`` and always: a save never changes a memory in another scope."""
        user = self.config.assistant.user_name
        scope = memory_scopes.normalize(scope, memory_scopes.PERSONAL) or memory_scopes.PERSONAL
        if review is not None and own_words and reviews.grounded(fact, own_words, user):
            log.info("kept from the user's own words although the chat read outside content: %r", fact)
            review = None
        if source == "conversation" and reviews.instruction_shaped(fact):
            log.info("held for review as an instruction: %r", fact)
            if review is None:
                review = reviews.instruction_note(reviews.best_snippet(own_words or "", fact) or fact, session_id)
            else:  # held for outside content already: it keeps that source, and says it is an instruction too
                review = {**review, "reason": reviews.INSTRUCTION}
        others: list[str] = []
        if source == origins.CONVERSATION and session_id:
            # a chat in a messaging app or on a device is that app's (#153), and also the apps and folders it read
            try:
                source, others = await origins.of_session(self.store, session_id)
            except Exception as exc:
                log.debug("sources of chat %s unknown: %s", session_id, exc)
            if review is not None and review.get("reason") != reviews.INSTRUCTION and others:
                # held because the chat read outside content: it came from what was read, not from the user
                source, others = others[0], others[1:]
            else:  # the user's own words, or the user's own notes it read
                others = [o for o in others if o.startswith("folder:")]
        if review is not None:
            result = await self._hold(fact, source=source, use_llm=use_llm, review=review, scope=scope)
        else:
            result = await self._remember(fact, source=source, use_llm=use_llm, scope=scope)
        if result.get("id") is not None and result["action"] in {"ADD", "UPDATE", "SKIP"}:
            # learned here too: forgetting one of its sources keeps it while another still backs it (#153)
            await self.link_sources(int(result["id"]), [source, *others])
        if result["action"] == "ADD":
            result["scope"] = scope
        if notify and result["action"] in {"ADD", "UPDATE", "DELETE"}:
            self.publish(result["action"], result["id"], result["content"], **self.event_extra(result, session_id))
        return result

    @staticmethod
    def event_extra(result: dict, session_id: str | None = None) -> dict[str, Any]:
        """The extra ``memory.updated`` fields for one save: ``status`` and ``reason`` when held, ``previous`` for an
        update, and the chat it came from (#234)."""
        extra: dict[str, Any] = {}
        if result.get("status") == "pending":
            extra["status"] = "pending"
            if result.get("reason"):
                extra["reason"] = result["reason"]
        if result.get("previous") is not None:
            extra["previous"] = result["previous"]
        if session_id:
            extra["session_id"] = session_id
        if result.get("scope"):
            extra["scope"] = result["scope"]
        return extra

    async def _hold(
        self, fact: str, *, source: str, use_llm: bool, review: dict, scope: str = memory_scopes.PERSONAL
    ) -> dict[str, Any]:
        """Save ``fact`` as pending. Duplicates of any fact (held or not) in its scope or always are skipped; nothing
        else changes."""
        fact = fact.strip()
        if not fact:
            return {"action": "SKIP", "id": None, "content": fact}
        await self.ensure_schema()
        family = memory_scopes.recall_scopes(scope)
        in_scope, scope_params = _in_scopes(family)
        exact = await self.store.fetchone(
            f"SELECT id FROM facts WHERE content = ? COLLATE NOCASE AND {in_scope}", (fact, *scope_params)
        )
        if exact:
            return {"action": "SKIP", "id": int(exact["id"]), "content": fact}
        neighbours = await self.recall(fact, top_k=1, min_similarity=0.0, track=False, scopes=family)
        if neighbours and self._duplicate(fact, neighbours[0]) and not self._corrects(fact, neighbours[0]):
            return {"action": "SKIP", "id": neighbours[0]["id"], "content": neighbours[0]["content"]}
        topics, memory_type, duration = await self.analyze(fact) if use_llm else ([DEFAULT_TOPIC], "long-term", None)
        fid = await self._insert(
            fact, [], source=source, topics=topics, memory_type=memory_type, duration=duration, review=review,
            scope=scope,
        )
        held = {"action": "ADD", "id": fid, "content": fact, "status": "pending"}
        return {**held, "reason": review["reason"]} if review.get("reason") else held

    async def _remember(
        self, fact: str, *, source: str, use_llm: bool, scope: str = memory_scopes.PERSONAL
    ) -> dict[str, Any]:
        fact = fact.strip()
        if not fact:
            return {"action": "SKIP", "id": None, "content": fact}
        family = memory_scopes.recall_scopes(scope)
        in_scope, scope_params = _in_scopes(family)
        exact = await self.store.fetchone(
            f"SELECT id FROM facts WHERE content = ? COLLATE NOCASE AND {ACTIVE} AND {in_scope}", (fact, *scope_params)
        )
        if exact:
            return {"action": "SKIP", "id": int(exact["id"]), "content": fact}
        [vec] = await self._embed([fact])
        neighbours = await self.recall(fact, top_k=5, min_similarity=0.0, track=False, scopes=family)
        # "Mira starts school in July" embeds almost exactly like "... in June": a correction is never a duplicate
        dup = next((n for n in neighbours if self._duplicate(fact, n) and not self._corrects(fact, n)), None)
        if dup is not None:
            return {"action": "SKIP", "id": dup["id"], "content": dup["content"]}

        action, fid, content = "ADD", None, fact
        topics: list[str] | None = None
        memory_type, duration = "long-term", None
        user = self.config.assistant.user_name
        close = [n for n in neighbours if n["similarity"] >= 0.5]
        related: list[dict] = []
        # The model may only update, skip or delete a fact about the same person or thing (#339): embedding models
        # score "प्रिया पुणे में रहती है" close to "Riya lives in Pune", and a small model then calls it an update.
        # Facts about someone else are left out of what it sees, and are still checked by ``covered_by`` below.
        others = {n["id"] for n in close if not same_subject(fact, n["content"], user)}
        if use_llm:
            try:
                related = await self._related_facts(fact, {n["id"] for n in close}, family)
            except Exception as exc:
                log.debug("related-fact lookup skipped: %s", exc)
            others |= {r["id"] for r in related if not same_subject(fact, r["content"], user)}
            if others:
                log.info("CUD for %r: facts %s are about someone else, not offered", fact, sorted(others))
        fixed = next(
            (
                (n, c) for n in [*close, *related]
                if n["id"] not in others and (c := correction_of(fact, n["content"], user))
            ),
            None,
        )
        if fixed is not None:
            # the user changed a date, place, number or name of something already saved (#364): small models
            # answer SKIP or ADD here, so this is decided without them
            target, corrected = fixed
            log.info("CUD for %r: corrects fact %s, updated without asking the model", fact, target["id"])
            analysis = None if use_llm else ([DEFAULT_TOPIC], "long-term", None)
            return await self._store_update(int(target["id"]), corrected, analysis, vec if corrected == fact else None)
        if use_llm:
            similar = [
                {"id": n["id"], "content": n["content"], "similarity": round(n["similarity"], 2)}
                | ({"same_subject_attribute": True} if conflict_strength(fact, n["content"], user) == 2 else {})
                | ({"changes_value": True} if changes_value(fact, n["content"], user) else {})
                for n in close if n["id"] not in others
            ] + [
                {"id": r["id"], "content": r["content"], "same_subject_attribute": True}
                | ({"changes_value": True} if changes_value(fact, r["content"], user) else {})
                for r in related if r["id"] not in others
            ]
            try:
                raw = await self.llm.complete_json(
                    "fast",
                    [
                        {"role": "system", "content": prompts.CUD_DECISION_SYSTEM},
                        {
                            "role": "user",
                            "content": prompts.CUD_DECISION_USER.format(
                                information=fact, similar_facts=json.dumps(similar) if similar else "[] (none)"
                            ),
                        },
                    ],
                )
                if isinstance(raw, dict) and str(raw.get("action", "")).upper() in {"ADD", "UPDATE", "DELETE", "SKIP"}:
                    action = str(raw["action"]).upper()
                    try:
                        fid = int(raw["fact_id"]) if raw.get("fact_id") is not None else None
                    except (TypeError, ValueError):
                        fid = None
                    content = str(raw.get("content") or fact).strip()
                    topics, memory_type, duration = _analysis_from(raw)
            except Exception as exc:
                log.warning("CUD decision failed, defaulting to ADD: %s", exc)

        candidates = [*close, *related]
        valid_ids = {n["id"] for n in candidates} - others
        if action in {"UPDATE", "DELETE", "SKIP"} and fid not in valid_ids:
            action, content = "ADD", fact  # the model pointed at something we did not show it
        if action == "SKIP":
            matched = next((n["content"] for n in candidates if n["id"] == fid), "")
            extra = new_details(fact, matched) | (script_words(fact) - script_words(matched))
            if extra:
                # the "duplicate" is missing a name, place or number the new fact carries (in Hindi and other
                # scripts without capitals, any word of it): keep it
                log.info("CUD SKIP overruled for %r (new details: %s)", fact, sorted(extra))
                action, content, topics = "ADD", fact, None
            else:
                return {"action": "SKIP", "id": fid, "content": content}
        if action == "UPDATE":
            matched = next((n["content"] for n in candidates if n["id"] == fid), "")
            dropped = new_details(matched, content or "") if matched else set()
            if matched and not (
                same_thing(fact, matched, user) and same_thing(content, matched, user)
                and not says_something_else(fact, matched, user)
            ):
                # a different person or action ("files written" vs "files deleted"), in the new fact or in the
                # model's rewrite that would be stored, or something unrelated about the same person ("a piano
                # recital in July" vs "starts school in July"): both facts hold
                log.info("CUD UPDATE of %s kept as ADD for %r (about something else)", fid, fact)
                action, content, topics = "ADD", fact, None
            elif dropped and not (
                new_details(fact, matched) or (script_words(fact) and new_details(content, matched))
            ):
                # the rewrite would lose a name, place or number that the new fact does not replace
                # ("moving to Berlin" rewritten as "pursuing her masters"): keep both facts instead. A Hindi fact's
                # places can't be read in Latin letters, so there the rewrite has to bring one in their place
                log.info("CUD UPDATE of %s kept as ADD for %r (would drop %s)", fid, fact, sorted(dropped))
                action, content, topics = "ADD", fact, None
        if action == "ADD":
            cover = next((n for n in candidates if covered_by(fact, n["content"], user)), None)
            if cover is not None:  # "sister lives in a city" next to "sister Meera lives in Lisbon"
                log.info("CUD ADD skipped for %r: fact %s already says it", fact, cover["id"])
                return {"action": "SKIP", "id": cover["id"], "content": cover["content"]}
        if action == "ADD" and use_llm:
            peer = self._residence_peer(fact, candidates)
            if peer is not None:  # one current home: "moved to X" replaces "lives in Y"
                log.info("CUD ADD turned into UPDATE of residence fact %s for %r", peer["id"], fact)
                action, fid = "UPDATE", peer["id"]
        if action == "DELETE":
            await self.forget(int(fid))  # type: ignore[arg-type]
            return {"action": "DELETE", "id": int(fid), "content": content}  # type: ignore[arg-type]
        if topics is None:
            topics, memory_type, duration = await self.analyze(content) if use_llm else ([DEFAULT_TOPIC], "long-term", None)
        if action == "UPDATE":
            analysis = (topics, memory_type, duration)
            return await self._store_update(int(fid), content, analysis, vec if content == fact else None)  # type: ignore[arg-type]
        if content != fact:
            [vec] = await self._embed([content])
        new_id = await self._insert(
            content, vec, source=source, topics=topics, memory_type=memory_type, duration=duration, scope=scope
        )
        return {"action": "ADD", "id": new_id, "content": content}

    async def _store_update(
        self,
        fid: int,
        content: str,
        analysis: tuple[list[str], str, str | None] | None,
        vec: list[float] | None = None,
    ) -> dict[str, Any]:
        """Replace fact ``fid`` with ``content`` (analysed now when ``analysis`` is None, embedded unless ``vec`` is
        given). The result carries ``previous``, the words it replaced, so the save can be undone."""
        topics, memory_type, duration = analysis if analysis is not None else await self.analyze(content)
        if vec is None:
            [vec] = await self._embed([content])
        before = await self.store.fetchone("SELECT content FROM facts WHERE id = ?", (fid,))
        await self._update(fid, content, vec, topics, memory_type, duration)
        previous = str(before["content"]) if before is not None else None
        return {"action": "UPDATE", "id": fid, "content": content, "previous": previous}

    def _corrects(self, fact: str, neighbour: dict) -> bool:
        return correction_of(fact, neighbour["content"], self.config.assistant.user_name) is not None

    async def update_content(self, fid: int, content: str, *, use_llm: bool = True) -> dict:
        """v2 update_memory: new content, re-analyzed topics/expiry, re-embedded; id kept."""
        content = content.strip()
        row = await self.store.fetchone("SELECT id FROM facts WHERE id = ?", (fid,))
        if row is None:
            raise KeyError(fid)
        if not content:
            raise ValueError("content is empty")
        topics, memory_type, duration = (
            await self.analyze(content) if use_llm else ([DEFAULT_TOPIC], "long-term", None)
        )
        [vec] = await self._embed([content])
        await self._update(fid, content, vec, topics, memory_type, duration)
        self.publish("UPDATE", fid, content)
        fact = await self.get_fact(fid)
        assert fact is not None
        return fact

    # ------------------------------------------------------------------ extraction
    @staticmethod
    def clean_fact(fact: str, username: str) -> str | None:
        """Post-filter model output: personalize leftovers, drop requests, clock/weather noise and "X is unknown"."""
        text = " ".join(str(fact).split()).strip(" -*•")
        if len(text) < 6:
            return None
        name = username.strip()
        if name:
            text = re.sub(r"\b(the user|USERNAME)'s\b", f"{name}'s", text, flags=re.IGNORECASE)
            text = _USER_REF_RE.sub(name, text)
        text = text[:1].upper() + text[1:]
        if _REQUEST_RE.search(text) or _NOISE_RE.search(text) or _UNKNOWN_RE.search(text) or _TAUTOLOGY_RE.match(text):
            return None
        return text

    def _facts_from(self, raw: Any, username: str) -> list[str]:
        items: list[Any] = []
        if isinstance(raw, dict):
            items = raw.get("facts") or []
        elif isinstance(raw, list):
            items = raw
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for f in items:
            if isinstance(f, dict):  # {"fact": "..."} drift
                f = f.get("fact") or f.get("content") or ""
            cleaned = self.clean_fact(str(f), username) if str(f).strip() else None
            if cleaned and cleaned not in out:
                out.append(cleaned)
        return out

    async def extract_facts(self, text: str, username: str) -> list[str]:
        name = username.strip() or "the user"
        raw = await self.llm.complete_json(
            "fast",
            [
                {"role": "system", "content": prompts.extraction_system(username)},
                {"role": "user", "content": prompts.FACT_EXTRACTION_USER.format(username=name, text=text)},
            ],
        )
        return self._facts_from(raw, username)

    @staticmethod
    def flush_transcript(transcript: str, max_chars: int) -> str:
        """Keep user and assistant turns (tool and system output dropped), bounded to ``max_chars``.
        Assistant turns are shortened first because the user's own words carry the facts."""
        blocks: list[list[str]] = []
        for line in (transcript or "").splitlines():
            m = _ROLE_LINE_RE.match(line)
            if m or not blocks:
                blocks.append([(m.group(1).lower() if m else "user"), line])
            else:
                blocks[-1].append(line)
        kept = [(b[0], "\n".join(b[1:]).strip()[:1200]) for b in blocks if b[0] in {"user", "assistant"}]
        kept = [(role, text) for role, text in kept if text]

        def size(items: list[tuple[str, str]]) -> int:
            return sum(len(t) + 1 for _, t in items)

        if size(kept) > max_chars:
            kept = [(r, t if r == "user" else t[:200]) for r, t in kept]
        text = "\n".join(t for _, t in kept)
        return text[-max_chars:] if len(text) > max_chars else text

    async def flush_conversation(
        self,
        transcript: str,
        user_name: str,
        *,
        review: dict | None = None,
        session_id: str | None = None,
        scope: str | None = None,
    ) -> list[dict]:
        """Save lasting facts from chat turns that are about to be compressed (docs/API.md section 10).

        One extraction call over the bounded transcript, then the normal CUD path per fact
        (exact and near duplicates short-circuit without a model call). Publishes ``memory.updated``.
        With ``review`` (a chat that read outside content) the facts are held for review.
        """
        cfg = self.config.memory
        if not cfg.flush_enabled or cfg.flush_max_facts <= 0:
            return []
        text = self.flush_transcript(transcript, cfg.flush_max_chars)
        if len(text.split()) < 4:
            return []
        name = user_name.strip() or "the user"
        try:
            raw = await self.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.extraction_system(user_name)},
                    {"role": "user", "content": prompts.FLUSH_USER.format(username=name, transcript=text)},
                ],
            )
        except Exception as exc:
            log.warning("memory flush extraction failed: %s", exc)
            return []
        results: list[dict] = []
        for fact in self._facts_from(raw, user_name)[: cfg.flush_max_facts]:
            try:
                results.append(await self.remember(
                    fact, source="conversation", notify=True, review=reviews.with_snippet(review, text, fact),
                    session_id=session_id, scope=scope,
                ))
            except Exception as exc:
                log.warning("memory flush could not store %r: %s", fact, exc)
        return results

    async def extract_and_store(
        self, text: str, username: str, source: str = "conversation", *, notify: bool = False,
        review: dict | None = None, own: bool = False, session_id: str | None = None, scope: str | None = None,
    ) -> list[dict]:
        """Pull atomic facts out of text and remember each (v2 cud_memory). With ``review`` they are held for review,
        each with the sentence of ``text`` it most likely came from. ``own``: ``text`` is the user's own words (a chat
        turn), so a fact they fully back is saved even with ``review`` (#234, ``review.grounded``)."""
        try:
            facts = await self.extract_facts(text, username)
        except Exception as exc:
            log.warning("fact extraction failed: %s", exc)
            return []
        results = []
        for f in facts[:20]:
            try:
                results.append(await self.remember(
                    f, source=source, notify=notify, review=reviews.with_snippet(review, text, f),
                    own_words=text if own else None, session_id=session_id, scope=scope,
                ))
            except Exception as exc:
                log.warning("failed to store fact %r: %s", f, exc)
        return results

    async def import_document(
        self, path: Path, *, username: str, source: str | None = None, scope: str | None = None
    ) -> dict:
        """v2 build_initial_memory for one file, without wiping existing memories. The facts are held for the
        user's review (``pending`` counts them), so nothing already remembered changes."""
        from sentient.files.extract import extract_text

        source = source or f"file:{path.name}"
        text = await asyncio.to_thread(extract_text, path, 200_000)
        text = re.sub(r"\[page \d+\]\n", "", text)
        counts = {"added": 0, "updated": 0, "skipped": 0, "pending": 0}
        review = reviews.note(path.name)
        for chunk in chunk_text(text, self.config.memory.import_chunk_chars):
            for r in await self.extract_and_store(chunk, username, source=source, review=review, scope=scope):
                key = {"ADD": "added", "UPDATE": "updated", "DELETE": "updated"}.get(r["action"], "skipped")
                counts[key] += 1
                counts["pending"] += r.get("status") == "pending"
        if counts["added"] or counts["updated"]:
            self.publish(
                "ADD", None, None, source=source, count=counts["added"] + counts["updated"], status="pending"
            )
        return {**counts, "source": source}

    # ------------------------------------------------------------------ reads
    async def get_fact(self, fid: int) -> dict | None:
        r = await self.store.fetchone(f"SELECT {MEMORY_COLUMNS} FROM facts WHERE id = ?", (fid,))
        return _row_to_memory(r) if r else None

    async def list_facts(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        topic: str | None = None,
        source: str | None = None,
        q: str | None = None,
        scope: str | None = None,
    ) -> list[dict]:
        """Active facts, newest first. ``scope`` keeps one scope only (the Memory screen's filter, #155)."""
        if topic:
            topic = normalize_topics([topic])[0]
        one = memory_scopes.normalize(scope)
        if q and q.strip():
            try:
                hits = await self._search(
                    q, min(offset + limit, 500) * (3 if topic else 1), self.config.memory.min_similarity, source,
                    [one] if one else None,
                )
                if topic:
                    hits = [h for h in hits if topic in h["topics"]]
                return hits[offset : offset + limit]
            except Exception as exc:
                log.debug("semantic list search unavailable, using keyword match: %s", exc)
        where = [ACTIVE, "(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [now_iso()]
        if source:
            where.append("source = ?")
            params.append(source)
        if one:
            where.append("scope = ?")
            params.append(one)
        if topic:
            where.append("EXISTS (SELECT 1 FROM json_each(facts.topics) WHERE value = ?)")
            params.append(topic)
        if q and q.strip():
            where.append("content LIKE ?")
            params.append(f"%{q.strip()}%")
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE {' AND '.join(where)}"
            " ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_row_to_memory(r) for r in rows]

    async def topic_counts(self) -> list[dict]:
        rows = await self.store.fetchall(
            f"SELECT topics FROM facts WHERE {ACTIVE} AND (expires_at IS NULL OR expires_at > ?)", (now_iso(),)
        )
        counts = {t["name"]: 0 for t in TOPICS}
        for r in rows:
            for t in normalize_topics(json.loads(r["topics"] or "[]")):
                counts[t] = counts.get(t, 0) + 1
        return [{"name": t["name"], "description": t["description"], "count": counts[t["name"]]} for t in TOPICS]

    async def expiring_soon(self, within: timedelta = timedelta(days=1)) -> list[dict]:
        now = datetime.now(UTC)
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE {ACTIVE} AND expires_at IS NOT NULL AND expires_at > ?"
            " AND expires_at <= ? ORDER BY expires_at",
            (now.isoformat(), (now + within).isoformat()),
        )
        return [_row_to_memory(r) for r in rows]

    async def graph(self, max_nodes: int = 1500) -> dict:
        """v2 create_memory_graph: nodes plus links where cosine similarity >= threshold."""
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE {ACTIVE} AND (expires_at IS NULL OR expires_at > ?)"
            " ORDER BY created_at DESC, id DESC LIMIT ?",
            (now_iso(), max_nodes),
        )
        nodes = []
        for r in rows:
            m = _row_to_memory(r)
            label = m["content"] if len(m["content"]) <= 25 else m["content"][:25].strip() + "..."
            nodes.append(
                {
                    "id": m["id"], "label": label, "title": m["content"], "content": m["content"],
                    "topics": m["topics"], "memory_type": m["memory_type"], "source": m["source"],
                    "created_at": m["created_at"],
                }
            )
        links: list[dict] = []
        vectors: dict[int, list[float]] = {}
        if nodes and (self.vec.ready or await self.vec.exists()):
            try:
                vectors = await self.vec.all_vectors()
            except Exception as exc:
                log.warning("graph vectors unavailable: %s", exc)
        ids = [n["id"] for n in nodes if n["id"] in vectors]
        threshold = self.config.memory.graph_link_similarity
        if len(ids) > 1:
            try:
                import numpy as np

                mat = np.array([vectors[i] for i in ids], dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                mat = mat / norms
                sim = mat @ mat.T
                ii, jj = np.where(np.triu(sim, k=1) >= threshold)
                for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
                    links.append({"source": ids[i], "target": ids[j], "value": round(float(sim[i, j]), 4)})
            except ImportError:  # numpy is optional
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        s = cosine(vectors[ids[a]], vectors[ids[b]])
                        if s >= threshold:
                            links.append({"source": ids[a], "target": ids[b], "value": round(s, 4)})
        return {"nodes": nodes, "links": links}

    # ------------------------------------------------------------------ housekeeping
    async def purge_expired(self) -> int:
        rows = await self.store.fetchall(
            "SELECT id FROM facts WHERE expires_at IS NOT NULL AND expires_at < ?", (now_iso(),)
        )
        for r in rows:
            await self.forget(int(r["id"]))
        if rows:
            self.publish("DELETE", None, None, reason="expired", count=len(rows))
        return len(rows)

    async def count(self) -> int:
        row = await self.store.fetchone(f"SELECT COUNT(*) AS n FROM facts WHERE {ACTIVE}")
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ review (ADR 0021)
    async def pending_facts(self, limit: int = 500) -> list[dict]:
        """Facts held for the user's review, newest first (``limit=-1``: all of them)."""
        await self.ensure_schema()
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE status = 'pending' ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_memory(r) for r in rows]

    async def pending_count(self) -> int:
        row = await self.store.fetchone("SELECT COUNT(*) AS n FROM facts WHERE status = 'pending'")
        return int(row["n"]) if row else 0

    async def approve(self, fid: int, content: str | None = None) -> dict | None:
        """The user approved a held fact, maybe in their own words: it becomes active and recall can find it.
        None when ``fid`` is not pending. A fact Sentient already remembers in the same words is not added twice."""
        row = await self.store.fetchone("SELECT content FROM facts WHERE id = ? AND status = 'pending'", (fid,))
        if row is None:
            return None
        text = " ".join(str(content or "").split()) or row["content"]
        # it is approved into the scope it was held in (#155): only a memory there or in always counts as the same
        in_scope, scope_params = _in_scopes(memory_scopes.recall_scopes(await self.scope_of(fid)))
        same = await self.store.fetchone(
            f"SELECT id FROM facts WHERE content = ? COLLATE NOCASE AND {ACTIVE} AND {in_scope}", (text, *scope_params)
        )
        if same is not None:
            await self.link_sources(int(same["id"]), await self.sources_of(fid))  # #153
            await self.forget(fid)
            self.publish("DELETE", fid, row["content"], reason="approved", merged_into=int(same["id"]))
            return await self.get_fact(int(same["id"]))
        [vec] = await self._embed([text])
        if text != row["content"]:
            topics, memory_type, duration = await self.analyze(text)
            await self.store.execute(
                "UPDATE facts SET content = ?, previous_content = ?, topics = ?, memory_type = ?, expires_at = ?"
                " WHERE id = ?",
                (text, row["content"], json.dumps(topics), memory_type, expires_for(memory_type, duration), fid),
            )
        await self.store.execute(
            "UPDATE facts SET status = 'active', updated_at = ?, embedding_model = ? WHERE id = ?",
            (now_iso(), self.llm.model_for("embedding"), fid),
        )
        await self.vec.upsert(fid, vec)
        self.publish("ADD", fid, text, reason="approved")
        return await self.get_fact(fid)

    async def classify_pending(self, fid: int) -> bool:
        """Topics and long/short-term for a held fact that was saved without them (an import does this after its
        model reads are done). False when ``fid`` is not pending."""
        row = await self.store.fetchone("SELECT content FROM facts WHERE id = ? AND status = 'pending'", (fid,))
        if row is None:
            return False
        topics, memory_type, duration = await self.analyze(row["content"])
        await self.store.execute(
            "UPDATE facts SET topics = ?, memory_type = ?, expires_at = ? WHERE id = ? AND status = 'pending'",
            (json.dumps(topics), memory_type, expires_for(memory_type, duration), fid),
        )
        return True

    async def reword_pending(self, fid: int, content: str, review: dict | None = None) -> bool:
        """Put fuller wording on a fact that still waits for review (an import found the same thing with more
        detail). Only a pending fact changes; nothing the user approved is touched. False when ``fid`` isn't pending."""
        content = " ".join(content.split())
        if not content:
            return False
        cur = await self.store.execute(
            "UPDATE facts SET content = ?, previous_content = content, updated_at = ?,"
            " review = COALESCE(?, review) WHERE id = ? AND status = 'pending'",
            (content, now_iso(), reviews.dump(review), fid),
        )
        return bool(getattr(cur, "rowcount", 0))

    async def discard(self, fid: int) -> bool:
        """The user turned a held fact down: it is deleted. False when ``fid`` is not pending."""
        row = await self.store.fetchone("SELECT content FROM facts WHERE id = ? AND status = 'pending'", (fid,))
        if row is None or not await self.forget(fid):
            return False
        self.publish("DELETE", fid, row["content"], reason="discarded")
        return True

    async def expire_pending(self, before: str) -> int:
        """Delete held facts created before ``before`` (ISO time) that nobody reviewed."""
        rows = await self.store.fetchall(
            "SELECT id FROM facts WHERE status = 'pending' AND created_at < ?", (before,)
        )
        for r in rows:
            await self.forget(int(r["id"]))
        if rows:
            self.publish("DELETE", None, None, reason="review_expired", count=len(rows))
        return len(rows)

    # ------------------------------------------------------------------ consolidation (dreaming)
    async def active_facts(self, limit: int = 500) -> list[dict]:
        """Unexpired facts, most recently updated first, with ``recall_count`` and ``vector`` (or None)."""
        await self.ensure_schema()
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS}, recall_count FROM facts WHERE {ACTIVE}"
            " AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC, id DESC LIMIT ?",
            (now_iso(), limit),
        )
        vectors: dict[int, list[float]] = {}
        if rows and (self.vec.ready or await self.vec.exists()):
            try:
                vectors = await self.vec.all_vectors()
            except Exception as exc:
                log.debug("fact vectors unavailable: %s", exc)
        out = []
        for r in rows:
            m = _row_to_memory(r)
            m["recall_count"] = int(r["recall_count"] or 0)
            m["vector"] = vectors.get(m["id"])
            out.append(m)
        return out

    async def merge_facts(self, keep_id: int, content: str, drop_ids: list[int]) -> dict | None:
        """Fold ``drop_ids`` into ``keep_id`` (id kept). Topics are united, long-term wins over
        short-term, recall counts are summed and the old text goes to ``previous_content``."""
        await self.ensure_schema()
        ids = [keep_id, *[i for i in drop_ids if i != keep_id]]
        marks = ",".join("?" * len(ids))
        rows = {
            int(r["id"]): r
            for r in await self.store.fetchall(
                f"SELECT {MEMORY_COLUMNS}, recall_count FROM facts WHERE id IN ({marks})", ids
            )
        }
        keep = rows.get(keep_id)
        if keep is None:
            return None
        content = content.strip() or keep["content"]
        group = [rows[i] for i in ids if i in rows]
        topics = normalize_topics([t for r in group for t in json.loads(r["topics"] or "[]")], limit=4)
        long_term = any(r["memory_type"] == "long-term" for r in group)
        expiries = [r["expires_at"] for r in group if r["expires_at"]]
        expires = None if long_term or not expiries else max(expiries)
        recalls = sum(int(r["recall_count"] or 0) for r in group)
        prev = keep["content"] if keep["content"] != content else keep["previous_content"]
        [vec] = await self._embed([content])
        await self.store.execute(
            "UPDATE facts SET content = ?, topics = ?, memory_type = ?, expires_at = ?, updated_at = ?,"
            " previous_content = ?, recall_count = ?, embedding_model = ? WHERE id = ?",
            (
                content, json.dumps(topics), "long-term" if long_term else "short-term", expires, now_iso(),
                prev, recalls, self.llm.model_for("embedding"), keep_id,
            ),
        )
        await self.vec.upsert(keep_id, vec)
        self.publish("UPDATE", keep_id, content)
        if len(ids) > 1:  # copies in other languages merged into a folded fact now belong to the kept one (#353)
            marks = ",".join("?" * (len(ids) - 1))
            await self.store.execute(
                f"UPDATE facts SET merged_into = ? WHERE status = 'merged' AND merged_into IN ({marks})",
                (keep_id, *ids[1:]),
            )
        for i in ids[1:]:
            if i in rows:
                await self.link_sources(keep_id, await self.sources_of(i))  # it backs the kept one now (#153)
                if await self.forget(i):
                    self.publish("DELETE", i, rows[i]["content"], reason="merged", merged_into=keep_id)
        return await self.get_fact(keep_id)

    async def supersede(self, keep_id: int, drop_id: int) -> bool:
        """Latest wins: delete ``drop_id`` and record its text as ``previous_content`` of ``keep_id``."""
        drop = await self.store.fetchone("SELECT content FROM facts WHERE id = ?", (drop_id,))
        keep = await self.store.fetchone("SELECT id FROM facts WHERE id = ?", (keep_id,))
        if drop is None or keep is None:
            return False
        await self.store.execute(
            "UPDATE facts SET previous_content = ?, updated_at = ? WHERE id = ?", (drop["content"], now_iso(), keep_id)
        )
        await self.forget(drop_id)
        self.publish("DELETE", drop_id, drop["content"], reason="contradicted", superseded_by=keep_id)
        return True

    async def merge_translation(self, keep_id: int, drop_id: int, by: str = "nightly tidying") -> bool:
        """``drop_id`` says what ``keep_id`` says in another language (#353): it leaves every read (status
        ``merged``, ``merged_into`` = ``keep_id``, out of the vector index) but stays in the review history, so
        :meth:`unmerge` can bring it back. The kept memory is not changed."""
        await self.ensure_schema()
        drop = await self.store.fetchone(f"SELECT content FROM facts WHERE id = ? AND {ACTIVE}", (drop_id,))
        keep = await self.store.fetchone(f"SELECT id FROM facts WHERE id = ? AND {ACTIVE}", (keep_id,))
        if drop is None or keep is None or keep_id == drop_id:
            return False
        indexed = self.vec.ready or await self.vec.exists()
        db = self.store.db
        try:  # the row and its vector change together (one commit)
            await db.execute(
                "UPDATE facts SET status = 'merged', merged_into = ?, merged_at = ?, merged_by = ? WHERE id = ?",
                (keep_id, now_iso(), by, drop_id),
            )
            if indexed:
                await db.execute(f"DELETE FROM {self.vec.name} WHERE rowid = ?", (drop_id,))
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self.publish("DELETE", drop_id, drop["content"], reason="merged", merged_into=keep_id)
        return True

    async def unmerge(self, fid: int) -> dict | None:
        """Undo :meth:`merge_translation`: the memory is active again, as it was. None when it isn't merged."""
        await self.ensure_schema()
        row = await self.store.fetchone(
            "SELECT content, merged_into FROM facts WHERE id = ? AND status = 'merged'"
            " AND (expires_at IS NULL OR expires_at > ?)",  # a short-term memory that ran out can't come back
            (fid, now_iso()),
        )
        if row is None:
            return None
        [vec] = await self._embed([row["content"]])  # also makes sure the vector table exists
        db = self.store.db
        try:  # the row and its vector change together (one commit)
            await db.execute(
                "UPDATE facts SET status = 'active', merged_into = NULL, merged_at = NULL, merged_by = NULL,"
                " embedding_model = ? WHERE id = ?",
                (self.llm.model_for("embedding"), fid),
            )
            await db.execute(f"DELETE FROM {self.vec.name} WHERE rowid = ?", (fid,))
            await db.execute(f"INSERT INTO {self.vec.name}(rowid, embedding) VALUES(?, ?)", (fid, pack(vec)))
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        if row["merged_into"] is not None:  # the user said these two differ: never merge them again
            apart = await self.kept_apart()
            apart.add(tuple(sorted((fid, int(row["merged_into"])))))
            await self.store.set_meta(KEPT_APART_META, json.dumps(sorted(apart)[-1000:]))
        self.publish("ADD", fid, row["content"], reason="unmerged")
        return await self.get_fact(fid)

    async def merged_facts(self, limit: int = 100) -> list[dict]:
        """Memories merged into another one, newest first, each with the memory it was merged into."""
        await self.ensure_schema()
        rows = await self.store.fetchall(
            "SELECT m.id, m.content, m.source, m.merged_into, m.merged_at, m.merged_by, k.content AS kept"
            " FROM facts m LEFT JOIN facts k ON k.id = m.merged_into WHERE m.status = 'merged'"
            " ORDER BY m.merged_at DESC, m.id DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": int(r["id"]), "text": r["content"], "source": r["source"], "merged_at": r["merged_at"],
             "by": r["merged_by"] or "nightly tidying",
             "merged_into": {"id": int(r["merged_into"]), "text": r["kept"]} if r["kept"] is not None else None}
            for r in rows
        ]

    async def kept_apart(self) -> set[tuple[int, int]]:
        """Pairs of memories the user split again after a merge (:meth:`unmerge`), smaller id first."""
        try:
            raw = json.loads(await self.store.get_meta(KEPT_APART_META) or "[]")
            return {(int(a), int(b)) for a, b in raw}
        except (TypeError, ValueError):
            return set()

    async def promote_recalled(self, min_recalls: int, limit: int = 50) -> list[dict]:
        """Short-term facts recalled at least ``min_recalls`` times become long-term (no expiry)."""
        await self.ensure_schema()
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE {ACTIVE} AND memory_type = 'short-term' AND recall_count >= ?"
            " AND (expires_at IS NULL OR expires_at > ?) ORDER BY recall_count DESC LIMIT ?",
            (min_recalls, now_iso(), limit),
        )
        out = []
        for r in rows:
            await self.store.execute(
                "UPDATE facts SET memory_type = 'long-term', expires_at = NULL, updated_at = ? WHERE id = ?",
                (now_iso(), int(r["id"])),
            )
            self.publish("UPDATE", int(r["id"]), r["content"], reason="promoted")
            out.append(_row_to_memory(r))
        return out
