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
from sentient.memory import prompts
from sentient.memory import review as reviews
from sentient.memory.episodic import EpisodicMemory
from sentient.memory.schema import ensure_memory_schema
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
    "id, content, source, topics, memory_type, created_at, updated_at, expires_at, previous_content, status, review"
)
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


def _subject_keys(subjects: set[str], users: set[str]) -> set[tuple[str, str]]:
    """(owner, relation) per subject, with every name for the user as "user": "maya's sister" -> ("user", "sister")."""
    out = set()
    for s in subjects:
        owner, _, rel = s.partition("'s ")
        out.add(("user" if owner in users else owner, rel))
    return out


def same_subject(a: str, b: str, user_name: str = "") -> bool:
    """False only when both facts clearly name different people or things ("Maya's sister" vs "Maya's brother").
    Unclear subjects count as the same, and so does "the user" next to a name while the user's name isn't set."""
    sa, sb = fact_subjects(a, user_name), fact_subjects(b, user_name)
    if not sa or not sb or sa & sb:
        return True
    users = {"user", "the user", (user_name or "").strip().lower()} - {""}
    unnamed = not (user_name or "").strip()
    ka, kb = _subject_keys(sa, users), _subject_keys(sb, users)
    for oa, ra in ka:
        for ob, rb in kb:
            if ra == rb and (oa == ob or (unnamed and "user" in (oa, ob))):
                return True
    return False


def same_thing(new_fact: str, existing: str, user_name: str = "") -> bool:
    """True when ``new_fact`` may replace ``existing``: the same subject and, where both name one, the same action.
    "does not want their files deleted" and "does not want files to be written" are two preferences, and so are
    "does not want her files deleted" and "does not want her emails deleted"."""
    if not same_subject(new_fact, existing, user_name):
        return False
    acts_new, acts_old = fact_actions(new_fact), fact_actions(existing)
    if not acts_new or not acts_old:
        return True
    if not acts_new & acts_old:
        return False
    if NEGATION_RE.search(new_fact) and NEGATION_RE.search(existing):
        # two "does not want X deleted" rules about different things ("files", "emails") both hold
        shared = acts_new & acts_old
        a, b = _object_words(new_fact, shared), _object_words(existing, shared)
        return not (a - b and b - a)
    return True


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
    new name, place or number, and every other word is in it (or a placeholder like "a city"). "Maya has a sister"
    is covered by any fact about Maya's sister."""
    if not new_fact.strip() or not existing.strip():
        return False
    if bool(NEGATION_RE.search(new_fact)) != bool(NEGATION_RE.search(existing)):
        return False
    if bool(PAST_RE.search(new_fact)) != bool(PAST_RE.search(existing)):
        return False
    if new_details(new_fact, existing):
        return False
    if not same_subject(new_fact, existing, user_name):
        return _only_names_relation(new_fact, existing, user_name)
    have = _plain_words(existing)
    user = (user_name or "").strip().lower()
    have.update(w for w in (user, "user") if w)
    left = {w for w in _plain_words(new_fact) - have if w not in _PLACEHOLDERS}
    return not left


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
    }
    if similarity is not None:
        d["similarity"] = round(similarity, 4)
    return d


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
            await self.vec.ensure(len(vecs[0]), self.reindex)
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
    async def _keyword_ids(self, tokens: list[str], limit: int, source: str | None) -> list[int]:
        if not tokens:
            return []
        sql = (
            "SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
            " WHERE facts_fts MATCH ? AND f.status = 'active'"
        )
        params: list[Any] = [" OR ".join(f'"{t}"' for t in tokens)]
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
        self, query: str, k: int, min_sim: float, source: str | None = None
    ) -> list[dict]:
        """Hybrid recall: vector neighbours plus FTS5 keyword matches, ranked by
        ``similarity + keyword_weight * share of query words present``. Embedding models
        compress similarities (unrelated text still scores ~0.65), so exact words help ordering."""
        await self.ensure_schema()
        qvec = await self.embed_query(query)
        now = now_iso()
        pool = max(k * 3, 10)
        if source is None:
            sims = dict(await self.vec.knn(qvec, pool))
        else:
            rows = await self.store.fetchall(
                "SELECT f.id, vec_distance_cosine(v.embedding, ?) AS distance"
                " FROM facts f JOIN facts_vec v ON v.rowid = f.id WHERE f.source = ? AND f.status = 'active'"
                " ORDER BY distance LIMIT ?",
                (pack(qvec), source, pool),
            )
            sims = {int(r["id"]): 1.0 - float(r["distance"]) for r in rows}
        weight = self.config.memory.keyword_weight
        tokens = keyword_tokens(query)
        kw_ids = await self._keyword_ids(tokens, pool, source) if weight > 0 else []
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
        rows = await self.store.fetchall(
            f"SELECT {MEMORY_COLUMNS} FROM facts WHERE id IN ({marks}) AND {ACTIVE}", list(sims)
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
    ) -> list[dict]:
        """Relevant facts for ``query``. ``track`` counts the recall (dreaming promotes
        short-term facts that keep being recalled); internal lookups pass False."""
        top_k = top_k if top_k is not None else self.config.memory.facts_top_k
        min_sim = min_similarity if min_similarity is not None else self.config.memory.min_similarity
        if top_k <= 0 or not query.strip():
            return []
        try:
            out = await self._search(query, top_k, min_sim, source)
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

    async def search_by_source(self, query: str, source: str, top_k: int = 5) -> list[dict]:
        return await self.recall(query, top_k=top_k, min_similarity=0.0, source=source, track=False)

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
    ) -> int:
        """A held fact (``review`` set) is saved as pending without a vector, so recall cannot find it."""
        await self.ensure_schema()
        ts = now_iso()
        cur = await self.store.execute(
            "INSERT INTO facts(content, source, topics, memory_type, created_at, updated_at, expires_at, embedding_model,"
            " status, review) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                content, source, json.dumps(topics), memory_type, ts, ts,
                expires_for(memory_type, duration), self.llm.model_for("embedding"),
                "pending" if review is not None else "active", reviews.dump(review),
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
        row = await self.store.fetchone("SELECT content FROM facts WHERE id = ?", (fid,))
        cur = await self.store.execute("DELETE FROM facts WHERE id = ?", (fid,))
        await self.vec.delete(fid)
        deleted = cur.rowcount > 0
        if deleted and notify:
            self.publish("DELETE", fid, row["content"] if row else None)
        return deleted

    async def forget_source(self, source: str) -> int:
        """v2 delete_memory_by_source."""
        rows = await self.store.fetchall("SELECT id FROM facts WHERE source = ?", (source,))
        for r in rows:
            await self.forget(int(r["id"]))
        if rows:
            self.publish("DELETE", None, None, source=source, count=len(rows))
        return len(rows)

    async def _related_facts(self, fact: str, exclude: set[int]) -> list[dict]:
        """Up to 3 facts about the same subject and attribute family as ``fact``, found by keyword rather
        than embedding ("moved to Bengaluru last month" scores low against "lives in Pune")."""
        user = self.config.assistant.user_name
        prof = profile_fact(fact, user)
        if not prof.subjects or not prof.families:
            return []
        terms = [t for s in sorted(prof.subjects) for t in keyword_tokens(s)]
        for fam in sorted(prof.families):
            terms.extend(FAMILY_TERMS.get(fam, []))
        ids = [i for i in await self._keyword_ids(list(dict.fromkeys(terms))[:24], 60, None) if i not in exclude]
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
    ) -> dict[str, Any]:
        """Store one fact with dedup / CUD logic. Returns {action, id, content}.

        With ``review`` (``memory.review.note``) the fact is held for the user's review instead: it is only ever
        added (never updates or deletes another fact), and the result carries ``status: "pending"``."""
        if review is not None:
            result = await self._hold(fact, source=source, use_llm=use_llm, review=review)
        else:
            result = await self._remember(fact, source=source, use_llm=use_llm)
        if notify and result["action"] in {"ADD", "UPDATE", "DELETE"}:
            extra = {"status": "pending"} if result.get("status") == "pending" else {}
            self.publish(result["action"], result["id"], result["content"], **extra)
        return result

    async def _hold(self, fact: str, *, source: str, use_llm: bool, review: dict) -> dict[str, Any]:
        """Save ``fact`` as pending. Duplicates of any fact (held or not) are skipped; nothing else changes."""
        fact = fact.strip()
        if not fact:
            return {"action": "SKIP", "id": None, "content": fact}
        await self.ensure_schema()
        exact = await self.store.fetchone("SELECT id FROM facts WHERE content = ? COLLATE NOCASE", (fact,))
        if exact:
            return {"action": "SKIP", "id": int(exact["id"]), "content": fact}
        neighbours = await self.recall(fact, top_k=1, min_similarity=0.0, track=False)
        if neighbours and neighbours[0]["similarity"] >= self.config.memory.duplicate_similarity:
            return {"action": "SKIP", "id": neighbours[0]["id"], "content": neighbours[0]["content"]}
        topics, memory_type, duration = await self.analyze(fact) if use_llm else ([DEFAULT_TOPIC], "long-term", None)
        fid = await self._insert(
            fact, [], source=source, topics=topics, memory_type=memory_type, duration=duration, review=review
        )
        return {"action": "ADD", "id": fid, "content": fact, "status": "pending"}

    async def _remember(self, fact: str, *, source: str, use_llm: bool) -> dict[str, Any]:
        fact = fact.strip()
        if not fact:
            return {"action": "SKIP", "id": None, "content": fact}
        exact = await self.store.fetchone(
            f"SELECT id FROM facts WHERE content = ? COLLATE NOCASE AND {ACTIVE}", (fact,)
        )
        if exact:
            return {"action": "SKIP", "id": int(exact["id"]), "content": fact}
        [vec] = await self._embed([fact])
        neighbours = await self.recall(fact, top_k=5, min_similarity=0.0, track=False)
        if neighbours and neighbours[0]["similarity"] >= self.config.memory.duplicate_similarity:
            return {"action": "SKIP", "id": neighbours[0]["id"], "content": neighbours[0]["content"]}

        action, fid, content = "ADD", None, fact
        topics: list[str] | None = None
        memory_type, duration = "long-term", None
        close = [n for n in neighbours if n["similarity"] >= 0.5]
        related: list[dict] = []
        if use_llm:
            user = self.config.assistant.user_name
            try:
                related = await self._related_facts(fact, {n["id"] for n in close})
            except Exception as exc:
                log.debug("related-fact lookup skipped: %s", exc)
            similar = [
                {"id": n["id"], "content": n["content"], "similarity": round(n["similarity"], 2)}
                | ({"same_subject_attribute": True} if conflict_strength(fact, n["content"], user) == 2 else {})
                for n in close
            ] + [{"id": r["id"], "content": r["content"], "same_subject_attribute": True} for r in related]
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
        valid_ids = {n["id"] for n in candidates}
        if action in {"UPDATE", "DELETE", "SKIP"} and fid not in valid_ids:
            action, content = "ADD", fact  # the model pointed at something we did not show it
        if action == "SKIP":
            matched = next((n["content"] for n in candidates if n["id"] == fid), "")
            if new_details(fact, matched):
                # the "duplicate" is missing a name, place or number the new fact carries: keep it
                log.info("CUD SKIP overruled for %r (new details: %s)", fact, sorted(new_details(fact, matched)))
                action, content, topics = "ADD", fact, None
            else:
                return {"action": "SKIP", "id": fid, "content": content}
        user = self.config.assistant.user_name
        if action == "UPDATE":
            matched = next((n["content"] for n in candidates if n["id"] == fid), "")
            dropped = new_details(matched, content or "") if matched else set()
            if matched and not same_thing(fact, matched, user):
                # a different person or action ("files written" vs "files deleted"): both facts hold
                log.info("CUD UPDATE of %s kept as ADD for %r (about something else)", fid, fact)
                action, content, topics = "ADD", fact, None
            elif dropped and not new_details(fact, matched):
                # the rewrite would lose a name, place or number that the new fact does not replace
                # ("moving to Berlin" rewritten as "pursuing her masters"): keep both facts instead
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
        if content != fact:
            [vec] = await self._embed([content])
        if action == "UPDATE":
            await self._update(int(fid), content, vec, topics, memory_type, duration)  # type: ignore[arg-type]
            return {"action": "UPDATE", "id": int(fid), "content": content}  # type: ignore[arg-type]
        new_id = await self._insert(
            content, vec, source=source, topics=topics, memory_type=memory_type, duration=duration
        )
        return {"action": "ADD", "id": new_id, "content": content}

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

    async def flush_conversation(self, transcript: str, user_name: str, *, review: dict | None = None) -> list[dict]:
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
                    fact, source="conversation", notify=True, review=reviews.with_snippet(review, text, fact)
                ))
            except Exception as exc:
                log.warning("memory flush could not store %r: %s", fact, exc)
        return results

    async def extract_and_store(
        self, text: str, username: str, source: str = "conversation", *, notify: bool = False, review: dict | None = None
    ) -> list[dict]:
        """Pull atomic facts out of text and remember each (v2 cud_memory). With ``review`` they are held for review,
        each with the sentence of ``text`` it most likely came from."""
        try:
            facts = await self.extract_facts(text, username)
        except Exception as exc:
            log.warning("fact extraction failed: %s", exc)
            return []
        results = []
        for f in facts[:20]:
            try:
                results.append(await self.remember(
                    f, source=source, notify=notify, review=reviews.with_snippet(review, text, f)
                ))
            except Exception as exc:
                log.warning("failed to store fact %r: %s", f, exc)
        return results

    async def import_document(self, path: Path, *, username: str, source: str | None = None) -> dict:
        """v2 build_initial_memory for one file, without wiping existing memories. The facts are held for the
        user's review (``pending`` counts them), so nothing already remembered changes."""
        from sentient.files.extract import extract_text

        source = source or f"file:{path.name}"
        text = await asyncio.to_thread(extract_text, path, 200_000)
        text = re.sub(r"\[page \d+\]\n", "", text)
        counts = {"added": 0, "updated": 0, "skipped": 0, "pending": 0}
        review = reviews.note(path.name)
        for chunk in chunk_text(text, self.config.memory.import_chunk_chars):
            for r in await self.extract_and_store(chunk, username, source=source, review=review):
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
    ) -> list[dict]:
        if topic:
            topic = normalize_topics([topic])[0]
        if q and q.strip():
            try:
                hits = await self._search(
                    q, min(offset + limit, 500) * (3 if topic else 1), self.config.memory.min_similarity, source
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
            " ORDER BY created_at DESC LIMIT ?",
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
        same = await self.store.fetchone(
            f"SELECT id FROM facts WHERE content = ? COLLATE NOCASE AND {ACTIVE}", (text,)
        )
        if same is not None:
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
        for i in ids[1:]:
            if i in rows and await self.forget(i):
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
