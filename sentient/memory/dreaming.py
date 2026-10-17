"""Nightly memory consolidation, called dreaming (docs/API.md section 15).

Once a day at ``dreaming.time`` (local), when the user has not chatted for
``require_idle_minutes``, or on demand via REST, a dream runs bounded steps:

a. merge near-duplicate facts: embedding similarity AND word overlap, with no conflicting
   names/places/numbers; pure duplicates merge without a model call, others ask the model
   for one merged fact that must keep every detail (ids stay stable: the kept fact keeps its id).
b. settle contradictions: facts are grouped by subject (name, "Maya's sister Riya", the user as
   "I"/"the user") and paired when they share an attribute family (residence, job, relationship, diet,
   health, ownership, schedule) or differ in details/negation; the model confirms and names the
   current fact; the newer wins unless only the older has explicit recent wording. Cleared pairs are
   remembered so they are not asked again until a fact changes.
c. promote short-term facts that keep being recalled to long-term.
d. purge expired short-term facts.
e. refresh the user model.
f. write a first-person journal and stats; notify only when something meaningful changed.

Raw cosine alone never decides a merge or a contradiction: local embedding models score
"lives in Pune" vs "lives in Mumbai" at 0.93.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from sentient.llm.jobs import detached
from sentient.memory import languages, prompts, scripts
from sentient.memory.facts import (
    TIME_WORDING_RE,
    conflict_strength,
    new_details,
    profile_fact_async,
    word_overlap,
)
from sentient.memory.schema import ensure_memory_schema
from sentient.memory.vectors import cosine
from sentient.services import Service, cancel_tasks
from sentient.store.db import new_id, now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

TICK_SECONDS = 300
MAX_CLUSTER = 5
MAX_JOURNAL_ITEMS = 8
MAX_CHECKED_PAIRS = 1000


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_hhmm(value: str) -> tuple[int, int]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", value or "")
    if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
        return int(m.group(1)), int(m.group(2))
    return 3, 0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1", "conflict", "y"}


def _flag(value: Any) -> bool | None:
    """A yes/no answer from a model: True, False, or None when it is neither."""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def similar_pairs(facts: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    """Index pairs (i, j, similarity) of facts whose vectors are at least ``threshold`` similar, best first."""
    idx = [n for n, f in enumerate(facts) if f.get("vector")]
    if len(idx) < 2:
        return []
    dims = {len(facts[n]["vector"]) for n in idx}
    if len(dims) != 1:
        dim = max(dims, key=lambda d: sum(1 for n in idx if len(facts[n]["vector"]) == d))
        idx = [n for n in idx if len(facts[n]["vector"]) == dim]
    pairs: list[tuple[int, int, float]] = []
    try:
        import numpy as np

        mat = np.array([facts[n]["vector"] for n in idx], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        sim = mat @ mat.T
        ii, jj = np.where(np.triu(sim, k=1) >= threshold)
        pairs = [(idx[a], idx[b], float(sim[a, b])) for a, b in zip(ii.tolist(), jj.tolist(), strict=True)]
    except ImportError:  # pragma: no cover - numpy is normally installed
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                s = cosine(facts[idx[a]]["vector"], facts[idx[b]]["vector"])
                if s >= threshold:
                    pairs.append((idx[a], idx[b], s))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def covers(candidate: str, others: list[str]) -> bool:
    """True when ``candidate`` mentions every name, place and number of ``others``."""
    return all(not new_details(o, candidate) for o in others)


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


class DreamingService(Service):
    name = "dreaming"
    model_kind = "memory"
    pause_on_stop = True  # no dream starts while Sentient is stopped (Stop everything)

    def __init__(self, app: SentientApp):
        super().__init__(app)
        self.clock = lambda: datetime.now(UTC)
        self._last_activity: datetime | None = None
        self._current: str | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def cfg(self):
        return self.app.config.dreaming

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await ensure_memory_schema(self.app.store)
        await self.app.store.execute(
            "UPDATE dreams SET status = 'error', error = 'interrupted by a restart', finished_at = ?"
            " WHERE status = 'running'",
            (now_iso(),),
        )
        if not self.app.enable_background:
            return
        self._loops.append(asyncio.create_task(self._listen(), name="dreaming:bus"))
        self.run_every(TICK_SECONDS, self._tick, name="schedule", initial_delay=120)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        await super().stop()

    async def halt(self) -> int:
        """Stop everything: cancel the dream that is running and record it as stopped."""
        cancelled = await super().halt() + await cancel_tasks([self._task] if self._task is not None else [])
        dream_id, self._current = self._current, None
        if dream_id is not None:
            await self.app.store.execute(
                "UPDATE dreams SET status = 'error', error = ?, finished_at = ? WHERE id = ? AND status = 'running'",
                ("Stopped by Stop everything.", now_iso(), dream_id),
            )
            dream = await self.get(dream_id)
            if dream is not None:
                self.app.bus.publish("dream.updated", dream)
        return cancelled

    async def _listen(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                if event.get("type") == "chat.turn_completed":
                    self.note_activity()

    async def _tick(self) -> None:
        await self.tick()

    # ------------------------------------------------------------------ scheduling
    def note_activity(self, when: datetime | None = None) -> None:
        self._last_activity = when or self.clock()

    def local_now(self, now: datetime) -> datetime:
        tz = (self.app.config.assistant.timezone or "").strip()
        if tz and tz.lower() != "auto":
            try:
                return now.astimezone(ZoneInfo(tz))
            except Exception:
                log.debug("unknown timezone %r; using the system zone", tz)
        return now.astimezone()

    async def last_activity(self) -> datetime | None:
        row = await self.app.store.fetchone("SELECT MAX(created_at) AS t FROM messages WHERE role = 'user'")
        seen = [t for t in (self._last_activity, _parse_ts(row["t"] if row else None)) if t is not None]
        return max(seen) if seen else None

    async def is_due(self, now: datetime | None = None) -> tuple[bool, str]:
        cfg = self.cfg
        if not cfg.enabled:
            return False, "disabled"
        if self.app.stopped:
            return False, "stopped"
        now = now or self.clock()
        local = self.local_now(now)
        hour, minute = parse_hhmm(cfg.time)
        if local < local.replace(hour=hour, minute=minute, second=0, microsecond=0):
            return False, "not yet time"
        if await self.app.store.get_meta("dreaming.last_scheduled_date") == local.date().isoformat():
            return False, "already dreamed today"
        if self._current is not None:
            return False, "a dream is running"
        last = await self.last_activity()
        if last is not None and now - last < timedelta(minutes=cfg.require_idle_minutes):
            return False, "user is active"
        return True, "due"

    async def tick(self, now: datetime | None = None) -> dict | None:
        """Scheduler step: run a dream if one is due. Returns the finished Dream or None."""
        now = now or self.clock()
        due, _reason = await self.is_due(now)
        if not due:
            return None
        await self.app.store.set_meta("dreaming.last_scheduled_date", self.local_now(now).date().isoformat())
        return await self.run(trigger="schedule")

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _dream(r: Any) -> dict:
        try:
            stats = json.loads(r["stats"] or "{}")
        except (TypeError, ValueError):
            stats = {}
        return {
            "id": r["id"], "started_at": r["started_at"], "finished_at": r["finished_at"], "status": r["status"],
            "trigger": r["trigger"], "stats": stats, "journal_md": r["journal_md"] or "", "error": r["error"],
        }

    async def get(self, dream_id: str) -> dict | None:
        await ensure_memory_schema(self.app.store)
        r = await self.app.store.fetchone("SELECT * FROM dreams WHERE id = ?", (dream_id,))
        return self._dream(r) if r else None

    async def list(self, limit: int = 20) -> list[dict]:
        await ensure_memory_schema(self.app.store)
        rows = await self.app.store.fetchall(
            "SELECT * FROM dreams ORDER BY started_at DESC LIMIT ?", (max(1, min(limit, 200)),)
        )
        return [self._dream(r) for r in rows]

    # ------------------------------------------------------------------ running
    async def _begin(self, trigger: str) -> dict:
        await ensure_memory_schema(self.app.store)
        did = new_id()
        await self.app.store.execute(
            "INSERT INTO dreams(id, started_at, status, trigger, stats, journal_md) VALUES(?,?,?,?,?,?)",
            (did, now_iso(), "running", trigger, "{}", ""),
        )
        self._current = did
        dream = await self.get(did)
        assert dream is not None
        self.app.bus.publish("dream.updated", dream)
        return dream

    async def start_run(self, trigger: str = "manual") -> dict:
        """Start a dream in the background and return it with status ``running`` (REST)."""
        async with self._lock:
            if self._current is not None:
                running = await self.get(self._current)
                if running is not None:
                    return running
            dream = await self._begin(trigger)
        # memory upkeep that waits for a quiet moment (#149); on battery only a critical level holds it, because
        # a person asked for it (#296). Nightly dreams keep the battery limits.
        requested = trigger == "manual"  # each press is its own request
        self._task = asyncio.create_task(
            self._execute(dream["id"]), name="dreaming:run", context=detached("memory", requested=requested)
        )
        return dream

    async def run(self, trigger: str = "manual") -> dict:
        """Run a dream to completion and return it."""
        async with self._lock:
            if self._current is not None:
                running = await self.get(self._current)
                if running is not None:
                    return running
            dream = await self._begin(trigger)
        return await self._execute(dream["id"])

    async def _execute(self, dream_id: str) -> dict:
        cfg = self.cfg
        stats = {
            "facts_reviewed": 0, "merged": 0, "contradictions_resolved": 0, "promoted": 0, "expired": 0,
            "insights_updated": 0, "languages_merged": 0,
        }
        notes: dict[str, Any] = {"merged": [], "languages": [], "contradictions": [], "promoted": [], "insights": None}
        errors: list[str] = []
        status = "completed"
        budget = [cfg.max_model_calls]
        mem = self.app.memory
        try:
            if mem is not None:
                steps = [
                    ("merge", self._merge_step),
                    ("languages", self._language_step),
                    ("contradictions", self._contradiction_step),
                ]
                for label, step in steps:
                    try:
                        await step(stats, notes, budget)
                    except Exception as exc:
                        log.exception("dream step %s failed", label)
                        errors.append(f"{label}: {exc}")
                try:
                    promoted = await mem.promote_recalled(cfg.promote_min_recalls)
                    stats["promoted"] = len(promoted)
                    notes["promoted"] = [p["content"] for p in promoted]
                except Exception as exc:
                    errors.append(f"promote: {exc}")
                try:
                    stats["expired"] = await mem.purge_expired()
                except Exception as exc:
                    errors.append(f"purge: {exc}")
            if cfg.refresh_user_model and self.app.config.user_model.enabled:
                try:
                    res = await self.app.user_model.refresh(trigger="dream")
                    stats["insights_updated"] = res["added"] + res["updated"] + res["disputed"]
                    notes["insights"] = res
                except Exception as exc:
                    errors.append(f"user model: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("dream failed")
            status = "error"
            errors.append(str(exc))
        journal = self.journal(stats, notes, errors)
        await self.app.store.execute(
            "UPDATE dreams SET status = ?, finished_at = ?, stats = ?, journal_md = ?, error = ? WHERE id = ?",
            (status, now_iso(), json.dumps(stats), journal, "; ".join(errors) or None, dream_id),
        )
        self._current = None
        dream = await self.get(dream_id)
        assert dream is not None
        self.app.bus.publish("dream.updated", dream)
        meaningful = stats["merged"] + stats["contradictions_resolved"] + stats["promoted"] + stats["insights_updated"]
        if cfg.notify and meaningful:
            try:
                await self.app.notify(
                    "info", journal.split("\n", 1)[0], title="Memory tidied while you were away",
                    payload={"dream_id": dream_id},
                )
            except Exception as exc:
                log.debug("dream notification failed: %s", exc)
        return dream

    # ------------------------------------------------------------------ steps
    def _name(self) -> str:
        return self.app.config.assistant.user_name.strip() or "the user"

    async def _merge_step(self, stats: dict, notes: dict, budget: list[int]) -> None:
        cfg, mem = self.cfg, self.app.memory
        assert mem is not None
        facts = await mem.active_facts(cfg.max_facts)
        stats["facts_reviewed"] = len(facts)
        parent = list(range(len(facts)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, _sim in similar_pairs(facts, cfg.merge_similarity):
            if facts[i].get("scope") != facts[j].get("scope"):
                continue  # memory scopes stay apart (#155): a work memory never folds into a personal one
            a, b = facts[i]["content"], facts[j]["content"]
            if word_overlap(a, b) < cfg.merge_min_overlap:
                continue
            if new_details(a, b) and new_details(b, a):
                continue  # each carries a detail the other lacks: a contradiction candidate, not a duplicate
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
        clusters: dict[int, list[dict]] = {}
        for n, f in enumerate(facts):
            clusters.setdefault(find(n), []).append(f)
        groups = [sorted(g, key=lambda f: f["id"])[:MAX_CLUSTER] for g in clusters.values() if len(g) > 1]
        for group in groups:
            ids = [f["id"] for f in group]
            contents = [f["content"] for f in group]
            coverers = sorted(
                (f for f in group if covers(f["content"], [c for c in contents if c != f["content"]])),
                key=lambda f: -len(f["content"]),
            )
            keep_id, merged = min(ids), ""
            pure = coverers and all(word_overlap(coverers[0]["content"], c) >= 0.85 for c in contents)
            if pure:
                merged = coverers[0]["content"]
            elif budget[0] > 0:
                budget[0] -= 1
                merged, chosen = await self._model_merge(group)
                if merged:
                    keep_id = chosen if chosen in ids else keep_id
                elif coverers:
                    merged = coverers[0]["content"]
            elif coverers:
                merged = coverers[0]["content"]
            if not merged:
                continue
            out = await mem.merge_facts(keep_id, merged, [i for i in ids if i != keep_id])
            if out is not None:
                stats["merged"] += len(ids) - 1
                notes["merged"].append({"count": len(ids), "content": merged})

    async def _model_merge(self, group: list[dict]) -> tuple[str, int | None]:
        listing = "\n".join(f"- id {f['id']}: {f['content']}" for f in group)
        try:
            raw = await self.app.llm.complete_json(
                self.cfg.role,
                [
                    {"role": "system", "content": prompts.DREAM_MERGE_SYSTEM.format(name=self._name())},
                    {"role": "user", "content": listing},
                ],
            )
        except Exception as exc:
            log.warning("dream merge call failed: %s", exc)
            return "", None
        if not isinstance(raw, dict):
            return "", None
        fact = " ".join(str(raw.get("fact") or raw.get("merged") or raw.get("content") or "").split())
        if not fact or fact.lower() in {"null", "none"}:
            return "", None
        contents = [f["content"] for f in group]
        if not covers(fact, contents) or new_details(fact, " ".join(contents)):
            log.info("dream merge rejected (lost or invented details): %r", fact)
            return "", None
        try:
            keep = int(raw.get("keep_id")) if raw.get("keep_id") is not None else None
        except (TypeError, ValueError):
            keep = None
        return fact, keep

    async def _language_step(self, stats: dict, notes: dict, budget: list[int]) -> None:
        """Merge the same memory saved in two languages (#353), only when the model says they mean exactly the same
        thing, neither adds a detail and they don't conflict. The merged-away one waits in the review history, where
        the user can undo it; a pair the user split again is never merged again."""
        cfg, mem, store = self.cfg, self.app.memory, self.app.store
        assert mem is not None
        if not cfg.merge_languages:
            return
        facts = await mem.active_facts(cfg.max_facts)
        model = self.app.llm.model_for("embedding")
        floor = cfg.language_merge_similarity if languages.multilingual(model) else None
        try:
            checked_list = [str(k) for k in json.loads(await store.get_meta("dreaming.language_pairs") or "[]")]
        except (TypeError, ValueError):
            checked_list = []
        checked = set(checked_list)
        kept_apart = await mem.kept_apart()
        calls = max(1, cfg.max_model_calls // 2)  # leave room for contradictions
        gone: set[int] = set()
        name = self._name()
        for _sim, i, j in languages.language_pairs(facts, self.app.config.assistant.user_name, floor):
            if budget[0] <= 0 or calls <= 0:
                break
            hindi, other = facts[i], facts[j]
            if hindi["id"] in gone or other["id"] in gone or hindi.get("scope") != other.get("scope"):
                continue
            ids = tuple(sorted((hindi["id"], other["id"])))
            if ids in kept_apart:
                continue
            key = f"{hindi['id']}@{hindi['updated_at']}|{other['id']}@{other['updated_at']}"
            if key in checked:
                continue
            budget[0] -= 1
            calls -= 1
            verdict = await self._same_meaning(hindi["content"], other["content"], name)
            if verdict is None:
                continue  # no clear answer: ask again next time
            if not verdict:
                checked.add(key)
                checked_list.append(key)
                continue
            keep, drop = self._clearer(hindi, other, [f for f in facts if f["id"] not in gone])
            if await mem.merge_translation(keep["id"], drop["id"]):
                gone.add(drop["id"])
                stats["merged"] += 1
                stats["languages_merged"] += 1
                notes["languages"].append({"kept": keep["content"], "merged": drop["content"]})
        if checked_list:
            await store.set_meta("dreaming.language_pairs", json.dumps(checked_list[-MAX_CHECKED_PAIRS:]))

    async def _same_meaning(self, a: str, b: str, name: str) -> bool | None:
        """True only when the model says the two mean exactly the same thing, with no extra detail and no conflict,
        and its English for ``a`` has the same words as ``b`` (small models call "every morning" no extra detail);
        False when it says otherwise; None when its answer is unclear."""
        try:
            raw = await self.app.llm.complete_json(
                self.cfg.role,
                [
                    {"role": "system", "content": prompts.DREAM_LANGUAGE_SYSTEM.format(name=name)},
                    {"role": "user", "content": prompts.DREAM_LANGUAGE_USER.format(a=a, b=b)},
                ],
            )
        except Exception as exc:
            log.warning("dream language check failed: %s", exc)
            return None
        if not isinstance(raw, dict) or "same" not in raw:
            return None
        same = _flag(raw.get("same"))
        extra = str(raw.get("extra") or "none").strip().lower()
        conflict = _flag(raw.get("conflict"))
        english = " ".join(str(raw.get("a_in_english") or "").split())
        log.info(
            "dream language check: same=%s extra=%s conflict=%s a_in_english=%r for %r / %r",
            same, extra, conflict, english, a, b,
        )
        if same is False:
            return False
        if same is None or conflict is None or not english:
            return None
        return extra in {"none", "", "neither", "null"} and not conflict and languages.same_words(english, b)

    def _clearer(self, hindi: dict, other: dict, facts: list[dict]) -> tuple[dict, dict]:
        """(kept, merged away): a long-term memory over a short-term one (so nothing is forgotten sooner), else the
        wording in the language most of the user's active memories (``facts``, without those merged away tonight) are
        in; on a tie the older one stays."""
        long_h, long_o = hindi["memory_type"] == "long-term", other["memory_type"] == "long-term"
        if long_h != long_o:
            return (hindi, other) if long_h else (other, hindi)
        written = sum(1 for f in facts if scripts.leads_with_devanagari(f["content"]))
        if written * 2 > len(facts):
            return hindi, other
        if written * 2 < len(facts):
            return other, hindi
        older_first = (hindi["created_at"], hindi["id"]) <= (other["created_at"], other["id"])
        return (hindi, other) if older_first else (other, hindi)

    @staticmethod
    def contradiction_candidate(a: str, b: str, user_name: str = "") -> bool:
        return conflict_strength(a, b, user_name) > 0

    async def contradiction_pairs(self, facts: list[dict]) -> list[tuple[int, float, int, int]]:
        """Candidate pairs ``(strength, similarity, i, j)``, strongest first.

        Facts are grouped by subject (the person or entity a fact is about, including possessive relations
        and the user's own name), so "Maya moved to Bengaluru" meets "Maya lives in Pune" without sharing
        a leading word or a high embedding score. Pairs sharing only a broad attribute also need
        ``contradiction_similarity``. A fact's attribute family falls back to embedding similarity when the
        English keyword families miss it, e.g. non-English text (``profile_fact_async``)."""
        user = self.app.config.assistant.user_name
        mem = self.app.memory
        embed_one = mem.embed_query if mem is not None else None
        model_key = mem.llm.model_for("embedding") if mem is not None else ""
        threshold = self.app.config.memory.family_similarity
        profiles = [await profile_fact_async(f["content"], embed_one, user, model_key, threshold) for f in facts]
        groups: dict[str, list[int]] = {}
        for n, prof in enumerate(profiles):
            for s in prof.subjects:
                groups.setdefault(s, []).append(n)
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, float, int, int]] = []
        for members in groups.values():
            for x, i in enumerate(members):
                for j in members[x + 1 :]:
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    strength = conflict_strength(profiles[i], profiles[j])
                    if not strength:
                        continue
                    va, vb = facts[i].get("vector"), facts[j].get("vector")
                    sim = cosine(va, vb) if va and vb and len(va) == len(vb) else None
                    if strength == 1 and (sim is None or sim < self.cfg.contradiction_similarity):
                        continue
                    out.append((strength, sim if sim is not None else 0.0, key[0], key[1]))
        out.sort(key=lambda c: (-c[0], -c[1]))
        return out

    async def _contradiction_step(self, stats: dict, notes: dict, budget: list[int]) -> None:
        cfg, mem, store = self.cfg, self.app.memory, self.app.store
        assert mem is not None
        facts = await mem.active_facts(cfg.max_facts)
        try:
            checked_list = [str(k) for k in json.loads(await store.get_meta("dreaming.checked_pairs") or "[]")]
        except (TypeError, ValueError):
            checked_list = []
        checked = set(checked_list)
        gone: set[int] = set()
        name = self._name()
        for _strength, _sim, i, j in await self.contradiction_pairs(facts):
            if budget[0] <= 0:
                break
            older, newer = sorted((facts[i], facts[j]), key=lambda f: (f["updated_at"], f["id"]))
            if older["id"] in gone or newer["id"] in gone or older.get("scope") != newer.get("scope"):
                continue
            pair_key = f"{older['id']}@{older['updated_at']}|{newer['id']}@{newer['updated_at']}"
            if pair_key in checked:
                continue  # the model already said these two can both be true
            budget[0] -= 1
            try:
                raw = await self.app.llm.complete_json(
                    cfg.role,
                    [
                        {"role": "system", "content": prompts.DREAM_CONTRADICTION_SYSTEM.format(name=name)},
                        {
                            "role": "user",
                            "content": prompts.DREAM_CONTRADICTION_USER.format(
                                a=older["content"], a_at=older["updated_at"][:10],
                                b=newer["content"], b_at=newer["updated_at"][:10],
                            ),
                        },
                    ],
                )
            except Exception as exc:
                log.warning("dream contradiction check failed: %s", exc)
                continue
            if not isinstance(raw, dict):
                continue  # malformed: ask again next time
            if not _truthy(raw.get("conflict", raw.get("contradiction"))):
                checked.add(pair_key)
                checked_list.append(pair_key)
                continue
            winner, loser = newer, older
            picked_older = str(raw.get("current") or "").strip().upper().startswith("A")
            if picked_older and TIME_WORDING_RE.search(older["content"]) and not TIME_WORDING_RE.search(newer["content"]):
                winner, loser = older, newer  # e.g. an old "moved back to Pune this month" vs a stale import
            if await mem.supersede(winner["id"], loser["id"]):
                gone.add(loser["id"])
                stats["contradictions_resolved"] += 1
                notes["contradictions"].append(
                    {"old": loser["content"], "new": winner["content"], "change": str(raw.get("change") or "").strip()}
                )
        if len(checked_list) > 0:
            await store.set_meta("dreaming.checked_pairs", json.dumps(checked_list[-MAX_CHECKED_PAIRS:]))

    # ------------------------------------------------------------------ journal
    def journal(self, stats: dict, notes: dict, errors: list[str]) -> str:
        name = self._name()
        did: list[str] = []
        duplicates = stats["merged"] - stats.get("languages_merged", 0)
        if duplicates:
            did.append(f"merged {duplicates + len(notes['merged'])} duplicate memories into {len(notes['merged'])}")
        if stats.get("languages_merged"):
            n = stats["languages_merged"]
            did.append(f"kept {n} memor{'ies' if n != 1 else 'y'} saved in two languages just once")
        if stats["contradictions_resolved"]:
            n = stats["contradictions_resolved"]
            did.append(f"settled {n} contradiction{'s' if n != 1 else ''}")
        if stats["promoted"]:
            did.append(f"kept {stats['promoted']} short-term memor{'ies' if stats['promoted'] != 1 else 'y'} for the long run")
        if stats["expired"]:
            did.append(f"let go of {stats['expired']} expired memor{'ies' if stats['expired'] != 1 else 'y'}")
        if stats["insights_updated"]:
            did.append(f"updated {stats['insights_updated']} things I understand about {name}")
        if did:
            joined = did[0] if len(did) == 1 else ", ".join(did[:-1]) + " and " + did[-1]
            opening = f"Tonight I {joined}."
        else:
            opening = f"Tonight I looked over {stats['facts_reviewed']} memories about {name} and nothing needed tidying."
        lines = [opening, ""]
        for m in notes["merged"][:MAX_JOURNAL_ITEMS]:
            lines.append(f'- {m["count"]} memories said the same thing, so I kept one: "{_short(m["content"])}"')
        for m in notes.get("languages", [])[:MAX_JOURNAL_ITEMS]:
            lines.append(
                f'- "{_short(m["merged"])}" and "{_short(m["kept"])}" said the same thing in two languages, so I kept '
                f'"{_short(m["kept"])}". You can undo this in Memory > Review.'
            )
        for c in notes["contradictions"][:MAX_JOURNAL_ITEMS]:
            change = f" ({c['change']})" if c["change"] else ""
            lines.append(f'- I used to remember "{_short(c["old"])}", but now "{_short(c["new"])}"{change}.')
        for p in notes["promoted"][:MAX_JOURNAL_ITEMS]:
            lines.append(f'- "{_short(p)}" keeps coming up, so I will keep it.')
        ins = notes.get("insights")
        if ins and (ins["added"] or ins["updated"] or ins["disputed"]):
            parts = []
            if ins["added"]:
                parts.append(f"{ins['added']} new")
            if ins["updated"]:
                parts.append(f"{ins['updated']} refined")
            if ins["disputed"]:
                parts.append(f"{ins['disputed']} I want to check with {name}")
            lines.append(f"- My picture of {name}: {', '.join(parts)}.")
        if errors:
            lines.append(f"- Some steps did not finish: {'; '.join(_short(e, 120) for e in errors)}.")
        if stats["facts_reviewed"] and did:
            lines.append(f"\nI went through {stats['facts_reviewed']} memories in all.")
        return "\n".join(lines).strip()
