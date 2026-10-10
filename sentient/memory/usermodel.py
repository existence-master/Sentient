"""Dialectic user model: evolving insights about the user (docs/API.md section 15).

Facts say what is true ("Maya lives in Pune"); insights say what the user is like
("Maya prefers short, direct answers"). Insights carry evidence and a confidence that
moves with new evidence:

- refresh: after ``refresh_after_turns`` completed chat turns (at most every
  ``min_refresh_hours``), during a dream, or via REST, one model call reads recent user
  messages, new or changed facts, conversation summaries and the current insights, and
  answers a short JSON list of operations: add, support, contradict, retire.
- contradicted insights lose confidence; below ``dispute_below`` they are marked disputed
  and a question is queued for the user. Confirmed or user-written insights are never
  changed automatically; evidence against them only raises a question.
- answering a question confirms, retires or rewrites the insight and stores a fact.
- ``context_for(text)`` is the prompt hook: no model call, a few hundred characters.
  ``context_with_sources(text)`` also returns the insights it used (memory sources).
- held for review (``pending``, ADR 0021): imported insights, and new insights the model draws only from outside
  material (messages or summaries of a chat that read outside content). Pending insights are never in a prompt,
  in a refresh or in ``get_state``; only the user approves or discards one. Support, contradict and retire
  operations that cite only outside material are ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sentient.memory import prompts
from sentient.memory import review as reviews
from sentient.memory.facts import FactMemory, word_overlap
from sentient.memory.schema import ensure_memory_schema
from sentient.memory.vectors import VecTable, cosine
from sentient.services import Service, cancel_tasks
from sentient.store.db import new_id, now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

DIMENSIONS = list(prompts.USER_MODEL_DIMENSIONS)
STATUSES = ("active", "confirmed", "disputed", "retired")
PENDING = "pending"  # held for the user's review; never one of STATUSES, so no reader but ``pending_insights`` sees it
VEC_TABLE = "user_insights_vec"
MAX_EVIDENCE = 8
SUMMARY_MAX_WORDS = 180
_COLUMNS = "rid, id, dimension, statement, confidence, status, source, evidence, created_at, updated_at, review"

_DIMENSION_ALIASES = {
    "preference": "preferences", "likes": "preferences", "tastes": "preferences",
    "communication_style": "communication", "tone": "communication", "style": "communication",
    "goal": "goals", "aspirations": "goals", "plans": "goals",
    "routine": "routines", "habits": "routines", "habit": "routines", "schedule": "routines",
    "relationship": "relationships", "people": "relationships", "family": "relationships",
    "value": "values", "beliefs": "values",
    "work": "work_style", "workstyle": "work_style", "working_style": "work_style",
    "dislike": "dislikes", "avoid": "dislikes",
    "background": "context", "life": "context", "situation": "context",
}
_OP_ALIASES = {
    "add": "add", "create": "add", "new": "add", "insert": "add",
    "support": "support", "reinforce": "support", "strengthen": "support", "confirm": "support",
    "contradict": "contradict", "weaken": "contradict", "dispute": "contradict", "conflict": "contradict",
    "retire": "retire", "remove": "retire", "delete": "retire", "drop": "retire",
}
_CONFIDENCE_WORDS = {"very high": 0.8, "high": 0.7, "medium": 0.5, "moderate": 0.5, "low": 0.35, "very low": 0.25}
_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|correct|right|true|exactly|absolutely|definitely|sure|still)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|not really|wrong|false|incorrect|never|not anymore|not any more)\b", re.I)


# ---------------------------------------------------------------------- tolerant parsing
def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def normalize_dimension(raw: Any) -> str:
    key = re.sub(r"[\s-]+", "_", str(raw or "").strip().lower())
    if key in DIMENSIONS:
        return key
    if key in _DIMENSION_ALIASES:
        return _DIMENSION_ALIASES[key]
    for d in DIMENSIONS:
        if key and (key.startswith(d.rstrip("s")) or d.startswith(key)):
            return d
    return "context"


def parse_confidence(raw: Any, default: float = 0.5) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int | float):
        val = float(raw)
    else:
        text = str(raw or "").strip().lower().rstrip("%")
        if text in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[text]
        try:
            val = float(text)
        except ValueError:
            return default
    if val > 1.0:  # "70" meaning 70%
        val = val / 100.0
    return max(0.0, min(1.0, val))


def as_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list | tuple):
        return list(raw)
    if isinstance(raw, str):
        return [p for p in re.split(r"[,;\s]+", raw) if p]
    return [raw]


def parse_operations(raw: Any) -> list[dict]:
    """Operations from a model reply, tolerating wrappers, synonyms and a single bare object."""
    ops: Any = []
    if isinstance(raw, dict):
        if any(k in raw for k in ("op", "action")) and not any(k in raw for k in ("operations", "ops")):
            ops = [raw]
        else:
            ops = raw.get("operations") or raw.get("ops") or raw.get("changes") or raw.get("insights") or []
    elif isinstance(raw, list):
        ops = raw
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list):
        return []
    out = []
    for item in ops:
        if not isinstance(item, dict):
            continue
        op = _OP_ALIASES.get(str(item.get("op") or item.get("action") or item.get("type") or "").strip().lower())
        if op:
            out.append({**item, "op": op})
    return out


def clip_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(",;:") + " ..."


def _clean_statement(text: Any, name: str) -> str:
    s = " ".join(str(text or "").split()).strip(" -*\"'")
    if name:
        s = re.sub(r"\b[Tt]he user's\b", f"{name}'s", s)
        s = re.sub(r"\b[Tt]he user\b", name, s)
    s = s[:240]
    return s[:1].upper() + s[1:] if s else ""


def _row(r: Any) -> dict:
    try:
        evidence = json.loads(r["evidence"] or "[]")
    except (TypeError, ValueError):
        evidence = []
    return {
        "id": r["id"],
        "dimension": r["dimension"],
        "statement": r["statement"],
        "confidence": round(float(r["confidence"]), 3),
        "status": r["status"],
        "source": r["source"],
        "evidence": evidence if isinstance(evidence, list) else [],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "review": reviews.load(r["review"]),
        "_rid": int(r["rid"]),
    }


def _public(ins: dict) -> dict:
    return {k: v for k, v in ins.items() if not k.startswith("_")}


def protected(ins: dict) -> bool:
    return ins["status"] == "confirmed" or ins["source"] == "user"


class UserModelService(Service):
    name = "user_model"

    def __init__(self, app: SentientApp):
        super().__init__(app)
        self.vec = VecTable(app.store, VEC_TABLE)
        self.clock = lambda: datetime.now(UTC)
        self._lock = asyncio.Lock()
        self._vectors: dict[int, list[float]] | None = None
        self._background: set[asyncio.Task] = set()

    @property
    def cfg(self):
        return self.app.config.user_model

    @property
    def user_name(self) -> str:
        return self.app.config.assistant.user_name.strip()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await ensure_memory_schema(self.app.store)
        if self.app.memory is not None and getattr(self.app.memory, "bus", None) is None:
            self.app.memory.bus = self.app.bus
        if self.app.enable_background:
            self._loops.append(asyncio.create_task(self._listen(), name="user_model:bus"))

    async def stop(self) -> None:
        for t in list(self._background):
            t.cancel()
        await super().stop()

    async def halt(self) -> int:
        return await cancel_tasks(self._background)

    async def _listen(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                if event.get("type") == "chat.turn_completed" and not self.app.stopped:
                    task = asyncio.create_task(self._safe_turn(event.get("data") or {}))
                    self._background.add(task)
                    task.add_done_callback(self._background.discard)

    async def _safe_turn(self, data: dict) -> None:
        try:
            await self.on_turn_completed(data)
        except Exception:
            log.exception("user model refresh after chat turn failed")

    async def on_turn_completed(self, data: dict | None = None, now: datetime | None = None) -> dict | None:
        """Count a finished turn; refresh when enough turns passed and the last refresh is old enough."""
        cfg = self.cfg
        if not cfg.enabled:
            return None
        store = self.app.store
        turns = int(await store.get_meta("user_model.turns_since_refresh") or 0) + 1
        await store.set_meta("user_model.turns_since_refresh", str(turns))
        if turns < cfg.refresh_after_turns or self._lock.locked():
            return None
        now = now or self.clock()
        last = _parse_ts(await store.get_meta("user_model.last_refresh_at"))
        if last is not None and now - last < timedelta(hours=cfg.min_refresh_hours):
            return None
        return await self.refresh(trigger="auto", now=now)

    # ------------------------------------------------------------------ reads
    async def _insights(self, statuses: tuple[str, ...] = STATUSES, limit: int = 500) -> list[dict]:
        await ensure_memory_schema(self.app.store)
        marks = ",".join("?" * len(statuses))
        rows = await self.app.store.fetchall(
            f"SELECT {_COLUMNS} FROM user_insights WHERE status IN ({marks})"
            " ORDER BY CASE status WHEN 'confirmed' THEN 0 WHEN 'active' THEN 1 WHEN 'disputed' THEN 2 ELSE 3 END,"
            " confidence DESC, updated_at DESC LIMIT ?",
            [*statuses, limit],
        )
        return [_row(r) for r in rows]

    async def _get(self, insight_id: str) -> dict | None:
        await ensure_memory_schema(self.app.store)
        r = await self.app.store.fetchone(f"SELECT {_COLUMNS} FROM user_insights WHERE id = ?", (insight_id,))
        return _row(r) if r else None

    async def get_insight(self, insight_id: str) -> dict | None:
        ins = await self._get(insight_id)
        return _public(ins) if ins else None

    async def open_questions(self) -> list[dict]:
        await ensure_memory_schema(self.app.store)
        rows = await self.app.store.fetchall(
            "SELECT id, question, insight_id, created_at FROM user_questions WHERE status = 'open' ORDER BY created_at"
        )
        return [dict(r) for r in rows]

    async def get_state(self) -> dict:
        store = self.app.store
        return {
            "summary": await store.get_meta("user_model.summary") or "",
            "updated_at": await store.get_meta("user_model.updated_at"),
            "insights": [_public(i) for i in await self._insights()],
            "questions": await self.open_questions(),
        }

    # ------------------------------------------------------------------ writes (shared)
    async def _touch(self, *, summary_changed: bool = False, changed: int = 0) -> None:
        await self.app.store.set_meta("user_model.updated_at", now_iso())
        self.app.bus.publish(
            "user_model.updated",
            {"summary_changed": summary_changed, "insights": changed, "questions": len(await self.open_questions())},
        )

    async def _embed_insight(self, rid: int, statement: str) -> None:
        if not self.app.store.vec_available:
            return
        try:
            vecs = await self.app.llm.embed([statement])
            if not vecs:
                return
            await self.vec.ensure(len(vecs[0]), self._reindex)
            await self.vec.upsert(rid, vecs[0])
        except Exception as exc:
            log.debug("insight embedding skipped: %s", exc)
        self._vectors = None

    async def _reindex(self) -> None:
        rows = await self.app.store.fetchall("SELECT rid, statement FROM user_insights")
        for r in rows:
            [v] = await self.app.llm.embed([r["statement"]])
            await self.vec.upsert(int(r["rid"]), v)

    async def _insert(
        self, statement: str, dimension: str, *, confidence: float, status: str, source: str, evidence: list[dict],
        review: dict | None = None,
    ) -> dict:
        iid, ts = new_id(), now_iso()
        cur = await self.app.store.execute(
            "INSERT INTO user_insights(id, dimension, statement, confidence, status, source, evidence, created_at,"
            " updated_at, review) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                iid, dimension, statement, confidence, status, source, json.dumps(evidence[-MAX_EVIDENCE:]), ts, ts,
                reviews.dump(review),
            ),
        )
        await self._embed_insight(int(cur.lastrowid), statement)
        ins = await self._get(iid)
        assert ins is not None
        return ins

    async def _save(self, ins: dict, *, reembed: bool = False) -> None:
        await self.app.store.execute(
            "UPDATE user_insights SET dimension = ?, statement = ?, confidence = ?, status = ?, source = ?,"
            " evidence = ?, updated_at = ? WHERE id = ?",
            (
                ins["dimension"], ins["statement"], round(float(ins["confidence"]), 4), ins["status"], ins["source"],
                json.dumps(ins["evidence"][-MAX_EVIDENCE:]), now_iso(), ins["id"],
            ),
        )
        if reembed:
            await self._embed_insight(ins["_rid"], ins["statement"])

    async def _ask(self, ins: dict, question: str | None = None) -> bool:
        """Queue one open question about an insight (never two for the same insight)."""
        store = self.app.store
        if await store.fetchone(
            "SELECT id FROM user_questions WHERE insight_id = ? AND status = 'open'", (ins["id"],)
        ):
            return False
        if len(await self.open_questions()) >= self.cfg.max_open_questions:
            return False
        text = " ".join(str(question or "").split())[:300]
        if len(text) < 8:
            text = f'Is this still right about you: "{ins["statement"]}"?'
        await store.execute(
            "INSERT INTO user_questions(id, question, insight_id, status, created_at) VALUES(?,?,?,?,?)",
            (new_id(), text, ins["id"], "open", now_iso()),
        )
        return True

    async def _close_questions(self, insight_id: str, status: str = "dismissed") -> None:
        await self.app.store.execute(
            "UPDATE user_questions SET status = ?, answered_at = ? WHERE insight_id = ? AND status = 'open'",
            (status, now_iso(), insight_id),
        )

    # ------------------------------------------------------------------ REST operations
    async def add_insight(self, statement: str, dimension: str = "context") -> dict:
        statement = _clean_statement(statement, self.user_name)
        if not statement:
            raise ValueError("statement is required")
        ins = await self._insert(
            statement, normalize_dimension(dimension), confidence=1.0, status="confirmed", source="user",
            evidence=[{"kind": "feedback", "ref": "manual", "quote": statement[:160], "at": now_iso()}],
        )
        await self._touch(changed=1)
        return _public(ins)

    async def import_insight(
        self, statement: str, *, source: str, dimension: str = "context", review: dict | None = None
    ) -> dict | None:
        """An insight brought over from another assistant (``source`` like ``import:hermes``): held for the user's
        review with medium confidence, so once approved new evidence can still change it. None when the same
        statement is already there."""
        statement = _clean_statement(statement, self.user_name)
        if not statement:
            return None
        await ensure_memory_schema(self.app.store)
        if await self.app.store.fetchone(
            "SELECT id FROM user_insights WHERE statement = ? COLLATE NOCASE AND status != 'retired'", (statement,)
        ):
            return None
        ins = await self._insert(
            statement, normalize_dimension(dimension), confidence=0.6, status=PENDING, source=source,
            evidence=[{"kind": "import", "ref": source, "quote": statement[:160], "at": now_iso()}],
            review=review or reviews.note(source),
        )
        await self._touch(changed=1)
        return _public(ins)

    # ------------------------------------------------------------------ review (ADR 0021)
    async def pending_insights(self, limit: int = 500) -> list[dict]:
        """Insights held for the user's review, newest first (``limit=-1``: all of them)."""
        await ensure_memory_schema(self.app.store)
        rows = await self.app.store.fetchall(
            f"SELECT {_COLUMNS} FROM user_insights WHERE status = ? ORDER BY created_at DESC LIMIT ?", (PENDING, limit)
        )
        return [_public(_row(r)) for r in rows]

    async def pending_count(self) -> int:
        await ensure_memory_schema(self.app.store)
        row = await self.app.store.fetchone("SELECT COUNT(*) AS n FROM user_insights WHERE status = ?", (PENDING,))
        return int(row["n"]) if row else 0

    async def approve_insight(self, insight_id: str, statement: str | None = None) -> dict | None:
        """The user approved a held insight: it becomes active, or confirmed in their own words when ``statement``
        rewords it. None when it is not pending."""
        ins = await self._get(insight_id)
        if ins is None or ins["status"] != PENDING:
            return None
        cleaned = _clean_statement(statement, self.user_name) if statement is not None else ""
        reembed = bool(cleaned) and cleaned != ins["statement"]
        if reembed:
            ins.update(statement=cleaned, source="user", status="confirmed", confidence=1.0)
        else:
            ins["status"] = "active"
        await self._save(ins, reembed=reembed)
        self._vectors = None
        await self._touch(changed=1)
        out = await self._get(insight_id)
        return _public(out) if out else None

    async def discard_insight(self, insight_id: str) -> bool:
        """The user turned a held insight down: it is deleted. False when it is not pending."""
        ins = await self._get(insight_id)
        if ins is None or ins["status"] != PENDING:
            return False
        return await self.delete_insight(insight_id)

    async def expire_pending(self, before: str) -> int:
        """Delete held insights created before ``before`` (ISO time) that nobody reviewed."""
        await ensure_memory_schema(self.app.store)
        rows = await self.app.store.fetchall(
            "SELECT id, rid FROM user_insights WHERE status = ? AND created_at < ?", (PENDING, before)
        )
        for r in rows:
            await self.app.store.execute("DELETE FROM user_insights WHERE id = ?", (r["id"],))
            with contextlib.suppress(Exception):
                await self.vec.delete(int(r["rid"]))
        if rows:
            self._vectors = None
            await self._touch(changed=len(rows))
        return len(rows)

    async def delete_by_source(self, source: str) -> int:
        """Delete every insight with this ``source`` (undo an import)."""
        await ensure_memory_schema(self.app.store)
        rows = await self.app.store.fetchall("SELECT id FROM user_insights WHERE source = ?", (source,))
        for r in rows:
            await self.delete_insight(r["id"])
        return len(rows)

    async def update_insight(self, insight_id: str, *, statement: str | None = None, status: str | None = None) -> dict:
        ins = await self._get(insight_id)
        if ins is None:
            raise KeyError(insight_id)
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        reembed = False
        if statement is not None:
            cleaned = _clean_statement(statement, self.user_name)
            if not cleaned:
                raise ValueError("statement is empty")
            if cleaned != ins["statement"]:
                ins["statement"], ins["source"], reembed = cleaned, "user", True
                ins["status"], ins["confidence"] = "confirmed", 1.0
        if status is not None:
            ins["status"] = status
            if status == "confirmed":
                ins["confidence"] = max(ins["confidence"], 0.9)
        await self._save(ins, reembed=reembed)
        if reembed or ins["status"] in {"confirmed", "retired"}:
            await self._close_questions(insight_id)
        await self._touch(changed=1)
        out = await self._get(insight_id)
        assert out is not None
        return _public(out)

    async def delete_insight(self, insight_id: str) -> bool:
        ins = await self._get(insight_id)
        if ins is None:
            return False
        await self.app.store.execute("DELETE FROM user_questions WHERE insight_id = ?", (insight_id,))
        await self.app.store.execute("DELETE FROM user_insights WHERE id = ?", (insight_id,))
        with contextlib.suppress(Exception):
            await self.vec.delete(ins["_rid"])
        self._vectors = None
        await self._touch(changed=1)
        return True

    async def dismiss_question(self, question_id: str) -> bool:
        await ensure_memory_schema(self.app.store)
        cur = await self.app.store.execute(
            "UPDATE user_questions SET status = 'dismissed', answered_at = ? WHERE id = ? AND status = 'open'",
            (now_iso(), question_id),
        )
        if cur.rowcount:
            await self._touch()
        return bool(cur.rowcount)

    async def answer_question(self, question_id: str, answer: str) -> dict:
        """Apply the user's answer: confirm, retire or rewrite the insight, and remember a fact."""
        await ensure_memory_schema(self.app.store)
        answer = " ".join(str(answer or "").split())
        if not answer:
            raise ValueError("answer is required")
        q = await self.app.store.fetchone(
            "SELECT id, question, insight_id FROM user_questions WHERE id = ? AND status = 'open'", (question_id,)
        )
        if q is None:
            raise KeyError(question_id)
        ins = await self._get(q["insight_id"]) if q["insight_id"] else None
        name = self.user_name or "the user"
        statement = ins["statement"] if ins else ""
        verdict, new_statement, fact = "", "", ""
        try:
            raw = await self.app.llm.complete_json(
                self.cfg.role,
                [
                    {"role": "system", "content": prompts.USER_MODEL_ANSWER_SYSTEM.format(name=name)},
                    {
                        "role": "user",
                        "content": prompts.USER_MODEL_ANSWER_USER.format(
                            statement=statement or "(none)", question=q["question"], answer=answer[:600], name=name
                        ),
                    },
                ],
            )
            if isinstance(raw, dict):
                verdict = str(raw.get("verdict") or raw.get("action") or "").strip().lower()
                new_statement = _clean_statement(raw.get("statement"), self.user_name)
                fact = str(raw.get("fact") or "").strip()
        except Exception as exc:
            log.warning("interpreting the answer failed, using a simple reading: %s", exc)
        if verdict not in {"confirm", "retire", "rewrite"}:
            verdict = "confirm" if _YES_RE.match(answer) else "retire" if _NO_RE.match(answer) else "rewrite"
        evidence = {"kind": "feedback", "ref": question_id, "quote": answer[:160], "at": now_iso()}
        if ins is not None:
            ins["evidence"].append(evidence)
            if verdict == "confirm":
                ins["status"], ins["confidence"] = "confirmed", max(ins["confidence"], 0.9)
                await self._save(ins)
            elif verdict == "retire":
                ins["status"] = "retired"
                await self._save(ins)
            else:
                if not new_statement or new_statement == ins["statement"]:
                    new_statement = _clean_statement(f"{name} says: {answer}", self.user_name)
                ins.update(statement=new_statement, status="confirmed", source="user", confidence=1.0)
                await self._save(ins, reembed=True)
        if not fact:
            if verdict == "confirm" and statement:
                fact = statement
            elif verdict == "rewrite" and ins is not None:
                fact = ins["statement"]
            elif statement:
                fact = f'{name} said this is not true: "{statement}"'
        await self.app.store.execute(
            "UPDATE user_questions SET status = 'answered', answer = ?, answered_at = ? WHERE id = ?",
            (answer, now_iso(), question_id),
        )
        stored = None
        cleaned = FactMemory.clean_fact(fact, self.user_name) if fact else None
        if cleaned and self.app.memory is not None:
            try:
                stored = await self.app.memory.remember(cleaned, source="user_model", notify=True)
            except Exception as exc:
                log.warning("could not store the answer as a fact: %s", exc)
        await self._touch(changed=1 if ins else 0)
        out = await self._get(ins["id"]) if ins else None
        return {"ok": True, "verdict": verdict, "insight": _public(out) if out else None, "fact": stored}

    # ------------------------------------------------------------------ refresh
    async def _gather(self, since: str) -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
        """Evidence since the last refresh: aliases (m1, f12, s1) -> evidence dict, prompt sections, and the aliases
        that are outside material (from a chat that read outside content, ADR 0018) -> where it came from."""
        cfg, store = self.cfg, self.app.store
        evidence: dict[str, dict] = {}
        outside: dict[str, str] = {}
        sections = {"messages": "", "facts": "", "summaries": ""}
        if cfg.recent_messages:
            rows = await store.fetchall(
                "SELECT m.id, m.content, m.created_at, s.untrusted FROM messages m LEFT JOIN sessions s"
                " ON s.id = m.session_id WHERE m.role = 'user' AND m.content IS NOT NULL"
                " AND m.content != '' AND m.created_at > ? ORDER BY m.created_at DESC LIMIT ?",
                (since, cfg.recent_messages),
            )
            lines = []
            for i, r in enumerate(reversed(rows), 1):
                text = " ".join(r["content"].split())[:300]
                evidence[f"m{i}"] = {"kind": "message", "ref": r["id"], "quote": text[:160], "at": r["created_at"]}
                if r["untrusted"]:
                    outside[f"m{i}"] = r["untrusted"]
                lines.append(f"- m{i}: {text}")
            sections["messages"] = "\n".join(lines)
        if cfg.recent_facts:
            rows = await store.fetchall(
                "SELECT id, content, updated_at FROM facts WHERE status = 'active' AND updated_at > ?"
                " AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                (since, now_iso(), cfg.recent_facts),
            )
            lines = []
            for r in reversed(rows):
                alias = f"f{int(r['id'])}"
                evidence[alias] = {"kind": "fact", "ref": str(r["id"]), "quote": r["content"][:160], "at": r["updated_at"]}
                lines.append(f"- {alias}: {r['content'][:240]}")
            sections["facts"] = "\n".join(lines)
        if cfg.recent_summaries:
            rows = await store.fetchall(
                "SELECT m.id, m.content, m.end_at, COALESCE(NULLIF(m.untrusted, ''), s.untrusted) AS untrusted"
                " FROM summaries m LEFT JOIN sessions s"
                " ON s.id = m.session_id WHERE m.created_at > ? ORDER BY m.created_at DESC LIMIT ?",
                (since, cfg.recent_summaries),
            )
            lines = []
            for i, r in enumerate(reversed(rows), 1):
                text = " ".join(r["content"].split())[:400]
                evidence[f"s{i}"] = {"kind": "summary", "ref": r["id"], "quote": text[:160], "at": r["end_at"]}
                if r["untrusted"]:
                    outside[f"s{i}"] = r["untrusted"]
                lines.append(f"- s{i}: {text}")
            sections["summaries"] = "\n".join(lines)
        return evidence, sections, outside

    @staticmethod
    def _evidence_for(op: dict, evidence: dict[str, dict]) -> list[dict]:
        refs = as_list(op.get("evidence") or op.get("evidence_ids") or op.get("refs"))
        out: list[dict] = []
        for ref in refs:
            if isinstance(ref, dict):
                ref = ref.get("ref") or ref.get("id") or ""
            key = str(ref).strip().strip("[]()\"'").lower().replace(":", "").replace("#", "")
            if key in evidence and evidence[key] not in out:
                out.append(evidence[key])
        return out

    async def refresh(self, *, trigger: str = "manual", now: datetime | None = None) -> dict:
        """Run one dialectic refresh. Returns ``{added, updated, disputed, questions, held}``."""
        counts = {"added": 0, "updated": 0, "disputed": 0, "questions": 0, "held": 0}
        if not self.cfg.enabled:
            return counts
        async with self._lock:
            return await self._refresh(counts, trigger, now or self.clock())

    async def _refresh(self, counts: dict, trigger: str, now: datetime) -> dict:
        cfg, store = self.cfg, self.app.store
        await ensure_memory_schema(store)
        since = await store.get_meta("user_model.last_refresh_at") or ""
        await store.set_meta("user_model.turns_since_refresh", "0")
        evidence, sections, outside = await self._gather(since)
        if not evidence:
            return counts
        outside_refs = [evidence[a] for a in outside]
        current = await self._insights(("confirmed", "active", "disputed"), limit=40)
        aliases = {f"i{n}": ins for n, ins in enumerate(current, 1)}
        by_id = {ins["id"]: ins for ins in current}
        name = self.user_name or "the user"
        summary = await store.get_meta("user_model.summary") or ""
        insight_lines = "\n".join(
            f"- {a} | {i['dimension']} | {i['confidence']:.2f} | {i['status']} | {i['statement']}"
            for a, i in aliases.items()
        )
        try:
            raw = await self.app.llm.complete_json(
                cfg.role,
                [
                    {
                        "role": "system",
                        "content": prompts.USER_MODEL_SYSTEM.format(name=name, dimensions=", ".join(DIMENSIONS)),
                    },
                    {
                        "role": "user",
                        "content": prompts.USER_MODEL_USER.format(
                            name=name,
                            summary=summary or "(none yet)",
                            insights=insight_lines or "(none yet)",
                            messages=sections["messages"] or "(none)",
                            facts=sections["facts"] or "(none)",
                            summaries=sections["summaries"] or "(none)",
                        ),
                    },
                ],
            )
        except Exception as exc:
            log.warning("user model refresh failed: %s", exc)
            return counts

        retired = await self._insights(("retired",), limit=200)
        held = await self.pending_insights()
        active_inferred = sum(1 for i in current if not protected(i))
        changed_ids: set[str] = set()

        def target(op: dict) -> dict | None:
            ref = str(op.get("id") or op.get("insight_id") or op.get("insight") or "").strip().lower()
            if ref.isdigit():
                ref = f"i{ref}"
            return aliases.get(ref) or by_id.get(ref)

        proposed: dict[str, str] = {}  # insight id -> question text offered earlier in this reply

        for op in parse_operations(raw)[: cfg.max_operations]:
            kind = op["op"]
            refs = self._evidence_for(op, evidence)
            trusted = [e for e in refs if e not in outside_refs]
            # only outside material behind it (or nothing, while outside material was offered): it may not change
            # anything the user relies on; a new insight waits for review (ADR 0021)
            untrusted_only = bool(outside) and not trusted
            aimed = target(op)
            if aimed is not None:
                if op.get("question"):
                    proposed[aimed["id"]] = str(op["question"])
                elif aimed["id"] in proposed:
                    op = {**op, "question": proposed[aimed["id"]]}
            if kind == "add":
                statement = _clean_statement(op.get("statement") or op.get("insight") or op.get("text"), self.user_name)
                if len(statement.split()) < 3:
                    continue
                twin = next(
                    (i for i in current if i["statement"].lower() == statement.lower()
                     or word_overlap(i["statement"], statement) >= 0.7),
                    None,
                )
                if twin is not None:  # already covered: treat as support
                    op, kind = {"op": "support", "id": twin["id"], "evidence": op.get("evidence")}, "support"
                elif any(word_overlap(i["statement"], statement) >= 0.7 for i in retired):
                    continue  # retired before (often by the user): do not bring it back
                elif any(word_overlap(i["statement"], statement) >= 0.7 for i in held):
                    continue  # already waiting for the user's review
                elif untrusted_only:
                    conf = min(max(parse_confidence(op.get("confidence"), 0.5), 0.2), 0.8 if refs else 0.4)
                    first = next((a for a in outside if evidence[a] in refs), next(iter(outside)))
                    ins = await self._insert(
                        statement, normalize_dimension(op.get("dimension")), confidence=conf, status=PENDING,
                        source="inferred", evidence=refs,
                        review=reviews.note(outside[first], evidence[first]["quote"]),
                    )
                    held.append(_public(ins))
                    counts["held"] += 1
                    changed_ids.add(ins["id"])
                    continue
                else:
                    if active_inferred >= cfg.max_active_insights:
                        continue
                    conf = min(max(parse_confidence(op.get("confidence"), 0.5), 0.2), 0.8)
                    if not refs:
                        conf = min(conf, 0.4)
                    ins = await self._insert(
                        statement, normalize_dimension(op.get("dimension")), confidence=conf,
                        status="active", source="inferred", evidence=refs,
                    )
                    current.append(ins)
                    by_id[ins["id"]] = ins
                    active_inferred += 1
                    counts["added"] += 1
                    changed_ids.add(ins["id"])
                    continue
            ins = target(op)
            if ins is None or ins["status"] == "retired" or untrusted_only:
                continue
            refs = trusted
            if kind == "support":
                if protected(ins):
                    continue
                step = cfg.support_step if refs else cfg.support_step / 2
                ins["confidence"] = min(0.95, ins["confidence"] + step)
                ins["evidence"].extend(e for e in refs if e not in ins["evidence"])
                if ins["status"] == "disputed" and ins["confidence"] >= cfg.dispute_below:
                    ins["status"] = "active"
                await self._save(ins)
                counts["updated"] += int(ins["id"] not in changed_ids)
                changed_ids.add(ins["id"])
            elif kind == "contradict":
                if protected(ins):
                    counts["questions"] += int(await self._ask(ins, op.get("question")))
                    continue
                ins["confidence"] = max(0.0, ins["confidence"] - cfg.contradict_step)
                ins["evidence"].extend(e for e in refs if e not in ins["evidence"])
                if ins["confidence"] < cfg.dispute_below and ins["status"] != "disputed":
                    ins["status"] = "disputed"
                    counts["disputed"] += 1
                    counts["questions"] += int(await self._ask(ins, op.get("question")))
                elif ins["id"] not in changed_ids:
                    counts["updated"] += 1
                await self._save(ins)
                changed_ids.add(ins["id"])
            elif kind == "retire":
                if protected(ins):
                    counts["questions"] += int(await self._ask(ins, op.get("question")))
                    continue
                ins["status"] = "retired"
                await self._save(ins)
                await self._close_questions(ins["id"])
                counts["updated"] += int(ins["id"] not in changed_ids)
                changed_ids.add(ins["id"])

        new_summary = self._summary_from(raw)
        if not new_summary and not summary:
            new_summary = await self._fallback_summary()
        summary_changed = bool(new_summary) and new_summary != summary
        if summary_changed:
            await store.set_meta("user_model.summary", new_summary)
        await store.set_meta("user_model.last_refresh_at", now.isoformat())
        if summary_changed or changed_ids or counts["questions"]:
            await self._touch(summary_changed=summary_changed, changed=len(changed_ids))
        log.info("user model refresh (%s): %s", trigger, counts)
        return counts

    @staticmethod
    def _summary_from(raw: Any) -> str:
        if not isinstance(raw, dict):
            return ""
        text = raw.get("summary") or raw.get("portrait") or ""
        if isinstance(text, list):
            text = "\n".join(f"- {t}" for t in text if t)
        return clip_words(str(text).strip(), SUMMARY_MAX_WORDS) if str(text).strip() else ""

    async def _fallback_summary(self) -> str:
        ins = await self._insights(("confirmed", "active"), limit=8)
        if not ins:
            return ""
        return clip_words("\n".join(f"- {i['statement']}" for i in ins), SUMMARY_MAX_WORDS)

    # ------------------------------------------------------------------ prompt hook
    async def _query_vector(self, text: str) -> list[float] | None:
        mem = self.app.memory
        try:
            if mem is not None:
                return await asyncio.wait_for(mem.embed_query(text[:2000]), timeout=2.0)
            vecs = await asyncio.wait_for(self.app.llm.embed([text[:2000]]), timeout=2.0)
            return vecs[0] if vecs else None
        except Exception as exc:
            log.debug("user model relevance falls back to confidence order: %s", exc)
            return None

    async def _insight_vectors(self) -> dict[int, list[float]]:
        if self._vectors is None:
            vectors: dict[int, list[float]] = {}
            if self.app.store.vec_available and (self.vec.ready or await self.vec.exists()):
                with contextlib.suppress(Exception):
                    vectors = await self.vec.all_vectors()
            self._vectors = vectors
        return self._vectors

    async def _ranked(self, text: str, limit: int = 100) -> tuple[list[dict], list[dict]]:
        """(confirmed insights, active insights ordered by relevance to ``text`` then confidence)."""
        rows = await self._insights(("confirmed", "active"), limit=limit)
        confirmed = [i for i in rows if protected(i)]
        active = [i for i in rows if not protected(i)]
        if not active or not text.strip():
            return confirmed, active
        vectors = await self._insight_vectors()
        if not vectors:
            return confirmed, active
        qvec = await self._query_vector(text)
        if not qvec:
            return confirmed, active
        floor = self.cfg.context_min_similarity
        sims = {
            i["id"]: cosine(qvec, vectors[i["_rid"]])
            for i in active
            if i["_rid"] in vectors and len(vectors[i["_rid"]]) == len(qvec)
        }
        relevant = sorted((i for i in active if sims.get(i["id"], 0.0) >= floor), key=lambda i: -sims[i["id"]])
        rest = [i for i in active if i not in relevant and i["confidence"] >= 0.6]
        return confirmed, relevant + rest

    async def context_for(self, text: str) -> str:
        """Short markdown block for the system prompt (no model call). Empty when disabled or unknown."""
        return (await self.context_with_sources(text))[0]

    async def context_with_sources(self, text: str) -> tuple[str, list[dict]]:
        """``context_for`` plus the insights that made it into the block (public shape), for memory sources."""
        cfg = self.cfg
        if not cfg.enabled or cfg.context_max_chars <= 0:
            return "", []
        try:
            confirmed, active = await self._ranked(text or "")
        except Exception as exc:
            log.debug("user model context unavailable: %s", exc)
            return "", []
        if not confirmed and not active:
            return "", []
        header = f"## What I have learned about {self.user_name or 'the user'}\n"
        budget = cfg.context_max_chars
        lines: list[str] = []
        shown: list[dict] = []
        used = len(header)
        for ins in [*confirmed, *active]:
            line = f"- {ins['statement']}" + (" (likely)" if not protected(ins) and ins["confidence"] < 0.6 else "")
            if used + len(line) + 1 > budget:
                continue
            lines.append(line)
            shown.append(_public(ins))
            used += len(line) + 1
        return (header + "\n".join(lines), shown) if lines else ("", [])

    # ------------------------------------------------------------------ tool
    async def ask(self, question: str) -> dict:
        """Answer "what would the user want" from insights and facts with one model call."""
        if not self.cfg.enabled:
            return {"error": "The user model is turned off in settings."}
        question = " ".join(str(question or "").split())
        if not question:
            return {"error": "question is required"}
        confirmed, active = await self._ranked(question)
        insights = [i["statement"] for i in [*confirmed, *active][:12]]
        facts: list[str] = []
        if self.app.memory is not None:
            with contextlib.suppress(Exception):
                facts = [f["content"] for f in await self.app.memory.recall(question, top_k=6)]
        summary = await self.app.store.get_meta("user_model.summary") or ""
        name = self.user_name or "the user"
        if not insights and not facts and not summary:
            return {"answer": f"I do not know enough about {name} yet to say.", "insights": [], "facts": []}
        context = "\n".join(
            [
                f"Portrait:\n{summary or '(none)'}",
                "Insights:\n" + ("\n".join(f"- {s}" for s in insights) or "(none)"),
                "Facts:\n" + ("\n".join(f"- {f}" for f in facts) or "(none)"),
                f"Question: {question}",
            ]
        )
        try:
            answer = await self.app.llm.complete_text(
                self.cfg.role,
                [
                    {"role": "system", "content": prompts.USER_MODEL_ASK_SYSTEM.format(name=name)},
                    {"role": "user", "content": context},
                ],
            )
        except Exception as exc:
            return {"error": f"could not answer: {exc}", "insights": insights, "facts": facts}
        return {"answer": answer.strip(), "insights": insights, "facts": facts}
