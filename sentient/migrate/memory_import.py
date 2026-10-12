"""Bring memory over from ChatGPT, Claude and OpenClaw (issue #156, docs/API.md section 19).

``preview(app, kind, path)`` reads an export (``exports.read``) and says what would happen, with counts and samples,
without calling a model. ``MemoryImports.start(kind, path, parts, skip)`` imports the picked parts in the background:

- memories the other assistant kept one by one (ChatGPT's saved memories, OpenClaw's MEMORY.md and USER.md) are
  imported as they are;
- everything else (conversations, Claude's memory summaries, custom instructions, project instructions, daily notes)
  goes to the fast model in small batches (``memory.import_chunk_chars``, ``BATCH_LINES``), which lists lasting
  facts. When the reading is done, the fast model sorts each new memory into topics.

Every memory waits in the review inbox (ADR 0021) with ``from`` = the app, the line it came from, the conversation or
note it was in, that day and the export's file name. Nothing is uploaded and the export is only read.

Duplicates are skipped before saving: the same words, a fact an existing one already says in full (``covered_by``,
the #257 rule), and near-identical ones (``FactMemory.remember`` with a review note). When a new fact says more than
one that still waits from an import ("Maya lives in Pune, India" after "Maya lives in Pune"), the waiting one gets the
fuller wording instead of a second card. A conversation or note that was read is remembered (``memory_import_seen``),
so importing a newer export of the same account only reads what's new.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from typing import Any

from sentient.llm.jobs import detached
from sentient.memory import prompts
from sentient.memory import review as memory_review
from sentient.memory.facts import (
    ATTRIBUTE_FAMILIES,
    NEGATION_RE,
    PAST_RE,
    FactMemory,
    attribute_families,
    covered_by,
    keyword_tokens,
    new_details,
    same_subject,
    word_overlap,
)
from sentient.memory.schema import ensure_memory_schema
from sentient.migrate import exports
from sentient.migrate.exports import APPS, Export, ExportError, Saved, Source
from sentient.migrate.hermes import personalize
from sentient.store.db import now_iso

log = logging.getLogger(__name__)

PARTS: dict[str, list[tuple[str, str]]] = {
    "chatgpt": [("memories", "Saved memories"), ("profile", "Custom instructions"), ("conversations", "Conversations")],
    "claude": [("memories", "Claude's memory"), ("projects", "Project instructions"), ("conversations", "Conversations")],
    "openclaw": [("memories", "Long-term memory (MEMORY.md)"), ("profile", "About you (USER.md)"),
                 ("notes", "Daily notes")],
}
SAMPLES = 6  # sources with a snippet in the preview
BATCH_LINES = 8  # messages or note lines the fast model reads at once
LISTED = 500  # sources listed per part in the preview (all of them are counted)
KNOWN_LIMIT = 20_000  # existing memories compared against
PARAPHRASE_OVERLAP = 0.7  # share of the shorter fact's words the other must have to count as the same thing


def source_of(kind: str) -> str:
    return f"import:{kind}"


# ---------------------------------------------------------------------------- what Sentient already knows
_PLURAL_RE = re.compile(r"\b([A-Za-z]{3,}?)s\b")


def _singular(text: str) -> str:
    return _PLURAL_RE.sub(r"\1", text)


class _Known:
    """Memories already there (and the ones this import adds), for the #257 "already says it" check."""

    def __init__(self, user: str):
        self.user = user
        self.rows: dict[Any, dict] = {}
        self.index: dict[str, set[Any]] = {}
        self.lower: dict[str, Any] = {}

    def drop(self, rid: Any) -> None:
        row = self.rows.pop(rid, None)
        if row is not None:
            self.lower.pop(row["text"].lower(), None)

    def add(self, rid: Any, text: str, *, waiting_import: bool) -> None:
        old = self.rows.get(rid)
        if old is not None:
            self.lower.pop(old["text"].lower(), None)
        self.rows[rid] = {"id": rid, "text": text, "waiting_import": waiting_import}
        self.lower[text.lower()] = rid
        for tok in keyword_tokens(text, None):
            self.index.setdefault(tok, set()).add(rid)

    def match(self, text: str) -> tuple[str, dict | None]:
        """("same", row) when ``row`` already says it, ("fuller", row) when ``text`` says more than a waiting
        import ``row``, else ("", None)."""
        rid = self.lower.get(text.lower())
        if rid is not None:
            return "same", self.rows[rid]
        cands: set[Any] = set()
        for tok in keyword_tokens(text, None):
            cands |= self.index.get(tok, set())
        ranked = sorted((self.rows[c] for c in cands if c in self.rows), key=lambda r: word_overlap(text, r["text"]), reverse=True)[:40]
        for row in ranked:
            if self._covered(text, row["text"]):
                return "same", row
        for row in ranked:
            if row["waiting_import"] and self._covered(row["text"], text):
                return "fuller", row
        for row in ranked:
            kind = self._paraphrase(text, row["text"])
            if kind == "fuller" and not row["waiting_import"]:
                continue  # a remembered fact is never reworded by an import; the fuller one waits for review
            if kind:
                return kind, row
        return "", None

    def _covered(self, new: str, old: str) -> bool:
        """``covered_by``, also when the two only differ in plurals ("on Wednesdays" and "the Wednesday critique")."""
        return covered_by(new, old, self.user) or covered_by(_singular(new), _singular(old), self.user)

    def _paraphrase(self, new: str, old: str) -> str:
        """Small models say one thing several ways across sources ("runs a weekly design critique on Wednesdays",
        "runs the Wednesday design critique at Northwind"). "same" when ``new`` says ``old`` in other words, "fuller"
        when it also adds a name, place or number, else "". Never for different people, a negation on one side,
        names or numbers that disagree ("5 km" and "10 km"), or facts too short to judge."""
        if not same_subject(new, old, self.user) or bool(NEGATION_RE.search(new)) != bool(NEGATION_RE.search(old)):
            return ""
        name = set(keyword_tokens(self.user, None))
        shared = attribute_families(new) & attribute_families(old)
        if shared and bool(PAST_RE.search(new)) == bool(PAST_RE.search(old)):
            # the same kind of fact with the same names and places and nothing else: "is based in Bengaluru" and
            # "lives in Bengaluru" (not "moved out of Bengaluru", not "lives in Bengaluru with her cat")
            def rest(text: str) -> tuple[set[str], set[str]]:
                details = {d.lower() for d in new_details(text, "")}
                for fam in shared:
                    text = ATTRIBUTE_FAMILIES[fam].sub(" ", text)
                return details, set(keyword_tokens(_singular(text), None)) - name - details

            (dn, wn), (do, wo) = rest(new), rest(old)
            if dn and dn == do and wn == wo:
                return "same"
        a = set(keyword_tokens(_singular(new), None)) - name
        b = set(keyword_tokens(_singular(old), None)) - name
        if min(len(a), len(b)) < 3 or len(a & b) / min(len(a), len(b)) < PARAPHRASE_OVERLAP:
            return ""
        added, missing = new_details(_singular(new), _singular(old)), new_details(_singular(old), _singular(new))
        if added and missing:
            return ""
        return "fuller" if added else "same"


async def _known_facts(app: Any, user: str) -> _Known:
    known = _Known(user)
    if app.memory is None:
        return known
    await ensure_memory_schema(app.store)
    rows = await app.store.fetchall(
        "SELECT id, content, status, source FROM facts WHERE status IN ('active', 'pending')"
        " AND (expires_at IS NULL OR expires_at > ?) ORDER BY id DESC LIMIT ?",
        (now_iso(), KNOWN_LIMIT),
    )
    for r in rows:
        known.add(int(r["id"]), r["content"],
                  waiting_import=r["status"] == "pending" and str(r["source"] or "").startswith("import:"))
    return known


async def _known_insights(app: Any, user: str) -> _Known:
    known = _Known(user)
    await ensure_memory_schema(app.store)
    for r in await app.store.fetchall(
        "SELECT id, statement FROM user_insights WHERE status != 'retired' ORDER BY rid DESC LIMIT ?", (KNOWN_LIMIT,)
    ):
        known.add(str(r["id"]), r["statement"], waiting_import=False)
    return known


async def _seen(app: Any, kind: str) -> set[str]:
    await ensure_memory_schema(app.store)
    rows = await app.store.fetchall("SELECT item FROM memory_import_seen WHERE app = ?", (kind,))
    return {r["item"] for r in rows}


# ChatGPT's saved memories often leave out the subject: "Is allergic to cashews.", "Works at Northwind Studio."
_VERB_START = frozenset({
    "is", "was", "has", "had", "works", "worked", "lives", "lived", "prefers", "likes", "loves", "wants", "enjoys",
    "plans", "owns", "uses", "dislikes", "hates", "does", "runs", "studies", "studied", "speaks", "goes", "drinks",
    "eats", "avoids", "needs", "believes", "grew", "moved", "started", "attends", "plays", "reads", "writes",
    "travels", "visits", "knows", "learned", "learns", "values", "feels", "keeps", "follows", "practices", "designs",
    "teaches", "manages", "leads", "volunteers", "cooks", "wears", "takes", "can't", "cannot", "doesn't", "does not",
})


def with_subject(text: str, user: str) -> str:
    """ "Is allergic to cashews." -> "Maya is allergic to cashews." (unchanged when it already names someone)."""
    text = " ".join(text.split())
    first = text.split(" ", 1)[0].lower().rstrip(",")
    if user and first in _VERB_START:
        return f"{user} {text[:1].lower()}{text[1:]}"
    return text


def _clean_fact(text: str, user: str) -> str | None:
    return FactMemory.clean_fact(personalize(with_subject(text, user), user), user)


# ---------------------------------------------------------------------------- batches
def batches(sources: list[Source], budget: int) -> list[list[tuple[Source, str]]]:
    """The texts of ``sources`` in model-sized batches: at most ``budget`` characters and ``BATCH_LINES`` lines, one
    part (conversations, notes...) per batch. Small models miss facts when a batch mixes many kinds of text."""
    out: list[list[tuple[Source, str]]] = []
    cur: list[tuple[Source, str]] = []
    size = 0
    for src in sources:
        for text in src.texts:
            if cur and (size + len(text) > budget or len(cur) >= BATCH_LINES or cur[-1][0].part != src.part):
                out.append(cur)
                cur, size = [], 0
            cur.append((src, text))
            size += len(text) + 1
    if cur:
        out.append(cur)
    return out


def _where(src: Source) -> str:
    kind = "Conversation" if src.part == "conversations" else "From"
    when = f" ({src.dated})" if src.dated else ""
    return f'{kind} "{src.title}"{when}:'


def render(batch: list[tuple[Source, str]]) -> str:
    lines: list[str] = []
    last: Source | None = None
    for src, text in batch:
        if src is not last:
            if lines:
                lines.append("")
            lines.append(_where(src))
            last = src
        lines.append(f"- {text}")
    return "\n".join(lines)


def _attribute(fact: str, batch: list[tuple[Source, str]], user: str) -> tuple[Source, str] | None:
    """The line ``fact`` most likely came from, or None when it shares no word with any line (made up, or copied
    from the prompt's examples)."""
    name = set(keyword_tokens(user, None))

    def score(text: str) -> float:
        a, b = set(keyword_tokens(fact, None)) - name, set(keyword_tokens(text, None)) - name
        return len(a & b) / len(a | b) if a and b else 0.0

    best = max(batch, key=lambda pair: score(pair[1]))
    return best if score(best[1]) > 0 else None


# ---------------------------------------------------------------------------- preview
def _saved_item(s: Saved, action: str, note: str, text: str) -> dict:
    return {"key": s.key, "action": action, "note": note, "kind": s.kind, "text": text, "title": s.title,
            "dated": s.dated}


async def _plan(app: Any, export: Export) -> dict:
    user = app.config.assistant.user_name.strip()
    facts = await _known_facts(app, user)
    insights = await _known_insights(app, user)
    seen = await _seen(app, export.app)
    budget = app.config.memory.import_chunk_chars
    in_file: set[str] = set()
    parts: list[dict] = []
    for pid, label in PARTS[export.app]:
        items: list[dict] = []
        for s in (x for x in export.saved if x.part == pid):
            text = _clean_fact(s.text, user) if s.kind == "fact" else personalize(s.text, user)
            where = "a memory" if s.kind == "fact" else "something Sentient knows about you"
            if not text:
                items.append(_saved_item(s, "skip", "Not something to remember.", s.text))
            elif text.lower() in in_file:
                items.append(_saved_item(s, "skip", "It's in there twice.", text))
            elif facts.match(text)[0] == "same" or insights.match(text)[0] == "same":
                items.append(_saved_item(s, "skip", "Sentient already knows this.", text))
            else:
                in_file.add(text.lower())
                items.append(_saved_item(s, "import", f"Becomes {where} once you approve it in Memory > Review.", text))
        sources: list[dict] = []
        todo: list[Source] = []
        for src in (x for x in export.sources if x.part == pid):
            read_before = src.key in seen
            if not read_before:
                todo.append(src)
            sources.append({
                "key": src.key, "title": src.title, "dated": src.dated, "messages": len(src.texts),
                "action": "skip" if read_before else "import",
                "note": "Read in an earlier import." if read_before else "Sentient reads it for facts about you.",
            })
        sources.sort(key=lambda x: x["dated"] or "", reverse=True)
        for row in sources[:SAMPLES]:
            src = next(s for s in export.sources if s.key == row["key"])
            row["snippet"] = exports.clip(src.texts[0], 160)
        n_items = sum(1 for i in items if i["action"] == "import")
        parts.append({
            "id": pid, "label": label, "count": n_items + len(todo), "items": items,
            "sources": sources[:LISTED], "sources_total": len(sources), "batches": len(batches(todo, budget)),
            "messages": sum(len(s.texts) for s in todo),
        })
    counts = {p["id"]: p["count"] for p in parts}
    return {
        "app": export.app, "label": export.label, "path": str(export.path), "export": export.name,
        "counts": counts, "batches": sum(p["batches"] for p in parts), "parts": parts, "stats": export.stats,
        "never_read": list(exports.NEVER_READ),
    }


async def _read(app: Any, kind: str, path: str | None) -> Export:
    if kind not in APPS:
        raise ExportError("Pick ChatGPT, Claude or OpenClaw.")
    if app.memory is None:
        raise ExportError("Memory is turned off on this computer, so there's nowhere to put imported memories.")
    return await asyncio.to_thread(exports.read, kind, path, user_name=app.config.assistant.user_name)


async def preview(app: Any, kind: str, path: str | None) -> dict:
    return await _plan(app, await _read(app, kind, path))


async def remove(app: Any, kind: str) -> dict:
    """Undo: every fact and insight imported from ``kind``, approved or not, and what was read (so it can be
    imported again)."""
    if kind not in APPS:
        raise ExportError("Pick ChatGPT, Claude or OpenClaw.")
    source = source_of(kind)
    facts = await app.memory.forget_source(source) if app.memory is not None else 0
    insights = await app.user_model.delete_by_source(source)
    await ensure_memory_schema(app.store)
    await app.store.execute("DELETE FROM memory_import_seen WHERE app = ?", (kind,))
    return {"facts": facts, "insights": insights}


# ---------------------------------------------------------------------------- the import itself
class ImportBusy(RuntimeError):
    """Another import is still running."""


class MemoryImports:
    """One import at a time, in the background; ``current()`` is its progress (``import.updated`` events too)."""

    def __init__(self, app: Any):
        self.app = app
        self.job: dict | None = None
        self._task: asyncio.Task | None = None
        self._cancel = False
        self._starting = False  # claimed before start() awaits anything, so two clicks can't start two imports
        self._added: set[int] = set()  # facts this job added (an insight may take one's place)

    def current(self) -> dict | None:
        return dict(self.job) if self.job else None

    def running(self) -> bool:
        return self._starting or (self._task is not None and not self._task.done())

    def cancel(self) -> dict | None:
        if self.running():
            self._cancel = True
            if self.job is not None:
                self.job["message"] = "Stopping after this batch."
                self._publish()
        return self.current()

    async def stop(self) -> None:
        await self.halt()

    async def halt(self) -> int:
        """Stop a running import now (Stop everything, shutdown): what it saved stays. Returns 1 when one ran."""
        if self._task is None or self._task.done():
            return 0
        self._cancel = True
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        return 1

    def _publish(self) -> None:
        if self.job is not None:
            self.app.bus.publish("import.updated", dict(self.job))

    async def start(self, kind: str, path: str | None, parts: list[str] | None, skip: list[str] | None = None) -> dict:
        if self.running():
            raise ImportBusy("An import is already running. Wait for it to finish or stop it first.")
        if getattr(self.app, "stopped", False):
            raise ImportBusy("Sentient is stopped. Resume it first, then import.")
        valid = {p for p, _ in PARTS.get(kind, [])}
        chosen = [p for p in (parts or []) if p in valid]
        if not chosen:
            raise ExportError("Pick at least one thing to import.")
        self._starting = True
        try:
            return await self._start(kind, path, chosen, skip)
        finally:
            self._starting = False

    async def _start(self, kind: str, path: str | None, chosen: list[str], skip: list[str] | None) -> dict:
        export = await _read(self.app, kind, path)
        plan = await _plan(self.app, export)
        left_out = set(skip or [])
        saved = [s for s in export.saved if s.part in chosen]
        planned = {i["key"]: i for p in plan["parts"] for i in p["items"]}
        saved = [s for s in saved if planned.get(s.key, {}).get("action") == "import" and s.key not in left_out]
        seen = await _seen(self.app, kind)
        sources = [s for s in export.sources if s.part in chosen and s.key not in seen and s.key not in left_out]
        # part by part, newest first: the most current facts land first
        order = [p for p, _ in PARTS[kind]]
        sources.sort(key=lambda s: s.dated or "", reverse=True)
        sources.sort(key=lambda s: order.index(s.part))
        work = batches(sources, self.app.config.memory.import_chunk_chars)
        backups = getattr(self.app, "backups", None)
        if backups is not None:  # an import can be undone by restoring this snapshot (issue #151)
            await backups.snapshot(f"before importing from {export.label}")
        self._cancel = False
        self._added = set()
        self.job = {
            "id": uuid.uuid4().hex[:12], "app": kind, "label": export.label, "export": export.name,
            "path": str(export.path), "parts": chosen, "status": "running", "total": len(work), "done": 0,
            "facts": 0, "insights": 0, "merged": 0, "skipped": 0, "failed": 0, "started_at": now_iso(),
            "finished_at": None, "message": "Starting.",
        }
        self._publish()
        self._task = asyncio.create_task(
            self._run(export, saved, work), name=f"import:{kind}", context=detached("memory", requested=True)
        )
        return self.current() or {}

    async def _run(self, export: Export, saved: list[Saved], work: list[list[tuple[Source, str]]]) -> None:
        job = self.job
        assert job is not None
        app = self.app
        user = app.config.assistant.user_name.strip()
        source = source_of(export.app)
        facts = await _known_facts(app, user)
        insights = await _known_insights(app, user)
        added_before = job["facts"]
        try:
            for s in saved:
                review = memory_review.note(export.label, s.text, title=s.title, dated=s.dated, export=export.name)
                if s.kind == "fact":
                    await self._keep_fact(s.text, review, facts, insights, source)
                else:
                    await self._keep_insight(s, review, insights, facts, source)
            if saved:
                job["message"] = f"Saved {job['facts'] + job['insights']} memories to review."
                self._publish()
            failed: set[str] = set()
            last_of = {}
            for n, batch in enumerate(work):
                for src, _ in batch:
                    last_of[src.key] = n
            for n, batch in enumerate(work):
                if self._cancel:
                    break
                job["message"] = f"Reading {batch[0][0].title}."
                self._publish()
                try:
                    found = await self._extract(export, batch, user)
                except Exception as exc:
                    log.warning("import batch %s from %s failed: %s", n, export.label, exc)
                    job["failed"] += 1
                    failed.update(src.key for src, _ in batch)
                    found = []
                log.info("import from %s: batch %d of %d gave %d facts", export.label, n + 1, len(work), len(found))
                for fact in found:
                    where = _attribute(fact, batch, user)
                    if where is None:
                        log.info("import: dropped %r (no line of the batch says it)", fact)
                        job["skipped"] += 1
                        continue
                    src, line = where
                    review = memory_review.note(export.label, line, title=src.title, dated=src.dated,
                                                export=export.name)
                    await self._keep_fact(fact, review, facts, insights, source)
                done = {src.key for src, _ in batch if last_of[src.key] == n and src.key not in failed}
                await self._mark_seen(export.app, done)
                job["done"] = n + 1
                self._publish()
            await self._sort_topics(job)
            job["status"] = "cancelled" if self._cancel and job["done"] < job["total"] else "done"
            job["message"] = _summary(job)
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["message"] = _summary(job)
            raise
        except Exception as exc:
            log.exception("memory import from %s failed", export.label)
            job["status"] = "error"
            job["message"] = f"The import stopped because of an error: {exc}"
        finally:
            job["finished_at"] = now_iso()
            new = job["facts"] - added_before
            if new and app.memory is not None:
                app.memory.publish("ADD", None, None, source=source, count=new, status="pending")
            self._publish()

    async def _sort_topics(self, job: dict) -> None:
        """Topics and long/short-term for what this import saved (held facts are saved without a model call so a
        big export stays quick to read; the fast model sorts them once the reading is done)."""
        todo = sorted(self._added)
        if not todo or self._cancel:
            return
        job["message"] = f"Sorting {len(todo)} memories into topics."
        self._publish()
        for fid in todo:
            if self._cancel:
                return
            try:
                await self.app.memory.classify_pending(fid)
            except Exception as exc:
                log.warning("could not sort imported memory %s into topics: %s", fid, exc)

    async def _extract(self, export: Export, batch: list[tuple[Source, str]], user: str) -> list[str]:
        name = user or "the user"
        raw = await self.app.llm.complete_json(
            "fast",
            [
                {"role": "system", "content": prompts.extraction_system(user)},
                {"role": "user", "content": prompts.IMPORT_USER.format(username=name, app=export.label,
                                                                       sections=render(batch))},
            ],
        )
        return self.app.memory._facts_from(raw, user)[:40]

    async def _keep_fact(self, text: str, review: dict, known: _Known, insights: _Known, source: str) -> None:
        job = self.job
        assert job is not None
        user = known.user
        clean = _clean_fact(text, user)
        if not clean:
            log.info("import: dropped %r (not something to remember)", text)
            job["skipped"] += 1
            return
        review = _without_echo(review, clean)
        kind, row = known.match(clean)
        if kind == "same" and row is not None:
            log.info("import: skipped %r (memory %s already says it: %r)", clean, row["id"], row["text"])
            job["skipped"] += 1
            return
        about, row_about = insights.match(clean)
        if about == "same" and row_about is not None:
            log.info("import: skipped %r (what Sentient knows about you already says it: %r)", clean, row_about["text"])
            job["skipped"] += 1
            return
        if kind == "fuller" and row is not None and await self.app.memory.reword_pending(row["id"], clean, review):
            log.info("import: %r fills in waiting memory %s (%r)", clean, row["id"], row["text"])
            known.add(row["id"], clean, waiting_import=True)
            self._added.add(int(row["id"]))
            job["merged"] += 1
            return
        result = await self.app.memory.remember(clean, source=source, use_llm=False, review=review)
        if result["action"] == "ADD":
            log.info("import: held %r for review (memory %s)", clean, result["id"])
            known.add(result["id"], clean, waiting_import=True)
            self._added.add(result["id"])
            job["facts"] += 1
        else:
            log.info("import: skipped %r (near-identical to memory %s)", clean, result["id"])
            job["skipped"] += 1

    async def _keep_insight(self, s: Saved, review: dict, known: _Known, facts: _Known, source: str) -> None:
        job = self.job
        assert job is not None
        text = personalize(s.text, known.user)
        for other in (known, facts):
            kind, row = other.match(text)
            if kind == "same" and row is not None:
                log.info("import: skipped %r (already known: %r)", text, row["text"])
                job["skipped"] += 1
                return
        made = await self.app.user_model.import_insight(
            text, source=source, dimension=s.dimension, review=_without_echo(review, text)
        )
        if made is None:
            job["skipped"] += 1
            return
        known.add(made["id"], made["statement"], waiting_import=False)
        job["insights"] += 1
        kind, row = facts.match(made["statement"])
        if kind == "fuller" and row is not None and row["waiting_import"]:
            # an imported memory that this insight says in full waits as one card, not two
            pending = await self.app.store.fetchone(
                "SELECT id FROM facts WHERE id = ? AND status = 'pending'", (row["id"],)
            )
            if pending is not None and await self.app.memory.forget(int(row["id"])):
                log.info("import: %r (about you) replaces waiting memory %s (%r)", made["statement"], row["id"], row["text"])
                facts.drop(row["id"])
                job["facts"] -= 1 if row["id"] in self._added else 0
                self._added.discard(row["id"])
                job["merged"] += 1

    async def _mark_seen(self, kind: str, keys: set[str]) -> None:
        if not keys:
            return
        ts = now_iso()
        for key in sorted(keys):
            await self.app.store.execute(
                "INSERT OR IGNORE INTO memory_import_seen(app, item, imported_at) VALUES(?,?,?)", (kind, key, ts)
            )


def _without_echo(review: dict, text: str) -> dict:
    """No "It came from" line on the review card when it would only repeat the memory word for word."""

    def norm(t: str) -> str:
        return " ".join(str(t or "").lower().split()).rstrip(".")

    return {**review, "snippet": None} if norm(review.get("snippet") or "") == norm(text) else review


def _summary(job: dict) -> str:
    n = job["facts"] + job["insights"]
    head = {"done": "Done.", "cancelled": "Stopped."}.get(job["status"], "")
    found = f"{n} {'memory waits' if n == 1 else 'memories wait'} for your review in Memory > Review."
    extra = []
    if job["merged"]:
        extra.append(f"{job['merged']} filled in one already waiting")
    if job["skipped"]:
        extra.append(f"{job['skipped']} Sentient already knew or couldn't use")
    if job["failed"]:
        extra.append(f"{job['failed']} {'batch' if job['failed'] == 1 else 'batches'} couldn't be read")
    tail = f" ({'; '.join(extra)}.)" if extra else ""
    return f"{head} {found}{tail}".strip()

