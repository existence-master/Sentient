"""REST routes for memory. OWNER: MEMORY/PROACTIVITY AGENT. Contract: docs/API.md section 7."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from sentient.gateway.deps import AUTH, get_core
from sentient.memory import export as memory_export
from sentient.memory import forget as memory_forget
from sentient.memory import review as memory_review
from sentient.memory import scopes as memory_scopes
from sentient.memory.episodic import EpisodicMemory
from sentient.memory.personas import PERSONAS, render_persona
from sentient.memory.topics import TOPICS

router = APIRouter(prefix="/api/memories", tags=["memory"], dependencies=AUTH)

IMPORT_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}
PROFILE_SCOPES = (memory_scopes.PERSONAL, memory_scopes.ALWAYS)  # what USER.md and MEMORY.md are built from


def _memory(request: Request):
    s = get_core(request)
    if s.memory is None:
        raise HTTPException(503, "memory is disabled (the sqlite-vec extension could not be loaded)")
    return s.memory


@router.get("")
async def list_memories(
    request: Request, topic: str = "", q: str = "", source: str = "", limit: int = 500, offset: int = 0,
    scope: str = "",
):
    s = get_core(request)
    if s.memory is None:
        return []
    return await s.memory.list_facts(
        max(1, min(limit, 2000)), max(0, offset), topic=topic or None, source=source or None, q=q or None,
        scope=scope or None,
    )


@router.get("/scopes")
async def scopes(request: Request):
    """Always, Personal and Work, then every project in use, each with how many memories it holds (#155)."""
    return await memory_scopes.list_scopes(get_core(request))


@router.delete("/scope/{scope:path}")
async def reset_scope(request: Request, scope: str):
    """Forget every memory in one scope, and nothing else (#155)."""
    target = memory_scopes.normalize(scope)
    if not target:
        raise HTTPException(400, "scope is required")
    mem = _memory(request)
    rows = await get_core(request).store.fetchall("SELECT content FROM facts WHERE scope = ?", (target,))
    gone = [r["content"] for r in rows]  # all of them, not one page
    deleted = await mem.forget_scope(target)
    if target in PROFILE_SCOPES:
        get_core(request).workspace.remove_learned(gone)
    return {"deleted": deleted}


@router.get("/topics")
async def topics(request: Request):
    s = get_core(request)
    if s.memory is None:
        return [{**t, "count": 0} for t in TOPICS]
    return await s.memory.topic_counts()


@router.get("/graph")
async def graph(request: Request):
    s = get_core(request)
    if s.memory is None:
        return {"nodes": [], "links": []}
    return await s.memory.graph()


@router.get("/summaries")
async def summaries(request: Request, limit: int = 50):
    s = get_core(request)
    episodic = s.memory.episodic if s.memory is not None else EpisodicMemory(s.store, s.llm, s.config)
    return await episodic.list(max(1, min(limit, 500)))


@router.get("/dreams")
async def dreams(request: Request, limit: int = 20):
    return await get_core(request).dreaming.list(limit)


@router.post("/dreams/run")
async def run_dream(request: Request):
    return await get_core(request).dreaming.start_run("manual")


@router.get("/dreams/{dream_id}")
async def dream(request: Request, dream_id: str):
    found = await get_core(request).dreaming.get(dream_id)
    if found is None:
        raise HTTPException(404, f"dream {dream_id} not found")
    return found


@router.get("/workspace")
async def workspace(request: Request):
    return get_core(request).workspace.read_full()


@router.put("/workspace/{which}")
async def write_workspace(request: Request, which: str, body: dict):
    if which not in {"soul", "user", "memory"}:
        raise HTTPException(400, "which must be soul|user|memory")
    get_core(request).workspace.write(which, str(body.get("content", "")))
    return {"saved": True}


@router.get("/personas")
async def personas(request: Request):
    cfg = get_core(request).config.assistant
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "soul_md": render_persona(p["id"], name=cfg.name, user=cfg.user_name),
        }
        for p in PERSONAS
    ]


@router.post("/import")
async def import_document(request: Request, file: UploadFile = File(...), scope: str = ""):
    s = get_core(request)
    mem = _memory(request)
    name = Path(file.filename or "document.txt").name
    suffix = Path(name).suffix.lower()
    if suffix not in IMPORT_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix or '(none)'}; use pdf, txt, md or docx")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        with path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            return await mem.import_document(
                path, username=s.config.assistant.user_name, source=f"file:{name}", scope=scope or None
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


@router.post("/export")
async def export_memory(request: Request, body: dict):
    """Everything Sentient remembers as a Markdown or SQLite file in ``<home>/files/exports`` (issue #156)."""
    try:
        return await memory_export.export(get_core(request), str(body.get("format") or "markdown"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/review")
async def review_inbox(request: Request):
    return await memory_review.inbox(get_core(request))


@router.post("/review/approve-all")
async def review_approve_all(request: Request, body: dict):
    source = str(body.get("from") or "").strip()
    if not source:
        raise HTTPException(400, "from is required")
    return {"approved": await memory_review.approve_from(get_core(request), source)}


@router.post("/review/merged/{item_id}/undo")
async def review_undo_merge(request: Request, item_id: str):
    try:
        memory = await memory_review.undo_merge(get_core(request), item_id)
    except KeyError:
        memory = None
    if memory is None:
        raise HTTPException(404, f"memory {item_id} was not merged")
    return {"ok": True, "memory": memory}


@router.post("/review/{kind}/{item_id}/approve")
async def review_approve(request: Request, kind: str, item_id: str, body: dict | None = None):
    content = (body or {}).get("content")
    if content is not None and not str(content).strip():
        raise HTTPException(400, "content is empty")
    try:
        ok = await memory_review.approve(get_core(request), kind, item_id, str(content) if content else None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError:
        ok = False
    if not ok:
        raise HTTPException(404, f"no {kind} {item_id} is waiting for review")
    return {"ok": True}


@router.delete("/review/{kind}/{item_id}")
async def review_discard(request: Request, kind: str, item_id: str):
    try:
        ok = await memory_review.discard(get_core(request), kind, item_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError:
        ok = False
    if not ok:
        raise HTTPException(404, f"no {kind} {item_id} is waiting for review")
    return {"ok": True}


@router.delete("/source/{source:path}")
async def delete_by_source(request: Request, source: str):
    """Forget the memories from ``source``. One another source backs too only loses this source (#153)."""
    mem = _memory(request)
    done = await mem.purge_source(source)
    if done["deleted"] or done["unlinked"]:
        mem.publish("DELETE", None, None, source=source, count=done["deleted"], unlinked=done["unlinked"])
    return done


@router.get("/forget-preview")
async def forget_preview(request: Request, source: str = ""):
    """What "Also forget what Sentient learned from it" would delete for ``source`` (``app:gmail``,
    ``channel:telegram``, ``device:<id>``, ``folder:<id>``), before the user disconnects it (#153)."""
    try:
        return await memory_forget.preview(get_core(request), source)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("")
async def create_memory(request: Request, body: dict):
    mem = _memory(request)
    content = str(body.get("content", "")).strip()
    if not content:
        raise HTTPException(400, "content is required")
    scope = memory_scopes.normalize(body.get("scope"), memory_scopes.PERSONAL)  # #155
    return await mem.remember(content, source=str(body.get("source") or "manual"), notify=True, scope=scope)


@router.put("/{memory_id}")
async def update_memory(request: Request, memory_id: int, body: dict):
    mem = _memory(request)
    content = str(body.get("content", "")).strip()
    if not content:
        raise HTTPException(400, "content is required")
    try:
        return await mem.update_content(memory_id, content)
    except KeyError as exc:
        raise HTTPException(404, f"memory {memory_id} not found") from exc


@router.patch("/{memory_id}")
async def move_memory(request: Request, memory_id: int, body: dict):
    """Move a memory to another scope (#155): ``{scope}``."""
    mem = _memory(request)
    scope = memory_scopes.normalize(body.get("scope"))
    if not scope:
        raise HTTPException(400, "scope is required")
    try:
        before = await mem.scope_of(memory_id)
        moved = await mem.set_scope(memory_id, scope)
    except KeyError as exc:
        raise HTTPException(404, f"memory {memory_id} not found") from exc
    if before in PROFILE_SCOPES and moved["scope"] not in PROFILE_SCOPES:
        # USER.md's learned facts are personal: one moved to work or a project leaves it. MEMORY.md is rewritten
        # from personal memories at its next upkeep.
        get_core(request).workspace.remove_learned([moved["content"]])
    return moved


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: int):
    mem = _memory(request)
    deleted = await mem.forget(memory_id, notify=True)
    if not deleted:
        raise HTTPException(404, f"memory {memory_id} not found")
    return {"deleted": True}
