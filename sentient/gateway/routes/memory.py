"""REST routes for memory. OWNER: MEMORY/PROACTIVITY AGENT. Contract: docs/API.md section 7."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from sentient.gateway.deps import AUTH, get_core
from sentient.memory import review as memory_review
from sentient.memory.episodic import EpisodicMemory
from sentient.memory.personas import PERSONAS, render_persona
from sentient.memory.topics import TOPICS

router = APIRouter(prefix="/api/memories", tags=["memory"], dependencies=AUTH)

IMPORT_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def _memory(request: Request):
    s = get_core(request)
    if s.memory is None:
        raise HTTPException(503, "memory is disabled (the sqlite-vec extension could not be loaded)")
    return s.memory


@router.get("")
async def list_memories(
    request: Request, topic: str = "", q: str = "", source: str = "", limit: int = 500, offset: int = 0
):
    s = get_core(request)
    if s.memory is None:
        return []
    return await s.memory.list_facts(
        max(1, min(limit, 2000)), max(0, offset), topic=topic or None, source=source or None, q=q or None
    )


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
async def import_document(request: Request, file: UploadFile = File(...)):
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
            return await mem.import_document(path, username=s.config.assistant.user_name, source=f"file:{name}")
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
    return {"deleted": await _memory(request).forget_source(source)}


@router.post("")
async def create_memory(request: Request, body: dict):
    mem = _memory(request)
    content = str(body.get("content", "")).strip()
    if not content:
        raise HTTPException(400, "content is required")
    return await mem.remember(content, source=str(body.get("source") or "manual"), notify=True)


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


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: int):
    mem = _memory(request)
    deleted = await mem.forget(memory_id, notify=True)
    if not deleted:
        raise HTTPException(404, f"memory {memory_id} not found")
    return {"deleted": True}
