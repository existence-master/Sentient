"""REST routes for skills and self-evolution. OWNER: MEMORY/PROACTIVITY AGENT. Contract: docs/API.md section 8."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.evolution.log import log_event, read_log
from sentient.gateway.deps import AUTH, get_core
from sentient.skills.loader import valid_name

router = APIRouter(prefix="/api/skills", tags=["skills"], dependencies=AUTH)


async def _changed(s, name: str, state: str) -> None:
    s.skills.reload({p.id for p in s.registry.plugins()})
    await s.skills.sync_stats(s.store)
    s.bus.publish("skill.updated", {"name": name, "state": state})


async def _skill(s, name: str) -> dict:
    skill = s.skills.get_any(name)
    if skill is None:
        raise HTTPException(404, f"no skill named {name}")
    stats = await s.skills.stats(s.store)
    return skill.to_dict(stats.get(skill.name), body=True)


@router.get("")
async def list_skills(request: Request):
    s = get_core(request)
    await s.skills.sync_stats(s.store)
    catalog = await s.skills.catalog(s.store)
    if catalog.get("pending"):
        # attach why each proposal was made (latest matching evolution-log entry)
        origin_keys = ("session_id", "task_id", "run_id", "curator", "merged_from")
        latest: dict[str, dict] = {}
        for entry in await read_log(s.store, 500):
            detail = entry.get("detail") or {}
            name = detail.get("name")
            if name and detail.get("pending") and name not in latest:
                latest[name] = {**detail, "ts": entry["ts"]}
        for item in catalog["pending"]:
            d = latest.get(item["name"])
            item["reason"] = (d or {}).get("reason")
            origin = {k: d[k] for k in origin_keys if d and d.get(k) is not None}
            if d and d.get("origin") == "repair":
                origin["repair"] = True
            item["origin"] = origin or None
            item["proposed_at"] = (d or {}).get("ts")
    return catalog


@router.get("/evolution-log")
async def evolution_log(request: Request, limit: int = 100):
    return await read_log(get_core(request).store, max(1, min(limit, 1000)))


@router.post("/review-now")
async def review_now(request: Request, body: dict | None = None):
    s = get_core(request)
    session_id = (body or {}).get("session_id") or None
    return await s.evolution.review_now(session_id)


@router.post("")
async def create_skill(request: Request, body: dict):
    s = get_core(request)
    name = str(body.get("name", "")).strip().lower()
    if not valid_name(name):
        raise HTTPException(400, "name must be lowercase letters, digits and dashes (2-64 chars)")
    if s.skills.get_active_file(name) or s.skills.get_pending(name) or s.skills.get_archived(name):
        raise HTTPException(409, f"a skill named {name} already exists")
    s.skills.write(
        name, str(body.get("description", "")), str(body.get("body", "")), author="user",
        tags=list(body.get("tags") or []), requires_tools=list(body.get("requires_tools") or []),
    )
    await _changed(s, name, "active")
    await log_event(s.store, "skill_created", {"name": name, "author": "user"})
    return await _skill(s, name)


@router.get("/{name}")
async def get_skill(request: Request, name: str):
    return await _skill(get_core(request), name)


@router.put("/{name}")
async def update_skill(request: Request, name: str, body: dict):
    s = get_core(request)
    target = request.query_params.get("target") or body.get("target")
    try:
        skill = s.skills.edit(
            name,
            description=body.get("description"),
            body=body.get("body"),
            tags=list(body["tags"]) if body.get("tags") is not None else None,
            requires_tools=list(body["requires_tools"]) if body.get("requires_tools") is not None else None,
            target=target,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no {target + ' ' if target else ''}skill named {name}") from exc
    if target == "pending":
        # the proposal changed but nothing is active yet: return the proposal itself
        await _changed(s, skill.name, "pending_review")
        return skill.to_dict((await s.skills.stats(s.store)).get(skill.name), body=True)
    if skill.state == "active":
        await s.skills.record_patch(skill.name, "active", s.store)
    await _changed(s, skill.name, skill.state)
    return await _skill(s, skill.name)


@router.post("/{name}/approve")
async def approve_skill(request: Request, name: str):
    s = get_core(request)
    had_active = s.skills.get_active_file(name) is not None
    try:
        s.skills.approve_pending(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending skill named {name}") from exc
    await s.skills.set_state(name, "active", s.store)
    await _changed(s, name, "active")
    await log_event(s.store, "skill_patched" if had_active else "skill_created", {"name": name, "approved": True})
    return await _skill(s, name)


@router.post("/{name}/reject")
async def reject_skill(request: Request, name: str):
    s = get_core(request)
    try:
        s.skills.reject_pending(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending skill named {name}") from exc
    state = "active" if s.skills.get_active_file(name) else "rejected"
    await _changed(s, name, state)
    return {"ok": True}


@router.post("/{name}/archive")
async def archive_skill(request: Request, name: str):
    s = get_core(request)
    try:
        s.skills.archive(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no active skill named {name}") from exc
    await s.skills.set_state(name, "archived", s.store)
    await _changed(s, name, "archived")
    await log_event(s.store, "skill_archived", {"name": name, "by": "user"})
    return await _skill(s, name)


@router.post("/{name}/restore")
async def restore_skill(request: Request, name: str):
    s = get_core(request)
    try:
        s.skills.restore(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no archived skill named {name}") from exc
    except FileExistsError as exc:
        raise HTTPException(409, f"an active skill named {name} already exists") from exc
    await s.skills.set_state(name, "active", s.store)
    await _changed(s, name, "active")
    return await _skill(s, name)


@router.delete("/{name}")
async def delete_skill(request: Request, name: str):
    s = get_core(request)
    if not s.skills.delete(name):
        raise HTTPException(404, f"no skill named {name}")
    await _changed(s, name, "deleted")
    return {"ok": True}


@router.get("/{name}/diff")
async def skill_diff(request: Request, name: str):
    s = get_core(request)
    try:
        return s.skills.diff(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending update for {name}") from exc
