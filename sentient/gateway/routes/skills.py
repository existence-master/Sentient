"""REST routes for skills and self-evolution. OWNER: MEMORY/PROACTIVITY AGENT. Contract: docs/API.md section 8."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.evolution.log import log_event, read_log
from sentient.gateway.deps import AUTH, get_core
from sentient.skills import checks as skill_checks
from sentient.skills import vet
from sentient.skills.loader import slugify, valid_name

router = APIRouter(prefix="/api/skills", tags=["skills"], dependencies=AUTH)


async def _changed(s, name: str, state: str) -> None:
    s.skills.reload(s._available_tools())  # plugin ids and tool names, as at startup
    await s.skills.sync_stats(s.store)
    s.bus.publish("skill.updated", {"name": name, "state": state})


async def _skill(s, name: str) -> dict:
    skill = s.skills.get_any(name)
    if skill is None:
        raise HTTPException(404, f"no skill named {name}")
    stats = await s.skills.stats(s.store)
    out = skill.to_dict(stats.get(skill.name), body=True)
    if skill.state == "active" and skill.dir.parent == s.skills.root:
        edit = await s.skills.edit_state(skill.name, s.store)
        out.update(user_edited=edit["user_edited"], can_reset=edit["can_reset"])
    out["lint"] = s.skills.lint(skill)
    out["vetting"] = await vet.report_for(s.store, skill.name, skill.dir)
    return out


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
    for item in catalog.get("pending") or []:  # skills from outside wait with their safety check (#146)
        item["vetting"] = await vet.report_for(s.store, item["name"], s.skills.pending_dir / item["name"])
    return catalog


@router.post("/import")
async def import_skill(request: Request, body: dict):
    """Add a skill from a web address (a SKILL.md, or a GitHub page, folder or repository that holds one). It waits in
    pending with its safety check until the user approves it."""
    s = get_core(request)
    url = str(body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "Paste the web address of a skill.")
    try:
        text, final = await vet.fetch_skill_text(url)
    except vet.SkillImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    base = slugify(vet.suggested_name(text, final))
    if not valid_name(base):
        base = "imported-skill"
    name, n = base, 1
    while s.skills.get_any(name) or (s.skills.pending_dir / name).exists():
        n += 1
        name = f"{base[:55]}-imported" if n == 2 else f"{base[:52]}-imported-{n - 1}"
    try:
        folder = vet.stage_text(s.skills.pending_dir, text, name)
    except vet.SkillImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    report = vet.vet_folder(folder, source="link", origin=url)
    await vet.record_import(s.store, name, report)
    await _changed(s, name, "pending_review")
    await log_event(s.store, "skill_imported", {"name": name, "source": "link", "origin": url,
                                                "verdict": report["verdict"], "pending": True})
    return await _skill(s, name)


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
    await s.skills.record_origin(name, s.store)  # what the user first saved is the version reset goes back to
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
    if target in (None, "active"):
        await s.skills.edit_state(name, s.store)  # make sure the version before this edit is on record for reset
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
        out = skill.to_dict((await s.skills.stats(s.store)).get(skill.name), body=True)
        return {**out, "lint": s.skills.lint(skill)}
    if skill.state == "active":
        await s.skills.record_patch(skill.name, "active", s.store)
    await _changed(s, skill.name, skill.state)
    return await _skill(s, skill.name)


@router.post("/{name}/approve")
async def approve_skill(request: Request, name: str, body: dict | None = None):
    s = get_core(request)
    had_active = s.skills.get_active_file(name) is not None
    imported = await vet.report_for(s.store, name, s.skills.pending_dir / name.strip().lower())
    if imported is not None and imported["verdict"] == "high" and (body or {}).get("accept_risk") is not True:
        # high-risk findings default to declined: approving needs an explicit "approve anyway" (#146)
        raise HTTPException(409, "The safety check found high-risk problems in this skill. Approve it anyway only if "
                                 "you trust where it came from.")
    pending = s.skills.get_pending(name)
    report = None
    if pending is not None and pending.validate:
        # the skill's own checks run before it is saved (#386); approving is the user's yes for a code check
        current = s.skills.get_active_file(name)
        folders = [pending.dir, *([current.dir] if current else [])]
        report = await skill_checks.run(s, pending.name, pending.validate, folders, run_code=True,
                                        outside=imported is not None)
        await s.skills.record_check(pending.name, report, s.store, pending=True)
        if reason := skill_checks.failure(report):
            s.bus.publish("skill.updated", {"name": pending.name, "state": "pending_review"})
            await log_event(s.store, "skill_check_failed", {"name": pending.name, "reason": reason, "by": "user"})
            raise HTTPException(409, f"The skill's check failed: {reason}. It is still waiting for your review.")
    try:
        s.skills.approve_pending(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending skill named {name}") from exc
    await s.skills.set_state(name, "active", s.store)
    await s.skills.record_origin(name, s.store)  # the accepted version is the new original
    await s.skills.record_check(name, report, s.store)  # its check report is now the active skill's
    await s.skills.record_check(name, None, s.store, pending=True)
    if imported is not None:
        await vet.mark_approved(s.store, name, s.skills.root / name.strip().lower())
    await _changed(s, name, "active")
    detail: dict = {"name": name, "approved": True}
    if imported is not None:
        detail.update(source=imported["source"], verdict=imported["verdict"])
    await log_event(s.store, "skill_patched" if had_active else "skill_created", detail)
    return await _skill(s, name)


@router.post("/{name}/reset")
async def reset_skill(request: Request, name: str):
    """Undo the user's edits: put back the version Sentient last wrote (or the one first saved)."""
    s = get_core(request)
    try:
        skill = await s.skills.reset_to_original(name, s.store)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no original version of {name} to go back to") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    await _changed(s, skill.name, "active")
    await log_event(s.store, "skill_reset", {"name": skill.name, "by": "user"})
    return await _skill(s, skill.name)


@router.post("/{name}/reject")
async def reject_skill(request: Request, name: str):
    s = get_core(request)
    imported = await vet.report_for(s.store, name, s.skills.pending_dir / name.strip().lower())
    try:
        s.skills.reject_pending(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending skill named {name}") from exc
    await s.skills.record_check(name, None, s.store, pending=True)
    if imported is not None and imported.get("decision") != "approved":
        await vet.forget(s.store, name)
        await log_event(s.store, "skill_import_declined", {"name": name, "source": imported["source"],
                                                            "verdict": imported["verdict"]})
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
    await vet.forget(s.store, name)
    await _changed(s, name, "deleted")
    return {"ok": True}


@router.get("/{name}/diff")
async def skill_diff(request: Request, name: str):
    s = get_core(request)
    try:
        out = s.skills.diff(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no pending update for {name}") from exc
    return {**out, **await s.skills.proposal_state(name, s.store)}
