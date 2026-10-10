"""Core REST routes. OWNER: core. Contract: docs/API.md sections 2 and 3.

health, bootstrap, stop everything, onboarding, config, sessions, chat (NDJSON),
approvals, files, tools catalog, usage.
"""

from __future__ import annotations

import json
import mimetypes
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from sentient import __version__, paths
from sentient.config import config_json_schema
from sentient.config.schema import SentientConfig
from sentient.gateway.auth import require_token
from sentient.gateway.deps import get_core
from sentient.memory.personas import render_persona

open_router = APIRouter(tags=["core"])
router = APIRouter(tags=["core"])
_auth = [Depends(require_token)]

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.get("/api/health")
async def health():
    return {"ok": True, "version": __version__, "name": "sentient"}


# ----------------------------------------------------------------------------- bootstrap / onboarding
@router.get("/api/bootstrap", dependencies=_auth)
async def bootstrap(request: Request):
    s = get_core(request)
    cfg = s.config
    return {
        "version": __version__,
        "home": str(paths.home()),
        "assistant": cfg.assistant.model_dump(),
        "models": cfg.models.roles.model_dump(),
        "memory_enabled": s.memory is not None,
        "unread_notifications": await s.notifications.unread_count(),
        "ui": cfg.ui.model_dump(),
        "features": {"voice": True, "proactivity": cfg.proactivity.enabled},
        "stop": dict(s.stop_state),
    }


@router.get("/api/system/hardware", dependencies=_auth)
async def system_hardware(request: Request, refresh: bool = False):
    """Memory, graphics cards and the local model that fits this computer (#131). Detected once, then cached."""
    return await get_core(request).hardware.get(refresh=refresh)


# ----------------------------------------------------------------------------- stop everything (section 17)
class StopBody(BaseModel):
    source: str = "desktop"


def _source(body: StopBody | None) -> str:
    return re.sub(r"[^a-z0-9_.-]", "", (body.source if body else "desktop").lower())[:30] or "desktop"


@router.get("/api/stop", dependencies=_auth)
async def stop_state(request: Request):
    return dict(get_core(request).stop_state)


@router.post("/api/stop-all", dependencies=_auth)
async def stop_all(request: Request, body: StopBody | None = None):
    return await get_core(request).stop_all(source=_source(body))


@router.post("/api/resume", dependencies=_auth)
async def resume(request: Request, body: StopBody | None = None):
    return await get_core(request).resume(source=_source(body))


class OnboardingBody(BaseModel):
    user_name: str
    assistant_name: str = "Sentient"
    timezone: str = "auto"
    location: str = ""
    professional_context: str = ""
    personal_context: str = ""
    persona: str = "friendly"
    daily_brief: bool = False


@router.post("/api/onboarding", dependencies=_auth)
async def onboarding(request: Request, body: OnboardingBody):
    s = get_core(request)
    cfg = s.config.model_copy(deep=True)
    cfg.assistant.user_name = body.user_name.strip()
    cfg.assistant.name = body.assistant_name.strip() or "Sentient"
    cfg.assistant.timezone = body.timezone or "auto"
    cfg.assistant.location = body.location.strip()
    cfg.assistant.onboarding_complete = True
    s.save_config(cfg)

    soul = render_persona(body.persona, name=cfg.assistant.name, user=cfg.assistant.user_name)
    if soul:
        s.workspace.write("soul", soul)
    lines = [f"# About {cfg.assistant.user_name}", "", f"- Name: {cfg.assistant.user_name}"]
    if cfg.assistant.location:
        lines.append(f"- Location: {cfg.assistant.location}")
    lines.append(f"- Timezone: {cfg.assistant.timezone}")
    if body.professional_context.strip():
        lines += ["", "## Work", body.professional_context.strip()]
    if body.personal_context.strip():
        lines += ["", "## Life", body.personal_context.strip()]
    s.workspace.write("user", "\n".join(lines) + "\n")

    if s.memory is not None:
        seed = "\n".join(
            x
            for x in (
                f"My name is {cfg.assistant.user_name}.",
                f"I live in {cfg.assistant.location}." if cfg.assistant.location else "",
                body.professional_context.strip(),
                body.personal_context.strip(),
            )
            if x
        )

        async def _seed() -> None:
            try:
                results = await s.memory.extract_and_store(seed, cfg.assistant.user_name, source="onboarding")
                for r in results:
                    s.bus.publish("memory.updated", {"action": r["action"], "id": r["id"], "content": r["content"]})
            except Exception:  # background best effort
                pass

        assert s.agent is not None
        s.agent._spawn(_seed())
    if body.daily_brief:  # the first opt-in: a recurring task the user can edit, pause or delete later
        await s.proactivity.brief.setup({})
    return {"ok": True}


# ----------------------------------------------------------------------------- config
@router.get("/api/config", dependencies=_auth)
async def get_config(request: Request):
    return get_core(request).config.model_dump(mode="json")


@router.get("/api/config/schema", dependencies=_auth)
async def get_config_schema():
    return config_json_schema()


@router.put("/api/config", dependencies=_auth)
async def put_config(request: Request, body: dict):
    s = get_core(request)
    try:
        cfg = SentientConfig.model_validate(body)
    except Exception as exc:
        raise HTTPException(422, _validation_detail(exc)) from exc
    s.save_config(cfg)
    return {"saved": True}


def _is_free_map(path: tuple[str, ...]) -> bool:
    """True when ``path`` points at a ``dict[str, ...]`` config field (fallbacks, reasoning,
    temperature, providers, mcp_servers) rather than a nested settings section."""
    import typing

    from pydantic import BaseModel

    model: type[BaseModel] = SentientConfig
    for i, part in enumerate(path):
        field = model.model_fields.get(part)
        if field is None:
            return False
        ann = field.annotation
        origin = typing.get_origin(ann)
        if i == len(path) - 1:
            return origin is dict
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            model = ann
        else:
            return False
    return False


def _deep_merge(base: dict, patch: dict, path: tuple[str, ...] = ()) -> dict:
    """Merge ``patch`` into ``base``. Inside free-form maps a ``null`` value removes the entry."""
    out = dict(base)
    in_map = bool(path) and _is_free_map(path)
    for k, v in patch.items():
        if v is None and in_map:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v, (*path, k))
        else:
            out[k] = v
    return out


def _validation_detail(exc: Exception) -> Any:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            return [
                {"loc": [str(x) for x in e.get("loc", ())], "msg": e.get("msg", ""), "type": e.get("type", "")}
                for e in errors()
            ]
        except Exception:
            pass
    return str(exc)


@router.patch("/api/config", dependencies=_auth)
async def patch_config(request: Request, body: dict):
    s = get_core(request)
    merged = _deep_merge(s.config.model_dump(mode="json"), body)
    try:
        cfg = SentientConfig.model_validate(merged)
    except Exception as exc:
        raise HTTPException(422, _validation_detail(exc)) from exc
    s.save_config(cfg)
    return {"saved": True, "config": cfg.model_dump(mode="json")}


# ----------------------------------------------------------------------------- sessions
@router.get("/api/sessions", dependencies=_auth)
async def sessions(request: Request, limit: int = 100):
    rows = await get_core(request).store.list_sessions(limit)
    for row in rows:  # stored as JSON text; the API gives a list (or null)
        raw = row.get("visited_hosts")
        try:
            row["visited_hosts"] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            row["visited_hosts"] = None
    return rows


@router.post("/api/sessions", dependencies=_auth)
async def create_session(request: Request, channel: str = "desktop"):
    sid = await get_core(request).store.create_session(channel=channel)
    return {"session_id": sid}


@router.get("/api/sessions/search", dependencies=_auth)
async def search_sessions(request: Request, q: str, limit: int = 30):
    s = get_core(request)
    safe = " ".join(f'"{w}"' for w in re.findall(r"\w+", q))
    if not safe:
        return []
    rows = await s.store.search_messages(safe, limit=limit)
    return [
        {
            "session_id": r["session_id"],
            "message_id": r["id"],
            "role": r["role"],
            "snippet": (r["content"] or "")[:240],
            "created_at": r["created_at"],
        }
        for r in rows
        if r["role"] in {"user", "assistant"}
    ]


class RenameBody(BaseModel):
    title: str


@router.patch("/api/sessions/{session_id}", dependencies=_auth)
async def rename_session(request: Request, session_id: str, body: RenameBody):
    s = get_core(request)
    await s.store.rename_session(session_id, body.title.strip()[:120])
    s.bus.publish("session.updated", {"session_id": session_id, "title": body.title.strip()[:120]})
    return {"ok": True}


@router.delete("/api/sessions/{session_id}", dependencies=_auth)
async def delete_session(request: Request, session_id: str):
    await get_core(request).store.delete_session(session_id)
    return {"ok": True}


@router.get("/api/sessions/{session_id}/messages", dependencies=_auth)
async def session_messages(request: Request, session_id: str, limit: int = 500):
    return await get_core(request).store.recent_messages(session_id, limit)


# ----------------------------------------------------------------------------- rules from chat (#130)
@router.get("/api/sessions/{session_id}/rule-proposals", dependencies=_auth)
async def rule_proposals(request: Request, session_id: str, status: str = "pending"):
    """Rules Sentient proposed from what was said in this chat; ``status=all`` includes answered ones."""
    if status not in {"pending", "accepted", "declined", "all"}:
        raise HTTPException(422, "status must be pending, accepted, declined or all")
    return await get_core(request).chat_rules.list(session_id, None if status == "all" else status)


class RuleDecisionBody(BaseModel):
    decision: Literal["accept", "decline"]


@router.post("/api/rule-proposals/{proposal_id}", dependencies=_auth)
async def decide_rule_proposal(request: Request, proposal_id: str, body: RuleDecisionBody):
    """The user's click on "Make it a rule" or "Not now". Nothing else creates a rule from chat."""
    try:
        return await get_core(request).chat_rules.decide(proposal_id, body.decision)
    except LookupError as exc:
        raise HTTPException(404, "no such rule proposal") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ----------------------------------------------------------------------------- chat fallback + approvals
class ChatBody(BaseModel):
    text: str = ""
    session_id: str | None = None
    channel: str = "desktop"
    attachments: list[str] = []
    model: str | None = None


@router.post("/api/chat", dependencies=_auth)
async def chat(request: Request, body: ChatBody):
    s = get_core(request)
    assert s.agent is not None
    if body.session_id and not body.attachments and s.agent.steer(body.session_id, body.text):
        # a reply is already running for this chat: the text steers it (docs/API.md section 10)
        ack = {"type": "steer_ack", "session_id": body.session_id, "queued": True}
        return StreamingResponse(iter([json.dumps(ack) + "\n"]), media_type="application/x-ndjson")
    session_id = body.session_id or await s.store.create_session(channel=body.channel)

    async def gen():
        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        async for event in s.agent.run_turn(
            session_id, body.text, channel=body.channel, attachments=body.attachments, model=body.model
        ):
            yield event.model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class ApprovalBody(BaseModel):
    approval_id: str
    decision: str


@router.post("/api/approvals", dependencies=_auth)
async def approve(request: Request, body: ApprovalBody):
    ok = get_core(request).approvals.resolve(body.approval_id, body.decision)
    if not ok:
        raise HTTPException(404, "no such pending approval")
    return {"resolved": True}


# ----------------------------------------------------------------------------- files
def uploads_dir() -> Path:
    d = paths.files_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\- ()]+", "_", base).strip() or "file"
    return base[:160]


def resolve_file(name: str) -> Path:
    """Files API names are relative to ~/.sentient/files (uploads live in uploads/)."""
    root = paths.files_dir().resolve()
    candidate = (root / name).resolve()
    if root not in candidate.parents:
        raise HTTPException(400, "invalid file name")
    return candidate


@router.post("/api/files", dependencies=_auth)
async def upload_file(request: Request, file: UploadFile = File(...)):
    name = _safe_name(file.filename or "upload")
    target = uploads_dir() / name
    stem, suffix = target.stem, target.suffix
    i = 1
    while target.exists():
        target = uploads_dir() / f"{stem} ({i}){suffix}"
        i += 1
    size = 0
    with target.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "file too large (50 MB max)")
            fh.write(chunk)
    rel = target.relative_to(paths.files_dir()).as_posix()
    return {"name": rel, "size": size, "mime": mimetypes.guess_type(target.name)[0] or "application/octet-stream"}


@router.get("/api/files", dependencies=_auth)
async def list_files(request: Request):
    root = paths.files_dir()
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out.append(
                {
                    "name": p.relative_to(root).as_posix(),
                    "size": st.st_size,
                    "mime": mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                    "modified_at": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
                }
            )
    return out


@router.get("/api/files/content/{name:path}", dependencies=_auth)
async def file_content(request: Request, name: str):
    p = resolve_file(name)
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@router.delete("/api/files/{name:path}", dependencies=_auth)
async def delete_file(request: Request, name: str):
    p = resolve_file(name)
    if p.is_file():
        p.unlink()
    return {"ok": True}


# ----------------------------------------------------------------------------- tools & usage
@router.get("/api/tools", dependencies=_auth)
async def tools(request: Request):
    # tools behind a "never" rule stay listed so Settings can show and change the rule
    return get_core(request).registry.catalog(include_blocked=True)


@router.get("/api/usage", dependencies=_auth)
async def usage(request: Request, days: int = 30):
    s = get_core(request)
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    async def grouped(col: str) -> list[dict[str, Any]]:
        rows = await s.store.fetchall(
            f"SELECT {col} AS k, SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c, COUNT(*) AS n"
            f" FROM usage WHERE created_at >= ? GROUP BY {col} ORDER BY p + c DESC",
            (since,),
        )
        return [{"key": r["k"], "prompt_tokens": r["p"] or 0, "completion_tokens": r["c"] or 0, "calls": r["n"]} for r in rows]

    by_model = [{"model": x.pop("key"), **x} for x in await grouped("model")]
    by_source = [{"source": x.pop("key"), **x} for x in await grouped("source")]
    by_day = [{"day": x.pop("key"), **x} for x in await grouped("substr(created_at, 1, 10)")]
    by_day.sort(key=lambda d: d["day"])
    return {
        "totals": {
            "prompt_tokens": sum(m["prompt_tokens"] for m in by_model),
            "completion_tokens": sum(m["completion_tokens"] for m in by_model),
        },
        "by_model": by_model,
        "by_source": by_source,
        "by_day": by_day,
    }
