"""Core REST routes. OWNER: core. Contract: docs/API.md sections 2 and 3.

health, bootstrap, stop everything, microphones and cameras, onboarding, config, sessions, chat (NDJSON),
approvals, files, tools catalog, usage.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from sentient import __version__, paths
from sentient.app_names import clean_app_name
from sentient.config import config_json_schema
from sentient.config.schema import SentientConfig
from sentient.files import names as file_names
from sentient.gateway.auth import require_token
from sentient.gateway.deps import MCP_OK, get_core
from sentient.llm import availability
from sentient.llm.events import Done, Error
from sentient.llm.spending import BudgetError
from sentient.memory import scopes as memory_scopes
from sentient.memory.personas import render_persona
from sentient.tools.rules import SCREEN_FOLDER, SELECTION_FOLDER

open_router = APIRouter(tags=["core"])
router = APIRouter(tags=["core"])
_auth = [Depends(require_token)]
log = logging.getLogger(__name__)

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


@router.post("/api/activity", dependencies=_auth)
async def activity(request: Request):
    """The window is in use (focused, clicked, typed). Resets the timer that pauses routines while away (#163)."""
    s = get_core(request)
    await s.user_seen("app")
    last = await s.tasks.away.last_active()
    return {"ok": True, "last_active_at": last.isoformat() if last else None}


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


# ----------------------------------------------------------------------------- microphones and cameras (section 24)
class SensorStopBody(BaseModel):
    id: str | None = None


@router.get("/api/sensors", dependencies=_auth)
async def sensors(request: Request):
    return get_core(request).sensors.state()


@router.post("/api/sensors/stop", dependencies=_auth)
async def sensors_stop(request: Request, body: SensorStopBody | None = None):
    """Turn off one live microphone or camera (``id``), or all of them. One that is already off is not an error."""
    registry = get_core(request).sensors
    result = await registry.stop(body.id if body else None)
    return {**result, "state": registry.state()}


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
        s.agent._spawn(_seed(), "memory")
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
    s = get_core(request)
    rows = await s.store.list_sessions(limit)
    for row in rows:  # stored as JSON text; the API gives a list (or null)
        row["memory_scope"] = memory_scopes.of_session(s.config, row)  # the scope it uses now (#155)
        raw = row.get("visited_hosts")
        try:
            row["visited_hosts"] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            row["visited_hosts"] = None
        row["incognito"] = bool(row.get("incognito"))
    return rows


def _chat_model(value: str | None) -> str | None:
    """A chat's model as given (``provider/model``), None for "follow the default" (#312). 422 when malformed."""
    try:
        return availability.model_id(value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _chat_scope(value: str | None) -> str | None:
    """A chat's memory scope (#155): None follows the default; 422 for "always"."""
    try:
        return memory_scopes.chat_scope(value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/sessions", dependencies=_auth)
async def create_session(
    request: Request,
    channel: str = "desktop",
    incognito: bool = False,
    model: str | None = None,
    memory_scope: str | None = None,
):
    sid = await get_core(request).store.create_session(
        channel=channel, incognito=incognito, model=_chat_model(model), memory_scope=_chat_scope(memory_scope)
    )
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


class SessionPatch(BaseModel):
    title: str | None = None
    # the chat's own model (#312): "provider/model" keeps it for this chat; null or "" follows the default again
    model: str | None = None
    # the memories it uses (#155): "personal", "work" or a project's name; null or "" follows the default again
    memory_scope: str | None = None


@router.patch("/api/sessions/{session_id}", dependencies=_auth)
async def update_session(request: Request, session_id: str, body: SessionPatch):
    """Rename a chat, or pick its model or memory scope. Only the fields sent change."""
    s = get_core(request)
    session = await s.store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "no such chat")
    # every field is checked before anything is written, so a 422 changes nothing
    model = _chat_model(body.model) if "model" in body.model_fields_set else None
    scope = _chat_scope(body.memory_scope) if "memory_scope" in body.model_fields_set else None
    changed: dict[str, Any] = {}
    if body.title is not None:
        changed["title"] = body.title.strip()[:120]
        await s.store.rename_session(session_id, changed["title"])
    if "model" in body.model_fields_set:
        changed["model"] = model
        await s.store.set_session_model(session_id, model)
    if "memory_scope" in body.model_fields_set:
        await s.store.set_session_scope(session_id, scope)
        changed["memory_scope"] = memory_scopes.of_session(s.config, {**session, "memory_scope": scope})
    if changed:
        s.bus.publish("session.updated", {"session_id": session_id, **changed})
    return {"ok": True, **changed}


class TrustBody(BaseModel):
    # the newest message the window had shown: outside content stored after it keeps the chat marked
    upto: str | None = None


@router.post("/api/sessions/{session_id}/trust", dependencies=_auth)
async def trust_session(request: Request, session_id: str, body: TrustBody | None = None):
    """"I've checked it, trust this chat again" (ADR 0029): clear the chat's outside-content mark, up to the message
    the user had seen. The user's own action in the window; the model has no way to call it."""
    s = get_core(request)
    if s.agent is None:
        raise HTTPException(503, "the assistant isn't ready yet")
    try:
        res = await s.agent.trust_chat(session_id, via="desktop", upto=(body.upto if body else None) or None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if res is None:
        raise HTTPException(404, "no such chat")
    return {"ok": True, "was": res.was, "trusted": res.trusted, "untrusted": res.still if not res.trusted else "",
            "writer": res.writer}


@router.delete("/api/sessions/{session_id}", dependencies=_auth)
async def delete_session(request: Request, session_id: str):
    await get_core(request).delete_chat(session_id)
    return {"ok": True}


@router.get("/api/sessions/{session_id}/messages", dependencies=_auth)
async def session_messages(request: Request, session_id: str, limit: int = 500):
    s = get_core(request)
    if await s.store.get_session(session_id) is None:  # deleted (an incognito chat that was closed, #154)
        raise HTTPException(404, "no such chat")
    return await s.store.recent_messages(session_id, limit)


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
    incognito: bool = False  # a new chat (no session_id) that learns nothing and is deleted when closed (#154)
    app: str | None = None  # an MCP client's name, display only: whose words these are (#443)


@router.post("/api/chat", dependencies=MCP_OK)
async def chat(request: Request, body: ChatBody):
    s = get_core(request)
    assert s.agent is not None
    is_mcp = getattr(request.state, "caller", "gateway") == "mcp"
    if is_mcp:
        # an MCP app's words are outside content, never the user's (issue #443): the turn runs on
        # the mcp channel whatever the body claims, and the loop marks it from the start
        body.channel = "mcp"
        # the app does not shape the turn either: attachments would pull files from the files
        # folder into it, a named model could pick a cloud model where the user keeps chats
        # local, and incognito is the user's own switch to flip, so all three are ignored
        body.model = None
        body.attachments = []
        body.incognito = False
        # the app's name is its own text, shown to the user in approvals and notes: cleaned
        # (capped, no control characters) before it is used anywhere (see app_names)
        body.app = clean_app_name(body.app)
        if body.session_id:
            # the session must be one of the app's own: naming one of the user's chats would hand
            # the app that chat's history and let its words into it, so anything else starts fresh
            session = await s.store.get_session(body.session_id)
            if session is None or session.get("channel") != "mcp":
                body.session_id = None
    if body.session_id and not body.attachments and not is_mcp and s.agent.steer(body.session_id, body.text):
        # a reply is already running for this chat: the text steers it (docs/API.md section 10)
        ack = {"type": "steer_ack", "session_id": body.session_id, "queued": True}
        return StreamingResponse(iter([json.dumps(ack) + "\n"]), media_type="application/x-ndjson")
    session_id = body.session_id or await s.store.create_session(channel=body.channel, incognito=body.incognito)

    async def gen():
        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        # "Make this a rule?" cards for this chat ride along in the /ws envelope (#311): a client without /ws can
        # show and answer them, before the approval requests they cause
        async with s.bus.subscribe() as q:

            def proposals() -> list[str]:
                out = []
                while not q.empty():
                    e = q.get_nowait()
                    if e["type"] == "rule_proposal.updated" and (e["data"] or {}).get("session_id") == session_id:
                        out.append(json.dumps(e, default=str) + "\n")
                return out

            ended = False  # the last line sent was a done
            try:
                async for event in s.agent.run_turn(
                    session_id, body.text, channel=body.channel, attachments=body.attachments, model=body.model,
                    app_name=body.app if is_mcp else None,
                ):
                    for line in proposals():
                        yield line
                    yield event.model_dump_json() + "\n"
                    ended = isinstance(event, Done)
            except Exception as exc:
                log.exception("chat turn failed")
                if not ended:  # a turn that broke still ends with error, then done (docs/API.md section 1, #334)
                    for line in proposals():  # rule cards made before it broke still come first
                        yield line
                    yield Error(message=str(exc), recoverable=False, session_id=session_id).model_dump_json() + "\n"
                    yield Done(content="", error=str(exc), session_id=session_id).model_dump_json() + "\n"
            for line in proposals():
                yield line

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


class UndoBody(BaseModel):
    undo_id: str
    action: str = "undo"  # "undo" | "send_now"


@router.post("/api/undo", dependencies=_auth)
async def undo_send(request: Request, body: UndoBody):
    """Undo (or send now) an email, message or post waiting in its undo window (#142)."""
    if not get_core(request).undo.respond(body.undo_id, body.action):
        raise HTTPException(404, "nothing is waiting with that id")
    return {"done": True, "action": body.action}


@router.get("/api/approvals/grants", dependencies=_auth)
async def list_grants(request: Request):
    """Live "Allow for this task" and "Allow for a while" answers (#139)."""
    grants = get_core(request).approvals.grants
    return {"grants": await grants.list() if grants is not None else []}


@router.delete("/api/approvals/grants/{grant_id}", dependencies=_auth)
async def revoke_grant(request: Request, grant_id: str):
    grants = get_core(request).approvals.grants
    if grants is None or not await grants.revoke(grant_id):
        raise HTTPException(404, "no such grant")
    return {"ok": True}


# ----------------------------------------------------------------------------- files
def uploads_dir(folder: str = "uploads") -> Path:
    d = paths.files_dir() / folder
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
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    source: Literal["upload", "screen", "selection"] = Form("upload"),
):
    # what the user shared from their screen (screens/) or selected in another app (selections/) marks the chat it is
    # sent in (ADR 0018)
    folder = uploads_dir({"screen": SCREEN_FOLDER, "selection": SELECTION_FOLDER}.get(source, "uploads"))
    # a name is never handed out again while a chat has it attached, and captures always get a random part (#458)
    target = await file_names.create(
        folder, _safe_name(file.filename or "upload"), store=get_core(request).store, unique=source != "upload"
    )
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


@router.api_route("/api/files/content/{name:path}", methods=["GET", "HEAD"], dependencies=_auth)
async def file_content(request: Request, name: str):
    # HEAD answers whether the file is still there: a chat shows "No longer available" for an attachment that is gone
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
            f"SELECT {col} AS k, SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c, COUNT(*) AS n,"
            f" SUM(cost_usd) AS usd FROM usage WHERE created_at >= ? GROUP BY {col} ORDER BY p + c DESC",
            (since,),
        )
        return [{"key": r["k"], "prompt_tokens": r["p"] or 0, "completion_tokens": r["c"] or 0, "calls": r["n"],
                 "cost_usd": r["usd"]} for r in rows]

    by_model = [{"model": x.pop("key"), **x} for x in await grouped("model")]
    by_source = [{"source": x.pop("key"), **x} for x in await grouped("source")]
    by_day = [{"day": x.pop("key"), **x} for x in await grouped("substr(created_at, 1, 10)")]
    by_day.sort(key=lambda d: d["day"])
    return {
        "totals": {
            "prompt_tokens": sum(m["prompt_tokens"] for m in by_model),
            "completion_tokens": sum(m["completion_tokens"] for m in by_model),
            "cost_usd": round(sum(m["cost_usd"] or 0.0 for m in by_model), 6),
        },
        "by_model": by_model,
        "by_source": by_source,
        "by_day": by_day,
    }


@router.get("/api/usage/cost", dependencies=_auth)
async def usage_cost(request: Request):
    """The cost meter (#148): today, this week and this month per model and role, and the monthly budget."""
    return await get_core(request).spending.report()


@router.post("/api/usage/budget/keep-going", dependencies=_auth)
async def budget_keep_going(request: Request):
    """Allow paid cloud models again until the month ends, when the budget is set to ask."""
    try:
        return await get_core(request).spending.keep_going()
    except BudgetError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
