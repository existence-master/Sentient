"""REST routes for the user model (owner: memory agent). Contract: docs/API.md section 15.

Dream routes live under ``/api/memories/dreams`` in ``routes/memory.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/user-model", tags=["user_model"], dependencies=AUTH)


def _um(request: Request):
    return get_core(request).user_model


@router.get("")
async def get_user_model(request: Request):
    return await _um(request).get_state()


@router.post("/insights")
async def add_insight(request: Request, body: dict):
    try:
        return await _um(request).add_insight(str(body.get("statement") or ""), str(body.get("dimension") or "context"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/insights/{insight_id}")
async def patch_insight(request: Request, insight_id: str, body: dict):
    statement = body.get("statement")
    status = body.get("status")
    try:
        return await _um(request).update_insight(
            insight_id,
            statement=str(statement) if statement is not None else None,
            status=str(status) if status is not None else None,
        )
    except KeyError as exc:
        raise HTTPException(404, f"insight {insight_id} not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/insights/{insight_id}")
async def delete_insight(request: Request, insight_id: str):
    if not await _um(request).delete_insight(insight_id):
        raise HTTPException(404, f"insight {insight_id} not found")
    return {"ok": True}


@router.post("/refresh")
async def refresh(request: Request):
    return await _um(request).refresh(trigger="manual")


@router.post("/questions/{question_id}")
async def answer_question(request: Request, question_id: str, body: dict):
    try:
        out = await _um(request).answer_question(question_id, str(body.get("answer") or ""))
    except KeyError as exc:
        raise HTTPException(404, f"open question {question_id} not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "verdict": out["verdict"], "insight": out["insight"]}


@router.delete("/questions/{question_id}")
async def dismiss_question(request: Request, question_id: str):
    if not await _um(request).dismiss_question(question_id):
        raise HTTPException(404, f"open question {question_id} not found")
    return {"ok": True}
