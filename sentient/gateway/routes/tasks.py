"""REST routes for tasks. OWNER: TASKS AGENT. Contract: docs/API.md section 4."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core
from sentient.llm.provider import ProviderError
from sentient.tasks.service import TaskConflict, TaskNotFound

router = APIRouter(prefix="/api/tasks", tags=["tasks"], dependencies=AUTH)


class CreateTaskBody(BaseModel):
    prompt: str
    is_swarm: bool = False
    assignee: str | None = "ai"
    model: str | None = None
    browser_profile: str | None = None


class PreviewBody(BaseModel):
    prompt: str


class ChatBody(BaseModel):
    message: str


class ClarificationAnswer(BaseModel):
    question_id: str
    answer_text: str = ""


class ClarificationsBody(BaseModel):
    answers: list[ClarificationAnswer] = Field(default_factory=list)


async def _guard[T](awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except TaskConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=f"The AI model is unavailable: {exc}") from exc
    except TypeError as exc:  # the model returned the wrong JSON shape
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _tasks(request: Request) -> Any:
    return get_core(request).tasks


@router.get("")
async def list_tasks(request: Request):
    return await _guard(_tasks(request).list())


@router.post("")
async def create_task(request: Request, body: CreateTaskBody):
    return await _guard(_tasks(request).create_task(
        body.prompt, is_swarm=body.is_swarm, model=body.model, browser_profile=body.browser_profile
    ))


@router.post("/preview")
async def preview_task(request: Request, body: PreviewBody):
    return await _guard(_tasks(request).preview(body.prompt))


@router.get("/{task_id}")
async def get_task(request: Request, task_id: str):
    return await _guard(_tasks(request).get(task_id))


@router.patch("/{task_id}")
async def update_task(request: Request, task_id: str, fields: dict[str, Any]):
    return await _guard(_tasks(request).update(task_id, fields))


@router.delete("/{task_id}")
async def delete_task(request: Request, task_id: str):
    return await _guard(_tasks(request).delete(task_id))


@router.post("/{task_id}/approve")
async def approve_task(request: Request, task_id: str):
    return await _guard(_tasks(request).approve(task_id))


@router.post("/{task_id}/decline")
async def decline_task(request: Request, task_id: str):
    return await _guard(_tasks(request).decline(task_id))


@router.post("/{task_id}/rerun")
async def rerun_task(request: Request, task_id: str):
    return await _guard(_tasks(request).rerun(task_id))


@router.post("/{task_id}/run-now")
async def run_task_now(request: Request, task_id: str):
    return await _guard(_tasks(request).run_now(task_id))


@router.post("/{task_id}/archive")
async def archive_task(request: Request, task_id: str):
    return await _guard(_tasks(request).archive(task_id))


@router.post("/{task_id}/chat")
async def task_chat(request: Request, task_id: str, body: ChatBody):
    return await _guard(_tasks(request).chat(task_id, body.message))


@router.post("/{task_id}/clarifications")
async def answer_clarifications(request: Request, task_id: str, body: ClarificationsBody):
    answers = [a.model_dump() for a in body.answers]
    return await _guard(_tasks(request).answer_clarifications(task_id, answers))


@router.post("/{task_id}/runs/{run_id}/cancel")
async def cancel_run(request: Request, task_id: str, run_id: str):
    return await _guard(_tasks(request).cancel_run(task_id, run_id))


class AnswerBody(BaseModel):
    answer: str = ""


@router.post("/{task_id}/runs/{run_id}/answer")
async def answer_question(request: Request, task_id: str, run_id: str, body: AnswerBody):
    return await _guard(_tasks(request).answer_question(task_id, run_id, body.answer))


@router.post("/{task_id}/runs/{run_id}/retry")
async def retry_run(request: Request, task_id: str, run_id: str):
    return await _guard(_tasks(request).retry_run(task_id, run_id))


class ScriptTestBody(BaseModel):
    code: str | None = None


@router.post("/{task_id}/script/test")
async def test_script(request: Request, task_id: str, body: ScriptTestBody | None = None):
    return await _guard(_tasks(request).test_script(task_id, body.code if body else None))


@router.get("/{task_id}/runs/{run_id}/events")
async def run_events(request: Request, task_id: str, run_id: str):
    return await _guard(_tasks(request).run_events(task_id, run_id))
