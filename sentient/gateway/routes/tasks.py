"""REST routes for tasks. OWNER: TASKS AGENT. Contract: docs/API.md section 4."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import MCP_OK, get_core
from sentient.llm.provider import PlainProviderError, ProviderError
from sentient.tasks.service import TaskConflict, TaskNotFound

router = APIRouter(prefix="/api/tasks", tags=["tasks"], dependencies=MCP_OK)


class CreateTaskBody(BaseModel):
    prompt: str
    is_swarm: bool = False
    assignee: str | None = "ai"
    model: str | None = None
    browser_profile: str | None = None
    memory_scope: str | None = None  # #155: personal (default), work or a project's name
    app: str | None = None  # the MCP app asking, when the caller is one (#443): names the task's source


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
    except PlainProviderError as exc:  # a refused job, no credits, a bad key...: say why and what to change
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    if getattr(request.state, "caller", "gateway") == "mcp":
        # an outside app's task (issue #443): made in the app's name and marked as outside
        # content, so its plan always waits for approval and every send asks, like a task by
        # email (#159). Only the MCP token can be this caller; the app name is display only.
        app_name = (body.app or "").strip() or "An MCP app"
        return await _guard(_tasks(request).create_from_mcp(body.prompt, app_name))
    return await _guard(_tasks(request).create_task(
        body.prompt, is_swarm=body.is_swarm, model=body.model, browser_profile=body.browser_profile,
        memory_scope=body.memory_scope,
    ))


@router.post("/preview")
async def preview_task(request: Request, body: PreviewBody):
    return await _guard(_tasks(request).preview(body.prompt))


@router.get("/away")
async def away_status(request: Request):
    """Routines paused while the user was away (#163), before ``/{task_id}`` so "away" isn't read as an id."""
    return await _tasks(request).away.status()


@router.post("/away/resume")
async def away_resume(request: Request):
    return await _tasks(request).away.resume()


@router.post("/away/keep")
async def away_keep(request: Request):
    return await _tasks(request).away.keep()


@router.get("/email")
async def email_status(request: Request, limit: int = 20):
    """Tasks by email (#159): the address, who may send tasks and what happened to the latest emails."""
    return await _tasks(request).inbox.status(limit=limit)


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


@router.post("/{task_id}/dry-run")
async def try_task(request: Request, task_id: str):
    """Try it first (#142): a run that only notes what it would send, change or run."""
    return await _guard(_tasks(request).dry_run(task_id))


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
    # the request being answered (its ``expires_at``): set, a run now waiting on another request refuses (#323)
    expires_at: str | None = None


@router.post("/{task_id}/runs/{run_id}/answer")
async def answer_question(request: Request, task_id: str, run_id: str, body: AnswerBody):
    return await _guard(_tasks(request).answer_question(task_id, run_id, body.answer, expect=body.expires_at))


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
