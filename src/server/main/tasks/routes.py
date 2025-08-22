import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import uuid
from typing import Tuple
from main.dependencies import auth_helper
from main.dependencies import mongo_manager, websocket_manager
from main.auth.utils import PermissionChecker
from workers.tasks import generate_plan_from_context, execute_task_plan, calculate_next_run, process_task_change_request, refine_task_details, refine_and_plan_ai_task, cud_memory_task, orchestrate_swarm_task, start_long_form_task

from .models import AddTaskRequest, UpdateTaskRequest, TaskIdRequest, TaskActionRequest, TaskChatRequest, ProgressUpdateRequest, AnswerClarificationRequest, ClarificationAnswerRequest, LongFormTaskActionRequest
from main.llm import run_agent
from main.tasks.models import AddTaskRequest, UpdateTaskRequest, TaskIdRequest, TaskActionRequest, TaskChatRequest, ProgressUpdateRequest
from workers.tasks import generate_plan_from_context, execute_task_plan, calculate_next_run, refine_and_plan_ai_task
from main.llm import run_agent, LLMProviderDownError
from json_extractor import JsonExtractor
from .prompts import TASK_CREATION_PROMPT
from workers.long_form_tasks import start_long_form_task, execute_orchestrator_cycle

class GeneratePlanRequest(BaseModel):
    prompt: str

router = APIRouter(
    prefix="/tasks",
    tags=["Agents & Tasks"]
)

logger = logging.getLogger(__name__)

@router.post("/generate-plan", status_code=status.HTTP_200_OK)
async def generate_plan_from_prompt(
    request: GeneratePlanRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    user_profile = await mongo_manager.get_user_profile(user_id)
    personal_info = user_profile.get("userData", {}).get("personalInfo", {}) if user_profile else {}
    user_name = personal_info.get("name", "User")
    user_timezone_str = personal_info.get("timezone", "UTC")
    try:
        user_timezone = ZoneInfo(user_timezone_str)
    except ZoneInfoNotFoundError:
        user_timezone = ZoneInfo("UTC")

    current_time_str = datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

    system_prompt = TASK_CREATION_PROMPT.format(
        user_name=user_name,
        user_timezone=user_timezone_str,
        current_time=current_time_str
    )

    try:
        messages = [{'role': 'user', 'content': request.prompt}]

        response_str = ""
        for chunk in run_agent(system_message=system_prompt, function_list=[], messages=messages):
            if isinstance(chunk, list) and chunk:
                last_message = chunk[-1]
                if last_message.get("role") == "assistant" and isinstance(last_message.get("content"), str):
                    response_str = last_message["content"]

        if not response_str:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")

        task_data = JsonExtractor.extract_valid_json(response_str)

        return JSONResponse(content=task_data)

    except LLMProviderDownError as e:
        logger.error(f"LLM provider down during plan generation for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Sorry, our AI provider is currently down. Please try again later.")
    except Exception as e:
        logger.error(f"Error generating plan from prompt for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_details(
    task_id: str,
    user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))
):
    """Fetches the full details of a single task by its ID."""
    task = await mongo_manager.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    return JSONResponse(content=task)

@router.post("/add-task", status_code=status.HTTP_201_CREATED)
async def add_task(
    request: AddTaskRequest,
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan
    # Note: Plan limit checks are removed as per the simplified plan. They can be added back later.

    if request.task_type == "swarm":
        if not request.prompt:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A prompt describing the goal is required for swarm tasks.")

        task_data = {
            "name": request.prompt, # Use prompt as the initial name
            "description": f"Swarm task to achieve the goal: {request.prompt}",
            "task_type": "swarm",
            "swarm_details": {
                "goal": request.prompt,
                "items": [],
                "total_agents": 0,
                "completed_agents": 0,
                "progress_updates": [],
                "aggregated_results": []
            }
        }
        task_id = await mongo_manager.add_task(user_id, task_data)
        if not task_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create swarm task.")

        orchestrate_swarm_task.delay(task_id, user_id)
        return {"message": "Swarm task initiated! Planning will begin shortly.", "task_id": task_id}

    elif request.task_type == "long_form":
        if not request.prompt:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A prompt describing the goal is required for long-form tasks.")

        task_data = {
            "name": request.prompt,
            "description": f"Long-form task to achieve the goal: {request.prompt}",
            "task_type": "long_form",
            "auto_approve_subtasks": request.auto_approve_subtasks,
            "orchestrator_state": {
                "main_goal": request.prompt,
                "current_state": "CREATED",
                "current_step": 0,
                "context_store": {},
                "waiting_config": None,
            },
            "dynamic_plan": [],
            "clarification_requests": [],
            "execution_log": []
        }
        task_id = await mongo_manager.add_task(user_id, task_data)
        if not task_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create long-form task.")
        start_long_form_task.delay(task_id, user_id)
        return {"message": "Long-form task initiated! Planning will begin shortly.", "task_id": task_id}

    elif request.task_type == "single":
        if not request.prompt:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A prompt is required for single tasks.")

        task_data = {
            "name": request.prompt, # Use prompt as initial name
            "description": request.prompt,
            "schedule": request.schedule, # Pass the entire schedule object
            "task_type": "single"
        }
        task_id = await mongo_manager.add_task(user_id, task_data)
        if not task_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create task.")

        # The existing worker is perfect for this, as it refines details from a prompt and schedule.
        refine_and_plan_ai_task.delay(task_id, user_id)
        return {"message": "Task created! Planning will begin shortly.", "task_id": task_id}

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid task_type: {request.task_type}")

@router.post("/fetch-tasks")
async def fetch_tasks(
    user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))
):
    tasks = await mongo_manager.get_all_tasks_for_user(user_id)
    return {"tasks": tasks}

@router.post("/update-task")
async def update_task(
    request: UpdateTaskRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    task = await mongo_manager.get_task(request.taskId, user_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")

    update_data = request.dict(exclude_unset=True)
    update_data.pop("taskId", None)

    # If the schedule is being updated, inject the user's timezone for recurring tasks.
    if 'schedule' in update_data and update_data['schedule']:
        if update_data['schedule'].get('type') == 'recurring':
            user_profile = await mongo_manager.get_user_profile(user_id)
            user_timezone_str = user_profile.get("userData", {}).get("personalInfo", {}).get("timezone", "UTC")
            update_data['schedule']['timezone'] = user_timezone_str

    # If assignee is changed to 'ai' and task is in a non-planned state, trigger planning
    if 'assignee' in update_data and update_data['assignee'] == 'ai':
        if task.get('status') == 'pending': # Assuming 'pending' is the status for user-assigned tasks
            update_data['status'] = 'planning'
            generate_plan_from_context.delay(request.taskId, user_id)
            logger.info(f"Task {request.taskId} reassigned to AI. Triggering planning.")

    # The plan is now a top-level field, so a simple update is sufficient.
    # The special logic for updating a plan within a run is no longer needed.
    success = await mongo_manager.update_task(request.taskId, update_data)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or no updates applied.")
    return {"message": "Task updated successfully."}

@router.post("/delete-task")
async def delete_task(
    request: TaskIdRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    task = await mongo_manager.get_task(request.taskId, user_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")

    # Always delete the entire task, regardless of its type or number of runs.
    message = await mongo_manager.delete_task(request.taskId, user_id)
    if not message:
        raise HTTPException(status_code=404, detail="Task not found or not deleted.")

    # Also delete any notifications related to the task.
    await mongo_manager.delete_notifications_for_task(user_id, request.taskId)

    return {"message": message}

@router.post("/task-action", status_code=status.HTTP_200_OK)
async def task_action(
    request: TaskActionRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    if request.action == "decline":
        message = await mongo_manager.decline_task(request.taskId, user_id)
        if not message:
            raise HTTPException(status_code=400, detail="Failed to decline task.")
        return JSONResponse(content={"message": message})
    elif request.action == "execute":
        # This implies immediate execution
        task = await mongo_manager.get_task(request.taskId, user_id)
        if not task:
            raise HTTPException(status_code=400, detail="Failed to execute task immediately.")

        # Create a new run object before dispatching
        now = datetime.now(timezone.utc)
        new_run = {
            "run_id": str(uuid.uuid4()),
            "status": "processing", # Set run status immediately
            "created_at": now,
            "execution_start_time": now
        }

        current_runs = task.get("runs", [])
        if not isinstance(current_runs, list):
            current_runs = []
        current_runs.append(new_run)

        update_payload = {
            "runs": current_runs,
            "status": "processing",
            "last_execution_at": now
        }
        await mongo_manager.update_task(request.taskId, update_payload)
        # Trigger the Celery task with the new run_id
        execute_task_plan.delay(request.taskId, user_id, new_run['run_id'])
        return JSONResponse(content={"message": "Task execution has been initiated."})
    else:
        raise HTTPException(status_code=400, detail="Invalid task action.")

@router.post("/rerun-task")
async def rerun_task(
    request: TaskIdRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    new_task_id = await mongo_manager.rerun_task(request.taskId, user_id)
    if not new_task_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original task not found or failed to duplicate.")

    # Trigger the planner for the new task
    generate_plan_from_context.delay(new_task_id, user_id)
    logger.info(f"Rerunning task {request.taskId}. New task {new_task_id} created and sent to planner.")
    return {"message": "Task has been duplicated for re-run.", "new_task_id": new_task_id}

@router.post("/approve-task", status_code=status.HTTP_200_OK)
async def approve_task(
    request: TaskIdRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    """Approves a task, scheduling it for execution."""
    task_id = request.taskId
    task_doc = await mongo_manager.get_task(task_id, user_id)
    if not task_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or user does not have permission.")

    if not task_doc.get("plan"):
        raise HTTPException(status_code=500, detail="Task has no plan to approve.")

    update_data = {}
    schedule_data = task_doc.get("schedule") or {}

    if schedule_data.get("type") == "recurring":
        # Activate the task. The scheduler will create runs.
        next_run, _ = calculate_next_run(schedule_data)
        if next_run:
            update_data["next_execution_at"] = next_run
            update_data["status"] = "active"
            update_data["enabled"] = True
        else:
            update_data["status"] = "error"
            update_data["error"] = "Could not calculate next run time for recurring task."
    elif schedule_data.get("type") == "triggered":
        # Activate the task. The trigger poller will create runs.
        update_data["status"] = "active"
        update_data["enabled"] = True
        logger.info(f"Approving triggered task {task_id}. Setting status to 'active'.")
    else: # One-off task
        run_at_time = None
        if schedule_data.get("type") == "once" and schedule_data.get("run_at"):
            run_at_time_str = schedule_data.get("run_at")
            if isinstance(run_at_time_str, str):
                if len(run_at_time_str) == 16: run_at_time_str += ":00"
                user_timezone_str = schedule_data.get("timezone", "UTC")
                try:
                    user_tz = ZoneInfo(user_timezone_str)
                except ZoneInfoNotFoundError:
                    user_tz = ZoneInfo("UTC")
                naive_run_at_time = datetime.fromisoformat(run_at_time_str)
                utc_run_at_time = naive_run_at_time.replace(tzinfo=user_tz).astimezone(timezone.utc)
                run_at_time = utc_run_at_time

        # If run_at is in the future, schedule it. Otherwise, run it now.
        if run_at_time and run_at_time > datetime.now(timezone.utc):
            # It's a future-scheduled task, so it becomes 'pending'.
            update_data["status"] = "pending"
            update_data["next_execution_at"] = run_at_time
            await mongo_manager.update_task(task_id, update_data)
            return JSONResponse(content={"message": "Task approved and scheduled for the future."})
        else:
            # It's an immediate task.
            now = datetime.now(timezone.utc)
            new_run = {
                "run_id": str(uuid.uuid4()),
                "status": "processing",
                "plan": task_doc.get("plan", []),
                "created_at": now,
                "execution_start_time": now
            }

            current_runs = task_doc.get("runs", [])
            if not isinstance(current_runs, list):
                current_runs = []
            current_runs.append(new_run)

            update_payload = {
                "runs": current_runs,
                "status": "processing",
                "last_execution_at": now,
                "next_execution_at": None,
            }
            await mongo_manager.update_task(task_id, update_payload)

            execute_task_plan.delay(task_id, user_id, new_run['run_id'])
            return JSONResponse(content={"message": "Task approved and execution has been initiated."})

    if update_data:
        success = await mongo_manager.update_task(task_id, update_data)
        if not success:
            logger.warning(f"Approve task for {task_id} resulted in 0 modified documents.")

    return JSONResponse(content={"message": "Task approved and scheduled."})

@router.post("/task-chat", status_code=status.HTTP_200_OK)
async def task_chat(
    request: TaskChatRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    """
    Handles a user's chat message for a task. If the task was completed,
    this will trigger a re-planning of the same task by adding a new 'run'.
    """
    task_id = request.taskId
    task = await mongo_manager.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")

    # Decryption is handled by get_task, so we can safely modify the list
    new_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now(timezone.utc)
    }

    chat_history = task.get("chat_history", [])
    if not isinstance(chat_history, list):
        # This can happen if the field was somehow corrupted or is an old format
        logger.warning(f"Task {task_id} chat_history is not a list. Resetting it.")
        chat_history = []
    chat_history.append(new_message)

    # Prepare the payload for a full field update
    update_payload = {
        "status": "planning", # Revert to planning for re-evaluation
        "chat_history": chat_history
    }

    # The update_task method will handle re-encrypting the entire chat_history field
    await mongo_manager.update_task(task_id, update_payload)

    # Re-trigger the planner for the same task
    generate_plan_from_context.delay(task_id, user_id)
    return JSONResponse(content={"message": "Change request received. The task is now being re-planned."})

@router.post("/internal/progress-update", include_in_schema=False)
async def internal_progress_update(request: ProgressUpdateRequest):
    # This is an internal endpoint called by workers. It should not be exposed publicly.
    # No auth check is performed, relying on network security (internal calls only).
    logger.info(f"Received internal progress update for task {request.task_id}")
    try:
        await websocket_manager.send_personal_json_message(
            {
                "type": "task_progress_update",
                "payload": {
                    "task_id": request.task_id,
                    "run_id": request.run_id,
                    "update": {
                        "message": request.message,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            },
            request.user_id,
            connection_type="notifications" # Use the existing notifications channel
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to push progress update via websocket for task {request.task_id}: {e}", exc_info=True)
        # Don't fail the worker, just log it.
        return {"status": "error", "detail": str(e)}

@router.post("/internal/task-update-push", include_in_schema=False)
async def internal_task_update_push(request: ProgressUpdateRequest): # Reusing model for convenience
    """
    Internal endpoint for workers to tell the main server to push a generic
    'tasks have changed' notification to the client via WebSocket.
    """
    logger.info(f"Received internal request to push task list update for user {request.user_id}")
    try:
        await websocket_manager.send_personal_json_message(
            {"type": "task_list_updated"},
            request.user_id,
            connection_type="notifications"
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to push task list update via websocket for user {request.user_id}: {e}", exc_info=True)
        return {"status": "error", "detail": str(e)}
@router.post("/answer-clarifications", status_code=status.HTTP_200_OK)
async def answer_clarifications(
    request: AnswerClarificationRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    task = await mongo_manager.get_task(request.task_id, user_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")

    if task['status'] != 'clarification_pending':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task is not awaiting clarification.")

    success = await mongo_manager.add_answers_to_task(request.task_id, [ans.dict() for ans in request.answers], user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save answers.")

    # Re-trigger the executor for the same run
    last_run = task.get("runs", [])[-1] if task.get("runs") else None
    if not last_run:
        raise HTTPException(status_code=500, detail="Cannot resume task: no previous run found.")

    await mongo_manager.update_task(request.task_id, {"status": "clarification_answered"})

    execute_task_plan.delay(request.task_id, user_id, last_run['run_id'])

    return JSONResponse(content={"message": "Answers submitted successfully. The task will now resume."})

@router.post("/{task_id}/clarifications", status_code=status.HTTP_200_OK)
async def answer_clarification(
    task_id: str,
    request: ClarificationAnswerRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    """Allows a user to answer a pending clarification request for a long-form task."""
    task = await mongo_manager.get_task(task_id, user_id)
    if not task or task.get("task_type") != "long_form":
        raise HTTPException(status_code=404, detail="Long-form task not found.")

    if task.get("orchestrator_state", {}).get("current_state") != "SUSPENDED":
        raise HTTPException(status_code=400, detail="Task is not currently awaiting clarification.")

    clarification_requests = task.get("clarification_requests", [])
    request_found = False
    for req in clarification_requests:
        if req.get("request_id") == request.request_id and req.get("status") == "pending":
            req["response"] = request.answer
            req["responded_at"] = datetime.now(timezone.utc)
            req["status"] = "answered"
            request_found = True
            break

    if not request_found:
        raise HTTPException(status_code=404, detail="Pending clarification request not found.")

    # Update task state and resume
    update_payload = {
        "clarification_requests": clarification_requests,
        "orchestrator_state.current_state": "ACTIVE"
    }
    await mongo_manager.update_task(task_id, update_payload)

    # Trigger the orchestrator to re-evaluate with the new information
    execute_orchestrator_cycle.delay(task_id)

    return JSONResponse(content={"message": "Answer received. Resuming task."})


@router.post("/{task_id}/action", status_code=status.HTTP_200_OK)
async def long_form_task_action(
    task_id: str,
    request: LongFormTaskActionRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))
):
    """Allows a user to manually pause or resume a long-form task."""
    task = await mongo_manager.get_task(task_id, user_id)
    if not task or task.get("task_type") != "long_form":
        raise HTTPException(status_code=404, detail="Long-form task not found.")

    action = request.action.lower()
    if action == "pause":
        await mongo_manager.update_task(task_id, {"orchestrator_state.current_state": "PAUSED"})
        return JSONResponse(content={"message": "Task paused."})
    elif action == "resume":
        await mongo_manager.update_task(task_id, {"orchestrator_state.current_state": "ACTIVE"})
        execute_orchestrator_cycle.delay(task_id)
        return JSONResponse(content={"message": "Task resumed."})
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'pause' or 'resume'.")