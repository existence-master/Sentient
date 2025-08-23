import logging
import uuid
import datetime
from typing import Dict, Any, List, Optional
from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import state_manager, waiting_manager, auth
from workers.utils.api_client import notify_user
from workers.tasks import refine_and_plan_ai_task
from mcp_hub.orchestrator.prompts import COMPLETION_EVALUATION_PROMPT
from main.llm import run_agent as run_main_agent
import json

logger = logging.getLogger(__name__)

# Note: @mcp.tool() decorator is applied in main.py

async def update_plan(ctx: Context, new_steps: List[Dict], reasoning: str, main_goal_update: str = None) -> Dict:
    """Update the dynamic plan with new steps or modified goal"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    await state_manager.update_dynamic_plan(task_id, user_id, new_steps, main_goal_update)
    await state_manager.add_execution_log(task_id, user_id, "plan_updated", {"new_step_count": len(new_steps)}, reasoning)
    return {"status": "success", "message": "Plan updated successfully."}

async def update_context(ctx: Context, key: str, value: Any, reasoning: str) -> Dict:
    """Store information in the task's context store"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    await state_manager.update_context_store(task_id, user_id, key, value)
    await state_manager.add_execution_log(task_id, user_id, "context_updated", {"key": key}, reasoning)
    return {"status": "success", "message": f"Context updated for key '{key}'."}

async def get_context(ctx: Context, key: str = None) -> Dict:
    """Retrieve information from task's context store"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    task = await state_manager.get_task_state(task_id, user_id)
    context_store = task.get("orchestrator_state", {}).get("context_store", {})
    if key:
        return {"status": "success", "result": context_store.get(key)}
    return {"status": "success", "result": context_store}

async def create_subtask(ctx: Context, step_id: str, subtask_description: str, context: Optional[Dict] = None, reasoning: str = "") -> Dict:
    """Create a sub-task for a specific step of a long-form task."""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    db = state_manager.PlannerMongoManager()
    try:
        # Get parent task to check for auto-approval flag
        parent_task = await db.get_task(task_id, user_id)
        if not parent_task:
            raise ToolError(f"Parent task with ID {task_id} not found for user.")
        auto_approve = parent_task.get("auto_approve_subtasks", False)

        sub_task_data = {
            "name": subtask_description,
            "description": subtask_description,
            "task_type": "single",  # Sub-tasks are always single-shot
            "original_context": {
                "source": "long_form_subtask",
                "parent_task_id": task_id,
                "parent_step_id": step_id,
                "context": context,
                "auto_approve": auto_approve
            }
        }
        sub_task_id = await db.add_task(user_id, sub_task_data)

        # Link sub_task_id to the parent task's step
        await db.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id, "dynamic_plan.step_id": step_id},
            {"$set": {"dynamic_plan.$.sub_task_id": sub_task_id}}
        )

        refine_and_plan_ai_task.delay(sub_task_id, user_id)
        await state_manager.add_execution_log(task_id, user_id, "subtask_created", {"sub_task_id": sub_task_id, "description": subtask_description}, reasoning)
        return {"status": "success", "result": {"sub_task_id": sub_task_id}}
    finally:
        await db.close()

async def wait_for_response(ctx: Context, waiting_for: str, timeout_minutes: int, max_retries: int = 3, reasoning: str = "") -> Dict:
    """Put the task in waiting state with timeout"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    await waiting_manager.set_waiting_state(task_id, user_id, waiting_for, timeout_minutes, max_retries)
    await state_manager.add_execution_log(task_id, user_id, "waiting_started", {"waiting_for": waiting_for, "timeout_minutes": timeout_minutes}, reasoning)
    return {"status": "success", "message": f"Task is now waiting for '{waiting_for}'."}

async def ask_user_clarification(ctx: Context, question: str, urgency: str = "normal", reasoning: str = "") -> Dict:
    """Suspend task and ask user for clarification"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    db = state_manager.PlannerMongoManager()
    try:
        request_id = str(uuid.uuid4())
        clarification_request = {
            "request_id": request_id,
            "question": question,
            "asked_at": datetime.datetime.now(datetime.timezone.utc),
            "response": None,
            "responded_at": None,
            "status": "pending"
        }
        await db.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {
                "$push": {"clarification_requests": clarification_request},
                "$set": {
                    "orchestrator_state.current_state": "SUSPENDED",
                    "status": "clarification_pending"
                }
            }
        )
        await state_manager.add_execution_log(task_id, user_id, "clarification_requested", {"question": question}, reasoning)
        
        await notify_user(
            user_id,
            f"A long-form task needs your input: {question}",
            task_id,
            notification_type="taskNeedsClarification",
            payload={"request_id": request_id}
        )

        return {"status": "success", "message": "Clarification requested from user. Task is suspended."}
    finally:
        await db.close()

async def mark_step_complete(ctx: Context, step_id: str, result: Dict, reasoning: str) -> Dict:
    """Mark a step as completed and store results"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    await state_manager.mark_step_as_complete(task_id, user_id, step_id, result)
    await state_manager.add_execution_log(task_id, user_id, "step_completed", {"step_id": step_id}, reasoning)
    return {"status": "success", "message": f"Step {step_id} marked as complete."}

async def evaluate_completion(ctx: Context, reasoning: str) -> Dict:
    """Evaluate if the main goal has been achieved"""
    user_id = auth.get_user_id_from_context(ctx)
    task_id = auth.get_task_id_from_context(ctx)
    task = await state_manager.get_task_state(task_id, user_id)
    
    dynamic_plan = task.get("dynamic_plan", [])
    recent_results_data = {}
    if dynamic_plan:
        recent_results_data = dynamic_plan[-1].get("result", {})

    prompt = COMPLETION_EVALUATION_PROMPT.format(
        main_goal=task.get("orchestrator_state", {}).get("main_goal"),
        context_store=json.dumps(task.get("orchestrator_state", {}).get("context_store", {})),
        recent_results=json.dumps(recent_results_data)
    )
    messages = [{'role': 'user', 'content': prompt}]
    final_content_str = ""
    for chunk in run_main_agent(system_message="You are a completion evaluation AI. Respond with JSON.", function_list=[], messages=messages):
        if isinstance(chunk, list) and chunk and chunk[-1].get("role") == "assistant":
            final_content_str = chunk[-1].get("content", "")
            
    decision = json.loads(final_content_str)
    is_complete = decision.get("is_complete", False)

    if is_complete:
        await state_manager.update_orchestrator_state(task_id, user_id, {"current_state": "COMPLETED"})
        await state_manager.add_execution_log(task_id, user_id, "task_completed", {}, reasoning)
        return {"status": "success", "result": {"is_complete": True}}
    else:
        await state_manager.add_execution_log(task_id, user_id, "completion_evaluation", {"is_complete": False}, reasoning)
        return {"status": "success", "result": {"is_complete": False}}
