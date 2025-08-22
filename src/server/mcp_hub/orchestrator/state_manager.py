import logging
import datetime
from typing import Dict, Any, List
from workers.planner.db import PlannerMongoManager

logger = logging.getLogger(__name__)

async def get_task_state(task_id: str, user_id: str) -> Dict:
    db = PlannerMongoManager()
    try:
        task = await db.get_task(task_id)
        if not task or task.get("user_id") != user_id:
            raise ValueError("Task not found or access denied.")
        return task
    finally:
        await db.close()

async def update_orchestrator_state(task_id: str, user_id: str, state_updates: Dict):
    db = PlannerMongoManager()
    try:
        payload = {f"orchestrator_state.{key}": value for key, value in state_updates.items()}

        # Also update main status if orchestrator state is changing
        if 'current_state' in state_updates:
            orchestrator_state = state_updates['current_state']
            status_map = {
                "COMPLETED": "completed",
                "FAILED": "error",
                "SUSPENDED": "clarification_pending",
                "PLANNING": "processing",
                "ACTIVE": "processing",
                "WAITING": "processing"
            }
            if orchestrator_state in status_map:
                payload['status'] = status_map[orchestrator_state]
        await db.update_task_field(task_id, user_id, payload)
    finally:
        await db.close()

async def add_execution_log(task_id: str, user_id: str, action: str, details: Dict, reasoning: str):
    db = PlannerMongoManager()
    try:
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "action": action,
            "details": details,
            "agent_reasoning": reasoning
        }
        await db.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {"$push": {"execution_log": log_entry}}
        )
    finally:
        await db.close()

async def update_dynamic_plan(task_id: str, user_id: str, new_steps: List[Dict], goal: str = None):
    db = PlannerMongoManager()
    try:
        update = {"dynamic_plan": new_steps}
        if goal:
            update["orchestrator_state.main_goal"] = goal
        await db.update_task_field(task_id, user_id, update)
    finally:
        await db.close()

async def update_context_store(task_id: str, user_id: str, key: str, value: Any):
    db = PlannerMongoManager()
    try:
        await db.update_task_field(task_id, user_id, {f"orchestrator_state.context_store.{key}": value})
    finally:
        await db.close()

async def mark_step_as_complete(task_id: str, user_id: str, step_id: str, result: Dict):
    db = PlannerMongoManager()
    try:
        # This requires finding the step in the array and updating it.
        # Using positional operator '$' is ideal here.
        await db.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id, "dynamic_plan.step_id": step_id},
            {
                "$set": {
                    "dynamic_plan.$.status": "completed",
                    "dynamic_plan.$.result": result,
                    "dynamic_plan.$.completed_at": datetime.datetime.now(datetime.timezone.utc)
                }
            }
        )
    finally:
        await db.close()
