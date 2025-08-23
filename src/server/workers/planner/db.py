import uuid
import datetime
import logging
import motor.motor_asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid

from workers.planner.config import MONGO_URI, MONGO_DB_NAME, INTEGRATIONS_CONFIG, ENVIRONMENT
from workers.utils.crypto import encrypt_doc, decrypt_doc


logger = logging.getLogger(__name__)

def get_date_from_text(text: str) -> str:
    """Extracts YYYY-MM-DD from text, defaults to today."""
    match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if match:
        return match.group(1)
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')

logger = logging.getLogger(__name__)

def get_all_mcp_descriptions() -> Dict[str, str]:
    """
    Creates a dictionary of all available services and their high-level descriptions
    from the main server's integration config.
    """
    if not INTEGRATIONS_CONFIG:
        logging.warning("INTEGRATIONS_CONFIG is empty. No tools will be available to the planner.")
        return {}
    
    mcp_descriptions = {}
    for name, config in INTEGRATIONS_CONFIG.items():
        # The planner agent should not have access to the tasks MCP,
        # as it can lead to recursive loops. The tasks MCP is for the chat agent.
        if name == "tasks":
            continue
        display_name = config.get("display_name")
        description = config.get("description")
        if display_name and description:
            # Use the simple name (e.g., 'gmail') as the key for the planner
            mcp_descriptions[name] = description
            
    return mcp_descriptions


class PlannerMongoManager:  # noqa: E501
    """A MongoDB manager for the planner worker."""
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.user_profiles_collection = self.db["user_profiles"]
        self.tasks_collection = self.db["tasks"]
        self.messages_collection = self.db["messages"]
        logger.info("PlannerMongoManager initialized.")

    async def add_task(self, user_id: str, task_data: dict) -> str:
        """Creates a new task document and returns its ID."""
        task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        schedule = task_data.get("schedule")
        if isinstance(schedule, str):
            try:
                schedule = json.loads(schedule)
            except json.JSONDecodeError:
                schedule = None

        task_doc = {
            "task_id": task_id,
            "user_id": user_id,
            "name": task_data.get("name", "New Task"),
            "description": task_data.get("description", ""),
            "status": "planning",
            "assignee": "ai",
            "priority": task_data.get("priority", 1),
            "plan": [],
            "runs": [],
            "schedule": schedule,
            "enabled": True,
            "original_context": task_data.get("original_context", {"source": "manual_creation"}),
            "created_at": now_utc,
            "updated_at": now_utc,
            "chat_history": [],
            "next_execution_at": None,
            "last_execution_at": None,
            "task_type": task_data.get("task_type", "single"),
        }

        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.insert_one(task_doc)
        logger.info(f"Created new task {task_id} (type: {task_doc['task_type']}) for user {user_id} with status 'planning'.")
        return task_id

    async def create_initial_task(self, user_id: str, name: str, description: str, action_items: list, topics: list, original_context: dict, source_event_id: str) -> Dict:
        """Creates an initial task document when an action item is first processed."""
        task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        task_doc = {
            "task_id": task_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "status": "planning", # As per spec
            "assignee": "ai", # Proactive tasks are assigned to AI
            "priority": 1,
            "plan": [],
            "runs": [],
            "original_context": original_context,
            "source_event_id": source_event_id,
            "created_at": now_utc,
            "updated_at": now_utc,
            "chat_history": [],
        }

        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.insert_one(task_doc)
        logger.info(f"Created initial task {task_id} for user {user_id}")
        return task_doc

    async def update_task_field(self, task_id: str, user_id: str, fields: dict):
        """Updates specific fields of a task document."""
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        # The encrypt_doc function modifies the dictionary in place
        encrypt_doc(fields, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {"$set": fields}
        )
        logger.info(f"Updated fields for task {task_id}: {list(fields.keys())}")

    async def update_task_with_plan(self, task_id: str, plan_data: dict, is_change_request: bool = False, chat_history: Optional[List[Dict]] = None):
        """Updates a task with a generated plan and sets it to pending approval."""
        plan_steps = plan_data.get("plan", [])

        update_doc = {
            "status": "approval_pending",
            "plan": plan_steps,
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        }

        if chat_history is not None:
            update_doc["chat_history"] = chat_history

        # Only set the main description for the very first run.
        if not is_change_request:
            name = plan_data.get("name", "Proactively generated plan")
            update_doc["name"] = name
            update_doc["description"] = plan_data.get("description", "")

        # Encrypt all sensitive fields at once before updating
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "chat_history"]
        encrypt_doc(update_doc, SENSITIVE_TASK_FIELDS)

        result = await self.tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": update_doc}
        )
        logger.info(f"Updated task {task_id} with a generated plan. Matched: {result.matched_count}")

    async def get_task(self, task_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """Fetches a single task by its ID."""
        query = {"task_id": task_id}
        if user_id:
            query["user_id"] = user_id
        doc = await self.tasks_collection.find_one(query)
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        decrypt_doc(doc, SENSITIVE_TASK_FIELDS)
        return doc

    async def update_task_status(self, task_id: str, status: str, details: Optional[Dict] = None):
        """Updates the status and optionally other fields of a task."""
        update_doc = {"status": status, "updated_at": datetime.datetime.now(datetime.timezone.utc)}
        if details:
            if "error" in details:
                update_doc["error"] = details["error"]

        SENSITIVE_TASK_FIELDS = ["error"]
        encrypt_doc(update_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.update_one({"task_id": task_id}, {"$set": update_doc})
        logger.info(f"Updated status of task {task_id} to {status}")

    async def save_plan_as_task(self, user_id: str, name: str, description: str, plan: list, original_context: dict, source_event_id: str):
        """Saves a generated plan to the tasks collection for user approval."""
        task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        task_doc = {
            "task_id": task_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "status": "approval_pending",
            "priority": 1,
            "plan": plan,
            "runs": [],
            "original_context": original_context,
            "source_event_id": source_event_id,
            "created_at": now_utc,
            "updated_at": now_utc,
            "agent_id": "planner_agent"
        }

        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "original_context", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.insert_one(task_doc)
        logger.info(f"Saved new plan with task_id: {task_id} for user: {user_id}")
        return task_id

    async def update_chat(self, user_id: str, chat_id: str, updates: Dict) -> bool:
        """Updates fields of a chat document, like the title."""
        updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
        result = await self.messages_collection.update_one(
            {"user_id": user_id, "chat_id": chat_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("Planner MongoDB connection closed.")