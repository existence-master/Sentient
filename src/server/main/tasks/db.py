import motor.motor_asyncio
import datetime
import uuid
import json
from json_extractor import JsonExtractor
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TaskDatabase:
    def __init__(self, mongo_uri: str, db_name: str = "task_manager_db"):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.task_collection = self.db["tasks"]
        self.user_collection = self.db["users"] # Assuming a user collection might exist or be needed

    async def _ensure_indexes(self):
        """Ensures that necessary indexes are created for efficient querying."""
        try:
            await self.task_collection.create_index("task_id", unique=True)
            await self.task_collection.create_index("user_id")
            await self.task_collection.create_index("status")
            await self.task_collection.create_index("next_execution_at")
            logger.info("Task indexes ensured.")
        except Exception as e:
            logger.error(f"Error ensuring task indexes: {e}")

    async def create_task(self, user_id: str, task_data: dict) -> dict:
        """Creates a new task document, typically from a manual user request."""
        task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        schedule = task_data.get("schedule")
        if isinstance(schedule, str):
            schedule = JsonExtractor.extract_valid_json(schedule)

        task_doc = {
            "task_id": task_id,
            "user_id": user_id,
            "description": task_data.get("description", "New Task"),
            "status": "planning",  # As per spec for new tasks
            "priority": task_data.get("priority", 1),
            "plan": [],
            "schedule": schedule,
            "enabled": True, 
            "original_context": {"source": "manual_creation"},
            "created_at": now_utc,
            "updated_at": now_utc,
            "clarifying_questions": [],
            "progress_updates": [],
            "result": None,
            "error": None,
            "next_execution_at": None,
            "last_execution_at": None,
            "task_type": task_data.get("task_type", "single"),
            "swarm_details": task_data.get("swarm_details"), # Will be None for single tasks
            "long_form_details": task_data.get("long_form_details")
        }
        
        await self.task_collection.insert_one(task_doc)
        logger.info(f"Created new manual task {task_id} for user {user_id} with status 'planning'.")
        return task_doc

    async def get_task(self, task_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """Fetches a single task by its ID, optionally filtered by user_id."""
        query = {"task_id": task_id}
        if user_id:
            query["user_id"] = user_id
        return await self.task_collection.find_one(query)

    async def get_tasks_for_user(self, user_id: str, status: Optional[str] = None) -> List[Dict]:
        """Fetches all tasks for a given user, optionally filtered by status."""
        query = {"user_id": user_id}
        if status:
            query["status"] = status
        
        cursor = self.task_collection.find(query).sort("created_at", -1)
        return await cursor.to_list(length=None)

    async def update_task(self, task_id: str, updates: Dict) -> bool:
        """Updates an existing task document."""
        # Ensure updated_at is always set
        updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
        
        result = await self.task_collection.update_one(
            {"task_id": task_id},
            {"$set": updates}
        )

        return result.modified_count > 0 or result.upserted_id is not None

    async def add_answers_to_task(self, task_id: str, answers: List[Dict], user_id: str) -> bool: # noqa
        """Finds a task and updates its top-level clarifying questions with user answers."""
        task = await self.get_task(task_id, user_id)
        if not task:
            return False

        current_questions = task.get("clarifying_questions", [])
        if not current_questions:
            # Fallback for legacy tasks where questions might be in the last run
            if task.get("runs"):
                current_questions = task["runs"][-1].get("clarifying_questions", [])
            if not current_questions:
                logger.warning(f"add_answers_to_task called for task {task_id}, but no questions found.")
                return False # Nothing to update

        answer_map = {ans.get("question_id"): ans.get("answer_text") for ans in answers}

        for question in current_questions:
            q_id = question.get("question_id")
            if q_id in answer_map:
                question["answer"] = answer_map[q_id]
        # Always write back to the top-level field for consistency
        return await self.update_task(task_id, {"clarifying_questions": current_questions})