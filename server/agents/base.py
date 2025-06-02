import uuid
import asyncio
import datetime
import os

from typing import Dict, List, Optional, Tuple, Union

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument # MODIFIED: Added ReturnDocument
from pymongo.results import UpdateResult, DeleteResult
from server.agents.functions import send_email, reply_email
# Global lock for thread-safe access to task data (Original, kept but unused by new TaskQueue)
task_lock = asyncio.Lock()

# Path to the JSON file for task persistence (Original, kept but unused by new TaskQueue)
TASKS_FILE = "agentic_operations.json"

# --- MongoDB Configuration ---
# These can be moved to environment variables or a config file
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_agent_db") # Changed DB name slightly
TASKS_COLLECTION_NAME = "agent_tasks"


class TaskQueue:
    def __init__(self): # Removed tasks_file parameter
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.collection = self.db[TASKS_COLLECTION_NAME]
        # self.tasks = [] # Replaced by MongoDB
        # self.task_id_counter = 0 # Replaced by UUIDs / MongoDB's _id
        # self.lock = asyncio.Lock() # MongoDB handles concurrency at DB level for these ops
        self.current_task_execution = None  # Kept as per original structure

    async def initialize_db(self):
        """
        Initializes the database by ensuring necessary indexes are created.
        Should be called once on application startup.
        """
        # Index for unique task_id (globally, if needed, or combined with user_id)
        # For true unique task_id: IndexModel([("task_id", ASCENDING)], unique=True)
        # However, if task_id is just a UUID, uniqueness is practically guaranteed.
        # Scoped queries will primarily use user_id.

        indexes = [
            IndexModel([("user_id", ASCENDING), ("task_id", ASCENDING)], name="user_task_id_idx", unique=True),
            IndexModel([("status", ASCENDING), ("priority", ASCENDING), ("created_at", ASCENDING)], name="global_pending_tasks_idx"), # For get_next_task
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING)], name="user_status_idx"),
            IndexModel([("user_id", ASCENDING), ("created_at", ASCENDING)], name="user_created_at_idx"),
            IndexModel([("status", ASCENDING), ("completed_at", ASCENDING)], name="global_completed_tasks_idx") # For cleanup
        ]
        try:
            await self.collection.create_indexes(indexes)
            print(f"[DB_INIT] Indexes ensured for collection: {self.collection.name}")
        except Exception as e:
            print(f"[DB_ERROR] Failed to create indexes for {self.collection.name}: {e}")

    async def get_task_by_id(self, user_id: str, task_id: str) -> Optional[Dict]:
        """Retrieves a specific task by its ID, scoped by user_id."""
        if not user_id or not task_id: # Basic validation
            return None
        return await self.collection.find_one({"user_id": user_id, "task_id": task_id})

    async def add_task(self, user_id: str, chat_id: str, description: str, priority: int, 
                       username: str,
                       use_personal_context: bool, internet: str) -> str:
        """Add a new task to the queue, scoped by user_id."""
        # Note: Original signature updated to include user_id, as app.py passes it.
        task_document_id = str(uuid.uuid4())
        task = {
            "task_id": task_document_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "description": description,
            "priority": priority,
            "status": "pending", # Initial status
            "username": username,
            "use_personal_context": use_personal_context,
            "internet": internet,
            "created_at": datetime.datetime.now(datetime.timezone.utc), # Store as datetime object
            "completed_at": None,
            "result": None,
            "error": None,
            "approval_data": None
        }
        await self.collection.insert_one(task)
        return task_document_id

    async def get_next_task(self) -> Optional[Dict]:
        """
        Get the next pending task with the highest priority from any user (global queue).
        Atomically updates the task status to "processing".
        """
        # Sort by priority (ascending) then by creation time (oldest first)
        return await self.collection.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "processing", "processing_started_at": datetime.datetime.now(datetime.timezone.utc)}},
            sort=[("priority", ASCENDING), ("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER # Get the updated document
        )

    async def complete_task(self, user_id: str, task_id: str, result: Optional[str] = None, 
                            error: Optional[str] = None, status: str = "completed"):
        """
        Mark a task with a final status (completed, error, cancelled) and save the result/error.
        Scoped by user_id.
        """
        update_doc = {
            "status": status,
            "result": result,
            "error": error,
            "completed_at": datetime.datetime.now(datetime.timezone.utc),
            "$unset": {"approval_data": "", "processing_started_at": ""}
        }
        # Remove None values for result/error if not provided, to avoid overwriting with None
        # if they were already set by some intermediate step (though less likely here)
        if result is None:
            del update_doc["result"]
        if error is None:
            del update_doc["error"]

        await self.collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {k: v for k,v in update_doc.items() if k != "$unset"}, 
             "$unset": update_doc["$unset"]} # Apply $set and $unset in one operation
        )


    async def set_task_approval_pending(self, user_id: str, task_id: str, approval_data: Dict):
        """Set a task to 'approval_pending' status and store approval data. Scoped by user_id."""
        # Note: `app.py` will need to be updated to pass `user_id`.
        result: UpdateResult = await self.collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"status": "approval_pending", "approval_data": approval_data}}
        )
        if result.matched_count == 0:
            # This matches original behavior of not raising error if task not found,
            # but consider if ValueError should be raised.
            print(f"Warning: Task {task_id} for user {user_id} not found during set_task_approval_pending.")
            # If strict error handling is needed: raise ValueError(f"Task {task_id} for user {user_id} not found.")


    async def approve_task(self, user_id: str, task_id: str) -> Optional[str]:
        """Approve a task, execute the tool with stored parameters, and complete it. Scoped by user_id."""
        # Note: Original signature updated to include user_id.
        task = await self.collection.find_one(
            {"user_id": user_id, "task_id": task_id, "status": "approval_pending"}
        )
        if not task:
            raise ValueError(f"Task {task_id} for user {user_id} not found or not in approval_pending status.")

        approval_data = task.get("approval_data")
        if not approval_data:
            await self.complete_task(user_id, task_id, error="Approval data missing", status="error")
            raise ValueError(f"Approval data missing for task {task_id}")

        tool_name = approval_data.get("tool_name")
        parameters = approval_data.get("parameters", {})
        
        tool_result_str: Optional[str] = None
        try:
            if tool_name == "send_email":
                tool_result = await send_email(**parameters) # Assuming send_email returns something serializable or None
                tool_result_str = str(tool_result) if tool_result is not None else "Email sent."
            elif tool_name == "reply_email":
                tool_result = await reply_email(**parameters) # Assuming reply_email returns something serializable or None
                tool_result_str = str(tool_result) if tool_result is not None else "Reply sent."
            else:
                # Original code raises ValueError here, which is then caught by app.py.
                # Mark task as error before raising.
                err_msg = f"Unknown tool for approval: {tool_name}"
                await self.complete_task(user_id, task_id, error=err_msg, status="error")
                raise ValueError(err_msg)
            
            await self.complete_task(user_id, task_id, result=tool_result_str, status="completed")
            return tool_result_str
        except Exception as e:
            # If tool execution itself fails after finding the tool
            await self.complete_task(user_id, task_id, error=f"Error executing approved tool {tool_name}: {str(e)}", status="error")
            raise # Re-raise the exception

    async def update_task(self, user_id: str, task_id: str, description: str, priority: int) -> bool:
        """Update a task's description and priority. Scoped by user_id."""
        # Note: Original signature updated to include user_id.
        task = await self.collection.find_one({"user_id": user_id, "task_id": task_id})
        if not task:
            raise ValueError(f"Task with id {task_id} for user {user_id} not found.")
        
        if task["status"] not in ["pending", "processing"]:
            # This matches original behavior of raising ValueError.
            raise ValueError(f"Cannot update task {task_id} with status: {task['status']}.")

        result: UpdateResult = await self.collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"description": description, "priority": priority}}
        )
        return result.modified_count > 0


    async def delete_task(self, user_id: str, task_id: str) -> bool: # Renamed taskID to task_id
        """Delete a task by its ID. Scoped by user_id."""
        result: DeleteResult = await self.collection.delete_one({"user_id": user_id, "task_id": task_id})
        if result.deleted_count == 0:
            raise ValueError(f"Task with id {task_id} for user {user_id} not found for deletion.")
        return True

    async def get_tasks_for_user(self, user_id: str) -> List[Dict]:
        """Return a list of all tasks for a specific user, newest first."""
        if not user_id: # Handle cases where user_id might be None or empty
            return []
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", DESCENDING)
        return await cursor.to_list(length=None) # Fetches all matching documents


    async def delete_old_completed_tasks(self, hours_threshold: int = 1):
        """Delete completed tasks older than the specified hours threshold (globally for all users)."""
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_threshold)
        query = {
            "status": "completed",
            "completed_at": {"$lt": cutoff_time}
        }
        result: DeleteResult = await self.collection.delete_many(query)
        print(f"[DB_CLEANUP] Deleted {result.deleted_count} old completed tasks.")

        stuck_processing_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_threshold * 4) # e.g., 4x threshold
        stuck_query = {
            "status": "processing",
            "processing_started_at": {"$lt": stuck_processing_cutoff}
        }
        # Option 2: Mark them as error
        stuck_update_result: UpdateResult = await self.collection.update_many(
            stuck_query,
            {"$set": {
                "status": "error", 
                "error": "Task timed out during processing cleanup.",
                "completed_at": datetime.datetime.now(datetime.timezone.utc)
            }}
        )
        if stuck_update_result.modified_count > 0:
            print(f"[DB_CLEANUP] Marked {stuck_update_result.modified_count} tasks stuck in processing as 'error'.")