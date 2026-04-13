# src/server/main/db.py
import os
import datetime
import uuid
import json
import logging
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from typing import Dict, List, Optional, Any, Tuple

# Import config from the current 'main' directory
from main.config import MONGO_URI, MONGO_DB_NAME, ENVIRONMENT
from workers.utils.crypto import encrypt_doc, decrypt_doc, encrypt_field, decrypt_field

DB_ENCRYPTION_ENABLED = ENVIRONMENT == 'stag'

def _decrypt_docs(docs: List[Dict], fields: List[str]):
    if not DB_ENCRYPTION_ENABLED or not docs:
        return
    for doc in docs:
        decrypt_doc(doc, fields)

USER_PROFILES_COLLECTION = "user_profiles"
NOTIFICATIONS_COLLECTION = "notifications"
PROCESSED_ITEMS_COLLECTION = "processed_items_log"
tasks_collection = "tasks"
MESSAGES_COLLECTION = "messages"

SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]

logger = logging.getLogger(__name__)

class MongoManager:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]

        self.user_profiles_collection = self.db[USER_PROFILES_COLLECTION]
        self.notifications_collection = self.db[NOTIFICATIONS_COLLECTION]
        self.processed_items_collection = self.db[PROCESSED_ITEMS_COLLECTION]
        self.tasks_collection = self.db[tasks_collection]
        self.messages_collection = self.db[MESSAGES_COLLECTION]

        print(f"[{datetime.datetime.now()}] [MainServer_MongoManager] Initialized. Database: {MONGO_DB_NAME}")

    async def initialize_db(self):
        print(f"[{datetime.datetime.now()}] [MainServer_DB_INIT] Ensuring indexes for MongoManager collections...")

        collections_with_indexes = {
            self.user_profiles_collection: [
                IndexModel([("user_id", ASCENDING)], unique=True, name="user_id_unique_idx"),
                IndexModel([("userData.last_active_timestamp", DESCENDING)], name="user_last_active_idx"),
                IndexModel([("userData.google_services.gmail.encrypted_refresh_token", ASCENDING)],
                           name="google_gmail_token_idx", sparse=True),
                IndexModel([("userData.onboardingComplete", ASCENDING)], name="user_onboarding_status_idx", sparse=True)
            ],
            self.notifications_collection: [
                IndexModel([("user_id", ASCENDING)], name="notification_user_id_idx"),
                IndexModel([("user_id", ASCENDING), ("notifications.timestamp", DESCENDING)], name="notification_timestamp_idx", sparse=True)
            ],
            self.processed_items_collection: [
                IndexModel([("user_id", ASCENDING), ("service_name", ASCENDING), ("item_id", ASCENDING)], unique=True, name="processed_item_unique_idx_main"),
                IndexModel([("processing_timestamp", DESCENDING)], name="processed_timestamp_idx_main", expireAfterSeconds=2592000) # 30 days
            ],
            self.tasks_collection: [
                IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="task_user_created_idx"),
                IndexModel([("user_id", ASCENDING), ("status", ASCENDING), ("priority", ASCENDING)], name="task_user_status_priority_idx"),
                IndexModel([("status", ASCENDING), ("agent_id", ASCENDING)], name="task_status_agent_idx", sparse=True),
                IndexModel([("task_id", ASCENDING)], unique=True, name="task_id_unique_idx"),
                IndexModel([("name", "text"), ("description", "text")], name="task_text_search_idx"),
            ],
            self.messages_collection: [
                IndexModel([("message_id", ASCENDING)], unique=True, name="message_id_unique_idx"),
                IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)], name="message_user_timestamp_idx"),
                IndexModel([("content", "text")], name="message_content_text_idx"),
            ],
        }

        for collection, indexes in collections_with_indexes.items():
            try:
                await collection.create_indexes(indexes)
                logger.info(f"Indexes ensured for collection: {collection.name}")
            except Exception as e:
                logger.error(f"Index creation failed for {collection.name}: {e}", exc_info=True)


    # --- User Profile Methods ---
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        if not user_id: return None
        doc = await self.user_profiles_collection.find_one({"user_id": user_id})
        if DB_ENCRYPTION_ENABLED and doc and "userData" in doc:
            user_data = doc["userData"]
            SENSITIVE_USER_DATA_FIELDS = ["onboardingAnswers", "personalInfo", "pwa_subscription", "privacyFilters"]
            for field in SENSITIVE_USER_DATA_FIELDS:
                if field in user_data and user_data[field] is not None:
                    user_data[field] = decrypt_field(user_data[field])
        return doc

    async def update_user_profile(self, user_id: str, profile_data: Dict) -> bool:
        if not user_id or not profile_data: return False
        if "_id" in profile_data: del profile_data["_id"]

        if DB_ENCRYPTION_ENABLED:
            SENSITIVE_USER_DATA_FIELDS = ["onboardingAnswers", "personalInfo", "pwa_subscription", "privacyFilters"]
            data_to_update = profile_data.copy()
            for key, value in profile_data.items():
                if key.startswith("userData.") and key.split('.')[1] in SENSITIVE_USER_DATA_FIELDS:
                    data_to_update[key] = encrypt_field(value)
            profile_data = data_to_update

        update_operations = {"$set": {}, "$setOnInsert": {}}
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        for key, value in profile_data.items():
            update_operations["$set"][key] = value

        update_operations["$set"]["last_updated"] = now_utc
        update_operations["$setOnInsert"]["user_id"] = user_id
        update_operations["$setOnInsert"]["createdAt"] = now_utc

        if "userData" not in profile_data and not any(k.startswith("userData.") for k in profile_data):
             update_operations["$setOnInsert"]["userData"] = {}

        for key_to_set in profile_data.keys():
            if key_to_set.startswith("userData.google_services."):
                parts = key_to_set.split('.')
                if len(parts) >= 3:
                    service_name_for_insert = parts[2]

                    user_data_on_insert = update_operations["$setOnInsert"].setdefault("userData", {})
                    google_services_on_insert = user_data_on_insert.setdefault("google_services", {})
                    google_services_on_insert.setdefault(service_name_for_insert, {})
                break

        if not update_operations["$set"]: del update_operations["$set"]
        if not update_operations["$setOnInsert"]: del update_operations["$setOnInsert"]

        if not update_operations.get("$set") and not update_operations.get("$setOnInsert"):
            return True

        result = await self.user_profiles_collection.update_one(
            {"user_id": user_id}, update_operations, upsert=True
        )
        return result.matched_count > 0 or result.upserted_id is not None

    async def update_user_last_active(self, user_id: str) -> bool:
        if not user_id: return False
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        update_payload = {
            "userData.last_active_timestamp": now_utc,
            "last_updated": now_utc
        }
        result = await self.user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_payload,
             "$setOnInsert": {"user_id": user_id, "createdAt": now_utc, "userData": {"last_active_timestamp": now_utc}}},
            upsert=True
        )
        return result.matched_count > 0 or result.upserted_id is not None

    async def get_completed_task_count_for_period(self, user_id: str, start_date: datetime.datetime, end_date: datetime.datetime) -> int:
        """Counts completed tasks for a user within a specific date range."""
        query = {
            "user_id": user_id,
            "status": "completed",
            "updated_at": {
                "$gte": start_date,
                "$lt": end_date
            }
        }
        count = await self.tasks_collection.count_documents(query)
        return count

    async def has_notification_type(self, user_id: str, notification_type: str) -> bool:
        """Checks if a user has ever received a notification of a specific type."""
        if not user_id or not notification_type:
            return False
        # Use find_one for better performance, as we only need to know if at least one exists.
        notification_doc = await self.notifications_collection.find_one(
            {"user_id": user_id, "notifications.type": notification_type}
        )
        return notification_doc is not None

    # --- Notification Methods ---
    async def get_notifications(self, user_id: str) -> List[Dict]:
        if not user_id: return []
        user_doc = await self.notifications_collection.find_one( # noqa: E501
            {"user_id": user_id}, {"notifications": 1}
        )
        notifications_list = user_doc.get("notifications", []) if user_doc else []

        if DB_ENCRYPTION_ENABLED:
            SENSITIVE_NOTIFICATION_FIELDS = ["message", "suggestion_payload"]
            for notification in notifications_list:
                for field in SENSITIVE_NOTIFICATION_FIELDS:
                    if field in notification and notification[field] is not None:
                        notification[field] = decrypt_field(notification[field])

        # Serialize datetime objects before returning, as they are not JSON-serializable by default.
        for notification in notifications_list:
            if isinstance(notification.get("timestamp"), datetime.datetime):
                notification["timestamp"] = notification["timestamp"].isoformat()
        return notifications_list

    async def add_notification(self, user_id: str, notification_data: Dict) -> Optional[Dict]:
        if not user_id or not notification_data: return None
        # Store timestamp as an ISO 8601 string to ensure timezone correctness
        # across all database and application layers.
        notification_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        notification_data["id"] = str(uuid.uuid4())

        if DB_ENCRYPTION_ENABLED:
            SENSITIVE_NOTIFICATION_FIELDS = ["message", "suggestion_payload"]
            for field in SENSITIVE_NOTIFICATION_FIELDS:
                if field in notification_data and notification_data[field] is not None:
                    notification_data[field] = encrypt_field(notification_data[field])

        result = await self.notifications_collection.update_one(
            {"user_id": user_id},
            {"$push": {
                "notifications": {
                    "$each": [notification_data],
                    "$position": 0, # Add to the beginning of the array
                    "$slice": -50
                }
            },
             "$setOnInsert": {"user_id": user_id, "created_at": datetime.datetime.now(datetime.timezone.utc)}},
            upsert=True
        )
        if result.matched_count > 0 or result.upserted_id is not None:
            return notification_data
        return None

    async def delete_notification(self, user_id: str, notification_id: str) -> bool:
        if not user_id or not notification_id: return False
        result = await self.notifications_collection.update_one(
            {"user_id": user_id},
            {"$pull": {"notifications": {"id": notification_id}}}
        )
        return result.modified_count > 0

    async def delete_all_notifications(self, user_id: str):
        """Deletes all notifications for a user by emptying the notifications array."""
        if not user_id:
            return
        # This operation is idempotent. If the user has no notification document,
        # it does nothing, which is the desired outcome.
        await self.notifications_collection.update_one(
            {"user_id": user_id},
            {"$set": {"notifications": []}}
        )

    async def delete_pwa_subscription(self, user_id: str, endpoint: str) -> bool:
        """Deletes a specific PWA push subscription object from a user's profile."""
        if not user_id or not endpoint:
            return False
        result = await self.user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$pull": {"userData.pwa_subscriptions": {"endpoint": endpoint}}}
        )
        logger.info(f"Removed expired PWA subscription for endpoint: {endpoint[-10:]}...")
        return result.modified_count > 0

    async def get_recent_completed_tasks_for_period(self, user_id: str, start_date: datetime.datetime, end_date: datetime.datetime, limit: int = 2) -> List[Dict]:
        """Fetches a few recent completed tasks for a user within a specific date range."""
        query = {
            "user_id": user_id,
            "status": "completed",
            "updated_at": {
                "$gte": start_date,
                "$lt": end_date
            }
        }
        cursor = self.tasks_collection.find(query, {"name": 1}).sort("updated_at", DESCENDING).limit(limit)
        tasks = await cursor.to_list(length=limit)
        decrypt_doc({"tasks": tasks}, ["tasks"]) # Decrypt the name field
        return tasks

    # --- Task Methods ---
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
            # Task type specific fields
            "task_type": task_data.get("task_type", "single"),
        }
    
        # Add type-specific fields based on the new schema
        task_type = task_doc["task_type"]
        if task_type == "swarm":
            task_doc["swarm_details"] = task_data.get("swarm_details", {})
        elif task_type == "long_form":
            task_doc["orchestrator_state"] = task_data.get("orchestrator_state")
            task_doc["dynamic_plan"] = task_data.get("dynamic_plan", [])
            task_doc["clarification_requests"] = task_data.get("clarification_requests", [])
            task_doc["execution_log"] = task_data.get("execution_log", [])
            task_doc["auto_approve_subtasks"] = task_data.get("auto_approve_subtasks", False)

        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.insert_one(task_doc) # noqa: E501
        logger.info(f"Created new task {task_id} (type: {task_doc['task_type']}) for user {user_id} with status 'planning'.")
        return task_id

    async def get_task(self, task_id: str, user_id: str) -> Optional[Dict]:
        """Fetches a single task by its ID, ensuring it belongs to the user."""
        doc = await self.tasks_collection.find_one({"task_id": task_id, "user_id": user_id})
        decrypt_doc(doc, SENSITIVE_TASK_FIELDS)
        return doc

    async def get_all_tasks_for_user(self, user_id: str) -> List[Dict]:
        """Fetches all tasks for a given user."""
        cursor = self.tasks_collection.find({"user_id": user_id}).sort("created_at", -1) # noqa: E501
        docs = await cursor.to_list(length=None)
        _decrypt_docs(docs, SENSITIVE_TASK_FIELDS)
        return docs

    async def update_task(self, task_id: str, user_id: str, updates: Dict) -> bool:
        """Updates an existing task document."""
        updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
        encrypt_doc(updates, SENSITIVE_TASK_FIELDS)
        result = await self.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def add_answers_to_task(self, task_id: str, answers: List[Dict], user_id: str) -> bool:
        """Finds a task and updates its clarifying questions with user answers."""
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
        return await self.update_task(task_id, user_id, {"clarifying_questions": current_questions})

    async def delete_task(self, task_id: str, user_id: str) -> Tuple[bool, List[str]]:
        """
        Deletes a parent task and all its sub-tasks.
        Returns a tuple: (success_flag, list_of_all_deleted_task_ids).
        """
        # 1. Find all sub-tasks to get their IDs for notification cleanup
        sub_task_query = {
            "user_id": user_id,
            "original_context.parent_task_id": task_id
        }
        sub_tasks_to_delete = await self.tasks_collection.find(sub_task_query, {"task_id": 1}).to_list(length=None)
        sub_task_ids = [st["task_id"] for st in sub_tasks_to_delete]

        # 2. Delete sub-tasks if any exist
        if sub_task_ids:
            delete_subtasks_result = await self.tasks_collection.delete_many(sub_task_query)
            logger.info(f"Deleted {delete_subtasks_result.deleted_count} sub-tasks for parent task {task_id}.")

        # 3. Delete the parent task
        delete_parent_result = await self.tasks_collection.delete_one({"task_id": task_id, "user_id": user_id})
        parent_deleted = delete_parent_result.deleted_count > 0

        if not parent_deleted and not sub_task_ids:
            return False, []

        all_deleted_ids = [task_id] if parent_deleted else []
        all_deleted_ids.extend(sub_task_ids)

        return True, all_deleted_ids

    async def decline_task(self, task_id: str, user_id: str) -> str:
        """Declines a task by setting its status to 'declined'."""
        success = await self.update_task(task_id, user_id, {"status": "declined"})
        return "Task declined." if success else None

    async def delete_tasks_by_tool(self, user_id: str, tool_name: str) -> int:
        """
        Deletes all tasks for a user that have a plan step using a specific tool.
        This is used when an integration is disconnected.
        """
        if not user_id or not tool_name:
            return 0
        query = {"user_id": user_id, "runs.plan.tool": tool_name}
        result = await self.tasks_collection.delete_many(query)
        logger.info(f"Deleted {result.deleted_count} tasks for user {user_id} using tool '{tool_name}'.")
        return result.deleted_count

    async def cancel_latest_run(self, task_id: str, user_id: str) -> bool:
        """Pops the last run from the array and reverts the task status to completed."""
        task = await self.get_task(task_id, user_id)
        if not task or not task.get("runs"):
            return False

        runs = task.get("runs", [])
        if not isinstance(runs, list) or not runs:
            return False

        runs.pop() # Remove the last run

        update_payload = {
            "status": "completed",
            "runs": runs
        }
        return await self.update_task(task_id, user_id, update_payload)

    async def delete_notifications_for_task(self, user_id: str, task_id: str):
        """Deletes all notifications associated with a specific task_id for a user."""
        if not user_id or not task_id:
            return
        await self.notifications_collection.update_one(
            {"user_id": user_id},
            {"$pull": {"notifications": {"task_id": task_id}}}
        )
        logger.info(f"Deleted notifications for task {task_id} for user {user_id}.")

    async def rerun_task(self, original_task_id: str, user_id: str) -> Optional[str]:
        """Duplicates a task to be re-run."""
        original_task = await self.get_task(original_task_id, user_id)
        if not original_task:
            return None

        new_task_doc = original_task.copy()
        new_task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # Reset fields for a new run
        if "_id" in new_task_doc:
            del new_task_doc["_id"] # Let Mongo generate a new one
        new_task_doc["task_id"] = new_task_id
        new_task_doc["status"] = "planning"
        new_task_doc["created_at"] = now_utc
        new_task_doc["updated_at"] = now_utc
        new_task_doc["last_execution_at"] = None
        new_task_doc["next_execution_at"] = None

        encrypt_doc(new_task_doc, SENSITIVE_TASK_FIELDS)
        await self.tasks_collection.insert_one(new_task_doc)
        return new_task_id

    async def create_initial_task(self, user_id: str, name: str, description: str, action_items: list, topics: list, original_context: dict, source_event_id: str) -> Dict:
        """Creates an initial task document when an action item is first processed."""
        task_id = str(uuid.uuid4())
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        task_doc = {
            "task_id": task_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "status": "planning",
            "assignee": "ai",
            "priority": 1,
            "plan": [],
            "runs": [],
            "original_context": original_context,
            "source_event_id": source_event_id,
            "created_at": now_utc,
            "updated_at": now_utc,
            "chat_history": [],
        }

        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.tasks_collection.insert_one(task_doc)
        logger.info(f"Created initial task {task_id} for user {user_id}")
        return task_doc

    async def update_task_with_plan(self, task_id: str, user_id: str, plan_data: dict, is_change_request: bool = False, chat_history: Optional[List[Dict]] = None):
        """Updates a task with a generated plan and sets it to pending approval."""
        plan_steps = plan_data.get("plan", [])

        update_doc = {
            "status": "approval_pending",
            "plan": plan_steps,
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
        }

        if chat_history is not None:
            update_doc["chat_history"] = chat_history

        if not is_change_request:
            name = plan_data.get("name", "Proactively generated plan")
            update_doc["name"] = name
            update_doc["description"] = plan_data.get("description", "")

        encrypt_doc(update_doc, SENSITIVE_TASK_FIELDS)

        result = await self.tasks_collection.update_one(
            {"task_id": task_id, "user_id": user_id},
            {"$set": update_doc}
        )
        logger.info(f"Updated task {task_id} with a generated plan. Matched: {result.matched_count}")

    async def save_plan_as_task(self, user_id: str, name: str, description: str, plan: list, original_context: dict, source_event_id: str) -> str:
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
        encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)
        await self.tasks_collection.insert_one(task_doc)
        logger.info(f"Saved new plan with task_id: {task_id} for user: {user_id}")
        return task_id

    # --- Message Methods ---
    async def add_message(self, user_id: str, role: str, content: str, message_id: Optional[str] = None, turn_steps: Optional[List[Dict]] = None) -> Dict:
        """
        Adds a single message to the messages collection.
        If a message_id is provided, it's used. Otherwise, a new one is generated.
        For user messages with a provided ID, it prevents duplicate insertions.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        final_message_id = message_id if message_id else str(uuid.uuid4())

        # Prevent duplicate user messages if client retries
        if role == "user" and message_id:
            existing = await self.messages_collection.find_one({"message_id": final_message_id, "user_id": user_id})
            if existing:
                logger.info(f"Message with ID {final_message_id} already exists for user {user_id}. Skipping.")
                return existing

        message_doc = {
            "message_id": final_message_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "is_summarized": False,
            "summary_id": None,
        }

        # --- CHANGED --- Add the new structured turn_steps field if provided.
        if turn_steps:
            message_doc["turn_steps"] = turn_steps

        SENSITIVE_MESSAGE_FIELDS = ["content", "turn_steps"]
        encrypt_doc(message_doc, SENSITIVE_MESSAGE_FIELDS)

        await self.messages_collection.insert_one(message_doc)
        logger.info(f"Added message for user {user_id} with role {role}")
        return message_doc

    async def get_message_history(self, user_id: str, limit: int, before_timestamp_iso: Optional[str] = None) -> List[Dict]:
        """Fetches a paginated history of messages for a user."""
        query = {"user_id": user_id}
        if before_timestamp_iso:
            try:
                # Ensure the timestamp is parsed correctly as UTC
                before_timestamp = datetime.datetime.fromisoformat(before_timestamp_iso.replace("Z", "+00:00"))
                query["timestamp"] = {"$lt": before_timestamp}
            except (ValueError, TypeError):
                logger.warning(f"Invalid before_timestamp format: {before_timestamp_iso}, ignoring.")
                pass

        cursor = self.messages_collection.find(query).sort("timestamp", DESCENDING).limit(limit)
        messages = await cursor.to_list(length=limit)

        SENSITIVE_MESSAGE_FIELDS = ["content", "turn_steps"]
        _decrypt_docs(messages, SENSITIVE_MESSAGE_FIELDS)

        # Serialize datetime and ObjectId objects for JSON response
        for msg in messages:
            if isinstance(msg.get("_id"), ObjectId):
                msg["_id"] = str(msg["_id"])
            if isinstance(msg.get("timestamp"), datetime.datetime):
                msg["timestamp"] = msg["timestamp"].isoformat()

        return messages

    async def delete_message(self, user_id: str, message_id: str) -> bool:
        """Deletes a single message by its ID for a specific user."""
        if not user_id or not message_id:
            return False
        result = await self.messages_collection.delete_one({"user_id": user_id, "message_id": message_id})
        return result.deleted_count > 0

    async def delete_all_messages(self, user_id: str) -> int:
        """Deletes all messages for a specific user."""
        if not user_id:
            return 0
        result = await self.messages_collection.delete_many({"user_id": user_id})
        logger.info(f"Deleted {result.deleted_count} messages for user {user_id}.")
        return result.deleted_count

    async def close(self):
        if self.client:
            self.client.close()
            print(f"[{datetime.datetime.now()}] [MainServer_MongoManager] MongoDB connection closed.")