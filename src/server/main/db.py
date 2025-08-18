# src/server/main/db.py
import os
import datetime
import uuid
import json
import logging
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from typing import Dict, List, Optional, Any, Tuple

# Import config from the current 'main' directory
from main.config import MONGO_URI, MONGO_DB_NAME, ENVIRONMENT
from main.auth.utils import aes_encrypt, aes_decrypt

DB_ENCRYPTION_ENABLED = ENVIRONMENT == 'stag'

def _datetime_serializer(obj):
    """JSON serializer for objects not serializable by default json code, like datetime."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def _encrypt_field(data: Any) -> Any:
    if not DB_ENCRYPTION_ENABLED or data is None:
        return data
    data_str = json.dumps(data, default=_datetime_serializer)
    return aes_encrypt(data_str)

def _decrypt_field(data: Any) -> Any:
    if not DB_ENCRYPTION_ENABLED or data is None or not isinstance(data, str):
        return data
    try:
        decrypted_str = aes_decrypt(data)
        return json.loads(decrypted_str)
    except Exception:
        return data

def _encrypt_doc(doc: Dict, fields: List[str]):
    if not DB_ENCRYPTION_ENABLED or not doc:
        return
    for field in fields:
        if field in doc and doc[field] is not None:
            doc[field] = _encrypt_field(doc[field])

def _decrypt_doc(doc: Optional[Dict], fields: List[str]):
    if not DB_ENCRYPTION_ENABLED or not doc:
        return
    for field in fields:
        if field in doc and doc[field] is not None:
            doc[field] = _decrypt_field(doc[field])

def _decrypt_docs(docs: List[Dict], fields: List[str]):
    if not DB_ENCRYPTION_ENABLED or not docs:
        return
    for doc in docs:
        _decrypt_doc(doc, fields)

USER_PROFILES_COLLECTION = "user_profiles" 
NOTIFICATIONS_COLLECTION = "notifications" 
POLLING_STATE_COLLECTION = "polling_state_store" 
DAILY_USAGE_COLLECTION = "daily_usage"
PROCESSED_ITEMS_COLLECTION = "processed_items_log" 
TASK_COLLECTION = "tasks"
MESSAGES_COLLECTION = "messages"

logger = logging.getLogger(__name__)

class MongoManager:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        
        self.user_profiles_collection = self.db[USER_PROFILES_COLLECTION]
        self.notifications_collection = self.db[NOTIFICATIONS_COLLECTION]
        self.polling_state_collection = self.db[POLLING_STATE_COLLECTION]
        self.daily_usage_collection = self.db[DAILY_USAGE_COLLECTION]
        self.processed_items_collection = self.db[PROCESSED_ITEMS_COLLECTION]
        self.task_collection = self.db[TASK_COLLECTION]
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
            self.polling_state_collection: [
                IndexModel([("user_id", ASCENDING), ("service_name", ASCENDING), ("poll_type", ASCENDING)], unique=True, name="polling_user_service_type_unique_idx"),
                IndexModel([
                    ("is_enabled", ASCENDING), ("next_scheduled_poll_time", ASCENDING),
                    ("is_currently_polling", ASCENDING), ("error_backoff_until_timestamp", ASCENDING)
                ], name="polling_due_tasks_idx"),
                IndexModel([("is_currently_polling", ASCENDING), ("last_attempted_poll_timestamp", ASCENDING)], name="polling_stale_locks_idx")
            ],
            self.daily_usage_collection: [
                IndexModel([("user_id", ASCENDING), ("date", DESCENDING)], unique=True, name="usage_user_date_unique_idx"),
                IndexModel([("date", DESCENDING)], name="usage_date_idx", expireAfterSeconds=2 * 24 * 60 * 60) # Expire docs after 2 days
            ],
            self.processed_items_collection: [ 
                IndexModel([("user_id", ASCENDING), ("service_name", ASCENDING), ("item_id", ASCENDING)], unique=True, name="processed_item_unique_idx_main"),
                IndexModel([("processing_timestamp", DESCENDING)], name="processed_timestamp_idx_main", expireAfterSeconds=2592000) # 30 days
            ],
            self.task_collection: [
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
                print(f"[{datetime.datetime.now()}] [MainServer_DB_INIT] Indexes ensured for: {collection.name}")
            except Exception as e:
                print(f"[{datetime.datetime.now()}] [MainServer_DB_ERROR] Index creation for {collection.name}: {e}")


    # --- User Profile Methods ---
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        if not user_id: return None
        doc = await self.user_profiles_collection.find_one({"user_id": user_id})
        if DB_ENCRYPTION_ENABLED and doc and "userData" in doc:
            user_data = doc["userData"]
            SENSITIVE_USER_DATA_FIELDS = ["onboardingAnswers", "personalInfo", "pwa_subscription", "privacyFilters"]
            for field in SENSITIVE_USER_DATA_FIELDS:
                if field in user_data and user_data[field] is not None:
                    user_data[field] = _decrypt_field(user_data[field])
        return doc

    async def update_user_profile(self, user_id: str, profile_data: Dict) -> bool:
        if not user_id or not profile_data: return False
        if "_id" in profile_data: del profile_data["_id"] 

        if DB_ENCRYPTION_ENABLED:
            SENSITIVE_USER_DATA_FIELDS = ["onboardingAnswers", "personalInfo", "pwa_subscription", "privacyFilters"]
            data_to_update = profile_data.copy()
            for key, value in profile_data.items():
                if key.startswith("userData.") and key.split('.')[1] in SENSITIVE_USER_DATA_FIELDS:
                    data_to_update[key] = _encrypt_field(value)
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

    # --- Usage Tracking Methods ---
    async def get_or_create_daily_usage(self, user_id: str) -> Dict[str, Any]:
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        usage_doc = await self.daily_usage_collection.find_one_and_update(
            {"user_id": user_id, "date": today_str},
            {"$setOnInsert": {"user_id": user_id, "date": today_str}},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        return usage_doc

    async def increment_daily_usage(self, user_id: str, feature: str, amount: int = 1):
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        await self.daily_usage_collection.update_one(
            {"user_id": user_id, "date": today_str},
            {"$inc": {feature: amount}},
            upsert=True
        )

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
                        notification[field] = _decrypt_field(notification[field])

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
                    notification_data[field] = _encrypt_field(notification_data[field])

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

    async def find_and_action_suggestion_notification(self, user_id: str, notification_id: str) -> Optional[Dict]:
        """
        Atomically finds a proactive_suggestion notification, marks it as actioned, and returns it.
        This prevents race conditions from multiple clicks.
        """
        if not user_id or not notification_id:
            return None

        # Find the document containing the notification
        user_notif_doc = await self.notifications_collection.find_one(
            {"user_id": user_id, "notifications.id": notification_id}
        )
        if not user_notif_doc:
            return None

        # Atomically update the specific notification inside the array
        result = await self.notifications_collection.find_one_and_update(
            {"user_id": user_id, "notifications.id": notification_id, "notifications.is_actioned": False},
            {"$set": {"notifications.$.is_actioned": True}},
        )

        return user_notif_doc if result else None

    # --- Polling State Store Methods ---
    async def get_polling_state(self, user_id: str, service_name: str) -> Optional[Dict[str, Any]]:
        if not user_id or not service_name: return None
        return await self.polling_state_collection.find_one(
            {"user_id": user_id, "service_name": service_name}
        )

    async def update_polling_state(self, user_id: str, service_name: str, poll_type: str, state_data: Dict[str, Any]) -> bool:
        if not user_id or not service_name or not poll_type or state_data is None: return False
        for key, value in state_data.items(): 
            if isinstance(value, datetime.datetime):
                state_data[key] = value.replace(tzinfo=datetime.timezone.utc) if value.tzinfo is None else value.astimezone(datetime.timezone.utc)
        
        # Prevent conflict errors by not trying to $set fields that are part of the unique index
        # or are immutable.
        if '_id' in state_data: del state_data['_id']
        if 'user_id' in state_data: del state_data['user_id']
        if 'service_name' in state_data: del state_data['service_name']
        if 'poll_type' in state_data: del state_data['poll_type']
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        state_data["last_updated_at"] = now_utc

        result = await self.polling_state_collection.update_one(
            {"user_id": user_id, "service_name": service_name, "poll_type": poll_type},
            {"$set": state_data, "$setOnInsert": {"created_at": now_utc, "user_id": user_id, "service_name": service_name, "poll_type": poll_type}},
            upsert=True
        )
        return result.matched_count > 0 or result.upserted_id is not None

    async def delete_polling_state_by_service(self, user_id: str, service_name: str) -> int:
        """Deletes all polling state documents for a user and a specific service."""
        if not user_id or not service_name:
            return 0
        query = {"user_id": user_id, "service_name": service_name}
        result = await self.polling_state_collection.delete_many(query)
        return result.deleted_count

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
            # NEW FIELDS
            "task_type": task_data.get("task_type", "single"),
            "swarm_details": task_data.get("swarm_details") # Will be None for single tasks
        }
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details"]
        _encrypt_doc(task_doc, SENSITIVE_TASK_FIELDS)

        await self.task_collection.insert_one(task_doc)
        logger.info(f"Created new task {task_id} (type: {task_doc['task_type']}) for user {user_id} with status 'planning'.")
        return task_id

    async def get_task(self, task_id: str, user_id: str) -> Optional[Dict]:
        """Fetches a single task by its ID, ensuring it belongs to the user."""
        doc = await self.task_collection.find_one({"task_id": task_id, "user_id": user_id})
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details"]
        _decrypt_doc(doc, SENSITIVE_TASK_FIELDS)
        return doc

    async def get_all_tasks_for_user(self, user_id: str) -> List[Dict]:
        """Fetches all tasks for a given user."""
        cursor = self.task_collection.find({"user_id": user_id}).sort("created_at", -1)
        docs = await cursor.to_list(length=None)
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details"]
        _decrypt_docs(docs, SENSITIVE_TASK_FIELDS)
        return docs

    async def update_task(self, task_id: str, updates: Dict) -> bool:
        """Updates an existing task document."""
        updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details"]
        _encrypt_doc(updates, SENSITIVE_TASK_FIELDS)
        result = await self.task_collection.update_one(
            {"task_id": task_id},
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
        return await self.update_task(task_id, {"clarifying_questions": current_questions})

    async def delete_task(self, task_id: str, user_id: str) -> str:
        """Deletes a task."""
        result = await self.task_collection.delete_one({"task_id": task_id, "user_id": user_id})
        return "Task deleted successfully." if result.deleted_count > 0 else None

    async def decline_task(self, task_id: str, user_id: str) -> str:
        """Declines a task by setting its status to 'declined'."""
        success = await self.update_task(task_id, {"status": "declined"})
        return "Task declined." if success else None

    async def delete_tasks_by_tool(self, user_id: str, tool_name: str) -> int:
        """
        Deletes all tasks for a user that have a plan step using a specific tool.
        This is used when an integration is disconnected.
        """
        if not user_id or not tool_name:
            return 0
        query = {"user_id": user_id, "runs.plan.tool": tool_name}
        result = await self.task_collection.delete_many(query)
        logger.info(f"Deleted {result.deleted_count} tasks for user {user_id} using tool '{tool_name}'.")
        return result.deleted_count

    async def cancel_latest_run(self, task_id: str) -> bool:
        """Pops the last run from the array and reverts the task status to completed."""
        update_payload = {
            "$set": {
                "status": "completed",
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
            },
            "$pop": {"runs": 1}  # Removes the last element from the 'runs' array
        }
        result = await self.task_collection.update_one({"task_id": task_id}, update_payload) # noqa: E501
        return result.modified_count > 0

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

        await self.task_collection.insert_one(new_task_doc)
        return new_task_id

    # --- Message Methods ---
    async def add_message(self, user_id: str, role: str, content: str, message_id: Optional[str] = None, thoughts: Optional[List[str]] = None, tool_calls: Optional[List[Dict]] = None, tool_results: Optional[List[Dict]] = None) -> Dict:
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

        # Add new structured fields if they are provided and not empty
        if thoughts:
            message_doc["thoughts"] = thoughts
        if tool_calls:
            message_doc["tool_calls"] = tool_calls
        if tool_results:
            message_doc["tool_results"] = tool_results

        SENSITIVE_MESSAGE_FIELDS = ["content", "thoughts", "tool_calls", "tool_results"]
        _encrypt_doc(message_doc, SENSITIVE_MESSAGE_FIELDS)

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

        SENSITIVE_MESSAGE_FIELDS = ["content", "thoughts", "tool_calls", "tool_results"]
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
