import asyncio
import datetime
import os
import uuid
from typing import Dict, List, Optional

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.results import UpdateResult, DeleteResult

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_agent_db")
MEMORY_COLLECTION_NAME = "memory_operations"

class MemoryQueue:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[MONGO_DB_NAME]
        self.collection = self.db[MEMORY_COLLECTION_NAME]
        self.current_operation_execution = None

    async def initialize_db(self):
        """
        Initializes the database by ensuring necessary indexes are created.
        Should be called once on application startup.
        """
        indexes = [
            IndexModel([("user_id", ASCENDING), ("operation_id", ASCENDING)], name="user_operation_id_idx", unique=True),
            IndexModel([("status", ASCENDING), ("timestamp", ASCENDING)], name="global_pending_operations_idx"),
            IndexModel([("user_id", ASCENDING), ("status", ASCENDING)], name="user_status_idx"),
            IndexModel([("user_id", ASCENDING), ("timestamp", ASCENDING)], name="user_timestamp_idx"),
            IndexModel([("status", ASCENDING), ("completed_at", ASCENDING)], name="global_completed_operations_idx")
        ]
        try:
            await self.collection.create_indexes(indexes)
            print(f"[DB_INIT] Indexes ensured for collection: {self.collection.name}")
        except Exception as e:
            print(f"[DB_ERROR] Failed to create indexes for {self.collection.name}: {e}")

    async def add_operation(self, user_id: str, memory_data: Dict) -> str:
        """Add a new memory operation to the queue, scoped by user_id."""
        operation_document_id = str(uuid.uuid4())
        operation = {
            "operation_id": operation_document_id,
            "user_id": user_id,
            "memory_data": memory_data,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "status": "pending"
        }
        await self.collection.insert_one(operation)
        return operation_document_id

    async def get_next_operation(self) -> Optional[Dict]:
        """
        Get the next pending memory operation from any user (global queue).
        Atomically updates the operation status to "processing".
        """
        return await self.collection.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "processing", "processing_started_at": datetime.datetime.now(datetime.timezone.utc)}},
            sort=[("timestamp", ASCENDING)],
            return_document=ReturnDocument.AFTER
        )

    async def complete_operation(self, user_id: str, operation_id: str, result: Optional[str] = None, error: Optional[str] = None, status: str = "completed"):
        """
        Mark a memory operation with a final status (completed, failed, cancelled) and save the result/error.
        Scoped by user_id.
        """
        update_doc = {
            "status": status,
            "result": result,
            "error": error,
            "completed_at": datetime.datetime.now(datetime.timezone.utc),
            "$unset": {"processing_started_at": ""}
        }
        if result is None:
            del update_doc["result"]
        if error is None:
            del update_doc["error"]
        
        await self.collection.update_one(
            {"user_id": user_id, "operation_id": operation_id},
            {"$set": {k: v for k,v in update_doc.items() if k != "$unset"}, "$unset": update_doc["$unset"]}
        )

    async def get_operations_for_user(self, user_id: str) -> List[Dict]:
        """Return a list of all memory operations for a specific user, newest first."""
        if not user_id:
            return []
        cursor = self.collection.find({"user_id": user_id}).sort("timestamp", DESCENDING)
        return await cursor.to_list(length=None)

    async def delete_old_completed_operations(self, hours_threshold: int = 1):
        """Delete completed memory operations older than the specified hours threshold (globally for all users)."""
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_threshold)
        query = {
            "status": "completed",
            "completed_at": {"$lt": cutoff_time}
        }
        result: DeleteResult = await self.collection.delete_many(query)
        print(f"[DB_CLEANUP] Deleted {result.deleted_count} old completed memory operations.")

        stuck_processing_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_threshold * 4)
        stuck_query = {
            "status": "processing",
            "processing_started_at": {"$lt": stuck_processing_cutoff}
        }
        stuck_update_result: UpdateResult = await self.collection.update_many(
            stuck_query,
            {"$set": {
                "status": "failed",
                "error": "Memory operation timed out during processing cleanup.",
                "completed_at": datetime.datetime.now(datetime.timezone.utc)
            }}
        )
        if stuck_update_result.modified_count > 0:
            print(f"[DB_CLEANUP] Marked {stuck_update_result.modified_count} memory operations stuck in processing as 'failed'.")