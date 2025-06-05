import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastmcp import FastMCP
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_memory_test")

# --- MongoDB Setup ---
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping') # Verify connection
    db = client[MONGO_DB_NAME]
    print(f"Successfully connected to MongoDB: {MONGO_URI}, Database: {MONGO_DB_NAME}")
except ConnectionFailure:
    print(f"Failed to connect to MongoDB at {MONGO_URI}. Please ensure MongoDB is running.")
    exit(1) # Or handle more gracefully

long_term_memories_collection = db["long_term_memories"]
short_term_memories_collection = db["short_term_memories"]

# Ensure TTL index for short-term memories (run once)
# This index will automatically delete documents from 'short_term_memories'
# when the 'expire_at' time is reached. 'expireAfterSeconds: 0' means
# delete immediately when 'expire_at' is in the past.
if "expire_at_ttl" not in short_term_memories_collection.index_information():
    short_term_memories_collection.create_index("expire_at", name="expire_at_ttl", expireAfterSeconds=0)
    print("Created TTL index on 'short_term_memories.expire_at'")

# Optional: Text indexes for searching (consider performance implications for large datasets)
# if "ltm_text_search" not in long_term_memories_collection.index_information():
#     long_term_memories_collection.create_index(
#         [("key", "text"), ("value_str", "text")], # Assuming value might be stored as value_str for text search
#         name="ltm_text_search"
#     )
#     print("Created text index on 'long_term_memories' for key and value_str")

# if "stm_text_search" not in short_term_memories_collection.index_information():
#     short_term_memories_collection.create_index(
#         [("content", "text"), ("category", "text")],
#         name="stm_text_search"
#     )
#     print("Created text index on 'short_term_memories' for content and category")


mcp = FastMCP(
    name="AICompanionMemoryServer",
    instructions="This server manages long-term and short-term memories for an AI companion. "
                 "All operations require a `user_id` to scope memories to a specific user."
)

def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return None

def _iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    if iso_str:
        try:
            return datetime.fromisoformat(iso_str.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
        except ValueError:
            return None # Or raise error
    return None

def _mongo_doc_to_dict(doc: Optional[Dict]) -> Optional[Dict]:
    if doc:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc:
            doc["created_at"] = _datetime_to_iso(doc["created_at"])
        if "updated_at" in doc:
            doc["updated_at"] = _datetime_to_iso(doc["updated_at"])
        if "due_date" in doc:
            doc["due_date"] = _datetime_to_iso(doc["due_date"])
        if "expire_at" in doc:
            doc["expire_at"] = _datetime_to_iso(doc["expire_at"])
        return doc
    return None

# --- Long-Term Memory Tools ---

@mcp.tool()
def set_long_term_memory(user_id: str, key: str, value: Any) -> Dict:
    """
    Adds or updates a long-term memory for a given user.
    Long-term memories are facts or preferences that change infrequently.
    If the key already exists, its value and updated_at timestamp will be updated.
    Otherwise, a new memory item will be created.

    :param user_id: The unique identifier for the user.
    :param key: The key or name of the memory (e.g., "name", "favorite_color", "hometown").
    :param value: The value associated with the key.
    :return: The created or updated memory document.
    """
    now = datetime.now(timezone.utc)
    query = {"user_id": user_id, "key": key}
    update = {
        "$set": {"value": value, "updated_at": now},
        "$setOnInsert": {"user_id": user_id, "key": key, "created_at": now}
    }
    result = long_term_memories_collection.find_one_and_update(
        query, update, upsert=True, return_document=ReturnDocument.AFTER
    )
    return _mongo_doc_to_dict(result)

@mcp.tool()
def get_long_term_memory(user_id: str, key: str) -> Optional[Dict]:
    """
    Retrieves a specific long-term memory for a user by its key.

    :param user_id: The unique identifier for the user.
    :param key: The key of the memory to retrieve.
    :return: The memory document if found, otherwise None.
    """
    memory = long_term_memories_collection.find_one({"user_id": user_id, "key": key})
    return _mongo_doc_to_dict(memory)

@mcp.tool()
def delete_long_term_memory(user_id: str, key: str) -> bool:
    """
    Deletes a specific long-term memory for a user by its key.

    :param user_id: The unique identifier for the user.
    :param key: The key of the memory to delete.
    :return: True if a memory was deleted, False otherwise.
    """
    result = long_term_memories_collection.delete_one({"user_id": user_id, "key": key})
    return result.deleted_count > 0

@mcp.tool()
def list_long_term_memories(user_id: str, query_text: Optional[str] = None) -> List[Dict]:
    """
    Lists all long-term memories for a user.
    Optionally, filters memories where the query_text is found in the key or (if string) value.

    :param user_id: The unique identifier for the user.
    :param query_text: Optional text to search for in memory keys or string values. Case-insensitive.
    :return: A list of memory documents.
    """
    find_filter = {"user_id": user_id}
    if query_text:
        # Simple regex search for flexibility across key and value (if value is string)
        # This is less efficient than a text index but more flexible for arbitrary values.
        # For production with large text values, consider text indexes and $text operator.
        regex_query = {"$regex": query_text, "$options": "i"}
        find_filter["$or"] = [
            {"key": regex_query},
            {"value": regex_query} # This part only works well if 'value' is often a string
        ]
        # If you expect 'value' to be non-string and want to search it,
        # you might need to store a string representation or use more complex queries.

    memories = long_term_memories_collection.find(find_filter)
    return [_mongo_doc_to_dict(mem) for mem in memories if mem]


# --- Short-Term Memory Tools ---

@mcp.tool()
def add_short_term_memory(
    user_id: str,
    content: str,
    ttl_seconds: int,
    category: Optional[str] = None,
    due_date_iso: Optional[str] = None
) -> Dict:
    """
    Adds a new short-term memory for a user.
    Short-term memories are temporary, like reminders or upcoming events.
    They will be automatically deleted after 'ttl_seconds'.

    :param user_id: The unique identifier for the user.
    :param content: The textual content of the memory (e.g., "Meeting with Jane at 3 PM").
    :param ttl_seconds: Time-to-live in seconds. The memory will be auto-deleted after this duration.
    :param category: Optional category for the memory (e.g., "reminder", "todo", "event").
    :param due_date_iso: Optional due date for the memory in ISO 8601 format (e.g., "2023-10-27T15:00:00Z").
    :return: The created short-term memory document.
    """
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(seconds=ttl_seconds)
    due_date = _iso_to_datetime(due_date_iso)

    memory_doc = {
        "user_id": user_id,
        "content": content,
        "category": category,
        "due_date": due_date,
        "expire_at": expire_at,
        "created_at": now,
    }
    result = short_term_memories_collection.insert_one(memory_doc)
    inserted_doc = short_term_memories_collection.find_one({"_id": result.inserted_id})
    return _mongo_doc_to_dict(inserted_doc)

@mcp.tool()
def get_short_term_memories(
    user_id: str,
    category: Optional[str] = None,
    query_text: Optional[str] = None,
    upcoming_days: Optional[int] = None
) -> List[Dict]:
    """
    Retrieves short-term memories for a user, with optional filters.
    Memories are sorted by due_date (if available, ascending) then created_at (ascending).

    :param user_id: The unique identifier for the user.
    :param category: Filter by a specific category.
    :param query_text: Filter by text content (case-insensitive search within 'content').
    :param upcoming_days: Filter for memories with a due_date within the next N days (inclusive of today).
    :return: A list of matching short-term memory documents.
    """
    find_filter: Dict[str, Any] = {"user_id": user_id}
    if category:
        find_filter["category"] = category
    if query_text:
        find_filter["content"] = {"$regex": query_text, "$options": "i"}
    if upcoming_days is not None:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        future_end = today_start + timedelta(days=upcoming_days + 1) # up to end of Nth day
        find_filter["due_date"] = {"$gte": today_start, "$lt": future_end}

    # Sort by due_date (nulls last), then by creation date
    sort_criteria = [("due_date", 1), ("created_at", 1)]

    memories = short_term_memories_collection.find(find_filter).sort(sort_criteria)
    return [_mongo_doc_to_dict(mem) for mem in memories if mem]

@mcp.tool()
def update_short_term_memory(
    user_id: str,
    memory_id: str,
    content: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    category: Optional[str] = None,
    due_date_iso: Optional[str] = None
) -> Optional[Dict]:
    """
    Updates an existing short-term memory.
    Only provided fields will be updated. The 'updated_at' timestamp is always updated.
    If 'ttl_seconds' is provided, 'expire_at' is recalculated from 'updated_at'.

    :param user_id: The unique identifier for the user.
    :param memory_id: The BSON ObjectId string of the memory to update.
    :param content: New content for the memory.
    :param ttl_seconds: New TTL in seconds. If set, 'expire_at' is recalculated.
    :param category: New category for the memory.
    :param due_date_iso: New due date in ISO 8601 format.
    :return: The updated memory document, or None if not found or not updated.
    """
    try:
        obj_id = ObjectId(memory_id)
    except Exception:
        print(f"Invalid memory_id format: {memory_id}")
        return None

    now = datetime.now(timezone.utc)
    update_fields: Dict[str, Any] = {"updated_at": now}
    if content is not None:
        update_fields["content"] = content
    if category is not None: # Allow setting category to None/empty
        update_fields["category"] = category
    if due_date_iso is not None:
        update_fields["due_date"] = _iso_to_datetime(due_date_iso)
    if ttl_seconds is not None:
        update_fields["expire_at"] = now + timedelta(seconds=ttl_seconds)

    if not update_fields: # Nothing to update other than timestamp
        return get_short_term_memories(user_id=user_id, memory_id_filter=memory_id) # type: ignore

    result = short_term_memories_collection.find_one_and_update(
        {"_id": obj_id, "user_id": user_id},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER
    )
    return _mongo_doc_to_dict(result)


@mcp.tool()
def delete_short_term_memory(user_id: str, memory_id: str) -> bool:
    """
    Manually deletes a specific short-term memory by its ID.

    :param user_id: The unique identifier for the user.
    :param memory_id: The BSON ObjectId string of the memory to delete.
    :return: True if a memory was deleted, False otherwise.
    """
    try:
        obj_id = ObjectId(memory_id)
    except Exception:
        print(f"Invalid memory_id format: {memory_id}")
        return False
    result = short_term_memories_collection.delete_one({"_id": obj_id, "user_id": user_id})
    return result.deleted_count > 0

# --- General Search Tool ---

@mcp.tool()
def search_all_memories(user_id: str, query_text: str) -> Dict[str, List[Dict]]:
    """
    Searches both long-term and short-term memories for a given query text.
    For long-term memories, it searches keys and string values.
    For short-term memories, it searches content and category.

    :param user_id: The unique identifier for the user.
    :param query_text: The text to search for.
    :return: A dictionary with 'long_term' and 'short_term' keys,
             each containing a list of matching memory documents.
    """
    ltm_results = list_long_term_memories(user_id, query_text)
    stm_results = get_short_term_memories(user_id, query_text=query_text)

    return {
        "long_term": ltm_results,
        "short_term": stm_results
    }


if __name__ == "__main__":
    print("Starting AI Companion Memory MCP Server...")
    # Example: Ensure user_id "test_user" has a long-term memory
    # This is just for testing and would typically be done by the LLM via tool calls
    if not get_long_term_memory("test_user", "name"):
        set_long_term_memory("test_user", "name", "Test User")
        print("Added initial 'name' LTM for 'test_user'")

    if not get_short_term_memories("test_user", query_text="initial test reminder"):
        add_short_term_memory("test_user", "This is an initial test reminder for test_user", ttl_seconds=3600, category="test")
        print("Added initial STM for 'test_user'")

    mcp.run(transport="sse", host="0.0.0.0", port=8001) # Using a different port, e.g., 8001