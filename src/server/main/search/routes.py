from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import asyncio
from bson import ObjectId
import datetime

from main.auth.utils import PermissionChecker
from main.search.models import UnifiedSearchRequest
from main.search.utils import perform_unified_search
from main.dependencies import mongo_manager
from main.memories import db as memories_db
from main.memories.utils import _get_normalized_embedding, _initialize_embedding_model
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/search",
    tags=["Search"]
)

def sanitize_dict(d: dict) -> dict:
    """Converts non-serializable types like ObjectId and datetime to strings."""
    for key, value in d.items():
        if isinstance(value, ObjectId):
            d[key] = str(value)
        elif isinstance(value, datetime.datetime):
            d[key] = value.isoformat()
    return d

@router.post("/unified", summary="Perform a unified search across all data sources")
async def unified_search_endpoint(
    request: UnifiedSearchRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"])) # Using a common permission
):
    try:
        return StreamingResponse(
            perform_unified_search(request.query, user_id),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            }
        )
    except Exception as e:
        # This part might not be reached if the error happens inside the generator,
        # but it's good for handling setup errors before the stream starts.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during search setup: {e}")
@router.get("/interactive", summary="Perform a fast, interactive search for the UI")
async def interactive_search(
    query: str = Query(..., min_length=3),
    user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks", "read:chat", "read:memory"]))
):
    try:
        # Ensure embedding model is ready for memory search
        _initialize_embedding_model()

        # Perform searches in parallel
        tasks_coro = mongo_manager.task_collection.find(
            {"user_id": user_id, "$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "name": 1, "description": 1, "task_id": 1, "status": 1, "created_at": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(10).to_list(length=10)

        chats_coro = mongo_manager.messages_collection.find(
            {"user_id": user_id, "role": "user", "$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "content": 1, "message_id": 1, "timestamp": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(10).to_list(length=10)

        async def search_memories_coro():
            pool = await memories_db.get_db_pool()
            async with pool.acquire() as conn:
                await register_vector(conn)
                query_embedding = _get_normalized_embedding(query, task_type="RETRIEVAL_QUERY")
                records = await conn.fetch(
                    """
                    SELECT id, content, created_at, 1 - (embedding <=> $2) AS similarity
                    FROM facts WHERE user_id = $1 ORDER BY similarity DESC LIMIT 10;
                    """, user_id, query_embedding
                )
                # Filter memories by a similarity threshold
                return [dict(r) for r in records if r['similarity'] > 0.6]

        tasks_res, chats_res, memories_res = await asyncio.gather(
            tasks_coro, chats_coro, search_memories_coro()
        )

        # Format and sanitize results
        formatted_tasks = [{"type": "task", **sanitize_dict(t)} for t in tasks_res]
        formatted_chats = [{"type": "chat", **sanitize_dict(c)} for c in chats_res]
        formatted_memories = [{"type": "memory", **sanitize_dict(m)} for m in memories_res]

        all_results = formatted_tasks + formatted_chats + formatted_memories

        # Simple sort: best match of any type comes first
        all_results.sort(key=lambda x: x.get('score', x.get('similarity', 0)), reverse=True)

        return JSONResponse(content={"results": all_results})

    except Exception as e:
        logger.error(f"Error during interactive search for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")