import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict
import datetime

from main.dependencies import auth_helper
from main.auth.utils import AuthHelper, PermissionChecker
from . import db, utils
from main.plans import PLAN_LIMITS
from .models import CreateMemoryRequest, UpdateMemoryRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memories",
    tags=["Memories"]
)

@router.on_event("startup")
async def startup_event():
    # Initialize agents and models when the memory routes are loaded
    utils._initialize_agents()
    utils._initialize_embedding_model()

@router.get("", summary="Get all memories for a user")
async def get_all_memories(
    user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))
):
    pool = None
    try:
        pool = await db.get_db_pool()
        async with pool.acquire() as connection:
            query = """
                SELECT
                    f.id, f.content, f.source, f.created_at, f.updated_at,
                    COALESCE(ARRAY_AGG(t.name) FILTER (WHERE t.name IS NOT NULL), '{}') as topics
                FROM facts f
                LEFT JOIN fact_topics ft ON f.id = ft.fact_id
                LEFT JOIN topics t ON ft.topic_id = t.id
                WHERE f.user_id = $1
                GROUP BY f.id ORDER BY f.created_at DESC;
            """
            records = await connection.fetch(query, user_id)
            memories = [
                {
                    "id": record["id"], "content": record["content"], "source": record["source"],
                    "created_at": record["created_at"].isoformat(), "updated_at": record["updated_at"].isoformat(),
                    "topics": record["topics"]
                } for record in records
            ]
            return JSONResponse(content={"memories": memories})
    except Exception as e:
        logger.error(f"Error fetching memories for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching memories.")

@router.get("/graph", summary="Get memory graph data for a user")
async def get_memory_graph(
    user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))
):
    try:
        graph_data = await utils.create_memory_graph(user_id)
        return JSONResponse(content=graph_data)
    except Exception as e:
        logger.error(f"Error generating memory graph for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating memory graph.")

@router.post("", summary="Create a new memory for a user")
async def create_memory(
    request: CreateMemoryRequest,
    user_id_and_plan: tuple = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan
    try:
        # --- Enforce Memory Limit ---
        limit = PLAN_LIMITS[plan].get("memories_total", 0)
        if limit != float('inf'):
            pool = await db.get_db_pool()
            async with pool.acquire() as conn:
                current_count = await conn.fetchval("SELECT COUNT(*) FROM facts WHERE user_id = $1", user_id)
            if current_count >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"You have reached your memory limit of {limit} facts. Please upgrade to Pro for unlimited memories."
                )
        result_message = await utils.create_memory(user_id, request.content, request.source)
        return JSONResponse(content={"message": result_message}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Error creating memory for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error creating memory.")

@router.put("/{memory_id}", summary="Update an existing memory")
async def update_memory(
    memory_id: int,
    request: UpdateMemoryRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    try:
        result_message = await utils.update_memory(user_id, memory_id, request.content)
        return JSONResponse(content={"message": result_message})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating memory {memory_id} for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error updating memory.")

@router.delete("/{memory_id}", summary="Delete a memory")
async def delete_memory(
    memory_id: int,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    try:
        result_message = await utils.delete_memory(user_id, memory_id)
        return JSONResponse(content={"message": result_message})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id} for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error deleting memory.")