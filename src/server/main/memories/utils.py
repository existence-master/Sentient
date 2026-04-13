import logging
import numpy as np
import json
import re
from datetime import datetime, timedelta, timezone
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Optional

from pgvector.asyncpg import register_vector
from json_extractor import JsonExtractor

from main.gemini_embed import ensure_embedding_ready, get_normalized_embedding
from . import db, llm
from .prompts import fact_analysis_user_prompt_template
from main.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# --- Module-level state ---
agents: Dict[str, Any] = {}

# A threshold to determine if two memories are connected in the graph.
SIMILARITY_THRESHOLD = 0.85

def _initialize_embedding_model():
    """Initializes the Gemini embedding model."""
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    ensure_embedding_ready()

def _initialize_agents():
    """Initializes all necessary LLM agents for memory operations."""
    global agents
    if not agents:
        logger.info("Initializing memory utility agents...")
        agents = {
            "fact_analysis": llm.get_fact_analysis_agent(),
        }

def clean_llm_output(data: Any) -> Any:
    """Cleans JSON string from LLM output."""
    if isinstance(data, str):
        # Strip <think>...</think> blocks (from reasoning models like qwen3)
        data = re.sub(r"<think>[\s\S]*?</think>", "", data)
        if "```json" in data:
            match = re.search(r"```json\s*([\s\S]+?)\s*```", data)
            if match:
                return match.group(1).strip()
        return data.strip()
    return data

def parse_duration(duration_str: Optional[str]) -> Optional[datetime]:
    """Parses a human-readable duration string."""
    if not duration_str: return None
    try:
        match = re.match(r"(\d+)\s+(hour|day|week|month)s?", duration_str, re.IGNORECASE)
        if not match: return None
        value, unit = int(match.group(1)), match.group(2).lower()
        delta_map = {
            "hour": timedelta(hours=value), "day": timedelta(days=value),
            "week": timedelta(weeks=value), "month": timedelta(days=value * 30),
        }
        if delta := delta_map.get(unit):
            return datetime.now(timezone.utc) + delta
    except Exception:
        return None
    return None

async def _insert_fact_with_analysis(conn, user_id: str, content: str, source: Optional[str], analysis: dict) -> str:
    """Internal function to insert a fact and its related metadata into the database."""
    expires_at = parse_duration(analysis.get("duration")) if analysis.get("memory_type") == "short-term" else None
    embedding = get_normalized_embedding(content, task_type="RETRIEVAL_DOCUMENT")

    await register_vector(conn)
    async with conn.transaction():
        fact_id = await conn.fetchval(
            "INSERT INTO facts (user_id, content, embedding, source, expires_at) VALUES ($1, $2, $3, $4, $5) RETURNING id",
            user_id, content, embedding, source, expires_at
        )
        topic_names = analysis.get("topics", ["Miscellaneous"])
        for topic_name in topic_names:
            topic_id = await conn.fetchval("SELECT id FROM topics WHERE name = $1", topic_name)
            if topic_id:
                await conn.execute("INSERT INTO fact_topics (fact_id, topic_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", fact_id, topic_id)
    return f"Fact added with ID {fact_id}."

# --- CRUD Functions ---

async def create_memory(user_id: str, content: str, source: Optional[str] = "manual") -> str:
    """Creates a new fact after analysis."""
    if not agents: _initialize_agents()
    logger.info(f"Creating new memory for user {user_id} from source '{source}'.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        prompt = fact_analysis_user_prompt_template.format(text=content)
        analysis_raw = llm.run_agent_with_prompt(agents["fact_analysis"], prompt)
        analysis_cleaned = clean_llm_output(analysis_raw)
        analysis = JsonExtractor.extract_valid_json(analysis_cleaned)
        if not analysis:
            logger.error(f"Create memory failed due to analysis JSON error. Output: {analysis_cleaned}")
            raise ValueError("Failed to analyze memory content.")
        return await _insert_fact_with_analysis(conn, user_id, content, source, analysis)

async def update_memory(user_id: str, memory_id: int, new_content: str) -> str:
    """Updates an existing fact's content, embedding, and topics."""
    if not agents: _initialize_agents()
    logger.info(f"Updating memory {memory_id} for user {user_id}.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchval("SELECT id FROM facts WHERE id = $1 AND user_id = $2", memory_id, user_id)
        if not existing:
            raise ValueError(f"Memory with ID {memory_id} not found for this user.")

        prompt = fact_analysis_user_prompt_template.format(text=new_content)
        analysis_raw = llm.run_agent_with_prompt(agents["fact_analysis"], prompt)
        analysis_cleaned = clean_llm_output(analysis_raw)
        analysis = JsonExtractor.extract_valid_json(analysis_cleaned)
        if not analysis:
            raise ValueError("Failed to analyze updated memory content.")

        new_embedding = get_normalized_embedding(new_content, task_type="RETRIEVAL_DOCUMENT")
        # Also re-evaluate the expiration based on the new content analysis
        expires_at = parse_duration(analysis.get("duration")) if analysis.get("memory_type") == "short-term" else None

        await register_vector(conn)
        async with conn.transaction():
            await conn.execute(
                # Update content, embedding, timestamp, and expiration
                "UPDATE facts SET content = $1, embedding = $2, updated_at = NOW(), expires_at = $4 WHERE id = $3",
                new_content, new_embedding, memory_id, expires_at
            )
            await conn.execute("DELETE FROM fact_topics WHERE fact_id = $1", memory_id)
            topic_names = analysis.get("topics", ["Miscellaneous"])
            for topic_name in topic_names:
                topic_id = await conn.fetchval("SELECT id FROM topics WHERE name = $1", topic_name)
                if topic_id:
                    await conn.execute("INSERT INTO fact_topics (fact_id, topic_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", memory_id, topic_id)
    
    return f"Memory {memory_id} updated successfully."

async def delete_memory(user_id: str, memory_id: int) -> str:
    """Deletes a fact from the database."""
    logger.info(f"Deleting memory {memory_id} for user {user_id}.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM facts WHERE id = $1 AND user_id = $2", memory_id, user_id)
        deleted_count = int(result.split(" ")[1]) if result else 0
        if deleted_count == 0:
            raise ValueError(f"Memory with ID {memory_id} not found for this user.")
    return f"Memory {memory_id} deleted successfully."

# --- Graph Generation (existing function) ---
def truncate_text(text: str, max_length: int = 25) -> str:
    """Truncates text to a max length and adds ellipsis if needed."""
    if len(text) > max_length:
        return text[:max_length].strip() + "..."
    return text

async def create_memory_graph(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches user's memories, calculates semantic similarity, and returns a graph structure.
    """
    logger.info(f"Generating memory graph for user_id: {user_id}")
    pool = await db.get_db_pool()
    async with pool.acquire() as connection:
        await register_vector(connection)
        query = """
            SELECT
                f.id, f.content, f.source, f.created_at, f.embedding,
                COALESCE(ARRAY_AGG(t.name) FILTER (WHERE t.name IS NOT NULL), '{}') as topics
            FROM facts f
            LEFT JOIN fact_topics ft ON f.id = ft.fact_id
            LEFT JOIN topics t ON ft.topic_id = t.id
            WHERE f.user_id = $1 AND f.embedding IS NOT NULL
            GROUP BY f.id ORDER BY f.created_at DESC;
        """
        records = await connection.fetch(query, user_id)

    if not records:
        return {"nodes": [], "links": []}

    fact_map = {record['id']: record for record in records}
    fact_ids = list(fact_map.keys())
    embeddings = np.array([record['embedding'] for record in records])

    nodes = [
        {
            "id": record["id"], "label": truncate_text(record["content"]),
            "title": record["content"], "content": record["content"],
            "created_at": record["created_at"].isoformat(), "source": record["source"],
            "topics": record.get("topics", [])
        } for record in fact_map.values()
    ]

    links = []
    if len(embeddings) > 1:
        similarity_matrix = cosine_similarity(embeddings)
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity >= SIMILARITY_THRESHOLD:
                    links.append({
                        "source": fact_ids[i], "target": fact_ids[j],
                        "value": float(similarity), "title": f"Similarity: {similarity:.2%}"
                    })

    logger.info(f"Generated graph with {len(nodes)} nodes and {len(links)} links.")
    return {"nodes": nodes, "links": links}