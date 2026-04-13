import os
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from pgvector.asyncpg import register_vector
from fastmcp.utilities.logging import get_logger

from main.gemini_embed import ensure_embedding_ready, get_normalized_embedding
from . import db, llm
from .prompts import (
    fact_relevance_user_prompt_template,
    fact_summarization_user_prompt_template,
    fact_extraction_user_prompt_template,
    cud_decision_user_prompt_template,
    fact_analysis_user_prompt_template,
)
from main.config import EMBEDDING_MODEL_NAME

logger = get_logger(__name__)

# --- Module-level state (initialized by lifespan event) ---
agents: Dict[str, Any] = {}

# --- Initialization Functions ---
def initialize_embedding_model():
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    ensure_embedding_ready()

def initialize_agents():
    global agents
    if not agents:
        logger.info("Initializing all memory agents...")
        agents = {
            "fact_summarization": llm.get_fact_summarization_agent(),
            "fact_extraction": llm.get_fact_extraction_agent(),
            "fact_analysis": llm.get_fact_analysis_agent(),
            "cud_decision": llm.get_cud_decision_agent(),
            "fact_relevance": llm.get_fact_relevance_agent(),
        }

def parse_duration(duration_str: Optional[str]) -> Optional[datetime]:
    """Parses a human-readable duration string and returns a future UTC datetime."""
    if not duration_str:
        return None
    logger.debug(f"Attempting to parse duration string: '{duration_str}'")
    try:
        match = re.match(r"(\d+)\s+(hour|day|week|month)s?", duration_str, re.IGNORECASE)
        if not match:
            logger.warning(f"Duration string '{duration_str}' does not match expected format.")
            return None

        value, unit = int(match.group(1)), match.group(2).lower()

        delta_map = {
            "hour": timedelta(hours=value),
            "day": timedelta(days=value),
            "week": timedelta(weeks=value),
            "month": timedelta(days=value * 30), # Approximation
        }
        if delta := delta_map.get(unit):
            result = datetime.now(timezone.utc) + delta
            logger.info(f"Parsed '{duration_str}' to an expiration date of {result.isoformat()}")
            return result
    except Exception as e:
        logger.warning(f"Could not parse duration string '{duration_str}': {e}")

    logger.warning(f"Failed to parse duration string '{duration_str}'.")
    return None

def clean_llm_output(data: Any) -> Any:
    """Recursively cleans strings in LLM output."""
    logger.debug(f"Cleaning LLM output. Input type: {type(data)}")
    if isinstance(data, str):
        cleaned_str = data
        original_str = str(data)
        # If the LLM wrapped the JSON in a markdown block, extract it.
        # This is helpful for pre-processing before a json.loads() call.
        if "```json" in cleaned_str:
            match = re.search(r"```json\s*([\s\S]+?)\s*```", cleaned_str)
            if match:
                cleaned_str = match.group(1)
                logger.debug("Extracted JSON content from markdown block.")

        # Remove any <think> blocks that might be present.
        cleaned_str = re.sub(r'<think>.*?</think>', '', cleaned_str, flags=re.DOTALL)
        cleaned_str = cleaned_str.strip()
        if cleaned_str != original_str:
            logger.debug("LLM string output was cleaned.")
        return cleaned_str
    if isinstance(data, dict):
        return {k: clean_llm_output(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_llm_output(i) for i in data]
    return data

async def search_memory(user_id: str, query: str) -> str: # noqa: E501
    """Searches memory by performing a semantic search, filtering for relevance, and summarizing results."""
    if not query or not query.strip():
        logger.warning(f"search_memory called with an empty query for user {user_id}. Aborting.")
        return "Please provide a search query to search your memory."

    logger.info(f"Executing search_memory for user_id='{user_id}' with query: '{query}'")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        await register_vector(conn)
        
        logger.info("Step 1/4: Performing semantic search in database.")
        query_embedding = get_normalized_embedding(query, task_type="RETRIEVAL_QUERY")
        
        records = await conn.fetch(
            """
            SELECT DISTINCT f.id, f.content, 1 - (f.embedding <=> $2) AS similarity
            FROM facts f
            WHERE f.user_id = $1
            ORDER BY similarity DESC
            LIMIT 5;
            """, user_id, query_embedding
        )
    
        found_facts = [r['content'] for r in records]
        logger.info(f"Found {len(found_facts)} potentially relevant facts from vector search.")

    if not found_facts:
        logger.info("No facts found from vector search. Returning.")
        return "No relevant information found in your memory."

    # Step 2: Relevance Checking
    logger.info("Step 2/4: Checking relevance of found facts.")
    relevant_facts = []
    for fact in found_facts:
        prompt = fact_relevance_user_prompt_template.format(query=query, fact=fact)
        relevance_raw = llm.run_agent_with_prompt(agents["fact_relevance"], prompt)
        try:
            relevance_json = json.loads(clean_llm_output(relevance_raw))
            if relevance_json.get("is_relevant") is True:
                relevant_facts.append(fact)
                logger.debug(f"Fact '{fact[:50]}...' is RELEVANT.")
            else:
                logger.debug(f"Fact '{fact[:50]}...' is NOT relevant.")
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Could not parse relevance check response: {relevance_raw}")

    logger.info(f"Step 3/4: Found {len(relevant_facts)} truly relevant facts after filtering.")

    if not relevant_facts:
        logger.info("No relevant facts remained after filtering. Returning message to user.")
        return "No relevant information found in your memory."

    # Step 4: Summarization
    logger.info("Step 4/4: Summarizing relevant facts into a coherent paragraph.")
    prompt = fact_summarization_user_prompt_template.format(query=query, facts=json.dumps(relevant_facts))
    summary_raw = llm.run_agent_with_prompt(agents["fact_summarization"], prompt)
    summary = clean_llm_output(summary_raw)

    logger.info(f"Summary generated: {summary[:100]}...")

    logger.info("Search complete. Returning summary.")
    return summary if isinstance(summary, str) and summary else "Could not generate a summary from the retrieved information."

async def search_memory_by_source(user_id: str, query: str, source_name: str) -> str:
    """Searches memory by performing a semantic search within a specific source."""
    logger.info(f"Executing search_memory_by_source for user_id='{user_id}', source='{source_name}' with query: '{query}'")
    pool = await db.get_db_pool()
    records = []
    async with pool.acquire() as conn:
        await register_vector(conn)

        if query and query.strip():
            logger.info(f"Step 1/2: Performing semantic search in database for source '{source_name}'.")
            query_embedding = get_normalized_embedding(query, task_type="RETRIEVAL_QUERY")

            records = await conn.fetch(
                """
                SELECT DISTINCT f.id, f.content, 1 - (f.embedding <=> $3) AS similarity
                FROM facts f
                WHERE f.user_id = $1 AND f.source = $2
                ORDER BY similarity DESC
                LIMIT 5;
                """, user_id, source_name, query_embedding
            )
        else:
            logger.info(f"Step 1/2: Query is empty. Fetching most recent facts for source '{source_name}'.")
            records = await conn.fetch(
                """
                SELECT id, content
                FROM facts
                WHERE user_id = $1 AND source = $2
                ORDER BY created_at DESC
                LIMIT 5;
                """, user_id, source_name
            )

    found_facts = {r['id']: r['content'] for r in records}
    logger.info(f"Found {len(found_facts)} relevant facts from search.")

    if not found_facts:
        logger.info(f"No relevant facts found for source '{source_name}'. Returning message to user.")
        return f"No relevant information found for your query within the source '{source_name}'."

    logger.info("Step 2/2: Summarizing search results into a coherent paragraph.")
    facts_list = list(found_facts.values())
    summary_query = query if query and query.strip() else f"a summary of information from {source_name}"
    prompt = fact_summarization_user_prompt_template.format(query=summary_query, facts=json.dumps(facts_list))
    summary_raw = llm.run_agent_with_prompt(agents["fact_summarization"], prompt)
    summary = clean_llm_output(summary_raw)

    logger.info("Search by source complete. Returning summary.")
    return summary if isinstance(summary, str) and summary else "Could not generate a summary from the retrieved information."

async def _insert_fact_with_analysis(conn, user_id: str, content: str, source: Optional[str], analysis: dict) -> str:
    """Internal function to insert a fact and its related metadata into the database."""
    logger.info("Executing _insert_fact_with_analysis.")
    
    expires_at = None
    if analysis.get("memory_type") == "short-term":
        expires_at = parse_duration(analysis.get("duration"))

    async with conn.transaction():
        embedding = get_normalized_embedding(content, task_type="RETRIEVAL_DOCUMENT")
        
        fact_id = await conn.fetchval(
            """
            INSERT INTO facts (user_id, content, embedding, source, expires_at)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
            """,
            user_id, content, embedding, source, expires_at
        )
        logger.info(f"Inserted new fact with ID: {fact_id}.")

        # Link topics
        topic_names = analysis.get("topics", ["Miscellaneous"])
        for topic_name in topic_names:
            topic_id = await conn.fetchval("SELECT id FROM topics WHERE name = $1", topic_name)
            if topic_id:
                await conn.execute("INSERT INTO fact_topics (fact_id, topic_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", fact_id, topic_id)

    message = f"Fact added with ID {fact_id}."
    if expires_at:
        message += f" This is a short-term memory and will be forgotten around {expires_at.strftime('%Y-%m-%d %H:%M %Z')}."
    return message

async def _update_fact_with_analysis(conn, user_id: str, fact_id: int, content: str, analysis: dict) -> str:
    """Internal function to update a fact and its related metadata."""
    logger.info(f"Executing _update_fact_with_analysis for fact_id {fact_id}.")
    
    expires_at = None
    if analysis.get("memory_type") == "short-term":
        expires_at = parse_duration(analysis.get("duration"))

    async with conn.transaction():
        embedding = get_normalized_embedding(content, task_type="RETRIEVAL_DOCUMENT")
        
        await conn.execute(
            """
            UPDATE facts SET
                content = $1,
                embedding = $2,
                expires_at = $3,
                updated_at = NOW()
            WHERE id = $4 AND user_id = $5
            """,
            content, embedding, expires_at, fact_id, user_id
        )
        logger.info(f"Updated fact record for ID: {fact_id}.")

        # Update topics: delete old, insert new
        await conn.execute("DELETE FROM fact_topics WHERE fact_id = $1", fact_id)
        
        topic_names = analysis.get("topics", ["Miscellaneous"])
        for topic_name in topic_names:
            topic_id = await conn.fetchval("SELECT id FROM topics WHERE name = $1", topic_name)
            if topic_id:
                await conn.execute("INSERT INTO fact_topics (fact_id, topic_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", fact_id, topic_id)
        
        logger.info(f"Updated topics for fact ID: {fact_id}.")
    
    message = f"Fact {fact_id} updated."
    if expires_at:
        message += f" It will now be forgotten around {expires_at.strftime('%Y-%m-%d %H:%M %Z')}."
    return message

async def _process_single_fact_cud(conn, user_id: str, fact_content: str, source: Optional[str] = None) -> str:
    """
    Processes a single atomic fact by deciding to ADD, UPDATE, or DELETE it.
    This is the core 3-step CUD logic.
    """
    logger.info(f"Starting CUD process for single fact: '{fact_content}'")
    await register_vector(conn)
    
    logger.info("Step 1/3: Finding potentially related facts via semantic search.")
    query_embedding = get_normalized_embedding(fact_content, task_type="RETRIEVAL_QUERY")
    similar_records = await conn.fetch(
        """
        SELECT DISTINCT f.id, f.content, 1 - (f.embedding <=> $2) AS similarity
        FROM facts f
        WHERE f.user_id = $1
        ORDER BY similarity DESC
        LIMIT 3;
        """, user_id, query_embedding
    )
    logger.info(f"Found {len(similar_records)} potentially related facts.")

    logger.info("Step 2/3: Using LLM to decide on action and perform analysis.")
    prompt = cud_decision_user_prompt_template.format(
        information=fact_content,
        similar_facts=json.dumps([dict(r) for r in similar_records])
    )
    decision_raw = llm.run_agent_with_prompt(agents["cud_decision"], prompt)
    cleaned_output = clean_llm_output(decision_raw)
    decision = {}
    try:
        decision = json.loads(cleaned_output)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse CUD decision JSON from LLM. Defaulting to ADD. Output: {cleaned_output}")
        decision = {"action": "ADD", "content": fact_content} # Fallback

    logger.info(f"Step 3/3: Executing action '{decision.get('action')}'.")
    action = decision.get("action")
    
    if action == "IGNORE":
        fact_id = decision.get("fact_id")
        logger.info(f"Action is IGNORE for fact '{fact_content[:50]}...'. It is a duplicate of fact {fact_id}. No changes made.")
        return f"Fact already exists (ID: {fact_id}) and was ignored."
    elif action == "ADD" and decision.get("content") and decision.get("analysis"):
        logger.info("Action is ADD. Inserting new fact with full analysis.")
        return await _insert_fact_with_analysis(conn, user_id, decision["content"], source, decision["analysis"])

    elif action == "UPDATE" and decision.get("fact_id") and decision.get("content") and decision.get("analysis"):
        fact_id = decision["fact_id"]
        logger.info(f"Action is UPDATE for fact_id {fact_id}. Updating fact in place.")
        return await _update_fact_with_analysis(conn, user_id, fact_id, decision["content"], decision["analysis"])

    elif action == "DELETE" and decision.get("fact_id"):
        fact_id = decision["fact_id"]
        logger.info(f"Action is DELETE for fact_id {fact_id}.")
        result = await conn.execute("DELETE FROM facts WHERE id = $1 AND user_id = $2", fact_id, user_id)
        
        deleted_count = int(result.split(" ")[1]) if result and " " in result else 0

        if deleted_count == 1:
            logger.info(f"Successfully deleted fact {fact_id}.")
            return f"Fact {fact_id} deleted."
        else:
            logger.warning(f"Attempted to delete fact {fact_id}, but it was not found or not owned by the user.")
            return f"Fact {fact_id} not found or not owned by user."

    else:
        logger.warning(f"LLM decision was ambiguous or invalid, falling back to simple ADD. Decision: {decision}")
        # Fallback to analyzing and adding the original information
        prompt = fact_analysis_user_prompt_template.format(text=fact_content)
        analysis_raw = llm.run_agent_with_prompt(agents["fact_analysis"], prompt)
        analysis_cleaned = clean_llm_output(analysis_raw)
        try:
            analysis = json.loads(analysis_cleaned)
            return await _insert_fact_with_analysis(conn, user_id, fact_content, source, analysis)
        except json.JSONDecodeError:
            logger.error(f"Fallback ADD failed due to analysis JSON error. Output: {analysis_cleaned}")
            return f"Failed to process fact '{fact_content}' due to an internal analysis error."

async def cud_memory(user_id: str, information: str, source: Optional[str] = None, username: Optional[str] = None) -> str:
    """Breaks down information into atomic facts, then for each fact, decides whether to add, update, or delete it in memory."""
    logger.info(f"Executing cud_memory for user_id='{user_id}' with source='{source}'.")
    logger.debug(f"CUD information: \"{information}\"")
    
    # Step 1: Extract atomic facts from the information.
    logger.info("Step 1/N: Extracting atomic facts from the provided information.")
    
    # Use the provided username for the prompt, fallback to user_id if not available.
    name_for_prompt = username if username else user_id
    prompt = fact_extraction_user_prompt_template.format(username=name_for_prompt, paragraph=information)
    
    facts_raw = llm.run_agent_with_prompt(agents["fact_extraction"], prompt)
    cleaned_output = clean_llm_output(facts_raw)
    facts = []
    try:
        parsed_facts = json.loads(cleaned_output)
        if isinstance(parsed_facts, list):
            facts = [str(f) for f in parsed_facts if f] # Ensure all are strings and not empty
        else:
            facts = [str(parsed_facts)] if parsed_facts else []
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse fact extraction JSON from LLM. Treating entire input as one fact. Output: {cleaned_output}")
        facts = [information]

    if not facts:
        logger.info("Fact extraction resulted in no facts. Nothing to do.")
        return "No information was processed as no facts were extracted."

    logger.info(f"Extracted {len(facts)} facts. Processing each one.")
    
    results = []
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        for i, fact_content in enumerate(facts):
            logger.info(f"Processing fact {i+1}/{len(facts)}: '{fact_content}'")
            try:
                result_message = await _process_single_fact_cud(conn, user_id, fact_content, source)
                results.append(result_message)
            except Exception as e:
                logger.error(f"Error processing fact '{fact_content}': {e}", exc_info=True)
                results.append(f"Failed to process fact: '{fact_content}'.")

    final_message = "\n".join(results)
    logger.info(f"cud_memory processing complete. Result: {final_message}")
    return final_message if final_message else "No facts were processed."

async def build_initial_memory(user_id: str, documents: List[Dict[str, str]], username: Optional[str] = None) -> str:
    """Builds memory from documents, clearing existing memory first."""
    logger.info(f"Executing build_initial_memory for user_id='{user_id}' with {len(documents)} documents.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        logger.info(f"Clearing all existing facts for user_id='{user_id}'.")
        # Use a transaction to ensure atomicity
        async with conn.transaction():
            # Cascading delete on 'facts' will also clear 'fact_topics'
            await conn.execute("DELETE FROM facts WHERE user_id = $1", user_id)
    
    total_facts_added = 0
    # Use the provided username for prompts, falling back to the user_id if not available.
    name_for_prompt = username if username else user_id

    for doc in documents:
        text, source = doc.get("text", ""), doc.get("source", "unknown")
        if not text: continue
        
        prompt = fact_extraction_user_prompt_template.format(username=name_for_prompt, paragraph=text)
        facts_raw = llm.run_agent_with_prompt(agents["fact_extraction"], prompt)
        cleaned_output = clean_llm_output(facts_raw)
        facts = []
        try:
            parsed_facts = json.loads(cleaned_output)
            if isinstance(parsed_facts, list):
                facts = parsed_facts
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse fact extraction JSON from LLM. Skipping document. Output: {cleaned_output}")

        logger.info(f"Extracted {len(facts)} facts from source '{source}'.")
        if isinstance(facts, list):
            # Analyze all facts from the document in a batch-like manner
            for fact_content in facts:
                if not fact_content: continue
                
                logger.debug(f"Analyzing and adding extracted fact: '{fact_content}'")
                # Use the streamlined CUD path with a direct ADD intent
                await cud_memory(user_id, fact_content, source=source, username=name_for_prompt)
                total_facts_added += 1
    
    logger.info(f"Finished building initial memory. Added {total_facts_added} total facts.")
    return f"Memory built successfully. Added {total_facts_added} facts."

async def delete_memory_by_source(user_id: str, source_name: str) -> str:
    """Deletes all facts associated with a specific source."""
    logger.info(f"Executing delete_memory_by_source for user_id='{user_id}' and source='{source_name}'.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM facts WHERE user_id = $1 AND source = $2", user_id, source_name)
    
    deleted_count = int(result.split(" ")[1]) if result and " " in result else 0
    logger.info(f"Deleted {deleted_count} facts from source: {source_name}")
    return f"Deleted {deleted_count} facts from source: {source_name}"

async def purge_expired_facts():
    """Deletes facts from the database that have passed their expiration time."""
    logger.info("Purge job: Checking for expired short-term memories.")
    pool = await db.get_db_pool()
    async with pool.acquire() as conn:
        # The index on expires_at makes this query very efficient.
        result = await conn.execute("DELETE FROM facts WHERE expires_at IS NOT NULL AND expires_at < NOW()")
        deleted_count = int(result.split(" ")[1]) if result and " " in result else 0
        logger.info(f"Purge job: Purged {deleted_count} expired short-term memories.")