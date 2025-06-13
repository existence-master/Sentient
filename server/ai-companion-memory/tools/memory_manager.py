# tools/memory_manager.py

import uuid
import json5
from datetime import datetime
from fastmcp import Context
from bson.objectid import ObjectId


from core.query_generator import generate_query_from_text
from core.security import validate_and_sanitize_query, SecurityException
from core.db_handler import db_handler
from core.semantic_search import semantic_search_manager

def get_user_id_from_context(ctx: Context) -> str:
    """Extracts user_id from request headers, with a fallback."""
    request = ctx.get_http_request()
    if not request:
        print("Warning: No HTTP request in context. Using fallback 'default_user'.")
        return "default_user"  # Fallback for non-HTTP contexts
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise ValueError("X-User-ID header is missing from the request.")
    return user_id

async def process_memory_request(task: str, ctx: Context) -> str:
    """
    Core function to process any memory request. It orchestrates the entire flow.
    """
    try:
        user_id = get_user_id_from_context(ctx)
        
        print(f"Generating query for user '{user_id}' with task: '{task}'")
        generated_json_str = generate_query_from_text(task)
        print(f"LLM Generated JSON: {generated_json_str}")
        
        query_spec = validate_and_sanitize_query(generated_json_str, user_id)
        
        # --- NEW, MORE ROBUST SEARCH LOGIC ---
        if query_spec['operation'] == 'find':
            parsed_spec = json5.loads(generated_json_str)
            if 'semantic_search_text' in parsed_spec:
                semantic_text = parsed_spec['semantic_search_text']
                print(f"Performing semantic search for: '{semantic_text}'")
                
                # We search across BOTH long-term and short-term memories for relevance
                relevant_mem_ids = semantic_search_manager.search_relevant_memories(user_id, semantic_text, k=5)
                
                if not relevant_mem_ids:
                    print("Semantic search found no relevant memories.")
                    return json5.dumps({"status": "success", "data": []}, ensure_ascii=False)

                print(f"Found relevant memory IDs: {relevant_mem_ids}")
                
                # We will now construct a new query to fetch these specific memories.
                # This overrides any filter the LLM generated, making the semantic result the source of truth.
                # We need to search both collections.
                
                long_term_results = db_handler.execute_query(
                    'long_term_memories', 'find', 
                    {'filter': {'user_id': user_id, 'memory_id': {'$in': relevant_mem_ids}}}
                )
                short_term_results = db_handler.execute_query(
                    'short_term_memories', 'find',
                    {'filter': {'user_id': user_id, 'memory_id': {'$in': relevant_mem_ids}}}
                )
                
                result = long_term_results + short_term_results
            else:
                # Fallback to the LLM's original query if no semantic text was provided
                print("No semantic_search_text provided. Running original filter.")
                result = db_handler.execute_query(
                    collection_name=query_spec['collection'],
                    operation=query_spec['operation'],
                    query=query_spec['query']
                )

        else: # Handle insert_one, update_one, etc. as before
            result = db_handler.execute_query(
                collection_name=query_spec['collection'],
                operation=query_spec['operation'],
                query=query_spec['query']
            )

            if query_spec['operation'] == 'insert_one' and 'inserted_id' in result:
                # ... (rest of the insertion logic remains exactly the same)
                doc_to_index = query_spec['query']['document']
                collection_name = query_spec['collection']
                inserted_oid = ObjectId(result['inserted_id'])

                if collection_name == 'contacts':
                    permanent_id = f"contact_{uuid.uuid4()}"
                    id_field = 'contact_id'
                else:
                    prefix = 'mem_l' if collection_name == 'long_term_memories' else 'mem_s'
                    permanent_id = f"{prefix}_{uuid.uuid4()}"
                    id_field = 'memory_id'

                db_handler.execute_query(
                    collection_name, 
                    'update_one', 
                    {
                        'filter': {'_id': inserted_oid, 'user_id': user_id},
                        'update': {'$set': {id_field: permanent_id}}
                    }
                )
                print(f"Replaced placeholder with permanent ID: {permanent_id}")
                
                content_to_embed = doc_to_index.get('content')
                if not content_to_embed and collection_name == 'contacts':
                    name = doc_to_index.get('name', '')
                    obs_text = ' '.join([obs.get('content', '') for obs in doc_to_index.get('observations', [])])
                    content_to_embed = f"{name}. {obs_text}".strip()
                
                if content_to_embed:
                    semantic_search_manager.add_memory_embedding(user_id, permanent_id, content_to_embed)

        # Format and return the result
        if isinstance(result, list):
            for item in result:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
        
        return json5.dumps({"status": "success", "data": result}, ensure_ascii=False)

    except (ValueError, SecurityException) as e:
        print(f"[ERROR] Validation/Security error for user '{get_user_id_from_context(ctx)}': {e}")
        return json5.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        print(f"[FATAL ERROR] for user '{get_user_id_from_context(ctx)}': {e}")
        import traceback
        traceback.print_exc()
        return json5.dumps({"status": "error", "message": "An internal server error occurred."}, ensure_ascii=False)