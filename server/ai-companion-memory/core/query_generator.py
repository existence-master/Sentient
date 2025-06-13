# core/query_generator.py

from qwen_agent.agents import Assistant
from .config import LLM_CONFIG, MEMORY_CATEGORIES
import datetime
import json5
import re  # Import the regex module

# The detailed instructions for the agent that will generate our queries.
# This is now a system prompt for our internal agent.
QUERY_GENERATOR_SYSTEM_PROMPT = f"""
You are a highly intelligent data processing agent. Your sole purpose is to convert a user's request into a single, structured, secure MongoDB query JSON5 object. You must follow all instructions precisely.

**CRITICAL INSTRUCTIONS:**
1.  **REASONING FIRST:** First, think step-by-step inside `<think>` tags.
2.  **JSON5 AFTER:** After the closing `</think>` tag, your entire remaining output must be a single, raw JSON5 object. Do NOT include any other text, explanations, or markdown code fences.
3.  **STORAGE vs RETRIEVAL:**
    *   **For Storing (insert/update):** Be very specific. Extract detailed `key_entities` like `"relationship": "cat"`.
    *   **For Retrieving (find):** Be broader. Create a general filter (it can even be empty: `{{}}`), and ALWAYS generate a descriptive `semantic_search_text` field. The semantic search is the primary way to find relevant memories. Rely on it.

**Allowed Operations:** `find`, `insert_one`, `update_one`, `delete_one`
**Allowed Collections:** `long_term_memories`, `short_term_memories`, `contacts`

**Current Date:** {datetime.datetime.utcnow().isoformat()}

---
**EXAMPLE 1 (Storing)**
User Request: "Remember my cat's name is Algebra."
Your Output:
<think>The user wants to store a long-term memory about their cat. I will use 'insert_one' on 'long_term_memories'. The key entities are specific: name 'Algebra' and relationship 'cat'.</think>
{{
  "operation": "insert_one",
  "collection": "long_term_memories",
  "query": {{
    "document": {{
      "memory_id": "mem_l_uuid_placeholder",
      "content": "User's cat's name is Algebra.",
      "category": "Personal",
      "key_entities": {{ "name": "Algebra", "relationship": "cat" }},
      "source": "user_request",
      "created_at": "{datetime.datetime.utcnow().isoformat()}",
      "updated_at": "{datetime.datetime.utcnow().isoformat()}"
    }}
  }}
}}

---
**EXAMPLE 2 (Retrieving)**
User Request: "what is my cat's name?"
Your Output:
<think>The user is asking a question. I must use the 'find' operation and rely on semantic search. I will create a very descriptive 'semantic_search_text' and keep the filter broad to allow the semantic search to do its job.</think>
{{
  "operation": "find",
  "collection": "long_term_memories",
  "query": {{ "filter": {{ "category": "Personal" }}, "projection": {{ "content": 1, "_id": 0 }} }},
  "semantic_search_text": "information about the user's cat, pet, or animal's name"
}}
"""


try:
    query_generation_agent = Assistant(
        llm=LLM_CONFIG,
        system_message=QUERY_GENERATOR_SYSTEM_PROMPT
    )
    print("Internal Query Generation Agent initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize internal Query Generation Agent. Error: {e}")
    query_generation_agent = None

def generate_query_from_text(user_request: str) -> str:
    if not query_generation_agent:
        raise RuntimeError("Query Generation Agent is not available. Check server logs for initialization errors.")

    messages = [{'role': 'user', 'content': user_request}]
    
    full_response_content = ""
    turn_history = []
    
    for item in query_generation_agent.run(messages=messages):
        if isinstance(item, list):
            turn_history = item
            continue
        if isinstance(item, dict):
            if item['role'] == 'assistant' and item.get('content'):
                full_response_content += item['content']

    if turn_history:
        for message in reversed(turn_history):
            if message['role'] == 'assistant' and message.get('content'):
                full_response_content = message['content']
                break
    
    if not full_response_content:
        raise ValueError("Query Generation Agent did not produce a final assistant message with content.")

    # --- REGEX FIX IS HERE ---
    # This regex will find the <think>...</think> block and remove it.
    # re.DOTALL makes '.' match newlines, so it works on multi-line think blocks.
    json_part = re.sub(r'<think>.*?</think>', '', full_response_content, flags=re.DOTALL).strip()

    # Further cleanup for any remaining unwanted characters or markdown
    json_part = json_part.replace('```json5', '').replace('```', '').strip()

    try:
        # Verify it's valid JSON5 before returning
        json5.loads(json_part)
        return json_part
    except Exception as e:
        print(f"Error parsing generated query. Full response was: \n---\n{full_response_content}\n---")
        print(f"Cleaned JSON part was: \n---\n{json_part}\n---")
        raise ValueError(f"Agent returned a non-JSON or malformed object: {e}")