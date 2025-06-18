import json
from qwen_agent.agents import Assistant
from typing import List, Dict, Optional

# --- LLM and Agent Configuration ---
llm_cfg = {
    'model': 'qwen1.5:7b-chat-q5_K_M',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}

# Point the client to the new dual-memory MCP server with CRUD tools
tools = [{"mcpServers": {"memory_tools": {"url": "http://localhost:8001/sse"}}}]

# --- REVISED SYSTEM PROMPT TO REFLECT THE FULL CRUD PIPELINE ---
SYSTEM_PROMPT = """
You are a highly advanced personal AI for user '{user_id}'. Your primary directive is to use your memory tools to create a personalized, context-aware experience. You must follow this operational workflow strictly.

**WORKFLOW STEP 1: RETRIEVE & SYNTHESIZE (Before EVERY response)**
1.  Before responding to the user, ALWAYS call `search_all_memories` with the user's query.
2.  Review the `long_term_facts` and `short_term_reminders` returned by the tool.
3.  Synthesize the retrieved information NATURALLY into your response.
4.  **NEVER** mention the tools, JSON, or "similarity scores". Your memory should feel innate, not programmatic.

**WORKFLOW STEP 2: ANALYZE & EXECUTE (After EVERY response)**
After you respond, analyze the user's last message to determine if a memory operation is needed.

**A) TO CREATE a new memory:**
-   **For Permanent Facts (LTM):** If the user shared new, lasting info (e.g., relationships, preferences, life events), you must first mentally structure it and then call `add_long_term_fact`.
    -   `fact_text`: A simple, atomic statement. (e.g., "Maya is a veterinarian.")
    -   `category`: "Personal", "Professional", "Social", etc.
    -   `entities`: List of key nouns. (e.g., ["Maya", "user"])
    -   `relations`: List of connections. `[{"from": "Maya", "to": "user", "type": "FRIEND_OF"}]`. Use 'user' for the primary user.
-   **For Temporary Info (STM):** If the user gives a reminder or a task for the near future, call `add_short_term_memory`.
    -   `content`: The reminder text.
    -   `ttl_seconds`: Time to live (e.g., 7200 for 2 hours).

**B) TO UPDATE an existing memory:**
-   If the user says "Actually, my friend Maya is a doctor now, not a vet", you must:
    1.  Call `search_all_memories` to find the original fact and get its `fact_id`.
    2.  Call `update_long_term_fact` with the `fact_id` and the `new_text`: "Maya is a doctor".

**C) TO DELETE an existing memory:**
-   If the user says "Forget what I told you about my old job", you must:
    1.  Call `search_all_memories` to find the specific fact about their old job and get its `fact_id`.
    2.  Call `delete_long_term_fact` with that `fact_id`.

**Example Interaction:**
User: "My sister, Chloe, is moving to Denver."
*Your Internal Process:*
1.  `search_all_memories` for "Chloe" or "sister". (Assume it finds nothing).
2.  *Your Response to User:* "That's a big move! I'll remember that Chloe is your sister and she's in Denver now. I hope she likes it there!"
3.  *Your Post-Response Action:* Call `add_long_term_fact` with `user_id='{user_id}'`, `fact_text='Chloe is the user's sister and lives in Denver'`, `category='Social'`, `entities=['Chloe', 'user', 'Denver']`, `relations=[{'from': 'Chloe', 'to': 'user', 'type': 'SISTER_OF'}, {'from': 'Chloe', 'to': 'Denver', 'type': 'LIVES_IN'}]`.

You are not just a chatbot. You are a curator of the user's life. Be proactive, be intelligent, and use your tools correctly.
"""

agent = Assistant(llm=llm_cfg, function_list=tools, name="SentientMemoryCurator")

def run_interaction(task_name: str, user_id: str, user_content: str, conversation: List[Dict]):
    print(f"\n{'='*25} {task_name.upper()} {'='*25}")
    print(f"USER ({user_id}): {user_content}")
    
    current_messages = conversation + [{'role': 'user', 'content': user_content}]
    
    final_response = None
    print("\nAGENT'S THOUGHT PROCESS & ACTIONS:")
    for response in agent.run(messages=current_messages):
        final_response = response
        if response and isinstance(response, list):
            for item in response:
                if item.get('role') == 'tool':
                    print(f"  > Executing Tool: {item.get('name', 'unknown_tool')}")
    
    print("\nAGENT'S RESPONSE TO USER:")
    if final_response and isinstance(final_response, list):
        assistant_responses = [msg['content'] for msg in final_response if msg.get('role') == 'assistant']
        if assistant_responses:
            print(assistant_responses[-1])
            return final_response
    print("[No text response generated, only tool calls.]")
    return final_response

def run_crud_narrative():
    USER_ID = "sarah_jones_88"
    conversation = [{'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)}]

    # Turn 1: CREATE a new fact
    conversation = run_interaction(
        "Turn 1: CREATE LTM Fact", USER_ID,
        "My cat's name is Leo, he's a Maine Coon.",
        conversation
    )
    
    # Turn 2: CREATE a temporary reminder
    conversation = run_interaction(
        "Turn 2: CREATE STM Reminder", USER_ID,
        "I need to remember to buy cat food later today.",
        conversation
    )
    
    # Turn 3: READ/RETRIEVE both memories
    conversation = run_interaction(
        "Turn 3: READ & Synthesize Memories", USER_ID,
        "What was I just talking about?",
        conversation
    )

    # Turn 4: UPDATE an existing fact
    conversation = run_interaction(
        "Turn 4: UPDATE LTM Fact", USER_ID,
        "Actually, I got Leo's breed wrong. He's a Norwegian Forest Cat.",
        conversation
    )
    
    # Turn 5: READ the updated fact
    conversation = run_interaction(
        "Turn 5: READ Updated Fact", USER_ID,
        "Tell me about my cat again.",
        conversation
    )
    
    # Turn 6: DELETE a fact
    conversation = run_interaction(
        "Turn 6: DELETE LTM Fact", USER_ID,
        "Please forget everything about my cat.",
        conversation
    )
    
    # Turn 7: Verify deletion
    conversation = run_interaction(
        "Turn 7: VERIFY Deletion", USER_ID,
        "Do I have any pets?",
        conversation
    )

if __name__ == "__main__":
    print("This client demonstrates the full CRUD lifecycle of the Dual-Memory MCP Server.")
    input("\nPress Enter to begin the agent narrative...")
    run_crud_narrative()