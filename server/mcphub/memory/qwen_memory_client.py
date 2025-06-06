import json
from qwen_agent.agents import Assistant
from typing import List, Dict

# --- LLM and Agent Configuration ---
llm_cfg = {
    'model': 'qwen3:4b',  # Using a slightly larger model like 7b is recommended for better reasoning
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}

tools = [{"mcpServers": {"memory_tools": {"url": "http://localhost:8001/sse"}}}]

# --- REWRITTEN & ENHANCED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a highly advanced personal AI companion. Your most critical function is to build a deep, personal understanding of your user, '{user_id}', through memory.

**Your Core Workflow - A Strict Mandate:**
1.  **LISTEN:** Analyze the user's message.
2.  **REMEMBER (MANDATORY):** Before crafting any response, you MUST use the `search_memories` tool. Ask yourself: "What do I already know about this person or topic?"
3.  **SYNTHESIZE (CRITICAL):**
    -   The `search_memories` tool will return clean facts and reminders. Review the `fact` and `content` fields.
    -   **DO NOT DESCRIBE THE TOOL's JSON OUTPUT.** Never mention "similarity scores," "long_term_facts," or "short_term_reminders" to the user. This is internal context for you only.
    -   Integrate the retrieved memories seamlessly and naturally into your response to show you remember the user.
4.  **RESPOND:** Talk to the user like a friend who remembers things.
5.  **UPDATE MEMORY:** After responding, decide if new, permanent information was shared.
    -   If yes, use the `save_long_term_fact` tool.
    -   Define relationships clearly. For the user, always use the keyword 'user' (e.g., `relations=[{{"from": "user", "to": "Jordan", "type": "FRIEND_OF"}}]`).
    -   For temporary items, use `add_short_term_memory`.

**Example Thought Process:**
User says: "My sister, Chloe, just got a dog."
Your internal thought process:
1.  `search_memories` for "Chloe" or "sister". Let's say it returns nothing.
2.  I will respond to the user and then update my memory.
3.  My response: "That's wonderful news! What kind of dog did Chloe get? I'll remember that she's your sister."
4.  After responding, call `save_long_term_fact` with `fact_text="Chloe has a dog"`, `category="Social"`, `relations=[{{"from": "Chloe", "to": "user", "type": "SISTER_OF"}}]`.

Your primary directive is to make the user feel remembered and understood. Your internal tool use should be invisible to them.
"""

agent = Assistant(
    llm=llm_cfg,
    function_list=tools,
    name="MemoryEnhancedAgent",
    description="Agent using a robust knowledge graph and vector search memory system."
)

def run_single_task(task_name: str, messages: List[Dict]):
    """Helper function to run a single agent task and print the output."""
    print(f"\n{'-'*20} {task_name} {'-'*20}")
    final_response = None
    # The loop consumes the generator to get the final response
    for response in agent.run(messages=messages):
        final_response = response

    if final_response:
        print("Final Agent Output:")
        # We only print the 'content' of the last message for cleaner output
        if isinstance(final_response, list) and final_response:
             print(json.dumps(final_response[-1]['content'], indent=2))
        else:
             print(json.dumps(final_response, indent=2))
    else:
        print("No response received for this task.")

def run_agent_narrative():
    """Runs a series of tasks to demonstrate the improved memory system's capabilities."""
    USER_ID = "user_alex_007"
    
    # Task 1: First interaction, learning the user's name and profession.
    run_single_task(
        "Task 1: First Meaningful Interaction",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "Hi there, you can call me Alex. I'm a graphic designer working at a company called DreamCanvas."}
        ]
    )

    # Task 2: Adding a preference
    run_single_task(
        "Task 2: Learning a Preference",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "I'm a huge fan of spicy food, especially Thai."}
        ]
    )

    # Task 3: Testing retrieval. The agent should synthesize its knowledge.
    run_single_task(
        "Task 3: Synthesis & Retrieval",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "I'm thinking of getting lunch and then getting back to my design work. Any ideas?"}
        ]
    )
    
    # Task 4: Adding a complex social fact with multiple relations.
    run_single_task(
        "Task 4: Complex Social Fact",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "My best friend, Jordan, who is a chef, is getting married next month!"}
        ]
    )

    # Task 5: A vague question requiring synthesis.
    run_single_task(
        "Task 5: Vague Question Requiring Memory",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "What's new with my friends and me?"}
        ]
    )

    # Task 6: Adding a short-term, categorized reminder.
    run_single_task(
        "Task 6: Adding a Short-Term Reminder",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "Can you remind me in an hour to check on the final design mockups for the new project? It's for work."}
        ]
    )
    
    # Task 7: Getting the full user profile, which should now be well-structured.
    run_single_task(
        "Task 7: Full Profile Summary",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT.format(user_id=USER_ID)},
            {'role': 'user', 'content': "Give me a summary of everything you know about me."}
        ]
    )

if __name__ == "__main__":
    print(f"Using LLM model: {llm_cfg['model']}")
    run_agent_narrative()