from qwen_agent.agents import Assistant
import asyncio
import json # For pretty printing the response

# Your LLM config (e.g., Qwen3)
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/', # Ollama endpoint
    'api_key': 'EMPTY', # Ollama doesn't require an API key by default
    # 'generate_cfg': {
    #     'stop': ['<|endoftext|>', '<|im_end|>'] # Add appropriate stop tokens for your model
    # }
}

# Define the remote MCP server endpoint URL for the memory server
memory_mcp_server_url = "http://localhost:8001/sse"  # Port 8001 as in memory_server.py

# Define tools connecting to the remote MCP server
tools = [{
    "mcpServers": {
        "memory_tools": {  # A descriptive name for this server connection
            "url": memory_mcp_server_url,
        }
    }
}]

agent = Assistant(
    llm=llm_cfg,
    function_list=tools,
    name="MemoryEnhancedAgent",
    description="Agent using Qwen model with a remote MCP server for memory management."
)

# --- Example Usage ---
def run_agent_tasks():
    # Unique user ID for this session/user
    USER_ID = "user_001"

    # test_messages_1 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'My name is Bob. Please remember this.'}
    # ]
    # print(f"\n--- Task 1: Remembering name (User ID: {USER_ID}) ---")
    # # Use agent.run() for asynchronous iteration
    # for responses_1 in agent.run(messages=test_messages_1):
    #     pass
    # if responses_1:
    #     print("Final Agent Output 1 (last yielded value):")
    #     print(json.dumps(responses_1, indent=2))
    # else:
    #     print("No responses received for Task 1.")


    # test_messages_2 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'I have a dentist appointment next Tuesday at 3 PM. Please remind me. Set a TTL of 7 days for this reminder.'}
    # ]
    # print(f"\n--- Task 2: Adding a short-term reminder (User ID: {USER_ID}) ---")
    # # Use agent.run()
    # for responses_2 in agent.run(messages=test_messages_2):
    #     pass
    # if responses_2:
    #     print("Final Agent Output 2 (last yielded value):")
    #     print(json.dumps(responses_2, indent=2))
    # else:
    #     print("No responses received for Task 2.")
        
    # test_messages_3 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'What is my name and do I have any upcoming appointments?'}
    # ]
    # print(f"\n--- Task 3: Retrieving memories (User ID: {USER_ID}) ---")
    # # Use agent.run()
    # for responses_3 in agent.run(messages=test_messages_3):
    #     pass
    # if responses_3:
    #     print("Final Agent Output 3 (last yielded value):")
    #     print(json.dumps(responses_3, indent=2))
    # else:
    #     print("No responses received for Task 3.")


    # test_messages_4 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'My favorite color is blue. Can you remember that for me?'}
    # ]
    # print(f"\n--- Task 4: Adding another long-term memory (User ID: {USER_ID}) ---")
    # # Use agent.run()
    # for responses_4 in agent.run(messages=test_messages_4):
    #     pass
    # if responses_4:
    #     print("Final Agent Output 4 (last yielded value):")
    #     print(json.dumps(responses_4, indent=2))
    # else:
    #     print("No responses received for Task 4.")


    # test_messages_5 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'Search all my memories for "blue" or "dentist".'}
    # ]
    # print(f"\n--- Task 5: Searching all memories (User ID: {USER_ID}) ---")
    # # Use agent.run()
    # for responses_5 in agent.run(messages=test_messages_5):
    #     pass
    # if responses_5:
    #     print("Final Agent Output 5 (last yielded value):")
    #     print(json.dumps(responses_5, indent=2))
    # else:
    #     print("No responses received for Task 5.")

    # test_messages_6 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'I have a temporary access code: "XYZ123". It is valid for only 10 seconds. Please store it.'} # Shorter TTL for testing
    # ]
    # print(f"\n--- Task 6: Adding a very short-lived STM (User ID: {USER_ID}) ---")
    # for responses_6 in agent.run(messages=test_messages_6):
    #     pass
    # if responses_6:
    #     print("Final Agent Output 6 (last yielded value):")
    #     print(json.dumps(responses_6, indent=2))
    # else:
    #     print("No responses received for Task 6.")
        
    # test_messages_7 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls.'},
    #     {'role': 'user', 'content': f'Do you remember my temporary access code "XYZ123"?'}
    # ]
    # print(f"\n--- Task 7: Checking for expired STM (User ID: {USER_ID}) ---")
    # for responses_7 in agent.run(messages=test_messages_7):
    #     pass
    # if responses_7:
    #     print("Final Agent Output 7 (last yielded value, should indicate code is not found or similar):")
    #     print(json.dumps(responses_7, indent=2))
    # else:
    #     print("No responses received for Task 7.")
    
    # test_messages_8 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls. You can now manage a knowledge graph.'},
    #     {'role': 'user', 'content': f'Please create an entity for "Alice Wonderland". She is a "person" and her observations are ["curious individual", "likes tea parties"]. Use user ID "{USER_ID}".'}
    # ]
    # print(f"\n--- Task 8: Creating KG Entity (User ID: {USER_ID}) ---")
    # for responses_8 in agent.run(messages=test_messages_8):
    #     pass # Get the last response
    # if responses_8:
    #     print("Final Agent Output 8:")
    #     print(json.dumps(responses_8, indent=2))
    # else:
    #     print("No responses received for Task 8.")

    # # --- Task 9: Create another KG entity and a relation ---
    # test_messages_9 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls. You can now manage a knowledge graph.'},
    #     {'role': 'user', 'content': f'Create an entity named "Mad Hatter" of type "character" with observations ["hosts tea parties", "wears a large hat"]. Then, create a relation: "Alice Wonderland" "knows" "Mad Hatter". Use user ID "{USER_ID}".'}
    # ]
    # print(f"\n--- Task 9: Creating KG Entity & Relation (User ID: {USER_ID}) ---")
    # for responses_9 in agent.run(messages=test_messages_9):
    #     pass
    # if responses_9:
    #     print("Final Agent Output 9:")
    #     print(json.dumps(responses_9, indent=2))
    # else:
    #     print("No responses received for Task 9.")

    # # --- Task 10: Search the KG ---
    # test_messages_10 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls. You can now manage a knowledge graph.'},
    #     {'role': 'user', 'content': f'Search my knowledge graph for "Alice" or "tea party". Use user ID "{USER_ID}".'}
    # ]
    # print(f"\n--- Task 10: Searching KG (User ID: {USER_ID}) ---")
    # for responses_10 in agent.run(messages=test_messages_10):
    #     pass
    # if responses_10:
    #     print("Final Agent Output 10:")
    #     print(json.dumps(responses_10, indent=2)) # This will call search_all_memories if LLM chooses it, or kg_search_nodes
    # else:
    #     print("No responses received for Task 10.")

    # # --- Task 11: Read the entire graph for the user ---
    # test_messages_11 = [
    #     {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls. You can now manage a knowledge graph.'},
    #     {'role': 'user', 'content': f'Show me my entire knowledge graph for user ID "{USER_ID}".'}
    # ]
    # print(f"\n--- Task 11: Reading KG (User ID: {USER_ID}) ---")
    # for responses_11 in agent.run(messages=test_messages_11):
    #     pass
    # if responses_11:
    #     print("Final Agent Output 11:")
    #     print(json.dumps(responses_11, indent=2))
    # else:
    #     print("No responses received for Task 11.")
    
    # --- Task 12: Testing Custom Task ---
    test_messages_12 = [
        {'role': 'system', 'content': f'You are a helpful assistant. Your current user_id is "{USER_ID}". ALWAYS use this user_id for memory operations and explicitly pass it to any memory tool calls. You can now manage a knowledge graph.'},
        {'role': 'user', 'content': f'Tell me about myself.'}
    ]
    print(f"\n--- Task 12: Telling User about Themself based on KG Profile (User ID: {USER_ID}) ---")
    for responses_12 in agent.run(messages=test_messages_12):
        pass
    if responses_12:
        print("Final Agent Output 12:")
        print(json.dumps(responses_12, indent=2))
    else:
        print("No responses received for Task 12.")


if __name__ == "__main__":
    # Ensure Ollama model name is correct, e.g., qwen2:1.5b, qwen2:7b etc.
    # Check `ollama list` for available models.
    print(f"Using LLM model: {llm_cfg['model']}")

    if hasattr(asyncio, 'run'):
        asyncio.run(run_agent_tasks())
    else:
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(run_agent_tasks())
        finally:
            loop.close()