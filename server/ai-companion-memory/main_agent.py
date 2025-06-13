# main_agent.py

from qwen_agent.agents import Assistant
from core.config import LLM_CONFIG, MCP_SERVER_PORT
import json
import logging

# Configure logging to see detailed output from the agent framework
logging.basicConfig(level=logging.INFO)
# Silence the overly verbose httpx logger unless there's an error
logging.getLogger("httpx").setLevel(logging.WARNING)


def run_chat_session():
    user_id = input("Enter your User ID for this session (e.g., sarthak): ").strip()
    if not user_id:
        print("User ID cannot be empty.")
        return

    print(f"\n--- Session started for user: {user_id} ---")
    print("You can now talk to the AI companion.")
    print("Try saying: 'Remember my dog is named Sparky' or 'my cat's name is algebra'.")
    print("Type 'exit' or 'quit' to end the session.\n")

    mcp_server_url = f"http://localhost:{MCP_SERVER_PORT}/sse"
    
    tools = [{
        "mcpServers": {
            "memory_system": {
                "url": mcp_server_url,
                "headers": {"X-User-ID": user_id}
            }
        }
    }]

    try:
        agent = Assistant(
            llm=LLM_CONFIG,
            function_list=tools,
            system_message='You are a helpful AI companion. Use the `manage_memory` tool to remember and recall information. Formulate the `task` parameter as a clear, self-contained instruction or question.'
        )
        print("Agent initialized successfully and connected to MCP server.")
    except Exception as e:
        print("\n--- AGENT INITIALIZATION FAILED ---")
        print(f"Failed to connect to the MCP server at {mcp_server_url}")
        print("Please ensure 'mcp_server.py' is running and accessible.")
        print(f"Error details: {e}")
        return

    # This will store the full conversation history between turns
    messages = []
    
    while True:
        try:
            query = input(f'[{user_id}]> ')
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if query.lower() in ['exit', 'quit']:
            break
        
        # Add the latest user message to the history for the agent to process
        messages.append({'role': 'user', 'content': query})
        
        print('\n[Companion]> ', end='')
        
        # This will hold the messages generated *during this specific turn*
        turn_history = []
        try:
            # The agent.run() is a generator that yields intermediate steps
            # The final item yielded is a list of all messages in the turn.
            for item in agent.run(messages=messages):
                # The final output is a list, intermediate steps are dicts
                if isinstance(item, list):
                    turn_history = item # This is the full history of the turn
                    continue

                # Process intermediate dictionaries for streaming output
                if isinstance(item, dict):
                    if item['role'] == 'assistant' and item.get('content'):
                        print(item['content'], end='', flush=True)
                    
                    elif item['role'] == 'tool_outputs':
                        print("\n[DEBUG: Tool Call Executed]")
                        content = item.get('content')
                        if not content:
                            print("[DEBUG] Tool output content is empty.")
                            continue
                        try:
                            # The content is a list containing one or more tool call dicts
                            if isinstance(content, str):
                                tool_outputs_list = json.loads(content)
                            else:
                                tool_outputs_list = content

                            if not isinstance(tool_outputs_list, list):
                                print(f"[DEBUG] Unexpected tool output format (not a list): {tool_outputs_list}")
                                continue

                            for tool_output in tool_outputs_list:
                                # The result from our MCP is a JSON string
                                result_str = tool_output.get('result', '{}')
                                result_json = json.loads(result_str)
                                print(json.dumps(result_json, indent=2))

                        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
                            print(f"\n[DEBUG] Error parsing tool output. Raw content: {content}. Error: {e}")
                        
                        print('[Companion]> ', end='', flush=True)

        except Exception as e:
            print(f"\n[ERROR] An error occurred during agent execution: {type(e).__name__}: {e}")
            messages.pop() # Remove the failing user message from history
            continue

        # Append the history of the completed turn to the main conversation history
        # We need to filter out the old messages that were passed in
        new_messages = [msg for msg in turn_history if msg not in messages]
        messages.extend(new_messages)

        print() # Final newline for the next user prompt

if __name__ == "__main__":
    run_chat_session()