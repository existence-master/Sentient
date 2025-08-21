import json
from qwen_agent.agents import Assistant

# --- Configuration ---
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}

# WhatsApp MCP Server Configuration
mcp_server_url = "http://127.0.0.1:9024/sse"

# IMPORTANT: Replace with a valid User ID from your MongoDB that has the WhatsApp agent connected
USER_ID = "YOUR_USER_ID_HERE"

# --- Agent Setup ---
tools = [{
    "mcpServers": {
        "whatsapp_server": {
            "url": mcp_server_url,
            "headers": {"X-User-ID": USER_ID},
        }
    }
}]

print("Initializing Qwen agent for WhatsApp...")
agent = Assistant(
    llm=llm_cfg,
    function_list=tools,
    name="WhatsAppAgent",
    description="An agent that can send WhatsApp messages.",
    system_message="You are an assistant that can send WhatsApp messages on the user's behalf using their connected number."
)

# --- Interactive Chat Loop ---
def run_agent_interaction():
    print("\n--- WhatsApp Agent Ready ---")
    print("What message would you like to send?")
    print("Type 'quit' or 'exit' to end the session.")
    print("\nExample commands:")
    print("  - what are my unread messages?")
    print("  - check my unread messages and summarize them for me")
    print("  - send the location of the Eiffel Tower (lat 48.8584, lon 2.2945) to John Doe")
    print("  - send a message to John Doe saying 'Hi there!'")
    print("  - send a WhatsApp message saying 'Hello from Sentient!'")
    print("-" * 25)

    messages = []
    while True:
        try:
            print("\nYou: ", end="")
            user_input = input()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋  Goodbye!")
                break

            messages.append({'role': 'user', 'content': user_input})
            print("\nAgent: ", end="", flush=True)
            
            last_assistant_text = ""
            final_response_from_run = None
            final_assistant_message = None
            for response in agent.run(messages=messages):
                if isinstance(response, list) and response and response[-1].get("role") == "assistant":
                    current_text = response[-1].get("content", "")
                    if isinstance(current_text, str):
                        delta = current_text[len(last_assistant_text):]
                        print(delta, end="", flush=True)
                        last_assistant_text = current_text
                    final_assistant_message = response[-1]

            print()
            if final_assistant_message:
                messages.append(final_assistant_message)
            else:
                print("I could not process that request.")
                messages.pop()

        except KeyboardInterrupt:
            print("\n👋  Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_agent_interaction()

