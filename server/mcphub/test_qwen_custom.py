from qwen_agent.agents import Assistant
import asyncio


# Your LLM config (e.g., Qwen3)
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}

# Define the remote MCP server endpoint URL
mcp_server_url = "http://localhost:8000/sse"  # Use your actual public IP or domain

# Define tools connecting to the remote MCP server
tools = [{
    "mcpServers": {
        "custom": {
            "url": mcp_server_url,
            # No command or args needed here since server is already running
        }
    }
}]

agent = Assistant(
    llm=llm_cfg,
    function_list=tools,
    name="AgentWithRemoteMCP",
    description="Agent using Qwen3 model with remote MCP server tools"
)

messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Calculate my BMI using the custom server tool. My weight is 70 kg and height is 1.75 m.'}
    ]
for responses in agent.run(messages=messages):
    pass
print(responses)  # Print the final response from the agent