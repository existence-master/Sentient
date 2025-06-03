from qwen_agent.agents import Assistant

# Define LLM configuration for Ollama
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}

# Define tools with MCP servers
tools = [
    {'mcpServers': {
        'time': {
            'command': 'uvx',
            'args': ['mcp-server-time', '--local-timezone=Asia/Kolkata']
        }
    }},
]

# Define the agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Example query to test the MCP server
messages = [{'role': 'user', 'content': 'What is the current time in Shanghai?'}]
for responses in bot.run(messages=messages):
    pass
print(responses)