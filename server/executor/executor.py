from qwen_agent.agents import Assistant
from typing import Optional, Dict, Any, Union, List
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('server/.env')

# Define LLM configuration for Ollama
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
}


def get_mcp_servers_for_user(user_id: str) -> Dict:
    mcp_servers = {
        'time': {
            'command': 'uvx',
            'args': ['mcp-server-time', '--local-timezone=Asia/Kolkata']
        },
        "brave-search": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-brave-search"
            ],
            "env": {
                "BRAVE_API_KEY": os.getenv('BRAVE_API_KEY')
            }
        }
    }
    google_tools = ["gmail", "gcalendar", "gdrive", "gdocs", "gslides", "gsheet"]
    for tool in google_tools:
        mcp_servers[tool] = {
            'command': 'uvx',
            'args': ['mcp-gsuite' if tool in ['gmail', 'gcalendar'] else 'mcp-' + tool, '--credentials-dir', f'credentials/{user_id}'],
            'env': {
                'GOOGLE_CLIENT_ID': os.getenv('GOOGLE_CLIENT_ID'),
                'GOOGLE_CLIENT_SECRET': os.getenv('GOOGLE_CLIENT_SECRET')
            }
        }
    return mcp_servers

def execute_plan_with_bot(plan: str, user_id: str, available_tools: List[str] = all_tools) -> Any:
    mcp_servers = get_mcp_servers_for_user(user_id)
    tools_config = [{'mcpServers': mcp_servers}]
    execute_bot = Assistant(llm=llm_cfg, function_list=tools_config)
    messages = [
        {'role': 'system', 'content': 'Execute this plan using the tools.'},
        {'role': 'user', 'content': plan}
    ]
    for responses in execute_bot.run(messages=messages):
        pass
    return responses