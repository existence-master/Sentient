import os
import asyncio
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from qwen_agent.agents import Assistant
from json_extractor import JsonExtractor

from . import auth, prompts

# --- LLM and Environment Configuration ---
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "qwen3:4b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

def get_generator_agent():
    """Initializes a Qwen agent specifically for generating JSON outlines internally."""
    llm_cfg = {
        'model': OPENAI_MODEL_NAME,
        'model_server': f"{OPENAI_API_BASE_URL.rstrip('/')}/v1",
        'api_key': OPENAI_API_KEY,
    }
    return Assistant(llm=llm_cfg, system_message=prompts.JSON_GENERATOR_SYSTEM_PROMPT, function_list=[])

mcp = FastMCP(
    name="GDocsServer",
    instructions="This server provides tools to create Google Docs in a two-step process.",
)

@mcp.resource("prompt://gdocs-agent-system")
def get_gdocs_system_prompt() -> str:
    """Provides the system prompt that instructs the main orchestrator agent on how to use the gdocs tools."""
    return prompts.MAIN_AGENT_SYSTEM_PROMPT

@mcp.tool()
async def generate_document_json(ctx: Context, topic: str, previous_tool_response: Optional[str] = "{}") -> Dict[str, Any]:
    """
    Step 1: Generates the structured JSON needed to create a Google Doc.
    Provide a topic and optionally the JSON result from a previous tool. This tool will use an internal AI to create the detailed JSON for the document's title and sections.
    The output of this tool should be passed directly to the `execute_document_creation` tool.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        user_profile = await auth.users_collection.find_one({"user_id": user_id})
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User") if user_profile else "User"
        
        agent = get_generator_agent()
        user_prompt = prompts.gdocs_internal_user_prompt.format(
            topic=topic, 
            username=username, 
            previous_tool_response=previous_tool_response
        )
        messages = [{'role': 'user', 'content': user_prompt}]
        
        final_content_str = ""
        for chunk in agent.run(messages=messages):
            if isinstance(chunk, list) and chunk:
                last_message = chunk[-1]
                if last_message.get("role") == "assistant" and isinstance(last_message.get("content"), str):
                    final_content_str = last_message["content"]
        
        if not final_content_str:
            raise Exception("The document generator agent returned an empty response.")
            
        document_json = JsonExtractor.extract_valid_json(final_content_str)
        if not document_json or "title" not in document_json or "sections_json" not in document_json:
             raise Exception(f"Generator agent failed to produce valid JSON. Response: {final_content_str}")

        return {"status": "success", "result": document_json}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def execute_document_creation(ctx: Context, title: str, sections_json: str) -> Dict[str, Any]:
    """
    Step 2: Creates the actual Google Document from the structured JSON.
    This tool takes the JSON output from the `generate_document_json` tool and creates the file.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        creds = await auth.get_google_creds(user_id)
        docs_service = auth.authenticate_gdocs(creds)
        drive_service = auth.authenticate_gdrive(creds)
        
        sections = json.loads(sections_json)

        document_result = await asyncio.to_thread(
            utils.create_google_document_sync, docs_service, drive_service, title, sections
        )
        
        return document_result
    except Exception as e:
        return {"status": "failure", "error": str(e)}

if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9004))
    
    print(f"Starting GDocs MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)