import os
import logging
from dotenv import load_dotenv
from fastmcp import FastMCP, Context

from . import tools

# --- Logging and Environment Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

# --- MCP Server Initialization ---
mcp = FastMCP(
    name="OrchestratorServer",
    instructions="Provides tools for the long-form task orchestrator agent to manage its state, plan, and interactions.",
)

# --- Register all tools from the tools module ---
mcp.tool(tools.update_plan)
mcp.tool(tools.update_context)
mcp.tool(tools.get_context)
mcp.tool(tools.create_subtask)
mcp.tool(tools.wait_for_response)
mcp.tool(tools.ask_user_clarification)
mcp.tool(tools.mark_step_complete)
mcp.tool(tools.evaluate_completion)
mcp.tool(tools.get_related_integrations_data)

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9027))

    logger.info(f"Starting Orchestrator MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)
