# Create new file: src/server/mcp_hub/gtasks/main.py
import os
import asyncio
import json
import inspect
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.utilities.logging import configure_logging, get_logger
from composio import Composio
from main.config import COMPOSIO_API_KEY

from . import auth, prompts

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# --- Composio Client ---
composio = Composio(api_key=COMPOSIO_API_KEY)

# Conditionally load .env for local development
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

# --- Server Initialization ---
mcp = FastMCP(
    name="GTasksServer",
    instructions="Provides tools to manage to-do lists and tasks in Google Tasks.",
)

@mcp.resource("prompt://gtasks-agent-system")
def get_gtasks_system_prompt() -> str:
    return prompts.gtasks_agent_system_prompt

async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    tool_name = inspect.stack()[1].function
    logger.info(f"Executing tool: {tool_name} with parameters: {kwargs}")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gtasks")

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        result = await asyncio.to_thread(
            composio.tools.execute,
            action_name,
            arguments=filtered_kwargs,
            connected_account_id=connection_id,
            dangerously_skip_version_check=True
        )
        
        serializable_result = json.loads(json.dumps(result, default=str))
        return {"status": "success", "result": serializable_result}
    except Exception as e:
        logger.error(f"Tool execution failed for action '{action_name}': {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# --- Tool Definitions ---

@mcp.tool()
async def clear_tasks(ctx: Context, tasklist: str) -> Dict:
    """Permanently clears all completed tasks from a specified google tasks list; this action is destructive and idempotent."""
    return await _execute_tool(ctx, "GOOGLETASKS_CLEAR_TASKS", tasklist=tasklist)

@mcp.tool()
async def create_task_list(ctx: Context, tasklist_title: str) -> Dict:
    """Creates a new task list with the specified title."""
    return await _execute_tool(ctx, "GOOGLETASKS_CREATE_TASK_LIST", tasklist_title=tasklist_title)

@mcp.tool()
async def delete_task(ctx: Context, task_id: str, tasklist_id: str) -> Dict:
    """Deletes a specified task from a given task list in google tasks."""
    return await _execute_tool(ctx, "GOOGLETASKS_DELETE_TASK", task_id=task_id, tasklist_id=tasklist_id)

@mcp.tool()
async def delete_task_list(ctx: Context, tasklist_id: str) -> Dict:
    """Permanently deletes an existing google task list, identified by `tasklist id`, along with all its tasks; this operation is irreversible."""
    return await _execute_tool(ctx, "GOOGLETASKS_DELETE_TASK_LIST", tasklist_id=tasklist_id)

@mcp.tool()
async def get_task(ctx: Context, task_id: str, tasklist_id: str) -> Dict:
    """Use to retrieve a specific google task if its `task id` and parent `tasklist id` are known."""
    return await _execute_tool(ctx, "GOOGLETASKS_GET_TASK", task_id=task_id, tasklist_id=tasklist_id)

@mcp.tool()
async def get_task_list(ctx: Context, tasklist_id: str) -> Dict:
    """Retrieves a specific task list from the user's google tasks if the `tasklist id` exists for the authenticated user."""
    return await _execute_tool(ctx, "GOOGLETASKS_GET_TASK_LIST", tasklist_id=tasklist_id)

@mcp.tool()
async def insert_task(ctx: Context, tasklist_id: str, title: str, status: str, completed: Optional[str] = None, deleted: Optional[bool] = None, due: Optional[str] = None, etag: Optional[str] = None, hidden: Optional[bool] = None, id: Optional[str] = None, notes: Optional[str] = None, task_parent: Optional[str] = None, task_previous: Optional[str] = None) -> Dict:
    """Creates a new task in a given `tasklist id`, optionally as a subtask of an existing `task parent` or positioned after an existing `task previous` sibling."""
    return await _execute_tool(ctx, "GOOGLETASKS_INSERT_TASK", tasklist_id=tasklist_id, title=title, status=status, completed=completed, deleted=deleted, due=due, etag=etag, hidden=hidden, id=id, notes=notes, task_parent=task_parent, task_previous=task_previous)

@mcp.tool()
async def list_tasks(ctx: Context, tasklist_id: str, completedMax: Optional[str] = None, completedMin: Optional[str] = None, dueMax: Optional[str] = None, dueMin: Optional[str] = None, maxResults: Optional[int] = None, pageToken: Optional[str] = None, showCompleted: Optional[bool] = None, showDeleted: Optional[bool] = None, showHidden: Optional[bool] = None, updatedMin: Optional[str] = None) -> Dict:
    """Retrieves tasks from a google tasks list; all date/time strings must be rfc3339 utc, and `showcompleted` must be true if `completedmin` or `completedmax` are specified."""
    return await _execute_tool(ctx, "GOOGLETASKS_LIST_TASKS", tasklist_id=tasklist_id, completedMax=completedMax, completedMin=completedMin, dueMax=dueMax, dueMin=dueMin, maxResults=maxResults, pageToken=pageToken, showCompleted=showCompleted, showDeleted=showDeleted, showHidden=showHidden, updatedMin=updatedMin)

@mcp.tool()
async def list_task_lists(ctx: Context, maxResults: int = 20, pageToken: Optional[str] = None) -> Dict:
    """Fetches the authenticated user's task lists from google tasks; results may be paginated."""
    return await _execute_tool(ctx, "GOOGLETASKS_LIST_TASK_LISTS", maxResults=maxResults, pageToken=pageToken)

@mcp.tool()
async def move_task(ctx: Context, task: str, tasklist: str, destinationTasklist: Optional[str] = None, parent: Optional[str] = None, previous: Optional[str] = None) -> Dict:
    """Moves the specified task to another position in the destination task list."""
    return await _execute_tool(ctx, "GOOGLETASKS_MOVE_TASK", task=task, tasklist=tasklist, destinationTasklist=destinationTasklist, parent=parent, previous=previous)

@mcp.tool()
async def patch_task(ctx: Context, task_id: str, tasklist_id: str, title: str, status: str, completed: Optional[str] = None, deleted: Optional[bool] = None, due: Optional[str] = None, etag: Optional[str] = None, hidden: Optional[bool] = None, id: Optional[str] = None, notes: Optional[str] = None) -> Dict:
    """Partially updates an existing task (identified by `task id`) within a specific google task list (identified by `tasklist id`)."""
    return await _execute_tool(ctx, "GOOGLETASKS_PATCH_TASK", task_id=task_id, tasklist_id=tasklist_id, title=title, status=status, completed=completed, deleted=deleted, due=due, etag=etag, hidden=hidden, id=id, notes=notes)

@mcp.tool()
async def patch_task_list(ctx: Context, tasklist_id: str, updated_title: str) -> Dict:
    """Updates the title of an existing google tasks task list."""
    return await _execute_tool(ctx, "GOOGLETASKS_PATCH_TASK_LIST", tasklist_id=tasklist_id, updated_title=updated_title)

@mcp.tool()
async def update_task(ctx: Context, task: str, tasklist: str, due: Optional[str] = None, notes: Optional[str] = None, status: Optional[str] = None, title: Optional[str] = None) -> Dict:
    """Updates the specified task."""
    return await _execute_tool(ctx, "GOOGLETASKS_UPDATE_TASK", task=task, tasklist=tasklist, due=due, notes=notes, status=status, title=title)

@mcp.tool()
async def update_task_list(ctx: Context, tasklist_id: str, title: str) -> Dict:
    """Updates the authenticated user's specified task list."""
    return await _execute_tool(ctx, "GOOGLETASKS_UPDATE_TASK_LIST", tasklist_id=tasklist_id, title=title)


# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9028))
    
    print(f"Starting GTasks MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)