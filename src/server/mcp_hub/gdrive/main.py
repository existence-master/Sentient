import os
import asyncio
import textract
from typing import Dict, Any, Optional
from pathlib import Path


from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.prompts.prompt import Message
from fastmcp.utilities.logging import configure_logging, get_logger
from composio import Composio
from main.config import COMPOSIO_API_KEY

# Local imports
from . import auth
from . import prompts

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
    name="GDriveServer",
    instructions="Provides tools to search for files in Google Drive and read their content.",
)

# --- Prompt Registration ---
@mcp.resource("prompt://gdrive-agent-system")
def get_gdrive_system_prompt() -> str:
    return prompts.gdrive_agent_system_prompt

async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gdrive")

        # Composio's execute method is synchronous, so we use asyncio.to_thread
        result = await asyncio.to_thread(
            composio.tools.execute,
            action_name,
            arguments=kwargs,
            connected_account_id=connection_id,
            dangerously_skip_version_check=True
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool execution failed for action '{action_name}': {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# --- Tool Definitions ---

@mcp.tool()
async def gdrive_search(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Searches for files and folders in Google Drive using a `query`. The query should follow the Google Drive API's search syntax (e.g., `name contains 'report'` or `mimeType='application/pdf'`). Returns a list of matching files with their metadata.
    """
    logger.info(f"Executing tool: gdrive_search with query='{query}'")
    # The Composio action is GOOGLEDRIVE_LIST_FILES and its parameter is 'q'
    return await _execute_tool(ctx, "GOOGLEDRIVE_LIST_FILES", q=query, pageSize=20)

@mcp.tool()
async def gdrive_read_file(ctx: Context, file_id: str) -> Dict[str, Any]:
    """
    Retrieves the content of a file from Google Drive, given its `file_id`. It automatically converts Google Docs/Sheets/Slides to a readable text format and returns other files (like PDFs or images) as base64-encoded strings.
    """
    temp_file_path: Optional[Path] = None
    logger.info(f"Executing tool: gdrive_read_file with file_id='{file_id}'")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gdrive")

        # Step 1: Get metadata to determine file type
        metadata_result = await asyncio.to_thread(
            lambda: composio.tools.execute(
                "GOOGLEDRIVE_GET_FILE_METADATA",
                arguments={"fileId": file_id},
                connected_account_id=connection_id,
                dangerously_skip_version_check=True
            )
        )
        if not metadata_result.get("successful"):
            raise ToolError(f"Failed to get file metadata: {metadata_result.get('error', 'Unknown error')}")

        file_metadata = metadata_result.get("data", {})
        mime_type = file_metadata.get("mimeType")
        file_name = file_metadata.get("name")
        if not file_name:
            raise ToolError("Could not determine file name from metadata.")

        # Step 2: Determine export format or direct download
        export_mime_type = None
        if mime_type == "application/vnd.google-apps.document":
            export_mime_type = "text/plain"
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            export_mime_type = "text/csv"
        elif mime_type == "application/vnd.google-apps.presentation":
            export_mime_type = "text/plain"

        # Step 3: Download/Export the file
        download_params = {"file_id": file_id}
        if export_mime_type:
            download_params["mime_type"] = export_mime_type

        download_result = await asyncio.to_thread(
            lambda: composio.tools.execute(
                "GOOGLEDRIVE_DOWNLOAD_FILE",
                arguments=download_params,
                connected_account_id=connection_id,
                dangerously_skip_version_check=True
            )
        )

        if not download_result.get("successful"):
            raise ToolError(f"Failed to download file: {download_result.get('error', 'Unknown error')}")

        # Step 4: Determine file path and read content
        temp_file_path_from_sdk = download_result.get("data", {}).get("file")

        if temp_file_path_from_sdk and os.path.exists(temp_file_path_from_sdk):
            temp_file_path = Path(temp_file_path_from_sdk)
        else:
            default_composio_output_dir = Path.home() / ".composio" / "outputs" / "googledrive" / "GOOGLEDRIVE_DOWNLOAD_FILE"
            temp_file_path = default_composio_output_dir / file_name
            logger.warning(f"SDK did not provide a valid file path. Assuming default location: {temp_file_path}")

        if not temp_file_path.exists():
            raise ToolError(f"File download did not return a valid file path, and the default path was not found. Path checked: {temp_file_path}")

        content = ""
        if export_mime_type and "text" in export_mime_type:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content_bytes = await asyncio.to_thread(textract.process, str(temp_file_path))
            content = content_bytes.decode('utf-8', errors='ignore')

        return {"status": "success", "result": {"content": content}}
    except Exception as e:
        logger.error(f"Tool gdrive_read_file failed: {e}", exc_info=True)
        return {"status": "failure", "error": f"Failed to read file: {e}"}
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9003))
    
    print(f"Starting GDrive MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)