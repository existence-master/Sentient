import os
from typing import Dict, Any
from fastmcp.exceptions import ToolError
import asyncio
import textract
import platform

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.prompts.prompt import Message
from pathlib import Path
from fastmcp.utilities.logging import configure_logging, get_logger

from . import auth

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# --- Environment Setup ---
# Load environment variables from a .env file.
# By default, load_dotenv() looks for .env in the current directory and its parents.
load_dotenv()

# Define the base temporary directory for file management
FILE_MANAGEMENT_TEMP_DIR = os.getenv('FILE_MANAGEMENT_TEMP_DIR', '/tmp/mcp_file_management_temp')

# Ensure the base temporary directory exists at startup
os.makedirs(FILE_MANAGEMENT_TEMP_DIR, exist_ok=True)

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

# --- MCP Server Initialization ---
mcp = FastMCP(
    name="FileManagementServer",
    instructions="Provides tools to read and write files in a temporary storage area, useful for handling uploads and generating files for download.",
)

def _get_safe_filepath(filename: str) -> Path:
    """
    Constructs a full, safe path to a file within the temp directory and validates it.
    Prevents path traversal attacks.
    """
    # Base directory where all user files are stored
    base_dir = Path(FILE_MANAGEMENT_TEMP_DIR).resolve()
    
    # Create the full path by joining the base directory and the relative filename
    # The filename from the agent already includes the user's subdirectory, e.g., "user-id/report.pdf"
    full_path = (base_dir / filename).resolve()
    
    # SECURITY CHECK: Ensure the resolved path is still inside the base directory
    if base_dir not in full_path.parents:
        raise ToolError("Access denied: Path traversal attempt detected.")

    return full_path

# --- Tool Definitions ---
@mcp.tool()
async def read_file(ctx: Context, filename: str) -> Dict[str, Any]:
    """
    Reads the content of a specified file from the temporary storage and extracts its text content. The `filename` should be one of the files returned by `list_files`.
    """
    logger.info(f"Executing tool: read_file with filename='{filename}'")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        # Prepend user_id to filename to create the relative path
        relative_path = os.path.join(safe_user_id, filename)
        safe_path = _get_safe_filepath(relative_path)
        
        if not safe_path.exists():
            return {"status": "failure", "error": "File not found."}
        
        def process_file():
            try:
                # Convert the Path object to a string. This is all that's needed.
                path_for_textract = str(safe_path)
                
                # On Windows, you might need to handle paths differently for some tools,
                # but doubling the slashes is incorrect. Let's try passing it directly.
                # If issues persist with specific file types, we might need a more nuanced fix,
                # but this is the correct first step.

                extracted_bytes = textract.process(path_for_textract)
                return extracted_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                return f"Error extracting text from file: {str(e)}"

        content = await asyncio.to_thread(process_file)

        return {"status": "success", "result": content}
    except Exception as e:
        logger.error(f"Tool read_file failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def write_file(ctx: Context, filename: str, content: str) -> Dict[str, Any]:
    """
    Writes or overwrites a file in the temporary storage with the provided `content`.
    """
    logger.info(f"Executing tool: write_file with filename='{filename}'")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        # Prepend user_id to filename to create the relative path
        relative_path = os.path.join(safe_user_id, filename)
        safe_path = _get_safe_filepath(relative_path)

        # Ensure the directory exists before writing
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_path.open("w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "result": f"File '{filename}' written successfully."}
    except Exception as e:
        logger.error(f"Tool write_file failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def list_files(ctx: Context) -> Dict[str, Any]:
    """
    Lists all files currently available in the user's temporary storage area.
    """
    logger.info("Executing tool: list_files")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        # Sanitize user_id to ensure it's safe for use in a file path
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        user_dir = os.path.join(FILE_MANAGEMENT_TEMP_DIR, safe_user_id)

        if not os.path.isdir(user_dir):
            return {"status": "success", "result": []} # User has no files yet

        files = [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))]
        
        # Return just the filenames, not the relative path
        return {"status": "success", "result": files}
    except Exception as e:
        logger.error(f"Tool list_files failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9026))
    
    print(f"Starting File Management MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)