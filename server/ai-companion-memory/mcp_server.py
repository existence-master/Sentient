from fastmcp import FastMCP, Context
from tools.memory_manager import process_memory_request
from core.config import MCP_SERVER_HOST, MCP_SERVER_PORT

# Initialize the FastMCP server
mcp = FastMCP(
    name="AICompanionMemorySystem",
    instructions="A server for managing long-term, short-term, and contact memory for an AI companion."
)

@mcp.tool()
async def manage_memory(task: str, ctx: Context | None = None) -> str:
    """
    A unified tool to save, retrieve, update, or delete memories and contacts.
    Use natural language for the 'task' parameter.
    For saving, be explicit. E.g., 'Remember my daughter's birthday is on March 12th.'
    For retrieving, ask a question. E.g., 'What is my daughter's birthday?' or 'Who is Alex?'
    For updating, state the change. E.g., 'Update Alex's phone number to 555-0101.'
    The system will automatically classify memory type (long-term, short-term, contact) and perform the correct database action.
    The user's identity is automatically handled by the system based on the 'X-User-ID' header.
    """
    return await process_memory_request(task, ctx)

if __name__ == "__main__":
    print(f"Starting MCP Server on {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    # Ensure FAISS index directory exists
    from core.config import FAISS_INDEX_PATH
    import os
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)
        
    mcp.run(transport="sse", host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)