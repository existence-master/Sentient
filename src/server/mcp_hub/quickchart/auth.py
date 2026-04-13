# server/mcp_hub/quickchart/auth.py

from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError

def get_user_id_from_context(ctx: Context) -> str:
    """
    Extracts the User ID from the 'X-User-ID' header in the HTTP request.
    Maintained for structural consistency.
    """
    http_request = _get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")

    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")

    return user_id