from fastmcp import Context
from fastmcp.exceptions import ToolError

def get_user_id_from_context(ctx: Context) -> str:
    """
    Extracts the User ID from the 'X-User-ID' header in the HTTP request.
    """
    http_request = ctx.get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")

    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")

    return user_id

def get_task_id_from_context(ctx: Context) -> str:
    """
    Extracts the Task ID from the 'X-Task-ID' header in the HTTP request.
    """
    http_request = ctx.get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")

    task_id = http_request.headers.get("X-Task-ID")
    if not task_id:
        raise ToolError("Context is missing: 'X-Task-ID' header is missing.")

    return task_id
