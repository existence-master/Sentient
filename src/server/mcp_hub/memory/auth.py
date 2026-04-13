from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

def get_user_id_from_context(ctx: Context) -> str:
    """
    Extracts the User ID from the 'X-User-ID' header in the HTTP request.
    """
    logger.debug("Attempting to extract User ID from context.")
    http_request = _get_http_request()
    if not http_request:
        logger.error("HTTP request context is not available.")
        raise ToolError("HTTP request context is not available.")

    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Authentication failed: 'X-User-ID' header is missing.")
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")

    logger.info(f"Successfully authenticated request for User ID: {user_id}")
    return user_id