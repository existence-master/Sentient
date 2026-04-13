import os
from dotenv import load_dotenv
from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError


# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
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

def get_news_api_key() -> str:
    """
    Retrieves the NewsAPI key from environment variables.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == "your_news_api_key":
        raise ToolError(
            "NewsAPI key is not configured. "
            "Please set NEWS_API_KEY in your server's .env file."
        )
    return api_key