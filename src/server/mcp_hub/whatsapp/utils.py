import httpx
import logging
from typing import Optional, Dict, Any
from fastmcp.exceptions import ToolError
import os
from dotenv import load_dotenv
from fastmcp.utilities.logging import configure_logging, get_logger

# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

WAHA_URL = os.getenv("WAHA_URL")
WAHA_API_KEY = os.getenv("WAHA_API_KEY")

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

async def waha_request(
    method: str,
    endpoint: str,
    session: str,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Centralized function to make authenticated requests to the WAHA API.
    Handles base URL, headers, session management, and error responses.
    """
    if not WAHA_URL or not WAHA_API_KEY:
        raise ToolError("WhatsApp service (WAHA) is not configured on the server.")

    headers = {"X-Api-Key": WAHA_API_KEY, "Content-Type": "application/json"}

    # Sanitize the session name for use in paths and query parameters
    sanitized_session = session.replace("|", "_")

    final_endpoint = endpoint.replace("{session}", sanitized_session)
    url = f"{WAHA_URL.rstrip('/')}{final_endpoint}"

    # Also sanitize the 'session' query parameter if it exists
    if params and "session" in params:
        params["session"] = sanitized_session

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Making WAHA request: {method} {url} | Session: {sanitized_session} | Params: {params} | JSON: {json_data}")
            res = await client.request(method, url, params=params, json=json_data, headers=headers)
            res.raise_for_status()
            
            if res.status_code == 204:
                return {"status": "success", "message": "Operation successful with no content."}
            
            return res.json()
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"WAHA API Error: {e.response.status_code} on {method} {url} - {error_text}")
            try:
                error_json = e.response.json()
                detail = error_json.get("message", error_text)
            except Exception:
                detail = error_text
            raise ToolError(f"WhatsApp API Error ({e.response.status_code}): {detail}")
        except httpx.RequestError as e:
            logger.error(f"Could not connect to WAHA API at {url}: {e}")
            raise ToolError(f"Unable to connect to the WhatsApp service: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during WAHA request: {e}", exc_info=True)
            raise ToolError(f"An unexpected error occurred: {e}")