import httpx
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException
from main.config import INTEGRATIONS_CONFIG, WAHA_URL, WAHA_API_KEY
import base64

logger = logging.getLogger(__name__)

async def call_mcp_tool(user_id: str, service_name: str, tool_name: str, payload: dict = None):
    """
    Makes an internal HTTP POST request to an MCP server to execute a tool.
    """
    service_config = INTEGRATIONS_CONFIG.get(service_name)
    if not service_config or "mcp_server_config" not in service_config:
        raise ValueError(f"MCP configuration for '{service_name}' not found.")
    
    mcp_url = service_config["mcp_server_config"]["url"]
    # The URL from config already ends in /sse, we need to call the /run endpoint
    base_url = mcp_url.replace("/sse", "")
    run_url = f"{base_url.rstrip('/')}/run"

    headers = {
        "X-User-ID": user_id,
        "Content-Type": "application/json"
    }
    
    request_body = {
        "tool": tool_name,
        "params": payload or {}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Calling MCP tool: {run_url} with tool: {tool_name}")
            # MCPs expect a POST request to the /run endpoint
            response = await client.post(run_url, json=request_body, headers=headers)
            response.raise_for_status()
            # The MCP returns a JSON response with status and result/error
            return response.json()
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"Error calling MCP tool {service_name}/{tool_name}: {e.response.status_code} - {error_text}")
            raise Exception(f"MCP Error: {error_text}")
        except Exception as e:
            logger.error(f"Failed to call MCP tool {service_name}/{tool_name}: {e}", exc_info=True)
            raise
async def waha_request_from_main(
    method: str,
    endpoint: str,
    session: str,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Centralized function to make authenticated requests to the WAHA API from the main server.
    """
    if not WAHA_URL or not WAHA_API_KEY:
        raise HTTPException(status_code=500, detail="WhatsApp service (WAHA) is not configured on the server.")

    headers = {"X-Api-Key": WAHA_API_KEY, "Content-Type": "application/json"}

    final_endpoint = endpoint.replace("{session}", session)
    url = f"{WAHA_URL.rstrip('/')}{final_endpoint}"

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            logger.info(f"Making WAHA request: {method} {url} | Session: {session} | Params: {params} | JSON: {json_data}")
            res = await client.request(method, url, params=params, json=json_data, headers=headers)
            res.raise_for_status()

            if res.status_code == 204:
                return {"status": "success", "message": "Operation successful with no content."}

            # --- MODIFICATION START ---
            # Conditionally handle binary image data vs. JSON data
            content_type = res.headers.get("content-type", "")
            if "image" in content_type:
                # It's an image, so we base64 encode the raw bytes
                encoded_image = base64.b64encode(res.content).decode("utf-8")
                return {"mimetype": content_type, "data": encoded_image}
            else:
                # It's JSON, so parse it as usual
                return res.json()
            # --- MODIFICATION END ---
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"WAHA API Error: {e.response.status_code} on {method} {url} - {error_text}")
            detail = error_text
            raise HTTPException(status_code=e.response.status_code, detail=f"WhatsApp API Error: {detail}")
        except httpx.RequestError as e:
            logger.error(f"Could not connect to WAHA API at {url}: {e}")
            raise HTTPException(status_code=503, detail=f"Unable to connect to the WhatsApp service.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during WAHA request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
