import os
from typing import Dict
from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError

from dotenv import load_dotenv
import motor.motor_asyncio

# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
# Configs
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# DB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]
users_collection = db["user_profiles"]

def get_user_id_from_context(ctx: Context) -> str:
    http_request = _get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")
    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")
    return user_id

async def get_google_api_key(user_id: str) -> str:
    """
    Validates user existence and returns the global Google API key for Maps.
    """
    user_doc = await users_collection.find_one({"user_id": user_id})
    if not user_doc:
        raise ToolError(f"User profile not found for user_id: {user_id}.")

    if not DEFAULT_GOOGLE_API_KEY:
        raise ToolError("Google API Key for Maps is not configured on the server.")

    return DEFAULT_GOOGLE_API_KEY