# server/mcp_hub/slack/auth.py

import os
import json
import base64
from typing import Dict, Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import motor.motor_asyncio
from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError
from dotenv import load_dotenv
from json_extractor import JsonExtractor


# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
AES_SECRET_KEY_HEX = os.getenv("AES_SECRET_KEY")
AES_IV_HEX = os.getenv("AES_IV")

AES_SECRET_KEY: Optional[bytes] = bytes.fromhex(AES_SECRET_KEY_HEX) if AES_SECRET_KEY_HEX and len(AES_SECRET_KEY_HEX) == 64 else None
AES_IV: Optional[bytes] = bytes.fromhex(AES_IV_HEX) if AES_IV_HEX and len(AES_IV_HEX) == 32 else None

def aes_decrypt(encrypted_data: str) -> str:
    if not AES_SECRET_KEY or not AES_IV:
        raise ValueError("AES encryption keys are not configured in the environment.")
    backend = default_backend()
    cipher = Cipher(algorithms.AES(AES_SECRET_KEY), modes.CBC(AES_IV), backend=backend)
    decryptor = cipher.decryptor()
    encrypted_bytes = base64.b64decode(encrypted_data)
    decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted) + unpadder.finalize()
    return unpadded_data.decode()

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

async def get_slack_creds(user_id: str) -> Dict[str, str]:
    """
    Fetches Slack credentials (token and team_id) from MongoDB.
    """
    user_doc = await users_collection.find_one({"user_id": user_id})

    if not user_doc:
        raise ToolError(f"User profile not found for user_id: {user_id}.")

    slack_data = user_doc.get("userData", {}).get("integrations", {}).get("slack", {})

    if not slack_data.get("connected") or "credentials" not in slack_data:
        raise ToolError(f"Slack integration not connected or credentials missing for user_id: {user_id}. Please connect it in settings.")

    try:
        decrypted_creds_str = aes_decrypt(slack_data["credentials"])
        token_info = JsonExtractor.extract_valid_json(decrypted_creds_str)
    except Exception as e:
        raise ToolError(f"Failed to decrypt or parse Slack credentials: {e}")

    if not token_info:
        raise ToolError("Failed to parse decrypted credentials for Slack.")

    authed_user = token_info.get("authed_user", {})
    access_token = authed_user.get("access_token")
    team_id = token_info.get("team", {}).get("id")

    if not access_token or not team_id:
        raise ToolError("Invalid Slack OAuth token data in database. Re-authentication may be required.")

    return {"token": access_token, "team_id": team_id}
