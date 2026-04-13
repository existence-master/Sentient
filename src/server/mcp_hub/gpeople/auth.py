import os
import json
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import motor.motor_asyncio
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from fastmcp import Context
from fastmcp.server.dependencies import get_http_request as _get_http_request
from fastmcp.exceptions import ToolError
from json_extractor import JsonExtractor

from typing import Optional
from dotenv import load_dotenv

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
        raise ValueError("AES encryption keys are not configured.")
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
    """Extracts the User ID from the 'X-User-ID' header in the HTTP request."""
    http_request = _get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")
    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")
    return user_id

async def get_google_creds(user_id: str) -> Credentials:
    """Fetches Google OAuth token from MongoDB for a given user_id."""
    user_doc = await users_collection.find_one({"user_id": user_id})
    if not user_doc or not user_doc.get("userData"):
        raise ToolError(f"User profile or userData not found for user_id: {user_id}.")
    
    user_data = user_doc["userData"]
    gpeople_data = user_data.get("integrations", {}).get("gpeople")
    if not gpeople_data or not gpeople_data.get("connected") or not gpeople_data.get("credentials"):
        raise ToolError(f"Google People integration not connected. Please use the default connect flow.")

    try:
        decrypted_creds_str = aes_decrypt(gpeople_data["credentials"])
        token_info = JsonExtractor.extract_valid_json(decrypted_creds_str)
        if not token_info:
            raise ToolError("Failed to parse decrypted credentials for Google People.")
        return Credentials.from_authorized_user_info(token_info)
    except Exception as e:
        raise ToolError(f"Failed to decrypt or parse default OAuth token for Google People: {e}")

def authenticate_gpeople(creds: Credentials) -> Resource:
    """
    Authenticates and returns the Google People API service using provided credentials.
    """
    return build("people", "v1", credentials=creds)