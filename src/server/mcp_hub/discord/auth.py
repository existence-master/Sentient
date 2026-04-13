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

# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
AES_SECRET_KEY_HEX = os.getenv("AES_SECRET_KEY")
AES_IV_HEX = os.getenv("AES_IV")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

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
    http_request = _get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")
    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")
    return user_id

async def get_discord_creds(user_id: str) -> Dict[str, str]:
    user_doc = await users_collection.find_one({"user_id": user_id})
    if not user_doc:
        raise ToolError(f"User profile not found for user_id: {user_id}.")

    discord_data = user_doc.get("userData", {}).get("integrations", {}).get("discord", {})
    if not discord_data.get("connected") or "credentials" not in discord_data:
        raise ToolError(f"Discord integration not connected or credentials missing for user_id: {user_id}. Please connect it in settings.")

    try:
        decrypted_creds_str = aes_decrypt(discord_data["credentials"])
        token_info = json.loads(decrypted_creds_str)
    except Exception as e:
        raise ToolError(f"Failed to decrypt or parse Discord credentials: {e}")

    access_token = token_info.get("access_token")
    if not access_token:
        raise ToolError("Invalid Discord OAuth token data in database. Re-authentication may be required.")

    # The bot token is static and comes from the environment, not the user's session
    bot_token = DISCORD_BOT_TOKEN
    if not bot_token:
        raise ToolError("Discord Bot Token is not configured on the server. Please set DISCORD_BOT_TOKEN.")

    return {"access_token": access_token, "bot_token": bot_token}