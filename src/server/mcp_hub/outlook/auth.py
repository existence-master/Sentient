import os
import json
import logging
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]

# Encryption key for credentials
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "your-32-byte-encryption-key-here").encode()

def aes_decrypt(encrypted_data: str) -> str:
    """Decrypt AES encrypted data."""
    try:
        # Decode from base64
        encrypted_bytes = bytes.fromhex(encrypted_data)
        
        # Extract IV and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(ENCRYPTION_KEY), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data.decode('utf-8')
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        raise

def get_user_id_from_context(ctx) -> str:
    """Extract user_id from MCP context."""
    try:
        # Extract user_id from context metadata
        user_id = ctx.metadata.get("user_id")
        if not user_id:
            raise ValueError("user_id not found in context metadata")
        return user_id
    except Exception as e:
        logger.error(f"Error extracting user_id from context: {e}")
        raise

async def get_outlook_credentials(user_id: str) -> Dict[str, Any]:
    """Get Outlook credentials for a user from MongoDB."""
    try:
        user_profile = await db.user_profiles.find_one({"user_id": user_id})
        if not user_profile:
            raise ValueError(f"User profile not found for user_id: {user_id}")
        
        integrations = user_profile.get("userData", {}).get("integrations", {})
        outlook_integration = integrations.get("outlook", {})
        
        if not outlook_integration.get("connected", False):
            raise ValueError("Outlook not connected for this user")
        
        encrypted_creds = outlook_integration.get("credentials")
        if not encrypted_creds:
            raise ValueError("No credentials found for Outlook integration")
        
        # Decrypt credentials
        decrypted_creds = aes_decrypt(encrypted_creds)
        return json.loads(decrypted_creds)
        
    except Exception as e:
        logger.error(f"Error getting Outlook credentials for user {user_id}: {e}")
        raise

async def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get user information including privacy filters."""
    try:
        user_profile = await db.user_profiles.find_one({"user_id": user_id})
        if not user_profile:
            return {}
        
        return user_profile.get("userData", {})
    except Exception as e:
        logger.error(f"Error getting user info for {user_id}: {e}")
        return {}
