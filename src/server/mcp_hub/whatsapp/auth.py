import os
import motor.motor_asyncio
from dotenv import load_dotenv
from fastmcp import Context
from fastmcp.exceptions import ToolError

# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]
users_collection = db["user_profiles"]

def get_user_id_from_context(ctx: Context) -> str:
    """
    Extracts the Sentient User ID from the 'X-User-ID' header.
    This user_id is used as the session name for the user's own WhatsApp account.
    """
    http_request = ctx.get_http_request()
    if not http_request:
        raise ToolError("HTTP request context is not available.")
    user_id = http_request.headers.get("X-User-ID")
    if not user_id:
        raise ToolError("Authentication failed: 'X-User-ID' header is missing.")
    return user_id

async def get_user_notification_chat_id(user_id: str) -> str:
    """
    Fetches the user's pre-configured WhatsApp chat ID for receiving system notifications.
    This is used by the 'send_message_to_self' tool.
    """
    user_doc = await users_collection.find_one({"user_id": user_id})

    if not user_doc or not user_doc.get("userData"):
        raise ToolError(f"User profile not found for user_id: {user_id}.")

    # This path is for system->user notifications, not the full-control integration
    wa_prefs = user_doc.get("userData", {}).get("notificationPreferences", {}).get("whatsapp", {})
    
    # NOTE: We don't check the `enabled` flag here. The tool should work if a number is configured,
    # regardless of the notification toggle setting.

    chat_id = wa_prefs.get("chatId")
    if not chat_id:
        raise ToolError("WhatsApp notification number is not configured for this user. Please set it in Settings.")

    return chat_id