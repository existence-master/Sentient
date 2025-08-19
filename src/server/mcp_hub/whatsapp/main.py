import os
import re
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import configure_logging, get_logger

from . import auth, utils

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# Load environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

mcp = FastMCP(
    name="WhatsAppServer",
    instructions="Provides a comprehensive suite of tools to fully control a user's WhatsApp account. It can manage chats, contacts, groups, and send various types of messages on the user's behalf.",
)

# ==============================================================================
# E. Existing Tool Preservation (System Notifications)
# ==============================================================================

@mcp.tool()
async def send_message_to_self(ctx: Context, message: str) -> Dict[str, Any]:
    """
    Sends a WhatsApp message FROM the Sentient system TO the user's configured notification number.
    This tool is for system alerts and notifications, not for sending messages to other people from the user's account.
    """
    logger.info(f"Executing tool: send_message_to_self")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        chat_id = await auth.get_user_notification_chat_id(user_id)
        
        # WAHA notifications are sent via a generic "default" session for the system
        result = await utils.waha_request("POST", "/api/sendText", session="default", json_data={"chatId": chat_id, "text": message})
        
        if result and result.get("id"):
            return {"status": "success", "result": f"Notification sent successfully. Message ID: {result['id']}"}
        else:
            raise ToolError("Failed to send notification via WAHA service or received an unexpected response.")
    except Exception as e:
        logger.error(f"Tool send_message_to_self failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# ==============================================================================
# A. Messaging & Chat Management Tools
# ==============================================================================

@mcp.tool()
async def send_text_message(ctx: Context, chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a text message from the user's WhatsApp account to a specified contact or group.
    The `chat_id` must be in the format 'phonenumber@c.us' for contacts or 'groupid@g.us' for groups.
    Optionally, provide a `reply_to_message_id` to reply to a specific message.
    """
    user_id = auth.get_user_id_from_context(ctx)
    payload = {"chatId": chat_id, "text": text}
    if reply_to_message_id:
        payload["reply_to"] = reply_to_message_id
    return await utils.waha_request("POST", "/api/sendText", session=user_id, json_data=payload)

@mcp.tool()
async def send_media(ctx: Context, chat_id: str, media_type: str, url: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a media file (image, video, or document) from a public URL from the user's WhatsApp account.
    `media_type` must be one of 'image', 'video', or 'file'.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    media_type_lower = media_type.lower()
    endpoint_map = {
        'image': '/api/sendImage',
        'video': '/api/sendVideo',
        'file': '/api/sendFile'
    }
    if media_type_lower not in endpoint_map:
        raise ToolError("Invalid media_type. Must be 'image', 'video', or 'file'.")
    
    payload = {"chatId": chat_id, "file": {"url": url}, "caption": caption}
    return await utils.waha_request("POST", endpoint_map[media_type_lower], session=user_id, json_data=payload)

@mcp.tool()
async def get_chats(ctx: Context, limit: int = 100) -> Dict[str, Any]:
    """
    Retrieves a list of the user's most recent chats, including direct messages and groups.
    Returns a list of chat objects, each containing an 'id' and 'name'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    return await utils.waha_request("GET", "/api/{session}/chats", session=user_id, params={"limit": limit})

@mcp.tool()
async def get_chat_history(ctx: Context, chat_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Fetches the most recent messages from a specific chat.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    endpoint = f"/api/{{session}}/chats/{chat_id}/messages"
    return await utils.waha_request("GET", endpoint, session=user_id, params={"limit": limit})

@mcp.tool()
async def manage_chat(ctx: Context, chat_id: str, action: str) -> Dict[str, Any]:
    """
    Performs an action on a chat. `action` must be one of 'archive', 'unarchive', 'delete', or 'mark_unread'.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    action_lower = action.lower()
    action_map = {
        'archive': ("POST", f"/api/{{session}}/chats/{chat_id}/archive"),
        'unarchive': ("POST", f"/api/{{session}}/chats/{chat_id}/unarchive"),
        'delete': ("DELETE", f"/api/{{session}}/chats/{chat_id}"),
        'mark_unread': ("POST", f"/api/{{session}}/chats/{chat_id}/unread"),
    }
    if action_lower not in action_map:
        raise ToolError("Invalid action. Must be 'archive', 'unarchive', 'delete', or 'mark_unread'.")
    
    method, endpoint = action_map[action_lower]
    return await utils.waha_request(method, endpoint, session=user_id)

@mcp.tool()
async def manage_message(ctx: Context, message_id: str, action: str, content: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs an action on a specific message.
    `action` can be 'react', 'edit', 'delete', 'pin', 'unpin'.
    The `content` parameter is used for 'react' (emoji) and 'edit' (new text).
    The `message_id` must be the full message ID string (e.g., 'true_12345@c.us_ABC...').
    """
    user_id = auth.get_user_id_from_context(ctx)
    action_lower = action.lower()
    
    chat_id_match = re.search(r'_(.+?@.+?\.us)_', message_id)
    if not chat_id_match:
        raise ToolError("Invalid message_id format. Could not extract chatId.")
    chat_id = chat_id_match.group(1)

    if action_lower == 'react':
        if not content: raise ToolError("Content (emoji) is required for 'react' action.")
        return await utils.waha_request("PUT", "/api/reaction", session=user_id, json_data={"messageId": message_id, "reaction": content})
    elif action_lower == 'edit':
        if not content: raise ToolError("Content (new text) is required for 'edit' action.")
        endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}"
        return await utils.waha_request("PUT", endpoint, session=user_id, json_data={"text": content})
    elif action_lower == 'delete':
        endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}"
        return await utils.waha_request("DELETE", endpoint, session=user_id)
    elif action_lower == 'pin':
        endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}/pin"
        return await utils.waha_request("POST", endpoint, session=user_id, json_data={"duration": 86400}) # Default 24h pin
    elif action_lower == 'unpin':
        endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}/unpin"
        return await utils.waha_request("POST", endpoint, session=user_id)
    else:
        raise ToolError("Invalid action. Must be 'react', 'edit', 'delete', 'pin', or 'unpin'.")

# ==============================================================================
# B. Contact Management Tools
# ==============================================================================

@mcp.tool()
async def get_contacts(ctx: Context) -> Dict[str, Any]:
    """
    Retrieves a list of all contacts from the user's WhatsApp account.
    """
    user_id = auth.get_user_id_from_context(ctx)
    return await utils.waha_request("GET", "/api/{session}/contacts/all", session=user_id)

@mcp.tool()
async def get_contact_info(ctx: Context, contact_id: str) -> Dict[str, Any]:
    """
    Gets detailed information about a specific contact, including their name, number, and profile picture URL.
    `contact_id` can be a phone number or a chatId (e.g., '14155552671' or '14155552671@c.us').
    """
    user_id = auth.get_user_id_from_context(ctx)
    info = await utils.waha_request("GET", "/api/contacts", session=user_id, params={"contactId": contact_id})
    pic = await utils.waha_request("GET", "/api/contacts/profile-picture", session=user_id, params={"contactId": contact_id})
    info['profilePictureURL'] = pic.get('profilePictureURL')
    return info

@mcp.tool()
async def check_whatsapp_number(ctx: Context, phone_number: str) -> Dict[str, Any]:
    """
    Checks if a phone number is registered on WhatsApp and returns its chatId.
    `phone_number` should include the country code but no '+' or spaces.
    """
    user_id = auth.get_user_id_from_context(ctx)
    return await utils.waha_request("GET", "/api/contacts/check-exists", session=user_id, params={"phone": phone_number})

@mcp.tool()
async def manage_contact(ctx: Context, contact_id: str, action: str) -> Dict[str, Any]:
    """
    Blocks or unblocks a contact. `action` must be 'block' or 'unblock'.
    `contact_id` can be a phone number or a chatId.
    """
    user_id = auth.get_user_id_from_context(ctx)
    action_lower = action.lower()
    if action_lower not in ['block', 'unblock']:
        raise ToolError("Invalid action. Must be 'block' or 'unblock'.")
    endpoint = f"/api/contacts/{action_lower}"
    return await utils.waha_request("POST", endpoint, session=user_id, json_data={"contactId": contact_id})

# ==============================================================================
# C. Group Management Tools
# ==============================================================================

@mcp.tool()
async def create_group(ctx: Context, name: str, participant_ids: List[str]) -> Dict[str, Any]:
    """
    Creates a new WhatsApp group with the given name and initial participants.
    `participant_ids` must be a list of contact chatIds (e.g., ['14155552671@c.us', '12125551234@c.us']).
    """
    user_id = auth.get_user_id_from_context(ctx)
    participants_payload = [{"id": pid} for pid in participant_ids]
    return await utils.waha_request("POST", "/api/{session}/groups", session=user_id, json_data={"name": name, "participants": participants_payload})

@mcp.tool()
async def get_group_info(ctx: Context, group_id: str) -> Dict[str, Any]:
    """
    Retrieves detailed information about a group, including its subject, description, and participants.
    `group_id` must be in the format 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    endpoint = f"/api/{{session}}/groups/{group_id}"
    return await utils.waha_request("GET", endpoint, session=user_id)

@mcp.tool()
async def manage_group_participants(ctx: Context, group_id: str, action: str, participant_ids: List[str]) -> Dict[str, Any]:
    """
    Manages group participants. `action` can be 'add', 'remove', 'promote_to_admin', or 'demote_from_admin'.
    `participant_ids` is a list of contact chatIds. `group_id` must be in the format 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    action_lower = action.lower()
    action_map = {
        'add': f"/api/{{session}}/groups/{group_id}/participants/add",
        'remove': f"/api/{{session}}/groups/{group_id}/participants/remove",
        'promote_to_admin': f"/api/{{session}}/groups/{group_id}/admin/promote",
        'demote_from_admin': f"/api/{{session}}/groups/{group_id}/admin/demote",
    }
    if action_lower not in action_map:
        raise ToolError("Invalid action. Must be 'add', 'remove', 'promote_to_admin', or 'demote_from_admin'.")
    
    participants_payload = [{"id": pid} for pid in participant_ids]
    return await utils.waha_request("POST", action_map[action_lower], session=user_id, json_data={"participants": participants_payload})

@mcp.tool()
async def update_group_info(ctx: Context, group_id: str, subject: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
    """
    Updates a group's subject (title) or description. At least one must be provided.
    `group_id` must be in the format 'groupid@g.us'.
    """
    user_id = auth.get_user_id_from_context(ctx)
    results = []
    if subject:
        endpoint = f"/api/{{session}}/groups/{group_id}/subject"
        result = await utils.waha_request("PUT", endpoint, session=user_id, json_data={"subject": subject})
        results.append({"subject_update": result})
    if description is not None:
        endpoint = f"/api/{{session}}/groups/{group_id}/description"
        result = await utils.waha_request("PUT", endpoint, session=user_id, json_data={"description": description})
        results.append({"description_update": result})
    if not results:
        raise ToolError("Either 'subject' or 'description' must be provided.")
    return {"status": "success", "result": results}

# ==============================================================================
# D. Profile & Status Tools
# ==============================================================================

@mcp.tool()
async def update_profile(ctx: Context, name: Optional[str] = None, status: Optional[str] = None, picture_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Updates the user's own WhatsApp profile name, status (about text), or profile picture from a URL.
    """
    user_id = auth.get_user_id_from_context(ctx)
    results = []
    if name:
        result = await utils.waha_request("PUT", "/api/{session}/profile/name", session=user_id, json_data={"name": name})
        results.append({"name_update": result})
    if status:
        result = await utils.waha_request("PUT", "/api/{session}/profile/status", session=user_id, json_data={"status": status})
        results.append({"status_update": result})
    if picture_url:
        result = await utils.waha_request("PUT", "/api/{session}/profile/picture", session=user_id, json_data={"file": {"url": picture_url}})
        results.append({"picture_update": result})
    if not results:
        raise ToolError("At least one of 'name', 'status', or 'picture_url' must be provided.")
    return {"status": "success", "result": results}

@mcp.tool()
async def post_status(ctx: Context, content: str, type: str = 'text') -> Dict[str, Any]:
    """
    Posts a new status (story). `type` can be 'text', 'image', or 'video'.
    For 'text', `content` is the text. For 'image' or 'video', `content` is a public URL to the media.
    """
    user_id = auth.get_user_id_from_context(ctx)
    type_lower = type.lower()
    if type_lower == 'text':
        return await utils.waha_request("POST", "/api/{session}/status/text", session=user_id, json_data={"text": content})
    elif type_lower in ['image', 'video']:
        endpoint = f"/api/{{session}}/status/{type_lower}"
        return await utils.waha_request("POST", endpoint, session=user_id, json_data={"file": {"url": content}})
    else:
        raise ToolError("Invalid type. Must be 'text', 'image', or 'video'.")

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9024))
    
    print(f"Starting WhatsApp MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)