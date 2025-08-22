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
    This tool is to be used only when the user says 'Send message to me', not for sending messages to other people from the user's account.
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
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        payload = {"chatId": chat_id, "text": text, "session": sanitized_session}
        if reply_to_message_id:
            payload["reply_to"] = reply_to_message_id
        # The session parameter here is for logging and path replacement (if any) in the util function.
        # The actual session for GOWS engine is passed in the JSON payload.
        result = await utils.waha_request("POST", "/api/sendText", session=sanitized_session, json_data=payload)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool send_text_message failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def send_media(ctx: Context, chat_id: str, media_type: str, url: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a media file (image, video, or document) from a public URL from the user's WhatsApp account.
    `media_type` must be one of 'image', 'video', or 'file'.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        media_type_lower = media_type.lower()
        endpoint_map = {
            'image': '/api/sendImage',
            'video': '/api/sendVideo',
            'file': '/api/sendFile'
        }
        if media_type_lower not in endpoint_map:
            raise ToolError("Invalid media_type. Must be 'image', 'video', or 'file'.")
        
        payload = {"chatId": chat_id, "file": {"url": url}, "caption": caption, "session": sanitized_session}
        result = await utils.waha_request("POST", endpoint_map[media_type_lower], session=sanitized_session, json_data=payload)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool send_media failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_chats(ctx: Context, limit: int = 100) -> Dict[str, Any]:
    """
    Retrieves a list of the user's most recent chats, including direct messages and groups.
    Returns a list of chat objects, each containing an 'id' and 'name'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        result = await utils.waha_request("GET", "/api/{session}/chats", session=sanitized_session, params={"limit": limit})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool get_chats failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_chat_history(ctx: Context, chat_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Fetches the most recent messages from a specific chat.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        endpoint = f"/api/{{session}}/chats/{chat_id}/messages"
        result = await utils.waha_request("GET", endpoint, session=sanitized_session, params={"limit": limit})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool get_chat_history failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def manage_chat(ctx: Context, chat_id: str, action: str) -> Dict[str, Any]:
    """
    Performs an action on a chat. `action` must be one of 'archive', 'unarchive', 'delete', or 'mark_unread'.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
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
        result = await utils.waha_request(method, endpoint, session=sanitized_session)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool manage_chat failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_unread_messages(ctx: Context, limit: int = 20) -> Dict[str, Any]:
    """
    Retrieves the most recent unread messages from all chats.
    Returns a list of message objects, including their content, sender, and chat_id.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        result = await utils.waha_request("GET", "/api/{session}/messages", session=sanitized_session, params={"unread": True, "limit": limit})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool get_unread_messages failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def set_typing_presence(ctx: Context, chat_id: str, duration_seconds: int = 10) -> Dict[str, Any]:
    """
    Shows a 'typing...' indicator in a chat for a specified duration to simulate a more natural interaction before sending a message.
    The `chat_id` must be in the format 'phonenumber@c.us' or 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        endpoint = f"/api/{{session}}/chats/{chat_id}/presence"
        result = await utils.waha_request("POST", endpoint, session=sanitized_session, json_data={"presence": "composing", "duration": duration_seconds})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool set_typing_presence failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def manage_message(ctx: Context, message_id: str, action: str, content: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs an action on a specific message.
    `action` can be 'react', 'edit', 'delete', 'pin', 'unpin'.
    The `content` parameter is used for 'react' (emoji) and 'edit' (new text).
    The `message_id` must be the full message ID string (e.g., 'true_12345@c.us_ABC...').
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        action_lower = action.lower()
        
        chat_id_match = re.search(r'_(.+?@.+?\.us)_', message_id)
        if not chat_id_match:
            raise ToolError("Invalid message_id format. Could not extract chatId.")
        chat_id = chat_id_match.group(1)

        if action_lower == 'react':
            if not content: raise ToolError("Content (emoji) is required for 'react' action.")
            result = await utils.waha_request("PUT", "/api/reaction", session=sanitized_session, json_data={"messageId": message_id, "reaction": content})
        elif action_lower == 'edit':
            if not content: raise ToolError("Content (new text) is required for 'edit' action.")
            endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}"
            result = await utils.waha_request("PUT", endpoint, session=sanitized_session, json_data={"text": content})
        elif action_lower == 'delete':
            endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}"
            result = await utils.waha_request("DELETE", endpoint, session=sanitized_session)
        elif action_lower == 'pin':
            endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}/pin"
            result = await utils.waha_request("POST", endpoint, session=sanitized_session, json_data={"duration": 86400}) # Default 24h pin
        elif action_lower == 'unpin':
            endpoint = f"/api/{{session}}/chats/{chat_id}/messages/{message_id}/unpin"
            result = await utils.waha_request("POST", endpoint, session=sanitized_session)
        else:
            raise ToolError("Invalid action. Must be 'react', 'edit', 'delete', 'pin', or 'unpin'.")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool manage_message failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# ==============================================================================
# B. Contact Management Tools
# ==============================================================================

@mcp.tool()
async def send_location(ctx: Context, chat_id: str, latitude: float, longitude: float, name: Optional[str] = None, address: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a location pin to a specified contact or group.
    Requires the `chat_id`, `latitude`, and `longitude`.
    Optionally include a `name` for the location (e.g., 'Eiffel Tower') and an `address`.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        payload = {"chatId": chat_id, "latitude": latitude, "longitude": longitude, "name": name, "address": address, "session": sanitized_session}
        result = await utils.waha_request("POST", "/api/sendLocation", session=sanitized_session, json_data=payload)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool send_location failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_contacts(ctx: Context) -> Dict[str, Any]:
    """
    Retrieves a list of all contacts from the user's WhatsApp account.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        result = await utils.waha_request("GET", "/api/contacts/all", session=sanitized_session, params={"session": sanitized_session})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool get_contacts failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_contact_info(ctx: Context, contact_id: str) -> Dict[str, Any]:
    """
    Gets detailed information about a specific contact, including their name, number, and profile picture URL.
    `contact_id` can be a phone number or a chatId (e.g., '14155552671' or '14155552671@c.us').
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        # Corrected: Added session as a query parameter to both calls.
        info = await utils.waha_request("GET", "/api/contacts", session=sanitized_session, params={"contactId": contact_id, "session": sanitized_session})
        pic = await utils.waha_request("GET", "/api/contacts/profile-picture", session=sanitized_session, params={"contactId": contact_id, "session": sanitized_session})
        info['profilePictureURL'] = pic.get('profilePictureURL')
        return {"status": "success", "result": info}
    except Exception as e:
        logger.error(f"Tool get_contact_info failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def check_whatsapp_number(ctx: Context, phone_number: str) -> Dict[str, Any]:
    """
    Checks if a phone number is registered on WhatsApp and returns its chatId.
    `phone_number` should include the country code but no '+' or spaces.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        # Corrected: Added session as a query parameter.
        result = await utils.waha_request("GET", "/api/contacts/check-exists", session=sanitized_session, params={"phone": phone_number, "session": sanitized_session})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool check_whatsapp_number failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_chat_id_by_name(ctx: Context, contact_name: str) -> Dict[str, Any]:
    """
    Searches for a contact by their name and returns their chat ID.
    The search is case-insensitive. If multiple contacts match, the first one found is returned.
    """
    user_id = auth.get_user_id_from_context(ctx)
    sanitized_session = user_id.replace("|", "_")

    try:
        contacts = await utils.waha_request("GET", "/api/contacts/all", session=sanitized_session, params={"session": sanitized_session})

        for contact in contacts:
            # Check both 'name' (saved name) and 'pushName' (their WhatsApp display name)
            if (contact.get('name') and contact_name.lower() in contact['name'].lower()) or \
               (contact.get('pushName') and contact_name.lower() in contact['pushName'].lower()):
                return {"status": "success", "result": {"chat_id": contact.get("id")}}

        return {"status": "failure", "error": f"Contact with name '{contact_name}' not found."}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def manage_contact(ctx: Context, contact_id: str, action: str) -> Dict[str, Any]:
    """
    Blocks or unblocks a contact. `action` must be 'block' or 'unblock'.
    `contact_id` can be a phone number or a chatId.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        action_lower = action.lower()
        if action_lower not in ['block', 'unblock']:
            raise ToolError("Invalid action. Must be 'block' or 'unblock'.")
        endpoint = f"/api/contacts/{action_lower}"
        result = await utils.waha_request("POST", endpoint, session=sanitized_session, json_data={"contactId": contact_id, "session": sanitized_session})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool manage_contact failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# ==============================================================================
# C. Group Management Tools
# ==============================================================================

@mcp.tool()
async def create_group(ctx: Context, name: str, participant_ids: List[str]) -> Dict[str, Any]:
    """
    Creates a new WhatsApp group with the given name and initial participants.
    `participant_ids` must be a list of contact chatIds (e.g., ['14155552671@c.us', '12125551234@c.us']).
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        participants_payload = [{"id": pid} for pid in participant_ids]
        result = await utils.waha_request("POST", "/api/{session}/groups", session=sanitized_session, json_data={"name": name, "participants": participants_payload})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool create_group failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def get_group_info(ctx: Context, group_id: str) -> Dict[str, Any]:
    """
    Retrieves detailed information about a group, including its subject, description, and participants.
    `group_id` must be in the format 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        endpoint = f"/api/{{session}}/groups/{group_id}"
        result = await utils.waha_request("GET", endpoint, session=sanitized_session)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool get_group_info failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def manage_group_participants(ctx: Context, group_id: str, action: str, participant_ids: List[str]) -> Dict[str, Any]:
    """
    Manages group participants. `action` can be 'add', 'remove', 'promote_to_admin', or 'demote_from_admin'.
    `participant_ids` is a list of contact chatIds. `group_id` must be in the format 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
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
        result = await utils.waha_request("POST", action_map[action_lower], session=sanitized_session, json_data={"participants": participants_payload})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool manage_group_participants failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def update_group_info(ctx: Context, group_id: str, subject: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
    """
    Updates a group's subject (title) or description. At least one must be provided.
    `group_id` must be in the format 'groupid@g.us'.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        results = []
        if subject:
            endpoint = f"/api/{{session}}/groups/{group_id}/subject"
            result = await utils.waha_request("PUT", endpoint, session=sanitized_session, json_data={"subject": subject})
            results.append({"subject_update": result})
        if description is not None:
            endpoint = f"/api/{{session}}/groups/{group_id}/description"
            result = await utils.waha_request("PUT", endpoint, session=sanitized_session, json_data={"description": description})
            results.append({"description_update": result})
        if not results:
            raise ToolError("Either 'subject' or 'description' must be provided.")
        return {"status": "success", "result": results}
    except Exception as e:
        logger.error(f"Tool update_group_info failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# ==============================================================================
# D. Profile & Status Tools
# ==============================================================================

@mcp.tool()
async def update_profile(ctx: Context, name: Optional[str] = None, status: Optional[str] = None, picture_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Updates the user's own WhatsApp profile name, status (about text), or profile picture from a URL.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        results = []
        if name:
            result = await utils.waha_request("PUT", "/api/{session}/profile/name", session=sanitized_session, json_data={"name": name})
            results.append({"name_update": result})
        if status:
            result = await utils.waha_request("PUT", "/api/{session}/profile/status", session=sanitized_session, json_data={"status": status})
            results.append({"status_update": result})
        if picture_url:
            result = await utils.waha_request("PUT", "/api/{session}/profile/picture", session=sanitized_session, json_data={"file": {"url": picture_url}})
            results.append({"picture_update": result})
        if not results:
            raise ToolError("At least one of 'name', 'status', or 'picture_url' must be provided.")
        return {"status": "success", "result": results}
    except Exception as e:
        logger.error(f"Tool update_profile failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def post_status(ctx: Context, content: str, type: str = 'text') -> Dict[str, Any]:
    """
    Posts a new status (story). `type` can be 'text', 'image', or 'video'.
    For 'text', `content` is the text. For 'image' or 'video', `content` is a public URL to the media.
    """
    try:
        user_id = auth.get_user_id_from_context(ctx)
        sanitized_session = user_id.replace("|", "_")
        type_lower = type.lower()
        if type_lower == 'text':
            result = await utils.waha_request("POST", "/api/{session}/status/text", session=sanitized_session, json_data={"text": content})
        elif type_lower in ['image', 'video']:
            endpoint = f"/api/{{session}}/status/{type_lower}"
            result = await utils.waha_request("POST", endpoint, session=sanitized_session, json_data={"file": {"url": content}})
        else:
            raise ToolError("Invalid type. Must be 'text', 'image', or 'video'.")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool post_status failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9024))
    
    print(f"Starting WhatsApp MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)