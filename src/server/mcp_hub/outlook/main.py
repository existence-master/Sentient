import os
import asyncio
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.prompts.prompt import Message
from fastmcp.utilities.logging import configure_logging, get_logger
from fastmcp.exceptions import ToolError

# Local imports
from . import auth
from . import prompts
from . import utils as helpers

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# Conditionally load .env for local development
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

# --- Server Initialization ---
mcp = FastMCP(
    name="OutlookServer",
    instructions="Provides a comprehensive suite of tools to read, search, send, and manage emails in Outlook using Microsoft Graph API.",
)

# --- Prompt Registration ---
@mcp.resource("prompt://outlook-agent-system")
def get_outlook_system_prompt() -> str:
    """Provides the system prompt for the Outlook agent."""
    return prompts.outlook_agent_system_prompt

@mcp.prompt(name="outlook_user_prompt_builder")
def build_outlook_user_prompt(query: str, username: str, previous_tool_response: str = "{}") -> Message:
    """Builds a formatted user prompt for the Outlook agent."""
    content = prompts.outlook_agent_user_prompt.format(
        query=query,
        username=username,
        previous_tool_response=previous_tool_response
    )
    return Message(role="user", content=content)

# --- Tool Helper ---
async def _execute_outlook_action(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all Outlook tools."""
    try:
        user_id = auth.get_user_id_from_context(ctx)
        credentials = await auth.get_outlook_credentials(user_id)
        
        if not credentials or "access_token" not in credentials:
            raise ToolError("Outlook not connected or access token missing")
        
        # Create Outlook API client
        outlook_api = helpers.OutlookAPI(credentials["access_token"])
        
        # Execute the requested action
        if action_name == "get_emails":
            return await outlook_api.get_emails(**kwargs)
        elif action_name == "get_email":
            return await outlook_api.get_email(**kwargs)
        elif action_name == "send_email":
            return await outlook_api.send_email(**kwargs)
        elif action_name == "get_folders":
            return await outlook_api.get_folders(**kwargs)
        elif action_name == "search_emails":
            return await outlook_api.search_emails(**kwargs)
        else:
            raise ToolError(f"Unknown action: {action_name}")
            
    except Exception as e:
        logger.error(f"Error executing Outlook action {action_name}: {e}")
        raise ToolError(f"Outlook action failed: {str(e)}")

# --- Tool Definitions ---
@mcp.tool()
async def get_emails(ctx: Context, folder: str = "inbox", top: int = 10, skip: int = 0, 
                    search: Optional[str] = None) -> Dict[str, Any]:
    """Get emails from a specific folder in Outlook."""
    try:
        result = await _execute_outlook_action(ctx, "get_emails", folder=folder, top=top, skip=skip, search=search)
        
        # Get user info for privacy filters
        user_id = auth.get_user_id_from_context(ctx)
        user_info = await auth.get_user_info(user_id)
        privacy_filters = user_info.get("privacy_filters", {})
        
        # Apply privacy filters and format emails
        emails = result.get("value", [])
        filtered_emails = helpers.apply_privacy_filters(emails, privacy_filters)
        formatted_emails = [helpers.format_email_summary(email) for email in filtered_emails]
        
        return {
            "emails": formatted_emails,
            "total_count": len(formatted_emails),
            "folder": folder
        }
        
    except Exception as e:
        logger.error(f"Error getting emails: {e}")
        raise ToolError(f"Failed to get emails: {str(e)}")

@mcp.tool()
async def get_email(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Get a specific email by ID."""
    try:
        result = await _execute_outlook_action(ctx, "get_email", message_id=message_id)
        
        # Format the email for better readability
        email = result
        from_info = email.get("from", {})
        to_info = email.get("toRecipients", [])
        cc_info = email.get("ccRecipients", [])
        bcc_info = email.get("bccRecipients", [])
        
        formatted_email = {
            "id": email.get("id"),
            "subject": email.get("subject", "No Subject"),
            "from": from_info.get("emailAddress", {}).get("address", "Unknown"),
            "from_name": from_info.get("emailAddress", {}).get("name", "Unknown"),
            "to": [recipient.get("emailAddress", {}).get("address", "") for recipient in to_info],
            "cc": [recipient.get("emailAddress", {}).get("address", "") for recipient in cc_info],
            "bcc": [recipient.get("emailAddress", {}).get("address", "") for recipient in bcc_info],
            "received_date": email.get("receivedDateTime"),
            "sent_date": email.get("sentDateTime"),
            "is_read": email.get("isRead", False),
            "has_attachments": email.get("hasAttachments", False),
            "body": email.get("body", {}).get("content", ""),
            "body_preview": email.get("bodyPreview", "")
        }
        
        return formatted_email
        
    except Exception as e:
        logger.error(f"Error getting email {message_id}: {e}")
        raise ToolError(f"Failed to get email: {str(e)}")

@mcp.tool()
async def send_email(ctx: Context, subject: str, body: str, to_recipients: List[str], 
                    cc_recipients: Optional[List[str]] = None,
                    bcc_recipients: Optional[List[str]] = None,
                    reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
    """Send an email through Outlook."""
    try:
        result = await _execute_outlook_action(
            ctx, "send_email",
            subject=subject,
            body=body,
            to_recipients=to_recipients,
            cc_recipients=cc_recipients,
            bcc_recipients=bcc_recipients,
            reply_to_message_id=reply_to_message_id
        )
        
        return {
            "success": True,
            "message": "Email sent successfully",
            "to": to_recipients,
            "subject": subject
        }
        
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise ToolError(f"Failed to send email: {str(e)}")

@mcp.tool()
async def get_folders(ctx: Context) -> Dict[str, Any]:
    """Get email folders in Outlook."""
    try:
        result = await _execute_outlook_action(ctx, "get_folders")
        
        folders = result.get("value", [])
        formatted_folders = []
        
        for folder in folders:
            formatted_folders.append({
                "id": folder.get("id"),
                "name": folder.get("displayName"),
                "message_count": folder.get("messageCount", 0),
                "unread_count": folder.get("unreadItemCount", 0)
            })
        
        return {
            "folders": formatted_folders,
            "total_folders": len(formatted_folders)
        }
        
    except Exception as e:
        logger.error(f"Error getting folders: {e}")
        raise ToolError(f"Failed to get folders: {str(e)}")

@mcp.tool()
async def search_emails(ctx: Context, query: str, top: int = 10) -> Dict[str, Any]:
    """Search emails in Outlook."""
    try:
        result = await _execute_outlook_action(ctx, "search_emails", query=query, top=top)
        
        # Get user info for privacy filters
        user_id = auth.get_user_id_from_context(ctx)
        user_info = await auth.get_user_info(user_id)
        privacy_filters = user_info.get("privacy_filters", {})
        
        # Apply privacy filters and format emails
        emails = result.get("value", [])
        filtered_emails = helpers.apply_privacy_filters(emails, privacy_filters)
        formatted_emails = [helpers.format_email_summary(email) for email in filtered_emails]
        
        return {
            "emails": formatted_emails,
            "total_count": len(formatted_emails),
            "search_query": query
        }
        
    except Exception as e:
        logger.error(f"Error searching emails: {e}")
        raise ToolError(f"Failed to search emails: {str(e)}")

@mcp.tool()
async def reply_to_email(ctx: Context, message_id: str, body: str, 
                        cc_recipients: Optional[List[str]] = None,
                        bcc_recipients: Optional[List[str]] = None) -> Dict[str, Any]:
    """Reply to an existing email."""
    try:
        # First get the original email to extract recipients
        original_email = await _execute_outlook_action(ctx, "get_email", message_id=message_id)
        
        # Extract the original sender as recipient for reply
        from_info = original_email.get("from", {})
        reply_to_email = from_info.get("emailAddress", {}).get("address")
        
        if not reply_to_email:
            raise ToolError("Could not determine recipient for reply")
        
        # Create reply subject
        original_subject = original_email.get("subject", "")
        reply_subject = f"Re: {original_subject}" if not original_subject.startswith("Re:") else original_subject
        
        # Send the reply
        result = await _execute_outlook_action(
            ctx, "send_email",
            subject=reply_subject,
            body=body,
            to_recipients=[reply_to_email],
            cc_recipients=cc_recipients,
            bcc_recipients=bcc_recipients,
            reply_to_message_id=message_id
        )
        
        return {
            "success": True,
            "message": "Reply sent successfully",
            "to": [reply_to_email],
            "subject": reply_subject,
            "original_message_id": message_id
        }
        
    except Exception as e:
        logger.error(f"Error replying to email: {e}")
        raise ToolError(f"Failed to reply to email: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.app, host="0.0.0.0", port=9027)
