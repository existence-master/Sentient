import os
from typing import Dict, Any, List, Optional
import datetime, inspect
import re
import asyncio
import re

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.prompts.prompt import Message
from fastmcp.utilities.logging import configure_logging, get_logger
from fastmcp.exceptions import ToolError
from composio import Composio
from main.config import COMPOSIO_API_KEY

# Local imports for modularity
from . import auth
from . import prompts
from . import utils as helpers

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# --- Composio Client ---
composio = Composio(api_key=COMPOSIO_API_KEY)

# Conditionally load .env for local development
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

# --- Server Initialization ---
mcp = FastMCP(
    name="GMailServer",
    instructions="Provides a comprehensive suite of tools to read, search, send, and manage emails and labels in a user's Gmail account.",
)


# --- Prompt Registration ---
@mcp.resource("prompt://gmail-agent-system")
def get_gmail_system_prompt() -> str:
    """Provides the system prompt for the GMail agent."""
    return prompts.gmail_agent_system_prompt

# --- Tool Helper ---
async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gmail")
        
        tool_name = inspect.stack()[1].function
        logger.info(f"Executing tool: {tool_name} with parameters: {kwargs}")

        # NEW: Fetch user info including privacy filters
        # Composio's execute method is synchronous, so we use asyncio.to_thread
        result = await asyncio.to_thread(
            composio.tools.execute,
            action_name,
            arguments=kwargs,
            connected_account_id=connection_id,
            dangerously_skip_version_check=True
        )

        if not result.get("successful"):
            raise ToolError(f"Composio action '{action_name}' failed: {result.get('error', 'Unknown error')}")

        data_payload = result.get("data")

        # Apply privacy filters and simplify if the action is fetching emails
        if action_name == "GMAIL_FETCH_EMAILS" and isinstance(data_payload, list):
            user_info = await auth.get_user_info(user_id)
            privacy_filters = user_info.get("privacy_filters", {})
            keyword_filters = privacy_filters.get("keywords", [])
            email_filters = [email.lower() for email in privacy_filters.get("emails", [])]
            label_filters = [label.lower() for label in privacy_filters.get("labels", [])]
            logger.info(f"Applying privacy filters for user {user_id}: Keywords={len(keyword_filters)}, Emails={len(email_filters)}, Labels={len(label_filters)}")

            emails = data_payload
            filtered_emails = []
            for email in emails:
                if not isinstance(email, dict):
                    logger.warning(f"Skipping non-dictionary item in email list: {type(email)}")
                    continue

                # Correctly extract fields for filtering based on the sample response
                subject = email.get("subject", "")
                body = email.get("messageText", "") # Use the full text body
                content_to_check = f"{subject} {body}".lower()

                if any(word.lower() in content_to_check for word in keyword_filters):
                    logger.info(f"Filtering email '{subject}' due to keyword match.")
                    continue
                
                sender_email = _extract_email_from_sender(email.get("sender", ""))
                if any(blocked_email in sender_email for blocked_email in email_filters):
                    logger.info(f"Filtering email '{subject}' due to sender email match.")
                    continue

                email_labels = [label.lower() for label in email.get("labelIds", [])]
                if any(blocked_label in email_labels for blocked_label in label_filters):
                    logger.info(f"Filtering email '{subject}' due to label match.")
                    continue

                filtered_emails.append(email)
            
            logger.info(f"Applied privacy filters. Kept {len(filtered_emails)} out of {len(emails)} emails.")

            # Now, simplify the filtered emails to provide a clean, useful structure to the LLM
            simplified_emails = [
                {
                    # Keys requested by the user
                    "attachmentList": email.get('attachmentList', []),
                    "labelIds": email.get('labelIds', []),
                    "messageId": email.get('messageId'),
                    "messageText": email.get('messageText', ''),
                    "messageTimestamp": email.get('messageTimestamp'),
                    # Also include a few other highly relevant fields for context
                    "sender": email.get('sender'),
                    "subject": email.get('subject', ''),
                    "snippet": email.get('preview', {}).get('body', ''),
                    "threadId": email.get('threadId'),
                } for email in filtered_emails
            ]
            
            # The final result for the agent should be a dictionary containing the list of simplified emails
            return {"status": "success", "result": {"messages": simplified_emails}}

        # For all other actions, just return the data payload
        return {"status": "success", "result": data_payload}
    except Exception as e:
        logger.error(f"Tool execution failed for action '{action_name}': {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

def _extract_email_from_sender(sender_string: str) -> str:
    """Extracts email from 'Name <email@example.com>' format."""
    if not isinstance(sender_string, str):
        return ""
    match = re.search(r'<(.+?)>', sender_string)
    if match:
        return match.group(1).lower()
    return sender_string.lower()

# --- Async Tool Definitions ---

@mcp.tool()
async def sendEmail(ctx: Context, to: str, subject: str, body: str) -> Dict[str, Any]:
    """Compose and send a new email message to one or more recipients."""
    logger.info(f"Executing tool: sendEmail to='{to}'")
    return await _execute_tool(ctx, "GMAIL_SEND_EMAIL", recipient_email=to, subject=subject, body=body)

@mcp.tool()
async def replyToEmail(ctx: Context, thread_id: str, to: str, body: str) -> Dict[str, Any]:
    """Reply to an email within a specific thread. You must provide the thread_id, recipient's email, and the message body."""
    logger.info(f"Executing tool: replyToEmail in thread_id='{thread_id}'")
    return await _execute_tool(ctx, "GMAIL_REPLY_TO_THREAD", thread_id=thread_id, recipient_email=to, message_body=body)

@mcp.tool()
async def getLatestEmails(ctx: Context, max_results: int = 10, inbox_type: str = "primary") -> Dict[str, Any]:
    """Retrieve the most recent email messages from your inbox, sorted by date received. Can specify inbox_type: 'primary', 'social', 'promotions', 'updates', 'forums'."""
    logger.info(f"Executing tool: getLatestEmails with max_results={max_results}, inbox_type='{inbox_type}'")
    timestamp_48h_ago = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)).timestamp())
    query = f"in:inbox category:{inbox_type} after:{timestamp_48h_ago}"
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query=query, max_results=max_results)

@mcp.tool()
async def getUnreadEmails(ctx: Context, max_results: int = 10, inbox_type: str = "primary") -> Dict[str, Any]:
    """Retrieve unread email messages from your inbox. Can specify inbox_type: 'primary', 'social', 'promotions', 'updates', 'forums'."""
    logger.info(f"Executing tool: getUnreadEmails with max_results={max_results}, inbox_type='{inbox_type}'")
    timestamp_48h_ago = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)).timestamp())
    query = f"is:unread in:inbox category:{inbox_type} after:{timestamp_48h_ago}"
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query=query, max_results=max_results)

@mcp.tool()
async def createLabel(ctx: Context, name: str) -> Dict[str, Any]:
    """Create a new Gmail label for organizing emails."""
    logger.info(f"Executing tool: createLabel with name='{name}'")
    return await _execute_tool(ctx, "GMAIL_CREATE_LABEL", label_name=name)

@mcp.tool()
async def applyLabels(ctx: Context, message_id: str, label_ids: List[str]) -> Dict[str, Any]:
    """Add one or more labels to a specific email message."""
    logger.info(f"Executing tool: applyLabels to message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_ADD_LABEL_TO_EMAIL", message_id=message_id, add_label_ids=label_ids)

@mcp.tool()
async def createDraft(ctx: Context, to: str, subject: str, body: str) -> Dict[str, Any]:
    """Create a new draft email that can be edited before sending."""
    logger.info(f"Executing tool: createDraft to='{to}'")
    return await _execute_tool(ctx, "GMAIL_CREATE_EMAIL_DRAFT", recipient_email=to, subject=subject, body=body)

@mcp.tool()
async def listDrafts(ctx: Context) -> Dict[str, Any]:
    """List all saved draft emails in the user's account."""
    logger.info("Executing tool: listDrafts")
    return await _execute_tool(ctx, "GMAIL_LIST_DRAFTS")

@mcp.tool()
async def sendDraft(ctx: Context, draft_id: str) -> Dict[str, Any]:
    """Sends an existing draft email."""
    logger.info(f"Executing tool: sendDraft for draft_id='{draft_id}'")
    return await _execute_tool(ctx, "GMAIL_SEND_DRAFT", draft_id=draft_id)

@mcp.tool()
async def markAsRead(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Mark an email message as read."""
    logger.info(f"Executing tool: markAsRead for message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_ADD_LABEL_TO_EMAIL", message_id=message_id, remove_label_ids=["UNREAD"])

@mcp.tool()
async def moveToTrash(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Move an email message to the trash."""
    logger.info(f"Executing tool: moveToTrash for message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_MOVE_TO_TRASH", message_id=message_id)

@mcp.tool()
async def archiveEmail(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Remove an email message from the inbox without deleting it (archive)."""
    logger.info(f"Executing tool: archiveEmail for message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_ADD_LABEL_TO_EMAIL", message_id=message_id, remove_label_ids=["INBOX"])

@mcp.tool()
async def searchWithAttachments(ctx: Context, max_results: int = 5) -> Dict[str, Any]:
    """Search for email messages that have file attachments."""
    logger.info(f"Executing tool: searchWithAttachments with max_results={max_results}")
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query="has:attachment", max_results=max_results)

@mcp.tool()
async def catchup(ctx: Context, inbox_type: str = "primary") -> Dict[str, Any]:
    """Get a quick compact summary of all unread emails from your inbox. Can specify inbox_type: 'primary', 'social', 'promotions', 'updates', 'forums'."""
    logger.info(f"Executing tool: catchup for inbox_type='{inbox_type}'")
    timestamp_48h_ago = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)).timestamp())
    query = f"is:unread in:inbox category:{inbox_type} after:{timestamp_48h_ago}"
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query=query, max_results=20)

@mcp.tool()
async def readEmail(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Retrieve and read the content of a specific email message by its unique ID."""
    logger.info(f"Executing tool: readEmail with message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", message_id=message_id)

@mcp.tool()
async def getEmailsByThread(ctx: Context, thread_id: str) -> Dict[str, Any]:
    """Retrieve all email messages that belong to the same conversation thread."""
    logger.info(f"Executing tool: getEmailsByThread with thread_id='{thread_id}'")
    return await _execute_tool(ctx, "GMAIL_FETCH_MESSAGE_BY_THREAD_ID", thread_id=thread_id)

@mcp.tool()
async def getEmailsBySender(ctx: Context, sender_email: str, max_results: int = 10) -> Dict[str, Any]:
    """Retrieve email messages from a specific sender email address."""
    logger.info(f"Executing tool: getEmailsBySender with sender_email='{sender_email}'")
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query=f"from:{sender_email}", max_results=max_results)

@mcp.tool()
async def searchEmails(ctx: Context, query: str, max_results: int = 10) -> Dict[str, Any]:
    """Search for email messages using Gmail search operators and syntax."""
    logger.info(f"Executing tool: searchEmails with query='{query}'")
    return await _execute_tool(ctx, "GMAIL_FETCH_EMAILS", query=query, max_results=max_results)

@mcp.tool()
async def listLabels(ctx: Context) -> Dict[str, Any]:
    """List all available Gmail labels in the user's account."""
    logger.info("Executing tool: listLabels")
    return await _execute_tool(ctx, "GMAIL_LIST_LABELS")

@mcp.tool()
async def getContacts(ctx: Context, person_fields: str = "emailAddresses,names") -> Dict[str, Any]:
    """
    Fetches contacts from the user's Google account.
    You can specify which fields to retrieve using person_fields (comma-separated, e.g., 'emailAddresses,names,phoneNumbers').
    """
    logger.info(f"Executing tool: getContacts with person_fields='{person_fields}'")
    return await _execute_tool(ctx, "GMAIL_GET_CONTACTS", person_fields=person_fields)

@mcp.tool()
async def removeLabels(ctx: Context, message_id: str, label_ids: List[str]) -> Dict[str, Any]:
    """Remove one or more labels from a specific email message."""
    logger.info(f"Executing tool: removeLabels from message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_ADD_LABEL_TO_EMAIL", message_id=message_id, remove_label_ids=label_ids)

@mcp.tool()
async def deleteDraft(ctx: Context, draft_id: str) -> Dict[str, Any]:
    """Delete a saved draft email."""
    logger.info(f"Executing tool: deleteDraft with draft_id='{draft_id}'")
    return await _execute_tool(ctx, "GMAIL_DELETE_DRAFT", draft_id=draft_id)

@mcp.tool()
async def markAsUnread(ctx: Context, message_id: str) -> Dict[str, Any]:
    """Mark an email message as unread."""
    logger.info(f"Executing tool: markAsUnread for message_id='{message_id}'")
    return await _execute_tool(ctx, "GMAIL_ADD_LABEL_TO_EMAIL", message_id=message_id, add_label_ids=["UNREAD"])

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9001))
    
    print(f"Starting GMail MCP Server on http://{host}:{port}")
    
    mcp.run(transport="sse", host=host, port=port)