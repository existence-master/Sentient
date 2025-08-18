import httpx
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class OutlookAPI:
    """Helper class for Microsoft Graph API calls."""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    async def get_emails(self, folder: str = "inbox", top: int = 10, skip: int = 0, 
                        search: Optional[str] = None) -> Dict[str, Any]:
        """Get emails from a specific folder."""
        try:
            url = f"{self.base_url}/me/mailFolders/{folder}/messages"
            params = {
                "$top": top,
                "$skip": skip,
                "$orderby": "receivedDateTime desc",
                "$select": "id,subject,from,toRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview"
            }
            
            if search:
                params["$filter"] = f"contains(subject,'{search}') or contains(body/content,'{search}')"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting emails: {e}")
            raise
    
    async def get_email(self, message_id: str) -> Dict[str, Any]:
        """Get a specific email by ID."""
        try:
            url = f"{self.base_url}/me/messages/{message_id}"
            params = {
                "$select": "id,subject,from,toRecipients,ccRecipients,bccRecipients,receivedDateTime,sentDateTime,isRead,hasAttachments,body,bodyPreview"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting email {message_id}: {e}")
            raise
    
    async def send_email(self, subject: str, body: str, to_recipients: List[str], 
                        cc_recipients: Optional[List[str]] = None,
                        bcc_recipients: Optional[List[str]] = None,
                        reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
        """Send an email."""
        try:
            url = f"{self.base_url}/me/sendMail"
            
            # Prepare recipients
            to_emails = [{"emailAddress": {"address": email}} for email in to_recipients]
            cc_emails = [{"emailAddress": {"address": email}} for email in (cc_recipients or [])]
            bcc_emails = [{"emailAddress": {"address": email}} for email in (bcc_recipients or [])]
            
            # Prepare message
            message = {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content": body
                },
                "toRecipients": to_emails
            }
            
            if cc_emails:
                message["ccRecipients"] = cc_emails
            if bcc_emails:
                message["bccRecipients"] = bcc_emails
            if reply_to_message_id:
                message["replyTo"] = [{"id": reply_to_message_id}]
            
            payload = {"message": message, "saveToSentItems": True}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return {"success": True, "message": "Email sent successfully"}
                
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise
    
    async def get_folders(self) -> Dict[str, Any]:
        """Get email folders."""
        try:
            url = f"{self.base_url}/me/mailFolders"
            params = {
                "$select": "id,displayName,messageCount,unreadItemCount"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting folders: {e}")
            raise
    
    async def search_emails(self, query: str, top: int = 10) -> Dict[str, Any]:
        """Search emails using Microsoft Graph search."""
        try:
            url = f"{self.base_url}/me/messages"
            params = {
                "$search": f'"{query}"',
                "$top": top,
                "$orderby": "receivedDateTime desc",
                "$select": "id,subject,from,toRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            raise

def format_email_summary(email: Dict[str, Any]) -> Dict[str, Any]:
    """Format email data for better readability."""
    try:
        from_info = email.get("from", {})
        to_info = email.get("toRecipients", [])
        
        return {
            "id": email.get("id"),
            "subject": email.get("subject", "No Subject"),
            "from": from_info.get("emailAddress", {}).get("address", "Unknown"),
            "from_name": from_info.get("emailAddress", {}).get("name", "Unknown"),
            "to": [recipient.get("emailAddress", {}).get("address", "") for recipient in to_info],
            "received_date": email.get("receivedDateTime"),
            "is_read": email.get("isRead", False),
            "has_attachments": email.get("hasAttachments", False),
            "body_preview": email.get("bodyPreview", "")
        }
    except Exception as e:
        logger.error(f"Error formatting email: {e}")
        return email

def apply_privacy_filters(emails: List[Dict[str, Any]], privacy_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply privacy filters to email list."""
    try:
        if not privacy_filters:
            return emails
        
        keyword_filters = privacy_filters.get("keywords", [])
        email_filters = [email.lower() for email in privacy_filters.get("emails", [])]
        
        filtered_emails = []
        for email in emails:
            # Skip if email contains filtered keywords
            subject = email.get("subject", "").lower()
            body = email.get("bodyPreview", "").lower()
            
            if any(keyword.lower() in subject or keyword.lower() in body for keyword in keyword_filters):
                continue
            
            # Skip if from filtered email addresses
            from_email = email.get("from", "").lower()
            if from_email in email_filters:
                continue
            
            filtered_emails.append(email)
        
        return filtered_emails
        
    except Exception as e:
        logger.error(f"Error applying privacy filters: {e}")
        return emails
