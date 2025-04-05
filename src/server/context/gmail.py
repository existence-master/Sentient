from server.context.base import BaseContextEngine
from server.agents.functions import authenticate_gmail
from server.context.runnables import get_gmail_context_runnable
from datetime import datetime
import asyncio

class GmailContextEngine(BaseContextEngine):
    """Context Engine for processing Gmail data."""

    def __init__(self, *args, **kwargs):
        print("GmailContextEngine.__init__ started")
        super().__init__(*args, **kwargs)
        print("GmailContextEngine.__init__ - calling authenticate_gmail()")
        self.gmail_service = authenticate_gmail()
        print("GmailContextEngine.__init__ - gmail_service authenticated")
        self.category = "gmail"
        print(f"GmailContextEngine.__init__ - category set to: {self.category}")
        print("GmailContextEngine.__init__ finished")
    
    async def start(self):
        """Start the engine, running periodically every hour."""
        print("BaseContextEngine.start started")
        while True:
            print("BaseContextEngine.start - running engine iteration")
            await self.run_engine()
            print("BaseContextEngine.start - engine iteration finished, sleeping for 3600 seconds")
            await asyncio.sleep(3600)  # Check every hour

    async def fetch_new_data(self):
        """Fetch new emails from Gmail since the last check."""
        print("GmailContextEngine.fetch_new_data started")
        print("GmailContextEngine.fetch_new_data - listing emails from inbox (max 5)")
        results = self.gmail_service.users().messages().list(userId="me", maxResults=5, q="in:inbox").execute()
        messages = results.get("messages", [])
        print(f"GmailContextEngine.fetch_new_data - fetched {len(messages)} messages")
        current_top_5_ids = [msg["id"] for msg in messages]
        print(f"GmailContextEngine.fetch_new_data - current top 5 email IDs: {current_top_5_ids}")
        gmail_context = self.context.get("gmail", {})
        previous_top_5_ids = gmail_context.get("emails", {}).get("last_email_ids", [])
        print(f"GmailContextEngine.fetch_new_data - previous top 5 email IDs: {previous_top_5_ids}")
        new_email_ids = [eid for eid in current_top_5_ids if eid not in previous_top_5_ids]
        print(f"GmailContextEngine.fetch_new_data - new email IDs to fetch: {new_email_ids}")
        new_emails = []
        for email_id in new_email_ids:
            print(f"GmailContextEngine.fetch_new_data - getting email with ID: {email_id}")
            email = self.gmail_service.users().messages().get(userId="me", id=email_id).execute()
            new_emails.append(email)
            print(f"GmailContextEngine.fetch_new_data - fetched email with ID: {email_id}")
        # Update context with the latest email IDs
        print("GmailContextEngine.fetch_new_data - updating context")
        self.context["gmail"] = {
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "emails": {"last_email_ids": current_top_5_ids}
        }
        print("GmailContextEngine.fetch_new_data - saving context")
        await self.save_context()
        print(f"GmailContextEngine.fetch_new_data - returning {len(new_emails)} new emails")
        print("GmailContextEngine.fetch_new_data finished")
        return new_emails

    async def process_new_data(self, new_emails):
        """Process new emails into summaries."""
        print("GmailContextEngine.process_new_data started")
        print(f"GmailContextEngine.process_new_data - processing {len(new_emails)} new emails")
        summaries = [await self.generate_email_summary(email) for email in new_emails]
        print(f"GmailContextEngine.process_new_data - generated {len(summaries)} summaries")
        joined_summaries = "\n".join(summaries)
        print(f"GmailContextEngine.process_new_data - joined summaries: {joined_summaries}")
        print("GmailContextEngine.process_new_data finished")
        return joined_summaries

    async def get_runnable(self):
        """Return the Gmail-specific runnable."""
        print("GmailContextEngine.get_runnable started")
        runnable = get_gmail_context_runnable()
        print(f"GmailContextEngine.get_runnable - returning runnable: {runnable}")
        print("GmailContextEngine.get_runnable finished")
        return runnable

    async def get_category(self):
        """Return the memory category for Gmail."""
        print("GmailContextEngine.get_category started")
        print(f"GmailContextEngine.get_category - returning category: {self.category}")
        print("GmailContextEngine.get_category finished")
        return self.category

    async def generate_email_summary(self, email):
        """Generate a summary for a single email."""
        print("GmailContextEngine.generate_email_summary started")
        snippet = email.get("snippet", "")
        print(f"GmailContextEngine.generate_email_summary - snippet: {snippet}")
        subject = next((header["value"] for header in email["payload"]["headers"] if header["name"] == "Subject"), "No Subject")
        print(f"GmailContextEngine.generate_email_summary - subject: {subject}")
        from_ = next((header["value"] for header in email["payload"]["headers"] if header["name"] == "From"), "Unknown Sender")
        print(f"GmailContextEngine.generate_email_summary - from: {from_}")
        summary = f"You have an email from {from_} with subject '{subject}' and snippet '{snippet}'"
        print(f"GmailContextEngine.generate_email_summary - summary: {summary}")
        print("GmailContextEngine.generate_email_summary finished")
        return summary