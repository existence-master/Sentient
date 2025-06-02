# src/server/context/gmail.py
from server.context.base import BaseContextEngine
from server.agents.functions import authenticate_gmail # Your existing Gmail auth
# from server.context.runnables import get_gmail_context_runnable # May not be needed if we just publish
from datetime import datetime, timezone
import asyncio
import json # For Kafka messages
from typing import Optional, Any, List, Tuple, Dict
from googleapiclient.errors import HttpError
import httpx # If you use httpx for other API calls; google-api-python-client handles its own.
import os
import uuid # For generating unique trace IDs for Kafka messages

# Assuming Kafka producer setup elsewhere, e.g., in app.py or a dedicated Kafka utils
# For now, we'll define a placeholder.
from server.utils.producers import KafkaProducerManager # Hypothetical
KAFKA_PRODUCER = KafkaProducerManager.get_producer()
GMAIL_KAFKA_TOPIC = os.getenv("GMAIL_KAFKA_TOPIC", "gmail_new_items")

GMAIL_POLL_HISTORY_MAX_RESULTS = int(os.getenv("GMAIL_POLL_HISTORY_MAX_RESULTS", 100))


class GmailContextEngine(BaseContextEngine):
    """Context Engine for processing Gmail data with dynamic polling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "gmail"
        print(f"[GMAIL_ENGINE_INIT] Initializing for user: {self.user_id}")
        try:
            # Make sure authenticate_gmail() is robust or wrapped
            self.gmail_service = authenticate_gmail() # This needs to be callable and return a service object
            print(f"[GMAIL_ENGINE_INIT] Gmail service authenticated for user: {self.user_id}")
        except Exception as e:
            print(f"[GMAIL_ENGINE_ERROR] Failed to authenticate Gmail for user {self.user_id}: {e}")
            self.gmail_service = None
        print(f"[GMAIL_ENGINE_INIT] Initialization complete for user: {self.user_id}")

    async def fetch_new_data(self, last_history_id: Optional[str], page_token: Optional[str]) -> Optional[Tuple[Optional[List[Any]], Optional[str], Optional[str]]]:
        if not self.gmail_service:
            print(f"[GMAIL_FETCH_ERROR] Gmail service not available for user {self.user_id}.")
            # Raise an error or return None to signal failure to the poll cycle
            raise ConnectionError("Gmail service not available for polling.")

        print(f"[GMAIL_FETCH] Fetching new Gmail history for user: {self.user_id}, startHistoryId: {last_history_id}, pageToken: {page_token}")
        
        history_records_payload: List[Dict[str, Any]] = []
        current_history_id_for_marker = last_history_id 
        next_page = page_token

        try:
            # Loop to handle pagination within a single fetch_new_data call, up to a limit
            # This is useful if a previous poll was interrupted mid-pagination.
            # The primary loop is managed by the scheduler for distinct polling intervals.
            # This inner loop is for fetching all pages for *this specific* poll attempt.
            MAX_PAGES_PER_POLL_CYCLE = 5 # Limit pages fetched in one go to avoid long running tasks
            pages_fetched = 0

            while pages_fetched < MAX_PAGES_PER_POLL_CYCLE:
                request_params = {
                    "userId": "me",
                    "maxResults": GMAIL_POLL_HISTORY_MAX_RESULTS,
                }
                if last_history_id and not next_page: # Only use startHistoryId if no pageToken
                    request_params["startHistoryId"] = last_history_id
                if next_page:
                    request_params["pageToken"] = next_page
                
                # The history.list method is synchronous, run in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    self.gmail_service.users().history().list(**request_params).execute
                )
                
                history_items = response.get("history", [])
                new_next_page_token = response.get("nextPageToken")
                current_history_id_for_marker = response.get("historyId", current_history_id_for_marker) # Gmail provides current historyId

                if history_items:
                    history_records_payload.extend(history_items)
                
                next_page = new_next_page_token # Update for next iteration of this inner loop
                pages_fetched += 1

                if not next_page: # No more pages in this batch
                    break
            
            print(f"[GMAIL_FETCH] Fetched {len(history_records_payload)} history records across {pages_fetched} pages for user {self.user_id}.")
            return history_records_payload, current_history_id_for_marker, next_page # next_page could be None

        except HttpError as e:
            print(f"[GMAIL_FETCH_API_ERROR] Gmail API error for user {self.user_id}: {e.resp.status} - {e._get_reason()}")
            # Re-raise as httpx.HTTPStatusError for consistency if your BaseContextEngine expects that
            # Or handle specific status codes here. For now, re-raise to be caught by run_poll_cycle
            # To simulate httpx.HTTPStatusError:
            # response_mock = httpx.Response(status_code=e.resp.status, text=e._get_reason())
            # raise httpx.HTTPStatusError(message=e._get_reason(), request=None, response=response_mock)
            raise # Let the run_poll_cycle handle HttpError and its status codes
        except Exception as e:
            print(f"[GMAIL_FETCH_ERROR] Unexpected error fetching Gmail history for user {self.user_id}: {e}")
            raise # Let the run_poll_cycle handle general errors

    async def process_and_publish_data(self, history_records: List[Dict[str, Any]]):
        """
        Process Gmail history records, check for duplicates, and publish new items to Kafka.
        """
        if not self.category: return # Should be set

        print(f"[GMAIL_PROCESS] Processing {len(history_records)} Gmail history records for user {self.user_id}.")
        
        # KAFKA_PRODUCER and GMAIL_KAFKA_TOPIC should be accessible here
        # For now, this is a placeholder for Kafka publishing.
        # You'll need to integrate your Kafka producer logic.
        
        # Example of how you might structure a Kafka producer call:
        # if not KAFKA_PRODUCER:
        #     print(f"[GMAIL_PROCESS_ERROR] Kafka producer not available for user {self.user_id}.")
        #     return

        new_items_published_count = 0
        for record in history_records:
            # Gmail history can contain various types of changes (messagesAdded, labelsAdded, etc.)
            # We are interested in 'messagesAdded' to detect new emails.
            if "messagesAdded" in record:
                for item in record["messagesAdded"]:
                    message = item.get("message", {})
                    message_id = message.get("id")
                    thread_id = message.get("threadId")
                    
                    if not message_id:
                        continue

                    # Check if this message_id has already been processed
                    if await self.mongo_manager.is_item_processed(self.user_id, self.category, message_id):
                        print(f"[GMAIL_PROCESS] Message {message_id} already processed for user {self.user_id}. Skipping.")
                        continue

                    # Construct Kafka message
                    kafka_message = {
                        "userId": self.user_id,
                        "sourceApp": self.category,
                        "eventType": "new_email_detected", # Or more specific based on history record type
                        "eventTimestamp": datetime.now(timezone.utc).isoformat(),
                        "eventData": {
                            "original_message_id": message_id,
                            "thread_id": thread_id,
                            # "historyId": record.get("id") # The history record ID itself
                            # Potentially add minimal metadata if available in message object
                            # For full details, Sentient Core will fetch using message_id
                        },
                        "traceId": str(uuid.uuid4()) # Unique trace ID for this event
                    }
                    
                    try:
                        # Placeholder for actual Kafka publish
                        # await KAFKA_PRODUCER.send_and_wait(GMAIL_KAFKA_TOPIC, json.dumps(kafka_message).encode('utf-8'))
                        print(f"[GMAIL_KAFKA_PUBLISH_SUCCESS] (Simulated) Published message_id {message_id} for user {self.user_id} to Kafka.")
                        
                        # Log as processed AFTER successful publish
                        await self.mongo_manager.log_processed_item(self.user_id, self.category, message_id)
                        new_items_published_count +=1
                    except Exception as e_kafka:
                        print(f"[GMAIL_KAFKA_PUBLISH_ERROR] Failed to publish message_id {message_id} for user {self.user_id}: {e_kafka}")
                        # Decide on retry logic or marking as failed for this item
                        break # Stop processing further items in this batch on Kafka error

        print(f"[GMAIL_PROCESS] Finished processing for user {self.user_id}. Published {new_items_published_count} new items to Kafka.")


    async def get_runnable(self):
        # This runnable was for the old context engine design.
        # For dynamic polling, this might not be directly used in the poll cycle itself,
        # as the primary output is Kafka messages.
        # It could be used if, after polling, we want an LLM to summarize what was found.
        print(f"[GMAIL_ENGINE_RUNNABLE] get_runnable called for user {self.user_id} (may not be used by dynamic poller).")
        # return get_gmail_context_runnable() # Assuming this exists
        return None 

    # execute_outputs is handled by BaseContextEngine or might be deprecated for polling engines
    # as the main output is publishing to Kafka. If specific notifications are needed based
    # on polling results (e.g. "You have 5 new important emails"), this could be adapted.