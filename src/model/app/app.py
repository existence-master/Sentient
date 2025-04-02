# -*- coding: utf-8 -*-
import time
from datetime import datetime, timezone
START_TIME = time.time()
print(f"[STARTUP] {datetime.now()}: Script execution started.")

import os
import json
import asyncio
import pickle
import multiprocessing
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union # Added Union
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import nest_asyncio
import uvicorn
import traceback # For detailed error printing

print(f"[STARTUP] {datetime.now()}: Basic imports completed.")

# Import specific functions, runnables, and helpers from respective folders
print(f"[STARTUP] {datetime.now()}: Importing model components...")
from model.agents.runnables import *
from model.agents.functions import *
from model.agents.prompts import *
from model.agents.formats import *
from model.agents.base import *
from model.agents.helpers import *

from model.memory.runnables import *
from model.memory.functions import *
from model.memory.prompts import *
from model.memory.constants import *
from model.memory.formats import *
from model.memory.backend import MemoryBackend

from model.utils.helpers import *

from model.scraper.runnables import *
from model.scraper.functions import *
from model.scraper.prompts import *
from model.scraper.formats import *

from model.auth.helpers import *

from model.common.functions import *
from model.common.runnables import *
from model.common.prompts import *
from model.common.formats import *

from model.chat.runnables import *
from model.chat.prompts import *
from model.chat.functions import *

from model.context.gmail import GmailContextEngine
from model.context.internet import InternetSearchContextEngine
from model.context.gcalendar import GCalendarContextEngine

from datetime import datetime, timezone


print(f"[STARTUP] {datetime.now()}: Model components import completed.")

# Define available data sources (can be extended in the future)
DATA_SOURCES = ["gmail", "internet_search", "gcalendar"]
print(f"[CONFIG] {datetime.now()}: Available data sources: {DATA_SOURCES}")

# Load environment variables from .env file
print(f"[STARTUP] {datetime.now()}: Loading environment variables from model/.env...")
load_dotenv("model/.env")
print(f"[STARTUP] {datetime.now()}: Environment variables loaded.")

# Apply nest_asyncio to allow nested event loops (useful for development environments)
print(f"[STARTUP] {datetime.now()}: Applying nest_asyncio...")
nest_asyncio.apply()
print(f"[STARTUP] {datetime.now()}: nest_asyncio applied.")

# --- Global Initializations ---
print(f"[INIT] {datetime.now()}: Starting global initializations...")

# Initialize embedding model for memory-related operations
print(f"[INIT] {datetime.now()}: Initializing HuggingFace Embedding model ({os.environ.get('EMBEDDING_MODEL_REPO_ID', 'N/A')})...")
try:
    embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])
    print(f"[INIT] {datetime.now()}: HuggingFace Embedding model initialized successfully.")
except Exception as e:
    print(f"[ERROR] {datetime.now()}: Failed to initialize Embedding model: {e}")
    embed_model = None # Handle potential failure gracefully

# Initialize Neo4j graph driver for knowledge graph interactions
print(f"[INIT] {datetime.now()}: Initializing Neo4j Graph Driver (URI: {os.environ.get('NEO4J_URI', 'N/A')})...")
try:
    graph_driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    graph_driver.verify_connectivity() # Test connection
    print(f"[INIT] {datetime.now()}: Neo4j Graph Driver initialized and connected successfully.")
except Exception as e:
    print(f"[ERROR] {datetime.now()}: Failed to initialize or connect Neo4j Driver: {e}")
    graph_driver = None # Handle potential failure gracefully

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        print(f"[WS_MANAGER] {datetime.now()}: WebSocketManager initialized.")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS_MANAGER] {datetime.now()}: WebSocket connected: {websocket.client}. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS_MANAGER] {datetime.now()}: WebSocket disconnected: {websocket.client}. Total connections: {len(self.active_connections)}")
        else:
            print(f"[WS_MANAGER] {datetime.now()}: WebSocket already disconnected or not found: {websocket.client}")


    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
            # print(f"[WS_MANAGER] {datetime.now()}: Sent personal message to {websocket.client}: {message[:100]}...") # Avoid logging large messages
        except Exception as e:
            print(f"[WS_MANAGER] {datetime.now()}: Error sending personal message to {websocket.client}: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        # print(f"[WS_MANAGER] {datetime.now()}: Broadcasting message: {message[:100]}...") # Avoid logging large messages
        disconnected_websockets = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"[WS_MANAGER] {datetime.now()}: Error broadcasting message to {connection.client}: {e}")
                disconnected_websockets.append(connection) # Mark for removal

        # Remove broken connections outside the iteration loop
        for ws in disconnected_websockets:
            self.disconnect(ws)
        # print(f"[WS_MANAGER] {datetime.now()}: Broadcast finished. Active connections: {len(self.active_connections)}")

manager = WebSocketManager()
print(f"[INIT] {datetime.now()}: WebSocketManager instance created.")

# Initialize runnables from agents
print(f"[INIT] {datetime.now()}: Initializing agent runnables...")
reflection_runnable = get_reflection_runnable()
print(f"[INIT] {datetime.now()}:   - Reflection Runnable initialized.")
inbox_summarizer_runnable = get_inbox_summarizer_runnable()
print(f"[INIT] {datetime.now()}:   - Inbox Summarizer Runnable initialized.")
priority_runnable = get_priority_runnable()
print(f"[INIT] {datetime.now()}:   - Priority Runnable initialized.")
print(f"[INIT] {datetime.now()}: Agent runnables initialization complete.")

# Initialize runnables from memory
print(f"[INIT] {datetime.now()}: Initializing memory runnables...")
graph_decision_runnable = get_graph_decision_runnable()
print(f"[INIT] {datetime.now()}:   - Graph Decision Runnable initialized.")
information_extraction_runnable = get_information_extraction_runnable()
print(f"[INIT] {datetime.now()}:   - Information Extraction Runnable initialized.")
graph_analysis_runnable = get_graph_analysis_runnable()
print(f"[INIT] {datetime.now()}:   - Graph Analysis Runnable initialized.")
text_dissection_runnable = get_text_dissection_runnable()
print(f"[INIT] {datetime.now()}:   - Text Dissection Runnable initialized.")
text_conversion_runnable = get_text_conversion_runnable()
print(f"[INIT] {datetime.now()}:   - Text Conversion Runnable initialized.")
query_classification_runnable = get_query_classification_runnable()
print(f"[INIT] {datetime.now()}:   - Query Classification Runnable initialized.")
fact_extraction_runnable = get_fact_extraction_runnable()
print(f"[INIT] {datetime.now()}:   - Fact Extraction Runnable initialized.")
text_summarizer_runnable = get_text_summarizer_runnable()
print(f"[INIT] {datetime.now()}:   - Text Summarizer Runnable initialized.")
text_description_runnable = get_text_description_runnable()
print(f"[INIT] {datetime.now()}:   - Text Description Runnable initialized.")
chat_history = get_chat_history()
print(f"[INIT] {datetime.now()}:   - Chat History retrieved.")
print(f"[INIT] {datetime.now()}: Memory runnables initialization complete.")

# Initialize chat, agent, and unified classification runnables
print(f"[INIT] {datetime.now()}: Initializing core interaction runnables...")
chat_runnable = get_chat_runnable(chat_history)
print(f"[INIT] {datetime.now()}:   - Chat Runnable initialized.")
agent_runnable = get_agent_runnable(chat_history)
print(f"[INIT] {datetime.now()}:   - Agent Runnable initialized.")
unified_classification_runnable = get_unified_classification_runnable(chat_history)
print(f"[INIT] {datetime.now()}:   - Unified Classification Runnable initialized.")
print(f"[INIT] {datetime.now()}: Core interaction runnables initialization complete.")

# Initialize runnables from scraper
print(f"[INIT] {datetime.now()}: Initializing scraper runnables...")
reddit_runnable = get_reddit_runnable()
print(f"[INIT] {datetime.now()}:   - Reddit Runnable initialized.")
twitter_runnable = get_twitter_runnable()
print(f"[INIT] {datetime.now()}:   - Twitter Runnable initialized.")
print(f"[INIT] {datetime.now()}: Scraper runnables initialization complete.")

# Initialize Internet Search related runnables
print(f"[INIT] {datetime.now()}: Initializing internet search runnables...")
internet_query_reframe_runnable = get_internet_query_reframe_runnable()
print(f"[INIT] {datetime.now()}:   - Internet Query Reframe Runnable initialized.")
internet_summary_runnable = get_internet_summary_runnable()
print(f"[INIT] {datetime.now()}:   - Internet Summary Runnable initialized.")
print(f"[INIT] {datetime.now()}: Internet search runnables initialization complete.")

# Tool handlers registry for agent tools
tool_handlers: Dict[str, callable] = {}
print(f"[INIT] {datetime.now()}: Tool handlers registry initialized.")

# Instantiate the task queue globally
print(f"[INIT] {datetime.now()}: Initializing TaskQueue...")
task_queue = TaskQueue()
print(f"[INIT] {datetime.now()}: TaskQueue initialized.")

def register_tool(name: str):
    """Decorator to register a function as a tool handler."""
    def decorator(func: callable):
        print(f"[TOOL_REGISTRY] {datetime.now()}: Registering tool '{name}' with handler '{func.__name__}'")
        tool_handlers[name] = func
        return func
    return decorator

# Google OAuth2 scopes and credentials (from auth and common)
print(f"[CONFIG] {datetime.now()}: Setting up Google OAuth2 configuration...")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive",
    "https://mail.google.com/",
]
print(f"[CONFIG] {datetime.now()}:   - SCOPES defined: {SCOPES}")

CREDENTIALS_DICT = {
    "installed": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "auth_uri": os.environ.get("GOOGLE_AUTH_URI"),
        "token_uri": os.environ.get("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost"] # Make sure this matches your setup
    }
}
# Mask sensitive parts for logging if desired
masked_creds = CREDENTIALS_DICT.copy()
masked_creds['installed']['client_secret'] = '***REDACTED***' if masked_creds['installed'].get('client_secret') else None
print(f"[CONFIG] {datetime.now()}:   - CREDENTIALS_DICT configured (secrets redacted): {masked_creds}")
print(f"[CONFIG] {datetime.now()}: Google OAuth2 configuration complete.")

# Auth0 configuration from utils
print(f"[CONFIG] {datetime.now()}: Setting up Auth0 configuration...")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")
print(f"[CONFIG] {datetime.now()}:   - AUTH0_DOMAIN: {AUTH0_DOMAIN}")
print(f"[CONFIG] {datetime.now()}:   - MANAGEMENT_CLIENT_ID: {'Set' if MANAGEMENT_CLIENT_ID else 'Not Set'}")
print(f"[CONFIG] {datetime.now()}:   - MANAGEMENT_CLIENT_SECRET: {'Set' if MANAGEMENT_CLIENT_SECRET else 'Not Set'}")
print(f"[CONFIG] {datetime.now()}: Auth0 configuration complete.")

# Database paths
print(f"[CONFIG] {datetime.now()}: Defining database file paths...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB = os.path.join(BASE_DIR, "..", "..", "userProfileDb.json")
CHAT_DB = "chatsDb.json"
NOTIFICATIONS_DB = "notificationsDB.json"
print(f"[CONFIG] {datetime.now()}:   - USER_PROFILE_DB: {USER_PROFILE_DB}")
print(f"[CONFIG] {datetime.now()}:   - CHAT_DB: {CHAT_DB}")
print(f"[CONFIG] {datetime.now()}:   - NOTIFICATIONS_DB: {NOTIFICATIONS_DB}")
print(f"[CONFIG] {datetime.now()}: Database file paths defined.")

db_lock = asyncio.Lock()  # Lock for synchronizing chat database access
notifications_db_lock = asyncio.Lock() # Lock for notifications database access
print(f"[INIT] {datetime.now()}: Database locks initialized.")

initial_db = {
    "chats": [],
    "active_chat_id": None,
    "next_chat_id": 1
}
print(f"[CONFIG] {datetime.now()}: Initial chat DB structure defined.")

# --- Helper Functions with Logging ---

def load_user_profile():
    """Load user profile data from userProfileDb.json."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to load user profile from {USER_PROFILE_DB}")
    try:
        with open(USER_PROFILE_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(f"[DB_HELPER] {datetime.now()}: User profile loaded successfully from {USER_PROFILE_DB}")
            return data
    except FileNotFoundError:
        print(f"[DB_HELPER] {datetime.now()}: User profile file not found ({USER_PROFILE_DB}). Returning default structure.")
        return {"userData": {}} # Return empty structure if file not found
    except json.JSONDecodeError as e:
        print(f"[ERROR] {datetime.now()}: Error decoding JSON from {USER_PROFILE_DB}: {e}. Returning default structure.")
        return {"userData": {}} # Handle case where JSON is corrupted or empty
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error loading user profile from {USER_PROFILE_DB}: {e}. Returning default structure.")
        return {"userData": {}}

def write_user_profile(data):
    """Write user profile data to userProfileDb.json."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to write user profile to {USER_PROFILE_DB}")
    try:
        with open(USER_PROFILE_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4) # Use indent for pretty printing
        # print(f"[DB_HELPER] {datetime.now()}: User profile written successfully to {USER_PROFILE_DB}")
        return True
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error writing user profile to {USER_PROFILE_DB}: {e}")
        return False

async def load_notifications_db():
    """Load the notifications database, initializing it if it doesn't exist."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to load notifications DB from {NOTIFICATIONS_DB}")
    try:
        with open(NOTIFICATIONS_DB, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "notifications" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'notifications' key missing in {NOTIFICATIONS_DB}, adding.")
                data["notifications"] = []
            if "next_notification_id" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'next_notification_id' key missing in {NOTIFICATIONS_DB}, adding.")
                data["next_notification_id"] = 1
            # print(f"[DB_HELPER] {datetime.now()}: Notifications DB loaded successfully from {NOTIFICATIONS_DB}")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[DB_HELPER] {datetime.now()}: Notifications DB ({NOTIFICATIONS_DB}) not found or invalid JSON ({e}). Initializing with default structure.")
        return {"notifications": [], "next_notification_id": 1}
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error loading notifications DB from {NOTIFICATIONS_DB}: {e}. Initializing with default structure.")
        return {"notifications": [], "next_notification_id": 1}

async def save_notifications_db(data):
    """Save the notifications database."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to save notifications DB to {NOTIFICATIONS_DB}")
    try:
        with open(NOTIFICATIONS_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # print(f"[DB_HELPER] {datetime.now()}: Notifications DB saved successfully to {NOTIFICATIONS_DB}")
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error saving notifications DB to {NOTIFICATIONS_DB}: {e}")

print(f"[INIT] {datetime.now()}: Initializing MemoryBackend...")
memory_backend = MemoryBackend()
print(f"[INIT] {datetime.now()}: MemoryBackend initialized. Performing cleanup...")
memory_backend.cleanup() # Ensure cleanup happens after initialization
print(f"[INIT] {datetime.now()}: MemoryBackend cleanup complete.")

async def load_db():
    """Load the chat database from chatsDb.json, initializing if it doesn't exist or is invalid."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to load chat DB from {CHAT_DB}")
    try:
        with open(CHAT_DB, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate structure
            if "chats" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'chats' key missing in {CHAT_DB}, adding.")
                data["chats"] = []
            if "active_chat_id" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'active_chat_id' key missing in {CHAT_DB}, setting to None.")
                data["active_chat_id"] = None
            if "next_chat_id" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'next_chat_id' key missing in {CHAT_DB}, adding.")
                data["next_chat_id"] = 1
            # print(f"[DB_HELPER] {datetime.now()}: Chat DB loaded successfully from {CHAT_DB}")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[DB_HELPER] {datetime.now()}: Chat DB ({CHAT_DB}) not found or invalid JSON ({e}). Initializing with default structure.")
        return initial_db.copy() # Return a copy to avoid modifying the global initial_db
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error loading chat DB from {CHAT_DB}: {e}. Initializing with default structure.")
        return initial_db.copy()


async def save_db(data):
    """Save the data to chatsDb.json."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to save chat DB to {CHAT_DB}")
    try:
        with open(CHAT_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # print(f"[DB_HELPER] {datetime.now()}: Chat DB saved successfully to {CHAT_DB}")
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error saving chat DB to {CHAT_DB}: {e}")


async def get_chat_history_messages() -> List[Dict[str, Any]]:
    """
    Function to retrieve the chat history of the currently active chat.
    Checks for inactivity and creates a new chat if needed.
    Returns the list of messages for the active chat, filtering out messages where isVisible is False.
    """
    # print(f"[CHAT_HISTORY] {datetime.now()}: get_chat_history_messages called.")
    async with db_lock:
        # print(f"[CHAT_HISTORY] {datetime.now()}: Acquired chat DB lock.")
        chatsDb = await load_db()
        active_chat_id = chatsDb["active_chat_id"]
        current_time = datetime.now(timezone.utc) # Use alias dt here

        # print(f"[CHAT_HISTORY] {datetime.now()}: Current active_chat_id: {active_chat_id}")

        # If no active chat exists, create a new one
        if active_chat_id is None or not chatsDb["chats"]:
            print(f"[CHAT_HISTORY] {datetime.now()}: No active chat found or chats list is empty. Creating a new chat.")
            new_chat_id = f"chat_{chatsDb['next_chat_id']}"
            chatsDb["next_chat_id"] += 1
            new_chat = {"id": new_chat_id, "messages": [], "created_at": current_time.isoformat() + "Z"}
            chatsDb["chats"].append(new_chat)
            chatsDb["active_chat_id"] = new_chat_id
            await save_db(chatsDb)
            print(f"[CHAT_HISTORY] {datetime.now()}: New chat created (ID: {new_chat_id}). Returning empty messages.")
            print(f"[CHAT_HISTORY] {datetime.now()}: Releasing chat DB lock.")
            return []  # Return empty messages for new chat

        # Find the active chat
        active_chat = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)

        # Check for inactivity (only if active chat exists and has messages)
        if active_chat and active_chat["messages"]:
            last_message = active_chat["messages"][-1]
            # Robust timestamp parsing
            try:
                 # Handle both 'Z' and '+00:00' formats
                ts_str = last_message["timestamp"].replace('Z', '+00:00')
                last_timestamp = datetime.fromisoformat(ts_str)
                # Ensure timezone aware comparison
                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

                time_diff_seconds = (current_time - last_timestamp).total_seconds()
                inactivity_threshold = 600 # 10 minutes

                # print(f"[CHAT_HISTORY] {datetime.now()}: Last message timestamp: {last_timestamp}, Current time: {current_time}, Diff (s): {time_diff_seconds}")

                if time_diff_seconds > inactivity_threshold:
                    print(f"[CHAT_HISTORY] {datetime.now()}: Inactivity period ({time_diff_seconds}s > {inactivity_threshold}s) exceeded for chat {active_chat_id}. Creating a new chat.")
                    new_chat_id = f"chat_{chatsDb['next_chat_id']}"
                    chatsDb["next_chat_id"] += 1
                    new_chat = {"id": new_chat_id, "messages": [], "created_at": current_time.isoformat() + "Z"}
                    chatsDb["chats"].append(new_chat)
                    chatsDb["active_chat_id"] = new_chat_id
                    await save_db(chatsDb)
                    print(f"[CHAT_HISTORY] {datetime.now()}: New chat created due to inactivity (ID: {new_chat_id}). Returning empty messages.")
                    print(f"[CHAT_HISTORY] {datetime.now()}: Releasing chat DB lock.")
                    return [] # Return empty messages for new chat
            except (ValueError, KeyError) as e:
                 print(f"[WARN] {datetime.now()}: Could not parse timestamp for inactivity check in chat {active_chat_id}: {e}. Skipping inactivity check.")


        # Return messages from the active chat, filtering out those with isVisible: False
        # Re-find active chat in case it was just created
        active_chat = next((chat for chat in chatsDb["chats"] if chat["id"] == chatsDb["active_chat_id"]), None)
        if active_chat and "messages" in active_chat:
            filtered_messages = [
                message for message in active_chat["messages"]
                # Default to True if 'isVisible' key is missing or its value is not explicitly False
                if message.get("isVisible", True) is not False
            ]
            # print(f"[CHAT_HISTORY] {datetime.now()}: Returning {len(filtered_messages)} visible messages for chat {chatsDb['active_chat_id']}.")
            # print(f"[CHAT_HISTORY] {datetime.now()}: Releasing chat DB lock.")
            return filtered_messages
        else:
            # This case should ideally not be reached if a new chat was created, but handles edge cases
            print(f"[CHAT_HISTORY] {datetime.now()}: Active chat ({chatsDb['active_chat_id']}) found but has no messages or 'messages' key is missing. Returning empty list.")
            print(f"[CHAT_HISTORY] {datetime.now()}: Releasing chat DB lock.")
            return []

# --- Background Task Processing ---

async def cleanup_tasks_periodically():
    """Periodically clean up old completed tasks."""
    print(f"[TASK_CLEANUP] {datetime.now()}: Starting periodic task cleanup loop.")
    while True:
        cleanup_interval = 60 * 60 # 1 hour
        print(f"[TASK_CLEANUP] {datetime.now()}: Running cleanup task. Next run in {cleanup_interval} seconds.")
        await task_queue.delete_old_completed_tasks()
        await asyncio.sleep(cleanup_interval)

async def process_queue():
    """Continuously process tasks from the queue."""
    print(f"[TASK_PROCESSOR] {datetime.now()}: Starting task processing loop.")
    while True:
        # print(f"[TASK_PROCESSOR] {datetime.now()}: Checking for next task...")
        task = await task_queue.get_next_task()
        if task:
            task_id = task.get("task_id", "N/A")
            task_desc = task.get("description", "N/A")
            print(f"[TASK_PROCESSOR] {datetime.now()}: Processing task ID: {task_id}, Description: {task_desc[:50]}...")
            try:
                # Execute task within a new asyncio task to handle cancellations
                task_queue.current_task_execution = asyncio.create_task(execute_agent_task(task))
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution started.")
                result = await task_queue.current_task_execution
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution finished successfully. Result length: {len(str(result))}")

                # Add results to chat
                print(f"[TASK_PROCESSOR] {datetime.now()}: Adding task description '{task_desc[:50]}...' to chat {task['chat_id']} as user message (hidden).")
                await add_result_to_chat(task["chat_id"], task["description"], True) # Add hidden user message
                print(f"[TASK_PROCESSOR] {datetime.now()}: Adding task result to chat {task['chat_id']} as assistant message.")
                await add_result_to_chat(task["chat_id"], result, False, task["description"]) # Add visible assistant message

                await task_queue.complete_task(task_id, result=result)
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} marked as completed in queue.")

                # --- WebSocket Message on Success ---
                task_completion_message = {
                    "type": "task_completed",
                    "task_id": task_id,
                    "description": task_desc,
                    "result": result
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task completion for {task_id}")
                await manager.broadcast(json.dumps(task_completion_message))

            except asyncio.CancelledError:
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution was cancelled.")
                await task_queue.complete_task(task_id, error="Task was cancelled", status="cancelled") # Update status to cancelled
                # --- WebSocket Message on Cancellation ---
                task_error_message = {
                    "type": "task_error", # Or maybe "task_cancelled"
                    "task_id": task_id,
                    "description": task_desc,
                    "error": "Task was cancelled"
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task cancellation for {task_id}")
                await manager.broadcast(json.dumps(task_error_message))

            except Exception as e:
                error_str = str(e)
                print(f"[ERROR] {datetime.now()}: Error processing task {task_id}: {error_str}")
                traceback.print_exc() # Print full traceback for debugging
                await task_queue.complete_task(task_id, error=error_str, status="error") # Update status to error
                # --- WebSocket Message on Error ---
                task_error_message = {
                    "type": "task_error",
                    "task_id": task_id,
                    "description": task_desc,
                    "error": error_str
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task error for {task_id}")
                await manager.broadcast(json.dumps(task_error_message))
            finally:
                 task_queue.current_task_execution = None # Reset current execution tracking
        else:
            # No task found, sleep briefly
            await asyncio.sleep(0.1)

async def process_memory_operations():
    """Continuously process memory operations from the memory backend queue."""
    print(f"[MEMORY_PROCESSOR] {datetime.now()}: Starting memory operation processing loop.")
    while True:
        # print(f"[MEMORY_PROCESSOR] {datetime.now()}: Checking for next memory operation...")
        operation = await memory_backend.memory_queue.get_next_operation()

        if operation:
            op_id = operation.get("operation_id", "N/A")
            user_id = operation.get("user_id", "N/A")
            memory_data = operation.get("memory_data", "N/A")
            print(f"[MEMORY_PROCESSOR] {datetime.now()}: Processing memory operation ID: {op_id} for user: {user_id}, Data: {str(memory_data)[:100]}...")

            try:
                # Perform the memory update
                await memory_backend.update_memory(user_id, memory_data)
                print(f"[MEMORY_PROCESSOR] {datetime.now()}: Memory update for user {user_id} successful (Op ID: {op_id}).")

                # Mark operation as complete
                await memory_backend.memory_queue.complete_operation(op_id, result="Success")
                print(f"[MEMORY_PROCESSOR] {datetime.now()}: Memory operation {op_id} marked as completed.")

                # --- WebSocket Notification on Success ---
                notification = {
                    "type": "memory_operation_completed",
                    "operation_id": op_id,
                    "status": "success",
                    "fact": memory_data
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting memory operation success for {op_id}")
                await manager.broadcast(json.dumps(notification))

            except Exception as e:
                error_str = str(e)
                print(f"[ERROR] {datetime.now()}: Error processing memory operation {op_id} for user {user_id}: {error_str}")
                traceback.print_exc() # Print full traceback

                # Mark operation as errored
                await memory_backend.memory_queue.complete_operation(op_id, error=error_str, status="error")
                print(f"[MEMORY_PROCESSOR] {datetime.now()}: Memory operation {op_id} marked as error.")

                # --- WebSocket Notification on Error ---
                notification = {
                    "type": "memory_operation_error",
                    "operation_id": op_id,
                    "error": error_str,
                    "fact": memory_data # Include the data that failed
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting memory operation error for {op_id}")
                await manager.broadcast(json.dumps(notification))

        else:
            # No operation found, sleep briefly
            await asyncio.sleep(0.1)


async def execute_agent_task(task: dict) -> str:
    """Execute the agent task asynchronously and handle approval for email tasks."""
    task_id = task.get("task_id", "N/A")
    task_desc = task.get("description", "N/A")
    print(f"[AGENT_EXEC] {datetime.now()}: Executing task ID: {task_id}, Description: {task_desc[:50]}...")
    print(f"[AGENT_EXEC] {datetime.now()}: Task details: {task}")

    # Explicitly reference globals needed within this function scope
    global agent_runnable, reflection_runnable, inbox_summarizer_runnable, graph_driver, embed_model
    global text_conversion_runnable, query_classification_runnable, internet_query_reframe_runnable, internet_summary_runnable

    transformed_input = task["description"]
    username = task["username"]
    personality = task["personality"]
    use_personal_context = task["use_personal_context"]
    internet = task["internet"]

    user_context = None
    internet_context = None

    # --- Compute User Context ---
    if use_personal_context:
        print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires personal context. Querying user profile...")
        try:
            if graph_driver and embed_model and text_conversion_runnable and query_classification_runnable:
                user_context = query_user_profile(
                    transformed_input,
                    graph_driver,
                    embed_model,
                    text_conversion_runnable,
                    query_classification_runnable
                )
                print(f"[AGENT_EXEC] {datetime.now()}: User context retrieved for task {task_id}. Length: {len(str(user_context)) if user_context else 0}")
            else:
                 print(f"[WARN] {datetime.now()}: Skipping user context query for task {task_id} due to missing dependencies (graph_driver, embed_model, etc.).")
                 user_context = "User context unavailable due to system configuration issues."
        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Error computing user_context for task {task_id}: {e}")
            traceback.print_exc()
            user_context = f"Error retrieving user context: {e}" # Pass error info downstream if needed

    # --- Compute Internet Context ---
    if internet == "Internet":
        print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires internet search.")
        try:
            print(f"[AGENT_EXEC] {datetime.now()}: Re-framing internet query for task {task_id}...")
            reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
            print(f"[AGENT_EXEC] {datetime.now()}: Reframed query for task {task_id}: '{reframed_query}'")

            print(f"[AGENT_EXEC] {datetime.now()}: Performing internet search for task {task_id}...")
            search_results = get_search_results(reframed_query)
            # print(f"[AGENT_EXEC] {datetime.now()}: Search results received for task {task_id}: {search_results}") # Can be verbose

            print(f"[AGENT_EXEC] {datetime.now()}: Summarizing search results for task {task_id}...")
            internet_context = get_search_summary(internet_summary_runnable, search_results)
            print(f"[AGENT_EXEC] {datetime.now()}: Internet context summary generated for task {task_id}. Length: {len(str(internet_context)) if internet_context else 0}")

        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Error computing internet_context for task {task_id}: {e}")
            traceback.print_exc()
            internet_context = f"Error retrieving internet context: {e}" # Pass error info

    # --- Invoke Agent Runnable ---
    print(f"[AGENT_EXEC] {datetime.now()}: Invoking main agent runnable for task {task_id}...")
    agent_input = {
        "query": transformed_input,
        "name": username,
        "user_context": user_context,
        "internet_context": internet_context,
        "personality": personality
    }
    # print(f"[AGENT_EXEC] {datetime.now()}: Agent Input for task {task_id}: {agent_input}") # Can be verbose
    try:
        response = agent_runnable.invoke(agent_input)
        print(f"[AGENT_EXEC] {datetime.now()}: Agent response received for task {task_id}.")
        # print(f"[AGENT_EXEC] {datetime.now()}: Agent Response content: {response}") # Log the raw response
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error invoking agent_runnable for task {task_id}: {e}")
        traceback.print_exc()
        return f"Error during agent execution: {e}"

    # --- Process Tool Calls ---
    if "tool_calls" not in response or not isinstance(response["tool_calls"], list):
        error_msg = f"Error: Invalid or missing 'tool_calls' list in agent response for task {task_id}."
        print(f"[AGENT_EXEC] {datetime.now()}: {error_msg} Response was: {response}")
        return error_msg

    all_tool_results = []
    previous_tool_result = None
    print(f"[AGENT_EXEC] {datetime.now()}: Processing {len(response['tool_calls'])} potential tool calls for task {task_id}.")

    for i, tool_call in enumerate(response["tool_calls"]):
        print(f"[AGENT_EXEC] {datetime.now()}: Processing tool call {i+1}/{len(response['tool_calls'])} for task {task_id}...")
        # print(f"[AGENT_EXEC] {datetime.now()}: Tool call data: {tool_call}")

        if not isinstance(tool_call, dict) or tool_call.get("response_type") != "tool_call":
            print(f"[AGENT_EXEC] {datetime.now()}: Skipping item {i+1} as it's not a valid tool call (type: {tool_call.get('response_type', 'N/A')}).")
            continue

        tool_content = tool_call.get("content")
        if not isinstance(tool_content, dict):
             print(f"[AGENT_EXEC] {datetime.now()}: Skipping tool call {i+1} due to invalid 'content' structure.")
             continue

        tool_name = tool_content.get("tool_name")
        task_instruction = tool_content.get("task_instruction")
        previous_tool_response_required = tool_content.get("previous_tool_response", False)

        print(f"[AGENT_EXEC] {datetime.now()}:   - Tool Name: {tool_name}")
        print(f"[AGENT_EXEC] {datetime.now()}:   - Task Instruction: {task_instruction[:100]}...")
        print(f"[AGENT_EXEC] {datetime.now()}:   - Previous Tool Response Required: {previous_tool_response_required}")

        tool_handler = tool_handlers.get(tool_name)
        if not tool_handler:
            error_msg = f"Error: Tool '{tool_name}' not found in registered handlers for task {task_id}."
            print(f"[AGENT_EXEC] {datetime.now()}: {error_msg}")
            # Decide whether to fail the whole task or just skip this tool call
            # Skipping for now, but might need adjustment based on desired behavior
            all_tool_results.append({"tool_name": tool_name, "task_instruction": task_instruction, "tool_result": error_msg, "status": "error"})
            continue # Skip to the next tool call

        # --- Prepare and Execute Tool Handler ---
        tool_input = {"input": task_instruction}
        if previous_tool_response_required:
            if previous_tool_result:
                print(f"[AGENT_EXEC] {datetime.now()}:   - Providing previous tool result to '{tool_name}'.")
                tool_input["previous_tool_response"] = previous_tool_result
            else:
                print(f"[WARN] {datetime.now()}: Tool '{tool_name}' requires previous result, but none is available. Passing 'None'.")
                tool_input["previous_tool_response"] = "Previous tool result was expected but not available." # Or pass None
        else:
            tool_input["previous_tool_response"] = "Not Required"

        print(f"[AGENT_EXEC] {datetime.now()}: Invoking tool handler '{tool_handler.__name__}' for tool '{tool_name}'...")
        try:
            tool_result_main = await tool_handler(tool_input)
            print(f"[AGENT_EXEC] {datetime.now()}: Tool handler '{tool_handler.__name__}' executed.")
            # print(f"[AGENT_EXEC] {datetime.now()}: Tool handler raw result: {tool_result_main}")
        except Exception as e:
            error_msg = f"Error executing tool handler '{tool_handler.__name__}' for tool '{tool_name}': {e}"
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            traceback.print_exc()
            all_tool_results.append({"tool_name": tool_name, "task_instruction": task_instruction, "tool_result": error_msg, "status": "error"})
            previous_tool_result = error_msg # Pass error as previous result if needed
            continue # Skip to next tool call

        # --- Handle Approval Flow ---
        if isinstance(tool_result_main, dict) and tool_result_main.get("action") == "approve":
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires approval for tool '{tool_name}'. Setting task to pending.")
            approval_data = tool_result_main.get("tool_call", {}) # Get the data needing approval
            await task_queue.set_task_approval_pending(task_id, approval_data)

            # --- WebSocket Notification for Approval ---
            notification = {
                "type": "task_approval_pending",
                "task_id": task_id,
                "description": f"Approval needed for: {tool_name} - {task_instruction[:50]}...", # Provide context
                "tool_name": tool_name,
                "approval_data": approval_data # Send data needing approval to frontend
            }
            print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task approval pending for {task_id}")
            await manager.broadcast(json.dumps(notification))
            return "Task requires approval." # Signal that execution is paused

        # --- Store Normal Tool Result ---
        else:
            # Extract the actual result, handling potential variations in the handler's return format
            if isinstance(tool_result_main, dict) and "tool_result" in tool_result_main:
                tool_result = tool_result_main["tool_result"]
            else:
                # Assume the handler returned the result directly
                tool_result = tool_result_main

            print(f"[AGENT_EXEC] {datetime.now()}: Tool '{tool_name}' executed successfully. Storing result.")
            previous_tool_result = tool_result # Store for potential use by the next tool
            all_tool_results.append({
                "tool_name": tool_name,
                "task_instruction": task_instruction,
                "tool_result": tool_result,
                "status": "success"
            })

    # --- Final Reflection/Summarization ---
    if not all_tool_results:
        print(f"[AGENT_EXEC] {datetime.now()}: No successful tool calls executed for task {task_id}. Returning empty result string.")
        # Maybe return a message indicating no tools were run or check agent's initial response?
        # For now, returning a generic message based on the agent's initial non-tool response might be better.
        # Let's check if the initial response had a direct answer before tool calls.
        if response.get("response_type") == "final_answer" and response.get("content"):
            print(f"[AGENT_EXEC] {datetime.now()}: Using agent's final answer as no tools were executed.")
            return response["content"]
        else:
            print(f"[AGENT_EXEC] {datetime.now()}: No tools run and no direct final answer from agent for task {task_id}. Returning generic message.")
            return "No specific actions were taken or information gathered."


    print(f"[AGENT_EXEC] {datetime.now()}: All tool calls processed for task {task_id}. Preparing final result.")
    final_result_str = "No final result generated." # Default value

    try:
        # Special handling for inbox search summarization
        if len(all_tool_results) == 1 and all_tool_results[0].get("tool_name") == "search_inbox" and all_tool_results[0].get("status") == "success":
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} involved only 'search_inbox'. Invoking inbox summarizer...")
            tool_result_data = all_tool_results[0]["tool_result"]
            # Ensure the result structure is as expected by the summarizer
            if isinstance(tool_result_data, dict) and "result" in tool_result_data and isinstance(tool_result_data["result"], dict):
                result_content = tool_result_data["result"]
                # Filter email data, keeping only non-body fields
                filtered_email_data = []
                if "email_data" in result_content and isinstance(result_content["email_data"], list):
                    filtered_email_data = [
                        {k: email[k] for k in email if k != "body"}
                        for email in result_content["email_data"] if isinstance(email, dict)
                    ]

                filtered_tool_result = {
                    "response": result_content.get("response", "No summary available."),
                    "email_data": filtered_email_data,
                    "gmail_search_url": result_content.get("gmail_search_url", "URL not available.")
                }
                # print(f"[AGENT_EXEC] {datetime.now()}: Filtered inbox data for summarizer: {filtered_tool_result}")
                final_result_str = inbox_summarizer_runnable.invoke({"tool_result": filtered_tool_result})
                print(f"[AGENT_EXEC] {datetime.now()}: Inbox summarizer finished for task {task_id}.")
            else:
                print(f"[WARN] {datetime.now()}: 'search_inbox' result format unexpected for summarization. Falling back to reflection. Result: {tool_result_data}")
                print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires reflection on tool results.")
                final_result_str = reflection_runnable.invoke({"tool_results": all_tool_results})
                print(f"[AGENT_EXEC] {datetime.now()}: Reflection finished for task {task_id}.")
        else:
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires reflection on multiple/different tool results.")
            # print(f"[AGENT_EXEC] {datetime.now()}: Input to reflection: {all_tool_results}") # Can be verbose
            final_result_str = reflection_runnable.invoke({"tool_results": all_tool_results})
            print(f"[AGENT_EXEC] {datetime.now()}: Reflection finished for task {task_id}.")

    except Exception as e:
        error_msg = f"Error during final result generation (reflection/summarization) for task {task_id}: {e}"
        print(f"[ERROR] {datetime.now()}: {error_msg}")
        traceback.print_exc()
        final_result_str = f"{error_msg}\n\nRaw Tool Results:\n{all_tool_results}" # Return error and raw results

    print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} execution complete. Final result length: {len(final_result_str)}")
    return final_result_str


async def add_result_to_chat(chat_id: str, result: str, isUser: bool, task_description: str = None):
    """Add the task result or hidden user message to the corresponding chat."""
    # print(f"[CHAT_UPDATE] {datetime.now()}: Adding message to chat_id '{chat_id}'. IsUser: {isUser}, Task Desc (if any): {task_description[:50] if task_description else 'N/A'}")
    async with db_lock:
        # print(f"[CHAT_UPDATE] {datetime.now()}: Acquired chat DB lock for '{chat_id}'.")
        chatsDb = await load_db()
        chat = next((c for c in chatsDb["chats"] if c.get("id") == chat_id), None)

        if chat:
            message_id = str(int(time.time() * 1000)) # Generate unique ID
            timestamp = datetime.now(timezone.utc).isoformat() + "Z" # Use alias dt

            if not isUser:
                # This is an assistant message, likely a tool result
                result_message = {
                    "id": message_id,
                    "type": "tool_result", # Indicate it's from a tool/agent task
                    "message": result,
                    "task": task_description, # Include the original task description
                    "isUser": False,
                    "memoryUsed": False, # Context about memory/internet usage isn't directly available here
                    "agentsUsed": True,  # Assume agent was used if it's a tool result
                    "internetUsed": False, # Context about memory/internet usage isn't directly available here
                    "timestamp": timestamp,
                    "isVisible": True # Tool results should be visible
                }
                # print(f"[CHAT_UPDATE] {datetime.now()}: Creating assistant message (tool result) for chat '{chat_id}'.")
            else:
                # This is a user message, likely the original task description being hidden
                result_message = {
                    "id": message_id,
                    "type": "user_message", # Standard user message type
                    "message": result, # This 'result' is the task description here
                    "isUser": True,
                    "isVisible": False, # Make this message hidden
                    "memoryUsed": False,
                    "agentsUsed": False,
                    "internetUsed": False,
                    "timestamp": timestamp
                }
                # print(f"[CHAT_UPDATE] {datetime.now()}: Creating hidden user message (task description) for chat '{chat_id}'.")

            # Append the message
            if "messages" not in chat:
                chat["messages"] = []
            chat["messages"].append(result_message)
            await save_db(chatsDb)
            # print(f"[CHAT_UPDATE] {datetime.now()}: Message (ID: {message_id}) added to chat '{chat_id}' and DB saved.")
        else:
            print(f"[ERROR] {datetime.now()}: Failed to add message. Chat with ID '{chat_id}' not found.")
        # print(f"[CHAT_UPDATE] {datetime.now()}: Releasing chat DB lock for '{chat_id}'.")


# --- FastAPI Application Setup ---
print(f"[FASTAPI] {datetime.now()}: Initializing FastAPI app...")
app = FastAPI(
    title="Sentient API",
    description="Monolithic API for the Sentient AI companion",
    docs_url="/docs",
    redoc_url=None # Disable Redoc if not needed
)
print(f"[FASTAPI] {datetime.now()}: FastAPI app initialized.")

# Add CORS middleware to allow cross-origin requests
print(f"[FASTAPI] {datetime.now()}: Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"]  # Allow all headers
)
print(f"[FASTAPI] {datetime.now()}: CORS middleware added.")

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Handles application startup procedures."""
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Application startup event triggered.")
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Loading tasks from storage...")
    await task_queue.load_tasks()
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Loading memory operations from storage...")
    await memory_backend.memory_queue.load_operations()

    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Creating background task for processing task queue...")
    asyncio.create_task(process_queue())
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Creating background task for processing memory operations...")
    asyncio.create_task(process_memory_operations())
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Creating background task for periodic task cleanup...")
    asyncio.create_task(cleanup_tasks_periodically())

    # Initialize and start context engines based on user profile settings
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Initializing context engines based on user profile...")
    user_id = "user1" # TODO: Replace with dynamic user ID retrieval if needed
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Using placeholder user_id: {user_id} for context engines.")
    user_profile = load_user_profile()
    enabled_data_sources = []

    # Check which data sources are enabled in the profile (default to True if key missing)
    if user_profile.get("userData", {}).get("gmailEnabled", True):
        enabled_data_sources.append("gmail")
    if user_profile.get("userData", {}).get("internetSearchEnabled", True):
        enabled_data_sources.append("internet_search")
    if user_profile.get("userData", {}).get("gcalendarEnabled", True):
        enabled_data_sources.append("gcalendar")

    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Enabled data sources for context engines: {enabled_data_sources}")

    for source in enabled_data_sources:
        engine = None
        print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Setting up context engine for source: {source}")
        try:
            if source == "gmail":
                engine = GmailContextEngine(user_id, task_queue, memory_backend, manager, db_lock, notifications_db_lock)
            elif source == "internet_search":
                 # Internet Search engine might not have a continuous process like Gmail/GCalendar
                 # Adjust if it needs a long-running task
                # engine = InternetSearchContextEngine(user_id, task_queue, memory_backend, manager, db_lock, notifications_db_lock)
                print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: InternetSearchContextEngine currently does not require a background task.")
                continue # Skip starting a task for this one for now
            elif source == "gcalendar":
                engine = GCalendarContextEngine(user_id, task_queue, memory_backend, manager, db_lock, notifications_db_lock)
            else:
                print(f"[WARN] {datetime.now()}: Unknown data source '{source}' encountered during context engine setup.")
                continue

            if engine:
                print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Starting background task for {source} context engine...")
                asyncio.create_task(engine.start())
            else:
                print(f"[WARN] {datetime.now()}: Failed to initialize engine for {source}.")


        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Failed to initialize or start context engine for {source}: {e}")
            traceback.print_exc()

    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Handles application shutdown procedures."""
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Application shutdown event triggered.")
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Saving pending tasks to storage...")
    await task_queue.save_tasks()
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Saving pending memory operations to storage...")
    await memory_backend.memory_queue.save_operations()
    # Close Neo4j driver if it was initialized
    if graph_driver:
        print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Closing Neo4j driver connection...")
        graph_driver.close()
        print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Neo4j driver closed.")
    print(f"[FASTAPI_LIFECYCLE] {datetime.now()}: Application shutdown complete.")

# --- Pydantic Models ---
# (No print statements needed in model definitions)
class Message(BaseModel):
    input: str
    pricing: str
    credits: int
    chat_id: Optional[str] = None # Make chat_id optional, will be determined if None

class ToolCall(BaseModel):
    input: str
    previous_tool_response: Optional[Any] = None

class ElaboratorMessage(BaseModel):
    input: str
    purpose: str

class EncryptionRequest(BaseModel):
    data: str

class DecryptionRequest(BaseModel):
    encrypted_data: str

class UserInfoRequest(BaseModel):
    user_id: str

class ReferrerStatusRequest(BaseModel):
    user_id: str
    referrer_status: bool

class BetaUserStatusRequest(BaseModel):
    user_id: str
    beta_user_status: bool

class SetReferrerRequest(BaseModel):
    referral_code: str

class DeleteSubgraphRequest(BaseModel):
    source: str

class GraphRequest(BaseModel):
    information: str

class GraphRAGRequest(BaseModel):
    query: str

class RedditURL(BaseModel):
    url: str

class TwitterURL(BaseModel):
    url: str

class LinkedInURL(BaseModel):
    url: str

class SetDataSourceEnabledRequest(BaseModel):
    source: str
    enabled: bool

class CreateTaskRequest(BaseModel):
    # chat_id: str # Removed, will be determined dynamically
    description: str
    # priority: int # Removed, will be determined dynamically
    # username: str # Removed, will be determined dynamically
    # personality: Union[Dict, str, None] # Removed, will be determined dynamically
    # use_personal_context: bool # Removed, will be determined dynamically
    # internet: str # Removed, will be determined dynamically

class UpdateTaskRequest(BaseModel):
    task_id: str
    description: str
    priority: int

class DeleteTaskRequest(BaseModel):
    task_id: str

class GetShortTermMemoriesRequest(BaseModel):
    user_id: str
    category: str
    limit: int

class UpdateUserDataRequest(BaseModel):
    data: Dict[str, Any]

class AddUserDataRequest(BaseModel):
    data: Dict[str, Any]

class AddMemoryRequest(BaseModel):
    user_id: str
    text: str
    category: str
    retention_days: int

class UpdateMemoryRequest(BaseModel):
    user_id: str
    category: str
    id: int # Assuming memory ID is an int
    text: str
    retention_days: int

class DeleteMemoryRequest(BaseModel):
    user_id: str
    category: str
    id: int # Assuming memory ID is an int

class TaskIdRequest(BaseModel):
    """Request model containing just a task ID."""
    task_id: str

# Define response models for clarity and validation (Optional but good practice)
class TaskApprovalDataResponse(BaseModel):
    approval_data: Optional[Dict[str, Any]] = None

class ApproveTaskResponse(BaseModel):
    message: str
    result: Any # Or be more specific if the result type is known

# --- API Endpoints ---

## Root Endpoint
@app.get("/", status_code=200)
async def main():
    """Root endpoint providing a welcome message."""
    print(f"[ENDPOINT /] {datetime.now()}: Root endpoint called.")
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }

@app.get("/get-history", status_code=200)
async def get_history():
    """
    Endpoint to retrieve the chat history. Calls the get_chat_history_messages function.
    """
    print(f"[ENDPOINT /get-history] {datetime.now()}: Endpoint called.")
    try:
        messages = await get_chat_history_messages()
        # print(f"[ENDPOINT /get-history] {datetime.now()}: Retrieved {len(messages)} messages.")
        return JSONResponse(status_code=200, content={"messages": messages})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /get-history: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")


@app.post("/clear-chat-history", status_code=200)
async def clear_chat_history():
    """Clear all chat history by resetting to the initial database structure."""
    print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Endpoint called.")
    async with db_lock:
        print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Acquired chat DB lock.")
        try:
            chatsDb = initial_db.copy() # Reset to initial state
            await save_db(chatsDb)
            print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Chat database reset and saved.")

            # Clear in-memory history components as well
            chat_runnable.clear_history()
            agent_runnable.clear_history()
            unified_classification_runnable.clear_history()
            print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: In-memory chat histories cleared.")

            print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Releasing chat DB lock.")
            return JSONResponse(status_code=200, content={"message": "Chat history cleared"})
        except Exception as e:
             print(f"[ERROR] {datetime.now()}: Error in /clear-chat-history: {e}")
             traceback.print_exc()
             # Ensure lock is released even on error
             print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Releasing chat DB lock due to error.")
             raise HTTPException(status_code=500, detail="Failed to clear chat history.")

@app.post("/chat", status_code=200)
async def chat(message: Message):
    """Handles incoming chat messages, classifies, and responds via streaming."""
    endpoint_start_time = time.time()
    print(f"[ENDPOINT /chat] {datetime.now()}: Endpoint called.")
    print(f"[ENDPOINT /chat] {datetime.now()}: Incoming message data: Input='{message.input[:50]}...', Pricing='{message.pricing}', Credits={message.credits}, ChatID='{message.chat_id}'")

    # Ensure global variables are accessible if modified or reassigned inside
    global embed_model, chat_runnable, fact_extraction_runnable, text_conversion_runnable
    global information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable
    global query_classification_runnable, agent_runnable, text_description_runnable
    global reflection_runnable, internet_query_reframe_runnable, internet_summary_runnable, priority_runnable
    global unified_classification_runnable, memory_backend

    try:
        # --- Load User Profile ---
        print(f"[ENDPOINT /chat] {datetime.now()}: Loading user profile...")
        user_profile_data = load_user_profile()
        if not user_profile_data or "userData" not in user_profile_data:
            print(f"[ERROR] {datetime.now()}: Failed to load valid user profile data.")
            raise HTTPException(status_code=500, detail="User profile could not be loaded.")
        db = user_profile_data # Use loaded data
        username = db.get("userData", {}).get("personalInfo", {}).get("name", "User") # Safer access
        print(f"[ENDPOINT /chat] {datetime.now()}: User profile loaded for username: {username}")

        # --- Determine Active Chat ---
        active_chat_id = message.chat_id
        if not active_chat_id:
            print(f"[ENDPOINT /chat] {datetime.now()}: No chat_id provided in request, determining active chat...")
            async with db_lock:
                chatsDb = await load_db()
                active_chat_id = chatsDb.get("active_chat_id")
                if not active_chat_id:
                     # If still no active chat, trigger history logic which creates one
                     print(f"[ENDPOINT /chat] {datetime.now()}: No active chat in DB, triggering history logic to potentially create one.")
                     await get_chat_history_messages() # This ensures a chat exists
                     chatsDb = await load_db() # Reload DB after potential creation
                     active_chat_id = chatsDb.get("active_chat_id")
                     if not active_chat_id:
                         print(f"[ERROR] {datetime.now()}: Failed to determine or create an active chat ID.")
                         raise HTTPException(status_code=500, detail="Could not determine active chat.")
            print(f"[ENDPOINT /chat] {datetime.now()}: Determined active chat ID: {active_chat_id}")
        else:
             print(f"[ENDPOINT /chat] {datetime.now()}: Using provided chat ID: {active_chat_id}")


        # --- Ensure Runnables use Correct History ---
        # This might be redundant if history is managed globally correctly, but good for clarity
        print(f"[ENDPOINT /chat] {datetime.now()}: Ensuring runnables use current history context...")
        current_chat_history = get_chat_history() # Ensure it reflects the latest state if needed
        # Re-initialize runnables if their history scope needs explicit update per request (depends on LangChain implementation)
        # Assuming get_chat_history() manages the state correctly for now.
        # chat_runnable = get_chat_runnable(current_chat_history)
        # agent_runnable = get_agent_runnable(current_chat_history)
        # unified_classification_runnable = get_unified_classification_runnable(current_chat_history)
        print(f"[ENDPOINT /chat] {datetime.now()}: Runnables ready.")

        # --- Unified Classification ---
        print(f"[ENDPOINT /chat] {datetime.now()}: Performing unified classification for input: '{message.input[:50]}...'")
        unified_output = unified_classification_runnable.invoke({"query": message.input})
        print(f"[ENDPOINT /chat] {datetime.now()}: Unified classification output: {unified_output}")
        category = unified_output.get("category", "chat") # Default to chat if missing
        use_personal_context = unified_output.get("use_personal_context", False)
        internet = unified_output.get("internet", "None")
        transformed_input = unified_output.get("transformed_input", message.input) # Use original if missing

        pricing_plan = message.pricing
        credits = message.credits

        # --- Streaming Response Generator ---
        async def response_generator():
            stream_start_time = time.time()
            print(f"[STREAM /chat] {datetime.now()}: Starting response generation stream for chat {active_chat_id}.")
            memory_used = False
            agents_used = False
            internet_used = False
            user_context = None
            internet_context = None
            pro_used = False # Tracks if a pro feature (memory update, internet) was successfully used
            note = "" # For credit limit messages

            # 1. Add User Message to DB
            user_msg_id = str(int(time.time() * 1000))
            user_msg = {
                "id": user_msg_id,
                "message": message.input,
                "isUser": True,
                "memoryUsed": False, # User message doesn't use these directly
                "agentsUsed": False,
                "internetUsed": False,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "isVisible": True # User messages are visible
            }
            async with db_lock:
                # print(f"[STREAM /chat] {datetime.now()}: Acquired chat DB lock to add user message.")
                chatsDb = await load_db()
                # Find the correct chat to add the message to
                active_chat_obj = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)
                if active_chat_obj:
                    if "messages" not in active_chat_obj: active_chat_obj["messages"] = []
                    active_chat_obj["messages"].append(user_msg)
                    await save_db(chatsDb)
                    # print(f"[STREAM /chat] {datetime.now()}: User message (ID: {user_msg_id}) added to chat {active_chat_id} in DB.")
                else:
                    print(f"[ERROR] {datetime.now()}: Could not find active chat {active_chat_id} in DB to add user message.")
                # print(f"[STREAM /chat] {datetime.now()}: Released chat DB lock.")

            # 2. Yield User Message Confirmation to Client
            print(f"[STREAM /chat] {datetime.now()}: Yielding user message confirmation.")
            yield json.dumps({
                "type": "userMessage",
                "message": message.input,
                "id": user_msg_id, # Send back the ID
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False,
                 "timestamp": user_msg["timestamp"] # Include timestamp
            }) + "\n"
            await asyncio.sleep(0.01) # Small delay

            # 3. Prepare Assistant Message Structure (will be filled and updated)
            assistant_msg_id = str(int(time.time() * 1000))
            assistant_msg = {
                "id": assistant_msg_id,
                "message": "",
                "isUser": False,
                "memoryUsed": False, # Will be updated
                "agentsUsed": False, # Will be updated
                "internetUsed": False, # Will be updated
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z", # Initial timestamp
                "isVisible": True
            }
            print(f"[STREAM /chat] {datetime.now()}: Prepared initial assistant message structure (ID: {assistant_msg_id}).")

            # --- Handle Agent Category (Task Creation) ---
            if category == "agent":
                print(f"[STREAM /chat] {datetime.now()}: Category is 'agent'. Preparing to add task to queue.")
                agents_used = True # Mark agent as used
                assistant_msg["agentsUsed"] = True
                personality_description = db.get("userData", {}).get("personality", "Default helpful assistant") # Get personality
                print(f"[STREAM /chat] {datetime.now()}: Determining task priority for: '{transformed_input[:50]}...'")
                try:
                    priority_response = priority_runnable.invoke({"task_description": transformed_input})
                    priority = priority_response.get("priority", 3) # Default priority if parse fails
                    print(f"[STREAM /chat] {datetime.now()}: Determined task priority: {priority}")
                except Exception as e:
                    print(f"[ERROR] {datetime.now()}: Failed to determine priority: {e}. Using default (3).")
                    priority = 3

                print(f"[STREAM /chat] {datetime.now()}: Adding task to queue...")
                try:
                    await task_queue.add_task(
                        chat_id=active_chat_id, # Use the determined active chat ID
                        description=transformed_input,
                        priority=priority,
                        username=username,
                        personality=personality_description,
                        use_personal_context=use_personal_context,
                        internet=internet
                    )
                    print(f"[STREAM /chat] {datetime.now()}: Task added to queue successfully.")
                    assistant_msg["message"] = "Got it! I'll work on that task for you." # Confirmation message
                except Exception as e:
                    print(f"[ERROR] {datetime.now()}: Failed to add task to queue: {e}")
                    traceback.print_exc()
                    assistant_msg["message"] = "Sorry, I encountered an error trying to schedule that task." # Error message

                # Add agent confirmation/error message to DB
                async with db_lock:
                    # print(f"[STREAM /chat] {datetime.now()}: Acquired chat DB lock to add agent confirmation message.")
                    chatsDb = await load_db()
                    active_chat_obj = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)
                    if active_chat_obj:
                        if "messages" not in active_chat_obj: active_chat_obj["messages"] = []
                        # Update timestamp just before saving
                        assistant_msg["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
                        active_chat_obj["messages"].append(assistant_msg)
                        await save_db(chatsDb)
                        # print(f"[STREAM /chat] {datetime.now()}: Agent confirmation message (ID: {assistant_msg_id}) added to chat {active_chat_id} in DB.")
                    else:
                        print(f"[ERROR] {datetime.now()}: Could not find active chat {active_chat_id} in DB to add agent confirmation.")
                    # print(f"[STREAM /chat] {datetime.now()}: Released chat DB lock.")

                # Yield final agent confirmation/error message
                print(f"[STREAM /chat] {datetime.now()}: Yielding final agent confirmation/error message.")
                yield json.dumps({
                    "type": "assistantMessage", # Final message, not stream
                    "message": assistant_msg["message"],
                    "id": assistant_msg_id,
                    "memoryUsed": memory_used, # False for agent category start
                    "agentsUsed": agents_used, # True for agent category start
                    "internetUsed": internet_used, # False for agent category start
                    "proUsed": pro_used,
                    "timestamp": assistant_msg["timestamp"]
                }) + "\n"
                await asyncio.sleep(0.01)
                print(f"[STREAM /chat] {datetime.now()}: Agent task creation flow finished.")
                return # End stream for agent category

            # --- Handle Memory/Context Retrieval ---
            if category == "memory" or use_personal_context:
                print(f"[STREAM /chat] {datetime.now()}: Category requires memory/personal context ({category=}, {use_personal_context=}).")
                if category == "memory" and pricing_plan == "free" and credits <= 0:
                    print(f"[STREAM /chat] {datetime.now()}: Memory category but free plan credits exhausted. Skipping memory update, retrieving only.")
                    note = "Sorry friend, memory updates are a pro feature and your daily credits have expired. Upgrade to pro in settings!"
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories (read-only)...", "id": assistant_msg_id}) + "\n"
                    memory_used = True # Still retrieving
                    user_context = memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Memory retrieved (read-only). Context length: {len(str(user_context)) if user_context else 0}")
                elif category == "memory": # Pro or has credits
                    print(f"[STREAM /chat] {datetime.now()}: Memory category with credits/pro plan. Retrieving and queueing update.")
                    yield json.dumps({"type": "intermediary", "message": "Retrieving and updating memories...", "id": assistant_msg_id}) + "\n"
                    memory_used = True
                    pro_used = True # Memory update is a pro feature use
                    # Retrieve existing memories first
                    user_context = memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Memory retrieved. Context length: {len(str(user_context)) if user_context else 0}")
                    # Queue memory update in the background
                    print(f"[STREAM /chat] {datetime.now()}: Queueing memory update operation for user '{username}'.")
                    asyncio.create_task(memory_backend.add_operation(username, transformed_input))
                else: # Just use_personal_context (not explicitly 'memory' category)
                    print(f"[STREAM /chat] {datetime.now()}: Retrieving personal context (not memory category).")
                    yield json.dumps({"type": "intermediary", "message": "Retrieving relevant context...", "id": assistant_msg_id}) + "\n"
                    memory_used = True # Mark as used context
                    user_context = memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Personal context retrieved. Context length: {len(str(user_context)) if user_context else 0}")
                assistant_msg["memoryUsed"] = memory_used # Update status

            # --- Handle Internet Search ---
            if internet == "Internet":
                print(f"[STREAM /chat] {datetime.now()}: Internet search required.")
                if pricing_plan == "free" and credits <= 0:
                    print(f"[STREAM /chat] {datetime.now()}: Internet search required but free plan credits exhausted. Skipping.")
                    note += " Sorry friend, could have searched the internet for more context, but your daily credits have expired. You can always upgrade to pro from the settings page"
                else:
                    print(f"[STREAM /chat] {datetime.now()}: Performing internet search (pro/credits available).")
                    yield json.dumps({"type": "intermediary", "message": "Searching the internet...", "id": assistant_msg_id}) + "\n"
                    try:
                        reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                        print(f"[STREAM /chat] {datetime.now()}: Internet query reframed: '{reframed_query}'")
                        search_results = get_search_results(reframed_query)
                        # print(f"[STREAM /chat] {datetime.now()}: Internet search results: {search_results}") # Can be verbose
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        print(f"[STREAM /chat] {datetime.now()}: Internet search summary generated. Length: {len(str(internet_context)) if internet_context else 0}")
                        internet_used = True
                        pro_used = True # Internet search is a pro feature use
                    except Exception as e:
                        print(f"[ERROR] {datetime.now()}: Error during internet search: {e}")
                        traceback.print_exc()
                        internet_context = f"Error searching internet: {e}" # Add error to context
                assistant_msg["internetUsed"] = internet_used # Update status

            # --- Handle Chat and Memory Categories (Generate Response) ---
            if category in ["chat", "memory"]:
                print(f"[STREAM /chat] {datetime.now()}: Category is '{category}'. Generating chat response...")
                personality_description = db.get("userData", {}).get("personality", "Default helpful assistant")
                print(f"[STREAM /chat] {datetime.now()}: Using personality: '{personality_description[:50]}...'")

                # Prepare input for the chat runnable
                chat_inputs = {
                    "query": transformed_input,
                    "user_context": user_context,
                    "internet_context": internet_context,
                    "name": username,
                    "personality": personality_description
                }
                # print(f"[STREAM /chat] {datetime.now()}: Input to chat_runnable: {chat_inputs}") # Can be verbose

                # Add placeholder assistant message to DB before streaming starts
                async with db_lock:
                    # print(f"[STREAM /chat] {datetime.now()}: Acquired chat DB lock to add placeholder assistant message.")
                    chatsDb = await load_db()
                    active_chat_obj = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)
                    if active_chat_obj:
                         if "messages" not in active_chat_obj: active_chat_obj["messages"] = []
                         # Ensure flags are set correctly before initial save
                         assistant_msg["memoryUsed"] = memory_used
                         assistant_msg["internetUsed"] = internet_used
                         assistant_msg["agentsUsed"] = agents_used # Should be False here
                         assistant_msg["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
                         active_chat_obj["messages"].append(assistant_msg.copy()) # Add a copy initially
                         await save_db(chatsDb)
                        #  print(f"[STREAM /chat] {datetime.now()}: Placeholder assistant message (ID: {assistant_msg_id}) added to chat {active_chat_id} in DB.")
                    else:
                         print(f"[ERROR] {datetime.now()}: Could not find active chat {active_chat_id} in DB to add placeholder.")
                    # print(f"[STREAM /chat] {datetime.now()}: Released chat DB lock.")


                # Stream the response
                print(f"[STREAM /chat] {datetime.now()}: Starting LLM stream generation...")
                full_response = ""
                try:
                    async for token in generate_streaming_response(
                        chat_runnable,
                        inputs=chat_inputs,
                        stream=True # Explicitly request streaming
                    ):
                        if isinstance(token, str):
                            full_response += token
                            # Yield the token to the client
                            yield json.dumps({
                                "type": "assistantStream",
                                "token": token,
                                "done": False,
                                "messageId": assistant_msg_id # Link token to message ID
                            }) + "\n"
                            await asyncio.sleep(0.01) # Small delay between tokens
                        else:
                            # End of stream signal or other object (handle potential errors/metadata if LangChain changes)
                            # print(f"[STREAM /chat] {datetime.now()}: Received non-string token (end of stream?): {token}")
                            pass # Assume end of stream for now
                except Exception as e:
                    print(f"[ERROR] {datetime.now()}: Error during LLM stream generation: {e}")
                    traceback.print_exc()
                    full_response += f"\n\n[Error generating response: {e}]" # Append error to output

                print(f"[STREAM /chat] {datetime.now()}: LLM stream generation finished. Full response length: {len(full_response)}")

                # Append notes if any
                if note:
                    full_response += "\n\n" + note.strip()
                    print(f"[STREAM /chat] {datetime.now()}: Appended note: '{note.strip()[:50]}...'")

                # Update the final assistant message in the DB
                assistant_msg["message"] = full_response
                assistant_msg["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z" # Final timestamp

                async with db_lock:
                    # print(f"[STREAM /chat] {datetime.now()}: Acquired chat DB lock to update final assistant message.")
                    chatsDb = await load_db()
                    active_chat_obj = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)
                    if active_chat_obj and "messages" in active_chat_obj:
                        # Find the message by ID and update it
                        message_updated = False
                        for i, msg in enumerate(active_chat_obj["messages"]):
                            if msg.get("id") == assistant_msg_id:
                                active_chat_obj["messages"][i] = assistant_msg.copy() # Update with final data
                                message_updated = True
                                break
                        if message_updated:
                           await save_db(chatsDb)
                        #    print(f"[STREAM /chat] {datetime.now()}: Final assistant message (ID: {assistant_msg_id}) updated in chat {active_chat_id} DB.")
                        else:
                            # This shouldn't happen if placeholder was added correctly
                            print(f"[ERROR] {datetime.now()}: Could not find message ID {assistant_msg_id} to update in chat {active_chat_id}.")
                            # Optionally append if not found?
                            # active_chat_obj["messages"].append(assistant_msg.copy())
                            # await save_db(chatsDb)

                    else:
                         print(f"[ERROR] {datetime.now()}: Could not find active chat {active_chat_id} or messages list in DB to update.")
                    # print(f"[STREAM /chat] {datetime.now()}: Released chat DB lock.")


                # Yield the final "done" signal with all metadata
                print(f"[STREAM /chat] {datetime.now()}: Yielding final 'done' signal.")
                yield json.dumps({
                    "type": "assistantStream",
                    "token": "", # No more tokens
                    "done": True,
                    "memoryUsed": memory_used,
                    "agentsUsed": agents_used,
                    "internetUsed": internet_used,
                    "proUsed": pro_used,
                    "messageId": assistant_msg_id # Link to the completed message
                }) + "\n"
                await asyncio.sleep(0.01)

            stream_duration = time.time() - stream_start_time
            print(f"[STREAM /chat] {datetime.now()}: Response generation stream finished for chat {active_chat_id}. Duration: {stream_duration:.2f}s")

        # Return the streaming response
        print(f"[ENDPOINT /chat] {datetime.now()}: Returning StreamingResponse.")
        return StreamingResponse(response_generator(), media_type="application/x-ndjson") # Use ndjson

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        print(f"[ERROR] {datetime.now()}: HTTPException in /chat: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[ERROR] {datetime.now()}: Unexpected error in /chat endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An internal server error occurred: {str(e)}"})
    finally:
        endpoint_duration = time.time() - endpoint_start_time
        print(f"[ENDPOINT /chat] {datetime.now()}: Endpoint execution finished. Duration: {endpoint_duration:.2f}s")

## Agents Endpoints
@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage):
    """Elaborates on an input string based on a specified purpose."""
    print(f"[ENDPOINT /elaborator] {datetime.now()}: Endpoint called.")
    print(f"[ENDPOINT /elaborator] {datetime.now()}: Input: '{message.input[:50]}...', Purpose: '{message.purpose}'")
    try:
        # Initialize runnable within the endpoint or ensure it's globally available
        # Assuming get_tool_runnable is lightweight or already cached
        elaborator_runnable = get_tool_runnable(
            elaborator_system_prompt_template,
            elaborator_user_prompt_template,
            None, # No specific format needed for simple elaboration?
            ["query", "purpose"]
        )
        print(f"[ENDPOINT /elaborator] {datetime.now()}: Elaborator runnable obtained.")
        # print(f"[ENDPOINT /elaborator] {datetime.now()}: Elaborator runnable details: {elaborator_runnable}") # Can be verbose

        output = elaborator_runnable.invoke({"query": message.input, "purpose": message.purpose})
        print(f"[ENDPOINT /elaborator] {datetime.now()}: Elaboration generated. Output length: {len(output)}")
        # print(f"[ENDPOINT /elaborator] {datetime.now()}: Elaborator output: {output}") # Log full output if needed

        return JSONResponse(status_code=200, content={"message": output})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /elaborator: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Elaboration failed: {str(e)}"})

## Tool Handlers (Registered via decorator)
# Note: These handlers are called internally by execute_agent_task, not directly via HTTP

@register_tool("gmail")
async def gmail_tool(tool_call_input: dict) -> Dict[str, Any]: # Renamed input var
    """Handles Gmail-related tasks with approval for send_email and reply_email."""
    tool_name = "gmail"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        # Load username dynamically within the handler
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Using username: {username}")

        # Get or initialize the specific tool runnable
        tool_runnable = get_tool_runnable(
            gmail_agent_system_prompt_template,
            gmail_agent_user_prompt_template,
            gmail_agent_required_format,
            ["query", "username", "previous_tool_response"]
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Gmail tool runnable obtained.")

        # Invoke the runnable to get the specific tool call details (like 'send_email')
        tool_call_str = tool_runnable.invoke({
            "query": tool_call_input["input"],
            "username": username,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided")
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        try:
            tool_call_dict = json.loads(tool_call_str)
            actual_tool_name = tool_call_dict.get("tool_name")
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Parsed tool call: {tool_call_dict}")
        except json.JSONDecodeError as json_e:
            error_msg = f"Failed to parse JSON response from tool runnable: {json_e}. Response was: {tool_call_str}"
            print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
            return {"status": "failure", "error": error_msg}
        except Exception as e: # Catch other potential errors during parsing/access
            error_msg = f"Error processing tool runnable response: {e}. Response was: {tool_call_str}"
            print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
            return {"status": "failure", "error": error_msg}


        # Check for approval requirement
        if actual_tool_name in ["send_email", "reply_email"]:
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Action '{actual_tool_name}' requires approval. Returning approval request.")
            return {"action": "approve", "tool_call": tool_call_dict}
        else:
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Action '{actual_tool_name}' does not require approval. Executing directly.")
            # Execute the tool call (e.g., search_inbox, get_drafts)
            tool_result = await parse_and_execute_tool_calls(tool_call_str) # Assuming this handles the actual execution
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
            # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")
            return {"tool_result": tool_result, "tool_call_str": tool_call_str} # Return result and original call str

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}

@register_tool("gdrive")
async def drive_tool(tool_call_input: dict) -> Dict[str, Any]:
    """Handles Google Drive interactions."""
    tool_name = "gdrive"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        tool_runnable = get_tool_runnable(
            gdrive_agent_system_prompt_template,
            gdrive_agent_user_prompt_template,
            gdrive_agent_required_format,
            ["query", "previous_tool_response"]
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GDrive tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
            "query": tool_call_input["input"],
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided")
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
        # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")

        # GDrive usually doesn't need approval, return result directly
        return {"tool_result": tool_result, "tool_call_str": None} # No need to return call str if not needed downstream

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}


@register_tool("gdocs")
async def gdoc_tool(tool_call_input: dict) -> Dict[str, Any]:
    """Handles Google Docs creation and text elaboration."""
    tool_name = "gdocs"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        tool_runnable = get_tool_runnable(
            gdocs_agent_system_prompt_template,
            gdocs_agent_user_prompt_template,
            gdocs_agent_required_format,
            ["query", "previous_tool_response"],
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GDocs tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
            "query": tool_call_input["input"],
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
        # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")

        return {"tool_result": tool_result, "tool_call_str": None}

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}

@register_tool("gsheets")
async def gsheet_tool(tool_call_input: dict) -> Dict[str, Any]:
    """Handles Google Sheets creation and data population."""
    tool_name = "gsheets"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        tool_runnable = get_tool_runnable(
            gsheets_agent_system_prompt_template,
            gsheets_agent_user_prompt_template,
            gsheets_agent_required_format,
            ["query", "previous_tool_response"],
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GSheets tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
             "query": tool_call_input["input"],
             "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
        # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")

        return {"tool_result": tool_result, "tool_call_str": None}

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}

@register_tool("gslides")
async def gslides_tool(tool_call_input: dict) -> Dict[str, Any]:
    """Handles Google Slides presentation creation."""
    tool_name = "gslides"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        # Load username dynamically
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Using username: {username}")

        tool_runnable = get_tool_runnable(
            gslides_agent_system_prompt_template,
            gslides_agent_user_prompt_template,
            gslides_agent_required_format,
            ["query", "user_name", "previous_tool_response"],
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GSlides tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
            "query": tool_call_input["input"],
            "user_name": username,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
        # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")

        return {"tool_result": tool_result, "tool_call_str": None}

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}

@register_tool("gcalendar")
async def gcalendar_tool(tool_call_input: dict) -> Dict[str, Any]:
    """Handles Google Calendar interactions."""
    tool_name = "gcalendar"
    input_instruction = tool_call_input.get("input", "No instruction provided")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Handler called.")
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {input_instruction[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        # Get current time and timezone dynamically
        current_time_iso = datetime.now(timezone.utc).isoformat() # Use UTC for consistency
        local_timezone_key = "UTC" # Default or get dynamically if possible/needed
        try:
            # Attempt to get local timezone; might not work reliably on all servers
            from tzlocal import get_localzone
            local_tz = get_localzone()
            local_timezone_key = local_tz.key
        except ImportError:
            print(f"[WARN] {datetime.now()}: [TOOL HANDLER {tool_name}] tzlocal not installed. Using UTC as timezone.")
        except Exception as tz_e:
             print(f"[WARN] {datetime.now()}: [TOOL HANDLER {tool_name}] Error getting local timezone: {tz_e}. Using UTC.")

        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Using Current Time (ISO): {current_time_iso}, Timezone Key: {local_timezone_key}")

        tool_runnable = get_tool_runnable(
            gcalendar_agent_system_prompt_template,
            gcalendar_agent_user_prompt_template,
            gcalendar_agent_required_format,
            ["query", "current_time", "timezone", "previous_tool_response"],
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GCalendar tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
            "query": tool_call_input["input"],
            "current_time": current_time_iso,
            "timezone": local_timezone_key,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string: {tool_call_str}")

        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool execution completed.")
        # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Tool result: {tool_result}")

        return {"tool_result": tool_result, "tool_call_str": None}

    except Exception as e:
        error_msg = f"Unexpected error in '{tool_name}' tool handler: {e}"
        print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
        traceback.print_exc()
        return {"status": "failure", "error": error_msg}


## Utils Endpoints
@app.post("/get-role")
async def get_role(request: UserInfoRequest) -> JSONResponse:
    """Retrieves a user's role from Auth0."""
    print(f"[ENDPOINT /get-role] {datetime.now()}: Endpoint called for user_id: {request.user_id}")
    try:
        token = get_management_token()
        if not token:
            print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
            raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-role] {datetime.now()}: Auth0 Management token obtained.")

        roles_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}/roles"
        headers = {"Authorization": f"Bearer {token}"}
        print(f"[ENDPOINT /get-role] {datetime.now()}: Making request to Auth0: GET {roles_url}")
        roles_response = requests.get(roles_url, headers=headers)

        print(f"[ENDPOINT /get-role] {datetime.now()}: Auth0 response status: {roles_response.status_code}")
        if roles_response.status_code != 200:
             print(f"[ERROR] {datetime.now()}: Auth0 API error ({roles_response.status_code}): {roles_response.text}")
             raise HTTPException(status_code=roles_response.status_code, detail=f"Auth0 API error: {roles_response.text}")

        roles = roles_response.json()
        print(f"[ENDPOINT /get-role] {datetime.now()}: Roles received: {roles}")
        if not roles:
            print(f"[ENDPOINT /get-role] {datetime.now()}: No roles found for user {request.user_id}.")
            return JSONResponse(status_code=404, content={"message": "No roles found for user."})

        # Assuming the first role is the primary one
        user_role = roles[0].get("name", "unknown").lower()
        print(f"[ENDPOINT /get-role] {datetime.now()}: Determined role: '{user_role}' for user {request.user_id}.")
        return JSONResponse(status_code=200, content={"role": user_role})

    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP exceptions
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-role: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})

@app.post("/get-beta-user-status")
async def get_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """Retrieves beta user status from Auth0 app_metadata."""
    print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Endpoint called for user_id: {request.user_id}")
    try:
        token = get_management_token()
        if not token:
            print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
            raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Auth0 Management token obtained.")

        user_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Making request to Auth0: GET {user_url}")
        response = requests.get(user_url, headers=headers)

        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Auth0 API error: {response.text}")

        user_data = response.json()
        # print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: User data received: {user_data}")
        beta_user_status = user_data.get("app_metadata", {}).get("betaUser")
        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Beta user status from metadata: {beta_user_status}")

        if beta_user_status is None:
            print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Beta user status not found for user {request.user_id}.")
            # Decide default: return 404 or default to False? Returning 404 for clarity.
            return JSONResponse(status_code=404, content={"message": "Beta user status not found in app_metadata."})

        # Ensure boolean response
        status_bool = str(beta_user_status).lower() == 'true'
        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Returning betaUserStatus: {status_bool}")
        return JSONResponse(status_code=200, content={"betaUserStatus": status_bool})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-beta-user-status: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})

@app.post("/get-referral-code")
async def get_referral_code(request: UserInfoRequest) -> JSONResponse:
    """Retrieves the referral code from Auth0 app_metadata."""
    print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Endpoint called for user_id: {request.user_id}")
    try:
        token = get_management_token()
        if not token:
            print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
            raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Auth0 Management token obtained.")

        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Making request to Auth0: GET {url}")
        response = requests.get(url, headers=headers)

        print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error fetching user info: {response.text}")

        user_data = response.json()
        referral_code = user_data.get("app_metadata", {}).get("referralCode")
        print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Referral code from metadata: {referral_code}")

        if not referral_code:
            print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Referral code not found for user {request.user_id}.")
            return JSONResponse(status_code=404, content={"message": "Referral code not found."})

        print(f"[ENDPOINT /get-referral-code] {datetime.now()}: Returning referralCode: {referral_code}")
        return JSONResponse(status_code=200, content={"referralCode": referral_code})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-referral-code: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/get-referrer-status")
async def get_referrer_status(request: UserInfoRequest) -> JSONResponse:
    """Retrieves the referrer status from Auth0 app_metadata."""
    print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Endpoint called for user_id: {request.user_id}")
    try:
        token = get_management_token()
        if not token:
            print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
            raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Auth0 Management token obtained.")

        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Making request to Auth0: GET {url}")
        response = requests.get(url, headers=headers)

        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error fetching user info: {response.text}")

        user_data = response.json()
        referrer_status = user_data.get("app_metadata", {}).get("referrer") # Key is 'referrer'
        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Referrer status from metadata: {referrer_status}")

        if referrer_status is None:
            print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Referrer status not found for user {request.user_id}.")
            return JSONResponse(status_code=404, content={"message": "Referrer status not found."})

        # Ensure boolean response
        status_bool = str(referrer_status).lower() == 'true'
        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Returning referrerStatus: {status_bool}")
        return JSONResponse(status_code=200, content={"referrerStatus": status_bool})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-referrer-status: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/set-referrer-status")
async def set_referrer_status(request: ReferrerStatusRequest) -> JSONResponse:
    """Sets the referrer status in Auth0 app_metadata."""
    print(f"[ENDPOINT /set-referrer-status] {datetime.now()}: Endpoint called for user_id: {request.user_id}, status: {request.referrer_status}")
    try:
        token = get_management_token()
        if not token:
            print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
            raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /set-referrer-status] {datetime.now()}: Auth0 Management token obtained.")

        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"app_metadata": {"referrer": request.referrer_status}} # Key is 'referrer'

        print(f"[ENDPOINT /set-referrer-status] {datetime.now()}: Making request to Auth0: PATCH {url} with payload: {payload}")
        response = requests.patch(url, headers=headers, json=payload)

        print(f"[ENDPOINT /set-referrer-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error updating referrer status: {response.text}")

        print(f"[ENDPOINT /set-referrer-status] {datetime.now()}: Referrer status updated successfully for user {request.user_id}.")
        return JSONResponse(status_code=200, content={"message": "Referrer status updated successfully."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /set-referrer-status: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/get-user-and-set-referrer-status")
async def get_user_and_set_referrer_status(request: SetReferrerRequest) -> JSONResponse:
    """Searches for a user by referral code and sets their referrer status to true."""
    print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Endpoint called for referral_code: {request.referral_code}")
    try:
        token = get_management_token()
        if not token:
             print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
             raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Auth0 Management token obtained.")

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        # --- Search for user by referral code ---
        search_query = f'app_metadata.referralCode:"{request.referral_code}"'
        search_url = f"https://{AUTH0_DOMAIN}/api/v2/users"
        params = {'q': search_query, 'search_engine': 'v3'} # Use v3 search engine
        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Making request to Auth0: GET {search_url} with query: {params}")
        search_response = requests.get(search_url, headers=headers, params=params)

        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Auth0 search response status: {search_response.status_code}")
        if search_response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API search error ({search_response.status_code}): {search_response.text}")
            raise HTTPException(status_code=search_response.status_code, detail=f"Error searching for user: {search_response.text}")

        users = search_response.json()
        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Found {len(users)} user(s) with referral code {request.referral_code}.")

        if not users:
            print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: No user found with referral code.")
            raise HTTPException(status_code=404, detail=f"No user found with referral code: {request.referral_code}")
        if len(users) > 1:
             print(f"[WARN] {datetime.now()}: Multiple users found with referral code {request.referral_code}. Using the first one.")

        user_id = users[0].get("user_id")
        if not user_id:
             print(f"[ERROR] {datetime.now()}: User found but user_id is missing in Auth0 response.")
             raise HTTPException(status_code=500, detail="Found user but could not retrieve user ID.")
        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Found user ID: {user_id}")

        # --- Set referrer status for the found user ---
        update_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{user_id}"
        update_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        update_payload = {"app_metadata": {"referrer": True}} # Set referrer to true

        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Making request to Auth0: PATCH {update_url} with payload: {update_payload}")
        set_status_response = requests.patch(update_url, headers=update_headers, json=update_payload)

        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Auth0 update response status: {set_status_response.status_code}")
        if set_status_response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API update error ({set_status_response.status_code}): {set_status_response.text}")
            raise HTTPException(status_code=set_status_response.status_code, detail=f"Error setting referrer status: {set_status_response.text}")

        print(f"[ENDPOINT /get-user-and-set-referrer-status] {datetime.now()}: Referrer status updated successfully for user {user_id}.")
        return JSONResponse(status_code=200, content={"message": "Referrer status updated successfully."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-user-and-set-referrer-status: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/set-beta-user-status")
def set_beta_user_status(request: BetaUserStatusRequest) -> JSONResponse:
    """Sets the beta user status in Auth0 app_metadata."""
    print(f"[ENDPOINT /set-beta-user-status] {datetime.now()}: Endpoint called for user_id: {request.user_id}, status: {request.beta_user_status}")
    try:
        token = get_management_token()
        if not token:
             print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
             raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /set-beta-user-status] {datetime.now()}: Auth0 Management token obtained.")

        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"app_metadata": {"betaUser": request.beta_user_status}} # Key is 'betaUser'

        print(f"[ENDPOINT /set-beta-user-status] {datetime.now()}: Making request to Auth0: PATCH {url} with payload: {payload}")
        response = requests.patch(url, headers=headers, json=payload)

        print(f"[ENDPOINT /set-beta-user-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error updating beta user status: {response.text}")

        print(f"[ENDPOINT /set-beta-user-status] {datetime.now()}: Beta user status updated successfully for user {request.user_id}.")
        return JSONResponse(status_code=200, content={"message": "Beta user status updated successfully."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /set-beta-user-status: {str(e)}")
        traceback.print_exc()
        # FastAPI needs async here, but this function is sync.
        # If running sync, this return is fine. If async context needed, refactor.
        # Assuming sync context is okay for this specific endpoint based on original code.
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/get-user-and-invert-beta-user-status")
def get_user_and_invert_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """Gets a user's current beta status and inverts it."""
    print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Endpoint called for user_id: {request.user_id}")
    try:
        token = get_management_token()
        if not token:
             print(f"[ERROR] {datetime.now()}: Failed to get Auth0 management token.")
             raise HTTPException(status_code=500, detail="Could not obtain management token.")
        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Auth0 Management token obtained.")

        # --- Get current status ---
        get_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        get_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Making request to Auth0: GET {get_url}")
        get_response = requests.get(get_url, headers=get_headers)

        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Auth0 get response status: {get_response.status_code}")
        if get_response.status_code != 200:
             print(f"[ERROR] {datetime.now()}: Auth0 API error getting user ({get_response.status_code}): {get_response.text}")
             raise HTTPException(status_code=get_response.status_code, detail=f"Error fetching user info: {get_response.text}")

        user_data = get_response.json()
        current_beta_status = user_data.get("app_metadata", {}).get("betaUser")
        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Current beta status from metadata: {current_beta_status}")

        # Determine the inverted status (handle None or non-boolean values)
        current_bool = str(current_beta_status).lower() == 'true'
        inverted_status = not current_bool
        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Inverted status calculated: {inverted_status}")

        # --- Set the inverted status ---
        set_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        set_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        set_payload = {"app_metadata": {"betaUser": inverted_status}}

        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Making request to Auth0: PATCH {set_url} with payload: {set_payload}")
        set_response = requests.patch(set_url, headers=set_headers, json=set_payload)

        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Auth0 set response status: {set_response.status_code}")
        if set_response.status_code != 200:
             print(f"[ERROR] {datetime.now()}: Auth0 API error setting status ({set_response.status_code}): {set_response.text}")
             raise HTTPException(status_code=set_response.status_code, detail=f"Error inverting beta user status: {set_response.text}")

        print(f"[ENDPOINT /get-user-and-invert-beta-user-status] {datetime.now()}: Beta user status inverted successfully for user {request.user_id}.")
        return JSONResponse(status_code=200, content={"message": "Beta user status inverted successfully."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /get-user-and-invert-beta-user-status: {str(e)}")
        traceback.print_exc()
        # Assuming sync context is okay based on original code.
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})

@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest) -> JSONResponse:
    """Encrypts data using AES encryption."""
    print(f"[ENDPOINT /encrypt] {datetime.now()}: Endpoint called. Data length: {len(request.data)}")
    try:
        encrypted_data = aes_encrypt(request.data)
        print(f"[ENDPOINT /encrypt] {datetime.now()}: Data encrypted successfully. Encrypted length: {len(encrypted_data)}")
        return JSONResponse(status_code=200, content={"encrypted_data": encrypted_data})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error during encryption: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Encryption failed: {str(e)}"})

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest) -> JSONResponse:
    """Decrypts data using AES decryption."""
    print(f"[ENDPOINT /decrypt] {datetime.now()}: Endpoint called. Encrypted data length: {len(request.encrypted_data)}")
    try:
        decrypted_data = aes_decrypt(request.encrypted_data)
        print(f"[ENDPOINT /decrypt] {datetime.now()}: Data decrypted successfully. Decrypted length: {len(decrypted_data)}")
        return JSONResponse(status_code=200, content={"decrypted_data": decrypted_data})
    except ValueError as ve: # Catch specific decryption errors (like padding)
         print(f"[ERROR] {datetime.now()}: Error during decryption (likely invalid data/key): {ve}")
         return JSONResponse(status_code=400, content={"message": f"Decryption failed: Invalid input data or key."})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error during decryption: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Decryption failed: {str(e)}"})

## Scraper Endpoints
@app.post("/scrape-linkedin", status_code=200)
async def scrape_linkedin(profile: LinkedInURL):
    """Scrapes and returns LinkedIn profile information."""
    print(f"[ENDPOINT /scrape-linkedin] {datetime.now()}: Endpoint called for URL: {profile.url}")
    try:
        # Ensure the scraping function exists and is imported
        # from model.scraper.functions import scrape_linkedin_profile # Assuming it's here
        print(f"[ENDPOINT /scrape-linkedin] {datetime.now()}: Starting LinkedIn scrape...")
        linkedin_profile = scrape_linkedin_profile(profile.url) # This function needs to be defined/imported
        print(f"[ENDPOINT /scrape-linkedin] {datetime.now()}: LinkedIn scrape completed.")
        # print(f"[ENDPOINT /scrape-linkedin] {datetime.now()}: Scraped profile data: {linkedin_profile}") # Can be verbose
        return JSONResponse(status_code=200, content={"profile": linkedin_profile})
    except NameError:
         error_msg = "LinkedIn scraping function (scrape_linkedin_profile) not available."
         print(f"[ERROR] {datetime.now()}: {error_msg}")
         return JSONResponse(status_code=501, content={"message": error_msg}) # 501 Not Implemented
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /scrape-linkedin: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"LinkedIn scraping failed: {str(e)}"})

@app.post("/scrape-reddit")
async def scrape_reddit(reddit_url: RedditURL):
    """Extracts topics of interest from a Reddit user's profile."""
    print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Endpoint called for URL: {reddit_url.url}")
    try:
        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Starting Reddit scrape...")
        subreddits = reddit_scraper(reddit_url.url) # Assuming reddit_scraper is imported
        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Reddit scrape completed. Found {len(subreddits)} potential subreddits/posts.")
        # print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Scraped subreddits/posts: {subreddits}")

        if not subreddits:
             print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: No subreddits found. Skipping LLM analysis.")
             return JSONResponse(status_code=200, content={"topics": []})

        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Invoking Reddit runnable for topic extraction...")
        response = reddit_runnable.invoke({"subreddits": subreddits})
        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Reddit runnable finished.")
        # print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Runnable response: {response}")

        if isinstance(response, list):
            print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Returning {len(response)} topics.")
            return JSONResponse(status_code=200, content={"topics": response})
        elif isinstance(response, dict) and 'topics' in response and isinstance(response['topics'], list):
             # Handle cases where the runnable might return a dict with a 'topics' key
             print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Runnable returned dict, extracting topics list. Returning {len(response['topics'])} topics.")
             return JSONResponse(status_code=200, content={"topics": response['topics']})
        else:
            error_msg = f"Invalid response format from the Reddit language model. Expected list or dict with 'topics' list, got {type(response)}."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    except NameError as ne:
         error_msg = f"Scraping or runnable function not available: {ne}"
         print(f"[ERROR] {datetime.now()}: {error_msg}")
         return JSONResponse(status_code=501, content={"message": error_msg})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /scrape-reddit: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error during Reddit scraping or analysis: {str(e)}")

@app.post("/scrape-twitter")
async def scrape_twitter(twitter_url: TwitterURL):
    """Extracts topics of interest from a Twitter user's profile."""
    print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Endpoint called for URL: {twitter_url.url}")
    num_tweets = 20 # Define how many tweets to fetch
    try:
        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Starting Twitter scrape for {num_tweets} tweets...")
        # Ensure scrape_twitter_data is imported/defined
        tweets = scrape_twitter_data(twitter_url.url, num_tweets)
        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Twitter scrape completed. Found {len(tweets)} tweets.")
        # print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Scraped tweets: {tweets}")

        if not tweets:
             print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: No tweets found. Skipping LLM analysis.")
             return JSONResponse(status_code=200, content={"topics": []})

        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Invoking Twitter runnable for topic extraction...")
        response = twitter_runnable.invoke({"tweets": tweets})
        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Twitter runnable finished.")
        # print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Runnable response: {response}")

        if isinstance(response, list):
            print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Returning {len(response)} topics.")
            return JSONResponse(status_code=200, content={"topics": response})
        elif isinstance(response, dict) and 'topics' in response and isinstance(response['topics'], list):
            print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Runnable returned dict, extracting topics list. Returning {len(response['topics'])} topics.")
            return JSONResponse(status_code=200, content={"topics": response['topics']})
        else:
            error_msg = f"Invalid response format from the Twitter language model. Expected list or dict with 'topics' list, got {type(response)}."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    except NameError as ne:
         error_msg = f"Scraping or runnable function not available: {ne}"
         print(f"[ERROR] {datetime.now()}: {error_msg}")
         return JSONResponse(status_code=501, content={"message": error_msg})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /scrape-twitter: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error during Twitter scraping or analysis: {str(e)}")

## Auth Endpoint
@app.get("/authenticate-google")
async def authenticate_google():
    """Authenticates with Google using OAuth 2.0."""
    print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Endpoint called.")
    token_file = "model/token.pickle"
    creds = None
    try:
        # 1. Check for existing, valid token
        if os.path.exists(token_file):
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Found existing token file: {token_file}")
            with open(token_file, "rb") as token:
                creds = pickle.load(token)
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Token loaded from file.")
            # Validate token
            if creds and creds.valid:
                print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Existing token is valid.")
                return JSONResponse(status_code=200, content={"success": True, "message": "Already authenticated."})
            else:
                print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Existing token is invalid or expired.")

        # 2. Refresh token if possible
        if creds and creds.expired and creds.refresh_token:
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Attempting to refresh expired token...")
            try:
                creds.refresh(Request())
                print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Token refreshed successfully.")
                # Save the refreshed token
                with open(token_file, "wb") as token:
                    pickle.dump(creds, token)
                print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Refreshed token saved to {token_file}.")
                return JSONResponse(status_code=200, content={"success": True, "message": "Authentication refreshed."})
            except Exception as refresh_err:
                 print(f"[ERROR] {datetime.now()}: Failed to refresh token: {refresh_err}. Proceeding to full authentication flow.")
                 creds = None # Ensure we trigger the flow

        # 3. Run full OAuth flow if no valid/refreshable token
        if not creds:
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: No valid token found. Starting OAuth flow...")
            # Ensure credentials dictionary is valid
            if not CREDENTIALS_DICT or not CREDENTIALS_DICT.get("installed") or not CREDENTIALS_DICT["installed"].get("client_id"):
                 error_msg = "Google API credentials configuration is missing or invalid."
                 print(f"[ERROR] {datetime.now()}: {error_msg}")
                 raise HTTPException(status_code=500, detail=error_msg)

            flow = InstalledAppFlow.from_client_config(CREDENTIALS_DICT, SCOPES)
            # The `run_local_server` will block until the user completes the flow in their browser.
            # It opens a local server temporarily (default port 8080, use port=0 for random).
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: Launching local server for user authentication...")
            creds = flow.run_local_server(port=0) # Use random available port
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: OAuth flow completed by user. Credentials obtained.")

            # Save the new credentials
            with open(token_file, "wb") as token:
                pickle.dump(creds, token)
            print(f"[ENDPOINT /authenticate-google] {datetime.now()}: New token saved to {token_file}.")
            return JSONResponse(status_code=200, content={"success": True, "message": "Authentication successful."})

    except FileNotFoundError:
        # This might happen if the pickle file is corrupted during load before check
        print(f"[WARN] {datetime.now()}: Token file not found during load (should have been caught by os.path.exists). Proceeding to auth flow.")
        # Rerun the flow part (could refactor this)
        flow = InstalledAppFlow.from_client_config(CREDENTIALS_DICT, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_file, "wb") as token:
            pickle.dump(creds, token)
        print(f"[ENDPOINT /authenticate-google] {datetime.now()}: New token saved after FileNotFoundError during load.")
        return JSONResponse(status_code=200, content={"success": True, "message": "Authentication successful."})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error during Google authentication: {e}")
        traceback.print_exc()
        # Return failure response
        return JSONResponse(status_code=500, content={"success": False, "error": f"Authentication failed: {str(e)}"})

## Memory Endpoints
@app.post("/graphrag", status_code=200)
async def graphrag(request: GraphRAGRequest):
    """Processes a user profile query using GraphRAG."""
    print(f"[ENDPOINT /graphrag] {datetime.now()}: Endpoint called with query: '{request.query[:50]}...'")
    try:
        # Check dependencies
        if not all([graph_driver, embed_model, text_conversion_runnable, query_classification_runnable]):
            error_msg = "GraphRAG dependencies (Neo4j, Embeddings, Runnables) are not initialized."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)

        print(f"[ENDPOINT /graphrag] {datetime.now()}: Querying user profile...")
        context = query_user_profile(
            request.query, graph_driver, embed_model,
            text_conversion_runnable, query_classification_runnable
        )
        print(f"[ENDPOINT /graphrag] {datetime.now()}: User profile query completed. Context length: {len(str(context)) if context else 0}")
        # print(f"[ENDPOINT /graphrag] {datetime.now()}: Retrieved context: {context}") # Can be verbose
        return JSONResponse(status_code=200, content={"context": context})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /graphrag: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"GraphRAG query failed: {str(e)}"})

@app.post("/initiate-long-term-memories", status_code=200)
async def create_graph():
    """Creates a knowledge graph from documents in the input directory."""
    print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Endpoint called.")
    input_dir = "model/input"
    extracted_texts = []
    try:
        # --- Load Username ---
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Using username: {username}")

        # --- Check Dependencies ---
        if not all([graph_driver, embed_model, text_dissection_runnable, information_extraction_runnable]):
             error_msg = "Create Graph dependencies (Neo4j, Embeddings, Runnables) are not initialized."
             print(f"[ERROR] {datetime.now()}: {error_msg}")
             raise HTTPException(status_code=503, detail=error_msg)

        # --- Read Input Files ---
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Reading text files from input directory: {input_dir}")
        if not os.path.exists(input_dir):
             print(f"[WARN] {datetime.now()}: Input directory '{input_dir}' does not exist. Creating it.")
             os.makedirs(input_dir)

        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(".txt"): # Process only .txt files
                print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Reading file: {file_name}")
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        text_content = file.read().strip()
                        if text_content:
                            extracted_texts.append({"text": text_content, "source": file_name})
                            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}:   - Added content from {file_name}. Length: {len(text_content)}")
                        else:
                             print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}:   - Skipped empty file: {file_name}")
                except Exception as read_e:
                     print(f"[ERROR] {datetime.now()}: Failed to read file {file_name}: {read_e}")
            else:
                 print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Skipping non-txt file or directory: {file_name}")


        if not extracted_texts:
            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: No text content found in input directory. Nothing to build.")
            return JSONResponse(status_code=200, content={"message": "No content found in input documents. Graph not modified."}) # Not really an error

        # --- Clear Existing Graph (Optional - Be Careful!) ---
        # Consider making this conditional based on a request parameter
        clear_graph = True # Set to False or make configurable if you don't want to wipe the graph every time
        if clear_graph:
            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Clearing existing graph in Neo4j...")
            try:
                with graph_driver.session(database="neo4j") as session: # Specify DB if not default
                    # Use write_transaction for safety
                    session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
                print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Existing graph cleared successfully.")
            except Exception as clear_e:
                 error_msg = f"Failed to clear existing graph: {clear_e}"
                 print(f"[ERROR] {datetime.now()}: {error_msg}")
                 raise HTTPException(status_code=500, detail=error_msg)
        else:
             print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Skipping graph clearing.")


        # --- Build Graph ---
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Building initial knowledge graph from {len(extracted_texts)} document(s)...")
        build_initial_knowledge_graph(
            username, extracted_texts, graph_driver, embed_model,
            text_dissection_runnable, information_extraction_runnable
        ) # This function should contain its own detailed logging
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Knowledge graph build process completed.")

        return JSONResponse(status_code=200, content={"message": f"Graph created/updated successfully from {len(extracted_texts)} documents."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /initiate-long-term-memories: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Graph creation failed: {str(e)}"})


@app.post("/delete-subgraph", status_code=200)
async def delete_subgraph(request: DeleteSubgraphRequest):
    """Deletes a subgraph from the knowledge graph based on a source name."""
    source_key = request.source # e.g., "linkedin", "reddit"
    print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Endpoint called for source key: {source_key}")
    input_dir = "model/input"

    try:
        # --- Load Username and Map Source Key to Filename ---
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User").lower()
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Using username: {username}")

        # Define the mapping from source key to expected filename pattern
        SOURCES = {
            "linkedin": f"{username}_linkedin_profile.txt",
            "reddit": f"{username}_reddit_profile.txt",
            "twitter": f"{username}_twitter_profile.txt",
            # Add mappings for personality traits if they should be deletable
            "extroversion": f"{username}_extroversion.txt",
            "introversion": f"{username}_introversion.txt",
            # ... add all personality traits used in create_document ...
            "personality": f"{username}_personality.txt", # If a unified file exists
        }
        # Add dynamic personality trait filenames based on constants if needed
        # for trait in PERSONALITY_DESCRIPTIONS.keys():
        #     SOURCES[trait.lower()] = f"{username}_{trait.lower()}.txt"

        file_name = SOURCES.get(source_key.lower()) # Use lower case for matching
        if not file_name:
            error_msg = f"No file mapping found for source key: '{source_key}'. Valid keys: {list(SOURCES.keys())}"
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            return JSONResponse(status_code=400, content={"message": error_msg})
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Mapped source key '{source_key}' to filename: {file_name}")

        # --- Check Dependency ---
        if not graph_driver:
             error_msg = "Neo4j driver is not initialized. Cannot delete subgraph."
             print(f"[ERROR] {datetime.now()}: {error_msg}")
             raise HTTPException(status_code=503, detail=error_msg)

        # --- Delete Subgraph from Neo4j ---
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Deleting subgraph related to source '{file_name}' from Neo4j...")
        # Assuming delete_source_subgraph handles the Neo4j deletion logic
        delete_source_subgraph(graph_driver, file_name)
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Subgraph deletion from Neo4j completed for '{file_name}'.")

        # --- Delete Corresponding Input File ---
        file_path_to_delete = os.path.join(input_dir, file_name)
        if os.path.exists(file_path_to_delete):
            try:
                os.remove(file_path_to_delete)
                print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Deleted input file: {file_path_to_delete}")
            except OSError as remove_e:
                 print(f"[WARN] {datetime.now()}: Failed to delete input file {file_path_to_delete}: {remove_e}. Subgraph was still deleted from Neo4j.")
        else:
             print(f"[WARN] {datetime.now()}: Input file {file_path_to_delete} not found. No file deleted.")


        return JSONResponse(status_code=200, content={"message": f"Subgraph related to source '{source_key}' (file: {file_name}) deleted successfully."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /delete-subgraph: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Subgraph deletion failed: {str(e)}"})


@app.post("/create-document", status_code=200)
async def create_document():
    """Creates and summarizes personality documents based on user profile data."""
    print(f"[ENDPOINT /create-document] {datetime.now()}: Endpoint called.")
    input_dir = "model/input"
    created_files = []
    unified_personality_description = ""

    try:
        # --- Load User Profile Data ---
        db = load_user_profile().get("userData", {})
        username = db.get("personalInfo", {}).get("name", "User")
        # Ensure personalityType is a list
        personality_type = db.get("personalityType", [])
        if isinstance(personality_type, str): # Handle case where it might be saved as string
            personality_type = [p.strip() for p in personality_type.split(',') if p.strip()]
        structured_linkedin_profile = db.get("linkedInProfile", {}) # Assuming dict/string
        # Ensure social profiles are lists of strings/topics
        reddit_profile = db.get("redditProfile", [])
        if isinstance(reddit_profile, str): reddit_profile = [reddit_profile]
        twitter_profile = db.get("twitterProfile", [])
        if isinstance(twitter_profile, str): twitter_profile = [twitter_profile]

        print(f"[ENDPOINT /create-document] {datetime.now()}: Processing for user: {username}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Personality Traits: {personality_type}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: LinkedIn data present: {bool(structured_linkedin_profile)}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Reddit topics: {reddit_profile}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Twitter topics: {twitter_profile}")

        # --- Ensure Input Directory Exists ---
        os.makedirs(input_dir, exist_ok=True)
        print(f"[ENDPOINT /create-document] {datetime.now()}: Ensured input directory exists: {input_dir}")

        # --- Clear Existing Files (Optional - matching /initiate-long-term-memories behavior) ---
        # clear_existing = True
        # if clear_existing:
        #     print(f"[ENDPOINT /create-document] {datetime.now()}: Clearing existing files in {input_dir}...")
        #     cleared_count = 0
        #     for file in os.listdir(input_dir):
        #          try:
        #              os.remove(os.path.join(input_dir, file))
        #              cleared_count += 1
        #          except OSError as rm_e:
        #              print(f"[WARN] Failed to remove file {file}: {rm_e}")
        #     print(f"[ENDPOINT /create-document] {datetime.now()}: Cleared {cleared_count} existing files.")


        # --- Process Personality Traits ---
        trait_descriptions = []
        print(f"[ENDPOINT /create-document] {datetime.now()}: Processing {len(personality_type)} personality traits...")
        for trait in personality_type:
            if trait in PERSONALITY_DESCRIPTIONS:
                description = f"{trait}: {PERSONALITY_DESCRIPTIONS[trait]}"
                trait_descriptions.append(description)
                filename = f"{username.lower()}_{trait.lower()}.txt"
                file_path = os.path.join(input_dir, filename)
                print(f"[ENDPOINT /create-document] {datetime.now()}: Summarizing description for trait '{trait}'...")
                try:
                    summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": description})
                    print(f"[ENDPOINT /create-document] {datetime.now()}: Writing summarized trait to {filename}...")
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(summarized_paragraph)
                    created_files.append(filename)
                except Exception as e:
                     print(f"[ERROR] {datetime.now()}: Failed to summarize or write file for trait '{trait}': {e}")
            else:
                 print(f"[WARN] {datetime.now()}: Personality trait '{trait}' not found in PERSONALITY_DESCRIPTIONS. Skipping.")

        unified_personality_description = f"{username}'s Personality Traits:\n\n" + "\n".join(trait_descriptions)
        # Optionally, save the unified description to a file as well?
        # unified_filename = f"{username.lower()}_personality_summary.txt"
        # with open(os.path.join(input_dir, unified_filename), "w", encoding="utf-8") as file:
        #     file.write(unified_personality_description)
        # created_files.append(unified_filename)

        # --- Process LinkedIn Profile ---
        if structured_linkedin_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing LinkedIn profile...")
            # Convert dict to string representation if necessary
            linkedin_text = json.dumps(structured_linkedin_profile, indent=2) if isinstance(structured_linkedin_profile, dict) else str(structured_linkedin_profile)
            linkedin_file = f"{username.lower()}_linkedin_profile.txt"
            file_path = os.path.join(input_dir, linkedin_file)
            print(f"[ENDPOINT /create-document] {datetime.now()}: Summarizing LinkedIn profile...")
            try:
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": linkedin_text})
                print(f"[ENDPOINT /create-document] {datetime.now()}: Writing summarized LinkedIn profile to {linkedin_file}...")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)
                created_files.append(linkedin_file)
            except Exception as e:
                 print(f"[ERROR] {datetime.now()}: Failed to summarize or write LinkedIn file: {e}")
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No LinkedIn profile data found.")


        # --- Process Reddit Profile ---
        if reddit_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing Reddit profile topics...")
            reddit_text = "User's Reddit Interests: " + ", ".join(reddit_profile)
            reddit_file = f"{username.lower()}_reddit_profile.txt"
            file_path = os.path.join(input_dir, reddit_file)
            print(f"[ENDPOINT /create-document] {datetime.now()}: Summarizing Reddit interests...")
            try:
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": reddit_text})
                print(f"[ENDPOINT /create-document] {datetime.now()}: Writing summarized Reddit interests to {reddit_file}...")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)
                created_files.append(reddit_file)
            except Exception as e:
                 print(f"[ERROR] {datetime.now()}: Failed to summarize or write Reddit file: {e}")
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No Reddit profile data found.")

        # --- Process Twitter Profile ---
        if twitter_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing Twitter profile topics...")
            twitter_text = "User's Twitter Interests: " + ", ".join(twitter_profile)
            twitter_file = f"{username.lower()}_twitter_profile.txt"
            file_path = os.path.join(input_dir, twitter_file)
            print(f"[ENDPOINT /create-document] {datetime.now()}: Summarizing Twitter interests...")
            try:
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": twitter_text})
                print(f"[ENDPOINT /create-document] {datetime.now()}: Writing summarized Twitter interests to {twitter_file}...")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)
                created_files.append(twitter_file)
            except Exception as e:
                 print(f"[ERROR] {datetime.now()}: Failed to summarize or write Twitter file: {e}")
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No Twitter profile data found.")

        print(f"[ENDPOINT /create-document] {datetime.now()}: Document creation process finished. Created {len(created_files)} files: {created_files}")
        return JSONResponse(status_code=200, content={
            "message": f"Documents created successfully: {', '.join(created_files)}",
            "personality": unified_personality_description # Return the combined description
        })

    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /create-document: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Document creation failed: {str(e)}"})


@app.post("/customize-long-term-memories", status_code=200)
async def customize_graph(request: GraphRequest):
    """Customizes the knowledge graph with new information."""
    print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Endpoint called with information: '{request.information[:50]}...'")
    try:
        # --- Load Username ---
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Using username: {username}")

        # --- Check Dependencies ---
        if not all([fact_extraction_runnable, graph_driver, embed_model, query_classification_runnable,
                   information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable,
                   text_description_runnable]):
             error_msg = "Customize Graph dependencies are not fully initialized."
             print(f"[ERROR] {datetime.now()}: {error_msg}")
             raise HTTPException(status_code=503, detail=error_msg)

        # --- Extract Facts ---
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracting facts from provided information...")
        points = fact_extraction_runnable.invoke({"paragraph": request.information, "username": username})
        if not isinstance(points, list):
             print(f"[WARN] {datetime.now()}: Fact extraction did not return a list. Got: {type(points)}. Assuming no facts extracted.")
             points = []
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracted {len(points)} potential facts.")
        # print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracted facts: {points}")

        if not points:
             return JSONResponse(status_code=200, content={"message": "No specific facts extracted from the information. Graph not modified."})

        # --- Apply Graph Operations ---
        processed_count = 0
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Applying CRUD operations for {len(points)} facts...")
        for i, point in enumerate(points):
            print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Processing fact {i+1}/{len(points)}: {str(point)[:100]}...")
            try:
                # crud_graph_operations should contain its own logging
                crud_graph_operations(
                    point, graph_driver, embed_model, query_classification_runnable,
                    information_extraction_runnable, graph_analysis_runnable,
                    graph_decision_runnable, text_description_runnable
                )
                processed_count += 1
            except Exception as crud_e:
                 print(f"[ERROR] {datetime.now()}: Failed to process fact {i+1} ('{str(point)[:50]}...'): {crud_e}")
                 # Decide whether to continue or stop on error
                 # traceback.print_exc() # Optionally print traceback for failed fact

        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Graph customization process completed. Applied operations for {processed_count}/{len(points)} facts.")
        return JSONResponse(status_code=200, content={"message": f"Graph customized successfully with {processed_count} facts."})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /customize-long-term-memories: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Graph customization failed: {str(e)}"})

## Task Queue Endpoints
@app.get("/fetch-tasks", status_code=200)
async def get_tasks():
    """Return the current state of all tasks."""
    # print(f"[ENDPOINT /fetch-tasks] {datetime.now()}: Endpoint called.")
    try:
        tasks = await task_queue.get_all_tasks()
        # print(f"[ENDPOINT /fetch-tasks] {datetime.now()}: Fetched {len(tasks)} tasks from queue.")
        return JSONResponse(content={"tasks": tasks})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error fetching tasks: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch tasks.")

@app.post("/add-task", status_code=201)
async def add_task(task_request: CreateTaskRequest): # Use CreateTaskRequest
    """
    Adds a new task with dynamically determined chat_id, personality, context needs, priority etc.
    Input only requires the task description.
    """
    print(f"[ENDPOINT /add-task] {datetime.now()}: Endpoint called with description: '{task_request.description[:50]}...'")
    try:
        # --- Determine Active Chat ID ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Determining active chat ID...")
        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id")
            if not active_chat_id:
                 # Ensure chat exists if none is active
                 await get_chat_history_messages()
                 chatsDb = await load_db()
                 active_chat_id = chatsDb.get("active_chat_id")
                 if not active_chat_id:
                     raise HTTPException(status_code=500, detail="Failed to determine or create an active chat ID.")
        print(f"[ENDPOINT /add-task] {datetime.now()}: Using active chat ID: {active_chat_id}")

        # --- Load User Profile Info ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Loading user profile...")
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        personality = user_profile.get("userData", {}).get("personality", "Default helpful assistant")
        print(f"[ENDPOINT /add-task] {datetime.now()}: Using username: '{username}', personality: '{str(personality)[:50]}...'")

        # --- Classify Task Needs ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Classifying task needs (context, internet)...")
        # Assuming unified classifier handles this based on description
        unified_output = unified_classification_runnable.invoke({"query": task_request.description})
        use_personal_context = unified_output.get("use_personal_context", False)
        internet = unified_output.get("internet", "None")
        # Use the potentially transformed input from classification? Or original description? Using original for now.
        # transformed_input = unified_output.get("transformed_input", task_request.description)
        print(f"[ENDPOINT /add-task] {datetime.now()}: Task needs: Use Context={use_personal_context}, Internet='{internet}'")

        # --- Determine Priority ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Determining task priority...")
        try:
            priority_response = priority_runnable.invoke({"task_description": task_request.description})
            priority = priority_response.get("priority", 3) # Default priority
            print(f"[ENDPOINT /add-task] {datetime.now()}: Determined priority: {priority}")
        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Failed to determine priority: {e}. Using default (3).")
            priority = 3

        # --- Add Task to Queue ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Adding task to queue...")
        task_id = await task_queue.add_task(
            chat_id=active_chat_id,
            description=task_request.description, # Use original description
            priority=priority,
            username=username,
            personality=personality,
            use_personal_context=use_personal_context,
            internet=internet
        )
        print(f"[ENDPOINT /add-task] {datetime.now()}: Task added successfully with ID: {task_id}")

        # --- Add User Message (Hidden) ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Adding hidden user message to chat {active_chat_id} for task description.")
        await add_result_to_chat(active_chat_id, task_request.description, isUser=True) # isVisible defaults to False for user=True

        # --- Add Assistant Confirmation (Visible) ---
        confirmation_message = f"OK, I've added the task: '{task_request.description[:40]}...' to my list."
        print(f"[ENDPOINT /add-task] {datetime.now()}: Adding visible assistant confirmation to chat {active_chat_id}.")
        await add_result_to_chat(active_chat_id, confirmation_message, isUser=False, task_description=task_request.description)

        # --- Broadcast Task Addition via WebSocket ---
        task_added_message = {
            "type": "task_added",
            "task_id": task_id,
            "description": task_request.description,
            "priority": priority,
            "status": "pending", # Initial status
            # Add other relevant fields if needed by the frontend
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task addition for {task_id}")
        await manager.broadcast(json.dumps(task_added_message))


        return JSONResponse(content={"task_id": task_id, "message": "Task added successfully"})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error adding task: {e}")
        traceback.print_exc()
        # Return error but maybe not raise HTTPException if adding message failed?
        # Or just raise the HTTP Exception
        raise HTTPException(status_code=500, detail=f"Failed to add task: {str(e)}")

@app.post("/update-task", status_code=200)
async def update_task(update_request: UpdateTaskRequest):
    """Updates an existing task's description and priority."""
    task_id = update_request.task_id
    new_desc = update_request.description
    new_priority = update_request.priority
    print(f"[ENDPOINT /update-task] {datetime.now()}: Endpoint called for task ID: {task_id}")
    print(f"[ENDPOINT /update-task] {datetime.now()}: New Description: '{new_desc[:50]}...', New Priority: {new_priority}")
    try:
        await task_queue.update_task(task_id, new_desc, new_priority)
        print(f"[ENDPOINT /update-task] {datetime.now()}: Task {task_id} updated successfully in queue.")

        # --- Broadcast Task Update via WebSocket ---
        task_update_message = {
            "type": "task_updated",
            "task_id": task_id,
            "description": new_desc,
            "priority": new_priority,
            # Include status if it might change or is relevant
            # "status": updated_task_status # Need to fetch updated status if needed
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task update for {task_id}")
        await manager.broadcast(json.dumps(task_update_message))

        return JSONResponse(content={"message": "Task updated successfully"})
    except ValueError as e: # Task not found
        print(f"[ERROR] {datetime.now()}: Error updating task {task_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) # Use 404 for not found
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error updating task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")

@app.post("/delete-task", status_code=200)
async def delete_task(delete_request: DeleteTaskRequest):
    """Deletes a task from the queue."""
    task_id = delete_request.task_id
    print(f"[ENDPOINT /delete-task] {datetime.now()}: Endpoint called for task ID: {task_id}")
    try:
        await task_queue.delete_task(task_id)
        print(f"[ENDPOINT /delete-task] {datetime.now()}: Task {task_id} deleted successfully from queue.")

         # --- Broadcast Task Deletion via WebSocket ---
        task_delete_message = {
            "type": "task_deleted",
            "task_id": task_id,
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task deletion for {task_id}")
        await manager.broadcast(json.dumps(task_delete_message))

        return JSONResponse(content={"message": "Task deleted successfully"})
    except ValueError as e: # Task not found or cannot be deleted
        print(f"[ERROR] {datetime.now()}: Error deleting task {task_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) # Use 404 for not found
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error deleting task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

## Short-Term Memory Endpoints
@app.post("/get-short-term-memories")
async def get_short_term_memories(request: GetShortTermMemoriesRequest) -> List[Dict]:
    """Fetches short-term memories for a user and category."""
    user_id = request.user_id
    category = request.category
    limit = request.limit
    print(f"[ENDPOINT /get-short-term-memories] {datetime.now()}: Endpoint called for User: {user_id}, Category: {category}, Limit: {limit}")
    try:
        memories = memory_backend.memory_manager.fetch_memories_by_category(
            user_id=user_id,
            category=category,
            limit=limit
        )
        print(f"[ENDPOINT /get-short-term-memories] {datetime.now()}: Fetched {len(memories)} memories.")
        # print(f"[ENDPOINT /get-short-term-memories] {datetime.now()}: Fetched memories: {memories}") # Can be verbose
        # Ensure return is JSON serializable (datetime objects might need conversion)
        serializable_memories = []
        for mem in memories:
             # Convert datetime objects to ISO format strings if they exist
             if isinstance(mem.get('created_at'), datetime):
                 mem['created_at'] = mem['created_at'].isoformat()
             if isinstance(mem.get('expires_at'), datetime):
                  mem['expires_at'] = mem['expires_at'].isoformat()
             serializable_memories.append(mem)

        return JSONResponse(content=serializable_memories) # Return as JSON response
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /get-short-term-memories: {e}")
        traceback.print_exc()
        # Return empty list in case of error, maybe should be 500?
        # Returning 500 might be better to signal failure
        raise HTTPException(status_code=500, detail=f"Failed to fetch memories: {str(e)}")


@app.post("/add-short-term-memory")
async def add_memory(request: AddMemoryRequest):
    """Add a new short-term memory."""
    user_id = request.user_id
    text = request.text
    category = request.category
    retention = request.retention_days
    print(f"[ENDPOINT /add-short-term-memory] {datetime.now()}: Endpoint called for User: {user_id}, Category: {category}, Retention: {retention} days")
    print(f"[ENDPOINT /add-short-term-memory] {datetime.now()}: Text: '{text[:50]}...'")
    try:
        memory_id = memory_backend.memory_manager.store_memory(
            user_id, text, category, retention
        )
        print(f"[ENDPOINT /add-short-term-memory] {datetime.now()}: Memory stored successfully with ID: {memory_id}")
        return JSONResponse(status_code=201, content={"memory_id": memory_id, "message": "Memory added successfully"})
    except ValueError as e: # Input validation errors
        print(f"[ERROR] {datetime.now()}: Value error adding memory: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error adding memory: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

@app.post("/update-short-term-memory")
async def update_memory(request: UpdateMemoryRequest):
    """Update an existing short-term memory."""
    user_id = request.user_id
    category = request.category
    mem_id = request.id
    text = request.text
    retention = request.retention_days
    print(f"[ENDPOINT /update-short-term-memory] {datetime.now()}: Endpoint called for User: {user_id}, Category: {category}, ID: {mem_id}")
    print(f"[ENDPOINT /update-short-term-memory] {datetime.now()}: New Text: '{text[:50]}...', New Retention: {retention} days")
    try:
        memory_backend.memory_manager.update_memory(
            user_id, category, mem_id, text, retention
        )
        print(f"[ENDPOINT /update-short-term-memory] {datetime.now()}: Memory ID {mem_id} updated successfully.")
        return JSONResponse(status_code=200, content={"message": "Memory updated successfully"})
    except ValueError as e: # Not found or invalid input
        print(f"[ERROR] {datetime.now()}: Value error updating memory {mem_id}: {e}")
        # Distinguish between Not Found (404) and Bad Request (400) if possible
        if "not found" in str(e).lower():
             raise HTTPException(status_code=404, detail=str(e))
        else:
             raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error updating memory {mem_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")

@app.post("/delete-short-term-memory")
async def delete_memory(request: DeleteMemoryRequest):
    """Delete a short-term memory."""
    user_id = request.user_id
    category = request.category
    mem_id = request.id
    print(f"[ENDPOINT /delete-short-term-memory] {datetime.now()}: Endpoint called for User: {user_id}, Category: {category}, ID: {mem_id}")
    try:
        memory_backend.memory_manager.delete_memory(
            user_id, category, mem_id
        )
        print(f"[ENDPOINT /delete-short-term-memory] {datetime.now()}: Memory ID {mem_id} deleted successfully.")
        return JSONResponse(status_code=200, content={"message": "Memory deleted successfully"})
    except ValueError as e: # Memory not found
        print(f"[ERROR] {datetime.now()}: Value error deleting memory {mem_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) # 404 Not Found
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error deleting memory {mem_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")

@app.post("/clear-all-short-term-memories")
async def clear_all_memories(request: Dict):
    """Clears all short-term memories for a given user."""
    user_id = request.get("user_id")
    print(f"[ENDPOINT /clear-all-short-term-memories] {datetime.now()}: Endpoint called for User: {user_id}")
    if not user_id:
        print(f"[ERROR] {datetime.now()}: 'user_id' is missing in the request.")
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        memory_backend.memory_manager.clear_all_memories(user_id)
        print(f"[ENDPOINT /clear-all-short-term-memories] {datetime.now()}: All memories cleared successfully for user {user_id}.")
        return JSONResponse(status_code=200, content={"message": "All memories cleared successfully"})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error clearing memories for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")

## User Profile DB Endpoints
@app.post("/set-user-data")
async def set_db_data(request: UpdateUserDataRequest) -> Dict[str, Any]:
    """
    Set data in the user profile database (overwrites existing keys at the top level of userData).
    """
    print(f"[ENDPOINT /set-user-data] {datetime.now()}: Endpoint called.")
    # print(f"[ENDPOINT /set-user-data] {datetime.now()}: Request data: {request.data}") # Careful logging potentially sensitive data
    try:
        db_data = load_user_profile()
        if "userData" not in db_data: # Ensure userData key exists
            db_data["userData"] = {}

        # Merge new data, overwriting existing keys at the same level
        # This performs a shallow merge. For deep merge, a recursive function would be needed.
        db_data["userData"].update(request.data)
        print(f"[ENDPOINT /set-user-data] {datetime.now()}: User data updated (shallow merge).")

        if write_user_profile(db_data):
            print(f"[ENDPOINT /set-user-data] {datetime.now()}: Data stored successfully.")
            return JSONResponse(status_code=200, content={"message": "Data stored successfully", "status": 200})
        else:
            print(f"[ERROR] {datetime.now()}: Failed to write updated user profile to disk.")
            raise HTTPException(status_code=500, detail="Error storing data: Failed to write to file")
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /set-user-data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")


@app.post("/add-db-data")
async def add_db_data(request: AddUserDataRequest) -> Dict[str, Any]:
    """
    Add data to the user profile database with merging logic for lists and dicts.
    """
    print(f"[ENDPOINT /add-db-data] {datetime.now()}: Endpoint called.")
    # print(f"[ENDPOINT /add-db-data] {datetime.now()}: Request data: {request.data}") # Careful logging
    try:
        db_data = load_user_profile()
        existing_data = db_data.get("userData", {}) # Ensure userData exists
        data_to_add = request.data

        print(f"[ENDPOINT /add-db-data] {datetime.now()}: Starting merge process...")
        for key, value in data_to_add.items():
            # print(f"[ENDPOINT /add-db-data] {datetime.now()}: Merging key: '{key}'")
            if key in existing_data:
                if isinstance(existing_data[key], list) and isinstance(value, list):
                    # Merge lists and remove duplicates
                    merged_list = existing_data[key] + [item for item in value if item not in existing_data[key]]
                    existing_data[key] = merged_list
                    # print(f"[ENDPOINT /add-db-data] {datetime.now()}:   - Merged list for key '{key}'. New length: {len(merged_list)}")
                elif isinstance(existing_data[key], dict) and isinstance(value, dict):
                    # Merge dictionaries (shallow merge)
                    existing_data[key].update(value)
                    # print(f"[ENDPOINT /add-db-data] {datetime.now()}:   - Merged dict for key '{key}'.")
                else:
                    # Overwrite if types don't match or aren't list/dict
                    existing_data[key] = value
                    # print(f"[ENDPOINT /add-db-data] {datetime.now()}:   - Overwrote value for key '{key}'.")
            else:
                # Add new key
                existing_data[key] = value
                # print(f"[ENDPOINT /add-db-data] {datetime.now()}:   - Added new key '{key}'.")

        db_data["userData"] = existing_data # Update userData in the main structure
        print(f"[ENDPOINT /add-db-data] {datetime.now()}: Merge process complete.")

        if write_user_profile(db_data):
            print(f"[ENDPOINT /add-db-data] {datetime.now()}: Data added/merged successfully.")
            return JSONResponse(status_code=200, content={"message": "Data added successfully", "status": 200})
        else:
            print(f"[ERROR] {datetime.now()}: Failed to write updated user profile to disk.")
            raise HTTPException(status_code=500, detail="Error adding data: Failed to write to file")

    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /add-db-data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding data: {str(e)}")


@app.post("/get-user-data")
# Request model not strictly needed as it takes no input, but good for consistency if used elsewhere
async def get_db_data() -> Dict[str, Any]:
    """Get all user profile database data."""
    print(f"[ENDPOINT /get-user-data] {datetime.now()}: Endpoint called.")
    try:
        db_data = load_user_profile()
        user_data = db_data.get("userData", {}) # Default to empty dict if not found
        print(f"[ENDPOINT /get-user-data] {datetime.now()}: Retrieved user data successfully.")
        # print(f"[ENDPOINT /get-user-data] {datetime.now()}: User data: {user_data}") # Careful logging
        return JSONResponse(status_code=200, content={"data": user_data, "status": 200})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error fetching user data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

## Graph Data Endpoint
@app.post("/get-graph-data") # Changed to POST maybe? Or keep GET if no body needed. Keeping POST for now.
async def get_graph_data_apoc():
    """Fetches graph data using APOC procedures."""
    print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Endpoint called.")
    if not graph_driver:
         print(f"[ERROR] {datetime.now()}: Neo4j driver not available.")
         raise HTTPException(status_code=503, detail="Database connection not available")

    # APOC query to get nodes and relationships in a suitable format
    apoc_query = """
    MATCH (n)
    WITH collect(DISTINCT n) as nodes // Collect distinct nodes first
    OPTIONAL MATCH (s)-[r]->(t) // Use OPTIONAL MATCH for relationships in case graph is only nodes
    WHERE s IN nodes AND t IN nodes // Ensure rels use nodes already collected
    WITH nodes, collect(DISTINCT r) as rels // Collect distinct relationships
    RETURN
        [node IN nodes | { id: elementId(node), label: coalesce(labels(node)[0], 'Unknown'), properties: properties(node) }] AS nodes_list,
        [rel IN rels | { id: elementId(rel), from: elementId(startNode(rel)), to: elementId(endNode(rel)), label: type(rel), properties: properties(rel) }] AS edges_list
    """
    # Note: Added elementId(rel) as 'id' for edges and properties(rel) for edge properties. Added coalesce for labels.

    print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Executing APOC query on Neo4j...")
    try:
        with graph_driver.session(database="neo4j") as session: # Specify DB if needed
            result = session.run(apoc_query).single() # Expecting a single row result

            if result:
                nodes = result['nodes_list'] if result['nodes_list'] else []
                edges = result['edges_list'] if result['edges_list'] else []
                print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Query successful. Found {len(nodes)} nodes and {len(edges)} edges.")

                # Optional: Further validation or transformation if needed
                # for node in nodes:
                #     if node.get('label') is None: node['label'] = 'Unknown' # Already handled by coalesce

                return JSONResponse(status_code=200, content={"nodes": nodes, "edges": edges})
            else:
                # Handle case where the query returns no rows (e.g., graph is completely empty)
                print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Query returned no result row (graph might be empty).")
                return JSONResponse(status_code=200, content={"nodes": [], "edges": []})

    except Exception as e:
        # Catch Cypher errors or connection issues
        error_msg = f"Error fetching graph data from Neo4j: {e}"
        print(f"[ERROR] {datetime.now()}: {error_msg}")
        # Check if it's an APOC specific error (optional)
        if "apoc" in str(e).lower():
            print(f"[ERROR] {datetime.now()}: This might be due to APOC procedures not being installed/available in Neo4j.")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching graph data.")

## Notifications Endpoint
@app.get("/get-notifications")
async def get_notifications():
    """Retrieves all stored notifications."""
    print(f"[ENDPOINT /get-notifications] {datetime.now()}: Endpoint called.")
    async with notifications_db_lock:
        # print(f"[ENDPOINT /get-notifications] {datetime.now()}: Acquired notifications DB lock.")
        try:
            notifications_db = await load_notifications_db()
            notifications = notifications_db.get("notifications", [])
            print(f"[ENDPOINT /get-notifications] {datetime.now()}: Retrieved {len(notifications)} notifications.")
            return JSONResponse(content={"notifications": notifications})
        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Failed to load notifications DB: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to retrieve notifications.")
        # finally:
            # print(f"[ENDPOINT /get-notifications] {datetime.now()}: Released notifications DB lock.")

## Task Approval Endpoints
@app.post("/approve-task", status_code=200, response_model=ApproveTaskResponse)
async def approve_task(request: TaskIdRequest):
    """Approves a task pending approval, triggering its final execution step."""
    task_id = request.task_id
    print(f"[ENDPOINT /approve-task] {datetime.now()}: Endpoint called for task ID: {task_id}")
    try:
        # The approve_task method in TaskQueue should handle the execution
        print(f"[ENDPOINT /approve-task] {datetime.now()}: Calling task_queue.approve_task for {task_id}...")
        result_data = await task_queue.approve_task(task_id)
        print(f"[ENDPOINT /approve-task] {datetime.now()}: Task {task_id} approved and execution completed by TaskQueue.")
        # print(f"[ENDPOINT /approve-task] {datetime.now()}: Result from approve_task: {result_data}")

        # --- Add results to chat after approval ---
        task_details = await task_queue.get_task_by_id(task_id) # Get details like description, chat_id
        if task_details:
            chat_id = task_details.get("chat_id")
            description = task_details.get("description", "N/A")
            if chat_id:
                print(f"[ENDPOINT /approve-task] {datetime.now()}: Adding approved task result to chat {chat_id}.")
                # Add hidden user message (original prompt) - might already exist? Check TaskQueue logic.
                # await add_result_to_chat(chat_id, description, True)
                # Add visible assistant message (final result)
                await add_result_to_chat(chat_id, result_data, False, description)
            else:
                print(f"[WARN] {datetime.now()}: Could not find chat_id for completed task {task_id}. Cannot add result to chat.")
        else:
            print(f"[WARN] {datetime.now()}: Could not retrieve details for completed task {task_id} after approval.")

        # --- Broadcast Task Completion via WebSocket ---
        # Note: execute_agent_task -> complete_task *might* already broadcast completion
        # if approve_task triggers it internally. If approve_task *returns* the result
        # and we want to broadcast here, we do it. Let's assume we broadcast here for clarity.
        task_completion_message = {
            "type": "task_completed", # Task is now fully complete
            "task_id": task_id,
            "description": description if task_details else "Task Approved", # Use description if found
            "result": result_data
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task completion (after approval) for {task_id}")
        await manager.broadcast(json.dumps(task_completion_message))

        print(f"[ENDPOINT /approve-task] {datetime.now()}: Returning success response for task {task_id}.")
        return ApproveTaskResponse(message="Task approved and completed", result=result_data)

    except ValueError as e:
        # Specific error for task not found or wrong state (e.g., not pending approval)
        print(f"[ERROR] {datetime.now()}: Error approving task {task_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) # Use 404 or 400 depending on error
    except Exception as e:
        # General server error during approval/execution
        print(f"[ERROR] {datetime.now()}: Unexpected error approving task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during task approval: {str(e)}")

@app.post("/get-task-approval-data", status_code=200, response_model=TaskApprovalDataResponse)
async def get_task_approval_data(request: TaskIdRequest):
    """Gets the data associated with a task pending approval."""
    task_id = request.task_id
    print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Endpoint called for task ID: {task_id}")
    try:
        print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Accessing task queue data...")
        async with task_queue.lock: # Ensure thread-safe access if needed
            print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Acquired task queue lock.")
            task = next((t for t in task_queue.tasks if str(t.get("task_id")) == str(task_id)), None)

            if task and task.get("status") == "approval_pending":
                approval_data = task.get("approval_data")
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Found task {task_id} pending approval. Returning data.")
                # print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Approval data: {approval_data}")
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Releasing task queue lock.")
                return TaskApprovalDataResponse(approval_data=approval_data)
            elif task:
                status = task.get("status", "unknown")
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Task {task_id} found but status is '{status}', not 'approval_pending'.")
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Releasing task queue lock.")
                raise HTTPException(status_code=400, detail=f"Task '{task_id}' is not pending approval (status: {status}).")
            else:
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Task {task_id} not found in the queue.")
                print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Releasing task queue lock.")
                raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    except HTTPException as http_exc:
         # Ensure lock is released if exception occurs after acquiring it
         if task_queue.lock.locked():
              task_queue.lock.release()
              print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Released task queue lock due to HTTPException.")
         raise http_exc
    except Exception as e:
        # Catch unexpected errors during lookup
        if task_queue.lock.locked():
             task_queue.lock.release()
             print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Released task queue lock due to unexpected exception.")
        print(f"[ERROR] {datetime.now()}: Unexpected error fetching approval data for task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

## Data Source Configuration Endpoints
@app.get("/get_data_sources")
async def get_data_sources_endpoint(): # Renamed to avoid conflict
    """Return the list of available data sources and their enabled states."""
    print(f"[ENDPOINT /get_data_sources] {datetime.now()}: Endpoint called.")
    try:
        user_profile = load_user_profile().get("userData", {})
        data_sources_status = []
        for source in DATA_SOURCES:
            key = f"{source}Enabled"
            # Default to True if the key doesn't exist in the profile
            enabled = user_profile.get(key, True)
            data_sources_status.append({"name": source, "enabled": enabled})
            # print(f"[ENDPOINT /get_data_sources] {datetime.now()}:   - Source: {source}, Enabled: {enabled}")

        print(f"[ENDPOINT /get_data_sources] {datetime.now()}: Returning status for {len(data_sources_status)} data sources.")
        return JSONResponse(content={"data_sources": data_sources_status})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error fetching data source statuses: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get data source statuses.")

@app.post("/set_data_source_enabled")
async def set_data_source_enabled_endpoint(request: SetDataSourceEnabledRequest): # Renamed
    """Update the enabled state of a specific data source."""
    source = request.source
    enabled = request.enabled
    print(f"[ENDPOINT /set_data_source_enabled] {datetime.now()}: Endpoint called. Source: {source}, Enabled: {enabled}")

    if source not in DATA_SOURCES:
        print(f"[ERROR] {datetime.now()}: Invalid data source provided: {source}")
        raise HTTPException(status_code=400, detail=f"Invalid data source: {source}. Valid sources are: {DATA_SOURCES}")

    try:
        db_data = load_user_profile()
        if "userData" not in db_data: db_data["userData"] = {}

        key = f"{source}Enabled"
        db_data["userData"][key] = enabled
        print(f"[ENDPOINT /set_data_source_enabled] {datetime.now()}: Updated '{key}' to {enabled} in user profile data.")

        if write_user_profile(db_data):
            print(f"[ENDPOINT /set_data_source_enabled] {datetime.now()}: User profile saved successfully.")
             # TODO: Trigger restart/reload of the corresponding context engine if necessary
             # This might involve signaling the engine's background task or restarting the server
            print(f"[ACTION_NEEDED] {datetime.now()}: Context engine for '{source}' might need restart/reload to reflect changes.")
            return JSONResponse(status_code=200, content={"status": "success", "message": f"Data source '{source}' status set to {enabled}. Restart may be needed."})
        else:
            print(f"[ERROR] {datetime.now()}: Failed to write updated user profile to disk.")
            raise HTTPException(status_code=500, detail="Failed to update user profile file.")

    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error setting data source status for {source}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to set data source status: {str(e)}")

## WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for real-time updates."""
    await manager.connect(websocket)
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    print(f"[WEBSOCKET /ws] {datetime.now()}: Client connected from {client_host}:{client_port}")
    try:
        while True:
            # Keep connection alive and receive potential messages from client
            data = await websocket.receive_text()
            print(f"[WEBSOCKET /ws] {datetime.now()}: Received message from {client_host}:{client_port}: {data[:100]}...")
            # Process client message if needed (e.g., ping/pong, specific commands)
            # Example: await manager.send_personal_message(f"You sent: {data}", websocket)
            # For now, primarily server-to-client communication driven by other events.
            await websocket.send_text(json.dumps({"type": "ack", "message": "Message received"})) # Send acknowledgment

    except WebSocketDisconnect:
        print(f"[WEBSOCKET /ws] {datetime.now()}: Client {client_host}:{client_port} disconnected.")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: WebSocket error for {client_host}:{client_port}: {e}")
        traceback.print_exc()
        manager.disconnect(websocket) # Ensure cleanup on error

# --- Server Startup Time and Run ---

END_TIME = time.time()
STARTUP_DURATION = END_TIME - START_TIME
print(f"[STARTUP] {datetime.now()}: ==============================================")
print(f"[STARTUP] {datetime.now()}: Server initialization completed.")
print(f"[STARTUP] {datetime.now()}: Total startup time: {STARTUP_DURATION:.2f} seconds")
print(f"[STARTUP] {datetime.now()}: ==============================================")


if __name__ == "__main__":
    print(f"[RUN] {datetime.now()}: Starting Uvicorn server...")
    multiprocessing.freeze_support() # For Windows compatibility if using multiprocessing
    # Consider number of workers based on CPU cores, but start with 1 for easier debugging
    # reload=True is useful for development but should be False in production
    uvicorn.run(
        app, # Point to the FastAPI app instance in this file (main.py)
        host="0.0.0.0",
        port=5000,
        reload=False, # Set to True for auto-reload during development
        workers=1     # Start with 1 worker for simplicity/debugging
        # log_level="info" # Uvicorn's own logging level
    )
    print(f"[RUN] {datetime.now()}: Uvicorn server stopped.")