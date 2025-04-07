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
# from datetime import datetime, timezone # Keep timezone explicit - Redundant import removed
from tzlocal import get_localzone # Keep if needed elsewhere, but use explicit UTC for DB timestamps
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse # Added HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Added StaticFiles
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, AsyncGenerator, Union # Added Union
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import nest_asyncio
import uvicorn
import traceback # For detailed error printing
import numpy as np
import gradio as gr # Still needed for FastRTC internals
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
import httpx

print(f"[STARTUP] {datetime.now()}: Basic imports completed.")

# Import specific functions, runnables, and helpers from respective folders
print(f"[STARTUP] {datetime.now()}: Importing model components...")
from server.agents.runnables import *
from server.agents.functions import *
from server.agents.prompts import *
from server.agents.formats import *
from server.agents.base import *
from server.agents.helpers import *

from server.memory.runnables import *
from server.memory.functions import *
from server.memory.prompts import *
from server.memory.constants import *
from server.memory.formats import *
from server.memory.backend import MemoryBackend

from server.utils.helpers import *

from server.scraper.runnables import *
from server.scraper.functions import *
from server.scraper.prompts import *
from server.scraper.formats import *

from server.auth.helpers import *

from server.common.functions import *
from server.common.runnables import *
from server.common.prompts import *
from server.common.formats import *

from server.chat.runnables import *
from server.chat.prompts import *
from server.chat.functions import *

from server.context.gmail import GmailContextEngine
from server.context.internet import InternetSearchContextEngine
from server.context.gcalendar import GCalendarContextEngine

from server.voice.stt import FasterWhisperSTT
from server.voice.orpheus_tts import OrpheusTTS, TTSOptions, VoiceId, AVAILABLE_VOICES

from datetime import datetime

APPROVAL_PENDING_SIGNAL = "Task requires approval."
# from datetime import datetime, timezone # Redundant import removed

print(f"[STARTUP] {datetime.now()}: Model components import completed.")

# Define available data sources (can be extended in the future)
DATA_SOURCES = ["gmail", "internet_search", "gcalendar"]
print(f"[CONFIG] {datetime.now()}: Available data sources: {DATA_SOURCES}")
# Import new/refactored voice modules

# Load environment variables from .env file
print(f"[STARTUP] {datetime.now()}: Loading environment variables from server/.env...")
load_dotenv("server/.env")
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

# --- WebSocket Manager --- V1 (Used by most of the app)
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_ids: Dict[WebSocket, str] = {} # Optional: Track clients if needed
        print(f"[WS_MANAGER] {datetime.now()}: WebSocketManager initialized.")

    async def connect(self, websocket: WebSocket, client_id: str = "anon"):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_ids[websocket] = client_id
        print(f"[WS_MANAGER] {datetime.now()}: WebSocket client connected: {client_id} ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        client_id = self.client_ids.pop(websocket, "unknown")
        if websocket in self.active_connections:
             self.active_connections.remove(websocket)
        print(f"[WS_MANAGER] {datetime.now()}: WebSocket client disconnected: {client_id} ({len(self.active_connections)} total)")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
             await websocket.send_text(message)
        except Exception as e:
             print(f"[WS_MANAGER] {datetime.now()}: Error sending personal message to {self.client_ids.get(websocket, 'unknown')}: {e}")
             self.disconnect(websocket) # Disconnect on send error

    async def broadcast(self, message: str):
        # Create a list of connections to iterate over, as disconnect modifies the list
        connections_to_send = list(self.active_connections)
        disconnected_websockets = []
        for connection in connections_to_send:
            try:
                await connection.send_text(message)
            except Exception as e:
                client_id = self.client_ids.get(connection, "unknown")
                print(f"[WS_MANAGER] {datetime.now()}: Error broadcasting to {client_id}, marking for disconnect: {e}")
                # Mark for removal instead of removing directly while iterating
                disconnected_websockets.append(connection)

        # Remove broken connections outside the iteration loop
        for ws in disconnected_websockets:
            self.disconnect(ws)
        # print(f"[WS_MANAGER] {datetime.now()}: Broadcast finished. Active connections: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        await self.broadcast(json.dumps(data))

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

# Voice Model Placeholders & Config
stt_model = None
tts_model = None
SELECTED_TTS_VOICE: VoiceId = "tara"
if SELECTED_TTS_VOICE not in AVAILABLE_VOICES:
    print(f"Warning: Selected voice '{SELECTED_TTS_VOICE}' not valid. Using default 'tara'.")
    SELECTED_TTS_VOICE = "tara"
ORPHEUS_EMOTION_TAGS = [
    "<giggle>", "<laugh>", "<chuckle>", "<sigh>", "<cough>",
    "<sniffle>", "<groan>", "<yawn>", "<gasp>"
]
EMOTION_TAG_LIST_STR = ", ".join(ORPHEUS_EMOTION_TAGS)

#--- STT/TTS Model Loading (Moved to lifespan for potentially faster startup) ---
print("Loading STT model...")
stt_model = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
print("STT model loaded.")

print("Loading TTS model...")
try:
    tts_model = OrpheusTTS(
        verbose=False,
        default_voice_id=SELECTED_TTS_VOICE
    )
    print("TTS model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load TTS model: {e}")
    exit(1)
# --- End Voice Specific Initializations ---

# --- Database and State Management ---
# Database paths
print(f"[CONFIG] {datetime.now()}: Defining database file paths...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB = os.path.join(BASE_DIR, "..", "..", "userProfileDb.json")
CHAT_DB = "chatsDb.json"
NOTIFICATIONS_DB = "notificationsDb.json"
print(f"[CONFIG] {datetime.now()}:   - USER_PROFILE_DB: {USER_PROFILE_DB}")
print(f"[CONFIG] {datetime.now()}:   - CHAT_DB: {CHAT_DB}")
print(f"[CONFIG] {datetime.now()}:   - NOTIFICATIONS_DB: {NOTIFICATIONS_DB}")
print(f"[CONFIG] {datetime.now()}: Database file paths defined.")

# Locks
db_lock = asyncio.Lock()  # Lock for synchronizing chat database access
notifications_db_lock = asyncio.Lock() # Lock for notifications database access
print(f"[INIT] {datetime.now()}: Database locks initialized.")

# Initial DB Structure - Updated active_chat_id and next_chat_id
initial_db = {
    "chats": [],
    "active_chat_id": 0, # Start with 0 (no active chat)
    "next_chat_id": 1   # The ID for the *first* chat to be created
}
print(f"[CONFIG] {datetime.now()}: Initial chat DB structure defined (active_chat_id=0, next_chat_id=1).")

# Global variable for active chat ID (reflecting the DB state) - Start at 0
# This might be redundant if we always read from DB, but can be useful for quick checks
# Let's remove it to rely solely on the DB read for consistency.
# active_chat_id = 0
# print(f"[INIT] {datetime.now()}: Global active_chat_id set to 0.")

# --- Memory Backend Initialization ---
print(f"[INIT] {datetime.now()}: Initializing MemoryBackend...")
memory_backend = MemoryBackend()
print(f"[INIT] {datetime.now()}: MemoryBackend initialized. Performing cleanup...")
memory_backend.cleanup() # Ensure cleanup happens after initialization
print(f"[INIT] {datetime.now()}: MemoryBackend cleanup complete.")

# Tool Registration Decorator
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

# Auth0 configuration from utils
print(f"[CONFIG] {datetime.now()}: Setting up Auth0 configuration...")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")
print(f"[CONFIG] {datetime.now()}:   - AUTH0_DOMAIN: {AUTH0_DOMAIN}")
print(f"[CONFIG] {datetime.now()}:   - MANAGEMENT_CLIENT_ID: {'Set' if MANAGEMENT_CLIENT_ID else 'Not Set'}")
print(f"[CONFIG] {datetime.now()}:   - MANAGEMENT_CLIENT_SECRET: {'Set' if MANAGEMENT_CLIENT_SECRET else 'Not Set'}")
print(f"[CONFIG] {datetime.now()}: Auth0 configuration complete.")


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


async def load_db():
    """Load the chat database from chatsDb.json, initializing if it doesn't exist or is invalid."""
    # print(f"[DB_HELPER] {datetime.now()}: Attempting to load chat DB from {CHAT_DB}")
    try:
        with open(CHAT_DB, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate structure (with new defaults)
            if "chats" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'chats' key missing in {CHAT_DB}, adding.")
                data["chats"] = []
            if "active_chat_id" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'active_chat_id' key missing in {CHAT_DB}, setting to 0.")
                data["active_chat_id"] = 0 # Default to 0
            if "next_chat_id" not in data:
                print(f"[DB_HELPER] {datetime.now()}: 'next_chat_id' key missing in {CHAT_DB}, setting to 1.")
                data["next_chat_id"] = 1 # Default to 1
            # Ensure IDs are integers if loaded from old file
            if data.get("active_chat_id") is not None:
                try:
                    data["active_chat_id"] = int(data["active_chat_id"])
                except (ValueError, TypeError):
                     print(f"[DB_HELPER] {datetime.now()}: Invalid 'active_chat_id' found ('{data['active_chat_id']}'), resetting to 0.")
                     data["active_chat_id"] = 0
            if data.get("next_chat_id") is not None:
                try:
                     data["next_chat_id"] = int(data["next_chat_id"])
                except (ValueError, TypeError):
                     print(f"[DB_HELPER] {datetime.now()}: Invalid 'next_chat_id' found ('{data['next_chat_id']}'), resetting to 1.")
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
        # Ensure IDs are integers before saving
        if data.get("active_chat_id") is not None:
             data["active_chat_id"] = int(data["active_chat_id"])
        if data.get("next_chat_id") is not None:
             data["next_chat_id"] = int(data["next_chat_id"])
        for chat in data.get("chats", []):
             if chat.get("id") is not None:
                 chat["id"] = int(chat["id"])

        with open(CHAT_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # print(f"[DB_HELPER] {datetime.now()}: Chat DB saved successfully to {CHAT_DB}")
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error saving chat DB to {CHAT_DB}: {e}")


async def get_chat_history_messages() -> List[Dict[str, Any]]:
    """
    Retrieves the chat history of the currently active chat.
    Handles initial state (active_chat_id=0), inactivity, and creates new chats as needed.
    Chat IDs are now integers starting from 1. Active ID 0 means no chat is currently active.
    Returns the list of messages for the active chat, filtering out invisible messages.
    """
    print(f"[CHAT_HISTORY] {datetime.now()}: get_chat_history_messages called.")
    async with db_lock:
        print(f"[CHAT_HISTORY] {datetime.now()}: Acquired chat DB lock.")
        chatsDb = await load_db()
        active_chat_id = chatsDb.get("active_chat_id", 0)  # Default to 0 if missing
        next_chat_id = chatsDb.get("next_chat_id", 1)     # Default to 1 if missing
        current_time = datetime.now(timezone.utc)
        existing_chats = chatsDb.get("chats", [])
        active_chat = None

        print(f"[CHAT_HISTORY] {datetime.now()}: Current DB state - active_chat_id: {active_chat_id}, next_chat_id: {next_chat_id}, num_chats: {len(existing_chats)}")

        # --- Handle Active Chat ID ---
        if active_chat_id == 0:
            # No chat is currently active
            if not existing_chats:
                # No chats exist at all, create the first one (ID 1)
                new_chat_id = next_chat_id  # Should be 1 initially
                print(f"[CHAT_HISTORY] {datetime.now()}: No chats exist. Creating first chat with ID: {new_chat_id}.")
                new_chat = {"id": new_chat_id, "messages": []}
                chatsDb["chats"] = [new_chat]  # Initialize chats list
                chatsDb["active_chat_id"] = new_chat_id
                chatsDb["next_chat_id"] = new_chat_id + 1  # Increment next ID
                await save_db(chatsDb)
                print(f"[CHAT_HISTORY] {datetime.now()}: First chat (ID: {new_chat_id}) created and activated. DB saved.")
                return []  # Return empty messages for the new chat
            else:
                # Chats exist, but none is active (e.g., after clearing history and getting new message). Activate the latest.
                latest_chat_id = existing_chats[-1]['id']  # Get ID of the last chat in the list
                print(f"[CHAT_HISTORY] {datetime.now()}: Active chat ID is 0, but chats exist. Activating the latest chat (ID: {latest_chat_id}).")
                chatsDb['active_chat_id'] = latest_chat_id
                active_chat_id = latest_chat_id  # Update local variable
                await save_db(chatsDb)
                print(f"[CHAT_HISTORY] {datetime.now()}: Activated latest chat (ID: {latest_chat_id}). DB saved.")
                # Proceed to find this chat below
        # else: active_chat_id is > 0

        # --- Find the Active Chat Object ---
        # (Could be set above if ID was 0, or loaded directly if ID was > 0)
        active_chat = next((chat for chat in existing_chats if chat.get("id") == active_chat_id), None)

        # Handle case where active_chat_id exists but the chat itself doesn't (data inconsistency)
        if not active_chat:
            print(f"[ERROR] {datetime.now()}: Active chat ID '{active_chat_id}' exists, but no corresponding chat found. Resetting active chat to 0.")
            chatsDb["active_chat_id"] = 0  # Reset to avoid repeated errors
            await save_db(chatsDb)
            # Return empty list; next call will try to activate latest or create new
            return []

        # --- Check for Inactivity (only if chat has messages) ---
        if active_chat.get("messages"):
            try:
                last_message = active_chat["messages"][-1]
                timestamp_str = last_message.get("timestamp")

                if not timestamp_str:
                    print(f"[CHAT_HISTORY] {datetime.now()}: Warning: Last message in chat {active_chat_id} has no timestamp. Skipping inactivity check.")
                else:
                    # Preprocess timestamp to handle invalid format (e.g., '+00:00Z')
                    if timestamp_str.endswith('Z') and ('+' in timestamp_str or '-' in timestamp_str):
                        timestamp_str = timestamp_str[:-1]  # Remove 'Z' if offset exists

                    try:
                        last_timestamp = datetime.fromisoformat(timestamp_str)
                        if last_timestamp.tzinfo is None:
                            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

                        # Check inactivity (e.g., 10 minutes)
                        inactivity_threshold = 600  # seconds
                        if (current_time - last_timestamp).total_seconds() > inactivity_threshold:
                            print(f"[CHAT_HISTORY] {datetime.now()}: Inactivity detected in chat {active_chat_id} (>{inactivity_threshold}s). Creating new chat.")
                            new_chat_id = next_chat_id
                            print(f"[CHAT_HISTORY] {datetime.now()}: Creating new chat due to inactivity with ID: {new_chat_id}.")
                            new_chat = {"id": new_chat_id, "messages": []}
                            chatsDb["chats"].append(new_chat)
                            chatsDb["active_chat_id"] = new_chat_id
                            chatsDb["next_chat_id"] = new_chat_id + 1
                            await save_db(chatsDb)
                            print(f"[CHAT_HISTORY] {datetime.now()}: New chat (ID: {new_chat_id}) created and activated due to inactivity. DB saved.")
                            return []  # Return empty messages for the new chat
                        # else: Not inactive, continue with current active chat

                    except ValueError as e_parse:
                        print(f"[CHAT_HISTORY] {datetime.now()}: Error parsing timestamp '{timestamp_str}' in chat {active_chat_id}: {e_parse}. Skipping inactivity check.")
                    except Exception as e_time:
                        print(f"[CHAT_HISTORY] {datetime.now()}: Error during timestamp comparison in chat {active_chat_id}: {e_time}. Skipping inactivity check.")

            except IndexError:
                print(f"[CHAT_HISTORY] {datetime.now()}: Chat {active_chat_id} has an empty messages list, cannot check inactivity.")
            except Exception as e:
                print(f"[CHAT_HISTORY] {datetime.now()}: Unexpected error during inactivity check for chat {active_chat_id}: {e}. Proceeding...")

        # --- Return Visible Messages from the Active Chat ---
        if active_chat and active_chat.get("messages"):
            filtered_messages = [
                message for message in active_chat["messages"]
                if message.get("isVisible", True) is not False  # Default isVisible to True if missing
            ]
            print(f"[CHAT_HISTORY] {datetime.now()}: Returning {len(filtered_messages)} visible messages for active chat {active_chat_id}.")
            return filtered_messages
        else:
            # Active chat exists but has no messages (or messages list is missing)
            print(f"[CHAT_HISTORY] {datetime.now()}: Active chat {active_chat_id} exists but has no messages. Returning empty list.")
            return []


# WebSocket Manager V2 (Duplicate from above, remove one if not needed)
# Removing the second definition as the first one (`manager`) is used globally.
# class WebSocketManager: ... (Removed duplicate definition)

async def add_message_to_db(chat_id: Union[int, str], message_text: str, is_user: bool, is_visible: bool = True, **kwargs):
    """Adds a message to the specified chat ID (now integer) in the database."""
    try:
        target_chat_id = int(chat_id)
    except (ValueError, TypeError):
        print(f"[ERROR] {datetime.now()}: Invalid chat_id format provided to add_message_to_db: '{chat_id}'. Expected integer.")
        return None

    async with db_lock:
        try:
            chatsDb = await load_db()
            active_chat = next((chat for chat in chatsDb["chats"] if chat.get("id") == target_chat_id), None)

            if active_chat:
                message_id = str(int(time.time() * 1000))
                new_message = {
                    "id": message_id,
                    "message": message_text,
                    "isUser": is_user,
                    "isVisible": is_visible,
                    # MODIFIED: Removed the redundant "+ "Z"" at the end
                    "timestamp": datetime.now(timezone.utc).isoformat(), # .isoformat() on timezone-aware datetime includes offset
                    "memoryUsed": kwargs.get("memoryUsed", False),
                    "agentsUsed": kwargs.get("agentsUsed", False),
                    "internetUsed": kwargs.get("internetUsed", False),
                }
                if kwargs.get("type"):
                    new_message["type"] = kwargs["type"]
                if kwargs.get("task"):
                    new_message["task"] = kwargs["task"]

                if "messages" not in active_chat:
                     active_chat["messages"] = []

                active_chat["messages"].append(new_message)
                await save_db(chatsDb)
                print(f"Message added to DB (Chat ID: {target_chat_id}, User: {is_user}): {message_text}...")
                return message_id # Return the ID if needed
            else:
                print(f"Error: Could not find chat with ID {target_chat_id} to add message.")
                return None
        except Exception as e:
            print(f"Error adding message to DB: {e}")
            traceback.print_exc()
            return None

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
        task = await task_queue.get_next_task()
        if task:
            task_id = task.get("task_id", "N/A")
            task_desc = task.get("description", "N/A")
            task_chat_id = task.get("chat_id", "N/A")
            print(f"[TASK_PROCESSOR] {datetime.now()}: Processing task ID: {task_id}, Chat ID: {task_chat_id}, Description: {task_desc}...")
            try:
                # Execute task within a new asyncio task to handle cancellations
                task_queue.current_task_execution = asyncio.create_task(execute_agent_task(task))
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution started.")
                result = await task_queue.current_task_execution
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution finished.") # Removed 'successfully' temporarily

                # --- Handle Approval Pending vs. Normal Completion ---
                if result == APPROVAL_PENDING_SIGNAL:
                    # Task requires approval, do not complete it here.
                    # execute_agent_task already set status to pending_approval and broadcasted.
                    print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} is now pending user approval. No further action in this loop.")
                    # The task remains 'processing' in the queue's eyes until approved/rejected/timed out,
                    # but the execute_agent_task set its *status* to 'pending_approval'.
                    # No complete_task call here.

                else:
                    # Task completed normally (or with an error handled within execute_agent_task returning a string)
                    print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} completed without needing approval. Result length: {len(str(result))}")

                    # Add results to chat (use task's chat_id)
                    if task_chat_id != "N/A":
                        print(f"[TASK_PROCESSOR] {datetime.now()}: Adding task description '{task_desc}...' to chat {task_chat_id} as user message (hidden).")
                        await add_message_to_db(task_chat_id, task["description"], is_user=True, is_visible=False)
                        print(f"[TASK_PROCESSOR] {datetime.now()}: Adding task result to chat {task_chat_id} as assistant message.")
                        await add_message_to_db(task_chat_id, result, is_user=False, is_visible=True, type="tool_result", task=task["description"], agentsUsed=True)
                    else:
                        print(f"[WARN] {datetime.now()}: Task {task_id} has no associated chat_id. Cannot add result to chat.")

                    # Mark task as completed in the queue
                    await task_queue.complete_task(task_id, result=result) # Status defaults to 'completed'
                    print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} marked as completed in queue.")

                    # --- WebSocket Message on Success ---
                    # This broadcast should ideally be handled within task_queue.complete_task
                    # to ensure consistency, especially when completion happens via approve_task.
                    # If task_queue.complete_task doesn't broadcast, uncomment and keep this block.
                    # task_completion_message = {
                    #     "type": "task_completed",
                    #     "task_id": task_id,
                    #     "description": task_desc,
                    #     "result": result
                    # }
                    # print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task completion for {task_id}")
                    # await manager.broadcast(json.dumps(task_completion_message))


            except asyncio.CancelledError:
                print(f"[TASK_PROCESSOR] {datetime.now()}: Task {task_id} execution was cancelled.")
                await task_queue.complete_task(task_id, error="Task was cancelled", status="cancelled")
                # --- WebSocket Message on Cancellation ---
                # Again, ideally handled by complete_task based on status. If not:
                task_error_message = {
                    "type": "task_cancelled", # Specific type for cancellation
                    "task_id": task_id,
                    "description": task_desc,
                    "error": "Task was cancelled"
                }
                print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task cancellation for {task_id}")
                await manager.broadcast(json.dumps(task_error_message))

            except Exception as e:
                error_str = str(e)
                print(f"[ERROR] {datetime.now()}: Error processing task {task_id}: {error_str}")
                traceback.print_exc()
                await task_queue.complete_task(task_id, error=error_str, status="error")
                # --- WebSocket Message on Error ---
                # Ideally handled by complete_task based on status. If not:
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
    """
    Execute the agent task asynchronously. Returns the final result string
    or APPROVAL_PENDING_SIGNAL if user approval is required.
    Handles setting the task status to 'pending_approval' and broadcasting.
    """
    task_id = task.get("task_id", "N/A")
    task_desc = task.get("description", "N/A")
    print(f"[AGENT_EXEC] {datetime.now()}: Executing task ID: {task_id}, Description: {task_desc}...")
    # ... (rest of the setup: globals, context computation - remains the same) ...
    global agent_runnable, reflection_runnable, inbox_summarizer_runnable, graph_driver, embed_model
    global text_conversion_runnable, query_classification_runnable, internet_query_reframe_runnable, internet_summary_runnable

    transformed_input = task["description"]
    username = task["username"]
    personality = task["personality"]
    use_personal_context = task["use_personal_context"]
    internet = task["internet"]

    user_context = None
    internet_context = None

    # --- Compute User Context --- (No changes needed here)
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
            user_context = f"Error retrieving user context: {e}"

    # --- Compute Internet Context --- (No changes needed here)
    if internet:
        print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires internet search.")
        try:
            print(f"[AGENT_EXEC] {datetime.now()}: Re-framing internet query for task {task_id}...")
            reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
            print(f"[AGENT_EXEC] {datetime.now()}: Reframed query for task {task_id}: '{reframed_query}'")

            print(f"[AGENT_EXEC] {datetime.now()}: Performing internet search for task {task_id}...")
            search_results = get_search_results(reframed_query)

            print(f"[AGENT_EXEC] {datetime.now()}: Summarizing search results for task {task_id}...")
            internet_context = get_search_summary(internet_summary_runnable, search_results)
            print(f"[AGENT_EXEC] {datetime.now()}: Internet context summary generated for task {task_id}. Length: {len(str(internet_context)) if internet_context else 0}")

        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Error computing internet_context for task {task_id}: {e}")
            traceback.print_exc()
            internet_context = f"Error retrieving internet context: {e}"
    else:
         print(f"[AGENT_EXEC] {datetime.now()}: Internet search not required for task {task_id}.")


    # --- Invoke Agent Runnable --- (No changes needed here)
    print(f"[AGENT_EXEC] {datetime.now()}: Invoking main agent runnable for task {task_id}...")
    agent_input = {
        "query": transformed_input,
        "name": username,
        "user_context": user_context,
        "internet_context": internet_context,
        "personality": personality
    }
    try:
        response = agent_runnable.invoke(agent_input)
        print(f"[AGENT_EXEC] {datetime.now()}: Agent response received for task {task_id}.")
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error invoking agent_runnable for task {task_id}: {e}")
        traceback.print_exc()
        return f"Error during agent execution: {e}" # Return error string

    # --- Process Tool Calls --- (Main changes here for approval)
    tool_calls = []
    if isinstance(response, dict) and "tool_calls" in response and isinstance(response["tool_calls"], list):
        tool_calls = response["tool_calls"]
        print(f"[AGENT_EXEC] {datetime.now()}: Found {len(tool_calls)} tool calls in agent response dict.")
    elif isinstance(response, list):
        tool_calls = response
        print(f"[AGENT_EXEC] {datetime.now()}: Agent response was a list, treating as {len(tool_calls)} tool calls.")
    else:
        if isinstance(response, str):
             print(f"[AGENT_EXEC] {datetime.now()}: Agent response is a direct string answer (no tool calls).")
             return response # Return the direct answer
        error_msg = f"Error: Invalid or missing 'tool_calls' list in agent response for task {task_id}. Response type: {type(response)}, Response: {str(response)[:200]}"
        print(f"[AGENT_EXEC] {datetime.now()}: {error_msg}")
        return error_msg

    all_tool_results = []
    previous_tool_result = None
    print(f"[AGENT_EXEC] {datetime.now()}: Processing {len(tool_calls)} potential tool calls for task {task_id}.")

    for i, tool_call in enumerate(tool_calls):
        # ... (Validation of tool_call structure - remains the same) ...
        tool_content = None
        if isinstance(tool_call, dict) and tool_call.get("response_type") == "tool_call":
            tool_content = tool_call.get("content")
        elif isinstance(tool_call, dict) and "tool_name" in tool_call and "task_instruction" in tool_call:
            tool_content = tool_call
            print(f"[AGENT_EXEC] {datetime.now()}: Interpreting dict as direct tool call content.")
        else:
            print(f"[AGENT_EXEC] {datetime.now()}: Skipping item {i+1} as it's not a valid tool call structure. Item: {str(tool_call)[:100]}")
            continue

        if not isinstance(tool_content, dict):
             print(f"[AGENT_EXEC] {datetime.now()}: Skipping tool call {i+1} due to invalid 'content' structure. Content: {tool_content}")
             continue

        tool_name = tool_content.get("tool_name")
        task_instruction = tool_content.get("task_instruction")
        previous_tool_response_required = tool_content.get("previous_tool_response", False)

        if not tool_name or not task_instruction:
            print(f"[AGENT_EXEC] {datetime.now()}: Skipping tool call {i+1} due to missing tool_name or task_instruction.")
            continue

        print(f"[AGENT_EXEC] {datetime.now()}:   - Tool Name: {tool_name}")
        print(f"[AGENT_EXEC] {datetime.now()}:   - Task Instruction: {str(task_instruction)[:100]}...")
        print(f"[AGENT_EXEC] {datetime.now()}:   - Previous Tool Response Required: {previous_tool_response_required}")

        tool_handler = tool_handlers.get(tool_name)
        if not tool_handler:
            error_msg = f"Error: Tool '{tool_name}' not found in registered handlers for task {task_id}."
            print(f"[AGENT_EXEC] {datetime.now()}: {error_msg}")
            all_tool_results.append({"tool_name": tool_name, "task_instruction": task_instruction, "tool_result": error_msg, "status": "error"})
            continue

        # --- Prepare and Execute Tool Handler ---
        tool_input = {"input": str(task_instruction)}
        if previous_tool_response_required:
             if previous_tool_result:
                 print(f"[AGENT_EXEC] {datetime.now()}:   - Providing previous tool result to '{tool_name}'.")
                 tool_input["previous_tool_response"] = previous_tool_result
             else:
                 print(f"[WARN] {datetime.now()}: Tool '{tool_name}' requires previous result, but none is available. Passing 'None'.")
                 tool_input["previous_tool_response"] = "Previous tool result was expected but not available."

        print(f"[AGENT_EXEC] {datetime.now()}: Invoking tool handler '{tool_handler.__name__}' for tool '{tool_name}'...")
        try:
            tool_result_main = await tool_handler(tool_input)
            print(f"[AGENT_EXEC] {datetime.now()}: Tool handler '{tool_handler.__name__}' executed.")
        except Exception as e:
            error_msg = f"Error executing tool handler '{tool_handler.__name__}' for tool '{tool_name}': {e}"
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            traceback.print_exc()
            all_tool_results.append({"tool_name": tool_name, "task_instruction": str(task_instruction), "tool_result": error_msg, "status": "error"})
            previous_tool_result = error_msg
            continue

        # --- <<< MODIFICATION START >>> Handle Approval Flow ---
        if isinstance(tool_result_main, dict) and tool_result_main.get("action") == "approve":
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires approval for tool '{tool_name}'. Setting task to pending.")
            approval_data = tool_result_main.get("tool_call", {})
            # Set the task status to 'pending_approval' in the queue
            await task_queue.set_task_approval_pending(task_id, approval_data)

            # --- WebSocket Notification for Approval ---
            notification = {
                "type": "task_approval_pending",
                "task_id": task_id,
                "description": f"Approval needed for: {tool_name} - {str(task_instruction)}...",
                "tool_name": tool_name,
                "approval_data": approval_data
            }
            print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task approval pending for {task_id}")
            await manager.broadcast(json.dumps(notification))

            # Return the specific signal to process_queue
            return APPROVAL_PENDING_SIGNAL
        # --- <<< MODIFICATION END >>> Handle Approval Flow ---

        # --- Store Normal Tool Result ---
        else:
            if isinstance(tool_result_main, dict) and "tool_result" in tool_result_main:
                tool_result = tool_result_main["tool_result"]
            else:
                tool_result = tool_result_main

            print(f"[AGENT_EXEC] {datetime.now()}: Tool '{tool_name}' executed successfully. Storing result.")
            previous_tool_result = tool_result
            all_tool_results.append({
                "tool_name": tool_name,
                "task_instruction": str(task_instruction),
                "tool_result": tool_result,
                "status": "success"
            })

    # --- Final Reflection/Summarization --- (No changes needed here, runs only if no approval was needed)
    if not all_tool_results:
        print(f"[AGENT_EXEC] {datetime.now()}: No successful tool calls executed for task {task_id}.")
        if isinstance(response, dict) and response.get("response_type") == "final_answer" and response.get("content"):
             print(f"[AGENT_EXEC] {datetime.now()}: Using agent's final answer from initial response dict.")
             return response["content"]
        else:
             print(f"[AGENT_EXEC] {datetime.now()}: No tools run and no direct final answer from agent for task {task_id}. Returning generic message.")
             return "No specific actions were taken or information gathered."


    print(f"[AGENT_EXEC] {datetime.now()}: All tool calls processed for task {task_id}. Preparing final result.")
    final_result_str = "No final result generated."

    try:
        # Special handling for inbox search summarization
        if len(all_tool_results) == 1 and all_tool_results[0].get("tool_name") == "search_inbox" and all_tool_results[0].get("status") == "success":
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} involved only 'search_inbox'. Invoking inbox summarizer...")
            tool_result_data = all_tool_results[0]["tool_result"]
            result_content = None
            if isinstance(tool_result_data, dict):
                if "result" in tool_result_data and isinstance(tool_result_data["result"], dict):
                    result_content = tool_result_data["result"]
                elif "email_data" in tool_result_data:
                    result_content = tool_result_data

            if result_content:
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
                final_result_str = inbox_summarizer_runnable.invoke({"tool_result": filtered_tool_result})
                print(f"[AGENT_EXEC] {datetime.now()}: Inbox summarizer finished for task {task_id}.")
            else:
                print(f"[WARN] {datetime.now()}: 'search_inbox' result format unexpected for summarization. Falling back to reflection. Result: {tool_result_data}")
                print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires reflection on tool results.")
                final_result_str = reflection_runnable.invoke({"tool_results": all_tool_results})
                print(f"[AGENT_EXEC] {datetime.now()}: Reflection finished for task {task_id}.")
        else:
            print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} requires reflection on multiple/different tool results.")
            final_result_str = reflection_runnable.invoke({"tool_results": all_tool_results})
            print(f"[AGENT_EXEC] {datetime.now()}: Reflection finished for task {task_id}.")

    except Exception as e:
        error_msg = f"Error during final result generation (reflection/summarization) for task {task_id}: {e}"
        print(f"[ERROR] {datetime.now()}: {error_msg}")
        traceback.print_exc()
        final_result_str = f"{error_msg}\n\nRaw Tool Results:\n{json.dumps(all_tool_results, indent=2)}"

    print(f"[AGENT_EXEC] {datetime.now()}: Task {task_id} execution complete. Final result length: {len(final_result_str)}")
    return final_result_str # Return the actual result string


async def add_result_to_chat(chat_id: Union[int, str], result: str, isUser: bool, task_description: str = None):
    """
    Add the task result or hidden user message to the corresponding chat.
    This function seems redundant as add_message_to_db handles this logic now.
    Refactor calls to use add_message_to_db directly with appropriate `isVisible` flag.
    """
    print(f"[DEPRECATED] {datetime.now()}: add_result_to_chat called. Use add_message_to_db instead.")
    is_visible = not isUser # User messages (task descriptions) should be hidden, assistant results visible
    await add_message_to_db(chat_id, result, is_user=isUser, is_visible=is_visible, task=task_description)


# --- FastAPI Application Setup ---
print(f"[FASTAPI] {datetime.now()}: Initializing FastAPI app...")
app = FastAPI(
    title="Sentient API",
    description="Monolithic API for the Sentient AI companion",
    docs_url="/docs",
    redoc_url=None # Disable Redoc if not needed
)
print(f"[FASTAPI] {datetime.now()}: FastAPI app initialized.")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "app://.",
]

# Add CORS middleware to allow cross-origin requests
print(f"[FASTAPI] {datetime.now()}: Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] + origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"]  # Allow all headers
)
print(f"[FASTAPI] {datetime.now()}: CORS middleware added.")

# --- Startup and Shutdown Events ---@app.on_event("startup")
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
    await task_queue.save_tasks()
    await memory_backend.memory_queue.save_operations()

# --- Pydantic Models ---
# (No print statements needed in model definitions)
class Message(BaseModel):
    input: str
    pricing: str
    credits: int

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
    # chat_id: Union[int, str] # Removed, will be determined dynamically
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

# --- Voice Chat Logic ---

# --- Configuration --- (Already defined globally)

# --- Async Handler using Chat Endpoint Logic ---
async def handle_audio_conversation(audio: tuple[int, np.ndarray]) -> AsyncGenerator[tuple[int, np.ndarray], None]:
    """
    Handles the voice conversation flow: STT -> LLM (using chat logic) -> TTS.
    Uses the same classification, context retrieval, and runnable invocation
    as the /chat endpoint, calling invoke() for the full response.
    Ensures async functions are awaited. Yields audio chunks.
    """
    # Make runnables and models accessible
    global embed_model, graph_driver, memory_backend, task_queue, stt_model, tts_model
    global reflection_runnable, inbox_summarizer_runnable, priority_runnable
    global graph_decision_runnable, information_extraction_runnable, graph_analysis_runnable
    global text_dissection_runnable, text_conversion_runnable, query_classification_runnable
    global fact_extraction_runnable, text_summarizer_runnable, text_description_runnable
    global reddit_runnable, twitter_runnable
    global internet_query_reframe_runnable, internet_summary_runnable
    global chat_runnable, agent_runnable, unified_classification_runnable # Core runnables

    print("\n--- Received audio chunk for processing ---")
    print("Transcribing...")
    if not stt_model:
        print("ERROR: STT model not loaded.")
        return
    user_text = stt_model.stt(audio)
    if not user_text or not user_text.strip():
        print("No valid text transcribed, skipping.")
        return

    print(f"User (STT): {user_text}")

    bot_response_text = ""
    active_chat_id = 0 # Start assuming no chat
    current_chat_runnable = None # Initialize

    try:
        # 1. Load User Profile & Determine Active Chat Info
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        personality_setting = user_profile.get("userData", {}).get("personality", "Default helpful assistant")

        # Get current active chat ID (will create/activate if needed)
        # This ensures active_chat_id is set correctly before proceeding
        await get_chat_history_messages() # Call this primarily for its side effect of setting active_chat_id
        async with db_lock: # Reload DB to get the *potentially updated* active_chat_id
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id", 0)

        if active_chat_id == 0: # Should not happen if get_chat_history_messages worked
            print("ERROR: Failed to determine or create an active chat ID. Cannot process voice message.")
            return

        print(f"Voice message will be added to active chat ID: {active_chat_id}")

        # 2. Add User Message to DB
        await add_message_to_db(active_chat_id, user_text, is_user=True, is_visible=True)

        # 3. Get Current Chat History & Initialize Context-Aware Runnables
        # It's crucial these use the history associated with the *active_chat_id*
        # get_chat_history() should implicitly handle this if it manages history correctly
        chat_history_obj = get_chat_history()
        try:
            # Re-initialize runnables with the potentially updated history context
            current_chat_runnable = get_chat_runnable(chat_history_obj)
            current_agent_runnable = get_agent_runnable(chat_history_obj)
            current_unified_classifier = get_unified_classification_runnable(chat_history_obj)

            # Basic validation
            if not hasattr(current_chat_runnable, 'invoke') or not callable(current_chat_runnable.invoke):
                 print(f"ERROR: get_chat_runnable did not return a valid runnable object with 'invoke'. Got type: {type(current_chat_runnable)}")
                 raise ValueError("Invalid chat runnable object received (missing invoke).")
            print("History-dependent runnables initialized/verified.")
        except Exception as e:
            print(f"ERROR initializing history-dependent runnables: {e}")
            traceback.print_exc()
            await add_message_to_db(active_chat_id, "[System Error: Failed to initialize core components]", is_user=False, error=True, is_visible=True)
            return

        # 4. Classify User Input
        print("Classifying input...")
        unified_output = current_unified_classifier.invoke({"query": user_text})
        category = unified_output.get("category", "chat")
        use_personal_context = unified_output.get("use_personal_context", False)
        internet = unified_output.get("internet", "None")
        transformed_input = unified_output.get("transformed_input", user_text)
        print(f"Classification: Category='{category}', Use Personal='{use_personal_context}', Internet='{internet}'")
        print(f"Transformed Input: {transformed_input}")


        # 5. Handle Agent Task Category
        if category == "agent":
            print("Input classified as agent task.")
            priority_response = priority_runnable.invoke({"task_description": transformed_input})
            priority = priority_response.get("priority", 3)

            print(f"Adding agent task to queue (Priority: {priority})...")
            await task_queue.add_task(
                chat_id=active_chat_id, # Use the determined active chat ID
                description=transformed_input,
                priority=priority,
                username=username,
                personality=personality_setting,
                use_personal_context=use_personal_context,
                internet=internet
            )
            print("Agent task added.")
            bot_response_text = "Okay, I'll get right on that." # Canned response for agent task

            # Add assistant confirmation to DB (visible)
            await add_message_to_db(active_chat_id, transformed_input, is_user=True, is_visible=False)
            await add_message_to_db(active_chat_id, bot_response_text, is_user=False, agentsUsed=True, is_visible=True)
            # Add hidden user message (task description) to DB
            

        # 6. Handle Chat/Memory Category (Generate Response using invoke)
        else: # category is "chat" or "memory"
            print("Input classified as chat/memory.")
            user_context = None
            internet_context = None
            memory_used_flag = False
            internet_used_flag = False

            # Fetch context if needed (Memory)
            if use_personal_context or category == "memory":
                print("Retrieving relevant memories...")
                try:
                    user_context = await memory_backend.retrieve_memory(username, transformed_input)
                    if user_context:
                        if isinstance(user_context, str):
                            print(f"Retrieved user context (memory): {user_context[:100]}...")
                        else:
                            print(f"Retrieved user context (memory) of type: {type(user_context)}")
                        memory_used_flag = True
                    else:
                        print("No relevant memories found.")
                except Exception as e:
                    print(f"Error retrieving user context: {e}")
                    traceback.print_exc()

                if category == "memory": # Queue update only if explicitly memory category
                    print("Queueing memory update operation...")
                    # Run in background, don't await here
                    asyncio.create_task(memory_backend.add_operation(username, transformed_input))

            # Fetch context if needed (Internet)
            if internet:
                print("Searching the internet...")
                try:
                    reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                    print(f"Reframed internet query: {reframed_query}")
                    search_results = get_search_results(reframed_query)
                    if search_results:
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        if internet_context and isinstance(internet_context, str):
                            print(f"Retrieved internet context (summary): {internet_context[:100]}...")
                        elif internet_context:
                            print(f"Retrieved internet context (summary) of type: {type(internet_context)}")
                        else:
                            print("Internet summary was empty.")
                        internet_used_flag = True
                    else:
                        print("No search results found or summary failed.")
                except Exception as e:
                    print(f"Error retrieving internet context: {e}")
                    traceback.print_exc()

            # Generate Response using invoke
            print("Generating response using chat runnable's invoke method...")
            try:
                # Ensure runnable is valid before calling
                if not current_chat_runnable or not hasattr(current_chat_runnable, 'invoke') or not callable(current_chat_runnable.invoke):
                     print(f"ERROR: current_chat_runnable is invalid before invoking. Value: {current_chat_runnable}")
                     raise ValueError("Chat runnable became invalid before invoke.")

                invocation_result = current_chat_runnable.invoke({
                    "query": transformed_input,
                    "user_context": user_context,
                    "internet_context": internet_context,
                    "name": username,
                    "personality": personality_setting
                })

                # Process the result
                if isinstance(invocation_result, str):
                    bot_response_text = invocation_result.strip()
                elif invocation_result is None:
                     print("Warning: Chat runnable invoke returned None.")
                     bot_response_text = "I couldn't generate a response for that."
                else:
                    print(f"Warning: Unexpected response type from invoke: {type(invocation_result)}. Converting to string.")
                    bot_response_text = str(invocation_result).strip()

                if not bot_response_text:
                    print("Warning: Generated response was empty.")
                    bot_response_text = "I don't have a specific response for that right now."

                print(f"Generated response text (invoke): {bot_response_text[:100]}...")

                # Add the final bot response to the DB (visible)
                await add_message_to_db(
                    active_chat_id,
                    bot_response_text,
                    is_user=False,
                    memoryUsed=memory_used_flag,
                    internetUsed=internet_used_flag,
                    is_visible=True # Explicitly visible
                )

            except Exception as e:
                print(f"Error during chat runnable invocation: {e}")
                traceback.print_exc()
                bot_response_text = "Sorry, I encountered an error while generating a response."
                # Add error message to DB (visible)
                await add_message_to_db(active_chat_id, bot_response_text, is_user=False, error=True, is_visible=True)

        # 7. Synthesize Speech (TTS)
        if not bot_response_text:
            print("No bot response text generated (or error occurred), skipping TTS.")
            return

        if not tts_model:
            print("ERROR: TTS model not loaded.")
            return

        print(f"Synthesizing speech for: {bot_response_text[:60]}...")
        try:
            tts_options: TTSOptions = {}
            chunk_count = 0
            # Use async for to handle the async generator from TTS
            async for sample_rate, audio_chunk in tts_model.stream_tts(bot_response_text, options=tts_options):
                if audio_chunk is not None and audio_chunk.size > 0:
                    if chunk_count == 0:
                         print(f"Streaming TTS audio chunk 1 (sr={sample_rate}, shape={audio_chunk.shape}, dtype={audio_chunk.dtype})...")
                    yield (sample_rate, audio_chunk) # Yield chunk to FastRTC
                    chunk_count += 1
                else:
                    # This might happen at the end of the stream or if there's an issue
                    print("Warning: Received empty audio chunk from TTS stream.")
            if chunk_count == 0:
                 print("Warning: TTS stream completed without yielding any audio chunks.")
            else:
                 print(f"Finished synthesizing and streaming {chunk_count} chunks.")

        except Exception as e:
            print(f"Error during TTS synthesis or streaming: {e}")
            traceback.print_exc()
            # Don't yield anything if TTS fails

    except Exception as e:
        print(f"--- Unhandled error in handle_audio_conversation: {e} ---")
        traceback.print_exc()
        if active_chat_id and active_chat_id != 0: # Check if chat ID was determined
            try:
                await add_message_to_db(active_chat_id, "[System Error in Voice Processing]", is_user=False, error=True, is_visible=True)
            except Exception as db_err:
                 print(f"Failed to log system error to DB: {db_err}")
        # Don't yield anything on major error


# --- Set up the Stream (Handler remains the same) ---
stream = Stream(
    ReplyOnPause(
        handle_audio_conversation,
        # Keep VAD settings as they worked in Gradio
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.25,
            speech_threshold=0.2
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=200,
            min_silence_duration_ms=1000 # Adjust silence duration if needed
        ),
        can_interrupt=False, # Keep interruption disabled for now
    ),
    mode="send-receive",
    modality="audio",
    additional_outputs=None,
    additional_outputs_handler=None
)

# Mount FastRTC stream AFTER FastAPI app is initialized
# stream.mount(app, path="/voice")
# print("FastRTC stream mounted at /voice.")
# Delay mounting until after lifespan setup to ensure models are loaded
# Mounting moved after lifespan context manager attachment

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
    Endpoint to retrieve the chat history for the currently active chat.
    Calls get_chat_history_messages which handles activation/creation.
    """
    print(f"[ENDPOINT /get-history] {datetime.now()}: Endpoint called.")
    try:
        messages = await get_chat_history_messages()
        # print(f"[ENDPOINT /get-history] {datetime.now()}: Retrieved {len(messages)} messages for active chat.")

        # Also return the current active chat ID to the frontend
        async with db_lock:
            chatsDb = await load_db()
            current_active_chat_id = chatsDb.get("active_chat_id", 0)

        return JSONResponse(status_code=200, content={
            "messages": messages,
            "activeChatId": current_active_chat_id
        })
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /get-history: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")


# Clear Chat History
@app.post("/clear-chat-history", status_code=200)
async def clear_chat_history():
    """
    Clear all chat history by resetting to the initial database structure.
    Resets active_chat_id to 0 and next_chat_id to 1.
    """
    print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Endpoint called.")
    async with db_lock:
        print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Acquired chat DB lock.")
        try:
            # Reset to the defined initial state
            chatsDb = initial_db.copy()
            await save_db(chatsDb)
            print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Chat database reset to initial state (active_id=0, next_id=1) and saved.")

            # Clear in-memory history components as well (important!)
            # Assuming these runnables have a method to clear internal history state
            if hasattr(chat_runnable, 'clear_history'): chat_runnable.clear_history()
            if hasattr(agent_runnable, 'clear_history'): agent_runnable.clear_history()
            if hasattr(unified_classification_runnable, 'clear_history'): unified_classification_runnable.clear_history()
            # If get_chat_history() manages state, reset it too if possible
            # chat_history = get_chat_history(force_reload=True) # Or similar mechanism if available

            print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: In-memory chat histories cleared (if methods exist).")

            # print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Releasing chat DB lock.") # Lock released automatically by 'async with'
            return JSONResponse(status_code=200, content={"message": "Chat history cleared", "activeChatId": 0}) # Return new active ID
        except Exception as e:
             print(f"[ERROR] {datetime.now()}: Error in /clear-chat-history: {e}")
             traceback.print_exc()
             # Ensure lock is released even on error (handled by 'async with')
             # print(f"[ENDPOINT /clear-chat-history] {datetime.now()}: Releasing chat DB lock due to error.")
             raise HTTPException(status_code=500, detail="Failed to clear chat history.")


@app.post("/chat", status_code=200)
async def chat(message: Message):
    """Handles incoming chat messages, classifies, determines active chat, and responds via streaming."""
    endpoint_start_time = time.time()
    print(f"[ENDPOINT /chat] {datetime.now()}: Endpoint called.")
    print(f"[ENDPOINT /chat] {datetime.now()}: Incoming message data: Input='{message.input}...', Pricing='{message.pricing}', Credits={message.credits}")

    # Ensure global variables are accessible if modified or reassigned inside
    global embed_model, chat_runnable, fact_extraction_runnable, text_conversion_runnable
    global information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable
    global query_classification_runnable, agent_runnable, text_description_runnable
    global reflection_runnable, internet_query_reframe_runnable, internet_summary_runnable, priority_runnable
    global unified_classification_runnable, memory_backend
    # Removed global active_chat_id, will read from DB

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

        # --- Determine Active Chat ID (Crucial Step) ---
        # Call get_chat_history_messages to ensure a chat is active/created
        # This function now handles the logic for active_chat_id=0
        print(f"[ENDPOINT /chat] {datetime.now()}: Ensuring an active chat exists...")
        await get_chat_history_messages() # Call for side effect of activating/creating chat

        # Now, load the guaranteed active chat ID from the DB
        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id", 0) # Should be > 0 now

        if active_chat_id == 0:
             # This case should ideally not be reached if get_chat_history_messages worked
             print(f"[ERROR] {datetime.now()}: Failed to determine or create an active chat ID even after check.")
             raise HTTPException(status_code=500, detail="Could not determine active chat.")

        print(f"[ENDPOINT /chat] {datetime.now()}: Confirmed active chat ID: {active_chat_id}")

        # --- Ensure Runnables use Correct History ---
        # Re-initialize runnables based on the current history state managed by get_chat_history()
        print(f"[ENDPOINT /chat] {datetime.now()}: Ensuring runnables use current history context...")
        current_chat_history = get_chat_history() # Get the history object
        chat_runnable = get_chat_runnable(current_chat_history)
        agent_runnable = get_agent_runnable(current_chat_history)
        unified_classification_runnable = get_unified_classification_runnable(current_chat_history)
        print(f"[ENDPOINT /chat] {datetime.now()}: Runnables ready.")

        # --- Unified Classification ---
        print(f"[ENDPOINT /chat] {datetime.now()}: Performing unified classification for input: '{message.input}...'")
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

            # 1. Add User Message to DB (Visible)
            user_msg_id = await add_message_to_db(
                active_chat_id,
                message.input,
                is_user=True,
                is_visible=True # User message is visible
            )
            if not user_msg_id:
                 print(f"[ERROR] {datetime.now()}: Failed to add user message to DB for chat {active_chat_id}. Aborting stream.")
                 # Optionally yield an error message?
                 yield json.dumps({"type": "error", "message": "Failed to save your message."}) + "\n"
                 return


            # 2. Yield User Message Confirmation to Client
            # Fetch the timestamp from the saved message if needed, or generate again
            user_msg_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
            print(f"[STREAM /chat] {datetime.now()}: Yielding user message confirmation.")
            yield json.dumps({
                "type": "userMessage",
                "message": message.input,
                "id": user_msg_id, # Send back the ID
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False,
                 "timestamp": user_msg_timestamp # Include timestamp
            }) + "\n"
            await asyncio.sleep(0.01) # Small delay

            # 3. Prepare Assistant Message Structure (will be filled and updated)
            assistant_msg_id_ts = str(int(time.time() * 1000)) # Timestamp-based message ID
            assistant_msg = {
                "id": assistant_msg_id_ts,
                "message": "",
                "isUser": False,
                "memoryUsed": False, # Will be updated
                "agentsUsed": False, # Will be updated
                "internetUsed": False, # Will be updated
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z", # Initial timestamp
                "isVisible": True # Assistant messages are typically visible
            }
            print(f"[STREAM /chat] {datetime.now()}: Prepared initial assistant message structure (ID: {assistant_msg_id_ts}).")

            # --- Handle Agent Category (Task Creation) ---
            if category == "agent":
                    with open("userProfileDb.json", "r", encoding="utf-8") as f:
                        db = json.load(f)
                    personality_description = db["userData"].get("personality", "None")
                    priority_response = priority_runnable.invoke({"task_description": transformed_input})
                    priority = priority_response["priority"]

                    print("Adding task to queue")
                    await task_queue.add_task(
                        chat_id=active_chat_id,
                        description=transformed_input,
                        priority=priority,
                        username=username,
                        personality=personality_description,
                        use_personal_context=use_personal_context,
                        internet=internet
                    )
                    print("Task added to queue")

                    assistant_msg["message"] = "On it"
                    async with db_lock:
                        chatsDb = await load_db()
                        active_chat = next(chat for chat in chatsDb["chats"] if chat["id"] == chatsDb["active_chat_id"])
                        active_chat["messages"].append(assistant_msg)
                        await save_db(chatsDb)

                    yield json.dumps({
                        "type": "assistantMessage",
                        "message": "On it",
                        "memoryUsed": False,
                        "agentsUsed": False,
                        "internetUsed": False
                    }) + "\n"
                    await asyncio.sleep(0.05)
                    return

            # --- Handle Memory/Context Retrieval ---
            if category == "memory" or use_personal_context:
                print(f"[STREAM /chat] {datetime.now()}: Category requires memory/personal context ({category=}, {use_personal_context=}).")
                if category == "memory" and pricing_plan == "free" and credits <= 0:
                    print(f"[STREAM /chat] {datetime.now()}: Memory category but free plan credits exhausted. Skipping memory update, retrieving only.")
                    note = "Sorry friend, memory updates are a pro feature and your daily credits have expired. Upgrade to pro in settings!"
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories (read-only)...", "id": assistant_msg_id_ts}) + "\n"
                    memory_used = True # Still retrieving
                    user_context = await memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Memory retrieved (read-only). Context length: {len(str(user_context)) if user_context else 0}")
                elif category == "memory": # Pro or has credits
                    print(f"[STREAM /chat] {datetime.now()}: Memory category with credits/pro plan. Retrieving and queueing update.")
                    yield json.dumps({"type": "intermediary", "message": "Retrieving and updating memories...", "id": assistant_msg_id_ts}) + "\n"
                    memory_used = True
                    pro_used = True # Memory update is a pro feature use
                    # Retrieve existing memories first
                    user_context = await memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Memory retrieved. Context length: {len(str(user_context)) if user_context else 0}")
                    # Queue memory update in the background
                    print(f"[STREAM /chat] {datetime.now()}: Queueing memory update operation for user '{username}'.")
                    asyncio.create_task(memory_backend.add_operation(username, transformed_input))
                else: # Just use_personal_context (not explicitly 'memory' category)
                    print(f"[STREAM /chat] {datetime.now()}: Retrieving personal context (not memory category).")
                    yield json.dumps({"type": "intermediary", "message": "Retrieving relevant context...", "id": assistant_msg_id_ts}) + "\n"
                    memory_used = True # Mark as used context
                    user_context = await memory_backend.retrieve_memory(username, transformed_input)
                    print(f"[STREAM /chat] {datetime.now()}: Personal context retrieved. Context length: {len(str(user_context)) if user_context else 0}")
                assistant_msg["memoryUsed"] = memory_used # Update status

            # --- Handle Internet Search ---
            if internet:
                print(f"[STREAM /chat] {datetime.now()}: Internet search required.")
                if pricing_plan == "free" and credits <= 0:
                    print(f"[STREAM /chat] {datetime.now()}: Internet search required but free plan credits exhausted. Skipping.")
                    note += " Sorry friend, could have searched the internet for more context, but your daily credits have expired. You can always upgrade to pro from the settings page"
                else:
                    print(f"[STREAM /chat] {datetime.now()}: Performing internet search (pro/credits available).")
                    yield json.dumps({"type": "intermediary", "message": "Searching the internet...", "id": assistant_msg_id_ts}) + "\n"
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
                with open("userProfileDb.json", "r", encoding="utf-8") as f:
                    db = json.load(f)
                personality_description = db["userData"].get("personality", "None")

                assistant_msg["memoryUsed"] = memory_used
                assistant_msg["internetUsed"] = internet_used
                print("USER CONTEXT: ", user_context)
                async for token in generate_streaming_response(
                    chat_runnable,
                    inputs={
                        "query": transformed_input,
                        "user_context": user_context,
                        "internet_context": internet_context,
                        "name": username,
                        "personality": personality_description
                    },
                    stream=True
                ):
                    if isinstance(token, str):
                        assistant_msg["message"] += token
                        async with db_lock:
                            chatsDb = await load_db()
                            active_chat = next(chat for chat in chatsDb["chats"] if chat["id"] == chatsDb["active_chat_id"])
                            for msg in active_chat["messages"]:
                                if msg["id"] == assistant_msg["id"]:
                                    msg["message"] = assistant_msg["message"]
                                    break
                            else:
                                active_chat["messages"].append(assistant_msg)
                            await save_db(chatsDb)
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": token,
                            "done": False,
                            "messageId": assistant_msg["id"]
                        }) + "\n"
                    else:
                        if note:
                            assistant_msg["message"] += "\n\n" + note
                        async with db_lock:
                            chatsDb = await load_db()
                            active_chat = next(chat for chat in chatsDb["chats"] if chat["id"] == chatsDb["active_chat_id"])
                            for msg in active_chat["messages"]:
                                if msg["id"] == assistant_msg["id"]:
                                    msg.update(assistant_msg)
                                    break
                            else:
                                active_chat["messages"].append(assistant_msg)
                            await save_db(chatsDb)
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": "\n\n" + note if note else "",
                            "done": True,
                            "memoryUsed": memory_used,
                            "agentsUsed": agents_used,
                            "internetUsed": internet_used,
                            "proUsed": pro_used,
                            "messageId": assistant_msg["id"]
                        }) + "\n"
                    await asyncio.sleep(0.05)

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
        # Try to return a JSON error response if possible
        try:
            return JSONResponse(status_code=500, content={"type": "error", "message": f"An internal server error occurred: {str(e)}"})
        except: # Fallback if JSON response fails
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        endpoint_duration = time.time() - endpoint_start_time
        print(f"[ENDPOINT /chat] {datetime.now()}: Endpoint execution finished. Duration: {endpoint_duration:.2f}s")

## Agents Endpoints
@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage):
    """Elaborates on an input string based on a specified purpose."""
    print(f"[ENDPOINT /elaborator] {datetime.now()}: Endpoint called.")
    print(f"[ENDPOINT /elaborator] {datetime.now()}: Input: '{message.input}...', Purpose: '{message.purpose}'")
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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...") # Ensure stringifiable
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
            "query": str(tool_call_input["input"]), # Ensure string
            "username": username,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided")
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

        try:
            # Parse the JSON string from the runnable
            # Use strict=False if the JSON might have control characters like newlines
            tool_call_dict = tool_call_str
            actual_tool_name = tool_call_dict.get("tool_name")
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Parsed tool call: {actual_tool_name}")
            # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Parsed tool call dict: {tool_call_dict}") # Can be verbose

        except json.JSONDecodeError as json_e:
            error_msg = f"Failed to parse JSON response from tool runnable: {json_e}. Response was: {tool_call_str[:500]}..." # Show start of problematic string
            print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
            return {"status": "failure", "error": error_msg}
        except Exception as e: # Catch other potential errors during parsing/access
            error_msg = f"Error processing tool runnable response: {e}. Response was: {tool_call_str[:500]}..."
            print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
            return {"status": "failure", "error": error_msg}


        # Check for approval requirement
        if actual_tool_name in ["send_email", "reply_email"]:
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Action '{actual_tool_name}' requires approval. Returning approval request.")
            # Ensure the dict being returned is valid
            if not isinstance(tool_call_dict, dict):
                 error_msg = f"Approval required but tool_call_dict is not a dict: {type(tool_call_dict)}"
                 print(f"[ERROR] {datetime.now()}: [TOOL HANDLER {tool_name}] {error_msg}")
                 return {"status": "failure", "error": error_msg}
            return {"action": "approve", "tool_call": tool_call_dict}
        else:
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Action '{actual_tool_name}' does not require approval. Executing directly.")
            # Execute the tool call (e.g., search_inbox, get_drafts)
            # parse_and_execute_tool_calls expects a string, so pass the original or re-serialized dict
            tool_result = await parse_and_execute_tool_calls(json.dumps(tool_call_dict)) # Re-serialize
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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...")
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
            "query": str(tool_call_input["input"]),
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided")
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...")
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
            "query": str(tool_call_input["input"]),
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...")
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
             "query": str(tool_call_input["input"]),
             "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...")
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
            "query": str(tool_call_input["input"]),
            "user_name": username,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

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
    print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Input instruction: {str(input_instruction)[:100]}...")
    # print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Full input: {tool_call_input}")

    try:
        # Get current time and timezone dynamically
        current_time_iso = datetime.now(timezone.utc).isoformat() + "Z" # Use UTC for consistency
        local_timezone_key = "UTC" # Default or get dynamically if possible/needed
        try:
            # Attempt to get local timezone; might not work reliably on all servers
            from tzlocal import get_localzone
            local_tz = get_localzone()
            local_timezone_key = str(local_tz) # Use string representation
            print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Detected local timezone: {local_timezone_key}")
        except ImportError:
            print(f"[WARN] {datetime.now()}: [TOOL HANDLER {tool_name}] tzlocal not installed. Using UTC as timezone.")
        except Exception as tz_e:
             print(f"[WARN] {datetime.now()}: [TOOL HANDLER {tool_name}] Error getting local timezone: {tz_e}. Using UTC.")

        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Using Current Time (ISO UTC): {current_time_iso}, Timezone Key: {local_timezone_key}")

        tool_runnable = get_tool_runnable(
            gcalendar_agent_system_prompt_template,
            gcalendar_agent_user_prompt_template,
            gcalendar_agent_required_format,
            ["query", "current_time", "timezone", "previous_tool_response"],
        )
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: GCalendar tool runnable obtained.")

        tool_call_str = tool_runnable.invoke({
            "query": str(tool_call_input["input"]),
            "current_time": current_time_iso,
            "timezone": local_timezone_key,
            "previous_tool_response": tool_call_input.get("previous_tool_response", "Not Provided"),
        })
        print(f"[TOOL HANDLER {tool_name}] {datetime.now()}: Runnable invoked. Raw output string length: {len(tool_call_str)}")

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

        # Use httpx for async requests
        async with httpx.AsyncClient() as client:
            roles_response = await client.get(roles_url, headers=headers)

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

        async with httpx.AsyncClient() as client:
            response = await client.get(user_url, headers=headers)

        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Auth0 API error: {response.text}")

        user_data = response.json()
        # print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: User data received: {user_data}")
        beta_user_status = user_data.get("app_metadata", {}).get("betaUser")
        print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Beta user status from metadata: {beta_user_status}")

        if beta_user_status is None:
            print(f"[ENDPOINT /get-beta-user-status] {datetime.now()}: Beta user status not found for user {request.user_id}. Defaulting to False.")
            # Default to False if not found
            status_bool = False
        else:
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

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)

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

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)

        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Auth0 response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] {datetime.now()}: Auth0 API error ({response.status_code}): {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error fetching user info: {response.text}")

        user_data = response.json()
        referrer_status = user_data.get("app_metadata", {}).get("referrer") # Key is 'referrer'
        print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Referrer status from metadata: {referrer_status}")

        if referrer_status is None:
            print(f"[ENDPOINT /get-referrer-status] {datetime.now()}: Referrer status not found for user {request.user_id}. Defaulting to False.")
            status_bool = False # Default to False
        else:
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

        async with httpx.AsyncClient() as client:
            response = await client.patch(url, headers=headers, json=payload)

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

        async with httpx.AsyncClient() as client:
            search_response = await client.get(search_url, headers=headers, params=params)

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

        async with httpx.AsyncClient() as client:
            set_status_response = await client.patch(update_url, headers=update_headers, json=update_payload)

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
async def set_beta_user_status(request: BetaUserStatusRequest) -> JSONResponse: # Made async
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

        async with httpx.AsyncClient() as client:
            response = await client.patch(url, headers=headers, json=payload)

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
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})


@app.post("/get-user-and-invert-beta-user-status")
async def get_user_and_invert_beta_user_status(request: UserInfoRequest) -> JSONResponse: # Made async
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

        async with httpx.AsyncClient() as client:
            get_response = await client.get(get_url, headers=get_headers)

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

        async with httpx.AsyncClient() as client:
            set_response = await client.patch(set_url, headers=set_headers, json=set_payload)

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
        from server.scraper.functions import scrape_linkedin_profile # Assuming it's here
        print(f"[ENDPOINT /scrape-linkedin] {datetime.now()}: Starting LinkedIn scrape...")
        # This function needs to be async or run in a threadpool if it's blocking
        linkedin_profile = await asyncio.to_thread(scrape_linkedin_profile, profile.url)
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
        # Assuming reddit_scraper is blocking, run in threadpool
        subreddits = await asyncio.to_thread(reddit_scraper, reddit_url.url)
        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Reddit scrape completed. Found {len(subreddits)} potential subreddits/posts.")
        # print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Scraped subreddits/posts: {subreddits}")

        if not subreddits:
             print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: No subreddits found. Skipping LLM analysis.")
             return JSONResponse(status_code=200, content={"topics": []})

        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Invoking Reddit runnable for topic extraction...")
        # Assuming runnable invoke is potentially blocking
        response = await asyncio.to_thread(reddit_runnable.invoke, {"subreddits": subreddits})
        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Reddit runnable finished.")
        # print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Runnable response: {response}")

        # Handle response format variations
        topics = []
        if isinstance(response, list):
            topics = response
        elif isinstance(response, dict) and 'topics' in response and isinstance(response['topics'], list):
             topics = response['topics']
        elif isinstance(response, str): # If LLM just returns a string
             print(f"[WARN] {datetime.now()}: Reddit runnable returned a string, attempting to parse.")
             # Try simple splitting, might need more robust parsing
             topics = [t.strip() for t in response.split(',') if t.strip()]
        else:
            error_msg = f"Invalid response format from the Reddit language model. Expected list or dict with 'topics', got {type(response)}."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        print(f"[ENDPOINT /scrape-reddit] {datetime.now()}: Returning {len(topics)} topics.")
        return JSONResponse(status_code=200, content={"topics": topics})

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
        # Ensure scrape_twitter_data is imported/defined and run in threadpool if blocking
        tweets = await asyncio.to_thread(scrape_twitter_data, twitter_url.url, num_tweets)
        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Twitter scrape completed. Found {len(tweets)} tweets.")
        # print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Scraped tweets: {tweets}")

        if not tweets:
             print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: No tweets found. Skipping LLM analysis.")
             return JSONResponse(status_code=200, content={"topics": []})

        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Invoking Twitter runnable for topic extraction...")
        # Run runnable in threadpool
        response = await asyncio.to_thread(twitter_runnable.invoke, {"tweets": tweets})
        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Twitter runnable finished.")
        # print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Runnable response: {response}")

        # Handle response format variations
        topics = []
        if isinstance(response, list):
            topics = response
        elif isinstance(response, dict) and 'topics' in response and isinstance(response['topics'], list):
            topics = response['topics']
        elif isinstance(response, str):
             print(f"[WARN] {datetime.now()}: Twitter runnable returned a string, attempting to parse.")
             topics = [t.strip() for t in response.split(',') if t.strip()]
        else:
            error_msg = f"Invalid response format from the Twitter language model. Expected list or dict with 'topics', got {type(response)}."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        print(f"[ENDPOINT /scrape-twitter] {datetime.now()}: Returning {len(topics)} topics.")
        return JSONResponse(status_code=200, content={"topics": topics})

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
    try:
        creds = None
        if os.path.exists("server/token.pickle"):
            with open("server/token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(CREDENTIALS_DICT, SCOPES)
                creds = flow.run_local_server(port=0)
            with open("server/token.pickle", "wb") as token:
                pickle.dump(creds, token)
        return JSONResponse(status_code=200, content={"success": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

## Memory Endpoints
@app.post("/graphrag", status_code=200)
async def graphrag(request: GraphRAGRequest):
    """Processes a user profile query using GraphRAG."""
    print(f"[ENDPOINT /graphrag] {datetime.now()}: Endpoint called with query: '{request.query}...'")
    try:
        # Check dependencies
        if not all([graph_driver, embed_model, text_conversion_runnable, query_classification_runnable]):
            error_msg = "GraphRAG dependencies (Neo4j, Embeddings, Runnables) are not initialized."
            print(f"[ERROR] {datetime.now()}: {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)

        print(f"[ENDPOINT /graphrag] {datetime.now()}: Querying user profile...")
        # Run blocking function in threadpool
        context = await asyncio.to_thread(
            query_user_profile,
            request.query, graph_driver, embed_model,
            text_conversion_runnable, query_classification_runnable
        )
        print(f"[ENDPOINT /graphrag] {datetime.now()}: User profile query completed. Context length: {len(str(context)) if context else 0}")
        # print(f"[ENDPOINT /graphrag] {datetime.now()}: Retrieved context: {context}") # Can be verbose
        return JSONResponse(status_code=200, content={"context": context or "No relevant context found."}) # Provide default message
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
    input_dir = "server/input"
    extracted_texts = []
    loop = asyncio.get_event_loop()

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

        # --- Read Input Files (Sync operation, potential block) ---
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Reading text files from input directory: {input_dir}")
        if not os.path.exists(input_dir):
             print(f"[WARN] {datetime.now()}: Input directory '{input_dir}' does not exist. Creating it.")
             await loop.run_in_executor(None, os.makedirs, input_dir) # Run mkdir in thread

        # Run file listing and reading in threadpool
        def read_files():
            texts = []
            files = os.listdir(input_dir)
            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Found {len(files)} items in {input_dir}.")
            for file_name in files:
                file_path = os.path.join(input_dir, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(".txt"): # Process only .txt files
                    print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Reading file: {file_name}")
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            text_content = file.read().strip()
                            if text_content:
                                texts.append({"text": text_content, "source": file_name})
                                print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}:   - Added content from {file_name}. Length: {len(text_content)}")
                            else:
                                print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}:   - Skipped empty file: {file_name}")
                    except Exception as read_e:
                         print(f"[ERROR] {datetime.now()}: Failed to read file {file_name}: {read_e}")
                else:
                     print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Skipping non-txt file or directory: {file_name}")
            return texts

        extracted_texts = await loop.run_in_executor(None, read_files)

        if not extracted_texts:
            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: No text content found in input directory. Nothing to build.")
            return JSONResponse(status_code=200, content={"message": "No content found in input documents. Graph not modified."}) # Not really an error

        # --- Clear Existing Graph (Potentially blocking Neo4j operation) ---
        clear_graph = True # Set to False or make configurable
        if clear_graph:
            print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Clearing existing graph in Neo4j...")
            try:
                def clear_neo4j_graph(driver):
                    with driver.session(database="neo4j") as session:
                        # Use write_transaction for safety
                        session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
                await loop.run_in_executor(None, clear_neo4j_graph, graph_driver)
                print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Existing graph cleared successfully.")
            except Exception as clear_e:
                 error_msg = f"Failed to clear existing graph: {clear_e}"
                 print(f"[ERROR] {datetime.now()}: {error_msg}")
                 raise HTTPException(status_code=500, detail=error_msg)
        else:
             print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Skipping graph clearing.")


        # --- Build Graph (Run blocking function in threadpool) ---
        print(f"[ENDPOINT /initiate-long-term-memories] {datetime.now()}: Building initial knowledge graph from {len(extracted_texts)} document(s)...")
        await loop.run_in_executor(
            None, # Use default executor
            build_initial_knowledge_graph, # Function to run
            # Arguments for the function:
            username, extracted_texts, graph_driver, embed_model,
            text_dissection_runnable, information_extraction_runnable
        )
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
    input_dir = "server/input"
    loop = asyncio.get_event_loop()

    try:
        # --- Load Username and Map Source Key to Filename ---
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User").lower()
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Using username: {username}")

        # Define the mapping (Ensure consistency with create_document filenames)
        SOURCES = {
            "linkedin": f"{username}_linkedin_profile.txt",
            "reddit": f"{username}_reddit_profile.txt",
            "twitter": f"{username}_twitter_profile.txt",
            # Add mappings for personality traits if they were saved as separate files
            "extroversion": f"{username}_extroversion.txt",
            "introversion": f"{username}_introversion.txt",
            "sensing": f"{username}_sensing.txt",
            "intuition": f"{username}_intuition.txt",
            "thinking": f"{username}_thinking.txt",
            "feeling": f"{username}_feeling.txt",
            "judging": f"{username}_judging.txt",
            "perceiving": f"{username}_perceiving.txt",
            # "personality_summary": f"{username}_personality_summary.txt", # If unified file exists
        }

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

        # --- Delete Subgraph from Neo4j (Run blocking in threadpool) ---
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Deleting subgraph related to source '{file_name}' from Neo4j...")
        await loop.run_in_executor(None, delete_source_subgraph, graph_driver, file_name)
        print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Subgraph deletion from Neo4j completed for '{file_name}'.")

        # --- Delete Corresponding Input File (Run blocking in threadpool) ---
        file_path_to_delete = os.path.join(input_dir, file_name)

        def remove_file_sync(path):
             if os.path.exists(path):
                 try:
                     os.remove(path)
                     return True, None
                 except OSError as remove_e:
                     return False, str(remove_e)
             else:
                 return False, "File not found"

        deleted, error = await loop.run_in_executor(None, remove_file_sync, file_path_to_delete)

        if deleted:
            print(f"[ENDPOINT /delete-subgraph] {datetime.now()}: Deleted input file: {file_path_to_delete}")
        else:
            if error == "File not found":
                print(f"[WARN] {datetime.now()}: Input file {file_path_to_delete} not found. No file deleted.")
            else:
                print(f"[WARN] {datetime.now()}: Failed to delete input file {file_path_to_delete}: {error}. Subgraph was still deleted from Neo4j.")


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
    input_dir = "server/input"
    created_files = []
    unified_personality_description = ""
    loop = asyncio.get_event_loop()

    try:
        # --- Load User Profile Data ---
        db = load_user_profile().get("userData", {})
        username = db.get("personalInfo", {}).get("name", "User")
        # Ensure personalityType is a list
        personality_type = db.get("personalityType", [])
        if isinstance(personality_type, str): # Handle case where it might be saved as string
            personality_type = [p.strip().lower() for p in personality_type.split(',') if p.strip()] # Normalize to lower
        structured_linkedin_profile = db.get("linkedInProfile", {}) # Assuming dict/string
        # Ensure social profiles are lists of strings/topics
        reddit_profile = db.get("redditProfile", [])
        if isinstance(reddit_profile, str): reddit_profile = [reddit_profile]
        twitter_profile = db.get("twitterProfile", [])
        if isinstance(twitter_profile, str): twitter_profile = [twitter_profile]

        print(f"[ENDPOINT /create-document] {datetime.now()}: Processing for user: {username}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Personality Traits: {personality_type}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: LinkedIn data present: {bool(structured_linkedin_profile)}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Reddit topics: {len(reddit_profile)}")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Twitter topics: {len(twitter_profile)}")

        # --- Ensure Input Directory Exists ---
        await loop.run_in_executor(None, os.makedirs, input_dir, True) # exist_ok=True
        print(f"[ENDPOINT /create-document] {datetime.now()}: Ensured input directory exists: {input_dir}")

        # --- Helper function to summarize and write (to run in threadpool) ---
        def summarize_and_write(user, text, filename):
            if not text:
                 print(f"[SUMMARIZE_WRITE] {datetime.now()}: Skipping empty text for {filename}.")
                 return False, None
            print(f"[SUMMARIZE_WRITE] {datetime.now()}: Summarizing content for {filename}...")
            try:
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": user, "text": text})
                file_path = os.path.join(input_dir, filename)
                print(f"[SUMMARIZE_WRITE] {datetime.now()}: Writing summarized content to {file_path}...")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)
                return True, filename
            except Exception as e:
                 print(f"[ERROR] {datetime.now()}: Failed to summarize or write file {filename}: {e}")
                 return False, filename

        # --- Process Personality Traits ---
        trait_descriptions = []
        tasks = []
        print(f"[ENDPOINT /create-document] {datetime.now()}: Processing {len(personality_type)} personality traits...")
        for trait in personality_type[0]:
            trait_lower = trait.lower() # Use lowercase consistently
            if trait_lower in PERSONALITY_DESCRIPTIONS:
                description = f"{trait}: {PERSONALITY_DESCRIPTIONS[trait_lower]}"
                trait_descriptions.append(description)
                filename = f"{username.lower()}_{trait_lower}.txt" # Use lowercase trait in filename
                # Add task to run summarize_and_write in threadpool
                tasks.append(loop.run_in_executor(None, summarize_and_write, username, description, filename))
            else:
                 print(f"[WARN] {datetime.now()}: Personality trait '{trait}' not found in PERSONALITY_DESCRIPTIONS. Skipping.")

        unified_personality_description = f"{username}'s Personality Traits:\n\n" + "\n".join(trait_descriptions)
        # Optionally, save the unified description (also in threadpool if needed)
        # unified_filename = f"{username.lower()}_personality_summary.txt"
        # tasks.append(loop.run_in_executor(None, summarize_and_write, username, unified_personality_description, unified_filename))

        # --- Process LinkedIn Profile ---
        if structured_linkedin_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing LinkedIn profile...")
            linkedin_text = json.dumps(structured_linkedin_profile, indent=2) if isinstance(structured_linkedin_profile, dict) else str(structured_linkedin_profile)
            linkedin_file = f"{username.lower()}_linkedin_profile.txt"
            tasks.append(loop.run_in_executor(None, summarize_and_write, username, linkedin_text, linkedin_file))
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No LinkedIn profile data found.")

        # --- Process Reddit Profile ---
        if reddit_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing Reddit profile topics...")
            reddit_text = "User's Reddit Interests: " + ", ".join(reddit_profile)
            reddit_file = f"{username.lower()}_reddit_profile.txt"
            tasks.append(loop.run_in_executor(None, summarize_and_write, username, reddit_text, reddit_file))
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No Reddit profile data found.")

        # --- Process Twitter Profile ---
        if twitter_profile:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Processing Twitter profile topics...")
            twitter_text = "User's Twitter Interests: " + ", ".join(twitter_profile)
            twitter_file = f"{username.lower()}_twitter_profile.txt"
            tasks.append(loop.run_in_executor(None, summarize_and_write, username, twitter_text, twitter_file))
        else:
            print(f"[ENDPOINT /create-document] {datetime.now()}: No Twitter profile data found.")

        # --- Wait for all summarize/write tasks to complete ---
        print(f"[ENDPOINT /create-document] {datetime.now()}: Waiting for {len(tasks)} summarization/write tasks to complete...")
        results = await asyncio.gather(*tasks)
        created_files = [filename for success, filename in results if success and filename]
        failed_files = [filename for success, filename in results if not success and filename]

        print(f"[ENDPOINT /create-document] {datetime.now()}: Document creation process finished.")
        print(f"[ENDPOINT /create-document] {datetime.now()}: Successfully created {len(created_files)} files: {created_files}")
        if failed_files:
            print(f"[ENDPOINT /create-document] {datetime.now()}: Failed to create/summarize {len(failed_files)} files: {failed_files}")

        return JSONResponse(status_code=200, content={
            "message": f"Documents processed. Created: {len(created_files)}, Failed: {len(failed_files)}.",
            "created_files": created_files,
            "failed_files": failed_files,
            "personality": unified_personality_description # Return the combined description
        })

    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Unexpected error in /create-document: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Document creation failed: {str(e)}"})


@app.post("/customize-long-term-memories", status_code=200)
async def customize_graph(request: GraphRequest):
    """Customizes the knowledge graph with new information."""
    print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Endpoint called with information: '{request.information}...'")
    loop = asyncio.get_event_loop()
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

        # --- Extract Facts (Run blocking in threadpool) ---
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracting facts from provided information...")
        points = await loop.run_in_executor(
            None, fact_extraction_runnable.invoke, {"paragraph": request.information, "username": username}
        )
        if not isinstance(points, list):
             print(f"[WARN] {datetime.now()}: Fact extraction did not return a list. Got: {type(points)}. Assuming no facts extracted.")
             points = []
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracted {len(points)} potential facts.")
        # print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Extracted facts: {points}")

        if not points:
             return JSONResponse(status_code=200, content={"message": "No specific facts extracted from the information. Graph not modified."})

        # --- Apply Graph Operations (Run each blocking operation in threadpool concurrently) ---
        tasks = []
        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Applying CRUD operations for {len(points)} facts...")
        for i, point in enumerate(points):
             # Pass necessary arguments to the threadpool function
             tasks.append(loop.run_in_executor(
                 None, crud_graph_operations, point, graph_driver, embed_model,
                 query_classification_runnable, information_extraction_runnable,
                 graph_analysis_runnable, graph_decision_runnable, text_description_runnable
             ))

        # Wait for all operations to complete and handle results/errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_count = 0
        errors = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                 print(f"[ERROR] {datetime.now()}: Failed to process fact {i+1} ('{str(points[i])}...'): {res}")
                 errors.append(str(res))
            else:
                 processed_count += 1

        print(f"[ENDPOINT /customize-long-term-memories] {datetime.now()}: Graph customization process completed. Applied operations for {processed_count}/{len(points)} facts.")
        message = f"Graph customized successfully for {processed_count} facts."
        if errors:
             message += f" Failed for {len(errors)} facts."

        return JSONResponse(status_code=200 if not errors else 207, content={"message": message, "errors": errors}) # 207 Multi-Status if partial success

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
        # Ensure tasks are JSON serializable (e.g., convert datetime)
        serializable_tasks = []
        for task in tasks:
             if isinstance(task.get('created_at'), datetime):
                 task['created_at'] = task['created_at'].isoformat()
             if isinstance(task.get('completed_at'), datetime):
                  task['completed_at'] = task['completed_at'].isoformat()
             serializable_tasks.append(task)
        return JSONResponse(content={"tasks": serializable_tasks})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error fetching tasks: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch tasks.")

@app.post("/add-task", status_code=201)
async def add_task(task_request: CreateTaskRequest): # Use CreateTaskRequest
    """
    Adds a new task with dynamically determined chat_id, personality, context needs, priority etc.
    Input only requires the task description. Associates task with the currently active chat.
    """
    print(f"[ENDPOINT /add-task] {datetime.now()}: Endpoint called with description: '{task_request.description}...'")
    loop = asyncio.get_event_loop()
    try:
        # --- Determine Active Chat ID ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Determining active chat ID...")
        # Ensure active chat exists and get its ID
        await get_chat_history_messages() # Ensure activation/creation
        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id", 0)
            if active_chat_id == 0:
                 raise HTTPException(status_code=500, detail="Failed to determine or create an active chat ID for the task.")
        print(f"[ENDPOINT /add-task] {datetime.now()}: Using active chat ID: {active_chat_id}")

        # --- Load User Profile Info ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Loading user profile...")
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        personality = user_profile.get("userData", {}).get("personality", "Default helpful assistant")
        print(f"[ENDPOINT /add-task] {datetime.now()}: Using username: '{username}', personality: '{str(personality)}...'")

        # --- Classify Task Needs (Run potentially blocking invoke in threadpool) ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Classifying task needs (context, internet)...")
        unified_output = await loop.run_in_executor(
            None, unified_classification_runnable.invoke, {"query": task_request.description}
        )
        use_personal_context = unified_output.get("use_personal_context", False)
        internet = unified_output.get("internet", "None")
        print(f"[ENDPOINT /add-task] {datetime.now()}: Task needs: Use Context={use_personal_context}, Internet='{internet}'")

        # --- Determine Priority (Run potentially blocking invoke in threadpool) ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Determining task priority...")
        try:
            priority_response = await loop.run_in_executor(
                None, priority_runnable.invoke, {"task_description": task_request.description}
            )
            priority = priority_response.get("priority", 3) # Default priority
            print(f"[ENDPOINT /add-task] {datetime.now()}: Determined priority: {priority}")
        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Failed to determine priority: {e}. Using default (3).")
            priority = 3

        # --- Add Task to Queue ---
        print(f"[ENDPOINT /add-task] {datetime.now()}: Adding task to queue...")
        task_id = await task_queue.add_task(
            chat_id=active_chat_id, # Use the active chat ID
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
        await add_message_to_db(active_chat_id, task_request.description, is_user=True, is_visible=False)

        # --- Add Assistant Confirmation (Visible) ---
        confirmation_message = f"OK, I've added the task: '{task_request.description[:40]}...' to my list."
        print(f"[ENDPOINT /add-task] {datetime.now()}: Adding visible assistant confirmation to chat {active_chat_id}.")
        await add_message_to_db(active_chat_id, confirmation_message, is_user=False, is_visible=True, agentsUsed=True, task=task_request.description)

        # --- Broadcast Task Addition via WebSocket ---
        task_details = await task_queue.get_task_by_id(task_id) # Get full details including timestamp
        task_added_message = {
            "type": "task_added",
            "task_id": task_id,
            "description": task_request.description,
            "priority": priority,
            "status": "pending", # Initial status
            "chat_id": active_chat_id,
            "created_at": task_details.get("created_at").isoformat() if task_details and task_details.get("created_at") else datetime.now(timezone.utc).isoformat(),
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task addition for {task_id}")
        await manager.broadcast(json.dumps(task_added_message))


        return JSONResponse(content={"task_id": task_id, "message": "Task added successfully"})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error adding task: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to add task: {str(e)}")

@app.post("/update-task", status_code=200)
async def update_task(update_request: UpdateTaskRequest):
    """Updates an existing task's description and priority."""
    task_id = update_request.task_id
    new_desc = update_request.description
    new_priority = update_request.priority
    print(f"[ENDPOINT /update-task] {datetime.now()}: Endpoint called for task ID: {task_id}")
    print(f"[ENDPOINT /update-task] {datetime.now()}: New Description: '{new_desc}...', New Priority: {new_priority}")
    try:
        updated_task = await task_queue.update_task(task_id, new_desc, new_priority)
        if not updated_task: # Check if update was successful (task found and modifiable)
             raise ValueError(f"Task '{task_id}' not found or could not be updated.")
        print(f"[ENDPOINT /update-task] {datetime.now()}: Task {task_id} updated successfully in queue.")

        # --- Broadcast Task Update via WebSocket ---
        task_update_message = {
            "type": "task_updated",
            "task_id": task_id,
            "description": new_desc,
            "priority": new_priority,
            "status": updated_task.get("status", "unknown"), # Include current status
            # Include other relevant fields if needed
        }
        print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task update for {task_id}")
        await manager.broadcast(json.dumps(task_update_message))

        return JSONResponse(content={"message": "Task updated successfully"})
    except ValueError as e: # Task not found or cannot be updated
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
        deleted = await task_queue.delete_task(task_id)
        if not deleted:
             raise ValueError(f"Task '{task_id}' not found or could not be deleted.")
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
    loop = asyncio.get_event_loop()
    try:
        # Run blocking DB fetch in threadpool
        memories = await loop.run_in_executor(
            None, memory_backend.memory_manager.fetch_memories_by_category,
            user_id, category, limit
        )
        print(f"[ENDPOINT /get-short-term-memories] {datetime.now()}: Fetched {len(memories)} memories.")
        # print(f"[ENDPOINT /get-short-term-memories] {datetime.now()}: Fetched memories: {memories}") # Can be verbose

        # Ensure return is JSON serializable (datetime objects might need conversion)
        serializable_memories = []
        for mem in memories:
             # Convert datetime objects to ISO format strings if they exist
             if isinstance(mem.get('created_at'), datetime):
                 mem['created_at'] = mem['created_at'].isoformat() + "Z"
             if isinstance(mem.get('expires_at'), datetime):
                  mem['expires_at'] = mem['expires_at'].isoformat() + "Z"
             serializable_memories.append(mem)

        return JSONResponse(content=serializable_memories) # Return as JSON response
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error in /get-short-term-memories: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch memories: {str(e)}")


@app.post("/add-short-term-memory")
async def add_memory(request: AddMemoryRequest):
    """Add a new short-term memory."""
    user_id = request.user_id
    text = request.text
    category = request.category
    retention = request.retention_days
    print(f"[ENDPOINT /add-short-term-memory] {datetime.now()}: Endpoint called for User: {user_id}, Category: {category}, Retention: {retention} days")
    print(f"[ENDPOINT /add-short-term-memory] {datetime.now()}: Text: '{text}...'")
    loop = asyncio.get_event_loop()
    try:
        # Run blocking DB store in threadpool
        memory_id = await loop.run_in_executor(
            None, memory_backend.memory_manager.store_memory,
            user_id, text, retention, category
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
    print(f"[ENDPOINT /update-short-term-memory] {datetime.now()}: New Text: '{text}...', New Retention: {retention} days")
    loop = asyncio.get_event_loop()
    try:
        # Run blocking DB update in threadpool
        await loop.run_in_executor(
            None, memory_backend.memory_manager.update_memory_crud,
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
    loop = asyncio.get_event_loop()
    try:
        # Run blocking DB delete in threadpool
        await loop.run_in_executor(
            None, memory_backend.memory_manager.delete_memory,
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
    loop = asyncio.get_event_loop()
    try:
        # Run blocking DB clear in threadpool
        await loop.run_in_executor(
            None, memory_backend.memory_manager.clear_all_memories, user_id
        )
        print(f"[ENDPOINT /clear-all-short-term-memories] {datetime.now()}: All memories cleared successfully for user {user_id}.")
        return JSONResponse(status_code=200, content={"message": "All memories cleared successfully"})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error clearing memories for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")

@app.post("/set-user-data")
async def set_db_data(request: UpdateUserDataRequest) -> Dict[str, Any]:
    """
    Set data in the user profile database (overwrites existing keys at the top level of userData).
    """
    print(f"[ENDPOINT /set-user-data] {datetime.now()}: Endpoint called.")
    # print(f"[ENDPOINT /set-user-data] {datetime.now()}: Request data: {request.data}") # Careful logging potentially sensitive data
    loop = asyncio.get_event_loop()
    try:
        db_data = await loop.run_in_executor(None, load_user_profile) # Load sync in thread
        if "userData" not in db_data: # Ensure userData key exists
            db_data["userData"] = {}

        # Merge new data (shallow merge)
        db_data["userData"].update(request.data)
        print(f"[ENDPOINT /set-user-data] {datetime.now()}: User data updated (shallow merge).")

        # Write sync in thread
        success = await loop.run_in_executor(None, write_user_profile, db_data)
        if success:
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
    loop = asyncio.get_event_loop()
    try:
        db_data = await loop.run_in_executor(None, load_user_profile) # Load sync in thread
        existing_data = db_data.get("userData", {}) # Ensure userData exists
        data_to_add = request.data

        print(f"[ENDPOINT /add-db-data] {datetime.now()}: Starting merge process...")
        # Merge logic is synchronous CPU bound, can stay in main thread or run in executor if complex
        for key, value in data_to_add.items():
            # print(f"[ENDPOINT /add-db-data] {datetime.now()}: Merging key: '{key}'")
            if key in existing_data:
                if isinstance(existing_data[key], list) and isinstance(value, list):
                    # Merge lists and remove duplicates
                    existing_items = set(existing_data[key])
                    for item in value:
                        if item not in existing_items:
                             existing_data[key].append(item)
                             existing_items.add(item) # Keep track of added items
                    # print(f"[ENDPOINT /add-db-data] {datetime.now()}:   - Merged list for key '{key}'. New length: {len(existing_data[key])}")
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

        # Write sync in thread
        success = await loop.run_in_executor(None, write_user_profile, db_data)
        if success:
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
async def get_db_data() -> Dict[str, Any]:
    """Get all user profile database data."""
    print(f"[ENDPOINT /get-user-data] {datetime.now()}: Endpoint called.")
    loop = asyncio.get_event_loop()
    try:
        db_data = await loop.run_in_executor(None, load_user_profile) # Load sync in thread
        user_data = db_data.get("userData", {}) # Default to empty dict if not found
        print(f"[ENDPOINT /get-user-data] {datetime.now()}: Retrieved user data successfully.")
        # print(f"[ENDPOINT /get-user-data] {datetime.now()}: User data: {user_data}") # Careful logging
        return JSONResponse(status_code=200, content={"data": user_data, "status": 200})
    except Exception as e:
        print(f"[ERROR] {datetime.now()}: Error fetching user data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

## Graph Data Endpoint
@app.post("/get-graph-data") # Changed back to GET as it fetches data without body
async def get_graph_data_apoc():
    """Fetches graph data using APOC procedures."""
    print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Endpoint called.")
    loop = asyncio.get_event_loop()
    if not graph_driver:
         print(f"[ERROR] {datetime.now()}: Neo4j driver not available.")
         raise HTTPException(status_code=503, detail="Database connection not available")

    apoc_query = """
    MATCH (n)
    WITH collect(DISTINCT n) as nodes // Collect distinct nodes first
    OPTIONAL MATCH (s)-[r]->(t) // Use OPTIONAL MATCH for relationships
    WHERE s IN nodes AND t IN nodes // Ensure rels use nodes already collected
    WITH nodes, collect(DISTINCT r) as rels // Collect distinct relationships
    RETURN
        [node IN nodes | { id: elementId(node), label: coalesce(labels(node)[0], 'Unknown'), properties: properties(node) }] AS nodes_list,
        [rel IN rels | { id: elementId(rel), from: elementId(startNode(rel)), to: elementId(endNode(rel)), label: type(rel), properties: properties(rel) }] AS edges_list
    """

    print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Executing APOC query on Neo4j...")

    def run_neo4j_query(driver, query):
        try:
            with driver.session(database="neo4j") as session:
                result = session.run(query).single()
                if result:
                    nodes = result['nodes_list'] if result['nodes_list'] else []
                    edges = result['edges_list'] if result['edges_list'] else []
                    return nodes, edges
                else:
                    return [], [] # Empty graph
        except Exception as e:
             # Propagate exception to be caught by the endpoint handler
             raise e

    try:
        nodes, edges = await loop.run_in_executor(None, run_neo4j_query, graph_driver, apoc_query)
        print(f"[ENDPOINT /get-graph-data] {datetime.now()}: Query successful. Found {len(nodes)} nodes and {len(edges)} edges.")
        return JSONResponse(status_code=200, content={"nodes": nodes, "edges": edges})

    except Exception as e:
        error_msg = f"Error fetching graph data from Neo4j: {e}"
        print(f"[ERROR] {datetime.now()}: {error_msg}")
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
            notifications_db = await load_notifications_db() # Already async
            notifications = notifications_db.get("notifications", [])
            # Serialize datetimes if necessary
            for notif in notifications:
                if isinstance(notif.get('timestamp'), datetime):
                     notif['timestamp'] = notif['timestamp'].isoformat() + "Z"

            print(f"[ENDPOINT /get-notifications] {datetime.now()}: Retrieved {len(notifications)} notifications.")
            return JSONResponse(content={"notifications": notifications})
        except Exception as e:
            print(f"[ERROR] {datetime.now()}: Failed to load/process notifications DB: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to retrieve notifications.")
        # finally: # Lock released by async with
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
        result_data = await task_queue.approve_task(task_id) # approve_task is already async
        print(f"[ENDPOINT /approve-task] {datetime.now()}: Task {task_id} approved and execution completed by TaskQueue.")
        # print(f"[ENDPOINT /approve-task] {datetime.now()}: Result from approve_task: {result_data}")

        # --- Add results to chat after approval ---
        task_details = await task_queue.get_task_by_id(task_id) # Get details like description, chat_id
        description = "Approved Task" # Default description
        chat_id = None
        if task_details:
            chat_id = task_details.get("chat_id")
            description = task_details.get("description", description) # Use task desc if available
            if chat_id:
                print(f"[ENDPOINT /approve-task] {datetime.now()}: Adding approved task result to chat {chat_id}.")
                # Add visible assistant message (final result)
                # The hidden user message (prompt) should have been added when the task was created.
                await add_message_to_db(
                    chat_id,
                    result_data,
                    is_user=False,
                    is_visible=True,
                    type="tool_result",
                    task=description,
                    agentsUsed=True # Agent was used to get to approval state
                )
            else:
                print(f"[WARN] {datetime.now()}: Could not find chat_id for completed task {task_id}. Cannot add result to chat.")
        else:
            print(f"[WARN] {datetime.now()}: Could not retrieve details for completed task {task_id} after approval.")

        # --- Broadcast Task Completion via WebSocket ---
        # TaskQueue's complete_task (called by approve_task) should already broadcast completion.
        # Broadcasting here might be redundant. If needed, uncomment below.
        # task_completion_message = {
        #     "type": "task_completed",
        #     "task_id": task_id,
        #     "description": description,
        #     "result": result_data,
        #     "status": "completed", # Explicitly set status
        # }
        # print(f"[WS_BROADCAST] {datetime.now()}: Broadcasting task completion (after approval) for {task_id}")
        # await manager.broadcast(json.dumps(task_completion_message))

        print(f"[ENDPOINT /approve-task] {datetime.now()}: Returning success response for task {task_id}.")
        return ApproveTaskResponse(message="Task approved and completed", result=result_data)

    except ValueError as e:
        # Specific error for task not found or wrong state (e.g., not pending approval)
        print(f"[ERROR] {datetime.now()}: Error approving task {task_id}: {e}")
        if "not found" in str(e).lower():
             raise HTTPException(status_code=404, detail=str(e))
        else: # Wrong state, etc.
             raise HTTPException(status_code=400, detail=str(e))
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
        # TaskQueue methods are async, no lock needed here externally
        task = await task_queue.get_task_by_id(task_id)

        if task and task.get("status") == "approval_pending":
            approval_data = task.get("approval_data")
            print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Found task {task_id} pending approval. Returning data.")
            # print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Approval data: {approval_data}")
            return TaskApprovalDataResponse(approval_data=approval_data)
        elif task:
            status = task.get("status", "unknown")
            print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Task {task_id} found but status is '{status}', not 'approval_pending'.")
            raise HTTPException(status_code=400, detail=f"Task '{task_id}' is not pending approval (status: {status}).")
        else:
            print(f"[ENDPOINT /get-task-approval-data] {datetime.now()}: Task {task_id} not found in the queue.")
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        # Catch unexpected errors during lookup
        print(f"[ERROR] {datetime.now()}: Unexpected error fetching approval data for task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

## Data Source Configuration Endpoints
@app.get("/get_data_sources")
async def get_data_sources_endpoint(): # Renamed to avoid conflict
    """Return the list of available data sources and their enabled states."""
    print(f"[ENDPOINT /get_data_sources] {datetime.now()}: Endpoint called.")
    loop = asyncio.get_event_loop()
    try:
        user_profile = await loop.run_in_executor(None, load_user_profile) # Load sync in thread
        user_data = user_profile.get("userData", {})
        data_sources_status = []
        for source in DATA_SOURCES:
            key = f"{source}Enabled"
            # Default to True if the key doesn't exist in the profile
            enabled = user_data.get(key, True)
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
    loop = asyncio.get_event_loop()

    if source not in DATA_SOURCES:
        print(f"[ERROR] {datetime.now()}: Invalid data source provided: {source}")
        raise HTTPException(status_code=400, detail=f"Invalid data source: {source}. Valid sources are: {DATA_SOURCES}")

    try:
        db_data = await loop.run_in_executor(None, load_user_profile) # Load sync
        if "userData" not in db_data: db_data["userData"] = {}

        key = f"{source}Enabled"
        db_data["userData"][key] = enabled
        print(f"[ENDPOINT /set_data_source_enabled] {datetime.now()}: Updated '{key}' to {enabled} in user profile data.")

        success = await loop.run_in_executor(None, write_user_profile, db_data) # Write sync
        if success:
            print(f"[ENDPOINT /set_data_source_enabled] {datetime.now()}: User profile saved successfully.")
             # TODO: Trigger restart/reload of the corresponding context engine if necessary
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
    # Simple connect/disconnect, broadcasting happens from other parts of the app
    client_id = websocket.query_params.get("clientId", "anon-" + str(int(time.time() * 1000))) # Example ID
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive, receive pings or other messages if needed
            data = await websocket.receive_text()
            # print(f"Received message on /ws from {client_id}: {data}")
            # Handle ping or other control messages if necessary
            try:
                 msg = json.loads(data)
                 if msg.get("type") == "ping":
                     await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                 # Ignore non-JSON messages or handle as needed
                 print(f"[WS /ws] {datetime.now()}: Received non-JSON message from {client_id}: {data}...")
            except Exception as e:
                 print(f"[WS /ws] {datetime.now()}: Error processing message from {client_id}: {e}")

    except WebSocketDisconnect:
        print(f"[WS /ws] {datetime.now()}: Client {client_id} disconnected (WebSocketDisconnect).")
    except Exception as e:
        # Catch other errors that might break the connection loop
        print(f"[WS /ws] {datetime.now()}: WebSocket error for client {client_id}: {e}")
        # Optionally try to send an error message before disconnecting? Risky.
    finally:
        manager.disconnect(websocket)

# Mount FastRTC stream AFTER FastAPI app is initialized and lifespan is attached
stream.mount(app, path="/voice")
print("FastRTC stream mounted at /voice.")

if __name__ == "__main__":
    # Ensure freeze_support is called for multiprocessing if packaging
    multiprocessing.freeze_support()

    # Configure Uvicorn logging to be less verbose or match project style if desired
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    log_config["loggers"]["uvicorn.access"]["handlers"] = ["access"] # Use custom access formatter
    log_config["loggers"]["uvicorn.error"]["level"] = "INFO" # Set error logger level

    print(f"[UVICORN] {datetime.now()}: Starting Uvicorn server on 0.0.0.0:5000...")
    uvicorn.run(
        "__main__:app", # Point to the app instance in this file
        host="0.0.0.0",
        port=5000,
        lifespan="on",
        reload=False, # Disable reload for production/stability
        workers=1, # Run with a single worker process
        log_config=log_config # Apply custom logging config
    )