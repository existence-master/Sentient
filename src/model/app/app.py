import time
START_TIME = time.time()

import os
import json
import asyncio
import pickle
import multiprocessing
import requests
from datetime import datetime, timezone # Keep timezone explicit
from tzlocal import get_localzone # Keep if needed elsewhere, but use explicit UTC for DB timestamps
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
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
import numpy as np
import gradio as gr # Still needed for FastRTC internals
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
import httpx
import traceback

# Import specific functions, runnables, and helpers from respective folders
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
from model.memory.dual_memory import MemoryManager

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

# Import new/refactored voice modules
from model.voice.stt import FasterWhisperSTT
from model.voice.orpheus_tts import OrpheusTTS, TTSOptions, VoiceId, AVAILABLE_VOICES


# Load environment variables from .env file
load_dotenv("model/.env")

# Apply nest_asyncio to allow nested event loops (useful for development environments)
nest_asyncio.apply()

# --- Global Initializations ---

# Initialize embedding model for memory-related operations
embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])

# Initialize Neo4j graph driver for knowledge graph interactions
graph_driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Initialize runnables from agents
reflection_runnable = get_reflection_runnable()
inbox_summarizer_runnable = get_inbox_summarizer_runnable()
priority_runnable = get_priority_runnable()

# Initialize runnables from memory
graph_decision_runnable = get_graph_decision_runnable()
information_extraction_runnable = get_information_extraction_runnable()
graph_analysis_runnable = get_graph_analysis_runnable()
text_dissection_runnable = get_text_dissection_runnable()
text_conversion_runnable = get_text_conversion_runnable()
query_classification_runnable = get_query_classification_runnable()
fact_extraction_runnable = get_fact_extraction_runnable()
text_summarizer_runnable = get_text_summarizer_runnable()
text_description_runnable = get_text_description_runnable()
chat_history = None

chat_runnable = None
agent_runnable = None
unified_classification_runnable = None
# Initialize runnables from scraper
reddit_runnable = get_reddit_runnable()
twitter_runnable = get_twitter_runnable()

internet_query_reframe_runnable = get_internet_query_reframe_runnable()
internet_summary_runnable = get_internet_summary_runnable()

# Tool handlers registry for agent tools
tool_handlers: Dict[str, callable] = {}

# Instantiate the task queue globally
task_queue = TaskQueue()

stt_model = None
tts_model = None
OLLAMA_API_URL = "http://localhost:11434/api/chat" # Keep for reference or potential fallback
OLLAMA_MODEL = "llama3.2:3b" # Keep for reference
OLLAMA_REQUEST_TIMEOUT = 60.0
SELECTED_TTS_VOICE: VoiceId = "tara"
if SELECTED_TTS_VOICE not in AVAILABLE_VOICES:
    print(f"Warning: Selected voice '{SELECTED_TTS_VOICE}' not valid. Using default 'tara'.")
    SELECTED_TTS_VOICE = "tara"
ORPHEUS_EMOTION_TAGS = [
    "<giggle>", "<laugh>", "<chuckle>", "<sigh>", "<cough>",
    "<sniffle>", "<groan>", "<yawn>", "<gasp>"
]
EMOTION_TAG_LIST_STR = ", ".join(ORPHEUS_EMOTION_TAGS)

# --- STT/TTS Model Loading ---
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

async def add_message_to_db(chat_id: str, message_text: str, is_user: bool, is_visible: bool = True, **kwargs):
    """Adds a message to the specified chat ID in the database."""
    async with db_lock:
        try:
            chatsDb = await load_db()
            active_chat = next((chat for chat in chatsDb["chats"] if chat["id"] == chat_id), None)

            if active_chat:
                message_id = str(int(time.time() * 1000)) # Simple timestamp-based ID
                new_message = {
                    "id": message_id,
                    "message": message_text,
                    "isUser": is_user,
                    "isVisible": is_visible,
                    "timestamp": datetime.datetime.now(timezone.utc).isoformat() + "Z",
                    "memoryUsed": kwargs.get("memoryUsed", False),
                    "agentsUsed": kwargs.get("agentsUsed", False),
                    "internetUsed": kwargs.get("internetUsed", False),
                    # Add other relevant fields if needed, like 'type' for tool results
                }
                if kwargs.get("type"):
                    new_message["type"] = kwargs["type"]
                if kwargs.get("task"):
                    new_message["task"] = kwargs["task"]

                active_chat["messages"].append(new_message)
                await save_db(chatsDb)
                print(f"Message added to DB (Chat ID: {chat_id}, User: {is_user}): {message_text[:50]}...")
                return message_id # Return the ID if needed
            else:
                print(f"Error: Could not find chat with ID {chat_id} to add message.")
                return None
        except Exception as e:
            print(f"Error adding message to DB: {e}")
            traceback.print_exc()
            return None

# Tool Registration Decorator
def register_tool(name: str):
    """Decorator to register a function as a tool handler."""
    def decorator(func: callable):
        tool_handlers[name] = func
        return func
    return decorator

# Google OAuth2 scopes and credentials (from auth and common)
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

CREDENTIALS_DICT = {
    "installed": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "auth_uri": os.environ.get("GOOGLE_AUTH_URI"),
        "token_uri": os.environ.get("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost"]
    }
}

# Auth0 configuration from utils
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB = os.path.join(BASE_DIR, "..", "..", "userProfileDb.json")

CHAT_DB = "chatsDb.json"
db_lock = asyncio.Lock()  # Lock for synchronizing database access

initial_db = {
    "chats": [],
    "active_chat_id": None,
    "next_chat_id": 1
}

def load_user_profile():
    """Load user profile data from userProfileDb.json."""
    try:
        with open(USER_PROFILE_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"userData": {}} # Return empty structure if file not found, similar to initial state
    except json.JSONDecodeError:
        return {"userData": {}} # Handle case where JSON is corrupted or empty

def write_user_profile(data):
    """Write user profile data to userProfileDb.json."""
    try:
        with open(USER_PROFILE_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4) # Use indent for pretty printing
        return True
    except Exception as e:
        print(f"Error writing to database: {e}")
        return False
    
memory_backend = MemoryBackend()
memory_backend.cleanup()

async def load_db():
    """Load the database from chatsDb.json, initializing if it doesn't exist or is invalid."""
    try:
        with open(CHAT_DB, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "chats" not in data:
                data["chats"] = []
            if "active_chat_id" not in data:
                data["active_chat_id"] = None
            if "next_chat_id" not in data:
                data["next_chat_id"] = 1
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print("DB NOT FOUND! Initializing with default structure.")
        return initial_db

async def save_db(data):
    """Save the data to chatsDb.json."""
    with open(CHAT_DB, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

async def get_chat_history_messages() -> List[Dict[str, Any]]:
    """
    Function to retrieve the chat history of the currently active chat.
    Checks for inactivity and creates a new chat if needed.
    Returns the list of messages for the active chat, filtering out messages where isVisible is False.
    Handles timestamp parsing robustly.
    """
    async with db_lock:
        chatsDb = await load_db()
        active_chat_id = chatsDb.get("active_chat_id") # Use .get for safety
        current_time = datetime.datetime.now(timezone.utc)

        # If no active chat exists, create a new one
        if not active_chat_id: # Check if None or empty
            # Find if any chats exist at all
            existing_chats = chatsDb.get("chats", [])
            if not existing_chats: # No chats exist at all
                 new_chat_id = f"chat_{chatsDb.get('next_chat_id', 1)}" # Default next_chat_id to 1
                 chatsDb["next_chat_id"] = chatsDb.get('next_chat_id', 1) + 1
                 new_chat = {"id": new_chat_id, "messages": []}
                 chatsDb["chats"] = existing_chats + [new_chat] # Append to potentially empty list
                 chatsDb["active_chat_id"] = new_chat_id
                 await save_db(chatsDb)
                 print(f"No active chat found, created and activated new chat: {new_chat_id}")
                 return []
            else:
                 # Chats exist, but none is active. Activate the latest one? Or first?
                 # Let's activate the last one added for simplicity.
                 active_chat_id = existing_chats[-1]['id']
                 chatsDb['active_chat_id'] = active_chat_id
                 await save_db(chatsDb)
                 print(f"No active chat ID set, activating the last chat: {active_chat_id}")
                 # Continue to fetch messages for this now active chat

        # Find the active chat
        active_chat = next((chat for chat in chatsDb.get("chats", []) if chat.get("id") == active_chat_id), None)

        # Handle case where active_chat_id exists but the chat itself doesn't (data inconsistency)
        if not active_chat:
             print(f"Error: Active chat ID '{active_chat_id}' exists, but no corresponding chat found. Resetting active chat.")
             chatsDb["active_chat_id"] = None # Reset to avoid repeated errors
             await save_db(chatsDb)
             # Recursively call or return empty? Returning empty is safer.
             # return await get_chat_history_messages() # Risky if DB error persists
             return []

        # Check for inactivity only if there are messages
        if active_chat.get("messages"): # Use .get for safety
            try:
                last_message = active_chat["messages"][-1]
                timestamp_str = last_message.get("timestamp") # Use .get

                if not timestamp_str:
                     print("Warning: Last message has no timestamp. Skipping inactivity check.")
                else:
                    # --- Robust Timestamp Parsing ---
                    if timestamp_str.endswith('Z'):
                        # Replace Z only if it's at the very end
                        timestamp_str = timestamp_str[:-1] + '+00:00'
                    # --- End Timestamp Parsing Fix ---

                    try:
                        last_timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        # Ensure last_timestamp is offset-aware for comparison
                        if last_timestamp.tzinfo is None:
                             # This case shouldn't happen if parsing worked with '+00:00'
                             # but as a safeguard, assume UTC if naive after parsing attempt
                             last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

                        if (current_time - last_timestamp).total_seconds() > 600:  # 10 minutes = 600 seconds
                            print(f"Inactivity detected in chat {active_chat_id}. Creating new chat.")
                            # Inactivity period exceeded, create a new chat
                            new_chat_id = f"chat_{chatsDb.get('next_chat_id', 1)}"
                            chatsDb["next_chat_id"] = chatsDb.get('next_chat_id', 1) + 1
                            new_chat = {"id": new_chat_id, "messages": []}
                            chatsDb["chats"].append(new_chat) # Append new chat
                            chatsDb["active_chat_id"] = new_chat_id
                            await save_db(chatsDb)
                            return [] # Return empty messages for new chat
                    except ValueError as e_parse:
                         print(f"Error parsing timestamp '{timestamp_str}': {e_parse}. Skipping inactivity check.")
                    except Exception as e_time:
                         print(f"Error during timestamp comparison: {e_time}. Skipping inactivity check.")

            except IndexError:
                 print("Chat has an empty messages list, cannot check inactivity.")
            except Exception as e:
                 print(f"Unexpected error during inactivity check: {e}. Proceeding...")


        # Return messages from the active chat, filtering out those with isVisible: False
        if active_chat and active_chat.get("messages"):
            filtered_messages = [
                message for message in active_chat["messages"]
                # Default isVisible to True if key is missing
                # Check if the value is explicitly False
                if message.get("isVisible", True) is not False
            ]
            return filtered_messages
        else:
            # Active chat exists but has no messages
            return []

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_ids: Dict[WebSocket, str] = {} # Optional: Track clients if needed

    async def connect(self, websocket: WebSocket, client_id: str = "anon"):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_ids[websocket] = client_id
        print(f"WebSocket client connected: {client_id} ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        client_id = self.client_ids.pop(websocket, "unknown")
        if websocket in self.active_connections:
             self.active_connections.remove(websocket)
        print(f"WebSocket client disconnected: {client_id} ({len(self.active_connections)} total)")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
             await websocket.send_text(message)
        except Exception as e:
             print(f"Error sending personal message: {e}")
             self.disconnect(websocket)

    async def broadcast(self, message: str):
        # Create a list of connections to iterate over, as disconnect modifies the list
        connections_to_send = list(self.active_connections)
        for connection in connections_to_send:
            try:
                await connection.send_text(message)
            except Exception as e:
                client_id = self.client_ids.get(connection, "unknown")
                print(f"Error broadcasting to {client_id}, disconnecting: {e}")
                # Disconnect broken connections
                self.disconnect(connection)

    async def broadcast_json(self, data: dict):
        await self.broadcast(json.dumps(data))

manager = WebSocketManager()

async def cleanup_tasks_periodically():
    """Periodically clean up old completed tasks."""
    while True:
        await task_queue.delete_old_completed_tasks()
        await asyncio.sleep(60 * 60) # Check every hour (or less for testing, e.g., 60 seconds)

async def process_queue():
    while True:
        task = await task_queue.get_next_task()
        if task:
            print(f"Processing task: {task}")
            try:
                task_queue.current_task_execution = asyncio.create_task(execute_agent_task(task))
                result = await task_queue.current_task_execution
                await add_result_to_chat(task["chat_id"], task["description"], True)
                await add_result_to_chat(task["chat_id"], result, False, task["description"])
                await task_queue.complete_task(task["task_id"], result=result)

                # --- WebSocket Message on Success ---
                task_completion_message = {
                    "type": "task_completed",
                    "task_id": task["task_id"],
                    "description": task["description"],
                    "result": result
                }
                await manager.broadcast(json.dumps(task_completion_message))
                print(f"WebSocket message sent for task completion: {task['task_id']}")

            except asyncio.CancelledError:
                await task_queue.complete_task(task["task_id"], error="Task was cancelled")
                # --- WebSocket Message on Cancellation ---
                task_error_message = {
                    "type": "task_error",
                    "task_id": task["task_id"],
                    "description": task["description"],
                    "error": "Task was cancelled"
                }
                await manager.broadcast(json.dumps(task_error_message))
                print(f"WebSocket message sent for task cancellation: {task['task_id']}")

            except Exception as e:
                error_str = str(e)
                await task_queue.complete_task(task["task_id"], error=error_str)
                # --- WebSocket Message on Error ---
                task_error_message = {
                    "type": "task_error",
                    "task_id": task["task_id"],
                    "description": task["description"],
                    "error": error_str
                }
                await manager.broadcast(json.dumps(task_error_message))
                print(f"WebSocket message sent for task error: {task['task_id']} - Error: {error_str}")
        await asyncio.sleep(0.1)

async def process_memory_operations():
    while True:
        operation = await memory_backend.memory_queue.get_next_operation()
        
        if operation:
            try:
                user_id = operation["user_id"]
                memory_data = operation["memory_data"]

                await memory_backend.update_memory(user_id, memory_data)

                await memory_backend.memory_queue.complete_operation(operation["operation_id"], result="Success")
                
                notification = {
                    "type": "memory_operation_completed",
                    "operation_id": operation["operation_id"],
                    "status": "success",
                    "fact": memory_data
                }
                
                await manager.broadcast(json.dumps(notification))
            except Exception as e:
                await memory_backend.memory_queue.complete_operation(operation["operation_id"], error=str(e))
                notification = {
                    "type": "memory_operation_error",
                    "operation_id": operation["operation_id"],
                    "error": str(e),
                    "fact": memory_data
                }
                await manager.broadcast(json.dumps(notification))
        await asyncio.sleep(0.1)

async def execute_agent_task(task: dict) -> str:
    """Execute the agent task asynchronously and return the result."""
    global agent_runnable, reflection_runnable, inbox_summarizer_runnable, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable, internet_query_reframe_runnable, internet_summary_runnable

    # Extract parameters from task
    transformed_input = task["description"]
    username = task["username"]
    personality = task["personality"]
    use_personal_context = task["use_personal_context"]
    internet = task["internet"]

    # Initialize context variables
    user_context = None
    internet_context = None

    # Compute user_context if required
    if use_personal_context:
        try:
            user_context = query_user_profile(
                transformed_input,
                graph_driver,
                embed_model,
                text_conversion_runnable,
                query_classification_runnable
            )
        except Exception as e:
            print(f"Error computing user_context: {e}")
            user_context = None

    # Compute internet_context if required
    if internet == "Internet":
        try:
            reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
            search_results = get_search_results(reframed_query)
            internet_context = get_search_summary(internet_summary_runnable, search_results)
        except Exception as e:
            print(f"Error computing internet_context: {e}")
            internet_context = None

    # Invoke agent_runnable with all required parameters
    response = agent_runnable.invoke({
        "query": transformed_input,
        "name": username,
        "user_context": user_context,
        "internet_context": internet_context,
        "personality": personality
    })

    print(f"Agent response: {response}")

    # Handle tool calls
    if "tool_calls" not in response or not isinstance(response["tool_calls"], list):
        return "Error: Invalid tool_calls format in response."

    print(f"Executing task: {transformed_input}")
    print(f"Tool calls: {response['tool_calls']}")

    all_tool_results = []
    previous_tool_result = None
    for tool_call in response["tool_calls"]:
        if tool_call["response_type"] != "tool_call":
            continue
        tool_name = tool_call["content"].get("tool_name")
        task_instruction = tool_call["content"].get("task_instruction")
        previous_tool_response_required = tool_call["content"].get("previous_tool_response", False)

        tool_handler = tool_handlers.get(tool_name)
        if not tool_handler:
            return f"Error: Tool {tool_name} not found."

        tool_input = {"input": task_instruction}
        if previous_tool_response_required and previous_tool_result:
            tool_input["previous_tool_response"] = previous_tool_result
        else:
            tool_input["previous_tool_response"] = "Not Required"

        tool_result_main = await tool_handler(tool_input)

        print(f"Tool result for {tool_name}: {tool_result_main}")

        tool_result = tool_result_main["tool_result"] if "tool_result" in tool_result_main else tool_result_main
        previous_tool_result = tool_result
        all_tool_results.append({"tool_name": tool_name, "task_instruction": task_instruction, "tool_result": tool_result})

    # Generate final response based on tool results
    if len(all_tool_results) == 1 and all_tool_results[0]["tool_name"] == "search_inbox":
        filtered_tool_result = {
            "response": all_tool_results[0]["tool_result"]["result"]["response"],
            "email_data": [{k: email[k] for k in email if k != "body"} for email in all_tool_results[0]["tool_result"]["result"]["email_data"]],
            "gmail_search_url": all_tool_results[0]["tool_result"]["result"]["gmail_search_url"]
        }
        # Assuming inbox_summarizer_runnable also supports non-streaming invocation
        result = inbox_summarizer_runnable.invoke({"tool_result": filtered_tool_result})
    else:
        # Invoke reflection_runnable without streaming
        result = reflection_runnable.invoke({"tool_results": all_tool_results})

    print(f"Final result: {result}")
    return result

async def add_result_to_chat(chat_id: str, result: str, isUser: bool, task_description: str = None):
    """Add the task result to the corresponding chat."""
    async with db_lock:
        chatsDb = await load_db()
        chat = next((c for c in chatsDb["chats"] if c["id"] == chat_id), None)
        if chat:
            if not isUser:
                result_message = {
                    "id": str(int(time.time() * 1000)),
                    "type": "tool_result",
                    "message": result,
                    "task": task_description,
                    "isUser": False,
                    "memoryUsed": False,
                    "agentsUsed": True,
                    "internetUsed": False,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                }
            else:
                result_message = {
                    "id": str(int(time.time() * 1000)),
                    "message": result,
                    "isUser": True,
                    "isVisible": False,
                    "memoryUsed": False,
                    "agentsUsed": False,
                    "internetUsed": False,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                }
            chat["messages"].append(result_message)
            await save_db(chatsDb)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Sentient API",
    description="Monolithic API for the Sentient AI companion",
    docs_url="/docs",
    redoc_url=None
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "app://.",
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] + origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Startup/Shutdown Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("--- Server Starting ---")
    # Load tasks and memory operations
    await task_queue.load_tasks()
    await memory_backend.memory_queue.load_operations()
    # print("Loading STT model...")
    # stt_model = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
    # print("STT model loaded.")

    # print("Loading TTS model...")
    # try:
    #     # --- Pass the selected default voice during initialization ---
    #     tts_model = OrpheusTTS(
    #         verbose=False,
    #         default_voice_id=SELECTED_TTS_VOICE, # Pass the selected voice here
    #     )
    #     # --- End TTS model initialization change ---
    #     print("TTS model loaded successfully.") # This confirmation comes after the voice is set in init
    # except Exception as e:
    #     print(f"FATAL ERROR: Could not load TTS model: {e}")
    #     exit(1)
    # Start background processors
    app.state.task_processor = asyncio.create_task(process_queue())
    app.state.memory_processor = asyncio.create_task(process_memory_operations())
    app.state.cleanup_processor = asyncio.create_task(cleanup_tasks_periodically())

    yield # Server runs here

    # Code to run on shutdown
    print("--- Server Shutting Down ---")
    # Gracefully shutdown background tasks (optional, depends on task nature)
    if hasattr(app.state, 'task_processor') and not app.state.task_processor.done():
        app.state.task_processor.cancel()
        try: await app.state.task_processor
        except asyncio.CancelledError: print("Task processor cancelled.")
    if hasattr(app.state, 'memory_processor') and not app.state.memory_processor.done():
        app.state.memory_processor.cancel()
        try: await app.state.memory_processor
        except asyncio.CancelledError: print("Memory processor cancelled.")
    if hasattr(app.state, 'cleanup_processor') and not app.state.cleanup_processor.done():
        app.state.cleanup_processor.cancel()
        try: await app.state.cleanup_processor
        except asyncio.CancelledError: print("Cleanup processor cancelled.")

    # Save pending tasks and memory operations
    await task_queue.save_tasks()
    await memory_backend.memory_queue.save_operations()
    # Close Neo4j driver
    if graph_driver:
        await asyncio.to_thread(graph_driver.close) # Use to_thread if close is blocking
    print("--- Server Shutdown Complete ---")

# --- Pydantic Models ---
class Message(BaseModel):
    input: str
    pricing: str
    credits: int
    chat_id: str

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

class CreateTaskRequest(BaseModel):
    chat_id: str
    description: str
    priority: int
    username: str
    personality: Union[Dict, str, None]
    use_personal_context: bool
    internet: str

class UpdateTaskRequest(BaseModel):
    task_id: str
    description: str
    priority: int

class DeleteTaskRequest(BaseModel):
    # No request body needed as per the function signature, but including for potential future use or consistency
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
    id: int
    text: str
    retention_days: int

class DeleteMemoryRequest(BaseModel):
    user_id: str
    category: str
    id: int

# --- API Endpoints ---

# --- Voice Chat Logic ---

# ... (Configuration, Model Init, Ollama function remain the same) ...
# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_REQUEST_TIMEOUT = 60.0

# --- Select TTS Voice ---
SELECTED_TTS_VOICE: VoiceId = "tara"
if SELECTED_TTS_VOICE not in AVAILABLE_VOICES:
    print(f"Warning: Selected voice '{SELECTED_TTS_VOICE}' not valid. Using default 'tara'.")
    SELECTED_TTS_VOICE = "tara"
# --- End Voice Selection ---


# --- Orpheus Supported Emotion Tags ---
ORPHEUS_EMOTION_TAGS = [
    "<giggle>", "<laugh>", "<chuckle>", "<sigh>", "<cough>",
    "<sniffle>", "<groan>", "<yawn>", "<gasp>"
]
EMOTION_TAG_LIST_STR = ", ".join(ORPHEUS_EMOTION_TAGS)

# --- Model Initialization ---
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

# --- Modified Async Handler using Chat Endpoint Logic ---
async def handle_audio_conversation(audio: tuple[int, np.ndarray]):
    """
    Handles the voice conversation flow: STT -> LLM (using chat logic) -> TTS.
    Uses the same classification, context retrieval, and runnable invocation
    as the /chat endpoint, calling invoke() for the full response.
    Ensures async functions are awaited.
    """
    # Make runnables accessible
    global embed_model, graph_driver, memory_backend, task_queue
    global reflection_runnable, inbox_summarizer_runnable, priority_runnable
    global graph_decision_runnable, information_extraction_runnable, graph_analysis_runnable
    global text_dissection_runnable, text_conversion_runnable, query_classification_runnable
    global fact_extraction_runnable, text_summarizer_runnable, text_description_runnable
    global reddit_runnable, twitter_runnable
    global internet_query_reframe_runnable, internet_summary_runnable

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
    active_chat_id = None
    current_chat_runnable = None

    try:
        # 1. Load User Profile & Active Chat Info
        user_profile = load_user_profile()
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        personality_setting = user_profile.get("userData", {}).get("personality", "Default helpful assistant")

        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id")

        if not active_chat_id:
            print("ERROR: No active chat ID found. Cannot process voice message.")
            return

        # 2. Add User Message to DB
        await add_message_to_db(active_chat_id, user_text, is_user=True, is_visible=True)

        # 3. Get Current Chat History & Initialize Context-Aware Runnables
        chat_history_obj = get_chat_history()
        try:
            # Initialize runnables - check chat runnable specifically
            temp_chat_runnable = get_chat_runnable(chat_history_obj)
            print(f"DEBUG: Type returned by get_chat_runnable: {type(temp_chat_runnable)}")
            # Check if it has the invoke method
            if not hasattr(temp_chat_runnable, 'invoke') or not callable(temp_chat_runnable.invoke):
                 print(f"ERROR: get_chat_runnable did not return a valid runnable object with a callable 'invoke' method. Got: {temp_chat_runnable}")
                 raise ValueError("Invalid chat runnable object received (missing invoke).")
            current_chat_runnable = temp_chat_runnable

            current_agent_runnable = get_agent_runnable(chat_history_obj)
            current_unified_classifier = get_unified_classification_runnable(chat_history_obj)
        except Exception as e:
            print(f"ERROR initializing history-dependent runnables: {e}")
            traceback.print_exc()
            await add_message_to_db(active_chat_id, "[System Error: Failed to initialize core components]", is_user=False, error=True)
            return

        # 4. Classify User Input
        print("Classifying input...")
        # Assuming invoke is synchronous here, adjust if classifier uses ainvoke
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
            # Assuming invoke is synchronous here
            priority_response = priority_runnable.invoke({"task_description": transformed_input})
            priority = priority_response.get("priority", 3)

            print(f"Adding agent task to queue (Priority: {priority})...")
            # ---> Use await here <---
            await task_queue.add_task(
                chat_id=active_chat_id,
                description=transformed_input,
                priority=priority,
                username=username,
                personality=personality_setting,
                use_personal_context=use_personal_context,
                internet=internet
            )
            print("Agent task added.")
            bot_response_text = "Okay, I'll get right on that." # Canned response for agent task

            await add_message_to_db(active_chat_id, bot_response_text, is_user=False, agentsUsed=True)

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
                    # ---> Use await here <---
                    user_context = await memory_backend.retrieve_memory(username, transformed_input)
                    # Now user_context should be the actual result (e.g., string or None)
                    if user_context:
                        # Check if it's a string before slicing
                        if isinstance(user_context, str):
                            print(f"Retrieved user context (memory): {user_context[:100]}...")
                        else:
                            # Handle other types if necessary, or just print type
                            print(f"Retrieved user context (memory) of type: {type(user_context)}")
                        memory_used_flag = True
                    else:
                        print("No relevant memories found.")
                except Exception as e:
                    print(f"Error retrieving user context: {e}")
                    traceback.print_exc() # Keep traceback for debugging
                # Starting background memory update task is correct with create_task
                if category == "memory":
                    print("Queueing memory update operation...")
                    asyncio.create_task(memory_backend.add_operation(username, transformed_input))

            # Fetch context if needed (Internet)
            # Assuming these helpers call synchronous invoke methods internally for now
            if internet == "Internet":
                print("Searching the internet...")
                try:
                    reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                    print(f"Reframed internet query: {reframed_query}")
                    search_results = get_search_results(reframed_query)
                    if search_results:
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        # Check if it's a string before slicing
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
                # Check the runnable again before calling invoke
                if not current_chat_runnable or not hasattr(current_chat_runnable, 'invoke') or not callable(current_chat_runnable.invoke):
                     print(f"ERROR: current_chat_runnable is invalid before invoking. Value: {current_chat_runnable}")
                     raise ValueError("Chat runnable became invalid before invoke.")

                # Call invoke (assuming it's synchronous)
                invocation_result = current_chat_runnable.invoke({
                    "query": transformed_input,
                    "user_context": user_context, # Pass the awaited result
                    "internet_context": internet_context, # Pass the retrieved result
                    "name": username,
                    "personality": personality_setting
                })

                # Process the result
                if isinstance(invocation_result, str):
                    bot_response_text = invocation_result.strip()
                    print(f"Generated response text (invoke): {bot_response_text}")
                elif invocation_result is None:
                     print("Warning: Chat runnable invoke returned None.")
                     bot_response_text = "I couldn't generate a response for that."
                else:
                    print(f"Warning: Unexpected response type from invoke: {type(invocation_result)}. Converting to string.")
                    bot_response_text = str(invocation_result).strip()

                if not bot_response_text:
                    print("Warning: Generated response was empty.")
                    bot_response_text = "I don't have a specific response for that right now."

                await add_message_to_db(
                    active_chat_id,
                    bot_response_text,
                    is_user=False,
                    memoryUsed=memory_used_flag,
                    internetUsed=internet_used_flag
                )

            except Exception as e:
                print(f"Error during chat runnable invocation: {e}")
                traceback.print_exc()
                bot_response_text = "Sorry, I encountered an error while generating a response."
                await add_message_to_db(active_chat_id, bot_response_text, is_user=False, error=True)

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
            # TTS still uses streaming internally (async for handles await)
            async for (sample_rate, audio_chunk) in tts_model.stream_tts(bot_response_text, options=tts_options):
                if audio_chunk is not None and audio_chunk.size > 0:
                    if chunk_count == 0:
                         print(f"Streaming TTS audio chunk 1 (sr={sample_rate}, shape={audio_chunk.shape}, dtype={audio_chunk.dtype})...")
                    yield (sample_rate, audio_chunk)
                    chunk_count += 1
                else:
                    print("Warning: Received empty audio chunk from TTS.")

            if chunk_count == 0:
                 print("Warning: TTS stream completed without yielding any audio chunks.")
            else:
                 print(f"Finished synthesizing and streaming {chunk_count} chunks.")

        except Exception as e:
            print(f"Error during TTS synthesis or streaming: {e}")
            traceback.print_exc()
            return

    except Exception as e:
        print(f"--- Unhandled error in handle_audio_conversation: {e} ---")
        traceback.print_exc()
        if active_chat_id:
            try:
                await add_message_to_db(active_chat_id, "[System Error in Voice Processing]", is_user=False, error=True)
            except:
                pass
        return

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
            min_silence_duration_ms=1000
        ),
        can_interrupt=False,
    ),
    mode="send-receive",
    modality="audio",
    additional_outputs=None,
    additional_outputs_handler=None
)

stream.mount(app, path="/voice")
print("FastRTC stream mounted at /voice.")

## Root Endpoint
@app.get("/", status_code=200)
async def main():
    """Root endpoint providing a welcome message."""
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }

@app.get("/get-history", status_code=200)
async def get_history():
    """
    Endpoint to retrieve the chat history. Calls the get_chat_history_messages function.
    """
    messages = await get_chat_history_messages()
    return JSONResponse(status_code=200, content={"messages": messages})

# Clear Chat History
@app.post("/clear-chat-history", status_code=200)
async def clear_chat_history():
    """Clear all chat history by resetting to the initial database structure."""
    async with db_lock:
        # Reset DB
        chatsDb = initial_db.copy()
        await save_db(chatsDb)

    # Clear in-memory history representations if necessary
    # This depends on how your runnables manage history.
    # If they read from DB on each request, this might be enough.
    # If they hold state, you need to clear them.
    try:
         # Re-initialize chat_history object if it holds state
         global chat_history
         chat_history = get_chat_history() # Re-create based on now empty DB (or initial state)
         print("In-memory history states cleared/reset.")
    except Exception as e:
         print(f"Warning: Error clearing in-memory history state: {e}")

    return JSONResponse(status_code=200, content={"message": "Chat history cleared"})

@app.post("/chat", status_code=200)
async def chat(message: Message):
    global embed_model, chat_runnable, fact_extraction_runnable, text_conversion_runnable
    global information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable
    global query_classification_runnable, agent_runnable, text_description_runnable
    global reflection_runnable, internet_query_reframe_runnable, internet_summary_runnable, priority_runnable
    global unified_classification_runnable, memory_backend

    try:
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)

        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb["active_chat_id"]
            if active_chat_id is None:
                raise HTTPException(status_code=400, detail="No active chat found. Please load the chat page first.")

            active_chat = next((chat for chat in chatsDb["chats"] if chat["id"] == active_chat_id), None)
            if active_chat is None:
                raise HTTPException(status_code=400, detail="Active chat not found.")

        chat_history = get_chat_history()

        chat_runnable = get_chat_runnable(chat_history)
        agent_runnable = get_agent_runnable(chat_history)
        unified_classification_runnable = get_unified_classification_runnable(chat_history)
        
        if chat_history is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

        username = db["userData"]["personalInfo"]["name"]
        unified_output = unified_classification_runnable.invoke({"query": message.input})
        
        print("Unified output: ", unified_output)
        category = unified_output["category"]
        use_personal_context = unified_output["use_personal_context"]
        internet = unified_output["internet"]
        transformed_input = unified_output["transformed_input"]

        pricing_plan = message.pricing
        credits = message.credits

        async def response_generator():
            memory_used = False
            agents_used = False
            internet_used = False
            user_context = None
            internet_context = None
            pro_used = False
            note = ""

            user_msg = {
                "id": str(int(time.time() * 1000)),
                "message": message.input,
                "isUser": True,
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            async with db_lock:
                chatsDb = await load_db()
                active_chat = next(chat for chat in chatsDb["chats"] if chat["id"] == chatsDb["active_chat_id"])
                active_chat["messages"].append(user_msg)
                await save_db(chatsDb)

            yield json.dumps({
                "type": "userMessage",
                "message": message.input,
                "memoryUsed": memory_used,
                "agentsUsed": agents_used,
                "internetUsed": internet_used
            }) + "\n"
            await asyncio.sleep(0.05)

            assistant_msg = {
                "id": str(int(time.time() * 1000)),
                "message": "",
                "isUser": False,
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }

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

            if category == "memory":
                if pricing_plan == "free" and credits <= 0:
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                    user_context = memory_backend.retrieve_memory(username, transformed_input)
                    note = "Sorry friend, memory updates are a pro feature and your daily credits have expired. Upgrade to pro in settings!"
                else:
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                    memory_used = True
                    pro_used = True
                    user_context = memory_backend.retrieve_memory(username, transformed_input)
                    # Queue memory update in the background
                    asyncio.create_task(memory_backend.add_operation(username, transformed_input))
            elif use_personal_context:
                yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                memory_used = True
                user_context = memory_backend.retrieve_memory(username, transformed_input)

            if internet == "Internet":
                if pricing_plan == "free" and credits <= 0:
                    note = "Sorry friend, could have searched the internet for more context, but your daily credits have expired. You can always upgrade to pro from the settings page"
                else:
                    yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                    reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                    search_results = get_search_results(reframed_query)
                    internet_context = get_search_summary(internet_summary_runnable, search_results)
                    internet_used = True
                    pro_used = True

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

        return StreamingResponse(response_generator(), media_type="application/json")

    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

## Agents Endpoints
@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage):
    """Elaborates on an input string based on a specified purpose."""
    print ("MESSAGE TO BE ELABORATED: ", message.input)
    try:
        elaborator_runnable = get_tool_runnable(
            elaborator_system_prompt_template,
            elaborator_user_prompt_template,
            None,
            ["query", "purpose"]
        )
        print ("Elaborator runnable: ", elaborator_runnable)
        output = elaborator_runnable.invoke({"query": message.input, "purpose": message.purpose})
        print (f"Elaborator output: {output}")
        return JSONResponse(status_code=200, content={"message": output})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

## Tool Handlers
@register_tool("gmail")
async def gmail_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """Handles Gmail-related tasks using multi-tool support."""
    try:
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"]["name"]
        tool_runnable = get_tool_runnable(
            gmail_agent_system_prompt_template,
            gmail_agent_user_prompt_template,
            gmail_agent_required_format,
            ["query", "username", "previous_tool_response"]
        )
        tool_call_str = tool_runnable.invoke({
            "query": tool_call["input"],
            "username": username,
            "previous_tool_response": tool_call["previous_tool_response"]
        })
        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        return {"tool_result": tool_result, "tool_call_str": tool_call_str}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

@register_tool("gdrive")
async def drive_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """Handles Google Drive interactions."""
    try:
        tool_runnable = get_tool_runnable(
            gdrive_agent_system_prompt_template,
            gdrive_agent_user_prompt_template,
            gdrive_agent_required_format,
            ["query", "previous_tool_response"]
        )
        tool_call_str = tool_runnable.invoke({
            "query": tool_call["input"],
            "previous_tool_response": tool_call["previous_tool_response"]
        })
        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        return {"tool_result": tool_result, "tool_call_str": None}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

@register_tool("gdocs")
async def gdoc_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GDocs Tool endpoint to handle Google Docs creation and text elaboration using multi-tool support.
    Registered as a tool with the name "gdocs".

    Args:
        tool_call (ToolCall): Request body containing the input for the gdocs tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """
    try:
        tool_runnable = get_tool_runnable(  # Initialize gdocs tool runnable
            gdocs_agent_system_prompt_template,
            gdocs_agent_user_prompt_template,
            gdocs_agent_required_format,
            ["query", "previous_tool_response"],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gdocs tool runnable
                "query": tool_call["input"],
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gdocs tool execution
        print(f"Error calling gdocs tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message

@register_tool("gsheets")
async def gsheet_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GSheets Tool endpoint to handle Google Sheets creation and data population using multi-tool support.
    Registered as a tool with the name "gsheets".

    Args:
        tool_call (ToolCall): Request body containing the input for the gsheets tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        tool_runnable = get_tool_runnable(  # Initialize gsheets tool runnable
            gsheets_agent_system_prompt_template,
            gsheets_agent_user_prompt_template,
            gsheets_agent_required_format,
            ["query", "previous_tool_response"],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gsheets tool runnable
                "query": tool_call["input"],
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gsheets tool execution
        print(f"Error calling gsheets tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message

@register_tool("gslides")
async def gslides_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GSlides Tool endpoint to handle Google Slides presentation creation using multi-tool support.
    Registered as a tool with the name "gslides".

    Args:
        tool_call (ToolCall): Request body containing the input for the gslides tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        with open(
            "userProfileDb.json", "r", encoding="utf-8"
        ) as f:  # Load user profile database
            db = json.load(f)

        username = db["userData"]["personalInfo"][
            "name"
        ]  # Extract username from user profile

        tool_runnable = get_tool_runnable(  # Initialize gslides tool runnable
            gslides_agent_system_prompt_template,
            gslides_agent_user_prompt_template,
            gslides_agent_required_format,
            [
                "query",
                "user_name",
                "previous_tool_response",
            ],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gslides tool runnable
                "query": tool_call["input"],
                "user_name": username,
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gslides tool execution
        print(f"Error calling gslides tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message

@register_tool("gcalendar")
async def gcalendar_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GCalendar Tool endpoint to handle Google Calendar interactions using multi-tool support.
    Registered as a tool with the name "gcalendar".

    Args:
        tool_call (ToolCall): Request body containing the input for the gcalendar tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        current_time = datetime.datetime.now().isoformat()  # Get current time in ISO format
        local_timezone = get_localzone()  # Get local timezone
        timezone = local_timezone.key  # Get timezone key

        tool_runnable = get_tool_runnable(  # Initialize gcalendar tool runnable
            gcalendar_agent_system_prompt_template,
            gcalendar_agent_user_prompt_template,
            gcalendar_agent_required_format,
            [
                "query",
                "current_time",
                "timezone",
                "previous_tool_response",
            ],  # Expected input parameters
        )

        tool_call_str = tool_runnable.invoke(  # Invoke the gcalendar tool runnable
            {
                "query": tool_call["input"],
                "current_time": current_time,
                "timezone": timezone,
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gcalendar tool execution
        print(f"Error calling gcalendar: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message

## Utils Endpoints
@app.post("/get-role")
async def get_role(request: UserInfoRequest) -> JSONResponse:
    """Retrieves a user's role from Auth0."""
    try:
        token = get_management_token()
        roles_response = requests.get(
            f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}/roles",
            headers={"Authorization": f"Bearer {token}"}
        )
        if roles_response.status_code != 200:
            raise HTTPException(status_code=roles_response.status_code, detail=roles_response.text)
        roles = roles_response.json()
        if not roles:
            return JSONResponse(status_code=404, content={"message": "No roles found for user."})
        return JSONResponse(status_code=200, content={"role": roles[0]["name"].lower()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/get-beta-user-status")
async def get_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """Retrieves beta user status from Auth0 app_metadata."""
    try:
        token = get_management_token()
        response = requests.get(
            f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        user_data = response.json()
        beta_user_status = user_data.get("app_metadata", {}).get("betaUser")
        if beta_user_status is None:
            return JSONResponse(status_code=404, content={"message": "Beta user status not found."})
        return JSONResponse(status_code=200, content={"betaUserStatus": beta_user_status})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/get-referral-code")
async def get_referral_code(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the referral code from Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: Referral code or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        referral_code = user_data.get("app_metadata", {}).get(
            "referralCode"
        )  # Extract referralCode from app_metadata
        if not referral_code:
            return JSONResponse(
                status_code=404, content={"message": "Referral code not found."}
            )  # Return 404 if referral code not found

        return JSONResponse(
            status_code=200, content={"referralCode": referral_code}
        )  # Return referral code
    except Exception as e:
        print(f"Error in get-referral-code: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-referrer-status")
async def get_referrer_status(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the referrer status from Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: Referrer status or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        referrer_status = user_data.get("app_metadata", {}).get(
            "referrer"
        )  # Extract referrer status from app_metadata
        if referrer_status is None:
            return JSONResponse(
                status_code=404, content={"message": "Referrer status not found."}
            )  # Return 404 if referrer status not found

        return JSONResponse(
            status_code=200, content={"referrerStatus": referrer_status}
        )  # Return referrer status
    except Exception as e:
        print(f"Error in referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/set-referrer-status")
async def set_referrer_status(request: ReferrerStatusRequest) -> JSONResponse:
    """
    Sets the referrer status in Auth0 app_metadata.

    Args:
        request (ReferrerStatusRequest): Request containing user_id and referrer_status.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Content-Type": "application/json",
        }

        payload = {
            "app_metadata": {
                "referrer": request.referrer_status  # Set referrer status in app_metadata
            }
        }

        response = requests.patch(
            url, headers=headers, json=payload
        )  # Use PATCH to update user metadata
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error updating referrer status: {response.text}",  # Raise HTTP exception if request fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Referrer status updated successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in set-referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-user-and-set-referrer-status")
async def get_user_and_set_referrer_status(request: SetReferrerRequest) -> JSONResponse:
    """
    Searches for a user by referral code and sets their referrer status to true.

    Args:
        request (SetReferrerRequest): Request containing referral_code.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        search_url = f"https://{AUTH0_DOMAIN}/api/v2/users?q=app_metadata.referralCode%3A%22{request.referral_code}%22"  # Search URL to find user by referral code
        search_response = requests.get(search_url, headers=headers)

        if search_response.status_code != 200:
            raise HTTPException(
                status_code=search_response.status_code,
                detail=f"Error searching for user: {search_response.text}",  # Raise HTTP exception if search fails
            )

        users = search_response.json()

        if not users or len(users) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No user found with referral code: {request.referral_code}",  # Raise 404 if no user found
            )

        user_id = users[0]["user_id"]  # Get user_id from search results

        referrer_status_payload = {"user_id": user_id, "referrer_status": True}

        set_status_url = f"http://localhost:5005/set-referrer-status"  # URL to set referrer status (assuming local service)
        set_status_response = requests.post(
            set_status_url, json=referrer_status_payload
        )  # Call local service to set referrer status

        if set_status_response.status_code != 200:
            raise HTTPException(
                status_code=set_status_response.status_code,
                detail=f"Error setting referrer status: {set_status_response.text}",  # Raise HTTP exception if setting status fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Referrer status updated successfully."},
        )  # Return success message

    except Exception as e:
        print(f"Error in get-user-and-set-referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/set-beta-user-status")
def set_beta_user_status(request: BetaUserStatusRequest) -> JSONResponse:
    """
    Sets the beta user status in Auth0 app_metadata.

    Args:
        request (BetaUserStatusRequest): Request containing user_id and beta_user_status.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Content-Type": "application/json",
        }

        payload = {
            "app_metadata": {
                "betaUser": request.beta_user_status  # Set betaUser status in app_metadata
            }
        }

        response = requests.patch(
            url, headers=headers, json=payload
        )  # Use PATCH to update user metadata
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error updating beta user status: {response.text}",  # Raise HTTP exception if request fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Beta user status updated successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in set-beta-user-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-user-and-invert-beta-user-status")
def get_user_and_invert_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """
    Searches for a user by user id and inverts the beta user status in Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing user_id.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        beta_user_status = user_data.get("app_metadata", {}).get(
            "betaUser"
        )  # Get current betaUser status

        # Invert the beta user status (string boolean to boolean and then invert)
        beta_user_status_payload = {
            "user_id": request.user_id,
            "beta_user_status": False
            if str(beta_user_status).lower() == "true"
            else True,
        }

        set_status_url = f"http://localhost:5005/set-beta-user-status"  # URL to set beta user status (assuming local service)
        set_status_response = requests.post(
            set_status_url, json=beta_user_status_payload
        )  # Call local service to set inverted beta user status

        if set_status_response.status_code != 200:
            raise HTTPException(
                status_code=set_status_response.status_code,
                detail=f"Error inverting beta user status: {set_status_response.text}",  # Raise HTTP exception if setting status fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Beta user status inverted successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in get-user-and-invert-beta-user-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions

@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest) -> JSONResponse:
    """Encrypts data using AES encryption."""
    try:
        encrypted_data = aes_encrypt(request.data)
        return JSONResponse(status_code=200, content={"encrypted_data": encrypted_data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest) -> JSONResponse:
    """Decrypts data using AES decryption."""
    try:
        decrypted_data = aes_decrypt(request.encrypted_data)
        return JSONResponse(status_code=200, content={"decrypted_data": decrypted_data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

## Scraper Endpoints
@app.post("/scrape-linkedin", status_code=200)
async def scrape_linkedin(profile: LinkedInURL):
    """Scrapes and returns LinkedIn profile information."""
    try:
        linkedin_profile = scrape_linkedin_profile(profile.url)
        return JSONResponse(status_code=200, content={"profile": linkedin_profile})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/scrape-reddit")
async def scrape_reddit(reddit_url: RedditURL):
    """Extracts topics of interest from a Reddit user's profile."""
    try:
        subreddits = reddit_scraper(reddit_url.url)
        response = reddit_runnable.invoke({"subreddits": subreddits})
        if isinstance(response, list):
            return JSONResponse(status_code=200, content={"topics": response})
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from the language model.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/scrape-twitter")
async def scrape_twitter(twitter_url: TwitterURL):
    """Extracts topics of interest from a Twitter user's profile."""
    try:
        tweets = scrape_twitter_data(twitter_url.url, 20)
        response = twitter_runnable.invoke({"tweets": tweets})
        if isinstance(response, list):
            return JSONResponse(status_code=200, content={"topics": response})
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from the language model.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

## Auth Endpoint
@app.get("/authenticate-google")
async def authenticate_google():
    """Authenticates with Google using OAuth 2.0."""
    try:
        creds = None
        if os.path.exists("model/token.pickle"):
            with open("model/token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(CREDENTIALS_DICT, SCOPES)
                creds = flow.run_local_server(port=0)
            with open("model/token.pickle", "wb") as token:
                pickle.dump(creds, token)
        return JSONResponse(status_code=200, content={"success": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

## Memory Endpoints
@app.post("/graphrag", status_code=200)
async def graphrag(request: GraphRAGRequest):
    """Processes a user profile query using GraphRAG."""
    try:
        context = query_user_profile(
            request.query, graph_driver, embed_model,
            text_conversion_runnable, query_classification_runnable
        )
        return JSONResponse(status_code=200, content={"context": context})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/create-graph", status_code=200)
async def create_graph():
    """Creates a knowledge graph from documents in the input directory."""
    try:
        input_dir = "model/input"
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User")
        extracted_texts = []
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text_content = file.read().strip()
                    if text_content:
                        extracted_texts.append({"text": text_content, "source": file_name})
        if not extracted_texts:
            return JSONResponse(status_code=400, content={"message": "No content found in input documents."})

        with graph_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        build_initial_knowledge_graph(
            username, extracted_texts, graph_driver, embed_model,
            text_dissection_runnable, information_extraction_runnable
        )
        return JSONResponse(status_code=200, content={"message": "Graph created successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/delete-subgraph", status_code=200)
async def delete_subgraph(request: DeleteSubgraphRequest):
    """Deletes a subgraph from the knowledge graph based on a source name."""
    try:
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User").lower()
        source_name = request.source
        SOURCES = {
            "linkedin": f"{username}_linkedin_profile.txt",
            "reddit": f"{username}_reddit_profile.txt",
            "twitter": f"{username}_twitter_profile.txt"
        }
        file_name = SOURCES.get(source_name)
        if not file_name:
            return JSONResponse(status_code=400, content={"message": f"No file mapping found for source: {source_name}"})
        delete_source_subgraph(graph_driver, file_name)
        os.remove(f"model/input/{file_name}")
        return JSONResponse(status_code=200, content={"message": f"Subgraph related to {file_name} deleted successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/create-document", status_code=200)
async def create_document():
    """Creates and summarizes personality documents based on user profile data."""
    try:
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User")
        personality_type = db["userData"].get("personalityType", "")
        structured_linkedin_profile = db["userData"].get("linkedInProfile", {})
        reddit_profile = db["userData"].get("redditProfile", [])
        twitter_profile = db["userData"].get("twitterProfile", [])
        input_dir = "model/input"
        os.makedirs(input_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            os.remove(os.path.join(input_dir, file))

        trait_descriptions = []
        for trait in personality_type:
            if trait in PERSONALITY_DESCRIPTIONS:
                description = f"{trait}: {PERSONALITY_DESCRIPTIONS[trait]}"
                trait_descriptions.append(description)
                filename = f"{username.lower()}_{trait.lower()}.txt"
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": description})
                with open(os.path.join(input_dir, filename), "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)

        unified_personality_description = f"{username}'s Personality:\n\n" + "\n".join(trait_descriptions)

        if structured_linkedin_profile:
            linkedin_file = os.path.join(input_dir, f"{username.lower()}_linkedin_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": structured_linkedin_profile})
            with open(linkedin_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)

        if reddit_profile:
            reddit_file = os.path.join(input_dir, f"{username.lower()}_reddit_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": "Interests: " + ",".join(reddit_profile)})
            with open(reddit_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)

        if twitter_profile:
            twitter_file = os.path.join(input_dir, f"{username.lower()}_twitter_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": "Interests: " + ",".join(twitter_profile)})
            with open(twitter_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)

        return JSONResponse(status_code=200, content={"message": "Documents created successfully", "personality": unified_personality_description})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/customize-graph", status_code=200)
async def customize_graph(request: GraphRequest):
    """Customizes the knowledge graph with new information."""
    try:
        with open("userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"]["name"]
        points = fact_extraction_runnable.invoke({"paragraph": request.information, "username": username})
        for point in points:
            crud_graph_operations(
                point, graph_driver, embed_model, query_classification_runnable,
                information_extraction_runnable, graph_analysis_runnable,
                graph_decision_runnable, text_description_runnable
            )
        return JSONResponse(status_code=200, content={"message": "Graph customized successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/fetch-tasks", status_code=200)
async def get_tasks():
    """Return the current state of all tasks."""
    tasks = await task_queue.get_all_tasks()
    return JSONResponse(content={"tasks": tasks})

@app.post("/add-task", status_code=201)
async def add_task(task_request: dict):
    """
    Adds a new task with dynamically determined chat_id, personality, use_personal_context, and internet.
    """
    try:
        async with db_lock:
            chatsDb = await load_db()
            active_chat_id = chatsDb.get("active_chat_id")
            if not active_chat_id:
                raise HTTPException(status_code=400, detail="No active chat found.")

        user_profile = load_user_profile()
        username = user_profile["userData"]["personalInfo"].get("name", "default_user")
        personality = user_profile["userData"].get("personality", "default_personality")

        unified_output = unified_classification_runnable.invoke({"query": task_request["description"]})
        use_personal_context = unified_output["use_personal_context"]
        internet = unified_output["internet"]

        priority_response = priority_runnable.invoke({"task_description": task_request["description"]})
        priority = priority_response["priority"]

        task_id = await task_queue.add_task(
            chat_id=active_chat_id,
            description=task_request["description"],
            priority=priority,
            username=username,
            personality=personality,
            use_personal_context=use_personal_context,
            internet=internet
        )

        return JSONResponse(content={"task_id": task_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/update-task", status_code=200)
async def update_task(update_request: UpdateTaskRequest): # Use UpdateTaskRequest as dependency
    try:
        await task_queue.update_task(update_request.task_id, update_request.description, update_request.priority)
        return JSONResponse(content={"message": "Task updated successfully"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/delete-task", status_code=200)
async def delete_task(delete_request: DeleteTaskRequest): # Use DeleteTaskRequest as dependency (even though it's empty)
    try:
        await task_queue.delete_task(delete_request.task_id)
        return JSONResponse(content={"message": "Task deleted successfully"})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/get-short-term-memories")
async def get_short_term_memories(request: GetShortTermMemoriesRequest) -> List[Dict]:
    try:
        memories = memory_backend.memory_manager.fetch_memories_by_category(
            user_id=request.user_id, 
            category=request.category, 
            limit=request.limit
        )
        return memories
    except Exception as e:
        print(f"Error in get_memories endpoint: {e}")
        return []

@app.post("/add-short-term-memory")
async def add_memory(request: AddMemoryRequest):
    """Add a new memory."""
    try:
        memory_id = memory_backend.memory_manager.store_memory(
            request.user_id, request.text, request.category, request.retention_days
        )
        return {"memory_id": memory_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

@app.post("/update-short-term-memory")
async def update_memory(request: UpdateMemoryRequest):
    """Update an existing memory."""
    try:
        memory_backend.memory_manager.update_memory(
            request.user_id, request.category, request.id, request.text, request.retention_days
        )
        return {"message": "Memory updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")

@app.post("/delete-short-term-memory")
async def delete_memory(request: DeleteMemoryRequest):
    """Delete a memory."""
    try:
        memory_backend.memory_manager.delete_memory(
            request.user_id, request.category, request.id
        )
        return {"message": "Memory deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")
    
@app.post("/clear-all-memories")
async def clear_all_memories(request: Dict):
    user_id = request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        memory_backend.memory_manager.clear_all_memories(user_id)
        return {"message": "All memories cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint for "set-db-data"
@app.post("/set-db-data")
async def set_db_data(request: UpdateUserDataRequest) -> Dict[str, Any]:
    """
    Set data in the user profile database.
    Merges the provided data with the existing userData.

    - **data**: An object containing the data to be set.
    """
    try:
        db_data = load_user_profile()
        db_data["userData"] = {
            **(db_data.get("userData", {})), # Ensure userData exists, default to empty object
            **(request.data) # Merge new data
        }
        if write_user_profile(db_data):
            return {"message": "Data stored successfully", "status": 200}
        else:
            raise HTTPException(status_code=500, detail="Error storing data")
    except Exception as e:
        print(f"Error setting data: {e}")
        raise HTTPException(status_code=500, detail="Error storing data")

# Endpoint for "add-db-data"
@app.post("/add-db-data")
async def add_db_data(request: AddUserDataRequest) -> Dict[str, Any]:
    """
    Add data to the user profile database.
    Handles array and object merging similar to the Electron IPC handler.

    - **data**: An object containing the data to be added.
    """
    try:
        db_data = load_user_profile()
        existing_data = db_data.get("userData", {}) # Ensure userData exists, default to empty object
        data_to_add = request.data

        for key, value in data_to_add.items():
            if key in existing_data:
                if isinstance(existing_data[key], list) and isinstance(value, list):
                    # For arrays, merge and remove duplicates
                    existing_data[key] = list(set(existing_data[key] + value))
                elif isinstance(existing_data[key], dict) and isinstance(value, dict):
                    # For objects, merge properties
                    existing_data[key] = {**existing_data[key], **value}
                else:
                    # For other types, simply overwrite or set new value
                    existing_data[key] = value
            else:
                # If key doesn't exist, set new value
                existing_data[key] = value

        db_data["userData"] = existing_data # Update userData in db_data
        if write_user_profile(db_data):
            return {"message": "Data added successfully", "status": 200}
        else:
            raise HTTPException(status_code=500, detail="Error adding data")
    except Exception as e:
        print(f"Error adding data: {e}")
        raise HTTPException(status_code=500, detail="Error adding data")

# Endpoint for "get-db-data"
@app.post("/get-db-data")
async def get_db_data() -> Dict[str, Any]: # Request is just for consistency, not used
    """
    Get all user profile database data.
    Returns the userData and the database path.
    """
    try:
        db_data = load_user_profile()
        user_data = db_data.get("userData", {}) # Ensure userData exists, default to empty object
        return {
            "data": user_data,
            "status": 200
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching data")

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
                 pass # Ignore non-JSON messages or handle as needed
    except WebSocketDisconnect:
        print(f"/ws client {client_id} disconnected.")
    except Exception as e:
        print(f"/ws error for client {client_id}: {e}")
    finally:
        manager.disconnect(websocket)

STARTUP_TIME = time.time() - START_TIME
print(f"Server startup time: {STARTUP_TIME:.2f} seconds")

# --- Run the Application ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=5000, lifespan="on", reload=False, workers=1)