# Record script start time for performance monitoring and logging
import time
import datetime # Moved for early use and combined

START_TIME = time.time()
print(f"[STARTUP] {datetime.datetime.now()}: Script execution started.")

# --- Core Python and System Imports ---
import os
import asyncio
import pickle
import multiprocessing
import traceback # For detailed error printing
import json # Keep json for websocket messages
from server.context.base import BaseContextEngine, POLLING_INTERVALS # Import POLLING_INTERVALS
from server.context.gmail import GmailContextEngine
from server.context.gcalendar import GCalendarContextEngine
from server.context.internet import InternetSearchContextEngine
from datetime import timezone, timedelta # For scheduling

# --- MongoDB Manager Import ---
from server.db import MongoManager

# --- Third-Party Library Imports ---
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket, # Keep for WebSocketManager type hint if needed, endpoint moved
    WebSocketDisconnect, # Keep for WebSocketManager type hint
    Depends,
    status,
    Security 
)
from fastapi.security import OAuth2PasswordBearer # SecurityScopes removed as PermissionChecker handles it
from fastapi.responses import JSONResponse # HTMLResponse and StreamingResponse moved to routers
from fastapi.middleware.cors import CORSMiddleware
# StaticFiles might not be needed if served by Gradio or other means
# from fastapi.staticfiles import StaticFiles 
# Pydantic models moved to respective routers
from typing import Optional, Any, Dict, List, Tuple # AsyncGenerator removed

# --- Authentication Specific Imports ---
import requests 
from jose import jwt, JWTError
from jose.exceptions import JOSEError

# --- Machine Learning and NLP Imports ---
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Google API Imports ---
# from google_auth_oauthlib.flow import InstalledAppFlow # Moved to utils/routes.py
# from google.auth.transport.requests import Request # Moved to utils/routes.py

# --- Environment and Configuration Imports ---
from dotenv import load_dotenv
import nest_asyncio
import uvicorn

# --- Utility and Other Imports ---
import numpy as np
import gradio as gr # Still needed for FastRTC internals
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions # For FastRTC components
# import httpx # httpx moved to routers where external calls are made

print(f"[STARTUP] {datetime.datetime.now()}: Basic imports completed.")

# --- Application-Specific Module Imports ---
print(f"[STARTUP] {datetime.datetime.now()}: Importing model components...")
from server.agents.runnables import *
from server.agents.functions import * # Contains Google API auth and functions (tool implementations)
from server.agents.prompts import *
from server.agents.formats import *
from server.agents.base import TaskQueue
from server.agents.helpers import * # Contains parse_and_execute_tool_calls
from server.agents.background import AgentTaskProcessor # APPROVAL_PENDING_SIGNAL not needed here

from server.memory.runnables import *
from server.memory.functions import *
from server.memory.prompts import *
from server.memory.constants import *
from server.memory.formats import *
from server.memory.backend import MemoryBackend

from server.utils.helpers import aes_encrypt, aes_decrypt, get_management_token # Specific utils

from server.auth.helpers import * # Contains AES, get_management_token

from server.common.functions import *
from server.common.runnables import *
from server.common.prompts import *
from server.common.formats import *

from server.chat.runnables import *
from server.chat.prompts import *
from server.chat.functions import * # get_chat_history, generate_streaming_response

# Context Engines are used by background tasks, not directly by app.py routes now
# from server.context.gmail import GmailContextEngine
# from server.context.internet import InternetSearchContextEngine
# from server.context.gcalendar import GCalendarContextEngine

from server.voice.stt import FasterWhisperSTT
from server.voice.orpheus_tts import OrpheusTTS, TTSOptions, VoiceId, AVAILABLE_VOICES
from server.voice.gcp_tts import GCPTTS
from server.voice.elevenlabs_tts import ElevenLabsTTS
import uuid # For new chat_id in helper functions

print(f"[STARTUP] {datetime.datetime.now()}: Loading environment variables from server/.env...")
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path)
print(f"[STARTUP] {datetime.datetime.now()}: Environment variables loaded from {dotenv_path}")

IS_DEV_ENVIRONMENT = os.getenv("IS_DEV_ENVIRONMENT", "false").lower() in ("true", "1", "t", "y")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "ORPHEUS").upper()
print(f"[STARTUP] {datetime.datetime.now()}: Development environment flag set to {IS_DEV_ENVIRONMENT}")
print(f"[STARTUP] {datetime.datetime.now()}: TTS Provider set to {TTS_PROVIDER}")

print(f"[STARTUP] {datetime.datetime.now()}: Applying nest_asyncio...")
nest_asyncio.apply()
print(f"[STARTUP] {datetime.datetime.now()}: nest_asyncio applied.")

print(f"[INIT] {datetime.datetime.now()}: Starting global initializations...")
DATA_SOURCES = ["gmail", "internet_search", "gcalendar"]

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
ALGORITHMS = ["RS256"]
# CUSTOM_CLAIMS_NAMESPACE logic moved to utils/routes.py where it's used

if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
    print("[ERROR] FATAL: AUTH0_DOMAIN or AUTH0_AUDIENCE not set!")
    exit(1)

jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
try:
    print(f"[INIT] {datetime.datetime.now()}: Fetching JWKS from {jwks_url}...")
    jwks_response = requests.get(jwks_url)
    jwks_response.raise_for_status()
    jwks = jwks_response.json()
    print(f"[INIT] {datetime.datetime.now()}: JWKS fetched successfully.")
except requests.exceptions.RequestException as e:
    print(f"[ERROR] FATAL: Could not fetch JWKS from Auth0: {e}")
    exit(1)
except Exception as e:
    print(f"[ERROR] FATAL: Error processing JWKS: {e}")
    exit(1)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Define DATA_SOURCES_CONFIG to map service_name to engine class
DATA_SOURCES_CONFIG: Dict[str, Dict[str, Any]] = {
    "gmail": {
        "engine_class": GmailContextEngine,
        "enabled_by_default": True, # Or fetch from user settings
        # Add other service-specific configs if needed
    },
    # "gcalendar": {
    #     "engine_class": GCalendarContextEngine,
    #     "enabled_by_default": True,
    # },
    # "internet_search": {
    #     "engine_class": InternetSearchContextEngine,
    #     "enabled_by_default": False, # Example: off by default
    # }
}

POLLING_SCHEDULER_INTERVAL_SECONDS = int(os.getenv("POLLING_SCHEDULER_INTERVAL_SECONDS", 30))
async def polling_scheduler_loop():
    print(f"[POLLING_SCHEDULER] Starting loop (interval: {POLLING_SCHEDULER_INTERVAL_SECONDS}s)")
    await mongo_manager.reset_stale_polling_locks() # Reset any locks from crashed previous runs
    
    while True:
        try:
            # print(f"[POLLING_SCHEDULER] Checking for due polling tasks at {datetime.now(timezone.utc).isoformat()}")
            due_tasks_states = await mongo_manager.get_due_polling_tasks() # Fetches tasks that are enabled, due, and not locked
            
            if not due_tasks_states:
                # print(f"[POLLING_SCHEDULER] No tasks due at this time.")
                pass
            else:
                print(f"[POLLING_SCHEDULER] Found {len(due_tasks_states)} due polling tasks.")

            for task_state in due_tasks_states:
                user_id = task_state["user_id"]
                service_name = task_state["service_name"] # Changed from engine_category
                
                print(f"[POLLING_SCHEDULER] Attempting to process task for {user_id}/{service_name}")

                # Atomically try to acquire the lock for this specific task
                locked_task_state = await mongo_manager.set_polling_status_and_get(user_id, service_name)
                
                if locked_task_state:
                    print(f"[POLLING_SCHEDULER] Acquired lock for {user_id}/{service_name}. Triggering poll.")
                    
                    engine_instance = active_context_engines.get(user_id, {}).get(service_name)
                    
                    if not engine_instance:
                        engine_config = DATA_SOURCES_CONFIG.get(service_name)
                        if engine_config and engine_config.get("engine_class"):
                            engine_class = engine_config["engine_class"]
                            print(f"[POLLING_SCHEDULER] Creating new {engine_class.__name__} instance for {user_id}/{service_name}")
                            engine_instance = engine_class(
                                user_id=user_id,
                                task_queue=task_queue, 
                                memory_backend=memory_backend, 
                                websocket_manager=manager, # Global websocket_manager
                                mongo_manager_instance=mongo_manager # Global mongo_manager
                            )
                            if user_id not in active_context_engines:
                                active_context_engines[user_id] = {}
                            active_context_engines[user_id][service_name] = engine_instance
                        else:
                            print(f"[POLLING_SCHEDULER_ERROR] No engine class configured for service: {service_name}")
                            # Release lock as we can't process it
                            await mongo_manager.update_polling_state(user_id, service_name, {"is_currently_polling": False})
                            continue
                    
                    # Run the poll cycle in a new asyncio task so the scheduler doesn't block
                    # The run_poll_cycle itself will handle releasing the lock via calculate_and_schedule_next_poll
                    asyncio.create_task(engine_instance.run_poll_cycle()) 
                else:
                    # This means another scheduler instance/worker picked it up, or it's no longer due.
                    print(f"[POLLING_SCHEDULER] Could not acquire lock for {user_id}/{service_name} (already processing or no longer due).")

        except Exception as e:
            print(f"[POLLING_SCHEDULER_ERROR] Error in scheduler loop: {e}")
            traceback.print_exc() # Log full error
        
        await asyncio.sleep(POLLING_SCHEDULER_INTERVAL_SECONDS)
        

class Auth:
    async def _validate_token_and_get_payload(self, token: str) -> dict:
        # ... (Auth class definition remains the same) ...
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            unverified_header = jwt.get_unverified_header(token)
            if "kid" not in unverified_header:
                print("[AUTH_VALIDATION_ERROR] 'kid' not found in token header.")
                raise credentials_exception

            token_kid = unverified_header["kid"]
            if not isinstance(jwks, dict) or "keys" not in jwks or not isinstance(jwks["keys"], list):
                print(f"[AUTH_VALIDATION_ERROR] JWKS structure is invalid.")
                raise credentials_exception

            rsa_key_data = {}
            found_matching_key = False
            for key_entry in jwks["keys"]:
                if isinstance(key_entry, dict) and key_entry.get("kid") == token_kid:
                    required_rsa_components = ["kty", "kid", "use", "n", "e"]
                    if all(comp in key_entry for comp in required_rsa_components):
                        rsa_key_data = {comp: key_entry[comp] for comp in required_rsa_components}
                        found_matching_key = True
                        break

            if not found_matching_key or not rsa_key_data:
                print(f"[AUTH_VALIDATION_ERROR] RSA key not found in JWKS for kid: {token_kid}")
                raise credentials_exception

            payload = jwt.decode(
                token, rsa_key_data, algorithms=ALGORITHMS,
                audience=AUTH0_AUDIENCE, 
                issuer=f"https://{AUTH0_DOMAIN}/"
            )
            return payload
        except JWTError as e:
            print(f"[AUTH_VALIDATION_ERROR] JWT Error: {e}")
            raise credentials_exception
        except JOSEError as e:
            print(f"[AUTH_VALIDATION_ERROR] JOSE Error: {e}")
            raise credentials_exception
        except HTTPException: 
            raise
        except Exception as e:
            print(f"[AUTH_VALIDATION_ERROR] Unexpected error: {e}")
            traceback.print_exc()
            raise credentials_exception

    async def get_current_user_with_permissions(self, token: str = Depends(oauth2_scheme)) -> Tuple[str, List[str]]:
        payload = await self._validate_token_and_get_payload(token)
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            print("[AUTH_VALIDATION_ERROR] Token payload missing 'sub' (user_id).")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found in token")

        permissions: List[str] = payload.get("permissions", [])
        return user_id, permissions

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> str:
        user_id, _ = await self.get_current_user_with_permissions(token=token)
        return user_id

    async def get_decoded_payload_with_claims(self, token: str = Depends(oauth2_scheme)) -> dict:
        payload = await self._validate_token_and_get_payload(token)
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found in token")
        payload["user_id"] = user_id 
        return payload

    async def ws_authenticate(self, websocket: WebSocket) -> Optional[str]:
        # ... (ws_authenticate logic remains the same) ...
        try:
            auth_data = await websocket.receive_text()
            message = json.loads(auth_data)
            if message.get("type") != "auth":
                await websocket.send_text(json.dumps({"type": "auth_failure", "message": "Auth message expected"}))
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None
            token = message.get("token")
            if not token:
                await websocket.send_text(json.dumps({"type": "auth_failure", "message": "Token missing"}))
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None

            payload = await self._validate_token_and_get_payload(token) 
            user_id: Optional[str] = payload.get("sub")
            if user_id:
                await websocket.send_text(json.dumps({"type": "auth_success", "user_id": user_id}))
                print(f"[WS_AUTH] WebSocket authenticated for user: {user_id}")
                return user_id
            else: 
                print(f"[WS_AUTH] Auth failed: Invalid token (user_id missing post-validation).")
                await websocket.send_text(json.dumps({"type": "auth_failure", "message": "Invalid token"}))
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None
        except WebSocketDisconnect:
            print("[WS_AUTH] Client disconnected during auth.")
            return None
        except json.JSONDecodeError:
            print("[WS_AUTH_ERROR] Non-JSON auth message.")
            # await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA) # Removed to avoid "already closed"
            return None
        except HTTPException as e: 
            print(f"[WS_AUTH_ERROR] Auth failed: {e.detail}")
            try:
                await websocket.send_text(json.dumps({"type": "auth_failure", "message": e.detail}))
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=e.detail[:123])
            except Exception: pass # NOSONAR
            return None
        except Exception as e:
            print(f"[WS_AUTH_ERROR] Unexpected error: {e}")
            traceback.print_exc()
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except: pass # NOSONAR
            return None


auth = Auth()
print(f"[INIT] {datetime.datetime.now()}: Authentication helper initialized.")

class PermissionChecker:
    # ... (PermissionChecker definition remains the same) ...
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = set(required_permissions)

    async def __call__(self, user_id_and_permissions: Tuple[str, List[str]] = Depends(auth.get_current_user_with_permissions)):
        user_id, token_permissions_list = user_id_and_permissions
        token_permissions_set = set(token_permissions_list)

        if not self.required_permissions.issubset(token_permissions_set):
            missing_perms = self.required_permissions - token_permissions_set
            print(f"User {user_id} missing permissions. Required: {self.required_permissions}, Has: {token_permissions_set}, Missing: {missing_perms}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Missing: {', '.join(missing_perms)}"
            )
        return user_id

print(f"[INIT] {datetime.datetime.now()}: Initializing HuggingFace Embedding model ({os.environ.get('EMBEDDING_MODEL_REPO_ID', 'N/A')})...")
try:
    embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])
    print(f"[INIT] {datetime.datetime.now()}: HuggingFace Embedding model initialized.")
except KeyError:
    print(f"[ERROR] {datetime.datetime.now()}: EMBEDDING_MODEL_REPO_ID not set. Embedding model not initialized.")
    embed_model = None
except Exception as e:
    print(f"[ERROR] {datetime.datetime.now()}: Failed to initialize Embedding model: {e}")
    embed_model = None

print(f"[INIT] {datetime.datetime.now()}: Initializing Neo4j Graph Driver (URI: {os.environ.get('NEO4J_URI', 'N/A')})...")
try:
    graph_driver = GraphDatabase.driver(uri=os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    graph_driver.verify_connectivity()
    print(f"[INIT] {datetime.datetime.now()}: Neo4j Graph Driver initialized and connected.")
except KeyError:
    print(f"[ERROR] {datetime.datetime.now()}: NEO4J_URI/USERNAME/PASSWORD not set. Neo4j Driver not initialized.")
    graph_driver = None
except Exception as e:
    print(f"[ERROR] {datetime.datetime.now()}: Failed to initialize Neo4j Driver: {e}")
    graph_driver = None

class WebSocketManager:
    # ... (WebSocketManager definition remains the same) ...
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        print(f"[WS_MANAGER] {datetime.datetime.now()}: WebSocketManager initialized.")
    async def connect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            print(f"[WS_MANAGER] User {user_id} already connected. Closing old one.")
            old_ws = self.active_connections[user_id]
            try:
                await old_ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="New connection")
            except Exception:
                pass
        self.active_connections[user_id] = websocket
        print(f"[WS_MANAGER] WebSocket connected: {user_id} ({len(self.active_connections)} total)")
    def disconnect(self, websocket: WebSocket):
        uid_to_remove = next((uid for uid, ws in self.active_connections.items() if ws == websocket), None)
        if uid_to_remove:
            del self.active_connections[uid_to_remove]
            print(f"[WS_MANAGER] WebSocket disconnected: {uid_to_remove} ({len(self.active_connections)} total)")
    async def send_personal_message(self, message: str, user_id: str):
        ws = self.active_connections.get(user_id)
        if ws:
            try:
                await ws.send_text(message)
            except Exception as e:
                print(f"[WS_MANAGER] Error sending to {user_id}: {e}")
                self.disconnect(ws) # Disconnect on send error
    async def broadcast(self, message: str):
        for ws in list(self.active_connections.values()): 
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws) 
    async def broadcast_json(self, data: dict):
        await self.broadcast(json.dumps(data))


manager = WebSocketManager()
print(f"[INIT] {datetime.datetime.now()}: WebSocketManager instance created.")

print(f"[INIT] {datetime.datetime.now()}: Initializing runnables...")
reflection_runnable = get_reflection_runnable()
inbox_summarizer_runnable = get_inbox_summarizer_runnable()
priority_runnable = get_priority_runnable()
graph_decision_runnable = get_graph_decision_runnable()
information_extraction_runnable = get_information_extraction_runnable()
graph_analysis_runnable = get_graph_analysis_runnable()
text_dissection_runnable = get_text_dissection_runnable()
text_conversion_runnable = get_text_conversion_runnable()
query_classification_runnable = get_query_classification_runnable()
fact_extraction_runnable = get_fact_extraction_runnable()
text_summarizer_runnable = get_text_summarizer_runnable()
text_description_runnable = get_text_description_runnable()
chat_runnable = get_chat_runnable()
agent_runnable = get_agent_runnable()
unified_classification_runnable = get_unified_classification_runnable()
internet_query_reframe_runnable = get_internet_query_reframe_runnable()
internet_summary_runnable = get_internet_summary_runnable()
print(f"[INIT] {datetime.datetime.now()}: Runnables initialization complete.")

tool_handlers: Dict[str, callable] = {} # Keep tool_handlers global for registration
print(f"[INIT] {datetime.datetime.now()}: Tool handlers registry initialized.")

print(f"[INIT] {datetime.datetime.now()}: Initializing TaskQueue...")
task_queue = TaskQueue()
print(f"[INIT] {datetime.datetime.now()}: TaskQueue initialized.")

print(f"[INIT] {datetime.datetime.now()}: Initializing MongoManager...")
mongo_manager = MongoManager()
print(f"[INIT] {datetime.datetime.now()}: MongoManager initialized.")

# Dictionary to hold active context engine instances per user
# Structure: { "user_id1": {"gmail": GmailEngineInstance, "gcalendar": GCalendarEngineInstance}, ... }
active_context_engines: Dict[str, Dict[str, BaseContextEngine]] = {}
polling_scheduler_task_handle: Optional[asyncio.Task] = None

POLLING_SCHEDULER_INTERVAL_SECONDS = 30 

# --- Helper Functions (User-Specific DB Operations) ---
async def load_user_profile(user_id: str) -> Dict[str, Any]:
    profile = await mongo_manager.get_user_profile(user_id)
    return profile if profile else {"user_id": user_id, "userData": {}}

async def write_user_profile(user_id: str, data: Dict[str, Any]) -> bool:
    return await mongo_manager.update_user_profile(user_id, data)

# get_chat_history_messages and add_message_to_db are crucial for /chat and agent background processor
async def get_chat_history_messages(user_id: str, chat_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    effective_chat_id = chat_id
    if effective_chat_id is None or effective_chat_id == "":
        user_profile = await mongo_manager.get_user_profile(user_id)
        if user_profile and "userData" in user_profile and user_profile["userData"].get("active_chat_id") not in [None, ""]:
            effective_chat_id = user_profile["userData"]["active_chat_id"]
        else:
            all_chat_ids = await mongo_manager.get_all_chat_ids_for_user(user_id)
            if all_chat_ids:
                effective_chat_id = all_chat_ids[0]
                await mongo_manager.update_user_profile(user_id, {"userData.active_chat_id": effective_chat_id})
            else:
                new_chat_id = str(uuid.uuid4())
                await mongo_manager.update_user_profile(user_id, {"userData.active_chat_id": new_chat_id})
                effective_chat_id = new_chat_id
    
    if effective_chat_id is None or effective_chat_id == "":
        print(f"[CHAT_HISTORY_ERROR] No chat_id determined for user {user_id} after all attempts.")
        return [], ""
 
    messages = await mongo_manager.get_chat_history(user_id, effective_chat_id)
    s_messages = []
    for m in messages:
        if not isinstance(m, dict): continue
        if isinstance(m.get('timestamp'), datetime.datetime):
            m['timestamp'] = m['timestamp'].isoformat()
        s_messages.append(m)
    return [m for m in s_messages if m.get("isVisible", True)], effective_chat_id


async def add_message_to_db(user_id: str, chat_id: Union[int, str], message_text: str, is_user: bool, is_visible: bool = True, **kwargs) -> Optional[str]:
    try:
        target_chat_id = str(chat_id)
    except (ValueError, TypeError):
        print(f"[ERROR] Invalid chat_id for add_message: {chat_id}")
        return None
    
    message_data = {
        "message": message_text,
        "isUser": is_user,
        "isVisible": is_visible,
        **kwargs }
    message_data = {k: v for k, v in message_data.items() if v is not None}

    try:
        message_id = await mongo_manager.add_chat_message(user_id, target_chat_id, message_data)
        return message_id
    except Exception as e:
        print(f"[ERROR] Adding message for {user_id}, chat {target_chat_id}: {e}")
        traceback.print_exc()
        return None

print(f"[INIT] {datetime.datetime.now()}: Initializing AgentTaskProcessor...")
agent_task_processor = AgentTaskProcessor(
    task_queue=task_queue,
    tool_handlers=tool_handlers,
    load_user_profile_func=load_user_profile,
    add_message_to_db_func=add_message_to_db,
    graph_driver_instance=graph_driver,
    embed_model_instance=embed_model,
    reflection_runnable_instance=reflection_runnable,
    agent_runnable_instance=agent_runnable,
    internet_query_reframe_runnable_instance=internet_query_reframe_runnable,
    internet_summary_runnable_instance=internet_summary_runnable,
    query_user_profile_func=query_user_profile,
    text_conversion_runnable_instance=text_conversion_runnable,
    query_classification_runnable_instance=query_classification_runnable,
    get_reframed_internet_query_func=get_reframed_internet_query,
    get_search_results_func=get_search_results,
    get_search_summary_func=get_search_summary
)
print(f"[INIT] {datetime.datetime.now()}: AgentTaskProcessor initialized.")

stt_model = None
tts_model = None
SELECTED_TTS_VOICE: VoiceId = "tara" # Default, can be changed
print(f"[CONFIG] {datetime.datetime.now()}: Defining database file paths...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB_DIR = os.path.join(BASE_DIR, "..", "..", "user_databases")
os.makedirs(USER_PROFILE_DB_DIR, exist_ok=True)
def get_user_profile_db_path(user_id: str) -> str:
    return os.path.join(USER_PROFILE_DB_DIR, f"{user_id}_profile.json")

print(f"[INIT] {datetime.datetime.now()}: Initializing MemoryBackend...")
memory_backend = MemoryBackend(mongo_manager=mongo_manager)
print(f"[INIT] {datetime.datetime.now()}: MemoryBackend initialized.")

def register_tool(name: str): # This must remain in app.py or be passed to AgentTaskProcessor
    def decorator(func: callable):
        print(f"[TOOL_REGISTRY] Registering tool '{name}'")
        tool_handlers[name] = func
        return func
    return decorator

# Tool function imports for registration need to happen AFTER register_tool is defined
# and tool_handlers is initialized.
# The actual tool functions are in server.agents.functions.
# Registration happens when server.agents.functions is imported IF the decorator is applied there.
# For this refactor, assuming tool registration happens in app.py by calling functions from agents.functions
# OR the decorator is applied directly in agents.functions.py.
# Let's assume registration is handled. The current structure suggests registration would
# happen when the `gmail_tool` etc. functions are defined with `@register_tool`.
# Since tool functions are moved to agents.functions, they should be decorated there.
# Or, if we keep register_tool here, then we'd call it here.
# For now, I'll assume that the @register_tool calls are done within the agents/functions.py or similar,
# and this `tool_handlers` dict is populated upon import.
# If tool functions are defined within app.py, they would be decorated here.
# The prompt mentions `gmail_tool` etc. as defined in `app.py` before,
# now they are methods inside AgentTaskProcessor or similar.
# The current structure of `AgentTaskProcessor` shows `tool_handlers` passed in.
# Let's ensure tool registration for GSuite tools happens explicitly here IF they are not auto-registered.
# The provided `app.py` already has @register_tool for gmail, gdocs etc.
# These definitions will need to be moved.
# For now, I'll keep the `@register_tool` and tool function definitions here
# and plan to move them with the `agents/routes.py` or into `agents/tools.py` in a later step if needed.
# The current request is about endpoint separation primarily.

# Re-defining tool handlers as they were in the original app.py
# These will be called by AgentTaskProcessor. They are not FastAPI endpoints.
@register_tool("gmail")
async def gmail_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = tool_call_input.get("input")
    if not user_id:
        return {"status": "failure", "error": "User_id missing for Gmail."}
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        tool_runnable = get_tool_runnable(gmail_agent_system_prompt_template, gmail_agent_user_prompt_template, gmail_agent_required_format, ["query", "username", "previous_tool_response"])
        tool_call_str_output = tool_runnable.invoke({"query": str(input_val), "username": username, "previous_tool_response": tool_call_input.get("previous_tool_response")})
        
        tool_call_dict: Dict[str, Any]
        if isinstance(tool_call_str_output, dict): 
            tool_call_dict = tool_call_str_output
        elif isinstance(tool_call_str_output, str):
            try:
                tool_call_dict = json.loads(tool_call_str_output)
            except json.JSONDecodeError:
                print(f"[ERROR] Gmail tool LLM output not JSON: {tool_call_str_output}")
                return {"status": "failure", "error": f"LLM output not valid JSON: {tool_call_str_output[:100]}..."}
        else:
            print(f"[ERROR] Gmail tool unexpected LLM output type: {type(tool_call_str_output)}")
            return {"status": "failure", "error": "Unexpected LLM output type for Gmail tool."}

        actual_tool_name = tool_call_dict.get("tool_name")
        if actual_tool_name in ["send_email", "reply_email"]: # Example approval tools
            return {"action": "approve", "tool_call": tool_call_dict}
        else:
            # parse_and_execute_tool_calls is in server.agents.helpers
            tool_result = await parse_and_execute_tool_calls(tool_call_dict) 
            return {"tool_result": tool_result} # Ensure this structure is what AgentTaskProcessor expects
    except Exception as e:
        print(f"[ERROR] Gmail tool {user_id}: {e}")
        traceback.print_exc()
        return {"status": "failure", "error": str(e)}

@register_tool("gdocs")
async def gdocs_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = str(tool_call_input.get("input",""))
    previous_tool_response = tool_call_input.get("previous_tool_response", {})
    if not user_id: return {"status": "failure", "error": "User_id missing."}
    try:
        tool_runnable = get_tool_runnable(gdocs_agent_system_prompt_template, gdocs_agent_user_prompt_template, gdocs_agent_required_format, ["query", "previous_tool_response"])
        tool_call_output = tool_runnable.invoke({"query": input_val, "previous_tool_response": previous_tool_response})
        tool_call_dict = json.loads(tool_call_output) if isinstance(tool_call_output, str) else tool_call_output
        tool_result = await parse_and_execute_tool_calls(tool_call_dict)
        return {"tool_result": tool_result}
    except Exception as e: return {"status": "failure", "error": str(e)}

@register_tool("gcalendar")
async def gcalendar_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = str(tool_call_input.get("input",""))
    previous_tool_response = tool_call_input.get("previous_tool_response", {})
    if not user_id: return {"status": "failure", "error": "User_id missing."}
    try:
        user_profile = await load_user_profile(user_id)
        timezone = user_profile.get("userData", {}).get("personalInfo", {}).get("timezone", "UTC") 
        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        tool_runnable = get_tool_runnable(gcalendar_agent_system_prompt_template, gcalendar_agent_user_prompt_template, gcalendar_agent_required_format, ["query", "current_time", "timezone", "previous_tool_response"])
        tool_call_output = tool_runnable.invoke({"query": input_val, "current_time": current_time, "timezone": timezone, "previous_tool_response": previous_tool_response})
        tool_call_dict = json.loads(tool_call_output) if isinstance(tool_call_output, str) else tool_call_output
        tool_result = await parse_and_execute_tool_calls(tool_call_dict)
        return {"tool_result": tool_result}
    except Exception as e: return {"status": "failure", "error": str(e)}

@register_tool("gsheets")
async def gsheets_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = str(tool_call_input.get("input",""))
    previous_tool_response = tool_call_input.get("previous_tool_response", {})
    if not user_id: return {"status": "failure", "error": "User_id missing."}
    try:
        tool_runnable = get_tool_runnable(gsheets_agent_system_prompt_template, gsheets_agent_user_prompt_template, gsheets_agent_required_format, ["query", "previous_tool_response"])
        tool_call_output = tool_runnable.invoke({"query": input_val, "previous_tool_response": previous_tool_response})
        tool_call_dict = json.loads(tool_call_output) if isinstance(tool_call_output, str) else tool_call_output
        tool_result = await parse_and_execute_tool_calls(tool_call_dict)
        return {"tool_result": tool_result}
    except Exception as e: return {"status": "failure", "error": str(e)}

@register_tool("gslides")
async def gslides_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = str(tool_call_input.get("input",""))
    previous_tool_response = tool_call_input.get("previous_tool_response", {})
    if not user_id: return {"status": "failure", "error": "User_id missing."}
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        tool_runnable = get_tool_runnable(gslides_agent_system_prompt_template, gslides_agent_user_prompt_template, gslides_agent_required_format, ["query", "user_name", "previous_tool_response"])
        tool_call_output = tool_runnable.invoke({"query": input_val, "user_name": username, "previous_tool_response": previous_tool_response})
        tool_call_dict = json.loads(tool_call_output) if isinstance(tool_call_output, str) else tool_call_output
        tool_result = await parse_and_execute_tool_calls(tool_call_dict)
        return {"tool_result": tool_result}
    except Exception as e: return {"status": "failure", "error": str(e)}

@register_tool("gdrive")
async def gdrive_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = str(tool_call_input.get("input",""))
    previous_tool_response = tool_call_input.get("previous_tool_response", {})
    if not user_id: return {"status": "failure", "error": "User_id missing."}
    try:
        tool_runnable = get_tool_runnable(gdrive_agent_system_prompt_template, gdrive_agent_user_prompt_template, gdrive_agent_required_format, ["query", "previous_tool_response"])
        tool_call_output = tool_runnable.invoke({"query": input_val, "previous_tool_response": previous_tool_response})
        tool_call_dict = json.loads(tool_call_output) if isinstance(tool_call_output, str) else tool_call_output
        tool_result = await parse_and_execute_tool_calls(tool_call_dict)
        return {"tool_result": tool_result}
    except Exception as e: return {"status": "failure", "error": str(e)}


print(f"[CONFIG] {datetime.datetime.now()}: Setting up Google OAuth2 configuration...")
SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/gmail.modify", "https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/presentations", "https://www.googleapis.com/auth/drive", "https://mail.google.com/"]
CREDENTIALS_DICT = {"installed": {"client_id": os.environ.get("GOOGLE_CLIENT_ID"), "project_id": os.environ.get("GOOGLE_PROJECT_ID"), "auth_uri": os.environ.get("GOOGLE_AUTH_URI"), "token_uri": os.environ.get("GOOGLE_TOKEN_URI"), "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"), "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"), "redirect_uris": ["http://localhost"]}} # Keep for utils/routes.py
print(f"[CONFIG] {datetime.datetime.now()}: Google OAuth2 config complete.")

# --- FastAPI Application Setup ---
print(f"[FASTAPI] {datetime.datetime.now()}: Initializing FastAPI app...")
app = FastAPI(title="Sentient API", description="API for Sentient application.", version="1.0.0", docs_url="/docs", redoc_url=None)
print(f"[FASTAPI] {datetime.datetime.now()}: FastAPI app initialized.")
app.add_middleware(CORSMiddleware, allow_origins=["app://.", "http://localhost:3000", "http://localhost"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
print(f"[FASTAPI] {datetime.datetime.now()}: CORS middleware added.")

agent_task_processor_instance: Optional[AgentTaskProcessor] = None 

async def polling_scheduler_loop():
    print(f"[POLLING_SCHEDULER] Starting loop (interval: {POLLING_SCHEDULER_INTERVAL_SECONDS}s)")
    if not mongo_manager:
        print("[POLLING_SCHEDULER_ERROR] MongoManager not initialized. Scheduler cannot run.")
        return

    await mongo_manager.reset_stale_polling_locks() 
    
    while True:
        try:
            # print(f"[POLLING_SCHEDULER] Checking for due polling tasks at {datetime.now(timezone.utc).isoformat()}")
            due_tasks_states = await mongo_manager.get_due_polling_tasks() 
            
            if not due_tasks_states:
                # print(f"[POLLING_SCHEDULER] No tasks due at this time.")
                pass
            else:
                print(f"[POLLING_SCHEDULER] Found {len(due_tasks_states)} due polling tasks.")

            for task_state in due_tasks_states:
                user_id = task_state["user_id"]
                service_name = task_state["service_name"] 
                
                print(f"[POLLING_SCHEDULER] Attempting to process task for {user_id}/{service_name}")

                locked_task_state = await mongo_manager.set_polling_status_and_get(user_id, service_name)
                
                if locked_task_state:
                    print(f"[POLLING_SCHEDULER] Acquired lock for {user_id}/{service_name}. Triggering poll.")
                    
                    engine_instance = active_context_engines.get(user_id, {}).get(service_name)
                    
                    if not engine_instance:
                        engine_config = DATA_SOURCES_CONFIG.get(service_name)
                        if engine_config and engine_config.get("engine_class"):
                            engine_class = engine_config["engine_class"]
                            print(f"[POLLING_SCHEDULER] Creating new {engine_class.__name__} instance for {user_id}/{service_name}")
                            # Ensure all dependencies for engine_class are passed correctly
                            engine_instance = engine_class(
                                user_id=user_id,
                                task_queue=task_queue, # Pass your global/initialized instances
                                memory_backend=memory_backend,
                                websocket_manager=manager, # Global websocket_manager
                                mongo_manager_instance=mongo_manager # Global mongo_manager
                            )
                            if user_id not in active_context_engines:
                                active_context_engines[user_id] = {}
                            active_context_engines[user_id][service_name] = engine_instance
                        else:
                            print(f"[POLLING_SCHEDULER_ERROR] No engine class configured for service: {service_name}")
                            await mongo_manager.update_polling_state(user_id, service_name, {"is_currently_polling": False})
                            continue
                    
                    asyncio.create_task(engine_instance.run_poll_cycle()) 
                else:
                    print(f"[POLLING_SCHEDULER] Could not acquire lock for {user_id}/{service_name} (already processing or no longer due).")

        except Exception as e:
            print(f"[POLLING_SCHEDULER_ERROR] Error in scheduler loop: {e}")
            traceback.print_exc() 
        
        await asyncio.sleep(POLLING_SCHEDULER_INTERVAL_SECONDS)

async def start_user_context_engines(user_id: str):
    """Ensures polling state exists for all enabled services for a user."""
    print(f"[CONTEXT_ENGINE_MGR] Starting/Ensuring context engines and polling states for user {user_id}.")
    if user_id not in active_context_engines: 
        active_context_engines[user_id] = {}
    
    if not mongo_manager:
        print(f"[CONTEXT_ENGINE_MGR_ERROR] MongoManager not initialized. Cannot start engines for user {user_id}.")
        return

    user_profile = await mongo_manager.get_user_profile(user_id)
    user_settings = user_profile.get("userData", {}) if user_profile else {}

    for service_name, config in DATA_SOURCES_CONFIG.items():
        # Default to enabled_by_default if the specific setting isn't in user_profile
        # This user-specific setting in user_profile would be set by /utils/set_data_source_enabled
        is_service_explicitly_enabled_in_profile = user_settings.get(f"{service_name}_polling_enabled")

        is_service_considered_enabled = False
        if is_service_explicitly_enabled_in_profile is not None:
            is_service_considered_enabled = is_service_explicitly_enabled_in_profile
        else: # Not set in profile, use default from config
            is_service_considered_enabled = config.get("enabled_by_default", True)


        if is_service_considered_enabled:
            engine_instance = active_context_engines.get(user_id, {}).get(service_name)
            if not engine_instance: # Engine instance not in memory, create one for initialization
                engine_class = config["engine_class"]
                print(f"[CONTEXT_ENGINE_MGR] Creating transient {engine_class.__name__} instance for {user_id}/{service_name} for state init.")
                engine_instance = engine_class(
                    user_id=user_id,
                    task_queue=task_queue, 
                    memory_backend=memory_backend,
                    websocket_manager=manager, # global websocket_manager
                    mongo_manager_instance=mongo_manager # global mongo_manager
                )
                # We might not need to store it in active_context_engines if scheduler recreates on demand
                # However, initialize_polling_state needs an instance.
            
            await engine_instance.initialize_polling_state() 
            print(f"[CONTEXT_ENGINE_MGR] Polling state ensured for {service_name} for user {user_id}.")
        else:
            print(f"[CONTEXT_ENGINE_MGR] Service {service_name} is disabled for user {user_id}. Skipping engine start/state init.")
            # If disabling, also ensure the polling state in DB is marked as is_enabled=False
            await mongo_manager.update_polling_state(user_id, service_name, {"is_enabled": False})


@app.on_event("startup")
async def startup_event():
    print(f"[FASTAPI_LIFECYCLE] App startup at {datetime.now(timezone.utc).isoformat()}")
    global agent_task_processor_instance, polling_scheduler_task_handle, mongo_manager, task_queue, memory_backend, manager
    # ... (STT, TTS init as before) ...
    print("[LIFECYCLE] Loading STT...")
    try:
        stt_model = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
        print("[LIFECYCLE] STT loaded.")
    except Exception as e:
        print(f"[ERROR] STT model failed to load. Voice features will be unavailable. Details: {e}")
        stt_model = None
    
    print("[LIFECYCLE] Loading TTS...")
    try:
        if IS_DEV_ENVIRONMENT:
            tts_model = OrpheusTTS(verbose=False, default_voice_id=SELECTED_TTS_VOICE)
            print("[LIFECYCLE] Orpheus TTS loaded (Developer Mode).")
        else:
            if TTS_PROVIDER == "GCP":
                tts_model = GCPTTS()
                print("[LIFECYCLE] GCP TTS loaded (Production Mode).")
            elif TTS_PROVIDER == "ELEVENLABS":
                tts_model = ElevenLabsTTS()
                print("[LIFECYCLE] ElevenLabs TTS loaded (Production Mode).")
            else: 
                tts_model = OrpheusTTS(verbose=False, default_voice_id=SELECTED_TTS_VOICE)
                print(f"[WARN] Unknown TTS_PROVIDER '{TTS_PROVIDER}'. Defaulting to Orpheus TTS.")
    except Exception as e:
        print(f"[ERROR] TTS model failed to load. Voice features will be unavailable. Details: {e}")
        tts_model = None

    mongo_manager = MongoManager() 
    await mongo_manager.initialize_db() 
    
    # Assuming task_queue and memory_backend are initialized globally or here using mongo_manager
    # task_queue = TaskQueue() # If it uses its own DB connection
    # memory_backend = MemoryBackend(mongo_manager=mongo_manager) # Pass the initialized manager
    
    # Re-initialize these if they depend on mongo_manager being set
    if 'task_queue' not in globals() or task_queue is None:
        task_queue = TaskQueue() # Initialize if not already
        await task_queue.initialize_db()

    if 'memory_backend' not in globals() or memory_backend is None:
        memory_backend = MemoryBackend(mongo_manager=mongo_manager)
        await memory_backend.memory_queue.initialize_db()

    if 'manager' not in globals() or manager is None: # Assuming 'manager' is your WebSocketManager
        manager = WebSocketManager()


    agent_task_processor_instance = agent_task_processor 

    if agent_task_processor_instance:
        asyncio.create_task(agent_task_processor_instance.process_queue())
        asyncio.create_task(agent_task_processor_instance.cleanup_tasks_periodically())
    else:
        print("[ERROR] AgentTaskProcessor not initialized. Task processing will not start.")
        
    asyncio.create_task(memory_backend.process_memory_operations())
    
    polling_scheduler_task_handle = asyncio.create_task(polling_scheduler_loop())
    print(f"[FASTAPI_LIFECYCLE] Central polling scheduler started.")
    
    print(f"[FASTAPI_LIFECYCLE] App startup complete at {datetime.now(timezone.utc).isoformat()}")


@app.on_event("shutdown")
async def shutdown_event():
    # ... (your existing shutdown code) ...
    global polling_scheduler_task_handle
    if polling_scheduler_task_handle and not polling_scheduler_task_handle.done():
        print("[FASTAPI_LIFECYCLE] Cancelling polling scheduler task...")
        polling_scheduler_task_handle.cancel()
        try:
            await polling_scheduler_task_handle
        except asyncio.CancelledError:
            print("[FASTAPI_LIFECYCLE] Polling scheduler task cancelled.")
        except Exception as e:
            print(f"[FASTAPI_LIFECYCLE_ERROR] Error during polling scheduler shutdown: {e}")
    # ... (rest of your shutdown) ...
    print(f"[FASTAPI_LIFECYCLE] App shutdown.")
    if mongo_manager.client:
        mongo_manager.client.close()
        print("[FASTAPI_LIFECYCLE] MongoManager client closed.")
    if task_queue.client: 
        task_queue.client.close()
        print("[FASTAPI_LIFECYCLE] TaskQueue MongoDB client closed.")
    if memory_backend.memory_queue.client: 
        memory_backend.memory_queue.client.close()
        print("[FASTAPI_LIFECYCLE] MemoryQueue MongoDB client closed.")
    print(f"[FASTAPI_LIFECYCLE] All MongoDB clients known to app.py closed.")


# --- API Endpoints ---
# Import and include routers
from server.agents import routes as agent_routes
from server.memory import routes as memory_routes
from server.common import routes as common_routes
from server.utils import routes as util_routes

app.include_router(agent_routes.router)
app.include_router(memory_routes.router)
app.include_router(common_routes.router)
app.include_router(util_routes.router)

# The root endpoint can stay in app.py or be moved to utils/routes.py
@app.get("/", status_code=status.HTTP_200_OK, summary="API Root", tags=["General"])
async def main_root():
    return {"message": "Sentient API is running."}

# --- Voice Endpoint (FastRTC - remains in app.py due to specific mounting) ---
async def handle_audio_conversation(audio: tuple[int, np.ndarray]) -> AsyncGenerator[tuple[int, np.ndarray], None]:
     user_id_for_voice = "PLACEHOLDER_VOICE_USER_ID" # Needs actual user_id integration
     print(f"\n--- [VOICE] Audio chunk received (User: {user_id_for_voice}) ---")

     if not stt_model:
         print("[VOICE_ERROR] STT model not available. Cannot process audio.")
         yield (0, np.array([], dtype=np.float32)) # Ensure correct dtype for empty array
         return
     
     user_transcribed_text = stt_model.stt(audio)
     if not user_transcribed_text or not user_transcribed_text.strip():
         print("[VOICE] Empty transcription from STT.")
         return 
         
     print(f"[VOICE] User (STT - {user_id_for_voice}): {user_transcribed_text}")
     
     # Placeholder for getting a response (e.g., from Ollama or another LLM)
     bot_response_text = f"Voice processing is a work in progress for user {user_id_for_voice}. You said: '{user_transcribed_text}'"
     # In a real scenario, you would call your chat logic here:
     # bot_response_text = await get_chat_response(user_id_for_voice, user_transcribed_text)
     
     if not tts_model:
         print("[VOICE_ERROR] TTS model not available. Cannot generate audio response.")
         yield (0, np.array([], dtype=np.float32))
         return
         
     tts_options: TTSOptions = {"voice_id": SELECTED_TTS_VOICE} 
     async for sr, chunk in tts_model.stream_tts(bot_response_text, options=tts_options): # OrpheusTTS.stream_tts is async
          if chunk is not None and chunk.size > 0: # Ensure chunk is not None and has data
              # Convert chunk to bytes if it's not already (FastRTC expects bytes)
              # OrpheusTTS yields (sr, np.ndarray[np.float32])
              # FastRTC expects (sr, np.ndarray[np.int16]) or bytes
              # Let's assume FastRTC can handle float32 or it needs conversion here.
              # For now, assuming FastRTC can handle it.
              # If not, convert: audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
              yield (sr, chunk)
              
     print(f"--- [VOICE] Finished processing audio response for {user_id_for_voice} ---")

voice_stream_handler = Stream(
    ReplyOnPause(
        handle_audio_conversation,
        algo_options=AlgoOptions(),
        model_options=SileroVadOptions(),
        can_interrupt=False
    ),
    mode="send-receive",
    modality="audio"
)
voice_stream_handler.mount(app, path="/voice")
print(f"[FASTAPI] {datetime.datetime.now()}: FastRTC voice stream mounted at /voice.")

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() 

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelname)s %(client_addr)s - \"%(request_line)s\" %(status_code)s" # Corrected format
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s [%(name)s] %(message)s" # Corrected format
    log_config["loggers"]["uvicorn.error"]["level"] = "INFO"

    print(f"[UVICORN] Starting Uvicorn server on host 0.0.0.0, port 5000...")
    print(f"[UVICORN] API Documentation available at http://localhost:5000/docs")
    
    uvicorn.run(
        "server.app.app:app", # Updated to point to the app instance within the refactored app.py
        host="0.0.0.0",
        port=5000,
        lifespan="on",
        reload=IS_DEV_ENVIRONMENT, # Enable reload only in dev
        workers=1, # For simplicity in dev; adjust for prod
        log_config=log_config
    )