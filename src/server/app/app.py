# Record script start time for performance monitoring and logging
import time
import datetime # Moved for early use and combined

START_TIME = time.time()
print(f"[STARTUP] {datetime.datetime.now()}: Script execution started.")

# --- Core Python and System Imports ---
import os
import json
import asyncio
import pickle
import multiprocessing
import traceback # For detailed error printing

# --- Third-Party Library Imports ---
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    status,
    Security # For SecurityScopes with Depends
)
from fastapi.security import OAuth2PasswordBearer, SecurityScopes # For Authentication
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse # For various response types
from fastapi.middleware.cors import CORSMiddleware # For Cross-Origin Resource Sharing
from fastapi.staticfiles import StaticFiles # For serving static files
from pydantic import BaseModel, Field # For data validation and settings management
from typing import Optional, Any, Dict, List, AsyncGenerator, Union, Tuple # For type hinting

# --- Authentication Specific Imports ---
import requests # For fetching JWKS (JSON Web Key Set)
from jose import jwt, JWTError # For JWT (JSON Web Token) validation
from jose.exceptions import JOSEError # For handling JOSE specific errors

# --- Machine Learning and NLP Imports ---
from neo4j import GraphDatabase # For Neo4j graph database interaction
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # For text embeddings

# --- Google API Imports ---
from google_auth_oauthlib.flow import InstalledAppFlow # For Google OAuth flow
from google.auth.transport.requests import Request # For Google API requests

# --- Environment and Configuration Imports ---
from dotenv import load_dotenv # For loading environment variables from .env files
import nest_asyncio # For allowing nested asyncio event loops (e.g., in Jupyter)
import uvicorn # For running the FastAPI application

# --- Utility and Other Imports ---
import numpy as np # For numerical operations, often used with ML models
import gradio as gr # Still needed for FastRTC internals (real-time communication)
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions # For FastRTC components
import httpx # For asynchronous HTTP requests

print(f"[STARTUP] {datetime.datetime.now()}: Basic imports completed.")

# --- Application-Specific Module Imports ---
# Assuming APPROVAL_PENDING_SIGNAL and PERSONALITY_DESCRIPTIONS are defined in these imports
print(f"[STARTUP] {datetime.datetime.now()}: Importing model components...")
from server.agents.runnables import *
from server.agents.functions import *
from server.agents.prompts import *
from server.agents.formats import *
from server.agents.base import *
from server.agents.helpers import *

# Memory: Short-term and long-term memory management
from server.memory.runnables import *
from server.memory.functions import *
from server.memory.prompts import *
from server.memory.constants import *
from server.memory.formats import *
from server.memory.backend import MemoryBackend

# Utils: General utility functions
from server.utils.helpers import *

# Auth: Authentication related helpers (e.g., AES encryption, management tokens)
from server.auth.helpers import * # Contains AES, get_management_token

# Common: Shared components across different modules
from server.common.functions import *
from server.common.runnables import *
from server.common.prompts import *
from server.common.formats import *

# Chat: Chat-specific logic and runnables
from server.chat.runnables import *
from server.chat.prompts import *
from server.chat.functions import *

# Context: Engines for pulling context from various sources
from server.context.gmail import GmailContextEngine
from server.context.internet import InternetSearchContextEngine
from server.context.gcalendar import GCalendarContextEngine

# Voice: Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities
from server.voice.stt import FasterWhisperSTT
from server.voice.orpheus_tts import OrpheusTTS, TTSOptions, VoiceId, AVAILABLE_VOICES

# --- Environment Variable Loading ---
print(f"[STARTUP] {datetime.datetime.now()}: Loading environment variables from server/.env...")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
print(f"[STARTUP] {datetime.datetime.now()}: Environment variables loaded from {dotenv_path}")

# --- Asyncio Configuration ---
print(f"[STARTUP] {datetime.datetime.now()}: Applying nest_asyncio...")
nest_asyncio.apply()
print(f"[STARTUP] {datetime.datetime.now()}: nest_asyncio applied.")

# --- Global Initializations ---
print(f"[INIT] {datetime.datetime.now()}: Starting global initializations...")
DATA_SOURCES = ["gmail", "internet_search", "gcalendar"]

# --- Auth0 Configuration & JWKS (JSON Web Key Set) ---
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE") # THIS IS NOW YOUR CUSTOM API AUDIENCE
ALGORITHMS = ["RS256"]
CUSTOM_CLAIMS_NAMESPACE = f"{AUTH0_AUDIENCE}/" if AUTH0_AUDIENCE and AUTH0_AUDIENCE.endswith('/') else f"{AUTH0_AUDIENCE}/" # Ensure trailing slash for namespace

if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
    print("[ERROR] FATAL: AUTH0_DOMAIN or AUTH0_AUDIENCE (for custom API) not set!")
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
    exit(1) # Exit if JWKS cannot be fetched
except Exception as e:
    print(f"[ERROR] FATAL: Error processing JWKS: {e}")
    exit(1) # Exit on other JWKS processing errors

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Dummy tokenUrl

# --- JWT Validation and Permission Handling Logic ---
class Auth:
    """
    Handles authentication and authorization using JWTs from Auth0 for the custom API.
    """
    async def _validate_token_and_get_payload(self, token: str) -> dict:
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
                audience=AUTH0_AUDIENCE, # Your custom API audience
                issuer=f"https://{AUTH0_DOMAIN}/"
            )
            return payload
        except JWTError as e:
            print(f"[AUTH_VALIDATION_ERROR] JWT Error: {e}")
            raise credentials_exception
        except JOSEError as e:
            print(f"[AUTH_VALIDATION_ERROR] JOSE Error: {e}")
            raise credentials_exception
        except HTTPException: # Catch specific HTTP exceptions first
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
        # print(f"[AUTH_VALIDATION] User: {user_id}, Permissions: {permissions}")
        return user_id, permissions

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> str:
        user_id, _ = await self.get_current_user_with_permissions(token=token)
        return user_id

    async def get_decoded_payload_with_claims(self, token: str = Depends(oauth2_scheme)) -> dict:
        payload = await self._validate_token_and_get_payload(token)
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found in token")
        payload["user_id"] = user_id # Make user_id easily accessible
        return payload

    async def ws_authenticate(self, websocket: WebSocket) -> Optional[str]:
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

            payload = await self._validate_token_and_get_payload(token) # Reuses core validation
            user_id: Optional[str] = payload.get("sub")
            if user_id:
                await websocket.send_text(json.dumps({"type": "auth_success", "user_id": user_id}))
                print(f"[WS_AUTH] WebSocket authenticated for user: {user_id}")
                return user_id
            else: # Should be caught by _validate_token if sub is missing
                print(f"[WS_AUTH] Auth failed: Invalid token (user_id missing post-validation).")
                await websocket.send_text(json.dumps({"type": "auth_failure", "message": "Invalid token"}))
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None
        except WebSocketDisconnect:
            print("[WS_AUTH] Client disconnected during auth.")
            return None
        except json.JSONDecodeError:
            print("[WS_AUTH_ERROR] Non-JSON auth message.")
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return None
        except HTTPException as e: # Catch validation errors from _validate_token_and_get_payload
            print(f"[WS_AUTH_ERROR] Auth failed: {e.detail}")
            await websocket.send_text(json.dumps({"type": "auth_failure", "message": e.detail}))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=e.detail[:123])
            return None
        except Exception as e:
            print(f"[WS_AUTH_ERROR] Unexpected error: {e}")
            traceback.print_exc()
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except: # NOSONAR
                pass
            return None

auth = Auth()
print(f"[INIT] {datetime.datetime.now()}: Authentication helper initialized for custom API.")

class PermissionChecker:
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
        return user_id # Return user_id if check passes for endpoint use


# --- Initialize embedding model, Neo4j driver, WebSocketManager, runnables, TaskQueue, voice models etc. ---
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
                self.disconnect(ws)
    async def broadcast(self, message: str):
        for ws in list(self.active_connections.values()): # Iterate on a copy
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws) # Disconnect on error
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
chat_history = get_chat_history()
chat_runnable = get_chat_runnable(chat_history)
agent_runnable = get_agent_runnable(chat_history)
unified_classification_runnable = get_unified_classification_runnable(chat_history)
internet_query_reframe_runnable = get_internet_query_reframe_runnable()
internet_summary_runnable = get_internet_summary_runnable()
print(f"[INIT] {datetime.datetime.now()}: Runnables initialization complete.")
tool_handlers: Dict[str, callable] = {}
print(f"[INIT] {datetime.datetime.now()}: Tool handlers registry initialized.")
print(f"[INIT] {datetime.datetime.now()}: Initializing TaskQueue...")
task_queue = TaskQueue()
print(f"[INIT] {datetime.datetime.now()}: TaskQueue initialized.")
stt_model = None
tts_model = None
SELECTED_TTS_VOICE: VoiceId = "tara"
print(f"[CONFIG] {datetime.datetime.now()}: Defining database file paths...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB_DIR = os.path.join(BASE_DIR, "..", "..", "user_databases")
CHAT_DB_DIR = os.path.join(BASE_DIR, "..", "..", "chat_databases")
NOTIFICATIONS_DB_DIR = os.path.join(BASE_DIR, "..", "..", "notification_databases")
os.makedirs(USER_PROFILE_DB_DIR, exist_ok=True)
os.makedirs(CHAT_DB_DIR, exist_ok=True)
os.makedirs(NOTIFICATIONS_DB_DIR, exist_ok=True)
def get_user_profile_db_path(user_id: str) -> str:
    return os.path.join(USER_PROFILE_DB_DIR, f"{user_id}_profile.json")
def get_user_chat_db_path(user_id: str) -> str:
    return os.path.join(CHAT_DB_DIR, f"{user_id}_chats.json")
def get_user_notifications_db_path(user_id: str) -> str:
    return os.path.join(NOTIFICATIONS_DB_DIR, f"{user_id}_notifications.json")
print(f"[CONFIG] {datetime.datetime.now()}: User-specific database directories set.")
db_lock = asyncio.Lock()
notifications_db_lock = asyncio.Lock()
profile_db_lock = asyncio.Lock()
print(f"[INIT] {datetime.datetime.now()}: Global database locks initialized.")
initial_db = {"chats": [], "active_chat_id": 0, "next_chat_id": 1}
print(f"[CONFIG] {datetime.datetime.now()}: Initial chat DB structure defined.")
print(f"[INIT] {datetime.datetime.now()}: Initializing MemoryBackend...")
memory_backend = MemoryBackend()
print(f"[INIT] {datetime.datetime.now()}: MemoryBackend initialized.")
def register_tool(name: str):
    def decorator(func: callable):
        print(f"[TOOL_REGISTRY] Registering tool '{name}'")
        tool_handlers[name] = func
        return func
    return decorator
print(f"[CONFIG] {datetime.datetime.now()}: Setting up Google OAuth2 configuration...")
SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/gmail.modify", "https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/presentations", "https://www.googleapis.com/auth/drive", "https://mail.google.com/"]
CREDENTIALS_DICT = {"installed": {"client_id": os.environ.get("GOOGLE_CLIENT_ID"), "project_id": os.environ.get("GOOGLE_PROJECT_ID"), "auth_uri": os.environ.get("GOOGLE_AUTH_URI"), "token_uri": os.environ.get("GOOGLE_TOKEN_URI"), "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"), "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"), "redirect_uris": ["http://localhost"]}}
print(f"[CONFIG] {datetime.datetime.now()}: Google OAuth2 config complete.")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID_M2M")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET_M2M")
MANAGEMENT_API_AUDIENCE = os.getenv("AUTH0_MANAGEMENT_API_AUDIENCE") # e.g. https://YOUR_DOMAIN/api/v2/
print(f"[CONFIG] {datetime.datetime.now()}: Auth0 Management API (M2M) config loaded.")

# --- Helper Functions (User-Specific File Operations) ---
async def load_user_profile(user_id: str) -> Dict[str, Any]:
    profile_path = get_user_profile_db_path(user_id)
    try:
        async with profile_db_lock:
            if not os.path.exists(profile_path):
                return {"userData": {}} # Return default if not exists
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"userData": {}}
    except Exception as e:
        print(f"[ERROR] Loading profile {user_id}: {e}")
        return {"userData": {}}
async def write_user_profile(user_id: str, data: Dict[str, Any]) -> bool:
    profile_path = get_user_profile_db_path(user_id)
    try:
        async with profile_db_lock:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
    except Exception as e:
        print(f"[ERROR] Writing profile {user_id}: {e}")
        return False
async def load_notifications_db(user_id: str) -> Dict[str, Any]:
    path = get_user_notifications_db_path(user_id)
    async with notifications_db_lock:
        try:
            if not os.path.exists(path):
                return {"notifications": [], "next_notification_id": 1}
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "notifications" not in data:
                data["notifications"] = []
            if "next_notification_id" not in data:
                data["next_notification_id"] = 1
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {"notifications": [], "next_notification_id": 1}
        except Exception as e:
            print(f"[ERROR] Loading notifications {user_id}: {e}")
            return {"notifications": [], "next_notification_id": 1}
async def save_notifications_db(user_id: str, data: Dict[str, Any]):
    path = get_user_notifications_db_path(user_id)
    async with notifications_db_lock:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Saving notifications {user_id}: {e}")
async def load_db(user_id: str) -> Dict[str, Any]: # Chat DB
    path = get_user_chat_db_path(user_id)
    async with db_lock:
        try:
            if not os.path.exists(path):
                return initial_db.copy()
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in ["active_chat_id", "next_chat_id"]:
                data[key] = int(data.get(key, 0 if key == "active_chat_id" else 1))
            if "chats" not in data:
                data["chats"] = []
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return initial_db.copy()
        except Exception as e:
            print(f"[ERROR] Loading chat DB {user_id}: {e}")
            return initial_db.copy()
async def save_db(user_id: str, data: Dict[str, Any]): # Chat DB
    path = get_user_chat_db_path(user_id)
    async with db_lock:
        try:
            for key in ["active_chat_id", "next_chat_id"]:
                data[key] = int(data.get(key, 0 if key == "active_chat_id" else 1))
            for chat_item in data.get("chats", []): # Renamed 'chat' to 'chat_item' to avoid conflict
                chat_item["id"] = int(chat_item.get("id", 0))
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Saving chat DB {user_id}: {e}")

async def update_neo4j_with_personal_info(user_id: str, personal_info: dict, graph_driver_instance: GraphDatabase.driver):
    if not graph_driver_instance:
        print(f"[NEO4J_PERSONAL_INFO_UPDATE_ERROR] Neo4j driver not available for user {user_id}.")
        return

    name = personal_info.get("name")
    location = personal_info.get("location")
    dob = personal_info.get("dateOfBirth") # Key from frontend

    if not name: # Name is a key piece of info for a User node
        print(f"[NEO4J_PERSONAL_INFO_UPDATE_WARN] User name not provided for {user_id}. Skipping Neo4j update for personal info.")
        return

    try:
        # Neo4j driver operations are typically synchronous, run in a separate thread
        def _update_graph_db(tx, user_id, name, location, dob):
            # MERGE User node based on userId. Set properties.
            # Create a UserProfile node if it doesn't exist and link it.
            # Basic info (name, location, DOB) can be on User or UserProfile.
            # Here, adding to User node and ensuring UserProfile exists and is linked.
            # Also, adding/updating a generic 'username' property on User node for consistency with other graph functions
            tx.run("""
                MERGE (u:User {userId: $userId})
                ON CREATE SET u.name = $name, u.location = $location, u.dateOfBirth = $dob, u.username = $name, u.createdAt = timestamp()
                ON MATCH SET u.name = $name, u.location = $location, u.dateOfBirth = $dob, u.username = $name, u.updatedAt = timestamp()
                MERGE (up:UserProfile {userId: $userId}) // Ensure UserProfile node exists, can add more props later
                ON CREATE SET up.createdAt = timestamp()
                ON MATCH SET up.updatedAt = timestamp()
                MERGE (u)-[:HAS_PROFILE]->(up)
            """, userId=user_id, name=name, location=location, dob=dob)

        with graph_driver_instance.session(database="neo4j") as session:
            await asyncio.to_thread(session.execute_write, _update_graph_db, user_id, name, location, dob)
        print(f"[NEO4J_PERSONAL_INFO_UPDATE] Successfully updated personal info for user {user_id} in Neo4j.")

    except Exception as e:
        print(f"[NEO4J_PERSONAL_INFO_UPDATE_ERROR] Failed to update personal info for user {user_id} in Neo4j: {e}")
        traceback.print_exc()

async def get_chat_history_messages(user_id: str) -> List[Dict[str, Any]]:
    # print(f"[CHAT_HISTORY] get_chat_history_messages for user {user_id}.")
    async with db_lock:
        chatsDb = await load_db(user_id)
        active_chat_id = chatsDb.get("active_chat_id", 0)
        next_chat_id = chatsDb.get("next_chat_id", 1)
        current_time = datetime.datetime.now(timezone.utc)
        existing_chats = chatsDb.get("chats", [])
        active_chat = None
        if active_chat_id == 0:
            if not existing_chats:
                new_chat_id = next_chat_id
                new_chat = {"id": new_chat_id, "messages": []}
                chatsDb["chats"] = [new_chat]
                chatsDb["active_chat_id"] = new_chat_id
                chatsDb["next_chat_id"] = new_chat_id + 1
                await save_db(user_id, chatsDb)
                return []
            else:
                latest_chat_id = existing_chats[-1]['id']
                chatsDb['active_chat_id'] = latest_chat_id
                active_chat_id = latest_chat_id
                await save_db(user_id, chatsDb)
        active_chat = next((c for c in existing_chats if c.get("id") == active_chat_id), None)
        if not active_chat:
            chatsDb["active_chat_id"] = 0
            await save_db(user_id, chatsDb)
            return []
        if active_chat.get("messages"):
            try:
                last_message_ts_str = active_chat["messages"][-1].get("timestamp")
                if last_message_ts_str:
                    last_timestamp = datetime.fromisoformat(last_message_ts_str.replace('Z', '+00:00'))
                    if (current_time - last_timestamp).total_seconds() > 600: # 10 min inactivity
                        new_chat_id = next_chat_id
                        new_chat = {"id": new_chat_id, "messages": []}
                        chatsDb["chats"].append(new_chat)
                        chatsDb["active_chat_id"] = new_chat_id
                        chatsDb["next_chat_id"] = new_chat_id + 1
                        await save_db(user_id, chatsDb)
                        return []
            except Exception as e:
                print(f"[WARN] Inactivity check error for {user_id}, chat {active_chat_id}: {e}")
        if active_chat and active_chat.get("messages"):
            return [m for m in active_chat["messages"] if m.get("isVisible", True)]
        else:
            return []
async def add_message_to_db(user_id: str, chat_id: Union[int, str], message_text: str, is_user: bool, is_visible: bool = True, **kwargs) -> Optional[str]:
    try:
        target_chat_id = int(chat_id)
    except (ValueError, TypeError):
        print(f"[ERROR] Invalid chat_id for add_message: {chat_id}")
        return None
    async with db_lock:
        try:
            chatsDb = await load_db(user_id)
            active_chat = next((c for c in chatsDb.get("chats", []) if c.get("id") == target_chat_id), None)
            if active_chat:
                message_id = str(int(time.time() * 1000))
                new_message = {"id": message_id, "message": message_text, "isUser": is_user, "isVisible": is_visible, "timestamp": datetime.datetime.now(timezone.utc).isoformat(), **kwargs}
                new_message = {k: v for k, v in new_message.items() if v is not None}
                if "messages" not in active_chat:
                    active_chat["messages"] = []
                active_chat["messages"].append(new_message)
                await save_db(user_id, chatsDb)
                return message_id
            else:
                print(f"[ERROR] Chat ID {target_chat_id} not found for user {user_id}")
                return None
        except Exception as e:
            print(f"[ERROR] Adding message for {user_id}, chat {target_chat_id}: {e}")
            traceback.print_exc()
            return None

# --- Background Task Processors ---
async def cleanup_tasks_periodically():
    print(f"[TASK_CLEANUP] Starting periodic task cleanup loop.")
    while True:
        await asyncio.sleep(3600)
        try:
            await task_queue.delete_old_completed_tasks()
            print(f"[TASK_CLEANUP] Cleanup complete.")
        except Exception as e:
            print(f"[ERROR] Task cleanup error: {e}")

async def process_queue():
    print(f"[TASK_PROCESSOR] Starting task processing loop.")
    while True:
        task = await task_queue.get_next_task()
        if task:
            task_id = task.get("task_id", "N/A")
            user_id = task.get("user_id", "N/A")
            chat_id = task.get("chat_id", "N/A")

            if user_id == "N/A":
                await task_queue.complete_task(task_id, error="Task missing user_id", status="error")
                continue
            try:
                task_queue.current_task_execution = asyncio.create_task(execute_agent_task(user_id, task))
                result = await task_queue.current_task_execution
                
                if result != APPROVAL_PENDING_SIGNAL:
                    if chat_id != "N/A":
                        await add_message_to_db(user_id, chat_id, task["description"], is_user=True, is_visible=False)
                        await add_message_to_db(user_id, chat_id, result, is_user=False, is_visible=True, type="tool_result", task=task["description"], agentsUsed=True)
                    await task_queue.complete_task(task_id, result=result)
            
            except asyncio.CancelledError:
                await task_queue.complete_task(task_id, error="Task cancelled", status="cancelled")
            except Exception as e:
                await task_queue.complete_task(task_id, error=str(e), status="error")
                traceback.print_exc()
            finally:
                task_queue.current_task_execution = None
        else:
            await asyncio.sleep(0.1)

async def process_memory_operations():
    print(f"[MEMORY_PROCESSOR] Starting memory op loop.")
    while True:
        operation = await memory_backend.memory_queue.get_next_operation()
        if operation:
            op_id = operation.get("operation_id", "N/A")
            user_id = operation.get("user_id", "N/A")
            memory_data = operation.get("memory_data", "N/A")

            if user_id == "N/A":
                await memory_backend.memory_queue.complete_operation(op_id, error="Op missing user_id", status="error")
                continue
            try:
                await memory_backend.update_memory(user_id, memory_data)
                await memory_backend.memory_queue.complete_operation(op_id, result="Success")
            except Exception as e:
                await memory_backend.memory_queue.complete_operation(op_id, error=str(e), status="error")
                traceback.print_exc()
        else:
            await asyncio.sleep(0.1)

# --- Task Execution Logic ---
async def execute_agent_task(user_id: str, task: dict) -> str:
    task_id = task.get("task_id", "N/A")
    print(f"[AGENT_EXEC] Executing task {task_id} for User: {user_id}...")
    user_profile = await load_user_profile(user_id)
    username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
    personality = user_profile.get("userData", {}).get("personality", "Default")
    transformed_input = task.get("description", "")
    use_personal_context = task.get("use_personal_context", False)
    internet_search_needed = task.get("internet", "None") != "None"
    user_context_str, internet_context_str = None, None
    if use_personal_context:
        try:
            if graph_driver and embed_model: # Add other runnables if query_user_profile needs them
                user_context_str = await asyncio.to_thread(query_user_profile, user_id, transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
            else:
                user_context_str = "User context unavailable (dependencies missing)."
        except Exception as e:
            user_context_str = f"Error retrieving user context: {e}"
    if internet_search_needed:
        try:
            reframed = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
            results = get_search_results(reframed)
            internet_context_str = get_search_summary(internet_summary_runnable, results)
        except Exception as e:
            internet_context_str = f"Error retrieving internet context: {e}"
    agent_input = {"query": transformed_input, "name": username, "user_context": user_context_str, "internet_context": internet_context_str, "personality": personality}
    try:
        response = agent_runnable.invoke(agent_input)
    except Exception as e:
        return f"Error: Agent failed: {e}"
    tool_calls = response.get("tool_calls", []) if isinstance(response, dict) else (response if isinstance(response, list) else [])
    if not tool_calls and isinstance(response, str):
        return response # Direct answer
    if not tool_calls:
        return "Agent decided no tools were needed or failed to propose tools."
    all_tool_results = []
    previous_tool_result = None
    for i, tool_call_item in enumerate(tool_calls):
        tool_content = tool_call_item.get("content") if isinstance(tool_call_item, dict) and tool_call_item.get("response_type") == "tool_call" else tool_call_item
        if not isinstance(tool_content, dict):
            all_tool_results.append({"tool_name": "unknown", "tool_result": "Invalid tool call format", "status": "error"})
            continue
        tool_name = tool_content.get("tool_name")
        task_instruction = tool_content.get("task_instruction")
        if not tool_name or not task_instruction:
            all_tool_results.append({"tool_name": tool_name or "unknown", "tool_result": "Missing name/instruction", "status": "error"})
            continue
        tool_handler = tool_handlers.get(tool_name)
        if not tool_handler:
            all_tool_results.append({"tool_name": tool_name, "tool_result": f"Tool '{tool_name}' not found.", "status": "error"})
            continue
        tool_input = {"input": str(task_instruction), "user_id": user_id}
        if tool_content.get("previous_tool_response", False) and previous_tool_result:
            tool_input["previous_tool_response"] = previous_tool_result
        try:
            tool_result_main = await tool_handler(tool_input)
        except Exception as e:
            tool_result_main = f"Error executing tool '{tool_name}': {e}"
            traceback.print_exc()
        if isinstance(tool_result_main, dict) and tool_result_main.get("action") == "approve":
            await task_queue.set_task_approval_pending(task_id, tool_result_main.get("tool_call", {}))
            return APPROVAL_PENDING_SIGNAL # Assuming APPROVAL_PENDING_SIGNAL is defined
        else:
            tool_result = tool_result_main.get("tool_result", tool_result_main) if isinstance(tool_result_main, dict) else tool_result_main
            previous_tool_result = tool_result
            all_tool_results.append({"tool_name": tool_name, "tool_result": tool_result, "status": "success" if not (isinstance(tool_result_main, dict) and tool_result_main.get("status") == "failure") else "failure"})
    if not all_tool_results:
        return "No successful tool actions performed."
    try:
        final_result_str = reflection_runnable.invoke({"tool_results": all_tool_results})
    except Exception as e:
        final_result_str = f"Error generating final summary: {e}"
    return final_result_str

# --- FastAPI Application Setup ---
print(f"[FASTAPI] {datetime.datetime.now()}: Initializing FastAPI app...")
app = FastAPI(title="Sentient API", description="API for Sentient application.", version="1.0.0", docs_url="/docs", redoc_url=None)
print(f"[FASTAPI] {datetime.datetime.now()}: FastAPI app initialized.")
app.add_middleware(CORSMiddleware, allow_origins=["app://.", "http://localhost:3000", "http://localhost"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
print(f"[FASTAPI] {datetime.datetime.now()}: CORS middleware added.")

@app.on_event("startup")
async def startup_event():
    print(f"[FASTAPI_LIFECYCLE] App startup.")
    global stt_model, tts_model
    print("[LIFECYCLE] Loading STT...")
    try:
        stt_model = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
        print("[LIFECYCLE] STT loaded.")
    except Exception as e:
        print(f"[ERROR] STT model failed to load. Voice features will be unavailable. Details: {e}")
        stt_model = None
    
    print("[LIFECYCLE] Loading TTS...")
    try:
        tts_model = OrpheusTTS(verbose=False, default_voice_id=SELECTED_TTS_VOICE)
        print("[LIFECYCLE] TTS loaded.")
    except Exception as e:
        print(f"[ERROR] TTS model failed to load. Voice features will be unavailable. Details: {e}")
        tts_model = None
    
    await task_queue.load_tasks()
    await memory_backend.memory_queue.load_operations()
    
    asyncio.create_task(process_queue())
    asyncio.create_task(process_memory_operations())
    asyncio.create_task(cleanup_tasks_periodically())
    
    print(f"[FASTAPI_LIFECYCLE] App startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    print(f"[FASTAPI_LIFECYCLE] App shutdown.")
    await task_queue.save_tasks()
    await memory_backend.memory_queue.save_operations()
    print(f"[FASTAPI_LIFECYCLE] Tasks/Ops saved.")

# --- Pydantic Models ---
class Message(BaseModel):
    input: str
    pricing: str
    credits: int
class ElaboratorMessage(BaseModel):
    input: str
    purpose: str
class EncryptionRequest(BaseModel):
    data: str
class DecryptionRequest(BaseModel):
    encrypted_data: str
class SetReferrerRequest(BaseModel):
    referral_code: str
class DeleteSubgraphRequest(BaseModel):
    source: str
class GraphRequest(BaseModel):
    information: str
class GraphRAGRequest(BaseModel):
    query: str
class SetDataSourceEnabledRequest(BaseModel):
    source: str
    enabled: bool
class CreateTaskRequest(BaseModel):
    description: str
class UpdateTaskRequest(BaseModel):
    task_id: str
    description: str
    priority: int = Field(..., ge=1, le=5)
class DeleteTaskRequest(BaseModel):
    task_id: str
class GetShortTermMemoriesRequest(BaseModel):
    category: str
    limit: int = Field(10, ge=1)
class UpdateUserDataRequest(BaseModel):
    data: Dict[str, Any]
class AddUserDataRequest(BaseModel):
    data: Dict[str, Any]
class AddMemoryRequest(BaseModel):
    text: str
    category: str
    retention_days: int = Field(..., ge=1)
class UpdateMemoryRequest(BaseModel):
    id: int
    text: str
    category: str
    retention_days: int = Field(..., ge=1)
class DeleteMemoryRequest(BaseModel):
    id: int
    category: str
class TaskIdRequest(BaseModel):
    task_id: str
class TaskApprovalDataResponse(BaseModel):
    approval_data: Optional[Dict[str, Any]] = None
class ApproveTaskResponse(BaseModel):
    message: str
    result: Any

# --- API Endpoints ---

@app.get("/", status_code=status.HTTP_200_OK, summary="API Root", tags=["General"])
async def main_root():
    return {"message": "Sentient API is running."}

@app.post("/get-history", status_code=status.HTTP_200_OK, summary="Get Chat History", tags=["Chat"])
async def get_history(user_id: str = Depends(PermissionChecker(required_permissions=["read:chat"]))):
    print(f"[ENDPOINT /get-history] Called by user {user_id}.")
    try:
        messages = await get_chat_history_messages(user_id)
        active_chat_id = 0 # Default
        async with db_lock:
            chatsDb = await load_db(user_id)
            active_chat_id = chatsDb.get("active_chat_id", 0)
        return JSONResponse(content={"messages": messages, "activeChatId": active_chat_id})
    except Exception as e:
        print(f"[ERROR] /get-history {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get chat history.")

@app.post("/clear-chat-history", status_code=status.HTTP_200_OK, summary="Clear Chat History", tags=["Chat"])
async def clear_chat_history_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["write:chat"]))):
    print(f"[ENDPOINT /clear-chat-history] Called by user {user_id}.")
    async with db_lock:
        try:
            await save_db(user_id, initial_db.copy())
            return JSONResponse(content={"message": "Chat history cleared.", "activeChatId": 0})
        except Exception as e:
            print(f"[ERROR] /clear-chat-history {user_id}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear chat history.")

@app.post("/chat", status_code=status.HTTP_200_OK, summary="Process Chat Message", tags=["Chat"])
async def chat(message: Message, user_id: str = Depends(PermissionChecker(required_permissions=["write:chat"]))):
    print(f"[ENDPOINT /chat] User {user_id}. Input: '{message.input[:50]}...'")
    try:
        user_profile_data = await load_user_profile(user_id)
        username = user_profile_data.get("userData", {}).get("personalInfo", {}).get("name", "User")
        personality_setting = user_profile_data.get("userData", {}).get("personality", "Default")
        await get_chat_history_messages(user_id) # Ensure active chat
        active_chat_id = 0 # Default
        async with db_lock:
            chatsDb = await load_db(user_id)
            active_chat_id = chatsDb.get("active_chat_id", 0)
        if active_chat_id == 0:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No active chat.")

        unified_output = unified_classification_runnable.invoke({"query": message.input})
        category = unified_output.get("category", "chat")
        use_personal_context = unified_output.get("use_personal_context", False)
        internet_search_type = unified_output.get("internet", "None")
        transformed_input = unified_output.get("transformed_input", message.input)

        async def response_generator():
            stream_start_time = time.time()
            memory_used, agents_used, internet_used, pro_features_used = False, False, False, False
            user_context_str, internet_context_str, additional_notes = None, None, ""

            user_msg_id = await add_message_to_db(user_id, active_chat_id, message.input, is_user=True, is_visible=True)
            if not user_msg_id:
                yield json.dumps({"type":"error", "message":"Failed to save message."})+"\n"
                return
            yield json.dumps({"type": "userMessage", "id": user_msg_id, "message": message.input, "timestamp": datetime.datetime.now(timezone.utc).isoformat()}) + "\n"
            await asyncio.sleep(0.01) # Allow UI to render user message
            assistant_msg_id_ts = str(int(time.time() * 1000))
            assistant_msg_base = {"id": assistant_msg_id_ts, "message": "", "isUser": False, "memoryUsed": False, "agentsUsed": False, "internetUsed": False, "timestamp": datetime.datetime.now(timezone.utc).isoformat(), "isVisible": True}

            if category == "agent":
                agents_used = True
                assistant_msg_base["agentsUsed"] = True
                priority_response = priority_runnable.invoke({"task_description": transformed_input})
                priority = priority_response.get("priority", 3)
                await task_queue.add_task(user_id=user_id, chat_id=active_chat_id, description=transformed_input, priority=priority, username=username, personality=personality_setting, use_personal_context=use_personal_context, internet=internet_search_type)
                assistant_msg_base["message"] = "Okay, I'll get right on that."
                await add_message_to_db(user_id, active_chat_id, transformed_input, is_user=True, is_visible=False) # Save agent task "input"
                await add_message_to_db(user_id, active_chat_id, assistant_msg_base["message"], is_user=False, is_visible=True, agentsUsed=True, task=transformed_input)
                yield json.dumps({"type": "assistantMessage", "messageId": assistant_msg_base["id"], "message": assistant_msg_base["message"], "done": True, "proUsed": False}) + "\n"
                return

            if category == "memory" or use_personal_context:
                memory_used = True
                assistant_msg_base["memoryUsed"] = True
                yield json.dumps({"type": "intermediary", "message": "Checking context...", "id": assistant_msg_id_ts}) + "\n"
                try:
                    user_context_str = await memory_backend.retrieve_memory(user_id, transformed_input)
                except Exception as e:
                    user_context_str = f"Error retrieving context: {e}"
                if category == "memory": # Only add to memory if explicitly a memory operation
                    if message.pricing == "free" and message.credits <= 0:
                        additional_notes += " (Memory update skipped: Pro)"
                    else:
                        pro_features_used = True
                        asyncio.create_task(memory_backend.add_operation(user_id, transformed_input)) # Add original input to memory

            if internet_search_type and internet_search_type != "None":
                internet_used = True
                assistant_msg_base["internetUsed"] = True
                if message.pricing == "free" and message.credits <= 0:
                    additional_notes += " (Search skipped: Pro)"
                else:
                    pro_features_used = True
                    yield json.dumps({"type": "intermediary", "message": "Searching internet...", "id": assistant_msg_id_ts}) + "\n"
                    try:
                        reframed = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                        results = get_search_results(reframed)
                        internet_context_str = get_search_summary(internet_summary_runnable, results)
                    except Exception as e:
                        internet_context_str = f"Error searching: {e}"

            if category in ["chat", "memory"]: # If it was memory, context is now loaded, proceed to chat
                yield json.dumps({"type": "intermediary", "message": "Thinking...", "id": assistant_msg_id_ts}) + "\n"
                full_response_text = ""
                try:
                    async for token_chunk in generate_streaming_response(chat_runnable, inputs={"query": transformed_input, "user_context": user_context_str, "internet_context": internet_context_str, "name": username, "personality": personality_setting}, stream=True):
                        if isinstance(token_chunk, str):
                            full_response_text += token_chunk
                            yield json.dumps({"type": "assistantStream", "token": token_chunk, "done": False, "messageId": assistant_msg_base["id"]}) + "\n"
                        await asyncio.sleep(0.01) # Small delay for stream effect
                    
                    final_message_content = full_response_text + (("\n\n" + additional_notes) if additional_notes else "")
                    assistant_msg_base["message"] = final_message_content
                    yield json.dumps({"type": "assistantStream", "token": ("\n\n" + additional_notes) if additional_notes else "", "done": True, "memoryUsed": memory_used, "agentsUsed": agents_used, "internetUsed": internet_used, "proUsed": pro_features_used, "messageId": assistant_msg_base["id"]}) + "\n"
                    await add_message_to_db(user_id, active_chat_id, final_message_content, is_user=False, is_visible=True, memoryUsed=memory_used, agentsUsed=agents_used, internetUsed=internet_used)
                except Exception as e:
                    error_msg_user = "Sorry, an error occurred while generating the response."
                    assistant_msg_base["message"] = error_msg_user
                    yield json.dumps({"type": "assistantStream", "token": error_msg_user, "done": True, "error": True, "messageId": assistant_msg_base["id"]}) + "\n"
                    await add_message_to_db(user_id, active_chat_id, error_msg_user, is_user=False, is_visible=True, error=True)
            # print(f"[STREAM /chat] User {user_id} - Stream finished. Duration: {time.time() - stream_start_time:.2f}s")
        
        return StreamingResponse(response_generator(), media_type="application/x-ndjson")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] /chat {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Chat processing error.")

@app.post("/elaborator", status_code=status.HTTP_200_OK, summary="Elaborate Text", tags=["Utilities"])
async def elaborate(message: ElaboratorMessage, user_id: str = Depends(PermissionChecker(required_permissions=["use:elaborator"]))):
    print(f"[ENDPOINT /elaborator] User {user_id}, input: '{message.input[:50]}...'")
    try:
        # Assuming elaborator_system_prompt_template and elaborator_user_prompt_template are defined
        elaborator_runnable = get_tool_runnable(elaborator_system_prompt_template, elaborator_user_prompt_template, None, ["query", "purpose"])
        output = elaborator_runnable.invoke({"query": message.input, "purpose": message.purpose})
        return JSONResponse(content={"message": output})
    except Exception as e:
        print(f"[ERROR] /elaborator: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Elaboration failed.")

@app.post("/encrypt", status_code=status.HTTP_200_OK, summary="Encrypt Data", tags=["Utilities"])
async def encrypt_data(request: EncryptionRequest):
    try:
        return JSONResponse(content={"encrypted_data": aes_encrypt(request.data)})
    except Exception as e:
        print(f"[ERROR] /encrypt: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Encryption failed.")

@app.post("/decrypt", status_code=status.HTTP_200_OK, summary="Decrypt Data", tags=["Utilities"])
async def decrypt_data(request: DecryptionRequest):
    try:
        return JSONResponse(content={"decrypted_data": aes_decrypt(request.encrypted_data)})
    except ValueError: # More specific error for bad data/key
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Decryption failed: Invalid data or key.")
    except Exception as e:
        print(f"[ERROR] /decrypt: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Decryption failed.")


# Tool Handlers
@register_tool("gmail")
async def gmail_tool(tool_call_input: dict) -> Dict[str, Any]:
    user_id = tool_call_input.get("user_id")
    input_val = tool_call_input.get("input")
    if not user_id:
        return {"status": "failure", "error": "User_id missing for Gmail."}
    # print(f"[TOOL gmail] User {user_id}, input: '{str(input_val)[:50]}...'")
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        # Assuming gmail_agent_system_prompt_template, etc. are defined
        tool_runnable = get_tool_runnable(gmail_agent_system_prompt_template, gmail_agent_user_prompt_template, gmail_agent_required_format, ["query", "username", "previous_tool_response"])
        tool_call_str = tool_runnable.invoke({"query": str(input_val), "username": username, "previous_tool_response": tool_call_input.get("previous_tool_response")})
        try:
            tool_call_dict = json.loads(tool_call_str)
        except json.JSONDecodeError:
            tool_call_dict = {"tool_name": "error_parsing_llm", "task_instruction": f"LLM output not JSON: {tool_call_str}"}
        
        actual_tool_name = tool_call_dict.get("tool_name")
        if actual_tool_name in ["send_email", "reply_email"]:
            return {"action": "approve", "tool_call": tool_call_dict}
        else:
            # Assuming parse_and_execute_tool_calls is defined and handles user_id
            tool_result = await parse_and_execute_tool_calls(user_id, json.dumps(tool_call_dict))
            return {"tool_result": tool_result}
    except Exception as e:
        print(f"[ERROR] Gmail tool {user_id}: {e}")
        traceback.print_exc()
        return {"status": "failure", "error": str(e)}

# --- Utility Endpoints (Now use Custom Claims or M2M for Management API) ---

@app.post("/get-role", status_code=status.HTTP_200_OK, summary="Get User Role from Token", tags=["User Management"])
async def get_role(payload: dict = Depends(auth.get_decoded_payload_with_claims)):
    user_id = payload["user_id"]
    print(f"[ENDPOINT /get-role] Called by user {user_id} (from token claims).")
    user_role = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}role", "free") # Default to "free"
    print(f"User {user_id} role from token claim: {user_role}")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"role": user_role})

@app.post("/get-beta-user-status", status_code=status.HTTP_200_OK, summary="Get Beta Status from Token", tags=["User Management"])
async def get_beta_user_status(payload: dict = Depends(auth.get_decoded_payload_with_claims)):
    user_id = payload["user_id"]
    print(f"[ENDPOINT /get-beta-user-status] Called by user {user_id} (from token claims).")
    beta_status = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}betaUserStatus", False) # Default to False
    print(f"User {user_id} beta status from token claim: {beta_status}")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"betaUserStatus": beta_status})

@app.post("/get-referral-code", status_code=status.HTTP_200_OK, summary="Get Referral Code from Token", tags=["User Management"])
async def get_referral_code(payload: dict = Depends(auth.get_decoded_payload_with_claims)):
    user_id = payload["user_id"]
    print(f"[ENDPOINT /get-referral-code] Called by user {user_id} (from token claims).")
    referral_code = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}referralCode", None)
    if not referral_code:
        print(f"User {user_id} referral code not found in token claims.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"referralCode": None}) # Or 404
    print(f"User {user_id} referral code from token claim found.")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"referralCode": referral_code})

@app.post("/get-referrer-status", status_code=status.HTTP_200_OK, summary="Get Referrer Status from Token", tags=["User Management"])
async def get_referrer_status(payload: dict = Depends(auth.get_decoded_payload_with_claims)):
    user_id = payload["user_id"]
    print(f"[ENDPOINT /get-referrer-status] Called by user {user_id} (from token claims).")
    referrer_status = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}referrerStatus", False) # Default to False
    print(f"User {user_id} referrer status from token claim: {referrer_status}")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"referrerStatus": referrer_status})

@app.post("/get-user-and-set-referrer-status", status_code=status.HTTP_200_OK, summary="Set Referrer Status by Code (Admin Action)", tags=["User Management"])
async def get_user_and_set_referrer_status(
    request: SetReferrerRequest,
    user_id_making_request: str = Depends(PermissionChecker(required_permissions=["admin:user_metadata"]))
):
    referral_code_to_find = request.referral_code
    print(f"[ENDPOINT /set-referrer-status] User {user_id_making_request} trying to set referrer for code {referral_code_to_find}.")
    try:
        mgmt_token = get_management_token() # Assumes get_management_token is defined
        if not mgmt_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="M2M token unavailable.")

        headers_search = {"Authorization": f"Bearer {mgmt_token}", "Accept": "application/json"}
        search_url = f"https://{AUTH0_DOMAIN}/api/v2/users"
        params = {'q': f'app_metadata.referralCode:"{referral_code_to_find}"', 'search_engine': 'v3'}
        async with httpx.AsyncClient() as client:
            search_response = await client.get(search_url, headers=headers_search, params=params)
        if search_response.status_code != 200:
            raise HTTPException(status_code=search_response.status_code, detail=f"Auth0 User Search Error: {search_response.text}")
        
        users_found = search_response.json()
        if not users_found:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No user with referral code: {referral_code_to_find}")

        user_to_update_id = users_found[0].get("user_id")
        if not user_to_update_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Found user, but ID missing.")

        update_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{user_to_update_id}"
        update_payload = {"app_metadata": {"referrer": True}}
        headers_update = {"Authorization": f"Bearer {mgmt_token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            update_response = await client.patch(update_url, headers=headers_update, json=update_payload)
        if update_response.status_code != 200:
            raise HTTPException(status_code=update_response.status_code, detail=f"Auth0 User Update Error: {update_response.text}")

        print(f"Referrer status set to True for user {user_to_update_id} (referred by code {referral_code_to_find}).")
        return JSONResponse(content={"message": "Referrer status updated successfully for the user who was referred."})
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] /set-referrer-status: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set referrer status.")

@app.post("/get-user-and-invert-beta-user-status", status_code=status.HTTP_200_OK, summary="Invert Beta User Status (Admin Action)", tags=["User Management"])
async def get_user_and_invert_beta_user_status(
    user_id_to_modify: str = Depends(PermissionChecker(required_permissions=["admin:user_metadata"]))
):
    print(f"[ENDPOINT /invert-beta-status] Admin action for user {user_id_to_modify}.")
    try:
        mgmt_token = get_management_token()
        if not mgmt_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="M2M token unavailable.")

        get_url = f"https://{AUTH0_DOMAIN}/api/v2/users/{user_id_to_modify}"
        headers_get = {"Authorization": f"Bearer {mgmt_token}", "Accept": "application/json"}
        async with httpx.AsyncClient() as client:
            get_response = await client.get(get_url, headers=headers_get)
        if get_response.status_code != 200:
            raise HTTPException(status_code=get_response.status_code, detail=f"Auth0 Get User Error: {get_response.text}")

        user_data = get_response.json()
        current_status = user_data.get("app_metadata", {}).get("betaUser", False)
        inverted_status = not current_status

        update_payload = {"app_metadata": {"betaUser": inverted_status}}
        headers_update = {"Authorization": f"Bearer {mgmt_token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            update_response = await client.patch(get_url, headers=headers_update, json=update_payload)
        if update_response.status_code != 200:
            raise HTTPException(status_code=update_response.status_code, detail=f"Auth0 Update User Error: {update_response.text}")

        print(f"Beta status for user {user_id_to_modify} inverted to {inverted_status}.")
        return JSONResponse(content={"message": "Beta user status inverted successfully.", "newStatus": inverted_status})
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] /invert-beta-status for {user_id_to_modify}: {e}")
        traceback.print_exc()
# --- Google Authentication Endpoint ---
@app.post("/authenticate-google", status_code=status.HTTP_200_OK, summary="Authenticate Google Services", tags=["Integrations"])
async def authenticate_google(user_id: str = Depends(PermissionChecker(required_permissions=["manage:google_auth"]))):
    print(f"[ENDPOINT /authenticate-google] User {user_id}.")
    token_path = os.path.join(BASE_DIR, f"token_{user_id}.pickle")
    creds = None
    try:
        if os.path.exists(token_path):
            with open(token_path, "rb") as token_file:
                creds = pickle.load(token_file)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request()) # google.auth.transport.requests.Request
            else:
                 raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google auth required or token expired. Setup flow needed.")
            with open(token_path, "wb") as token_file:
                pickle.dump(creds, token_file)
        return JSONResponse(content={"success": True, "message": "Google auth successful."})
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] Google auth for {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Google auth failed: {str(e)}")


# --- Memory and Knowledge Graph Endpoints ---
@app.post("/graphrag", status_code=status.HTTP_200_OK, summary="Query Knowledge Graph (RAG)", tags=["Knowledge Graph"])
async def graphrag(request: GraphRAGRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /graphrag] User {user_id}, Query: '{request.query[:50]}...'")
    try:
        if not all([graph_driver, embed_model, text_conversion_runnable, query_classification_runnable]):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="GraphRAG dependencies unavailable.")
        # Assuming query_user_profile is defined
        context = await asyncio.to_thread(query_user_profile, user_id, request.query, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
        return JSONResponse(content={"context": context or "No relevant context found."})
    except Exception as e:
        print(f"[ERROR] /graphrag {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GraphRAG query failed.")

@app.post("/initiate-long-term-memories", status_code=status.HTTP_200_OK, summary="Initialize/Reset Knowledge Graph", tags=["Knowledge Graph"])
async def create_graph(request_data: Optional[Dict[str, bool]] = None, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    clear_graph_flag = request_data.get("clear_graph", False) if request_data else False
    action = "Resetting/rebuilding" if clear_graph_flag else "Initiating/Updating"
    print(f"[ENDPOINT /initiate-LTM] {action} KG for user {user_id}.")
    loop = asyncio.get_event_loop()
    input_dir = os.path.join(BASE_DIR, "input") # TODO: user-specific input dir
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", user_id)
        if not all([graph_driver, embed_model, text_dissection_runnable, information_extraction_runnable]):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph building dependencies unavailable.")

        def read_files_sync(udir, uname):
            fpath = os.path.join(udir, f"{uname}_sample.txt") # This needs to be user-specific if multi-user
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    return [{"text": f.read(), "source": os.path.basename(fpath)}]
            return [{"text": f"Sample data for {uname}", "source": "default_sample.txt"}] # Generic sample
        
        extracted_texts = await loop.run_in_executor(None, read_files_sync, input_dir, username)

        if clear_graph_flag:
            print(f"Clearing KG for user scope: {username} (user_id: {user_id})...")
            def clear_neo4j_user_graph_sync(driver, uid_scope: str): # uid_scope is username here
                with driver.session(database="neo4j") as session:
                    query = "MATCH (n {username: $username_scope}) DETACH DELETE n" # Scoped by username
                    session.execute_write(lambda tx: tx.run(query, username_scope=uid_scope))
            await loop.run_in_executor(None, clear_neo4j_user_graph_sync, graph_driver, username)
            print(f"Graph cleared for user scope: {username}.")

        if extracted_texts:
            # Assuming build_initial_knowledge_graph is defined
            await loop.run_in_executor(None, build_initial_knowledge_graph, username, extracted_texts, graph_driver, embed_model, text_dissection_runnable, information_extraction_runnable)
            return JSONResponse(content={"message": f"Knowledge Graph {action.lower()} for user {username} completed."})
        else:
            return JSONResponse(content={"message": "No input documents found. Graph not modified."})
    except Exception as e:
        print(f"[ERROR] /initiate-LTM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Knowledge Graph operation failed.")

@app.post("/delete-subgraph", status_code=status.HTTP_200_OK, summary="Delete Subgraph by Source", tags=["Knowledge Graph"])
async def delete_subgraph(request: DeleteSubgraphRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    source_key = request.source.lower()
    print(f"[ENDPOINT /delete-subgraph] User {user_id}, source: {source_key}")
    loop = asyncio.get_event_loop()
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", user_id).lower()
        
        # This mapping and fallback needs to be robust and match how sources are named in the graph
        SOURCE_FILENAME_MAPPINGS = { key: f"{username}_{key.replace(' ', '_')}.txt" for key in ["extroversion", "introversion", "sensing", "intuition", "thinking", "feeling", "judging", "perceiving"] }
        source_identifier_in_graph = SOURCE_FILENAME_MAPPINGS.get(source_key, f"{username}_{source_key.replace(' ', '_')}.txt")

        if not graph_driver:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neo4j driver unavailable.")
        
        # Assuming delete_source_subgraph is defined and correctly scopes by username and source_identifier
        await loop.run_in_executor(None, delete_source_subgraph, graph_driver, source_identifier_in_graph, username)

        input_dir_path = os.path.join(BASE_DIR, "input") # TODO: user-specific input directory
        file_to_delete_on_disk = os.path.join(input_dir_path, source_identifier_in_graph)
        
        def remove_file_sync(path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    return True, None
                except OSError as e_os:
                    return False, str(e_os)
            return False, "File not found on disk."
        
        deleted, ferr = await loop.run_in_executor(None, remove_file_sync, file_to_delete_on_disk)
        if not deleted:
            print(f"[WARN] Could not delete source file {file_to_delete_on_disk}: {ferr}")
        
        return JSONResponse(content={"message": f"Subgraph for source '{source_key}' (graph id: {source_identifier_in_graph}) processed for deletion."})
    except Exception as e:
        print(f"[ERROR] /delete-subgraph {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Subgraph deletion failed.")

@app.post("/create-document", status_code=status.HTTP_200_OK, summary="Create Input Documents from Profile", tags=["Knowledge Graph"])
async def create_document(user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /create-document] User {user_id}.")
    input_dir = os.path.join(BASE_DIR, "input") # TODO: User-specific input directory
    loop = asyncio.get_event_loop()
    try:
        user_profile = await load_user_profile(user_id)
        db_user_data = user_profile.get("userData", {})
        username = db_user_data.get("personalInfo", {}).get("name", user_id).lower()
        await loop.run_in_executor(None, os.makedirs, input_dir, True) # Use exist_ok=True

        # Assuming PERSONALITY_DESCRIPTIONS and summarize_and_write_sync are defined
        personality_type_list = db_user_data.get("personalityType", [])
        bg_tasks = []
        trait_descs_for_summary = []

        if isinstance(personality_type_list, list):
            for trait in personality_type_list:
                trait_lower = trait.lower()
                if trait_lower in PERSONALITY_DESCRIPTIONS:
                    desc_content = f"{trait.capitalize()}: {PERSONALITY_DESCRIPTIONS[trait_lower]}"
                    trait_descs_for_summary.append(desc_content)
                    filename = f"{username}_{trait_lower}.txt"
                    bg_tasks.append(loop.run_in_executor(None, summarize_and_write_sync, username, desc_content, filename, input_dir))
        
        results = await asyncio.gather(*bg_tasks, return_exceptions=True)
        created = [fname for res_tuple in results if isinstance(res_tuple, tuple) and res_tuple[0] and isinstance(res_tuple[1], str) and not isinstance(res_tuple[1], Exception) for fname in [res_tuple[1]]] # Adjusted based on typical return of (success, filename_or_error)
        failed = [str(item) for item in results if isinstance(item, Exception) or (isinstance(item, tuple) and not item[0])]


        summary_text = "\n".join(trait_descs_for_summary)

        return JSONResponse(content={"message": "Input documents processed.", "created_files": created, "failed_files": failed, "personality_summary_generated": summary_text})
    except Exception as e:
        print(f"[ERROR] /create-document {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document creation failed.")

@app.post("/customize-long-term-memories", status_code=status.HTTP_200_OK, summary="Customize KG with Text", tags=["Knowledge Graph"])
async def customize_graph(request: GraphRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /customize-LTM] User {user_id}, Info: '{request.information[:50]}...'")
    loop = asyncio.get_event_loop()
    try:
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", user_id)
        required_deps = [graph_driver, embed_model, fact_extraction_runnable, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable]
        if not all(required_deps):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph customization dependencies unavailable.")

        # Assuming fact_extraction_runnable and crud_graph_operations are defined
        extracted_points = await loop.run_in_executor(None, fact_extraction_runnable.invoke, {"paragraph": request.information, "username": username})
        if not isinstance(extracted_points, list):
            extracted_points = []
        if not extracted_points:
            return JSONResponse(content={"message": "No facts extracted. Graph not modified."})

        crud_tasks = [loop.run_in_executor(None, crud_graph_operations, user_id, point, graph_driver, embed_model, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable) for point in extracted_points]
        crud_results = await asyncio.gather(*crud_tasks, return_exceptions=True)
        
        processed = sum(1 for r in crud_results if not isinstance(r, Exception))
        errors = [str(r) for r in crud_results if isinstance(r, Exception)]
        msg = f"Knowledge Graph customization processed {processed}/{len(extracted_points)} facts." + (f" Errors: {len(errors)}" if errors else "")
        
        return JSONResponse(status_code=status.HTTP_200_OK if not errors else status.HTTP_207_MULTI_STATUS, content={"message": msg, "errors": errors})
    except Exception as e:
        print(f"[ERROR] /customize-LTM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Knowledge Graph customization failed.")


# --- Task Queue Endpoints ---
@app.post("/fetch-tasks", status_code=status.HTTP_200_OK, summary="Fetch User Tasks", tags=["Tasks"])
async def get_tasks(user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))):
    print(f"[ENDPOINT /fetch-tasks] User {user_id}.")
    try:
        user_tasks = await task_queue.get_tasks_for_user(user_id)
        s_tasks = []
        for t in user_tasks:
            if isinstance(t.get('created_at'), datetime):
                t['created_at'] = t['created_at'].isoformat()
            if isinstance(t.get('completed_at'), datetime):
                t['completed_at'] = t['completed_at'].isoformat()
            s_tasks.append(t)
        return JSONResponse(content={"tasks": s_tasks})
    except Exception as e:
        print(f"[ERROR] /fetch-tasks {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch tasks.")

@app.post("/add-task", status_code=status.HTTP_201_CREATED, summary="Add New Task", tags=["Tasks"])
async def add_task_endpoint(task_request: CreateTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /add-task] User {user_id}, Desc: '{task_request.description[:50]}...'")
    loop = asyncio.get_event_loop()
    try:
        await get_chat_history_messages(user_id) # Ensure active chat context
        active_chat_id = 0
        async with db_lock:
            chatsDb = await load_db(user_id)
            active_chat_id = chatsDb.get("active_chat_id", 0)
        
        if active_chat_id == 0:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No active chat found for task creation.")
        
        user_profile = await load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", user_id)
        personality = user_profile.get("userData", {}).get("personality", "Default")
        
        unified_output = await loop.run_in_executor(None, unified_classification_runnable.invoke, {"query": task_request.description})
        use_personal_context = unified_output.get("use_personal_context", False)
        internet_needed = unified_output.get("internet", "None")
        
        task_priority = 3 # Default priority
        try:
            priority_response = await loop.run_in_executor(None, priority_runnable.invoke, {"task_description": task_request.description})
            task_priority = priority_response.get("priority", 3)
        except Exception: # Default to 3 if priority runnable fails
            pass 
            
        new_task_id = await task_queue.add_task(user_id=user_id, chat_id=active_chat_id, description=task_request.description, priority=task_priority, username=username, personality=personality, use_personal_context=use_personal_context, internet=internet_needed)
        
        await add_message_to_db(user_id, active_chat_id, task_request.description, is_user=True, is_visible=False) # Log original "task" as hidden user message
        confirm_msg = f"Task added: '{task_request.description[:40]}...'"
        await add_message_to_db(user_id, active_chat_id, confirm_msg, is_user=False, is_visible=True, agentsUsed=True, task=task_request.description)
        
        return JSONResponse(content={"task_id": new_task_id, "message": "Task added successfully."}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        print(f"[ERROR] /add-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add task.")

@app.post("/update-task", status_code=status.HTTP_200_OK, summary="Update Task", tags=["Tasks"])
async def update_task_endpoint(request: UpdateTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /update-task] User {user_id}, Task: {request.task_id}, Prio: {request.priority}")
    try:
        updated = await task_queue.update_task(user_id, request.task_id, request.description, request.priority)
        if not updated:
            raise ValueError("Task not found or user unauthorized to update this task.")
        return JSONResponse(content={"message": "Task updated successfully."})
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /update-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update task.")

@app.post("/delete-task", status_code=status.HTTP_200_OK, summary="Delete Task", tags=["Tasks"])
async def delete_task_endpoint(request: DeleteTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /delete-task] User {user_id}, Task: {request.task_id}")
    try:
        deleted = await task_queue.delete_task(user_id, request.task_id)
        if not deleted:
            raise ValueError("Task not found or user unauthorized to delete this task.")
        return JSONResponse(content={"message": "Task deleted successfully."})
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /delete-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete task.")


# --- Short-Term Memory Endpoints ---
@app.post("/get-short-term-memories", status_code=status.HTTP_200_OK, summary="Get Short-Term Memories", tags=["Short-Term Memory"])
async def get_short_term_memories(request: GetShortTermMemoriesRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /get-STM] User {user_id}, Cat: {request.category}, Lim: {request.limit}")
    loop = asyncio.get_event_loop()
    try:
        memories = await loop.run_in_executor(None, memory_backend.memory_manager.fetch_memories_by_category, user_id, request.category, request.limit)
        s_mem = []
        for m in memories:
            if isinstance(m.get('created_at'), datetime):
                m['created_at'] = m['created_at'].isoformat()
            if isinstance(m.get('expires_at'), datetime):
                m['expires_at'] = m['expires_at'].isoformat()
            s_mem.append(m)
        return JSONResponse(content=s_mem)
    except Exception as e:
        print(f"[ERROR] /get-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch short-term memories.")

@app.post("/add-short-term-memory", status_code=status.HTTP_201_CREATED, summary="Add STM", tags=["Short-Term Memory"])
async def add_memory(request: AddMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /add-STM] User {user_id}, Cat: {request.category}")
    loop = asyncio.get_event_loop()
    try:
        mem_id = await loop.run_in_executor(None, memory_backend.memory_manager.store_memory, user_id, request.text, request.retention_days, request.category)
        return JSONResponse(content={"memory_id": mem_id, "message": "Short-term memory added."}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        print(f"[ERROR] /add-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add short-term memory.")

@app.post("/update-short-term-memory", status_code=status.HTTP_200_OK, summary="Update STM", tags=["Short-Term Memory"])
async def update_memory(request: UpdateMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /update-STM] User {user_id}, ID: {request.id}")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, memory_backend.memory_manager.update_memory_crud, user_id, request.category, request.id, request.text, request.retention_days)
        return JSONResponse(content={"message": "Short-term memory updated."})
    except ValueError as ve: # E.g., memory not found
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /update-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update short-term memory.")

@app.post("/delete-short-term-memory", status_code=status.HTTP_200_OK, summary="Delete STM", tags=["Short-Term Memory"])
async def delete_memory(request: DeleteMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /delete-STM] User {user_id}, ID: {request.id}, Cat: {request.category}")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, memory_backend.memory_manager.delete_memory, user_id, request.category, request.id)
        return JSONResponse(content={"message": "Short-term memory deleted."})
    except ValueError as ve: # E.g., memory not found
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /delete-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete short-term memory.")

@app.post("/clear-all-short-term-memories", status_code=status.HTTP_200_OK, summary="Clear All STM", tags=["Short-Term Memory"])
async def clear_all_memories(user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /clear-all-STM] User {user_id}.")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, memory_backend.memory_manager.clear_all_memories, user_id)
        return JSONResponse(content={"message": "All short-term memories cleared."})
    except Exception as e:
        print(f"[ERROR] /clear-all-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear all short-term memories.")


# --- User Profile Database Endpoints ---
@app.post("/set-user-data", status_code=status.HTTP_200_OK, summary="Set User Profile Data", tags=["User Profile"])
async def set_db_data(request: UpdateUserDataRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /set-user-data] User {user_id}, Data: {str(request.data)[:100]}...")
    try:
        profile = await load_user_profile(user_id) # AWAIT
        if "userData" not in profile:
            profile["userData"] = {}
        profile["userData"].update(request.data)
        success = await write_user_profile(user_id, profile)
        
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to write user profile to JSON.")

        # After successful JSON write, update Neo4j with personalInfo if present
        if "personalInfo" in profile["userData"] and graph_driver:
            personal_info_data = profile["userData"]["personalInfo"]
            if isinstance(personal_info_data, dict): # Ensure it's a dictionary
                print(f"[NEO4J_INITIATE_PERSONAL_INFO_UPDATE] Initiating Neo4j update for personal info for user {user_id}.")
                await update_neo4j_with_personal_info(user_id, personal_info_data, graph_driver)
            else:
                print(f"[WARN] personalInfo for user {user_id} is not a dictionary. Skipping Neo4j update.")
        
        return JSONResponse(content={"message": "User data stored successfully.", "status": 200})
    except Exception as e:
        print(f"[ERROR] /set-user-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set user data.")

@app.post("/add-db-data", status_code=status.HTTP_200_OK, summary="Add/Merge User Profile Data", tags=["User Profile"])
async def add_db_data(request: AddUserDataRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /add-db-data] User {user_id}, Data: {str(request.data)[:100]}...")
    try:
        profile = await load_user_profile(user_id) # AWAIT
        existing_udata = profile.get("userData", {})
        
        # Deep merge logic (simple version from original)
        for key, new_val in request.data.items():
            if key in existing_udata and isinstance(existing_udata[key], list) and isinstance(new_val, list):
                existing_udata[key].extend(item for item in new_val if item not in existing_udata[key])
            elif key in existing_udata and isinstance(existing_udata[key], dict) and isinstance(new_val, dict):
                existing_udata[key].update(new_val) # Note: this is a shallow update for nested dicts
            else:
                existing_udata[key] = new_val
        
        profile["userData"] = existing_udata
        success = await write_user_profile(user_id, profile) # AWAIT
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to write merged user profile.")
        return JSONResponse(content={"message": "User data merged successfully.", "status": 200})
    except Exception as e:
        print(f"[ERROR] /add-db-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to merge user data.")

@app.post("/get-user-data", status_code=status.HTTP_200_OK, summary="Get User Profile Data", tags=["User Profile"])
async def get_db_data(user_id: str = Depends(PermissionChecker(required_permissions=["read:profile"]))):
    print(f"[ENDPOINT /get-user-data] User {user_id}.")
    # loop = asyncio.get_event_loop() # Not needed if you directly await
    try:
        # CORRECT:
        profile = await load_user_profile(user_id) # <--- Add await here

        return JSONResponse(content={"data": profile.get("userData", {}), "status": 200})
    except Exception as e:
        print(f"[ERROR] /get-user-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user data.")

# --- Graph Data Endpoint ---
@app.post("/get-graph-data", status_code=status.HTTP_200_OK, summary="Get KG Data (Visualization)", tags=["Knowledge Graph"])
async def get_graph_data_apoc(user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /get-graph-data] User {user_id}.")
    loop = asyncio.get_event_loop()
    if not graph_driver:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neo4j driver unavailable.")
    
    # Query needs to be user-scoped. This example assumes nodes have a 'userId' property. Adjust to your model.
    graph_visualization_query = """
    MATCH (n) WHERE n.userId = $userId // Crucial: Scope data to the requesting user
    WITH collect(DISTINCT n) as nodes
    OPTIONAL MATCH (s)-[r]->(t) WHERE s IN nodes AND t IN nodes // Get relationships only between user's nodes
    WITH nodes, collect(DISTINCT r) as rels
    RETURN [node IN nodes | { id: elementId(node), label: coalesce(labels(node)[0], 'Unknown'), properties: properties(node) }] AS nodes_list,
           [rel IN rels | { id: elementId(rel), from: elementId(startNode(rel)), to: elementId(endNode(rel)), label: type(rel), properties: properties(rel) }] AS edges_list
    """
    # If nodes might not have userId, but are connected to a user-identifiable node (e.g., UserProfile node):
    # MATCH (u:UserProfile {id: $userId})-[:HAS_DATA*0..]->(n)
    # This is just an example, actual scoping depends on your graph schema.

    def run_q(driver, query, params):
        with driver.session(database="neo4j") as session: # Ensure correct database if not default
            res = session.run(query, params).single() # Use single() if one row is expected
            return (res['nodes_list'] if res else [], res['edges_list'] if res else [])
    try:
        nodes, edges = await loop.run_in_executor(None, run_q, graph_driver, graph_visualization_query, {"userId": user_id})
        return JSONResponse(content={"nodes": nodes, "edges": edges})
    except Exception as e:
        print(f"[ERROR] /get-graph-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get graph data.")

# --- Notifications Endpoint ---
@app.post("/get-notifications", status_code=status.HTTP_200_OK, summary="Get User Notifications", tags=["Notifications"])
async def get_notifications_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["read:notifications"]))):
    print(f"[ENDPOINT /get-notifications] User {user_id}.")
    try:
        db_data = await load_notifications_db(user_id)
        notifs = db_data.get("notifications", [])
        for n in notifs:
            if isinstance(n.get('timestamp'), datetime): # Should already be string from DB save
                n['timestamp'] = n['timestamp'].isoformat()
        return JSONResponse(content={"notifications": notifs})
    except Exception as e:
        print(f"[ERROR] /get-notifications {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get notifications.")


# --- Task Approval Endpoints ---
@app.post("/approve-task", response_model=ApproveTaskResponse, summary="Approve Pending Task", tags=["Tasks"])
async def approve_task_endpoint(request: TaskIdRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /approve-task] User {user_id}, Task: {request.task_id}")
    try:
        result_data = await task_queue.approve_task(user_id, request.task_id) # This should verify ownership
        task_details = await task_queue.get_task_by_id(request.task_id) # Fetch task details to get chat_id
        
        # If task was approved and we have details, add result to chat
        if task_details and task_details.get("user_id") == user_id and task_details.get("chat_id"):
            await add_message_to_db(user_id, task_details["chat_id"], str(result_data), is_user=False, is_visible=True, type="tool_result", task=task_details.get("description"), agentsUsed=True)
        
        return ApproveTaskResponse(message="Task approved and completed.", result=result_data)
    except ValueError as ve: # Raised by task_queue.approve_task for invalid state/ownership
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /approve-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task approval failed.")

@app.post("/get-task-approval-data", response_model=TaskApprovalDataResponse, summary="Get Task Approval Data", tags=["Tasks"])
async def get_task_approval_data_endpoint(request: TaskIdRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))):
    print(f"[ENDPOINT /get-task-approval-data] User {user_id}, Task: {request.task_id}")
    try:
        task = await task_queue.get_task_by_id(request.task_id)
        if task and task.get("user_id") == user_id: # Verify ownership
            if task.get("status") == "approval_pending":
                return TaskApprovalDataResponse(approval_data=task.get("approval_data"))
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task is not pending approval. Current status: {task.get('status')}")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or access denied.")
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] /get-task-approval-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get task approval data.")


# --- Data Source Configuration Endpoints ---
@app.post("/get_data_sources", status_code=status.HTTP_200_OK, summary="Get Data Source Statuses", tags=["Configuration"])
async def get_data_sources_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["read:config"]))):
    print(f"[ENDPOINT /get_data_sources] User {user_id}.")
    loop = asyncio.get_event_loop()
    try:
        profile = await load_user_profile(user_id)
        settings = profile.get("userData", {})
        statuses = [{"name": src, "enabled": settings.get(f"{src}Enabled", True)} for src in DATA_SOURCES] # Default to True if not set
        return JSONResponse(content={"data_sources": statuses})
    except Exception as e:
        print(f"[ERROR] /get_data_sources {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get data source statuses.")

@app.post("/set_data_source_enabled", status_code=status.HTTP_200_OK, summary="Enable/Disable Data Source", tags=["Configuration"])
async def set_data_source_enabled_endpoint(request: SetDataSourceEnabledRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))):
    print(f"[ENDPOINT /set_data_source_enabled] User {user_id}, Source: {request.source}, Enabled: {request.enabled}")
    if request.source not in DATA_SOURCES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data source: {request.source}")
    loop = asyncio.get_event_loop()
    try:
        profile = await load_user_profile(user_id)
        if "userData" not in profile:
            profile["userData"] = {}
        profile["userData"][f"{request.source}Enabled"] = request.enabled
        success = await loop.run_in_executor(None, write_user_profile, user_id, profile)
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save data source status.")
        return JSONResponse(content={"status": "success", "message": f"Data source '{request.source}' status set to {request.enabled}."})
    except Exception as e:
        print(f"[ERROR] /set_data_source_enabled {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set data source status.")


# --- WebSocket Endpoint (Authentication updated) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    authenticated_user_id: Optional[str] = None
    try:
        authenticated_user_id = await auth.ws_authenticate(websocket) # Uses updated auth class
        if not authenticated_user_id:
            print("[WS /ws] WebSocket authentication failed or connection closed during auth.")
            return # Exit if authentication fails
        
        await manager.connect(websocket, authenticated_user_id)
        while True:
            data = await websocket.receive_text()
            try:
                message_payload = json.loads(data)
                if message_payload.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                # Add other message type handlers here if needed
            except json.JSONDecodeError:
                print(f"[WS /ws] Received non-JSON message from {authenticated_user_id}. Ignoring.")
                pass # Ignore non-JSON messages gracefully
            except Exception as e_ws_loop:
                print(f"[WS /ws] Error in WebSocket loop for user {authenticated_user_id}: {e_ws_loop}")
                # Optionally send an error to client or break, depending on severity
    except WebSocketDisconnect:
        print(f"[WS /ws] Client disconnected (User: {authenticated_user_id or 'unknown'}).")
    except Exception as e: # Catch other unexpected errors during WS setup or handling
        print(f"[WS /ws] Unexpected WebSocket error (User: {authenticated_user_id or 'unknown'}): {e}")
        traceback.print_exc()
        # Try to close gracefully if not already closed
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        manager.disconnect(websocket)
        print(f"[WS /ws] WebSocket connection cleaned up for user: {authenticated_user_id or 'unknown'}")


# --- Voice Endpoint (FastRTC - Authentication still a TODO here) ---
async def handle_audio_conversation(audio: tuple[int, np.ndarray]) -> AsyncGenerator[tuple[int, np.ndarray], None]:
     # TODO: Implement robust user_id retrieval for voice sessions.
     # This could involve passing a token with the initial FastRTC connection,
     # similar to WebSocket authentication, but FastRTC's mechanism might differ.
     user_id_for_voice = "PLACEHOLDER_VOICE_USER_ID" # Needs a real, authenticated user ID
     print(f"\n--- [VOICE] Audio chunk received (User: {user_id_for_voice}) ---")

     if not stt_model:
         print("[VOICE_ERROR] STT model not available. Cannot process audio.")
         yield (0, np.array([])) 
         return
     
     user_transcribed_text = stt_model.stt(audio)
     if not user_transcribed_text or not user_transcribed_text.strip():
         print("[VOICE] Empty transcription from STT.")
         return 
         
     print(f"[VOICE] User (STT - {user_id_for_voice}): {user_transcribed_text}")
     
     # Placeholder for full chat/agent logic using user_id_for_voice
     # This should ideally invoke the same chat logic as the /chat endpoint,
     # potentially passing user_id_for_voice, transcribed_text, and any voice-specific context.
     bot_response_text = f"Voice processing is a work in progress for user {user_id_for_voice}. You said: '{user_transcribed_text}'"
     
     if not tts_model:
         print("[VOICE_ERROR] TTS model not available. Cannot generate audio response.")
         yield (0, np.array([]))
         return
         
     tts_options: TTSOptions = {"voice_id": SELECTED_TTS_VOICE} # Or user-selected voice
     async for sr, chunk in tts_model.stream_tts(bot_response_text, options=tts_options):
          if chunk is not None and chunk.size > 0:
              yield (sr, chunk)
              
     print(f"--- [VOICE] Finished processing audio response for {user_id_for_voice} ---")

# Assuming Stream, ReplyOnPause, etc. are correctly imported and configured
voice_stream_handler = Stream(
    ReplyOnPause(
        handle_audio_conversation,
        algo_options=AlgoOptions(), # Configure VAD algorithm options
        model_options=SileroVadOptions(), # Configure VAD model options
        can_interrupt=False # Consider if interruption is desired
    ),
    mode="send-receive", # Bidirectional audio stream
    modality="audio"
)
voice_stream_handler.mount(app, path="/voice") # Mounts the Gradio/FastRTC interface
print(f"[FASTAPI] {datetime.datetime.now()}: FastRTC voice stream mounted at /voice.")

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Important for Windows when using multiprocessing

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s [%(name)s] %(message)s" # Customize default for other logs
    log_config["loggers"]["uvicorn.error"]["level"] = "INFO" # Set error logger level
    # log_config["loggers"]["uvicorn.access"]["handlers"] = ["access"] # Default is usually fine

    print(f"[UVICORN] Starting Uvicorn server on host 0.0.0.0, port 5000...")
    print(f"[UVICORN] API Documentation available at http://localhost:5000/docs")
    
    uvicorn.run(
        "__main__:app", # Points to the app instance in the current file
        host="0.0.0.0",
        port=5000,
        lifespan="on", # Handles startup/shutdown events
        reload=False,  # Set to True for development if auto-reload on code changes is needed
        workers=1,     # Adjust worker count based on deployment needs and CPU cores
        log_config=log_config
    )