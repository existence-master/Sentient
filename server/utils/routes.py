from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import traceback
import pickle # For Google Auth
import os

# Assuming these are initialized in app.py and can be imported
from server.app.app import (
    auth,
    PermissionChecker,
    aes_encrypt, 
    aes_decrypt, 
    get_management_token, 
    # load_user_profile, # Not directly used here, mongo_manager is used
    # write_user_profile, # Not directly used here, mongo_manager is used
    DATA_SOURCES, 
    CREDENTIALS_DICT, 
    SCOPES, 
    BASE_DIR,
    mongo_manager, # Import global mongo_manager
    DATA_SOURCES_CONFIG, # Import for initializing engine state
    task_queue, 
    memory_backend, 
    manager as websocket_manager # Ensure this is your WebSocketManager instance
)


from server.agents.runnables import get_tool_runnable # For elaborator
from server.agents.prompts import elaborator_system_prompt_template, elaborator_user_prompt_template
# Pydantic models moved from app.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import httpx # For Auth0 M2M calls
from google_auth_oauthlib.flow import InstalledAppFlow # For Google OAuth flow
from google.auth.transport.requests import Request # For Google API requests


router = APIRouter(
    prefix="/utils",
    tags=["Utilities & Configuration"]
)

# --- Pydantic Models for Utility Endpoints ---
class ElaboratorMessage(BaseModel):
    input: str
    purpose: str

class EncryptionRequest(BaseModel):
    data: str

class DecryptionRequest(BaseModel):
    encrypted_data: str

class SetReferrerRequest(BaseModel):
    referral_code: str

class SetDataSourceEnabledRequest(BaseModel):
    source: str
    enabled: bool

# --- Utility Endpoints ---
@router.get("/", status_code=status.HTTP_200_OK, summary="API Root (Utils)")
async def utils_root(): # Renamed to avoid conflict if main app has "/"
    # This endpoint is mostly for testing the router. The main "/" is in app.py.
    return {"message": "Utilities API section is running."}


@router.post("/elaborator", status_code=status.HTTP_200_OK, summary="Elaborate Text")
async def elaborate(message: ElaboratorMessage, user_id: str = Depends(PermissionChecker(required_permissions=["use:elaborator"]))):
    print(f"[ENDPOINT /utils/elaborator] User {user_id}, input: '{message.input[:50]}...'")
    try:
        # get_tool_runnable is defined in agents/runnables.py and imported via app.py
        elaborator_runnable_instance = get_tool_runnable(elaborator_system_prompt_template, elaborator_user_prompt_template, None, ["query", "purpose"])
        output = elaborator_runnable_instance.invoke({"query": message.input, "purpose": message.purpose})
        return JSONResponse(content={"message": output})
    except Exception as e:
        print(f"[ERROR] /utils/elaborator: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Elaboration failed.")

@router.post("/encrypt", status_code=status.HTTP_200_OK, summary="Encrypt Data")
async def encrypt_data_route(request: EncryptionRequest): # Added _route to avoid conflict
    try:
        return JSONResponse(content={"encrypted_data": aes_encrypt(request.data)})
    except Exception as e:
        print(f"[ERROR] /utils/encrypt: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Encryption failed.")

@router.post("/decrypt", status_code=status.HTTP_200_OK, summary="Decrypt Data")
async def decrypt_data_route(request: DecryptionRequest): # Added _route to avoid conflict
    try:
        return JSONResponse(content={"decrypted_data": aes_decrypt(request.encrypted_data)})
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Decryption failed: Invalid data or key.")
    except Exception as e:
        print(f"[ERROR] /utils/decrypt: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Decryption failed.")

@router.post("/get-role", status_code=status.HTTP_200_OK, summary="Get User Role from Token")
async def get_role_route(payload: dict = Depends(auth.get_decoded_payload_with_claims)): # Added _route
    user_id = payload["user_id"]
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    CUSTOM_CLAIMS_NAMESPACE = f"{AUTH0_AUDIENCE}/" if AUTH0_AUDIENCE and not AUTH0_AUDIENCE.endswith('/') else AUTH0_AUDIENCE
    print(f"[ENDPOINT /utils/get-role] Called by user {user_id} (from token claims).")
    user_role = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}role", "free")
    print(f"User {user_id} role from token claim: {user_role}")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"role": user_role})

@router.post("/get-referral-code", status_code=status.HTTP_200_OK, summary="Get Referral Code from Token")
async def get_referral_code_route(payload: dict = Depends(auth.get_decoded_payload_with_claims)): # Added _route
    user_id = payload["user_id"]
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    CUSTOM_CLAIMS_NAMESPACE = f"{AUTH0_AUDIENCE}/" if AUTH0_AUDIENCE and not AUTH0_AUDIENCE.endswith('/') else AUTH0_AUDIENCE
    print(f"[ENDPOINT /utils/get-referral-code] Called by user {user_id} (from token claims).")
    referral_code = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}referralCode", None)
    if not referral_code:
        print(f"User {user_id} referral code not found in token claims.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"referralCode": None})
    print(f"User {user_id} referral code from token claim found.")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"referralCode": referral_code})

@router.post("/get-referrer-status", status_code=status.HTTP_200_OK, summary="Get Referrer Status from Token")
async def get_referrer_status_route(payload: dict = Depends(auth.get_decoded_payload_with_claims)): # Added _route
    user_id = payload["user_id"]
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    CUSTOM_CLAIMS_NAMESPACE = f"{AUTH0_AUDIENCE}/" if AUTH0_AUDIENCE and not AUTH0_AUDIENCE.endswith('/') else AUTH0_AUDIENCE
    print(f"[ENDPOINT /utils/get-referrer-status] Called by user {user_id} (from token claims).")
    referrer_status = payload.get(f"{CUSTOM_CLAIMS_NAMESPACE}referrerStatus", False)
    print(f"User {user_id} referrer status from token claim: {referrer_status}")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"referrerStatus": referrer_status})

@router.post("/set-referrer-status", status_code=status.HTTP_200_OK, summary="Set Referrer Status by Code (Admin Action)")
async def set_referrer_status_route( # Renamed to avoid conflict with app.py if it was there
    request: SetReferrerRequest,
    user_id_making_request: str = Depends(PermissionChecker(required_permissions=["admin:user_metadata"]))
):
    referral_code_to_find = request.referral_code
    AUTH0_DOMAIN_LOCAL = os.getenv("AUTH0_DOMAIN") # Use local var to avoid shadowing
    print(f"[ENDPOINT /utils/set-referrer-status] User {user_id_making_request} trying to set referrer for code {referral_code_to_find}.")
    try:
        mgmt_token = get_management_token()
        if not mgmt_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="M2M token unavailable.")

        headers_search = {"Authorization": f"Bearer {mgmt_token}", "Accept": "application/json"}
        search_url = f"https://{AUTH0_DOMAIN_LOCAL}/api/v2/users"
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

        update_url = f"https://{AUTH0_DOMAIN_LOCAL}/api/v2/users/{user_to_update_id}"
        update_payload = {"app_metadata": {"referrer": True}} # Assuming this is the correct claim name
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
        print(f"[ERROR] /utils/set-referrer-status: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set referrer status.")


# --- Google Authentication Endpoint ---
@router.post("/authenticate-google", status_code=status.HTTP_200_OK, summary="Authenticate Google Services")
async def authenticate_google_route(user_id: str = Depends(PermissionChecker(required_permissions=["manage:google_auth"]))): # Added _route
    print(f"[ENDPOINT /utils/authenticate-google] User {user_id}.")
    # Path adjustments might be needed if BASE_DIR is not correctly resolved here.
    # For now, assume BASE_DIR is correctly defined and imported from app.py.
    token_dir = os.path.join(BASE_DIR, "..", "..", "tokens") 
    os.makedirs(token_dir, exist_ok=True)
    token_path = os.path.join(token_dir, f"token_{user_id}.pickle")
    
    creds = None
    try:
        if os.path.exists(token_path):
            with open(token_path, "rb") as token_file:
                creds = pickle.load(token_file)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                 # This typically should initiate an OAuth flow, which is hard to do server-side without user interaction.
                 # For now, if token is invalid/expired and cannot be refreshed, raise an error.
                 # The frontend should handle the initial OAuth flow.
                 raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google auth required or token expired. Please re-authenticate via the frontend.")
            with open(token_path, "wb") as token_file:
                pickle.dump(creds, token_file)
        return JSONResponse(content={"success": True, "message": "Google auth successful."})
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] Google auth for {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Google auth failed: {str(e)}")


# --- Data Source Configuration Endpoints ---
@router.post("/set_data_source_enabled", status_code=status.HTTP_200_OK, summary="Enable/Disable Data Source")
async def set_data_source_enabled_endpoint(request: SetDataSourceEnabledRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))):
    print(f"[ENDPOINT /utils/set_data_source_enabled] User {user_id}, Source: {request.source}, Enabled: {request.enabled}")
    if request.source not in DATA_SOURCES_CONFIG: # Check against DATA_SOURCES_CONFIG keys
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data source: {request.source}")
    try:
        # 1. Update the user's profile setting (optional, but good for UI consistency)
        profile = await mongo_manager.get_user_profile(user_id)
        if not profile: # Should not happen if user is authenticated and profile exists
             profile = {"user_id": user_id, "userData": {}} # Create a basic profile structure
        
        if "userData" not in profile: profile["userData"] = {}
        profile["userData"][f"{request.source}_polling_enabled"] = request.enabled # Store user preference
        
        profile_update_success = await mongo_manager.update_user_profile(user_id, profile)
        if not profile_update_success:
             print(f"[WARN] Failed to update user profile for {user_id} with {request.source} enabled status.")
             # Continue to update polling state regardless, as it's critical

        # 2. Update the polling_state_store directly for is_enabled
        polling_state_update = {"is_enabled": request.enabled}
        if request.enabled:
            # If enabling, also set next_scheduled_poll_time to now to trigger immediate consideration by scheduler
            polling_state_update["next_scheduled_poll_time"] = datetime.datetime.now(datetime.timezone.utc)
            polling_state_update["is_currently_polling"] = False # Ensure it's not locked
            polling_state_update["error_backoff_until_timestamp"] = None 
            polling_state_update["consecutive_empty_polls_count"] = 0 
            # Consider if current_polling_interval_seconds should be reset to default
            # polling_state_update["current_polling_interval_seconds"] = DEFAULT_INITIAL_INTERVAL_SECONDS 
        
        success_polling_state = await mongo_manager.update_polling_state(user_id, request.source, polling_state_update)
        
        if not success_polling_state and request.enabled:
            # If state didn't exist and we are enabling, it means upsert happened.
            # Or if update_one somehow didn't modify (e.g., no change needed), still re-init state for safety.
            print(f"[SET_DATA_SOURCE] Polling state for {user_id}/{request.source} might have been newly created or unchanged. Ensuring initialization.")
        
        # If it was just enabled, ensure its polling state is fully initialized (this creates if not exists).
        if request.enabled:
            engine_config = DATA_SOURCES_CONFIG.get(request.source)
            if engine_config and engine_config.get("engine_class"):
                engine_class = engine_config["engine_class"]
                # Create a temporary instance just to initialize/ensure state
                temp_engine = engine_class(user_id, task_queue, memory_backend, websocket_manager, mongo_manager)
                await temp_engine.initialize_polling_state() 
                print(f"[SET_DATA_SOURCE] Polling state initialized/verified for {user_id}/{request.source} after enabling.")
            else:
                print(f"[SET_DATA_SOURCE_ERROR] No engine_class found for {request.source} in DATA_SOURCES_CONFIG.")
                # This is a configuration error on the server side.

        return JSONResponse(content={"status": "success", "message": f"Data source '{request.source}' status set to {request.enabled}."})
    except Exception as e:
        print(f"[ERROR] /utils/set_data_source_enabled {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set data source status.")