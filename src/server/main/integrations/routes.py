import os
import datetime
import json
import base64
import asyncio
import time
import uuid
import httpx
import logging
from composio import Composio, types
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from typing import Tuple

from main.integrations.models import (ManualConnectRequest, OAuthConnectRequest, DisconnectRequest,
                                      ComposioInitiateRequest, ComposioFinalizeRequest)
from main.dependencies import mongo_manager, auth_helper
from main.auth.utils import aes_encrypt, PermissionChecker
from main.config import (
    INTEGRATIONS_CONFIG,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET,
    TRELLO_CLIENT_ID, COMPOSIO_API_KEY,
    GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, SLACK_CLIENT_ID,
    SLACK_CLIENT_SECRET, NOTION_CLIENT_ID, NOTION_CLIENT_SECRET,
)
from workers.tasks import execute_triggered_task
from workers.proactive.utils import event_pre_filter
from main.plans import PRO_ONLY_INTEGRATIONS
from .utils import waha_request_from_main

logger = logging.getLogger(__name__)

# Initialize Composio SDK
composio = Composio(api_key=COMPOSIO_API_KEY)

router = APIRouter(
    prefix="/integrations",
    tags=["Integrations Management"]
)

@router.get("/sources", summary="Get all available integration sources and their status")
async def get_integration_sources(user_id: str = Depends(auth_helper.get_current_user_id)):
    user_profile = await mongo_manager.get_user_profile(user_id)
    user_integrations = user_profile.get("userData", {}).get("integrations", {}) if user_profile else {}

    all_sources = []
    for name, config in INTEGRATIONS_CONFIG.items():
        source_info = config.copy()
        source_info["name"] = name
        user_connection = user_integrations.get(name, {})
        source_info["connected"] = user_connection.get("connected", False)

        # Add Composio-specific config for the client
        if source_info["auth_type"] == "composio":
            # Use the explicitly defined environment variable name from the config
            env_var_name = config.get("auth_config_env_var")
            if env_var_name:
                source_info["auth_config_id"] = os.getenv(env_var_name)
            else:
                source_info["auth_config_id"] = None

        # Add public config needed by the client for OAuth flow
        if source_info["auth_type"] == "oauth":
            # List of Google services that still use the standard OAuth flow
            google_oauth_services = ["gpeople"]
            if name in google_oauth_services:
                 source_info["client_id"] = GOOGLE_CLIENT_ID
            elif name == 'github':
                 source_info["client_id"] = GITHUB_CLIENT_ID
            elif name == 'slack':
                source_info["client_id"] = SLACK_CLIENT_ID
            elif name == 'notion':
                source_info["client_id"] = NOTION_CLIENT_ID
            elif name == 'trello':
                source_info["client_id"] = TRELLO_CLIENT_ID
            elif name == 'discord':
                source_info["client_id"] = DISCORD_CLIENT_ID

        all_sources.append(source_info)

    return JSONResponse(content={"integrations": all_sources})


@router.post("/connect/manual", summary="Connect an integration using manual credentials")
async def connect_manual_integration(
    request: ManualConnectRequest,
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan
    service_name = request.service_name
    service_config = INTEGRATIONS_CONFIG.get(service_name)

    if not service_config:
        raise HTTPException(status_code=400, detail="Invalid service name.")

    # --- Check Plan Limit ---
    if service_name in PRO_ONLY_INTEGRATIONS and plan == "free":
        raise HTTPException(
            status_code=403,
            detail=f"The {service_config.get('display_name', service_name)} integration is a Pro feature. Please upgrade your plan."
        )

    # Allow Trello to use this endpoint despite being 'oauth' type, as its flow provides a token directly.
    if service_config["auth_type"] != "manual" and service_name != "trello":
        raise HTTPException(status_code=400, detail=f"Service '{service_name}' does not support this connection method.")

    try:
        encrypted_creds = aes_encrypt(json.dumps(request.credentials))
        update_payload = {
            f"userData.integrations.{service_name}.credentials": encrypted_creds,
            f"userData.integrations.{service_name}.connected": True,
            f"userData.integrations.{service_name}.auth_type": "manual"
        }
        success = await mongo_manager.update_user_profile(user_id, update_payload)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save integration credentials.")

        return JSONResponse(content={"message": f"{service_name} connected successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect/oauth", summary="Finalize OAuth2 connection by exchanging code for token")
async def connect_oauth_integration(
    request: OAuthConnectRequest,
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan
    service_name = request.service_name
    if service_name not in INTEGRATIONS_CONFIG or INTEGRATIONS_CONFIG[service_name]["auth_type"] != "oauth":
        raise HTTPException(status_code=400, detail="Invalid service name or auth type is not OAuth.")

    # --- Check Plan Limit ---
    if service_name in PRO_ONLY_INTEGRATIONS and plan == "free":
        raise HTTPException(
            status_code=403,
            detail=f"The {INTEGRATIONS_CONFIG[service_name].get('display_name', service_name)} integration is a Pro feature. Please upgrade your plan."
        )

    token_url = ""
    token_payload = {}
    request_headers = {}
    creds_to_save = {}

    if not request.code:
        raise HTTPException(status_code=400, detail="Authorization code is missing.")

    if service_name.startswith('g') and service_name != 'github': # Google Services
        token_url = "https://oauth2.googleapis.com/token"
        token_payload = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": request.code,
            "grant_type": "authorization_code",
            "redirect_uri": request.redirect_uri,
        }
    elif service_name == 'github':
        token_url = "https://github.com/login/oauth/access_token"
        token_payload = {
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": request.code,
            "redirect_uri": request.redirect_uri
        }
        request_headers = {"Accept": "application/json"}

        # 🔍 DEBUG: Log the exact payload being sent to GitHub
        print(f"[DEBUG] GitHub OAuth request to {token_url}")
        print(f"[DEBUG] Headers: {request_headers}")
        print(f"[DEBUG] Payload: {token_payload}")
        print(f"[DEBUG] redirect_uri from frontend: {request.redirect_uri}")

    elif service_name == 'slack':
        token_url = "https://slack.com/api/oauth.v2.access"
        token_payload = {
            "client_id": SLACK_CLIENT_ID,
            "client_secret": SLACK_CLIENT_SECRET,
            "code": request.code,
            "redirect_uri": request.redirect_uri,
        }
    elif service_name == 'notion':
        token_url = "https://api.notion.com/v1/oauth/token"
        auth_string = f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}"
        auth_bytes = auth_string.encode("ascii")
        base64_string = base64.b64encode(auth_bytes).decode("ascii")
        request_headers = {
            "Authorization": f"Basic {base64_string}",
            "Content-Type": "application/json",
        }
        token_payload = {
            "grant_type": "authorization_code",
            "code": request.code,
            "redirect_uri": request.redirect_uri,
        }
    elif service_name == 'discord':
        token_url = "https://discord.com/api/oauth2/token"
        token_payload = {
            "client_id": DISCORD_CLIENT_ID,
            "client_secret": DISCORD_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": request.code,
            "redirect_uri": request.redirect_uri
        }
    else:
        raise HTTPException(status_code=400, detail=f"OAuth flow not implemented for {service_name}")

    try:
        async with httpx.AsyncClient() as client:
            if service_name == 'notion':
                token_response = await client.post(token_url, json=token_payload, headers=request_headers)
            else:
                token_response = await client.post(token_url, data=token_payload, headers=request_headers)
            token_response.raise_for_status()
            token_data = token_response.json()
        
        if service_name.startswith('g') and service_name != 'github':
            # Extract granted scopes from the token response
            granted_scopes = token_data.get("scope", "").split(" ")
            # Update the credentials object to include scopes
            creds_to_save = {
                "token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "token_uri": token_url,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "scopes": granted_scopes,  # <--- SAVE THE SCOPES HERE
            }
        elif service_name == 'github':
             if "access_token" not in token_data:
                raise HTTPException(status_code=400, detail=f"GitHub OAuth error: {token_data.get('error_description', 'No access token in response.')}")
             creds_to_save = {"access_token": token_data["access_token"]}
        elif service_name == 'slack':
            if not token_data.get("ok"):
                 raise HTTPException(status_code=400, detail=f"Slack OAuth error: {token_data.get('error', 'Unknown error.')}")
            # The user token is nested inside authed_user
            creds_to_save = token_data # Store the whole response for now
        elif service_name == 'notion':
             if "access_token" not in token_data:
                raise HTTPException(status_code=400, detail=f"Notion OAuth error: {token_data.get('error_description', 'No access token in response.')}")
             creds_to_save = token_data # Store the whole object (access_token, workspace_id, etc.)
        elif service_name == 'discord':
            if "access_token" not in token_data:
                raise HTTPException(status_code=400, detail=f"Discord OAuth error: {token_data.get('error_description', 'No access token.')}")
            creds_to_save = token_data # This includes access_token, refresh_token, and the 'bot' object with bot token

        encrypted_creds = aes_encrypt(json.dumps(creds_to_save))

        update_payload = {
            f"userData.integrations.{service_name}.credentials": encrypted_creds,
            f"userData.integrations.{service_name}.connected": True,
            f"userData.integrations.{service_name}.auth_type": "oauth"
        }
        success = await mongo_manager.update_user_profile(user_id, update_payload)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save integration credentials.")
        
        if service_name in ['gmail', 'gcalendar']:
            await mongo_manager.update_polling_state(
                user_id,
                service_name,
                "triggers",
                {
                    "is_enabled": True,
                    "is_currently_polling": False,
                    "next_scheduled_poll_time": datetime.datetime.now(datetime.timezone.utc), # Poll immediately
                    "last_successful_poll_timestamp_unix": None,
                }
            )

        return JSONResponse(content={"message": f"{service_name} connected successfully."})

    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("error_description", f"Failed to exchange token with {service_name}.")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect", summary="Disconnect an integration")
async def disconnect_integration(request: DisconnectRequest, user_id: str = Depends(auth_helper.get_current_user_id)):
    service_name = request.service_name
    if service_name not in INTEGRATIONS_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid service name.")

    try:
        # Delete tasks that rely on this tool
        deleted_tasks_count = await mongo_manager.delete_tasks_by_tool(user_id, service_name)
        logger.info(f"Deleted {deleted_tasks_count} tasks for user {user_id} associated with disconnected tool '{service_name}'.")

        # Unset the specific integration object from the user profile
        update_payload = {f"userData.integrations.{service_name}": ""}
        result = await mongo_manager.user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$unset": update_payload}
        )

        if result.modified_count == 0 and deleted_tasks_count == 0:
            # This can happen if the field didn't exist, which is not an error.
            return JSONResponse(content={"message": f"{service_name} was not connected or already disconnected."})

        return JSONResponse(content={"message": f"{service_name} disconnected successfully."})
    except Exception as e:
        logger.error(f"Error disconnecting integration {service_name} for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect/composio/initiate", summary="Initiate Composio OAuth flow")
async def initiate_composio_connection(
    request: ComposioInitiateRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    service_name = request.service_name
    service_config = INTEGRATIONS_CONFIG.get(service_name)
    if not service_config or service_config.get("auth_type") != "composio":
        raise HTTPException(status_code=400, detail="Invalid service for Composio connection.")

    auth_config_id = os.getenv(f"{service_name.upper()}_AUTH_CONFIG_ID")
    if not auth_config_id:
        raise HTTPException(status_code=500, detail=f"Auth Config ID for {service_name} is not configured on the server.")

    try:
        logger.info(f"Initiating Composio for {service_name} with auth_config_id={auth_config_id}")
        callback_url = f"{os.getenv('APP_BASE_URL', 'http://localhost:3000')}/integrations"
        connection_request = composio.connected_accounts.initiate(
            user_id=user_id,
            auth_config_id=auth_config_id,
            callback_url=callback_url,
            config={
                "authScheme": "OAUTH2"
            }
        )

        return JSONResponse(content={"redirect_url": connection_request.redirect_url})
    except Exception as e:
        logger.error(f"Error initiating Composio connection for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect/composio/finalize", summary="Finalize Composio OAuth flow")
async def finalize_composio_connection(
    request: ComposioFinalizeRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    service_name = request.service_name
    connected_account_id = request.connectedAccountId

    try:
        logger.info(f"Waiting for Composio connection {connected_account_id} to become active...")

        
        # USING A MANUAL POLLING LOOP since the SDK's internal wait_for_connection method is broken.
        start_time = time.time()
        timeout = 120  # seconds
        connected_account = None

        while time.time() - start_time < timeout:
            # Use asyncio.to_thread to run the synchronous SDK call in a separate thread
            # The .get() method is an alias for .retrieve() and fetches the account by its ID.
            connected_account = await asyncio.to_thread(
                composio.connected_accounts.get, connected_account_id
            )

            if connected_account and connected_account.status == "ACTIVE":
                break  # Success!

            if connected_account and connected_account.status == "FAILED":
                raise HTTPException(status_code=400, detail="Connection failed during authentication with the provider.")

            await asyncio.sleep(2)  # Wait for 2 seconds before polling again
        else:
            # This block runs if the while loop finishes without a `break`
            raise TimeoutError("Connection verification timed out.")

        logger.info(f"Finalized Composio connection for {service_name}: ID {connected_account_id}")

        trigger_id = None
        if service_name in ["gcalendar", "gmail"]:
            slug_map = {
                "gcalendar": "GOOGLECALENDAR_GOOGLE_CALENDAR_EVENT_SYNC_TRIGGER",
                "gmail": "GMAIL_NEW_GMAIL_MESSAGE"
            }
            trigger_config = {"calendarId": "primary"} if service_name == "gcalendar" else {}
            try:
                logger.info(f"Setting up Composio trigger for {service_name} for user {user_id}")
                trigger = await asyncio.to_thread(
                    composio.triggers.create,
                    slug=slug_map[service_name],
                    user_id=user_id,
                    trigger_config=trigger_config
                )
                trigger_id = trigger.id
                logger.info(f"Successfully created Composio trigger {trigger_id} for {service_name} for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to create Composio trigger for {service_name} for user {user_id}: {e}", exc_info=True)
                # Do not fail the entire connection if trigger creation fails, just log it.

        update_payload = {
            f"userData.integrations.{service_name}.connection_id": connected_account.id,
            f"userData.integrations.{service_name}.connected": True,
            f"userData.integrations.{service_name}.auth_type": "composio"
        }
        if trigger_id:
            update_payload[f"userData.integrations.{service_name}.trigger_id"] = trigger_id

        await mongo_manager.update_user_profile(user_id, update_payload)

        return JSONResponse(content={"message": f"{service_name} connected successfully via Composio."})
    except Exception as e:
        logger.error(f"Error finalizing Composio connection for {user_id}: {e}", exc_info=True)
        if isinstance(e, TimeoutError) or "timed out" in str(e).lower():
            raise HTTPException(status_code=408, detail="Connection verification timed out. Please try again.")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/composio/webhook", summary="Webhook receiver for Composio triggers", include_in_schema=False)
async def composio_webhook(request: Request):
    """
    Receives event payloads from Composio triggers, filters them, and dispatches them
    to the appropriate Celery worker for processing.
    """
    try:
        payload = await request.json()
        user_id = payload.get("userId")
        trigger_slug = payload.get("triggerSlug")
        event_data = payload.get("payload")

        if not all([user_id, trigger_slug, event_data]):
            raise HTTPException(status_code=400, detail="Missing required fields in webhook payload.")

        service_name_map = {
            "GOOGLECALENDAR_GOOGLE_CALENDAR_EVENT_SYNC_TRIGGER": "gcalendar",
            "GMAIL_NEW_GMAIL_MESSAGE": "gmail"
        }
        event_type_map = {
            "GOOGLECALENDAR_GOOGLE_CALENDAR_EVENT_SYNC_TRIGGER": "new_event",
            "GMAIL_NEW_GMAIL_MESSAGE": "new_email"
        }

        service_name = service_name_map.get(trigger_slug)
        event_type = event_type_map.get(trigger_slug)

        if not service_name:
            logger.warning(f"Received webhook for unhandled trigger slug: {trigger_slug}")
            return JSONResponse(content={"status": "ignored", "reason": "unhandled trigger"})

        logger.info(f"Received Composio trigger for user '{user_id}' - Service: '{service_name}', Event: '{event_type}'")

        # --- Filtering Logic (similar to old poller) ---
        user_profile = await mongo_manager.get_user_profile(user_id)
        if not user_profile:
            logger.error(f"Webhook received for non-existent user '{user_id}'. Ignoring.")
            return JSONResponse(content={"status": "ignored", "reason": "user not found"})

        user_email = user_profile.get("userData", {}).get("personalInfo", {}).get("email")

        # 1. Apply user-defined privacy filters
        # (This logic would be similar to the one in the old poller service.py)

        # 2. Apply system-wide pre-filter
        if not event_pre_filter(event_data, service_name, user_email):
            logger.info(f"Event for user '{user_id}' was discarded by the pre-filter.")
            return JSONResponse(content={"status": "ignored", "reason": "pre-filter discard"})

        # 3. Dispatch to Celery worker
        execute_triggered_task.delay(
            user_id=user_id,
            source=service_name,
            event_type=event_type,
            event_data=event_data
        )
        logger.info(f"Dispatched event to triggered task worker for user '{user_id}'.")
        return JSONResponse(content={"status": "received"})
    except Exception as e:
        logger.error(f"Error processing Composio webhook: {e}", exc_info=True)
        # Return a 200 to Composio to prevent retries on our internal errors.
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=200)

# --- NEW WHATSAPP FULL CONTROL ROUTES ---

@router.post("/whatsapp/connect/initiate", summary="Start a WAHA session and get a QR code")
async def initiate_whatsapp_connection(user_id: str = Depends(auth_helper.get_current_user_id)):
    try:
        # Step 1: Start the session for the user
        # --- MODIFICATION START (v2) ---
        # The user_id from Auth0 (e.g., 'google-oauth2|123...') contains invalid characters ('|')
        # for a WAHA session name. We must sanitize it first.
        sanitized_session_name = user_id.replace("|", "_")

        create_payload = {
            "name": sanitized_session_name,
            "start": True
        }
        try:
            await waha_request_from_main("POST", "/api/sessions", session=sanitized_session_name, json_data=create_payload)
        except HTTPException as e:
            if e.status_code == 422 and "already exists" in e.detail:
                logger.info(f"WAHA session '{sanitized_session_name}' already exists. Proceeding to get QR code.")
                # Session exists, we can ignore this error and continue.
            else:
                raise # Re-raise other HTTP exceptions

        await asyncio.sleep(1)  # Give it a moment to initialize
        status_res = await waha_request_from_main("GET", "/api/sessions/{session}", session=sanitized_session_name)
        session_status = status_res.get("status")
        if session_status not in ["STARTING", "SCAN_QR_CODE", "WORKING"]:
            raise HTTPException(status_code=500, detail=f"Failed to start WhatsApp session. Status: {session_status}")

        # Step 2: Get the QR code
        await asyncio.sleep(5) 
        qr_result = await waha_request_from_main("GET", "/api/{session}/auth/qr", session=sanitized_session_name, params={"format": "image"}) # noqa
        return JSONResponse(content=qr_result)

    except Exception as e:
        logger.error(f"Error initiating WhatsApp connection for {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/whatsapp/connect/status", summary="Check the status of a WAHA session")
async def get_whatsapp_connection_status(user_id: str = Depends(auth_helper.get_current_user_id)):
    try:
        sanitized_session_name = user_id.replace("|", "_")
        status_result = await waha_request_from_main("GET", "/api/sessions/{session}", session=sanitized_session_name)
        session_status = status_result.get("status", "UNKNOWN")

        if session_status == "WORKING":
            await mongo_manager.update_user_profile(user_id, {"userData.integrations.whatsapp.connected": True})
        
        return JSONResponse(content={"status": session_status})
    except HTTPException as e:
        # If the session is not found (404), it's not an error, just means it's not connected.
        if e.status_code == 404:
            return JSONResponse(content={"error": "Session not found. Please try connecting again."}, status_code=404)
        logger.error(f"Error checking WhatsApp status for {user_id}: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error checking WhatsApp status for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/whatsapp/disconnect", summary="Delete a WAHA session")
async def disconnect_whatsapp_connection(user_id: str = Depends(auth_helper.get_current_user_id)):
    try:
        sanitized_session_name = user_id.replace("|", "_")
        await waha_request_from_main("DELETE", "/api/sessions/{session}", session=sanitized_session_name)
        await mongo_manager.update_user_profile(user_id, {"userData.integrations.whatsapp.connected": False})
        return JSONResponse(content={"message": "WhatsApp session disconnected successfully."})
    except Exception as e:
        # Even if deleting fails (e.g., session was already deleted), mark as disconnected.
        await mongo_manager.update_user_profile(user_id, {"userData.integrations.whatsapp.connected": False})
        logger.error(f"Error disconnecting WhatsApp for {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            # It's okay if deleting fails, the main goal is to mark as disconnected.
            return JSONResponse(content={"message": "WhatsApp session disconnected."})
        raise HTTPException(status_code=500, detail=str(e))