from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
import asyncio
import json # For WebSocket messages
import uuid # For new chat_id
import datetime # For timestamps

# Assuming these are initialized in app.py and can be imported
from server.app.app import (
    auth,
    PermissionChecker,
    mongo_manager,
    manager as websocket_manager, # Renamed to avoid conflict if a local 'manager' is defined
    get_chat_history_messages,
    add_message_to_db,
    load_user_profile, # Required for chat
    write_user_profile, # Required for chat if active_chat_id needs update
    chat_runnable, # For /chat
    unified_classification_runnable, # For /chat
    priority_runnable, # For /chat if agent category
    task_queue, # For /chat if agent category
    memory_backend, # For /chat if memory category
    internet_query_reframe_runnable, # For /chat if internet search
    internet_summary_runnable, # For /chat if internet search
    get_reframed_internet_query, # from common.functions, imported in app.py
    get_search_results, # from common.functions, imported in app.py
    get_search_summary, # from common.functions, imported in app.py
    update_neo4j_with_onboarding_data, # For onboarding
    update_neo4j_with_personal_info, # for set-user-data
    graph_driver, # for Neo4j updates
    embed_model, # for Neo4j updates
    start_user_context_engines
)

# Pydantic models moved from app.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union, Tuple
import time # For assistant_msg_id_ts in chat

router = APIRouter(
    tags=["Common (Chat, DB, Notifications, Voice, User Activity)"]
)

# --- Pydantic Models for Common Endpoints ---
class Message(BaseModel): # Used by /chat
    input: str
    pricing: str # Example, might be part of user claims or profile
    credits: int # Example

class OnboardingData(BaseModel): # Used by /onboarding
    data: Dict[str, Any]

class UpdateUserDataRequest(BaseModel): # Used by /set-user-data
    data: Dict[str, Any]

class AddUserDataRequest(BaseModel): # Used by /add-db-data
    data: Dict[str, Any]


# --- DB related endpoints (Onboarding, Profile) ---
@router.post("/onboarding", status_code=status.HTTP_200_OK, summary="Save Onboarding Data")
async def save_onboarding_data_endpoint(onboarding_data: OnboardingData, user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /onboarding] User {user_id}, Onboarding Data: {str(onboarding_data.data)[:100]}...")
    try:
        existing_profile = await mongo_manager.get_user_profile(user_id)
        user_data_to_update = existing_profile.get("userData", {}) if existing_profile else {}

        for key, new_val in onboarding_data.data.items():
            if key in user_data_to_update and isinstance(user_data_to_update[key], list) and isinstance(new_val, list):
                user_data_to_update[key].extend(item for item in new_val if item not in user_data_to_update[key])
            elif key in user_data_to_update and isinstance(user_data_to_update[key], dict) and isinstance(new_val, dict):
                user_data_to_update[key].update(new_val)
            else:
                user_data_to_update[key] = new_val
        
        # Update only the 'userData' field
        # Also set onboardingComplete to true
        user_data_to_update["onboardingComplete"] = True
        success = await mongo_manager.update_user_profile(user_id, {"userData": user_data_to_update})

        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save onboarding data to user profile.")

        if graph_driver and embed_model:
            print(f"[NEO4J_ONBOARDING] Initiating Neo4j update for onboarding data for user {user_id}.")
            await update_neo4j_with_onboarding_data(user_id, onboarding_data.data, graph_driver, embed_model)
        else:
            print(f"[WARN] Neo4j driver/embed_model not available. Skipping Neo4j update for onboarding data for user {user_id}.")

        return JSONResponse(content={"message": "Onboarding data saved successfully.", "status": 200})
    except Exception as e:
        print(f"[ERROR] /onboarding {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save onboarding data.")

@router.post("/check-user-profile", status_code=status.HTTP_200_OK, summary="Check and Create User Profile")
async def check_and_create_user_profile(user_id: str = Depends(PermissionChecker(required_permissions=["read:profile", "write:profile"]))):
    print(f"[ENDPOINT /check-user-profile] Called by user {user_id}.")
    try:
        profile = await mongo_manager.get_user_profile(user_id)
        if profile:
            print(f"[USER_PROFILE] Profile exists for user {user_id}.")
            # Check if onboardingComplete field exists, if not, assume it's not complete.
            onboarding_complete = profile.get("userData", {}).get("onboardingComplete", False)
            return JSONResponse(content={"profile_exists": True, "onboarding_complete": onboarding_complete})
        else:
            print(f"[USER_PROFILE] Profile does NOT exist for user {user_id}. Creating initial profile.")
            initial_profile_data = {
                "user_id": user_id,
                "createdAt": datetime.datetime.now(datetime.timezone.utc),
                "userData": {
                    "personalInfo": {"name": "New User", "email": user_id},
                    "onboardingComplete": False,
                    "active_chat_id": str(uuid.uuid4())
                }
            }
            success = await mongo_manager.update_user_profile(user_id, initial_profile_data)
            if not success:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create initial user profile.")
            print(f"[USER_PROFILE] Initial profile created for user {user_id}.")
            return JSONResponse(content={"profile_exists": False, "onboarding_complete": False})
    except Exception as e:
        print(f"[ERROR] /check-user-profile {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to check or create user profile.")

@router.post("/set-user-data", status_code=status.HTTP_200_OK, summary="Set User Profile Data")
async def set_db_data(request: UpdateUserDataRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /set-user-data] User {user_id}, Data: {str(request.data)[:100]}...")
    try:
        profile = await load_user_profile(user_id)
        if "userData" not in profile: # Should not happen if check-user-profile is called first
            profile["userData"] = {}
        
        # Merge new data into existing userData, overwriting existing keys at the top level of request.data
        for key, value in request.data.items():
            profile["userData"][key] = value
            
        success = await write_user_profile(user_id, profile)
        
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to write user profile.")

        if "personalInfo" in profile["userData"] and graph_driver:
            personal_info_data = profile["userData"]["personalInfo"]
            if isinstance(personal_info_data, dict):
                print(f"[NEO4J_INITIATE_PERSONAL_INFO_UPDATE] Initiating Neo4j update for personal info for user {user_id}.")
                await update_neo4j_with_personal_info(user_id, personal_info_data, graph_driver)
            else:
                print(f"[WARN] personalInfo for user {user_id} is not a dictionary. Skipping Neo4j update.")
        
        return JSONResponse(content={"message": "User data stored successfully.", "status": 200})
    except Exception as e:
        print(f"[ERROR] /set-user-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set user data.")

@router.post("/add-db-data", status_code=status.HTTP_200_OK, summary="Add/Merge User Profile Data")
async def add_db_data_route(request: AddUserDataRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /add-db-data] User {user_id}, Data: {str(request.data)[:100]}...")
    try:
        profile = await load_user_profile(user_id)
        existing_udata = profile.get("userData", {})
        
        for key, new_val in request.data.items():
            if key in existing_udata and isinstance(existing_udata[key], list) and isinstance(new_val, list):
                existing_udata[key].extend(item for item in new_val if item not in existing_udata[key])
            elif key in existing_udata and isinstance(existing_udata[key], dict) and isinstance(new_val, dict):
                existing_udata[key].update(new_val)
            else:
                existing_udata[key] = new_val
        
        profile["userData"] = existing_udata
        success = await write_user_profile(user_id, profile)
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to write merged user profile.")
        return JSONResponse(content={"message": "User data merged successfully.", "status": 200})
    except Exception as e:
        print(f"[ERROR] /add-db-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to merge user data.")

@router.post("/get-user-data", status_code=status.HTTP_200_OK, summary="Get User Profile Data")
async def get_db_data(user_id: str = Depends(PermissionChecker(required_permissions=["read:profile"]))):
    print(f"[ENDPOINT /get-user-data] User {user_id}.")
    try:
        profile = await load_user_profile(user_id)
        return JSONResponse(content={"data": profile.get("userData", {}), "status": 200})
    except Exception as e:
        print(f"[ERROR] /get-user-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user data.")


# --- Chat Endpoints ---
@router.post("/get-history", status_code=status.HTTP_200_OK, summary="Get Chat History")
async def get_history(user_id: str = Depends(PermissionChecker(required_permissions=["read:chat"]))):
    print(f"[ENDPOINT /get-history] Called by user {user_id}.")
    try:
        messages, effective_chat_id = await get_chat_history_messages(user_id, None)
        return JSONResponse(content={"messages": messages, "activeChatId": effective_chat_id})
    except Exception as e:
        print(f"[ERROR] /get-history {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get chat history.")

@router.post("/clear-chat-history", status_code=status.HTTP_200_OK, summary="Clear Chat History")
async def clear_chat_history_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["write:chat"]))):
    print(f"[ENDPOINT /clear-chat-history] Called by user {user_id}.")
    try:
        user_profile = await mongo_manager.get_user_profile(user_id)
        active_chat_id = user_profile.get("userData", {}).get("active_chat_id", None)

        if active_chat_id:
            deleted = await mongo_manager.delete_chat_history(user_id, active_chat_id)
            if deleted:
                # Create a new active_chat_id
                new_active_chat_id = str(uuid.uuid4())
                await mongo_manager.update_user_profile(user_id, {"userData.active_chat_id": new_active_chat_id})
                print(f"[CHAT_HISTORY] Cleared active chat {active_chat_id} for user {user_id}. New active chat ID: {new_active_chat_id}")
                return JSONResponse(content={"message": "Active chat history cleared.", "activeChatId": new_active_chat_id})
            else:
                # If deletion failed but chat_id existed, this is unexpected.
                # For robustness, still assign a new active_chat_id.
                new_active_chat_id = str(uuid.uuid4())
                await mongo_manager.update_user_profile(user_id, {"userData.active_chat_id": new_active_chat_id})
                print(f"[CHAT_HISTORY_WARN] No chat found to delete with ID {active_chat_id} for user {user_id}, but assigning new active chat ID: {new_active_chat_id}")
                return JSONResponse(content={"message": "No active chat found to clear. New chat session started.", "activeChatId": new_active_chat_id})
        else:
            # No active chat was set, create one
            new_active_chat_id = str(uuid.uuid4())
            await mongo_manager.update_user_profile(user_id, {"userData.active_chat_id": new_active_chat_id})
            print(f"[CHAT_HISTORY] No active chat to clear for user {user_id}. New active chat ID: {new_active_chat_id}")
            return JSONResponse(content={"message": "No active chat to clear. New chat session started.", "activeChatId": new_active_chat_id})

    except Exception as e:
        print(f"[ERROR] /clear-chat-history {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear chat history.")


@router.post("/get-all-chat-ids", status_code=status.HTTP_200_OK, summary="Get All Chat IDs for User")
async def get_all_chat_ids_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["read:chat"]))):
    print(f"[ENDPOINT /get-all-chat-ids] Called by user {user_id}.")
    try:
        chat_ids = await mongo_manager.get_all_chat_ids_for_user(user_id)
        return JSONResponse(content={"chat_ids": chat_ids})
    except Exception as e:
        print(f"[ERROR] /get-all-chat-ids {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat IDs.")

@router.post("/chat", status_code=status.HTTP_200_OK, summary="Process Chat Message")
async def chat_endpoint(message: Message, user_id: str = Depends(PermissionChecker(required_permissions=["write:chat"]))):
    print(f"[ENDPOINT /chat] User {user_id}. Input: '{message.input[:50]}...'")
    try:
        user_profile_data = await load_user_profile(user_id)
        username = user_profile_data.get("userData", {}).get("personalInfo", {}).get("name", "User")
        _, active_chat_id = await get_chat_history_messages(user_id, None) # Ensures active_chat_id is set
        
        if not active_chat_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not determine active chat ID.")

        current_chat_history_for_llm, _ = await get_chat_history_messages(user_id, active_chat_id) # Fetch history for current chat
        
        unified_output = unified_classification_runnable.invoke({"query": message.input, "chat_history": current_chat_history_for_llm})
        category = unified_output.get("category", "chat")
        use_personal_context = unified_output.get("use_personal_context", False)
        internet_search_type = unified_output.get("internet", "None") # From boolean to string
        transformed_input = unified_output.get("transformed_input", message.input)

        async def response_generator():
            # ... (response_generator logic from app.py, ensure all dependencies like task_queue, memory_backend are available) ...
            # This part is extensive and will be copied with necessary import adjustments
            memory_used, agents_used, internet_used, pro_features_used = False, False, False, False
            user_context_str, internet_context_str, additional_notes = None, None, ""

            user_msg_id = await add_message_to_db(user_id, active_chat_id, message.input, is_user=True, is_visible=True)
            if not user_msg_id:
                yield json.dumps({"type":"error", "message":"Failed to save user message."})+"\n"
                return
            yield json.dumps({"type": "userMessage", "id": user_msg_id, "message": message.input, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n"
            await asyncio.sleep(0.01) # Give client time to process
            
            assistant_msg_id_ts = str(int(time.time() * 1000)) # Temporary ID for streaming
            assistant_msg_base = {"id": assistant_msg_id_ts, "message": "", "isUser": False, "memoryUsed": False, "agentsUsed": False, "internetUsed": False, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "isVisible": True}

            if category == "agent":
                agents_used = True
                assistant_msg_base["agentsUsed"] = True
                try:
                    priority_response = priority_runnable.invoke({"task_description": transformed_input})
                    priority = priority_response.get("priority", 2)
                    await task_queue.add_task(user_id=user_id, chat_id=active_chat_id, description=transformed_input, priority=priority, username=username, use_personal_context=use_personal_context, internet=internet_search_type)
                    assistant_msg_base["message"] = "Okay, I'll get right on that."
                    # Save assistant message to DB
                    db_assistant_msg_id = await add_message_to_db(user_id, active_chat_id, assistant_msg_base["message"], is_user=False, is_visible=True, agentsUsed=True, task=transformed_input)
                    assistant_msg_base["id"] = db_assistant_msg_id or assistant_msg_id_ts # Use DB ID if available
                    yield json.dumps({"type": "assistantMessage", "messageId": assistant_msg_base["id"], "message": assistant_msg_base["message"], "done": True, "proUsed": False}) + "\n"
                    return
                except Exception as e_agent:
                    print(f"[ERROR] Agent task processing failed: {e_agent}")
                    traceback.print_exc()
                    yield json.dumps({"type":"error", "message":"Failed to process agent task."})+"\n"
                    return

            if category == "memory" or use_personal_context:
                memory_used = True
                assistant_msg_base["memoryUsed"] = True
                yield json.dumps({"type": "intermediary", "message": "Checking context...", "id": assistant_msg_id_ts}) + "\n"
                try:
                    user_context_str = await memory_backend.retrieve_memory(user_id, transformed_input)
                except Exception as e_mem_ret:
                    user_context_str = f"Error retrieving context: {e_mem_ret}"
                if category == "memory": # Store memory if it's a memory-categorized input
                    if message.pricing == "free" and message.credits <= 0:
                        additional_notes += " (Memory update skipped: Pro)"
                    else:
                        pro_features_used = True
                        asyncio.create_task(memory_backend.add_operation(user_id, {"type": "store", "query_text": transformed_input}))


            if internet_search_type and internet_search_type != "None": # Check if boolean is true or string is not "None"
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
                    except Exception as e_search:
                        internet_context_str = f"Error searching: {e_search}"
            
            if category in ["chat", "memory"]: # Also respond for memory category after potential storage
                yield json.dumps({"type": "intermediary", "message": "Thinking...", "id": assistant_msg_id_ts}) + "\n"
                full_response_text = ""
                try:
                    # Fetch latest history again for LLM prompt context, EXCLUDING current user input
                    full_chat_history_from_db_for_llm, _ = await get_chat_history_messages(user_id, active_chat_id)
                    history_for_llm_prompt = []
                    if full_chat_history_from_db_for_llm:
                        # Ensure the last message isn't the one we just added.
                        # The user_msg_id is the ID of the current user's message.
                        history_for_llm_prompt = [m for m in full_chat_history_from_db_for_llm if m.get('_id') != user_msg_id and m.get('id') != user_msg_id]
                        # If user_msg_id is not _id, then it means some messages might be missed.
                        # A robust way is to slice off the last user message if it matches current input
                        if history_for_llm_prompt and history_for_llm_prompt[-1].get("isUser") and history_for_llm_prompt[-1].get("message") == message.input:
                            history_for_llm_prompt = history_for_llm_prompt[:-1]

                    async for token_chunk in chat_runnable.stream_response(inputs={"query": transformed_input, "user_context": user_context_str, "internet_context": internet_context_str, "name": username, "chat_history": history_for_llm_prompt}):
                        if isinstance(token_chunk, str):
                            full_response_text += token_chunk
                            yield json.dumps({"type": "assistantStream", "token": token_chunk, "done": False, "messageId": assistant_msg_base["id"]}) + "\n"
                        await asyncio.sleep(0.001) # Shorter sleep
                    
                    final_message_content = full_response_text + (("\n\n" + additional_notes) if additional_notes else "")
                    assistant_msg_base["message"] = final_message_content
                    
                    # Save final assistant message to DB
                    db_assistant_final_msg_id = await add_message_to_db(user_id, active_chat_id, final_message_content, is_user=False, is_visible=True, memoryUsed=memory_used, agentsUsed=agents_used, internetUsed=internet_used)
                    assistant_msg_base["id"] = db_assistant_final_msg_id or assistant_msg_base["id"] # Update ID with DB one

                    yield json.dumps({"type": "assistantStream", "token": ("\n\n" + additional_notes) if additional_notes else "", "done": True, "memoryUsed": memory_used, "agentsUsed": agents_used, "internetUsed": internet_used, "proUsed": pro_features_used, "messageId": assistant_msg_base["id"]}) + "\n"
                
                except Exception as e_stream:
                    print(f"[ERROR] Error during streaming response generation: {e_stream}")
                    traceback.print_exc()
                    error_msg_user = "Sorry, an error occurred while generating the response."
                    assistant_msg_base["message"] = error_msg_user
                    db_assistant_err_msg_id = await add_message_to_db(user_id, active_chat_id, error_msg_user, is_user=False, is_visible=True, error=True)
                    assistant_msg_base["id"] = db_assistant_err_msg_id or assistant_msg_base["id"]
                    yield json.dumps({"type": "assistantStream", "token": error_msg_user, "done": True, "error": True, "messageId": assistant_msg_base["id"]}) + "\n"
            
        return StreamingResponse(response_generator(), media_type="application/x-ndjson")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] /chat {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Chat processing error.")


# --- Notification Endpoints ---
@router.post("/get-notifications", status_code=status.HTTP_200_OK, summary="Get User Notifications")
async def get_notifications_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["read:notifications"]))):
    print(f"[ENDPOINT /get-notifications] User {user_id}.")
    try:
        notifications = await mongo_manager.get_notifications(user_id)
        s_notifications = []
        for n in notifications:
            if isinstance(n.get('timestamp'), datetime.datetime):
                n['timestamp'] = n['timestamp'].isoformat()
            s_notifications.append(n)
        return JSONResponse(content={"notifications": s_notifications})
    except Exception as e:
        print(f"[ERROR] /get-notifications {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get notifications.")

@router.post("/add-notification", status_code=status.HTTP_201_CREATED, summary="Add New Notification")
async def add_notification_endpoint(notification_data: Dict[str, Any], user_id: str = Depends(PermissionChecker(required_permissions=["write:notifications"]))):
    print(f"[ENDPOINT /add-notification] User {user_id}, Data: {str(notification_data)[:100]}...")
    try:
        # Ensure notification_data includes at least a 'message' or 'text' field
        if not notification_data.get("message") and not notification_data.get("text"):
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Notification data must include 'message' or 'text'.")

        success = await mongo_manager.add_notification(user_id, notification_data)
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add notification.")
        return JSONResponse(content={"message": "Notification added successfully."}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        print(f"[ERROR] /add-notification {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add notification.")

@router.post("/clear-notifications", status_code=status.HTTP_200_OK, summary="Clear All Notifications")
async def clear_notifications_endpoint(user_id: str = Depends(PermissionChecker(required_permissions=["write:notifications"]))):
    print(f"[ENDPOINT /clear-notifications] User {user_id}.")
    try:
        success = await mongo_manager.clear_notifications(user_id)
        if not success:
            print(f"[WARN] No notifications found to clear for user {user_id}, or no document to update.")
        return JSONResponse(content={"message": "All notifications cleared successfully."})
    except Exception as e:
        print(f"[ERROR] /clear-notifications {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear notifications.")


# --- WebSocket Endpoint ---
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    authenticated_user_id: Optional[str] = None
    try:
        authenticated_user_id = await auth.ws_authenticate(websocket)
        if not authenticated_user_id:
            print("[WS /ws] WebSocket authentication failed or connection closed during auth.")
            return
        
        await websocket_manager.connect(websocket, authenticated_user_id)
        while True:
            data = await websocket.receive_text()
            try:
                message_payload = json.loads(data)
                if message_payload.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                print(f"[WS /ws] Received non-JSON message from {authenticated_user_id}. Ignoring.")
            except Exception as e_ws_loop:
                print(f"[WS /ws] Error in WebSocket loop for user {authenticated_user_id}: {e_ws_loop}")
                # Consider breaking or specific error handling
    except WebSocketDisconnect:
        print(f"[WS /ws] Client disconnected (User: {authenticated_user_id or 'unknown'}).")
    except Exception as e:
        print(f"[WS /ws] Unexpected WebSocket error (User: {authenticated_user_id or 'unknown'}): {e}")
        traceback.print_exc()
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except RuntimeError as re:
            print(f"[WS /ws] Error during explicit close in exception handler (likely already closing/closed): {re}")
        except Exception as e_close:
            print(f"[WS /ws] Unexpected error during explicit close in exception handler: {e_close}")
    finally:
        if authenticated_user_id: # Only disconnect if user was authenticated
            websocket_manager.disconnect(websocket)
        print(f"[WS /ws] WebSocket connection cleaned up for user: {authenticated_user_id or 'unknown'}")
        
@router.post("/users/activity/heartbeat", status_code=status.HTTP_200_OK, summary="User Activity Heartbeat")
async def user_activity_heartbeat(user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))):
    print(f"[ENDPOINT /users/activity/heartbeat] Received heartbeat for user {user_id}")
    try:
        success = await mongo_manager.update_user_last_active(user_id)
        if success:
            return JSONResponse(content={"message": "User activity timestamp updated."})
        else:
            print(f"[ERROR] Failed to update last_active_timestamp for user {user_id} via MongoManager.")
            # This could be due to user_id not existing in user_profiles or a DB issue.
            # update_user_last_active has upsert=True, so it should create if not exist.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update user activity.")
    except Exception as e:
        print(f"[ERROR] /users/activity/heartbeat for {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing heartbeat.")

@router.post("/set_data_source_enabled", status_code=status.HTTP_200_OK, summary="Enable/Disable Data Source")
async def set_data_source_enabled_endpoint(request: SetDataSourceEnabledRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))):
    # ... (existing code from your utils/routes.py) ...
    # ADD an import from server.app.app import DATA_SOURCES_CONFIG, task_queue, memory_backend, manager as websocket_manager, mongo_manager
    # Inside the try block, after successfully setting in user profile:
    try:
        # ... (your existing logic to update profile in mongo_manager.user_profiles_collection) ...
        # profile = await mongo_manager.get_user_profile(user_id)
        # if "userData" not in profile: profile["userData"] = {}
        # profile["userData"][f"{request.source}Enabled"] = request.enabled
        # success = await mongo_manager.update_user_profile(user_id, profile)
        # if not success:
        #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save data source status in user profile.")

        # NEW: Update the polling_state_store directly for is_enabled
        # and trigger re-initialization if being enabled.
        polling_state_update = {"is_enabled": request.enabled}
        if request.enabled:
            # If enabling, also set next_scheduled_poll_time to now to trigger immediate consideration by scheduler
            polling_state_update["next_scheduled_poll_time"] = datetime.datetime.now(datetime.timezone.utc)
            polling_state_update["is_currently_polling"] = False # Ensure it's not locked
            polling_state_update["error_backoff_until_timestamp"] = None # Clear any error backoff
            polling_state_update["consecutive_empty_polls_count"] = 0 # Reset empty polls
            # Optionally reset interval to default initial, or let it continue from where it was
            # polling_state_update["current_polling_interval_seconds"] = DEFAULT_INITIAL_INTERVAL_SECONDS


        success_polling_state = await mongo_manager.update_polling_state(user_id, request.source, polling_state_update)
        
        if not success_polling_state:
            print(f"[WARN] Failed to update is_enabled in polling_state_store for {user_id}/{request.source}. May need manual sync if enabling.")
        
        # If it was just enabled and state didn't exist, initialize it fully
        if request.enabled:
            from server.app.app import DATA_SOURCES_CONFIG, task_queue, memory_backend, manager, mongo_manager as global_mongo_manager # Add imports
            engine_config = DATA_SOURCES_CONFIG.get(request.source)
            if engine_config and engine_config.get("engine_class"):
                engine_class = engine_config["engine_class"]
                # Create a temporary instance just to initialize state
                temp_engine = engine_class(user_id, task_queue, memory_backend, manager, global_mongo_manager)
                await temp_engine.initialize_polling_state() # This will create if not exists and set to poll soon
                print(f"[set_data_source_enabled] Initialized polling state for {user_id}/{request.source} after enabling.")


        return JSONResponse(content={"status": "success", "message": f"Data source '{request.source}' status set to {request.enabled}."})
    except Exception as e:
        # ... (your existing error handling) ...
        print(f"[ERROR] /utils/set_data_source_enabled {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set data source status.")


@router.post("/users/force-sync/{engine_category}", status_code=status.HTTP_200_OK, summary="Force Sync for a Service")
async def force_sync_service(engine_category: str, user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))): # Assuming config write for this
    print(f"[ENDPOINT /users/force-sync] User {user_id} requests force sync for {engine_category}")
    
    # Validate engine_category
    from server.app.app import DATA_SOURCES_CONFIG # Import here to avoid circularity at module level
    if engine_category not in DATA_SOURCES_CONFIG:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid service category: {engine_category}")
        
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # Update the next_scheduled_poll_timestamp to now to trigger an immediate poll by the scheduler
        # Also ensure is_currently_polling is false so the scheduler can pick it up.
        updated = await mongo_manager.update_polling_state(
            user_id, 
            engine_category, 
            {
                "next_scheduled_poll_timestamp": now_utc,
                "is_currently_polling": False, # Ensure it's not locked
                "current_polling_tier": "forced_sync" # Optional: Mark the tier
            }
        )
        if updated:
            print(f"[FORCE_SYNC] Polling for {user_id}/{engine_category} scheduled immediately.")
            return JSONResponse(content={"message": f"Sync requested for {engine_category}. It will be processed shortly."})
        else:
            # This could happen if the polling state document doesn't exist yet (should be created by initialize_polling_state)
            # Or if there's a DB issue. For robustness, try to initialize and schedule.
            print(f"[FORCE_SYNC_WARN] Failed to directly update polling state for {user_id}/{engine_category}. Attempting initialization.")
            # Ensure the engine instance exists and its state is initialized
            from server.app.app import active_context_engines, task_queue, memory_backend, manager as ws_manager, mongo_manager as global_mongo_manager
            
            engine_instance = active_context_engines.get(user_id, {}).get(engine_category)
            if not engine_instance:
                 engine_class = DATA_SOURCES_CONFIG[engine_category]['engine_class']
                 engine_instance = engine_class(
                     user_id=user_id, task_queue=task_queue, memory_backend=memory_backend, 
                     websocket_manager=ws_manager, mongo_manager_instance=global_mongo_manager
                 )
                 if user_id not in active_context_engines: active_context_engines[user_id] = {}
                 active_context_engines[user_id][engine_category] = engine_instance
            
            await engine_instance.initialize_polling_state() # This will set next_poll_at to now if new
            # Re-attempt the update, or just rely on initialize_polling_state to have set it to now.
            await mongo_manager.update_polling_state(user_id, engine_category, {"next_scheduled_poll_timestamp": now_utc, "is_currently_polling": False, "current_polling_tier": "forced_sync_init"})

            print(f"[FORCE_SYNC] Polling for {user_id}/{engine_category} (after init) scheduled immediately.")
            return JSONResponse(content={"message": f"Sync requested for {engine_category} (initialized). It will be processed shortly."})

    except Exception as e:
        print(f"[ERROR] /users/force-sync for {user_id}/{engine_category}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing force sync request.")