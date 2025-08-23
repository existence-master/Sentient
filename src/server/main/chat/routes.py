import datetime
import uuid
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import asyncio
from typing import Tuple
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from main.chat.models import ChatMessageInput, DeleteMessageRequest # noqa: E501
from main.chat.utils import generate_chat_llm_stream # No longer need parse_assistant_response here
from main.auth.utils import PermissionChecker, AuthHelper
from main.dependencies import mongo_manager, auth_helper
from main.plans import PLAN_LIMITS

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)
logger = logging.getLogger(__name__)

@router.post("/message", summary="Process Chat Message (Overlay Chat)")
async def chat_endpoint(
    request_body: ChatMessageInput, 
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan

    if not any(msg.get("role") == "user" for msg in request_body.messages):
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    usage = await mongo_manager.get_or_create_daily_usage(user_id)
    limit = PLAN_LIMITS[plan].get("text_messages_daily", 0)
    current_count = usage.get("text_messages", 0)

    if current_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You have reached your daily message limit of {limit}. Please upgrade or try again tomorrow."
        )

    for msg in reversed(request_body.messages):
        if msg.get("role") == "assistant":
            break
        if msg.get("role") == "user":
            await mongo_manager.add_message(
                user_id=user_id,
                role="user",
                content=msg.get("content", ""),
                message_id=msg.get("id")
            )
            await mongo_manager.increment_daily_usage(user_id, "text_messages")

    clean_history_for_llm = list(reversed(await mongo_manager.get_message_history(user_id, limit=20)))

    user_profile = await mongo_manager.get_user_profile(user_id)
    user_data = user_profile.get("userData", {}) if user_profile else {}
    personal_info = user_data.get("personalInfo", {})

    user_context = {
        "name": personal_info.get("name", "User"),
        "timezone": personal_info.get("timezone", "UTC"),
    }

    async def event_stream_generator():
        # --- CHANGED --- We no longer need a buffer for parsing.
        assistant_message_id = None
        final_content_to_save = None
        turn_steps_to_save = None

        try:
            async for event in generate_chat_llm_stream(
                user_id,
                clean_history_for_llm,
                user_context,
                db_manager=mongo_manager
            ):
                if not event:
                    continue
                
                # --- CHANGED --- Logic to capture the final parsed data from the last event.
                if event.get("type") == "assistantStream":
                    if not assistant_message_id and event.get("messageId"):
                        assistant_message_id = event["messageId"]
                    
                    # When the stream is done, capture the parsed data sent from the generator.
                    if event.get("done"):
                        final_content_to_save = event.get("final_content")
                        turn_steps_to_save = event.get("turn_steps")

                yield json.dumps(event) + "\n"
        except asyncio.CancelledError:
            logger.info(f"Client disconnected, stream cancelled for user {user_id}.")
        except Exception as e:
            logger.error(f"Error in chat stream for user {user_id}: {e}")
            error_response = {
                "type": "error",
                "message": "Sorry, I encountered an error while processing your request."
            }
            yield json.dumps(error_response) + "\n"
        finally:
            # --- CHANGED --- The saving logic is now simpler and more robust.
            if final_content_to_save is not None and turn_steps_to_save is not None and assistant_message_id:
                # Note: Ensure your `add_message` function and DB schema can handle a `turn_steps` field.
                await mongo_manager.add_message(
                    user_id=user_id,
                    role="assistant",
                    content=final_content_to_save,
                    message_id=assistant_message_id,
                    turn_steps=turn_steps_to_save  # Pass the structured turn steps directly
                )
                logger.info(f"Saved parsed assistant response for user {user_id} with ID {assistant_message_id}")

    return StreamingResponse(
        event_stream_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )

@router.get("/history", summary="Get message history for a user")
async def get_chat_history(
    request: Request,
    user_id: str = Depends(PermissionChecker(required_permissions=["read:chat"]))
):
    limit = int(request.query_params.get("limit", 30))
    before_timestamp = request.query_params.get("before_timestamp")

    messages = await mongo_manager.get_message_history(user_id, limit, before_timestamp)
    return JSONResponse(content={"messages": messages[::-1]})

@router.post("/delete", summary="Delete a message or clear chat history")
async def delete_message(
    request: DeleteMessageRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:chat"]))
):
    if request.clear_all:
        deleted_count = await mongo_manager.delete_all_messages(user_id)
        return JSONResponse(content={"message": f"Successfully deleted {deleted_count} messages."})

    if request.message_id:
        success = await mongo_manager.delete_message(user_id, request.message_id)
        if success:
            return JSONResponse(content={"message": "Message deleted successfully."})
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found or you do not have permission to delete it."
            )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="You must provide either a 'message_id' or 'clear_all: true'."
    )
