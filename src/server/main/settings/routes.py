import logging
from typing import Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.responses import JSONResponse
from main.dependencies import auth_helper
from main.dependencies import mongo_manager
from main.auth.utils import PermissionChecker, AuthHelper
from main.notifications.whatsapp_client import check_phone_number_exists, send_whatsapp_message
from main.settings.models import WhatsAppMcpRequest, WhatsAppNotificationNumberRequest, ProfileUpdateRequest, WhatsAppNotificationRequest, CompleteProfileRequest, WhatsAppVerifyRequest
from main.settings.google_sheets_utils import update_onboarding_data_in_sheet, update_plan_in_sheet, check_if_contact_is_missing

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/settings",
    tags=["User Settings"]
)

@router.post("/whatsapp-mcp", summary="Connect or disconnect WhatsApp for the agent (MCP)")
async def set_whatsapp_mcp_number(
    request: WhatsAppMcpRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))
):
    whatsapp_number = request.whatsapp_mcp_number.strip() if request.whatsapp_mcp_number else ""

    if not whatsapp_number:
        update_payload = {
            "userData.integrations.whatsapp": {
                "connected": False,
                "credentials": None
            }
        }
        await mongo_manager.update_user_profile(user_id, update_payload)
        return JSONResponse(content={"message": "WhatsApp Agent disconnected."})

    try:
        validation_result = await check_phone_number_exists(whatsapp_number)
        if not validation_result or not validation_result.get("numberExists"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This phone number does not appear to be on WhatsApp.")

        chat_id = validation_result.get("chatId")
        if not chat_id:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not retrieve Chat ID for the number.")

        # For MCP, we store credentials under integrations
        update_payload = {
            "userData.integrations.whatsapp": {
                "connected": True,
                "auth_type": "manual_config",
                "credentials": { # Store unencrypted, as it's just a number/ID
                    "number": whatsapp_number,
                    "chatId": chat_id
                }
            }
        }
        await mongo_manager.update_user_profile(user_id, update_payload)
        return JSONResponse(content={"message": "WhatsApp Agent connected successfully."})

    except ConnectionError as e:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to WhatsApp service: {e}")
    except Exception as e:
        logger.error(f"Error setting WhatsApp MCP number for user {user_id}: {e}", exc_info=True)
        # Re-raise HTTPException or handle others
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


@router.get("/whatsapp-mcp", summary="Get WhatsApp MCP connection status")
async def get_whatsapp_mcp_number(
    user_id: str = Depends(PermissionChecker(required_permissions=["read:config"]))
):
    user_profile = await mongo_manager.get_user_profile(user_id)
    if not user_profile:
        return JSONResponse(content={"whatsapp_mcp_number": "", "connected": False})

    wa_integration = user_profile.get("userData", {}).get("integrations", {}).get("whatsapp", {})
    return JSONResponse(content={
        "whatsapp_mcp_number": wa_integration.get("credentials", {}).get("number", ""),
        "connected": wa_integration.get("connected", False)
    })


@router.post("/whatsapp-notifications", summary="Set or Update WhatsApp number for notifications")
async def set_whatsapp_notification_number(
    request: WhatsAppNotificationNumberRequest,
    payload: dict = Depends(auth_helper.get_decoded_payload_with_claims)
):
    # 1. First, ensure the user has the required permission from the token payload
    required_permission = "write:config"
    if required_permission not in payload.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required permission: {required_permission}"
        )

    # 2. Extract user_id and email directly from the validated token payload
    user_id = payload.get("sub")
    user_email = payload.get("email") # This is the reliable email from Auth0    
    
    whatsapp_number = request.whatsapp_notifications_number.strip() if request.whatsapp_notifications_number else ""
    if not whatsapp_number:
        # If the user clears the number, disable notifications and clear the fields.
        update_payload = {
            "userData.notificationPreferences.whatsapp.number": "",
            "userData.notificationPreferences.whatsapp.chatId": "",
            "userData.notificationPreferences.whatsapp.enabled": False,
        }
        await mongo_manager.update_user_profile(user_id, update_payload)
        return JSONResponse(content={"message": "WhatsApp notification number removed."})

    try:
        validation_result = await check_phone_number_exists(whatsapp_number)
        if not validation_result or not validation_result.get("numberExists"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This phone number does not appear to be on WhatsApp.")

        chat_id = validation_result.get("chatId")
        if not chat_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not retrieve Chat ID for the number.")

        update_payload = {
            "userData.notificationPreferences.whatsapp.number": whatsapp_number,
            "userData.notificationPreferences.whatsapp.chatId": chat_id,
            "userData.notificationPreferences.whatsapp.enabled": True, # Always enable when number is set/updated
        }
        await mongo_manager.update_user_profile(user_id, update_payload)
        
        return JSONResponse(content={"message": "WhatsApp notification number updated successfully."})

    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to WhatsApp service: {e}")
    except Exception as e:
        logger.error(f"Error setting WhatsApp notification number for user {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

@router.post("/whatsapp-notifications/verify", summary="Verify if a WhatsApp number exists")
async def verify_whatsapp_notification_number(
    request: WhatsAppVerifyRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    phone_number = request.phone_number
    if not phone_number:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="phone_number is required.")

    try:
        validation_result = await check_phone_number_exists(phone_number)
        if validation_result is None:
             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Could not connect to WhatsApp service to verify number.")

        return validation_result
    except Exception as e:
        logger.error(f"Error verifying WhatsApp number for user {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/whatsapp-notifications", summary="Get WhatsApp Notification settings")
async def get_whatsapp_notification_settings(
    user_id: str = Depends(PermissionChecker(required_permissions=["read:config"]))
):
    user_profile = await mongo_manager.get_user_profile(user_id)
    if not user_profile:
        return JSONResponse(content={"whatsapp_notifications_number": "", "notifications_enabled": False})

    wa_prefs = user_profile.get("userData", {}).get("notificationPreferences", {}).get("whatsapp", {})
    return JSONResponse(content={
        "whatsapp_notifications_number": wa_prefs.get("number", ""),
        "notifications_enabled": wa_prefs.get("enabled", False)
    })

@router.post("/whatsapp-notifications/toggle", summary="Toggle WhatsApp notifications on/off")
async def toggle_whatsapp_notifications(
    request: WhatsAppNotificationRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:config"]))
):
    """
    Updates the enabled status of WhatsApp notifications for the user.
    """
    try:
        # Check if a number is configured before allowing enable
        if request.enabled:
            user_profile = await mongo_manager.get_user_profile(user_id)
            wa_prefs = user_profile.get("userData", {}).get("notificationPreferences", {}).get("whatsapp", {})
            if not wa_prefs.get("number") or not wa_prefs.get("chatId"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot enable notifications without a configured phone number."
                )

        update_payload = {"userData.notificationPreferences.whatsapp.enabled": request.enabled}
        success = await mongo_manager.update_user_profile(user_id, update_payload)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update notification preference.")
        
        status_text = "enabled" if request.enabled else "disabled"
        return JSONResponse(content={"message": f"WhatsApp notifications {status_text}."})
    except Exception as e:
        logger.error(f"Error toggling WhatsApp notifications for user {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@router.post("/complete-profile", summary="Complete profile for existing users with missing data")
async def complete_profile(
    request: CompleteProfileRequest,
    payload: dict = Depends(auth_helper.get_decoded_payload_with_claims)
):
    user_id = payload.get("sub")
    user_email = payload.get("email")
    plan = payload.get("plan", "free")

    try:
        # 1. Validate WhatsApp number
        whatsapp_number = request.whatsapp_notifications_number.strip()
        validation_result = await check_phone_number_exists(whatsapp_number)
        if not validation_result or not validation_result.get("numberExists"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This phone number does not appear to be on WhatsApp.")
        chat_id = validation_result.get("chatId")
        if not chat_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not retrieve Chat ID for the number.")

        # 2. Fetch existing onboarding data from MongoDB
        user_profile = await mongo_manager.get_user_profile(user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found.")

        existing_onboarding_data = user_profile.get("userData", {}).get("onboardingAnswers", {})

        # 3. Consolidate old and new data
        consolidated_data = {
            **existing_onboarding_data,
            "needs-pa": request.needs_pa,
            "whatsapp_notifications_number": whatsapp_number
        }

        # 4. Update MongoDB with new answers and notification prefs
        update_payload = {
            "userData.onboardingAnswers.needs-pa": request.needs_pa,
            "userData.onboardingAnswers.whatsapp_notifications_number": whatsapp_number,
            "userData.notificationPreferences.whatsapp": {
                "number": whatsapp_number,
                "chatId": chat_id,
                "enabled": True
            }
        }
        await mongo_manager.update_user_profile(user_id, update_payload)

        # 5. Update Google Sheet with the full consolidated data
        if user_email:
            await update_onboarding_data_in_sheet(user_email, consolidated_data, plan)
        else:
            logger.warning(f"Could not update GSheet for {user_id} during profile completion: no email.")

        # 6. Send conditional WhatsApp message
        if request.needs_pa == "yes":
            feedback_message = "Hi there, I am Sarthak from team Sentient. Thanks for using our app, are you okay with giving feedback and helping us improve the platform to better suit your needs?"
            await send_whatsapp_message(chat_id, feedback_message)
            logger.info(f"Sent feedback request to existing user {user_id} after profile completion.")

        return JSONResponse(content={"message": "Profile completed successfully."})

    except Exception as e:
        logger.error(f"Error completing profile for user {user_id}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@router.post("/profile", summary="Update User Profile and Onboarding Data")
async def update_profile_data(
    request: ProfileUpdateRequest,
    user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))
):
    """
    Updates the user's profile, including personal info, preferences,
    and the original onboarding answers.
    """
    try:
        update_payload = {
            "userData.onboardingAnswers": request.onboardingAnswers,
            "userData.personalInfo": request.personalInfo,
            "userData.preferences": request.preferences,
        }
        success = await mongo_manager.update_user_profile(user_id, update_payload)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update profile.")
        return JSONResponse(content={"message": "Profile updated successfully."})
    except Exception as e:
        logger.error(f"Error updating profile for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")