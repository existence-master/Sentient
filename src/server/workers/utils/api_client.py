import httpx
import logging
import os
import motor.motor_asyncio
import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional

logger = logging.getLogger(__name__)

MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://localhost:5000")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_agent_db")

async def get_user_preferences_from_db(user_id: str):
    """Helper to fetch user preferences directly."""
    client = None
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        user_profile = await db.user_profiles.find_one(
            {"user_id": user_id},
            {"userData.preferences": 1}
        )
        return user_profile.get("userData", {}).get("preferences", {}) if user_profile else {}
    finally:
        if client:
            client.close()

async def notify_user(user_id: str, message: str, task_id: Optional[str] = None, notification_type: str = "general"):
    """
    Calls the main server to create a notification for the user, after checking preferences.
    """
    prefs = await get_user_preferences_from_db(user_id)
    if not prefs:
        logger.warning(f"Could not retrieve preferences for user {user_id}. Sending notification by default.")
    else:
        # Check Quiet Hours
        quiet_hours = prefs.get("quietHours", {})
        if quiet_hours.get("enabled"):
            try:
                user_tz = ZoneInfo(prefs.get("timezone", "UTC"))
                now_user_time = datetime.datetime.now(user_tz).time()
                start_time = datetime.time.fromisoformat(quiet_hours.get("start", "22:00"))
                end_time = datetime.time.fromisoformat(quiet_hours.get("end", "08:00"))

                # Handle overnight quiet hours
                if start_time > end_time:
                    if now_user_time >= start_time or now_user_time < end_time:
                        logger.info(f"Notification for user {user_id} suppressed due to quiet hours.")
                        return
                else: # Same day quiet hours
                    if start_time <= now_user_time < end_time:
                        logger.info(f"Notification for user {user_id} suppressed due to quiet hours.")
                        return
            except (ZoneInfoNotFoundError, ValueError) as e:
                logger.error(f"Error processing quiet hours for user {user_id}: {e}")

        # Check Notification Controls
        controls = prefs.get("notificationControls", {})
        if notification_type in controls and not controls[notification_type]:
            logger.info(f"Notification type '{notification_type}' disabled for user {user_id}. Suppressing.")
            return

    endpoint = f"{MAIN_SERVER_URL}/notifications/internal/create"
    payload = {
        "user_id": user_id,
        "message": message,
        "task_id": task_id
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully sent notification for user {user_id}: {message}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to send notification for user {user_id}. Status: {e.response.status_code}, Response: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending notification for {user_id}: {e}", exc_info=True)