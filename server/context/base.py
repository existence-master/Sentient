# src/server/context/base.py
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import asyncio
import time 
import random 
import json
from server.db.mongo_manager import MongoManager # Ensure this is your updated MongoManager
from typing import Optional, Dict, Any, List, Tuple
import os # For os.getenv

# --- Polling Configuration (Loaded from .env) ---
DEFAULT_INITIAL_INTERVAL_SECONDS = int(os.getenv("DEFAULT_INITIAL_INTERVAL_SECONDS", "900"))
MIN_POLLING_INTERVAL_SECONDS = int(os.getenv("MIN_POLLING_INTERVAL_SECONDS", "300"))
MAX_POLLING_INTERVAL_SECONDS = int(os.getenv("MAX_POLLING_INTERVAL_SECONDS", "86400"))
ACTIVITY_FACTOR = float(os.getenv("ACTIVITY_FACTOR", "2.0"))
INACTIVITY_THRESHOLD_POLLS = int(os.getenv("INACTIVITY_THRESHOLD_POLLS", "3"))
INACTIVITY_FACTOR = float(os.getenv("INACTIVITY_FACTOR", "1.5"))
RATE_LIMIT_BACKOFF_FACTOR = float(os.getenv("RATE_LIMIT_BACKOFF_FACTOR", "3.0"))
MAX_POLL_INTERVAL_RATE_LIMIT_SECONDS = int(os.getenv("MAX_POLL_INTERVAL_RATE_LIMIT_SECONDS", "86400"))
TRANSIENT_ERROR_BACKOFF_SECONDS = int(os.getenv("TRANSIENT_ERROR_BACKOFF_SECONDS", "900"))
POLLING_JITTER_SECONDS = int(os.getenv("POLLING_JITTER_SECONDS", "30")) 

# User activity thresholds (can also be from config)
ACTIVE_USER_SECONDS = int(os.getenv("ACTIVE_USER_SECONDS", "60")) # For highly active users
RECENTLY_ACTIVE_SECONDS = int(os.getenv("RECENTLY_ACTIVE_SECONDS", "900")) # 15 mins
PEAK_HOURS_SECONDS = int(os.getenv("PEAK_HOURS_SECONDS", "1800")) # 30 mins
OFF_PEAK_SECONDS = int(os.getenv("OFF_PEAK_SECONDS", "7200")) # 2 hours

ACTIVE_THRESHOLD_MINUTES = int(os.getenv("ACTIVE_THRESHOLD_MINUTES", "15"))
RECENTLY_ACTIVE_THRESHOLD_HOURS = int(os.getenv("RECENTLY_ACTIVE_THRESHOLD_HOURS", "2"))
PEAK_HOURS_START = int(os.getenv("PEAK_HOURS_START", "8")) # User's local time
PEAK_HOURS_END = int(os.getenv("PEAK_HOURS_END", "22")) # User's local time


class BaseContextEngine(ABC):
    def __init__(self, user_id: str, task_queue: Any, memory_backend: Any, 
                 websocket_manager: Any, mongo_manager_instance: MongoManager):
        self.user_id: str = user_id
        self.task_queue = task_queue 
        self.memory_backend = memory_backend 
        self.websocket_manager = websocket_manager 
        self.mongo_manager = mongo_manager_instance
        self.category: Optional[str] = None 
        print(f"[BaseContextEngine] Initialized for user {self.user_id}, category: {getattr(self, 'category', 'NOT_SET')}")

    async def initialize_polling_state(self):
        """Ensure an initial polling state exists for this user/engine."""
        if not self.category:
            print(f"[POLLING_INIT_ERROR] Category not set for engine for user {self.user_id}.")
            return

        existing_state = await self.mongo_manager.get_polling_state(self.user_id, self.category)
        if not existing_state:
            print(f"[POLLING_INIT] No polling state for {self.user_id}/{self.category}. Creating initial state.")
            
            random_offset_seconds = random.randint(0, MIN_POLLING_INTERVAL_SECONDS // 2)
            initial_next_poll_time = datetime.now(timezone.utc) + timedelta(seconds=random_offset_seconds)
            
            initial_state = {
                "user_id": self.user_id,
                "service_name": self.category,
                "last_polled_timestamp": None,
                "last_processed_item_id": None,
                "next_page_token": None,
                "last_successful_poll_status_message": "Polling state initialized.",
                "next_scheduled_poll_time": initial_next_poll_time,
                "current_polling_interval_seconds": DEFAULT_INITIAL_INTERVAL_SECONDS,
                "min_polling_interval_seconds": MIN_POLLING_INTERVAL_SECONDS,
                "max_polling_interval_seconds": MAX_POLLING_INTERVAL_SECONDS,
                "consecutive_empty_polls_count": 0,
                "last_activity_timestamp": None,
                "is_rate_limited": False,
                "error_backoff_until_timestamp": None,
                "is_enabled": True, 
                "is_currently_polling": False, # Ensure it's not locked initially
                "last_attempted_poll_timestamp": None, # Track when a poll cycle starts
                "created_at": datetime.now(timezone.utc),
            }
            await self.mongo_manager.update_polling_state(self.user_id, self.category, initial_state)
            print(f"[POLLING_INIT] Initial polling state created for {self.user_id}/{self.category}, next poll at {initial_next_poll_time.isoformat()}.")
        else:
            if existing_state.get("is_currently_polling", False):
                print(f"[POLLING_INIT_WARN] Polling lock was stale for {self.user_id}/{self.category}. Resetting.")
                await self.mongo_manager.update_polling_state(self.user_id, self.category, {"is_currently_polling": False})
            
            # Ensure all necessary fields from the plan exist, add if missing
            # This helps migrate older states if new fields are added to the schema
            updated_fields = {}
            default_fields_for_check = {
                "min_polling_interval_seconds": MIN_POLLING_INTERVAL_SECONDS,
                "max_polling_interval_seconds": MAX_POLLING_INTERVAL_SECONDS,
                "is_enabled": existing_state.get("is_enabled", True), # Keep existing or default to true
                "is_currently_polling": False # Always reset if it was true
            }
            for key, default_val in default_fields_for_check.items():
                if key not in existing_state:
                    updated_fields[key] = default_val
            if "service_name" not in existing_state and self.category: # Ensure service_name
                updated_fields["service_name"] = self.category

            if updated_fields:
                print(f"[POLLING_INIT_MIGRATE] Adding missing fields to polling state for {self.user_id}/{self.category}: {updated_fields}")
                await self.mongo_manager.update_polling_state(self.user_id, self.category, updated_fields)

            print(f"[POLLING_INIT] Polling state already exists for {self.user_id}/{self.category}.")
            if not existing_state.get("is_enabled", True):
                 print(f"[POLLING_INIT] Service {self.category} is disabled for {self.user_id}. No polling will occur until enabled.")

    async def run_poll_cycle(self):
        if not self.category:
            print(f"[POLL_CYCLE_ERROR] Category not set for engine user {self.user_id}")
            return

        print(f"[POLL_CYCLE] Starting poll for {self.user_id}/{self.category}")
        
        polling_state = await self.mongo_manager.get_polling_state(self.user_id, self.category)
        if not polling_state:
            print(f"[POLL_CYCLE_ERROR] Polling state missing for {self.user_id}/{self.category}. Attempting re-initialization.")
            await self.initialize_polling_state()
            await self.mongo_manager.update_polling_state(self.user_id, self.category, {
                "is_currently_polling": False, # Release hypothetical lock
                "next_scheduled_poll_time": datetime.now(timezone.utc) 
            })
            return
        
        if not polling_state.get("is_enabled", False): # Check if enabled
            print(f"[POLL_CYCLE_SKIP] Service {self.category} is disabled for user {self.user_id}. Skipping poll.")
            # Schedule next poll far out or based on a 'disabled' interval if desired
            await self.mongo_manager.update_polling_state(self.user_id, self.category, {
                "is_currently_polling": False,
                "next_scheduled_poll_time": datetime.now(timezone.utc) + timedelta(hours=MAX_POLLING_INTERVAL_SECONDS / 3600)
            })
            return

        last_marker_from_db = polling_state.get("last_processed_item_id")
        next_page_token_from_db = polling_state.get("next_page_token")

        new_data_found_in_cycle = False
        api_error_occurred = False
        rate_limited_in_cycle = False
        new_last_marker_for_db = last_marker_from_db # Start with old, update on success
        
        try:
            print(f"[POLL_CYCLE] Fetching data for {self.user_id}/{self.category} (Marker: {last_marker_from_db}, PageToken: {next_page_token_from_db})")
            fetch_result = await self.fetch_new_data(last_marker_from_db, next_page_token_from_db)
            
            if fetch_result is None: 
                print(f"[POLL_CYCLE_API_ERROR] API error during fetch_new_data for {self.user_id}/{self.category}.")
                api_error_occurred = True
            else:
                new_data_payload, new_last_marker_from_fetch, new_next_page_token_for_db = fetch_result

                if new_data_payload: 
                    new_data_found_in_cycle = True
                    print(f"[POLL_CYCLE] Processing {len(new_data_payload)} new items for {self.user_id}/{self.category}")
                    await self.process_and_publish_data(new_data_payload) 
                    new_last_marker_for_db = new_last_marker_from_fetch 
                    print(f"[POLL_CYCLE] Data processed. New marker: {new_last_marker_for_db}. Next page token: {new_next_page_token_for_db}")
                    await self.mongo_manager.update_polling_state(
                        self.user_id, self.category, {"next_page_token": new_next_page_token_for_db}
                    )
                else: # No new data in this specific fetch
                    print(f"[POLL_CYCLE] No new data returned by fetch for {self.user_id}/{self.category}.")
                    if next_page_token_from_db: # If we were paginating and got no data, clear token
                         await self.mongo_manager.update_polling_state(self.user_id, self.category, {"next_page_token": None})
        
        except httpx.HTTPStatusError as e: 
            print(f"[POLL_CYCLE_HTTP_ERROR] HTTP error for {self.user_id}/{self.category}: {e.response.status_code} - {e.response.text}")
            api_error_occurred = True
            status_message_on_error = f"API Client Error {e.response.status_code}."
            if e.response.status_code == 429: 
                rate_limited_in_cycle = True
                retry_after_seconds = int(e.response.headers.get("Retry-After", str(TRANSIENT_ERROR_BACKOFF_SECONDS)))
                error_backoff_ts = datetime.now(timezone.utc) + timedelta(seconds=retry_after_seconds)
                status_message_on_error = f"Rate limited. Retry after {retry_after_seconds}s."
                await self.mongo_manager.update_polling_state(
                    self.user_id, self.category, 
                    {"is_rate_limited": True, "error_backoff_until_timestamp": error_backoff_ts, "last_successful_poll_status_message": status_message_on_error }
                )
            elif e.response.status_code >= 500: 
                 error_backoff_ts = datetime.now(timezone.utc) + timedelta(seconds=TRANSIENT_ERROR_BACKOFF_SECONDS)
                 status_message_on_error = f"Server error {e.response.status_code}."
                 await self.mongo_manager.update_polling_state(
                    self.user_id, self.category, 
                    {"error_backoff_until_timestamp": error_backoff_ts, "last_successful_poll_status_message": status_message_on_error}
                )
            else: # Other client error
                await self.mongo_manager.update_polling_state(
                    self.user_id, self.category,
                    {"last_successful_poll_status_message": status_message_on_error}
                )
        except Exception as e:
            print(f"[POLL_CYCLE_ERROR] General error during poll for {self.user_id}/{self.category}: {e}")
            traceback.print_exc()
            api_error_occurred = True
            error_backoff_ts = datetime.now(timezone.utc) + timedelta(seconds=TRANSIENT_ERROR_BACKOFF_SECONDS)
            await self.mongo_manager.update_polling_state(
                self.user_id, self.category,
                {"error_backoff_until_timestamp": error_backoff_ts,
                 "last_successful_poll_status_message": f"Polling cycle error: {str(e)[:100]}"}
            )
        finally:
            print(f"[POLL_CYCLE] Finalizing poll for {self.user_id}/{self.category}. API Success: {not api_error_occurred}, New Data Found: {new_data_found_in_cycle}, Rate Limited: {rate_limited_in_cycle}")
            await self.calculate_and_schedule_next_poll(
                api_call_succeeded=(not api_error_occurred), 
                data_found_this_cycle=new_data_found_in_cycle, 
                is_rate_limited_this_cycle=rate_limited_in_cycle,
                processed_marker_this_cycle=new_last_marker_for_db # Pass marker from *this* successful poll
            )
            print(f"[POLL_CYCLE] Poll cycle finished for {self.user_id}/{self.category}")

    async def calculate_and_schedule_next_poll(self, api_call_succeeded: bool, data_found_this_cycle: bool, 
                                               is_rate_limited_this_cycle: bool, processed_marker_this_cycle: Optional[str]):
        if not self.category:
            print(f"[SCHEDULE_NEXT_ERROR] Category not set for user {self.user_id}")
            return

        polling_state = await self.mongo_manager.get_polling_state(self.user_id, self.category)
        if not polling_state:
            print(f"[SCHEDULE_NEXT_ERROR] No polling state found for {self.user_id}/{self.category}. Cannot schedule.")
            return

        # Get current values from DB state, with defaults from constants file
        base_interval = polling_state.get("current_polling_interval_seconds", DEFAULT_INITIAL_INTERVAL_SECONDS)
        consecutive_empty = polling_state.get("consecutive_empty_polls_count", 0)
        
        status_message = "Scheduled."
        tier = "default"
        error_backoff_ts = polling_state.get("error_backoff_until_timestamp") # Might have been set by error handling

        # 1. Handle API Errors (rate limit or other) - These dictate immediate next step
        if is_rate_limited_this_cycle:
            # Interval increases, backoff_ts is already set by error handler
            next_interval_seconds = min(int(base_interval * RATE_LIMIT_BACKOFF_FACTOR), MAX_POLL_INTERVAL_RATE_LIMIT_SECONDS)
            tier = "rate_limited_backoff"
            status_message = f"Rate limited. Interval increased to ~{next_interval_seconds/60:.1f}m."
            consecutive_empty = 0 # Reset empty counter on error
        elif not api_call_succeeded:
            # Other API error, interval might not change, backoff_ts is set by error handler
            next_interval_seconds = base_interval 
            tier = "api_error_backoff"
            status_message = f"API error. Interval: ~{next_interval_seconds/60:.1f}m. Will retry after backoff."
            consecutive_empty = 0 # Reset empty counter on error
        else: # API call was successful
            # Clear previous error states
            error_backoff_ts = None 
            await self.mongo_manager.update_polling_state(self.user_id, self.category, {"is_rate_limited": False, "error_backoff_until_timestamp": None})
            
            # 2. Determine base interval from user activity and timezone
            user_activity = await self.mongo_manager.get_user_activity_and_timezone(self.user_id)
            last_active_ts_obj = user_activity.get("last_active_timestamp")
            user_timezone_str = user_activity.get("timezone")
            now_utc = datetime.now(timezone.utc)

            is_highly_active = last_active_ts_obj and (now_utc - last_active_ts_obj) < timedelta(minutes=ACTIVE_THRESHOLD_MINUTES)
            is_recently_active = last_active_ts_obj and (now_utc - last_active_ts_obj) < timedelta(hours=RECENTLY_ACTIVE_THRESHOLD_HOURS)

            if is_highly_active:
                target_interval_from_tier = ACTIVE_USER_SECONDS
                tier = "user_highly_active"
            elif is_recently_active:
                target_interval_from_tier = RECENTLY_ACTIVE_SECONDS
                tier = "user_recently_active"
            else: # Less active, apply timezone
                current_hour_user_tz = now_utc.hour # Default to UTC hour
                if user_timezone_str:
                    try:
                        # Use zoneinfo if available (Python 3.9+)
                        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
                        try:
                            user_tz = ZoneInfo(user_timezone_str)
                            current_hour_user_tz = now_utc.astimezone(user_tz).hour
                        except ZoneInfoNotFoundError:
                             print(f"[WARN] ZoneInfo not found for '{user_timezone_str}'. Trying pytz.")
                             import pytz # Fallback for older Python or if zoneinfo db is minimal
                             user_tz = pytz.timezone(user_timezone_str)
                             current_hour_user_tz = now_utc.astimezone(user_tz).hour
                    except ImportError: # Fallback for < Python 3.9 or if zoneinfo not available
                        try:
                            import pytz
                            user_tz = pytz.timezone(user_timezone_str)
                            current_hour_user_tz = now_utc.astimezone(user_tz).hour
                        except Exception as tz_err:
                            print(f"[WARN] Invalid timezone '{user_timezone_str}' or pytz not installed for user {self.user_id}. Defaulting to UTC. Error: {tz_err}")
                    except Exception as e_tz:
                         print(f"[WARN] Error processing timezone {user_timezone_str} for user {self.user_id}: {e_tz}. Defaulting to UTC.")
                
                if PEAK_HOURS_START <= current_hour_user_tz < PEAK_HOURS_END:
                    target_interval_from_tier = PEAK_HOURS_SECONDS
                    tier = "less_active_peak"
                else:
                    target_interval_from_tier = OFF_PEAK_SECONDS
                    tier = "less_active_off_peak"
            
            # 3. Adjust the target_interval_from_tier based on data_found
            if data_found_this_cycle:
                next_interval_seconds = max(MIN_POLLING_INTERVAL_SECONDS, int(target_interval_from_tier / ACTIVITY_FACTOR))
                consecutive_empty = 0 
                status_message = f"New data found ({tier}). Freq. increased to ~{next_interval_seconds/60:.1f}m."
                await self.mongo_manager.update_polling_state(self.user_id, self.category, {"last_activity_timestamp": datetime.now(timezone.utc)})
            else: # Success, but no new data
                consecutive_empty += 1
                if consecutive_empty >= INACTIVITY_THRESHOLD_POLLS:
                    next_interval_seconds = min(MAX_POLLING_INTERVAL_SECONDS, int(target_interval_from_tier * INACTIVITY_FACTOR))
                    consecutive_empty = 0 
                    status_message = f"No new data ({tier}, {INACTIVITY_THRESHOLD_POLLS-1} empty before this). Freq. decreased to ~{next_interval_seconds/60:.1f}m."
                else:
                    next_interval_seconds = target_interval_from_tier # Keep base tier interval
                    status_message = f"No new data ({tier}, {consecutive_empty} empty polls)."

        # 4. Apply global min/max bounds and calculate next poll time with jitter
        final_interval_seconds = int(max(MIN_POLLING_INTERVAL_SECONDS, min(next_interval_seconds, MAX_POLLING_INTERVAL_SECONDS)))
        jitter = random.randint(-POLLING_JITTER_SECONDS, POLLING_JITTER_SECONDS)
        
        effective_next_poll_time = datetime.now(timezone.utc) + timedelta(seconds=final_interval_seconds + jitter)
        
        if error_backoff_ts:
            if isinstance(error_backoff_ts, str):
                error_backoff_ts = datetime.fromisoformat(error_backoff_ts.replace("Z", "+00:00"))
            if error_backoff_ts.tzinfo is None: error_backoff_ts = error_backoff_ts.replace(tzinfo=timezone.utc)
            
            if effective_next_poll_time < error_backoff_ts:
                effective_next_poll_time = error_backoff_ts + timedelta(seconds=random.randint(0, POLLING_JITTER_SECONDS)) # Add positive jitter after backoff
                status_message += f" Next poll respects error_backoff until {error_backoff_ts.isoformat()}."

        # 5. Prepare and save update to MongoDB
        update_data: Dict[str, Any] = {
            "next_scheduled_poll_time": effective_next_poll_time,
            "current_polling_interval_seconds": final_interval_seconds,
            "consecutive_empty_polls_count": consecutive_empty,
            "last_successful_poll_status_message": status_message,
            "is_currently_polling": False, # Release the lock
            "current_polling_tier": tier, # Store the determined tier
        }
        if api_call_succeeded: # Only update these on a successful API call
            update_data["last_polled_timestamp"] = datetime.now(timezone.utc)
            if processed_marker_this_cycle is not None:
                update_data["last_processed_item_id"] = processed_marker_this_cycle
        
        await self.mongo_manager.update_polling_state(self.user_id, self.category, update_data)
        print(f"[SCHEDULE_NEXT] User {self.user_id}/{self.category} - Tier: {tier}. Next poll: {effective_next_poll_time.isoformat()} (Interval: {final_interval_seconds}s, Jitter: {jitter}s). Status: {status_message}")

    @abstractmethod
    async def fetch_new_data(self, last_marker: Optional[str], page_token: Optional[str]) -> Optional[Tuple[Optional[List[Any]], Optional[str], Optional[str]]]:
        # ... (no change) ...
        pass

    @abstractmethod
    async def process_and_publish_data(self, new_data_payload: List[Any]):
        # ... (no change) ...
        pass

    @abstractmethod
    async def get_runnable(self):
        # ... (no change) ...
        pass

    async def execute_outputs(self, output):
        # ... (no change, likely simplified or removed for polling) ...
        if not self.category:
            print("[BASE_CONTEXT_ENGINE_ERROR] Category not set. Cannot execute outputs.")
            return
        print(f"BaseContextEngine.execute_outputs for {self.category}: {output if output else 'No output from runnable.'}")


    async def get_chat_history(self):
        # ... (no change) ...
        print(f"BaseContextEngine.get_chat_history started for user {self.user_id}")
        user_profile = await self.mongo_manager.get_user_profile(self.user_id)
        active_chat_id = None
        if user_profile and "userData" in user_profile and "active_chat_id" in user_profile["userData"]:
            active_chat_id = user_profile["userData"]["active_chat_id"]

        if not active_chat_id:
            print(f"BaseContextEngine.get_chat_history - no active chat ID found for user {self.user_id}, returning empty list")
            return []

        print(f"BaseContextEngine.get_chat_history - active_chat_id: {active_chat_id} for user_id: {self.user_id}")
        history = await self.mongo_manager.get_chat_history(self.user_id, active_chat_id)
        last_10_messages = history[-10:] if history else []
        print(f"BaseContextEngine.get_chat_history - returning last {len(last_10_messages)} messages.")
        print("BaseContextEngine.get_chat_history finished")
        return last_10_messages