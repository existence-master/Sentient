import asyncio
import logging
import uuid
import json
import re
import datetime
import os
import httpx
from dateutil import rrule
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Dict, Any, Optional, List, Tuple
from bson import ObjectId
from celery import group, chord
from main.analytics import capture_event
from json_extractor import JsonExtractor
from workers.utils.api_client import notify_user, push_task_list_update
from main.plans import PLAN_LIMITS
from main.config import INTEGRATIONS_CONFIG
from main.tasks.prompts import TASK_CREATION_PROMPT
from mcp_hub.memory.utils import initialize_embedding_model, initialize_agents, cud_memory
from main.llm import run_agent as run_main_agent, LLMProviderDownError
from main.db import MongoManager
from workers.celery_app import celery_app
from workers.planner.llm import get_planner_agent
from workers.planner.db import PlannerMongoManager, get_all_mcp_descriptions
from workers.executor.tasks import execute_task_plan, run_single_item_worker, aggregate_results_callback
from main.vector_db import get_conversation_summaries_collection
from mcp_hub.tasks.prompts import ITEM_EXTRACTOR_SYSTEM_PROMPT, RESOURCE_MANAGER_SYSTEM_PROMPT
from workers.utils.text_utils import clean_llm_output

# Imports for poller logic
from workers.poller.gmail.service import GmailPollingService
from workers.poller.gcalendar.service import GCalendarPollingService
from workers.poller.gmail.db import PollerMongoManager as GmailPollerDB
from workers.poller.gcalendar.db import PollerMongoManager as GCalPollerDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_date_from_text(text: str) -> str:
    """Extracts YYYY-MM-DD from text, defaults to today."""
    match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if match:
        return match.group(1)
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')

# Helper to run async code in Celery's sync context
def run_async(coro):
    # Always create a new loop for each task to ensure isolation and prevent conflicts.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Ensure the connection pool for this specific loop is closed.
        from mcp_hub.memory.db import close_db_pool_for_loop
        loop.run_until_complete(close_db_pool_for_loop(loop))
        loop.close()
        asyncio.set_event_loop(None)

async def async_cud_memory_task(user_id: str, information: str, source: Optional[str] = None):
    """The async logic for the CUD memory task."""
    db_manager = MongoManager()
    username = user_id  # Default fallback
    try:
        # --- Enforce Memory Limit ---
        user_profile = await db_manager.get_user_profile(user_id)
        plan = user_profile.get("userData", {}).get("plan", "free") if user_profile else "free"
        limit = PLAN_LIMITS[plan].get("memories_total", 0)

        if limit != float('inf'):
            from mcp_hub.memory import db as memory_db
            pool = await memory_db.get_db_pool()
            async with pool.acquire() as conn:
                current_count = await conn.fetchval("SELECT COUNT(*) FROM facts WHERE user_id = $1", user_id)
            if current_count >= limit:
                logger.warning(f"User {user_id} on '{plan}' plan reached memory limit of {limit}. CUD operation aborted.")
                await notify_user(user_id, f"You've reached your memory limit of {limit} facts. Please upgrade to Pro for unlimited memories.")
                return

        # --- Fetch user's name before calling cud_memory ---
        if user_profile:
            # Use the name from personalInfo, which is set during onboarding and can be updated in settings.
            username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", user_id)

    except Exception as e:
        logger.error(f"Error during pre-CUD setup for user {user_id}: {e}", exc_info=True)
        # We can still proceed with the CUD operation, just using the user_id as the name.
    finally:
        await db_manager.close()

    # Initialize models required for the CUD operation
    initialize_embedding_model()
    initialize_agents()
    # Pass the fetched username to the cud_memory function
    await cud_memory(user_id, information, source, username)

@celery_app.task(name="cud_memory_task")
def cud_memory_task(user_id: str, information: str, source: Optional[str] = None):
    """
    Celery task wrapper for the CUD (Create, Update, Delete) memory operation.
    This runs the core memory management logic asynchronously.
    """
    logger.info(f"Celery worker received cud_memory_task for user_id: {user_id}")
    # this single call to run_async wraps the entire asynchronous logic,
    # ensuring the event loop and DB connections are managed correctly for the task's lifecycle.
    run_async(async_cud_memory_task(user_id, information, source))

@celery_app.task(name="start_long_form_task")
def start_long_form_task(task_id: str, user_id: str):
    """
    Celery task entry point for initializing a new long-form task.
    This will eventually become the orchestrator's first step.
    """
    logger.info(f"Celery worker received 'start_long_form_task' for task_id: {task_id}")
    run_async(async_start_long_form_task(task_id, user_id))

async def async_start_long_form_task(task_id: str, user_id: str):
    """
    The async logic for initializing a long-form task.
    For Phase 1, it just logs and updates the state.
    """
    db_manager = MongoManager()
    try:
        logger.info(f"Starting long-form task {task_id} for user {user_id}. This is the initial step of the orchestrator.")

        initial_log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "action": "Task Initiated",
            "details": {"message": "Orchestrator has started and is in the initial planning phase."},
            "agent_reasoning": "The task has been created. The first step is to analyze the main goal and create an initial high-level plan."
        }

        update_payload = {
            "long_form_details.current_state": "ACTIVE",
            "long_form_details.execution_log": [initial_log_entry]
        }
        await db_manager.update_task(task_id, update_payload)
        await push_task_list_update(user_id, task_id, "long_form_started")
        logger.info(f"Long-form task {task_id} moved to ACTIVE state.")
    except Exception as e:
        logger.error(f"Error starting long-form task {task_id}: {e}", exc_info=True)
        await db_manager.update_task(task_id, {
            "long_form_details.current_state": "FAILED",
            "error": f"Failed to start orchestrator: {str(e)}"
        })
    finally:
        await db_manager.close()

@celery_app.task(name="orchestrate_swarm_task")
def orchestrate_swarm_task(task_id: str, user_id: str):
    """
    Celery task entry point for orchestrating a swarm task.
    """
    logger.info(f"Celery worker received task 'orchestrate_swarm_task' for task_id: {task_id}")
    run_async(async_orchestrate_swarm_task(task_id, user_id))

async def async_orchestrate_swarm_task(task_id: str, user_id: str):
    """
    The main orchestration logic for a swarm task.
    1. Fetches the task.
    2. Invokes the Resource Manager agent to get an execution plan.
    3. Updates the task with the plan.
    4. Dispatches a Celery chord of sub-agent workers.
    """
    db_manager = MongoManager()
    try:
        task = await db_manager.get_task(task_id, user_id)
        if not task or task.get("task_type") != "swarm":
            logger.error(f"Orchestrator: Task {task_id} not found or is not a swarm task.")
            return

        # --- Get user plan to check limits ---
        user_profile = await db_manager.get_user_profile(user_id)
        plan = user_profile.get("userData", {}).get("plan", "free") if user_profile else "free"
        sub_agent_limit = PLAN_LIMITS[plan].get("swarm_sub_agents_max", 10)

        swarm_details = task.get("swarm_details", {})
        goal = swarm_details.get("goal")
        items = swarm_details.get("items")

        # --- NEW: Item Extraction Step ---
        if not items: # If items list is empty, try to extract from goal
            logger.info(f"Task {task_id}: Items list is empty. Attempting to extract items from goal.")

            if not goal:
                 raise ValueError("Swarm task is missing a goal to extract items from.")

            extractor_response_str = ""
            messages = [{'role': 'user', 'content': goal}]
            for chunk in run_main_agent(system_message=ITEM_EXTRACTOR_SYSTEM_PROMPT, function_list=[], messages=messages):
                if isinstance(chunk, list) and chunk:
                    last_message = chunk[-1]
                    if last_message.get("role") == "assistant" and isinstance(last_message.get("content"), str):
                        extractor_response_str = last_message["content"]

            extractor_response_str = clean_llm_output(extractor_response_str).strip()
            extracted_items = JsonExtractor.extract_valid_json(extractor_response_str)

            if not isinstance(extracted_items, list) or not extracted_items:
                raise ValueError(f"Item Extractor agent failed to extract a list of items from the goal. Response: {extractor_response_str}")

            items = extracted_items # Update local variable
            logger.info(f"Task {task_id}: Extracted {len(items)} items. Updating task in DB.")

            # Update the task in the DB with the extracted items
            await db_manager.task_collection.update_one(
                {"task_id": task_id},
                {"$set": {"swarm_details.items": items}}
            )

        if not goal or not items: # Re-check after potential extraction attempt
            raise ValueError("Swarm task is missing goal or items after extraction attempt.")

        # --- 1. Resource Manager Agent ---
        logger.info(f"Invoking Resource Manager for user {user_id} with goal: {goal}")
        
        user_profile = await db_manager.get_user_profile(user_id)
        user_integrations = user_profile.get("userData", {}).get("integrations", {}) if user_profile else {}
        
        available_tools = {}
        for tool_name, config in INTEGRATIONS_CONFIG.items():
            if tool_name in ["tasks", "progress_updater"]: continue
            is_builtin = config.get("auth_type") == "builtin"
            is_connected = user_integrations.get(tool_name, {}).get("connected", False)
            if is_builtin or is_connected:
                available_tools[tool_name] = config.get("description", "")
        
        system_prompt = RESOURCE_MANAGER_SYSTEM_PROMPT.format(
            available_tools_json=json.dumps(list(available_tools.keys()))
        )
        
        items_sample = items[:5]
        user_prompt = f"Goal: \"{goal}\"\n\nItems (sample of {len(items)} total):\n{json.dumps(items_sample, indent=2, default=str)}"
        
        messages = [{'role': 'user', 'content': user_prompt}]
        
        manager_response_str = ""
        for chunk in run_main_agent(system_message=system_prompt, function_list=[], messages=messages):
            if isinstance(chunk, list) and chunk:
                last_message = chunk[-1]
                if last_message.get("role") == "assistant" and isinstance(last_message.get("content"), str):
                    manager_response_str = last_message["content"]

        manager_response_str = clean_llm_output(manager_response_str).strip()

        if not manager_response_str:
            raise Exception("Resource Manager agent returned an empty response.")
        
        swarm_plan = JsonExtractor.extract_valid_json(manager_response_str)

        if not isinstance(swarm_plan, list) or not all(isinstance(item, dict) for item in swarm_plan):
            raise Exception(f"Resource Manager returned an invalid plan. Response: {manager_response_str}")

        logger.info(f"Resource Manager created execution plan with {len(swarm_plan)} sub-task(s).")

        # --- 2. Dispatch Worker Tasks ---
        all_worker_groups = []
        total_agents = 0
        for config in swarm_plan:
            item_indices = config.get("item_indices", [])
            worker_prompt = config.get("worker_prompt")
            required_tools = config.get("required_tools", [])
            
            # --- Enforce sub-agent limit ---
            if total_agents > sub_agent_limit:
                raise Exception(f"Swarm plan exceeds the sub-agent limit for your plan ({total_agents} > {sub_agent_limit}). Please reduce the number of items or upgrade your plan.")

            if not all([isinstance(item_indices, list), worker_prompt, isinstance(required_tools, list)]):
                logger.warning(f"Skipping invalid worker configuration: {config}")
                continue

            valid_indices = [i for i in item_indices if i < len(items)]
            total_agents += len(valid_indices)
            task_group = group(run_single_item_worker.s(parent_task_id=task_id, user_id=user_id, item=items[i], worker_prompt=worker_prompt, worker_tools=required_tools) for i in valid_indices)
            if task_group:
                all_worker_groups.append(task_group)

        if not all_worker_groups:
            raise Exception("The execution plan resulted in no valid tasks to run.")
        
        parent_run_id = str(uuid.uuid4())
        run_doc = {
            "run_id": parent_run_id,
            "status": "processing",
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "execution_start_time": datetime.datetime.now(datetime.timezone.utc),
            "plan": swarm_plan,
            "progress_updates": [{
                "message": {"type": "info", "content": f"Resource manager created a plan for {total_agents} agents."},
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            }]
        }
        current_runs = task.get("runs", [])
        if not isinstance(current_runs, list):
            current_runs = []
        current_runs.append(run_doc)

        update_payload = {
            "runs": current_runs,
            "status": "processing",
            "swarm_details.total_agents": total_agents,
        }
        await db_manager.update_task(task_id, update_payload)
        await push_task_list_update(user_id, task_id, "swarm_plan_created")

        header = group(all_worker_groups)
        callback = aggregate_results_callback.s(parent_task_id=task_id, user_id=user_id, parent_run_id=parent_run_id)
        chord_task = chord(header, callback)
        
        logger.info(f"Dispatching a chord of {total_agents} parallel workers for task {task_id}, run {parent_run_id}.")
        chord_task.apply_async()

    except LLMProviderDownError as e:
        logger.error(f"LLM provider down during swarm orchestration for {task_id}: {e}", exc_info=True)
        await db_manager.update_task(task_id, {"status": "error", "error": "Sorry, our AI provider is currently down. Please try again later."})
    except Exception as e:
        logger.error(f"Error in orchestrate_swarm_task for task {task_id}: {e}", exc_info=True)
        await db_manager.update_task(task_id, {"status": "error", "error": str(e)})
    finally:
        await db_manager.close()

@celery_app.task(name="refine_and_plan_ai_task")
def refine_and_plan_ai_task(task_id: str, user_id: str):
    """
    Asynchronously refines an AI-assigned task's details using an LLM,
    updates the task in the DB, and then triggers the planning process.
    """
    logger.info(f"Refining and planning for AI task_id: {task_id}")
    run_async(async_refine_and_plan_ai_task(task_id, user_id))

async def async_refine_and_plan_ai_task(task_id: str, user_id: str):
    """Async logic for refining an AI task and then kicking off the planner."""
    db_manager = PlannerMongoManager()
    try:
        task = await db_manager.get_task(task_id)
        if not task or task.get("assignee") != "ai":
            logger.warning(f"Skipping refine/plan for task {task_id}: not found or not assigned to AI.")
            return

        # --- FIX: Store the original schedule provided by the user ---
        original_schedule = task.get("schedule")

        user_id = task["user_id"]
        user_profile = await db_manager.user_profiles_collection.find_one({"user_id": user_id})
        personal_info = user_profile.get("userData", {}).get("personalInfo", {}) if user_profile else {}
        user_name = personal_info.get("name", "User")
        user_timezone_str = personal_info.get("timezone", "UTC")
        try:
            user_timezone = ZoneInfo(user_timezone_str)
        except ZoneInfoNotFoundError:
            user_timezone = ZoneInfo("UTC")
        current_time_str = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        system_prompt = TASK_CREATION_PROMPT.format(
            user_name=user_name,
            user_timezone=user_timezone_str,
            current_time=current_time_str
        )
        # Use the raw prompt from the task's description for parsing
        messages = [{'role': 'user', 'content': task["description"]}]

        response_str = ""
        for chunk in run_main_agent(system_message=system_prompt, function_list=[], messages=messages):
            if isinstance(chunk, list) and chunk and chunk[-1].get("role") == "assistant":
                response_str = chunk[-1].get("content", "")

        parsed_data = JsonExtractor.extract_valid_json(clean_llm_output(response_str))
        if parsed_data:
            # Safeguard: never allow the refiner to nullify or change the user_id
            # --- ADD POSTHOG EVENT TRACKING ---
            schedule_type = parsed_data.get("schedule", {}).get("type", "once")
            capture_event(
                user_id,
                "task_created",
                {"task_id": task_id, "schedule_type": schedule_type, "source": "prompt"}
            )
            # --- END POSTHOG EVENT TRACKING ---

            if "user_id" in parsed_data:
                del parsed_data["user_id"]

            # Safeguard: never allow the refiner to nullify or change the user_id
            if "user_id" in parsed_data:
                del parsed_data["user_id"]

            # --- FIX: If an original schedule existed, restore it. ---
            # This ensures the user's explicit schedule from the UI is not overwritten by the LLM's interpretation.
            if original_schedule:
                parsed_data["schedule"] = original_schedule
            else:
                # This branch handles cases where no schedule was provided initially (e.g., future chat-based creation)
                schedule = parsed_data.get('schedule')
                if schedule and schedule.get('type') == 'once' and schedule.get('run_at') is None:
                    schedule['run_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Inject user's timezone into the schedule object if it exists
            if 'schedule' in parsed_data and parsed_data.get('schedule'):
                parsed_data['schedule']['timezone'] = user_timezone_str
            # Ensure description is not empty, fall back to original prompt if needed
            if not parsed_data.get("name"):
                parsed_data["name"] = task["description"] or task["name"]
            await db_manager.update_task_field(task_id, user_id, parsed_data)
            logger.info(f"Successfully refined and updated AI task {task_id} with new details.")
        else:
            logger.warning(f"Could not parse details for AI task {task_id}, proceeding with raw description.")

        # Now trigger the planning step, which is the main purpose of AI-assigned tasks
        generate_plan_from_context.delay(task_id, user_id)
        logger.info(f"Refinement complete for AI task {task_id}, dispatched to planner.")

    except LLMProviderDownError as e:
        logger.error(f"LLM provider down during task refinement for {task_id}: {e}", exc_info=True)
        await db_manager.update_task_field(task_id, user_id, {"status": "error", "error": "Sorry, our AI provider is currently down. Please try again later."})
    except Exception as e:
        logger.error(f"Error refining and planning AI task {task_id}: {e}", exc_info=True)
        await db_manager.update_task_field(task_id, user_id, {"status": "error", "error": "Failed during initial refinement."})
    finally:
        await db_manager.close()

@celery_app.task(name="process_task_change_request")
def process_task_change_request(task_id: str, user_id: str, user_message: str):
    """Processes a user's change request on a completed task via the in-task chat."""
    logger.info(f"Task Change Request: Received for task {task_id} from user {user_id}. Message: '{user_message}'")
    run_async(async_process_change_request(task_id, user_id, user_message))

async def async_process_change_request(task_id: str, user_id: str, user_message: str):
    """DEPRECATED: This logic is now handled by the main planner flow."""
    db_manager = PlannerMongoManager()
    try:
        # 1. Fetch the task and its full history
        task = await db_manager.get_task(task_id)
        if not task:
            logger.error(f"Task Change Request: Task {task_id} not found.")
            return

        # 2. Append user message to the task's chat history
        chat_history = task.get("chat_history", [])
        chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })

        # 3. Update task status and chat history in DB
        await db_manager.update_task_field(task_id, user_id, {
            "chat_history": chat_history,
            "status": "planning" # Revert to planning to re-evaluate
        })

        await notify_user(user_id, f"I've received your changes for '{task.get('description', 'the task')}' and will start working on them.", task_id)

        # 4. Create a consolidated context for the planner
        # This is similar to the initial context creation but includes much more history
        original_context = task.get("original_context", {})
        if isinstance(original_context, str):
            original_context = JsonExtractor.extract_valid_json(original_context) or {"source": "unknown", "raw_context": original_context}

        original_context["previous_plan"] = task.get("plan", [])
        original_context["previous_result"] = task.get("result", "")
        original_context["chat_history"] = chat_history

        # 5. Trigger the planner with this rich context
        # The planner will now see the change request and all prior history
        # The action_items can be just the user's new message.
        process_action_item.delay(user_id, [user_message], [], task_id, original_context)
        logger.info(f"Task Change Request: Dispatched task {task_id} back to planner with new context.")

    except Exception as e:
        logger.error(f"Error in async_process_change_request for task {task_id}: {e}", exc_info=True)
    finally:
        await db_manager.close()

@celery_app.task(name="process_action_item")
def process_action_item(user_id: str, action_items: list, topics: list, source_event_id: str, original_context: dict):
    """Orchestrates the pre-planning phase for a new proactive task."""
    run_async(async_process_action_item(user_id, action_items, topics, source_event_id, original_context))

async def async_process_action_item(user_id: str, action_items: list, topics: list, source_event_id: str, original_context: dict):
    """Async logic for the proactive task orchestrator."""
    db_manager = PlannerMongoManager()
    task_id = None
    try:
        task_description = " ".join(map(str, action_items))
        # For proactive tasks, the initial name is the description, and the description is the context.
        # The planner will generate a better name and description later.
        full_description = json.dumps(original_context, indent=2, default=str)
        # Per spec, create task with 'planning' status and immediately trigger planner.
        task = await db_manager.create_initial_task(user_id, task_description, full_description, action_items, topics, original_context, source_event_id)
        task_id = task["task_id"]
        generate_plan_from_context.delay(task_id, user_id)
        logger.info(f"Task {task_id} created with status 'planning' and dispatched to planner worker.")

    except Exception as e:
        logger.error(f"Error in process_action_item: {e}", exc_info=True)
        if task_id:
            await db_manager.update_task_status(task_id, "error", {"error": str(e)})
    finally:
        await db_manager.close()

@celery_app.task(name="generate_plan_from_context")
def generate_plan_from_context(task_id: str, user_id: str):
    """Generates a plan for a task once all context is available."""
    run_async(async_generate_plan(task_id, user_id))

async def async_generate_plan(task_id: str, user_id: str):
    """Async logic for plan generation."""
    db_manager = PlannerMongoManager()
    try:
        task = await db_manager.get_task(task_id)
        if not task:
            logger.error(f"Cannot generate plan: Task {task_id} not found.")
            return

        if task.get("runs") is None:
             logger.error(f"Cannot generate plan: Task {task_id} is malformed (missing 'runs' array).")
             await db_manager.update_task_status(task_id, "error", {"error": "Task data is malformed: missing 'runs' array."})
             return

        # Determine if this is a change request before proceeding.
        is_change_request = bool(task.get("chat_history"))

        # Prevent re-planning if it's not in a plannable state
        plannable_statuses = ["planning"]
        if task.get("status") in ["clarification_pending", "clarification_answered"]:
            plannable_statuses.append(task.get("status"))
        if task.get("status") not in plannable_statuses:
             logger.warning(f"Task {task_id} is not in a plannable state (current: {task.get('status')}). Aborting plan generation.")
             return

        # Trust the user_id passed to the Celery task.
        # If the document is missing it for some reason, log a warning and proceed.
        if not task.get("user_id"):
            logger.warning(f"Task {task_id} document is missing user_id. Proceeding with passed user_id '{user_id}'.")
            # Attempt to heal the document
            await db_manager.update_task_field(task_id, user_id, {"user_id": user_id})

        original_context = task.get("original_context", {})

        # For re-planning, add previous results and chat history to the context
        if task.get("chat_history"):
            original_context["chat_history"] = task.get("chat_history")
            original_context["previous_plan"] = task.get("plan")
            original_context["previous_result"] = task.get("result")

        user_profile = await db_manager.user_profiles_collection.find_one(
            {"user_id": user_id},
            {"userData.personalInfo": 1} # Projection to get only necessary data
        )
        if not user_profile:
            logger.error(f"User profile not found for user_id '{user_id}' associated with task {task_id}. Cannot generate plan.")
            await db_manager.update_task_status(task_id, "error", {"error": f"User profile not found for user_id '{user_id}'."})
            return

        if not user_profile:
            logger.error(f"User profile not found for user_id '{user_id}' associated with task {task_id}. Cannot generate plan.")
            await db_manager.update_task_status(task_id, "error", {"error": f"User profile not found for user_id '{user_id}'."})
            return

        personal_info = user_profile.get("userData", {}).get("personalInfo", {})
        user_name = personal_info.get("name", "User")
        user_location_raw = personal_info.get("location", "Not specified")
        if isinstance(user_location_raw, dict):
            user_location = f"latitude: {user_location_raw.get('latitude')}, longitude: {user_location_raw.get('longitude')}"
        else:
            user_location = user_location_raw

        user_timezone_str = personal_info.get("timezone", "UTC")
        try:
            user_timezone = ZoneInfo(user_timezone_str)
        except ZoneInfoNotFoundError:
            logger.warning(f"Invalid timezone '{user_timezone_str}' for user {user_id}. Defaulting to UTC.")
            user_timezone = ZoneInfo("UTC")

        current_user_time = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        action_items = task.get("action_items", [])
        if not action_items:
            # This is likely a manually created task. Use its description as the action item.
            logger.info(f"Task {task_id}: No 'action_items' field found. Using main description as the action.")
            action_items = [task.get("description", "")]

        available_tools = get_all_mcp_descriptions()

        agent_config = get_planner_agent(available_tools, current_user_time, user_name, user_location)

        user_prompt_content = "Please create a plan for the following action items:\n- " + "\n- ".join(action_items)
        messages = [{'role': 'user', 'content': user_prompt_content}]

        final_response_str = ""
        for chunk in run_main_agent(system_message=agent_config["system_message"], function_list=agent_config["function_list"], messages=messages):
            if isinstance(chunk, list) and chunk and chunk[-1].get("role") == "assistant":
                final_response_str = chunk[-1].get("content", "")

        if not final_response_str:
            raise Exception("Planner agent returned no response.")

        plan_data = JsonExtractor.extract_valid_json(clean_llm_output(final_response_str))
        if not plan_data or "plan" not in plan_data:
            raise Exception(f"Planner agent returned invalid JSON: {final_response_str}")

        await db_manager.update_task_with_plan(task_id, plan_data, is_change_request)
        capture_event(user_id, "proactive_task_generated", {
            "task_id": task_id,
            "source": task.get("original_context", {}).get("source", "unknown"),
            "plan_steps": len(plan_data.get("plan", []))
        })

        # Notify user that a plan is ready for their approval
        await notify_user(
            user_id, f"I've created a new plan for you: '{plan_data.get('name', '...')[:50]}...'", task_id,
            notification_type="taskNeedsApproval"
        )

        # CRITICAL: Notify the frontend to refresh its task list.
        # A run_id doesn't exist yet, as this is the planning stage. We pass a placeholder.
        await push_task_list_update(user_id, task_id, "plan_generated")
        logger.info(f"Sent task_list_updated push notification for user {user_id}")


    except LLMProviderDownError as e:
        logger.error(f"LLM provider down during plan generation for task {task_id}: {e}", exc_info=True)
        await db_manager.update_task_status(task_id, "error", {"error": "Sorry, our AI provider is currently down. Please try again later."})
    except Exception as e:
        logger.error(f"Error generating plan for task {task_id}: {e}", exc_info=True)
        await db_manager.update_task_status(task_id, "error", {"error": str(e)})
    finally:
        await db_manager.close()

@celery_app.task(name="refine_task_details")
def refine_task_details(task_id: str):
    """
    Asynchronously refines a user-created task by using an LLM to parse
    the description, priority, and schedule from the initial prompt.
    """
    logger.info(f"Refining details for task_id: {task_id}")
    run_async(async_refine_task_details(task_id))

async def async_refine_task_details(task_id: str):
    db_manager = PlannerMongoManager()
    try:
        task = await db_manager.get_task(task_id)
        if not task or task.get("assignee") != "user":
            logger.warning(f"Skipping refinement for task {task_id}: not found or not assigned to user.")
            return

        user_id = task["user_id"]
        user_profile = await db_manager.user_profiles_collection.find_one({"user_id": user_id})
        personal_info = user_profile.get("userData", {}).get("personalInfo", {}) if user_profile else {}
        user_name = personal_info.get("name", "User")
        user_timezone_str = personal_info.get("timezone", "UTC")
        user_timezone = ZoneInfo(user_timezone_str)
        current_time_str = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        system_prompt = TASK_CREATION_PROMPT.format(
            user_name=user_name,
            user_timezone=user_timezone_str,
            current_time=current_time_str
        )
        messages = [{'role': 'user', 'content': task["description"]}] # Use the raw prompt for parsing

        response_str = ""
        for chunk in run_main_agent(system_message=system_prompt, function_list=[], messages=messages):
            if isinstance(chunk, list) and chunk and chunk[-1].get("role") == "assistant":
                response_str = chunk[-1].get("content", "")

        parsed_data = JsonExtractor.extract_valid_json(clean_llm_output(response_str))
        if parsed_data:
            # Inject user's timezone into the schedule object if it exists
            if 'schedule' in parsed_data and parsed_data.get('schedule'):
                parsed_data['schedule']['timezone'] = user_timezone_str
            await db_manager.update_task_field(task_id, user_id, parsed_data)
            logger.info(f"Successfully refined and updated user task {task_id} with new details.")

    except Exception as e:
        logger.error(f"Error refining task {task_id}: {e}", exc_info=True)
    finally:
        await db_manager.close()

@celery_app.task(name="execute_triggered_task")
def execute_triggered_task(user_id: str, source: str, event_type: str, event_data: Dict[str, Any]):
    """
    Checks for and executes any tasks triggered by a new event.
    """
    logger.info(f"Checking for triggered tasks for user '{user_id}' from source '{source}' event '{event_type}'.")
    run_async(async_execute_triggered_task(user_id, source, event_type, event_data))

def _event_matches_filter(event_data: Dict[str, Any], task_filter: Dict[str, Any], source: str) -> bool:
    """
    Checks if an event's data matches the conditions defined in a task's filter.
    Supports complex, MongoDB-like query syntax including $or, $and, $not,
    and operators like $eq, $ne, $in, $nin, $contains, $regex.
    """
    if not task_filter:
        return True  # An empty filter matches everything.

    def _extract_email(header_string: str) -> str:
        """Extracts the email address from a header string like 'Name <email@example.com>'."""
        if not isinstance(header_string, str):
            return ""
        match = re.search(r'<(.+?)>', header_string)
        if match:
            return match.group(1).lower().strip()
        return header_string.lower().strip()

    def _evaluate(condition: Any, data: Dict[str, Any]) -> bool:
        if not isinstance(condition, dict):
            return False

        # Check for top-level logical operators first
        if "$or" in condition:
            if not isinstance(condition["$or"], list): return False
            return any(_evaluate(sub_cond, data) for sub_cond in condition["$or"])
        if "$and" in condition:
            if not isinstance(condition["$and"], list): return False
            return all(_evaluate(sub_cond, data) for sub_cond in condition["$and"])
        if "$not" in condition:
            return not _evaluate(condition["$not"], data)

        # If no logical operators, it's an implicit AND of field conditions
        for field, query in condition.items():
            event_value = None
            # Special remapping for gmail 'from' filter to match Composio's 'sender' field
            if field == 'from' and source == 'gmail':
                event_value = data.get('sender')
            else:
                event_value = data.get(field)

            # Special handling for 'from' field
            if field == 'from' and source == 'gmail' and isinstance(event_value, str):
                event_value = _extract_email(event_value) # This will now work correctly

            if isinstance(query, dict): # Field has operators like {$contains: ...}
                for op, op_val in query.items():
                    if op == "$eq":
                        if (field == 'from' and source == 'gmail' and isinstance(op_val, str) and event_value != op_val.lower().strip()) or \
                           (not (field == 'from' and source == 'gmail') and event_value != op_val):
                            return False
                    elif op == "$ne":
                        if event_value == op_val: return False
                    elif op == "$in":
                        if not isinstance(op_val, list) or event_value not in op_val: return False
                    elif op == "$nin":
                        if not isinstance(op_val, list) or event_value in op_val: return False
                    elif op == "$contains":
                        if not isinstance(event_value, str) or not isinstance(op_val, str) or op_val.lower() not in event_value.lower(): return False
                    elif op == "$regex":
                        if not isinstance(event_value, str): return False
                        try:
                            if not re.search(op_val, event_value, re.IGNORECASE): return False
                        except re.error: return False
                    else: return False
            else: # Simple equality check
                if field == 'from' and source == 'gmail' and isinstance(query, str):
                    if event_value != query.lower().strip(): return False
                elif event_value != query: return False

        return True

    return _evaluate(task_filter, event_data)

async def async_execute_triggered_task(user_id: str, source: str, event_type: str, event_data: Dict[str, Any]):
    db_manager = MongoManager()
    try:
        # Find all active tasks for this user that are triggered by this source and event
        query = {
            "user_id": user_id,
            "status": "active",
            "enabled": True,
            "schedule.type": "triggered",
            "schedule.source": source,
            "schedule.event": event_type
        }
        triggered_tasks_cursor = db_manager.task_collection.find(query)
        tasks_to_check = await triggered_tasks_cursor.to_list(length=None)

        if not tasks_to_check:
            logger.info(f"No triggered tasks found for user '{user_id}' matching this event.")
            return

        logger.info(f"Found {len(tasks_to_check)} potential triggered tasks for user '{user_id}'.")

        for task in tasks_to_check:
            task_id = task['task_id']
            schedule = task.get('schedule', {})
            task_filter = schedule.get('filter', {})

            if _event_matches_filter(event_data, task_filter, source):
                logger.info(f"Event matches filter for triggered task {task_id}. Queuing for execution.")

                # The event data becomes the context for this execution run.
                # We need to create a new "run" for this triggered execution.
                new_run = {
                    "run_id": str(uuid.uuid4()),
                    "status": "processing",
                    "trigger_event_data": event_data,
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "execution_start_time": datetime.datetime.now(datetime.timezone.utc)
                }

                current_runs = task.get("runs", [])
                if not isinstance(current_runs, list):
                    current_runs = []
                current_runs.append(new_run)
                update_payload = {
                    "runs": current_runs,
                    "status": "processing",
                }
                await db_manager.update_task(task_id, update_payload)

                # Queue the executor with the new run_id
                execute_task_plan.delay(task_id, user_id, new_run['run_id'])
            else:
                logger.debug(f"Event did not match filter for triggered task {task_id}.")

    except Exception as e:
        logger.error(f"Error during async_execute_triggered_task for user {user_id}: {e}", exc_info=True)
    finally:
        await db_manager.close()

# --- Polling Tasks ---
@celery_app.task(name="poll_gmail_for_triggers")
def poll_gmail_for_triggers(user_id: str, polling_state: dict):
    logger.info(f"Polling Gmail for triggers for user {user_id}")
    db_manager = GmailPollerDB()
    service = GmailPollingService(db_manager)
    run_async(service._run_single_user_poll_cycle(user_id, polling_state, mode='triggers'))

@celery_app.task(name="poll_gcalendar_for_triggers")
def poll_gcalendar_for_triggers(user_id: str, polling_state: dict):
    logger.info(f"Polling GCalendar for triggers for user {user_id}")
    db_manager = GCalPollerDB()
    service = GCalendarPollingService(db_manager)
    run_async(service._run_single_user_poll_cycle(user_id, polling_state, mode='triggers'))

# --- Scheduler Tasks ---
@celery_app.task(name="schedule_trigger_polling")
def schedule_trigger_polling():
    """Celery Beat task for the frequent triggered workflow polling."""
    logger.info("Trigger Polling Scheduler: Checking for due tasks...")
    run_async(async_schedule_polling('triggers'))

async def async_schedule_polling(mode: str):
    """Generic scheduler logic for both proactivity and triggers."""
    db_manager = GmailPollerDB() # Can use either DB manager as they are identical
    try:
        supported_polling_services = ["gmail", "gcalendar"]
        await db_manager.reset_stale_polling_locks("gmail", mode)
        await db_manager.reset_stale_polling_locks("gcalendar", mode)

        for service_name in supported_polling_services:
            due_tasks_states = await db_manager.get_due_polling_tasks_for_service(service_name, mode)
            logger.info(f"Found {len(due_tasks_states)} due '{mode}' tasks for {service_name}.")

            for task_state in due_tasks_states:
                user_id = task_state["user_id"]
                locked_task_state = await db_manager.set_polling_status_and_get(user_id, service_name, mode)

                if locked_task_state and '_id' in locked_task_state and isinstance(locked_task_state['_id'], ObjectId):
                    locked_task_state['_id'] = str(locked_task_state['_id'])

                if locked_task_state:
                    if service_name == "gmail":
                        poll_gmail_for_triggers.delay(user_id, locked_task_state)
                    elif service_name == "gcalendar":
                        poll_gcalendar_for_triggers.delay(user_id, locked_task_state)
                    logger.info(f"Dispatched '{mode}' polling task for {user_id} - service: {service_name}")
    finally:
        await db_manager.close()

def calculate_next_run(schedule: Dict[str, Any], last_run: Optional[datetime.datetime] = None) -> Tuple[Optional[datetime.datetime], Optional[str]]:
    """Calculates the next execution time for a scheduled task in UTC."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    user_timezone_str = schedule.get("timezone", "UTC")

    try:
        user_tz = ZoneInfo(user_timezone_str)
    except ZoneInfoNotFoundError:
        logger.warning(f"Invalid timezone '{user_timezone_str}'. Defaulting to UTC.")
        user_timezone_str = "UTC"
        user_tz = ZoneInfo("UTC")

    time_str = schedule.get("time", "09:00")
    if not isinstance(time_str, str) or ":" not in time_str:
        logger.warning(f"Invalid or missing 'time' in recurring schedule: {schedule}. Defaulting to 09:00.")
        time_str = "09:00"

    # The reference time for 'after' should be in the user's timezone to handle day boundaries correctly
    start_time_user_tz = (last_run or now_utc).astimezone(user_tz)

    try:
        frequency = schedule.get("frequency")
        hour, minute = map(int, time_str.split(':'))

        # Create the start datetime in the user's timezone
        dtstart_user_tz = start_time_user_tz.replace(hour=hour, minute=minute, second=0, microsecond=0)

        rule = None
        if frequency == 'daily':
            rule = rrule.rrule(rrule.DAILY, dtstart=dtstart_user_tz, until=start_time_user_tz + datetime.timedelta(days=365))
        elif frequency == 'weekly':
            days = schedule.get("days", [])
            if not days: return None, user_timezone_str
            weekday_map = {"Sunday": rrule.SU, "Monday": rrule.MO, "Tuesday": rrule.TU, "Wednesday": rrule.WE, "Thursday": rrule.TH, "Friday": rrule.FR, "Saturday": rrule.SA}
            byweekday = [weekday_map[day] for day in days if day in weekday_map]
            if not byweekday: return None, user_timezone_str
            rule = rrule.rrule(rrule.WEEKLY, dtstart=dtstart_user_tz, byweekday=byweekday, until=start_time_user_tz + datetime.timedelta(days=365))

        if rule:
            next_run_user_tz = rule.after(start_time_user_tz)
            if next_run_user_tz:
                # Convert the result back to UTC for storage
                return next_run_user_tz.astimezone(datetime.timezone.utc), user_timezone_str
    except Exception as e:
        logger.error(f"Error calculating next run time for schedule {schedule}: {e}")
    return None, user_timezone_str

@celery_app.task(name="run_due_tasks")
def run_due_tasks():
    """Celery Beat task to check for and queue user-defined tasks (recurring and scheduled-once)."""
    logger.info("Scheduler: Checking for due user-defined tasks...")
    run_async(async_run_due_tasks())


async def async_run_due_tasks():
    db_manager = PlannerMongoManager()
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        # Fetch tasks that are due and are either 'active' (recurring) or 'pending' (scheduled-once)
        query = {
            "status": {"$in": ["active", "pending"]},
            "enabled": True,
            "next_execution_at": {"$lte": now}
        }
        due_tasks_cursor = db_manager.tasks_collection.find(query).sort("next_execution_at", 1)
        due_tasks = await due_tasks_cursor.to_list(length=100) # Process up to 100 at a time

        if not due_tasks:
            logger.info("Scheduler: No user-defined tasks are due.")
            return

        logger.info(f"Scheduler: Found {len(due_tasks)} due user-defined tasks.")
        for task in due_tasks:
            # Try to lock this task
            lock_result = await db_manager.tasks_collection.update_one(
                {"_id": task["_id"], "status": {"$in": ["active", "pending"]}}, # Ensure it hasn't been picked up
                {"$set": {"status": "processing", "last_execution_at": now}}
            )

            if lock_result.modified_count == 0:
                logger.info(f"Scheduler: Task {task['_id']} was already picked up by another worker. Skipping.")
                continue

            task_id = task['task_id']
            user_id = task['user_id']
            logger.info(f"Scheduler: Locked and queuing task {task_id} for execution.")

            new_run = {
                "run_id": str(uuid.uuid4()),
                "status": "processing",
                "created_at": now,
                "execution_start_time": now
            }
            current_runs = task.get("runs", [])
            if not isinstance(current_runs, list):
                current_runs = []
            current_runs.append(new_run)
            await db_manager.update_task(task_id, {"runs": current_runs})
            execute_task_plan.delay(task_id, user_id, new_run['run_id'])

    except Exception as e:
        logger.error(f"Scheduler: An error occurred checking user-defined tasks: {e}", exc_info=True)
    finally:
        await db_manager.close()


# --- Chat History Summarization Task ---
@celery_app.task(name="summarize_old_conversations")
def summarize_old_conversations():
    """
    Celery Beat task to find old, unsummarized messages, group them into chunks,
    summarize them, and store the summaries and embeddings in ChromaDB.
    """
    logger.info("Summarization Task: Starting to look for old conversations to summarize.")
    run_async(async_summarize_conversations())

# src/server/workers/tasks.py

async def async_summarize_conversations():
    db_manager = PlannerMongoManager()  # Re-using for its mongo access
    try:
        # 1. Find users with recent, unsummarized messages
        # We process one user at a time to keep the task manageable.
        # This aggregation finds the first user with unsummarized messages older than 1 day.
        # Temporarily changed for testing
        one_day_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        pipeline = [
            {"$match": {"is_summarized": False, "timestamp": {"$lt": one_day_ago}}},
            {"$group": {"_id": "$user_id"}},
            {"$limit": 1} # Process one user per run
        ]
        user_to_process = await db_manager.messages_collection.aggregate(pipeline).to_list(length=1)

        if not user_to_process:
            logger.info("Summarization Task: No users with old, unsummarized messages found.")
            return

        user_id = user_to_process[0]['_id']
        logger.info(f"Summarization Task: Found user {user_id} with messages to summarize.")

        # 2. Fetch all unsummarized messages for this user
        messages_cursor = db_manager.messages_collection.find({
            "user_id": user_id,
            "is_summarized": False
        }).sort("timestamp", 1) # Get them in chronological order
        
        messages_to_process = await messages_cursor.to_list(length=None)
        
        if not messages_to_process:
            logger.info(f"Summarization Task: No unsummarized messages to process for user {user_id}.")
            return

        # 3. Group messages into chunks (e.g., of 30 messages)
        chunk_size = 30
        message_chunks = [messages_to_process[i:i + chunk_size] for i in range(0, len(messages_to_process), chunk_size)]
        
        # 4. Process each chunk
        
        for chunk in message_chunks:
            if len(chunk) < 2: # Don't summarize single messages
                continue

            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chunk])
            
            # 4a. Summarize with LLM using the new, narrative-style prompt
            messages = [{'role': 'user', 'content': conversation_text}]
            summary_text = ""
            
            # --- MODIFIED PROMPT ---
            system_prompt = """
You are the AI assistant in the provided conversation log. Your task is to write a summary of the conversation from your own perspective, as if you are recalling the memory of the interaction.

Core Instructions:
1.  Adopt a First-Person Narrative: Use "I", "me", and "my" to refer to your own actions and thoughts. Refer to the other party as "the user".
2.  Describe the Flow: Recount the conversation as a sequence of events. For example: "The user told me about their project...", "I then asked for clarification on...", "We then discussed...".
3.  CRITICAL INSTRUCTION FOR FILE UPLOADS: If a user message involves uploading a file (e.g., "user: (Attached file for context: report.pdf) Can you summarize this?"), your summary must NOT state that you cannot process it. Instead, you MUST describe the user's action factually. For example: "The user uploaded a file named 'report.pdf' and asked for a summary."
4.  Goal: The goal is to create a dense, narrative paragraph that captures the key information, decisions, and flow of the conversation from your point of view. Focus on information that would be useful for future context.
5.  Format: Do not add any preamble or sign-off. Respond only with the summary paragraph.
"""
            # --- END MODIFIED PROMPT ---

            for response_chunk in run_main_agent(system_message=system_prompt, function_list=[], messages=messages):
                 if isinstance(response_chunk, list) and response_chunk and response_chunk[-1].get("role") == "assistant":
                    summary_text = response_chunk[-1].get("content", "")

            summary_text = clean_llm_output(summary_text)
            
            if not summary_text:
                logger.warning(f"Summarization for user {user_id} produced an empty result. Skipping chunk.")
                continue

            # 4b. Store summary and embedding in ChromaDB
            summary_id = str(uuid.uuid4())
            message_ids = [msg['message_id'] for msg in chunk]
            
            collection = get_conversation_summaries_collection()
            collection.add(
                ids=[summary_id],
                documents=[summary_text],
                metadatas=[{
                    "user_id": user_id,
                    "start_timestamp": chunk[0]['timestamp'].isoformat(),
                    "end_timestamp": chunk[-1]['timestamp'].isoformat(),
                    "message_ids_json": json.dumps(message_ids)
                }]
            )
            logger.info(f"Summarization Task: Stored summary {summary_id} in ChromaDB for user {user_id}.")
            
            # 4c. Update original messages in MongoDB
            message_object_ids = [msg['_id'] for msg in chunk]
            await db_manager.messages_collection.update_many(
                {"_id": {"$in": message_object_ids}},
                {"$set": {"is_summarized": True, "summary_id": summary_id}}
            )

    except Exception as e:
        logger.error(f"Error during conversation summarization: {e}", exc_info=True)
    finally:
        await db_manager.close()