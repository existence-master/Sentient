### `src\server\workers\executor\tasks.py`

import os
import json
import datetime
import asyncio
import motor.motor_asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import re
from json_extractor import JsonExtractor

from main.analytics import capture_event
from qwen_agent.agents import Assistant
from workers.celery_app import celery_app
from workers.executor.prompts import RESULT_GENERATOR_SYSTEM_PROMPT # noqa: E501
from workers.utils.api_client import notify_user, push_progress_update, push_task_list_update
from workers.utils.text_utils import clean_llm_output
from celery import chord, group
from main.llm import run_agent as run_main_agent, LLMProviderDownError
from workers.utils.crypto import aes_decrypt, encrypt_doc, decrypt_doc

# Load environment variables for the worker from its own config
from workers.executor.config import (MONGO_URI, MONGO_DB_NAME,
                                     INTEGRATIONS_CONFIG, OPENAI_API_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME, ENVIRONMENT)

# Setup logger for this module
logger = logging.getLogger(__name__)

DB_ENCRYPTION_ENABLED = ENVIRONMENT == 'stag'

# --- Database Connection within Celery Task ---
def get_db_client():
    return motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)[MONGO_DB_NAME]

# Helper to run async functions from sync Celery tasks
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

async def update_task_run_status(db, task_id: str, run_id: str, status: str, user_id: str, details: Dict = None, block_id: Optional[str] = None):
    """
    Fetches a task, updates a specific run's status and details in memory,
    and then saves the entire modified runs array back to the database.
    This approach is compatible with field-level encryption.
    """
    task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})
    if not task:
        logger.error(f"Cannot update run status: Task {task_id} not found for user {user_id}.")
        return

    SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
    decrypt_doc(task, SENSITIVE_TASK_FIELDS)

    runs = task.get("runs", [])
    if not isinstance(runs, list):
        logger.error(f"Task {task_id} 'runs' field is not a list after decryption. Cannot update status.")
        return

    run_found = False
    for run in runs:
        if run.get("run_id") == run_id:
            run["status"] = status
            if details:
                if "result" in details:
                    run["result"] = details["result"]
                if "error" in details:
                    run["error"] = details["error"]
            run_found = True
            break

    if not run_found:
        logger.warning(f"Run {run_id} not found in task {task_id} to update status.")
        return

    update_payload = {
        "status": status,
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
        "runs": runs
    }

    # Encrypt the modified runs array before updating
    encrypt_doc(update_payload, ["runs"])

    logger.info(f"Updating task {task_id} run {run_id} status to '{status}' with details: {details}")
    await db.tasks.update_one(
        {"task_id": task_id, "user_id": user_id},
        {"$set": update_payload}
    )

    task_description = task.get("name", "Unnamed Task")

    if status in ["completed", "error"]:
        notification_message = f"Task '{task_description}' has finished with status: {status}."
        notification_type = "taskCompleted" if status == "completed" else "taskFailed"
        await notify_user(user_id, notification_message, task_id, notification_type=notification_type)

async def add_progress_update(db, task_id: str, run_id: str, user_id: str, message: Any, block_id: Optional[str] = None):
    logger.info(f"Adding progress update to task {task_id}: '{message}'")
    progress_update = {"message": message, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    
    # 1. Push to frontend via WebSocket in real-time
    await push_progress_update(user_id, task_id, run_id, message)

    # 2. Fetch, decrypt, modify, encrypt, and save to DB for persistence
    task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})
    if not task:
        logger.error(f"Cannot add progress update: Task {task_id} not found for user {user_id}.")
        return

    SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
    decrypt_doc(task, SENSITIVE_TASK_FIELDS)

    runs = task.get("runs", [])
    if not isinstance(runs, list):
        logger.error(f"Task {task_id} 'runs' field is not a list after decryption. Cannot add progress update.")
        return

    run_found = False
    for run in runs:
        if run.get("run_id") == run_id:
            if "progress_updates" not in run or not isinstance(run["progress_updates"], list):
                run["progress_updates"] = []
            run["progress_updates"].append(progress_update)
            run_found = True
            break

    if not run_found:
        logger.warning(f"Run {run_id} not found in task {task_id} to add progress update.")
        return

    update_payload = {"runs": runs}
    encrypt_doc(update_payload, ["runs"])

    await db.tasks.update_one(
        {"task_id": task_id, "user_id": user_id},
        {"$set": update_payload}
    )

    if block_id:
        await db.tasks_blocks_collection.update_one(
            {"block_id": block_id, "user_id": user_id},
            {"$push": {"task_progress": progress_update}}
        )

@celery_app.task(name="execute_task_plan")
def execute_task_plan(task_id: str, user_id: str, run_id: str):
    logger.info(f"Celery worker received task 'execute_task_plan' for task_id: {task_id}, run_id: {run_id}")
    return run_async(async_execute_task_plan(task_id, user_id, run_id))

def parse_agent_string_to_updates(content: str) -> List[Dict[str, Any]]:
    """
    Parses a string containing agent's XML-like tags into a list of structured update objects.
    This logic is inspired by the frontend's ChatBubble.js parser.
    """
    if not content or not isinstance(content, str):
        return []

    updates = []
    # This regex finds all tags and captures the tag name in group 2
    regex = re.compile(r'(<(think|tool_code|tool_result|answer)[\s\S]*?>[\s\S]*?<\/\2>)', re.DOTALL)

    last_index = 0
    for match in regex.finditer(content):
        # Capture text between tags as an 'info' update
        preceding_text = content[last_index:match.start()].strip()
        if preceding_text:
            updates.append({"type": "info", "content": preceding_text})

        tag_content = match.group(1)

        think_match = re.search(r'<think>([\s\S]*?)</think>', tag_content, re.DOTALL)
        if think_match:
            updates.append({"type": "thought", "content": think_match.group(1).strip()})
            last_index = match.end()
            continue

        tool_code_match = re.search(r'<tool_code name="([^"]+)">([\s\S]*?)</tool_code>', tag_content, re.DOTALL)
        if tool_code_match:
            params_str = tool_code_match.group(2).strip()
            params = JsonExtractor.extract_valid_json(params_str)
            if not params:
                params = {"raw_parameters": params_str}
            updates.append({
                "type": "tool_call",
                "tool_name": tool_code_match.group(1),
                "parameters": params
            })
            last_index = match.end()
            continue

        tool_result_match = re.search(r'<tool_result tool_name="([^"]+)">([\s\S]*?)</tool_result>', tag_content, re.DOTALL)
        if tool_result_match:
            result_str = tool_result_match.group(2).strip()
            result_content = JsonExtractor.extract_valid_json(result_str)
            if not result_content:
                result_content = {"raw_result": result_str}
            is_error = isinstance(result_content, dict) and result_content.get("status") == "failure"
            updates.append({
                "type": "tool_result",
                "tool_name": tool_result_match.group(1),
                "result": result_content.get("result", result_content.get("error", result_content)),
                "is_error": is_error
            })
            last_index = match.end()
            continue

        answer_match = re.search(r'<answer>([\s\S]*?)</answer>', tag_content, re.DOTALL)
        if answer_match:
            updates.append({"type": "final_answer", "content": answer_match.group(1).strip()})
            last_index = match.end()
            continue

        last_index = match.end()

    # Capture any trailing text after the last tag
    trailing_text = content[last_index:].strip()
    if trailing_text:
        updates.append({"type": "info", "content": trailing_text})

    return updates

async def async_execute_task_plan(task_id: str, user_id: str, run_id: str):
    db = get_db_client()
    task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})

    # Decrypt sensitive fields before processing # noqa: E501
    SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
    decrypt_doc(task, SENSITIVE_TASK_FIELDS)

    if not task:
        logger.error(f"Executor: Task {task_id} not found for user {user_id}.")
        return {"status": "error", "message": "Task not found."}

    # Find the specific run to execute
    current_run = next((r for r in task.get("runs", []) if r["run_id"] == run_id), None)
    if not current_run:
        logger.error(f"Executor: Run {run_id} not found for task {task_id}. Aborting.")
        return {"status": "error", "message": f"Run ID {run_id} not found."}

    original_context_data = task.get("original_context", {})
    block_id = None
    if original_context_data.get("source") == "tasks_block":
        block_id = original_context_data.get("block_id")

    # --- NEW: Inject trigger event data into the context for the executor ---
    trigger_event_data_str = ""
    if current_run.get("trigger_event_data"):
        trigger_event_data_str = json.dumps(current_run["trigger_event_data"], indent=2, default=str)
        trigger_event_prompt_section = f"**Triggering Event Data (Your primary context for this run):**\n---BEGIN TRIGGER DATA---\n{trigger_event_data_str}\n---END TRIGGER DATA---\n\n"
    else:
        trigger_event_prompt_section = ""

    logger.info(f"Executor started processing task {task_id} (block_id: {block_id}) for user {user_id}.")
    await update_task_run_status(db, task_id, run_id, "processing", user_id, block_id=block_id)
    await add_progress_update(db, task_id, run_id, user_id, {"type": "info", "content": "Executor has picked up the task and is starting execution."}, block_id=block_id)

    user_profile = await db.user_profiles.find_one({"user_id": user_id})
    personal_info = user_profile.get("userData", {}).get("personalInfo", {}) if user_profile else {}
    user_name = personal_info.get("name", "User")
    user_location_raw = personal_info.get("location", "Not specified")
    if isinstance(user_location_raw, dict):
        user_location = f"latitude: {user_location_raw.get('latitude')}, longitude: {user_location_raw.get('longitude')}"
    else:
        user_location = user_location_raw

    required_tools_from_plan = {step['tool'] for step in task.get('plan', [])}
    logger.info(f"Task {task_id}: Plan requires tools: {required_tools_from_plan}")
    
    user_integrations = user_profile.get("userData", {}).get("integrations", {}) if user_profile else {}
    
    active_mcp_servers = {}

    for tool_name in required_tools_from_plan:
        if tool_name not in INTEGRATIONS_CONFIG:
            logger.warning(f"Task {task_id}: Plan requires tool '{tool_name}' which is not in server config.")
            continue
        config = INTEGRATIONS_CONFIG[tool_name]
        mcp_config = config.get("mcp_server_config")
        if not mcp_config:
            continue
        is_builtin = config.get("auth_type") == "builtin"
        is_connected_via_oauth = user_integrations.get(tool_name, {}).get("connected", False)
        if is_builtin or is_connected_via_oauth:
            active_mcp_servers[mcp_config["name"]] = {"url": mcp_config["url"], "headers": {"X-User-ID": user_id}}
        else:
            logger.warning(f"Task {task_id}: Plan requires tool '{tool_name}' but it is not available/connected for user {user_id}.")
    tools_config = [{"mcpServers": active_mcp_servers}]
    logger.info(f"Task {task_id}: Executor configured with tools: {list(active_mcp_servers.keys())}")

    user_timezone_str = personal_info.get("timezone", "UTC")
    try:
        user_timezone = ZoneInfo(user_timezone_str)
    except ZoneInfoNotFoundError:
        logger.warning(f"Invalid timezone '{user_timezone_str}' for user {user_id}. Defaulting to UTC.")
        user_timezone = ZoneInfo("UTC")
    current_user_time = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

    plan_description = task.get("name", "Unnamed plan")
    original_context_str = json.dumps(original_context_data, indent=2, default=str) if original_context_data else "No original context provided."

    full_plan_prompt = (
        f"You are Sentient, a resourceful and autonomous executor agent. Your goal is to complete the user's request by intelligently following the provided plan.\n\n" # noqa
        f"**User Context:**\n- **User's Name:** {user_name}\n- **User's Location:** {user_location}\n- **Current Date & Time:** {current_user_time}\n\n"
        f"{trigger_event_prompt_section}"
        f"Your task ID is '{task_id}' and the current run ID is '{run_id}'.\n\n"
        f"The original context that triggered this plan is:\n---BEGIN CONTEXT---\n{original_context_str}\n---END CONTEXT---\n\n"
        f"**Primary Objective:** '{plan_description}'\n\n"
        f"**The Plan to Execute:**\n" + "\n".join([f"- Step {i+1}: Use the '{step['tool']}' tool to '{step['description']}'" for i, step in enumerate(task.get("plan", []))]) + "\n\n" # noqa
        "**EXECUTION STRATEGY:**\n" # noqa
        "1.  **Think Step-by-Step:** Before each action, you MUST explain your reasoning and what you are about to do. Your thought process MUST be wrapped in `<think>` tags.\n" # noqa
        "2.  **Execution Flow:** You MUST start by executing the first step of the plan. Do not summarize the plan or provide a final answer until you have executed all steps. Follow the plan sequentially. SEARCH FOR ANY RELEVANT CONTEXT THAT YOU NEED TO COMPLETE THE EXECUTION. If you are resuming a task after the user answered a clarifying question, the answered questions will be in the `clarifying_questions` field of the task context. Use this new information to proceed.\n" # noqa
        "3.  **Map Plan to Tools:** The plan provides a high-level tool name (e.g., 'gmail', 'gdrive'). You must map this to the specific functions available to you (e.g., `gmail_server-sendEmail`, `gdrive_server-gdrive_search`).\n" # noqa
        "4.  **Be Resourceful & Fill Gaps:** The plan is a guideline. If a step is missing information (e.g., an email address for a manager, a document name), your first action for that step MUST be to use the `memory-search_memory` tool to find the missing information. Do not proceed with incomplete information.\n" # noqa
        "5.  **Remember New Information:** If you discover a new, permanent fact about the user during your execution (e.g., you find their manager's email is 'boss@example.com'), you MUST use `memory-cud_memory` to save it.\n" # noqa
        "6.  **Handle Failures:** If a tool fails, analyze the error, think about an alternative approach, and try again. Do not give up easily. Your thought process and the error will be logged automatically.\n" # noqa
        "7.  **Provide a Final, Detailed Answer:** ONLY after all steps are successfully completed, you MUST provide a final, comprehensive answer to the user. This is not a tool call. Your final response MUST be wrapped in `<answer>` tags. For example: `<answer>I have successfully scheduled the meeting and sent an invitation to John Doe.</answer>`.\n" # noqa
        "8.  **Contact Information:** To find contact details like phone numbers or emails, use the `gpeople` tool before attempting to send an email or make a call.\n" # noqa
        "9. **Ask for Help**: If you have exhausted all tool options (including memory and search) and still lack critical information to proceed, you MUST stop and ask for clarification. Your entire final output in this case must be a single JSON object with the key `clarifying_questions`. The value should be a list of question objects, where each object has a `text` key. Example: `{\"clarifying_questions\": [{\"text\": \"What is the email address of the client?\"}]}`\n"
        "\nNow, begin your work. Think step-by-step and start executing the plan."
    )
    
    try:
        logger.info(f"Task {task_id}: Starting agent run.")
        initial_messages = [{'role': 'user', 'content': "Begin executing the plan. Follow your instructions meticulously."}]

        last_history_len = len(initial_messages)
        final_history = None

        for current_history in run_main_agent(
            system_message=full_plan_prompt,
            function_list=tools_config,
            messages=initial_messages
        ):
            final_history = current_history # Keep track of the latest state

            if not isinstance(current_history, list):
                continue

            # Process new messages for UI updates
            new_messages = current_history[last_history_len:]
            for msg in new_messages:
                updates_to_push = []
                if msg.get("role") == "assistant":
                    if msg.get("content"):
                        updates_to_push.extend(parse_agent_string_to_updates(msg["content"]))
                    if msg.get("function_call"):
                        fc = msg["function_call"]
                        params_str = fc.get("arguments", "{}")
                        params = JsonExtractor.extract_valid_json(params_str) or {"raw_parameters": params_str}
                        updates_to_push.append({
                            "type": "tool_call",
                            "tool_name": fc.get("name"),
                            "parameters": params
                        })
                elif msg.get("role") == "function":
                    result_str = msg.get("content", "{}")
                    result_content = JsonExtractor.extract_valid_json(result_str) or {"raw_result": result_str}
                    is_error = isinstance(result_content, dict) and result_content.get("status") == "failure"
                    updates_to_push.append({
                        "type": "tool_result",
                        "tool_name": msg.get("name"),
                        "result": result_content.get("result", result_content.get("error", result_content)),
                        "is_error": is_error
                    })
                
                for update in updates_to_push:
                    if update.get("type") != "thought":
                        # Use asyncio.create_task to send updates without blocking the agent loop
                        asyncio.create_task(add_progress_update(db, task_id, run_id, user_id, update, block_id))

            last_history_len = len(current_history)

        if not final_history:
            raise Exception("Agent run produced no history, indicating an immediate failure.")

        final_assistant_message = next((msg for msg in reversed(final_history) if msg.get("role") == "assistant"), None)
        
        has_final_answer = False
        final_content = final_assistant_message.get("content", "") if final_assistant_message else ""

        # Try to parse for clarification questions
        clarification_data = JsonExtractor.extract_valid_json(clean_llm_output(final_content))
        if isinstance(clarification_data, dict) and "clarifying_questions" in clarification_data and isinstance(clarification_data["clarifying_questions"], list):
            questions = clarification_data["clarifying_questions"]
            # Add unique IDs to questions if they don't have one
            for q in questions:
                if "question_id" not in q:
                    q["question_id"] = str(uuid.uuid4())

            logger.info(f"Task {task_id} requires user clarification. Suspending task.")
            await add_progress_update(db, task_id, run_id, user_id, {"type": "info", "content": "Task suspended. Waiting for user input."})

            # Update task with questions and set status to 'clarification_pending'
            await db.tasks.update_one(
                {"task_id": task_id},
                {"$set": {
                    "status": "clarification_pending",
                    "clarifying_questions": questions,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc)
                }}
            )

            # Notify user
            question_text = " ".join([q['text'] for q in questions])
            await notify_user(
                user_id,
                f"Task '{task.get('name', '...')}' needs your input: {question_text}",
                task_id,
                notification_type="taskNeedsClarification"
            )
            return {"status": "success", "message": "Task suspended for clarification."}
        if final_assistant_message and final_assistant_message.get("content"):
            parsed_updates = parse_agent_string_to_updates(final_assistant_message["content"])
            if any(upd.get("type") == "final_answer" for upd in parsed_updates):
                has_final_answer = True

        if not has_final_answer:
            error_message = "Agent finished execution without providing a final answer as required by its instructions. The task may be incomplete."
            logger.error(f"Task {task_id}: {error_message}. Final history: {final_history}")
            await add_progress_update(db, task_id, run_id, user_id, {"type": "error", "content": error_message}, block_id=block_id)
            await update_task_run_status(db, task_id, run_id, "error", user_id, details={"error": error_message}, block_id=block_id)
            
            from workers.tasks import calculate_next_run
            schedule_type = task.get('schedule', {}).get('type')
            if schedule_type == 'recurring':
                next_run_time, _ = calculate_next_run(task['schedule'], last_run=datetime.datetime.now(datetime.timezone.utc))
                await db.tasks.update_one({"_id": task["_id"]}, {"$set": {"status": "active", "next_execution_at": next_run_time}})
            elif schedule_type == 'triggered':
                await db.tasks.update_one({"_id": task["_id"]}, {"$set": {"status": "active", "next_execution_at": None}})
            else:
                await db.tasks.update_one({"_id": task["_id"]}, {"$set": {"status": "error", "next_execution_at": None}})
            return {"status": "error", "message": error_message}

        # If we have a final answer, the execution was successful.
        logger.info(f"Task {task_id} execution phase completed. Dispatching to result generator.")
        await add_progress_update(db, task_id, run_id, user_id, {"type": "info", "content": "Execution finished. Generating final report..."}, block_id=block_id)
        await update_task_run_status(db, task_id, run_id, "completed", user_id, block_id=block_id)
        capture_event(user_id, "task_execution_succeeded", {"task_id": task_id, "run_id": run_id})

        # Call the new result generator task
        generate_task_result.delay(task_id, run_id, user_id)

        from workers.tasks import calculate_next_run
        schedule_type = task.get('schedule', {}).get('type')
        if schedule_type == 'recurring':
            next_run_time, _ = calculate_next_run(task['schedule'], last_run=datetime.datetime.now(datetime.timezone.utc))
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "active", "next_execution_at": next_run_time}}
            )
            logger.info(f"Executor: Rescheduled recurring task {task_id} for {next_run_time}.")
        elif schedule_type == 'triggered':
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "active", "next_execution_at": None}}
            )
            logger.info(f"Executor: Reset triggered task {task_id} to 'active' state.")
        else: # One-off task
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "completed", "next_execution_at": None}}
            )

        return {"status": "success", "message": "Task execution complete, generating report."}

    except LLMProviderDownError as e:
        error_message = "Sorry, our AI provider is currently down. Please try again later."
        logger.error(f"Task {task_id}: LLM provider down: {e}", exc_info=True)
        await add_progress_update(db, task_id, run_id, user_id, {"type": "error", "content": error_message}, block_id=block_id)
        await update_task_run_status(db, task_id, run_id, "error", user_id, details={"error": error_message}, block_id=block_id)
    except Exception as e:
        error_message = f"Executor agent failed: {str(e)}"
        logger.error(f"Task {task_id}: {error_message}", exc_info=True)
        await add_progress_update(db, task_id, run_id, user_id, {"type": "error", "content": f"An error occurred during execution: {error_message}"}, block_id=block_id)
        await update_task_run_status(db, task_id, run_id, "error", user_id, details={"error": error_message}, block_id=block_id)
        
        # Also update parent task status on failure
        from workers.tasks import calculate_next_run
        schedule_type = task.get('schedule', {}).get('type')
        if schedule_type == 'recurring':
            # It failed, but we should still reschedule it. The failure is logged in the run.
            next_run_time, _ = calculate_next_run(task['schedule'], last_run=datetime.datetime.now(datetime.timezone.utc))
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "active", "next_execution_at": next_run_time}}
            )
        elif schedule_type == 'triggered':
            # If a triggered run fails, it should also go back to 'active' to wait for the next trigger.
            # The error is logged in the specific run.
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "active", "next_execution_at": None}}
            )
            logger.info(f"Executor: Reset failed triggered task {task_id} to 'active' state.")
        else:
            await db.tasks.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "error", "next_execution_at": None}}
            )
        return {"status": "error", "message": error_message}

@celery_app.task(name="aggregate_results_callback")
def aggregate_results_callback(results, parent_task_id: str, user_id: str, parent_run_id: str):
    """Celery callback task to aggregate results from a swarm chord."""
    logger.info(f"Aggregating results for swarm task {parent_task_id}. Received {len(results)} results.")
    run_async(async_aggregate_results(results, parent_task_id, user_id, parent_run_id))

async def async_aggregate_results(results, parent_task_id: str, user_id: str, parent_run_id: str):
    db = get_db_client()
    try:
        # Check for errors in results
        failed_count = sum(1 for r in results if isinstance(r, dict) and 'error' in r)
        final_status = "completed_with_errors" if failed_count > 0 else "completed"

        progress_update = {
            "worker_id": "aggregator",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "status": "aggregating",
            "message": f"All {len(results)} agents have completed. Generating final report."
        }

        task = await db.tasks.find_one({"task_id": parent_task_id, "user_id": user_id})
        if not task:
            logger.error(f"Cannot aggregate results: Task {parent_task_id} not found.")
            return

        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        decrypt_doc(task, SENSITIVE_TASK_FIELDS)

        runs = task.get("runs", [])
        if not isinstance(runs, list):
            logger.error(f"Task {parent_task_id} 'runs' field is not a list. Cannot aggregate.")
            return

        run_found = False
        for run in runs:
            if run.get("run_id") == parent_run_id:
                run["status"] = final_status
                if "progress_updates" not in run or not isinstance(run["progress_updates"], list):
                    run["progress_updates"] = []
                run["progress_updates"].append(progress_update)
                run_found = True
                break

        if not run_found:
            logger.warning(f"Run {parent_run_id} not found in task {parent_task_id} to aggregate results.")
            return

        update_payload = {
            "status": final_status,
            "runs": runs
        }
        encrypt_doc(update_payload, ["runs"])

        await db.tasks.update_one(
            {"task_id": parent_task_id},
            {"$set": update_payload}
        )
        
        generate_task_result.delay(parent_task_id, parent_run_id, user_id, aggregated_results=results)
        task_name = task.get("name", "Swarm Task")

        await notify_user(user_id, f"Swarm task '{task_name}' has completed.", parent_task_id, notification_type="taskCompleted")

    except Exception as e:
        logger.error(f"Error in aggregate_results_callback for task {parent_task_id}: {e}", exc_info=True)
        # Update task to error state if aggregation fails
        await db.tasks.update_one(
            {"task_id": parent_task_id},
            {"$set": {"status": "error", "error": f"Failed during result aggregation: {str(e)}"}}
        )

@celery_app.task(name="generate_task_result")
def generate_task_result(task_id: str, run_id: str, user_id: str, aggregated_results: Optional[List[Any]] = None):
    """
    Analyzes a completed task run and generates a structured, user-friendly result.
    """
    logger.info(f"Generating final result for task {task_id}, run {run_id}")
    run_async(async_generate_task_result(task_id, run_id, user_id, aggregated_results))

async def async_generate_task_result(task_id: str, run_id: str, user_id: str, aggregated_results: Optional[List[Any]] = None):
    db = get_db_client()
    try:
        task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})
        SENSITIVE_TASK_FIELDS = ["name", "description", "plan", "runs", "original_context", "chat_history", "error", "clarifying_questions", "result", "swarm_details", "orchestrator_state", "dynamic_plan", "clarification_requests", "execution_log"]
        decrypt_doc(task, SENSITIVE_TASK_FIELDS)

        if not task:
            logger.error(f"ResultGenerator: Task {task_id} not found.")
            return

        current_run = next((r for r in task.get("runs", []) if r["run_id"] == run_id), None)
        if not current_run:
            logger.error(f"ResultGenerator: Run {run_id} not found in task {task_id}.")
            return

        # 1. Compile context for the LLM
        context_for_llm = {
            "goal": task.get("name", "No goal specified."),
            "plan": current_run.get("plan", task.get("plan", [])), # Use run-specific plan if available
            "execution_log": current_run.get("progress_updates", []),
            "aggregated_results": aggregated_results # This will be present for swarm tasks
        }

        # 2. Call the Result Generator Agent
        messages = [{'role': 'user', 'content': json.dumps(context_for_llm, default=str)}]

        response_str = ""
        # This is a non-streaming call, so we just need the final result.
        for chunk in run_main_agent(system_message=RESULT_GENERATOR_SYSTEM_PROMPT, function_list=[], messages=messages):
            if isinstance(chunk, list) and chunk and chunk[-1].get("role") == "assistant":
                response_str = chunk[-1].get("content", "")

        if not response_str:
            raise Exception("Result Generator agent returned an empty response.")

        structured_result = JsonExtractor.extract_valid_json(clean_llm_output(response_str))
        if not structured_result:
            # Fallback: use the raw text as the summary
            structured_result = {"summary": response_str, "links_created": [], "links_found": [], "files_created": [], "tools_used": []}
            logger.warning(f"ResultGenerator for task {task_id} failed to parse JSON, using raw output as summary.")

        # 3. Update the task run with the structured result
        runs = task.get("runs", [])
        if not isinstance(runs, list):
            logger.error(f"Task {task_id} 'runs' field is not a list. Cannot save result.")
            return

        run_found = False
        for run in runs:
            if run.get("run_id") == run_id:
                run["result"] = structured_result
                run_found = True
                break

        if not run_found:
            logger.warning(f"Run {run_id} not found in task {task_id} to save result.")
            return

        update_payload = {"runs": runs}
        encrypt_doc(update_payload, ["runs"])

        await db.tasks.update_one(
            {"task_id": task_id, "user_id": user_id},
            {"$set": update_payload}
        )
        logger.info(f"Successfully generated and saved structured result for task {task_id}, run {run_id}.")
        await push_task_list_update(user_id, task_id, run_id)
    except LLMProviderDownError as e:
        logger.error(f"LLM provider down during result generation for task {task_id}: {e}", exc_info=True)
        error_result = {"summary": "Sorry, our AI provider is currently down, so a final report could not be generated."}
        # Fetch, modify, save pattern for error case
        task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})
        if task:
            decrypt_doc(task, ["runs"])
            runs = task.get("runs", [])
            if isinstance(runs, list):
                for run in runs:
                    if run.get("run_id") == run_id:
                        run["result"] = error_result
                        break
                update_payload = {"runs": runs}
                encrypt_doc(update_payload, ["runs"])
                await db.tasks.update_one({"task_id": task_id, "user_id": user_id}, {"$set": update_payload})
    except Exception as e:
        logger.error(f"Error in async_generate_task_result for task {task_id}: {e}", exc_info=True)
        error_result = {"summary": f"Failed to generate final report: {str(e)}"}
        # Fetch, modify, save pattern for error case
        task = await db.tasks.find_one({"task_id": task_id, "user_id": user_id})
        if task:
            decrypt_doc(task, ["runs"])
            runs = task.get("runs", [])
            if isinstance(runs, list):
                for run in runs:
                    if run.get("run_id") == run_id:
                        run["result"] = error_result
                        break
                update_payload = {"runs": runs}
                encrypt_doc(update_payload, ["runs"])
                await db.tasks.update_one({"task_id": task_id, "user_id": user_id}, {"$set": update_payload})

@celery_app.task(name="run_single_item_worker", bind=True)
def run_single_item_worker(self, parent_task_id: str, user_id: str, item: Any, worker_prompt: str, worker_tools: List[str]):
    """
    A Celery worker that executes a sub-task for a single item from a collection.
    This worker runs a self-contained agent to fulfill the worker_prompt using a specific set of tools.
    """
    worker_id = self.request.id
    logger.info(f"Running single item worker {worker_id} for parent task {parent_task_id} on item: {str(item)[:100]}")
    return run_async(async_run_single_item_worker(parent_task_id, user_id, item, worker_prompt, worker_tools, worker_id))

async def async_run_single_item_worker(parent_task_id: str, user_id: str, item: Any, worker_prompt: str, worker_tools: List[str], worker_id: str):
    """
    Async logic for the single item worker.
    """
    db = get_db_client()

    async def push_update(status: str, message: str):
        update = {
            "worker_id": worker_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "status": status,
            "message": message
        }
        await db.tasks.update_one(
            {"task_id": parent_task_id},
            {
                "$push": {"swarm_details.progress_updates": update},
                "$inc": {"swarm_details.completed_agents": 1 if status in ["completed", "error"] else 0}
            }
        )
        await push_task_list_update(user_id, parent_task_id, "swarm_progress")

    try:
        # 1. Get user context
        user_profile = await db.user_profiles.find_one({"user_id": user_id})
        user_integrations = user_profile.get("userData", {}).get("integrations", {}) if user_profile else {}

        # 2. Configure tools for the sub-agent based on the provided list
        active_mcp_servers = {}
        for tool_name in worker_tools:
            config = INTEGRATIONS_CONFIG.get(tool_name)
            if not config:
                logger.warning(f"Worker for user {user_id} requested unknown tool '{tool_name}'.")
                continue
            
            mcp_config = config.get("mcp_server_config")
            if not mcp_config: continue

            is_builtin = config.get("auth_type") == "builtin"
            is_connected = user_integrations.get(tool_name, {}).get("connected", False)

            if is_builtin or is_connected:
                active_mcp_servers[mcp_config["name"]] = {"url": mcp_config["url"], "headers": {"X-User-ID": user_id}}
            else:
                logger.warning(f"Worker for user {user_id} needs tool '{tool_name}' but it is not connected/available.")

        tools_config = [{"mcpServers": active_mcp_servers}]

        # 3. Construct prompts for the sub-agent
        system_prompt = (
            "You are an autonomous sub-agent. Your goal is to complete a specific task given to you as part of a larger parallel operation. "
            "You have access to a specific, limited suite of tools. Follow the user's prompt precisely. "
            "Your final output should be a single, concise result (e.g., a string, a number, a JSON object, or null). Do not add conversational filler. "
            "If you generate a final answer, wrap it in <answer> tags."
        )
        
        item_context = json.dumps(item, indent=2, default=str)
        full_worker_prompt = f"**Task:**\n{worker_prompt}\n\n**Input Data for this Task:**\n```json\n{item_context}\n```"
        
        messages = [{'role': 'user', 'content': full_worker_prompt}]

        await push_update("processing", f"Starting work on item: {str(item)[:100]}")

        # 4. Run the agent
        agent = Assistant(llm=llm_cfg, function_list=tools_config, system_message=system_prompt)
        
        final_content = ""
        final_response_list = []
        for response in agent.run(messages=messages):
            if isinstance(response, list) and response and response[-1].get("role") == "assistant":
                final_content = response[-1].get("content", "")
            final_response_list = response

        # 5. Parse and return the result
        answer_match = re.search(r'<answer>([\s\S]*?)</answer>', final_content, re.DOTALL)
        if answer_match:
            result_str = answer_match.group(1).strip()
            parsed_result = JsonExtractor.extract_valid_json(result_str)
            if parsed_result is not None:
                final_result = parsed_result
            elif result_str.lower() == 'null':
                final_result = None
            else:
                final_result = result_str
        else:
            last_message = final_response_list[-1] if final_response_list else {}
            if last_message.get("role") == "function":
                tool_result = JsonExtractor.extract_valid_json(last_message.get("content", "{}"))
                if isinstance(tool_result, dict):
                    final_result = tool_result.get("result")
                else:
                    final_result = last_message.get("content")
            else:
                cleaned_content = clean_llm_output(final_content)
                if cleaned_content.lower() == 'null':
                    final_result = None
                else:
                    final_result = cleaned_content
        
        await push_update("completed", f"Finished work. Result: {str(final_result)[:100]}")
        return final_result

    except LLMProviderDownError as e:
        error_str = "Sorry, our AI provider is currently down. Please try again later."
        logger.error(f"LLM provider down in single item worker {worker_id} for task {parent_task_id}: {e}", exc_info=True)
        await push_update("error", error_str)
        return {"error": error_str, "item": item}
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error in single item worker {worker_id} for task {parent_task_id}: {e}", exc_info=True)
        await push_update("error", f"An error occurred: {error_str}")
        return {"error": error_str, "item": item}