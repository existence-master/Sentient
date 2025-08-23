import logging
from typing import Dict, Any
import datetime
from workers.celery_app import celery_app
from workers.utils.api_client import notify_user
from workers.planner.db import PlannerMongoManager
import httpx
import json
from main.llm import run_agent as run_main_agent
from mcp_hub.orchestrator.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT, STEP_PLANNING_PROMPT,
    COMPLETION_EVALUATION_PROMPT, FOLLOW_UP_DECISION_PROMPT
)
from main.config import INTEGRATIONS_CONFIG

logger = logging.getLogger(__name__)

# Helper to run async code in Celery's sync context
def run_async(coro):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        from mcp_hub.memory.db import close_db_pool_for_loop
        loop.run_until_complete(close_db_pool_for_loop(loop))
        loop.close()
        asyncio.set_event_loop(None)

@celery_app.task(name="start_long_form_task")
def start_long_form_task(task_id: str, user_id: str):
    """Initialize and start a long-form task."""
    logger.info(f"Initializing long-form task {task_id} for user {user_id}")
    run_async(async_start_long_form_task(task_id, user_id))

async def async_start_long_form_task(task_id: str, user_id: str):
    db_manager = PlannerMongoManager()
    try:
        await db_manager.update_task_field(task_id, user_id, {
            "orchestrator_state.current_state": "PLANNING"
        })
        # Immediately trigger the first orchestrator cycle to generate the initial plan
        execute_orchestrator_cycle.delay(task_id)
    except Exception as e:
        logger.error(f"Error starting long-form task {task_id}: {e}", exc_info=True)
    finally:
        await db_manager.close()

@celery_app.task(name="execute_orchestrator_cycle")
def execute_orchestrator_cycle(task_id: str):
    """Main orchestrator execution cycle."""
    logger.info(f"Executing orchestrator cycle for task {task_id}")
    run_async(async_execute_orchestrator_cycle(task_id))

async def async_execute_orchestrator_cycle(task_id: str):
    db_manager = PlannerMongoManager()
    try:
        task = await db_manager.get_task(task_id)
        if not task:
            logger.error(f"Orchestrator cycle: Task {task_id} not found.")
            return
        
        user_id = task.get("user_id")
        current_state = task.get("orchestrator_state", {}).get("current_state")

        if current_state in ["COMPLETED", "FAILED", "SUSPENDED", "PAUSED", "WAITING"]:
            logger.info(f"Orchestrator cycle for task {task_id} skipped. State is '{current_state}'.")
            return

        # 1. Construct context for the agent
        orchestrator_state = task.get("orchestrator_state", {})
        context_for_prompt = {
            "task_id": task_id,
            "main_goal": orchestrator_state.get("main_goal"),
            "current_state": current_state,
            "dynamic_plan": json.dumps(task.get("dynamic_plan", []), default=str),
            "context_store": orchestrator_state.get("context_store", {}),
            "execution_log": task.get("execution_log", [])[-5:], # Last 5 logs
            "clarification_history": json.dumps(task.get("clarification_requests", []), default=str)
        }

        # 2. Determine the right prompt based on the current state
        user_prompt = ""
        if current_state == "WAITING":
            waiting_config = orchestrator_state.get("waiting_config", {})

            started_at = waiting_config.get("started_at", datetime.datetime.now(datetime.timezone.utc))
            # Ensure the datetime from DB is timezone-aware before subtraction
            if started_at and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=datetime.timezone.utc)

            time_elapsed_delta = datetime.datetime.now(datetime.timezone.utc) - started_at

            user_prompt = FOLLOW_UP_DECISION_PROMPT.format( # noqa: E501
                waiting_for=waiting_config.get("waiting_for"),
                time_elapsed=str(time_elapsed_delta),
                previous_attempts=waiting_config.get("current_retries", 0),
                context=json.dumps({k: v for k, v in context_for_prompt.items() if k != 'task_id'}, default=str)
            )
        else: # PLANNING or ACTIVE
            user_prompt = STEP_PLANNING_PROMPT.format(**context_for_prompt)

        # 3. Run the orchestrator agent to get the next tool call
        orchestrator_config = INTEGRATIONS_CONFIG.get("orchestrator", {})
        mcp_config = orchestrator_config.get("mcp_server_config", {})

        if not mcp_config or not mcp_config.get("url"):
            raise ValueError("Orchestrator MCP server URL is not configured in INTEGRATIONS_CONFIG.")

        mcp_servers_to_use = {
            mcp_config["name"]: {
                "url": mcp_config["url"],
                "headers": {"X-User-ID": user_id, "X-Task-ID": task_id}
            }
        }
        function_list = [{"mcpServers": mcp_servers_to_use}]

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            **context_for_prompt
        )
        messages = [{'role': 'user', 'content': user_prompt}]
        
        final_agent_response = None
        for chunk in run_main_agent(system_message=system_prompt, function_list=function_list, messages=messages):
            if isinstance(chunk, list) and chunk:
                final_agent_response = chunk

        # Find the last tool call made by the agent in this turn
        last_tool_call_message = None
        if final_agent_response:
            last_tool_call_message = next((msg for msg in reversed(final_agent_response) if msg.get("role") == "assistant" and msg.get("function_call")), None)

        if not last_tool_call_message:
            logger.error(f"Orchestrator agent for task {task_id} did not produce a valid tool call. Last response: {final_agent_response}")
            raise Exception("Orchestrator agent failed to decide on a next step.")

        # 4. Extract the tool call result from the agent's response history
        tool_call = last_tool_call_message["function_call"]
        tool_name = tool_call.get("name")

        logger.info(f"Orchestrator agent for task {task_id} decided to call tool: {tool_name}")

        # Find the corresponding function/tool result message in the history
        tool_result_message = next((msg for msg in reversed(final_agent_response) if msg.get("role") == "function" and msg.get("name") == tool_name), None)

        if not tool_result_message:
            logger.error(f"Orchestrator agent for task {task_id} made a tool call but no result was found. History: {final_agent_response}")
            raise Exception("Orchestrator agent tool call did not yield a result.")

        tool_content = tool_result_message.get("content", "{}")
        tool_result = None
        if not tool_content or not tool_content.strip():
            logger.warning(f"Orchestrator tool '{tool_name}' for task {task_id} returned empty content. Assuming failure.")
            tool_result = {"status": "failure", "error": "Tool returned an empty response."}
        else:
            try:
                tool_result = json.loads(tool_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from tool '{tool_name}' for task {task_id}. Content: {tool_content}")
                raise Exception(f"Orchestrator tool '{tool_name}' returned invalid JSON: {tool_content}")

        if tool_result.get("status") != "success":
            raise Exception(f"Orchestrator tool '{tool_name}' failed: {tool_result.get('error')}")

        logger.info(f"Orchestrator tool '{tool_name}' for task {task_id} executed successfully.")

        # 5. Reschedule the next cycle if the task is still active
        updated_task = await db_manager.get_task(task_id)
        current_state = updated_task.get("orchestrator_state", {}).get("current_state")
        if current_state == "ACTIVE":
            # Re-queue the next cycle with a short delay if the task is still actively processing steps.
            execute_orchestrator_cycle.apply_async(args=[task_id], countdown=5)

    except Exception as e:
        logger.error(f"Error in orchestrator cycle for task {task_id}: {e}", exc_info=True)
        await db_manager.update_task_field(task_id, user_id, {"orchestrator_state.current_state": "FAILED", "error": f"Orchestrator cycle failed: {str(e)}"})
    finally:
        await db_manager.close()

@celery_app.task(name="handle_waiting_timeout")
def handle_waiting_timeout(task_id: str, waiting_type: str):
    """Handle timeout for waiting tasks."""
    logger.info(f"Handling timeout for task {task_id}, waiting for {waiting_type}")
    # Logic to decide on follow-up action (retry, escalate, etc.)
    # This would trigger another orchestrator cycle to re-evaluate.
    execute_orchestrator_cycle.delay(task_id)

@celery_app.task(name="process_external_trigger")
def process_external_trigger(task_id: str, trigger_data: Dict):
    """Process external triggers (emails, calendar events, etc.)"""
    logger.info(f"Processing external trigger for task {task_id}")
    run_async(async_process_external_trigger(task_id, trigger_data))

async def async_process_external_trigger(task_id: str, trigger_data: Dict):
    db_manager = PlannerMongoManager()
    try:
        task = await db_manager.get_task(task_id)
        if not task:
            logger.error(f"Cannot process external trigger for task {task_id}: Task not found.")
            return
        user_id = task.get("user_id")
        if not user_id:
            logger.error(f"Cannot process external trigger for task {task_id}: user_id not found in task.")
            return
        # 1. Update context store with new information
        await db_manager.update_task_field(task_id, user_id, {f"orchestrator_state.context_store.trigger_{datetime.datetime.now(datetime.timezone.utc).isoformat()}": trigger_data})
        # 2. Resume task execution
        execute_orchestrator_cycle.delay(task_id)
    finally:
        await db_manager.close()