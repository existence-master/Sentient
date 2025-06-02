import asyncio
import datetime
import os
import json
import traceback
from typing import Dict, List, Optional, Union, Any
import uuid

# MongoDB specific imports
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.results import UpdateResult, DeleteResult

# Import TaskQueue from its new location
from server.agents.base import TaskQueue

# Define the signal constant
APPROVAL_PENDING_SIGNAL = "APPROVAL_PENDING"

class AgentTaskProcessor:
    def __init__(self, task_queue: TaskQueue, tool_handlers: Dict, 
                 load_user_profile_func: callable, add_message_to_db_func: callable,
                 graph_driver_instance: Any, embed_model_instance: Any,
                 reflection_runnable_instance: Any, agent_runnable_instance: Any,
                 internet_query_reframe_runnable_instance: Any, internet_summary_runnable_instance: Any,
                 query_user_profile_func: callable, text_conversion_runnable_instance: Any,
                 query_classification_runnable_instance: Any,
                 get_reframed_internet_query_func: callable, get_search_results_func: callable,
                 get_search_summary_func: callable):
        self.task_queue = task_queue
        self.tool_handlers = tool_handlers
        self.load_user_profile = load_user_profile_func
        self.add_message_to_db = add_message_to_db_func
        self.graph_driver = graph_driver_instance
        self.embed_model = embed_model_instance
        self.reflection_runnable = reflection_runnable_instance
        self.agent_runnable = agent_runnable_instance
        self.internet_query_reframe_runnable = internet_query_reframe_runnable_instance
        self.internet_summary_runnable = internet_summary_runnable_instance
        self.query_user_profile = query_user_profile_func
        self.text_conversion_runnable = text_conversion_runnable_instance
        self.query_classification_runnable = query_classification_runnable_instance
        self.get_reframed_internet_query = get_reframed_internet_query_func
        self.get_search_results = get_search_results_func
        self.get_search_summary = get_search_summary_func
        print(f"[AGENT_TASK_PROCESSOR] AgentTaskProcessor initialized.")

    async def cleanup_tasks_periodically(self):
        print(f"[TASK_CLEANUP] Starting periodic task cleanup loop.")
        while True:
            await asyncio.sleep(3600) # Clean up every hour
            try:
                await self.task_queue.delete_old_completed_tasks()
                print(f"[TASK_CLEANUP] Cleanup complete.")
            except Exception as e:
                print(f"[ERROR] Task cleanup error: {e}")
                traceback.print_exc()


    async def process_queue(self):
        print(f"[TASK_PROCESSOR] Starting task processing loop.")
        while True:
            task = await self.task_queue.get_next_task()
            if task:
                task_id = task.get("task_id", "N/A")
                user_id = task.get("user_id", "N/A")
                chat_id = task.get("chat_id", "N/A")

                if user_id == "N/A":
                    await self.task_queue.complete_task(user_id, task_id, error="Task missing user_id", status="error") # user_id might be "N/A" here, but task_id should exist
                    continue
                try:
                    self.task_queue.current_task_execution = asyncio.create_task(self.execute_agent_task(user_id, task))
                    result = await self.task_queue.current_task_execution
                    
                    if result != APPROVAL_PENDING_SIGNAL:
                        if chat_id != "N/A": # Ensure chat_id is valid before adding messages
                            await self.add_message_to_db(user_id, chat_id, task["description"], is_user=True, is_visible=False)
                            await self.add_message_to_db(user_id, chat_id, result, is_user=False, is_visible=True, type="tool_result", task=task["description"], agentsUsed=True)
                        await self.task_queue.complete_task(user_id, task_id, result=result)
                
                except asyncio.CancelledError:
                    print(f"[TASK_PROCESSOR] Task {task_id} for user {user_id} was cancelled.")
                    await self.task_queue.complete_task(user_id, task_id, error="Task cancelled", status="cancelled")
                except Exception as e:
                    print(f"[TASK_PROCESSOR_ERROR] Error processing task {task_id} for user {user_id}: {e}")
                    await self.task_queue.complete_task(user_id, task_id, error=str(e), status="error")
                    traceback.print_exc()
                finally:
                    self.task_queue.current_task_execution = None # Reset current execution tracking
            else:
                await asyncio.sleep(0.1) # Wait briefly if no task

    async def execute_agent_task(self, user_id: str, task: dict) -> str:
        task_id = task.get("task_id", "N/A")
        print(f"[AGENT_EXEC] Executing task {task_id} for User: {user_id}...")
        user_profile = await self.load_user_profile(user_id)
        username = user_profile.get("userData", {}).get("personalInfo", {}).get("name", "User")
        transformed_input = task.get("description", "")
        use_personal_context = task.get("use_personal_context", False)
        internet_search_needed = task.get("internet", "None") != "None"
        user_context_str, internet_context_str = None, None
        if use_personal_context:
            try:
                if self.graph_driver and self.embed_model and self.query_user_profile and self.text_conversion_runnable and self.query_classification_runnable:
                    user_context_str = await asyncio.to_thread(self.query_user_profile, user_id, transformed_input, self.graph_driver, self.embed_model, self.text_conversion_runnable, self.query_classification_runnable)
                else:
                    user_context_str = "User context unavailable (dependencies missing)."
            except Exception as e:
                print(f"[AGENT_EXEC_ERROR] Error retrieving user context for task {task_id}: {e}")
                user_context_str = f"Error retrieving user context: {e}"
        if internet_search_needed:
            try:
                if self.internet_query_reframe_runnable and self.get_reframed_internet_query and self.get_search_results and self.get_search_summary and self.internet_summary_runnable:
                    reframed = self.get_reframed_internet_query(self.internet_query_reframe_runnable, transformed_input)
                    results = self.get_search_results(reframed)
                    internet_context_str = self.get_search_summary(self.internet_summary_runnable, results)
                else:
                    internet_context_str = "Internet search unavailable (dependencies missing)."
            except Exception as e:
                print(f"[AGENT_EXEC_ERROR] Error retrieving internet context for task {task_id}: {e}")
                internet_context_str = f"Error retrieving internet context: {e}"
        agent_input = {"query": transformed_input, "name": username, "user_context": user_context_str, "internet_context": internet_context_str}
        try:
            response = self.agent_runnable.invoke(agent_input)
        except Exception as e:
            print(f"[AGENT_EXEC_ERROR] Agent invocation failed for task {task_id}: {e}")
            return f"Error: Agent failed: {e}"
        tool_calls = response.get("tool_calls", []) if isinstance(response, dict) else (response if isinstance(response, list) else [])
        if not tool_calls and isinstance(response, str):
            return response # Direct answer
        if not tool_calls:
            return "Agent decided no tools were needed or failed to propose tools."
        all_tool_results = []
        previous_tool_result = None
        for i, tool_call_item in enumerate(tool_calls):
            tool_content = tool_call_item.get("content") if isinstance(tool_call_item, dict) and tool_call_item.get("response_type") == "tool_call" else tool_call_item
            if not isinstance(tool_content, dict):
                all_tool_results.append({"tool_name": "unknown", "tool_result": "Invalid tool call format", "status": "error"})
                continue
            tool_name = tool_content.get("tool_name")
            task_instruction = tool_content.get("task_instruction")
            if not tool_name or not task_instruction:
                all_tool_results.append({"tool_name": tool_name or "unknown", "tool_result": "Missing name/instruction", "status": "error"})
                continue
            tool_handler = self.tool_handlers.get(tool_name)
            if not tool_handler:
                all_tool_results.append({"tool_name": tool_name, "tool_result": f"Tool '{tool_name}' not found.", "status": "error"})
                continue
            tool_input = {"input": str(task_instruction), "user_id": user_id} # Pass user_id to tool_handler
            if tool_content.get("previous_tool_response", False) and previous_tool_result:
                tool_input["previous_tool_response"] = previous_tool_result
            try:
                tool_result_main = await tool_handler(tool_input)
            except Exception as e:
                print(f"[AGENT_EXEC_ERROR] Error executing tool '{tool_name}' for task {task_id}: {e}")
                tool_result_main = f"Error executing tool '{tool_name}': {e}"
                traceback.print_exc()
            if isinstance(tool_result_main, dict) and tool_result_main.get("action") == "approve":
                await self.task_queue.set_task_approval_pending(user_id, task_id, tool_result_main.get("tool_call", {})) # Pass user_id here
                return APPROVAL_PENDING_SIGNAL
            else:
                tool_result = tool_result_main.get("tool_result", tool_result_main) if isinstance(tool_result_main, dict) else tool_result_main
                previous_tool_result = tool_result
                all_tool_results.append({"tool_name": tool_name, "tool_result": tool_result, "status": "success" if not (isinstance(tool_result_main, dict) and tool_result_main.get("status") == "failure") else "failure"})
        if not all_tool_results:
            return "No successful tool actions performed."
        try:
            final_result_str = self.reflection_runnable.invoke({"tool_results": all_tool_results})
        except Exception as e:
            print(f"[AGENT_EXEC_ERROR] Error in reflection for task {task_id}: {e}")
            final_result_str = f"Error generating final summary: {e}"
        return final_result_str