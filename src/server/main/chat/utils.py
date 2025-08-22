import datetime
import uuid
import os
import json
import asyncio
import logging
import datetime
import threading
import time
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Callable, Coroutine, Union
import httpx
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI, APIError

from main.chat.prompts import STAGE_1_SYSTEM_PROMPT, STAGE_2_SYSTEM_PROMPT # noqa: E501
from main.db import MongoManager
from main.llm import run_agent, LLMProviderDownError
from main.config import (INTEGRATIONS_CONFIG, ENVIRONMENT, OPENAI_API_KEY, OPENAI_API_BASE_URL, OPENAI_MODEL_NAME)
from json_extractor import JsonExtractor
from workers.utils.text_utils import clean_llm_output
import re

logger = logging.getLogger(__name__)

@register_tool('json_validator')
class JsonValidatorTool(BaseTool):
    description = (
        "Validates and cleans a string that is supposed to be a JSON object or list. "
        "Use this tool to fix any syntax errors in a JSON string before passing it to another tool that requires valid JSON."
    )
    parameters = [{
        'name': 'json_string',
        'type': 'string',
        'description': 'The string to be validated and cleaned as JSON.',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        logger.info(f"JsonValidatorTool called with params: {params}")
        try:
            if isinstance(params, dict):
                 parsed_params = params
            else:
                 parsed_params = JsonExtractor.extract_valid_json(params)
            if not parsed_params:
                return json.dumps({"status": "failure", "error": "Invalid JSON in params."})
            json_string_to_validate = parsed_params.get('json_string', '')
            if not json_string_to_validate:
                return json.dumps({"status": "failure", "error": "Input json_string is empty."})
            valid_json = JsonExtractor.extract_valid_json(json_string_to_validate)
            if valid_json:
                cleaned_json_string = json.dumps(valid_json)
                logger.info(f"Successfully cleaned JSON: {cleaned_json_string}")
                return json.dumps({"status": "success", "cleaned_json": cleaned_json_string})
            else:
                logger.warning(f"Could not extract valid JSON from: {json_string_to_validate}")
                return json.dumps({"status": "failure", "error": "Could not extract any valid JSON from the input string."})
        except Exception as e:
            logger.error(f"JsonValidatorTool encountered an unexpected error: {e}", exc_info=True)
            return json.dumps({"status": "failure", "error": str(e)})

async def _get_stage1_response(messages: List[Dict[str, Any]], connected_tools_map: Dict[str, str], disconnected_tools_map: Dict[str, str], user_id: str) -> Dict[str, Any]:
    """
    Uses the Stage 1 LLM to detect topic changes and select relevant tools.
    Returns a dictionary containing a 'topic_changed' boolean and a 'tools' list.
    """
    if not OPENAI_API_KEY:
        raise ValueError("No OpenAI API key configured for Stage 1.")

    formatted_messages = [
        {"role": "system", "content": STAGE_1_SYSTEM_PROMPT}
    ]
    for msg in messages:
        if 'role' in msg and 'content' in msg:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    client = OpenAI(base_url=OPENAI_API_BASE_URL, api_key=OPENAI_API_KEY)

    try:
        logger.info(f"Stage 1: Attempting LLM call")

        def sync_api_call():
            return client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=formatted_messages,
            )

        completion = await asyncio.to_thread(sync_api_call)

        # FIX: Add a check to handle cases where the API returns a successful
        # response but with an empty 'choices' list (e.g., due to content filtering).
        if not completion.choices:
            raise Exception("LLM response was successful but contained no choices.")

        final_content_str = completion.choices[0].message.content

        logger.info(f"Stage 1 LLM output for user {user_id}: {final_content_str}")
        cleaned_output = clean_llm_output(final_content_str)
        logger.info(f"Cleaned Stage 1 output for user {user_id}: {cleaned_output}")
        stage1_result = JsonExtractor.extract_valid_json(cleaned_output)

        if isinstance(stage1_result, dict) and "topic_changed" in stage1_result and "tools" in stage1_result:
            selected_tools = stage1_result.get("tools", [])
            connected_tools_selected = [tool for tool in selected_tools if tool in connected_tools_map]
            disconnected_tools_selected = [tool for tool in selected_tools if tool in disconnected_tools_map]
            return {
                "topic_changed": stage1_result.get("topic_changed", False),
                "connected_tools": connected_tools_selected,
                "disconnected_tools": disconnected_tools_selected
            }
    except Exception as e:
        logger.error(f"An unexpected error occurred during Stage 1 call: {e}", exc_info=True)

    # --- FAILURE PATH ---
    logger.error(f"Stage 1 LLM call failed for user {user_id}.")
    # Fallback to avoid crashing the chat
    return {"topic_changed": False, "connected_tools": [], "disconnected_tools": []}

import re
from typing import List, Dict, Any

def parse_assistant_response(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses conversation history and stores reasoning, tool calls, and tool results
    in chronological order as `turn_steps`, along with the final user-facing content.
    """
    turn_steps = []
    final_content = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # ✅ Thoughts from <think> tags
        if role == "assistant" and "<think>" in content:
            think_matches = re.findall(r"<think>([\s\S]*?)</think>", content, re.DOTALL)
            for match in think_matches:
                turn_steps.append({
                    "type": "thought",
                    "content": match.strip()
                })

        # ✅ Tool calls
        if role == "assistant" and "function_call" in msg:
            tool_call = msg["function_call"]
            turn_steps.append({
                "type": "tool_call",
                "tool_name": tool_call.get("name"),
                "arguments": tool_call.get("arguments")
            })

        # ✅ Tool results
        if role == "function":
            turn_steps.append({
                "type": "tool_result",
                "tool_name": msg.get("name"),
                "result": msg.get("content", "").strip()
            })

    # ✅ Extract final user-facing message (last assistant message without function_call)
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "function_call" not in msg:
            text = msg.get("content", "").strip()
            if text:
                final_content = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
                break

    return {
        "final_content": final_content,
        "turn_steps": turn_steps
    }


def _extract_answer_from_llm_response(llm_output: str) -> str:
    """
    Extracts content from the first <answer> tag in the LLM's output.
    This ensures only the user-facing response is used for TTS.
    """
    if not llm_output:
        return ""
    match = re.search(r'<answer>([\s\S]*?)</answer>', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: if no answer tag, strip all other tags and return what's left.
    return re.sub(r'<(think|tool_code|tool_result)>.*?</\1>', '', llm_output, flags=re.DOTALL).strip()

def _get_tool_lists(user_integrations: Dict) -> Tuple[Dict, Dict]:
    """Separates tools into connected and disconnected lists."""
    connected_tools = {} # Tools that are built-in or connected by the user
    disconnected_tools = {}
    for tool_name, config in INTEGRATIONS_CONFIG.items():
        auth_type = config.get("auth_type")
        # Skip internal tools that shouldn't be exposed to the planner/user
        if tool_name in ["progress_updater", "chat_tools"]:
            continue

        # Built-in tools are always considered "connected" and available
        if auth_type == "builtin":
            connected_tools[tool_name] = config.get("description", "")
        # For user-configured tools, check the 'connected' flag from the DB
        elif auth_type in ["oauth", "manual", "composio"]:
            if user_integrations.get(tool_name, {}).get("connected", False):
                connected_tools[tool_name] = config.get("description", "")
            else:
                disconnected_tools[tool_name] = config.get("description", "")
    return connected_tools, disconnected_tools

async def generate_chat_llm_stream(
    user_id: str,
    messages: List[Dict[str, Any]],
    user_context: Dict[str, Any], # Basic context like name, timezone
    db_manager: MongoManager) -> AsyncGenerator[Dict[str, Any], None]:
    assistant_message_id = str(uuid.uuid4())

    try:
        yield {"type": "status", "message": "Analyzing context..."}

        username = user_context.get("name", "User")
        timezone_str = user_context.get("timezone", "UTC")
        location_raw = user_context.get("location")

        if isinstance(location_raw, dict) and 'latitude' in location_raw:
            location = f"latitude: {location_raw.get('latitude')}, longitude: {location_raw.get('longitude')}"
        elif isinstance(location_raw, str):
            location = location_raw
        else:
            location = "Not specified"
        try:
            user_timezone = ZoneInfo(timezone_str)
        except ZoneInfoNotFoundError:
            user_timezone = ZoneInfo("UTC")

        current_user_time = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        user_profile = await db_manager.get_user_profile(user_id)
        user_integrations = user_profile.get("userData", {}).get("integrations", {}) if user_profile else {}

        # Get both connected and disconnected tools
        connected_tools, disconnected_tools = _get_tool_lists(user_integrations)

        yield {"type": "status", "message": "Thinking..."}

        # --- STAGE 1 ---
        stage1_result = await _get_stage1_response(messages, connected_tools, disconnected_tools, user_id)

        topic_changed = stage1_result.get("topic_changed", False) # noqa
        relevant_tool_names = stage1_result.get("connected_tools", [])
        disconnected_requested_tools = stage1_result.get("disconnected_tools", [])

        # --- TOOL SELECTION & HISTORY TRUNCATION, PROCEED TO STAGE 2 ---
        tool_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in relevant_tool_names if t != 'memory']
        if tool_display_names:
            yield {"type": "status", "message": f"Using: {', '.join(tool_display_names)}"}

        # Stage 2 agent also needs access to core tools
        mandatory_tools = {"memory", "history", "tasks"}
        final_tool_names = set(relevant_tool_names) | mandatory_tools
        # Build the list of tools for the agent, including MCPs and local tools, ensuring headers are included.
        filtered_mcp_servers = {}
        for tool_name in final_tool_names:
            config = INTEGRATIONS_CONFIG.get(tool_name, {})
            if not config:
                continue

            mcp_config = config.get("mcp_server_config", {})
            if not (mcp_config and mcp_config.get("url") and mcp_config.get("name")):
                continue

            server_name = mcp_config["name"]
            base_url = mcp_config["url"]
            headers = {"X-User-ID": user_id}

            # If we've made it this far, the tool is configured correctly.
            filtered_mcp_servers[server_name] = {
                "url": base_url,
                "headers": headers,
                "transport": "sse"
            }
        tools = [{"mcpServers": filtered_mcp_servers}]

        logger.info(f"Final tools for agent: {list(filtered_mcp_servers.keys())}")
        
    except Exception as e:
        logger.error(f"Failed during initial setup for chat stream for user {user_id}: {e}", exc_info=True)
        yield {"type": "error", "message": "Failed to set up chat stream."}
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
    
    # --- Prepare message history for Stage 2 ---
    stage_2_expanded_messages = []
    messages_for_stage2 = []

    if topic_changed:
        logger.info(f"Topic change detected for user {user_id}. Truncating history for Stage 2.")
        # Find the last user message and send only that
        last_user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_message:
            messages_for_stage2 = [last_user_message]
    else:
        logger.info(f"No topic change detected for user {user_id}. Using full history for Stage 2.")
        messages_for_stage2 = messages

    # --- REFACTORED HISTORY RECONSTRUCTION ---
    # This loop "unrolls" our stored message format into the multi-message sequence the agent expects.
    for msg in messages_for_stage2:
        if msg.get("role") == "user":
            stage_2_expanded_messages.append({
                "role": "user",
                "content": msg.get("content", "")
            })
        elif msg.get("role") == "assistant":
            # This is the fix. We are now only adding the final, clean 'content'
            # field from the assistant's message to the history.
            # The thoughts, tool_calls, and tool_results are ignored,
            # so the LLM will not see its own internal monologue from the previous turn.
            if msg.get("content"):
                stage_2_expanded_messages.append({
                    "role": "assistant",
                    "content": msg.get("content")
                })

    if not any(msg.get("role") == "user" for msg in stage_2_expanded_messages):
        logger.error(f"Message history for Stage 2 is empty for user {user_id}. This should not happen.")

    # Inject a system note if some requested tools are disconnected
    if disconnected_requested_tools:
        disconnected_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in disconnected_requested_tools]
        system_note = (
            f"System Note: The user's request mentioned functionality requiring the following tools which are currently disconnected: "
            f"{', '.join(disconnected_display_names)}. You MUST inform the user that you cannot complete that part of the request "
            f"and suggest they connect the tool(s) in the Integrations page. Then, proceed with the rest of the request using the available tools."
        )
        # Prepend this note to the user's last message to make it highly visible to the agent.
        if stage_2_expanded_messages and stage_2_expanded_messages[-1]['role'] == 'user':
            stage_2_expanded_messages[-1]['content'] = f"{system_note}\n\nUser's original message: {stage_2_expanded_messages[-1]['content']}"
        else:
            # This case is unlikely but safe to handle.
            stage_2_expanded_messages.append({'role': 'system', 'content': system_note})
            
    logger.info(f"Reconstructed history for Stage 2 Agent (user: {user_id}):\\n{json.dumps(stage_2_expanded_messages, indent=2)}")

    system_prompt = STAGE_2_SYSTEM_PROMPT.format(
        username=username,
        location=location,
        current_user_time=current_user_time
    )

    # --- ADDED LOGGING ---
    logger.info(f"Reconstructed history for Stage 2 Agent (user: {user_id}):\n{json.dumps(stage_2_expanded_messages, indent=2)}")
    # --- END LOGGING ---

    def worker():
        try:
            # The agent expects a list of message dicts, which is what stage_2_expanded_messages is.
            for new_history_step in run_agent(system_message=system_prompt, function_list=tools, messages=stage_2_expanded_messages):
                loop.call_soon_threadsafe(queue.put_nowait, new_history_step)
        except Exception as e:
            logger.error(f"Error in chat worker thread for user {user_id}: {e}", exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, {"_error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        first_chunk = True
        last_yielded_content_str = ""
        while True:
            current_history = await queue.get()
            if current_history is None:
                break
            if isinstance(current_history, dict) and "_error" in current_history:
                raise Exception(f"Qwen Agent worker failed: {current_history['_error']}")
            if not isinstance(current_history, list):
                continue
            
            assistant_turn_start_index = next((i + 1 for i in range(len(current_history) - 1, -1, -1) if current_history[i].get('role') == 'user'), 0)
            assistant_messages = current_history[assistant_turn_start_index:]
            current_turn_str = "".join(str(m) for m in assistant_messages)

            if len(current_turn_str) > len(last_yielded_content_str):
                new_chunk = current_turn_str[len(last_yielded_content_str):]
                event_payload = {"type": "assistantStream", "token": new_chunk, "done": False, "messageId": assistant_message_id}
                if first_chunk and new_chunk.strip():
                    event_payload["tools"] = list(final_tool_names)
                    first_chunk = False
                yield event_payload
                last_yielded_content_str = current_turn_str

        print("Last yielded content string:", last_yielded_content_str)

    except asyncio.CancelledError:
        stream_interrupted = True
        raise
    except LLMProviderDownError as e:
        logger.error(f"LLM provider is down for user {user_id}: {e}", exc_info=True)
        yield json.dumps({"type": "error", "message": "Sorry, our AI provider is currently down. Please try again later."}) + "\n"
    except Exception as e:
        logger.error(f"Error during main chat agent run for user {user_id}: {e}", exc_info=True)
        yield {"type": "error", "message": "An unexpected error occurred in the chat agent."}
    finally:
        yield {"type": "assistantStream", "token": "", "done": True, "messageId": assistant_message_id}

async def process_voice_command(
    user_id: str,
    transcribed_text: str,
    send_status_update: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]],
    db_manager: MongoManager
) -> Tuple[str, str]:
    """
    Processes a transcribed voice command with full agentic capabilities,
    providing status updates and returning a final text response for TTS.
    """
    assistant_message_id = str(uuid.uuid4())
    logger.info(f"Processing voice command for user {user_id}: '{transcribed_text}'")

    try:
        # 1. Save user message and a placeholder for the assistant's response
        await db_manager.add_message(user_id=user_id, role="user", content=transcribed_text)
        await db_manager.add_message(user_id=user_id, role="assistant", content="[Thinking...]", message_id=assistant_message_id)

        # 2. Fetch history and user context
        history_from_db = await db_manager.get_message_history(user_id, limit=30)
        messages = list(reversed(history_from_db))
        qwen_formatted_history = [msg for msg in messages if msg.get("message_id") != assistant_message_id]

        user_profile = await db_manager.get_user_profile(user_id)
        user_data = user_profile.get("userData", {}) if user_profile else {}
        personal_info = user_data.get("personalInfo", {})
        
        username = personal_info.get("name", "User")
        timezone_str = personal_info.get("timezone", "UTC")
        location_raw = personal_info.get("location")

        if isinstance(location_raw, dict) and 'latitude' in location_raw:
            location = f"latitude: {location_raw.get('latitude')}, longitude: {location_raw.get('longitude')}"
        elif isinstance(location_raw, str):
            location = location_raw
        else:
            location = "Not specified"
        try:
            user_timezone = ZoneInfo(timezone_str)
        except ZoneInfoNotFoundError:
            user_timezone = ZoneInfo("UTC")
        current_user_time = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        # 3. Full tool selection logic (Stage 1)
        await send_status_update({"type": "status", "message": "choosing_tools"})
        
        user_integrations = user_data.get("integrations", {})
        connected_tools, disconnected_tools = _get_tool_lists(user_integrations)

        stage1_result = await _get_stage1_response(qwen_formatted_history, connected_tools, disconnected_tools, user_id)

        topic_changed = stage1_result.get("topic_changed", False)
        relevant_tool_names = stage1_result.get("connected_tools", [])
        disconnected_requested_tools = stage1_result.get("disconnected_tools", [])

        # 4. Stage 2 setup
        mandatory_tools = {"memory", "history", "tasks"}
        final_tool_names = set(relevant_tool_names) | mandatory_tools

        filtered_mcp_servers = {}
        for tool_name in final_tool_names:
            config = INTEGRATIONS_CONFIG.get(tool_name, {})
            if config:
                mcp_config = config.get("mcp_server_config", {})
                if mcp_config and mcp_config.get("url") and mcp_config.get("name"):
                    server_name = mcp_config["name"]
                    filtered_mcp_servers[server_name] = {
                        "url": mcp_config["url"],
                        "headers": {"X-User-ID": user_id},
                        "transport": "sse"
                    }
        tools = [{"mcpServers": filtered_mcp_servers}]
        logger.info(f"Voice Command Tools (Stage 2): {list(filtered_mcp_servers.keys())}")

        # --- Prepare message history for Stage 2 ---
        stage_2_expanded_messages = []
        messages_for_stage2 = []

        if topic_changed:
            logger.info(f"Topic change detected for user {user_id}. Truncating history for Stage 2.")
            # Find the last user message and send only that
            last_user_message = next((msg for msg in reversed(qwen_formatted_history) if msg.get("role") == "user"), None)
            if last_user_message:
                messages_for_stage2 = [last_user_message]
        else:
            logger.info(f"No topic change detected for user {user_id}. Using full history for Stage 2.")
            messages_for_stage2 = qwen_formatted_history

        # --- REFACTORED HISTORY RECONSTRUCTION ---
        # This loop "unrolls" our stored message format into the multi-message sequence the agent expects.
        for msg in messages_for_stage2:
            if msg.get("role") == "user":
                stage_2_expanded_messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif msg.get("role") == "assistant":
                # An assistant turn can have thoughts, multiple tool calls, and a final answer.
                # We must reconstruct this sequence for the agent's context.

                # 1. Add a preliminary message containing all thoughts for that turn.
                if msg.get("thoughts"):
                    thought_content = "\n".join([f"<think>{thought}</think>" for thought in msg["thoughts"]])
                    stage_2_expanded_messages.append({
                        "role": "assistant",
                        "content": thought_content
                    })

                # 2. Add the sequence of tool calls and their corresponding results.
                tool_calls = msg.get("tool_calls", [])
                tool_results = msg.get("tool_results", [])

                for i in range(len(tool_calls)):
                    tool_call = tool_calls[i]
                    stage_2_expanded_messages.append({
                        "role": "assistant",
                        "content": None,  # Important for the agent to recognize this as a tool call message
                        "function_call": {
                            "name": tool_call.get("tool_name"),
                            "arguments": tool_call.get("parameters")
                        }
                    })

                    # Add the corresponding tool result.
                    if i < len(tool_results):
                        tool_result = tool_results[i]
                        result_content = tool_result.get("result")
                        if not isinstance(result_content, str):
                            result_content = json.dumps(result_content)

                        stage_2_expanded_messages.append({
                            "role": "function",
                            "name": tool_result.get("tool_name"),
                            "content": result_content
                        })

                # 3. Add the final user-facing answer as a separate assistant message if it exists.
                if msg.get("content"):
                    stage_2_expanded_messages.append({
                        "role": "assistant",
                        "content": msg.get("content")
                    })

        if not any(msg.get("role") == "user" for msg in stage_2_expanded_messages):
            logger.error(f"Message history for Stage 2 is empty for user {user_id}. This should not happen.")

        # Handle disconnected tools note
        if disconnected_requested_tools:
            disconnected_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in disconnected_requested_tools]
            system_note = (
                f"System Note: The user's request mentioned functionality requiring the following tools which are currently disconnected: "
                f"{', '.join(disconnected_display_names)}. You MUST inform the user that you cannot complete that part of the request "
                f"and suggest they connect the tool(s) in the Integrations page. Then, proceed with the rest of the request using the available tools."
            )
            if stage_2_expanded_messages and stage_2_expanded_messages[-1]['role'] == 'user':
                stage_2_expanded_messages[-1]['content'] = f"{system_note}\n\nUser's original message: {stage_2_expanded_messages[-1]['content']}"
            else:
                stage_2_expanded_messages.append({'role': 'system', 'content': system_note})

        system_prompt = STAGE_2_SYSTEM_PROMPT.format(
            username=username,
            location=location,
            current_user_time=current_user_time
        )

        # --- ADDED LOGGING ---
        logger.info(f"Reconstructed history for Stage 2 Agent (user: {user_id}):\n{json.dumps(stage_2_expanded_messages, indent=2)}")
        # --- END LOGGING ---

        await send_status_update({"type": "status", "message": "thinking"})

        # 5. Agent Execution in a thread
        loop = asyncio.get_running_loop()
        def agent_worker():
            final_run_response = None
            try:
                for response in run_agent(system_message=system_prompt, function_list=tools, messages=stage_2_expanded_messages):
                    final_run_response = response
                    if isinstance(response, list) and response:
                        last_message = response[-1]
                        if last_message.get('role') == 'assistant' and last_message.get('function_call'):
                            tool_name = last_message['function_call']['name']
                            asyncio.run_coroutine_threadsafe(
                                send_status_update({"type": "status", "message": f"using_tool_{tool_name}"}),
                                loop
                            )
                return final_run_response
            except Exception as e:
                logger.error(f"Error inside agent_worker thread for voice command: {e}", exc_info=True)
                return None

        final_run_response = await asyncio.to_thread(agent_worker)

        # 6. Process result
        full_response_str = ""
        if final_run_response and isinstance(final_run_response, list):
            assistant_content_parts = [
                msg.get('content', '')
                for msg in final_run_response
                if msg.get('role') == 'assistant' and msg.get('content')
            ]
            full_response_str = "".join(assistant_content_parts)

        final_text_response = _extract_answer_from_llm_response(full_response_str)

        if not final_text_response:
            last_message = final_run_response[-1] if final_run_response else {}
            if last_message.get('role') == 'function':
                final_text_response = "The action has been completed."
            else:
                final_text_response = "I'm sorry, I couldn't process that."

        await db_manager.messages_collection.update_one(
            {"message_id": assistant_message_id, "user_id": user_id},
            {"$set": {"content": final_text_response}}
        )

        return final_text_response, assistant_message_id
    except Exception as e:
        logger.error(f"Error processing voice command for {user_id}: {e}", exc_info=True)
        error_msg = "I encountered an error while processing your request."
        await db_manager.messages_collection.update_one(
            {"message_id": assistant_message_id},
            {"$set": {"content": error_msg}}
        )
        return error_msg, assistant_message_id