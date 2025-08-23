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

        if not completion.choices:
            raise Exception("LLM response was successful but contained no choices.")

        final_content_str = completion.choices[0].message.content

        cleaned_output = clean_llm_output(final_content_str)
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

    logger.error(f"Stage 1 LLM call failed for user {user_id}.")
    return {"topic_changed": False, "connected_tools": [], "disconnected_tools": []}

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
        if role == "assistant" and content and "<think>" in content:
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
            if not text:
                continue

            # Priority 1: Check for an explicit <answer> tag.
            answer_match = re.search(r'<answer>([\s\S]*?)</answer>', text, re.DOTALL)
            if answer_match:
                final_content = answer_match.group(1).strip()
            else:
                # Fallback: Use content outside of <think> tags.
                final_content = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
            break # Stop after finding the first valid final content

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
    # Priority 1: Look for the <answer> tag.
    match = re.search(r'<answer>([\s\S]*?)</answer>', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: If no <answer> tag, strip out all other known tags.
    # This correctly handles cases where the final response is mixed with <think> tags.
    return re.sub(r'<think>[\s\S]*?</think>', '', llm_output, flags=re.DOTALL).strip()

def _get_tool_lists(user_integrations: Dict) -> Tuple[Dict, Dict]:
    """Separates tools into connected and disconnected lists."""
    connected_tools = {}
    disconnected_tools = {}
    for tool_name, config in INTEGRATIONS_CONFIG.items():
        auth_type = config.get("auth_type")
        if tool_name in ["progress_updater", "chat_tools"]:
            continue
        if auth_type == "builtin":
            connected_tools[tool_name] = config.get("description", "")
        elif auth_type in ["oauth", "manual", "composio"]:
            if user_integrations.get(tool_name, {}).get("connected", False):
                connected_tools[tool_name] = config.get("description", "")
            else:
                disconnected_tools[tool_name] = config.get("description", "")
    return connected_tools, disconnected_tools

async def generate_chat_llm_stream(
    user_id: str,
    messages: List[Dict[str, Any]],
    user_context: Dict[str, Any],
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
        connected_tools, disconnected_tools = _get_tool_lists(user_integrations)
        yield {"type": "status", "message": "Thinking..."}
        stage1_result = await _get_stage1_response(messages, connected_tools, disconnected_tools, user_id)
        topic_changed = stage1_result.get("topic_changed", False) # noqa
        relevant_tool_names = stage1_result.get("connected_tools", [])
        disconnected_requested_tools = stage1_result.get("disconnected_tools", [])
        tool_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in relevant_tool_names if t != 'memory']
        if tool_display_names:
            yield {"type": "status", "message": f"Using: {', '.join(tool_display_names)}"}
        mandatory_tools = {"memory", "history", "tasks"}
        final_tool_names = set(relevant_tool_names) | mandatory_tools
        filtered_mcp_servers = {}
        for tool_name in final_tool_names:
            config = INTEGRATIONS_CONFIG.get(tool_name, {})
            if not config: continue
            mcp_config = config.get("mcp_server_config", {})
            if not (mcp_config and mcp_config.get("url") and mcp_config.get("name")): continue
            server_name = mcp_config["name"]
            base_url = mcp_config["url"]
            headers = {"X-User-ID": user_id}
            filtered_mcp_servers[server_name] = {"url": base_url, "headers": headers, "transport": "sse"}
        tools = [{"mcpServers": filtered_mcp_servers}]
        logger.info(f"Final tools for agent: {list(filtered_mcp_servers.keys())}")
    except Exception as e:
        logger.error(f"Failed during initial setup for chat stream for user {user_id}: {e}", exc_info=True)
        yield {"type": "error", "message": "Failed to set up chat stream."}
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
    
    stage_2_expanded_messages = []
    messages_for_stage2 = []
    if topic_changed:
        logger.info(f"Topic change detected for user {user_id}. Truncating history for Stage 2.")
        last_user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_message:
            messages_for_stage2 = [last_user_message]
    else:
        logger.info(f"No topic change detected for user {user_id}. Using full history for Stage 2.")
        messages_for_stage2 = messages

    for msg in messages_for_stage2:
        if msg.get("role") == "user":
            stage_2_expanded_messages.append({
                "role": "user",
                "content": msg.get("content", "")
            })
        elif msg.get("role") == "assistant":
            # --- CHANGED --- Unroll the assistant's turn from turn_steps into the multi-message format.
            turn_steps = msg.get("turn_steps", [])
            thought_buffer = []

            for step in turn_steps:
                if step.get("type") == "thought":
                    # Buffer thoughts to combine them into a single <think> block if they are consecutive.
                    thought_buffer.append(step.get("content", "").strip())
                else:
                    # If we encounter a non-thought step, flush the thought buffer first.
                    if thought_buffer:
                        combined_thoughts = "<think>\n" + "\n\n".join(thought_buffer) + "\n</think>"
                        stage_2_expanded_messages.append({"role": "assistant", "content": combined_thoughts})
                        thought_buffer = [] # Reset buffer

                    if step.get("type") == "tool_call":
                        stage_2_expanded_messages.append({
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": step.get("tool_name"),
                                "arguments": step.get("arguments")
                            }
                        })
                    elif step.get("type") == "tool_result":
                        stage_2_expanded_messages.append({
                            "role": "function",
                            "name": step.get("tool_name"),
                            "content": step.get("result", "")
                        })

            # After the loop, add the final user-facing content.
            # Combine any remaining thoughts with the final content.
            if msg.get("content"):
                final_content_with_thoughts = "\n".join([f"<think>{thought}</think>" for thought in thought_buffer]) + "\n" + msg.get("content")
                stage_2_expanded_messages.append({
                    "role": "assistant",
                    "content": final_content_with_thoughts.strip()
                })
            elif thought_buffer: # If there's no final content but there were thoughts
                combined_thoughts = "<think>\n" + "\n\n".join(thought_buffer) + "\n</think>"
                stage_2_expanded_messages.append({"role": "assistant", "content": combined_thoughts})

    if not any(msg.get("role") == "user" for msg in stage_2_expanded_messages):
        logger.error(f"Message history for Stage 2 is empty for user {user_id}. This should not happen.")

    if disconnected_requested_tools:
        disconnected_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in disconnected_requested_tools]
        system_note = (f"System Note: The user's request mentioned functionality requiring the following tools which are currently disconnected: {', '.join(disconnected_display_names)}. You MUST inform the user that you cannot complete that part of the request and suggest they connect the tool(s) in the Integrations page. Then, proceed with the rest of the request using the available tools.")
        if stage_2_expanded_messages and stage_2_expanded_messages[-1]['role'] == 'user':
            stage_2_expanded_messages[-1]['content'] = f"{system_note}\n\nUser's original message: {stage_2_expanded_messages[-1]['content']}"
        else:
            stage_2_expanded_messages.append({'role': 'system', 'content': system_note})
            

    system_prompt = STAGE_2_SYSTEM_PROMPT.format(username=username, location=location, current_user_time=current_user_time)

    def worker():
        try:
            for new_history_step in run_agent(system_message=system_prompt, function_list=tools, messages=stage_2_expanded_messages):
                loop.call_soon_threadsafe(queue.put_nowait, new_history_step)
        except Exception as e:
            logger.error(f"Error in chat worker thread for user {user_id}: {e}", exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, {"_error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_yielded_content_str = ""
    final_assistant_messages = []

    try:
        first_chunk = True
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
            final_assistant_messages = assistant_messages

            current_turn_str = "".join(str(m) for m in assistant_messages)

            if len(current_turn_str) > len(last_yielded_content_str):
                new_chunk = current_turn_str[len(last_yielded_content_str):]
                event_payload = {"type": "assistantStream", "token": new_chunk, "done": False, "messageId": assistant_message_id}
                if first_chunk and new_chunk.strip():
                    event_payload["tools"] = list(final_tool_names)
                    first_chunk = False
                yield event_payload
                last_yielded_content_str = current_turn_str

    except asyncio.CancelledError:
        raise
    except LLMProviderDownError as e:
        logger.error(f"LLM provider is down for user {user_id}: {e}", exc_info=True)
        yield json.dumps({"type": "error", "message": "Sorry, our AI provider is currently down. Please try again later."}) + "\n"
    except Exception as e:
        logger.error(f"Error during main chat agent run for user {user_id}: {e}", exc_info=True)
        yield {"type": "error", "message": "An unexpected error occurred in the chat agent."}
    finally:
        if final_assistant_messages:
            parsed_data = parse_assistant_response(final_assistant_messages)
            final_payload = {
                "type": "assistantStream", 
                "token": "", 
                "done": True, 
                "messageId": assistant_message_id,
                "final_content": parsed_data.get("final_content"),
                "turn_steps": parsed_data.get("turn_steps")
            }
            yield final_payload
        else:
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
        await db_manager.add_message(user_id=user_id, role="user", content=transcribed_text)
        await db_manager.add_message(user_id=user_id, role="assistant", content="[Thinking...]", message_id=assistant_message_id)
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
        await send_status_update({"type": "status", "message": "choosing_tools"})
        user_integrations = user_data.get("integrations", {})
        connected_tools, disconnected_tools = _get_tool_lists(user_integrations)
        stage1_result = await _get_stage1_response(qwen_formatted_history, connected_tools, disconnected_tools, user_id)
        topic_changed = stage1_result.get("topic_changed", False)
        relevant_tool_names = stage1_result.get("connected_tools", [])
        disconnected_requested_tools = stage1_result.get("disconnected_tools", [])
        mandatory_tools = {"memory", "history", "tasks"}
        final_tool_names = set(relevant_tool_names) | mandatory_tools
        filtered_mcp_servers = {}
        for tool_name in final_tool_names:
            config = INTEGRATIONS_CONFIG.get(tool_name, {})
            if config:
                mcp_config = config.get("mcp_server_config", {})
                if mcp_config and mcp_config.get("url") and mcp_config.get("name"):
                    server_name = mcp_config["name"]
                    filtered_mcp_servers[server_name] = {"url": mcp_config["url"], "headers": {"X-User-ID": user_id}, "transport": "sse"}
        tools = [{"mcpServers": filtered_mcp_servers}]
        logger.info(f"Voice Command Tools (Stage 2): {list(filtered_mcp_servers.keys())}")
        stage_2_expanded_messages = []
        messages_for_stage2 = []
        if topic_changed:
            logger.info(f"Topic change detected for user {user_id}. Truncating history for Stage 2.")
            last_user_message = next((msg for msg in reversed(qwen_formatted_history) if msg.get("role") == "user"), None)
            if last_user_message:
                messages_for_stage2 = [last_user_message]
        else:
            logger.info(f"No topic change detected for user {user_id}. Using full history for Stage 2.")
            messages_for_stage2 = qwen_formatted_history

        for msg in messages_for_stage2:
            if msg.get("role") == "user":
                stage_2_expanded_messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif msg.get("role") == "assistant":
                # --- CHANGED --- Unroll the assistant's turn from turn_steps into the multi-message format.
                turn_steps = msg.get("turn_steps", [])
                thought_buffer = []

                for step in turn_steps:
                    if step.get("type") == "thought":
                        thought_buffer.append(step.get("content", "").strip())
                    else:
                        if thought_buffer:
                            combined_thoughts = "<think>\n" + "\n\n".join(thought_buffer) + "\n</think>"
                            stage_2_expanded_messages.append({"role": "assistant", "content": combined_thoughts})
                            thought_buffer = []

                        if step.get("type") == "tool_call":
                            stage_2_expanded_messages.append({
                                "role": "assistant",
                                "content": None,
                                "function_call": { "name": step.get("tool_name"), "arguments": step.get("arguments") }
                            })
                        elif step.get("type") == "tool_result":
                            result_content = step.get("result", "")
                            if not isinstance(result_content, str):
                                result_content = json.dumps(result_content)
                            stage_2_expanded_messages.append({
                                "role": "function", "name": step.get("tool_name"), "content": result_content
                            })

                if msg.get("content"):
                    final_content_with_thoughts = "\n".join([f"<think>{thought}</think>" for thought in thought_buffer]) + "\n" + msg.get("content")
                    stage_2_expanded_messages.append({
                        "role": "assistant", "content": final_content_with_thoughts.strip()
                    })
                elif thought_buffer:
                    combined_thoughts = "<think>\n" + "\n\n".join(thought_buffer) + "\n</think>"
                    stage_2_expanded_messages.append({"role": "assistant", "content": combined_thoughts})

        if not any(msg.get("role") == "user" for msg in stage_2_expanded_messages):
            logger.error(f"Message history for Stage 2 is empty for user {user_id}. This should not happen.")

        if disconnected_requested_tools:
            disconnected_display_names = [INTEGRATIONS_CONFIG.get(t, {}).get('display_name', t) for t in disconnected_requested_tools]
            system_note = (f"System Note: The user's request mentioned functionality requiring the following tools which are currently disconnected: {', '.join(disconnected_display_names)}. You MUST inform the user that you cannot complete that part of the request and suggest they connect the tool(s) in the Integrations page. Then, proceed with the rest of the request using the available tools.")
            if stage_2_expanded_messages and stage_2_expanded_messages[-1]['role'] == 'user':
                stage_2_expanded_messages[-1]['content'] = f"{system_note}\n\nUser's original message: {stage_2_expanded_messages[-1]['content']}"
            else:
                stage_2_expanded_messages.append({'role': 'system', 'content': system_note})

        system_prompt = STAGE_2_SYSTEM_PROMPT.format(username=username, location=location, current_user_time=current_user_time)
        await send_status_update({"type": "status", "message": "thinking"})

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
                            asyncio.run_coroutine_threadsafe(send_status_update({"type": "status", "message": f"using_tool_{tool_name}"}), loop)
                return final_run_response
            except Exception as e:
                logger.error(f"Error inside agent_worker thread for voice command: {e}", exc_info=True)
                return None

        final_run_response = await asyncio.to_thread(agent_worker)

        if not final_run_response or not isinstance(final_run_response, list):
            final_run_response = []

        assistant_turn_start_index = next((i + 1 for i in range(len(final_run_response) - 1, -1, -1) if final_run_response[i].get('role') == 'user'), 0)
        assistant_messages = final_run_response[assistant_turn_start_index:]

        parsed_response = parse_assistant_response(assistant_messages)
        final_text_for_tts = parsed_response.get("final_content", "I'm sorry, I couldn't process that.")

        if not final_text_for_tts and assistant_messages:
            last_message = assistant_messages[-1]
            if last_message.get('role') == 'function':
                final_text_for_tts = "The action has been completed."

        await db_manager.messages_collection.update_one(
            {"message_id": assistant_message_id, "user_id": user_id},
            {"$set": {
                "content": final_text_for_tts,
                "turn_steps": parsed_response.get("turn_steps", [])
            }}
        )

        return final_text_for_tts, assistant_message_id
    except Exception as e:
        logger.error(f"Error processing voice command for {user_id}: {e}", exc_info=True)
        error_msg = "I encountered an error while processing your request."
        await db_manager.messages_collection.update_one(
            {"message_id": assistant_message_id},
            {"$set": {"content": error_msg}}
        )
        return error_msg, assistant_message_id