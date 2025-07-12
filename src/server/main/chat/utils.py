import datetime
import uuid
import os
import json
import asyncio
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import threading
import logging
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional

from main.db import MongoManager
from main.llm import get_qwen_assistant
from main.config import INTEGRATIONS_CONFIG, SUPERMEMORY_MCP_BASE_URL, SUPERMEMORY_MCP_ENDPOINT_SUFFIX

logger = logging.getLogger(__name__)

async def generate_chat_llm_stream(
    user_id: str,
    messages: List[Dict[str, Any]], # This is now the history from the client
    user_context: Dict[str, Any],
    db_manager: MongoManager,
    enable_internet: bool = False,
    enable_weather: bool = False,
    enable_news: bool = False,
    enable_maps: bool = False,
    enable_shopping: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
    assistant_message_id = str(uuid.uuid4())

    def step_event(step, status):
        return {
            "type": "step",
            "step": step,
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    try:
        yield step_event("Initializing chat stream", "in_progress")
        # --- Construct detailed system prompt from user context ---
        username = user_context.get("name", "User")
        timezone_str = user_context.get("timezone", "UTC")
        comm_style = user_context.get("communication_style", "friendly and professional")
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
            logger.warning(f"Invalid timezone '{timezone_str}' for user {user_id}. Defaulting to UTC.")
            user_timezone = ZoneInfo("UTC")

        current_user_time = datetime.datetime.now(user_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')

        user_profile = await db_manager.get_user_profile(user_id)
        yield step_event("Fetched user profile", "done")
        supermemory_user_id = user_profile.get("userData", {}).get("supermemory_user_id") if user_profile else None
        
        active_mcp_servers = {}

        if supermemory_user_id:
            full_supermemory_mcp_url = f"{SUPERMEMORY_MCP_BASE_URL.rstrip('/')}/{supermemory_user_id}{SUPERMEMORY_MCP_ENDPOINT_SUFFIX}"
            active_mcp_servers["supermemory"] = {
                "transport": "sse",
                "url": full_supermemory_mcp_url
            }
        yield step_event("Configured MCP servers", "done")
        
        active_mcp_servers["chat_tools"] = {"url": INTEGRATIONS_CONFIG["chat_tools"]["mcp_server_config"]["url"], "headers": {"X-User-ID": user_id}}
        # ADDED: Journal Server
        active_mcp_servers["journal_server"] = {"url": INTEGRATIONS_CONFIG["journal"]["mcp_server_config"]["url"], "headers": {"X-User-ID": user_id}}

        tool_flags = {
            "internet_search": enable_internet,
            "accuweather": enable_weather,
            "news": enable_news,
            "gmaps": enable_maps,
            "gshopping": enable_shopping
        }

        for service_name, is_enabled in tool_flags.items():
            if is_enabled:
                config = INTEGRATIONS_CONFIG.get(service_name)
                if config and "mcp_server_config" in config:
                    mcp_config = config["mcp_server_config"]
                    active_mcp_servers[mcp_config["name"]] = {"url": mcp_config["url"], "headers": {"X-User-ID": user_id}}
        yield step_event("Prepared tool flags and servers", "done")
        tools = [{"mcpServers": active_mcp_servers}]
        yield step_event("System prompt and tools ready", "done")

    except Exception as e:
        logger.error(f"Failed during initial setup for chat stream for user {user_id}: {e}", exc_info=True)
        yield {"type": "error", "message": "Failed to set up chat stream."}
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()

    stream_interrupted = False
    try:
        yield step_event("Starting LLM worker thread", "in_progress")
        def worker():
            try:
                system_prompt = ( # noqa
                    f"You are Sentient, a personalized AI assistant. Your primary goal is to provide helpful, personalized, and context-aware responses. You MUST use your memory to achieve this.\n\n"
                    f"**Critical Instructions:**\n"
                    f"1.  **ALWAYS Use Memory First:** Before answering ANY query, you MUST use the `supermemory-search` tool to check if you already know relevant information about the user or the topic. This is not optional. Personalize your response based on what you find.\n"
                    f"2.  **Continuously Learn:** If you learn a new, permanent fact about the user (their preferences, relationships, personal details, goals, etc.), you MUST use the `supermemory-addToSupermemory` tool to remember it for the future. For example, if the user says 'my wife's name is Jane', you must call the tool to save this fact.\n"
                    f"3.  **Delegate Complex Tasks:** For requests that require multiple steps, research, or actions over time (e.g., 'plan my trip', 'summarize this topic'), use the `create_task_from_description` tool. Do not try to perform complex tasks yourself.\n"
                    f"4.  **Use Your Journal:** For daily notes, simple reminders, or retrieving information from a specific day, use the journal tools (`search_journal`, `summarize_day`, `add_journal_entry`).\n"
                    f"5.  **Final Answer Format:** When you have a complete, final answer for the user that is not a tool call, you MUST wrap it in <answer> tags. For example: <answer>The weather in London is 15°C and cloudy.</answer>. Any self-correction or thought process should be inside <think> tags, which can precede the final answer.\n\n"
                    f"**User Context (for your reference):**\n"
                    f"-   **User's Name:** {username}\n"
                    f"-   **User's Location:** {location}\n"
                    f"-   **Current Time:** {current_user_time}\n"
                    f"-   **Your Persona:** You must adopt a '{comm_style}' communication style.\n\n"
                    f"Your primary directive is to be as personalized and helpful as possible by actively using your memory. Do not ask questions you should already know the answer to; search your memory instead."
                )

                qwen_formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

                qwen_assistant = get_qwen_assistant(system_message=system_prompt, function_list=tools)
                for new_history_step in qwen_assistant.run(messages=qwen_formatted_history):
                    loop.call_soon_threadsafe(queue.put_nowait, new_history_step)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"_error": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        yield step_event("LLM worker thread started", "done")

        last_yielded_content_str, final_history_state = "", None
        while True:
            current_history = await queue.get()
            if current_history is None: break
            if isinstance(current_history, dict) and "_error" in current_history: raise Exception(f"Qwen Agent failed: {current_history['_error']}")
            if not isinstance(current_history, list): continue

            final_history_state = current_history
            assistant_turn_start_index = next((i + 1 for i in range(len(current_history) - 1, -1, -1) if current_history[i].get('role') == 'user'), 0)
            assistant_messages = current_history[assistant_turn_start_index:]
            current_turn_str = "".join(msg_to_str(m) for m in assistant_messages)
            if len(current_turn_str) > len(last_yielded_content_str):
                new_chunk = current_turn_str[len(last_yielded_content_str):]
                yield {"type": "assistantStream", "token": new_chunk, "done": False, "messageId": assistant_message_id}
                last_yielded_content_str = current_turn_str
    except asyncio.CancelledError:
        stream_interrupted = True; raise
    finally:
        yield step_event("Chat stream complete", "done")
        yield {"type": "assistantStream", "token": "", "done": True, "messageId": assistant_message_id}
        await asyncio.to_thread(thread.join)

def msg_to_str(msg: Dict[str, Any]) -> str:
    if msg.get('role') == 'assistant' and msg.get('function_call'):
        args_str = msg['function_call'].get('arguments', '')
        try: args_pretty = json.dumps(json.loads(args_str), indent=2)
        except: args_pretty = args_str
        return f"<tool_code name=\"{msg['function_call'].get('name')}\">\n{args_pretty}\n</tool_code>\n"
    elif msg.get('role') == 'function':
        content = msg.get('content', '')
        try: content_pretty = json.dumps(json.loads(content), indent=2)
        except: content_pretty = content
        return f"<tool_result tool_name=\"{msg.get('name')}\">\n{content_pretty}\n</tool_result>\n"
    elif msg.get('role') == 'assistant' and msg.get('content'):
        return msg.get('content', '')
    return ''