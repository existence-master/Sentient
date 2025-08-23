import re
from typing import List, Dict, Any

def clean_llm_output(text: str) -> str:
    """
    Removes reasoning tags (e.g., <think>...</think>) and trims whitespace from LLM output.
    """
    if not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()
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
