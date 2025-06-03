# server/planner/plan.py
from typing import Optional, Dict, Any, Union, List
from qwen_agent.agents import Assistant
import re

# Define LLM configuration for Ollama
llm_cfg = {
    'model': 'qwen3:4b',
    'model_server': 'http://localhost:11434/v1/',
    'api_key': 'EMPTY',
    'generate_cfg': {
        # This parameter will affect the tool-call parsing logic. Default is False:
          # Set to True: when content is `<think>this is the thought</think>this is the answer`
          # Set to False: when response consists of reasoning_content and content
          # Not working?
        # 'thought_in_content': True, 

        # tool-call template: default is nous (recommended for qwen3):
        # 'fncall_prompt_type': 'nous'

        # Maximum input length, messages will be truncated if they exceed this length, please adjust according to model API:
        # 'max_input_tokens': 58000

        # Parameters that will be passed directly to the model API, such as top_p, enable_thinking, etc., according to the API specifications:
        # 'top_p': 0.8
    }
}

bot = Assistant(llm=llm_cfg)

action_items = [
    "The user needs to prepare a presentation for the Quarterly Report meeting.",
]

available_tools = ["gmail", "gcalendar", "gdrive", "gdocs", "gslides", "gsheet", "get_short_term_memories", "get_long_term_memories", "notion", "web_search"]
context_sources = ["short_term_memories (short-term information about the user)", "long_term_memories (long-term information about the user)", "gdrive (user's Google Drive)", "notion (user's Notion workspace)"]

def generate_plan_text(action_items: List[str], tools: List[str] = available_tools) -> Dict[str, Any]:
    """
    A simple planner that takes in a list of action items and returns a plan.
    """
    system_prompt = f"""You are Sentient, an agent adept at making plans that achieve a user's long-term goals. You will take in an action item and a list of tools that you have access to. You can then create a plan that an  executor can follow to use these tools to provide the best possible outcome for the user. Given an action item, decide the optimal outcome that a user will prefer to have at the end of the plan. Then, create a plan that efficiently combines available tools to achieve this outcome. The plan should be a simple list of steps that the executor can follow. Each step should be clear and actionable, and should involve the usage of one or more tools. If a step does not involve the usage of tools, it should be left out of the plan. The executor will have access to the tools you have access to, so you can use them in your plan.
    
    The executor will have access to the following tools: {available_tools}. You will only use these tools as context to define a list of steps.
    
    The executor will also have access to context sources about the user that can provide additional context for the execution of tasks. These context sources are as follows: {context_sources}.
    
    Do not use Markdown."""
    
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': ''.join(action_items)},]
    for responses in bot.run(messages=messages):
        pass
    return responses


plan = generate_plan_text(action_items, available_tools)

thinking_trace = re.search(r"<think>(.*?)</think>", plan[0]["content"], re.DOTALL).group(1).strip()
final_plan = re.sub(r"<think>.*?</think>", "", plan[0]["content"], flags=re.DOTALL).strip()


# print ("Plan generated based on action items:\n")

print ("\nPlan:\n" + final_plan)

print ("\n\n")

print ("Thinking trace of the model:\n" + thinking_trace)