import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from server.app.base import *
from .prompts import *

load_dotenv("server/.env")

def get_chat_runnable(chat_history: List[Dict[str, str]]) -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class based on the selected model.

    Determines the model provider (Ollama, OpenAI, Claude, Gemini) based on the
    selected model name from `get_selected_model()`. It then returns an instance
    of the corresponding Runnable class, configured with API URLs and default prompts.

    Args:
        chat_history (List[Dict[str, str]]): The chat history to initialize the runnable with.

    Returns:
        BaseRunnable: An instance of a Runnable class (OllamaRunnable, OpenAIRunnable, ClaudeRunnable, or GeminiRunnable).
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    model_name, provider = get_selected_model()

     # Extract provider from model name (e.g., 'openai' from 'openai:gpt-3.5-turbo')


    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=chat_system_prompt_template,
        user_prompt_template=chat_user_prompt_template,
        input_variables=[
            "query",
            "user_context",
            "internet_context",
            "name",
            "personality",
        ],
        response_type="chat",
        stream=True,
        stateful=True,
    )

    runnable.add_to_history(chat_history)
    return runnable