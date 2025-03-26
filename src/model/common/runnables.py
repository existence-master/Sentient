import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library

from .prompts import *  # Importing prompt templates and related utilities from prompts.py
from .formats import *  # Importing format specifications or utilities from formats.py
from model.app.base import *
from model.app.helpers import *  # Importing helper functions from helpers.py

load_dotenv("model/.env")

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

    provider: Optional[str] = None

    model_name: str = get_selected_model()

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


# --- Runnable Initialization Functions ---
# These functions create and configure instances of CustomRunnable for different tasks.
# They encapsulate the specific prompts, model settings, and response formats for each functionality.

def get_unified_classification_runnable(chat_history):
    model_name, provider = get_selected_model()
    model_mapping = {
        "openai": OpenAIRunnable,
        "claude": ClaudeRunnable,
        "gemini": GeminiRunnable,
    }
    runnable_class = model_mapping.get(provider, OllamaRunnable)
    model_url = os.getenv("BASE_MODEL_URL") if provider not in model_mapping else os.getenv(f"{provider.upper()}_API_URL")

    unified_classification_runnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=unified_classification_system_prompt_template,  # Defined above
        user_prompt_template=unified_classification_user_prompt_template,      # Defined above
        input_variables=["query"],
        required_format=unified_classification_format,                # Defined above
        response_type="json",
        stateful=True,
    )
    unified_classification_runnable.add_to_history(chat_history)
    return unified_classification_runnable


def get_internet_query_reframe_runnable():
    """
    Creates and configures a CustomRunnable for reframing user queries to be more effective for internet searches.

    This runnable takes a user's original query and reformulates it into a query that is optimized for web search engines.
    It utilizes prompts and settings tailored for query reframing.

    :return: A configured CustomRunnable instance for internet query reframing.
    """
    model_name, provider=get_selected_model()
    
    model_mapping: Dict[str, type[BaseRunnable]] = {
        "openai": OpenAIRunnable,
        "claude": ClaudeRunnable,
        "gemini": GeminiRunnable,
    }
    runnable_class = model_mapping.get(provider, OllamaRunnable)
    model_url = os.getenv("BASE_MODEL_URL") if provider not in model_mapping else os.getenv(f"{provider.upper()}_API_URL")

    internet_query_reframe_runnable = runnable_class(
        model_url=model_url,  # Get model URL from environment variables
        model_name=model_name,  # Get model name from environment variables
        system_prompt_template=internet_query_reframe_system_prompt_template,  # System prompt for query reframing
        user_prompt_template=internet_query_reframe_user_prompt_template,  # User prompt template for query reframing
        input_variables=["query"],  # Input variable for query reframing prompt
        response_type="chat",  # Response type is chat (text)
    )

    return internet_query_reframe_runnable  # Return the configured internet query reframe runnable


def get_internet_summary_runnable():
    """
    Creates and configures a CustomRunnable for summarizing content retrieved from internet searches.

    This runnable is designed to take web search results and condense them into a concise summary.
    It is configured with prompts and settings optimized for summarization tasks.

    :return: A configured CustomRunnable instance for internet summary generation.
    """
    model_name, provider=get_selected_model()
    
    model_mapping: Dict[str, type[BaseRunnable]] = {
        "openai": OpenAIRunnable,
        "claude": ClaudeRunnable,
        "gemini": GeminiRunnable,
        "llama3": OllamaRunnable
    }
    runnable_class = model_mapping.get(provider, OllamaRunnable)
    model_url = os.getenv("BASE_MODEL_URL") if provider not in model_mapping else os.getenv(f"{provider.upper()}_API_URL")

    internet_summary_runnable = runnable_class(
        model_url=model_url,  # Get model URL from environment variables
        model_name=model_name,  # Get model name from environment variables
        system_prompt_template=internet_summary_system_prompt_template,  # System prompt for internet summary
        user_prompt_template=internet_summary_user_prompt_template,  # User prompt template for internet summary
        input_variables=[
            "query"
        ],  # Input variable for summary prompt (expects search results as input)
        response_type="chat",  # Response type is chat (text)
    )

    return internet_summary_runnable  # Return the configured internet summary runnable