import os
from server.app.base import *
from server.context.prompts import *
from server.context.formats import *

def get_gmail_context_runnable():
    """Configure and return a Runnable for the Gmail Context Engine."""
    # Example assumes you have a BaseRunnable and specific implementations like OllamaRunnable
    # Adjust imports and classes based on your actual LLM setup
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    model_name, provider = get_selected_model()  # Assume this function exists to get model config

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback, adjust as needed

    return runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=gmail_context_engine_system_prompt_template,
        user_prompt_template=gmail_context_engine_user_prompt_template,
        input_variables=["new_information", "related_memories", "ongoing_tasks", "chat_history"],
        required_format=gmail_context_engine_required_format,  
        response_type="json",
        stateful=False,
    )

def get_internet_search_context_runnable():
    """Configure and return a Runnable for the Internet Search Context Engine."""
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    model_name, provider = get_selected_model()  # Assume this function exists

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback

    return runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=internet_search_context_engine_system_prompt_template,
        user_prompt_template=internet_search_context_engine_user_prompt_template,
        input_variables=["new_information", "related_memories", "ongoing_tasks", "chat_history"],
        required_format=internet_search_context_engine_required_format,
        response_type="json",
        stateful=False,
    )

def get_gcalendar_context_runnable():
    """Configure and return a Runnable for the GCalendar Context Engine."""
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    model_name, provider = get_selected_model()  # Assume this function exists

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback, adjust as needed

    return runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=gcalendar_context_engine_system_prompt_template,
        user_prompt_template=gcalendar_context_engine_user_prompt_template,
        input_variables=["new_information", "related_memories", "ongoing_tasks", "chat_history"],
        required_format=gcalendar_context_engine_required_format,
        response_type="json",
        stateful=False,
    )