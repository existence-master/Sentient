import os
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library (not explicitly used in this file)
from typing import Optional, Dict
from dotenv import load_dotenv

from .prompts import *  # Importing prompt templates and related utilities from prompts.py
from .formats import *  # Importing format specifications or utilities from formats.py
from server.app.helpers import *  # Importing helper functions from helpers.py
from server.app.base import *

load_dotenv("server/.env")  # Load environment variables from .env file


def get_reddit_runnable() -> BaseRunnable:
    """
    Creates and configures a stateless Runnable for interacting with Reddit data based on selected model.

    This runnable is specifically set up for tasks related to Reddit, using Reddit-specific prompts
    and configurations. It is stateless, meaning it does not retain conversation history across calls.

    :return: A configured stateless BaseRunnable instance for Reddit interactions.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable


    reddit_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=reddit_system_prompt_template,
        user_prompt_template=reddit_user_prompt_template,
        input_variables=["subreddits"],
        required_format=reddit_required_format,
        response_type="json",
        stream=False, # Stateless runnables generally don't need streaming
        stateful=False
    )

    return reddit_runnable


def get_twitter_runnable() -> BaseRunnable:
    """
    Creates and configures a stateless Runnable for interacting with Twitter data based on selected model.

    This runnable is specifically set up for tasks related to Twitter, using Twitter-specific prompts
    and configurations. Like the Reddit runnable, it is stateless and does not maintain conversation history.

    :return: A configured stateless BaseRunnable instance for Twitter interactions.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable


    twitter_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=twitter_system_prompt_template,
        user_prompt_template=twitter_user_prompt_template,
        input_variables=["tweets"],
        required_format=twitter_required_format,
        response_type="json",
        stream=False, # Stateless runnables generally don't need streaming
        stateful=False
    )

    return twitter_runnable