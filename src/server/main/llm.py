import os
import logging
import time
import httpx
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model

from main.config import (OPENAI_API_KEY, OPENAI_API_BASE_URL,
                         OPENAI_MODEL_NAME)

logger = logging.getLogger(__name__)

class LLMProviderDownError(Exception):
    """Custom exception for when all LLM providers are down."""
    pass


def run_agent(system_message: str, function_list: list, messages: list):
    """
    Initializes and runs a Qwen Assistant.
    Relies on the underlying LLM provider (e.g., LiteLLM) to handle fallbacks and retries.
    """
    if not OPENAI_API_KEY:
        raise ValueError("No OpenAI API key configured.")

    llm_cfg = {
        'model': OPENAI_MODEL_NAME,
        'model_server': OPENAI_API_BASE_URL,
        'api_key': OPENAI_API_KEY,
        'generate_cfg': {
            'max_input_tokens': 128000, # Set a high limit to avoid truncation errors
        }
    }

    try:
        logger.info(f"Running agent with model: {OPENAI_MODEL_NAME}")
        bot = Assistant(llm=llm_cfg, system_message=system_message, function_list=function_list or [])
        yield from bot.run(messages=messages)
    except Exception as e:
        error_message = f"Agent run failed: {e}"
        logger.error(error_message, exc_info=True)
        # Re-raise as a specific exception to be caught by the caller
        raise LLMProviderDownError(error_message) from e
