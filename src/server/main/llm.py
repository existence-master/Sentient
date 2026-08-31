import os
import logging
import time
import httpx
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model

from main.config import (OPENAI_API_KEY, OPENAI_API_BASE_URL,
                         OPENAI_MODEL_NAME)

logger = logging.getLogger(__name__)

# Omniroute Blood Flow Integration
OMNIROUTE_URL = os.environ.get("OMNIROUTE_URL", "http://127.0.0.1:20128")
OMNIROUTE_TIMEOUT = int(os.environ.get("OMNIROUTE_TIMEOUT", "60"))

class LLMProviderDownError(Exception):
    """Custom exception for when all LLM providers are down."""
    pass


def check_omniroute_available() -> bool:
    """Check if Omniroute blood flow is running."""
    try:
        response = httpx.get(f"{OMNIROUTE_URL}/v1/models", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def run_agent(system_message: str, function_list: list, messages: list):
    """
    Initializes and runs a Qwen Assistant with Omniroute blood flow routing.
    
    If Omniroute is available on :20128, routes through it.
    Falls back to configured LLM provider if Omniroute is down.
    """
    # Check if Omniroute blood flow is available
    if check_omniroute_available():
        logger.info("Routing through Omniroute blood flow on :20128")
        # Use Omniroute as the model server
        llm_cfg = {
            'model': OPENAI_MODEL_NAME,
            'model_server': OMNIROUTE_URL,
            'api_key': OPENAI_API_KEY or 'omniroute-key',
            'generate_cfg': {
                'max_input_tokens': 128000,
                'tools': [{"urlContext": {}}]
            }
        }
    else:
        logger.info("Omniroute not available, using configured LLM provider")
        llm_cfg = {
            'model': OPENAI_MODEL_NAME,
            'model_server': OPENAI_API_BASE_URL,
            'api_key': OPENAI_API_KEY,
            'generate_cfg': {
                'max_input_tokens': 128000,
                'tools': [{"urlContext": {}}]
            }
        }

    try:
        logger.info(f"Running agent with model: {llm_cfg['model']}")
        bot = Assistant(llm=llm_cfg, system_message=system_message, function_list=function_list or [])
        yield from bot.run(messages=messages)
    except Exception as e:
        error_message = f"Agent run failed: {e}"
        logger.error(error_message, exc_info=True)
        # Re-raise as a specific exception to be caught by the caller
        raise LLMProviderDownError(error_message) from e
