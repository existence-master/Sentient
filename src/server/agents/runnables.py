import os
from typing import Dict, List, Optional, Tuple, Type
from dotenv import load_dotenv

from .formats import *
from .prompts import *
from server.app.base import *

load_dotenv("server/.env")

def get_chat_runnable(chat_history: List[Dict[str, str]]) -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class for chat interactions based on the selected model.

    Determines the model provider (Ollama, OpenAI, Claude, Gemini) based on the
    selected model name from `get_selected_model()`. It then returns an instance
    of the corresponding Runnable class, configured with API URLs and chat prompts.
    This function is designed for conversational chat interfaces.

    Args:
        chat_history (List[Dict[str, str]]): The chat history to initialize the runnable with. This provides context
                                            from previous turns in the conversation.

    Returns:
        BaseRunnable: An instance of a Runnable class (OllamaRunnable, OpenAIRunnable, ClaudeRunnable, or GeminiRunnable)
                      that is configured for chat interactions, including streaming and stateful conversation management.
    """
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    """Mapping of model providers to their API URLs and Runnable classes."""

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

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
    """Configures the Runnable instance with chat-specific settings: streaming, stateful, and chat prompts."""

    runnable.add_to_history(chat_history)
    return runnable


def get_agent_runnable(chat_history: List[Dict[str, str]]) -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class for agent-like behavior based on the selected model.

    Determines the model provider and returns a Runnable configured for agent-like tasks,
    including JSON response format and agent-specific prompts. Agents typically require structured output for tool use.

    Args:
        chat_history (List[Dict[str, str]]): Initial chat history to provide context to the agent.
                                            This history might include initial instructions or context for the agent's task.

    Returns:
        BaseRunnable: A Runnable instance configured for agent behavior. This instance is set up to expect
                      JSON responses and is initialized with agent-specific prompts and state management.
    """
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    """Mapping of model providers to their API URLs and Runnable classes."""

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable # Or potentially CustomRunnable if you want to force CustomRunnable for agent

    runnable: BaseRunnable = runnable_class( # Enforce CustomRunnable for agent features
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=agent_system_prompt_template,
        user_prompt_template=agent_user_prompt_template,
        input_variables=[
            "query",
            "name",
            "user_context",
            "internet_context", # Note: Typo in original function 'internet_contextname' is kept for consistency
            "personality",
        ],
        required_format=agent_required_format,
        response_type="json",
        stateful=True,
    )
    """Configures the Runnable instance for agent-like behavior, expecting JSON responses and maintaining state."""

    runnable.add_to_history(chat_history)
    return runnable


def get_tool_runnable(
    system_prompt_template: str,
    user_prompt_template: str,
    required_format: dict,
    input_variables: List[str],
) -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class for tool execution based on the selected model.

    Determines the model provider and returns a Runnable configured for tool calls,
    with JSON response format and stateless operation. Tool execution typically requires structured JSON output
    and each call is often independent without needing to retain conversation history.

    Args:
        system_prompt_template (str): System prompt template specific to the tool. Provides context and instructions
                                     for the model when executing the tool.
        user_prompt_template (str): User prompt template for tool interaction. Formats user input for the tool.
        required_format (dict): Required JSON format for tool responses. Defines the structure of the JSON output
                                expected from the model when it uses the tool.
        input_variables (List[str]): List of input variables for tool prompts. Variables to be replaced in the
                                     prompt templates with actual values when the tool is invoked.

    Returns:
        BaseRunnable: A Runnable instance configured for tool execution. This instance is set up to expect
                      JSON responses and operates in a stateless manner, suitable for independent tool calls.
    """
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    """Mapping of model providers to their API URLs and Runnable classes."""

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable # Or potentially CustomRunnable

    runnable: BaseRunnable = runnable_class( # Enforce CustomRunnable for tool features
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        input_variables=input_variables,
        required_format=required_format,
        response_type= "json" if required_format else "chat",
    )
    """Configures the Runnable instance for tool execution, expecting JSON responses and operating stateless."""

    return runnable


def get_reflection_runnable() -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class for reflection tasks based on the selected model.

    Determines the model provider and returns a Runnable configured for reflection,
    with non-streaming chat responses and stateless operation.

    Returns:
        BaseRunnable: A Runnable instance configured for reflection tasks.
    """
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider = get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback

    runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=reflection_system_prompt_template,
        user_prompt_template=reflection_user_prompt_template,
        input_variables=["tool_results"],
        response_type="chat",
        stream=False,  # Changed to False for non-streaming output
    )

    return runnable


def get_inbox_summarizer_runnable() -> BaseRunnable:
    """
    Factory function to get the appropriate Runnable class for inbox summarization tasks based on the selected model.

    Determines the model provider and returns a Runnable configured for inbox summarization,
    with streaming chat responses and stateless operation. Inbox summarization is typically a stateless task
    where each summarization is independent, and streaming allows for quicker feedback.

    Returns:
        BaseRunnable: A Runnable instance configured for inbox summarization. This instance is set up for streaming
                      chat responses and operates in a stateless manner, suitable for independent summarization calls.
    """
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    """Mapping of model providers to their API URLs and Runnable classes."""

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable # Or potentially CustomRunnable

    runnable: BaseRunnable = runnable_class( # Enforce CustomRunnable for inbox summarizer features
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=inbox_summarizer_system_prompt_template,
        user_prompt_template=inbox_summarizer_user_prompt_template,
        input_variables=["tool_result"],
        response_type="chat",
        stream=True,
    )
    """Configures the Runnable instance for inbox summarization, using streaming chat responses and operating stateless."""

    return runnable

def get_priority_runnable() -> BaseRunnable:
    """
    Creates a Runnable instance for determining task priority using an LLM.

    Returns:
        BaseRunnable: A configured runnable for priority determination.
    """
    # Model selection logic (adapt to your existing setup)
    model_mapping: Dict[str, Tuple[Optional[str], Type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider = get_selected_model()  # Assume this function exists in your codebase

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback

    # Create the runnable
    runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=priority_system_prompt_template,
        user_prompt_template=priority_user_prompt_template,
        input_variables=["task_description"],
        required_format=priority_required_format,
        response_type="json",
        stream=False,  # We want a single, non-streaming response
    )

    return runnable