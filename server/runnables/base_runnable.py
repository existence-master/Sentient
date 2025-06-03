from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator
from abc import ABC, abstractmethod

class BaseRunnable(ABC):
    """
    Abstract base class for all runnable components in the system.

    Defines the core interface for components that can process input,
    generate output, and optionally stream results.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the BaseRunnable with an optional model name.

        Args:
            model_name (Optional[str]): The name of the model to be used by the runnable.
                                         Defaults to None.
        """
        self.model_name = model_name

    @abstractmethod
    def run(self,
            user_input: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Abstract method to run the runnable component.

        Args:
            user_input (str): The main input string for the runnable.
            conversation_history (Optional[List[Dict[str, Any]]]): A list of previous
                                                                    conversation turns,
                                                                    each a dictionary
                                                                    with 'role' and 'content'.
                                                                    Defaults to None.
            stream (bool): If True, the method should yield chunks of output as they
                           become available. If False, it should return the complete
                           output at once. Defaults to False.

        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
                If stream is False, returns a dictionary containing the complete output.
                If stream is True, returns a generator that yields dictionaries
                representing chunks of the output.
        """
        pass

    @abstractmethod
    def get_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """
        Abstract method to get the token count for a list of messages.

        Args:
            messages (List[Dict[str, Any]]): A list of message dictionaries.

        Returns:
            int: The total token count for the messages.
        """
        pass

    @abstractmethod
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Abstract method to calculate the cost based on prompt and completion tokens.

        Args:
            prompt_tokens (int): The number of tokens in the prompt.
            completion_tokens (int): The number of tokens in the completion.

        Returns:
            float: The calculated cost.
        """
        pass

def call_model(
    user_message: str,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    response_type: str = "text",
    required_format: Optional[Union[dict, list]] = None,
    stream: bool = False,
    stateful: bool = False,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> Union[Dict[str, Any], List[Any], str, None]:
    """
    Calls the appropriate language model based on configuration or overrides.

    Args:
        user_message (str): The user's message to send to the model.
        provider_override (Optional[str]): Override the default provider (e.g., "ollama", "openrouter").
        model_override (Optional[str]): Override the default model for the chosen provider.
        response_type (str): The expected type of the model's response ('text' or 'json').
        required_format (Optional[Union[dict, list]]): Required format for JSON responses.
        stream (bool): Whether to enable streaming responses.
        stateful (bool): Whether the conversation is stateful.
        chat_history (Optional[List[Dict[str, Any]]]): Previous chat history for stateful conversations.

    Returns:
        Union[Dict[str, Any], List[Any], str, None]: The response from the language model.
    """
    # Moved imports inside the function to avoid circular dependency
    from server.runnables.ollama_runnable import OllamaRunnable
    from server.runnables.openrouter_runnable import OpenRouterRunnable
    from server.config.models_config import DEFAULT_PROVIDER, DEFAULT_OLLAMA_MODEL, DEFAULT_OPENROUTER_MODEL

    chosen_provider = provider_override if provider_override else DEFAULT_PROVIDER
    runnable = None
    model_name = None

    system_prompt_template = "You are a helpful AI assistant."
    user_prompt_template = "{user_input}"
    input_variables = ["user_input"]
    
    inputs = {
        "user_input": user_message,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "input_variables": input_variables,
        "response_type": response_type,
        "required_format": required_format,
        "stream": stream,
        "stateful": stateful
    }
    if chat_history:
        inputs["chat_history"] = chat_history

    if chosen_provider == "ollama":
        model_name = model_override if model_override else DEFAULT_OLLAMA_MODEL
        runnable = OllamaRunnable(model_name=model_name)
    elif chosen_provider == "openrouter":
        model_name = model_override if model_override else DEFAULT_OPENROUTER_MODEL
        runnable = OpenRouterRunnable(model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider: {chosen_provider}")

    if runnable:
        print(f"Calling model '{model_name}' from provider '{chosen_provider}'...")
        if stream:
            return runnable.run(
                user_input=inputs["user_input"],
                conversation_history=inputs.get("chat_history"),
                stream=True
            )
        else:
            return runnable.run(
                user_input=inputs["user_input"],
                conversation_history=inputs.get("chat_history"),
                stream=False
            )
    return None