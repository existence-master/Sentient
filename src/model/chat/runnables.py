import os
from prompts import *
from wrapt_timeout_decorator import *
from helpers import *
import json
import requests
import asyncio
from typing import Dict, Any, List, Union, Optional, Generator, AsyncGenerator
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

# Helper function to get selected model
def get_selected_model():
    """Fetch the selected model name from userProfile.json."""
    db_path = os.getenv("USER_PROFILE_DB_PATH")
    if not db_path or not os.path.exists(db_path):
        raise ValueError("USER_PROFILE_DB_PATH not set or file not found")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    return db["userData"].get("selectedModel", "llama3.2:3b")  # Default to llama3.2:3b

# --- CustomRunnable Class Definition ---
class CustomRunnable:
    """
    A custom runnable class to interact with language models via API requests.

    This class handles building prompts, sending requests to a model API,
    and processing responses. It supports both standard and streaming responses,
    and can maintain stateful conversations.
    """

    def __init__(
        self,
        model_url: str,
        model_name: str,
        system_prompt_template: str,
        user_prompt_template: str,
        input_variables: List[str],
        response_type: str,
        required_format: Optional[Union[dict, list]] = None,
        stream: bool = False,
        stateful: bool = False,
    ):
        """
        Initialize the CustomRunnable instance.

        Args:
            model_url (str): URL of the model API endpoint.
            model_name (str): Name of the model to use.
            system_prompt_template (str): Template for the system prompt.
            user_prompt_template (str): Template for the user prompt.
            input_variables (List[str]): List of variables required by the prompt templates.
            response_type (str): Expected response type, e.g., "json", "chat".
            required_format (Optional[Union[dict, list]]): Expected output format for JSON responses (optional).
            stream (bool): Whether to use streaming for responses (default: False).
            stateful (bool): Whether the runnable should maintain conversation state (default: False).
        """
        self.model_url: str = model_url  # Model API endpoint URL
        self.model_name: str = model_name  # Model name to be used in API requests
        self.system_prompt_template: str = (
            system_prompt_template  # System prompt template string
        )
        self.user_prompt_template: str = (
            user_prompt_template  # User prompt template string
        )
        self.input_variables: List[str] = (
            input_variables  # List of input variable names
        )
        self.required_format: Optional[Union[dict, list]] = (
            required_format  # Required format for JSON responses, if applicable
        )
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt_template}
        ]  # Initialize messages list with system prompt
        self.response_type: str = (
            response_type  # Expected response type ("json", "chat", etc.)
        )
        self.stream: bool = stream  # Flag to enable streaming responses
        self.stateful: bool = stateful  # Flag to indicate if the runnable is stateful

    def build_prompt(self, inputs: Dict[str, Any]):
        """
        Build the prompt by substituting input variables into the user prompt template.

        If the runnable is stateful, it appends the new user prompt to the existing message history.
        Otherwise, it resets the message history to just the system prompt before adding the new user prompt.

        Args:
            inputs (Dict[str, Any]): Dictionary of input variables to replace in the prompt template.
        """
        if self.stateful:  # If stateful, maintain conversation history
            user_prompt: str = self.user_prompt_template.format(
                **inputs
            )  # Format user prompt with inputs

            new_messages: List[Dict[str, str]] = [  # Create new message in chat format
                {"role": "user", "content": user_prompt},
            ]

            for message in new_messages:  # Append new messages to existing history
                self.messages.append(message)
        else:  # If not stateful, start with a fresh system prompt each time
            self.messages = [
                {"role": "system", "content": self.system_prompt_template}
            ]  # Reset messages to just system prompt

            user_prompt: str = self.user_prompt_template.format(
                **inputs
            )  # Format user prompt with inputs
            self.messages.append(
                {"role": "user", "content": user_prompt}
            )  # Append new user prompt

    def add_to_history(self, chat_history: List[Dict[str, str]]):
        """
        Add chat history to the messages list.

        This method is used to incorporate previous chat turns into the current prompt,
        maintaining context for stateful conversations.

        Args:
            chat_history (List[Dict[str, str]]): A list of chat messages, each a dictionary
                                                 with 'role' ("user" or "assistant") and 'content'.
        """
        for chat in chat_history:  # Iterate through provided chat history
            self.messages.append(
                chat
            )  # Append each message to the current messages list

    def invoke(
        self, inputs: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Any], str, None]:
        """
        Execute the model call, process the response, and return the output.

        This method builds the prompt, sends a request to the model API,
        and then processes the JSON response to extract and return the content.
        It handles potential JSON decoding errors and HTTP request failures.

        Args:
            inputs (Dict[str, Any]): Dictionary of input values for prompt formatting.

        Returns:
            Union[Dict[str, Any], List[Any], str, None]: Processed output from the model response.
                                                         Returns a dict or list if response is JSON,
                                                         a string for other response types, or None on failure.

        Raises:
            ValueError: If the HTTP request fails or if JSON decoding fails and response_type is "json".
        """
        self.build_prompt(inputs)  # Build the prompt using the input

        payload: Dict[str, Any] = {  # Construct the payload for the API request
            "model": self.model_name,
            "messages": self.messages,
            "stream": False,  # Set stream to False for standard response
            "options": {  # Model options
                "num_ctx": 4096  # Context window size
            },
        }

        if self.response_type == "json":  # If expecting a JSON response, set the format
            if (
                platform == "win32"
            ):  # Conditional format setting based on platform (Windows specific handling)
                payload["format"] = (
                    self.required_format
                )  # Set format directly for Windows
            else:
                payload["format"] = json.dumps(
                    self.required_format
                )  # Serialize format to JSON string for non-Windows

        headers: Dict[str, str] = {
            "Content-Type": "application/json"
        }  # Set headers for JSON request

        response: requests.Response = requests.post(
            self.model_url, headers=headers, data=json.dumps(payload)
        )  # Send POST request to model API

        if response.status_code == 200:  # Check if the request was successful
            try:
                data: str = (
                    response.json().get("message", {}).get("content", "")
                )  # Extract content from JSON response
                try:  # Attempt to parse response data as JSON
                    parsed_data: Union[Dict[str, Any], List[Any]] = (
                        extract_and_fix_json(data)
                    )  # Parse and fix JSON, if necessary
                    return parsed_data  # Return parsed JSON data
                except (
                    Exception
                ):  # If JSON parsing fails, proceed to next parsing attempt
                    pass  # Fallback to literal evaluation if JSON parsing fails

                try:  # Attempt to parse response data as Python literal (e.g., list, dict, string)
                    parsed_data = ast.literal_eval(
                        data
                    )  # Safely evaluate string as Python literal
                    return parsed_data  # Return parsed Python literal data
                except (
                    ValueError,
                    SyntaxError,
                ):  # Catch errors during literal evaluation
                    pass  # If literal evaluation fails, return raw data

                return data  # Return raw string data if no parsing is successful

            except json.JSONDecodeError:  # Handle JSONDecodeError specifically
                raise ValueError(
                    f"Failed to decode JSON response: {response.text}"
                )  # Raise ValueError for JSON decode failure
        else:  # Handle non-200 status codes
            raise ValueError(
                f"Request failed with status code {response.status_code}: {response.text}"
            )  # Raise ValueError for HTTP error

    def stream_response(
        self, inputs: Dict[str, Any]
    ) -> Generator[Optional[str], None, None]:
        """
        Generate a streaming response from the model API.

        This method is similar to `invoke` but handles streaming responses.
        It yields tokens as they are received from the API, allowing for real-time processing
        of the model's output.

        Args:
            inputs (Dict[str, Any]): Dictionary of input values for prompt formatting.

        Yields:
            Generator[Optional[str], None, None]: A generator that yields tokens (strings) from the streaming response.
                                                 Yields None to signal the end of the stream.
        """
        self.build_prompt(inputs)  # Build the prompt using the inputs

        payload: Dict[str, Any] = {  # Construct payload for streaming request
            "model": self.model_name,
            "messages": self.messages,
            "stream": True,  # Set stream to True for streaming response
            "options": {  # Model options
                "num_ctx": 4096  # Context window size
            },
        }

        if (
            self.response_type == "json"
        ):  # If expecting JSON response format, set format in payload
            payload["format"] = (
                self.required_format
            )  # Set the required format for JSON response

        with requests.post(
            self.model_url, json=payload, stream=True
        ) as response:  # Send streaming POST request
            for line in response.iter_lines(
                decode_unicode=True
            ):  # Iterate over response lines
                if line:  # Check if line is not empty
                    try:
                        data: Dict[str, Any] = json.loads(
                            line
                        )  # Load JSON data from line

                        if data.get("done", True):  # Check if stream is done
                            if (
                                data.get("done_reason") == "load"
                            ):  # Skip if done reason is 'load'
                                continue  # Continue to next line
                            else:  # If done for other reasons, yield None and break
                                token: None = (
                                    None  # Set token to None to signal end of stream
                                )
                                yield token  # Yield None token
                                break  # Break from loop

                        token: Optional[str] = data["message"][
                            "content"
                        ]  # Extract token content from message
                        if token:  # Check if token is not empty
                            yield token  # Yield the extracted token

                    except (
                        json.JSONDecodeError
                    ):  # Handle JSONDecodeError during line processing
                        continue  # Continue to next line if JSON decode fails

def get_chat_runnable(chat_history: List[Dict[str, str]]) -> CustomRunnable:
    """
    Initialize and configure a CustomRunnable for chat interactions.

    This runnable is designed for conversational exchanges, using a chat-specific system prompt
    and user prompt templates. It is configured for streaming responses and maintains stateful conversation history.

    Args:
        chat_history (List[Dict[str, str]]): Initial chat history to prime the conversation.

    Returns:
        CustomRunnable: Configured CustomRunnable instance for chat interactions.
    """
    chat_runnable: CustomRunnable = CustomRunnable(  # Initialize CustomRunnable for chat
        model_url=os.getenv("BASE_MODEL_URL"),
        model_name=get_selected_model(),
        system_prompt_template=chat_system_prompt_template,  # System prompt for chat
        user_prompt_template=chat_user_prompt_template,  # User prompt template for chat
        input_variables=[
            "query",
            "user_context",
            "internet_context",
            "name",
            "personality",
        ],  # Input variables for chat prompts
        response_type="chat",  # Response type is chat
        stream=True,  # Enable streaming for chat responses
        stateful=True,  # Enable stateful conversation
    )

    chat_runnable.add_to_history(chat_history)  # Add initial chat history to runnable
    return chat_runnable  # Return configured chat runnable
