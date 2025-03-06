import os
from prompts import *  # Importing prompt templates and related utilities from prompts.py
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library (not explicitly used in this file)
from helpers import *  # Importing helper functions from helpers.py
from runnables import *  # Importing other runnable classes or functions from runnables.py
import requests  # For making HTTP requests
from formats import *  # Importing format specifications or utilities from formats.py
import ast  # For Abstract Syntax Tree manipulation, used for safely evaluating strings as Python literals
import json  # For working with JSON data
from sys import platform  # To get system platform information
from typing import Optional, Dict, Any, List, Union, Generator, Tuple  # For type hints
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import keyring


load_dotenv("../.env")  # Load environment variables from .env file


def get_selected_model() -> Tuple[str, str]:
    """
    Fetches the selected model name and provider from the user profile database.

    Reads the `userProfileDb.json` file to determine the currently selected
    language model. If the database file is not found or the 'selectedModel'
    key is missing, it defaults to 'llama3.2:3b' as the model and provider.

    Returns:
        Tuple[str, str]: A tuple containing the selected model name and the provider.
                         For example: ('gpt-4o', 'openai') or ('llama3.2:3b', 'llama3.2:3b').

    Raises:
        ValueError: If the `userProfileDb.json` file path is not set or the file does not exist.
    """
    db_path = "../../userProfileDb.json"
    if not db_path or not os.path.exists(db_path):
        raise ValueError("USER_PROFILE_DB_PATH not set or file not found")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    selected_model = db["userData"].get("selectedModel", "llama3.2:3b")  # Default to llama3.2:3b
    if selected_model == "openai":
        return "gpt-4o", "openai"
    elif selected_model == "claude":
        return "claude-3-7-sonnet-20250219", "claude"
    else:
        return selected_model, selected_model

class BaseRunnable(ABC):
    """
    Abstract base class for runnable language model interactions.

    This class defines the interface for interacting with different language models,
    handling prompt construction, API calls, and response processing. It is designed
    to be subclassed for specific model providers like Ollama, OpenAI, Claude, and Gemini.
    """
    @abstractmethod
    def __init__(self, model_url: str, model_name: str, system_prompt_template: str,
                 user_prompt_template: str, input_variables: List[str], response_type: str,
                 required_format: Optional[Union[dict, list]] = None, stream: bool = False,
                 stateful: bool = False):
        """
        Initializes a BaseRunnable instance.

        Args:
            model_url (str): The URL of the language model API endpoint.
            model_name (str): The name or identifier of the language model.
            system_prompt_template (str): The template for the system prompt, providing context to the model.
            user_prompt_template (str): The template for the user prompt, where user inputs are inserted.
            input_variables (List[str]): A list of variable names to be replaced in the prompt templates.
            response_type (str): The expected type of the model's response ('text' or 'json').
            required_format (Optional[Union[dict, list]], optional):  Required format for JSON responses. Defaults to None.
            stream (bool, optional): Whether to enable streaming responses. Defaults to False.
            stateful (bool, optional): Whether the conversation is stateful, maintaining message history. Defaults to False.
        """
        self.model_url: str = model_url
        """The URL of the language model API endpoint."""
        self.model_name: str = model_name
        """The name or identifier of the language model."""
        self.system_prompt_template: str = system_prompt_template
        """The template for the system prompt."""
        self.user_prompt_template: str = user_prompt_template
        """The template for the user prompt."""
        self.input_variables: List[str] = input_variables
        """A list of variable names to be replaced in the prompts."""
        self.response_type: str = response_type
        """The expected type of the model's response ('text' or 'json')."""
        self.required_format: Optional[Union[dict, list]] = required_format
        """Required format for JSON responses, if applicable."""
        self.stream: bool = stream
        """Whether to use streaming for API requests."""
        self.stateful: bool = stateful
        """Whether the conversation is stateful, maintaining message history."""
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt_template}
        ]
        """Message history for stateful conversations, starting with the system prompt."""

    def build_prompt(self, inputs: Dict[str, Any]) -> None:
        """
        Builds the prompt for the language model by substituting input variables into the templates.

        Formats the user prompt template with the provided inputs and constructs
        the message history based on whether the conversation is stateful or not.
        For stateful conversations, it appends the new user prompt to the existing message history.
        For stateless conversations, it resets the message history to just the system prompt and the current user prompt.

        Args:
            inputs (Dict[str, Any]): A dictionary of input variable names and their values.
        """
        user_prompt = self.user_prompt_template.format(**inputs)

        if self.stateful:
            self.messages.append({"role": "user", "content": user_prompt})
        else:
            self.messages = [{"role": "system", "content": self.messages[0]["content"]}]
            self.messages.append({"role": "user", "content": user_prompt})

    def add_to_history(self, chat_history: List[Dict[str, str]]) -> None:
        """
        Adds chat history to the message list to maintain conversation context.

        Extends the current message list with previous conversation turns, which is
        crucial for stateful conversations where context needs to be preserved across multiple interactions.

        Args:
            chat_history (List[Dict[str, str]]): A list of message dictionaries representing the chat history.
                                                Each dictionary should have 'role' and 'content' keys.
        """
        self.messages.extend(chat_history)

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], List[Any], str, None]:
        """
        Abstract method to invoke the language model with the given inputs and get a complete response.

        This method must be implemented by subclasses to handle the specific API
        call and response processing for each language model provider. It is responsible for sending
        the prompt to the model and returning the full response.

        Args:
            inputs (Dict[str, Any]): A dictionary of input variable names and their values for the prompt.

        Returns:
            Union[Dict[str, Any], List[Any], str, None]: The response from the language model.
                                                        The type of response depends on the 'response_type'
                                                        and could be a JSON object (dict), a list, a string, or None in case of error.
        """
        pass

    @abstractmethod
    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Abstract method to invoke the language model and get a stream of responses.

        This method must be implemented by subclasses to handle streaming responses
        from the language model API. It should send the prompt and yield chunks of the response as they are received.

        Args:
            inputs (Dict[str, Any]): A dictionary of input variable names and their values for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator that yields chunks of the response as strings.
                                                Yields None when the stream ends or encounters an error.
        """
        pass


class OllamaRunnable(BaseRunnable):
    """
    Runnable class for interacting with Ollama language models.

    This class extends BaseRunnable and implements the specific logic for calling
    the Ollama API, handling requests and responses, and streaming. It provides methods to invoke Ollama models
    for both complete responses and streaming responses.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an OllamaRunnable instance.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)

    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the Ollama model to get a complete, non-streaming response.

        Constructs the payload for the Ollama API, sends the POST request, and processes
        the JSON response to extract and return the model's output.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Ollama model, either JSON (dict) or text (str),
                                             or None if there is an error in the API call or response processing.
        """
        self.build_prompt(inputs)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": 4096}, # Set context window size
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

        response = requests.post(self.model_url, json=payload)
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Invokes the Ollama model to get a stream of responses.

        Sends a streaming POST request to the Ollama API and yields chunks of the response
        as they are received. This allows for real-time processing of the model's output.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings.
                                                Yields None if a chunk cannot be processed or the stream ends.
        """
        self.build_prompt(inputs)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": True,
            "options": {"num_ctx": 4096}, # Set context window size
        }

        with requests.post(self.model_url, json=payload, stream=True) as response:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    yield self._handle_stream_line(line)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the Ollama API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the Ollama API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON (dict) or text (str).
                                             Returns None if the response status is not 200 or JSON decoding fails.

        Raises:
            ValueError: If the request fails or the JSON response cannot be decoded.
        """
        if response.status_code == 200:
            try:
                data = response.json().get("message", {}).get("content", "")
                if self.response_type == "json":
                    return json.loads(data)
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON response: {data}. Error: {e}")
        raise ValueError(f"Request failed with status {response.status_code}: {response.text}")

    def _handle_stream_line(self, line: str) -> Optional[str]:
        """
        Handles each line of a streaming response from the Ollama API.

        Parses each line as JSON, extracts the content chunk, and returns it.
        Handles 'done' signals which indicate the end of the stream.

        Args:
            line (str): A line from the streaming response, expected to be a JSON string.

        Returns:
            Optional[str]: The extracted content chunk as a string.
                           Returns None if the line is not valid JSON, or if the stream is marked as 'done'.
        """
        try:
            data = json.loads(line)
            if data.get("done", False): # Changed from True to False, as done:True indicates stream is finished.
                return None
            return data["message"]["content"]
        except json.JSONDecodeError:
            return None


class OpenAIRunnable(BaseRunnable):
    """
    Runnable class for interacting with OpenAI language models.

    This class extends BaseRunnable and implements the specific logic for calling
    the OpenAI API, including authentication, request formatting, response handling, and streaming.
    It supports both text and JSON response formats and handles API key retrieval.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an OpenAIRunnable instance.
        Retrieves the OpenAI API key from environment variables.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")  # only in development
        """OpenAI API key, retrieved from environment variables."""

    def clean_schema_for_openai(self, schema: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Recursively processes the JSON schema to remove or adjust disallowed keywords like 'oneOf'
        for compatibility with OpenAI's API. OpenAI's API has limitations on certain JSON schema keywords.
        This function aims to simplify the schema to increase compatibility.

        Args:
            schema: The JSON schema to clean (can be a dict or list).

        Returns:
            The cleaned schema with unsupported fields like 'oneOf' removed or transformed.
        """
        if isinstance(schema, dict):
            if "oneOf" in schema:
                print("Warning: 'oneOf' found in schema. Replacing with first subschema.")
                # Replace 'oneOf' with the first subschema to maintain basic compatibility
                return self.clean_schema_for_openai(schema["oneOf"][0])
            # Recursively clean all other key-value pairs, excluding 'oneOf'
            return {k: self.clean_schema_for_openai(v) for k, v in schema.items() if k != "oneOf"}
        elif isinstance(schema, list):
            # Recursively clean each item in the list
            return [self.clean_schema_for_openai(item) for item in schema]
        # Return non-dict/list values unchanged (e.g., strings, numbers)
        return schema

    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the OpenAI model to get a complete, non-streaming response.

        Constructs the headers and payload for the OpenAI API, sends the POST request,
        and processes the JSON response to extract and return the model's output.
        Handles structured JSON response formatting if `response_type` is set to "json".

        Args:
            inputs: Input variables for the prompt.

        Returns:
            The response from the OpenAI model, either JSON (dict) or text (str), or None on error.

        Raises:
            ValueError: If the OpenAI API request fails or returns an error status.
        """
        # Build the prompt from inputs (assumes this sets self.messages)
        self.build_prompt(inputs)

        # Set up headers with API key
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Construct the payload
        payload = {
            "model": self.model_name,  # e.g., "gpt-4o-2024-08-06"
            "messages": self.messages,  # Contains system and user messages
            "temperature": 0.7,        # Adjustable as needed
        }

        # Apply structured JSON response format if response_type is "json"
        if self.response_type == "json":
            # Clean the schema to remove unsupported keywords like 'oneOf'
            clean_schema = self.clean_schema_for_openai(self.required_format)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "json_response",
                    "strict": True,
                    "schema": clean_schema
                }
            }

        # Send the request to the OpenAI API
        response = requests.post(self.model_url, headers=headers, json=payload)

        # Handle and return the response
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Invokes the OpenAI model to get a stream of responses.

        Sends a streaming POST request to the OpenAI API and yields content chunks
        as they are received. This enables real-time, chunk-wise processing of the model's output.

        Args:
            inputs: Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings.
                                                Yields None if a chunk is not valid or the stream ends.
        """
        self.build_prompt(inputs)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.7,
            "stream": True
        }

        with requests.post(self.model_url, headers=headers, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    yield self._handle_stream_line(line)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the OpenAI API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response: The HTTP response object from the OpenAI API.

        Returns:
            The processed response content, either JSON (dict) or text (str), or None on error.

        Raises:
            ValueError: If the request to OpenAI API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Model did not return valid JSON. Error: {e}")
            return content
        raise ValueError(f"OpenAI API Error: {response.text}")

    def _handle_stream_line(self, line: bytes) -> Optional[str]:
        """
        Handles each line of a streaming response from the OpenAI API.

        Parses each line, extracts the content delta (the incremental content chunk), and returns it.
        Streaming responses from OpenAI are in a specific 'data: ...' format, which this function parses.

        Args:
            line: A line from the streaming response in bytes.

        Returns:
            Optional[str]: The extracted content chunk as a string.
                           Returns None if the line is not a data line or if the content delta is empty.
        """
        if line.startswith(b"data: "):
            chunk = json.loads(line[6:])
            return chunk["choices"][0]["delta"].get("content", "")
        return None

class ClaudeRunnable(BaseRunnable):
    """
    Runnable class for interacting with Claude language models.

    This class extends BaseRunnable and implements the specific logic for calling
    the Claude API, including authentication, request formatting, response handling, and streaming.
    It handles API key retrieval, request headers specific to Claude, and response parsing for both
    complete and streaming responses.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a ClaudeRunnable instance.
        Retrieves the Claude API key from environment variables.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)
        self.api_key: Optional[str] = os.getenv("CLAUDE_API_KEY") # only in development
        """Claude API key, retrieved from environment variables."""

        # # Retrieve the encrypted API key from Keyring - commented out for now.
        # encrypted_key = keyring.get_password("electron-openid-oauth", "claude")

        # # Check if the encrypted key exists
        # if encrypted_key:
        #     try:
        #         # Define the utility server URL and endpoint
        #         url = "http://localhost:5005/decrypt"
        #         # Prepare the JSON payload with the encrypted data
        #         payload = {"encrypted_data": encrypted_key}
        #         # Make a POST request to the /decrypt endpoint
        #         response = requests.post(url, json=payload)

        #         # Check if the request was successful
        #         if response.status_code == 200:
        #             # Extract the decrypted data from the response
        #             decrypted_data = response.json().get("decrypted_data")
        #             self.api_key = decrypted_data
        #         else:
        #             # Handle non-200 status codes (e.g., 500 from server errors)
        #             print(f"Failed to decrypt API key: {response.status_code} - {response.text}")
        #             self.api_key = None
        #     except requests.exceptions.RequestException as e:
        #         # Handle network-related errors (e.g., server down, connection issues)
        #         print(f"Error connecting to decryption service: {e}")
        #         self.api_key = None
        # else:
        #     # Handle the case where no encrypted key is found in Keyring
        #     print("No encrypted API key found in Keyring.")
        #     self.api_key = None

    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the Claude model to get a complete, non-streaming response.

        Constructs the headers and payload for the Claude API, sends the POST request,
        and processes the JSON response to extract and return the model's output.
        If `response_type` is "json", it adds JSON formatting instructions to the prompt to guide the model.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Claude model, either JSON (dict) or text (str),
                                             or None if there is an error in the API call or response processing.

        Raises:
            ValueError: If the Claude API request fails or returns an error status.
        """
        # Build the initial prompt from inputs
        self.build_prompt(inputs)

        # If response_type is "json", modify the prompt to include formatting instructions
        if self.response_type == "json" and self.required_format:
            # Convert self.required_format to a JSON string for inclusion in the prompt
            schema_str = json.dumps(self.required_format, indent=2)
            # Add instructions to the last message (assumed to be the user message)
            instruction = (
                f"\n\nPlease format your response as a JSON object that conforms to the following schema:\n"
                f"```json\n{schema_str}\n```"
            )
            self.messages[-1]["content"] += instruction

        # Set up headers with API key and required Claude-specific headers
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01", # Required Claude API version header
            "content-type": "application/json" # Explicitly set content type to JSON
        }

        # Construct the payload
        payload = {
            "model": self.model_name,  # Hardcoded model name for Claude
            "messages": self.messages,
            "max_tokens": 4096 # Maximum number of tokens in the response
        }

        # Send the request to the Claude API
        response = requests.post(self.model_url, headers=headers, json=payload)

        # Handle and return the response
        return self._handle_response(response)
    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Invokes the Claude model to get a stream of responses.

        Sends a streaming POST request to the Claude API and yields content chunks
        as they are received. This allows for real-time processing of Claude's output.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings.
                                                Yields None if a chunk is not valid or the stream ends.
        """
        self.build_prompt(inputs)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01", # Required Claude API version header
            "content-type": "application/json" # Explicitly set content type to JSON
        }
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "max_tokens": 4096, # Maximum number of tokens in the response
            "stream": True
        }

        response = requests.post(self.model_url, headers=headers, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                yield self._handle_stream_line(line)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the Claude API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the Claude API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON (dict) or text (str).
                                             Returns None if the response status is not 200 or JSON decoding fails.

        Raises:
            ValueError: If the request to Claude API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = " ".join([block["text"] for block in data["content"]]) # Claude returns content as a list of blocks
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Model did not return valid JSON. Error: {e}")
            return content
        raise ValueError(f"Claude API Error: {response.text}")

    def _handle_stream_line(self, line: bytes) -> Optional[str]:
        """
        Handles each line of a streaming response from the Claude API.

        Parses each line as JSON, extracts the content blocks, and concatenates their text.
        Claude's streaming API sends updates as JSON lines, which need to be parsed and processed.

        Args:
            line (bytes): A line from the streaming response in bytes.

        Returns:
            Optional[str]: The extracted content chunk as a string.
                           Returns None if the line is not valid JSON or if the content is empty.
        """
        try:
            data = json.loads(line.decode("utf-8"))
            return " ".join([block["text"] for block in data.get("content", [])]) # Extract text from content blocks
        except json.JSONDecodeError:
            return None

class GeminiRunnable(BaseRunnable):
    """
    Runnable class for interacting with Gemini language models.

    This class extends BaseRunnable and implements the specific logic for calling
    the Gemini API, including authentication, request formatting, response handling, and streaming.
    It manages API key authorization, request payloads formatted for Gemini, and response parsing.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a GeminiRunnable instance.
        Retrieves the Gemini API key from environment variables.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY")  # only in development
        """Gemini API key, retrieved from environment variables."""
        # # Retrieve the encrypted API key from Keyring - commented out for now.
        # encrypted_key = keyring.get_password("electron-openid-oauth", "gemini")

        # # Check if the encrypted key exists
        # if encrypted_key:
        #     try:
        #         # Define the utility server URL and endpoint
        #         url = "http://localhost:5005/decrypt"
        #         # Prepare the JSON payload with the encrypted data
        #         payload = {"encrypted_data": encrypted_key}
        #         # Make a POST request to the /decrypt endpoint
        #         response = requests.post(url, json=payload)

        #         # Check if the request was successful
        #         if response.status_code == 200:
        #             # Extract the decrypted data from the response
        #             decrypted_data = response.json().get("decrypted_data")
        #             self.api_key = decrypted_data
        #         else:
        #             # Handle non-200 status codes (e.g., 500 from server errors)
        #             print(f"Failed to decrypt API key: {response.status_code} - {response.text}")
        #             self.api_key = None
        #     except requests.exceptions.RequestException as e:
        #         # Handle network-related errors (e.g., server down, connection issues)
        #         print(f"Error connecting to decryption service: {e}")
        #         self.api_key = None
        # else:
        #     # Handle the case where no encrypted key is found in Keyring
        #     print("No encrypted API key found in Keyring.")
        #     self.api_key = None

    def clean_schema_for_gemini(self, schema: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """
        Cleans and adapts a JSON schema for compatibility with the Gemini API.

        Gemini has specific requirements and limitations on JSON schemas, particularly for function calling and structured output.
        This function removes unsupported keywords and adjusts the schema to align with Gemini's capabilities.
        It specifically handles 'oneOf' and ensures that object types are correctly defined.

        Args:
            schema: The JSON schema to be cleaned (can be a dict or list).

        Returns:
            The cleaned JSON schema, compatible with Gemini API requirements.
        """
        supported_keywords = {"enum", "items", "maxItems", "nullable", "properties", "required", "type"}
        if isinstance(schema, dict):
            # Handle 'oneOf' by selecting the first subschema and cleaning it
            if "oneOf" in schema:
                print("Warning: 'oneOf' found in schema. Replacing with first subschema.")
                subschema = schema["oneOf"][0]
                return self.clean_schema_for_gemini(subschema)

            cleaned = {}
            for k, v in schema.items():
                if k == "properties":
                    cleaned["properties"] = {prop: self.clean_schema_for_gemini(prop_schema)
                                            for prop, prop_schema in v.items()}
                elif k in supported_keywords:
                    cleaned[k] = self.clean_schema_for_gemini(v)

            # Ensure 'type' is set for objects with properties
            if "properties" in cleaned and "type" not in cleaned:
                cleaned["type"] = "object"

            # Handle object type validation and conversion for Gemini
            if cleaned.get("type") == "object":
                if "properties" not in cleaned or not cleaned["properties"]:
                    # Convert empty object to string type for JSON workaround
                    print("Warning: Empty object detected. Converting to JSON string schema for Gemini compatibility.")
                    return {
                        "type": "string",
                        "description": cleaned.get("description", "Dynamic parameters as a JSON string") + " (JSON string)"
                    }
                if "required" in cleaned:
                    defined_properties = set(cleaned["properties"].keys())
                    cleaned["required"] = [prop for prop in cleaned["required"] if prop in defined_properties]
                    if not cleaned["required"]:
                        del cleaned["required"]

            return cleaned
        elif isinstance(schema, list):
            return [self.clean_schema_for_gemini(item) for item in schema]
        return schema

    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the Gemini model to get a complete, non-streaming response.

        Constructs the payload for the Gemini API, sends the POST request, and processes
        the JSON response to extract and return the model's output.
        Handles system instructions and formats the payload according to Gemini's API requirements.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Gemini model, either JSON (dict) or text (str),
                                             or None if there is an error in the API call or response processing.

        Raises:
            ValueError: If the Gemini API request fails or returns an error status.
        """
        self.build_prompt(inputs)
        system_instruction = None
        if self.messages and self.messages[0]["role"].lower() == "system":
            system_content = self.messages[0]["content"]
            system_instruction = {"parts": [{"text": system_content}]}
            conversation_messages = self.messages[1:]
        else:
            conversation_messages = self.messages

        def map_role(role: str) -> str:
            """Maps a role name to Gemini's role names ('user' or 'model')."""
            return "model" if role.lower() == "assistant" else "user"

        contents = [{"role": map_role(msg["role"]), "parts": [{"text": msg["content"]}]}
                    for msg in conversation_messages]
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if self.response_type == "json" and self.required_format:
            generation_config = {"response_mime_type": "application/json"}
            if self.required_format is not None:
                clean_schema = self.clean_schema_for_gemini(self.required_format)
                generation_config["response_schema"] = clean_schema
            payload["generationConfig"] = generation_config

        response = requests.post(
            f"{self.model_url}?key={self.api_key}", # API key is passed as a query parameter
            json=payload,
            headers={"Content-Type": "application/json"} # Explicitly set content type to JSON
        )
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the Gemini API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the Gemini API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON (dict) or text (str).
                                             Returns None if the response status is not 200 or JSON decoding fails.

        Raises:
            ValueError: If the request to Gemini API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = "".join([part["text"] for part in data["candidates"][0]["content"]["parts"]]) # Gemini returns content in 'parts'
            if self.response_type == "json" and self.required_format:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Model did not return valid JSON. Error: {e}")
            return content
        raise ValueError(f"Gemini API Error: {response.text}")

    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Union[Dict[str, Any], str, None], None, None]:
        """
        Invokes the Gemini model to get a stream of responses.

        Currently implemented as a single yield due to limited streaming support,
        but can be extended for true streaming later. In the current implementation, it behaves like a non-streaming call.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Union[Dict[str, Any], str, None], None, None]: A generator yielding the response.
                                                                    In the current implementation, it yields only once.
        """
        self.build_prompt(inputs)

        system_instruction = None
        if self.messages and self.messages[0]["role"].lower() == "system":
            system_content = self.messages[0]["content"]
            system_instruction = {"parts": [{"text": system_content}]}
            conversation_messages = self.messages[1:]
        else:
            conversation_messages = self.messages

        def map_role(role: str) -> str:
            """Maps a role name to Gemini's role names ('user' or 'model')."""
            role_lower = role.lower()
            return "model" if role_lower == "assistant" else "user"

        contents = [{"role": map_role(msg["role"]), "parts": [{"text": msg["content"]}]}
                    for msg in conversation_messages]

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        response = requests.post(
            f"{self.model_url}?key={self.api_key}", # API key is passed as a query parameter
            json=payload,
            headers={"Content-Type": "application/json"} # Explicitly set content type to JSON
        )

        yield self._handle_response(response)

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