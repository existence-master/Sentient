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
from typing import Optional, Dict, Any, List, Union, Generator  # For type hints
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import keyring


load_dotenv("../.env")  # Load environment variables from .env file


def get_selected_model() -> str:
    """
    Fetches the selected model name from the user profile database.

    Reads the `userProfileDb.json` file to determine the currently selected
    language model. If the database file is not found or the 'selectedModel'
    key is missing, it defaults to 'llama3.2:3b'.

    Returns:
        str: The name of the selected model.

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
        self.model_name: str = model_name
        self.user_prompt_template: str = user_prompt_template
        self.input_variables: List[str] = input_variables
        self.response_type: str = response_type
        self.required_format: Optional[Union[dict, list]] = required_format
        self.stream: bool = stream
        self.stateful: bool = stateful
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt_template}
        ]

    def build_prompt(self, inputs: Dict[str, Any]) -> None:
        """
        Builds the prompt for the language model by substituting input variables.

        Formats the user prompt template with the provided inputs and constructs
        the message history based on whether the conversation is stateful or not.

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
        crucial for stateful conversations where context needs to be preserved.

        Args:
            chat_history (List[Dict[str, str]]): A list of message dictionaries representing the chat history.
                                                Each dictionary should have 'role' and 'content' keys.
        """
        self.messages.extend(chat_history)

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], List[Any], str, None]:
        """
        Abstract method to invoke the language model with the given inputs and get a complete response.

        This method should be implemented by subclasses to handle the specific API
        call and response processing for each language model provider.

        Args:
            inputs (Dict[str, Any]): A dictionary of input variable names and their values for the prompt.

        Returns:
            Union[Dict[str, Any], List[Any], str, None]: The response from the language model.
                                                        The type of response depends on the 'response_type'
                                                        and could be a JSON object (dict), a list, a string, or None.
        """
        pass

    @abstractmethod
    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Abstract method to invoke the language model and get a stream of responses.

        This method should be implemented by subclasses to handle streaming responses
        from the language model API.

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
    the Ollama API, handling requests and responses, and streaming.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an OllamaRunnable instance.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)

    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the Ollama model to get a complete response.

        Constructs the payload for the Ollama API, sends the request, and processes
        the response to return the model's output.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Ollama model, either JSON or text, or None on error.
        """
        self.build_prompt(inputs)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": 4096},
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

        Sends a streaming request to the Ollama API and yields chunks of the response
        as they are received.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings, or None for errors/end.
        """
        self.build_prompt(inputs)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": True,
            "options": {"num_ctx": 4096},
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
            Union[Dict[str, Any], str, None]: The processed response content, either JSON or text, or None on error.

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
        Handles 'done' signals and JSON decoding errors.

        Args:
            line (str): A line from the streaming response.

        Returns:
            Optional[str]: The extracted content chunk as a string, or None if the line is not valid or stream is done.
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
    the OpenAI API, handling authentication, requests, responses, and streaming.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an OpenAIRunnable instance.
        Retrieves the OpenAI API key from environment variables or Keyring.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY") # only in development

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
        Invokes the OpenAI model to get a complete response.

        Constructs the headers and payload for the OpenAI API, sends the request,
        and processes the response.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the OpenAI model, either JSON or text, or None on error.

        Raises:
            ValueError: If the OpenAI API request fails.
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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "json_response",
                    "strict": True,
                    "schema": self.required_format  # Assumes this is the schema from your query
                }
            }
        
        # Send the request to the OpenAI API
        response = requests.post(self.model_url, headers=headers, json=payload)
        
        # Handle and return the response
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Invokes the OpenAI model to get a stream of responses.

        Sends a streaming request to the OpenAI API and yields content chunks
        as they are received.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings, or None for errors/end.
        """
        print(inputs)
        self.build_prompt(inputs)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.7,
            "stream": True
        }

        with requests.post(self.model_url, headers=headers, json=payload, stream=True) as response:
            print("Response: ", response)
            for line in response.iter_lines():
                print("Line: ", line)
                if line:
                    yield self._handle_stream_line(line)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the OpenAI API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the OpenAI API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON or text, or None on error.

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

        Parses each line, extracts the content delta, and returns it.

        Args:
            line (bytes): A line from the streaming response in bytes.

        Returns:
            Optional[str]: The extracted content chunk as a string, or None if the line is not a data line or content is empty.
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
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a ClaudeRunnable instance.
        Retrieves the Claude API key from environment variables or Keyring.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)
        self.api_key: Optional[str] = os.getenv("CLAUDE_API_KEY") # only in development

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
        Invokes the Claude model to get a complete response.

        Constructs the headers and payload for the Claude API, sends the request,
        and processes the response. Adds JSON formatting instructions to the prompt if response_type is "json".

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Claude model, either JSON or text, or None on error.

        Raises:
            ValueError: If the Claude API request fails.
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
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Construct the payload
        payload = {
            "model": self.model_name,  # Hardcoded model name for Claude
            "messages": self.messages,
            "max_tokens": 4096
        }

        # Send the request to the Claude API
        response = requests.post(self.model_url, headers=headers, json=payload)

        # Handle and return the response
        return self._handle_response(response)
    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        """
        Invokes the Claude model to get a stream of responses.

        Sends a streaming request to the Claude API and yields content chunks
        as they are received.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Optional[str], None, None]: A generator yielding response chunks as strings, or None for errors/end.
        """
        self.build_prompt(inputs)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "max_tokens": 4096,
            "stream": True
        }

        response = requests.post(self.model_url, headers=headers, json=payload, stream=True)
        print("Response: ", response)
        for line in response.iter_lines():
            print("Line: ", line)
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
            Union[Dict[str, Any], str, None]: The processed response content, either JSON or text, or None on error.

        Raises:
            ValueError: If the request to Claude API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = " ".join([block["text"] for block in data["content"]])
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

        Args:
            line (bytes): A line from the streaming response in bytes.

        Returns:
            Optional[str]: The extracted content chunk as a string, or None if the line is not valid or content is empty.
        """
        try:
            data = json.loads(line.decode("utf-8"))
            return " ".join([block["text"] for block in data.get("content", [])])
        except json.JSONDecodeError:
            return None

class GeminiRunnable(BaseRunnable):
    """
    Runnable class for interacting with Gemini language models.

    This class extends BaseRunnable and implements the specific logic for calling
    the Gemini API, including authentication, request formatting, response handling, and streaming.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a GeminiRunnable instance.
        Retrieves the Gemini API key from environment variables or Keyring.
        Inherits arguments from BaseRunnable.
        """
        super().__init__(*args, **kwargs)

        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY") # only in development

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
    def remove_unsupported_fields(self, schema):
        """
        Recursively removes unsupported fields like 'additionalProperties' from the schema.

        Args:
            schema (Union[Dict[str, Any], List[Any]]): The schema to clean.

        Returns:
            Union[Dict[str, Any], List[Any]]: The cleaned schema.
        """
        if isinstance(schema, dict):
            return {
                key: self.remove_unsupported_fields(value)
                for key, value in schema.items()
                if key != "additionalProperties"
            }
        elif isinstance(schema, list):
            return [self.remove_unsupported_fields(item) for item in schema]
        else:
            return schema
        
    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], str, None]:
        """
        Invokes the Gemini model to get a complete response.

        Constructs the payload for the Gemini API, including system instructions and
        structured JSON output configuration if required. Sends the request and processes the response.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Returns:
            Union[Dict[str, Any], str, None]: The response from the Gemini model, either JSON or text, or None on error.

        Raises:
            ValueError: If the Gemini API request fails.
        """
        # Build the prompt (e.g., append user input to messages list)
        self.build_prompt(inputs)

        # Extract system instruction (first message if role is "system")
        system_instruction = None
        if self.messages and self.messages[0]["role"].lower() == "system":
            system_content = self.messages[0]["content"]
            system_instruction = {"parts": [{"text": system_content}]}
            conversation_messages = self.messages[1:]
        else:
            conversation_messages = self.messages

        # Map roles: "user" -> "user", "assistant" -> "model"
        def map_role(role: str) -> str:
            """Maps generic role names to Gemini specific role names."""
            role_lower = role.lower()
            if role_lower == "assistant":
                return "model"
            elif role_lower == "user":
                return "user"
            else:
                raise ValueError(f"Unsupported role: {role}")

        # Construct contents list from messages
        contents = [
            {
                "role": map_role(msg["role"]),
                "parts": [{"text": msg["content"]}]
            }
            for msg in conversation_messages
        ]

        # Build the payload
        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Add generationConfig for structured JSON output if response_type is "json"
        if self.response_type == "json" and self.required_format:
            # Clean the schema to remove unsupported fields
            clean_schema = self.remove_unsupported_fields(self.required_format)
            payload["generationConfig"] = {
                "response_mime_type": "application/json",
                "response_schema": clean_schema
            }

        # Make the API request
        response = requests.post(
            f"{self.model_url}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Handle the response
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the Gemini API for non-streaming requests.

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the Gemini API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON or text, or None on error.

        Raises:
            ValueError: If the request to Gemini API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = "".join([part["text"] for part in data["candidates"][0]["content"]["parts"]])
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Model did not return valid JSON. Error: {e}")
            return content
        raise ValueError(f"Gemini API Error: {response.text}")

    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Union[Dict[str, Any], str, None], None, None]:
        """
        Intended for streaming responses from Gemini, but currently returns a non-streaming response.
        Streaming functionality for Gemini is not fully implemented in this version.

        Args:
            inputs (Dict[str, Any]): Input variables for the prompt.

        Yields:
            Generator[Union[Dict[str, Any], str, None], None, None]: A generator that currently yields a single non-streaming response.
        """
        # Build the prompt (e.g., append user input to messages list)
        self.build_prompt(inputs)

        # Extract system instruction (assuming it’s the first message if role is "system")
        system_instruction = None
        if self.messages and self.messages[0]["role"].lower() == "system":
            system_content = self.messages[0]["content"]
            system_instruction = {"parts": [{"text": system_content}]}
            conversation_messages = self.messages[1:]
        else:
            conversation_messages = self.messages

        # Map roles for conversation messages ("user" stays "user", "assistant" becomes "model")
        def map_role(role: str) -> str:
            """Maps generic role names to Gemini specific role names for streaming."""
            role_lower = role.lower()
            if role_lower == "assistant":
                return "model"
            elif role_lower == "user":
                return "user"
            else:
                raise ValueError(f"Unsupported role: {role}")

        # Construct the contents list from remaining messages
        contents = [
            {
                "role": map_role(msg["role"]),
                "parts": [{"text": msg["content"]}]
            }
            for msg in conversation_messages
        ]

        # Build the payload
        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Make the API request
        response = requests.post(
            f"{self.model_url}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Handle the response (assuming _handle_response exists)
        yield self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> Union[Dict[str, Any], str, None]:
        """
        Handles the HTTP response from the Gemini API (same as non-streaming).

        Parses the JSON response, extracts the content, and handles potential errors
        such as JSON decoding failures or non-200 status codes.

        Args:
            response (requests.Response): The HTTP response object from the Gemini API.

        Returns:
            Union[Dict[str, Any], str, None]: The processed response content, either JSON or text, or None on error.

        Raises:
            ValueError: If the request to Gemini API fails or JSON response is invalid.
        """
        if response.status_code == 200:
            data = response.json()
            content = "".join([part["text"] for part in data["candidates"][0]["content"]["parts"]])
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Model did not return valid JSON. Error: {e}")
            return content
        raise ValueError(f"Gemini API Error: {response.text}")

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