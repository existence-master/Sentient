import os
import json
import requests
from typing import Dict, Any, List, Union, Optional, Generator
from abc import ABC, abstractmethod
from helpers import *
from dotenv import load_dotenv
import keyring 

load_dotenv("../.env")

def get_selected_model():
    """Fetch the selected model name from userProfile.json."""
    db_path = "../../userProfileDb.json"
    if not db_path or not os.path.exists(db_path):
        raise ValueError("USER_PROFILE_DB_PATH not set or file not found")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    return db["userData"].get("selectedModel", "llama3.2:3b")  # Default to llama3.2:3b

class BaseRunnable(ABC):
    @abstractmethod
    def __init__(self, model_url: str, model_name: str, system_prompt_template: str,
                 user_prompt_template: str, input_variables: List[str], response_type: str,
                 required_format: Optional[Union[dict, list]] = None, stream: bool = False,
                 stateful: bool = False):
        self.model_url = model_url
        self.model_name = model_name
        self.user_prompt_template = user_prompt_template
        self.input_variables = input_variables
        self.response_type = response_type
        self.required_format = required_format
        self.stream = stream
        self.stateful = stateful

        if self.response_type == "json" and self.required_format:
            schema_str = json.dumps(self.required_format, indent=2)
            system_prompt_template += f"\n\nPlease respond in JSON format following this schema:\n{schema_str}"
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt_template}]

    def build_prompt(self, inputs: Dict[str, Any]):
        """Build the prompt by substituting input variables into templates."""
        user_prompt = self.user_prompt_template.format(**inputs)

        if self.stateful:
            self.messages.append({"role": "user", "content": user_prompt})
        else:
            self.messages = [{"role": "system", "content": self.messages[0]["content"]}]
            self.messages.append({"role": "user", "content": user_prompt})

    def add_to_history(self, chat_history: List[Dict[str, str]]):
        """Add chat history to maintain context."""
        self.messages.extend(chat_history)

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], List[Any], str, None]:
        pass

    @abstractmethod
    def stream_response(self, inputs: Dict[str, Any]) -> Generator[Optional[str], None, None]:
        pass


class OllamaRunnable(BaseRunnable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, inputs: Dict[str, Any]):
        self.build_prompt(inputs)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }

        if self.response_type == "json" and self.required_format:
            payload["format"] = json.dumps(self.required_format)

        response = requests.post(self.model_url, json=payload)
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]):
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

    def _handle_response(self, response: requests.Response):
        if response.status_code == 200:
            try:
                data = response.json().get("message", {}).get("content", "")
                if self.response_type == "json":
                    return json.loads(data)
                return data
            except json.JSONDecodeError:
                raise ValueError(f"Failed to decode JSON response: {data}")
        raise ValueError(f"Request failed with status {response.status_code}: {response.text}")

    def _handle_stream_line(self, line: str):
        try:
            data = json.loads(line)
            if data.get("done", True):
                return None
            return data["message"]["content"]
        except json.JSONDecodeError:
            return None


class OpenAIRunnable(BaseRunnable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY") # Old way from environment variable
        # self.api_key = keyring.get_password("openai", "api_key") # Get from keyring, service name "openai", username "api_key"

    def invoke(self, inputs: Dict[str, Any]):
        self.build_prompt(inputs)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.7,
        }
        if self.response_type == "json":
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(self.model_url, headers=headers, json=payload)
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]):
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

    def _handle_response(self, response: requests.Response):
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("Model did not return valid JSON.")
            return content
        raise ValueError(f"OpenAI API Error: {response.text}")

    def _handle_stream_line(self, line: bytes):
        if line.startswith(b"data: "):
            chunk = json.loads(line[6:])
            return chunk["choices"][0]["delta"].get("content", "")
        return None


class ClaudeRunnable(BaseRunnable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = os.getenv("CLAUDE_API_KEY") # Old way from environment variable
        # self.api_key = keyring.get_password("claude", "api_key") # Get from keyring, service name "claude", username "api_key"

    def invoke(self, inputs: Dict[str, Any]):
        self.build_prompt(inputs)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "max_tokens": 4096
        }

        response = requests.post(self.model_url, headers=headers, json=payload)
        return self._handle_response(response)

    def stream_response(self, inputs: Dict[str, Any]):
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

    def _handle_response(self, response: requests.Response):
        if response.status_code == 200:
            data = response.json()
            content = " ".join([block["text"] for block in data["content"]])
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("Model did not return valid JSON.")
            return content
        raise ValueError(f"Claude API Error: {response.text}")

    def _handle_stream_line(self, line: bytes):
        try:
            data = json.loads(line.decode("utf-8"))
            return " ".join([block["text"] for block in data.get("content", [])])
        except json.JSONDecodeError:
            return None

class GeminiRunnable(BaseRunnable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Retrieve the encrypted API key from Keyring
        encrypted_key = keyring.get_password("electron-openid-oauth", "gemini")
        print("Encrypted Key: ", encrypted_key)
        
        # Check if the encrypted key exists
        if encrypted_key:
            try:
                # Decrypt the key using the locally defined aes_decrypt function
                decrypted_data = aes_decrypt(encrypted_key)
                self.api_key = decrypted_data
            except Exception as e:
                # Handle decryption errors
                print(f"Failed to decrypt API key: {str(e)}")
                self.api_key = None
        else:
            # Handle the case where no encrypted key is found in Keyring
            print("No encrypted API key found in Keyring.")
            self.api_key = None
    
    def invoke(self, inputs):
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
        def map_role(role):
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
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Add generationConfig for structured JSON output if response_type is "json"
        if self.response_type == "json" and self.required_format:
            payload["generationConfig"] = {
                "response_mime_type": "application/json",
                "response_schema": self.required_format
            }

        # Make the API request
        response = requests.post(
            f"{self.model_url}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Handle the response
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response):
        if response.status_code == 200:
            data = response.json()
            content = "".join([part["text"] for part in data["candidates"][0]["content"]["parts"]])
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("Model did not return valid JSON.")
            return content
        raise ValueError(f"Gemini API Error: {response.text}")

    def stream_response(self, inputs):
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
        def map_role(role):
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
        payload = {"contents": contents}
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

    def _handle_response(self, response: requests.Response):
        if response.status_code == 200:
            data = response.json()
            content = "".join([part["text"] for part in data["candidates"][0]["content"]["parts"]])
            if self.response_type == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("Model did not return valid JSON.")
            return content
        raise ValueError(f"Gemini API Error: {response.text}")

def get_chat_runnable(chat_history: List[Dict[str, str]]) -> BaseRunnable:
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    
    print("Model Mapping: ", model_mapping)

    provider = None

    model_name = get_selected_model()
    
    print("Model Name: ", model_name)

    provider = model_name.lower()
    
    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable
        
    print(model_url)
    print(provider)
    print(runnable_class)

    system_prompt = "You are a helpful AI assistant."
    user_prompt = "{query}"

    runnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=system_prompt,
        user_prompt_template=user_prompt,
        input_variables=["query"],
        response_type="text",
        stream=True,
        stateful=True
    )

    runnable.add_to_history(chat_history)
    return runnable