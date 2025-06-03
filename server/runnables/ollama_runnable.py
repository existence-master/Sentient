import json
import os
import requests
import platform
from typing import Dict, Any, Union, Generator, Optional, List

from .base_runnable import BaseRunnable
# Absolute import for DEFAULT_OLLAMA_MODEL
from server.config.models_config import DEFAULT_OLLAMA_MODEL

class OllamaRunnable(BaseRunnable):
    """
    A runnable component for interacting with Ollama models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the OllamaRunnable with an optional model name.

        Args:
            model_name (Optional[str]): The name of the Ollama model to use.
                                         If None, uses DEFAULT_OLLAMA_MODEL.
        """
        actual_model_name = model_name if model_name is not None else DEFAULT_OLLAMA_MODEL
        super().__init__(model_name=actual_model_name)
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.session = requests.Session()

    def _generate_ollama_chat_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates the payload for the Ollama chat API.
        """
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "think": True,  # Enable thinking mode for better responses
            "options": {
                "temperature": 0.0
            }
        }

    def _stream_ollama_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """
        Streams the response from the Ollama API.
        """
        for chunk in response.iter_lines():
            if chunk:
                try:
                    json_chunk = json.loads(chunk.decode('utf-8'))
                    yield json_chunk
                except json.JSONDecodeError:
                    continue

    def run(self,
            user_input: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Runs the Ollama model with the given input and history.
        """
        messages = conversation_history if conversation_history is not None else []
        messages.append({"role": "user", "content": user_input})

        payload = self._generate_ollama_chat_payload(messages)
        url = f"{self.ollama_host}/api/chat"

        try:
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()

            if stream:
                return self._stream_ollama_response(response)
            else:
                full_response_content = ""
                thinking_content = ""
                for chunk in self._stream_ollama_response(response):
                    if "content" in chunk.get("message", {}):
                        full_response_content += chunk["message"]["content"]
                    if "thinking" in chunk.get("message", {}):
                        thinking_content += chunk["message"]["thinking"]
                return {"role": "assistant", "content": full_response_content, "thinking": thinking_content}

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return {"role": "assistant", "content": f"Error: Could not connect to Ollama. {e}"}

    def get_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimates token count for Ollama messages.
        Note: Ollama doesn't expose a direct tokenization endpoint.
        This is a rough estimate based on character count.
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # A very rough estimate: 1 token ~= 4 characters
        return total_chars // 4

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Ollama models are typically run locally and are free to use.
        """
        return 0.0