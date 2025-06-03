import json
import os
import requests
from typing import Dict, Any, Union, Generator, Optional, List

from .base_runnable import BaseRunnable
# Absolute import for DEFAULT_OPENROUTER_MODEL
from server.config.models_config import DEFAULT_OPENROUTER_MODEL

from dotenv import load_dotenv
load_dotenv(dotenv_path='server/.env')

class OpenRouterRunnable(BaseRunnable):
    """
    A runnable component for interacting with OpenRouter models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the OpenRouterRunnable with an optional model name.

        Args:
            model_name (Optional[str]): The name of the OpenRouter model to use.
                                         If None, uses DEFAULT_OPENROUTER_MODEL.
        """
        actual_model_name = model_name if model_name is not None else DEFAULT_OPENROUTER_MODEL
        super().__init__(model_name=actual_model_name)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
        self.openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.session = requests.Session()

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        if not self.openrouter_site_url:
            print("Warning: OPENROUTER_SITE_URL environment variable not set. OpenRouter may rate limit or block requests without it.")
        if not self.openrouter_site_name:
            print("Warning: OPENROUTER_SITE_NAME environment variable not set. OpenRouter may rate limit or block requests without it.")

    def _generate_openrouter_chat_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates the payload for the OpenRouter chat API.
        """
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.0
        }

    def _stream_openrouter_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """
        Streams the response from the OpenRouter API.
        """
        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded_chunk = chunk.decode('utf-8')
                    if decoded_chunk.startswith("data:"):
                        json_chunk = json.loads(decoded_chunk[len("data:"):])
                        yield json_chunk
                except json.JSONDecodeError:
                    continue

    def run(self,
            user_input: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Runs the OpenRouter model with the given input and history.
        """
        messages = conversation_history if conversation_history is not None else []
        messages.append({"role": "user", "content": user_input})

        payload = self._generate_openrouter_chat_payload(messages)
        url = f"{self.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_site_name,
        }

        try:
            response = self.session.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            if stream:
                return self._stream_openrouter_response(response)
            else:
                full_response_content = ""
                for chunk in self._stream_openrouter_response(response):
                    if "content" in chunk.get("choices", [{}])[0].get("delta", {}):
                        full_response_content += chunk["choices"][0]["delta"]["content"]
                return {"role": "assistant", "content": full_response_content}

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with OpenRouter: {e}")
            return {"role": "assistant", "content": f"Error: Could not connect to OpenRouter. {e}"}

    def get_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimates token count for OpenRouter messages.
        Note: OpenRouter doesn't expose a direct tokenization endpoint.
        This is a rough estimate based on character count.
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # A very rough estimate: 1 token ~= 4 characters
        return total_chars // 4

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculates the cost for OpenRouter based on prompt and completion tokens.
        Note: This is a placeholder. Actual costs depend on the specific model and OpenRouter's pricing.
        """
        # Example pricing (replace with actual OpenRouter pricing for the model)
        # For demonstration, let's assume a hypothetical cost per token
        cost_per_input_token = 0.000001  # Example: $0.001 per 1k input tokens
        cost_per_output_token = 0.000002 # Example: $0.002 per 1k output tokens

        return (prompt_tokens * cost_per_input_token) + (completion_tokens * cost_per_output_token)