import requests
import os
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional, Type, Any

# Load environment variables at the start
load_dotenv()

# --- Base Runnable Class ---
class BaseRunnable:
    def __init__(
        self,
        model_url: Optional[str], # URL might be different for different providers
        model_name: str,
        system_prompt_template: str,
        user_prompt_template: str,
        input_variables: List[str],
        required_format: Optional[Dict],
        response_type: str = "chat",
        stream: bool = False,
        stateful: bool = False,
        max_tokens: Optional[int] = None, # Added max_tokens
        temperature: float = 0.7 # Added temperature
    ):
        self.model_url = model_url # Kept for Ollama, not strictly needed for OpenRouter (hardcoded URL)
        self.model_name = model_name
        self.system_prompt_template = system_prompt_template # Use template for formatting later
        self.user_prompt_template = user_prompt_template   # Use template for formatting later
        self.input_variables = input_variables
        self.required_format = required_format
        self.response_type = response_type
        self.stream = stream
        self.stateful = stateful
        self.history: List[Dict[str, str]] = [] # Type hint for history
        self.max_tokens = max_tokens
        self.temperature = temperature

    def add_to_history(self, chat_history: List[Dict[str, str]]) -> None:
        self.history.extend(chat_history)

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]: # Return type hint
        raise NotImplementedError("Subclasses must implement invoke method")

# --- Ollama Runnable (Kept for comparison/alternative) ---
class OllamaRunnable(BaseRunnable):
    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        # Format prompts with input variables
        system_prompt = self.system_prompt_template.format(**inputs)
        user_prompt = self.user_prompt_template.format(**inputs)

        # Ollama typically uses a single prompt string or messages list depending on endpoint/version
        # This example uses a single prompt string for simplicity with generate endpoint
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Prepare payload for Ollama API
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": self.stream,
            "options": { # Ollama options object for temperature, max_tokens etc.
                "temperature": self.temperature,
                "num_predict": self.max_tokens, # Ollama uses num_predict for max_tokens
            }
        }

        # Add format if requested (Ollama requires "json" string, not required_format dict)
        if self.response_type == "json":
             payload["format"] = "json"

        # Make API call to Ollama
        try:
            # Ensure model_url is set for Ollama
            ollama_url = self.model_url or "http://localhost:11434"
            response = requests.post(f"{ollama_url}/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()

            if self.response_type == "json":
                 # Ollama's /api/generate with format="json" still returns JSON wrapper
                response_content = result.get("response", "{}")
                try:
                    # Need to parse the string inside the "response" key
                    return json.loads(response_content)
                except json.JSONDecodeError:
                     print(f"Warning: Ollama returned non-JSON response despite format='json'. Content: {response_content}")
                     return {"error": "Model did not return valid JSON."}
            else:
                # For chat response, just return the 'response' key content
                return {"response": result.get("response", "")}

        except requests.RequestException as e:
            return {"error": f"Failed to invoke Ollama model {self.model_name}: {str(e)}"}
        except Exception as e:
             return {"error": f"An unexpected error occurred with Ollama: {str(e)}"}


# --- OpenRouter Runnable ---
class OpenRouterRunnable(BaseRunnable):
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

        # Ensure response_type is 'chat' for OpenRouter chat completions endpoint
        if self.response_type not in ["chat", "json"]:
             print(f"Warning: OpenRouter chat completion supports 'chat' or 'json' response_type, not '{self.response_type}'. Using 'chat'.")
             self.response_type = "chat" # Default to chat if something else was passed

        # OpenRouter handles JSON formatting if the prompt guides the model to produce it
        # We don't set a specific 'format' parameter in the OpenRouter payload like Ollama
        # The 'required_format' is used as instruction in the system/user prompt

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "OpenRouter API key is not configured."}

        # Format prompts with input variables
        system_prompt = self.system_prompt_template.format(**inputs)
        user_prompt = self.user_prompt_template.format(**inputs)

        # OpenRouter uses the standard OpenAI chat messages format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            # You could add self.history here if stateful is True
        ]

        # Prepare payload for OpenRouter API
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.max_tokens is not None:
             payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Make API call to OpenRouter
        try:
            print(f"Calling OpenRouter model: {self.model_name}...")
            response = requests.post(self.OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            # OpenRouter (like OpenAI) returns the response in result['choices'][0]['message']['content']
            if result and result.get('choices'):
                response_content = result['choices'][0]['message']['content']

                if self.response_type == "json":
                    try:
                        # Attempt to parse the content as JSON
                        return json.loads(response_content)
                    except json.JSONDecodeError:
                        print(f"Warning: Model response was not valid JSON despite request for json type. Content: {response_content[:200]}...") # Print truncated content
                        return {"error": "Model response was not valid JSON.", "raw_response": response_content}
                else:
                    # Return as plain text response
                    return {"response": response_content}
            else:
                # Handle cases where 'choices' is missing or empty
                print(f"Warning: OpenRouter response missing 'choices'. Full response: {result}")
                return {"error": "Unexpected API response structure."}

        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API request failed: {e}")
            # Try to get more details from the response if available in the exception
            if e.response is not None:
                 try:
                      error_details = e.response.json()
                      print(f"OpenRouter Error Details: {error_details}")
                      return {"error": f"OpenRouter API request failed: {e}", "details": error_details}
                 except json.JSONDecodeError:
                       return {"error": f"OpenRouter API request failed: {e}", "raw_response": e.response.text}
            return {"error": f"OpenRouter API request failed: {e}"}
        except Exception as e:
             print(f"An unexpected error occurred with OpenRouter: {str(e)}")
             return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Configuration and Runnable Selection ---

# Environment variable to select provider (e.g., in .env: LLM_PROVIDER=openrouter)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # Default to ollama if not set

# Environment variables for model names
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen:4b") # Default Ollama Qwen
# Confirmed OpenRouter name for Qwen 1.5 4B Chat
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "Qwen/Qwen1.5-4B-Chat")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Updated Prompt Templates and Format ---
# System prompt guiding the extraction and JSON format
SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant designed to process user input and extract specific information.
Your task is to read the provided text content and the user's query.
Based on the content, identify:
1.  **Action Items:** Tasks or activities suggested or implied that require follow-up or use of tools (like creating a presentation, drafting an email, scheduling). These should be clear, actionable phrases.
2.  **Memory Items:** Key facts, details, names, dates, or concepts from the text that the user might want to remember or reference later. These should be concise summaries.

Structure your response strictly as a JSON object with two keys: "action_items" and "memory_items".
Each key should map to a JSON array (list) of strings.
If no items are found for a category, the corresponding array should be empty ([]).
Do NOT include any other text or formatting outside the JSON object.
Example required format:
```json
{
    "action_items": ["Create a presentation on Agriculture Visit", "Draft follow-up email"],
    "memory_items": ["Visited Maharashtra farms", "Meeting with Mr. Patil on Oct 26th"]
}
```
"""