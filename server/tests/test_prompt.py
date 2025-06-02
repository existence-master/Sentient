import os
import json
import requests
from typing import Dict, Any, List, Union, Optional, Generator, Tuple
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library
from sys import platform  # To get system platform information
from fastapi import WebSocket

load_dotenv("server/.env")

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator

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
        self.messages: List[Dict[str, str]] = []
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
            self.messages = [{"role": "system", "content": self.system_prompt_template}]
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
        self.messages = [{"role": "system", "content": self.system_prompt_template}]
        self.messages.extend(chat_history)

    def clear_history(self) -> None:
        """
        Clears the message history, resetting the conversation to its initial state.

        This method is useful for starting a new conversation with the language model
        without any context from previous interactions. It resets the message history
        to contain only the initial system prompt.
        """
        self.messages = []

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
            print(f"Response type: {self.response_type}")
            print(f"Platform: {platform}")
            if (
                platform == "win32"
            ):  # Conditional format setting based on platform (Windows specific handling)
                payload["format"] = (
                    self.required_format
                )  # Set format directly for Windows
            else:
                payload["format"] = self.required_format
                  # Serialize format to JSON string for non-Windows

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

unified_classification_format = {
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["chat", "memory", "agent"]
    },
    "use_personal_context": {
      "type": "boolean"
    },
    "internet": {
      "type": "boolean",
    },
    "transformed_input": {
      "type": "string"
    }
  },
  "required": ["category", "use_personal_context", "internet", "transformed_input"],
  "additionalProperties": False
}


# ======================
# PROMPT DEFINITIONS
# ======================

# Base user prompt template used by multiple configurations - UNCHANGED

base_user_prompt = """Classify the following user input based on the criteria provided.

Key Instructions:
- Classify into only one category.
- Memory is *only* for facts *about the user*.
- Agent is *only* for explicit requests involving the specified Google tools.
- Crucially: Set internet to true ONLY when external, dynamic, or very specific factual lookup is the clear primary need. Do NOT use it for greetings, general chat, personal info queries, or tool requests. If in doubt about whether base knowledge suffices, lean towards internet: false.
- Transform Agent input only for follow-ups like "Retry," using implicit history context. If context is unclear, classify as chat.

Input: {query}

Output:
"""

# System Prompts
# Original System Prompt - UNCHANGED

original_system = """You are an input classification system for a personalized AI assistant. Your task is to analyze the user's input, classify it based on the criteria provided, and, if needed, transform it by incorporating relevant context from the implicit chat history.

Follow these instructions carefully:

1. Read the user input thoroughly.

2. Determine the classification category based on these rules:
   - memory: For statements that include any information about the user that is not a request for action. This includes personal preferences, experiences, or any other information that can be stored in memory.
   - agent: For explicit requests to perform an action using one of these tools: Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, or Google Slides. If these tools are not mentioned, you can't classify it as agent
   - chat: If you can't classify the input as memory or agent, classify it as chat. This includes general conversation, greetings, or any other non-specific requests.

3. Set the flag `use_personal_context`:
   - true if the query involves details specific to the user's personal life or stored memories.
   - false if the query is general or does not depend on personal context.

4. Set the flag `internet`:
   - true ONLY when the primary request is for highly dynamic, time-sensitive, or external factual information (e.g., current news, weather, live data).
   - false in all other cases (greetings, basic questions, personal info queries, or tool requests).

5. Determine the `transformed_input`:
   - For agent requests that are follow-ups (e.g., “Retry that”), incorporate the intended original action using context from the chat history. If the context isn't clear, classify as chat.
   - For chat and memory categories, use the user’s input exactly as provided.

6. Output Requirement:
   - Produce only a valid JSON object with exactly these keys:
     - `"category"`: must be one of `"chat"`, `"memory"`, or `"agent"`.
     - `"use_personal_context"`: either `true` or `false`.
     - `"internet"`: either `true` or `false`.
     - `"transformed_input"`: a string.
   - Do not output any additional text or explanation."""

# --- Improved Prompts Based on Original Logic ---

# Improved Few-Shot Prompt
# Adds more detailed instructions before examples and includes more diverse examples covering key rules.

few_shot_system = """You are an input classification system for a personalized AI assistant. Your task is to analyze the user's input, classify it based on the criteria provided, and, if needed, transform it by incorporating relevant context from the implicit chat history.

Follow these instructions carefully:

1.  **Classification Categories:**
    *   `memory`: Input contains information *about the user* (preferences, facts, experiences) to be remembered. Not an action request.
    *   `agent`: Explicit request using specific Google tools: **Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, Google Slides**. Tool *must* be mentioned.
    *   `chat`: Default category for greetings, general conversation, questions, or requests not fitting `memory` or `agent`.

2.  **Flags:**
    *   `use_personal_context`: `true` if the query relates to the user's specific life, preferences, or stored memories. `false` otherwise.
    *   `internet`: `true` ONLY for queries needing *external, dynamic, real-time, or very specific factual data* (e.g., current news, live weather, stock prices). `false` for general knowledge, greetings, personal info, tool requests. If unsure, lean towards `false`.

3.  **Input Transformation:**
    *   `transformed_input`: Usually the original user input. Exception: For `agent` follow-ups (like "Retry that"), use chat history to fill in the original request context. If context is ambiguous, classify as `chat`.

4.  **Output Format:** Strict JSON with keys: `"category"`, `"use_personal_context"`, `"internet"`, `"transformed_input"`. No extra text.

Here are some examples:

Example 1:
Input: "I like classical music."
Output: {"category": "memory", "use_personal_context": true, "internet": false, "transformed_input": "I like classical music."}

Example 2:
Input: "Can you draft an email in Google Mail?"
Output: {"category": "agent", "use_personal_context": false, "internet": false, "transformed_input": "Can you draft an email in Google Mail?"}

Example 3:
Input: "What's the weather like right now in London?"
Output: {"category": "chat", "use_personal_context": false, "internet": true, "transformed_input": "What's the weather like right now in London?"}

Example 4:
Input: "Tell me a fun fact."
Output: {"category": "chat", "use_personal_context": false, "internet": false, "transformed_input": "Tell me a fun fact."}

Example 5:
Input: "My favorite color is blue."
Output: {"category": "memory", "use_personal_context": true, "internet": false, "transformed_input": "My favorite color is blue."}

Example 6:
Input: "Can you check Google Calendar for my schedule today?"
Output: {"category": "agent", "use_personal_context": true, "internet": false, "transformed_input": "Can you check Google Calendar for my schedule today?"}

Example 7 (Assuming previous action was 'Create a doc called Meeting Notes'):
Input: "Retry that."
Output: {"category": "agent", "use_personal_context": false, "internet": false, "transformed_input": "Create a doc called Meeting Notes"}

Example 8:
Input: "Who is the president of the USA?"
Output: {"category": "chat", "use_personal_context": false, "internet": false, "transformed_input": "Who is the president of the USA?"} # Assumes base knowledge, not dynamic

Example 9:
Input: "What are the latest headlines?"
Output: {"category": "chat", "use_personal_context": false, "internet": true, "transformed_input": "What are the latest headlines?"}
"""

# Improved Chain-of-Thought Prompt
# Expands the steps with the specific criteria from the original prompt.

chain_of_thought_system = """You are an input classification system for a personalized AI assistant. Analyze the input step-by-step and provide the final JSON output.

**Reasoning Steps:**

1.  **Analyze Input Purpose:** Is the user providing information about themselves to be remembered?
    *   If YES (preferences, experiences, personal facts): Preliminary Category = `memory`. Go to Step 4.
    *   If NO (it's a question or request): Go to Step 2.

2.  **Check for Agent Request:** Does the input explicitly mention a request involving one of these specific Google tools: **Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, Google Slides**?
    *   If YES: Preliminary Category = `agent`. Go to Step 4.
    *   If NO: Preliminary Category = `chat`. Go to Step 3.

3.  **Refine Chat Classification:** Confirm it's not `memory` or `agent`. This includes general chat, greetings, questions not needing dynamic external data. Final Category = `chat`. Go to Step 4.

4.  **Determine `use_personal_context`:** Does the query involve details specific to the user's personal life, stored memories, or preferences mentioned now or previously?
    *   If YES: `use_personal_context` = `true`.
    *   If NO: `use_personal_context` = `false`.

5.  **Determine `internet` Flag:** Is the *primary need* of the query for highly dynamic, time-sensitive, external factual information (e.g., current news, live weather, stock prices, very specific/niche facts)?
    *   If YES: `internet` = `true`.
    *   If NO (includes greetings, general knowledge questions, personal info queries, tool requests): `internet` = `false`.

6.  **Determine `transformed_input`:**
    *   If Category is `agent` AND the input is a follow-up like "Retry that": Use chat history context to determine the original intended action for `transformed_input`. If context is unclear, change Category to `chat` and use original input.
    *   Otherwise (Category is `chat` or `memory`, or it's an initial `agent` request): `transformed_input` is the original user input.

7.  **Final Output:** Construct the JSON object with `"category"`, `"use_personal_context"`, `"internet"`, and `"transformed_input"`. Ensure it's *only* the JSON object.
"""

# Improved XML Tagged Prompt
# Adds more detail within the tags, mirroring the original prompt's instructions more closely.

xml_tagged_system = """<system>
<role>Input Classification System for Personalized AI Assistant</role>
<task>Analyze user input, classify it, set flags, potentially transform input, and output strict JSON.</task>

<instructions>
  <classification_rules>
    <rule category="memory">
      <condition>Input contains information *about the user* (preferences, experiences, personal facts) not requiring an action.</condition>
      <output>category: "memory"</output>
    </rule>
    <rule category="agent">
      <condition>Input is an *explicit request* involving one of these Google tools: **Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, Google Slides**. The tool name must be present.</condition>
      <output>category: "agent"</output>
    </rule>
    <rule category="chat">
      <condition>Input does not meet criteria for `memory` or `agent`. Includes general conversation, greetings, questions, non-specific requests.</condition>
      <output>category: "chat"</output>
    </rule>
  </classification_rules>

  <flag_rules>
    <flag name="use_personal_context">
      <condition_true>Query involves details specific to the user's personal life, stored memories, or preferences.</condition_true>
      <condition_false>Query is general or doesn't depend on personal context.</condition_false>
    </flag>
    <flag name="internet">
      <condition_true>Primary need is for highly dynamic, time-sensitive, or external factual information (e.g., current news, weather, live data). Explicit external lookup required.</condition_true>
      <condition_false>All other cases: greetings, general knowledge, personal info queries, tool requests. Lean towards false if unsure.</condition_false>
    </flag>
  </flag_rules>

  <transformation_rules>
    <rule type="agent_followup">
      <condition>Category is `agent` and input is a follow-up (e.g., "Retry that").</condition>
      <action>Use chat history context to reconstruct the original request for `transformed_input`. If context unclear, classify as `chat` with original input.</action>
    </rule>
    <rule type="default">
      <condition>Category is `chat` or `memory`, or initial `agent` request.</condition>
      <action>`transformed_input` is the original user input.</action>
    </rule>
  </transformation_rules>

  <output_format>
    <requirement>Produce *only* a valid JSON object.</requirement>
    <keys>
      <key name="category" type="string" values="chat|memory|agent" />
      <key name="use_personal_context" type="boolean" />
      <key name="internet" type="boolean" />
      <key name="transformed_input" type="string" />
    </keys>
    <instruction>No additional text or explanations outside the JSON.</instruction>
  </output_format>
</instructions>
</system>"""

# Improved Markdown Structured Prompt
# Uses Markdown sections to structure the detailed instructions from the original prompt.

markdown_system = """## Role: Input Classifier for Personalized AI Assistant

**Task**: Analyze the user's input, classify it according to the rules below, set the required flags, determine the transformed input, and output *only* a valid JSON object.

### 1. Classification Categories

*   **`memory`**: Use this if the input is a statement providing information *about the user* (e.g., preferences, experiences, personal facts) that should be remembered. It's not a request for action.
    *   *Example*: "I am allergic to peanuts."
*   **`agent`**: Use this *only* if the input is an *explicit request* to perform an action using one of these specific Google tools:
    *   **Google Drive**
    *   **Google Mail**
    *   **Google Docs**
    *   **Google Sheets**
    *   **Google Calendar**
    *   **Google Slides**
    *   If the tool isn't mentioned, do NOT classify as `agent`.
    *   *Example*: "Create a new spreadsheet in Google Sheets called 'Budget'."
*   **`chat`**: Use this if the input doesn't fit `memory` or `agent`. This includes:
    *   General conversation, greetings ("Hello")
    *   Questions (unless they strictly require `internet`)
    *   Requests not involving the specified Google tools.
    *   *Example*: "Tell me a joke.", "What is photosynthesis?"

### 2. Flag Settings

*   **`use_personal_context`** (boolean):
    *   `true`: If the query relates to the user's specific life details, preferences, or information previously stored in memory.
    *   `false`: If the query is general, factual, or doesn't rely on knowing the user personally.
*   **`internet`** (boolean):
    *   `true`: ONLY when the primary goal is to get *highly dynamic, time-sensitive, or external factual information*. Think current news, live weather/stock data, very specific external lookups.
    *   `false`: For **all** other cases. This includes greetings, general knowledge questions (if likely in base model), personal information queries ("What's my favorite color?"), and `agent` tool requests. **If in doubt, set to `false`**.

### 3. Input Transformation

*   **`transformed_input`** (string):
    *   Normally, this is the user's original input verbatim.
    *   **Exception**: If the category is `agent` and the input is a follow-up command like "Try that again" or "Retry", use the chat history context to determine the *original action* the user wants repeated. This becomes the `transformed_input`.
    *   If the context for an agent follow-up is unclear, classify the input as `chat` and use the original text ("Retry that").

### 4. Output Requirement

*   Produce **only** a valid JSON object.
*   The JSON object must contain exactly these keys: `"category"`, `"use_personal_context"`, `"internet"`, `"transformed_input"`.
*   Do not include any explanations, comments, or text before or after the JSON object.
"""

# Improved Schema Example Prompt
# Provides the full instructions first, then uses the schema as a formatting guide.

schema_example_system = """You are an input classification system for a personalized AI assistant. Your task is to analyze the user's input, classify it based on the criteria provided, potentially transform it, and output a JSON object matching the specified schema.

**Instructions:**

1.  **Classification:**
    *   `memory`: Info *about the user* (prefs, facts, experiences). Not a request.
    *   `agent`: Explicit request using **Google Drive, Mail, Docs, Sheets, Calendar, or Slides**. Tool must be named.
    *   `chat`: Fallback for general chat, greetings, questions, non-Google tool requests.

2.  **Flags:**
    *   `use_personal_context` (boolean): `true` if query needs user-specific info/memory, `false` otherwise.
    *   `internet` (boolean): `true` *only* for external, dynamic, real-time data (news, weather, live data). `false` for general knowledge, greetings, personal queries, tool requests. Default to `false` if unsure.

3.  **Transformation:**
    *   `transformed_input` (string): Usually the original input. Exception: For `agent` follow-ups ("Retry"), reconstruct original action from history. If unclear, classify as `chat`.

4.  **Output Format:** Adhere strictly to the JSON schema below. Output *only* the JSON.

**Output JSON Schema Example:**
```json
{
  "category": "agent",
  "use_personal_context": false,
  "internet": false,
  "transformed_input": "Create Google Doc with meeting notes"
}


Note: The values in the example are illustrative. Determine the correct values based on the input and the rules above. Ensure your output is solely the JSON object.
"""

# Improved Error Corrective Prompt
# Presents the positive instructions first, then lists common errors to avoid as reinforcement.

error_corrective_system = """You are an input classification system for a personalized AI assistant. Follow these primary instructions precisely, and pay attention to the common errors listed afterwards.

Primary Instructions:

Role & Task: Analyze user input, classify it, set flags, potentially transform input, and output strict JSON.

Classification Categories:

memory: Statements containing information about the user (preferences, experiences, personal facts) for storage. Not an action request.

agent: Explicit requests involving Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, or Google Slides. The specific tool must be mentioned.

chat: Default category for greetings, general conversation, questions, or requests not fitting memory or agent.

Flag Settings:

use_personal_context: true if the query relates to the user's specific life, preferences, or stored memories. false otherwise.

internet: true ONLY for queries needing external, dynamic, real-time, or very specific factual data (e.g., current news, live weather, stock prices). false for general knowledge, greetings, personal info queries, tool requests. If unsure, lean towards false.

Input Transformation:

transformed_input: Usually the original user input. Exception: For agent follow-ups (like "Retry that"), use chat history to fill in the original request context. If context is ambiguous, classify as chat.

Output Format: Strict JSON with keys: "category", "use_personal_context", "internet", "transformed_input". No extra text.

Common Errors to Avoid:

Incorrect Agent Classification: Do NOT classify as agent if one of the specific Google Workspace tools (Drive, Mail, Docs, Sheets, Calendar, Slides) is not explicitly mentioned in the request.

Overuse of internet=true: Do NOT set internet to true for greetings, general knowledge questions (if likely answerable from base model), personal info retrieval ("What's my favorite color?"), or requests for the agent to perform actions (unless the action itself requires fetching live data, which is rare for the defined agent tools). It's specifically for external, dynamic lookups.

Missing use_personal_context=true: If the query clearly references the user's personal details, preferences, or past interactions stored in memory, use_personal_context MUST be true.

Incorrect transformed_input: Only transform the input for agent follow-up commands where history provides clear context. Otherwise, use the exact original input.

Invalid Output: Ensure the output is only the JSON object, correctly formatted, with no additional explanatory text, greetings, or markdown formatting.
"""

# Improved Role Contextual Prompt
# Enhances the policy section with the full detailed instructions from the original prompt.

role_contextual_system = """You are Clara, Senior Data Classifier at AI Assistant Inc. Your primary responsibility is to accurately classify incoming user inputs according to strict company policy to ensure correct routing and processing by the AI assistant. Adhere precisely to the following classification guidelines and output specifications.

Company Classification Policy (Strict Adherence Required):

Classification Categories:

memory: Input contains information about the user (preferences, experiences, personal facts) intended for storage in the user's profile. This is informational, not an action request.

agent: Input is an explicit request for automation involving one of the following approved Google Workspace tools: Google Drive, Google Mail, Google Docs, Google Sheets, Google Calendar, Google Slides. The tool name MUST be explicitly stated in the request. No other tools trigger this category.

chat: This is the default category for all inputs not classified as memory or agent. This includes general conversation, greetings, questions (factual or otherwise, unless needing live data), and requests not involving the specified Google tools.

Context & Data Flags:

use_personal_context (Boolean): Set to true if processing the request requires accessing or utilizing information specific to the user's personal life, stated preferences, or previously stored memories. Set to false for general requests or information.

internet (Boolean): Set to true ONLY when the user's primary request necessitates accessing external, dynamic, real-time, or highly specific factual information not typically held in a static knowledge base (e.g., current news headlines, live weather updates, current stock prices). Set to false for all other cases, including general knowledge questions, memory inputs, agent requests, and personal context queries. Err on the side of false if uncertain.

Input Transformation Protocol:

transformed_input (String): In most cases, this field should contain the user's input exactly as received.

Exception: For inputs classified as agent that represent follow-up commands (e.g., "Retry that", "Do it again"), consult the recent interaction history to identify the specific original action being referred to. Populate transformed_input with that original action's description. If the context is ambiguous or unavailable, re-classify the input as chat and use the literal follow-up command (e.g., "Retry that") as the transformed_input.

Output Requirements:

Your output MUST be a single, valid JSON object.

The JSON object must contain exactly these four keys: "category", "use_personal_context", "internet", "transformed_input".

Absolutely NO commentary, explanations, or any text outside the JSON structure is permitted in the output.
"""

# User Prompts - Minor adjustments for clarity where needed

xml_user_prompt = "<input>{query}</input>\n<response_format>JSON</response_format>" # Keep as is
markdown_user_prompt = "Input: {query}\nRequired Output Format:\njson\n{{\n \"category\": \"...\",\n \"use_personal_context\": ...,\n \"internet\": ...,\n \"transformed_input\": \"...\"\n}}\n" # Added all keys for clarity
cot_user_prompt = base_user_prompt + "\nShow your reasoning steps before the final JSON answer." # Keep as is, requests reasoning

# ======================
# PROMPT CONFIGURATIONS
# ======================

prompt_configs = [
# Original Prompts - UNCHANGED
{
"name": "original",
"system_prompt": original_system,
"user_prompt": base_user_prompt
},

# --- Improved Prompt Variants ---
{
    "name": "few_shot_improved", # Renamed for clarity
    "system_prompt": few_shot_system,
    "user_prompt": base_user_prompt # Uses base user prompt now
},
{
    "name": "chain_of_thought_improved", # Renamed
    "system_prompt": chain_of_thought_system,
    "user_prompt": base_user_prompt # Uses base user prompt
},
{
    "name": "xml_tagged_improved", # Renamed
    "system_prompt": xml_tagged_system,
    "user_prompt": xml_user_prompt # Uses specific XML user prompt
},
{
    "name": "markdown_structured_improved", # Renamed
    "system_prompt": markdown_system,
    "user_prompt": markdown_user_prompt # Uses specific Markdown user prompt
},
{
    "name": "cot_explicit_reasoning", # Renamed for clarity (combines CoT system + specific user request)
    "system_prompt": chain_of_thought_system, # Uses the improved CoT system prompt
    "user_prompt": cot_user_prompt # Uses the user prompt requesting reasoning
},
{
    "name": "schema_example_improved", # Renamed
    "system_prompt": schema_example_system,
    "user_prompt": base_user_prompt # Uses base user prompt
},
{
    "name": "error_corrective_improved", # Renamed
    "system_prompt": error_corrective_system,
    "user_prompt": base_user_prompt # Uses base user prompt
},
{
    "name": "role_contextual_improved", # Renamed
    "system_prompt": role_contextual_system,
    "user_prompt": "Input to Classify: {query}\n\nRequired JSON Output:" # Slightly more direct user prompt
}
]
# ======================
# TEST INFRASTRUCTURE (Relevant Part)
# ======================
model_name = "mistral:7b"

def get_unified_classification_runnable(chat_history, system_prompt, user_prompt):
    # Using a potentially smaller/faster model for local testing - adjust if needed
     # Changed model example
    runnable_class = OllamaRunnable
    # Ensure your Ollama service is running and accessible at this URL
    model_url = "http://localhost:11434/api/chat" # Default local Ollama URL

    unified_classification_runnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=system_prompt,
        user_prompt_template=user_prompt,
        input_variables=["query"],
        required_format=unified_classification_format, # Assumed defined elsewhere
        response_type="json",
        stateful=True, # Stateful allows history, but we manage it explicitly per call
    )
    # Clear any potential residual history from the stateful instance
    # Note: OllamaRunnable might not have explicit clear_history.
    # Re-creating the instance per call is the safest way if state persists unexpectedly.
    # The current approach passes history at creation, implying a fresh state.

    # Add the provided history for this specific invocation
    if chat_history:
        # Assuming add_to_history correctly sets the context for the *next* call
        # If stateful=True causes issues, creating a new instance *inside* the loop is safer.
        unified_classification_runnable.add_to_history(chat_history)
    return unified_classification_runnable

# --- Define Example Chat Histories for Testing ---
# History 1: Clear agent action, suitable for testing "Retry that last action."
history_agent_retry_clear = [
    {"role": "user", "content": "Can you create a document in Google Docs named 'Project Plan'?"},
    # Simulating a natural language confirmation from the assistant after *executing* the request.
    # The context for "Retry that" is the user's previous request.
    {"role": "assistant", "content": "Okay, I've created the Google Doc named 'Project Plan' for you."}
]

# History 2: Memory input followed by a related query context
history_memory_context = [
    {"role": "user", "content": "My favorite vacation spot is Hawaii."},
    # Simulating the assistant confirming it stored the memory.
    {"role": "assistant", "content": "Got it! I'll remember that Hawaii is your favorite vacation spot."}
]

# History 3: Internet query context
history_chat_internet = [
    {"role": "user", "content": "What's the weather in Paris?"},
    # Simulating the assistant providing the information. The content itself isn't crucial, just that it was a chat/internet turn.
    {"role": "assistant", "content": "The current weather in Paris is partly cloudy with a temperature of 15°C."}
]

# History 4: Ambiguous context where "Retry" might be classified as chat
history_agent_retry_ambiguous = [
    {"role": "user", "content": "Do that thing we talked about earlier."},
    # Simulating the assistant indicating confusion, making the context ambiguous for a subsequent "Retry".
    {"role": "assistant", "content": "I'm sorry, I'm not sure which specific thing you're referring to. Could you please clarify?"}
]

# History 5: Multiple turns with different types
history_mixed = [
    {"role": "user", "content": "Hello there!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "Remind me that I prefer Earl Grey tea."},
    {"role": "assistant", "content": "Okay, I've noted that you prefer Earl Grey tea."},
    {"role": "user", "content": "Now, create a meeting invite in Google Calendar for tomorrow at 3 PM called 'Sync Up'."},
    # Provides clear context for a potential "Retry that last action" related to the calendar event.
    {"role": "assistant", "content": "Sure thing. I've created the 'Sync Up' event in your Google Calendar for tomorrow at 3 PM."}
]

# Empty history for baseline tests
history_empty = []

# --- Map specific inputs to relevant histories ---
# (Used within the execution loop)
def select_history_for_input(input_text):
    input_lower = input_text.lower()
    if "retry that" in input_lower or "try again" in input_lower:
        # Use clear history first, could switch to ambiguous to test that case
        return history_agent_retry_clear
    elif "check my google calendar" in input_lower or "my anniversary" in input_lower:
        # Inputs likely requiring personal context might benefit from memory history
        return history_memory_context
    elif "what's happening" in input_lower or "latest tech news" in input_lower or "stock market" in input_lower:
        # Follow-up to potential internet queries
        return history_chat_internet
    elif "allergic to nuts" in input_lower or "i prefer tea" in input_lower or "i use android" in input_lower:
         # Test memory function without specific prior history needing recall
         return history_empty
    # Default to empty history for general inputs
    else:
        return history_empty

# Test inputs (copied from previous state for context, no changes needed here)
test_inputs = [
    # Original tests
    "What is the capital of France?", # chat, internet: false (base knowledge)
    "I like to play chess.", # memory, context: true
    "Can you schedule a meeting in Google Calendar?", # agent, context: false/true depends on details
    "Tell me about the latest tech news.", # chat, internet: true
    "Retry that last action.", # agent (if history clear), context: false, transformed: (previous action)

    # New test cases
    "Add this to my preferences: I'm allergic to nuts", # memory, context: true
    "Make a spreadsheet in Google Sheets for Q2 sales", # agent, context: false
    "What's happening in the stock market today?", # chat, internet: true
    "Remember that I use Android not iPhone", # memory, context: true
    # "Can you check my calendar?", # Needs explicit "Google Calendar" per rules
    "Check my Google Calendar for today", # agent, context: true
    "How does photosynthesis work?", # chat, internet: false (base knowledge)
    "What time is sunset in Tokyo tomorrow?", # chat, internet: true (specific external data)
    "Please email the report to my team using Google Mail", # agent, context: false/true depending on 'my team'
    "I prefer tea over coffee", # memory, context: true
    # "Create a presentation in Slides about Q3 results", # Needs explicit "Google Slides" per rules
    "Create a presentation in Google Slides about Q3 results", # agent, context: false
    "Just saying hello", # chat, internet: false
    "What was the score of the big game last night?", # chat, internet: true (dynamic, recent)
    "Remind me that my anniversary is June 5th", # memory, context: true
]

# ======================
# EXECUTION (Modified Part)
# ======================

output_filename = "prompt_experiment_results_improved_with_history.txt" # New filename
print(f"Starting prompt evaluation with targeted history. Results will be saved to {output_filename}")

# Clear the file for a fresh run
with open(output_filename, "w", encoding="utf-8") as f:
     f.write(f"Prompt Experiment Results with Targeted History\nModel: {model_name}\n===\n") # Use model_name directly

# Main execution loop
with open(output_filename, "a", encoding="utf-8") as f:
    for i, input_text in enumerate(test_inputs):
        print(f"\nProcessing input {i+1}/{len(test_inputs)}: '{input_text}'")
        f.write(f"\n---\nInput: {input_text}\n---\n")

        # Select the appropriate history for this input
        current_history = select_history_for_input(input_text)
        history_name = "Empty" # Default
        if current_history == history_agent_retry_clear: history_name = "AgentRetryClear"
        elif current_history == history_memory_context: history_name = "MemoryContext"
        elif current_history == history_chat_internet: history_name = "ChatInternet"
        elif current_history == history_agent_retry_ambiguous: history_name = "AgentRetryAmbiguous"
        elif current_history == history_mixed: history_name = "Mixed"

        f.write(f"Selected History Context: {history_name}\n---\n")
        print(f"  Using history context: {history_name}")

        for config in prompt_configs:
            print(f"  Testing prompt: {config['name']}...")
            try:
                # Create a new runnable instance FOR EACH test case (input + config + history)
                # This ensures history isolation between different tests.
                runnable = get_unified_classification_runnable(
                    current_history,
                    config["system_prompt"],
                    config["user_prompt"]
                )

                output = runnable.invoke(inputs={"query": input_text})
                output_json = json.dumps(output, indent=2)
                f.write(f"--- {config['name']} ---\n{output_json}\n\n")
                # print(f"    Output: {output_json}") # Optionally print output to console too
            except Exception as e:
                error_msg = f"Error invoking runnable {config['name']} for input '{input_text}' with history '{history_name}': {e}"
                print(f"    {error_msg}")
                f.write(f"--- {config['name']} ---\nERROR: {error_msg}\n\n")

print(f"\nEvaluation complete. Results saved to {output_filename}")