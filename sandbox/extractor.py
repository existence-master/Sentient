# file: extract_actions_memories.py

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import requests

load_dotenv()

# -------- Step 1: Load Raw Data from Text File -------- #
def load_raw_data() -> List[str]:
    import user_data
    return user_data.raw_data_list

# -------- Step 2: Define Prompt Templates -------- #
system_prompt_template = """
You are an expert assistant specialized in extracting structured information from various types of user communication data such as emails, messages, queries, and notifications.

Your goal is to analyze the provided raw user data carefully and extract two specific categories of information:

1. Action Items:
   - Clear, actionable tasks or requests the user needs to complete.
   - Often involve interactions with tools like calendars, emails, reminders, documents, or appointments.
   - Provide concise, unambiguous task descriptions.

2. Memory Items:
   - Personal notes, facts, or context about the user's life: health, family, career, events, preferences.
   - Informational and non-actionable, useful for future reference.

Important Instructions:
- Only extract information explicitly or strongly implied by the data.
- Do NOT add assumptions or unrelated details.
- Format the output strictly as a JSON object matching the required schema.
- If no relevant items exist in a category, return an empty list for that category.
- The input data may be formal or informal, questions, notifications, or any mix, so interpret accordingly.

Examples:

Example 1:
Input:
\"\"\"
Subject: Invoice #INV-2023-01 Received
To whom it may concern,

We have received your invoice #INV-2023-01 dated October 26, 2023. Payment processing is underway and is expected within 10 business days.

Regards,
Accounts Payable Department
\"\"\"

Output:
{
  "action_items": [
    "Track payment processing for invoice #INV-2023-01"
  ],
  "memory_items": [
    "Invoice #INV-2023-01 was received on October 26, 2023",
    "Payment expected within 10 business days"
  ]
}

Example 2:
Input:
\"\"\"
Hi Sarah,

Are you doing anything fun this weekend? Thought we might grab pizza if you're free.

Best,
Mike
\"\"\"

Output:
{
  "action_items": [
    "Ask Sarah if she is free to grab pizza this weekend"
  ],
  "memory_items": [
    "User wants to make weekend plans with Sarah"
  ]
}

Your structured output must contain exactly two fields:
- "action_items": an array of actionable task strings.
- "memory_items": an array of personal or contextual information strings.

Focus on precision, clarity, and relevance in your extraction.
"""

user_prompt_template = """
Given the following raw user data:
{raw_data}

Analyze it thoroughly and extract all relevant action items and memory items as per the instructions.

Return the output ONLY as a JSON object following this structure:
{{
  "action_items": [ ... ],
  "memory_items": [ ... ]
}}
"""


required_format = """
{
  "type": "object",
  "properties": {
    "action_items": {
      "type": "array",
      "items": {
        "type": "string",
        "description": "A task or action the user needs to complete"
      },
      "description": "List of tasks or actions the user needs to take"
    },
    "memory_items": {
      "type": "array",
      "items": {
        "type": "string",
        "description": "A personal note or memory"
      },
      "description": "List of personal notes or memories"
    }
"""

# -------- Step 3: Create a Runnable Class for OpenRouter -------- #
class OpenRouterRunnable:
    def __init__(self, model_url: str, model_name: str):
        self.api_url = model_url
        self.model = model_name
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, List[str]]:
        messages = [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": user_prompt_template.format(raw_data=inputs["raw_data"]) + "\n" + required_format},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()

        try:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print("Failed to parse model response:", e)
            print("Raw output:", response.text)
            return {"action_items": [], "memory_items": []}

# -------- Step 4: Main Processing Logic -------- #
def extract_action_and_memory_items() -> Dict[str, List[str]]:
    raw_data_list = load_raw_data()
    if not raw_data_list:
        return {"action_items": [], "memory_items": []}

    # Example: process the first item (index 0)
    raw_data_str = raw_data_list[10]

    runnable = OpenRouterRunnable(
        model_url="https://openrouter.ai/api/v1/chat/completions",
        model_name="qwen/qwen3-8b"
    )

    return runnable.invoke({"raw_data": raw_data_str, "required_format": required_format})


# -------- Optional Entry Point -------- #
if __name__ == "__main__":
    result = extract_action_and_memory_items()
    print(json.dumps(result, indent=2))
