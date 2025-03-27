# Append to model/agents/runnables.py
import os

def get_gmail_context_runnable():
    """Configure and return a Runnable for the Gmail Context Engine."""
    from model.agents.prompts import gmail_context_engine_system_prompt_template, gmail_context_engine_user_prompt_template
    # Example assumes you have a BaseRunnable and specific implementations like OllamaRunnable
    # Adjust imports and classes based on your actual LLM setup
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    model_name, provider = get_selected_model()  # Assume this function exists to get model config

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback, adjust as needed

    return runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=gmail_context_engine_system_prompt_template,
        user_prompt_template=gmail_context_engine_user_prompt_template,
        input_variables=["new_information", "related_memories", "ongoing_tasks", "chat_history"],
        required_format=None,  # JSON structure is enforced by prompt
        response_type="json",
        stateful=False,
    )

# Placeholder for get_selected_model; implement based on your system
def get_selected_model():
    return "default_model", None  # Replace with actual logic