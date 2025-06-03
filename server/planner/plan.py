# server/planner/plan.py
from typing import Optional, Dict, Any, Union, List
from server.runnables.base_runnable import call_model

if __name__ == "__main__":
    # Example Usage:

    # 1. Call using default config (Ollama, qwen3:4b)
    print("\n--- Calling with default config (Ollama) ---")
    response_default = call_model("What is the capital of France?")
    print(f"Response: {response_default}")