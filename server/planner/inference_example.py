# server/planner/plan.py
from typing import Optional, Dict, Any, Union, List
from server.runnables.base_runnable import call_model

if __name__ == "__main__":
    # Example Usage:

    # 1. Call using default config (Ollama, qwen3:4b)
    print("\n--- Calling with default config (Ollama) ---")
    response_default = call_model("What is the capital of France?")
    print(f"Response (default): {response_default}")

    # 2. Override model for Ollama
    print("\n--- Calling Ollama with overridden model (llama3) ---")
    response_ollama_override = call_model(
        "Tell me a short story about a brave knight.",
        provider_override="ollama",
        model_override="llama3" # Ensure llama3 is available in your Ollama setup
    )
    print(f"Response (Ollama override): {response_ollama_override}")

    # 3. Call using OpenRouter (default model from config)
    print("\n--- Calling with OpenRouter (default model) ---")
    # Ensure OPENROUTER_API_KEY and OPENROUTER_SITE_URL/OPENROUTER_SITE_NAME are set in your environment
    response_openrouter = call_model(
        "What is the largest ocean on Earth?",
        provider_override="openrouter"
    )
    print(f"Response (OpenRouter): {response_openrouter}")

    # 4. Call OpenRouter with overridden model
    print("\n--- Calling OpenRouter with overridden model (mistralai/mistral-7b-instruct) ---")
    response_openrouter_override = call_model(
        "Write a haiku about a sunset.",
        provider_override="openrouter",
        model_override="mistralai/mistral-7b-instruct" # Example OpenRouter model
    )
    print(f"Response (OpenRouter override): {response_openrouter_override}")

    # 5. Example of JSON response
    print("\n--- Calling with JSON response type ---")
    json_response = call_model(
        "Give me a JSON object with 'name' and 'age' for a person named Alice who is 30.",
        response_type="json",
        required_format={"name": "string", "age": "number"}
    )
    print(f"Response (JSON): {json_response}")
    print(f"Type of JSON response: {type(json_response)}")

    # 6. Example with chat history (stateful=True is not strictly needed for this example,
    #    as history is passed per-request, but demonstrates the parameter)
    print("\n--- Calling with chat history ---")
    history = [
        {"isUser": True, "message": "Hello, how are you?"},
        {"isUser": False, "message": "I am doing well, thank you! How can I help you today?"}
    ]
    response_with_history = call_model(
        "What is your purpose?",
        chat_history=history
    )
    print(f"Response (with history): {response_with_history}")

    # 7. Example of streaming response (Ollama)
    print("\n--- Calling Ollama with streaming response ---")
    print("Streaming response (Ollama): ", end="")
    stream_gen_ollama = call_model(
        "Explain the concept of recursion in programming.",
        stream=True,
        provider_override="ollama"
    )
    if stream_gen_ollama:
        for chunk in stream_gen_ollama:
            if chunk:
                print(chunk, end="")
        print() # Newline after stream

    # 8. Example of streaming response (OpenRouter)
    print("\n--- Calling OpenRouter with streaming response ---")
    print("Streaming response (OpenRouter): ", end="")
    stream_gen_openrouter = call_model(
        "Describe the water cycle.",
        stream=True,
        provider_override="openrouter"
    )
    if stream_gen_openrouter:
        for chunk in stream_gen_openrouter:
            if chunk:
                print(chunk, end="")
            print() # Newline after stream