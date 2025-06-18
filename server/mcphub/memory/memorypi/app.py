import os
from memoripy import MemoryManager, JSONStorage
from memoripy.implemented_models import ChatCompletionsModel, OllamaEmbeddingModel

# --- Configuration ---
# This example now runs entirely on local models using Ollama.
# Make sure Ollama is installed and running on your machine.

# Model for chat completions. Make sure you have pulled it: `ollama pull llama3.1`
OLLAMA_CHAT_MODEL_NAME = "llama3.1"
# Ollama's OpenAI-compatible API endpoint
OLLAMA_API_ENDPOINT = "http://localhost:11434/v1"

# Model for embeddings. Make sure you have pulled it: `ollama pull mxbai-embed-large`
OLLAMA_EMBEDDING_MODEL_NAME = "mxbai-embed-large"

# Number of recent interactions to use as immediate context
CONTEXT_WINDOW_SIZE = 5
# Storage file for persistent memory
STORAGE_FILE = "interaction_history.json"


def main():
    """
    Main function to run the chat agent application using Ollama.
    """
    print("--- Memoripy Chat Agent (100% Ollama) ---")
    print("This agent runs entirely locally. Ensure Ollama is running.")
    print(f"Chat Model: {OLLAMA_CHAT_MODEL_NAME}, Embedding Model: {OLLAMA_EMBEDDING_MODEL_NAME}")
    print("Initializing models and memory...")

    # 1. Initialize Chat and Embedding Models
    # For chat, we use the ChatCompletionsModel pointed at Ollama's OpenAI-compatible endpoint.
    chat_model = ChatCompletionsModel(
        api_endpoint=OLLAMA_API_ENDPOINT,
        api_key="ollama",  # The API key is required by the client but not used by Ollama
        model_name=OLLAMA_CHAT_MODEL_NAME
    )
    # For embeddings, we use the dedicated OllamaEmbeddingModel.
    embedding_model = OllamaEmbeddingModel(OLLAMA_EMBEDDING_MODEL_NAME)

    # 2. Initialize Storage
    # JSONStorage provides persistence by saving memories to a file.
    storage_option = JSONStorage(STORAGE_FILE)

    # 3. Initialize Memory Manager
    # The MemoryManager orchestrates memory operations.
    memory_manager = MemoryManager(
        chat_model=chat_model,
        embedding_model=embedding_model,
        storage=storage_option
    )

    print("\nInitialization complete. You can start chatting.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 30)

    # 4. Start the Chat Loop
    while True:
        try:
            new_prompt = input("You: ")
            if new_prompt.lower() in ['quit', 'exit']:
                print("Agent: Goodbye! Your memories have been saved.")
                break
            
            print("Agent: Thinking...")

            # Load the last N interactions from history for immediate context
            short_term, _ = memory_manager.load_history()
            last_interactions = short_term[-CONTEXT_WINDOW_SIZE:] if len(short_term) >= CONTEXT_WINDOW_SIZE else short_term

            # Retrieve relevant past interactions, excluding those already in our immediate context
            relevant_interactions = memory_manager.retrieve_relevant_interactions(
                new_prompt, 
                exclude_last_n=len(last_interactions)
            )

            # Generate a response using both immediate and retrieved long-term context
            response = memory_manager.generate_response(
                new_prompt, 
                last_interactions, 
                relevant_interactions
            )

            print(f"Agent: {response}")
            
            # This part is crucial for making the agent remember the new interaction
            print("\nAgent: [Saving to memory...]")
            
            # Create a combined text for concept extraction and embedding
            combined_text = f"{new_prompt} {response}"
            
            # Extract concepts
            concepts = memory_manager.extract_concepts(combined_text)
            
            # Generate an embedding for the new interaction
            new_embedding = memory_manager.get_embedding(combined_text)
            
            # Add the full interaction to the memory store, which also saves it via the storage handler
            memory_manager.add_interaction(new_prompt, response, new_embedding, concepts)
            print("Agent: [Memory saved.]\n")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure Ollama is running and the models are downloaded.")
            print(f"You can run `ollama pull {OLLAMA_CHAT_MODEL_NAME}` and `ollama pull {OLLAMA_EMBEDDING_MODEL_NAME}`.")
            print("Restart the application after starting Ollama.")


if __name__ == "__main__":
    main()