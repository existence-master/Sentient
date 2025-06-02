from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import asyncio
import traceback
from dotenv import load_dotenv
from server.memory.runnables import *
from server.memory.functions import *
from server.memory.dual_memory import MemoryManager
from server.memory.base import *  # Import the new MemoryQueue
from server.db.mongo_manager import MongoManager # Import MongoManager
from server.app.base import *

load_dotenv("server/.env")

class MemoryBackend:
    def __init__(self, mongo_manager: MongoManager):
        """Initialize the memory backend with both long-term and short-term agents and MemoryQueue."""
        self.mongo_manager = mongo_manager # Use the passed MongoManager instance
        print("MongoManager instance received by MemoryBackend.")
        print("Initializing MemoryBackend...")
        print("Initializing MemoryBackend...")
        
        # Long-term memory (Neo4j)
        print("Initializing HuggingFaceEmbedding...")
        self.embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])
        print("HuggingFaceEmbedding initialized.")
        print("Initializing Neo4j GraphDriver...")
        self.graph_driver = GraphDatabase.driver(
            uri=os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
        )
        print("Neo4j GraphDriver initialized.")
        print("Initializing graph runnables...")
        self.graph_decision_runnable = get_graph_decision_runnable()
        self.info_extraction_runnable = get_information_extraction_runnable()
        self.graph_analysis_runnable = get_graph_analysis_runnable()
        self.query_class_runnable = get_query_classification_runnable()
        self.fact_extraction_runnable = get_fact_extraction_runnable()
        self.text_desc_runnable = get_text_description_runnable()
        self.text_conv_runnable = get_text_conversion_runnable()
        print("Graph runnables initialized.")
        # Memory type classifiers (removed)

        # Short-term memory (MongoDB)
        self.memory_manager = MemoryManager(mongo_manager=self.mongo_manager, model_name=os.environ["BASE_MODEL_REPO_ID"])

        # Initialize MemoryQueue
        print("Initializing MemoryQueue...")
        self.memory_queue = MemoryQueue() # No longer needs memory_file parameter
        print("MemoryQueue initialized.")
        print("MemoryBackend initialization complete.")

    def classify_memory(self, fact: str) -> str:
        print(f"Classifying memory type for fact: '{fact}'")
        return "Long Term"

    def classify_query_memory_type(self, query: str) -> str:
        print(f"Classifying query memory type for query: '{query}'")
        return "Long Term"

    def extract_memory_facts(self, query: str) -> list:
        """Extract multiple factual statements from a memory-related query."""
        print(f"Extracting memory facts for query: '{query}'")
        try:
            username = "User" # Fallback username

            response = self.fact_extraction_runnable.invoke({"paragraph": query, "username": username})
            print(f"Raw fact extraction response: {response}")
            facts = response
            if isinstance(facts, dict) and "facts" in facts and isinstance(facts["facts"], list):
                facts = facts["facts"]
            elif not isinstance(facts, list):
                 print(f"Warning: Extracted facts are not in list format, attempting to use as is: {facts}")
                 facts = [str(facts)] if facts is not None else []

            print(f"Extracted facts: {facts}")
            return facts
        except Exception as e:
            print(f"Error extracting memory facts: {e}")
            return []

    async def store_memory(self, user_id: str, query: str):
        """Extract and store multiple facts in the appropriate memory system."""
        print(f"Storing memory for user ID: '{user_id}', query: '{query}'")
        facts = self.extract_memory_facts(query)

        if not facts:
            print("No facts extracted from the query. Memory storage aborted.")
            return

        for fact in facts:
            memory_type = "Long Term" # Default to Long Term after removing classification
            print(f"Extracted fact: '{fact}' | Classified as: {memory_type}")

            if memory_type == "Short Term": # This block will now effectively be skipped
                print("Storing fact in Short Term memory...")
                expiry_info = self.memory_manager.expiry_date_decision(fact)
                retention_days = expiry_info if isinstance(expiry_info, int) else expiry_info.get("retention_days", 7)

                memory_info = self.memory_manager.extract_and_invoke_memory(fact)
                category = memory_info.get("memories", [{}])[0].get("category", "tasks")
                self.memory_manager.store_memory(user_id, fact, retention_days, category)
                print("Fact stored in Short Term memory.")
            else:
                print("Storing fact in Long Term memory (Neo4j)...")
                if not all([self.graph_driver, self.embed_model, self.query_class_runnable,
                            self.info_extraction_runnable, self.graph_analysis_runnable,
                            self.graph_decision_runnable, self.text_desc_runnable]):
                    print("Error: One or more Neo4j runnables/drivers are not initialized. Skipping long-term memory storage.")
                    continue

                await asyncio.to_thread(
                    crud_graph_operations,
                    user_id,
                    fact, self.graph_driver, self.embed_model, self.query_class_runnable,
                    self.info_extraction_runnable, self.graph_analysis_runnable,
                    self.graph_decision_runnable, self.text_desc_runnable
                )
                print("Fact stored in Long Term memory (Neo4j).")
        print("Memory storage process completed.")
    
    async def update_memory(self, user_id: str, query: str):
        """Extract and update multiple facts in the memory system."""
        print(f"Updating memory for user ID: '{user_id}', query: '{query}'")
        facts = self.extract_memory_facts(query)

        if not facts:
            print("No facts extracted from the query. Memory update aborted.")
            return

        for fact in facts:
            memory_type = "Long Term" # Default to Long Term after removing classification
            print(f"Extracted fact: '{fact}' | Classified as: {memory_type}")

            if memory_type == "Short Term": # This block will now effectively be skipped
                print("Updating Short Term memory...")
                self.memory_manager.update_memory(user_id, fact)
                print("Short Term memory updated.")
            else:
                print("Updating Long Term memory (Neo4j)...")
                if not all([self.graph_driver, self.embed_model, self.query_class_runnable,
                            self.info_extraction_runnable, self.graph_analysis_runnable,
                            self.graph_decision_runnable, self.text_desc_runnable]):
                    print("Error: One or more Neo4j runnables/drivers are not initialized. Skipping long-term memory update.")
                    continue
                await asyncio.to_thread(
                    crud_graph_operations,
                    user_id,
                    fact, self.graph_driver, self.embed_model, self.query_class_runnable,
                    self.info_extraction_runnable, self.graph_analysis_runnable,
                    self.graph_decision_runnable, self.text_desc_runnable
                )
                print("Long Term memory (Neo4j) updated.")
        print("Memory update process completed.")

    async def retrieve_memory(self, user_id: str, query: str, type: str = None) -> str:
        """Retrieve relevant memories synchronously from the appropriate store."""
        print(f"Retrieving memory for user ID: '{user_id}', query: '{query}'")
        memory_type = "Long Term" # Default to Long Term after removing classification
            
        print(f"Query classified as: {memory_type} memory query.")
        if memory_type == "Short Term": # This block will now effectively be skipped
            print("Retrieving from Short Term memory...")
            context = self.memory_manager.process_user_query(user_id, query)
            if context and context != "I'm having trouble processing your question. Please try again.":
                print(f"Retrieved short-term context: {context}")
                return context
            else:
                print("No relevant short-term memories found.")
                return None
        else:
            print("Retrieving from Long Term memory (Neo4j)...")
            if not all([self.graph_driver, self.embed_model, self.text_conv_runnable, self.query_class_runnable]):
                print("Error: One or more Neo4j runnables/drivers are not initialized for long-term memory retrieval.")
                return None
            context = await asyncio.to_thread(
                query_user_profile,
                user_id,
                query, self.graph_driver, self.embed_model,
                self.text_conv_runnable, self.query_class_runnable
            )
            print(f"Retrieved long-term context: {context}")
            return context
    
    async def add_operation(self, user_id: str, query: str):
        """Add a memory operation to the queue."""
        memory_op_data = {"type": "store", "query_text": query} # Default to 'store' operation
        await self.memory_queue.add_operation(user_id, memory_op_data)

    async def process_memory_operations(self):
        """Continuously process memory operations from the queue."""
        print("[MEMORY_BACKEND_PROCESSOR] Starting memory operation processing loop.")
        while True:
            operation = await self.memory_queue.get_next_operation()
            if operation:
                user_id = operation.get("user_id")
                memory_data = operation.get("memory_data")
                operation_id = operation.get("operation_id")
                
                if not isinstance(memory_data, dict):
                    print(f"[MEMORY_BACKEND_PROCESSOR_ERROR] Invalid memory_data format for op {operation_id}: {memory_data}. Expected Dict.")
                    await self.memory_queue.complete_operation(user_id, operation_id, error="Invalid memory_data format.")
                    continue
                
                op_type = memory_data.get("type")
                query_text = memory_data.get("query_text")

                if not query_text:
                    print(f"[MEMORY_BACKEND_PROCESSOR_ERROR] Missing query_text in memory_data for op {operation_id}.")
                    await self.memory_queue.complete_operation(user_id, operation_id, error="Missing query_text.")
                    continue
                
                print(f"[MEMORY_BACKEND_PROCESSOR] Processing operation {operation_id} (type: {op_type}) for user {user_id}: {query_text[:50]}...")
                try:
                    if op_type == "store":
                        await self.store_memory(user_id, query_text)
                        await self.memory_queue.complete_operation(user_id, operation_id, result="Stored successfully")
                    elif op_type == "update":
                        await self.update_memory(user_id, query_text)
                        await self.memory_queue.complete_operation(user_id, operation_id, result="Updated successfully")
                    else:
                        print(f"[MEMORY_BACKEND_PROCESSOR_WARN] Unknown operation type '{op_type}' for op {operation_id}.")
                        await self.memory_queue.complete_operation(user_id, operation_id, error=f"Unknown operation type: {op_type}")
                    
                    print(f"[MEMORY_BACKEND_PROCESSOR] Completed operation {operation_id}.")
                except Exception as e:
                    print(f"[MEMORY_BACKEND_PROCESSOR_ERROR] Error processing memory operation {operation_id}: {e}")
                    traceback.print_exc()
                    await self.memory_queue.complete_operation(user_id, operation_id, error=str(e))
            else:
                await asyncio.sleep(0.5) # Wait if queue is empty

    def cleanup(self):
        """Clean up expired short-term memories."""
        print("Cleaning up expired short-term memories...")
        self.memory_manager.cleanup_expired_memories()
        print("Short-term memory cleanup completed.")