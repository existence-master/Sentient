from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv
import json
from model.memory.runnables import *
from model.memory.functions import *
from model.memory.dual_memory import MemoryManager
from  model.memory.base import *  # Import the new MemoryQueue
from model.app.base import *

load_dotenv("model/.env")

class MemoryBackend:
    def __init__(self):
        """Initialize the memory backend with both long-term and short-term agents and MemoryQueue."""
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

        # Short-term memory (SQLite)
        print("Initializing MemoryManager (Short-term memory)...")
        self.memory_manager = MemoryManager(db_path="memory.db", model_name=os.environ["BASE_MODEL_REPO_ID"])
        print("MemoryManager initialized.")

        # Memory type classifiers
        print("Initializing memory type classifiers...")
        self.memory_type_runnable = self._initialize_memory_type_classifier()
        self.query_memory_type_runnable = self._initialize_query_memory_type_classifier()
        print("Memory type classifiers initialized.")

        # Initialize MemoryQueue
        print("Initializing MemoryQueue...")
        self.memory_queue = MemoryQueue()
        print("MemoryQueue initialized.")
        print("MemoryBackend initialization complete.")

    def _initialize_memory_type_classifier(self):
        """Initialize the classifier for short-term vs long-term memories."""
        print("Initializing memory type classifier (Short Term vs Long Term for facts)...")
        classifier = OllamaRunnable(
            model_url="http://localhost:11434/api/chat/",
            model_name="llama3.2:3b",
            system_prompt_template="""
            You are an AI designed to classify user-provided facts into 'Short Term' or 'Long Term' memory types.
            - 'Short Term' memories are transient (e.g., tasks, recent events) and typically expire within days or weeks.
            - 'Long Term' memories are persistent (e.g., preferences, personal traits) and stored indefinitely.
            Provide your classification as a string: "Short Term" or "Long Term". Do not add quotes. Simply respond with the memory type.
            """,
            user_prompt_template="Classify this fact: {fact}",
            input_variables=["fact"],
            response_type="chat"
        )
        print("Memory type classifier initialized.")
        return classifier

    def _initialize_query_memory_type_classifier(self):
        """Initialize the classifier for short-term vs long-term queries."""
        print("Initializing query memory type classifier (Short Term vs Long Term for queries)...")
        classifier = OllamaRunnable(
            model_url="http://localhost:11434/api/chat/",
            model_name="llama3.2:3b",
            system_prompt_template="""
            You are an AI designed to classify user queries into 'Short Term' or 'Long Term' memory types based on the kind of information being requested.
            - 'Short Term' queries are about recent events, tasks, or transient information (e.g., days or weeks). Examples: "What did I have for lunch yesterday?" or "Do I have meetings tomorrow?"
            - 'Long Term' queries are about persistent information, preferences, or knowledge (e.g., habits, traits). Examples: "What's my favorite color?" or "What do I usually order at restaurants?"
            Provide your classification as a string: "Short Term" or "Long Term". Do not add quotes. Simply respond with the memory type.
            """,
            user_prompt_template="Classify this query: {query}",
            input_variables=["query"],
            response_type="chat"
        )
        print("Query memory type classifier initialized.")
        return classifier

    def extract_memory_facts(self, query: str) -> list:
        """Extract multiple factual statements from a memory-related query."""
        print(f"Extracting memory facts for query: '{query}'")
        try:
            with open("userProfileDb.json", "r", encoding="utf-8") as f:
                user_db = json.load(f)
            username = user_db["userData"]["personalInfo"]["name"]
            response = self.fact_extraction_runnable.invoke({"paragraph": query, "username": username})
            print(f"Raw fact extraction response: {response}")
            facts = response
            if not isinstance(facts, list):
                raise ValueError("Extracted facts are not in list format")
            print(f"Extracted facts: {facts}")
            return facts
        except Exception as e:
            print(f"Error extracting memory facts: {e}")
            return []

    def classify_memory(self, fact: str) -> str:
        """Classify a fact as short-term or long-term."""
        print(f"Classifying memory type for fact: '{fact}'")
        try:
            response = self.memory_type_runnable.invoke({"fact": fact})
            classification = response.strip()
            print(f"Memory classification response: '{response}', Classified as: '{classification}'")
            return classification
        except Exception as e:
            print(f"Error classifying memory: {e}")
            print("Defaulting to Long Term memory.")
            return "Long Term"

    def classify_query_memory_type(self, query: str) -> str:
        """Classify a query as short-term or long-term."""
        print(f"Classifying query memory type for query: '{query}'")
        try:
            response = self.query_memory_type_runnable.invoke({"query": query})
            print(response)
            classification = response.strip()
            print(classification)
            print(f"Query memory type classification response: '{response}', Classified as: '{classification}'")
            return classification
        except Exception as e:
            print(f"Error classifying query memory type: {e}")
            print("Defaulting to Long Term query type.")
            return "Long Term"

    async def store_memory(self, user_id: str, query: str):
        """Extract and store multiple facts in the appropriate memory system."""
        print(f"Storing memory for user ID: '{user_id}', query: '{query}'")
        facts = self.extract_memory_facts(query)

        if not facts:
            print("No facts extracted from the query. Memory storage aborted.")
            return

        for fact in facts:
            memory_type = self.classify_memory(fact)
            print(f"Extracted fact: '{fact}' | Classified as: {memory_type}")

            if memory_type == "Short Term":
                print("Storing fact in Short Term memory...")
                expiry_info = self.memory_manager.expiry_date_decision(fact)
                retention_days = expiry_info.get("retention_days", 7)
                memory_info = self.memory_manager.extract_and_invoke_memory(fact)
                category = memory_info.get("memories", [{}])[0].get("category", "tasks")
                self.memory_manager.store_memory(user_id, fact, {"retention_days": retention_days}, category)
                print("Fact stored in Short Term memory.")
            else:
                print("Storing fact in Long Term memory (Neo4j)...")
                crud_graph_operations(
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
            memory_type = self.classify_memory(fact)
            print(f"Extracted fact: '{fact}' | Classified as: {memory_type}")

            if memory_type == "Short Term":
                print("Updating Short Term memory...")
                self.memory_manager.update_memory(user_id, fact)
                print("Short Term memory updated.")
            else:
                print("Updating Long Term memory (Neo4j)...")
                crud_graph_operations(
                    fact, self.graph_driver, self.embed_model, self.query_class_runnable,
                    self.info_extraction_runnable, self.graph_analysis_runnable,
                    self.graph_decision_runnable, self.text_desc_runnable
                )
                print("Long Term memory (Neo4j) updated.")
        print("Memory update process completed.")

    async def retrieve_memory(self, user_id: str, query: str) -> str:
        """Retrieve relevant memories synchronously from the appropriate store."""
        print(f"Retrieving memory for user ID: '{user_id}', query: '{query}'")
        memory_type = self.classify_query_memory_type(query)
        print(f"Query classified as: {memory_type} memory query.")
        if memory_type == "Short Term":
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
            context = query_user_profile(
                query, self.graph_driver, self.embed_model,
                self.text_conv_runnable, self.query_class_runnable
            )
            print(f"Retrieved long-term context: {context}")
            return context
    
    async def add_operation(self, user_id: str, query: str):
        await self.memory_queue.add_operation(user_id, query)

    def cleanup(self):
        """Clean up expired short-term memories."""
        print("Cleaning up expired short-term memories...")
        self.memory_manager.cleanup_expired_memories()
        print("Short-term memory cleanup completed.")