from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from datetime import datetime, date, timedelta
import spacy
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import numpy as np
from numpy.linalg import norm
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel
from server.db.mongo_manager import MongoManager
from server.memory.base import MEMORY_COLLECTION_NAME, MONGO_DB_NAME, MONGO_URI

from .formats import *
from .helpers import *
from .prompts import *
from server.app.base import *

load_dotenv("server/.env")

nlp = spacy.load("en_core_web_sm")

class MemoryManager:
    def __init__(self, mongo_manager: MongoManager, model_name: str = os.environ["BASE_MODEL_REPO_ID"]):
        print("Initializing MemoryManager...")
        self.mongo_manager = mongo_manager
        self.memory_collection = self.mongo_manager.db[MEMORY_COLLECTION_NAME]
        self.model_name = model_name
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")
        self.categories = {
            "PERSONAL_WELLBEING": ["personal", "health", "lifestyle", "values", "preferences", "well-being", "self-care", "emotions", "habits", "routines", "diet", "exercise", "mental health", "physical health"],
            "PROFESSIONAL_ACADEMIC": ["career", "work", "education", "achievements", "challenges", "job", "study", "school", "college", "university", "degree", "course", "project", "promotion", "skill", "learning", "academic", "professional development"],
            "SOCIAL_RELATIONSHIPS": ["relationships", "socials", "social", "friends", "family", "partner", "colleagues", "networking", "community", "gatherings", "events", "interactions"],
            "FINANCIAL": ["financial", "money", "bank", "loan", "debt", "payment", "investment", "budget", "income", "expense", "savings"],
            "GOALS_TASKS": ["goals", "tasks", "objectives", "targets", "to-do", "deadline", "appointment", "schedule", "reminder", "aspirations", "planning"],
            "INTERESTS_HOBBIES": ["interests", "hobbies", "entertainment", "recreation", "leisure", "movies", "games", "music", "books", "art", "sports", "travel", "creative pursuits"],
            "LOGISTICS_PRACTICAL": ["transportation", "technology", "miscellaneous", "logistics", "practical", "devices", "software", "apps", "car", "travel", "home", "daily errands", "general information"],
            "SPIRITUAL": ["spiritual", "faith", "meditation", "religion", "beliefs", "mindfulness", "inner peace"]
        }
        print("Ensuring MongoDB indexes for memories...")
        self.initialize_mongodb_indexes()
        print("MemoryManager initialized.")

    def initialize_mongodb_indexes(self):
        """
        Ensures necessary indexes for the memory collection.
        This is called during MemoryManager initialization.
        """
        indexes = [
            IndexModel([("user_id", ASCENDING), ("category", ASCENDING)], name="user_category_idx"),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="user_created_at_desc_idx"),
            IndexModel([("expiry_at", ASCENDING)], name="expiry_at_idx", expireAfterSeconds=0) # TTL index
        ]
        try:
            # Use asyncio.run to run the async index creation in sync context
            # This is generally okay for startup tasks
            import asyncio
            asyncio.run(self.memory_collection.create_indexes(indexes))
            print(f"[DB_INIT] Indexes ensured for memory collection: {self.memory_collection.name}")
        except Exception as e:
            print(f"[DB_ERROR] Failed to create indexes for {self.memory_collection.name}: {e}")

    def compute_embedding(self, text: str) -> bytes:
        print(f"Computing embedding for text: '{text}...'")
        embedding = np.array(self.embedding_model.encode(text)).tobytes()
        print("Embedding computed.")
        return embedding

    def bytes_to_array(self, embedding_bytes: bytes) -> np.ndarray:
        return np.frombuffer(embedding_bytes, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (norm(a) * norm(b))

    def extract_keywords(self, text: str) -> List[str]:
        print(f"Extracting keywords from text: '{text}...'")
        doc = nlp(text.lower())
        keywords = [ent.text for ent in doc.ents]
        keywords.extend([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop and len(token.text) > 2])
        unique_keywords = list(set(keywords))
        print(f"Extracted keywords: {unique_keywords}")
        return unique_keywords

    def determine_category(self, keywords: List[str]) -> str:
        print(f"Determining category from keywords: {keywords}")
        category_scores = {category: 0 for category in self.categories}
        for keyword in keywords:
            for category, category_keywords in self.categories.items():
                if any(cat_keyword in keyword for cat_keyword in category_keywords):
                    category_scores[category] += 1
        max_score = max(category_scores.values())
        determined_category = "tasks" if max_score == 0 else max(category_scores.items(), key=lambda x: x[1])[0]
        print(f"Determined category: {determined_category}")
        return determined_category

    def expiry_date_decision(self, query: str) -> int:
        today = date.today()
        formatted_date = today.strftime("%d %B %Y %A")
        try:
            modified_system_expiry_template = system_memory_expiry_template.replace(
                "Your response must strictly adhere to the following rules:\n1. The minimum storage time is 1 day and the maximum is 90 days.",
                "Return a JSON object with a single key 'retention_days' and the value as the number of days (minimum 1, maximum 90)."
            )
            runnable = OllamaRunnable(
                model_url="http://localhost:11434/api/chat",
                model_name=self.model_name,
                system_prompt_template=modified_system_expiry_template,
                user_prompt_template=user_memory_expiry_template,
                input_variables=["query", "formatted_date"],
                response_type="json"
            )
            print(f"Invoking expiry date decision for query: '{query}...'")
            response = runnable.invoke({"query": query, "formatted_date": formatted_date})
            print(f"Expiry date decision response: {response}")
            return response.get("retention_days", 7) if isinstance(response, dict) else 7
        except Exception as e:
            print(f"Error in expiry_date_decision: {e}")
            return {"retention_days": 7}

    def extract_and_invoke_memory(self, current_query: str) -> Dict:
        date_today = date.today()
        try:
            runnable = OllamaRunnable(
                model_url="http://localhost:11434/api/chat",
                model_name=self.model_name,
                system_prompt_template=extract_memory_system_prompt_template,
                user_prompt_template=extract_memory_user_prompt_template,
                input_variables=["current_query", "date_today"],
                response_type="json",
                required_format=extract_memory_required_format
            )
            print(f"Invoking memory extraction for query: '{current_query}...'")
            response = runnable.invoke({"current_query": current_query, "date_today": date_today})
            print(f"Memory extraction response: {response}")
            return response if isinstance(response, dict) else {"memories": []}
        except Exception as e:
            print(f"Error in extract_and_invoke_memory: {e}")
            return {"memories": []}

    async def update_memory(self, user_id: str, current_query: str) -> Optional[Dict]:
        memories = self.extract_and_invoke_memory(current_query)
        for mem in memories.get('memories', []):
            category = mem['category'].lower()
            relevant_memories = await self.get_relevant_memories(user_id, mem['text'], category)
            if not relevant_memories:
                retention_days = self.expiry_date_decision(mem['text'])
                await self.store_memory(user_id, mem['text'], retention_days, category)
                continue

            memory_context = [
                f"Memory {idx+1}: {memory['text']} (ID: {memory['_id']}, Created: {memory['created_at']}, Expires: {memory['expiry_at']})"
                for idx, memory in enumerate(relevant_memories)
            ]

            def get_processed_json_response_update(mem_text: str, memory_context: List[str]) -> Dict:
                try:
                    runnable = OllamaRunnable(
                        model_url="http://localhost:11434/api/chat",
                        model_name=self.model_name,
                        system_prompt_template=update_decision_system_prompt,
                        user_prompt_template=update_user_prompt_template,
                        input_variables=["current_query", "memory_context"],
                        response_type="json",
                        required_format=update_required_format
                    )
                    print(f"Invoking memory update decision for query: '{mem_text}...'")
                    response = runnable.invoke({"current_query": mem_text, "memory_context": memory_context})
                    print(f"Memory update decision response: {response}")
                    return response if isinstance(response, dict) else {"update": []}
                except Exception as e:
                    print(f"Error in get_processed_json_response_update: {e}")
                    return {"update": []}

            update_details = get_processed_json_response_update(mem['text'], memory_context)
            updates = update_details.get('update', [])
            if updates:
                for update in updates:
                    memory_id = update["id"]
                    updated_text = update["text"]
                    # For MongoDB, we don't need to determine original_category from DB
                    # as it's part of the document or can be passed directly.
                    # Assuming 'category' from the initial 'mem' is the target category.
                    original_category = category # Use the category determined earlier

                    new_embedding = self.compute_embedding(updated_text)
                    query_keywords = self.extract_keywords(updated_text)
                    retention_days = self.expiry_date_decision(updated_text)
                    
                    # Convert memory_id to ObjectId if it's a string representation of ObjectId
                    # If it's a custom string ID, use it directly.
                    from bson.objectid import ObjectId
                    try:
                        obj_memory_id = ObjectId(memory_id)
                    except:
                        obj_memory_id = memory_id # Assume it's a string ID if not ObjectId-compatible

                    await self.update_memory_crud(
                        user_id=user_id,
                        category=original_category,
                        memory_id=obj_memory_id,
                        new_text=updated_text,
                        retention_days=retention_days
                    )
            
    async def update_memory_crud(self, user_id: str, category: str, memory_id: Any, new_text: str, retention_days: int):
        """
        Updates an existing memory record using direct CRUD operations in MongoDB.

        Args:
            user_id (str): The ID of the user owning the memory.
            category (str): The category of the memory.
            memory_id (Any): The ID of the memory record to update (can be ObjectId or str).
            new_text (str): The new text content for the memory.
            retention_days (int): The new retention period in days (e.g., 1-90).

        Raises:
            ValueError: If the category is invalid, retention_days is out of range,
                        the memory is not found, or the memory does not belong to the user.
            Exception: For other unexpected errors.
        """
        print(f"Attempting CRUD update for memory ID {memory_id} in category '{category}' for user '{user_id}'")
        print(f"New text: '{new_text}...', New retention: {retention_days} days")

        # 1. Validate Inputs
        category_lower = category.lower()
        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
            raise ValueError(f"Invalid category specified: {category}")

        if not (1 <= retention_days <= 90):
            raise ValueError("Retention days must be between 1 and 90.")

        if not new_text:
            raise ValueError("Memory text cannot be empty.")

        try:
            # 2. Prepare Data for Update
            new_embedding = self.compute_embedding(new_text)
            new_keywords = self.extract_keywords(new_text)
            new_expiry_at = datetime.now() + timedelta(days=retention_days)

            # 3. Verify Ownership and Existence (find_one_and_update handles this implicitly)
            # We'll use the _id field for MongoDB documents
            query = {"_id": memory_id, "user_id": user_id, "category": category_lower}
            
            update_data = {
                "original_text": new_text,
                "keywords": new_keywords,
                "embedding": new_embedding.tolist(), # Store as list for MongoDB
                "expiry_at": new_expiry_at,
                "last_updated": datetime.now()
            }

            result = await self.memory_collection.update_one(query, {"$set": update_data})

            if result.matched_count == 0:
                raise ValueError(f"Memory with ID {memory_id} not found or does not belong to user {user_id} in category '{category_lower}'")
            
            if result.modified_count == 0:
                print(f"No changes made to memory ID {memory_id}. Document already up-to-date or no fields changed.")

            print(f"Successfully updated memory ID {memory_id} in category '{category_lower}'")

        except Exception as e:
            print(f"Error during CRUD update for memory ID {memory_id}: {e}")
            raise # Re-raise the specific exception

    async def store_memory(self, user_id: str, text: str, retention_days: int, category: str) -> bool:
        print(f"Attempting to store memory: '{text}...' in category '{category}'")
        try:
            keywords = self.extract_keywords(text)
            embedding = self.compute_embedding(text)
            current_time = datetime.now()
            expiry_time = current_time + timedelta(days=retention_days)

            memory_document = {
                "user_id": user_id,
                "original_text": text,
                "keywords": keywords,
                "embedding": embedding.tolist(), # Convert numpy array to list for MongoDB
                "entities": [], # Placeholder, can be extracted if needed
                "created_at": current_time,
                "expiry_at": expiry_time,
                "is_active": True,
                "category": category.lower() # Store category in the document
            }
            
            await self.memory_collection.insert_one(memory_document)
            print(f"Inserted memory into MongoDB collection '{self.memory_collection.name}': '{text}...' with expiry at {expiry_time}")
            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    async def get_relevant_memories(self, user_id: str, query: str, category: str, similarity_threshold: float = 0.5) -> List[Dict]:
        print(f"Retrieving memories for user '{user_id}' in category '{category}' for query: '{query}...'")
        query_embedding = self.embedding_model.encode(query)
        
        # Find memories for the user and category that are active and not expired
        current_time = datetime.now()
        cursor = self.memory_collection.find({
            "user_id": user_id,
            "category": category.lower(),
            "is_active": True,
            "expiry_at": {"$gt": current_time}
        })
        
        memories = []
        async for doc in cursor:
            memory_embedding = np.array(doc['embedding'], dtype=np.float32) # Convert list back to numpy array
            similarity = self.cosine_similarity(query_embedding, memory_embedding)
            if similarity >= similarity_threshold:
                memories.append({
                    '_id': str(doc['_id']), # Convert ObjectId to string for consistent ID handling
                    'text': doc['original_text'],
                    'keywords': doc['keywords'],
                    'similarity': similarity,
                    'created_at': doc['created_at'],
                    'expiry_at': doc['expiry_at'],
                    'category': doc['category']
                })
        
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"Retrieved {len(memories)} relevant memories for query '{query}...' in category '{category}'")
        return memories

    async def process_user_query(self, user_id: str, query: str) -> str:
        print(f"Processing user query: '{query}' for user ID: '{user_id}'")
        query_keywords = self.extract_keywords(query)
        determined_category = self.determine_category(query_keywords)
        relevant_memories = await self.get_relevant_memories(user_id, query, determined_category)

        if not relevant_memories:
            print(f"No relevant memories found in category '{determined_category}'. Falling back to 'personal' category.")
            relevant_memories_personal = await self.get_relevant_memories(user_id, query, 'personal')
            if relevant_memories_personal:
                print(f"Found relevant memories in 'personal' category.")
                memory_context = "\n".join([f"- {memory['text']}" for memory in relevant_memories_personal])
            else:
                print("No relevant memories found in 'personal' category either.")
                memory_context = "" # No context available
        else:
            print(f"Found relevant memories in category '{determined_category}'.")
            memory_context = "\n".join([f"- {memory['text']}" for memory in relevant_memories])

        print(f"Memory context for query '{query}':\n{memory_context}")

        if not memory_context:
            # Fallback response when no context is available
            response = None
        else:
            runnable = OllamaRunnable(
                model_url="http://localhost:11434/api/chat",
                model_name=self.model_name,
                system_prompt_template="Use the provided memory context to answer the user's query:\n{memory_context}",
                user_prompt_template="{query}",
                input_variables=["query", "memory_context"],
                response_type="text"
            )
            print("Invoking LLM to answer query using memory context.")
            response = runnable.invoke({"query": query, "memory_context": memory_context})
            print(f"LLM response: '{response}'")
        return response

    async def cleanup_expired_memories(self):
        # This method will be re-implemented using MongoDB's TTL feature in a later subtask.
        # For now, it's a no-op as the TTL index handles automatic cleanup.
        print("MongoDB TTL index handles expired memory cleanup automatically.")
        pass

    async def delete_memory(self, user_id: str, category: str, memory_id: Any):
        """Delete a memory by ID and category for a specific user in MongoDB."""
        if category.lower() not in [cat.lower() for cat in self.categories.keys()]:
            raise ValueError("Invalid category")
        
        # Convert memory_id to ObjectId if it's a string representation of ObjectId
        from bson.objectid import ObjectId
        try:
            obj_memory_id = ObjectId(memory_id)
        except:
            obj_memory_id = memory_id # Assume it's a string ID if not ObjectId-compatible

        query = {"_id": obj_memory_id, "user_id": user_id, "category": category.lower()}
        result = await self.memory_collection.delete_one(query)
        
        if result.deleted_count == 0:
            raise ValueError(f"Memory with ID {memory_id} not found or not owned by user {user_id} in category '{category}'")
        
        print(f"Deleted memory ID {memory_id} from {category.lower()} for user {user_id}")

    async def fetch_memories_by_category(self, user_id: str, category: str, limit: int = 50) -> List[Dict]:
        """
        Fetch memories for a specific user and category from the MongoDB database.
        
        Args:
            user_id (str): The ID of the user
            category (str): Memory category to fetch
            limit (int, optional): Maximum number of memories to retrieve. Defaults to 50.
        
        Returns:
            List[Dict]: List of memory dictionaries
        """
        print(f"Fetching memories for user '{user_id}' in category '{category}'")
        
        category_lower = category.lower()
        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
            print(f"Invalid category: {category}")
            return []
        
        current_time = datetime.now()
        cursor = self.memory_collection.find({
            "user_id": user_id,
            "category": category_lower,
            "is_active": True,
            "expiry_at": {"$gt": current_time}
        }).sort("created_at", DESCENDING).limit(limit)
        
        memories = []
        async for doc in cursor:
            memories.append({
                'id': str(doc['_id']), # Convert ObjectId to string
                'original_text': doc['original_text'],
                'keywords': doc['keywords'],
                'created_at': doc['created_at'],
                'expiry_at': doc['expiry_at'],
                'category': doc['category']
            })
        
        print(f"Retrieved {len(memories)} memories")
        return memories
            
    async def clear_all_memories(self, user_id: str):
        """Deletes all memories for a specific user across all categories in MongoDB."""
        print(f"Clearing all memories for user ID: {user_id}")
        try:
            result = await self.memory_collection.delete_many({"user_id": user_id})
            print(f"Cleared a total of {result.deleted_count} memories for user {user_id}.")
        except Exception as e:
            print(f"Error clearing memories for user {user_id}: {e}")
            raise # Re-raise the error