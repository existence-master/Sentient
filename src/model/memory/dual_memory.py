from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import sqlite3
from datetime import datetime, date, timedelta
import spacy
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

from .helpers import *
from .prompts import *
from model.app.base import *

load_dotenv("model/.env")

nlp = spacy.load("en_core_web_sm")

# SQLite adapters and converters (unchanged)
def adapt_date_iso(val: date) -> str:
    return val.isoformat()

def adapt_datetime_iso(val: datetime) -> str:
    return val.isoformat()

def adapt_datetime_epoch(val: datetime) -> int:
    return int(val.timestamp())

sqlite3.register_adapter(date, adapt_date_iso)
sqlite3.register_adapter(datetime, adapt_datetime_iso)

def convert_date(val: bytes) -> date:
    return date.fromisoformat(val.decode())

def convert_datetime(val: bytes) -> datetime:
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val: bytes) -> datetime:
    return datetime.datetime.fromtimestamp(int(val))

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", model_name: str = os.environ["BASE_MODEL_REPO_ID"]):
        print("Initializing MemoryManager...")
        self.db_path = db_path
        self.model_name = model_name
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")
        self.categories = {
            "PERSONAL": ["home", "hobby", "diary", "self", "goals", "habit", "routine", "personal"],
            "WORK": ["office", "business", "client", "report", "presentation", "deadline", "manager", "workplace"],
            "SOCIAL": ["meetup", "gathering", "party", "social", "community", "group", "network"],
            "RELATIONSHIP": ["friend", "family", "partner", "colleague", "neighbor"],
            "FINANCE": ["money", "bank", "loan", "debt", "payment", "buy", "sell"],
            "SPIRITUAL": ["pray", "meditation", "temple", "church", "mosque"],
            "CAREER": ["job", "work", "interview", "meeting", "project"],
            "TECHNOLOGY": ["phone", "computer", "laptop", "device", "software"],
            "HEALTH": ["doctor", "medicine", "exercise", "diet", "hospital"],
            "EDUCATION": ["study", "school", "college", "course", "learn"],
            "TRANSPORTATION": ["car", "bike", "bus", "train", "flight"],
            "ENTERTAINMENT": ["movie", "game", "music", "party", "book"],
            "TASKS": ["todo", "deadline", "appointment", "schedule", "reminder"]
        }
        print("Initializing database...")
        self.initialize_database()
        print("MemoryManager initialized.")

    def initialize_database(self):
        print("Initializing SQLite database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for category in self.categories.keys():
            print(f"Creating table for category: {category.lower()}...")
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {category.lower()} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                original_text TEXT NOT NULL,
                keywords TEXT NOT NULL,
                embedding BLOB NOT NULL,
                entities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expiry_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
            ''')
            print(f"Table for category: {category.lower()} created.")
        conn.commit()
        conn.close()
        print("SQLite database initialized.")

    def compute_embedding(self, text: str) -> bytes:
        print(f"Computing embedding for text: '{text[:50]}...'")
        embedding = np.array(self.embedding_model.encode(text)).tobytes()
        print("Embedding computed.")
        return embedding

    def bytes_to_array(self, embedding_bytes: bytes) -> np.ndarray:
        return np.frombuffer(embedding_bytes, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (norm(a) * norm(b))

    def extract_keywords(self, text: str) -> List[str]:
        print(f"Extracting keywords from text: '{text[:50]}...'")
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

    def expiry_date_decision(self, query: str) -> Dict:
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
            print(f"Invoking expiry date decision for query: '{query[:50]}...'")
            response = runnable.invoke({"query": query, "formatted_date": formatted_date})
            print(f"Expiry date decision response: {response}")
            return response if isinstance(response, dict) else {"retention_days": 7}
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
            print(f"Invoking memory extraction for query: '{current_query[:50]}...'")
            response = runnable.invoke({"current_query": current_query, "date_today": date_today})
            print(f"Memory extraction response: {response}")
            return response if isinstance(response, dict) else {"memories": []}
        except Exception as e:
            print(f"Error in extract_and_invoke_memory: {e}")
            return {"memories": []}

    def update_memory(self, user_id: str, current_query: str) -> Optional[Dict]:
        memories = self.extract_and_invoke_memory(current_query)
        for mem in memories.get('memories', []):
            category = mem['category'].lower()
            relevant_memories = self.get_relevant_memories(user_id, mem['text'], category)
            if not relevant_memories:
                retention_days = self.expiry_date_decision(mem['text'])
                self.store_memory(user_id, mem['text'], retention_days, category)
                continue

            memory_context = [
                f"Memory {idx+1}: {memory['text']} (ID: {memory['id']}, Created: {memory['created_at']}, Expires: {memory['expiry_at']})"
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
                    print(f"Invoking memory update decision for query: '{mem_text[:50]}...'")
                    response = runnable.invoke({"current_query": mem_text, "memory_context": memory_context})
                    print(f"Memory update decision response: {response}")
                    return response if isinstance(response, dict) else {"update": []}
                except Exception as e:
                    print(f"Error in get_processed_json_response_update: {e}")
                    return {"update": []}

            def get_memory_category(cursor: sqlite3.Cursor, memory_id: int) -> Optional[str]:
                category_tables = list(self.categories.keys())
                for cat in category_tables:
                    cursor.execute(f'SELECT 1 FROM {cat.lower()} WHERE id = ?', (memory_id,))
                    if cursor.fetchone():
                        return cat.lower()
                return None

            update_details = get_processed_json_response_update(mem['text'], memory_context)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            updates = update_details.get('update', [])
            if updates:
                for update in updates:
                    memory_id = update["id"]
                    updated_text = update["text"]
                    original_category = get_memory_category(cursor, memory_id)
                    if not original_category:
                        continue
                    new_embedding = self.compute_embedding(updated_text)
                    query_keywords = self.extract_keywords(updated_text)
                    expiry_info = self.expiry_date_decision(updated_text)
                    retention_days = expiry_info.get("retention_days", 7)
                    cursor.execute(f'''
                    UPDATE {original_category}
                    SET original_text = ?, embedding = ?, keywords = ?, expiry_at = datetime('now', '+{retention_days} days')
                    WHERE id = ?
                    ''', (updated_text, new_embedding, ','.join(query_keywords), memory_id))
            conn.commit()
            conn.close()

    def store_memory(self, user_id: str, text: str, retention_days: Dict, category: str) -> bool:
        print(f"Attempting to store memory: '{text[:50]}...' in category '{category}'")
        try:
            keywords = self.extract_keywords(text)
            embedding = self.compute_embedding(text)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.datetime.now()
            expiry_time = current_time + timedelta(days=int(retention_days.get("retention_days", 7)))
            cursor.execute(f'''
            INSERT INTO {category.lower()} (user_id, original_text, keywords, embedding, created_at, expiry_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, text, ','.join(keywords), embedding, current_time, expiry_time))
            conn.commit()
            print(f"Inserted memory into {category.lower()}: '{text[:50]}...' with expiry at {expiry_time}")
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    def get_relevant_memories(self, user_id: str, query: str, category: str, similarity_threshold: float = 0.5) -> List[Dict]:
        print(f"Retrieving memories for user '{user_id}' in category '{category}' for query: '{query[:50]}...'")
        query_embedding = self.embedding_model.encode(query)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f'''
        SELECT id, original_text, keywords, embedding, created_at, expiry_at
        FROM {category.lower()}
        WHERE user_id = ? AND is_active = 1 AND datetime('now') < expiry_at
        ''', (user_id,))
        memories = []
        for row in cursor.fetchall():
            memory_embedding = self.bytes_to_array(row[3])
            similarity = self.cosine_similarity(query_embedding, memory_embedding)
            if similarity >= similarity_threshold:
                memories.append({
                    'id': row[0], 'text': row[1], 'keywords': row[2].split(','),
                    'similarity': similarity, 'created_at': row[4], 'expiry_at': row[5]
                })
        conn.close()
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"Retrieved {len(memories)} relevant memories for query '{query[:50]}...' in category '{category}'")
        return memories

    def process_user_query(self, user_id: str, query: str) -> str:
        print(f"Processing user query: '{query}' for user ID: '{user_id}'")
        query_keywords = self.extract_keywords(query)
        determined_category = self.determine_category(query_keywords)
        relevant_memories = self.get_relevant_memories(user_id, query, determined_category)

        if not relevant_memories:
            print(f"No relevant memories found in category '{determined_category}'. Falling back to 'personal' category.")
            relevant_memories_personal = self.get_relevant_memories(user_id, query, 'personal')
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

    def cleanup_expired_memories(self):
        print("Cleaning up expired memories...")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.datetime.now()
            for category in self.categories.keys():
                print(f"Cleaning up expired memories in category: {category.lower()}...")
                cursor.execute(f'DELETE FROM {category.lower()} WHERE expiry_at < ?', (current_time,))
                print(f"Deleted {cursor.rowcount} expired memories from {category.lower()}")
            conn.commit()
            conn.close()
            print("Expired memory cleanup completed.")
        except Exception as e:
            print(f"Error during memory cleanup: {e}")

    def delete_memory(self, user_id: str, category: str, memory_id: int):
        """Delete a memory by ID and category."""
        if category.lower() not in [cat.lower() for cat in self.categories.keys()]:
            raise ValueError("Invalid category")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verify memory exists and belongs to the user
        cursor.execute(f'SELECT user_id FROM {category.lower()} WHERE id = ?', (memory_id,))
        result = cursor.fetchone()
        if not result or result[0] != user_id:
            conn.close()
            raise ValueError("Memory not found or not owned by the user")
        
        cursor.execute(f'DELETE FROM {category.lower()} WHERE id = ?', (memory_id,))
        conn.commit()
        conn.close()
        print(f"Deleted memory ID {memory_id} from {category.lower()}")

    def fetch_memories_by_category(self, user_id: str, category: str, limit: int = 50) -> List[Dict]:
        """
        Fetch memories for a specific user and category from the SQLite database.
        
        Args:
            user_id (str): The ID of the user
            category (str): Memory category to fetch
            limit (int, optional): Maximum number of memories to retrieve. Defaults to 50.
        
        Returns:
            List[Dict]: List of memory dictionaries
        """
        print(f"Fetching memories for user '{user_id}' in category '{category}'")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            category = category.lower()
            if category not in [cat.lower() for cat in self.categories.keys()]:
                print(f"Invalid category: {category}")
                return []
            
            cursor.execute(f'''
            SELECT id, original_text, keywords, created_at, expiry_at
            FROM {category}
            WHERE user_id = ? AND is_active = 1 AND datetime('now') < expiry_at
            ORDER BY created_at DESC
            LIMIT ?
            ''', (user_id, limit))
            
            memories = [
                {
                    'id': row[0],
                    'original_text': row[1],
                    'keywords': row[2].split(','),
                    'created_at': row[3],
                    'expiry_at': row[4],
                    'category': category  # Added category field
                }
                for row in cursor.fetchall()
            ]
            
            print(f"Retrieved {len(memories)} memories")
            return memories
        
        except Exception as e:
            print(f"Error fetching memories: {e}")
            return []
        finally:
            conn.close()
            
    def clear_all_memories(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for table in self.categories.values():
            cursor.execute(f'DELETE FROM {table.lower()} WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()