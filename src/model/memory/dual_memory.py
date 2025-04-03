import logging
import os
import sqlite3
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import numpy as np
import spacy
from dotenv import load_dotenv
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from .helpers import *  # Assuming helpers.py exists in the same directory
from .prompts import *   # Assuming prompts.py exists
from model.app.base import OllamaRunnable # Assuming base.py exists

# --- Logging Configuration ---
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
app_env = os.getenv("APP_ENV", "development")

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use the module name for the logger
logger.setLevel(log_level)

# Clear existing handlers to avoid duplicates if reloaded
if logger.hasHandlers():
    logger.handlers.clear()

if app_env == "production":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "memory_manager.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info("Production environment detected. Logging to file: %s", log_file)
else:
    stream_handler = logging.StreamHandler() # Defaults to sys.stderr
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.info("Development environment detected. Logging to stdout.")

# --- Environment Variables & Model Loading ---
logger.info("Loading environment variables from model/.env...")
load_dotenv("model/.env") # Ensure this path is correct relative to execution
base_model_repo_id = os.environ.get("BASE_MODEL_REPO_ID")
if not base_model_repo_id:
    logger.error("BASE_MODEL_REPO_ID not found in environment variables.")
    raise ValueError("BASE_MODEL_REPO_ID must be set in the environment.")
else:
    logger.info("BASE_MODEL_REPO_ID loaded successfully.")


logger.info("Loading spaCy model 'en_core_web_sm'...")
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully.")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    raise

# --- SQLite Adapters and Converters (Unchanged) ---
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
    # Ensure datetime.datetime is used for clarity
    return datetime.fromisoformat(val.decode())

def convert_timestamp(val: bytes) -> datetime:
    # Ensure datetime.datetime is used for clarity
    return datetime.fromtimestamp(int(val))

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)
logger.info("SQLite type adapters and converters registered.")


# --- Memory Manager Class ---
class MemoryManager:
    def __init__(self, db_path: str = "memory.db", model_name: str = base_model_repo_id):
        self.logger = logging.getLogger(self.__class__.__name__) # Logger specific to this class instance
        self.logger.info("Initializing MemoryManager...")
        self.db_path = db_path
        self.model_name = model_name
        self.logger.info(f"Using LLM model: {self.model_name}")
        self.logger.info("Loading SentenceTransformer model 'sentence-transformers/all-MiniLM-L6-v2'...")
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            raise
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
        self.logger.info(f"Defined categories: {list(self.categories.keys())}")
        self.logger.info("Initializing database...")
        self.initialize_database()
        self.logger.info("MemoryManager initialized successfully.")

    def initialize_database(self):
        self.logger.info(f"Initializing SQLite database at path: {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            cursor = conn.cursor()
            for category in self.categories.keys():
                table_name = category.lower()
                self.logger.info(f"Ensuring table exists for category: {table_name}")
                cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    entities TEXT, -- Consider JSON for entities if structured data is needed
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
                ''')
                self.logger.debug(f"Table '{table_name}' checked/created.")
            conn.commit()
            self.logger.info("SQLite database initialization complete.")
        except sqlite3.Error as e:
            self.logger.error(f"Database error during initialization: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    def compute_embedding(self, text: str) -> bytes:
        self.logger.debug(f"Computing embedding for text: '{text[:50]}...'")
        try:
            embedding_array = self.embedding_model.encode(text)
            # Ensure the array is float32 for consistency, although MiniLM usually is
            embedding_bytes = np.array(embedding_array, dtype=np.float32).tobytes()
            self.logger.debug(f"Embedding computed successfully, size: {len(embedding_bytes)} bytes.")
            return embedding_bytes
        except Exception as e:
            self.logger.error(f"Error computing embedding for text '{text[:50]}...': {e}", exc_info=True)
            raise # Re-raise exception as embedding is critical

    def bytes_to_array(self, embedding_bytes: bytes) -> np.ndarray:
        # Assuming embeddings were stored as float32
        array = np.frombuffer(embedding_bytes, dtype=np.float32)
        self.logger.debug(f"Converted {len(embedding_bytes)} bytes to numpy array, shape: {array.shape}")
        return array

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        self.logger.debug(f"Calculating cosine similarity between vectors of shape {a.shape} and {b.shape}")
        dot_product = np.dot(a, b)
        norm_a = norm(a)
        norm_b = norm(b)
        if norm_a == 0 or norm_b == 0:
            self.logger.warning("Cannot compute cosine similarity with zero vector(s).")
            return 0.0
        similarity = dot_product / (norm_a * norm_b)
        self.logger.debug(f"Cosine similarity result: {similarity:.4f}")
        return float(similarity) # Ensure float return type

    def extract_keywords(self, text: str) -> List[str]:
        self.logger.debug(f"Extracting keywords from text: '{text[:50]}...'")
        try:
            doc = nlp(text.lower())
            # Extract named entities
            entities = [ent.text for ent in doc.ents]
            # Extract nouns and verbs (lemmatized, not stop words, length > 2)
            tokens = [
                token.lemma_ for token in doc
                if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop and len(token.text) > 2
            ]
            # Combine and make unique
            keywords = list(set(entities + tokens))
            self.logger.debug(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            self.logger.error(f"Error extracting keywords from text '{text[:50]}...': {e}", exc_info=True)
            return [] # Return empty list on error

    def determine_category(self, keywords: List[str]) -> str:
        self.logger.debug(f"Determining category from keywords: {keywords}")
        if not keywords:
            self.logger.warning("No keywords provided, defaulting category to 'tasks'.")
            return "tasks"

        category_scores: Dict[str, int] = {category: 0 for category in self.categories}
        for keyword in keywords:
            # Simple substring matching for keywords within categories
            for category, category_keywords in self.categories.items():
                if any(cat_keyword in keyword for cat_keyword in category_keywords):
                    category_scores[category] += 1
                    self.logger.debug(f"Keyword '{keyword}' matched category '{category}'. Score: {category_scores[category]}")

        max_score = max(category_scores.values())

        # Default to 'tasks' if no keywords match any category strongly
        if max_score == 0:
            determined_category = "tasks"
            self.logger.info(f"No category keywords matched. Defaulting to category: {determined_category}")
        else:
            # Find the category with the highest score (first one in case of ties)
            determined_category = max(category_scores.items(), key=lambda item: item[1])[0]
            self.logger.info(f"Determined category: {determined_category} with score {max_score}")

        return determined_category.lower() # Return lowercase category name

    def expiry_date_decision(self, query: str) -> Dict:
        today = date.today()
        formatted_date = today.strftime("%d %B %Y %A")
        self.logger.info(f"Requesting LLM for expiry date decision for query: '{query[:50]}...'")
        self.logger.debug(f"Current date for expiry context: {formatted_date}")

        # Modified system prompt to explicitly ask for JSON with 'retention_days'
        modified_system_expiry_template = system_memory_expiry_template.replace(
            "Your response must strictly adhere to the following rules:\n1. The minimum storage time is 1 day and the maximum is 90 days.",
            "Return ONLY a JSON object with a single key 'retention_days' and the value as an integer representing the number of days (minimum 1, maximum 90). Example: {\"retention_days\": 30}"
        )

        try:
            # Assuming OllamaRunnable is correctly imported and configured
            runnable = OllamaRunnable(
                model_url=os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", # Make URL configurable
                model_name=self.model_name,
                system_prompt_template=modified_system_expiry_template,
                user_prompt_template=user_memory_expiry_template,
                input_variables=["query", "formatted_date"],
                response_type="json",
                required_format={"retention_days": "int"} # Add required format for validation if supported
            )
            response = runnable.invoke({"query": query, "formatted_date": formatted_date})
            self.logger.info(f"LLM response for expiry date: {response}")

            # Validate response
            if isinstance(response, dict) and "retention_days" in response and isinstance(response["retention_days"], int):
                 # Clamp the value between 1 and 90
                retention_days = max(1, min(90, response["retention_days"]))
                validated_response = {"retention_days": retention_days}
                self.logger.info(f"Validated expiry decision: {validated_response}")
                return validated_response
            else:
                self.logger.warning(f"Invalid expiry response format from LLM: {response}. Defaulting to 7 days.")
                return {"retention_days": 7}

        except Exception as e:
            self.logger.error(f"Error invoking Ollama for expiry date decision: {e}", exc_info=True)
            self.logger.warning("Defaulting to 7 days retention due to error.")
            return {"retention_days": 7}

    def extract_and_invoke_memory(self, current_query: str) -> Dict:
        date_today = date.today().isoformat() # Use ISO format for consistency
        self.logger.info(f"Requesting LLM to extract potential memories from query: '{current_query[:50]}...'")
        self.logger.debug(f"Current date for memory extraction context: {date_today}")

        try:
             # Assuming OllamaRunnable is correctly imported and configured
            runnable = OllamaRunnable(
                model_url=os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", # Make URL configurable
                model_name=self.model_name,
                system_prompt_template=extract_memory_system_prompt_template,
                user_prompt_template=extract_memory_user_prompt_template,
                input_variables=["current_query", "date_today"],
                response_type="json",
                required_format=extract_memory_required_format # Validate against expected structure
            )
            response = runnable.invoke({"current_query": current_query, "date_today": date_today})
            self.logger.info(f"LLM response for memory extraction: {response}")

            # Basic validation
            if isinstance(response, dict) and "memories" in response and isinstance(response["memories"], list):
                self.logger.info(f"Extracted {len(response['memories'])} potential memories.")
                return response
            else:
                self.logger.warning(f"Invalid memory extraction response format from LLM: {response}. Returning empty list.")
                return {"memories": []}

        except Exception as e:
            self.logger.error(f"Error invoking Ollama for memory extraction: {e}", exc_info=True)
            self.logger.warning("Returning empty memory list due to error.")
            return {"memories": []}

    def update_memory(self, user_id: str, current_query: str) -> None:
        """Extracts potential memories, checks against existing ones, and updates or stores them."""
        self.logger.info(f"Starting memory update process for user '{user_id}' and query: '{current_query[:50]}...'")
        extracted_data = self.extract_and_invoke_memory(current_query)
        extracted_memories = extracted_data.get('memories', [])

        if not extracted_memories:
            self.logger.info("No potential memories extracted from the query. Nothing to update or store.")
            return

        conn = None # Initialize conn to None
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            cursor = conn.cursor()

            for mem in extracted_memories:
                if not isinstance(mem, dict) or 'text' not in mem or 'category' not in mem:
                    self.logger.warning(f"Skipping invalid memory structure in extraction result: {mem}")
                    continue

                mem_text = mem['text']
                category = mem['category'].lower() # Use lowercase category

                # Ensure category is valid before proceeding
                if category not in [cat.lower() for cat in self.categories.keys()]:
                    self.logger.warning(f"Invalid category '{category}' extracted for memory '{mem_text[:50]}...'. Determining category based on keywords.")
                    # Re-determine category if the LLM provided an invalid one
                    keywords_for_category = self.extract_keywords(mem_text)
                    category = self.determine_category(keywords_for_category)
                    self.logger.info(f"Re-determined category as '{category}' based on keywords.")


                self.logger.info(f"Processing extracted memory: '{mem_text[:50]}...' for category '{category}'")
                relevant_memories = self.get_relevant_memories(user_id, mem_text, category)

                if not relevant_memories:
                    self.logger.info(f"No existing relevant memories found for '{mem_text[:50]}...'. Storing as new memory.")
                    # Need to drop the connection before calling store_memory which opens its own
                    conn.close()
                    conn = None
                    retention_info = self.expiry_date_decision(mem_text)
                    self.store_memory(user_id, mem_text, retention_info, category)
                    # Re-establish connection for the next iteration
                    conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
                    cursor = conn.cursor()
                    continue

                # If relevant memories exist, decide whether to update
                self.logger.info(f"Found {len(relevant_memories)} relevant existing memories. Checking for updates.")
                memory_context = [
                    f"Memory {idx+1}: {memory['text']} (ID: {memory['id']}, Created: {memory['created_at']}, Expires: {memory['expiry_at']})"
                    for idx, memory in enumerate(relevant_memories)
                ]
                self.logger.debug(f"Memory context for update decision:\n{memory_context}")

                update_details = self._invoke_update_decision_llm(mem_text, memory_context)
                updates_to_perform = update_details.get('update', [])

                if not updates_to_perform:
                    self.logger.info(f"LLM decided no update needed for memory based on '{mem_text[:50]}...'.")
                    # Potentially store as new if LLM didn't identify it as an update?
                    # Or assume the extraction was redundant if no update needed. Let's assume redundancy for now.
                    continue

                self.logger.info(f"LLM suggested {len(updates_to_perform)} update(s) for memory based on '{mem_text[:50]}...'.")
                for update_action in updates_to_perform:
                    if not isinstance(update_action, dict) or "id" not in update_action or "text" not in update_action:
                         self.logger.warning(f"Skipping invalid update action structure: {update_action}")
                         continue

                    memory_id = update_action["id"]
                    updated_text = update_action["text"]

                    # Find the original category of the memory being updated
                    original_category = self._get_memory_category_by_id(cursor, memory_id)
                    if not original_category:
                        self.logger.warning(f"Could not find original category for memory ID {memory_id}. Skipping update.")
                        continue

                    self.logger.info(f"Updating memory ID {memory_id} in category '{original_category}' with new text: '{updated_text[:50]}...'")

                    try:
                        new_embedding = self.compute_embedding(updated_text)
                        query_keywords = self.extract_keywords(updated_text)
                        expiry_info = self.expiry_date_decision(updated_text)
                        retention_days = expiry_info.get("retention_days", 7) # Default 7 days
                        expiry_time = datetime.now() + timedelta(days=retention_days)

                        cursor.execute(f'''
                        UPDATE {original_category}
                        SET original_text = ?, embedding = ?, keywords = ?, expiry_at = ?, updated_at = CURRENT_TIMESTAMP -- Add updated_at if needed
                        WHERE id = ? AND user_id = ?
                        ''', (updated_text, new_embedding, ','.join(query_keywords), expiry_time, memory_id, user_id))
                        conn.commit()
                        self.logger.info(f"Successfully updated memory ID {memory_id}. Rows affected: {cursor.rowcount}")
                        if cursor.rowcount == 0:
                             self.logger.warning(f"Update for memory ID {memory_id} affected 0 rows. Check user_id or ID existence.")

                    except sqlite3.Error as db_err:
                        self.logger.error(f"Database error updating memory ID {memory_id}: {db_err}", exc_info=True)
                        conn.rollback() # Rollback the specific failed update
                    except Exception as e:
                        self.logger.error(f"Unexpected error updating memory ID {memory_id}: {e}", exc_info=True)
                        conn.rollback()

            self.logger.info(f"Finished memory update process for user '{user_id}'.")

        except sqlite3.Error as e:
            self.logger.error(f"Database error during update process for user '{user_id}': {e}", exc_info=True)
            if conn:
                conn.rollback() # Rollback any pending changes if connection exists
        except Exception as e:
            self.logger.error(f"Unexpected error during update process for user '{user_id}': {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after update process.")


    def _invoke_update_decision_llm(self, mem_text: str, memory_context: List[str]) -> Dict:
        """Helper function to invoke the LLM for the update decision."""
        self.logger.info(f"Requesting LLM for update decision on memory: '{mem_text[:50]}...'")
        try:
            runnable = OllamaRunnable(
                model_url=os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", # Make URL configurable
                model_name=self.model_name,
                system_prompt_template=update_decision_system_prompt,
                user_prompt_template=update_user_prompt_template,
                input_variables=["current_query", "memory_context"],
                response_type="json",
                required_format=update_required_format # Validate response structure
            )
            response = runnable.invoke({"current_query": mem_text, "memory_context": "\n".join(memory_context)}) # Pass context as single string
            self.logger.info(f"LLM response for update decision: {response}")

            # Basic Validation
            if isinstance(response, dict) and "update" in response and isinstance(response["update"], list):
                return response
            else:
                self.logger.warning(f"Invalid update decision response format from LLM: {response}. Returning empty update list.")
                return {"update": []}

        except Exception as e:
            self.logger.error(f"Error invoking Ollama for update decision: {e}", exc_info=True)
            self.logger.warning("Returning empty update list due to error.")
            return {"update": []}

    def _get_memory_category_by_id(self, cursor: sqlite3.Cursor, memory_id: int) -> Optional[str]:
        """Finds the category table containing a specific memory ID."""
        self.logger.debug(f"Searching for category containing memory ID: {memory_id}")
        category_tables = [cat.lower() for cat in self.categories.keys()]
        for table_name in category_tables:
            try:
                cursor.execute(f'SELECT 1 FROM {table_name} WHERE id = ?', (memory_id,))
                if cursor.fetchone():
                    self.logger.debug(f"Memory ID {memory_id} found in category '{table_name}'.")
                    return table_name
            except sqlite3.Error as e:
                self.logger.error(f"Database error checking table '{table_name}' for memory ID {memory_id}: {e}", exc_info=True)
                # Continue checking other tables even if one fails
        self.logger.warning(f"Memory ID {memory_id} not found in any category table.")
        return None


    def store_memory(self, user_id: str, text: str, retention_days_info: Dict, category: str) -> bool:
        category_lower = category.lower()
        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
             self.logger.error(f"Attempted to store memory in invalid category: '{category}'.")
             return False

        retention_days = retention_days_info.get("retention_days", 7) # Default 7 days
        self.logger.info(f"Attempting to store memory for user '{user_id}' in category '{category_lower}': '{text[:50]}...' with retention {retention_days} days.")

        conn = None
        try:
            keywords = self.extract_keywords(text)
            embedding = self.compute_embedding(text)
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            cursor = conn.cursor()
            current_time = datetime.now()
            expiry_time = current_time + timedelta(days=int(retention_days)) # Ensure days is int

            cursor.execute(f'''
            INSERT INTO {category_lower} (user_id, original_text, keywords, embedding, created_at, expiry_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (user_id, text, ','.join(keywords), embedding, current_time, expiry_time))
            conn.commit()
            last_id = cursor.lastrowid
            self.logger.info(f"Successfully inserted memory into '{category_lower}' with ID {last_id}. Text: '{text[:50]}...'. Expires at: {expiry_time.isoformat()}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing memory in '{category_lower}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        except Exception as e:
             self.logger.error(f"Unexpected error storing memory: {e}", exc_info=True)
             if conn:
                 conn.rollback()
             return False
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after store_memory.")


    def get_relevant_memories(self, user_id: str, query: str, category: str, similarity_threshold: float = 0.5) -> List[Dict]:
        category_lower = category.lower()
        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
            self.logger.error(f"Attempted to retrieve memories from invalid category: '{category}'.")
            return []

        self.logger.info(f"Retrieving relevant memories for user '{user_id}' in category '{category_lower}' for query: '{query[:50]}...'")
        self.logger.debug(f"Using similarity threshold: {similarity_threshold}")

        conn = None
        try:
            query_embedding = self.embedding_model.encode(query)
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            # Ensure connection uses the custom converters
            conn.row_factory = sqlite3.Row # Use row factory for dict-like access
            cursor = conn.cursor()

            # Select potentially relevant memories (active and not expired)
            cursor.execute(f'''
            SELECT id, original_text, keywords, embedding, created_at, expiry_at
            FROM {category_lower}
            WHERE user_id = ? AND is_active = 1 AND expiry_at > datetime('now') -- Use SQLite datetime comparison
            ''', (user_id,))

            rows = cursor.fetchall()
            self.logger.debug(f"Fetched {len(rows)} candidate memories from DB for category '{category_lower}'.")

            relevant_memories = []
            for row in rows:
                try:
                    memory_embedding = self.bytes_to_array(row['embedding'])
                    similarity = self.cosine_similarity(query_embedding, memory_embedding)

                    if similarity >= similarity_threshold:
                        self.logger.debug(f"Memory ID {row['id']} meets threshold ({similarity:.4f} >= {similarity_threshold}). Text: '{row['original_text'][:50]}...'")
                        memories.append({
                            'id': row['id'],
                            'text': row['original_text'],
                            'keywords': row['keywords'].split(','), # Assuming comma-separated
                            'similarity': similarity,
                            # Ensure datetime objects are correctly converted if needed
                            'created_at': row['created_at'], # Should be datetime if PARSE_DECLTYPES works
                            'expiry_at': row['expiry_at']   # Should be datetime
                        })
                    else:
                        self.logger.debug(f"Memory ID {row['id']} below threshold ({similarity:.4f} < {similarity_threshold}).")

                except Exception as e: # Catch errors during processing of a single row
                    self.logger.error(f"Error processing memory row ID {row.get('id', 'N/A')}: {e}", exc_info=True)

            # Sort by similarity descending
            relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
            self.logger.info(f"Found {len(relevant_memories)} relevant memories meeting threshold {similarity_threshold} for query '{query[:50]}...' in category '{category_lower}'.")
            return relevant_memories

        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving memories from '{category_lower}': {e}", exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving relevant memories: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after get_relevant_memories.")

    def process_user_query(self, user_id: str, query: str) -> Optional[str]: # Return Optional[str] as response can be None
        """Processes a user query, retrieves context, and generates a response using LLM."""
        self.logger.info(f"Processing user query for user ID '{user_id}': '{query[:50]}...'")

        try:
            query_keywords = self.extract_keywords(query)
            if not query_keywords:
                self.logger.warning("No keywords extracted from query, cannot determine category accurately.")
                determined_category = "personal" # Default if no keywords
                self.logger.info(f"Defaulting to category: '{determined_category}'")
            else:
                determined_category = self.determine_category(query_keywords)
                self.logger.info(f"Determined category based on keywords: '{determined_category}'")

            relevant_memories = self.get_relevant_memories(user_id, query, determined_category)

            # Fallback to 'personal' category if primary category yields no results
            if not relevant_memories and determined_category != 'personal':
                self.logger.warning(f"No relevant memories found in category '{determined_category}'. Falling back to 'personal' category.")
                relevant_memories = self.get_relevant_memories(user_id, query, 'personal')

            if not relevant_memories:
                self.logger.info("No relevant memories found in determined or fallback categories.")
                memory_context = "" # No context available
            else:
                self.logger.info(f"Found {len(relevant_memories)} relevant memories to use as context.")
                # Format context clearly for the LLM
                memory_context = "\n".join([f"- Memory: {memory['text']} (Similarity: {memory['similarity']:.2f})" for memory in relevant_memories])
                self.logger.debug(f"Memory context for LLM:\n{memory_context}")

            # Invoke LLM for response generation ONLY if context was found
            if not memory_context:
                self.logger.info("No memory context available, cannot generate contextual response.")
                # Decide on behavior: return None, a default message, or pass query directly to LLM without context?
                # Returning None seems appropriate if the goal is context-based answers.
                return None
            else:
                self.logger.info("Invoking LLM to generate response using memory context.")
                try:
                    runnable = OllamaRunnable(
                        model_url=os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
                        model_name=self.model_name,
                        # Simple system prompt using the context
                        system_prompt_template="You are a helpful assistant. Use the following relevant memories provided as context to answer the user's query accurately and concisely:\n--- Context Memories ---\n{memory_context}\n--- End Context ---",
                        user_prompt_template="{query}",
                        input_variables=["query", "memory_context"],
                        response_type="text" # Expecting a text response
                    )
                    response = runnable.invoke({"query": query, "memory_context": memory_context})
                    self.logger.info(f"LLM generated response: '{response[:100]}...'")
                    return str(response) # Ensure it's a string
                except Exception as e:
                    self.logger.error(f"Error invoking Ollama for query response generation: {e}", exc_info=True)
                    return "I encountered an error while trying to process your request with memory context." # Provide error feedback

        except Exception as e:
            self.logger.error(f"Unexpected error during process_user_query for user '{user_id}': {e}", exc_info=True)
            return "I encountered an unexpected error while processing your query." # Generic error message


    def cleanup_expired_memories(self):
        """Deletes memories from all category tables where expiry_at is in the past."""
        self.logger.info("Starting cleanup of expired memories...")
        conn = None
        deleted_count_total = 0
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            cursor = conn.cursor()
            # No need for current_time variable, SQLite can compare directly
            for category in self.categories.keys():
                table_name = category.lower()
                self.logger.info(f"Cleaning up expired memories in category: {table_name}...")
                try:
                    # Delete rows where expiry_at is less than the current time
                    cursor.execute(f"DELETE FROM {table_name} WHERE expiry_at < datetime('now')")
                    deleted_count = cursor.rowcount
                    conn.commit() # Commit after each table cleanup
                    deleted_count_total += deleted_count
                    self.logger.info(f"Deleted {deleted_count} expired memories from {table_name}.")
                except sqlite3.Error as e:
                    self.logger.error(f"Database error cleaning up table '{table_name}': {e}", exc_info=True)
                    conn.rollback() # Rollback changes for this table on error
            self.logger.info(f"Expired memory cleanup completed. Total memories deleted: {deleted_count_total}.")
        except sqlite3.Error as e:
            self.logger.error(f"Database error during memory cleanup connection/setup: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error during memory cleanup: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after cleanup.")

    def delete_memory(self, user_id: str, category: str, memory_id: int):
        """Deletes a specific memory by ID, category, and user ID."""
        category_lower = category.lower()
        self.logger.info(f"Attempting to delete memory ID {memory_id} from category '{category_lower}' for user '{user_id}'.")

        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
            self.logger.error(f"Invalid category '{category_lower}' provided for deletion.")
            raise ValueError("Invalid category provided for deletion.")

        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Verify memory exists and belongs to the user before deleting
            cursor.execute(f'SELECT 1 FROM {category_lower} WHERE id = ? AND user_id = ?', (memory_id, user_id))
            result = cursor.fetchone()

            if not result:
                self.logger.warning(f"Memory ID {memory_id} not found in category '{category_lower}' for user '{user_id}'. Cannot delete.")
                conn.close() # Close connection before raising
                raise ValueError("Memory not found or does not belong to the specified user.")

            # Perform the deletion
            cursor.execute(f'DELETE FROM {category_lower} WHERE id = ? AND user_id = ?', (memory_id, user_id))
            conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                self.logger.info(f"Successfully deleted memory ID {memory_id} from category '{category_lower}' for user '{user_id}'.")
            else:
                # Should not happen due to the check above, but log just in case
                 self.logger.warning(f"Delete operation for memory ID {memory_id} affected 0 rows, despite prior check.")

        except sqlite3.Error as e:
            self.logger.error(f"Database error deleting memory ID {memory_id} from '{category_lower}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise # Re-raise database errors
        except ValueError as ve: # Catch the specific ValueError raised above
             raise ve # Re-raise it so caller knows why it failed
        except Exception as e:
             self.logger.error(f"Unexpected error deleting memory: {e}", exc_info=True)
             if conn:
                 conn.rollback()
             raise # Re-raise other unexpected errors
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after delete_memory.")

    def fetch_memories_by_category(self, user_id: str, category: str, limit: int = 50) -> List[Dict]:
        """Fetches active, non-expired memories for a user in a specific category, ordered by creation date."""
        category_lower = category.lower()
        self.logger.info(f"Fetching up to {limit} memories for user '{user_id}' in category '{category_lower}'.")

        if category_lower not in [cat.lower() for cat in self.categories.keys()]:
            self.logger.error(f"Invalid category '{category_lower}' requested for fetching.")
            return []

        conn = None
        memories = []
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            conn.row_factory = sqlite3.Row # Use row factory for dict-like access
            cursor = conn.cursor()

            cursor.execute(f'''
            SELECT id, original_text, keywords, created_at, expiry_at
            FROM {category_lower}
            WHERE user_id = ? AND is_active = 1 AND expiry_at > datetime('now') -- Compare expiry with current time
            ORDER BY created_at DESC -- Order by most recent first
            LIMIT ?
            ''', (user_id, limit))

            rows = cursor.fetchall()
            memories = [
                {
                    'id': row['id'],
                    'original_text': row['original_text'],
                    'keywords': row['keywords'].split(','), # Assuming comma-separated
                    'created_at': row['created_at'], # Should be datetime
                    'expiry_at': row['expiry_at'],   # Should be datetime
                    'category': category_lower  # Add category field consistently
                }
                for row in rows
            ]
            self.logger.info(f"Retrieved {len(memories)} memories for user '{user_id}' in category '{category_lower}'.")
            return memories

        except sqlite3.Error as e:
            self.logger.error(f"Database error fetching memories from '{category_lower}': {e}", exc_info=True)
            return [] # Return empty list on error
        except Exception as e:
             self.logger.error(f"Unexpected error fetching memories: {e}", exc_info=True)
             return []
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after fetch_memories_by_category.")

    def clear_all_memories(self, user_id: str):
        """Deletes ALL memories for a specific user across ALL categories."""
        self.logger.warning(f"Initiating deletion of ALL memories for user ID: '{user_id}'. This is irreversible.")
        conn = None
        deleted_count_total = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for category in self.categories.keys():
                table_name = category.lower()
                self.logger.info(f"Deleting memories for user '{user_id}' from table '{table_name}'...")
                try:
                    cursor.execute(f'DELETE FROM {table_name} WHERE user_id = ?', (user_id,))
                    deleted_count = cursor.rowcount
                    deleted_count_total += deleted_count
                    self.logger.info(f"Deleted {deleted_count} memories from {table_name} for user '{user_id}'.")
                except sqlite3.Error as e:
                    self.logger.error(f"Database error clearing table '{table_name}' for user '{user_id}': {e}", exc_info=True)
                    # Decide whether to continue or stop on error. Let's continue but log.

            conn.commit() # Commit all deletions at the end
            self.logger.warning(f"Successfully deleted a total of {deleted_count_total} memories across all categories for user '{user_id}'.")

        except sqlite3.Error as e:
            self.logger.error(f"Database error during clear_all_memories for user '{user_id}': {e}", exc_info=True)
            if conn:
                conn.rollback() # Rollback if commit hasn't happened
            raise # Re-raise error to indicate failure
        except Exception as e:
             self.logger.error(f"Unexpected error during clear_all_memories: {e}", exc_info=True)
             if conn:
                 conn.rollback()
             raise
        finally:
            if conn:
                conn.close()
                self.logger.debug("Database connection closed after clear_all_memories.")