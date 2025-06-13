import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
LLM_CONFIG = {
    'model': os.getenv("QWEN_MODEL", 'qwen3:4b'),
    'model_server': os.getenv("OLLAMA_BASE_URL", 'http://localhost:11434/v1/'),
    'api_key': 'EMPTY',
}

# --- Database Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ai_companion_memory"

# --- Memory System Configuration ---
MEMORY_CATEGORIES = [
    "Personal", "Professional", "Social", "Financial", 
    "Health", "Preferences", "Events", "General"
]

# --- Semantic Search Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "./faiss_indexes"

# --- Security Configuration ---
ALLOWED_MONGO_OPERATIONS = ['find', 'insert_one', 'update_one', 'delete_one']
ALLOWED_COLLECTIONS = ['long_term_memories', 'short_term_memories', 'contacts']

# --- MCP Server Configuration ---
MCP_SERVER_HOST = "0.0.0.0"
MCP_SERVER_PORT = 8000
