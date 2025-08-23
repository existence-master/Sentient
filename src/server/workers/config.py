import os
from dotenv import load_dotenv


# Load .env file for 'dev-local' environment.
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_local_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
    if os.path.exists(dotenv_local_path):
        load_dotenv(dotenv_path=dotenv_local_path)
MEMORY_MCP_SERVER_URL = os.getenv("MEMORY_MCP_SERVER_URL", "http://localhost:8001/sse")
