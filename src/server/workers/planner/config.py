import os
from dotenv import load_dotenv
from datetime import datetime

import logging

# --- Environment Loading Logic ---
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
logging.info(f"[PlannerConfig] Initializing configuration for ENVIRONMENT='{ENVIRONMENT}'")

server_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if ENVIRONMENT == 'dev-local':
    # Prefer .env.local, fall back to .env
    dotenv_local_path = os.path.join(server_root, '.env.local')
    dotenv_path = os.path.join(server_root, '.env')
    load_path = dotenv_local_path if os.path.exists(dotenv_local_path) else dotenv_path
    logging.info(f"[{datetime.now()}] [PlannerConfig] Loading .env file for 'dev-local' mode from: {load_path}")
    if os.path.exists(load_path):
        load_dotenv(dotenv_path=load_path)
elif ENVIRONMENT == 'selfhost':
    dotenv_path = os.path.join(server_root, '.env.selfhost')
    logging.info(f"[PlannerConfig] Loading .env file for 'selfhost' mode from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    logging.info(f"[PlannerConfig] Skipping dotenv loading for '{ENVIRONMENT}' mode.")
# OpenAI API Standard Configuration
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1/")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "qwen3:4b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(',')
ACTION_ITEMS_TOPIC = os.getenv("ACTION_ITEMS_TOPIC", "action_items")
KAFKA_CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP_ID", "planner_worker_group")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_db")

# This is a replication of the logic from main/config.py to make the worker self-contained.
INTEGRATIONS_CONFIG = {
    "github": {
        "display_name": "GitHub",
        "description": "Connect to manage repositories, issues, and more. Enables the agent to search public and private repos, list your repos, view repository details, list issues, create new issues, and read file contents.",
        "auth_type": "oauth",
        "icon": "IconBrandGithub",
        "mcp_server_config": {
            "name": "github_server",
            "url": os.getenv("GITHUB_MCP_SERVER_URL", "http://localhost:9010/sse")
        }
    },
    "gdrive": { # User-configurable OAuth
        "display_name": "Google Drive",
        "description": "Access and manage files in your Google Drive. Allows the agent to search for files by name or content and read the contents of various file types like documents and spreadsheets.",
        "auth_type": "oauth",
        "icon": "IconBrandGoogleDrive",
        "mcp_server_config": {
            "name": "gdrive_server",
            "url": os.getenv("GDRIVE_MCP_SERVER_URL", "http://localhost:9003/sse")
        }
    },
    "gcalendar": { # User-configurable OAuth
        "display_name": "Google Calendar",
        "description": "Read and manage events on your Google Calendar. Enables the agent to list upcoming events, add new events, search for specific events, update event details, and delete events.",
        "auth_type": "oauth",
        "icon": "IconCalendarEvent",
        "mcp_server_config": {
            "name": "gcal_server",
            "url": os.getenv("GCAL_MCP_SERVER_URL", "http://localhost:9002/sse")
        }
    },
    "gmail": { # User-configurable OAuth
        "display_name": "Gmail",
        "description": "Read, send, and manage emails. The agent can search your inbox, send new emails, create drafts, reply to threads, forward messages, and manage emails by deleting them or marking them as read/unread.",
        "auth_type": "oauth",
        "icon": "IconMail",
        "mcp_server_config": {
            "name": "gmail_server",
            "url": os.getenv("GMAIL_MCP_SERVER_URL", "http://localhost:9001/sse")
        }
    },
    "gdocs": {
        "display_name": "Google Docs",
        "description": "Create and manage documents in Google Docs. Allows the agent to generate new, multi-section documents with titles, headings, paragraphs, and bullet points.",
        "auth_type": "oauth",
        "icon": "IconFileText",
        "mcp_server_config": {
            "name": "gdocs_server",
            "url": os.getenv("GDOCS_MCP_SERVER_URL", "http://localhost:9004/sse")
        }
    },
    "gslides": {
        "display_name": "Google Slides",
        "description": "Create and manage presentations in Google Slides. The agent can build new slide decks with titles, content, images, and charts based on a structured outline you provide.",
        "auth_type": "oauth",
        "icon": "IconPresentation",
        "mcp_server_config": {
            "name": "gslides_server",
            "url": os.getenv("GSLIDES_MCP_SERVER_URL", "http://localhost:9014/sse")
        }
    },
    "gsheets": {
        "display_name": "Google Sheets",
        "description": "Create and manage spreadsheets in Google Sheets. The agent can help organize data by creating new spreadsheets with one or more sheets, including headers and rows.",
        "auth_type": "oauth",
        "icon": "IconTable",
        "mcp_server_config": {
            "name": "gsheets_server",
            "url": os.getenv("GSHEETS_MCP_SERVER_URL", "http://localhost:9015/sse")
        }
    },
    "gpeople": {
        "display_name": "Google People",
        "description": "Manage your contacts. The agent can search, create, update, and delete contacts in your Google account, helping you keep your address book organized.",
        "auth_type": "oauth",
        "icon": "IconUsers",
        "mcp_server_config": {
            "name": "gpeople_server",
            "url": os.getenv("GPEOPLE_MCP_SERVER_URL", "http://localhost:9019/sse")
        }
    },
    "gmaps": {
        "display_name": "Google Maps",
        "description": "Search for places and get directions. The agent can look up addresses, points of interest, and find routes for driving, walking, bicycling, or transit.",
        "auth_type": "builtin",
        "icon": "IconMapPin",
        "mcp_server_config": {
            "name": "gmaps_server",
            "url": os.getenv("GMAPS_MCP_SERVER_URL", "http://localhost:9016/sse")
        }
    },
    "slack": { # User-configurable Manual
        "display_name": "Slack",
        "description": "Connect to your Slack workspace. Allows the agent to list channels, post messages, reply in threads, add reactions, read channel history, and get user information.",
        "auth_type": "manual",
        "icon": "IconBrandSlack",
        "mcp_server_config": {
            "name": "slack_server",
            "url": os.getenv("SLACK_MCP_SERVER_URL", "http://localhost:9006/sse")
        }
    },
    "notion": {
        "display_name": "Notion",
        "description": "Connect to your Notion workspace. The agent can search for pages and databases, read page content, create new pages, append content to existing pages, and query databases with filters.",
        "auth_type": "manual",
        "icon": "IconBrandNotion",
        "mcp_server_config": {
            "name": "notion_server",
            "url": os.getenv("NOTION_MCP_SERVER_URL", "http://localhost:9009/sse")
        }
    },
    "news": { # Built-in
        "display_name": "News",
        "description": "Fetches top headlines and news articles from around the world. The agent can get top headlines by country or category, or search for articles on any topic.",
        "auth_type": "builtin",
        "icon": "IconNews",
        "mcp_server_config": {
            "name": "news_server",
            "url": os.getenv("NEWS_MCP_SERVER_URL", "http://localhost:9012/sse")
        }
    },
    "internet_search": { # Built-in
        "display_name": "Internet Search",
        "description": "Allows the agent to search the web using Google Search to find real-time, factual information on any topic.",
        "auth_type": "builtin",
        "icon": "IconWorldSearch",
        "mcp_server_config": {
            "name": "google_search",
            "url": os.getenv("GOOGLE_SEARCH_MCP_SERVER_URL", "http://localhost:9005/sse")
        }
    },
    "accuweather": { # Built-in
        "display_name": "AccuWeather",
        "description": "Provides current weather conditions and daily forecasts. The agent can get the current weather for any location or a forecast for the next 1-5 days.",
        "auth_type": "builtin",
        "icon": "IconCloud", # Frontend needs to map this string to the icon component
        "mcp_server_config": {
            "name": "weather_server",
            "url": os.getenv("ACCUWEATHER_MCP_SERVER_URL", "http://localhost:9007/sse")
        }
    },
    "quickchart": { # Built-in
        "display_name": "QuickChart",
        "description": "Generates charts and data visualizations on the fly. The agent can create bar charts, line charts, pie charts, and more, then provide a URL or download the image.",
        "auth_type": "builtin",
        "icon": "IconChartPie", # Frontend needs to map this
        "mcp_server_config": {
            "name": "quickchart_server",
            "url": os.getenv("QUICKCHART_MCP_SERVER_URL", "http://localhost:9008/sse")
        }
    },
    "memory": {
        "display_name": "Memory",
        "description": "Manages the user's memory. Use 'search_memory' to find facts, and 'cud_memory' to add, update, or delete information. This is critical for personalization.",
        "auth_type": "builtin",
        "icon": "IconBrain",
        "mcp_server_config": {
            "name": "memory_mcp",
            "url": os.getenv("MEMORY_MCP_SERVER_URL", "http://localhost:8001/sse")
        }
    },
    "file_management": {
        "display_name": "File Management",
        "description": "Read and write files to a temporary storage area. Useful for handling uploads, generating files for download, and data analysis.",
        "auth_type": "builtin",
        "icon": "IconFile",
        "mcp_server_config": {
            "name": "file_management_server",
            "url": os.getenv("FILE_MANAGEMENT_MCP_SERVER_URL", "http://localhost:9026/sse")
        }
    },
    "whatsapp": {
        "display_name": "WhatsApp",
        "description": "Connect a WhatsApp number to allow your agent to send messages on your behalf as a tool. This is different from your notification number.",
        "auth_type": "manual",
        "icon": "IconBrandWhatsapp",
        "mcp_server_config": {
            "name": "whatsapp_server",
            "url": os.getenv("WHATSAPP_MCP_SERVER_URL", "http://localhost:9024/sse")
        }
    },
    "outlook": {
        "display_name": "Outlook",
        "description": "Connect to read, send, and manage emails in Outlook. The agent can list emails, read message content, send new emails, reply to messages, and manage folders.",
        "auth_type": "oauth",
        "icon": "IconMail",
        "mcp_server_config": {
            "name": "outlook_server",
            "url": os.getenv("OUTLOOK_MCP_SERVER_URL", "http://localhost:9027/sse")
        }
    }
}


logging.info(f"Planner Worker configured. Input Topic: {ACTION_ITEMS_TOPIC}")