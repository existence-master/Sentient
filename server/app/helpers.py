from typing import Dict, Any, Optional, Callable, List, Union
import base64
import re
from html import unescape
import importlib
import asyncio

# The following imports and functions related to JSON file handling for chat and notifications
# are no longer needed as the logic has been migrated to MongoDB in BaseContextEngine.
# Therefore, they are removed from this file.
# import os
# import json
# CHAT_DB = "chatsDb.json"
# initial_db = {
#     "chats": [],
#     "active_chat_id": None,
#     "next_chat_id": 1
# }
# async def load_db():
#     """Load the database from chatsDb.json, initializing if it doesn't exist or is invalid."""
#     try:
#         with open(CHAT_DB, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             if "chats" not in data:
#                 data["chats"] = []
#             if "active_chat_id" not in data:
#                 data["active_chat_id"] = None
#             if "next_chat_id" not in data:
#                 data["next_chat_id"] = 1
#             return data
#     except (FileNotFoundError, json.JSONDecodeError):
#         print("DB NOT FOUND! Initializing with default structure.")
#         return initial_db
# async def save_db(data):
#     """Save the data to chatsDb.json."""
#     with open(CHAT_DB, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4)
# NOTIFICATIONS_DB = "notificationsDb.json"
# async def load_notifications_db():
#     """Load the notifications database, initializing it if it doesn't exist."""
#     try:
#         with open(NOTIFICATIONS_DB, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             if "notifications" not in data:
#                 data["notifications"] = []
#             if "next_notification_id" not in data:
#                 data["next_notification_id"] = 1
#             return data
#     except (FileNotFoundError, json.JSONDecodeError):
#         print("Notifications DB NOT FOUND! Initializing with default structure.")
#         return {"notifications": [], "next_notification_id": 1}
# async def save_notifications_db(data):
#     """Save the notifications database."""
#     with open(NOTIFICATIONS_DB, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4)