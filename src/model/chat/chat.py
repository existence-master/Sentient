import os
import uvicorn
import json
import asyncio
from runnables import *
from functions import *
from externals import *
from helpers import *
from prompts import *
import nest_asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, AsyncGenerator
from dotenv import load_dotenv
import time

load_dotenv("../.env")  # Load environment variables from .env file

# --- FastAPI Application ---
app = FastAPI(
    title="Chat API", description="API for chat functionalities",
    docs_url="/docs",
    redoc_url=None
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Bodies ---
class Message(BaseModel):
    """
    Pydantic model for the chat message request body.
    """
    original_input: str
    transformed_input: str
    pricing: str
    credits: int
    chat_id: str

# --- Global Variables and Database Setup ---
db_path = os.path.join(os.path.dirname(__file__), "..", "..", "chatsDb.json")
db_lock = asyncio.Lock()  # Lock for synchronizing database access

def load_db():
    """Load the database from chatsDb.json, initializing with {"messages": []} if it doesn't exist or is invalid."""
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print ("DB NOT FOUND!")
        return {"messages": []}

def save_db(data):
    """Save the data to chatsDb.json."""
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

db_data = load_db()  # Load database into memory at startup
chat_runnable = None  # Global chat runnable, initialized later

# --- Apply nest_asyncio ---
nest_asyncio.apply()

# --- API Endpoints ---
@app.get("/", status_code=200)
async def main() -> Dict[str, str]:
    """Root endpoint of the Chat API."""
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }

@app.post("/initiate", status_code=200)
async def initiate() -> JSONResponse:
    """Endpoint to initiate the Chat API model."""
    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )
    except Exception as e:
        print(f"Error initiating chat: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/get-chat-history", status_code=200)
async def get_chat_history():
    """Retrieve the chat history."""
    async with db_lock:
        return JSONResponse(status_code=200, content={"messages": db_data["messages"]})

@app.post("/clear-chat-history", status_code=200)
async def clear_chat_history():
    """Clear the chat history."""
    async with db_lock:
        db_data["messages"] = []
        save_db(db_data)
    return JSONResponse(status_code=200, content={"message": "Chat history cleared"})

@app.post("/chat", status_code=200)
async def chat(message: Message):
    """Handle chat interactions with streaming responses."""
    global chat_runnable
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            user_db = json.load(f)
        
        chat_history = await get_chat_history()
        chat_runnable = get_chat_runnable(chat_history)

        username = user_db["userData"]["personalInfo"]["name"]
        transformed_input = message.transformed_input
        pricing_plan = message.pricing
        credits = message.credits

        async def response_generator():
            memory_used = False
            agents_used = False
            internet_used = False
            user_context = None
            internet_context = None
            pro_used = False
            note = ""

            # Save user message
            user_msg = {
                "id": str(int(time.time() * 1000)),
                "message": message.original_input,
                "isUser": True,
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False
            }
            async with db_lock:
                db_data["messages"].append(user_msg)
                save_db(db_data)

            yield json.dumps({
                "type": "userMessage",
                "message": message.original_input,
                "memoryUsed": False,
                "agentsUsed": False,
                "internetUsed": False
            }) + "\n"
            await asyncio.sleep(0.05)

            yield json.dumps({"type": "intermediary", "message": "Processing chat response..."}) + "\n"
            await asyncio.sleep(0.05)

            context_classification = await classify_context(transformed_input, "category")
            if "personal" in context_classification["class"]:
                yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                memory_used = True
                user_context = await perform_graphrag(transformed_input)
                
            note = ""
            internet_classification = await classify_context(transformed_input, "internet")
            if pricing_plan == "free" and internet_classification["class"] == "Internet" and credits > 0:
                yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                internet_context = await perform_internet_search(transformed_input)
                internet_used = True
                pro_used = True
            elif pricing_plan != "free" and internet_classification["class"] == "Internet":
                yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                internet_context = await perform_internet_search(transformed_input)
                internet_used = True
                pro_used = True
            else:
                note = "Sorry, internet search is a pro feature and requires credits on the free plan."

            personality = user_db["userData"].get("personality", "None")
            assistant_msg = {
                "id": str(int(time.time() * 1000)),
                "message": "",
                "isUser": False,
                "memoryUsed": memory_used,
                "agentsUsed": agents_used,
                "internetUsed": internet_used
            }
            async with db_lock:
                db_data["messages"].append(assistant_msg)
                save_db(db_data)

            async for token in generate_streaming_response(
                chat_runnable,
                inputs={
                    "query": transformed_input,
                    "user_context": user_context,
                    "internet_context": internet_context,
                    "name": username,
                    "personality": personality
                },
                stream=True
            ):
                if isinstance(token, str):
                    assistant_msg["message"] += token
                    async with db_lock:
                        save_db(db_data)
                    yield json.dumps({
                        "type": "assistantStream",
                        "token": token,
                        "done": False,
                        "messageId": assistant_msg["id"]
                    }) + "\n"
                    await asyncio.sleep(0.05)
                else:
                    # Streaming is done, append the full assistant message
                    if note:
                        assistant_msg["message"] += "\n\n" + note
                    chat_runnable.messages.append({"role": "assistant", "content": assistant_msg["message"]})
                    async with db_lock:
                        save_db(db_data)
                    yield json.dumps({
                        "type": "assistantStream",
                        "token": "\n\n" + note if note else "",
                        "done": True,
                        "memoryUsed": memory_used,
                        "agentsUsed": agents_used,
                        "internetUsed": internet_used,
                        "proUsed": pro_used,
                        "messageId": assistant_msg["id"]
                    }) + "\n"

        return StreamingResponse(response_generator(), media_type="application/json")
    except Exception as e:
        print(f"Error executing chat: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, port=5003)