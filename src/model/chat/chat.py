import os
import uvicorn
import json
import asyncio  # Import asyncio for asynchronous operations
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
from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Dict, Any, AsyncGenerator  # Import specific types for clarity
from dotenv import load_dotenv
from lowdb import LowDB, JSONFile


load_dotenv("../.env")  # Load environment variables from .env file

# --- FastAPI Application ---
app = FastAPI(
    title="Chat API", description="API for chat functionalities",
    docs_url="/docs", 
    redoc_url=None
)  # Initialize FastAPI application

# --- CORS Middleware ---
# Configure CORS to allow cross-origin requests.
# In a production environment, you should restrict the `allow_origins` to specific domains for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - configure this for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods - configure this for production
    allow_headers=["*"],  # Allows all headers - configure this for production
)

# --- Pydantic Models for Request Bodies ---


class Message(BaseModel):
    """
    Pydantic model for the chat message request body.

    Attributes:
        original_input (str): The original user input message.
        transformed_input (str): The transformed user input message, potentially after preprocessing.
        pricing (str): The pricing plan of the user (e.g., "free", "pro").
        credits (int): The number of credits the user has.
        chat_id (str): Unique identifier for the chat session.
    """

    original_input: str
    transformed_input: str
    pricing: str
    credits: int
    chat_id: str


# --- Global Variables ---
# These global variables hold the runnables and chat history for the chat functionality.
# It is initialized to None and will be set in the `/chat` endpoint.
db_path = os.path.join(os.path.dirname(__file__), "chatsDb.json")
db = LowDB(JSONFile(db_path))
db.data = db.data or {"messages": []}
chat_runnable = None

# --- Apply nest_asyncio ---
# nest_asyncio is used to allow asyncio.run() to be called from within a jupyter notebook or another async environment.
# It's needed here because uvicorn runs in an asyncio event loop, and we might need to run async functions within the API endpoints.
nest_asyncio.apply()

# --- API Endpoints ---


@app.get("/", status_code=200)
async def main() -> Dict[str, str]:
    """
    Root endpoint of the Chat API.

    Returns:
        JSONResponse: A simple greeting message.
    """
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }


@app.post("/initiate", status_code=200)
async def initiate() -> JSONResponse:
    """
    Endpoint to initiate the Chat API model.
    Currently, it only returns a success message as there's no specific initialization needed for this API beyond startup.

    Returns:
        JSONResponse: Success or error message in JSON format.
    """
    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )
    except Exception as e:
        print(f"Error initiating chat: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/get-chat-history", status_code=200)
async def get_chat_history():
    return JSONResponse(status_code=200, content={"messages": db.data["messages"]})

@app.post("/clear-chat-history", status_code=200)
async def clear_chat_history():
    db.data["messages"] = []
    db.write()
    return JSONResponse(status_code=200, content={"message": "Chat history cleared"})

@app.post("/chat", status_code=200)
async def chat(message: Message):
    global chat_runnable
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            user_db = json.load(f)

        if not chat_runnable:
            chat_runnable = get_chat_runnable(db.data["messages"])

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
            user_msg = {"id": str(int(time.time() * 1000)), "message": message.original_input, "isUser": True, "memoryUsed": False, "agentsUsed": False, "internetUsed": False}
            db.data["messages"].append(user_msg)
            db.write()

            yield json.dumps({"type": "userMessage", "message": message.original_input, "memoryUsed": False, "agentsUsed": False, "internetUsed": False}) + "\n"
            await asyncio.sleep(0.05)

            yield json.dumps({"type": "intermediary", "message": "Processing chat response..."}) + "\n"
            await asyncio.sleep(0.05)

            context_classification = await classify_context(transformed_input, "category")
            if "personal" in context_classification["class"]:
                yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                memory_used = True
                user_context = await perform_graphrag(transformed_input)

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
            assistant_msg = {"id": str(int(time.time() * 1000)), "message": "", "isUser": False, "memoryUsed": memory_used, "agentsUsed": agents_used, "internetUsed": internet_used}
            db.data["messages"].append(assistant_msg)
            db.write()

            async for token in generate_streaming_response(chat_runnable, inputs={"query": transformed_input, "user_context": user_context, "internet_context": internet_context, "name": username, "personality": personality}, stream=True):
                if isinstance(token, str):
                    assistant_msg["message"] += token
                    db.write()
                    yield json.dumps({"type": "assistantStream", "token": token, "done": False, "messageId": assistant_msg["id"]}) + "\n"
                    await asyncio.sleep(0.05)
                else:
                    assistant_msg["message"] += "\n\n" + note
                    db.write()
                    yield json.dumps({"type": "assistantStream", "token": "\n\n" + note, "done": True, "memoryUsed": memory_used, "agentsUsed": agents_used, "internetUsed": internet_used, "proUsed": pro_used, "messageId": assistant_msg["id"]}) + "\n"

        return StreamingResponse(response_generator(), media_type="application/json")
    except Exception as e:
        print(f"Error executing chat: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, port=5003)  # Run the FastAPI application using Uvicorn server