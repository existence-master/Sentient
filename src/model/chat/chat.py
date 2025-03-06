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
from typing import Dict, AsyncGenerator # Import specific types for clarity
from dotenv import load_dotenv

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
chat_history = None  # Placeholder for chat history object
chat_runnable = None  # Runnable for handling chat conversations

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


@app.post("/chat", status_code=200)
async def chat(message: Message) -> StreamingResponse:
    """
    Endpoint to handle chat messages and generate responses using the AI model.

    This is the main chat endpoint that processes user messages, retrieves context,
    and streams back the response.

    Args:
        message (Message): Request body containing the chat message details.

    Returns:
        StreamingResponse: A streaming response containing different types of messages
                           (user message, intermediary messages, assistant messages).
    """
    global chat_runnable  # Access the global chat_runnable

    try:
        with open(
            "../../userProfileDb.json", "r", encoding="utf-8"
        ) as f:  # Load user profile database
            db = json.load(f)

        chat_history = get_chat_history(
            message.chat_id
        )  # Retrieve chat history for the given chat ID

        chat_runnable = get_chat_runnable(
            chat_history
        )  # Initialize chat runnable with chat history

        username = db["userData"]["personalInfo"][
            "name"
        ]  # Extract username from user profile
        transformed_input = (
            message.transformed_input
        )  # Get transformed input from message

        pricing_plan = message.pricing  # Get pricing plan from message
        credits = message.credits  # Get credits from message

        async def response_generator() -> AsyncGenerator[str, None]:
            """
            Asynchronous generator to produce streaming responses for the chat endpoint.

            Yields:
                str: JSON string representing different types of messages in the chat flow.
            """
            memory_used = False  # Flag to track if memory was used
            agents_used = False  # Flag to track if agents were used (not used in this simplified chat)
            internet_used = False  # Flag to track if internet search was used
            user_context = None  # Placeholder for user context retrieved from memory
            internet_context = None  # Placeholder for internet search results
            pro_used = False  # Flag to track if pro features were used
            note = ""  # Placeholder for notes or messages to the user

            yield (
                json.dumps(
                    {
                        "type": "userMessage",
                        "message": message.original_input,
                        "memoryUsed": False,
                        "agentsUsed": False,
                        "internetUsed": False,
                    }
                )
                + "\n"
            )  # Yield user message
            await asyncio.sleep(0.05)  # Small delay for streaming effect

            yield (
                json.dumps(
                    {"type": "intermediary", "message": "Processing chat response..."}
                )
                + "\n"
            )  # Yield intermediary message
            await asyncio.sleep(0.05)  # Small delay

            context_classification = await classify_context(
                transformed_input, "category"
            )  # Classify context for category
            
            print("Context classification: ", context_classification)
            context_classification_category = context_classification[
                "class"
            ]  # Extract classification category

            if (
                "personal" in context_classification_category
            ):  # If context is personal, retrieve memory
                yield (
                    json.dumps(
                        {"type": "intermediary", "message": "Retrieving memories..."}
                    )
                    + "\n"
                )  # Yield intermediary message
                await asyncio.sleep(0.05)  # Small delay
                memory_used = True  # Mark memory as used
                user_context = await perform_graphrag(
                    transformed_input
                )  # Perform graph-based RAG for memory retrieval
            else:
                user_context = None  # No user context needed

            internet_classification = await classify_context(
                transformed_input, "internet"
            )  # Classify context for internet
            internet_classification_internet = internet_classification[
                "class"
            ]  # Extract internet classification
            
            print("Internet classification: ", internet_classification_internet)

            if pricing_plan == "free":  # Free plan logic
                if (
                    internet_classification_internet == "Internet"
                ):  # If internet search is relevant
                    if credits > 0:  # Check for credits
                        yield (
                            json.dumps(
                                {
                                    "type": "intermediary",
                                    "message": "Searching the internet...",
                                }
                            )
                            + "\n"
                        )  # Yield intermediary message
                        internet_context = await perform_internet_search(
                            transformed_input
                        )  # Perform internet search
                        internet_used = True  # Mark internet as used
                        pro_used = True  # Mark pro feature as used
                    else:
                        note = "Sorry friend, could have searched the internet for this query for more context. But, that is a pro feature and your daily credits have expired. You can always upgrade to pro from the settings page"  # Note for free users without credits
                else:
                    internet_context = None  # No internet context needed for free plan

            else:  # Pro plan logic
                if (
                    internet_classification_internet == "Internet"
                ):  # If internet search is relevant
                    yield (
                        json.dumps(
                            {
                                "type": "intermediary",
                                "message": "Searching the internet...",
                            }
                        )
                        + "\n"
                    )  # Yield intermediary message
                    internet_context = await perform_internet_search(
                        transformed_input
                    )  # Perform internet search
                    internet_used = True  # Mark internet as used
                    pro_used = True  # Mark pro feature as used
                else:
                    internet_context = None  # No internet context needed for pro plan

            personality_description = db["userData"].get(
                "personality", "None"
            )  # Extract personality description from user profile

            try:
                async for (
                    token
                ) in generate_streaming_response(  # Stream response from chat runnable
                    chat_runnable,
                    inputs={  # Input parameters for chat runnable
                        "query": transformed_input,
                        "user_context": user_context,
                        "internet_context": internet_context,
                        "name": username,
                        "personality": personality_description,
                    },
                    stream=True,  # Enable streaming response
                ):
                    if isinstance(token, str):  # Yield assistant stream tokens
                        yield (
                            json.dumps(
                                {
                                    "type": "assistantStream",
                                    "token": token,
                                    "done": False,
                                }
                            )
                            + "\n"
                        )
                        await asyncio.sleep(0.05)  # Small delay
                    else:  # Yield final assistant stream message with metadata
                        yield (
                            json.dumps(
                                {
                                    "type": "assistantStream",
                                    "token": "\n\n" + note,
                                    "done": True,
                                    "memoryUsed": memory_used,
                                    "agentsUsed": agents_used,
                                    "internetUsed": internet_used,
                                    "proUsed": pro_used,
                                }
                            )
                            + "\n"
                        )
            except Exception as e:  # Handle exceptions during chat generation
                print(f"Error executing chat in chat: {e}")
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Generation error: {str(e)}",  # Error message for generation failure
                        }
                    )
                    + "\n"
                )

        return StreamingResponse(
            response_generator(), media_type="application/json"
        )  # Return streaming response

    except Exception as e:  # Handle exceptions during chat processing
        print(f"Error executing chat in chat: {e}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return JSON error response


if __name__ == "__main__":
    uvicorn.run(app, port=5003)  # Run the FastAPI application using Uvicorn server