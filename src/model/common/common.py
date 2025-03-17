import os
import uvicorn
from runnables import *  # Importing runnables (likely Langchain Runnable sequences) from runnables.py
from functions import *  # Importing custom functions from functions.py
from helpers import *  # Importing helper functions from helpers.py
from prompts import *  # Importing prompt templates from prompts.py
import nest_asyncio  # For running asyncio event loop within another event loop (needed for FastAPI in some environments)
from fastapi import FastAPI  # Importing FastAPI for creating the API application
from pydantic import (
    BaseModel,
)  # Importing BaseModel from Pydantic for request body validation and data modeling
from fastapi.responses import (
    JSONResponse,
)  # Importing JSONResponse for sending JSON responses from API endpoints
from fastapi.middleware.cors import (
    CORSMiddleware,
)  # Importing CORSMiddleware to handle Cross-Origin Resource Sharing
from fastapi import FastAPI  # Re-importing FastAPI (likely a typo and redundant)
from pydantic import BaseModel  # Re-importing BaseModel (likely a typo and redundant)
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

# --- FastAPI Application Initialization ---
app = FastAPI(
    docs_url="/docs", 
    redoc_url=None
    )  # Creating a FastAPI application instance

# --- CORS Middleware Configuration ---
# Configuring CORS to allow requests from any origin.
# This is generally fine for open-source projects or APIs intended for broad use, but be cautious in production environments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows credentials (cookies, authorization headers) to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers to be included in requests
)


# --- Pydantic Models for Request Body Validation ---
class InternetSearchRequest(BaseModel):
    """
    Pydantic model for validating internet search requests.
    Requires a 'query' string field.
    """

    query: str  # User's search query string


class ContextClassificationRequest(BaseModel):
    """
    Pydantic model for validating context classification requests.
    Requires a 'query' string and a 'context' string field.
    """

    query: str  # User's query string
    context: str  # Context for classification (e.g., "category", "internet")


class ChatClassificationRequest(BaseModel):
    """
    Pydantic model for validating chat classification requests.
    Requires an 'input' string and a 'chat_id' string field.
    """

    input: str  # User's chat input string
    chat_id: str  # Identifier for the chat session


# --- Global Variables for Runnables and Chat History ---
# These global variables store initialized Langchain Runnable sequences and chat history.
# Initialized in the `/initiate` endpoint.
chat_history = (
    None  # Stores chat history, likely as a list of messages or a database connection
)
chat_runnable = None  # Runnable for general chat interactions (not currently used in the provided endpoints)
orchestrator_runnable = (
    None  # Runnable for orchestrating different tasks based on user input
)
context_classification_runnable = (
    None  # Runnable for classifying the context of a query (e.g., category)
)
internet_search_runnable = None  # Runnable for performing internet searches
internet_query_reframe_runnable = None  # Runnable for reframing internet search queries
internet_summary_runnable = None  # Runnable for summarizing internet search results
orchestrator_runnable = None  # Runnable for orchestrating tasks (redundant declaration)
internet_classification_runnable = (
    None  # Runnable for classifying if a query requires internet search
)

# --- Apply nest_asyncio ---
# Applying nest_asyncio to allow asyncio event loops to be nested.
# This is often needed when running FastAPI applications in environments that may already have an event loop running,
# such as Jupyter notebooks or certain testing frameworks.
nest_asyncio.apply()


# --- API Endpoints ---
@app.get("/")
async def main():
    """
    Root endpoint of the API.
    Returns a simple welcome message.
    """
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }


@app.post("/initiate", status_code=200)
async def initiate():
    """
    Endpoint to initialize the AI model and runnables.
    This endpoint sets up the global runnable variables required for other API calls.
    Returns a success message or an error message if initialization fails.
    """
    global \
        chat_history, \
        context_classification_runnable, \
        internet_search_runnable, \
        internet_query_reframe_runnable, \
        internet_summary_runnable, \
        internet_classification_runnable, \
        orchestrator_runnable

    # Initialize different runnables using functions from runnables.py
    context_classification_runnable = (
        get_context_classification_runnable()
    )  # Get runnable for context classification
    internet_classification_runnable = (
        get_internet_classification_runnable()
    )  # Get runnable for internet classification
    internet_search_runnable = get_internet_classification_runnable()  # Get runnable for internet search (typo in original code, should likely be `get_internet_search_runnable()`)
    internet_query_reframe_runnable = (
        get_internet_query_reframe_runnable()
    )  # Get runnable for reframing internet queries
    internet_summary_runnable = (
        get_internet_summary_runnable()
    )  # Get runnable for summarizing internet search results

    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )  # Return success message
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/chat-classify", status_code=200)
async def chat_classify(request: ChatClassificationRequest):
    """
    Endpoint to classify user chat input and determine the appropriate response strategy.
    Uses an orchestrator runnable to classify the input and potentially transform it.
    Returns the classification and transformed input.
    """
    try:
        global orchestrator_runnable, chat_history

        chat_history = get_chat_history()
        orchestrator_runnable = get_orchestrator_runnable(
            chat_history
        )  # Initialize orchestrator runnable with chat history
        orchestrator_output = orchestrator_runnable.invoke(
            {"query": request.input}
        )  # Invoke orchestrator runnable with user input
        classification = orchestrator_output[
            "class"
        ]  # Extract classification from orchestrator output
        transformed_input = orchestrator_output[
            "input"
        ]  # Extract transformed input from orchestrator output

        return JSONResponse(
            status_code=200,
            content={
                "classification": classification,
                "transformed_input": transformed_input,
            },
        )  # Return classification and transformed input
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/context-classify", status_code=200)
async def context_classify(request: ContextClassificationRequest):
    """
    Endpoint to classify user query based on the provided context.
    Uses different classification runnables based on the context.
    Returns the classification result.
    """
    try:
        global context_classification_runnable, internet_classification_runnable

        if request.context == "category":  # Classify based on category context
            classification = context_classification_runnable.invoke(
                {"query": request.query}
            )  # Invoke context classification runnable
        else:  # Classify based on internet context (or default to internet classification if context is not "category")
            classification = internet_classification_runnable.invoke(
                {"query": request.query}
            )  # Invoke internet classification runnable

        return JSONResponse(
            status_code=200, content={"classification": classification}
        )  # Return classification result
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/internet-search", status_code=200)
async def internet_search(request: InternetSearchRequest):
    """
    Endpoint to perform an internet search based on the user query.
    Reframes the query, fetches search results, and summarizes them to provide internet context.
    Returns the summarized internet context.
    """
    try:
        reframed_query = get_reframed_internet_query(
            internet_query_reframe_runnable, request.query
        )  # Reframe the user query for better internet search
        search_results = get_search_results(
            reframed_query
        )  # Fetch search results using the reframed query
        internet_context = get_search_summary(
            internet_summary_runnable, search_results
        )  # Summarize the search results to get internet context

        return JSONResponse(
            status_code=200, content={"internet_context": internet_context}
        )  # Return the summarized internet context

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error performing internet search: {str(e)}"},
        )  # Return error message if exception occurs


# --- Main execution block ---
if __name__ == "__main__":
    """
    This block is executed when the script is run directly (not imported as a module).
    It starts the uvicorn server to run the FastAPI application.
    """
    uvicorn.run(app, port=5006)  # Start uvicorn server on port 5006
