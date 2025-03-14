import httpx
import os
from helpers import *
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

async def classify_context(query: str, context: str) -> Dict[str, Any]:
    """
    Asynchronously calls the /context-classify endpoint to classify a query.

    This function sends a POST request to the context classification service
    to determine the context of the given query.

    Args:
        query (str): The query string to be classified.
        context (str): The context in which the query is being made (e.g., "category", "internet").

    Returns:
        Dict[str, Any]: A dictionary containing the classification result.
                         If successful, it returns a dictionary with the 'classification' key.
                         If an error occurs, it returns a dictionary with an 'error' key
                         containing the error message.
    """
    try:
        port: Optional[str] = os.environ.get(
            "APP_SERVER_PORT"
        )  # Get the port from environment variables
        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create an async HTTP client with no timeout
            response = await client.post(
                f"http://localhost:{port}/context-classify",
                json={"query": query, "context": context},
            )  # Send POST request to classify context

        if response.status_code == 200:  # Check if the request was successful
            return response.json()[
                "classification"
            ]  # Return the classification from the response
        else:
            return {
                "error": response.text
            }  # Return an error dictionary with the response text

    except Exception as e:  # Catch any exceptions during the process
        print(f"Error calling classify_context: {e}")
        return {
            "error": f"Error calling context-classify: {str(e)}"
        }  # Return an error dictionary with the exception message


async def perform_internet_search(query: str) -> Dict[str, Any]:
    """
    Asynchronously calls the /internet-search endpoint to fetch search results and summarize them.

    This function sends a POST request to the internet search service to
    retrieve and summarize internet search results for the given query.

    Args:
        query (str): The query string to be used for internet search.

    Returns:
        Dict[str, Any]: A dictionary containing the internet search context.
                         If successful, it returns a dictionary with the 'internet_context' key.
                         If an error occurs, it returns a dictionary with an 'error' key
                         containing the error message.
    """
    try:
        port: Optional[str] = os.environ.get(
            "APP_SERVER_PORT"
        )  # Get the port from environment variables
        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create an async HTTP client with no timeout
            response = await client.post(
                f"http://localhost:{port}/internet-search", json={"query": query}
            )  # Send POST request for internet search

        if response.status_code == 200:  # Check if the request was successful
            return response.json()[
                "internet_context"
            ]  # Return the internet context from the response
        else:
            return {
                "error": response.text
            }  # Return an error dictionary with the response text

    except Exception as e:  # Catch any exceptions during the process
        print(f"Error performing internet search: {e}")
        return {
            "error": f"Error calling internet-search: {str(e)}"
        }  # Return an error dictionary with the exception message


async def perform_graphrag(query: str) -> Dict[str, Any]:
    """
    Asynchronously calls the /graphrag endpoint to query the user profile and get graphrag context.

    This function sends a POST request to the graphrag service to query the user profile
    and retrieve relevant context using graph-based retrieval-augmented generation (RAG).

    Args:
        query (str): The query string to be used for graph-based RAG.

    Returns:
        Dict[str, Any]: A dictionary containing the graphrag context.
                         If successful, it returns a dictionary with the 'context' key.
                         If an error occurs, it returns a dictionary with an 'error' key
                         containing the error message.
    """
    try:
        port: Optional[str] = os.environ.get(
            "APP_SERVER_PORT"
        )  # Get the port from environment variables
        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create an async HTTP client with no timeout
            response = await client.post(
                f"http://localhost:{port}/graphrag", json={"query": query}
            )  # Send POST request for graphrag

        if response.status_code == 200:  # Check if the request was successful
            return response.json()["context"]  # Return the context from the response
        else:
            return {
                "error": response.text
            }  # Return an error dictionary with the response text

    except Exception as e:  # Catch any exceptions during the process
        print(f"Error performing graphrag: {e}")
        return {
            "error": f"Error calling graphrag: {str(e)}"
        }  # Return an error dictionary with the exception message
