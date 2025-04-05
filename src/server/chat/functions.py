import os
from wrapt_timeout_decorator import *
import json
import requests
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv

from .prompts import *
from server.app.helpers import *

load_dotenv("server/.env")  # Load environment variables from .env file

async def generate_streaming_response(
    runnable, inputs: Dict[str, Any], stream: bool = False
) -> AsyncGenerator[Any, None]:
    """
    Generic function to generate a streaming response from any runnable.

    This function abstracts the process of invoking a runnable and handling its response,
    whether it's a standard response or a streaming one. It checks if the runnable supports
    streaming and calls the appropriate method.

    Args:
        runnable: The runnable object (e.g., chain, agent runnable) to invoke.
        inputs (Dict[str, Any]): Input dictionary for the runnable.
        stream (bool): If True, attempt to generate a streaming response if supported by the runnable (default is False).

    Yields:
        AsyncGenerator[Any, None]: Asynchronously yields tokens or the full response depending on streaming.
                                  Yields None if an error occurs during response generation.
    """
    try:
        if stream and hasattr(
            runnable, "stream_response"
        ):  # Check if streaming is enabled and runnable supports it
            for token in await asyncio.to_thread(lambda: runnable.stream_response(inputs)):
                yield token  # Yield each token from the streaming response
        else:  # Handle non-streaming response
            response = runnable.invoke(
                inputs
            )  # Invoke the runnable to get a standard response
            yield response  # Yield the complete response

    except Exception as e:  # Catch any exceptions during response generation
        print(f"An error occurred: {e}")
        yield None  # Yield None to indicate an error occurred


def generate_response(
    runnable,
    message: str,
    user_context: Optional[str],
    internet_context: Optional[str],
    username: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a response using the provided runnable, incorporating user and internet context.

    This function retrieves user personality description from a user profile database,
    and invokes the given runnable with the message, context information, username, and personality.

    Args:
        runnable: The runnable object (e.g., agent runnable) to use for response generation.
        message (str): The user's input message.
        user_context (Optional[str]): Context retrieved from user's personal memory or graph (optional).
        internet_context (Optional[str]): Context retrieved from internet search (optional).
        username (str): The username of the user.

    Returns:
        Optional[Dict[str, Any]]: The response generated by the runnable, or None if an error occurs.
    """
    try:
        with open(
            "userProfileDb.json", "r", encoding="utf-8"
        ) as f:  # Open and load user profile database
            db = json.load(f)  # Load user profile data from JSON file

        personality_description: str = db["userData"].get(
            "personality", "None"
        )  # Extract personality description from database, default to "None"

        response = runnable.invoke(
            {
                "query": message,
                "user_context": user_context,
                "internet_context": internet_context,
                "name": username,
                "personality": personality_description,
            }
        )  # Invoke runnable with all inputs

        return response  # Return the generated response
    except Exception as e:  # Catch any exceptions during response generation
        print(f"An error occurred in generating response: {e}")
        return None  # Return None to indicate an error occurred


def get_reframed_internet_query(internet_query_reframe_runnable, input: str) -> str:
    """
    Reframes the internet query using the provided runnable.

    This function takes a user input and uses a runnable, specifically designed
    for reframing internet queries, to generate a more effective search query.

    Args:
        internet_query_reframe_runnable: The runnable object designed for reframing internet queries.
        input (str): The original user input to be reframed into an internet search query.

    Returns:
        str: The reframed internet search query.
    """
    reframed_query: str = internet_query_reframe_runnable.invoke(
        {"query": input}
    )  # Invoke the query reframe runnable
    return reframed_query  # Return the reframed query


def get_search_results(reframed_query: str) -> List[Dict[str, Optional[str]]]:
    """
    Fetch and clean descriptions from a web search API based on the provided query.

    This function uses the Brave Search API to fetch web search results for a given query.
    It extracts titles, URLs, and descriptions from the API response and cleans the descriptions
    to remove HTML tags and unescape HTML entities.

    Args:
        reframed_query (str): The search query string to be used for fetching web search results.

    Returns:
        List[Dict[str, Optional[str]]]: A list of dictionaries, each containing the 'title', 'url', and 'description'
                                         of a search result. Returns an empty list if there's an error or no results.
    """
    try:
        params: Dict[str, str] = {  # Parameters for the Brave Search API request
            "q": reframed_query,  # The search query
        }

        headers: Dict[str, str] = {  # Headers for the Brave Search API request
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": os.getenv(
                "BRAVE_SUBSCRIPTION_TOKEN"
            ),  # API token for Brave Search
        }

        response: requests.Response = requests.get(
            os.getenv("BRAVE_BASE_URL"), headers=headers, params=params
        )  # Send GET request to Brave Search API

        if response.status_code == 200:  # Check if the API request was successful
            results = response.json()  # Parse JSON response

            descriptions: List[
                Dict[str, Optional[str]]
            ] = []  # Initialize list to store descriptions
            for item in results.get("web", {}).get("results", [])[
                :5
            ]:  # Iterate through the top 5 web search results
                descriptions.append(
                    {  # Append extracted and raw data to descriptions list
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "description": item.get("description"),
                    }
                )

            clean_descriptions: List[
                Dict[str, Optional[str]]
            ] = [  # Clean descriptions to remove html tags and unescape html characters
                {
                    "title": entry["title"],
                    "url": entry["url"],
                    "description": clean_description(
                        entry["description"]
                    ),  # Clean the description text
                }
                for entry in descriptions  # Iterate over descriptions to clean each description
            ]

            return clean_descriptions  # Return the list of cleaned descriptions

        else:  # Handle non-200 status codes
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )  # Raise exception with error details

    except Exception as e:  # Catch any exceptions during search or processing
        print(f"Error fetching or processing descriptions: {e}")
        return []  # Return empty list in case of error


def get_search_summary(
    internet_summary_runnable, search_results: List[Dict[str, Optional[str]]]
) -> Optional[Dict[str, Any]]:
    """
    Summarize internet search results using the provided runnable.

    This function takes a list of search results and uses a runnable, specifically designed
    for summarizing internet search results, to generate a concise summary.

    Args:
        internet_summary_runnable: The runnable object designed for summarizing internet search results.
        search_results (List[Dict[str, Optional[str]]]): A list of dictionaries, each containing search result details.

    Returns:
        Optional[Dict[str, Any]]: The summary of the search results generated by the runnable,
                                  or None if an error occurs during summarization.
    """
    search_summary = internet_summary_runnable.invoke(
        {"query": search_results}
    )  # Invoke the internet summary runnable with search results

    return search_summary  # Return the generated search summary


def get_chat_history() -> Optional[List[Dict[str, str]]]:
    """
    Retrieve the chat history from the active chat for use with the Ollama backend.

    Reads the chat history from "chatsDb.json" and formats it into a list of dictionaries
    suitable for conversational models, indicating 'user' or 'assistant' role for each message.

    Returns:
        Optional[List[Dict[str, str]]]: Formatted chat history as a list of dictionaries, where each
                                        dictionary has 'role' ('user' or 'assistant') and 'content'
                                        (message text). Returns None if retrieval fails or no active chat exists.
    """
    try:
        with open("chatsDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)  # Load chat database from JSON file

        active_chat_id = db.get("active_chat_id")
        if active_chat_id is None:
            return []  # No active chat, return empty history

        # Find the active chat
        active_chat = next((chat for chat in db["chats"] if chat["id"] == active_chat_id), None)
        if active_chat is None:
            return []  # Active chat not found, return empty history

        messages = active_chat.get("messages", [])  # Get messages from active chat

        formatted_chat_history: List[Dict[str, str]] = [
            {
                "role": "user" if entry["isUser"] else "assistant",
                "content": entry["message"],
            }
            for entry in messages  # Format all messages from active chat
        ]

        return formatted_chat_history

    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return None  # Return None in case of error

