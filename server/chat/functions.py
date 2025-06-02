import os
from wrapt_timeout_decorator import *
# import requests # httpx is preferred if async, or use a consistent sync client
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv

from server.app.helpers import *
from server.db import MongoManager # Import MongoManager

load_dotenv("server/.env")  # Load environment variables from .env file

async def generate_streaming_response(
    runnable, inputs: Dict[str, Any], stream: bool = False # This function is now in common.functions
) -> AsyncGenerator[Any, None]:
    from server.common.functions import generate_streaming_response as common_generate_streaming_response
    async for item in common_generate_streaming_response(runnable, inputs, stream):
        yield item


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
                                         This function uses synchronous `requests`.
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
        import requests # Ensure requests is imported for this sync call

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


async def get_chat_history(user_id: str, chat_id: str, mongo_manager_instance: MongoManager) -> List[Dict[str, str]]:
    """
    Retrieve the chat history for a specific user and chat_id from MongoDB.

    Fetches messages from the MongoDB chat_history collection and formats them
    into a list of dictionaries suitable for conversational models, indicating
    'user' or 'assistant' role for each message.

    Args:
        user_id (str): The ID of the user.
        chat_id (str): The ID of the chat to retrieve history for.

    Returns:
        List[Dict[str, str]]: Formatted chat history as a list of dictionaries, where each
                               dictionary has 'role' ('user' or 'assistant') and 'content'
                               (message text). Returns an empty list if retrieval fails or no messages exist.
    """
    try:
        messages = await mongo_manager_instance.get_chat_history(user_id, chat_id)
        if not messages:
            return []

        formatted_chat_history: List[Dict[str, str]] = []
        for entry in messages:
            # Ensure 'message' key exists and is a string
            message_content = str(entry.get("message", ""))
            if message_content: # Only add if message content is not empty
                formatted_chat_history.append(
                    {
                        "role": "user" if entry.get("isUser") else "assistant",
                        "content": message_content,
                    }
                )
        return formatted_chat_history

    except Exception as e:
        print(f"Error retrieving chat history from MongoDB: {str(e)}")
        return [] # Return empty list in case of error

