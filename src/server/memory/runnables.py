import os
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library
from typing import Optional, Dict
from dotenv import load_dotenv

from .formats import *  # Importing format specifications or utilities from formats.py
from .prompts import *  # Importing prompt templates and related utilities from prompts.py
from .constants import *  # Importing constant variables from constants.py
from server.app.base import *
from server.app.helpers import *  # Importing helper functions from helpers.py

load_dotenv("server/.env")

def get_chat_runnable(chat_history: list[dict]) -> BaseRunnable:
    """
    Creates and configures a Runnable for handling chat interactions based on selected model.

    This runnable is designed for general chat purposes, utilizing chat-specific prompts and settings.
    It is stateful to maintain conversation history and supports streaming responses for interactive chat experiences.

    :param chat_history: The chat history to be added to the runnable's context.
    :return: A configured BaseRunnable instance for chat interactions.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=chat_system_prompt_template,
        user_prompt_template=chat_user_prompt_template,
        input_variables=[
            "query",
            "user_context",
            "internet_context",
            "name",
            "personality",
        ],
        response_type="chat",
        stream=True,
        stateful=True,
    )
    runnable.add_to_history(chat_history)
    return runnable


def get_graph_decision_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for making decisions about graph operations (CRUD) based on selected model.

    This runnable is responsible for deciding on Create, Read, Update, Delete (CRUD) operations
    based on analysis of graph data. It is configured with prompts and settings specific to graph decision-making.

    :return: A configured BaseRunnable instance for graph decision-making.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    graph_decision_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=graph_decision_system_prompt_template,
        user_prompt_template=graph_decision_user_prompt_template,
        input_variables=["analysis"],
        required_format=graph_decision_required_format,
        response_type="json",
    )

    return graph_decision_runnable


def get_graph_analysis_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for analyzing graph data based on selected model.

    This runnable is designed to analyze and compare graph data, such as identifying differences
    between an existing graph and new information. It is configured with prompts and settings
    optimized for graph analysis tasks.

    :return: A configured BaseRunnable instance for graph analysis.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable


    graph_analysis_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=graph_analysis_system_prompt_template,
        user_prompt_template=graph_analysis_user_prompt_template,
        input_variables=["related_graph", "extracted_data"],
        required_format=graph_analysis_required_format,
        response_type="json",
    )

    return graph_analysis_runnable


def get_text_dissection_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for dissecting text into predefined categories based on selected model.

    This runnable is used to categorize unstructured text into predefined categories, which is useful
    for organizing and processing large volumes of text data. It is configured with prompts and settings
    specific to text dissection tasks.

    :return: A configured BaseRunnable instance for text dissection.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    text_dissection_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=text_dissection_system_prompt_template,
        user_prompt_template=text_dissection_user_prompt_template,
        input_variables=["user_name", "text"],
        required_format=text_dissection_required_format,
        response_type="json",
    )

    return text_dissection_runnable


def get_information_extraction_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for extracting structured information (entities and relationships) from text based on selected model.

    This runnable is designed to identify and extract entities and the relationships between them from unstructured text.
    It is configured with prompts and settings optimized for information extraction tasks, and expects a JSON format response.

    :return: A configured BaseRunnable instance for information extraction.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    information_extraction_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=information_extraction_system_prompt_template,
        user_prompt_template=information_extraction_user_prompt_template,
        input_variables=["category", "text"],
        required_format=information_extraction_required_format,
        response_type="json",
    )

    return information_extraction_runnable


def get_text_conversion_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for converting structured graph data into unstructured text based on selected model.

    This runnable takes graph data as input and transforms it into human-readable, unstructured text.
    It is used for generating textual summaries or descriptions from knowledge graph information and is
    configured with prompts and settings for text conversion.

    :return: A configured BaseRunnable instance for text conversion.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    text_conversion_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=text_conversion_system_prompt_template,
        user_prompt_template=text_conversion_user_prompt_template,
        input_variables=["graph_data"],
        response_type="chat",
    )

    return text_conversion_runnable

def get_interest_extraction_runnable():
    """Configure and return a Runnable for extracting user interests from context."""
    # Model selection logic (assumes get_selected_model exists)
    model_mapping = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }
    model_name, provider = get_selected_model()

    # Choose the appropriate model provider or fallback
    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable  # Default fallback

    # Configure and return the runnable
    interest_extraction_runnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=interest_extraction_system_prompt_template,
        user_prompt_template=interest_extraction_user_prompt_template,
        input_variables=["context"],  # The variable to substitute in the user prompt
        required_format=interest_extraction_required_format,
        response_type="json",  # Expect JSON output
        stateful=False,  # No need to maintain state for this task
    )
    
    return interest_extraction_runnable


def get_query_classification_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for classifying user queries into predefined categories based on selected model.

    This runnable is used to determine the category or intent of a user query, which is crucial for
    routing queries to the appropriate processing logic. It is configured with prompts and settings
    for query classification tasks and expects a JSON response containing the classification.

    :return: A configured BaseRunnable instance for query classification.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable
        
    query_classification_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=query_classification_system_prompt_template,
        user_prompt_template=query_classification_user_prompt_template,
        input_variables=["query"],
        required_format=query_classification_required_format,
        response_type="json",
    )

    return query_classification_runnable


def get_fact_extraction_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for extracting factual points from a paragraph of text based on selected model.

    This runnable is designed to identify and extract key factual statements from a given text paragraph.
    It is used for populating the knowledge graph with structured facts and is configured with prompts
    and settings for fact extraction, expecting a JSON format response.

    :return: A configured BaseRunnable instance for fact extraction.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    fact_extraction_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=fact_extraction_system_prompt_template,
        user_prompt_template=fact_extraction_user_prompt_template,
        input_variables=["paragraph", "username"],
        required_format=fact_extraction_required_format,
        response_type="json",
    )

    return fact_extraction_runnable


def get_text_summarizer_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for summarizing text based on selected model.

    This runnable is used to generate concise summaries of longer texts. It is configured with prompts
    and settings optimized for text summarization tasks and is expected to return chat-style text output.

    :return: A configured BaseRunnable instance for text summarization.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    text_summarizer_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=text_summarization_system_prompt_template,
        user_prompt_template=text_summarization_user_prompt_template,
        input_variables=["user_name", "text"],
        response_type="chat",
    )

    return text_summarizer_runnable


def get_text_description_runnable() -> BaseRunnable:
    """
    Creates and configures a Runnable for generating descriptive text for entities or queries based on selected model.

    This runnable is designed to produce human-readable descriptions, often used to provide context or
    elaborate on entities or concepts. It is configured with prompts and settings for text description
    generation and is expected to return chat-style text output.

    :return: A configured BaseRunnable instance for text description generation.
    """
    model_mapping: Dict[str, tuple[Optional[str], type[BaseRunnable]]] = {
        "openai": (os.getenv("OPENAI_API_URL"), OpenAIRunnable),
        "claude": (os.getenv("CLAUDE_API_URL"), ClaudeRunnable),
        "gemini": (os.getenv("GEMINI_API_URL"), GeminiRunnable),
    }

    provider: Optional[str] = None
    model_name, provider=get_selected_model()

    if provider and provider in model_mapping:
        model_url, runnable_class = model_mapping[provider]
    else:
        model_url = os.getenv("BASE_MODEL_URL")
        runnable_class = OllamaRunnable

    text_description_runnable: BaseRunnable = runnable_class(
        model_url=model_url,
        model_name=model_name,
        system_prompt_template=text_description_system_prompt_template,
        user_prompt_template=text_description_user_prompt_template,
        input_variables=["query"],
        response_type="chat",
    )

    return text_description_runnable