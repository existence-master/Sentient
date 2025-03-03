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
from typing import (
    Optional,
    Any,
    Dict,
    List,
    AsyncGenerator,
)  # Import specific types for clarity
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from tzlocal import get_localzone
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

# --- FastAPI Application ---
app = FastAPI(
    title="Sentient API", description="API for the Sentient AI companion"
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

# --- Tool Handlers Registry ---
# Dictionary to store registered tool handler functions.
tool_handlers: Dict[str, callable] = {}


def register_tool(name: str):
    """
    Decorator to register a function as a tool handler.

    Args:
        name (str): The name of the tool to register.

    Returns:
        callable: A decorator that registers the decorated function as a tool handler.
    """

    def decorator(func: callable):
        """
        The actual decorator function that registers the tool.

        Args:
            func (callable): The function to be registered as a tool handler.

        Returns:
            callable: The original function, after registering it in tool_handlers.
        """
        tool_handlers[name] = func
        return func

    return decorator


# --- Apply nest_asyncio ---
# nest_asyncio is used to allow asyncio.run() to be called from within a jupyter notebook or another async environment.
# It's needed here because uvicorn runs in an asyncio event loop, and we might need to run async functions within the API endpoints.
nest_asyncio.apply()

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


class ToolCall(BaseModel):
    """
    Pydantic model for tool call requests.

    Attributes:
        input (str): The input string for the tool.
        previous_tool_response (Optional[Any]): The response from a previous tool call, if required. Defaults to None.
    """

    input: str
    previous_tool_response: Optional[Any] = None


class ElaboratorMessage(BaseModel):
    """
    Pydantic model for the elaborator message request body.

    Attributes:
        input (str): The input string to be elaborated.
        purpose (str): The purpose of elaboration.
    """

    input: str
    purpose: str


# --- Global Variables ---
# These global variables hold the runnables for different parts of the application.
# It is initialized to None and will be set in the `/initiate` endpoint.
chat_history = None  # Placeholder for chat history object
chat_runnable = None  # Runnable for handling chat conversations
agent_runnable = None  # Runnable for the main agent logic
tool_runnable = None  # Runnable for handling tool calls
reflection_runnable = None  # Runnable for reflection process
inbox_summarizer_runnable = None  # Runnable for summarizing inbox contents

# --- API Endpoints ---


@app.get("/", status_code=200)
async def main() -> Dict[str, str]:
    """
    Root endpoint of the API.

    Returns:
        JSONResponse: A simple greeting message.
    """
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }


@app.post("/initiate", status_code=200)
async def initiate() -> JSONResponse:
    """
    Endpoint to initiate the AI model and related runnables.

    This endpoint initializes the reflection and inbox summarizer runnables.
    It is intended to be called once at the start of a session or application lifecycle.

    Returns:
        JSONResponse: Success or error message in JSON format.
    """
    global chat_history, tool_runnable, reflection_runnable, inbox_summarizer_runnable

    reflection_runnable = get_reflection_runnable()  # Initialize reflection runnable
    inbox_summarizer_runnable = (
        get_inbox_summarizer_runnable()
    )  # Initialize inbox summarizer runnable

    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )
    except Exception as e:
        print(f"Error in initiating agents: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage) -> JSONResponse:
    try:
        elaborator_runnable = get_tool_runnable(
            elaborator_system_prompt_template,
            elaborator_user_prompt_template,
            None,
            ["query", "purpose"],
        )
        output = elaborator_runnable.invoke(
            {"query": message.input, "purpose": message.purpose}
        )
        print(f"Elaborator output: {output}")  # Debug the raw output
        return JSONResponse(status_code=200, content={"message": output})
    except Exception as e:
        print(f"Error in elaborator: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/chat", status_code=200)
async def chat(message: Message) -> StreamingResponse:
    """
    Endpoint to handle chat messages and generate responses using the AI model.

    This is the main chat endpoint that processes user messages, retrieves context,
    calls tools if necessary, and streams back the response.

    Args:
        message (Message): Request body containing the chat message details.

    Returns:
        StreamingResponse: A streaming response containing different types of messages
                           (user message, intermediary messages, assistant messages, tool results).
    """
    global chat_runnable, agent_runnable, reflection_runnable

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
        agent_runnable = get_agent_runnable(
            chat_history
        )  # Initialize agent runnable with chat history
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
            agents_used = False  # Flag to track if agents were used
            internet_used = False  # Flag to track if internet search was used
            user_context = None  # Placeholder for user context retrieved from memory
            internet_context = None  # Placeholder for internet search results
            pro_used = False  # Flag to track if pro features were used
            note = ""  # Placeholder for notes or messages to the user

            agents_used = True  # Mark agents as used for this interaction

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

            context_classification = await classify_context(
                transformed_input, "category"
            )  # Classify context for memory retrieval

            if (
                "personal" in context_classification
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
            )  # Classify context for internet search

            if pricing_plan == "free":  # Free plan logic
                if (
                    internet_classification == "Internet"
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
                        note = "Could have searched the internet too. But, that is a pro feature too :)"  # Note for free users without credits
                else:
                    internet_context = None  # No internet context needed for free plan
            else:  # Pro plan logic
                internet_classification = await classify_context(
                    transformed_input, "category"
                )  # Classify context for internet search

                if (
                    internet_classification == "Internet"
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

            response = generate_response(
                agent_runnable,
                transformed_input,
                user_context,
                internet_context,
                username,
            )  # Generate agent response

            if "tool_calls" not in response or not isinstance(
                response["tool_calls"], list
            ):  # Check for valid tool calls in response
                yield (
                    json.dumps(
                        {
                            "type": "assistantMessage",
                            "message": "Error: Invalid tool_calls format in response.",  # Error message for invalid tool calls
                        }
                    )
                    + "\n"
                )
                return

            previous_tool_result = None  # Placeholder for previous tool result
            all_tool_results: List[
                Dict
            ] = []  # List to store results from all tool calls

            if (
                len(response["tool_calls"]) > 1 and pricing_plan == "free"
            ):  # Check for multiple tool calls in free plan
                if credits <= 0:  # Check for credits in free plan for multiple tools
                    yield (
                        json.dumps(
                            {
                                "type": "assistantMessage",
                                "message": "Sorry friend, but the query requires multiple tools to be called. This is a pro feature and you are out of daily credits for pro. You can upgrade to pro from the settings page."
                                + f"\n\n{note}",  # Message for free users without credits for multiple tools
                            }
                        )
                        + "\n"
                    )
                    return
                else:
                    pro_used = True  # Mark pro feature as used if credits are available

            for tool_call in response[
                "tool_calls"
            ]:  # Iterate through tool calls in the response
                if (
                    tool_call["response_type"] != "tool_call"
                ):  # Skip if not a tool call response
                    continue

                tool_name = tool_call["content"].get(
                    "tool_name"
                )  # Get tool name from tool call content

                if (
                    tool_name != "gmail" and pricing_plan == "free"
                ):  # Check for tool usage in free plan (excluding gmail)
                    if credits <= 0:  # Check for credits in free plan for tool usage
                        yield (
                            json.dumps(
                                {
                                    "type": "assistantMessage",
                                    "message": "Sorry friend but the query requires a tool to be called which is only available in the pro version. This is a pro feature and you are out of daily credits for pro. You can upgrade to pro from the settings page."
                                    + f"\n\n{note}",  # Message for free users without credits for tool usage
                                }
                            )
                            + "\n"
                        )
                        return
                    else:
                        pro_used = (
                            True  # Mark pro feature as used if credits are available
                        )

                task_instruction = tool_call["content"].get(
                    "task_instruction"
                )  # Get task instruction from tool call content
                previous_tool_response_required = tool_call["content"].get(
                    "previous_tool_response", False
                )  # Check if previous tool response is required

                if (
                    not tool_name or not task_instruction
                ):  # Check for required fields in tool call
                    yield (
                        json.dumps(
                            {
                                "type": "assistantMessage",
                                "message": "Error: Tool call is missing required fields.",  # Error message for missing tool call fields
                            }
                        )
                        + "\n"
                    )
                    continue

                yield (
                    json.dumps(
                        {
                            "type": "intermediary-flow-update",
                            "message": f"Calling tool: {tool_name}...",
                        }
                    )
                    + "\n"
                )  # Yield intermediary message - tool call update
                await asyncio.sleep(0.05)  # Small delay

                tool_handler = tool_handlers.get(
                    tool_name
                )  # Get tool handler function from registry
                if not tool_handler:  # Check if tool handler exists
                    yield (
                        json.dumps(
                            {
                                "type": "assistantMessage",
                                "message": f"Error: Tool {tool_name} not found.",  # Error message for tool not found
                            }
                        )
                        + "\n"
                    )
                    continue

                tool_input = {"input": task_instruction}  # Prepare tool input
                if (
                    previous_tool_response_required and previous_tool_result
                ):  # Add previous tool response to input if required
                    tool_input["previous_tool_response"] = previous_tool_result
                else:
                    tool_input["previous_tool_response"] = (
                        "Not Required"  # Indicate previous tool response is not required
                    )

                try:
                    tool_result_main = await tool_handler(
                        tool_input
                    )  # Execute tool handler function
                    tool_result = None
                    tool_call_str = None
                    if (
                        tool_result_main["tool_call_str"] is not None
                    ):  # Check if tool call string is present in tool result
                        tool_call_str = tool_result_main[
                            "tool_call_str"
                        ]  # Get tool call string
                        tool_result = tool_result_main["tool_result"]  # Get tool result
                        tool_name = tool_call_str[
                            "tool_name"
                        ]  # Get tool name from tool call string
                        if (
                            tool_name == "search_inbox"
                        ):  # Handle search inbox tool result
                            yield (
                                json.dumps(
                                    {
                                        "type": "toolResult",
                                        "tool_name": tool_name,
                                        "result": tool_result["result"],
                                        "gmail_search_url": tool_result["result"][
                                            "gmail_search_url"
                                        ],  # Include gmail search URL in result
                                    }
                                )
                                + "\n"
                            )
                        elif (
                            tool_name == "get_email_details"
                        ):  # Handle get email details tool result
                            yield (
                                json.dumps(
                                    {
                                        "type": "toolResult",
                                        "tool_name": tool_name,
                                        "result": tool_result["result"],
                                    }
                                )
                                + "\n"
                            )
                        await asyncio.sleep(0.05)  # Small delay
                    else:
                        tool_result = tool_result_main[
                            "tool_result"
                        ]  # Get tool result if no tool call string
                    previous_tool_result = tool_result  # Update previous tool result
                    all_tool_results.append(
                        {  # Append tool result to list
                            "tool_name": tool_name,
                            "task_instruction": task_instruction,
                            "tool_result": tool_result,
                        }
                    )

                except Exception as e:  # Handle exceptions during tool execution
                    print(f"Error executing tool {tool_name}")
                    yield (
                        json.dumps(
                            {
                                "type": "assistantMessage",
                                "message": f"Error executing tool {tool_name}: {str(e)}",  # Error message for tool execution failure
                            }
                        )
                        + "\n"
                    )
                    continue

            yield (
                json.dumps({"type": "intermediary-flow-end"}) + "\n"
            )  # Yield intermediary message - flow end
            await asyncio.sleep(0.05)  # Small delay

            try:
                if (
                    len(all_tool_results) == 1
                    and all_tool_results[0]["tool_name"] == "search_inbox"
                ):  # Handle inbox summarization for single search inbox tool result
                    filtered_tool_result = {
                        "response": all_tool_results[0]["tool_result"]["result"][
                            "response"
                        ],  # Extract response from tool result
                        "email_data": [  # Filter email data to exclude email body
                            {key: email[key] for key in email if key != "body"}
                            for email in all_tool_results[0]["tool_result"]["result"][
                                "email_data"
                            ]
                        ],
                        "gmail_search_url": all_tool_results[0]["tool_result"][
                            "result"
                        ]["gmail_search_url"],  # Extract gmail search URL
                    }

                    async for token in generate_streaming_response(  # Stream response from inbox summarizer runnable
                        inbox_summarizer_runnable,
                        inputs={"tool_result": filtered_tool_result},
                        stream=True,
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
                            await asyncio.sleep(0.05)
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
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.05)
                else:  # Handle reflection for other tool results or multiple tool results
                    async for token in generate_streaming_response(  # Stream response from reflection runnable
                        reflection_runnable,
                        inputs={"tool_results": all_tool_results},
                        stream=True,
                    ):
                        if isinstance(token, str):  # Yield assistant stream tokens
                            yield (
                                json.dumps(
                                    {
                                        "type": "assistantStream",
                                        "token": token,
                                        "done": False,
                                        "memoryUsed": memory_used,
                                        "agentsUsed": agents_used,
                                        "internetUsed": internet_used,
                                        "proUsed": pro_used,
                                    }
                                )
                                + "\n"
                            )
                            await asyncio.sleep(0.05)
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
                        await asyncio.sleep(0.05)
                await asyncio.sleep(0.05)
            except (
                Exception
            ) as e:  # Handle exceptions during reflection or summarization
                print(f"Error during reflection: {e}")
                yield (
                    json.dumps(
                        {
                            "type": "assistantMessage",
                            "message": f"Error during reflection: {str(e)}",  # Error message for reflection failure
                        }
                    )
                    + "\n"
                )

        return StreamingResponse(
            response_generator(), media_type="application/json"
        )  # Return streaming response

    except Exception as e:  # Handle exceptions during chat processing
        print(f"Error during chat: {e}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return JSON error response


@register_tool("gmail")
async def gmail_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    Gmail Tool endpoint to handle email related tasks using multi-tool support.
    Registered as a tool with the name "gmail".

    Args:
        tool_call (ToolCall): Request body containing the input for the gmail tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string.
                         Returns status "failure" and error message if an exception occurs.
    """
    try:
        with open(
            "../../userProfileDb.json", "r", encoding="utf-8"
        ) as f:  # Load user profile database
            db = json.load(f)

        username = db["userData"]["personalInfo"][
            "name"
        ]  # Extract username from user profile

        tool_runnable = get_tool_runnable(  # Initialize gmail tool runnable
            gmail_agent_system_prompt_template,
            gmail_agent_user_prompt_template,
            gmail_agent_required_format,
            [
                "query",
                "username",
                "previous_tool_response",
            ],  # Expected input parameters
        )

        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gmail tool runnable
                "query": tool_call["input"],
                "username": username,
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response

        return {
            "tool_result": tool_result,
            "tool_call_str": tool_call_str,
        }  # Return tool result and tool call string
    except Exception as e:  # Handle exceptions during gmail tool execution
        print(f"Error calling gmail tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


@register_tool("gdrive")
async def drive_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    Drive Tool endpoint to handle Google Drive interactions using multi-tool support.
    Registered as a tool with the name "gdrive".

    Args:
        tool_call (ToolCall): Request body containing the input for the drive tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """
    try:
        tool_runnable = get_tool_runnable(  # Initialize gdrive tool runnable
            gdrive_agent_system_prompt_template,
            gdrive_agent_user_prompt_template,
            gdrive_agent_required_format,
            ["query", "previous_tool_response"],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gdrive tool runnable
                "query": tool_call["input"],
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gdrive tool execution
        print(f"Error calling gdrive tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


@register_tool("gslides")
async def gslides_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GSlides Tool endpoint to handle Google Slides presentation creation using multi-tool support.
    Registered as a tool with the name "gslides".

    Args:
        tool_call (ToolCall): Request body containing the input for the gslides tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        with open(
            "../../userProfileDb.json", "r", encoding="utf-8"
        ) as f:  # Load user profile database
            db = json.load(f)

        username = db["userData"]["personalInfo"][
            "name"
        ]  # Extract username from user profile

        tool_runnable = get_tool_runnable(  # Initialize gslides tool runnable
            gslides_agent_system_prompt_template,
            gslides_agent_user_prompt_template,
            gslides_agent_required_format,
            [
                "query",
                "user_name",
                "previous_tool_response",
            ],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gslides tool runnable
                "query": tool_call["input"],
                "user_name": username,
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gslides tool execution
        print(f"Error calling gslides tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


@register_tool("gdocs")
async def gdoc_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GDocs Tool endpoint to handle Google Docs creation and text elaboration using multi-tool support.
    Registered as a tool with the name "gdocs".

    Args:
        tool_call (ToolCall): Request body containing the input for the gdocs tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """
    try:
        tool_runnable = get_tool_runnable(  # Initialize gdocs tool runnable
            gdocs_agent_system_prompt_template,
            gdocs_agent_user_prompt_template,
            gdocs_agent_required_format,
            ["query", "previous_tool_response"],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gdocs tool runnable
                "query": tool_call["input"],
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gdocs tool execution
        print(f"Error calling gdocs tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


@register_tool("gsheets")
async def gsheet_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GSheets Tool endpoint to handle Google Sheets creation and data population using multi-tool support.
    Registered as a tool with the name "gsheets".

    Args:
        tool_call (ToolCall): Request body containing the input for the gsheets tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        tool_runnable = get_tool_runnable(  # Initialize gsheets tool runnable
            gsheets_agent_system_prompt_template,
            gsheets_agent_user_prompt_template,
            gsheets_agent_required_format,
            ["query", "previous_tool_response"],  # Expected input parameters
        )
        tool_call_str = tool_runnable.invoke(
            {  # Invoke the gsheets tool runnable
                "query": tool_call["input"],
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gsheets tool execution
        print(f"Error calling gsheets tool: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


@register_tool("gcalendar")
async def gcalendar_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """
    GCalendar Tool endpoint to handle Google Calendar interactions using multi-tool support.
    Registered as a tool with the name "gcalendar".

    Args:
        tool_call (ToolCall): Request body containing the input for the gcalendar tool.

    Returns:
        Dict[str, Any]: A dictionary containing the tool result and tool call string (None in this case).
                         Returns status "failure" and error message if an exception occurs.
    """

    try:
        current_time = datetime.now().isoformat()  # Get current time in ISO format
        local_timezone = get_localzone()  # Get local timezone
        timezone = local_timezone.key  # Get timezone key

        tool_runnable = get_tool_runnable(  # Initialize gcalendar tool runnable
            gcalendar_agent_system_prompt_template,
            gcalendar_agent_user_prompt_template,
            gcalendar_agent_required_format,
            [
                "query",
                "current_time",
                "timezone",
                "previous_tool_response",
            ],  # Expected input parameters
        )

        tool_call_str = tool_runnable.invoke(  # Invoke the gcalendar tool runnable
            {
                "query": tool_call["input"],
                "current_time": current_time,
                "timezone": timezone,
                "previous_tool_response": tool_call["previous_tool_response"],
            }
        )

        tool_result = await parse_and_execute_tool_calls(
            tool_call_str
        )  # Parse and execute tool calls from the response
        return {
            "tool_result": tool_result,
            "tool_call_str": None,
        }  # Return tool result and None for tool call string
    except Exception as e:  # Handle exceptions during gcalendar tool execution
        print(f"Error calling gcalendar: {e}")
        return {"status": "failure", "error": str(e)}  # Return error status and message


if __name__ == "__main__":
    # --- Run the application ---
    # This block is executed when the script is run directly (not imported as a module).
    # It starts the uvicorn server to serve the FastAPI application. Note that 5001 is a port used by Sentient in production. Make sure to change it
    uvicorn.run(app, port=5001)
