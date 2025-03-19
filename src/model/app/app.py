import os
import json
import asyncio
import pickle
import multiprocessing
import requests
from datetime import datetime
from tzlocal import get_localzone
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, AsyncGenerator
from neo4j import GraphDatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import nest_asyncio
import uvicorn

# Import specific functions, runnables, and helpers from respective folders
from agents.runnables import *
from agents.functions import *
from agents.prompts import *
from agents.formats import *

from memory.runnables import *
from memory.functions import *
from memory.prompts import *
from memory.constants import *
from memory.formats import *

from utils.helpers import *

from scraper.runnables import *
from scraper.functions import *
from scraper.prompts import *
from scraper.formats import *

from auth.helpers import *

from common.functions import *
from common.runnables import *
from common.prompts import *
from common.formats import *

from chat.runnables import *
from chat.prompts import *
from chat.functions import *

# Load environment variables from .env file
load_dotenv("../.env")

# Apply nest_asyncio to allow nested event loops (useful for development environments)
nest_asyncio.apply()

# --- Global Initializations ---
# Perform all initializations before defining endpoints, replacing the /initiate endpoints

# Initialize embedding model for memory-related operations
embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])

# Initialize Neo4j graph driver for knowledge graph interactions
graph_driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Initialize runnables from agents
reflection_runnable = get_reflection_runnable()
inbox_summarizer_runnable = get_inbox_summarizer_runnable()

# Initialize runnables from memory
graph_decision_runnable = get_graph_decision_runnable()
information_extraction_runnable = get_information_extraction_runnable()
graph_analysis_runnable = get_graph_analysis_runnable()
text_dissection_runnable = get_text_dissection_runnable()
text_conversion_runnable = get_text_conversion_runnable()
query_classification_runnable = get_query_classification_runnable()
fact_extraction_runnable = get_fact_extraction_runnable()
text_summarizer_runnable = get_text_summarizer_runnable()
text_description_runnable = get_text_description_runnable()

# Initialize runnables from scraper
reddit_runnable = get_reddit_runnable()
twitter_runnable = get_twitter_runnable()

context_classification_runnable = get_context_classification_runnable()
internet_query_reframe_runnable = get_internet_query_reframe_runnable()
internet_summary_runnable = get_internet_summary_runnable()
internet_search_runnable = get_internet_classification_runnable()

# Tool handlers registry for agent tools
tool_handlers: Dict[str, callable] = {}

def register_tool(name: str):
    """Decorator to register a function as a tool handler."""
    def decorator(func: callable):
        tool_handlers[name] = func
        return func
    return decorator

# Google OAuth2 scopes and credentials (from auth and common)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive",
    "https://mail.google.com/",
]

CREDENTIALS_DICT = {
    "installed": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "auth_uri": os.environ.get("GOOGLE_AUTH_URI"),
        "token_uri": os.environ.get("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost"]
    }
}

# Auth0 configuration from utils
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILE_DB = os.path.join(BASE_DIR, "..", "..", "userProfileDb.json")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Sentient API",
    description="Monolithic API for the Sentient AI companion",
    docs_url="/docs",
    redoc_url=None
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic Models ---
class Message(BaseModel):
    original_input: str
    transformed_input: str
    pricing: str
    credits: int
    chat_id: str

class ToolCall(BaseModel):
    input: str
    previous_tool_response: Optional[Any] = None

class ElaboratorMessage(BaseModel):
    input: str
    purpose: str

class EncryptionRequest(BaseModel):
    data: str

class DecryptionRequest(BaseModel):
    encrypted_data: str

class UserInfoRequest(BaseModel):
    user_id: str

class ReferrerStatusRequest(BaseModel):
    user_id: str
    referrer_status: bool

class BetaUserStatusRequest(BaseModel):
    user_id: str
    beta_user_status: bool

class SetReferrerRequest(BaseModel):
    referral_code: str

class DeleteSubgraphRequest(BaseModel):
    source: str

class GraphRequest(BaseModel):
    information: str

class GraphRAGRequest(BaseModel):
    query: str

class RedditURL(BaseModel):
    url: str

class TwitterURL(BaseModel):
    url: str

class LinkedInURL(BaseModel):
    url: str

# --- API Endpoints ---

## Root Endpoint
@app.get("/", status_code=200)
async def main():
    """Root endpoint providing a welcome message."""
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }

## Chat Endpoint (Combining agents and memory logic)
@app.post("/chat", status_code=200)
async def chat(message: Message):
    global embed_model, chat_runnable, context_classification_runnable
    global fact_extraction_runnable, text_conversion_runnable, information_extraction_runnable
    global graph_analysis_runnable, graph_decision_runnable, query_classification_runnable, agent_runnable, orchestrator_runnable, text_description_runnable, reflection_runnable, internet_search_runnable, internet_query_reframe_runnable, internet_summary_runnable

    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)

        chat_history = get_chat_history(message.chat_id)
        chat_runnable = get_chat_runnable(chat_history)
        agent_runnable = get_agent_runnable(chat_history)
        orchestrator_runnable = get_orchestrator_runnable(chat_history)

        username = db["userData"]["personalInfo"]["name"]
        output = orchestrator_runnable.invoke({"query": message.input})
        category = output["class"]
        transformed_input = output["input"]

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

            yield json.dumps({"type": "userMessage", "message": message.input, "memoryUsed": memory_used, "agentsUsed": agents_used, "internetUsed": internet_used}) + "\n"
            await asyncio.sleep(0.05)

            if category == "chat":
                yield json.dumps({"type": "intermediary", "message": "Processing chat response..."}) + "\n"
                await asyncio.sleep(0.05)

                context_classification = context_classify(context_classification_runnable, transformed_input)
                context_classification = context_classification["class"]

                if "personal" in context_classification:
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                    await asyncio.sleep(0.05)
                    memory_used = True
                    user_context = query_user_profile(transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
                else:
                    user_context = None
                
                internet_classification = context_classify(internet_search_runnable, transformed_input)
                internet_classification = internet_classification["class"]

                if pricing_plan == "free":
                    if internet_classification == "Internet":
                        if credits > 0:
                                yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                                reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)

                                search_results = get_search_results(reframed_query)
                                internet_context = get_search_summary(internet_summary_runnable, search_results)
                                internet_used = True
                                pro_used = True
                        else:
                            note = "Sorry friend, could have searched the internet for this query for more context. But, that is a pro feature and your daily credits have expired. You can always upgrade to pro from the settings page"
                    else:
                        internet_context = None

                else:
                    if internet_classification == "Internet":
                        yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                        reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)
                        search_results = get_search_results(reframed_query)
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        internet_used = True
                        pro_used = True
                    else:
                        internet_context = None
                with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
                    db = json.load(f)

                personality_description = db["userData"].get("personality", "None")

                async for token in generate_streaming_response(
                    chat_runnable,
                    inputs= {
                        "query": transformed_input,
                        "user_context": user_context,
                        "internet_context": internet_context,
                        "name": username,
                        "personality": personality_description
                    },
                    stream=True
                ):
                    if isinstance(token, str):
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": token,
                            "done": False
                        }) + "\n"
                    else:
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": "\n\n" + note,
                            "done": True,
                            "memoryUsed": memory_used,
                            "agentsUsed": agents_used,
                            "internetUsed": internet_used,
                            "proUsed": pro_used,
                        }) + "\n"
                    await asyncio.sleep(0.05)  
                await asyncio.sleep(0.05)

            elif category == "memory":
                if pricing_plan == "free":
                    if credits <= 0:
                        yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                        await asyncio.sleep(0.05)

                        user_context = query_user_profile(transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
                        note = "Sorry friend, could have updated my memory for this query. But, that is a pro feature and your daily credits have expired. You can always upgrade to pro from the settings page"
                    
                    else:
                        yield json.dumps({"type": "intermediary", "message": "Updating memories..."}) + "\n"
                        await asyncio.sleep(0.05)

                        memory_used = True
                        pro_used = True
                        points = fact_extraction_runnable.invoke({"paragraph": transformed_input, "username": username})

                        for point in points:
                            crud_graph_operations(point, graph_driver, embed_model, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable)

                        yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                        await asyncio.sleep(0.05)

                        user_context = query_user_profile(transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)

                        internet_classification = context_classify(internet_search_runnable, transformed_input)["class"]

                        if internet_classification == "Internet":
                            yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                            reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)

                            search_results = get_search_results(reframed_query)
                            internet_context = get_search_summary(internet_summary_runnable, search_results)
                            internet_used = True
                        else:
                            internet_context = None
                else:
                    yield json.dumps({"type": "intermediary", "message": "Updating memories..."}) + "\n"
                    await asyncio.sleep(0.05)

                    memory_used = True
                    pro_used = True
                    points = fact_extraction_runnable.invoke({"paragraph": transformed_input, "username": username})

                    for point in points:
                        crud_graph_operations(point, graph_driver, embed_model, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable)

                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                    await asyncio.sleep(0.05)

                    user_context = query_user_profile(transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)

                    internet_classification = context_classify(internet_search_runnable, transformed_input)["class"]

                    if internet_classification == "Internet":
                        yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                        reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)

                        search_results = get_search_results(reframed_query)
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        internet_used = True
                    else:
                        internet_context = None

                with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
                    db = json.load(f)

                personality_description = db["userData"].get("personality", "None")

                async for token in generate_streaming_response(
                    chat_runnable,
                    inputs= {
                        "query": transformed_input,
                        "user_context": user_context,
                        "internet_context": internet_context,
                        "name": username,
                        "personality": personality_description
                    },
                    stream=True
                ):
                    if isinstance(token, str):
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": token,
                            "done": False
                        }) + "\n"
                    else:
                        yield json.dumps({
                            "type": "assistantStream",
                            "token": "\n\n" + note,
                            "done": True,
                            "memoryUsed": memory_used,
                            "agentsUsed": agents_used,
                            "internetUsed": internet_used,
                            "proUsed": pro_used,
                        }) + "\n"
                    await asyncio.sleep(0.05)  
                await asyncio.sleep(0.05)

            elif category == "agent":
                agents_used = True

                context_classification = context_classify(context_classification_runnable, transformed_input)["class"]

                if "personal" in context_classification:
                    yield json.dumps({"type": "intermediary", "message": "Retrieving memories..."}) + "\n"
                    await asyncio.sleep(0.05)
                    memory_used = True
                    user_context = query_user_profile(transformed_input, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
                else:
                    user_context = None

                internet_classification = context_classify(internet_search_runnable, transformed_input)["class"]

                if pricing_plan == "free":
                    if internet_classification == "Internet":
                        if credits > 0:
                            yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                            reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)

                            search_results = get_search_results(reframed_query)
                            internet_context = get_search_summary(internet_summary_runnable, search_results)
                            internet_used = True
                            pro_used = True
                        else:
                            note = "Could have searched the internet too. But, that is a pro feature too :)"
                    else:
                        internet_context = None

                else:
                    internet_classification = context_classify(internet_search_runnable, transformed_input)["class"]

                    if internet_classification == "Internet":
                        yield json.dumps({"type": "intermediary", "message": "Searching the internet..."}) + "\n"
                        reframed_query = get_reframed_internet_query(internet_query_reframe_runnable, transformed_input)

                        search_results = get_search_results(reframed_query)
                        internet_context = get_search_summary(internet_summary_runnable, search_results)
                        internet_used = True
                        pro_used = True
                    else:
                        internet_context = None

                response = generate_response(agent_runnable, transformed_input, user_context, internet_context, username)
                
                if "tool_calls" not in response or not isinstance(response["tool_calls"], list):
                    yield json.dumps({
                        "type": "assistantMessage",
                        "message": "Error: Invalid tool_calls format in response."
                    }) + "\n"
                    return

                previous_tool_result = None
                all_tool_results = []

                if len(response["tool_calls"]) > 1 and pricing_plan == "free":
                    if credits <= 0:
                        yield json.dumps({
                            "type": "assistantMessage",
                            "message": "Sorry friend, but the query requires multiple tools to be called. This is a pro feature and you are out of daily credits for pro. You can upgrade to pro from the settings page." + f"\n\n{note}"
                        }) + "\n"
                        return
                    else:
                        pro_used = True

                for tool_call in response["tool_calls"]:
                    if tool_call["response_type"] != "tool_call":
                        continue

                    tool_name = tool_call["content"].get("tool_name")

                    if tool_name != "gmail" and pricing_plan == "free":
                        if credits <= 0:
                            yield json.dumps({
                                "type": "assistantMessage",
                                "message": "Sorry friend but the query requires a tool to be called which is only available in the pro version. This is a pro feature and you are out of daily credits for pro. You can upgrade to pro from the settings page." + f"\n\n{note}"
                            }) + "\n"
                            return
                        else:
                            pro_used = True

                    task_instruction = tool_call["content"].get("task_instruction")
                    previous_tool_response_required = tool_call["content"].get("previous_tool_response", False)

                    if not tool_name or not task_instruction:
                        yield json.dumps({
                            "type": "assistantMessage",
                            "message": "Error: Tool call is missing required fields."
                        }) + "\n"
                        continue

                    yield json.dumps({"type": "intermediary-flow-update", "message": f"Calling tool: {tool_name}..."}) + "\n"
                    await asyncio.sleep(0.05)

                    tool_handler = tool_handlers.get(tool_name)
                    if not tool_handler:
                        yield json.dumps({
                            "type": "assistantMessage",
                            "message": f"Error: Tool {tool_name} not found."
                        }) + "\n"
                        continue

                    tool_input = {"input": task_instruction}
                    if previous_tool_response_required and previous_tool_result:
                        tool_input["previous_tool_response"] = previous_tool_result
                    else:
                        tool_input["previous_tool_response"] = "Not Required"

                    try:
                        tool_result_main = await tool_handler(tool_input)
                        tool_result = None
                        tool_call_str = None
                        if tool_result_main["tool_call_str"] is not None:
                            tool_call_str = tool_result_main["tool_call_str"]
                            tool_result = tool_result_main["tool_result"]
                            tool_name = tool_call_str["tool_name"]
                            if tool_name == "search_inbox":
                                yield json.dumps({
                                    "type": "toolResult",
                                    "tool_name": tool_name,
                                    "result": tool_result["result"],
                                    "gmail_search_url": tool_result["result"]["gmail_search_url"]
                                }) + "\n"
                            elif tool_name == "get_email_details":
                                yield json.dumps({
                                    "type": "toolResult",
                                    "tool_name": tool_name,
                                    "result": tool_result["result"]
                                }) + "\n"
                            await asyncio.sleep(0.05)
                        else:
                            tool_result = tool_result_main
                            
                        write_to_log(f"Tool {tool_name} executed successfully.")

                        previous_tool_result = tool_result
                        all_tool_results.append({
                            "tool_name": tool_name,
                            "task_instruction": task_instruction,
                            "tool_result": tool_result
                        })
                    except Exception as e:
                        write_to_log(f"Error executing tool {tool_name}: {str(e)}")
                        yield json.dumps({
                            "type": "assistantMessage",
                            "message": f"Error executing tool {tool_name}: {str(e)}"
                        }) + "\n"
                        continue

                yield json.dumps({"type": "intermediary-flow-end"}) + "\n"
                await asyncio.sleep(0.05)

                try:
                    if len(all_tool_results) == 1 and all_tool_results[0]["tool_name"] == "search_inbox":
                        filtered_tool_result = {
                            "response": all_tool_results[0]["tool_result"]["result"]["response"],
                            "email_data": [
                                {key: email[key] for key in email if key != "body"}
                                for email in all_tool_results[0]["tool_result"]["result"]["email_data"]
                            ],
                            "gmail_search_url": all_tool_results[0]["tool_result"]["result"]["gmail_search_url"]
                        }

                        async for token in generate_streaming_response(
                            inbox_summarizer_runnable,
                            inputs = {"tool_result": filtered_tool_result},
                            stream=True  
                        ):
                            if isinstance(token, str):
                                yield json.dumps({
                                    "type": "assistantStream",
                                    "token": token,
                                    "done": False
                                }) + "\n"
                                await asyncio.sleep(0.05)  
                            else:
                                yield json.dumps({
                                    "type": "assistantStream",
                                    "token": "\n\n" + note,
                                    "done": True,
                                    "memoryUsed": memory_used,
                                    "agentsUsed": agents_used,
                                    "internetUsed": internet_used,
                                    "proUsed": pro_used,
                                }) + "\n"
                            await asyncio.sleep(0.05)
                        await asyncio.sleep(0.05)
                    else:
                        async for token in generate_streaming_response(
                            reflection_runnable,
                            inputs = {"tool_results": all_tool_results},
                            stream=True  
                        ):
                            if isinstance(token, str):
                                yield json.dumps({
                                    "type": "assistantStream",
                                    "token": token,
                                    "done": False
                                }) + "\n"
                                await asyncio.sleep(0.05)  
                            else:
                                yield json.dumps({
                                    "type": "assistantStream",
                                    "token": "\n\n" + note,
                                    "done": True,
                                    "memoryUsed": memory_used,
                                    "agentsUsed": agents_used,
                                    "internetUsed": internet_used,
                                    "proUsed": pro_used,
                                }) + "\n"
                            await asyncio.sleep(0.05)
                    await asyncio.sleep(0.05)
                except Exception as e:
                    write_to_log(f"Error during reflection: {str(e)}")
                    yield json.dumps({
                        "type": "assistantMessage",
                        "message": f"Error during reflection: {str(e)}"
                    }) + "\n"

        return StreamingResponse(response_generator(), media_type="application/json")

    except Exception as e:
        write_to_log(f"Error in chat: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

## Agents Endpoints
@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage):
    """Elaborates on an input string based on a specified purpose."""
    try:
        elaborator_runnable = get_tool_runnable(
            elaborator_system_prompt_template,
            elaborator_user_prompt_template,
            None,
            ["query", "purpose"]
        )
        output = elaborator_runnable.invoke({"query": message.input, "purpose": message.purpose})
        return JSONResponse(status_code=200, content={"message": output})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

## Tool Handlers
@register_tool("gmail")
async def gmail_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """Handles Gmail-related tasks using multi-tool support."""
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"]["name"]
        tool_runnable = get_tool_runnable(
            gmail_agent_system_prompt_template,
            gmail_agent_user_prompt_template,
            gmail_agent_required_format,
            ["query", "username", "previous_tool_response"]
        )
        tool_call_str = tool_runnable.invoke({
            "query": tool_call.input,
            "username": username,
            "previous_tool_response": tool_call.previous_tool_response
        })
        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        return {"tool_result": tool_result, "tool_call_str": tool_call_str}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

@register_tool("gdrive")
async def drive_tool(tool_call: ToolCall) -> Dict[str, Any]:
    """Handles Google Drive interactions."""
    try:
        tool_runnable = get_tool_runnable(
            gdrive_agent_system_prompt_template,
            gdrive_agent_user_prompt_template,
            gdrive_agent_required_format,
            ["query", "previous_tool_response"]
        )
        tool_call_str = tool_runnable.invoke({
            "query": tool_call.input,
            "previous_tool_response": tool_call.previous_tool_response
        })
        tool_result = await parse_and_execute_tool_calls(tool_call_str)
        return {"tool_result": tool_result, "tool_call_str": None}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

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

# Additional tool handlers (gslides, gdocs, gsheets, gcalendar) follow the same pattern
# They are omitted here for brevity but should be included similarly

## Utils Endpoints
@app.post("/get-role")
async def get_role(request: UserInfoRequest) -> JSONResponse:
    """Retrieves a user's role from Auth0."""
    try:
        token = get_management_token()
        roles_response = requests.get(
            f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}/roles",
            headers={"Authorization": f"Bearer {token}"}
        )
        if roles_response.status_code != 200:
            raise HTTPException(status_code=roles_response.status_code, detail=roles_response.text)
        roles = roles_response.json()
        if not roles:
            return JSONResponse(status_code=404, content={"message": "No roles found for user."})
        return JSONResponse(status_code=200, content={"role": roles[0]["name"].lower()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/get-beta-user-status")
async def get_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """Retrieves beta user status from Auth0 app_metadata."""
    try:
        token = get_management_token()
        response = requests.get(
            f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        user_data = response.json()
        beta_user_status = user_data.get("app_metadata", {}).get("betaUser")
        if beta_user_status is None:
            return JSONResponse(status_code=404, content={"message": "Beta user status not found."})
        return JSONResponse(status_code=200, content={"betaUserStatus": beta_user_status})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Additional utils endpoints (get-referral-code, set-referrer-status, etc.) follow similar patterns
# They are omitted here for brevity but should be included

@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest) -> JSONResponse:
    """Encrypts data using AES encryption."""
    try:
        encrypted_data = aes_encrypt(request.data)
        return JSONResponse(status_code=200, content={"encrypted_data": encrypted_data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest) -> JSONResponse:
    """Decrypts data using AES decryption."""
    try:
        decrypted_data = aes_decrypt(request.encrypted_data)
        return JSONResponse(status_code=200, content={"decrypted_data": decrypted_data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

## Scraper Endpoints
@app.post("/scrape-linkedin", status_code=200)
async def scrape_linkedin(profile: LinkedInURL):
    """Scrapes and returns LinkedIn profile information."""
    try:
        linkedin_profile = scrape_linkedin_profile(profile.url)
        return JSONResponse(status_code=200, content={"profile": linkedin_profile})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/scrape-reddit")
async def scrape_reddit(reddit_url: RedditURL):
    """Extracts topics of interest from a Reddit user's profile."""
    try:
        subreddits = reddit_scraper(reddit_url.url)
        response = reddit_runnable.invoke({"subreddits": subreddits})
        if isinstance(response, list):
            return JSONResponse(status_code=200, content={"topics": response})
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from the language model.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/scrape-twitter")
async def scrape_twitter(twitter_url: TwitterURL):
    """Extracts topics of interest from a Twitter user's profile."""
    try:
        tweets = scrape_twitter_data(twitter_url.url, 20)
        response = twitter_runnable.invoke({"tweets": tweets})
        if isinstance(response, list):
            return JSONResponse(status_code=200, content={"topics": response})
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from the language model.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

## Auth Endpoint
@app.get("/authenticate-google")
async def authenticate_google():
    """Authenticates with Google using OAuth 2.0."""
    try:
        creds = None
        if os.path.exists("../token.pickle"):
            with open("../token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_config(CREDENTIALS_DICT, SCOPES)
                creds = flow.run_local_server(port=0)
            with open("../token.pickle", "wb") as token:
                pickle.dump(creds, token)
        return JSONResponse(status_code=200, content={"success": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

## Memory Endpoints
@app.post("/graphrag", status_code=200)
async def graphrag(request: GraphRAGRequest):
    """Processes a user profile query using GraphRAG."""
    try:
        context = query_user_profile(
            request.query, graph_driver, embed_model,
            text_conversion_runnable, query_classification_runnable
        )
        return JSONResponse(status_code=200, content={"context": context})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/create-graph", status_code=200)
async def create_graph():
    """Creates a knowledge graph from documents in the input directory."""
    try:
        input_dir = "../input"
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User")
        extracted_texts = []
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text_content = file.read().strip()
                    if text_content:
                        extracted_texts.append({"text": text_content, "source": file_name})
        if not extracted_texts:
            return JSONResponse(status_code=400, content={"message": "No content found in input documents."})
        
        with graph_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        build_initial_knowledge_graph(
            username, extracted_texts, graph_driver, embed_model,
            text_dissection_runnable, information_extraction_runnable
        )
        return JSONResponse(status_code=200, content={"message": "Graph created successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/delete-subgraph", status_code=200)
async def delete_subgraph(request: DeleteSubgraphRequest):
    """Deletes a subgraph from the knowledge graph based on a source name."""
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User").lower()
        source_name = request.source
        SOURCES = {
            "linkedin": f"{username}_linkedin_profile.txt",
            "reddit": f"{username}_reddit_profile.txt",
            "twitter": f"{username}_twitter_profile.txt"
        }
        file_name = SOURCES.get(source_name)
        if not file_name:
            return JSONResponse(status_code=400, content={"message": f"No file mapping found for source: {source_name}"})
        delete_source_subgraph(graph_driver, file_name)
        os.remove(f"../input/{file_name}")
        return JSONResponse(status_code=200, content={"message": f"Subgraph related to {file_name} deleted successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/create-document", status_code=200)
async def create_document():
    """Creates and summarizes personality documents based on user profile data."""
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"].get("name", "User")
        personality_type = db["userData"].get("personalityType", "")
        structured_linkedin_profile = db["userData"].get("linkedInProfile", {})
        reddit_profile = db["userData"].get("redditProfile", [])
        twitter_profile = db["userData"].get("twitterProfile", [])
        input_dir = "../input"
        os.makedirs(input_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            os.remove(os.path.join(input_dir, file))

        trait_descriptions = []
        for trait in personality_type:
            if trait in PERSONALITY_DESCRIPTIONS:
                description = f"{trait}: {PERSONITY_DESCRIPTIONS[trait]}"
                trait_descriptions.append(description)
                filename = f"{username.lower()}_{trait.lower()}.txt"
                summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": description})
                with open(os.path.join(input_dir, filename), "w", encoding="utf-8") as file:
                    file.write(summarized_paragraph)

        unified_personality_description = f"{username}'s Personality:\n\n" + "\n".join(trait_descriptions)
        
        if structured_linkedin_profile:
            linkedin_file = os.path.join(input_dir, f"{username.lower()}_linkedin_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": structured_linkedin_profile})
            with open(linkedin_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)
        
        if reddit_profile:
            reddit_file = os.path.join(input_dir, f"{username.lower()}_reddit_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": "Interests: " + ",".join(reddit_profile)})
            with open(reddit_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)
        
        if twitter_profile:
            twitter_file = os.path.join(input_dir, f"{username.lower()}_twitter_profile.txt")
            summarized_paragraph = text_summarizer_runnable.invoke({"user_name": username, "text": "Interests: " + ",".join(twitter_profile)})
            with open(twitter_file, "w", encoding="utf-8") as file:
                file.write(summarized_paragraph)

        return JSONResponse(status_code=200, content={"message": "Documents created successfully", "personality": unified_personality_description})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/customize-graph", status_code=200)
async def customize_graph(request: GraphRequest):
    """Customizes the knowledge graph with new information."""
    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        username = db["userData"]["personalInfo"]["name"]
        points = fact_extraction_runnable.invoke({"paragraph": request.information, "username": username})
        for point in points:
            crud_graph_operations(
                point, graph_driver, embed_model, query_classification_runnable,
                information_extraction_runnable, graph_analysis_runnable,
                graph_decision_runnable, text_description_runnable
            )
        return JSONResponse(status_code=200, content={"message": "Graph customized successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# --- Run the Application ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False, workers=1)