import os
import sys
import httpx
import time
from typing import Union
import uvicorn
import asyncio
import psutil
import subprocess
import json
from helpers import *
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, List, Any, AsyncGenerator
import multiprocessing
import traceback
import sys
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

# --- Service Configuration ---
# Mapping of service categories to their respective port environment variables.
CATEGORY_SERVERS: Dict[str, str] = {
    "chat": os.getenv("CHAT_SERVER_PORT"),
    "memory": os.getenv("MEMORY_SERVER_PORT"),
    "agents": os.getenv("AGENTS_SERVER_PORT"),
    "scraper": os.getenv("SCRAPER_SERVER_PORT"),
    "utils": os.getenv("UTILS_SERVER_PORT"),
    "common": os.getenv("COMMON_SERVER_PORT"),
}

# Mapping of port environment variables to their executable file names (Windows .exe files).
SERVICE_MODULES: Dict[str, str] = {
    "AGENTS_SERVER_PORT": "agents.exe",
    "MEMORY_SERVER_PORT": "memory.exe",
    "CHAT_SERVER_PORT": "chat.exe",
    "SCRAPER_SERVER_PORT": "scraper.exe",
    "UTILS_SERVER_PORT": "utils.exe",
    "COMMON_SERVER_PORT": "common.exe",
}

# --- FastAPI Application ---
app = FastAPI(
    title="Orchestrator API",
    description="Orchestrates different services to provide a seamless AI experience.",
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
# Define Pydantic models for request validation and data structure.


class Message(BaseModel):
    """BaseModel for chat messages."""

    input: str
    pricing: str
    credits: int
    chat_id: str


class ChatId(BaseModel):
    """BaseModel for chat IDs."""

    id: str


class InternetSearchRequest(BaseModel):
    """BaseModel for internet search requests."""

    query: str


class ContextClassificationRequest(BaseModel):
    """BaseModel for context classification requests."""

    query: str
    context: str


class ElaboratorMessage(BaseModel):
    """BaseModel for elaborator messages."""

    input: str
    purpose: str


class DeleteSubgraphRequest(BaseModel):
    """BaseModel for delete subgraph requests."""

    source: str


class GraphRequest(BaseModel):
    """BaseModel for generic graph requests."""

    information: str


class RedditURL(BaseModel):
    """BaseModel for Reddit URL requests."""

    url: str


class TwitterURL(BaseModel):
    """BaseModel for Twitter URL requests."""

    url: str


class Profile(BaseModel):
    """BaseModel for social media profile URLs."""

    url: str


class EncryptionRequest(BaseModel):
    """BaseModel for encryption requests."""

    data: str


class DecryptionRequest(BaseModel):
    """BaseModel for decryption requests."""

    encrypted_data: str


class UserInfoRequest(BaseModel):
    """BaseModel for user info requests (user_id)."""

    user_id: str


class ReferrerStatusRequest(BaseModel):
    """BaseModel for referrer status update requests."""

    user_id: str
    referrer_status: bool


class SetReferrerRequest(BaseModel):
    """BaseModel for setting referrer using referral code."""

    referral_code: str


class GraphRAGRequest(BaseModel):
    """BaseModel for GraphRAG (Graph-based Retrieval Augmented Generation) requests."""

    query: str


class InternetSearchRequest(BaseModel):
    """BaseModel for internet search requests."""

    query: str


class ContextClassificationRequest(BaseModel):
    """BaseModel for context classification requests."""

    query: str
    context: str
    
class CustomRAGRequest(BaseModel):
    """BaseModel for GraphRAG (Graph-based Retrieval Augmented Generation) requests."""

    query: str


class BetaUserStatusRequest(BaseModel):
    """BaseModel for beta user status update requests."""

    user_id: str
    beta_user_status: bool


# --- Global Variables ---
# Global variables to maintain chat context and runnables.
chat_id: Optional[str] = (
    None  # Global variable to store the current chat ID, initialized to None
)
chat_history = (
    None  # Placeholder for chat history object, currently unused in this orchestrator
)
chat_runnable = (
    None  # Placeholder for chat runnable, currently unused in this orchestrator
)
orchestrator_runnable = None  # Placeholder for orchestrator runnable, currently unused
context_classification_runnable = (
    None  # Placeholder for context classification runnable
)
internet_search_runnable = None  # Placeholder for internet search runnable
internet_query_reframe_runnable = (
    None  # Placeholder for internet query reframe runnable
)
internet_summary_runnable = None  # Placeholder for internet summary runnable


# --- Asyncio Integration ---
nest_asyncio.apply()  # Apply nest_asyncio to allow nested asyncio event loops


# --- Server State Management Functions ---
# Functions to manage the lifecycle of backend services (start, check status, stop).


async def is_server_live(port: str) -> bool:
    """
    Checks if a server is live at the given port by sending a GET request to the root URL.

    Args:
        port (str): The port number where the server is expected to be running.

    Returns:
        bool: True if the server responds with a 200 status code, False otherwise.
    """
    url = f"http://localhost:{port}/"  # Construct URL to check server liveness
    try:
        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create async HTTP client with no timeout
            response = await client.get(
                url, timeout=None
            )  # Send GET request to server root
            return (
                response.status_code == 200
            )  # Return True if status code is 200 (OK), False otherwise
    except httpx.ConnectError:  # Handle connection errors (server not reachable)
        return False  # Server is not live if connection error occurs
    except Exception as e:  # Catch any other exceptions during server liveness check
        print(f"Error checking server liveness at port {port}: {e}")
        return False  # Server is considered not live in case of any error


async def spawn_server(port_env_var: str) -> bool:
    """
    Spawns a server as a Windows executable if it's not already live.

    This function checks if a server is already running on the specified port.
    If not, it attempts to start the server by launching a Windows executable defined in SERVICE_MODULES.

    Args:
        port_env_var (str): Environment variable name that holds the port number for the service.

    Returns:
        bool: True if the server is live after checking or spawning, False if spawning fails or times out.
    """
    port: Optional[str] = os.environ.get(
        port_env_var
    )  # Get port number from environment variable
    exe_file: Optional[str] = SERVICE_MODULES.get(
        port_env_var
    )  # Get executable file name from SERVICE_MODULES mapping

    if not port:  # Check if port is defined
        print(f"{port_env_var} not defined in environment variables.")
        return False  # Return False if port is not defined
    if not exe_file:  # Check if executable file is defined
        print(f"No executable defined for {port_env_var}.")
        return False  # Return False if executable file is not defined

    if await is_server_live(port):  # Check if server is already live
        print(f"Server at port {port} is already live.")
        return True  # Return True if server is already live

    print(f"Spawning server for {port_env_var} on port {port}...")
    print(
        f"Starting server with: {os.path.join(os.path.dirname(sys.executable), exe_file)}"
    )

    try:
        process: subprocess.Popen = subprocess.Popen(  # Start server as a subprocess
            [
                os.path.join(os.path.dirname(sys.executable), exe_file)
            ],  # Path to the executable
            env=os.environ.copy(),  # Inherit environment variables
            cwd=os.path.dirname(
                sys.executable
            ),  # Set current working directory to executable's directory
            creationflags=subprocess.CREATE_NO_WINDOW,  # Run in background without a console window (Windows specific)
        )

        start_time: float = time.time()  # Record start time for timeout
        timeout: int = 120  # Set timeout for server to become live (seconds)

        while (
            time.time() - start_time < timeout
        ):  # Wait for server to become live within timeout period
            if await is_server_live(port):  # Check if server is live
                print(f"Server on port {port} spawned successfully.")
                return True  # Return True if server spawned successfully and is live
            await asyncio.sleep(2)  # Wait for 2 seconds before checking again

        print(f"Server on port {port} did not become live within {timeout} seconds.")
        return False  # Return False if server did not become live within timeout

    except Exception as e:  # Catch any exceptions during server spawning
        print(f"Error spawning server on port {port}: {e}")
        return False  # Return False if server spawning fails


async def stop_server(port_env_var: str) -> bool:
    """
    Stops a server running as a Windows executable based on the port and executable path.

    This function iterates through running processes to find and terminate a server
    process that matches the executable path defined for the given port environment variable.

    Args:
        port_env_var (str): Environment variable name that holds the port number for the service to stop.

    Returns:
        bool: True if the server was successfully stopped or if no server was found running, False if an error occurred during termination.
    """
    port: Optional[str] = os.environ.get(
        port_env_var
    )  # Get port number from environment variable
    exe_file: Optional[str] = SERVICE_MODULES.get(
        port_env_var
    )  # Get executable file name from SERVICE_MODULES mapping

    if not port:  # Check if port is defined
        print(f"{port_env_var} not defined in environment variables.")
        return False  # Return False if port is not defined
    if not exe_file:  # Check if executable file is defined
        print(f"No executable defined for {port_env_var}.")
        return False  # Return False if executable file is not defined

    if not await is_server_live(port):  # Check if server is live
        print(f"No live server found on port {port}.")
        return True  # Return True if no live server found (considered stopped)

    target_path: str = os.path.join(
        os.path.dirname(sys.executable), exe_file
    )  # Construct full path to executable

    try:
        print(f"Stopping server on port {port} (Path: {target_path})...")

        for proc in psutil.process_iter(
            attrs=["pid", "name", "exe"]
        ):  # Iterate through running processes
            if (
                proc.info["exe"]
                and os.path.normcase(proc.info["exe"]) == os.path.normcase(target_path)
            ):  # Check if process executable path matches target path (case-insensitive)
                proc.terminate()  # Terminate the process
                proc.wait(timeout=30)  # Wait for process to terminate, with timeout
                print(f"Server on port {port} stopped successfully.")
                return True  # Return True if server stopped successfully

        print(f"No running process found for {target_path}.")
        return False  # Return False if no matching process found

    except Exception as e:  # Catch any exceptions during server stopping
        print(f"Error stopping server on port {port}: {e}")
        return False  # Return False if error occurred during server stopping


# --- Service Call Functions ---
# Functions to call and manage interactions with backend services (initiate, call endpoints).


async def call_service_initiate(port_env_var: str) -> JSONResponse:
    """
    Calls the '/initiate' endpoint of a service after ensuring it's live.

    This function first ensures that the service at the given port is live,
    spawning it if necessary. Then, it calls the '/initiate' endpoint of the service.

    Args:
        port_env_var (str): Environment variable name that holds the port number of the service.

    Returns:
        JSONResponse: A FastAPI JSONResponse object containing the service's response,
                      or an error message if initiation fails.
    """
    port: Optional[str] = os.environ.get(
        port_env_var
    )  # Get port number from environment variable
    if not port:  # Check if port is defined
        print(f"{port_env_var} not configured.")
        return JSONResponse(
            status_code=500, content={"message": f"{port_env_var} not configured."}
        )  # Return error response if port is not configured

    if not await spawn_server(port_env_var):  # Ensure the server is spawned and live
        print(f"Failed to spawn service on port {port}.")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to start service on port {port}."},
        )  # Return error response if server spawning fails

    initiate_url: str = (
        f"http://localhost:{port}/initiate"  # Construct initiate endpoint URL
    )
    try:
        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create async HTTP client with no timeout
            response: httpx.Response = await client.post(
                initiate_url
            )  # Send POST request to initiate endpoint
            if response.status_code == 200:  # Check if request was successful
                print(f"Service on port {port} initiated successfully.")
                return JSONResponse(
                    status_code=200, content=response.json()
                )  # Return success response with service response
            else:  # Handle non-200 status codes
                print(f"Error initiating service on port {port}: {response.text}")
                return JSONResponse(
                    status_code=response.status_code, content={"message": response.text}
                )  # Return error response with service error message
    except Exception as e:  # Catch any exceptions during service initiation call
        print(f"Error calling initiate endpoint at port {port}: {e}")
        return JSONResponse(
            status_code=500, content={"message": f"Error initiating service: {str(e)}"}
        )  # Return error response with exception message


async def call_service_endpoint(
    port_env_var: str, endpoint: str, payload: Optional[dict] = None
) -> JSONResponse:
    """
    Calls a specific endpoint of a service, ensuring the service is live and initiated.

    This function improves upon `call_service_initiate` by adding retry logic for service initiation
    and more robust error handling for both service availability and HTTP requests.

    Args:
        port_env_var (str): Environment variable name for the service's port.
        endpoint (str): The API endpoint to call on the service (e.g., "/chat", "/graphrag").
        payload (Optional[dict]): Optional dictionary payload to send with the POST request as JSON.

    Returns:
        JSONResponse: A FastAPI JSONResponse object containing the service's response,
                      or a detailed error response if the call fails at any stage.
    """
    port: Optional[str] = os.environ.get(
        port_env_var
    )  # Get port number from environment variable
    if not port:  # Check if port is configured
        return JSONResponse(
            status_code=500,
            content={
                "message": f"{port_env_var} not configured."
            },  # Return error response if port is not configured
        )

    try:
        server_available: bool = (
            await asyncio.wait_for(  # Wait for server to be available, with timeout
                spawn_server(
                    port_env_var
                ),  # Attempt to spawn server if not already running
                timeout=None,  # No timeout for server spawning
            )
        )
        if (
            not server_available
        ):  # Check if server is available after attempting to spawn
            return JSONResponse(
                status_code=503,
                content={
                    "message": f"Service on port {port} unavailable after startup attempt"
                },  # Return error response if server is unavailable
            )

        max_retries: int = 3  # Set maximum number of retries for service initiation
        for attempt in range(max_retries):  # Retry loop for service initiation
            initiate_response: JSONResponse = await call_service_initiate(
                port_env_var
            )  # Call service initiate endpoint
            if (
                initiate_response.status_code == 200
            ):  # Check if initiate call was successful
                break  # Break retry loop if initiation is successful
            await asyncio.sleep(
                1 + attempt * 2
            )  # Wait before retrying, with increasing backoff
        else:  # Else block executed if loop completes without break (all retries failed)
            print(
                f"Failed to initiate service on port {port} after {max_retries} attempts"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "message": f"Service initiation failed after {max_retries} attempts"
                },  # Return error response if initiation fails after retries
            )

        service_url: str = (
            f"http://localhost:{port}{endpoint}"  # Construct full service endpoint URL
        )

        async with httpx.AsyncClient(
            timeout=None
        ) as client:  # Create async HTTP client with no timeout
            response: httpx.Response = (
                await client.post(  # Send POST request to service endpoint
                    service_url,
                    json=payload,  # Payload for the request
                    headers={
                        "Content-Type": "application/json"
                    },  # Set JSON content type header
                )
            )

            try:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return JSONResponse(  # Return success JSONResponse with content from service
                    status_code=response.status_code, content=response.json()
                )
            except httpx.HTTPStatusError as e:  # Catch HTTP status errors
                print(f"Service returned error: {e.response.text}")
                return JSONResponse(  # Return error JSONResponse with details from service error
                    status_code=e.response.status_code,
                    content={"message": f"Service returned error: {str(e)}"},
                )

    except asyncio.TimeoutError:  # Handle asyncio timeout errors
        print(f"Service timeout")
        return JSONResponse(
            status_code=504,
            content={"message": f"Service timeout"},  # Return timeout error response
        )
    except httpx.RequestError as e:  # Handle httpx request errors (network issues)
        print(f"Network error calling {endpoint}: {e}")
        return JSONResponse(
            status_code=502,
            content={
                "message": f"Network error: {str(e)}"
            },  # Return network error response
        )
    except Exception as e:  # Catch any other unexpected exceptions
        print(f"Critical error calling {endpoint}: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal server error"
            },  # Return internal server error response
        )


async def fetch_streaming_response(url: str, payload: dict) -> StreamingResponse:
    """
    Proxies a streaming response from a backend service to the client.

    This function sets up an asynchronous generator to stream data from a backend service endpoint
    and forwards it as a StreamingResponse. It handles JSON decoding and yields JSON formatted lines.

    Args:
        url (str): The full URL of the streaming endpoint on the backend service.
        payload (dict): The dictionary payload to send as JSON in the POST request to the streaming endpoint.

    Returns:
        StreamingResponse: A FastAPI StreamingResponse that streams data from the backend service.
    """

    async def generate() -> AsyncGenerator[str, None]:
        """
        Asynchronous generator that streams data from the backend service.

        Yields:
            str: JSON-formatted string representing a line of data from the streaming response.
                 In case of errors, yields JSON-formatted error messages.
        """
        try:
            async with httpx.AsyncClient(
                timeout=None
            ) as client:  # Create async HTTP client with no timeout
                async with client.stream(
                    "POST", url, json=payload
                ) as response:  # Open a streaming POST request
                    async for line in (
                        response.aiter_lines()
                    ):  # Asynchronously iterate over lines in the response
                        if line:  # Check if line is not empty
                            try:
                                data: Dict[str, Any] = json.loads(
                                    line
                                )  # Parse each line as JSON
                                yield (
                                    json.dumps(data) + "\n"
                                )  # Yield JSON data as a string, with newline for streaming
                                await asyncio.sleep(
                                    0.05
                                )  # Small delay to control stream rate
                            except (
                                json.JSONDecodeError
                            ):  # Handle JSON decode errors (malformed JSON)
                                continue  # Skip to the next line if JSON decode fails

        except httpx.HTTPStatusError as e:  # Handle HTTP status errors during streaming
            error_msg: str = f"HTTP error {e.response.status_code}: {e.response.text}"  # Format error message
            print(f"HTTP error: {error_msg}")
            yield (
                json.dumps({"type": "error", "message": error_msg}) + "\n"
            )  # Yield JSON error message
        except Exception as e:  # Catch any other exceptions during streaming
            error_msg: str = (
                f"Connection error: {str(e)}"  # Format connection error message
            )
            print(f"Connection error: {error_msg}")
            yield (
                json.dumps({"type": "error", "message": error_msg}) + "\n"
            )  # Yield JSON error message

    return StreamingResponse(
        generate(), media_type="application/json"
    )  # Return StreamingResponse with the generator


async def stream_yield(data: dict) -> StreamingResponse:
    """
    Yields a single streaming JSON response for any provided data.

    This is a utility function to quickly create a StreamingResponse that yields a single JSON object.
    Useful for immediate, non-streaming responses that need to be in a streaming format for consistency.

    Args:
        data (dict): The dictionary data to be yielded as a single JSON object.

    Returns:
        StreamingResponse: A FastAPI StreamingResponse that yields the provided data as JSON.
    """

    async def generate() -> AsyncGenerator[str, None]:
        """
        Asynchronous generator that yields the provided data as JSON.

        Yields:
            str: JSON-formatted string of the provided data, followed by a newline.
        """
        yield json.dumps(data) + "\n"  # Yield data as JSON string with newline
        await asyncio.sleep(
            0.05
        )  # Small delay, though likely unnecessary for single yield

    return StreamingResponse(
        generate(), media_type="application/json"
    )  # Return StreamingResponse for the data


# --- API Endpoints ---
# Define FastAPI endpoints for the orchestrator service.


@app.get("/", status_code=200)
async def main() -> Dict[str, str]:
    """
    Root endpoint of the orchestrator API.

    Returns:
        JSONResponse: A simple greeting message in JSON format.
    """
    return {
        "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
    }  # Return a greeting message


@app.post("/initiate", status_code=200)
async def initiate() -> JSONResponse:
    """
    Initiate the orchestration server itself.

    This endpoint currently just returns a success message, as the orchestrator
    doesn't have a complex initiation process beyond starting the FastAPI app.

    Returns:
        JSONResponse: Success message indicating the orchestrator has been initiated.
    """
    print("Orchestration Model initiated successfully")
    return JSONResponse(
        status_code=200,
        content={"message": "Orchestration Model initiated successfully"},
    )  # Return success message


@app.post("/set-chat", status_code=200)
async def set_chat(id: ChatId) -> JSONResponse:
    """
    Set the current chat ID for the session.

    This endpoint allows clients to set a chat ID, which can be used to maintain
    context across chat messages.

    Args:
        id (ChatId): Pydantic model containing the chat ID to set.

    Returns:
        JSONResponse: Success message indicating the chat ID has been set,
                      or an error message if setting the chat ID fails.
    """
    global chat_id  # Access the global chat_id variable
    try:
        chat_id = id.id  # Set the global chat_id to the ID provided in the request
        print(f"Chat set to {chat_id}")
        return JSONResponse(
            status_code=200, content={"message": "Chat set successfully"}
        )  # Return success message
    except Exception as e:  # Catch any exceptions during chat ID setting
        print(f"Error in set-chat: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error response with exception message


@app.post("/initiate-agents", status_code=200)
async def initiate_agents() -> JSONResponse:
    """
    Endpoint to initiate the Agent Service.

    Forwards the initiation request to the Agent Service and returns its response.

    Returns:
        JSONResponse: Response from the Agent Service's initiate endpoint.
    """
    return await call_service_initiate(
        "AGENTS_SERVER_PORT"
    )  # Call Agent Service initiate endpoint


@app.post("/initiate-memory", status_code=200)
async def initiate_memory() -> JSONResponse:
    """
    Endpoint to initiate the Memory Service.

    Forwards the initiation request to the Memory Service and returns its response.

    Returns:
        JSONResponse: Response from the Memory Service's initiate endpoint.
    """
    return await call_service_initiate(
        "MEMORY_SERVER_PORT"
    )  # Call Memory Service initiate endpoint


@app.post("/initiate-chat", status_code=200)
async def initiate_chat() -> JSONResponse:
    """
    Endpoint to initiate the Chat Service.

    Forwards the initiation request to the Chat Service and returns its response.

    Returns:
        JSONResponse: Response from the Chat Service's initiate endpoint.
    """
    return await call_service_initiate(
        "CHAT_SERVER_PORT"
    )  # Call Chat Service initiate endpoint


@app.post("/initiate-scraper", status_code=200)
async def initiate_scraper() -> JSONResponse:
    """
    Endpoint to initiate the Scraper Service.

    Forwards the initiation request to the Scraper Service and returns its response.

    Returns:
        JSONResponse: Response from the Scraper Service's initiate endpoint.
    """
    return await call_service_initiate(
        "SCRAPER_SERVER_PORT"
    )  # Call Scraper Service initiate endpoint


@app.post("/initiate-utils", status_code=200)
async def initiate_utils() -> JSONResponse:
    """
    Endpoint to initiate the Utils Service.

    Forwards the initiation request to the Utils Service and returns its response.

    Returns:
        JSONResponse: Response from the Utils Service's initiate endpoint.
    """
    return await call_service_initiate(
        "UTILS_SERVER_PORT"
    )  # Call Utils Service initiate endpoint


@app.post("/elaborator", status_code=200)
async def elaborate(message: ElaboratorMessage) -> JSONResponse:
    """
    Endpoint to proxy elaboration requests to the Agent Service.

    Forwards elaboration requests to the Agent Service and returns its response.
    The elaborator service is now integrated within the Agent Service.

    Args:
        message (ElaboratorMessage): Request body containing the input for elaboration.

    Returns:
        JSONResponse: Response from the Agent Service's elaborator endpoint.
    """
    payload: Dict[str, str] = (
        message.model_dump()
    )  # Extract payload from ElaboratorMessage model
    return await call_service_endpoint(
        "AGENTS_SERVER_PORT", "/elaborator", payload
    )  # Call Agent Service elaborator endpoint


@app.post("/create-graph", status_code=200)
async def create_graph() -> JSONResponse:
    """
    Endpoint to proxy graph creation requests to the Memory Service.

    Forwards requests to create a new graph to the Memory Service and returns its response.
    The graph creation service is now part of the Memory Service.

    Returns:
        JSONResponse: Response from the Memory Service's create-graph endpoint.
    """
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/create-graph"
    )  # Call Memory Service create-graph endpoint


@app.post("/delete-subgraph", status_code=200)
async def delete_subgraph(request: DeleteSubgraphRequest) -> JSONResponse:
    """
    Endpoint to proxy subgraph deletion requests to the Memory Service.

    Forwards requests to delete a subgraph to the Memory Service and returns its response.
    The delete subgraph service is part of the Memory Service.

    Args:
        request (DeleteSubgraphRequest): Request body specifying the source of the subgraph to delete.

    Returns:
        JSONResponse: Response from the Memory Service's delete-subgraph endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from DeleteSubgraphRequest model
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/delete-subgraph", payload
    )  # Call Memory Service delete-subgraph endpoint


@app.post("/create-document", status_code=200)
async def create_document() -> JSONResponse:
    """
    Endpoint to proxy document creation requests to the Memory Service.

    Forwards requests to create a new document to the Memory Service and returns its response.
    The create document service is now part of the Memory Service.

    Returns:
        JSONResponse: Response from the Memory Service's create-document endpoint.
    """
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/create-document"
    )  # Call Memory Service create-document endpoint


@app.post("/scrape-linkedin", status_code=200)
async def scrape_linkedin(profile: Profile) -> JSONResponse:
    """
    Endpoint to proxy LinkedIn scraping requests to the Scraper Service.

    Forwards requests to scrape LinkedIn profiles to the Scraper Service and returns its response.

    Args:
        profile (Profile): Request body containing the LinkedIn profile URL to scrape.

    Returns:
        JSONResponse: Response from the Scraper Service's scrape-linkedin endpoint.
    """
    payload: Dict[str, str] = profile.model_dump()  # Extract payload from Profile model
    return await call_service_endpoint(
        "SCRAPER_SERVER_PORT", "/scrape-linkedin", payload
    )  # Call Scraper Service scrape-linkedin endpoint


@app.post("/customize-graph", status_code=200)
async def customize_graph(request: GraphRequest) -> JSONResponse:
    """
    Endpoint to proxy graph customization requests to the Memory Service.

    Forwards requests to customize the graph to the Memory Service and returns its response.
    The customize graph service is now part of the Memory Service.

    Args:
        request (GraphRequest): Request body containing information for graph customization.

    Returns:
        JSONResponse: Response from the Memory Service's customize-graph endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from GraphRequest model
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/customize-graph", payload
    )  # Call Memory Service customize-graph endpoint


@app.post("/scrape-reddit")
async def scrape_reddit(reddit_url: RedditURL) -> JSONResponse:
    """
    Endpoint to proxy Reddit scraping requests to the Scraper Service.

    Forwards requests to scrape Reddit URLs to the Scraper Service and returns its response.

    Args:
        reddit_url (RedditURL): Request body containing the Reddit URL to scrape.

    Returns:
        JSONResponse: Response from the Scraper Service's scrape-reddit endpoint.
    """
    payload: Dict[str, str] = (
        reddit_url.model_dump()
    )  # Extract payload from RedditURL model
    return await call_service_endpoint(
        "SCRAPER_SERVER_PORT", "/scrape-reddit", payload
    )  # Call Scraper Service scrape-reddit endpoint


@app.post("/scrape-twitter")
async def scrape_twitter(twitter_url: TwitterURL) -> JSONResponse:
    """
    Endpoint to proxy Twitter scraping requests to the Scraper Service.

    Forwards requests to scrape Twitter URLs to the Scraper Service and returns its response.

    Args:
        twitter_url (TwitterURL): Request body containing the Twitter URL to scrape.

    Returns:
        JSONResponse: Response from the Scraper Service's scrape-twitter endpoint.
    """
    payload: Dict[str, str] = (
        twitter_url.model_dump()
    )  # Extract payload from TwitterURL model
    return await call_service_endpoint(
        "SCRAPER_SERVER_PORT", "/scrape-twitter", payload
    )  # Call Scraper Service scrape-twitter endpoint


@app.post("/get-role")
async def get_role_endpoint(request: UserInfoRequest) -> JSONResponse:
    """
    Endpoint to proxy user role retrieval requests to the Utils Service.

    Forwards requests to get user roles to the Utils Service and returns its response.

    Args:
        request (UserInfoRequest): Request body containing the user ID for role retrieval.

    Returns:
        JSONResponse: Response from the Utils Service's get-role endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from UserInfoRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-role", payload
    )  # Call Utils Service get-role endpoint


@app.post("/get-referral-code")
async def get_referral_code_endpoint(request: UserInfoRequest) -> JSONResponse:
    """
    Endpoint to proxy referral code retrieval requests to the Utils Service.

    Forwards requests to get referral codes to the Utils Service and returns its response.

    Args:
        request (UserInfoRequest): Request body containing the user ID for referral code retrieval.

    Returns:
        JSONResponse: Response from the Utils Service's get-referral-code endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from UserInfoRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-referral-code", payload
    )  # Call Utils Service get-referral-code endpoint


@app.post("/get-referrer-status")
async def get_referrer_status_endpoint(request: UserInfoRequest) -> JSONResponse:
    """
    Endpoint to proxy referrer status retrieval requests to the Utils Service.

    Forwards requests to get referrer status to the Utils Service and returns its response.

    Args:
        request (UserInfoRequest): Request body containing the user ID for referrer status retrieval.

    Returns:
        JSONResponse: Response from the Utils Service's get-referrer-status endpoint.
    """
    payload: Dict[str, bool] = (
        request.model_dump()
    )  # Extract payload from UserInfoRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-referrer-status", payload
    )  # Call Utils Service get-referrer-status endpoint


@app.post("/set-referrer-status")
async def set_referrer_status_endpoint(request: ReferrerStatusRequest) -> JSONResponse:
    """
    Endpoint to proxy referrer status update requests to the Utils Service.

    Forwards requests to set referrer status to the Utils Service and returns its response.

    Args:
        request (ReferrerStatusRequest): Request body containing user ID and new referrer status.

    Returns:
        JSONResponse: Response from the Utils Service's set-referrer-status endpoint.
    """
    payload: Dict[str, Union[str, bool]] = (
        request.model_dump()
    )  # Extract payload from ReferrerStatusRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/set-referrer-status", payload
    )  # Call Utils Service set-referrer-status endpoint


@app.post("/get-user-and-set-referrer-status")
async def get_user_and_set_referrer_status_endpoint(
    request: SetReferrerRequest,
) -> JSONResponse:
    """
    Endpoint to proxy user retrieval and referrer status setting requests to the Utils Service.

    Forwards requests to get user info and set referrer status based on referral code to the Utils Service and returns its response.

    Args:
        request (SetReferrerRequest): Request body containing referral code to set referrer status.

    Returns:
        JSONResponse: Response from the Utils Service's get-user-and-set-referrer-status endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from SetReferrerRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-user-and-set-referrer-status", payload
    )  # Call Utils Service get-user-and-set-referrer-status endpoint


@app.post("/get-user-and-invert-beta-user-status")
async def get_user_and_invert_beta_user_status_endpoint(
    request: UserInfoRequest,
) -> JSONResponse:
    """
    Endpoint to proxy user retrieval and beta user status inversion requests to the Utils Service.

    Forwards requests to get user info and invert beta user status to the Utils Service and returns its response.

    Args:
        request (UserInfoRequest): Request body containing user ID for beta user status inversion.

    Returns:
        JSONResponse: Response from the Utils Service's get-user-and-invert-beta-user-status endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from UserInfoRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-user-and-invert-beta-user-status", payload
    )  # Call Utils Service get-user-and-invert-beta-user-status endpoint


@app.post("/set-beta-user-status")
async def set_beta_user_status_endpoint(request: BetaUserStatusRequest) -> JSONResponse:
    """
    Endpoint to proxy beta user status update requests to the Utils Service.

    Forwards requests to set beta user status to the Utils Service and returns its response.

    Args:
        request (BetaUserStatusRequest): Request body containing user ID and new beta user status.

    Returns:
        JSONResponse: Response from the Utils Service's set-beta-user-status endpoint.
    """
    payload: Dict[str, Union[str, bool]] = (
        request.model_dump()
    )  # Extract payload from BetaUserStatusRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/set-beta-user-status", payload
    )  # Call Utils Service set-beta-user-status endpoint


@app.post("/get-beta-user-status")
async def get_beta_user_status_endpoint(request: UserInfoRequest) -> JSONResponse:
    """
    Endpoint to proxy beta user status retrieval requests to the Utils Service.

    Forwards requests to get beta user status to the Utils Service and returns its response.

    Args:
        request (UserInfoRequest): Request body containing user ID for beta user status retrieval.

    Returns:
        JSONResponse: Response from the Utils Service's get-beta-user-status endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from UserInfoRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/get-beta-user-status", payload
    )  # Call Utils Service get-beta-user-status endpoint


@app.post("/encrypt")
async def encrypt_endpoint(request: EncryptionRequest) -> JSONResponse:
    """
    Endpoint to proxy encryption requests to the Utils Service.

    Forwards requests to encrypt data to the Utils Service and returns its response.

    Args:
        request (EncryptionRequest): Request body containing data to be encrypted.

    Returns:
        JSONResponse: Response from the Utils Service's encrypt endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from EncryptionRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/encrypt", payload
    )  # Call Utils Service encrypt endpoint


@app.post("/decrypt")
async def decrypt_endpoint(request: DecryptionRequest) -> JSONResponse:
    """
    Endpoint to proxy decryption requests to the Utils Service.

    Forwards requests to decrypt data to the Utils Service and returns its response.

    Args:
        request (DecryptionRequest): Request body containing encrypted data to be decrypted.

    Returns:
        JSONResponse: Response from the Utils Service's decrypt endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from DecryptionRequest model
    return await call_service_endpoint(
        "UTILS_SERVER_PORT", "/decrypt", payload
    )  # Call Utils Service decrypt endpoint


@app.post("/graphrag")
async def graphrag_endpoint(request: GraphRAGRequest) -> JSONResponse:
    """
    Endpoint to proxy GraphRAG requests to the Memory Service.

    Forwards requests for graph-based retrieval-augmented generation to the Memory Service and returns its streaming response.

    Args:
        request (GraphRAGRequest): Request body containing the query for GraphRAG.

    Returns:
        StreamingResponse: Streaming response from the Memory Service's graphrag endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from GraphRAGRequest model
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/graphrag", payload
    )  # Fetch and return streaming response


@app.post("/internet-search")
async def internet_search_endpoint(request: InternetSearchRequest) -> JSONResponse:
    """
    Endpoint to proxy internet search requests to the Common Service.

    Forwards requests for internet searches to the Common Service and returns its response.

    Args:
        request (InternetSearchRequest): Request body containing the query for internet search.

    Returns:
        JSONResponse: Response from the Common Service's internet-search endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from InternetSearchRequest model
    return await call_service_endpoint(
        "COMMON_SERVER_PORT", "/internet-search", payload
    )  # Call Common Service internet-search endpoint


@app.post("/context-classify")
async def context_classify_endpoint(
    request: ContextClassificationRequest,
) -> JSONResponse:
    """
    Endpoint to proxy context classification requests to the Common Service.

    Forwards requests to classify context to the Common Service and returns its response.

    Args:
        request (ContextClassificationRequest): Request body containing query and context for classification.

    Returns:
        JSONResponse: Response from the Common Service's context-classify endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from ContextClassificationRequest model
    return await call_service_endpoint(
        "COMMON_SERVER_PORT", "/context-classify", payload
    )  # Call Common Service context-classify endpoint
    
@app.post("/refresh-rag-context")
async def refresh_rag_context():
    try:
        response = await call_service_endpoint("MEMORY_SERVER_PORT", "/refresh-rag-context")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing RAG context: {str(e)}")
    
@app.post("/custom-rag")
async def custom_rag_endpoint(request: CustomRAGRequest) -> JSONResponse:
    """
    Endpoint to proxy CustomRAG requests to the Memory Service.

    Forwards requests for text based retrieval-augmented generation to the Memory Service and returns its streaming response.

    Args:
        request (CustomRAGRequest): Request body containing the query for CustomRAG.

    Returns:
        StreamingResponse: Streaming response from the Memory Service's custom-rag endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from GraphRAGRequest model
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/custom-rag", payload
    )  # Fetch and return streaming response

@app.post("/temp-rag")
async def custom_rag_endpoint(request: CustomRAGRequest) -> JSONResponse:
    """
    Endpoint to proxy TempRAG requests to the Memory Service.

    Forwards requests for text based retrieval-augmented generation to the Memory Service and returns its streaming response.

    Args:
        request (TempRAGRequest): Request body containing the query for TempRAG.

    Returns:
        StreamingResponse: Streaming response from the Memory Service's custom-rag endpoint.
    """
    payload: Dict[str, str] = (
        request.model_dump()
    )  # Extract payload from GraphRAGRequest model
    return await call_service_endpoint(
        "MEMORY_SERVER_PORT", "/temp-rag", payload
    )  # Fetch and return streaming response
    
@app.post("/chat", status_code=200)
async def chat(message: Message) -> StreamingResponse:
    """
    Main chat endpoint to route chat requests to the appropriate service based on classification.

    This endpoint classifies the chat message using the Common Service and then routes the request
    to either the Chat, Memory, or Agent Service based on the classification result.
    It returns a StreamingResponse, proxying the streaming output from the relevant service.

    Args:
        message (Message): Request body containing the chat message and related context.

    Returns:
        StreamingResponse: Streaming response from the Chat, Memory, or Agent Service,
                           depending on the classification of the chat message.

    Raises:
        HTTPException: If chat classification fails or if the determined category is invalid.
    """
    global chat_id  # Access the global chat_id variable

    try:
        response: JSONResponse = await call_service_endpoint(
            "COMMON_SERVER_PORT",
            "/chat-classify",
            {"input": message.input, "chat_id": chat_id},
        )  # Classify chat message category

        if response.status_code != 200:  # Check if chat classification was successful
            print(f"Chat classification failed: {json.loads(response.body)['message']}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Chat classification failed: {json.loads(response.body)['message']}",
            )  # Raise HTTPException if classification fails

        response_data: Dict[str, Any] = json.loads(
            response.body
        )  # Load JSON response from classification service
        category: str = response_data[
            "classification"
        ]  # Extract classification category
        transformed_input: str = response_data[
            "transformed_input"
        ]  # Extract transformed input

        category_port_env_var: Optional[str] = (
            None  # Initialize category port environment variable
        )
        if category == "chat":  # Route to Chat Service if category is 'chat'
            category_port_env_var = "CHAT_SERVER_PORT"
        elif category == "memory":  # Route to Memory Service if category is 'memory'
            category_port_env_var = "MEMORY_SERVER_PORT"
        elif category == "agent":  # Route to Agent Service if category is 'agent'
            category_port_env_var = "AGENTS_SERVER_PORT"
        else:  # Handle invalid categories
            print(f"Invalid category determined by orchestrator: {category}")
            raise HTTPException(
                status_code=400, detail="Invalid category determined by orchestrator"
            )  # Raise HTTPException for invalid category

        category_port: Optional[str] = os.environ.get(
            category_port_env_var
        )  # Get category service port from environment variable
        if not category_port:  # Check if category port is configured
            print(f"{category_port_env_var} not configured.")
            raise HTTPException(
                status_code=500, detail=f"{category_port_env_var} not configured."
            )  # Raise HTTPException if category port is not configured

        if not await spawn_server(
            category_port_env_var
        ):  # Ensure category service is spawned and live
            print(f"Service on port {category_port} is not available.")
            raise HTTPException(
                status_code=500,
                detail=f"Service on port {category_port} is not available.",
            )  # Raise HTTPException if service is not available

        await call_service_initiate(
            category_port_env_var
        )  # Initiate the category service

        category_url: str = f"http://localhost:{category_port}/chat"  # Construct URL for category-specific chat endpoint

        payload: Dict[str, Union[str, int]] = {  # Construct payload for chat endpoint
            "chat_id": chat_id,
            "original_input": message.input,
            "transformed_input": transformed_input,
            "pricing": message.pricing,
            "credits": message.credits,
        }

        return await fetch_streaming_response(
            category_url, payload
        )  # Fetch and return streaming response from category service

    except HTTPException as http_exc:  # Catch HTTPExceptions raised during processing
        print(http_exc)
        return JSONResponse(
            status_code=http_exc.status_code, content={"message": http_exc.detail}
        )  # Return JSONResponse for HTTP Exceptions
    except Exception as e:  # Catch any other exceptions during chat processing
        print(f"Error in chat endpoint: {e}")
        return JSONResponse(
            status_code=500, content={"message": f"Error processing chat: {str(e)}"}
        )  # Return JSONResponse for general exceptions


# --- Server Shutdown Endpoint ---
# Endpoint to gracefully stop backend services when the orchestrator app closes.

SERVERS_TO_STOP_ON_APP_CLOSE: List[str] = [
    "AGENTS_SERVER_PORT",
    "SCRAPER_SERVER_PORT",
    "MEMORY_SERVER_PORT",
    "CHAT_SERVER_PORT",
    "UTILS_SERVER_PORT",
    "COMMON_SERVER_PORT",
]  # List of servers to stop on app close


@app.post("/stop-servers-on-app-close")
async def stop_servers_endpoint() -> JSONResponse:
    """
    Endpoint to stop specific backend servers when the orchestrator application is closing.

    This endpoint iterates through a predefined list of server port environment variables,
    stopping each server asynchronously. It aggregates the results and returns a JSON response
    indicating the success or failure of stopping each server.

    Returns:
        JSONResponse: A FastAPI JSONResponse object indicating the status of stopping each server.
                      Returns a success message if all servers are stopped successfully,
                      or an error message if some servers failed to stop.
    """
    results: Dict[
        str, str
    ] = {}  # Initialize dictionary to store results of stopping each server

    try:
        for (
            server
        ) in SERVERS_TO_STOP_ON_APP_CLOSE:  # Iterate through list of servers to stop
            result: bool = await stop_server(server)  # Stop each server asynchronously
            results[server] = (
                "Stopped" if result else "Failed"
            )  # Record stop status in results dictionary

        if all(
            status == "Stopped" for status in results.values()
        ):  # Check if all servers were stopped successfully
            print("All servers stopped successfully")
            return JSONResponse(
                status_code=200, content={"message": "All servers stopped successfully"}
            )  # Return success response if all servers stopped
        else:  # If not all servers stopped successfully
            print("Some servers failed to stop")
            return JSONResponse(
                status_code=500, content={"message": "Some servers failed to stop"}
            )  # Return error response if some servers failed to stop

    except Exception as e:  # Catch any exceptions during server stopping process
        print(f"Error stopping servers: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error stopping servers: {str(e)}"
        )  # Raise HTTPException with error details


# --- Main Application Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows executables created with PyInstaller
    uvicorn.run(
        app, host="0.0.0.0", port=5000, reload=False, workers=1
    )  # Run the FastAPI application using Uvicorn server
