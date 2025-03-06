import os
import uvicorn
import json
from externals import *  # Importing external service integrations or utilities from externals.py
from runnables import *  # Importing runnable classes or functions from runnables.py
from functions import *  # Importing custom functions from functions.py
from constants import *  # Importing constant variables from constants.py
from helpers import *  # Importing helper functions from helpers.py
from prompts import *  # Importing prompt templates and related utilities from prompts.py
import nest_asyncio  # For running asyncio event loop within another event loop (needed for FastAPI in some environments)
from fastapi import FastAPI, HTTPException  # Importing FastAPI for creating the API application
from pydantic import (
    BaseModel,
)  # Importing BaseModel from Pydantic for request body validation and data modeling
from neo4j import (
    GraphDatabase,
)  # Importing GraphDatabase from neo4j for Neo4j interaction
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)  # Importing JSONResponse and StreamingResponse for sending JSON responses from API endpoints
from fastapi.middleware.cors import (
    CORSMiddleware,
)  # Importing CORSMiddleware to handle Cross-Origin Resource Sharing
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)  # Importing HuggingFaceEmbedding for embedding model from llama_index
from fastapi import FastAPI  # Re-importing FastAPI (likely a typo and redundant)
from pydantic import BaseModel  # Re-importing BaseModel (likely a typo and redundant)
import os  # Re-importing os (likely a typo and redundant)
from neo4j import (
    GraphDatabase,
)  # Re-importing GraphDatabase (likely a typo and redundant)
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

UPLOAD_DIR = "../../interface/uploads"

# --- Pydantic Models for Request Body Validation ---
class DeleteSubgraphRequest(BaseModel):
    """
    Pydantic model for validating delete subgraph requests.
    Requires a 'source' string field.
    """

    source: str  # Source identifier for the subgraph to be deleted


class Message(BaseModel):
    """
    Pydantic model for validating chat message requests.
    Defines the structure for incoming chat messages.
    """

    original_input: str  # The original user input message
    transformed_input: str  # The transformed/processed user input message
    pricing: str  # Pricing plan of the user (e.g., "free", "pro")
    credits: int  # User's available credits
    chat_id: str  # Identifier for the chat session


class GraphRequest(BaseModel):
    """
    Pydantic model for validating generic graph requests.
    Requires an 'information' string field.
    """

    information: str  # Information string for graph operations


class GraphRAGRequest(BaseModel):
    """
    Pydantic model for validating GraphRAG (Retrieval-Augmented Generation) requests.
    Requires a 'query' string field.
    """

    query: str  # User's query for GraphRAG
    
class CustomRAGRequest(BaseModel):
    query: str


# --- Global Variables for Application State ---
# These global variables store initialized models, runnables, database connections, and chat history.
# Initialized in the `/initiate` endpoint.
index = None  # Llama-Index index (not currently used in the provided endpoints)
retriever = None  # Llama-Index retriever (not currently used in the provided endpoints)
embed_model = None  # Embedding model instance (HuggingFaceEmbedding)
chat_history = (
    None  # Stores chat history, likely as a list of messages or a database connection
)
graph_driver = None  # Neo4j graph driver instance for database interaction
chat_runnable = None  # Runnable for handling chat interactions
information_extraction_runnable = (
    None  # Runnable for extracting information (entities, relationships) from text
)
graph_decision_runnable = (
    None  # Runnable for making decisions about graph operations (CRUD)
)
graph_analysis_runnable = None  # Runnable for analyzing graph data
text_dissection_runnable = None  # Runnable for dissecting text into categories
text_conversion_runnable = (
    None  # Runnable for converting structured graph data to unstructured text
)
query_classification_runnable = (
    None  # Runnable for classifying user queries into categories
)
fact_extraction_runnable = None  # Runnable for extracting facts from text
text_summarizer_runnable = None  # Runnable for summarizing text
text_description_runnable = None  # Runnable for generating descriptions for entities

# --- Apply nest_asyncio ---
# Applying nest_asyncio to allow asyncio event loops to be nested.
# This is often needed when running FastAPI applications in environments that may already have an event loop running,
# such as Jupyter notebooks or certain testing frameworks.
# nest_asyncio.apply()


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
    Endpoint to initialize the AI model, graph driver, and runnables.
    This endpoint sets up the global variables required for other API calls.
    Returns a success message or an error message if initialization fails.
    """
    global embed_model, graph_driver, chat_history, index, information_extraction_runnable, graph_decision_runnable, graph_analysis_runnable, text_dissection_runnable, text_conversion_runnable, query_classification_runnable, fact_extraction_runnable, text_summarizer_runnable, text_description_runnable

    # Initialize HuggingFace embedding model
    embed_model = HuggingFaceEmbedding(model_name=os.environ["EMBEDDING_MODEL_REPO_ID"])

    # Initialize Neo4j graph driver
    graph_driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    # Initialize various runnables using functions from runnables.py
    graph_decision_runnable = (
        get_graph_decision_runnable()
    )  # Get runnable for graph decision making (CRUD operations)
    information_extraction_runnable = (
        get_information_extraction_runnable()
    )  # Get runnable for information extraction
    graph_analysis_runnable = (
        get_graph_analysis_runnable()
    )  # Get runnable for graph analysis
    text_dissection_runnable = (
        get_text_dissection_runnable()
    )  # Get runnable for text dissection
    text_conversion_runnable = (
        get_text_conversion_runnable()
    )  # Get runnable for text conversion (graph to text)
    query_classification_runnable = (
        get_query_classification_runnable()
    )  # Get runnable for query classification
    fact_extraction_runnable = (
        get_fact_extraction_runnable()
    )  # Get runnable for fact extraction
    text_summarizer_runnable = (
        get_text_summarizer_runnable()
    )  # Get runnable for text summarization
    text_description_runnable = (
        get_text_description_runnable()
    )  # Get runnable for text description generation

    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )  # Return success message
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/chat", status_code=200)
async def chat(message: Message):
    """
    Endpoint to handle user chat messages and generate responses.
    This is the main chat endpoint that processes user input, interacts with the knowledge graph,
    performs internet searches if necessary, and streams back the AI assistant's response.
    """
    global index, embed_model, chat_runnable
    global \
        fact_extraction_runnable, \
        text_conversion_runnable, \
        information_extraction_runnable
    global \
        graph_analysis_runnable, \
        graph_decision_runnable, \
        query_classification_runnable, \
        text_description_runnable
        
    print("Query classification runnable", query_classification_runnable)

    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)  # Load user profile database

        chat_history = get_chat_history(
            message.chat_id
        )  # Retrieve chat history for the given chat_id

        chat_runnable = get_chat_runnable(
            chat_history
        )  # Initialize chat runnable with chat history

        username = db["userData"]["personalInfo"][
            "name"
        ]  # Get username from user profile

        transformed_input = (
            message.transformed_input
        )  # Get transformed user input (pre-processed query)

        pricing_plan = message.pricing  # Get user's pricing plan
        credits = message.credits  # Get user's available credits

        async def response_generator():
            """
            Async generator function to stream the AI assistant's response.
            Handles memory updates, internet search, and response generation based on user's pricing plan and credits.
            """
            memory_used = (
                False  # Flag to track if memory (knowledge graph) was used for updates
            )
            agents_used = (
                False  # Flag to track if agents (not used in this endpoint) were used
            )
            internet_used = False  # Flag to track if internet search was performed
            user_context = None  # Context retrieved from user's knowledge graph
            internet_context = None  # Context retrieved from internet search
            pro_used = False  # Flag to track if pro features were used (memory updates)
            note = ""  # Note to append to the response, e.g., for credit expiry

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
            )  # Yield user's message
            await asyncio.sleep(0.05)  # Small delay for UI responsiveness

            if pricing_plan == "free":
                if credits <= 0:
                    # Free plan and no credits left: retrieval from knowledge graph only, no memory updates
                    yield (
                        json.dumps(
                            {
                                "type": "intermediary",
                                "message": "Retrieving memories...",
                            }
                        )
                        + "\n"
                    )  # Yield intermediary message
                    await asyncio.sleep(0.05)  # Small delay for UI responsiveness

                    user_context = query_user_profile(
                        transformed_input,
                        graph_driver,
                        embed_model,
                        text_conversion_runnable,
                        query_classification_runnable,
                    )  # Retrieve context from knowledge graph
                    note = "Sorry friend, could have updated my memory for this query. But, that is a pro feature and your daily credits have expired. You can always upgrade to pro from the settings page"  # Set note about credit expiry

                else:
                    # Free plan with credits: memory update and retrieval, internet search if classified as needed
                    yield (
                        json.dumps(
                            {"type": "intermediary", "message": "Updating memories..."}
                        )
                        + "\n"
                    )  # Yield intermediary message
                    await asyncio.sleep(0.05)  # Small delay for UI responsiveness

                    memory_used = True  # Mark memory as used
                    pro_used = True  # Mark pro features as used
                    points = fact_extraction_runnable.invoke(
                        {"paragraph": transformed_input, "username": username}
                    )  # Extract facts from user input

                    for point in points:
                        crud_graph_operations(
                            point,
                            graph_driver,
                            embed_model,
                            query_classification_runnable,
                            information_extraction_runnable,
                            graph_analysis_runnable,
                            graph_decision_runnable,
                            text_description_runnable,
                        )  # Perform CRUD operations to update graph

                    yield (
                        json.dumps(
                            {
                                "type": "intermediary",
                                "message": "Retrieving memories...",
                            }
                        )
                        + "\n"
                    )  # Yield intermediary message
                    await asyncio.sleep(0.05)  # Small delay for UI responsiveness

                    user_context = query_user_profile(
                        transformed_input,
                        graph_driver,
                        embed_model,
                        text_conversion_runnable,
                        query_classification_runnable,
                    )  # Retrieve context from knowledge graph

                    internet_classification = await classify_context(
                        transformed_input, "internet"
                    )  # Classify if internet search is needed

                    if internet_classification == "Internet":
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
                    else:
                        internet_context = None  # No internet context needed
            else:
                # Pro plan: memory update and retrieval, internet search if classified as needed
                yield (
                    json.dumps(
                        {"type": "intermediary", "message": "Updating memories..."}
                    )
                    + "\n"
                )  # Yield intermediary message
                await asyncio.sleep(0.05)  # Small delay for UI responsiveness

                memory_used = True  # Mark memory as used
                pro_used = True  # Mark pro features as used
                points = fact_extraction_runnable.invoke(
                    {"paragraph": transformed_input, "username": username}
                )  # Extract facts from user input

                for point in points:
                    crud_graph_operations(
                        point,
                        graph_driver,
                        embed_model,
                        query_classification_runnable,
                        information_extraction_runnable,
                        graph_analysis_runnable,
                        graph_decision_runnable,
                        text_description_runnable,
                    )  # Perform CRUD operations to update graph

                yield (
                    json.dumps(
                        {"type": "intermediary", "message": "Retrieving memories..."}
                    )
                    + "\n"
                )  # Yield intermediary message
                await asyncio.sleep(0.05)  # Small delay for UI responsiveness

                user_context = query_user_profile(
                    transformed_input,
                    graph_driver,
                    embed_model,
                    text_conversion_runnable,
                    query_classification_runnable,
                )  # Retrieve context from knowledge graph

                internet_classification = await classify_context(
                    transformed_input, "internet"
                )  # Classify if internet search is needed

                if internet_classification == "Internet":
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
                else:
                    internet_context = None  # No internet context needed

            with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
                db = json.load(
                    f
                )  # Load user profile database again (redundant load, consider optimizing)

            personality_description = db["userData"].get(
                "personality", "None"
            )  # Get personality description from user profile

            # Stream response from chat runnable
            async for token in generate_streaming_response(
                chat_runnable,
                inputs={
                    "query": transformed_input,
                    "user_context": user_context,
                    "internet_context": internet_context,
                    "name": username,
                    "personality": personality_description,
                },
                stream=True,  # Enable streaming response
            ):
                if isinstance(token, str):
                    yield (
                        json.dumps(
                            {"type": "assistantStream", "token": token, "done": False}
                        )
                        + "\n"
                    )  # Yield assistant's stream token
                else:
                    yield (
                        json.dumps(
                            {
                                "type": "assistantStream",
                                "token": "\n\n"
                                + note,  # Append note to the final token
                                "done": True,
                                "memoryUsed": memory_used,
                                "agentsUsed": agents_used,
                                "internetUsed": internet_used,
                                "proUsed": pro_used,
                            }
                        )
                        + "\n"
                    )  # Yield final token with flags and note
                await asyncio.sleep(0.05)  # Small delay for UI responsiveness
            await asyncio.sleep(0.05)  # Small delay after response generation

        return StreamingResponse(
            response_generator(), media_type="application/json"
        )  # Return streaming response

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/graphrag", status_code=200)
async def graphrag(request: GraphRAGRequest):
    """
    Endpoint to process a user profile query using GraphRAG (Retrieval-Augmented Generation).

    This endpoint takes a user query, retrieves relevant context from the knowledge graph using GraphRAG,
    and returns the context as a JSON response. This is useful for directly querying the knowledge graph
    without engaging in a full chat conversation.
    """
    global \
        graph_driver, \
        embed_model, \
        text_conversion_runnable, \
        query_classification_runnable

    try:
        context = query_user_profile(
            request.query,
            graph_driver,
            embed_model,
            text_conversion_runnable,
            query_classification_runnable,
        )  # Query user profile using GraphRAG
        return JSONResponse(
            status_code=200, content={"context": context}
        )  # Return context in JSON response
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/create-graph", status_code=200)
async def create_graph():
    """
    Endpoint to create a knowledge graph from documents in the input directory.

    This endpoint initiates the process of building a knowledge graph by processing text documents
    found in the "../input" directory. It clears any existing graph, then dissects, extracts, and loads
    information from these documents into the Neo4j graph database.
    """
    global \
        graph_driver, \
        embed_model, \
        text_dissection_runnable, \
        information_extraction_runnable, \
        text_summarizer_runnable

    try:
        input_dir = "../input"  # Define input directory for documents

        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)  # Load user profile database to get username

        username = db["userData"]["personalInfo"].get(
            "name", "User"
        )  # Get username from user profile

        extracted_texts = []  # Initialize list to store extracted texts
        for file_name in os.listdir(
            input_dir
        ):  # Iterate through files in the input directory
            file_path = os.path.join(input_dir, file_name)  # Construct file path
            if os.path.isfile(file_path):  # Check if it's a file
                with open(file_path, "r", encoding="utf-8") as file:
                    text_content = (
                        file.read().strip()
                    )  # Read file content and strip whitespace
                    if text_content:
                        extracted_texts.append(
                            {"text": text_content, "source": file_name}
                        )  # Append text and source to list

        if not extracted_texts:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "No content found in input documents to create the graph."
                },  # Return error if no content found
            )

        def clear_graph():
            """Clears all nodes and relationships from the Neo4j graph database."""
            with graph_driver.session() as session:  # Open Neo4j session
                session.run(
                    "MATCH (n) DETACH DELETE n"
                )  # Cypher query to delete all nodes and relationships
                print(
                    "All nodes and relationships deleted successfully."
                )  # Print confirmation message

        clear_graph()  # Clear existing graph before creating a new one

        build_initial_knowledge_graph(
            username,
            extracted_texts,
            graph_driver,
            embed_model,
            text_dissection_runnable,
            information_extraction_runnable,
        )  # Build initial knowledge graph from extracted texts

        return JSONResponse(
            status_code=200, content={"message": "Graph created successfully."}
        )  # Return success message

    except FileNotFoundError as e:
        print(e)
        return JSONResponse(
            status_code=404,
            content={"message": "Input directory or documents not found."},
        )  # Return error if input directory not found
    except Exception as e:
        print(f"Error creating graph: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while creating the graph.",
                "error": str(e),
            },  # Return general error message
        )


@app.post("/delete-subgraph", status_code=200)
async def delete_subgraph(request: DeleteSubgraphRequest):
    """
    Endpoint to delete a subgraph from the knowledge graph based on a source name.

    This endpoint allows for the removal of specific parts of the knowledge graph that are associated
    with a given data source (e.g., LinkedIn profile, Reddit data). It identifies nodes by their 'source'
    property and deletes them along with their relationships.
    """
    global graph_driver

    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)  # Load user profile database to get username

        username = (
            db["userData"]["personalInfo"].get("name", "User").lower()
        )  # Get lowercase username from user profile

        source_name = request.source  # Get source name from request

        if not source_name:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Missing source_name parameter."
                },  # Return error if source_name is missing
            )

        SOURCES = {
            "linkedin": f"{username}_linkedin_profile.txt",
            "reddit": f"{username}_reddit_profile.txt",
            "twitter": f"{username}_twitter_profile.txt",
        }  # Mapping of source names to file names

        file_name = SOURCES[source_name]  # Get file name corresponding to source name
        if not file_name:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"No file mapping found for source name: {source_name}"
                },  # Return error if no file mapping found
            )

        delete_source_subgraph(
            graph_driver, file_name
        )  # Delete subgraph from Neo4j based on file name

        os.remove(f"../input/{file_name}")  # Remove the corresponding input file

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Subgraph related to {file_name} deleted successfully."
            },  # Return success message
        )

    except Exception as e:
        print(f"Error deleting subgraph: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while deleting the subgraph.",
                "error": str(e),
            },  # Return general error message
        )


@app.post("/create-document", status_code=200)
async def create_document():
    """
    Endpoint to create and summarize personality documents based on user profile data.

    This endpoint generates text documents representing different aspects of the user's personality
    and profile (e.g., personality traits, LinkedIn profile summary, Reddit interests, Twitter interests).
    It uses a text summarizer runnable to condense the information and saves these documents to the "../input" directory.
    """
    global text_summarizer_runnable

    try:
        with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
            db = json.load(f)  # Load user profile database

        username = db["userData"]["personalInfo"].get(
            "name", "User"
        )  # Get username from user profile
        personality_type = db["userData"].get(
            "personalityType", ""
        )  # Get personality type from user profile
        structured_linkedin_profile = db["userData"].get(
            "linkedInProfile", {}
        )  # Get LinkedIn profile data
        reddit_profile = db["userData"].get(
            "redditProfile", []
        )  # Get Reddit profile data (interests)
        twitter_profile = db["userData"].get(
            "twitterProfile", []
        )  # Get Twitter profile data (interests)
        input_dir = "../input"  # Define input directory

        os.makedirs(input_dir, exist_ok=True)  # Ensure input directory exists

        for file in os.listdir(
            input_dir
        ):  # Clear existing files in the input directory
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        trait_descriptions = []  # Initialize list to store personality trait descriptions

        for trait in personality_type:  # Iterate through each personality trait
            if trait in PERSONALITY_DESCRIPTIONS:
                description = f"{trait}: {PERSONALITY_DESCRIPTIONS[trait]}"  # Get description for the trait
                trait_descriptions.append(description)  # Append description to list
                filename = f"{username.lower()}_{trait.lower()}.txt"  # Construct filename for trait document
                filepath = os.path.join(input_dir, filename)  # Construct file path
                summarized_paragraph = text_summarizer_runnable.invoke(
                    {"user_name": username, "text": description}
                )  # Summarize trait description

                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(
                        summarized_paragraph
                    )  # Write summarized description to file

        unified_personality_description = (
            f"{username}'s Personality:\n\n"
            + "\n".join(
                trait_descriptions
            )  # Create unified personality description string
        )

        if structured_linkedin_profile:
            linkedin_profile_file = os.path.join(
                input_dir,
                f"{username.lower()}_linkedin_profile.txt",  # Construct filename for LinkedIn profile document
            )
            summarized_paragraph = text_summarizer_runnable.invoke(
                {"user_name": username, "text": structured_linkedin_profile}
            )  # Summarize LinkedIn profile

            with open(linkedin_profile_file, "w", encoding="utf-8") as file:
                file.write(
                    summarized_paragraph
                )  # Write summarized LinkedIn profile to file

        if reddit_profile:
            reddit_profile_file = os.path.join(
                input_dir,
                f"{username.lower()}_reddit_profile.txt",  # Construct filename for Reddit profile document
            )
            summarized_paragraph = text_summarizer_runnable.invoke(
                {
                    "user_name": username,
                    "text": "Interests: " + (",").join(reddit_profile),
                }
            )  # Summarize Reddit interests

            with open(reddit_profile_file, "w", encoding="utf-8") as file:
                file.write(
                    summarized_paragraph
                )  # Write summarized Reddit interests to file

        if twitter_profile:
            twitter_profile_file = os.path.join(
                input_dir,
                f"{username.lower()}_twitter_profile.txt",  # Construct filename for Twitter profile document
            )
            summarized_paragraph = text_summarizer_runnable.invoke(
                {
                    "user_name": username,
                    "text": "Interests: " + (",").join(twitter_profile),
                }
            )  # Summarize Twitter interests

            with open(twitter_profile_file, "w", encoding="utf-8") as file:
                file.write(
                    summarized_paragraph
                )  # Write summarized Twitter interests to file

        return JSONResponse(
            status_code=200,
            content={
                "message": "Documents created and personality saved successfully",
                "personality": unified_personality_description,  # Return unified personality description in response
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return error message if exception occurs


@app.post("/customize-graph", status_code=200)
async def customize_graph(request: GraphRequest):
    """
    Endpoint to customize the knowledge graph with new information provided in the request.

    This endpoint takes unstructured text information from the request, extracts facts from it,
    and performs CRUD operations on the knowledge graph to incorporate this new information.
    This allows users to interactively add or modify information in their personal knowledge graph.
    """
    global \
        information_extraction_runnable, \
        graph_decision_runnable, \
        graph_driver, \
        embed_model, \
        query_classification_runnable, \
        graph_analysis_runnable, \
        text_description_runnable, \
        fact_extraction_runnable
    with open("../../userProfileDb.json", "r", encoding="utf-8") as f:
        db = json.load(f)  # Load user profile database to get username

    username = db["userData"]["personalInfo"]["name"]  # Get username from user profile

    try:
        points = fact_extraction_runnable.invoke(
            {"paragraph": request.information, "username": username}
        )  # Extract facts from the provided information

        for point in points:
            crud_graph_operations(
                point,
                graph_driver,
                embed_model,
                query_classification_runnable,
                information_extraction_runnable,
                graph_analysis_runnable,
                graph_decision_runnable,
                text_description_runnable,
            )  # Perform CRUD operations to customize graph

        return JSONResponse(
            status_code=200,
            content={
                "message": "Graph customized successfully."
            },  # Return success message
        )

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={
                "message": f"An error occurred: {str(e)}"
            },  # Return error message if exception occurs
        )

@app.post("/refresh-rag-context")
async def refresh_rag_context():
    try:
        # Delete and recreate the collection to ensure a fresh state
        client.delete_collection("documents_collection")
        global collection
        collection = client.get_or_create_collection(
            name="documents_collection",
            embedding_function=sentence_transformer_ef
        )
        # Process all files in the uploads folder and add to ChromaDB
        process_and_add_documents(collection, UPLOAD_DIR)
        return {"status": "success", "message": "RAG context refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing RAG context: {str(e)}")

app.post("/custom-rag")
async def custom_rag(request: CustomRAGRequest):
    """
    Endpoint to perform a similarity search in ChromaDB and return formatted context with sources.

    Args:
        request (CustomRAGRequest): Request body containing the query string.

    Returns:
        dict: JSON response with 'context' key containing the formatted context string.

    Raises:
        HTTPException: If an error occurs during processing, returns a 500 status code with error details.
    """
    try:
        query = request.query
        # Perform similarity search
        results = semantic_search(collection, query)
        # Format context with sources
        formatted_context = format_context_with_sources(results)
        # Return JSON response
        return {"context": formatted_context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- Main execution block ---
if __name__ == "__main__":
    """
    This block is executed when the script is run directly (not imported as a module).
    It starts the uvicorn server to run the FastAPI application.
    """
    uvicorn.run(app, port=5002)  # Start uvicorn server on port 5002
