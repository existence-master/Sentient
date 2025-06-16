# graphiti_agent.py
import asyncio
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, List

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# --- Graphiti Imports ---
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

# --- Global Variables & Setup ---
load_dotenv()
USER_NAME = "Alex"
graphiti: Graphiti = None
alex_node_uuid: str = None

# --- Agent Tools for Graphiti CRUD Operations ---

@tool
async def search_knowledge_graph(query: str) -> str:
    """
    Use this tool to answer questions about Alex or retrieve stored facts.
    It searches the knowledge graph for relevant information.
    """
    print(f"--- TOOL: Searching graph for: '{query}' ---")
    if not graphiti:
        return "Graphiti client not initialized."
    # Center the search on Alex's node for personalized results
    edge_results = await graphiti.search(query, center_node_uuid=alex_node_uuid, num_results=5)
    if not edge_results:
        return "No relevant information found."
    facts = "\n- ".join([edge.fact for edge in edge_results])
    return f"Found the following facts:\n- {facts}"

@tool
async def add_or_update_fact(fact: str) -> str:
    """
    Use this tool to add new information or update an existing fact about Alex.
    The fact should be a complete, declarative sentence.
    For example: 'Alex now lives in Berlin' or 'Alex no longer likes pop music'.
    This will automatically update the knowledge graph, invalidating old contradictory facts.
    """
    print(f"--- TOOL: Adding/Updating fact: '{fact}' ---")
    if not graphiti:
        return "Graphiti client not initialized."
    await graphiti.add_episode(
        name=f"FactUpdate from chat",
        episode_body=f"{USER_NAME} states: {fact}",
        source=EpisodeType.text,
        reference_time=datetime.now(),
        source_description="Personal Assistant Chat",
    )
    return f"Successfully updated knowledge graph with the fact: '{fact}'"


# --- LangGraph State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Agent Definition ---

async def agent_node(state: AgentState):
    """The primary node of the agent that decides what to do."""
    # Fetch current context from the graph to inform the LLM
    last_user_message = state["messages"][-1].content
    context_facts = await search_knowledge_graph.ainvoke({"query": last_user_message})

    system_prompt = f"""You are a helpful personal assistant for a user named {USER_NAME}.
Your memory is a Graphiti knowledge graph. You have tools to read from and write to this graph.

- To answer questions or recall information, use the `search_knowledge_graph` tool.
- To save new information or update facts, use the `add_or_update_fact` tool. Frame the input as a complete sentence.

When a user tells you something new, first use `add_or_update_fact` to remember it, then confirm to the user that you've learned it.
When a user asks a question, use `search_knowledge_graph` to find the answer and then respond.

Current relevant information from the knowledge graph:
{context_facts}
"""
    
    messages_with_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = await llm.ainvoke(messages_with_prompt)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Conditional edge to decide whether to call tools or end the turn."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# --- Database and Graph Setup ---

async def setup_database():
    """Initializes Graphiti, clears old data, and populates the graph."""
    global graphiti, alex_node_uuid
    print("--- Setting up Graphiti and Neo4j Database ---")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    graphiti = Graphiti(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD'),
        llm_client=GeminiClient(config=LLMConfig(api_key=google_api_key, model="gemini-2.0-flash")),
        embedder=GeminiEmbedder(config=GeminiEmbedderConfig(api_key=google_api_key, embedding_model="embedding-001")),
        cross_encoder=BGERerankerClient()
    )

    print("Clearing any existing data...")
    await clear_data(graphiti.driver)
    print("Building indices and constraints...")
    await graphiti.build_indices_and_constraints()

    print("Populating initial knowledge about Alex...")
    initial_facts = [
        f"{USER_NAME} lives in London.",
        f"{USER_NAME}'s favorite hobby is hiking.",
        f"{USER_NAME} works as a Software Engineer.",
        f"{USER_NAME} likes dogs.",
    ]
    for i, fact in enumerate(initial_facts):
        await graphiti.add_episode(
            name=f"InitialFact_{i}",
            episode_body=fact,
            source=EpisodeType.text,
            reference_time=datetime.now(),
        )

    # Get Alex's node UUID to use for personalized search
    alex_nodes = await graphiti.get_nodes_by_query(USER_NAME)
    if not alex_nodes:
        raise Exception("Could not create and find the user node for Alex.")
    alex_node_uuid = alex_nodes[0].uuid
    print(f"Database setup complete. Found user '{USER_NAME}' with node UUID: {alex_node_uuid}")
    print("-" * 50)


# --- Main Execution ---

if __name__ == "__main__":
    # Initialize the LLM and bind the tools
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    tools = [search_knowledge_graph, add_or_update_fact]
    llm_with_tools = llm.bind_tools(tools)
    
    # Build the LangGraph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))
    
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    graph_builder.add_edge("tools", "agent")
    
    graph = graph_builder.compile()

    # --- Interactive Chat Loop ---
    async def chat_loop():
        await setup_database()
        print("\nHi! I'm your personal AI assistant. I can remember things you tell me.")
        print("Try asking 'Where do I live?' or telling me 'My new hobby is painting.'")
        print("Type 'quit' to exit.")

        while True:
            user_input = input("\nAlex: ")
            if user_input.lower() == "quit":
                break

            try:
                # The `astream` method lets us process the agent's steps
                async for event in graph.astream({"messages": [("user", user_input)]}):
                    for key, value in event.items():
                        if key == "agent":
                            if value['messages'][-1].tool_calls:
                                print(f"--- Agent decided to use a tool ---")
                            else:
                                print(f"\nAssistant: {value['messages'][-1].content}")
                        elif key == "tools":
                            # The tool output is automatically added back to the state
                            print(f"--- Tool output: {value['messages'][-1].content} ---")

            except Exception as e:
                print(f"\nAn error occurred: {e}")
        
        await graphiti.close()
        print("\nConnection closed. Goodbye!")

    # Run the async chat loop
    asyncio.run(chat_loop())