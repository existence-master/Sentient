# graphiti_explorer.py
import asyncio
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Graphiti Core Imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# ADDED: Imports for Google Gemini configuration
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

# Local Sample Data
from sample_data import get_quickstart_episodes, get_bulk_episodes_from_product_data

# --- Helper Functions (No changes here) ---

def print_header(title):
    print("\n" + "="*50)
    print(f"// {title.upper()} //")
    print("="*50)

def print_results(results):
    if not results:
        print("\nNo results found.")
        return
    print("\n--- Search Results ---")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Fact: {getattr(result, 'fact', 'N/A')}")
        print(f"    UUID: {result.uuid}")
        if hasattr(result, 'valid_at') and result.valid_at:
            print(f"    Valid From: {result.valid_at}")
        if hasattr(result, 'invalid_at') and result.invalid_at:
            print(f"    Valid Until: {result.invalid_at}")
    print("----------------------")

def print_nodes(nodes):
    if not nodes:
        print("\nNo nodes found.")
        return
    print("\n--- Node Results ---")
    for i, node in enumerate(nodes):
        print(f"\n[{i+1}] Name: {node.name}")
        print(f"    UUID: {node.uuid}")
        print(f"    Labels: {', '.join(node.labels)}")
        if hasattr(node, 'attributes') and node.attributes:
            print("    Attributes:")
            for key, value in node.attributes.items():
                print(f"      - {key}: {value}")
    print("--------------------")

# --- Feature Handlers (No changes in these functions) ---

async def handle_initialize_db(graphiti: Graphiti):
    print_header("Initialize Database")
    print("This will build Graphiti's required indices and constraints.")
    print("This only needs to be done once for a new database.")
    if input("Proceed? (y/n): ").lower() == 'y':
        await graphiti.build_indices_and_constraints()
        print("\nDatabase initialized successfully.")

async def handle_clear_db(graphiti: Graphiti):
    print_header("Clear Entire Database")
    print("\n!!! WARNING: THIS IS A DESTRUCTIVE OPERATION AND WILL WIPE ALL DATA !!!")
    if input("Are you absolutely sure you want to proceed? (type 'yes' to confirm): ") == 'yes':
        await clear_data(graphiti.driver)
        print("\nDatabase has been cleared.")

async def handle_add_episode(graphiti: Graphiti):
    print_header("Add a Single Episode")
    group_id = input("Enter a group_id for namespacing (or press Enter for none): ") or None
    name = input("Episode name (e.g., 'Customer Interaction 1'): ")
    source_description = input("Source description (e.g., 'Support Chat'): ")

    print("Choose episode type: 1. Text, 2. Message, 3. JSON")
    choice = input("> ")
    if choice == '1':
        source = EpisodeType.text
        body = input("Enter the text content:\n> ")
    elif choice == '2':
        source = EpisodeType.message
        print("Enter conversational messages (e.g., 'Speaker: Message'). Press Enter on an empty line to finish.")
        lines = []
        while True:
            line = input("> ")
            if not line:
                break
            lines.append(line)
        body = "\n".join(lines)
    elif choice == '3':
        source = EpisodeType.json
        print("Enter the JSON content:")
        body_str = input("> ")
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError:
            print("Invalid JSON. Aborting.")
            return
    else:
        print("Invalid choice.")
        return

    await graphiti.add_episode(
        name=name,
        episode_body=body,
        source=source,
        source_description=source_description,
        reference_time=datetime.now(),
        group_id=group_id
    )
    print(f"\nEpisode '{name}' added successfully.")

async def handle_bulk_load(graphiti: Graphiti):
    print_header("Bulk Load Episodes")
    print("This will load sample product data using `add_episode_bulk`.")
    print("Note: Bulk loading is for empty graphs or when edge invalidation is not needed.")
    if input("Proceed? (y/n): ").lower() == 'y':
        episodes = get_bulk_episodes_from_product_data()
        await graphiti.add_episode_bulk(episodes)
        print(f"\nSuccessfully bulk-loaded {len(episodes)} episodes.")

async def handle_add_triplet(graphiti: Graphiti):
    print_header("Add a Fact Triplet Manually")
    print("This adds a (source)-[edge]->(target) relationship.")
    source_name = input("Enter source node name (e.g., 'Bob'): ")
    edge_name = input("Enter edge name/verb (e.g., 'LIKES'): ")
    target_name = input("Enter target node name (e.g., 'bananas'): ")
    edge_fact = input(f"Enter the full fact for the edge (e.g., '{source_name} {edge_name.lower()} {target_name}'): ")
    group_id = input("Enter a group_id for namespacing (or press Enter for none): ") or None
    
    source_node = EntityNode(uuid=str(uuid.uuid4()), name=source_name, group_id=group_id)
    target_node = EntityNode(uuid=str(uuid.uuid4()), name=target_name, group_id=group_id)
    edge = EntityEdge(
        group_id=group_id,
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        created_at=datetime.now(),
        name=edge_name,
        fact=edge_fact
    )
    
    await graphiti.add_triplet(source_node, edge, target_node)
    print("\nFact triplet added successfully.")

async def handle_hybrid_search(graphiti: Graphiti):
    print_header("Hybrid Search (Edges/Facts)")
    query = input("Enter your search query: ")
    group_id = input("Enter a group_id to search within (or press Enter for all): ") or None
    results = await graphiti.search(query, group_id=group_id)
    print_results(results)

async def handle_node_distance_search(graphiti: Graphiti):
    print_header("Hybrid Search with Node Distance Reranking")
    node_name = input("First, enter the name of a central entity to focus on (e.g., 'Kamala Harris'): ")
    
    search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    search_config.limit = 1
    node_results = await graphiti._search(query=node_name, config=search_config)

    if not node_results.nodes:
        print(f"Could not find a node named '{node_name}'.")
        return

    center_node = node_results.nodes[0]
    print(f"\nFound focal node: {center_node.name} (UUID: {center_node.uuid})")

    query = input("Now, enter your search query to be reranked around this node: ")
    reranked_results = await graphiti.search(query, center_node_uuid=center_node.uuid)
    print_results(reranked_results)

async def handle_advanced_node_search(graphiti: Graphiti):
    print_header("Advanced Node Search with a Recipe")
    print("This uses `_search` with the `NODE_HYBRID_SEARCH_RRF` recipe to find nodes.")
    query = input("Enter your node search query (e.g., 'California Governor'): ")
    group_id = input("Enter a group_id to search within (or press Enter for all): ") or None

    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5
    
    node_search_results = await graphiti._search(query=query, config=node_search_config, group_id=group_id)
    print_nodes(node_search_results.nodes)

async def handle_build_communities(graphiti: Graphiti):
    print_header("Build/Rebuild Communities")
    print("This uses the Leiden algorithm to group related nodes.")
    if input("Proceed? (y/n): ").lower() == 'y':
        await graphiti.build_communities()
        print("\nCommunities built successfully.")

async def handle_custom_entities(graphiti: Graphiti):
    print_header("Demonstrate Custom Entity Types (Ontology)")

    class Customer(BaseModel):
        """A customer of the service"""
        name: str | None = Field(..., description="The name of the customer")
        email: str | None = Field(..., description="The email address of the customer")
        subscription_tier: str | None = Field(..., description="The customer's subscription level")

    class Product(BaseModel):
        """A product or service offering"""
        name: str | None = Field(..., description="The name of the product")
        price: float | None = Field(..., description="The price of the product or service")
        category: str | None = Field(..., description="The category of the product")

    entity_types = {"Customer": Customer, "Product": Product}
    print("Defined two custom entity types: 'Customer' and 'Product'.")

    episode_body = "New customer John Doe (john@example.com) signed up for premium tier and purchased our Analytics Pro product ($199.99) from the Software category."
    print(f"\nAdding an episode with this content:\n'{episode_body}'")
    
    await graphiti.add_episode(
        name='CustomTypeDemo',
        episode_body=episode_body,
        source=EpisodeType.text,
        entity_types=entity_types
    )
    print("\nEpisode added. Graphiti extracted entities and classified them.")
    
    print("\nVerifying by searching for 'John Doe' and inspecting the node...")
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 1
    results = await graphiti._search("John Doe", config=node_search_config)
    print_nodes(results.nodes)

async def handle_list_all_nodes(graphiti: Graphiti):
    print_header("List All Nodes (limit 25)")
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (n) RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels LIMIT 25"
    )
    if not records:
        print("No nodes found in the graph.")
        return
    print("\n--- All Nodes ---")
    for i, record in enumerate(records):
        print(f"[{i+1}] Name: {record['name']}")
        print(f"    UUID: {record['uuid']}")
        print(f"    Labels: {', '.join(record['labels'])}")
    print("-----------------")


async def main():
    load_dotenv()
    
    # MODIFIED: Check for Google API key instead of OpenAI
    required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'GOOGLE_API_KEY']
    if any(not os.getenv(var) for var in required_vars):
        print("Error: Required environment variables are not set.")
        print("Please create a .env file with:", ", ".join(required_vars))
        return

    # MODIFIED: Initialize Graphiti with explicit Gemini clients
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    graphiti = Graphiti(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD'),
        llm_client=GeminiClient(
            config=LLMConfig(
                api_key=google_api_key,
                model="gemini-2.0-flash" # Use a modern, cost-effective model
            )
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=google_api_key,
                embedding_model="embedding-001"
            )
        ),
        cross_encoder=BGERerankerClient()
    )
    
    # Load sample episodes from the quickstart guide on startup
    print("Loading sample data from the Quickstart guide (using Gemini)...")
    for episode_data in get_quickstart_episodes():
        await graphiti.add_episode(
            name=episode_data['name'],
            episode_body=json.dumps(episode_data['content']) if isinstance(episode_data['content'], dict) else episode_data['content'],
            source=episode_data['type'],
            source_description=episode_data['description'],
            reference_time=datetime.now()
        )
    print("Sample data loaded.")


    menu = {
        "1": ("Add a Single Episode", handle_add_episode),
        "2": ("Bulk Load Product Episodes", handle_bulk_load),
        "3": ("Add a Manual Fact Triplet", handle_add_triplet),
        "4": ("Hybrid Search (for Facts/Edges)", handle_hybrid_search),
        "5": ("Search with Node Distance Reranking", handle_node_distance_search),
        "6": ("Advanced Node Search (with Recipe)", handle_advanced_node_search),
        "7": ("Build/Rebuild Communities", handle_build_communities),
        "8": ("Demonstrate Custom Entity Types", handle_custom_entities),
        "9": ("List All Nodes (utility)", handle_list_all_nodes),
        "10": ("Initialize Database (run once)", handle_initialize_db),
        "11": ("!!! WIPE ENTIRE DATABASE !!!", handle_clear_db),
        "q": ("Quit", None)
    }

    try:
        while True:
            print_header("Graphiti Explorer Main Menu (Gemini Mode)")
            for key, (desc, _) in menu.items():
                print(f"  {key}. {desc}")
            
            choice = input("\nEnter your choice: ").lower()
            
            if choice == 'q':
                break
            
            if choice in menu:
                desc, func = menu[choice]
                await func(graphiti)
            else:
                print("Invalid choice. Please try again.")

    finally:
        await graphiti.close()
        print("\nConnection closed. Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")