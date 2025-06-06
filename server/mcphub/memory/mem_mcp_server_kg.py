import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from bson import ObjectId
from fastmcp import FastMCP
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from neo4j import GraphDatabase, exceptions as Neo4jExceptions
from sentence_transformers import SentenceTransformer

# --- General Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the Neo4j notifications logger to a higher level to hide informational messages
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

# --- Constants ---
class Cfg:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_memory_test")
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    CATEGORIES = ["Personal", "Professional", "Social", "Financial", "Health", "Preferences", "Events", "General"]

# --- Embedding Model Singleton ---
try:
    embedding_model = SentenceTransformer(Cfg.EMBEDDING_MODEL)
    logger.info(f"Successfully loaded SentenceTransformer model: {Cfg.EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Failed to load sentence-transformer model. Error: {e}")
    exit(1)

def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text, convert_to_tensor=False).tolist()

# --- Neo4j Knowledge Graph Memory Manager ---
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info(f"Successfully connected to Neo4j: {uri}")
        self._setup_database()

    def _setup_database(self):
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            session.run("CREATE VECTOR INDEX `observation-embeddings` IF NOT EXISTS FOR (o:Observation) ON (o.embedding) OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } }")
            session.run("UNWIND $categories as category_name MERGE (c:Category {name: category_name})", categories=Cfg.CATEGORIES)
            logger.info("Ensured Neo4j constraints, indexes, and categories.")

    def close(self):
        if self._driver: self._driver.close()

    def _sanitize_relation_type(self, relation_type: str) -> str:
        return relation_type.upper().replace(' ', '_').replace('-', '_')

    def upsert_memory_fact(self, user_id: str, fact_text: str, category: str, relations: List[Dict]) -> Dict[str, Any]:
        fact_embedding = get_embedding(fact_text)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._driver.session() as session:
            session.run("""
                MERGE (u:User:Entity {user_id: $user_id})
                ON CREATE SET u.created_at = datetime($now_iso), u.name = 'User Profile'
                
                CREATE (o:Observation {
                    text: $fact_text, embedding: $fact_embedding,
                    created_at: datetime($now_iso), last_accessed_at: datetime($now_iso)
                })
                
                MERGE (u)-[:HAS_OBSERVATION]->(o)
                WITH u, o
                MATCH (c:Category {name: $category})
                MERGE (o)-[:IN_CATEGORY]->(c)
            """, user_id=user_id, now_iso=now_iso, fact_text=fact_text, fact_embedding=fact_embedding, category=category)

            relations_created_summary = []
            for rel in relations:
                rel_type = self._sanitize_relation_type(rel['type'])
                # Use execute_write (the modern replacement for write_transaction)
                tx_result = session.execute_write(
                    self._create_relation_tx, user_id, rel['from'], rel['to'], rel_type
                )
                if tx_result:
                    relations_created_summary.append(tx_result)

        return {"status": "success", "fact_added": fact_text, "relations_processed": relations_created_summary}

    @staticmethod
    def _create_relation_tx(tx, user_id, from_entity_name, to_entity_name, rel_type):
        from_node_alias = 'a'
        from_match_clause = f"MATCH ({from_node_alias}:User {{user_id: $user_id}})" if from_entity_name.lower() == 'user' \
                            else f"MERGE ({from_node_alias}:Entity {{user_id: $user_id, name: $from_name}})"
        
        to_node_alias = 'b'
        to_match_clause = f"MATCH ({to_node_alias}:User {{user_id: $user_id}})" if to_entity_name.lower() == 'user' \
                          else f"MERGE ({to_node_alias}:Entity {{user_id: $user_id, name: $to_name}})"

        # --- FIX #2 ---
        # Added WITH clause to pass the context from the first MATCH/MERGE to the second.
        query = f"""
        {from_match_clause}
        WITH {from_node_alias}
        {to_match_clause}
        MERGE ({from_node_alias})-[r:`{rel_type}`]->({to_node_alias})
        RETURN $from_name as from_name, $to_name as to_name, $rel_type_orig as type
        """
        result = tx.run(query, user_id=user_id, from_name=from_entity_name, to_name=to_entity_name, rel_type_orig=rel_type)
        return result.single()

    def semantic_search(self, user_id: str, query_text: str, categories: Optional[List[str]] = None, limit: int = 5) -> List[Dict]:
        query_embedding = get_embedding(query_text)
        
        category_match_clause = ""
        if categories:
            category_match_clause = "MATCH (o)-[:IN_CATEGORY]->(c:Category) WHERE c.name IN $categories"

        # --- FIX #1 ---
        # Added WITH clause between SET and MATCH.
        query = f"""
            CALL db.index.vector.queryNodes('observation-embeddings', $limit, $query_embedding) YIELD node AS o, score
            MATCH (e:Entity {{user_id: $user_id}})-[:HAS_OBSERVATION]->(o)
            SET o.last_accessed_at = datetime()
            WITH o, e, score // Explicitly pass o, e, and score forward
            {category_match_clause}
            RETURN
                o.text as fact,
                c.name as category,
                score as similarity
            LIMIT $limit
        """
        
        with self._driver.session() as session:
            results = session.run(query, limit=limit, query_embedding=query_embedding, user_id=user_id, categories=categories)
            return [
                {
                    "fact": r["fact"],
                    "category": r["category"],
                    "similarity": round(r["similarity"], 2)
                }
                for r in results
            ]
            
    def get_full_user_profile(self, user_id: str) -> Dict[str, List[str]]:
        query = """
            MATCH (u:User:Entity {user_id: $user_id})-[:HAS_OBSERVATION]->(o:Observation)-[:IN_CATEGORY]->(c:Category)
            RETURN c.name as category, collect(o.text) as facts
        """
        with self._driver.session() as session:
            results = session.run(query, user_id=user_id)
            return {r["category"]: r["facts"] for r in results}

# --- MongoDB Short-Term Memory Manager (with custom cosine) ---
class MongoShortTermMemory:
    def __init__(self, client):
        self.db = client[Cfg.MONGO_DB_NAME]
        self.collection = self.db["short_term_memories"]
        self._setup_database()

    def _setup_database(self):
        if "expire_at_ttl" not in self.collection.index_information():
            self.collection.create_index("expire_at", name="expire_at_ttl", expireAfterSeconds=0)
        if "user_category_idx" not in self.collection.index_information():
            self.collection.create_index([("user_id", 1), ("category", 1)], name="user_category_idx")

    def _cosine_similarity(self, vec1, vec2):
        vec1_np, vec2_np = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
        return float(dot_product / (norm_vec1 * norm_vec2))

    def add_memory(self, user_id, content, ttl_seconds, category):
        embedding = get_embedding(content)
        now = datetime.now(timezone.utc)
        doc = {
            "user_id": user_id, "content": content, "content_embedding": embedding,
            "category": category, "expire_at": now + timedelta(seconds=ttl_seconds),
            "created_at": now, "last_accessed_at": now
        }
        self.collection.insert_one(doc)
        return self._serialize_mongo_doc(doc)

    def search_memories(self, user_id, query_text, categories=None, limit=5):
        query_embedding = get_embedding(query_text)
        mongo_filter = {"user_id": user_id}
        if categories: mongo_filter["category"] = {"$in": categories}
        
        candidate_docs = list(self.collection.find(mongo_filter))
        if not candidate_docs: return []

        scored_docs = [
            {'similarity': self._cosine_similarity(query_embedding, doc['content_embedding']), 'doc': doc}
            for doc in candidate_docs if 'content_embedding' in doc and doc['content_embedding']
        ]
        
        sorted_results = sorted(scored_docs, key=lambda x: x['similarity'], reverse=True)
        top_results = sorted_results[:limit]
        
        final_docs_to_return = []
        if top_results:
            doc_ids_to_update = [item['doc']['_id'] for item in top_results]
            self.collection.update_many(
                {"_id": {"$in": doc_ids_to_update}},
                {"$set": {"last_accessed_at": datetime.now(timezone.utc)}}
            )
            for item in top_results:
                final_docs_to_return.append({
                    "content": item['doc']['content'],
                    "category": item['doc']['category'],
                    "similarity": round(item['similarity'], 2)
                })
        return final_docs_to_return

    def _serialize_mongo_doc(self, doc):
        if not doc: return None
        if "_id" in doc: doc["_id"] = str(doc["_id"])
        if "content_embedding" in doc: del doc["content_embedding"]
        for key, value in doc.items():
            if isinstance(value, datetime): doc[key] = value.isoformat()
        return doc

# --- Initialize Managers and MCP Server ---
kg_manager = Neo4jKnowledgeGraph(Cfg.NEO4J_URI, Cfg.NEO4J_USER, Cfg.NEO4J_PASSWORD)
mongo_client = MongoClient(Cfg.MONGO_URI)
stm_manager = MongoShortTermMemory(mongo_client)

mcp = FastMCP(
    name="SentientMemoryCompanionServer",
    instructions="This server provides a robust, dual-system memory for an AI companion..."
)

# --- MCP Tools ---
@mcp.tool()
def save_long_term_fact(user_id: str, fact_text: str, category: str, relations: Optional[List[Dict]] = None) -> Dict:
    """
    Saves a permanent fact to the user's knowledge graph. Use for preferences, relationships, or key personal/professional details.
    To connect entities, use the 'relations' field. Use 'user' to refer to the primary user.
    Example relations: [{"from": "user", "to": "Innovatech", "type": "WORKS_AT"}, {"from": "Jordan", "to": "user", "type": "FRIEND_OF"}].
    
    :param user_id: The user's unique identifier.
    :param fact_text: The string of information to remember. E.g., "I work at Innovatech as a software engineer."
    :param category: The category of this fact. Must be one of: Personal, Professional, Social, Financial, Health, Preferences, Events, General.
    :param relations: (Optional) A list of relationships to create between entities.
    """
    if category not in Cfg.CATEGORIES: raise ValueError(f"Invalid category '{category}'. Must be one of {Cfg.CATEGORIES}")
    return kg_manager.upsert_memory_fact(user_id, fact_text, category, relations or [])

@mcp.tool()
def add_short_term_memory(user_id: str, content: str, ttl_seconds: int, category: str) -> Dict:
    """Saves a temporary, expiring piece of information like a reminder or to-do item."""
    if category not in Cfg.CATEGORIES: raise ValueError(f"Invalid category '{category}'. Must be one of {Cfg.CATEGORIES}")
    return stm_manager.add_memory(user_id, content, ttl_seconds, category)

@mcp.tool()
def search_memories(user_id: str, query_text: str, categories: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """
    Performs a semantic search across all memories to find relevant information. Use this before answering any user query.
    
    :param user_id: The user's unique identifier.
    :param query_text: The question or topic to search for, e.g., "What do I do for work?".
    :param categories: (Optional) A list of categories to restrict the search to.
    """
    ltm_results = kg_manager.semantic_search(user_id, query_text, categories)
    stm_results = stm_manager.search_memories(user_id, query_text, categories)
    return {"long_term_facts": ltm_results, "short_term_reminders": stm_results}

@mcp.tool()
def get_user_profile_summary(user_id: str) -> Dict[str, List[str]]:
    """Retrieves a structured summary of everything known about the user from the long-term knowledge graph."""
    return kg_manager.get_full_user_profile(user_id)

if __name__ == "__main__":
    logger.info("Starting Sentient Memory Companion MCP Server...")
    try:
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
    finally:
        if kg_manager: kg_manager.close()
        if mongo_client: mongo_client.close()
        logger.info("MCP Server shut down and DB connections closed.")