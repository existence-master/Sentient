import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from bson import ObjectId
from fastmcp import FastMCP
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DualMemoryMCP")
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

class Cfg:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_dual_memory_v2")
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384

# --- Singleton for Embedding Model ---
embedding_model = SentenceTransformer(Cfg.EMBEDDING_MODEL)
logger.info(f"Loaded SentenceTransformer model: {Cfg.EMBEDDING_MODEL}")

def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text, convert_to_tensor=False).tolist()

# --- Neo4j Manager with Full CRUD ---
class Neo4jManager:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._setup_database()
        logger.info("Neo4j Manager initialized and connected.")

    def _setup_database(self):
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            session.run(f"CREATE VECTOR INDEX `fact_embeddings` IF NOT EXISTS FOR (f:Fact) ON (f.embedding) OPTIONS {{ indexConfig: {{ `vector.dimensions`: {Cfg.EMBEDDING_DIM}, `vector.similarity_function`: 'cosine' }} }}")

    def close(self): self._driver.close()

    def add_fact(self, user_id: str, fact_text: str, category: str, entities: List[str], relations: List[Dict]) -> Dict:
        with self._driver.session() as session:
            return session.execute_write(self._add_fact_tx, user_id, fact_text, category, entities, relations)

    @staticmethod
    def _add_fact_tx(tx, user_id, fact_text, category, entities, relations):
        fact_embedding = get_embedding(fact_text)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Create the Fact node and link it to the user
        result = tx.run("""
            MERGE (u:User {user_id: $user_id})
            CREATE (f:Fact {
                text: $fact_text, embedding: $fact_embedding,
                category: $category, created_at: datetime($now_iso)
            })
            MERGE (u)-[:HAS_FACT]->(f)
            RETURN id(f) as fact_id
        """, user_id=user_id, fact_text=fact_text, fact_embedding=fact_embedding, category=category)
        fact_id = result.single()["fact_id"]

        # Link the Fact to its mentioned entities
        if entities:
            tx.run("""
                MATCH (f:Fact) WHERE id(f) = $fact_id
                UNWIND $entities as entity_name
                MERGE (e:Entity {user_id: $user_id, name: entity_name})
                MERGE (f)-[:MENTIONS]->(e)
            """, fact_id=fact_id, entities=entities, user_id=user_id)

        # Create relationships between entities
        for rel in relations:
            rel_type = ''.join(c for c in rel['type'].upper().replace(' ', '_') if c.isalnum() or c == '_')
            tx.run(f"""
                MERGE (a:Entity {{user_id: $user_id, name: $from_entity}})
                MERGE (b:Entity {{user_id: $user_id, name: $to_entity}})
                MERGE (a)-[:`{rel_type}`]->(b)
            """, user_id=user_id, from_entity=rel['from'], to_entity=rel['to'])
        
        return {"status": "success", "fact_added": fact_text, "fact_id": fact_id}

    def retrieve_facts(self, user_id: str, query_text: str, limit: int = 5) -> List[Dict]:
        with self._driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('fact_embeddings', $limit, $query_embedding) YIELD node AS f, score
                MATCH (:User {user_id: $user_id})-[:HAS_FACT]->(f)
                RETURN f.text as fact, f.category as category, id(f) as fact_id, score as similarity
                ORDER BY score DESC
            """, limit=limit, query_embedding=get_embedding(query_text), user_id=user_id)
            return [dict(res) for res in results]

    def update_fact(self, fact_id: int, new_text: str, new_category: Optional[str] = None) -> Optional[Dict]:
        with self._driver.session() as session:
            result = session.run("""
                MATCH (f:Fact) WHERE id(f) = $fact_id
                SET f.text = $new_text, f.embedding = $new_embedding
                """ + ("SET f.category = $new_category" if new_category else "") + """
                RETURN f.text as fact, f.category as category, id(f) as fact_id
            """, fact_id=fact_id, new_text=new_text, new_embedding=get_embedding(new_text), new_category=new_category)
            return result.single(as_dictionary=True)

    def delete_fact(self, fact_id: int) -> bool:
        with self._driver.session() as session:
            result = session.run("MATCH (f:Fact) WHERE id(f) = $fact_id DETACH DELETE f", fact_id=fact_id)
            return result.consume().counters.nodes_deleted > 0

# --- MongoDB Manager with Full CRUD ---
class MongoManager:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["short_term_memories"]
        self._setup_database()
        logger.info("MongoDB Manager initialized.")

    def _setup_database(self):
        if "expire_at_ttl" not in self.collection.index_information():
            self.collection.create_index("expire_at", name="expire_at_ttl", expireAfterSeconds=0)
        # Assuming Atlas for vector search. If not, fallback search is used.
        try:
            self.collection.create_index([("embedding", "vector")], name="vector_index_fallback", background=True) # Basic index
        except Exception:
            pass # Index creation may fail on non-vector-supporting versions

    def close(self): self.client.close()

    @staticmethod
    def _serialize(doc):
        if doc:
            doc["_id"] = str(doc["_id"])
            if "created_at" in doc: doc["created_at"] = doc["created_at"].isoformat()
            if "expire_at" in doc: doc["expire_at"] = doc["expire_at"].isoformat()
        return doc

    def add_memory(self, user_id: str, content: str, ttl_seconds: int, category: str) -> Dict:
        doc = {
            "user_id": user_id, "content": content, "embedding": get_embedding(content),
            "category": category, "expire_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            "created_at": datetime.now(timezone.utc)
        }
        self.collection.insert_one(doc)
        return self._serialize(doc)

    def retrieve_memories(self, user_id: str, query_text: str, limit: int = 5) -> List[Dict]:
        pipeline = [
            {"$vectorSearch": {
                "index": "vector_search_index", "path": "embedding", "queryVector": get_embedding(query_text),
                "numCandidates": limit * 15, "limit": limit, "filter": {"user_id": {"$eq": user_id}}
            }},
            {"$project": {"_id": 1, "content": 1, "category": 1, "similarity": {"$meta": "vectorSearchScore"}}}
        ]
        try:
            return [self._serialize(doc) for doc in self.collection.aggregate(pipeline)]
        except Exception:
            logger.warning("Atlas $vectorSearch failed. Using manual cosine similarity fallback.")
            docs = list(self.collection.find({"user_id": user_id}))
            if not docs: return []
            query_vec = np.array(get_embedding(query_text))
            for doc in docs:
                doc['similarity'] = np.dot(query_vec, doc['embedding'])
            return [self._serialize(d) for d in sorted(docs, key=lambda x: x['similarity'], reverse=True)[:limit]]

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        return self._serialize(self.collection.find_one({"_id": ObjectId(memory_id)}))

    def update_memory(self, memory_id: str, new_content: Optional[str] = None, new_ttl_seconds: Optional[int] = None) -> Optional[Dict]:
        update_fields = {}
        if new_content:
            update_fields["content"] = new_content
            update_fields["embedding"] = get_embedding(new_content)
        if new_ttl_seconds:
            update_fields["expire_at"] = datetime.now(timezone.utc) + timedelta(seconds=new_ttl_seconds)

        if not update_fields: return self.get_memory_by_id(memory_id)

        return self._serialize(self.collection.find_one_and_update(
            {"_id": ObjectId(memory_id)}, {"$set": update_fields}, return_document=ReturnDocument.AFTER
        ))

    def delete_memory(self, memory_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(memory_id)})
        return result.deleted_count > 0

# --- Initialize Managers & MCP Server ---
kg_manager = Neo4jManager(Cfg.NEO4J_URI, Cfg.NEO4J_USER, Cfg.NEO4J_PASSWORD)
mongo_manager = MongoManager(Cfg.MONGO_URI, Cfg.MONGO_DB_NAME)
mcp = FastMCP(name="SentientDualMemoryServer", instructions="A server providing full CRUD and search for long-term (Graph) and short-term (Vector DB) memories.")

# --- Long-Term Memory (LTM) Tools ---
@mcp.tool()
def add_long_term_fact(user_id: str, fact_text: str, category: str, entities: List[str], relations: List[Dict]) -> Dict:
    """Saves a permanent, structured fact to the knowledge graph. Requires pre-extracted entities and relations."""
    return kg_manager.add_fact(user_id, fact_text, category, entities, relations)

@mcp.tool()
def update_long_term_fact(fact_id: int, new_text: str, new_category: Optional[str] = None) -> Dict:
    """Updates the text and/or category of an existing fact in the knowledge graph using its unique ID."""
    updated = kg_manager.update_fact(fact_id, new_text, new_category)
    return {"status": "success", "updated_fact": updated} if updated else {"status": "failure", "error": "Fact not found."}

@mcp.tool()
def delete_long_term_fact(fact_id: int) -> Dict:
    """Permanently deletes a fact from the knowledge graph using its unique ID."""
    was_deleted = kg_manager.delete_fact(fact_id)
    return {"status": "success" if was_deleted else "failure"}

# --- Short-Term Memory (STM) Tools ---
@mcp.tool()
def add_short_term_memory(user_id: str, content: str, ttl_seconds: int, category: str = "General") -> Dict:
    """Saves a temporary, expiring memory. Useful for reminders or recent conversation context."""
    return mongo_manager.add_memory(user_id, content, ttl_seconds, category)

@mcp.tool()
def update_short_term_memory(memory_id: str, new_content: Optional[str] = None, new_ttl_seconds: Optional[int] = None) -> Dict:
    """Updates the content or extends the expiry time of a short-term memory using its unique ID."""
    updated = mongo_manager.update_memory(memory_id, new_content, new_ttl_seconds)
    return {"status": "success", "updated_memory": updated} if updated else {"status": "failure", "error": "Memory not found."}

@mcp.tool()
def delete_short_term_memory(memory_id: str) -> Dict:
    """Manually deletes a short-term memory before it expires using its unique ID."""
    was_deleted = mongo_manager.delete_memory(memory_id)
    return {"status": "success" if was_deleted else "failure"}

# --- Unified Search Tool ---
@mcp.tool()
def search_all_memories(user_id: str, query_text: str, limit: int = 5) -> Dict[str, List[Dict]]:
    """Performs a semantic search across both LTM (facts) and STM (reminders) to retrieve relevant context."""
    ltm_results = kg_manager.retrieve_facts(user_id, query_text, limit)
    stm_results = mongo_manager.retrieve_memories(user_id, query_text, limit)
    return {"long_term_facts": ltm_results, "short_term_reminders": stm_results}

if __name__ == "__main__":
    logger.info("Starting Sentient Dual-Memory MCP Server with Full CRUD...")
    try:
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
    finally:
        kg_manager.close()
        mongo_manager.close()