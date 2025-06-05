import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from fastmcp import FastMCP
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure
from neo4j import GraphDatabase, exceptions as Neo4jExceptions
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentient_memory_test")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") # CHANGE THIS

# --- MongoDB Setup (for Short-Term Memory) ---
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command('ping')
    mongo_db = mongo_client[MONGO_DB_NAME]
    logger.info(f"Successfully connected to MongoDB: {MONGO_URI}, Database: {MONGO_DB_NAME}")
    short_term_memories_collection = mongo_db["short_term_memories"]
    if "expire_at_ttl" not in short_term_memories_collection.index_information():
        short_term_memories_collection.create_index("expire_at", name="expire_at_ttl", expireAfterSeconds=0)
        logger.info("Created TTL index on 'short_term_memories.expire_at'")
except ConnectionFailure:
    logger.error(f"Failed to connect to MongoDB at {MONGO_URI}. Please ensure MongoDB is running.")
    exit(1)

# --- Neo4j Setup (for Long-Term Knowledge Graph Memory) ---
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j: {uri}")
            self._ensure_constraints()
        except Neo4jExceptions.AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}. Check NEO4J_USER and NEO4J_PASSWORD.")
            exit(1)
        except Neo4jExceptions.ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {uri}: {e}. Ensure Neo4j is running.")
            exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during Neo4j setup: {e}")
            exit(1)

    def _ensure_constraints(self):
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT entity_user_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.user_id, e.name) IS UNIQUE")
            session.run("CREATE INDEX entity_user_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.user_id)")
            session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entityType)")
            # For full-text search on observations (Neo4j 5.x+):
            # session.run("CREATE FULLTEXT INDEX entity_observations_ft_idx IF NOT EXISTS FOR (n:Entity) ON EACH [n.observations]")
            logger.info("Ensured Neo4j constraints and indexes.")

    def close(self):
        if self._driver:
            self._driver.close()

    def _sanitize_relation_type(self, relation_type: str) -> str:
        # Basic sanitization for Cypher relationship types
        return relation_type.upper().replace(' ', '_').replace('-', '_').replace('.', '_')

    def _serialize_node(self, node):
        if not node:
            return None
        # Assuming 'Entity' nodes
        data = dict(node.items()) # Get all properties
        if 'created_at' in data and data['created_at']:
            data['created_at'] = data['created_at'].isoformat()
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = data['updated_at'].isoformat()
        return data
    
    def _serialize_relation_for_output(self, record_rel_from_query: Dict): # Renamed for clarity
        if not record_rel_from_query:
            return None
        # The input `record_rel_from_query` comes from the Cypher queries in read_graph/get_subgraph
        # which produce keys: 'from', 'to', 'relationType'
        return {
            "from": record_rel_from_query.get("from"), # Use .get() for safety, though 'from' should exist if valid_relations filter worked
            "to": record_rel_from_query.get("to"),
            "relationType": record_rel_from_query.get("relationType")
        }


    def create_entities(self, user_id: str, entities_data: List[Dict]) -> List[Dict]:
        # Ignores entities with existing names (for properties other than name)
        # but MERGE on name ensures entity is created if not present.
        # Returns list of entities that were processed (either newly created or already existing ones that were matched).
        query = """
        UNWIND $entities_data as entity_props
        MERGE (e:Entity {user_id: $user_id, name: entity_props.name})
        ON CREATE SET
            e.entityType = entity_props.entityType,
            e.observations = entity_props.observations,
            e.created_at = datetime(),
            e.updated_at = datetime()
        ON MATCH SET // Only update timestamp if matched, other props are from create or manual update
            e.updated_at = datetime()
        RETURN properties(e) as entity
        """
        with self._driver.session() as session:
            results = session.run(query, user_id=user_id, entities_data=entities_data)
            created_or_matched = [self._serialize_node(record["entity"]) for record in results]
        return created_or_matched

    def create_relations(self, user_id: str, relations_data: List[Dict]) -> List[Dict]:
        # Skips duplicate relations. Returns list of successfully created/merged relations.
        created_relations_info = []
        with self._driver.session() as session:
            for rel_props in relations_data:
                sanitized_rel_type = self._sanitize_relation_type(rel_props["relationType"])
                # MERGE will create if not exists.
                # Note: Using dynamic relationship type with backticks.
                # Storing original relationType as a property if needed for exact retrieval later,
                # or rely on consistent sanitization. For now, we use sanitized one.
                query = f"""
                MATCH (from_node:Entity {{user_id: $user_id, name: $from_name}})
                MATCH (to_node:Entity {{user_id: $user_id, name: $to_name}})
                MERGE (from_node)-[r:`{sanitized_rel_type}`]->(to_node)
                ON CREATE SET r.created_at = datetime(), r.original_relation_type = $original_relation_type
                RETURN $from_name AS from_name, $to_name AS to_name, type(r) AS type, r.original_relation_type as original_type
                """
                try:
                    result = session.run(query, user_id=user_id, from_name=rel_props["from"], to_name=rel_props["to"], original_relation_type=rel_props["relationType"])
                    record = result.single()
                    if record:
                        created_relations_info.append({
                            "from": record["from_name"],
                            "to": record["to_name"],
                            "relationType": record["original_type"] # return original for consistency with input
                        })
                except Exception as e:
                    logger.error(f"Failed to create relation {rel_props}: {e}")
        return created_relations_info

    def add_observations(self, user_id: str, observations_to_add: List[Dict]) -> List[Dict]:
        # Returns added observations per entity. Fails if entity doesn't exist (implicitly, as MATCH won't find it).
        results_summary = []
        with self._driver.session() as session:
            for item in observations_to_add:
                entity_name = item["entityName"]
                contents = item["contents"]
                if not contents: continue

                # Add only new observations
                query = """
                MATCH (e:Entity {user_id: $user_id, name: $entity_name})
                WITH e, [obs IN $new_observations WHERE NOT obs IN e.observations] AS unique_new_obs
                SET e.observations = e.observations + unique_new_obs
                SET e.updated_at = datetime()
                RETURN $entity_name AS entityName, unique_new_obs AS addedObservations
                """
                result = session.run(query, user_id=user_id, entity_name=entity_name, new_observations=contents)
                record = result.single()
                if record and record["addedObservations"]: # Only add to summary if something was actually added
                    results_summary.append({
                        "entityName": record["entityName"],
                        "addedObservations": record["addedObservations"]
                    })
        return results_summary

    def delete_entities(self, user_id: str, entity_names: List[str]) -> Dict[str, Any]:
        # Cascading deletion of associated relations.
        query = """
        UNWIND $entity_names as entity_name
        MATCH (e:Entity {user_id: $user_id, name: entity_name})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """
        with self._driver.session() as session:
            result = session.run(query, user_id=user_id, entity_names=entity_names)
            deleted_count = result.single()["deleted_count"] if result.peek() else 0
        return {"status": "success", "deleted_count": deleted_count}


    def delete_observations(self, user_id: str, deletions_data: List[Dict]) -> Dict[str, Any]:
        updated_entities_count = 0
        with self._driver.session() as session:
            for item in deletions_data:
                entity_name = item["entityName"]
                observations_to_remove = item["observations"]
                if not observations_to_remove: continue

                query = """
                MATCH (e:Entity {user_id: $user_id, name: $entity_name})
                WITH e, e.observations AS current_observations
                SET e.observations = [obs IN current_observations WHERE NOT obs IN $observations_to_remove]
                SET e.updated_at = datetime()
                RETURN CASE WHEN current_observations <> e.observations THEN 1 ELSE 0 END as updated
                """
                result = session.run(query, user_id=user_id, entity_name=entity_name, observations_to_remove=observations_to_remove)
                record = result.single()
                if record and record["updated"] > 0:
                    updated_entities_count +=1
        return {"status": "success", "updated_entities_count": updated_entities_count}


    def delete_relations(self, user_id: str, relations_to_delete: List[Dict]) -> Dict[str, Any]:
        deleted_count = 0
        with self._driver.session() as session:
            for rel_spec in relations_to_delete:
                sanitized_rel_type = self._sanitize_relation_type(rel_spec["relationType"])
                query = f"""
                MATCH (from_node:Entity {{user_id: $user_id, name: $from_name}})
                      -[r:`{sanitized_rel_type}`]->
                      (to_node:Entity {{user_id: $user_id, name: $to_name}})
                WHERE r.original_relation_type = $original_relation_type OR type(r) = $sanitized_rel_type 
                DELETE r
                RETURN count(r) as deleted
                """ # Check original_relation_type if present, otherwise sanitized type
                # This might need refinement if original_relation_type is not always set or if types can collide after sanitization
                # For now, assuming original_relation_type is the source of truth for deletion if it exists on the rel.
                
                result = session.run(query, user_id=user_id, from_name=rel_spec["from"], to_name=rel_spec["to"],
                                     original_relation_type=rel_spec["relationType"], sanitized_rel_type=sanitized_rel_type)
                record = result.single()
                if record:
                    deleted_count += record["deleted"]
        return {"status": "success", "deleted_relations_count": deleted_count}


    def read_graph(self, user_id: str) -> Dict[str, List[Dict]]:
        query = """
        MATCH (e:Entity {user_id: $user_id})
        OPTIONAL MATCH (e)-[r]->(e2:Entity {user_id: $user_id}) // Relations only between user's entities
        WITH e, कलेक्ट(DISTINCT r { .*, from_name:startNode(r).name, to_name:endNode(r).name, type:COALESCE(r.original_relation_type, type(r)) }) AS rels_data
        RETURN कलेक्ट(DISTINCT properties(e)) AS entities, 
               [rel IN apoc.coll.flatten(collect(rels_data)) WHERE rel IS NOT NULL] AS relations
        """
        # Using "कलेक्ट" (collect in Hindi/Devanagari) as a placeholder since direct collect(properties(r)) with conditions is tricky
        # A better way using APOC if available for unique relations:
        query_apoc_preferred = """
            MATCH (e:Entity {user_id: $user_id})
            WITH collect(DISTINCT properties(e)) as entity_list
            MATCH (n1:Entity {user_id: $user_id})-[r]->(n2:Entity {user_id: $user_id})
            RETURN entity_list as entities, 
                   collect(DISTINCT { from: n1.name, to: n2.name, relationType: COALESCE(r.original_relation_type, type(r)) }) as relations
        """
        # Simpler query without APOC, might duplicate entities if they have no relations
        query_simple = """
            MATCH (e:Entity {user_id: $user_id})
            WITH collect(DISTINCT properties(e)) as entity_list
            OPTIONAL MATCH (n1:Entity {user_id: $user_id})-[r]->(n2:Entity {user_id: $user_id})
            RETURN entity_list as entities, 
                   collect(DISTINCT { from: n1.name, to: n2.name, relationType: COALESCE(r.original_relation_type, type(r)) }) as relations_list
        """

        with self._driver.session() as session:
            # Using query_simple for now to avoid APOC dependency, will fix relation serialization
            results = session.run(query_simple, user_id=user_id)
            record = results.single()
            if not record:
                return {"entities": [], "relations": []}

            entities = [self._serialize_node(e) for e in record["entities"]]
            
            relations_output = []
            if record["relations_list"]:
                # Filter out null relations that can occur from OPTIONAL MATCH if no relations exist
                valid_relations = [rel for rel in record["relations_list"] if rel and rel.get("from") and rel.get("to")]
                relations_output = [self._serialize_relation_for_output(r) for r in valid_relations]
                
            return {"entities": entities, "relations": relations_output}


    def _get_subgraph(self, user_id: str, entity_names_list: List[str]) -> Dict[str, List[Dict]]:
        # Helper to get entities by name and relations ONLY between them
        query = """
        MATCH (e:Entity {user_id: $user_id})
        WHERE e.name IN $entity_names
        WITH collect(DISTINCT properties(e)) AS entity_list, $entity_names AS input_names
        MATCH (n1:Entity {user_id: $user_id})-[r]->(n2:Entity {user_id: $user_id})
        WHERE n1.name IN input_names AND n2.name IN input_names // Relations between the selected nodes
        RETURN entity_list AS entities, 
               collect(DISTINCT { from: n1.name, to: n2.name, relationType: COALESCE(r.original_relation_type, type(r)) }) AS relations_list
        """
        with self._driver.session() as session:
            results = session.run(query, user_id=user_id, entity_names=entity_names_list)
            record = results.single()
            if not record or not record["entities"]: # If no entities found, return empty
                 return {"entities": [], "relations": []}

            entities = [self._serialize_node(e) for e in record["entities"]]
            relations_output = []
            if record["relations_list"]:
                valid_relations = [rel for rel in record["relations_list"] if rel and rel.get("from") and rel.get("to")]
                relations_output = [self._serialize_relation_for_output(r) for r in valid_relations]

            return {"entities": entities, "relations": relations_output}

    def search_nodes(self, user_id: str, search_text: str) -> Dict[str, List[Dict]]: # Renamed search_query to search_text
        # Searches entity names, types, and observations.
        # Returns matching entities and relations *between* those matching entities.
        find_matching_entities_query = """
        MATCH (e:Entity {user_id: $user_id})
        WHERE toLower(e.name) CONTAINS toLower($cypher_search_param)
           OR toLower(e.entityType) CONTAINS toLower($cypher_search_param)
           OR ANY(obs IN e.observations WHERE toLower(obs) CONTAINS toLower($cypher_search_param))
        RETURN collect(DISTINCT e.name) AS matching_entity_names
        """
        with self._driver.session() as session:
            # Pass search_text as the value for the Cypher parameter $cypher_search_param
            results = session.run(
                find_matching_entities_query, 
                user_id=user_id, 
                cypher_search_param=search_text  # Use a different name for the Cypher parameter
            )
            record = results.single()
            if not record or not record["matching_entity_names"]:
                return {"entities": [], "relations": []}
            
            matching_names = record["matching_entity_names"]
            return self._get_subgraph(user_id, matching_names)

    def open_nodes(self, user_id: str, names: List[str]) -> Dict[str, List[Dict]]:
        # Retrieves specific nodes by name and relations *between* them.
        if not names:
            return {"entities": [], "relations": []}
        return self._get_subgraph(user_id, names)


kg_manager = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

mcp = FastMCP(
    name="AICompanionMemoryServer",
    instructions="This server manages memories for an AI companion. "
                 "It uses a Knowledge Graph (Neo4j) for long-term, structured memories (entities, relations, observations) "
                 "and MongoDB for short-term, temporary memories (expiring reminders, todos). "
                 "All operations MUST include a `user_id` to scope memories to a specific user."
)

# --- Helper Functions (mostly for MongoDB STM, kept from original) ---
def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return None

def _iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    if iso_str:
        try:
            dt_obj = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
            if dt_obj.tzinfo is None:
                return dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        except ValueError:
            logger.warning(f"Could not parse ISO string to datetime: {iso_str}")
            return None
    return None

def _mongo_doc_to_dict(doc: Optional[Dict]) -> Optional[Dict]:
    if doc:
        if "_id" in doc and isinstance(doc["_id"], ObjectId):
            doc["_id"] = str(doc["_id"])
        for date_field in ["created_at", "updated_at", "due_date", "expire_at"]:
            if date_field in doc and isinstance(doc[date_field], datetime):
                doc[date_field] = _datetime_to_iso(doc[date_field])
        return doc
    return None

# --- Knowledge Graph (Long-Term Memory) Tools ---
@mcp.tool()
def kg_create_entities(user_id: str, entities: List[Dict[str, Any]]) -> List[Dict]:
    """
    Create multiple new entities in the user's knowledge graph.
    Each entity object must contain: name (string, unique ID for the entity for this user),
    entityType (string, e.g., 'person', 'organization'), and observations (list of strings).
    If an entity name already exists for the user, its timestamp is updated, but other properties are not changed by this call.
    Returns a list of the processed (created or matched) entity objects.
    """
    # Basic validation
    if not all("name" in e and "entityType" in e and "observations" in e for e in entities):
        raise ValueError("Each entity must have 'name', 'entityType', and 'observations'.")
    return kg_manager.create_entities(user_id, entities)

@mcp.tool()
def kg_create_relations(user_id: str, relations: List[Dict[str, str]]) -> List[Dict]:
    """
    Create multiple new relations between entities in the user's knowledge graph.
    Each relation object must contain: from (string, source entity name), to (string, target entity name),
    and relationType (string, e.g., 'works_at', 'knows'). Relations are directed.
    Source and target entities must exist for this user. Skips duplicate relations.
    Returns a list of successfully created/merged relation objects.
    """
    if not all("from" in r and "to" in r and "relationType" in r for r in relations):
        raise ValueError("Each relation must have 'from', 'to', and 'relationType'.")
    return kg_manager.create_relations(user_id, relations)

@mcp.tool()
def kg_add_observations(user_id: str, observations_to_add: List[Dict[str, Any]]) -> List[Dict]:
    """
    Add new observations to existing entities in the user's knowledge graph.
    Input is a list of objects, each containing: entityName (string, target entity)
    and contents (list of new observation strings to add).
    Only adds observations not already present for that entity. Fails if an entity doesn't exist.
    Returns a list of objects, each detailing the entityName and the actual observations added to it.
    """
    if not all("entityName" in o and "contents" in o for o in observations_to_add):
        raise ValueError("Each observation item must have 'entityName' and 'contents'.")
    return kg_manager.add_observations(user_id, observations_to_add)

@mcp.tool()
def kg_delete_entities(user_id: str, entity_names: List[str]) -> Dict[str, Any]:
    """
    Remove entities and all their associated relations from the user's knowledge graph.
    Input is a list of entity names (strings) to delete.
    Operation is silent if an entity doesn't exist.
    Returns a status and count of deleted entities.
    """
    return kg_manager.delete_entities(user_id, entity_names)

@mcp.tool()
def kg_delete_observations(user_id: str, deletions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Remove specific observations from entities in the user's knowledge graph.
    Input is a list of objects, each containing: entityName (string, target entity)
    and observations (list of observation strings to remove).
    Operation is silent if an observation or entity doesn't exist.
    Returns a status and count of entities whose observations were updated.
    """
    if not all("entityName" in d and "observations" in d for d in deletions):
        raise ValueError("Each deletion item must have 'entityName' and 'observations'.")
    return kg_manager.delete_observations(user_id, deletions)

@mcp.tool()
def kg_delete_relations(user_id: str, relations_to_delete: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Remove specific relations from the user's knowledge graph.
    Input is a list of relation objects, each containing: from (string, source entity name),
    to (string, target entity name), and relationType (string).
    Operation is silent if a relation doesn't exist.
    Returns a status and count of deleted relations.
    """
    if not all("from" in r and "to" in r and "relationType" in r for r in relations_to_delete):
        raise ValueError("Each relation to delete must have 'from', 'to', and 'relationType'.")
    return kg_manager.delete_relations(user_id, relations_to_delete)

@mcp.tool()
def kg_read_graph(user_id: str) -> Dict[str, List[Dict]]:
    """
    Read the entire knowledge graph (all entities and relations) for the specified user.
    No input parameters other than user_id.
    Returns a dictionary with 'entities' (list of entity objects) and 'relations' (list of relation objects).
    """
    return kg_manager.read_graph(user_id)

@mcp.tool()
def kg_search_nodes(user_id: str, query: str) -> Dict[str, List[Dict]]: # LLM will call with 'query'
    """
    Search for nodes (entities) in the user's knowledge graph based on a query string.
    The search is case-insensitive and checks entity names, entity types, and observation content.
    Returns a dictionary with 'entities' (list of matching entity objects)
    and 'relations' (list of relation objects *between* these matching entities).
    """
    # 'query' from LLM is passed as 'search_text' to the kg_manager method
    return kg_manager.search_nodes(user_id, search_text=query)

@mcp.tool()
def kg_open_nodes(user_id: str, names: List[str]) -> Dict[str, List[Dict]]:
    """
    Retrieve specific nodes (entities) by their names from the user's knowledge graph.
    Input is a list of entity names (strings).
    Returns a dictionary with 'entities' (list of found entity objects)
    and 'relations' (list of relation objects *between* these retrieved entities).
    Silently skips non-existent entity names.
    """
    return kg_manager.open_nodes(user_id, names)

# --- Short-Term Memory Tools (MongoDB - unchanged from your original) ---
@mcp.tool()
def add_short_term_memory(
    user_id: str, content: str, ttl_seconds: int,
    category: Optional[str] = None, due_date_iso: Optional[str] = None
) -> Optional[Dict]:
    """... (docstring as before) ..."""
    now = datetime.now(timezone.utc)
    expire_at = now + timedelta(seconds=ttl_seconds)
    due_date = _iso_to_datetime(due_date_iso)

    memory_doc = {
        "user_id": user_id, "content": content, "category": category,
        "due_date": due_date, "expire_at": expire_at, "created_at": now,
    }
    result = short_term_memories_collection.insert_one(memory_doc)
    inserted_doc = short_term_memories_collection.find_one({"_id": result.inserted_id})
    return _mongo_doc_to_dict(inserted_doc)

@mcp.tool()
def get_short_term_memories(
    user_id: str, category: Optional[str] = None, query_text: Optional[str] = None,
    upcoming_days: Optional[int] = None
) -> List[Dict]:
    """... (docstring as before) ..."""
    find_filter: Dict[str, Any] = {"user_id": user_id}
    if category: find_filter["category"] = category
    if query_text: find_filter["content"] = {"$regex": query_text, "$options": "i"}
    if upcoming_days is not None:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        future_end = today_start + timedelta(days=upcoming_days + 1)
        find_filter["due_date"] = {"$gte": today_start, "$lt": future_end}
    sort_criteria = [("due_date", 1), ("created_at", 1)]
    memories = short_term_memories_collection.find(find_filter).sort(sort_criteria)
    return [_mongo_doc_to_dict(mem) for mem in memories if mem]


@mcp.tool()
def update_short_term_memory(
    user_id: str, memory_id: str, content: Optional[str] = None,
    ttl_seconds: Optional[int] = None, category: Optional[str] = None,
    due_date_iso: Optional[str] = None
) -> Optional[Dict]:
    """... (docstring as before) ..."""
    try: obj_id = ObjectId(memory_id)
    except Exception:
        logger.warning(f"Invalid memory_id format for STM update: {memory_id}")
        return None
    now = datetime.now(timezone.utc)
    update_fields: Dict[str, Any] = {"updated_at": now}
    if content is not None: update_fields["content"] = content
    if category is not None: update_fields["category"] = category # Allow "" or None
    if due_date_iso is not None: update_fields["due_date"] = _iso_to_datetime(due_date_iso)
    if ttl_seconds is not None: update_fields["expire_at"] = now + timedelta(seconds=ttl_seconds)

    # Check if there's anything to update beyond the timestamp
    if len(update_fields) == 1 and "updated_at" in update_fields:
        # If only timestamp, we might just want to fetch the doc or do nothing if no other changes
        # For now, we'll proceed to update the timestamp, or LLM might provide specific instructions.
        pass
    
    result = short_term_memories_collection.find_one_and_update(
        {"_id": obj_id, "user_id": user_id},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER
    )
    return _mongo_doc_to_dict(result)

@mcp.tool()
def delete_short_term_memory(user_id: str, memory_id: str) -> bool:
    """... (docstring as before) ..."""
    try: obj_id = ObjectId(memory_id)
    except Exception:
        logger.warning(f"Invalid memory_id format for STM delete: {memory_id}")
        return False
    result = short_term_memories_collection.delete_one({"_id": obj_id, "user_id": user_id})
    return result.deleted_count > 0

# --- General Search Tool (Updated) ---
@mcp.tool()
def search_all_memories(user_id: str, query_text: str) -> Dict[str, Any]:
    """
    Searches both long-term knowledge graph memories and short-term memories for a given query text.
    For long-term memories, it searches entity names, types, and observations.
    For short-term memories, it searches content.

    :param user_id: The unique identifier for the user.
    :param query_text: The text to search for.
    :return: A dictionary with 'long_term_graph' (containing 'entities' and 'relations' lists)
             and 'short_term_list' (list of matching STM documents).
    """
    ltm_graph_results = kg_manager.search_nodes(user_id, query_text)
    stm_list_results = get_short_term_memories(user_id, query_text=query_text)

    return {
        "long_term_graph": ltm_graph_results, # This is now a dict {"entities": [], "relations": []}
        "short_term_list": stm_list_results
    }

# --- Old LTM tools (commented out, replaced by KG tools) ---
# @mcp.tool()
# def set_long_term_memory(user_id: str, key: str, value: Any) -> Dict: ...
# @mcp.tool()
# def get_long_term_memory(user_id: str, key: str) -> Optional[Dict]: ...
# @mcp.tool()
# def delete_long_term_memory(user_id: str, key: str) -> bool: ...
# @mcp.tool()
# def list_long_term_memories(user_id: str, query_text: Optional[str] = None) -> List[Dict]: ...


if __name__ == "__main__":
    logger.info("Starting AI Companion Memory MCP Server with Knowledge Graph support...")
    # Example: You might want to pre-populate or test KG functions here
    # For instance, ensuring a test user exists or creating some initial entities/relations.
    # kg_manager.create_entities("test_user_kg", [{"name": "TestPerson", "entityType": "person", "observations": ["Likes Neo4j"]}])

    try:
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
    finally:
        if kg_manager:
            kg_manager.close()
        if mongo_client:
            mongo_client.close()
        logger.info("MCP Server shut down and DB connections closed.")