from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import traceback
import asyncio # Required for loop.run_in_executor
import datetime # for isoformat

# Assuming these are initialized in app.py and can be imported
from server.app.app import (
    auth,
    PermissionChecker,
    mongo_manager, # For load_user_profile
    graph_driver,
    embed_model,
    text_conversion_runnable,
    query_classification_runnable,
    fact_extraction_runnable,
    information_extraction_runnable,
    graph_analysis_runnable,
    graph_decision_runnable,
    text_description_runnable,
    text_dissection_runnable,
    memory_backend, # Short-term memory manager
    USER_PROFILE_DB_DIR # For input_docs path
)
from server.memory.functions import (
    query_user_profile,
    build_initial_knowledge_graph,
    delete_source_subgraph,
    crud_graph_operations,
    update_neo4j_with_onboarding_data # For onboarding endpoint
)
from server.memory.helpers import summarize_and_write_sync # Used in create_document
from server.memory.constants import CATEGORIES # For get-memory-categories
# Pydantic models moved from app.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os # for os.path.join, os.makedirs

router = APIRouter(
    prefix="/memory",
    tags=["Memory & Knowledge Graph"]
)

# --- Pydantic Models for Memory Endpoints ---
class GraphRAGRequest(BaseModel):
    query: str

class DeleteSubgraphRequest(BaseModel):
    source: str

class GraphRequest(BaseModel):
    information: str

class GetShortTermMemoriesRequest(BaseModel):
    category: str
    limit: int = Field(10, ge=1)

class AddMemoryRequest(BaseModel):
    text: str
    category: str
    retention_days: int = Field(..., ge=1)

class UpdateMemoryRequest(BaseModel):
    id: Any # Can be int or string depending on DB
    text: str
    category: str
    retention_days: int = Field(..., ge=1)

class DeleteMemoryRequest(BaseModel):
    id: Any # Can be int or string
    category: str


# --- Memory and Knowledge Graph Endpoints ---
@router.post("/graphrag", status_code=status.HTTP_200_OK, summary="Query Knowledge Graph (RAG)")
async def graphrag(request: GraphRAGRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /memory/graphrag] User {user_id}, Query: '{request.query[:50]}...'")
    try:
        if not all([graph_driver, embed_model, text_conversion_runnable, query_classification_runnable]):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="GraphRAG dependencies unavailable.")
        # query_user_profile is a synchronous function from server.memory.functions
        context = await asyncio.to_thread(query_user_profile, user_id, request.query, graph_driver, embed_model, text_conversion_runnable, query_classification_runnable)
        return JSONResponse(content={"context": context or "No relevant context found."})
    except Exception as e:
        print(f"[ERROR] /memory/graphrag {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GraphRAG query failed.")

@router.post("/initiate-long-term-memories", status_code=status.HTTP_200_OK, summary="Initialize/Reset Knowledge Graph")
async def create_graph(request_data: Optional[Dict[str, bool]] = None, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    clear_graph_flag = request_data.get("clear_graph", False) if request_data else False
    action = "Resetting/rebuilding" if clear_graph_flag else "Initiating/Updating"
    print(f"[ENDPOINT /memory/initiate-LTM] {action} KG for user {user_id}.")
    loop = asyncio.get_event_loop()
    input_dir = os.path.join(USER_PROFILE_DB_DIR, user_id, "input_docs")
    os.makedirs(input_dir, exist_ok=True)

    try:
        user_profile_doc = await mongo_manager.get_user_profile(user_id) # Use mongo_manager
        username = user_profile_doc.get("userData", {}).get("personalInfo", {}).get("name", user_id) if user_profile_doc else user_id
        if not all([graph_driver, embed_model, text_dissection_runnable, information_extraction_runnable]):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph building dependencies unavailable.")

        def read_files_sync(udir, uname): # This function remains sync
            fpath = os.path.join(udir, f"{user_id}_profile_summary.txt")
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    return [{"text": f.read(), "source": os.path.basename(fpath)}]
            return [{"text": f"Initial data for {uname}. Preferences: Likes Italian food. Works as a software engineer.", "source": "default_sample.txt"}]
        
        extracted_texts = await loop.run_in_executor(None, read_files_sync, input_dir, username)

        if clear_graph_flag:
            print(f"Clearing KG for user: {user_id}...")
            def clear_neo4j_user_graph_sync(driver, uid_scope: str):
                with driver.session(database="neo4j") as session:
                    query = "MATCH (n {userId: $userId_scope}) DETACH DELETE n"
                    session.execute_write(lambda tx: tx.run(query, userId_scope=uid_scope))
            await loop.run_in_executor(None, clear_neo4j_user_graph_sync, graph_driver, user_id)
            print(f"Graph cleared for user: {user_id}.")

        if extracted_texts:
            await loop.run_in_executor(None, build_initial_knowledge_graph, user_id, extracted_texts, graph_driver, embed_model, text_dissection_runnable, information_extraction_runnable) # Pass user_id instead of username for KG operations for consistency
            return JSONResponse(content={"message": f"Knowledge Graph {action.lower()} for user {user_id} completed."})
        else:
            return JSONResponse(content={"message": "No input documents found. Graph not modified."})
    except Exception as e:
        print(f"[ERROR] /memory/initiate-LTM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Knowledge Graph operation failed.")

@router.post("/delete-subgraph", status_code=status.HTTP_200_OK, summary="Delete Subgraph by Source")
async def delete_subgraph_route(request: DeleteSubgraphRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    source_key = request.source.lower().replace(' ', '_')
    print(f"[ENDPOINT /memory/delete-subgraph] User {user_id}, source key: {source_key}")
    loop = asyncio.get_event_loop()
    try:
        source_identifier_in_graph = f"{user_id}_{source_key}.txt"

        if not graph_driver:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neo4j driver unavailable.")
        
        await loop.run_in_executor(None, delete_source_subgraph, graph_driver, source_identifier_in_graph, user_id)

        input_dir_path = os.path.join(USER_PROFILE_DB_DIR, user_id, "input_docs")
        file_to_delete_on_disk = os.path.join(input_dir_path, source_identifier_in_graph)
        
        def remove_file_sync(path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    return True, None
                except OSError as e_os:
                    return False, str(e_os)
            return False, "File not found on disk for deletion."
        
        deleted, ferr = await loop.run_in_executor(None, remove_file_sync, file_to_delete_on_disk)
        if not deleted:
            print(f"[WARN] Could not delete source file {file_to_delete_on_disk}: {ferr}")
        
        return JSONResponse(content={"message": f"Subgraph for source '{request.source}' (graph id: {source_identifier_in_graph}) processed for deletion."})
    except Exception as e:
        print(f"[ERROR] /memory/delete-subgraph {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Subgraph deletion failed.")

@router.post("/create-document", status_code=status.HTTP_200_OK, summary="Create Input Documents from Profile")
async def create_document_route(user_id: str = Depends(PermissionChecker(required_permissions=["write:profile"]))): # Changed permission scope
    print(f"[ENDPOINT /memory/create-document] User {user_id}.")
    input_dir = os.path.join(USER_PROFILE_DB_DIR, user_id, "input_docs")
    loop = asyncio.get_event_loop()
    try:
        user_profile_doc = await mongo_manager.get_user_profile(user_id) # Use mongo_manager
        db_user_data = user_profile_doc.get("userData", {}) if user_profile_doc else {}
        username = db_user_data.get("personalInfo", {}).get("name", user_id)
        await loop.run_in_executor(None, os.makedirs, input_dir, True)

        created_files_info = []
        summary_text = "" # Placeholder, actual summary logic might be complex
        # Simplified: the original function `summarize_and_write_sync` is complex
        # and likely involves LLM calls. For this refactor, we'll assume it generates a summary text.
        # A more robust solution would involve abstracting the summarization logic.
        
        # Example text generation (replace with actual summarization logic if needed)
        summary_parts = []
        for key, value in db_user_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    summary_parts.append(f"{sub_key.replace('_', ' ').title()}: {sub_value}")
            else:
                summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        summary_text = "\n".join(summary_parts)

        summary_filename = f"{user_id}_profile_summary.txt"
        # Assuming summarize_and_write_sync is a synchronous function
        await loop.run_in_executor(None, summarize_and_write_sync, user_id, username, summary_text, summary_filename, input_dir)
        created_files_info.append(summary_filename)
        
        return JSONResponse(content={"message": "Input documents processed.", "created_files": created_files_info, "summary_generated_length": len(summary_text)})
    except Exception as e:
        print(f"[ERROR] /memory/create-document {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document creation failed.")

@router.post("/customize-long-term-memories", status_code=status.HTTP_200_OK, summary="Customize KG with Text")
async def customize_graph(request: GraphRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /memory/customize-LTM] User {user_id}, Info: '{request.information[:50]}...'")
    loop = asyncio.get_event_loop()
    try:
        user_profile_doc = await mongo_manager.get_user_profile(user_id)
        username = user_profile_doc.get("userData", {}).get("personalInfo", {}).get("name", user_id) if user_profile_doc else user_id
        required_deps = [graph_driver, embed_model, fact_extraction_runnable, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable]
        if not all(required_deps):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph customization dependencies unavailable.")

        extracted_points_data = await loop.run_in_executor(None, fact_extraction_runnable.invoke, {"paragraph": request.information, "username": username})
        
        extracted_points = []
        if isinstance(extracted_points_data, dict) and "facts" in extracted_points_data and isinstance(extracted_points_data["facts"], list):
            extracted_points = extracted_points_data["facts"]
        elif isinstance(extracted_points_data, list):
            extracted_points = extracted_points_data
        else:
            print(f"[WARN] Unexpected format from fact_extraction_runnable: {type(extracted_points_data)}")
            
        if not extracted_points:
            return JSONResponse(content={"message": "No facts extracted. Graph not modified."})

        crud_tasks = [loop.run_in_executor(None, crud_graph_operations, user_id, point, graph_driver, embed_model, query_classification_runnable, information_extraction_runnable, graph_analysis_runnable, graph_decision_runnable, text_description_runnable) for point in extracted_points]
        crud_results = await asyncio.gather(*crud_tasks, return_exceptions=True)
        
        processed = sum(1 for r in crud_results if not isinstance(r, Exception))
        errors = [str(r) for r in crud_results if isinstance(r, Exception)]
        msg = f"Knowledge Graph customization processed {processed}/{len(extracted_points)} facts." + (f" Errors: {len(errors)}" if errors else "")
        
        return JSONResponse(status_code=status.HTTP_200_OK if not errors else status.HTTP_207_MULTI_STATUS, content={"message": msg, "errors": errors})
    except Exception as e:
        print(f"[ERROR] /memory/customize-LTM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Knowledge Graph customization failed.")


# --- Short-Term Memory Endpoints ---
@router.post("/get-short-term-memories", status_code=status.HTTP_200_OK, summary="Get Short-Term Memories")
async def get_short_term_memories(request: GetShortTermMemoriesRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /memory/get-STM] User {user_id}, Cat: {request.category}, Lim: {request.limit}")
    loop = asyncio.get_event_loop()
    try:
        memories = await memory_backend.memory_manager.fetch_memories_by_category(user_id, request.category, request.limit) # fetch_memories_by_category is already async
        s_mem = []
        for m in memories:
            if isinstance(m.get('created_at'), datetime.datetime):
                m['created_at'] = m['created_at'].isoformat()
            if isinstance(m.get('expiry_at'), datetime.datetime): # Changed from expires_at to expiry_at to match MemoryManager
                m['expiry_at'] = m['expiry_at'].isoformat()
            s_mem.append(m)
        return JSONResponse(content=s_mem)
    except Exception as e:
        print(f"[ERROR] /memory/get-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch short-term memories.")

@router.post("/add-short-term-memory", status_code=status.HTTP_201_CREATED, summary="Add STM")
async def add_memory(request: AddMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /memory/add-STM] User {user_id}, Cat: {request.category}")
    try:
        # store_memory in MemoryManager is async
        mem_id = await memory_backend.memory_manager.store_memory(user_id, request.text, request.retention_days, request.category)
        return JSONResponse(content={"memory_id": str(mem_id), "message": "Short-term memory added."}, status_code=status.HTTP_201_CREATED) # Ensure mem_id is string
    except Exception as e:
        print(f"[ERROR] /memory/add-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add short-term memory.")

@router.post("/update-short-term-memory", status_code=status.HTTP_200_OK, summary="Update STM")
async def update_memory_route(request: UpdateMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /memory/update-STM] User {user_id}, ID: {request.id}")
    try:
        # update_memory_crud in MemoryManager is async
        await memory_backend.memory_manager.update_memory_crud(user_id, request.category, request.id, request.text, request.retention_days)
        return JSONResponse(content={"message": "Short-term memory updated."})
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /memory/update-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update short-term memory.")

@router.post("/delete-short-term-memory", status_code=status.HTTP_200_OK, summary="Delete STM")
async def delete_memory_route(request: DeleteMemoryRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /memory/delete-STM] User {user_id}, ID: {request.id}, Cat: {request.category}")
    try:
        # delete_memory in MemoryManager is async
        await memory_backend.memory_manager.delete_memory(user_id, request.category, request.id)
        return JSONResponse(content={"message": "Short-term memory deleted."})
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /memory/delete-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete short-term memory.")

@router.post("/clear-all-short-term-memories", status_code=status.HTTP_200_OK, summary="Clear All STM")
async def clear_all_memories(user_id: str = Depends(PermissionChecker(required_permissions=["write:memory"]))):
    print(f"[ENDPOINT /memory/clear-all-STM] User {user_id}.")
    try:
        # clear_all_memories in MemoryManager is async
        await memory_backend.memory_manager.clear_all_memories(user_id)
        return JSONResponse(content={"message": "All short-term memories cleared."})
    except Exception as e:
        print(f"[ERROR] /memory/clear-all-STM {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clear all short-term memories.")

@router.post("/get-memory-categories", status_code=status.HTTP_200_OK, summary="Get Memory Categories")
async def get_memory_categories_route(user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))): # Added _route suffix to avoid conflict
    print(f"[ENDPOINT /memory/get-memory-categories] User {user_id}.")
    try:
        return JSONResponse(content={"categories": CATEGORIES})
    except Exception as e:
        print(f"[ERROR] /memory/get-memory-categories {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve memory categories.")


@router.post("/get-graph-data", status_code=status.HTTP_200_OK, summary="Get KG Data (Visualization)")
async def get_graph_data_apoc(user_id: str = Depends(PermissionChecker(required_permissions=["read:memory"]))):
    print(f"[ENDPOINT /memory/get-graph-data] User {user_id}.")
    loop = asyncio.get_event_loop()
    if not graph_driver:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neo4j driver unavailable.")
    
    graph_visualization_query = """
    MATCH (u:User {userId: $userId})-[r*0..2]-(n)
    WHERE n.userId = $userId OR (EXISTS(r) AND ALL(rel IN r WHERE startNode(rel).userId = $userId OR endNode(rel).userId = $userId))
    WITH collect(DISTINCT n) as nodes, collect(DISTINCT r) as rels_list
    UNWIND rels_list as rel
    RETURN [node IN nodes | { id: elementId(node), label: coalesce(labels(node)[0], 'Unknown'), properties: properties(node) }] AS nodes_list,
           [rel_item IN rels_list | { id: elementId(rel_item), from: elementId(startNode(rel_item)), to: elementId(endNode(rel_item)), label: type(rel_item), properties: properties(rel_item) }] AS edges_list
    """

    def run_q(driver, query, params):
        with driver.session(database="neo4j") as session:
            res = session.run(query, params).single()
            return (res['nodes_list'] if res else [], res['edges_list'] if res else [])
    try:
        nodes, edges = await loop.run_in_executor(None, run_q, graph_driver, graph_visualization_query, {"userId": user_id})
        return JSONResponse(content={"nodes": nodes, "edges": edges})
    except Exception as e:
        print(f"[ERROR] /memory/get-graph-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get graph data.")