from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import traceback
import asyncio # Required for loop.run_in_executor

# Assuming these are initialized in app.py and can be imported
from server.app.app import (
    auth,
    PermissionChecker,
    task_queue,
    mongo_manager, # For loading user profile and active_chat_id
    unified_classification_runnable, # For add_task_endpoint
    priority_runnable, # For add_task_endpoint
    add_message_to_db, # For add_task_endpoint
    get_chat_history_messages # For add_task_endpoint
)
# Pydantic models moved from app.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import datetime # for isoformat

router = APIRouter(
    prefix="/agents",
    tags=["Agents & Tasks"]
)

# --- Pydantic Models for Agent/Task Endpoints ---
class CreateTaskRequest(BaseModel):
    description: str

class UpdateTaskRequest(BaseModel):
    task_id: str
    description: str
    priority: int = Field(..., ge=0, le=2)

class DeleteTaskRequest(BaseModel):
    task_id: str

class TaskIdRequest(BaseModel):
    task_id: str

class TaskApprovalDataResponse(BaseModel):
    approval_data: Optional[Dict[str, Any]] = None

class ApproveTaskResponse(BaseModel):
    message: str
    result: Any

# --- Task Queue Endpoints ---
@router.post("/fetch-tasks", status_code=status.HTTP_200_OK, summary="Fetch User Tasks")
async def get_tasks(user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))):
    print(f"[ENDPOINT /agents/fetch-tasks] User {user_id}.")
    try:
        user_tasks = await task_queue.get_tasks_for_user(user_id)
        s_tasks = []
        for t in user_tasks:
            # Ensure datetime objects are converted to ISO format strings
            if isinstance(t.get('created_at'), datetime.datetime):
                t['created_at'] = t['created_at'].isoformat()
            if isinstance(t.get('completed_at'), datetime.datetime):
                t['completed_at'] = t['completed_at'].isoformat()
            if isinstance(t.get('processing_started_at'), datetime.datetime):
                t['processing_started_at'] = t['processing_started_at'].isoformat()
            s_tasks.append(t)
        return JSONResponse(content={"tasks": s_tasks})
    except Exception as e:
        print(f"[ERROR] /agents/fetch-tasks {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch tasks.")

@router.post("/add-task", status_code=status.HTTP_201_CREATED, summary="Add New Task")
async def add_task_endpoint(task_request: CreateTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /agents/add-task] User {user_id}, Desc: '{task_request.description[:50]}...'")
    loop = asyncio.get_event_loop()
    try:
        user_profile_for_task = await mongo_manager.get_user_profile(user_id)
        active_chat_id = user_profile_for_task.get("userData", {}).get("active_chat_id", None)

        # Ensure active_chat_id is available.
        # get_chat_history_messages will create/assign one if not present.
        _, active_chat_id_from_history = await get_chat_history_messages(user_id, active_chat_id)
        
        # Use the chat ID obtained from history logic, as it's guaranteed to be set.
        final_active_chat_id = active_chat_id_from_history
        if not final_active_chat_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not determine active chat ID for task creation.")

        username = user_profile_for_task.get("userData", {}).get("personalInfo", {}).get("name", user_id)
        
        current_chat_history_for_task, _ = await get_chat_history_messages(user_id, final_active_chat_id)
        
        unified_output = await loop.run_in_executor(None, unified_classification_runnable.invoke, {"query": task_request.description, "chat_history": current_chat_history_for_task})
        use_personal_context = unified_output.get("use_personal_context", False)
        internet_needed = unified_output.get("internet", "None")
        
        task_priority = 2
        try:
            priority_response = await loop.run_in_executor(None, priority_runnable.invoke, {"task_description": task_request.description})
            task_priority = priority_response.get("priority", 2)
        except Exception as e_prio:
            print(f"[WARN] Priority determination failed for task '{task_request.description[:30]}...': {e_prio}. Defaulting to low.")
        
        new_task_id = await task_queue.add_task(user_id=user_id, chat_id=final_active_chat_id, description=task_request.description, priority=task_priority, username=username, use_personal_context=use_personal_context, internet=internet_needed)
        
        await add_message_to_db(user_id, final_active_chat_id, task_request.description, is_user=True, is_visible=False)
        confirm_msg = f"Task added: '{task_request.description[:40]}...'"
        await add_message_to_db(user_id, final_active_chat_id, confirm_msg, is_user=False, is_visible=True, agentsUsed=True, task=task_request.description)
        
        return JSONResponse(content={"task_id": new_task_id, "message": "Task added successfully."}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        print(f"[ERROR] /agents/add-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add task.")

@router.post("/update-task", status_code=status.HTTP_200_OK, summary="Update Task")
async def update_task_endpoint(request: UpdateTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /agents/update-task] User {user_id}, Task: {request.task_id}, Prio: {request.priority}")
    try:
        updated = await task_queue.update_task(user_id, request.task_id, request.description, request.priority)
        if not updated:
            task = await task_queue.get_task_by_id(user_id, request.task_id)
            if not task:
                 raise ValueError(f"Task with id {request.task_id} for user {user_id} not found.")
            else:
                 raise ValueError(f"Cannot update task {request.task_id} with status: {task['status']}.")
        return JSONResponse(content={"message": "Task updated successfully."})
    except ValueError as ve:
        if "not found" in str(ve).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /agents/update-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update task.")

@router.post("/delete-task", status_code=status.HTTP_200_OK, summary="Delete Task")
async def delete_task_endpoint(request: DeleteTaskRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /agents/delete-task] User {user_id}, Task: {request.task_id}")
    try:
        deleted = await task_queue.delete_task(user_id, request.task_id)
        return JSONResponse(content={"message": "Task deleted successfully."})
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /agents/delete-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete task.")

@router.post("/approve-task", response_model=ApproveTaskResponse, summary="Approve Pending Task")
async def approve_task_endpoint(request: TaskIdRequest, user_id: str = Depends(PermissionChecker(required_permissions=["write:tasks"]))):
    print(f"[ENDPOINT /agents/approve-task] User {user_id}, Task: {request.task_id}")
    try:
        result_data = await task_queue.approve_task(user_id, request.task_id)
        task_details = await task_queue.get_task_by_id(user_id, request.task_id)
        
        if task_details and task_details.get("chat_id"):
            await add_message_to_db(user_id, task_details["chat_id"], str(result_data), is_user=False, is_visible=True, type="tool_result", task=task_details.get("description"), agentsUsed=True)
        
        return ApproveTaskResponse(message="Task approved and completed.", result=result_data)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        print(f"[ERROR] /agents/approve-task {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task approval failed.")

@router.post("/get-task-approval-data", response_model=TaskApprovalDataResponse, summary="Get Task Approval Data")
async def get_task_approval_data_endpoint(request: TaskIdRequest, user_id: str = Depends(PermissionChecker(required_permissions=["read:tasks"]))):
    print(f"[ENDPOINT /agents/get-task-approval-data] User {user_id}, Task: {request.task_id}")
    try:
        task = await task_queue.get_task_by_id(user_id, request.task_id)
        if task:
            if task.get("status") == "approval_pending":
                return TaskApprovalDataResponse(approval_data=task.get("approval_data"))
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task is not pending approval. Current status: {task.get('status')}")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or access denied.")
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] /agents/get-task-approval-data {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get task approval data.")