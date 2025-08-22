from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TaskStep(BaseModel):
    tool: str
    description: str

class Answer(BaseModel):
    question_id: str
    answer_text: str

class AddTaskRequest(BaseModel):
    prompt: str  # The main goal or description
    task_type: str  # "single", "swarm", or "long_form"
    schedule: Optional[Dict[str, Any]] = None # For one-time, recurring, triggered
    auto_approve_subtasks: Optional[bool] = False

class ClarificationAnswerRequest(BaseModel):
    request_id: str
    answer: str

class LongFormTaskActionRequest(BaseModel):
    action: str # "pause" or "resume"
class ClarificationAnswerRequest(BaseModel):
    request_id: str
    answer: str

class LongFormTaskActionRequest(BaseModel):
    action: str # "pause" or "resume"
class UpdateTaskRequest(BaseModel):
    taskId: str
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None
    plan: Optional[List[TaskStep]] = None
    schedule: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    assignee: Optional[str] = None
    status: Optional[str] = None

class TaskIdRequest(BaseModel):
    taskId: str
class TaskActionRequest(BaseModel):
    taskId: str
    action: str

class TaskChatRequest(BaseModel):
    taskId: str
    message: str

class AnswerClarificationRequest(BaseModel):
    task_id: str
    answers: List[Answer]

class ProgressUpdateRequest(BaseModel):
    user_id: str
    task_id: str
    run_id: str
    message: Any # Changed from str to Any to allow structured updates