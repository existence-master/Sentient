import heapq
import uuid
import asyncio
import datetime
import aiofiles
import json
from typing import Dict, List, Optional, Tuple, Union

# Global lock for thread-safe access to task data
task_lock = asyncio.Lock()

# Path to the JSON file for task persistence
TASKS_FILE = "tasks.json"

class TaskQueue:
    def __init__(self, tasks_file="tasks.json"):
        self.tasks_file = tasks_file
        self.tasks = []
        self.task_id_counter = 0
        self.lock = asyncio.Lock()
        self.current_task_execution = None  # To hold the currently executing task

    async def load_tasks(self):
        """Load tasks from the tasks.json file."""
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
                self.tasks = data.get('tasks', [])
                self.task_id_counter = data.get('task_id_counter', 0)
        except FileNotFoundError:
            self.tasks = []
            self.task_id_counter = 0
            await self.save_tasks() # Create file if not exists
        except json.JSONDecodeError:
            self.tasks = []
            self.task_id_counter = 0
            print("Error decoding tasks.json, initializing with empty tasks.")
            await self.save_tasks()

    async def save_tasks(self):
        """Save tasks to the tasks.json file."""
        data = {'tasks': self.tasks, 'task_id_counter': self.task_id_counter}
        with open(self.tasks_file, 'w') as f:
            json.dump(data, f, indent=4)

    async def add_task(self, chat_id: str, description: str, priority: int, username: str, personality: Union[Dict, str, None], use_personal_context: bool, internet: str) -> str:
        """Add a new task to the queue."""
        async with self.lock:
            task_id = f"task_{self.task_id_counter}"
            task = {
                "task_id": task_id,
                "chat_id": chat_id,
                "description": description,
                "priority": priority,
                "status": "pending",
                "username": username,
                "personality": personality,
                "use_personal_context": use_personal_context,
                "internet": internet,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "completed_at": None,
                "result": None,
                "error": None
            }
            self.tasks.append(task)
            self.task_id_counter += 1
            await self.save_tasks()
            return task_id

    async def get_next_task(self) -> Optional[Dict]:
        """Get the next pending task with the highest priority."""
        async with self.lock:
            pending_tasks = [task for task in self.tasks if task["status"] == "pending"]
            if not pending_tasks:
                return None

            # Sort by priority (lower number = higher priority) and then by creation time (FIFO for same priority)
            pending_tasks.sort(key=lambda task: (task["priority"], task["created_at"]))
            next_task = pending_tasks[0]
            next_task["status"] = "processing"
            await self.save_tasks() # Update task status immediately when processing starts
            return next_task

    async def complete_task(self, task_id: str, result: Optional[str] = None, error: Optional[str] = None):
        """Mark a task as completed and save the result."""
        async with self.lock:
            for task in self.tasks:
                if task["task_id"] == task_id:
                    task["status"] = "completed"
                    task["result"] = result
                    task["error"] = error
                    task["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                    break # Task ID is unique, no need to continue searching
            await self.save_tasks()

    async def update_task(self, task_id: str, description: str, priority: int):
        """Update a task's description and priority."""
        async with self.lock:
            for task in self.tasks:
                if task["task_id"] == task_id:
                    if task["status"] not in ["pending", "processing"]:
                        raise ValueError(f"Cannot update task with status: {task['status']}. Only pending or processing tasks can be updated.")
                    task["description"] = description
                    task["priority"] = priority
                    break
            else:
                raise ValueError(f"Task with id {task_id} not found.")
            await self.save_tasks()

    async def delete_task(self, task_id: str):
        """Delete a task by its ID."""
        async with self.lock:
            self.tasks = [task for task in self.tasks if task["task_id"] != task_id]
            await self.save_tasks()

    async def get_all_tasks(self) -> List[Dict]:
        """Return a list of all tasks."""
        async with self.lock:
            return list(self.tasks) # Return a copy to avoid external modification

    async def delete_old_completed_tasks(self, hours_threshold: int = 1):
        """Delete completed tasks older than the specified hours threshold."""
        async with self.lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            tasks_to_keep = []
            deleted_task_ids = []
            for task in self.tasks:
                if task["status"] == "completed" and task["completed_at"]:
                    completed_at_dt = datetime.datetime.fromisoformat(task["completed_at"].replace('Z', '+00:00'))
                    if now - completed_at_dt > datetime.timedelta(hours=hours_threshold):
                        deleted_task_ids.append(task["task_id"])
                        continue # Skip adding to tasks_to_keep, effectively deleting it
                tasks_to_keep.append(task)
            self.tasks = tasks_to_keep
            await self.save_tasks()
            if deleted_task_ids:
                print(f"Deleted completed tasks with IDs: {deleted_task_ids} older than {hours_threshold} hours.")