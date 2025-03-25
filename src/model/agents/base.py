import heapq
import uuid
import asyncio
import datetime
import aiofiles
import json
from typing import Dict, List, Optional, Tuple

# Global lock for thread-safe access to task data
task_lock = asyncio.Lock()

# Path to the JSON file for task persistence
TASKS_FILE = "tasks.json"

class TaskQueue:
    def __init__(self):
        # Heapq list for "to do" tasks: (priority, task_id)
        self.todo_tasks: List[Tuple[int, str]] = []
        # Dictionary for all tasks: task_id -> task_dict
        self.all_tasks: Dict[str, dict] = {}
        # Current task being processed: task_id or None
        self.current_task: Optional[str] = None

    async def add_task(self, chat_id: str, description: str, priority: int, username: str, personality: str, use_personal_context: bool, internet: str) -> str:
        task_id = str(uuid.uuid4())
        task_dict = {
            "task_id": task_id,
            "chat_id": chat_id,
            "description": description,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "priority": priority,
            "status": "to do",
            "username": username,
            "personality": personality,
            "use_personal_context": use_personal_context,
            "internet": internet
        }
        
        async with task_lock:
            self.all_tasks[task_id] = task_dict
            heapq.heappush(self.todo_tasks, (priority, task_id))
            await self.save_tasks()
            
        return task_id

    async def update_task(self, task_id: str, description: str, priority: int):
        async with task_lock:
            if task_id not in self.all_tasks:
                raise ValueError("Task not found")
            task = self.all_tasks[task_id]
            if task["status"] == "in progress":
                raise ValueError("Cannot update a task in progress")
            task["description"] = description
            task["priority"] = priority
            if task["status"] == "to do":
                self.todo_tasks = [t for t in self.todo_tasks if t[1] != task_id]
                heapq.heappush(self.todo_tasks, (priority, task_id))
            await self.save_tasks()

    async def delete_task(self, task_id: str):
        async with task_lock:
            if task_id not in self.all_tasks:
                raise ValueError("Task not found")
            task = self.all_tasks[task_id]
            if task["status"] == "to do":
                self.todo_tasks = [t for t in self.todo_tasks if t[1] != task_id]
            elif task["status"] == "in progress" and self.current_task == task_id:
                if self.current_task_execution:
                    self.current_task_execution.cancel()
                self.current_task = None
            del self.all_tasks[task_id]
            await self.save_tasks()

    async def get_next_task(self):
        async with task_lock:
            if self.current_task is None and self.todo_tasks:
                priority, task_id = heapq.heappop(self.todo_tasks)
                self.current_task = task_id
                self.all_tasks[task_id]["status"] = "in progress"
                await self.save_tasks()
                return self.all_tasks[task_id]
        return None

    async def complete_task(self, task_id: str, result: str = None, error: str = None):
        async with task_lock:
            if task_id in self.all_tasks:
                task = self.all_tasks[task_id]
                task["status"] = "done" if not error else "error"
                if result:
                    task["result"] = result
                if error:
                    task["error"] = error
                if self.current_task == task_id:
                    self.current_task = None
                    self.current_task_execution = None
                await self.save_tasks()

    async def get_all_tasks(self) -> List[dict]:
        """Return a list of all tasks for frontend access."""
        async with task_lock:
            return list(self.all_tasks.values())

    async def save_tasks(self):
        """Save all tasks to the JSON file without acquiring the lock."""
        async with aiofiles.open(TASKS_FILE, "w") as f:
            await f.write(json.dumps(self.all_tasks))

    async def load_tasks(self):
        """Load tasks from the JSON file and reconstruct the heapq list."""
        try:
            with open(TASKS_FILE, "r") as f:
                loaded_tasks = json.load(f)
            async with task_lock:
                self.all_tasks = loaded_tasks
                self.todo_tasks = []
                for task_id, task in loaded_tasks.items():
                    if task["status"] == "to do":
                        heapq.heappush(self.todo_tasks, (task["priority"], task_id))
                    elif task["status"] == "in progress":
                        # Reset in-progress tasks to "to do" since processing was interrupted
                        task["status"] = "to do"
                        heapq.heappush(self.todo_tasks, (task["priority"], task_id))
                # Reset current_task since processing was interrupted
                self.current_task = None
        except FileNotFoundError:
            # No tasks to load if the file doesn’t exist yet
            pass