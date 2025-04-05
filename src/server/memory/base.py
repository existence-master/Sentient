import asyncio
import json
import datetime
from typing import Dict, List, Optional

class MemoryQueue:
    def __init__(self, memory_file="memory_operations.json"):
        self.memory_file = memory_file
        self.operations = []
        self.operation_id_counter = 0
        self.lock = asyncio.Lock()
        self.current_operation_execution = None

    async def load_operations(self):
        """Load memory operations from the JSON file."""
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.operations = data.get('operations', [])
                self.operation_id_counter = data.get('operation_id_counter', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.operations = []
            self.operation_id_counter = 0
            await self.save_operations()

    async def save_operations(self):
        """Save memory operations to the JSON file."""
        data = {'operations': self.operations, 'operation_id_counter': self.operation_id_counter}
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=4)

    async def add_operation(self, user_id: str, memory_data: Dict) -> str:
        """Add a new memory operation to the queue."""
        async with self.lock:
            operation_id = f"mem_op_{self.operation_id_counter}"
            operation = {
                "operation_id": operation_id,
                "user_id": user_id,
                "memory_data": memory_data,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "status": "pending"
            }
            self.operations.append(operation)
            self.operation_id_counter += 1
            await self.save_operations()
            return operation_id

    async def get_next_operation(self) -> Optional[Dict]:
        """Get the next pending memory operation."""
        async with self.lock:
            pending_operations = [op for op in self.operations if op["status"] == "pending"]
            if not pending_operations:
                return None
            next_operation = pending_operations[0]
            next_operation["status"] = "processing"
            await self.save_operations()
            return next_operation

    async def complete_operation(self, operation_id: str, result: Optional[str] = None, error: Optional[str] = None):
        """Mark a memory operation as completed or failed."""
        async with self.lock:
            for operation in self.operations:
                if operation["operation_id"] == operation_id:
                    operation["status"] = "completed" if not error else "failed"
                    operation["result"] = result
                    operation["error"] = error
                    operation["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                    break
            await self.save_operations()