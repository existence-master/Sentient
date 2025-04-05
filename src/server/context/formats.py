gmail_context_engine_required_format = {
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "description": "Description of the task"
          },
          "priority": {
            "type": "integer",
            "description": "Priority of the task",
            "enum": [0, 1, 2]
          }
        },
        "required": [
          "description",
          "priority"
        ],
        "additionalProperties": False
      },
      "description": "List of tasks"
    },
    "memory_operations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string",
            "description": "Text for memory operation"
          }
        },
        "required": [
          "text"
        ],
        "additionalProperties": False
      },
      "description": "List of memory operations"
    },
    "message": {
        "type": "string",
        "description": "Message to the user"
      },
  },
  "additionalProperties": False,
  "description": "JSON format for tasks, memory operations and messages"
}

internet_search_context_engine_required_format = {
  "type": "object",
  "properties": {
    "message": {
        "type": "string",
        "description": "Message to the user"
    }
  },
  "additionalProperties": False,
  "description": "JSON format for only messages"
}

gcalendar_context_engine_required_format = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the task"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority of the task",
                        "enum": [0, 1, 2]
                    }
                },
                "required": ["description", "priority"],
                "additionalProperties": False
            },
            "description": "List of tasks"
        },
        "memory_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text for memory operation"
                    }
                },
                "required": ["text"],
                "additionalProperties": False
            },
            "description": "List of memory operations"
        },
        "message": {
            "type": "string",
            "description": "Message to the user"
        }
    },
    "additionalProperties": False,
    "description": "JSON format for tasks, memory operations, and message"
}