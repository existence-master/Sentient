unified_classification_format = {
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["chat", "memory", "agent"]
    },
    "use_personal_context": {
      "type": "boolean"
    },
    "internet": {
      "type": "string",
      "enum": ["Internet", "None"]
    }
  },
  "required": ["category", "use_personal_context", "internet"],
  "additionalProperties": False
}