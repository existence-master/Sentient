text_dissection_required_format = {
    "type": "object",
    "properties": {
        "user_name": {"type": "string"},
        "categories": {
            "type": "object",
            "properties": {
                "Personal & Well-being": {"type": "string"},
                "Professional & Academic": {"type": "string"},
                "Social & Relationships": {"type": "string"},
                "Financial": {"type": "string"},
                "Goals & Tasks": {"type": "string"},
                "Interests & Hobbies": {"type": "string"},
                "Logistics & Practical": {"type": "string"},
                "Spiritual": {"type": "string"},
            },
            "required": [
                "Personal & Well-being",
                "Professional & Academic",
                "Social & Relationships",
                "Financial",
                "Goals & Tasks",
                "Interests & Hobbies",
                "Logistics & Practical",
                "Spiritual",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["user_name", "categories"],
    "additionalProperties": False,
}

information_extraction_required_format = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "root_node": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
            "additionalProperties": False,
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relationship": {"type": "string"},
                    "target": {"type": "string"},
                    "source_properties": {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "relationship_properties": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                    "target_properties": {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "source",
                    "relationship",
                    "target",
                    "source_properties",
                    "relationship_properties",
                    "target_properties",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["category", "root_node", "relationships"],
    "additionalProperties": False,
}

graph_analysis_required_format = {
    "type": "array",
    "items": {
        "type": "object",
        "oneOf": [
            {
                "properties": {
                    "node": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "delete"],
                    },
                    "properties": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["node", "action"],
                "additionalProperties": False,
            },
            {
                "properties": {
                    "relationship": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["type", "source", "target"],
                        "additionalProperties": False,
                    },
                    "action": {"type": "string", "enum": ["create"]},
                },
                "required": ["relationship", "action"],
                "additionalProperties": False,
            },
        ],
    },
}

graph_decision_required_format = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "update", "delete"]},
            "node": {"type": "string"},
            "properties": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "type": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["create", "update", "delete"],
                        },
                        "target_properties": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["target", "type", "action"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["action", "node", "properties", "relationships"],
        "additionalProperties": False,
    },
}

query_classification_required_format = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "category": {"type": "string"}},
    "required": ["query", "category"],
    "additionalProperties": False,
}

fact_extraction_required_format = {"type": "array", "items": {"type": "string"}}

interest_extraction_required_format = {
  "type": "array",
  "items": {
    "type": "string"
  }
}

extract_memory_required_format = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The complete memory statement with context, reasoning, and specific dates"
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "personal_wellbeing",
                            "professional_academic",
                            "social_relationships",
                            "financial",
                            "goals_tasks",
                            "interests_hobbies",
                            "logistics_practical",
                            "spiritual"
                        ],
                        "description": "The category that best describes the memory"
                    }
                },
                "required": ["text", "category"]
            },
            "description": "A list of extracted memories with their categories based on the user query. Each memory should be clear, standalone, and factual."
        }
    },
    "required": ["memories"]
}


update_required_format = {
    "type": "object",
    "properties": {
        "update": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "number"},  # Changed from string to number
                    "text": {"type": "string"}
                },
                "required": ["id", "text"]
            }
        }
    },
    "required": ["update"]
}
