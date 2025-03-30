agent_required_format = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "response_type": {"type": "string", "enum": ["tool_call"]},
                    "content": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "enum": [
                                    "gmail",
                                    "gdocs",
                                    "gcalendar",
                                    "gsheets",
                                    "gslides",
                                    "gdrive",
                                ],
                            },
                            "task_instruction": {"type": "string"},
                            "previous_tool_response": {"type": "boolean"},
                        },
                        "required": [
                            "tool_name",
                            "task_instruction",
                            "previous_tool_response",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["response_type", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tool_calls"],
    "additionalProperties": False,
}

gmail_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "enum": [
                "send_email",
                "create_draft",
                "search_inbox",
                "reply_email",
                "forward_email",
                "delete_email",
                "mark_email_as_read",
                "mark_email_as_unread",
                "delete_spam_emails",
            ],
        },
        "parameters": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "format": "email"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False,
                    "description": "Parameters for sending an email.",
                },
                {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "format": "email"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False,
                    "description": "Parameters for drafting an email.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Parameters for searching the inbox.",
                },
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["query", "body"],
                    "additionalProperties": False,
                    "description": "Parameters for replying to an email.",
                },
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "to": {"type": "string", "format": "email"},
                    },
                    "required": ["query", "to"],
                    "additionalProperties": False,
                    "description": "Parameters for forwarding an email.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Parameters for deleting an email.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Parameters for marking an email as read.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Parameters for marking an email as unread.",
                },
            ]
        },
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False,
}

gdrive_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "enum": [
                "upload_file_to_gdrive",
                "search_and_download_file_from_gdrive",
                "search_file_in_gdrive",
            ],
        },
        "parameters": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "folder_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                    "additionalProperties": False,
                    "description": "Uploads a file to a specified folder in Google Drive. If folder_name is not provided, uploads to the root.",
                },
                {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["file_name", "destination"],
                    "additionalProperties": False,
                    "description": "Searches for a file in Google Drive by file name and downloads it to the specified destination.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Searches for files in Google Drive that match the given query.",
                },
            ]
        },
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False,
}

gdocs_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "enum": ["create_google_doc"]},
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "heading": {"type": "string"},
                                    "heading_level": {"type": "string", "enum": ["H1", "H2"]},
                                    "paragraphs": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "bullet_points": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "image_description": {"type": "string"}
                                },
                                "required": ["heading", "heading_level", "paragraphs", "bullet_points", "image_description"],
                                "additionalProperties": False
                            },
                            "minItems": 4,
                            "maxItems": 5
                        }
                    },
                    "required": ["title", "sections"],
                    "additionalProperties": False
                }
            },
            "required": ["content"],
            "additionalProperties": False
        }
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False
}
gsheets_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "enum": ["create_google_sheet"]},
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sheets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "table": {
                                "type": "object",
                                "properties": {
                                    "headers": {"type": "array", "items": {"type": "string"}},
                                    "rows": {
                                        "type": "array",
                                        "items": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "required": ["headers", "rows"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["title", "table"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["title", "sheets"],
            "additionalProperties": False
        }
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False
}
gslides_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "enum": ["create_google_presentation"]},
        "parameters": {
            "type": "object",
            "properties": {
                "outline": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "username": {"type": "string"},
                        "slides": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {"type": "array", "items": {"type": "string"}}
                                        ]
                                    },
                                    "image_description": {"type": "string"},
                                    "chart": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": ["bar", "pie", "line"]},
                                            "categories": {"type": "array", "items": {"type": "string"}},
                                            "data": {"type": "array", "items": {"type": "number"}}
                                        },
                                        "required": ["type", "categories", "data"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["title", "content","image_description", "chart"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["topic", "username", "slides"],
                    "additionalProperties": False
                }
            },
            "required": ["outline"],
            "additionalProperties": False
        }
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False
}

gcalendar_agent_required_format = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "enum": ["add_event", "search_events", "list_upcoming_events"],
        },
        "parameters": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "description": {"type": "string"},
                        "start": {"type": "string", "format": "date-time"},
                        "end": {"type": "string", "format": "date-time"},
                        "timezone": {"type": "string"},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string", "format": "email"},
                        },
                    },
                    "required": ["summary", "start", "end", "timezone"],
                    "additionalProperties": False,
                    "description": "Adds an event to Google Calendar with optional attendees and description.",
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                    "description": "Searches for events in Google Calendar that match the given query.",
                },
                {
                    "type": "object",
                    "properties": {"days": {"type": "integer", "minimum": 1}},
                    "required": ["days"],
                    "additionalProperties": False,
                    "description": "Lists all upcoming events within the specified number of days.",
                },
            ]
        },
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False,
}

priority_required_format = {
    "type": "object",
    "properties": {
        "priority": {
            "type": "integer",
            "enum": [0, 1, 2],
        }
    },
    "required": ["priority"],
    "additionalProperties": False,
}