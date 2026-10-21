"""Tasks made by an MCP app (issue #443): another app asks Sentient to do something, through the
``task_create`` tool.

The app's words are outside content (ADR 0018, and ADR 0033 for tasks by email), so such a task is
treated exactly like a task by email:

- it is never refined (refining would read the app's prompt as the user's own words);
- its plan always waits for the user's approval, whatever ``tasks.require_plan_approval`` says;
- every run is marked from the start with the app's name, so anything that sends asks first;
- its brief is not the user's own words to the second look.
"""

from __future__ import annotations

SOURCE = "mcp"  # tasks.source and original_context.source of a task an MCP app made


def is_mcp_task(task: dict | None) -> bool:
    return isinstance(task, dict) and (task.get("original_context") or {}).get("source") == SOURCE


def app_name(task: dict) -> str:
    """The app that asked, as the approval card names it."""
    return str((task.get("original_context") or {}).get("app") or "An MCP app")
