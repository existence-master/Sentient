"""Shorter versions of known, very long MCP results (#264)."""

from __future__ import annotations

import json

from sentient.integrations.mcp_shorten import composio_search, shortener_for


def _param(desc: str) -> dict:
    return {"type": "string", "description": desc}


def composio_like() -> dict:
    """A search result shaped like Composio's COMPOSIO_SEARCH_TOOLS (made-up values), about 13k characters."""
    big_props = {f"option_{i}": _param("A long explanation of this option. " * 6) for i in range(30)}
    data = {
        "results": [{
            "index": 1,
            "use_case": "list today's calendar events",
            "execution_guidance": "IMPORTANT: Follow the recommended plan below.",
            "recommended_plan_steps": [
                "[Required] [Step]: List the day's agenda using CALENDAR_EVENTS_LIST_ALL (set single_events=true).",
                "[Optional] [Step]: Use CALENDAR_EVENTS_LIST for one calendar only.",
            ],
            "known_pitfalls": [f"[CALENDAR_EVENTS_LIST_ALL] pitfall {i} " + "detail " * 40 for i in range(5)],
            "difficulty": "easy",
            "primary_tool_slugs": ["CALENDAR_EVENTS_LIST_ALL"],
            "related_tool_slugs": ["CALENDAR_EVENTS_LIST", "CALENDAR_GET_TIME"],
            "toolkits": ["calendar"],
            "plan_id": "plan123",
        }],
        "toolkit_connection_statuses": [{
            "toolkit": "calendar",
            "description": "A calendar app. " * 10,
            "has_active_connection": False,
            "status_message": "No Active connection for toolkit=calendar. Call COMPOSIO_MANAGE_CONNECTIONS first.",
        }],
        "tool_schemas": {
            "CALENDAR_EVENTS_LIST_ALL": {
                "toolkit": "CALENDAR",
                "tool_slug": "CALENDAR_EVENTS_LIST_ALL",
                "description": "Return every event across calendars for a time range. " * 5,
                "input_schema": {
                    "type": "object",
                    "required": ["time_min", "time_max"],
                    "properties": {
                        "time_min": _param("Lower bound, RFC3339 with offset."),
                        "time_max": _param("Upper bound, RFC3339 with offset."),
                        "event_types": {"type": "array", "items": {"type": "string", "enum": ["default", "focusTime"]},
                                        "description": "Event types to return."},
                    },
                },
            },
            "CALENDAR_EVENTS_LIST": {
                "toolkit": "CALENDAR",
                "tool_slug": "CALENDAR_EVENTS_LIST",
                "description": "Lists events from one calendar.",
                "input_schema": {"type": "object", "properties": big_props},
            },
            "CALENDAR_GET_TIME": {"toolkit": "CALENDAR", "description": "Current time.", "hasFullSchema": False},
        },
        "time_info": {"current_time_utc": "2026-10-10T12:00:00Z", "message": "Use this time. " * 10},
        "session": {"id": "abcd", "generate_id": True, "instructions": 'Pass session_id "abcd" in later calls.'},
        "next_steps_guidance": ["1) CALL COMPOSIO_MANAGE_CONNECTIONS for toolkits that are not active"],
        "success": True,
        "error": None,
    }
    text = json.dumps({"successful": True, "data": data, "error": None, "log_id": "log_1"})
    return {"content": text + "\nNo exact fit? Any endpoint can be connected as a custom MCP."}


def test_composio_search_keeps_plan_slugs_and_connection_status():
    full = composio_like()
    assert 12_000 < len(json.dumps(full)) < 16_000
    short = composio_search(full)
    text = json.dumps(short)
    assert len(text) < len(json.dumps(full)) // 3
    result = short["results"][0]
    assert result["recommended_plan_steps"][0].startswith("[Required] [Step]: List the day's agenda")
    assert result["primary_tool_slugs"] == ["CALENDAR_EVENTS_LIST_ALL"]
    assert result["related_tool_slugs"] == ["CALENDAR_EVENTS_LIST", "CALENDAR_GET_TIME"]
    assert len(result["known_pitfalls"]) == 3
    assert short["toolkit_connection_statuses"] == [{
        "toolkit": "calendar", "has_active_connection": False,
        "status_message": "No Active connection for toolkit=calendar. Call COMPOSIO_MANAGE_CONNECTIONS first.",
    }]
    main = short["tool_schemas"]["CALENDAR_EVENTS_LIST_ALL"]
    assert main["required"] == ["time_min", "time_max"]
    assert main["parameters"]["event_types"].startswith("array of string (one of default, focusTime)")
    assert set(short["tool_schemas"]["CALENDAR_EVENTS_LIST"]) == {"description"}  # not a main tool: no parameters
    assert short["session"]["id"] == "abcd" and short["current_time_utc"] == "2026-10-10T12:00:00Z"
    assert short["next_steps_guidance"] and "custom MCP" not in text


def test_other_shapes_are_left_to_the_plain_cut():
    assert composio_search({"content": "plain text"}) is None
    assert composio_search({"content": json.dumps({"data": {"items": []}})}) is None
    assert composio_search("text") is None
    assert composio_search({"error": "failed"}) is None


def test_shortener_is_picked_by_the_tools_own_name():
    assert shortener_for("COMPOSIO_SEARCH_TOOLS") is composio_search
    assert shortener_for("composio_search_tools") is composio_search
    assert shortener_for("COMPOSIO_MULTI_EXECUTE_TOOL") is None
    assert shortener_for("") is None


def test_structured_content_that_repeats_the_text_is_left_out():
    from types import SimpleNamespace

    from sentient.integrations.mcp import _result_to_json

    def result(text: str, structured):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], structured_content=structured,
                               is_error=False)

    assert _result_to_json(result("long text", {"result": "long text"})) == {"content": "long text"}
    assert _result_to_json(result("5", {"result": 5})) == {"content": "5"}
    assert _result_to_json(result('{"a": 1}', {"a": 1})) == {"content": '{"a": 1}'}
    assert _result_to_json(result("3 events", {"events": [1, 2, 3]})) == {
        "content": "3 events", "structured": {"events": [1, 2, 3]}}


def test_next_step_details_come_first_and_other_tools_are_bounded():
    full = composio_like()
    outer = json.loads(full["content"].split("\nNo exact fit?")[0])
    for i in range(20):
        outer["data"]["tool_schemas"][f"OTHER_TOOL_{i}"] = {"description": "Another tool. " * 20}
    short = composio_search({"content": json.dumps(outer)})
    keys = list(short)
    assert keys.index("session") < keys.index("results") < keys.index("tool_schemas")
    assert keys.index("toolkit_connection_statuses") < keys.index("results")
    assert keys.index("next_steps_guidance") < keys.index("tool_schemas")
    slugs = list(short["tool_schemas"])
    assert slugs[0] == "CALENDAR_EVENTS_LIST_ALL" and len(slugs) == 9  # the main tool, then at most 8 others
