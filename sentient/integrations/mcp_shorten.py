"""Shorter versions of known, very long MCP results for the model to read (#264).

A tool result that does not fit what the model may read is cut from the end (``Agent._tool_content``). For a few
well-known MCP tools a plain cut loses the useful part, so they get a shortener here: it keeps what the model
needs for its next step and drops the rest. The full result is still saved and still shown in the window.

Shorteners are picked by the MCP tool's own name, whatever the user called the server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

PARAM_DESC_CHARS = 70
TOOL_DESC_CHARS = 120
MAX_PITFALLS = 3
MAX_OTHER_TOOLS = 8  # tools besides the main ones, by description only
PITFALL_CHARS = 160


def _parse_text(text: str) -> Any:
    """The JSON object at the start of an MCP text result (Composio adds plain guidance after it), or None."""
    start = text.lstrip()
    if not start.startswith("{"):
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(start)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def _param(spec: Any) -> str:
    """One parameter as a short line: ``type (one of a, b): description``."""
    if not isinstance(spec, dict):
        return ""
    items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
    kind = spec.get("type") or ""
    if kind == "array" and items.get("type"):
        kind = f"array of {items['type']}"
    enum = spec.get("enum") or items.get("enum")
    head = str(kind)
    if isinstance(enum, list) and enum:
        head += " (one of " + ", ".join(str(e) for e in enum[:8]) + ")"
    desc = _clip(spec.get("description"), PARAM_DESC_CHARS)
    return f"{head}: {desc}" if desc else head


def _schema(tool: Any, full: bool) -> dict:
    """A tool from ``tool_schemas``: its short description, and for a main tool its parameters in one line each."""
    if not isinstance(tool, dict):
        return {}
    out: dict[str, Any] = {"description": _clip(tool.get("description"), TOOL_DESC_CHARS)}
    schema = tool.get("input_schema")
    if full and isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        if schema.get("required"):
            out["required"] = list(schema["required"])
        out["parameters"] = {name: _param(spec) for name, spec in schema["properties"].items()}
    return out


def composio_search(result: Any) -> dict | None:
    """``COMPOSIO_SEARCH_TOOLS``: keep the recommended plan, the tool names, which apps are connected, the session
    id and the time, with parameters only for the main tools. None when the result has another shape."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        return None
    outer = _parse_text(result["content"])
    data = outer.get("data") if outer else None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return None
    results: list[dict] = []
    main: list[str] = []
    for r in data["results"]:
        if not isinstance(r, dict):
            continue
        item = {
            k: r[k]
            for k in ("use_case", "recommended_plan_steps", "known_pitfalls", "primary_tool_slugs",
                      "related_tool_slugs", "error")
            if r.get(k)
        }
        if isinstance(item.get("known_pitfalls"), list):
            item["known_pitfalls"] = [_clip(x, PITFALL_CHARS) for x in item["known_pitfalls"][:MAX_PITFALLS]]
        results.append(item)
        main += [s for s in r.get("primary_tool_slugs") or [] if isinstance(s, str) and s not in main]
    # what the next call needs comes first, so a cut at the end only loses other tools' descriptions
    out: dict[str, Any] = {"successful": outer.get("successful", data.get("success"))}
    if outer.get("error") or data.get("error"):
        out["error"] = outer.get("error") or data.get("error")
    statuses = data.get("toolkit_connection_statuses")
    if isinstance(statuses, list):
        out["toolkit_connection_statuses"] = [
            {k: s[k] for k in ("toolkit", "has_active_connection", "status_message") if k in s}
            for s in statuses
            if isinstance(s, dict)
        ]
    session = data.get("session")
    if isinstance(session, dict):
        out["session"] = {k: session[k] for k in ("id", "instructions") if k in session}
    time_info = data.get("time_info")
    if isinstance(time_info, dict) and time_info.get("current_time_utc"):
        out["current_time_utc"] = time_info["current_time_utc"]
    if data.get("next_steps_guidance"):
        out["next_steps_guidance"] = data["next_steps_guidance"]
    out["results"] = results
    schemas = data.get("tool_schemas")
    if isinstance(schemas, dict):
        first = [slug for slug in schemas if slug in main]
        others = [slug for slug in schemas if slug not in main][:MAX_OTHER_TOOLS]
        out["tool_schemas"] = {slug: _schema(schemas[slug], slug in main) for slug in first + others}
    out["note"] = "Shortened by Sentient: other tools' parameters left out; search again with a narrower use case if needed."
    return out


SHORTENERS: dict[str, Callable[[Any], Any]] = {
    "COMPOSIO_SEARCH_TOOLS": composio_search,
}


def shortener_for(mcp_name: str) -> Callable[[Any], Any] | None:
    """The shortener for an MCP tool's own name, or None."""
    return SHORTENERS.get(str(mcp_name or "").upper())
