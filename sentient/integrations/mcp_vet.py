"""Safety check for the tools of MCP servers (#146).

An MCP tool's name, description and input schema go straight into the model's prompt, so a server can steer the
model with them ("tool poisoning"), or change them after the user trusted it (a "rug pull"). Every tool is checked
deterministically each time a server connects:

- its description, title and input schema texts against the skill linter's safety rules (hidden text, instructions
  to ignore other instructions, hide things, send data to a fixed address, read secrets...), plus orders about which
  tools to call and when ("always call this first"), which no honest tool description needs;
- its name against Sentient's own tools (a tool that shadows ``web_search`` or ``memory_save``);
- a SHA-256 hash of name, title, description, input schema and hints, compared with the hash the user approved.

A tool is offered to the model only once the user approved that exact hash. New tools and tools whose hash changed
are held back until the user looks; findings at ``error`` level make the suggested answer "decline".
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sentient.skills.lint import _INVISIBLE, security_findings

SCANNER_VERSION = "1"
MAX_TEXT = 20_000

_ORDERS = re.compile(
    r"\b(?:always|first|must|need to|have to|make sure to)\b[^.\n]{0,40}\b(?:call|use|run|invoke|execute)\b[^.\n]{0,50}"
    r"\b(?:first|before)\b"
    r"|\bbefore\s+(?:using|calling|running|invoking|answering|responding|replying|doing)\b[^.\n]{0,50}"
    r"\b(?:call|use|run|invoke)\b"
    r"|\binstead of (?:the |any )?(?:built-?in|other|sentient'?s?|default) tools?\b"
    r"|\b(?:do not|don'?t|never)\s+(?:use|call)\s+(?:any\s+)?other tools?\b",
    re.IGNORECASE,
)
_ORDERS_MESSAGE = "It tells Sentient which tools to call and when. A tool's description shouldn't give orders."


def _schema_texts(schema: Any, out: list[str]) -> list[str]:
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in {"description", "title"} and isinstance(value, str):
                out.append(value)
            elif key in {"default", "examples", "enum", "const"}:
                _strings(value, out)
            else:
                _schema_texts(value, out)
    elif isinstance(schema, list):
        for value in schema:
            _schema_texts(value, out)
    return out


def _strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)


def _annotations(tool: Any) -> dict:
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return {}
    if hasattr(ann, "model_dump"):
        return {k: v for k, v in ann.model_dump().items() if v is not None}
    return dict(ann) if isinstance(ann, dict) else {}


def tool_text(tool: Any) -> str:
    """Everything a tool says that the model reads: description, title and the texts in its input schema."""
    parts = [str(getattr(tool, "description", None) or ""), str(getattr(tool, "title", None) or "")]
    parts += _schema_texts(getattr(tool, "input_schema", None) or {}, [])
    return "\n".join(p for p in parts if p)[:MAX_TEXT]


def tool_hash(tool: Any) -> str:
    data = {
        "name": getattr(tool, "name", ""),
        "title": getattr(tool, "title", None),
        "description": getattr(tool, "description", None),
        "input_schema": getattr(tool, "input_schema", None) or {},
        "annotations": _annotations(tool),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def visible(text: str) -> str:
    """The text with invisible characters shown as ``[U+200B]``, so the user sees them in the report."""
    return _INVISIBLE.sub(lambda m: "".join(f"[U+{ord(c):04X}]" for c in m.group(0)), text or "")


def vet_tool(tool: Any, builtin: set[str], *, sanitize: Any = None) -> dict:
    """``{findings: [{rule, level, message}], level: "high"|"check"|"clear"}`` for one MCP tool. ``builtin`` holds
    the names of Sentient's own tools."""
    text = tool_text(tool)
    findings = [f for f in security_findings(text) if f["rule"] in {"danger", "hidden", "outbound"}]
    if _ORDERS.search(text) or _ORDERS.search(re.sub(_INVISIBLE, "", text)):
        findings.insert(0, {"rule": "orders", "level": "error", "message": _ORDERS_MESSAGE})
    name = str(getattr(tool, "name", "") or "")
    names = {name, name.lower(), sanitize(name) if sanitize else name}
    clash = sorted(n for n in names if n in builtin)
    if clash:
        findings.append({"rule": "shadows", "level": "error",
                         "message": f"It has the same name as Sentient's own tool {clash[0]}, so the two could be "
                                    "mixed up."})
    if _INVISIBLE.search(name):
        findings.append({"rule": "hidden", "level": "error", "message": "Its name has invisible characters in it."})
    level = "high" if any(f["level"] == "error" for f in findings) else ("check" if findings else "clear")
    return {"findings": findings, "level": level}
