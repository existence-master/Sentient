"""Tool calls a small model wrote out as text instead of making them (#532).

Real qwen3:8b sometimes answers with the call itself as text: ``{"name": "memory_remember", "arguments": {...}}``,
the same inside ``<tool_call>`` tags or a code fence, or ``memory_remember({...})``, now and then after a short
sentence ("I'll note that.") and once wrapped in a made-up ``[inside content CODE ... ]`` marker. Nothing ran, and
the user saw the raw call.

Two parts, both deterministic:

- ``held_from`` tells the loop which tail of a streaming reply could still turn out to be such a call, so that part
  is held back instead of shown. Anything that clearly is not a call (prose, other code) is shown at once.
- ``find`` reads a finished reply. When it is (essentially) only written-out calls of known tools, with arguments
  that fit each tool, the loop makes them real calls through the usual path, so risk, rules and approvals all apply.
  A written call that can't be run (broken JSON, wrong arguments, an unknown tool in ``<tool_call>`` tags) gets one
  nudge instead. A code example inside a normal prose answer is left alone: only a reply that is just the call, or
  a short sentence around it, counts.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from sentient.llm.provider import parse_json_loose, tool_arguments

# keys that name the tool, in order; "type" only counts when its value is a tool's name (qwen wrote that once)
NAME_KEYS = ("name", "tool", "tool_name", "function", "type")
ARG_KEYS = ("arguments", "parameters", "args", "input", "params")
TAGS = ("<tool_call>", "<tool_calls>")
TAG_RE = re.compile(r"<tool_calls?>(.*?)(?:</tool_calls?>|$)", re.DOTALL)
# prose around written calls, in characters: up to this the call is the reply's action ("I'll note that." + call)
MAX_AROUND = 400
# a JSON object this long with no tool name yet is not a call
MAX_UNNAMED = 300
NUDGE = (
    "You wrote the {name} call out as text, so it did not run and the user can't see it. "
    "Make the tool call itself now, or answer the user in plain words."
)
NOT_RUN_REPLY = "I tried to use one of my tools, but the step didn't go through. Please ask me again."
SchemaFn = Callable[[str], "dict | None"]  # a known tool's JSON schema for its arguments, None for unknown names

_WRAPPER_LINE = re.compile(r"^\s*(?:```[\w-]*|</?tool_calls?>|\[[^\n]*)\s*$")
_TRAILING_WRAPPERS = re.compile(r"(?:\s*(?:```[\w-]*|<tool_calls?>|\[[^\]\n]*|`))*\s*$")
_IDENT = re.compile(r"(?:functions\.)?([A-Za-z_]\w*)")
_NAME_FIELD = re.compile(r'"(name|tool|tool_name|function|type)"\s*:\s*"((?:[^"\\]|\\.)*)("?)')
_CALL_START = re.compile(r"(?<![\w.])(?:functions\.)?([A-Za-z_]\w*)\s*\(")


@dataclass
class Found:
    """Written-out calls in a finished reply."""

    calls: list[tuple[str, dict]] = field(default_factory=list)  # (tool, arguments) that fit their tool
    names: list[str] = field(default_factory=list)  # every tool named in written calls, runnable or not
    before: str = ""  # the prose before the first written call, without wrapper bits ("```json", "<tool_call>")
    cleaned: str = ""  # the reply without the written calls
    runnable: bool = False  # the reply is the call(s), and every one fits its tool: run them
    is_action: bool = False  # the reply is the call(s) (alone or with a short sentence), not prose with an example


def _strip_prefix(name: str) -> str:
    return name.removeprefix("functions.").strip()


# ---------------------------------------------------------------------- while streaming
def held_from(text: str, names: Collection[str], start: int = 0) -> int:
    """Index in ``text`` from which the reply must be held back because it could still be a written-out call of one
    of ``names``; ``len(text)`` when all of it can be shown. Only lines from ``start`` on are looked at."""
    tag = _tag_start(text, start)
    ls = start
    while ls < tag:
        if _could_be_call(text[ls:], names):
            return ls
        nl = text.find("\n", ls)
        if nl == -1:
            break
        ls = nl + 1
    return tag


def line_start(text: str, pos: int) -> int:
    """Start of the line that holds ``pos``."""
    return text.rfind("\n", 0, pos) + 1


def _tag_start(text: str, start: int) -> int:
    """Where a ``<tool_call>`` tag (or the start of one, at the very end) begins; ``len(text)`` when none."""
    idx = text.find("<tool_call", start)
    if idx != -1:
        return idx
    lt = text.rfind("<", start)
    if lt != -1 and "<tool_call>".startswith(text[lt:]):
        return lt
    return len(text)


def _could_be_call(s: str, names: Collection[str]) -> bool:
    """True while the text from a line start can still be a written call (undecided counts as yes)."""
    body = s.lstrip(" \t")
    if not body:
        return True
    if body[0] == "\n":
        return False  # a blank line; the next line is looked at on its own
    if body.startswith("```"):
        nl = body.find("\n")
        if nl == -1:
            return True  # the fence line is still coming
        return _inner_could_be_call(body[nl + 1 :], names)
    if body[0] == "[" and len(body) > 1 and body[1].isalpha():
        nl = body.find("\n")  # a marker line like "[inside content CODE": it depends on the next line
        if nl == -1:
            return True
        return _inner_could_be_call(body[nl + 1 :], names)
    if any(t.startswith(body[: len(t)]) for t in TAGS):
        return True
    return _inner_could_be_call(body, names)


def _inner_could_be_call(s: str, names: Collection[str]) -> bool:
    inner = s.lstrip()
    if not inner:
        return True
    if inner[0] == "[":
        inner = inner[1:].lstrip()
        if not inner:
            return True
    if inner[0] == "{":
        for m in _NAME_FIELD.finditer(inner):
            key, value, closed = m.group(1), _strip_prefix(m.group(2)), m.group(3)
            if not closed:
                return any(n.startswith(value) for n in names) or (key in ("type", "function") and "function".startswith(value))
            if value in names:
                return True
            if value == "function" and key in ("type", "function"):
                continue  # the OpenAI shape: the name follows inside "function"
            return False
        return _balanced_end(inner, 0) == -1 and len(inner) <= MAX_UNNAMED
    if TAGS[0].startswith(inner[: len(TAGS[0])]):
        return True
    m = _IDENT.match(inner)
    if not m:
        return False
    ident = m.group(1)
    if m.end() == len(inner):  # the word is still coming
        whole = inner[: m.end()]
        return any(n.startswith(ident) for n in names) or "functions.".startswith(whole)
    rest = inner[m.end() :].lstrip(" ")
    return ident in names and (not rest or rest[0] == "(")


def _balanced_end(s: str, i: int) -> int:
    """Index just past the bracket that closes the one at ``s[i]``, minding JSON and Python strings; -1 if open."""
    pairs = {"{": "}", "[": "]", "(": ")"}
    stack: list[str] = []
    quote = ""
    j = i
    while j < len(s):
        c = s[j]
        if quote:
            if c == "\\":
                j += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return j + 1
        j += 1
    return -1


# ---------------------------------------------------------------------- a finished reply
def find(text: str, schema_for: SchemaFn) -> Found | None:
    """The written-out calls in a finished reply, or None when it has none (or a tool name is only mentioned)."""
    if not text or not text.strip():
        return None
    tagged = bool(TAG_RE.search(text))
    spans: list[tuple[int, int, list[tuple[str, Any]]]] = []  # (start, end, [(tool, arguments or None)])
    for m in TAG_RE.finditer(text):
        calls = _calls_in(m.group(1), schema_for, loose=True)
        spans.append((m.start(), m.end(), calls or [(_named(m.group(1)), None)]))
    for start, end, calls in _json_spans(text, schema_for) + _paren_spans(text, schema_for):
        if not any(start < e and s < end for s, e, _ in spans):
            spans.append((start, end, calls))
    if not spans:
        broken = _broken(text, schema_for)
        if broken is None:
            return None
        spans.append(broken)
    spans.sort(key=lambda sp: sp[0])
    names: list[str] = []
    calls: list[tuple[str, dict]] = []
    fits = True
    for _s, _e, written in spans:
        for name, args in written:
            names.append(name or "tool")
            schema = schema_for(name) if name else None
            if schema is None or not isinstance(args, dict) or not _fits(args, schema):
                fits = False
            else:
                calls.append((name, args))
    pieces, last = [], 0
    for s, e, _ in spans:
        pieces.append(text[last:s])
        last = e
    pieces.append(text[last:])
    before = _TRAILING_WRAPPERS.sub("", pieces[0]).rstrip()
    after = "\n".join(_prose(p) for p in pieces[1:]).strip()
    around = len(_prose(pieces[0])) + len(after)
    is_action = tagged or around <= MAX_AROUND
    cleaned = "\n\n".join(p for p in (before, after) if p)
    return Found(calls=calls if fits else [], names=names, before=before, cleaned=cleaned,
                 runnable=is_action and fits and bool(calls), is_action=is_action)


def _prose(piece: str) -> str:
    """A piece of the reply without wrapper lines (fences, tags, ``[...]`` markers)."""
    return "\n".join(line for line in piece.splitlines() if not _WRAPPER_LINE.match(line)).strip()


def _named(body: str) -> str:
    """The tool a broken written call names, for the nudge; "" when it names none."""
    m = _NAME_FIELD.search(body)
    name = _strip_prefix(m.group(2)) if m else ""
    return "" if name == "function" else name


def _json_spans(text: str, schema_for: SchemaFn) -> list[tuple[int, int, list[tuple[str, Any]]]]:
    """Every complete JSON object or array in ``text`` that is a written call of a known tool."""
    decoder = json.JSONDecoder()
    out: list[tuple[int, int, list[tuple[str, Any]]]] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                end = -1
            if end != -1:
                calls = _as_calls(obj, schema_for)
                if calls:
                    out.append((i, end, calls))
                i = end
                continue
            closing = _balanced_end(text, i)  # loosely written JSON (single quotes, trailing commas)
            if closing != -1:
                calls = _calls_in(text[i:closing], schema_for, loose=True)
                if calls:
                    out.append((i, closing, calls))
                    i = closing
                    continue
        i += 1
    return out


def _paren_spans(text: str, schema_for: SchemaFn) -> list[tuple[int, int, list[tuple[str, Any]]]]:
    """``memory_remember({...})`` or ``memory_remember(fact="...")`` for a known tool."""
    out: list[tuple[int, int, list[tuple[str, Any]]]] = []
    for m in _CALL_START.finditer(text):
        name = m.group(1)
        schema = schema_for(name)
        if schema is None:
            continue
        close = _balanced_end(text, m.end() - 1)
        if close == -1:
            continue
        args = _paren_arguments(text[m.end() : close - 1], schema)
        if args is not None:  # "memory_remember (the memory tool)" in a sentence is a mention, not a call
            out.append((m.start(), close, [(name, args)]))
    return out


def _broken(text: str, schema_for: SchemaFn) -> tuple[int, int, list[tuple[str, Any]]] | None:
    """A written call of a known tool that doesn't parse (cut off, broken JSON): its span, to nudge about."""
    for m in _NAME_FIELD.finditer(text):
        name = _strip_prefix(m.group(2))
        if not m.group(3) or schema_for(name) is None:
            continue
        brace = text.rfind("{", 0, m.start())
        if brace == -1:
            continue
        end = _balanced_end(text, brace)
        return brace, (len(text) if end == -1 else end), [(name, None)]
    return None


def _paren_arguments(inner: str, schema: dict) -> dict | None:
    inner = inner.strip()
    if not inner:
        return {}
    if inner.startswith("{"):
        try:
            args = parse_json_loose(inner)
        except ValueError:
            return None
        return args if isinstance(args, dict) else None
    try:
        node = ast.parse(f"f({inner})", mode="eval").body
        assert isinstance(node, ast.Call)
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        if node.args:  # one plain value for a tool with one argument: memory_remember("Maya ...")
            props = list((schema.get("properties") or {}).keys())
            if len(node.args) != 1 or len(props) != 1 or len(node.keywords):
                return None
            args = {props[0]: ast.literal_eval(node.args[0])}
        return args
    except (SyntaxError, ValueError, AssertionError, TypeError):
        return None


def _calls_in(body: str, schema_for: SchemaFn, *, loose: bool = False) -> list[tuple[str, Any]]:
    """The calls written in one block (a tag's or a loosely written object's text)."""
    body = re.sub(r"^\s*```[\w-]*\s*|\s*```\s*$", "", body.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        if not loose:
            return []
        try:
            obj = parse_json_loose(body)
        except ValueError:
            spans = _paren_spans(body, schema_for)
            return spans[0][2] if spans else []
    return _as_calls(obj, schema_for)


def _as_calls(obj: Any, schema_for: SchemaFn) -> list[tuple[str, Any]]:
    """``obj`` as written calls of known tools; [] when it isn't one (or names a tool that doesn't exist)."""
    if isinstance(obj, list):
        calls = [_as_call(o, schema_for) for o in obj]
        return calls if calls and all(c is not None for c in calls) else []  # type: ignore[misc]
    call = _as_call(obj, schema_for)
    return [call] if call else []


def _as_call(obj: Any, schema_for: SchemaFn) -> tuple[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # {"type": "function", "function": {"name": ..., "arguments": ...}}
        return _as_call(obj["function"], schema_for)
    name_key = next(
        (k for k in NAME_KEYS if isinstance(obj.get(k), str) and schema_for(_strip_prefix(obj[k])) is not None), None
    )
    if name_key is None:
        return None
    name = _strip_prefix(obj[name_key])
    arg_key = next((k for k in ARG_KEYS if k in obj), None)
    if arg_key is None:  # {"name": "memory_remember", "fact": "..."}: the other keys are the arguments
        args: Any = {k: v for k, v in obj.items() if k not in (name_key, "id", "type")}
    else:
        args = obj[arg_key]
        if isinstance(args, str):
            args = tool_arguments(args)
            if "_raw" in args:
                args = None
    return name, args


def _fits(args: dict, schema: dict) -> bool:
    """The arguments name only the tool's own parameters and every required one."""
    props = schema.get("properties")
    closed = isinstance(props, dict) and props and not schema.get("additionalProperties")
    if closed and any(k not in props for k in args):
        return False
    return all(k in args for k in schema.get("required") or ())
