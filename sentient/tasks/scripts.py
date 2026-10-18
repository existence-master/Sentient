"""Script jobs: watch-style tasks that run a short Python check without model calls (docs/API.md section 16).

A script job is a task with ``task_type: "script"`` and a ``script`` object::

    {code, condition: "changed"|"noteworthy"|"alert"|"every_run", then: "notify"|"run", last_result, last_run_at,
     last_error, checks}

The code runs through ``app.sandbox.run`` and may call read-only Sentient tools with
``from sentient_tools import tools, result``. This module holds the pure helpers:
validation, normalization, reading the sandbox outcome, deciding whether to act and the short history of checks.

A ``noteworthy`` job is a quiet monitor (#157): like ``changed`` it compares each result with the last one with no
model, so an unchanged source costs nothing; only when something changed does the ``fast`` model judge whether the
change matters (``JUDGE_PROMPT`` on a line diff), and only then does the job act.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import sys
from typing import Any

CONDITIONS = ("alert", "changed", "noteworthy", "every_run")
COMPARING = ("changed", "noteworthy")  # act on a difference from the last result
THEN = ("notify", "run")
MAX_CODE_CHARS = 20000
MAX_RESULT_CHARS = 20000
REPORT_CHARS = 3000  # an every-run report shows this much of the script's output
SCRIPT_STATE_FIELDS = ("last_result", "last_run_at", "last_error", "checks")
MAX_CHECKS = 30  # the history keeps this many entries; a row of quiet checks is one entry
QUIET_OUTCOMES = ("nothing_new", "quiet")  # repeats of these fold into one entry with a count
DIFF_CHARS = 3000  # the judge sees at most this much of the change
JUDGE_PROMPT = (
    "You check a change in something the user asked Sentient to watch. First say what changed, then decide whether "
    "the user would want a notification about it.\n"
    "Worth telling (true): an item was added or removed, or a price, date, status or result the user asked about "
    "changed.\n"
    "Not worth telling (false): only a counter changed (page views, visitors, likes, comments), a clock or "
    "\"last updated\" time, an ad, the order of items, spacing, or the same words said differently.\n"
    'Reply with JSON only: {"change": "one short sentence naming what changed", "noteworthy": true or false}'
)


class ScriptInvalid(ValueError):
    """The script code or settings cannot be used."""


def validate_code(code: Any) -> str:
    """Return the code when it is non-empty Python that compiles; raise ScriptInvalid otherwise."""
    if not isinstance(code, str) or not code.strip():
        raise ScriptInvalid("The check script is empty.")
    if len(code) > MAX_CODE_CHARS:
        raise ScriptInvalid(f"The check script is too long (over {MAX_CODE_CHARS} characters).")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        raise ScriptInvalid(f"The check script is not valid Python{where}: {exc.msg}") from exc
    return code


_SLEEPS = {"sleep", "time_sleep"}
_TOOL_CALL = re.compile(r"\btools\.([A-Za-z_]\w*)")


def tools_named(code: Any) -> list[str]:
    """The tools a check script uses as ``tools.<name>``, in order and once each. Works on code that does not
    compile, since it is read from the text."""
    return list(dict.fromkeys(_TOOL_CALL.findall(str(code or ""))))


_IMPORTABLE = set(sys.stdlib_module_names) | {"sentient_tools"}

# What the read-only tools check scripts call most return (#366), so a script reading a key that is never there
# (``web_fetch(...)['html']``) is rejected instead of returning "" on every check. The planner's script rules say the
# same; tests/integrations/test_keyless.py checks these against the real tools. A tool that fails raises in the
# script, so a result never holds "error".
ARTICLE_KEYS = ("title", "source", "url", "published_at", "summary")
RESULT_KEYS: dict[str, tuple[str, ...]] = {
    "web_fetch": ("url", "title", "content", "truncated"),
    "web_search": ("query", "provider", "results", "note"),
    "news_top_headlines": ("provider", "country", "category", "articles"),
    "news_search": ("provider", "query", "articles"),
}
ITEM_KEYS: dict[tuple[str, str], tuple[str, ...]] = {  # (tool, list key): the keys of each item in that list
    ("web_search", "results"): ("title", "url", "snippet"),
    ("news_top_headlines", "articles"): ARTICLE_KEYS,
    ("news_search", "articles"): ARTICLE_KEYS,
}
_HARMLESS_KEYS = {"error"}  # reading these finds nothing but breaks nothing


class _Shape:
    """A value whose keys are known: a tool's result (``keys``), or a list in it whose items have ``item`` keys."""

    def __init__(self, what: str, keys: tuple[str, ...] = (), item: _Shape | None = None, tool: str = "") -> None:
        self.what, self.keys, self.item, self.tool = what, keys, item, tool


def _result_shape(tool: str) -> _Shape:
    return _Shape(f"the result of `tools.{tool}`", RESULT_KEYS[tool], tool=tool)


def _key_read(node: ast.AST) -> tuple[ast.expr, str] | None:
    """``(value, key)`` for ``value['key']`` and ``value.get('key', ...)``."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.value, node.slice.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        return node.func.value, node.args[0].value
    return None


def _shape_of(node: ast.AST, env: dict[str, _Shape]) -> _Shape | None:
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):  # page.get('results') or []
        return _shape_of(node.values[0], env)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tools" and node.func.attr in RESULT_KEYS):
        return _result_shape(node.func.attr)
    if isinstance(node, ast.Subscript) and not _key_read(node):  # results[0], results[:5]
        base = _shape_of(node.value, env)
        if base is not None and base.item is not None:
            return base if isinstance(node.slice, ast.Slice) else base.item
        return None
    read = _key_read(node)
    if read:
        base = _shape_of(read[0], env)
        items = ITEM_KEYS.get((base.tool, read[1])) if base is not None and base.item is None else None
        if items:
            item = _Shape(f"each item of `{read[1]}` from `tools.{base.tool}`", items, tool=base.tool)
            return _Shape(f"`{read[1]}` from `tools.{base.tool}`", item=item, tool=base.tool)
    return None


def _bindings(tree: ast.AST) -> dict[str, list[tuple[str, ast.expr | None]]]:
    """Every way each name gets a value: ``("value", expr)``, ``("item", iterable)`` for a loop variable, or
    ``("other", None)`` for anything this check does not follow (unpacking, arguments, ``+=``...)."""
    out: dict[str, list[tuple[str, ast.expr | None]]] = {}
    seen: set[int] = set()

    def bind(target: ast.AST, how: str, expr: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            out.setdefault(target.id, []).append((how, expr))
            seen.add(id(target))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, "value", node.value)
        elif isinstance(node, ast.NamedExpr) or (isinstance(node, ast.AnnAssign) and node.value is not None):
            bind(node.target, "value", node.value)
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            bind(node.target, "item", node.iter)
        elif isinstance(node, ast.arg):
            out.setdefault(node.arg, []).append(("other", None))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in seen:
            out.setdefault(node.id, []).append(("other", None))
        # a result the script changes itself (page["price"] = ..., page.update(...)) may hold keys of its own
        changed = node if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load) else None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"update", "setdefault", "__setitem__"}):
            changed = node.func.value
        while isinstance(changed, ast.Subscript | ast.Attribute):
            changed = changed.value
        if isinstance(changed, ast.Name):
            out.setdefault(changed.id, []).append(("other", None))
    return out


def _known_shapes(tree: ast.AST) -> dict[str, _Shape]:
    """Names that always hold a tool result (or a list or item in one). A name that also gets anything else is left
    out, so the check never guesses."""
    bindings = _bindings(tree)
    env: dict[str, _Shape] = {}
    for _ in range(len(bindings) + 1):  # until nothing changes: page = tools.web_search(...); rows = page['results']...
        found: dict[str, _Shape] = {}
        for name, ways in bindings.items():
            shapes = []
            for how, expr in ways:
                shape = _shape_of(expr, env) if expr is not None else None
                if how == "item":
                    shape = shape.item if shape is not None else None
                shapes.append(shape)
            first = shapes[0]
            if first is not None and all(s is not None and s.what == first.what for s in shapes):
                found[name] = first
        if found.keys() == env.keys():
            break
        env = found
    return env


def _and(words: tuple[str, ...]) -> str:
    quoted = [f"`{w}`" for w in words]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _check_result_keys(tree: ast.AST) -> None:
    """Reject a read of a key the tool never returns (#366), naming the keys it does return."""
    env = _known_shapes(tree)
    for node in ast.walk(tree):
        read = _key_read(node)
        if not read or (isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load)):
            continue
        shape = _shape_of(read[0], env)
        key = read[1]
        if shape is None or key in _HARMLESS_KEYS:
            continue
        if shape.item is not None:
            raise ScriptInvalid(
                f"The check script reads `{key}` from {shape.what}, which is a list. Loop over it and read "
                f"{_and(shape.item.keys)} from each item."
            )
        if key not in shape.keys:
            hint = " The page's text is in `content`." if shape.tool == "web_fetch" else ""
            raise ScriptInvalid(
                f"The check script reads `{key}` from {shape.what}, which has no `{key}`: it has only "
                f"{_and(shape.keys)}.{hint}"
            )


# A request to watch something for news (#375): "watch", "track", "monitor", "keep an eye on", "tell me when"...
_WATCH = re.compile(
    r"\b(?:watch|track|monitor|keep\s+an\s+eye|keep\s+tabs)\w*\b"
    r"|\b(?:tell|let|notify|alert|ping|message|text)\s+(?:me|us)\s+(?:know\s+)?(?:when|if|whenever|as\s+soon\s+as)\b",
    re.IGNORECASE,
)
# ...that still asks for a report on every check: "a report every hour", "each time you check", "daily summary".
_EVERY_REPORT = re.compile(
    r"\b(?:report|summary|digest|update|recap)s?\s+(?:every|each)\b"
    r"|\b(?:every|each)\s+(?:run|check)\b|\b(?:every|each)\s+time\s+(?:you|it)\s+(?:check|run|look)s?\b"
    r"|\b(?:hourly|daily|nightly|weekly|monthly)\s+(?:report|summary|digest|update|recap)\b"
    r"|\beven\s+(?:if|when)\s+nothing\b",
    re.IGNORECASE,
)
# The request names the part of the page to watch, so keeping only it is right.
_NAMES_PART = re.compile(
    r"\b(?:first|top|opening|1st)\s+(?:line|item|entry|row|post|story|event|result|article|paragraph)s?\b"
    r"|\b(?:heading|headline|title|header|banner)s?\b",
    re.IGNORECASE,
)
_LINE_SPLITS = {"split", "rsplit", "splitlines", "partition", "rpartition"}
_SAME_TEXT = {"strip", "lstrip", "rstrip", "lower", "upper", "casefold"}
SHORT_CUT_CHARS = 500  # content[:n] with a smaller n keeps only the top of the page


def is_watch_request(request: str) -> bool:
    """The user asks to watch something for news, not for a report on every run (#375). The request holds what they
    said later too (a change in the task's chat), so the last of "watch" and "report every..." wins."""
    text = request or ""
    watch = [m.start() for m in _WATCH.finditer(text)]
    report = [m.start() for m in _EVERY_REPORT.finditer(text)]
    return bool(watch) and (not report or watch[-1] > report[-1])


def _page_text_names(tree: ast.AST, env: dict[str, _Shape]) -> tuple[set[str], set[str]]:
    """``(texts, lines)``: names that always hold the fetched page's text, and names that always hold it split into
    pieces (``content.splitlines()``)."""
    bindings = _bindings(tree)
    texts: set[str] = set()
    lines: set[str] = set()
    for _ in range(len(bindings) + 1):
        found_texts = {n for n, ways in bindings.items()
                       if all(how == "value" and expr is not None and _is_page_text(expr, env, texts)
                              for how, expr in ways)}
        found_lines = {n for n, ways in bindings.items()
                       if all(how == "value" and expr is not None and _splits_page(expr, env, found_texts)
                              for how, expr in ways)}
        if found_texts == texts and found_lines == lines:
            break
        texts, lines = found_texts, found_lines
    return texts, lines


def _is_page_text(node: ast.AST, env: dict[str, _Shape], texts: set[str]) -> bool:
    """The whole text of a ``tools.web_fetch`` page: ``page['content']``, ``page.get('content', '')``, a name holding
    it, ``... or ''``, ``str(...)`` or ``.strip()``/``.lower()`` of it."""
    if isinstance(node, ast.Name):
        return node.id in texts
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _is_page_text(node.values[0], env, texts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
        return _is_page_text(node.args[0], env, texts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _SAME_TEXT:
        return _is_page_text(node.func.value, env, texts)
    read = _key_read(node)
    if read and read[1] == "content":
        shape = _shape_of(read[0], env)
        return shape is not None and shape.tool == "web_fetch" and shape.item is None
    return False


def _splits_page(node: ast.AST, env: dict[str, _Shape], texts: set[str]) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LINE_SPLITS
            and _is_page_text(node.func.value, env, texts))


def _first_piece(node: ast.Subscript) -> bool:
    """``x[0]`` or ``x[:1]``."""
    s = node.slice
    if isinstance(s, ast.Constant):
        return s.value == 0
    return (isinstance(s, ast.Slice) and s.lower is None and s.step is None and isinstance(s.upper, ast.Constant)
            and s.upper.value == 1)


def _returned_page_keys(tree: ast.AST, env: dict[str, _Shape]) -> set[str] | None:
    """The keys of a ``tools.web_fetch`` result that reach what the script returns (``result(...)`` or ``print(...)``),
    following names through their assignments; None when the script returns nothing."""
    bindings = _bindings(tree)
    keys: set[str] = set()

    def collect(node: ast.AST, seen: frozenset[str]) -> None:
        if isinstance(node, ast.Name):
            if node.id not in seen:
                for how, expr in bindings.get(node.id, ()):
                    if how == "value" and expr is not None:
                        collect(expr, seen | {node.id})
            return
        read = _key_read(node)
        if read:
            shape = _shape_of(read[0], env)
            if shape is not None and shape.tool == "web_fetch" and shape.item is None:
                keys.add(read[1])
        for child in ast.iter_child_nodes(node):
            collect(child, seen)

    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id in ("result", "print")]
    for call in returns:
        for arg in [*call.args, *(k.value for k in call.keywords)]:
            collect(arg, frozenset())
    return keys if returns else None


def _check_whole_page(tree: ast.AST) -> None:
    """Reject a script that cuts a fetched page down to its first line, its heading or its first few characters
    (#375): a new item further down the page would never be noticed."""
    env = _known_shapes(tree)
    texts, lines = _page_text_names(tree, env)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
            continue
        base = node.value
        split = _splits_page(base, env, texts) or (isinstance(base, ast.Name) and base.id in lines)
        if split and _first_piece(node):
            raise ScriptInvalid(
                f"The check script keeps only the first line of the page (`{ast.unparse(node)}`), which is usually "
                "its heading, so something new further down the page would never be noticed. Return the page's whole "
                "`content` instead."
            )
        s = node.slice
        if (_is_page_text(base, env, texts) and isinstance(s, ast.Slice) and s.lower is None
                and isinstance(s.upper, ast.Constant) and isinstance(s.upper.value, int)
                and 0 <= s.upper.value < SHORT_CUT_CHARS):
            raise ScriptInvalid(
                f"The check script keeps only the first {s.upper.value} characters of the page "
                f"(`{ast.unparse(node)}`), so something new further down the page would never be noticed. Return the "
                "page's whole `content` instead."
            )
    returned = _returned_page_keys(tree, env)
    if returned is not None and "title" in returned and "content" not in returned:
        raise ScriptInvalid(
            "The check script reads only the page's `title`, so it would never notice anything new on the page "
            "itself. Return the page's whole `content` instead."
        )


# Tools that read the web: a script using them polls a page, a search or the news (#389).
WEB_TOOLS = ("web_fetch", "web_search", "news_top_headlines", "news_search")
# Words that name a trigger's app in a request: "when I get an email", "when a meeting is added to my calendar".
_SOURCE_WORDS: dict[str, re.Pattern] = {
    "gmail": re.compile(r"\b(?:e-?mails?|mail|inbox|gmail)\b", re.IGNORECASE),
    "email_imap": re.compile(r"\b(?:e-?mails?|mail|inbox)\b", re.IGNORECASE),
    "gcalendar": re.compile(r"\b(?:calendars?|meetings?|appointments?)\b", re.IGNORECASE),
    "home_assistant": re.compile(
        r"\b(?:home\s+assistant|smart\s+home|doors?|lights?|locks?|sensors?|thermostats?|garage|alarms?|switch(?:es)?)\b",
        re.IGNORECASE),
    "webhook": re.compile(r"\b(?:webhooks?|hooks?)\b", re.IGNORECASE),
}


def _reads_trigger_event(code: str) -> bool:
    """The code reads the ``trigger_event`` variable; a comment or a string that only mentions it does not count."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and node.id == "trigger_event" and isinstance(node.ctx, ast.Load)
               for node in ast.walk(tree))


def polls_the_web(tools: list[str]) -> bool:
    """A script (or the AI-run task that replaced one) that reads pages, a search or the news."""
    return any(t in WEB_TOOLS for t in tools)


def trigger_is_real(schedule: dict, code: str, request: str) -> bool:
    """Whether a ``triggered`` schedule can be what a web check runs on (#389): the script reads the event that
    started it (``trigger_event``), or the user's request names the trigger's app ("when I get an email"). Small
    models give a page watch like "tell me when a new event is added" a Google Calendar trigger, so it would only
    run when a calendar event arrives and never look at the page."""
    if _reads_trigger_event(code):
        return True
    words = _SOURCE_WORDS.get(str(schedule.get("source") or "").strip().lower())
    return bool(words and words.search(request or ""))


def check_rules(code: str, condition: str, request: str = "") -> None:
    """Reject what small models write into check scripts that cannot work (#305), with a plain reason the planner
    can act on: a package that is not installed, an endless loop, a wait, memory calls, an alert result on a
    comparing condition, or a key a tool's result never has (#366).

    With the user's ``request`` (the planner, #375) it also rejects ``every_run`` for a watch request, which would
    notify on every check, and a script that cuts a page down to its heading or first line when the request does
    not name that part."""
    if request and condition == "every_run" and is_watch_request(request):
        raise ScriptInvalid(
            "The user asked to watch for something new, but the condition \"every_run\" would send a notification "
            "after every check, even when nothing changed. Use \"noteworthy\" to watch a page, a feed or the news, "
            "\"changed\" for one exact value, or \"alert\" with a test."
        )
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in names:
                top = module.split(".")[0]
                if top and not getattr(node, "level", 0) and top not in _IMPORTABLE:
                    raise ScriptInvalid(
                        f"The check script imports `{top}`, which is not installed. Use only the Python standard "
                        "library and `tools.<function>` (read a page with `tools.web_fetch(url=...)`)."
                    )
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
            raise ScriptInvalid(
                "The check script loops forever (`while True`). It must run once and finish; Sentient runs it "
                "again on the schedule."
            )
        name = node.attr if isinstance(node, ast.Attribute) else ""  # tools.memory_recall, even uncalled
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if called in _SLEEPS:
                raise ScriptInvalid(
                    f"The check script waits (`{ast.unparse(func)}`). It must not sleep; Sentient runs it again "
                    "on the schedule."
                )
            name = name or called
        if name.startswith("memory_"):
            raise ScriptInvalid(
                f"The check script calls `{name}`. Check scripts must not use memory; for \"changed\" and "
                "\"noteworthy\" Sentient remembers the last result and compares it for you."
            )
        if condition in COMPARING and isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "alert" for k in node.keys
        ):
            raise ScriptInvalid(
                f"The check script returns an alert (`{{\"alert\": ...}}`) but its condition is \"{condition}\". "
                f"Use the condition \"alert\", or return only the watched value and let Sentient compare it."
            )
    _check_result_keys(tree)
    if request and not _NAMES_PART.search(request):
        _check_whole_page(tree)


def normalize_script(
    raw: Any, previous: dict | None = None, *, model_written: bool = False, request: str = ""
) -> dict:
    """Merge user/planner input over ``previous``. Changing the code resets the stored state. Code a model wrote
    (the planner, chat tools) must also pass ``check_rules``, with the user's ``request`` for the planner; a person
    may write what they like."""
    if not isinstance(raw, dict):
        raise ScriptInvalid("script must be an object with code, condition and then.")
    prev = dict(previous or {})
    code = raw.get("code", prev.get("code"))
    code = validate_code(code)
    condition = str(raw.get("condition") or prev.get("condition") or "alert").strip().lower()
    if condition not in CONDITIONS:
        raise ScriptInvalid("script.condition must be 'alert', 'changed', 'noteworthy' or 'every_run'.")
    then = str(raw.get("then") or prev.get("then") or "notify").strip().lower()
    if then not in THEN:
        raise ScriptInvalid("script.then must be 'notify' or 'run'.")
    if model_written:
        check_rules(code, condition, request)
    out = {"code": code, "condition": condition, "then": then}
    same_code = previous is not None and prev.get("code") == code
    # switching between changed and noteworthy keeps the baseline: both compare the same result
    keep = same_code and (prev.get("condition") == condition or {prev.get("condition"), condition} <= set(COMPARING))
    for key in SCRIPT_STATE_FIELDS:
        out[key] = prev.get(key) if keep else None
    if same_code:
        out["checks"] = prev.get("checks")  # the history of checks stays while only the condition changes
    return out


# The newest request about updates wins. A question about a page is not a request to change its watch setting;
# neither is a bare use of "change" as a verb. Requests for every update must say that positively.
_NOTEWORTHY_REQUEST = re.compile(
    r"\b(?:only\s+)?(?:important|big|major|significant|meaningful)\s+(?:\w+\s+)?"
    r"(?:changes?|updates?|stuff|ones?)\b"
    r"|\b(?:ignore|skip|don't|do not)\s+(?:\w+\s+)?(?:minor|small|little)\s+(?:changes?|updates?)\b",
    re.IGNORECASE,
)
_EVERY_CHANGE_REQUEST = re.compile(
    r"\b(?:any|every|all)\s+(?:\w+\s+){0,3}?(?:changes?|updates?)\b"
    r"|\b(?:when|whenever|if)\b[^\n]{0,160}?\b(?:changes?|updates?)\b"
    r"|\b(?:tell me|notify me|let me know|alert me|report)\b[^\n]{0,160}?\b(?:changes?|updates?)\b"
    r"|\b(?:tell me|notify me|let me know|alert me|report)\b[^\n]{0,160}?\b"
    r"(?:anything|everything)\s+(?:at\s+all|new)\b",
    re.IGNORECASE,
)
_NEGATED_REQUEST_PREFIX = re.compile(
    r"\b(?:not|never)\s*$|\b(?:don't|do not)\b[^,.!?;]{0,60}$", re.IGNORECASE,
)


def _change_preference(message: str) -> str | None:
    """``changed`` or ``noteworthy`` when one user message says which updates to report."""
    preferences = [(match.start(), "noteworthy") for match in _NOTEWORTHY_REQUEST.finditer(message)]
    preferences += [
        (match.start(), "changed") for match in _EVERY_CHANGE_REQUEST.finditer(message)
        if not _NEGATED_REQUEST_PREFIX.search(message[:match.start()])
    ]
    if message.rstrip().endswith("?") and not preferences:
        return None
    return max(preferences, default=(0, None), key=lambda preference: preference[0])[1]


def _latest_change_preference(request: str | list[str]) -> str | None:
    messages = request.split("\n") if isinstance(request, str) else request
    for message in reversed(messages):
        preference = _change_preference(str(message or ""))
        if preference is not None:
            return preference
    return None


def _is_fetch(node: ast.AST, pages: set[str]) -> bool:
    """``tools.web_fetch(...)``, a name holding it, or its ``content`` (``page["content"]``, ``page.get("content")``)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "web_fetch":
        return isinstance(node.func.value, ast.Name) and node.func.value.id == "tools"
    if isinstance(node, ast.Name):
        return node.id in pages
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("strip", "lstrip", "rstrip") and not node.args):
        return _is_fetch(node.func.value, pages)  # page["content"].strip() is still the whole page
    if isinstance(node, ast.Subscript):
        key, base = node.slice, node.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
        key, base = node.args[0], node.func.value
    else:
        return False
    return isinstance(key, ast.Constant) and key.value == "content" and _is_fetch(base, pages)


def returns_whole_page(code: str) -> bool:
    """True when the script hands back the whole result of ``tools.web_fetch`` (the page or its content) rather than
    a value taken out of it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    pages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_fetch(node.value, pages):
            pages.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("result", "print")
        and len(node.args) == 1 and _is_fetch(node.args[0], pages)
        for node in ast.walk(tree)
    )


def planner_default(script: dict, request: str | list[str], previous: dict | None = None) -> dict:
    """A ``changed`` check the planner proposes that returns a whole fetched page becomes ``noteworthy`` (#298): a
    view counter or a timestamp there would notify on every check. One value taken out of the page, a search, a
    request whose newest update preference is every change keeps ``changed``. A newer request for important changes
    switches an existing ``changed`` task back to ``noteworthy``."""
    if script.get("condition") != "changed":
        return script
    if not returns_whole_page(script.get("code") or ""):
        return script
    preference = _latest_change_preference(request)
    if preference == "changed" or (preference is None and (previous or {}).get("condition") == "changed"):
        return script
    return {**script, "condition": "noteworthy"}


def trigger_prelude(trigger_data: dict | None) -> str:
    """Lines prepended to the code of a triggered script so it can read the event as ``trigger_event``."""
    payload = json.dumps(trigger_data if isinstance(trigger_data, dict) else None, ensure_ascii=False, default=str)
    return f"import json as _sentient_json\ntrigger_event = _sentient_json.loads({payload!r})\n"


def outcome_value(outcome: dict) -> Any:
    """The value a script produced: ``result(...)`` when set, else its trimmed stdout."""
    value = outcome.get("result")
    if value is None:
        stdout = outcome.get("stdout")
        value = stdout.strip() if isinstance(stdout, str) and stdout.strip() else None
    return cap(value)


def cap(value: Any) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)[:MAX_RESULT_CHARS]
    if len(text) <= MAX_RESULT_CHARS:
        return json.loads(text)  # plain JSON types only
    return text[:MAX_RESULT_CHARS] + " ... [truncated]"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def outcome_error(outcome: dict) -> str | None:
    """A short, user-readable failure message, or None when the script succeeded."""
    if outcome.get("ok"):
        return None
    detail = str(outcome.get("error") or "").strip()
    if not detail:
        stderr = str(outcome.get("stderr") or "").strip()
        detail = stderr.splitlines()[-1] if stderr else "it stopped without saying why"
    return f"The check script failed: {detail[:500]}"


def should_act(condition: str, value: Any, previous: Any) -> bool:
    """``alert``: the script returned ``{"alert": true, ...}``. ``changed``: the value differs from the
    previous successful result. The first successful result of a ``changed`` job is only the baseline, and an
    empty (``None``) result is never a change (a page that briefly returns nothing should not alert).
    ``every_run``: every successful check that produced something, even when it is the same as last time; a
    check that prints nothing stays quiet. ``noteworthy`` is ``changed`` here; the judge decides afterwards."""
    if condition == "every_run":
        return value is not None and not (isinstance(value, str) and not value.strip())
    if condition in COMPARING:
        return value is not None and previous is not None and canonical(value) != canonical(previous)
    return isinstance(value, dict) and bool(value.get("alert"))


def alert_message(task_name: str, condition: str, value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("message"), str) and value["message"].strip():
        return value["message"].strip()
    if condition == "every_run" and isinstance(value, str):
        return value if len(value) <= REPORT_CHARS else value[: REPORT_CHARS - 3] + "..."  # the output is the report
    shown = value if isinstance(value, str) else canonical(value)
    shown = shown if len(shown) <= 500 else shown[:497] + "..."
    if condition in COMPARING:
        return f"'{task_name}' noticed a change: {shown}"
    if condition == "every_run":
        return f"'{task_name}' reported: {shown}"
    return f"'{task_name}' raised an alert: {shown}"


def describe_for_approval(script: dict) -> str:
    """Plain-language summary shown next to the code in a plan card."""
    action = "send you a notification" if script.get("then") == "notify" else "start a full task run"
    if script.get("condition") == "every_run":
        return f"Runs a small check script with no AI calls; after every check that prints something, it will {action}."
    if script.get("condition") == "noteworthy":
        return (
            "Runs a small check script with no AI calls. When the result changes, your fast AI model decides whether "
            f"the change matters; if it does, it will {action}."
        )
    when = "the script raises an alert" if script.get("condition") == "alert" else "the result changes"
    return f"Runs a small check script with no AI calls; when {when}, it will {action}."


# ---------------------------------------------------------------------------- noteworthy: the judge
def _lines(value: Any) -> list[str]:
    """One line per line of text, also inside dicts and lists, so a page held in a dict (a whole ``web_fetch``
    result) diffs line by line and the judge sees only what changed (#298)."""
    if isinstance(value, dict):
        return [f"{k}: {line}" for k in sorted(value, key=str) for line in _lines(value[k])]
    if isinstance(value, list):
        return [line for item in value for line in _lines(item)]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def diff_text(previous: Any, value: Any) -> str:
    """The changed lines only (``- removed``, ``+ added``), at most ``DIFF_CHARS``; "" when nothing differs."""
    # skip the two file headers by position; hunk headers start with "@@", changed lines with "-" or "+"
    diff = list(difflib.unified_diff(_lines(previous), _lines(value), lineterm="", n=0))[2:]
    lines = [line for line in diff if line.startswith(("-", "+"))]
    text = "\n".join(f"{line[0]} {line[1:].strip()}" for line in lines)
    return text if len(text) <= DIFF_CHARS else text[: DIFF_CHARS - 4] + "\n..."


def judge_messages(goal: str, previous: Any, value: Any) -> list[dict]:
    """The judge's short prompt: what the user watches and the line diff."""
    user = (
        f"What the user watches: {goal[:500]}\n"
        f"Changes since the last check (- removed, + added):\n{diff_text(previous, value) or '(only spacing)'}"
    )
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]


def read_verdict(data: Any) -> tuple[bool | None, str]:
    """``(noteworthy, change)`` from the judge's JSON; ``None`` when it gave no usable answer."""
    if not isinstance(data, dict):
        return None, ""
    raw = data.get("noteworthy", data.get("notify", data.get("material")))
    if isinstance(raw, str):
        raw = {"true": True, "yes": True, "false": False, "no": False}.get(raw.strip().lower())
    change = " ".join(str(data.get("change") or data.get("summary") or "").split())[:300]
    return (raw if isinstance(raw, bool) else None), change


# ---------------------------------------------------------------------------- the history of checks
def add_check(checks: Any, outcome: str, now: str, note: str = "") -> list[dict]:
    """Append ``{at, outcome, note?}`` to the history, newest last. A quiet check right after one with the same
    outcome folds into it (``count``, ``last_at``), so a monitor that checks every 5 minutes stays readable."""
    out = [dict(c) for c in checks if isinstance(c, dict)] if isinstance(checks, list) else []
    last = out[-1] if out else None
    if last and outcome in QUIET_OUTCOMES and last.get("outcome") == outcome:
        last["count"] = int(last.get("count") or 1) + 1
        last["last_at"] = now
        return out
    entry: dict[str, Any] = {"at": now, "outcome": outcome}
    if note:
        entry["note"] = note[:300]
    out.append(entry)
    return out[-MAX_CHECKS:]
