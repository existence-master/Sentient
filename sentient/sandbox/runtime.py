"""Files written into each run's working folder.

- ``sentient_tools.py``: the client half of the tool bridge. Standard library only, so it works
  in the engine's Python and in ``python:3.12-slim``. It talks to the engine over loopback HTTP
  with a one-time token, or, for a Docker container without a network, through request/response
  files in ``.rpc/`` inside the mounted working folder.
- ``_sentient_run.py``: runs ``script.py`` as ``__main__`` with the working folder importable,
  prints tracebacks without the runner's own frames and records the failing line for a friendly
  error message.
"""

from __future__ import annotations

import json

SCRIPT_FILE = "script.py"
RUNNER_FILE = "_sentient_run.py"
CLIENT_FILE = "sentient_tools.py"
RESULT_FILE = "_sentient_result.json"
ERROR_FILE = "_sentient_error.json"
RPC_DIR = ".rpc"
TMP_DIR = ".tmp"
HOME_DIR = ".home"

# Never reported or copied as files the script created.
INTERNAL_NAMES = frozenset(
    {SCRIPT_FILE, RUNNER_FILE, CLIENT_FILE, RESULT_FILE, ERROR_FILE, RPC_DIR, TMP_DIR, HOME_DIR, "__pycache__"}
)

_CLIENT_TEMPLATE = '''"""Sentient tools for this script (generated for one run; the token stops working when it ends).

    from sentient_tools import tools, result
    hits = tools.internet_search(query="python 3.13 release date")
    result({"answer": hits})

tools.<name>(**kwargs) calls a Sentient tool and returns its JSON result.
It raises ToolRefused when the tool cannot run from a script (call that tool directly instead)
and ToolError when the tool fails or reports an error. tools.available() lists what you can call.
result(value) sets the value handed back to the assistant.
"""

import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request

_CONFIG = json.loads(__CONFIG_JSON__)
_HERE = os.path.dirname(os.path.abspath(__file__))
_COUNTER = itertools.count(1)
_LOCK = threading.Lock()

__all__ = ["ToolError", "ToolRefused", "call_tool", "result", "tools"]


class ToolError(Exception):
    """A tool failed or returned an error."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class ToolRefused(ToolError):
    """The tool is not allowed inside scripts. Call it directly as a normal tool call."""


def _http(payload):
    request = urllib.request.Request(
        _CONFIG["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Sentient-Token": _CONFIG["token"]},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=_CONFIG["timeout"]) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ToolError("Sentient refused the connection (HTTP %s)." % exc.code) from None
    except (urllib.error.URLError, OSError) as exc:
        raise ToolError("Could not reach Sentient's tools: %s" % exc) from None


def _mailbox(payload):
    box = os.path.join(_HERE, _CONFIG["mailbox"])
    with _LOCK:
        request_id = "%s-%s" % (os.getpid(), next(_COUNTER))
    tmp = os.path.join(box, "req-%s.tmp" % request_id)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(payload, token=_CONFIG["token"]), fh)
    os.replace(tmp, os.path.join(box, "req-%s.json" % request_id))
    answer = os.path.join(box, "resp-%s.json" % request_id)
    deadline = time.monotonic() + _CONFIG["timeout"]
    delay = 0.005
    while not os.path.exists(answer):
        if time.monotonic() > deadline:
            raise ToolError("Sentient's tools did not answer in time.")
        time.sleep(delay)
        delay = min(0.05, delay * 1.5)
    with open(answer, encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        os.remove(answer)
    except OSError:
        pass
    return data


def call_tool(name, **arguments):
    """Call the Sentient tool ``name`` with keyword arguments and return its result."""
    payload = {"tool": name, "arguments": arguments}
    data = _mailbox(payload) if _CONFIG.get("mailbox") else _http(payload)
    if data.get("refused"):
        raise ToolRefused(data.get("error") or "%s is not allowed in scripts." % name)
    if not data.get("ok"):
        raise ToolError("%s failed: %s" % (name, data.get("error") or "unknown error"))
    value = data.get("result")
    if isinstance(value, dict) and value.get("error"):
        raise ToolError("%s reported an error: %s" % (name, value.get("error")), value)
    return value


class _Tools:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            if args:
                raise TypeError("Pass tool arguments by name, for example tools.%s(query=...)." % name)
            return call_tool(name, **kwargs)

        call.__name__ = name
        return call

    def available(self):
        """Names of the tools this script may call."""
        return list(_CONFIG["tools"])

    def __dir__(self):
        return [*_CONFIG["tools"], "available"]


tools = _Tools()


def result(value):
    """Hand ``value`` back to the assistant as this script's result (must be JSON-friendly)."""
    with open(os.path.join(_HERE, _CONFIG["result_file"]), "w", encoding="utf-8") as fh:
        json.dump(value, fh, default=str, ensure_ascii=False)
'''

RUNNER_SOURCE = '''import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)
sys.argv = ["__SCRIPT__"]
sys.dont_write_bytecode = True


def _main():
    with open(os.path.join(_HERE, "__SCRIPT__"), encoding="utf-8") as fh:
        source = fh.read()
    code = compile(source, "__SCRIPT__", "exec")
    namespace = {"__name__": "__main__", "__file__": os.path.join(_HERE, "__SCRIPT__"), "__builtins__": __builtins__}
    try:  # small models often forget the import: make tools and result available anyway
        import sentient_tools as _bridge
        namespace["tools"] = _bridge.tools
        namespace["result"] = _bridge.result
    except Exception:
        pass
    exec(code, namespace)


try:
    _main()
except SystemExit:
    raise
except BaseException as exc:
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "__SCRIPT__":
            line = frame.lineno
    if line is None and isinstance(exc, SyntaxError):
        line = exc.lineno
    tb = exc.__traceback__
    for _ in range(2):  # hide this runner's frames
        if tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
    traceback.print_exception(type(exc), exc, tb)
    try:
        with open(os.path.join(_HERE, "__ERROR__"), "w", encoding="utf-8") as fh:
            json.dump({"type": type(exc).__name__, "message": str(exc), "line": line}, fh)
    except OSError:
        pass
    sys.stderr.flush()
    sys.exit(1)
'''.replace("__SCRIPT__", SCRIPT_FILE).replace("__ERROR__", ERROR_FILE)


def client_source(*, url: str | None, token: str, mailbox: str | None, tool_names: list[str], timeout_s: float) -> str:
    config = {
        "url": url,
        "token": token,
        "mailbox": mailbox,
        "tools": sorted(tool_names),
        "timeout": max(5.0, float(timeout_s)),
        "result_file": RESULT_FILE,
    }
    return _CLIENT_TEMPLATE.replace("__CONFIG_JSON__", repr(json.dumps(config)))
