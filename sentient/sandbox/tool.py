"""The ``code`` tool plugin: ``execute_code`` (docs/API.md section 11)."""

from __future__ import annotations

import inspect

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

DESCRIPTION = (
    "Write and run a short Python 3 script, then get back what it printed, its result and any files it saved. "
    "Prefer this over many separate tool calls when you need to call tools repeatedly (for example search ten "
    "things and compare them), process or reshape data, do exact calculations, or make a chart or file. "
    "Inside the script: `from sentient_tools import tools, result`; `tools.<tool_name>(arg=value)` calls one of "
    "your tools and returns its JSON result (raises ToolError on failure); `result(value)` returns a value to "
    "you, otherwise the printed output is what you get. From a script only tools that look things up or change "
    "Sentient's own memory, files and tasks run; tools that send, delete, change outside services or run code "
    "are refused, so call those directly. Files saved in the current folder are copied to files/outputs/. "
    "Use the standard library; pass tool arguments by name. "
    "`purpose`: one plain sentence telling the user what the script does."
)


@tool("execute_code", risk=Risk.exec, description=DESCRIPTION)
async def execute_code(ctx: ToolContext, code: str, purpose: str) -> dict:
    app = (ctx.extra or {}).get("app")
    sandbox = getattr(app, "sandbox", None)
    if sandbox is None:
        return {"ok": False, "error": "Running code is not available right now."}
    # ctx.progress streams tool_progress events inside a chat turn and is a no-op elsewhere (docs/API.md section 10)
    progress = getattr(ctx, "progress", None)
    on_output = None
    if callable(progress):
        async def on_output(kind: str, text: str) -> None:
            maybe = progress({"kind": kind, "text": text})
            if inspect.isawaitable(maybe):
                await maybe

    return await sandbox.run(code, session_id=ctx.session_id, channel=ctx.channel, on_output=on_output)


class CodePlugin(ToolPlugin):
    id = "code"
    display_name = "Code"
    description = "Write and run short Python scripts that can use Sentient's tools."
    category = "utilities"
    icon = "IconCode"
    selection_hint = (
        "many similar tool calls in one go, processing or analysing data, exact calculations, "
        "charts, converting or generating files"
    )
    tools = [execute_code]


PLUGIN = CodePlugin()
