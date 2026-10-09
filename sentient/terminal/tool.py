"""The ``terminal`` tool plugin: ``terminal_run`` (docs/API.md section 18)."""

from __future__ import annotations

from typing import Any

from sentient.terminal.guard import IS_WINDOWS
from sentient.terminal.service import TOOL_NAME
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

SHELL_WORDS = "PowerShell" if IS_WINDOWS else "bash, zsh or sh"
DESCRIPTION = (
    f"Run a command in a terminal on the user's computer ({SHELL_WORDS}) and get its exit code and output. Use it "
    "for git, builds, tests, package managers, scripts and file chores in the user's project folders. "
    "`command`: the exact command line. `cwd`: the folder to run in, absolute or relative to the default folder; "
    "leave it empty for the default folder. Each call is a new shell, so `cd` does not carry over: pass `cwd`. "
    "Commands can't answer questions, so use flags that don't ask (for example `git commit -m`, `npm init -y`). "
    "The user approves each command unless it is on their list of commands that never need asking."
)


def _service(ctx: ToolContext | Any) -> Any:
    return getattr((getattr(ctx, "extra", None) or {}).get("app"), "terminal", None)


def _risk(arguments: dict, ctx: ToolContext) -> Risk | None:
    svc = _service(ctx)
    return svc.risk_for(arguments or {}, ctx) if svc is not None else None


def _describe(arguments: dict, ctx: ToolContext) -> dict | None:
    svc = _service(ctx)
    return svc.describe(arguments or {}, ctx) if svc is not None else None


# every command asks again: "Allow for this chat" never covers the terminal (ADR 0019)
@tool(TOOL_NAME, risk=Risk.exec, description=DESCRIPTION, risk_fn=_risk, describe_fn=_describe, allow_for_chat=False)
async def terminal_run(ctx: ToolContext, command: str, cwd: str | None = None) -> dict:
    svc = _service(ctx)
    if svc is None:
        return {"ok": False, "error": "Running commands on this computer is not available right now."}
    return await svc.run(command, cwd, ctx)


class TerminalPlugin(ToolPlugin):
    id = "terminal"
    display_name = "Terminal"
    description = "Run commands on this computer in the folders you allow, after you approve them."
    category = "utilities"
    icon = "IconTerminal2"
    selection_hint = (
        "running commands on this computer: git, builds, tests, npm or pip, scripts, files and folders in a project"
    )
    tools = [terminal_run]


PLUGIN = TerminalPlugin()
