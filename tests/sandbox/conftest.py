"""Fixtures for the sandbox package: a started app with a fake tool kit covering every risk."""

from __future__ import annotations

import sys

import pytest

from sentient.app import SentientApp
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider

CALLS: list[tuple[str, dict]] = []


@tool("kit_lookup", risk=Risk.read)
async def kit_lookup(ctx: ToolContext, q: str) -> dict:
    """Look something up."""
    CALLS.append(("kit_lookup", {"q": q, "session_id": ctx.session_id}))
    return {"echo": q, "session_id": ctx.session_id}


@tool("kit_note_save", risk=Risk.write, internal=True)
async def kit_note_save(ctx: ToolContext, text: str) -> dict:
    """Save a note inside Sentient."""
    CALLS.append(("kit_note_save", {"text": text}))
    return {"saved": text}


@tool("kit_calendar_create", risk=Risk.write)
async def kit_calendar_create(ctx: ToolContext, title: str) -> dict:
    """Create a calendar event."""
    CALLS.append(("kit_calendar_create", {"title": title}))
    return {"created": title}


@tool("kit_email_send", risk=Risk.send)
async def kit_email_send(ctx: ToolContext, to: str) -> dict:
    """Send an email."""
    CALLS.append(("kit_email_send", {"to": to}))
    return {"sent": to}


@tool("kit_shell", risk=Risk.exec)
async def kit_shell(ctx: ToolContext, cmd: str) -> dict:
    """Run a command."""
    CALLS.append(("kit_shell", {"cmd": cmd}))
    return {"ran": cmd}


@tool("kit_click", risk=Risk.read)
async def kit_click(ctx: ToolContext, label: str) -> dict:
    """Click a button."""
    CALLS.append(("kit_click", {"label": label}))
    return {"clicked": label}


kit_click.risk_fn = lambda arguments, ctx: Risk.send if "order" in str(arguments.get("label", "")).lower() else None  # type: ignore[attr-defined]


@tool("kit_boom", risk=Risk.read)
async def kit_boom(ctx: ToolContext) -> dict:
    """Always fails."""
    raise RuntimeError("the kit exploded")


@tool("kit_soft_fail", risk=Risk.read)
async def kit_soft_fail(ctx: ToolContext) -> dict:
    """Reports an error instead of raising."""
    return {"error": "not connected"}


class KitPlugin(ToolPlugin):
    id = "kit"
    display_name = "Kit"
    tools = [kit_lookup, kit_note_save, kit_calendar_create, kit_email_send, kit_shell, kit_click, kit_boom, kit_soft_fail]


@pytest.fixture
def calls():
    CALLS.clear()
    return CALLS


@pytest.fixture
async def sandbox_app(config, isolated_home, calls):
    config.tools.approvals.mode = "ask"
    config.sandbox.backend = "process"
    app = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "sandbox.db", enable_background=False)
    await app.start()
    app.registry.register(KitPlugin())
    yield app
    await app.stop()


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            k32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child of ours counts as dead
        waited, _ = os.waitpid(pid, os.WNOHANG)
        return waited == 0
    except ChildProcessError:
        return True
