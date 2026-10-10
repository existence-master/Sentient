from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sentient import paths
from sentient.app import SentientApp
from sentient.integrations.mcp import mcp_tool_name
from tests.conftest import FakeProvider

SERVER = Path(__file__).with_name("mcp_echo_server.py")


def test_tool_name_sanitizing():
    assert mcp_tool_name("My Server!", "Get-Weather.now") == "mcp_my_server_get_weather_now"
    long = mcp_tool_name("s" * 40, "t" * 40)
    assert len(long) == 64 and long.startswith("mcp_sss")
    assert long != mcp_tool_name("s" * 40, "t" * 41)


async def _wait_status(app, name: str, status: str, timeout: float = 60) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        server = next(s for s in app.integrations.mcp.list() if s["name"] == name)
        if server["status"] == status:
            return server
        await asyncio.sleep(0.2)
    raise AssertionError(f"{name} never reached {status}: {server}")


async def test_mcp_stdio_server_lifecycle(config, isolated_home, keychain):
    gate = isolated_home / "echo-may-start"
    config.integrations.mcp_servers = {
        "Echo Test": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER), "--wait-for", str(gate)],
                      "enabled": True},
    }
    app = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "mcp.db", enable_background=False)
    # the server can't answer until the gate opens, so start-up only returns if it doesn't wait for MCP servers
    await asyncio.wait_for(app.start(), 120)
    try:
        waiting = next(s for s in app.integrations.mcp.list() if s["name"] == "Echo Test")
        assert waiting["status"] in {"disconnected", "connecting"}
        gate.touch()
        server = await _wait_status(app, "Echo Test", "connected")
        risks = {t["name"]: t["risk"] for t in server["tools"]}
        assert risks == {"mcp_echo_test_echo": "read", "mcp_echo_test_add_numbers": "write",
                         "mcp_echo_test_explode": "write"}
        assert "mcp_echo_test" in await app.integrations.connected_plugins()
        assert await app.integrations.is_connected("mcp_echo_test")

        ctx = app.agent.tool_context(None, "test")
        add = app.registry.get("mcp_echo_test_add_numbers")
        params = add.openai_schema()["function"]["parameters"]
        assert set(params["properties"]) == {"a", "b"} and params["required"] == ["a", "b"]
        assert (await add.call(ctx, {"a": 2, "b": 3}))["content"] == "5"
        assert (await app.registry.get("mcp_echo_test_echo").call(ctx, {"text": "hi"}))["content"] == "echo: hi"
        assert "error" in await app.registry.get("mcp_echo_test_explode").call(ctx, {})

        tested = await app.integrations.mcp.test("Echo Test")
        assert tested["ok"] is True and "echo" in tested["tools"]

        assert await app.integrations.mcp.remove("Echo Test")
        assert app.registry.get("mcp_echo_test_add_numbers") is None
        assert "Echo Test" not in app.config.integrations.mcp_servers
    finally:
        await app.stop()


async def test_mcp_add_with_env_secret_and_broken_server(app, keychain):
    added = await app.integrations.mcp.add(
        "echo2", {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)], "env": {"ECHO_TOKEN": "s3cret"}},
        wait_s=60)
    assert added["status"] == "connected" and added["env_keys"] == ["ECHO_TOKEN"]
    assert "s3cret" in keychain["mcp:echo2"]
    assert app.config.integrations.mcp_servers["echo2"]["env_keys"] == ["ECHO_TOKEN"]
    assert "s3cret" not in paths.config_file().read_text(encoding="utf-8")

    broken = await app.integrations.mcp.add(
        "broken", {"transport": "stdio", "command": "definitely-not-a-real-command-xyz", "args": []}, wait_s=30)
    assert broken["status"] == "error" and broken["error"]
    assert (await app.integrations.mcp.test("broken"))["ok"] is False
    await app.integrations.mcp.remove("broken")
    await app.integrations.mcp.remove("echo2")
    assert "mcp:echo2" not in keychain
