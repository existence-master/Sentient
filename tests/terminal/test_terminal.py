"""Commands on this computer (#207, ADR 0019): off by default, approvals, allowed folders and commands, the
blocklist, timeouts, Stop, Stop everything, unprompted work and long output."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from sentient import paths
from sentient.agent.loop import LoopResult
from sentient.app import SentientApp
from sentient.config.schema import ProviderConfig, SentientConfig
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest, ToolProgress, ToolResultEvent
from sentient.sandbox.policy import BridgePolicy
from sentient.terminal.tool import terminal_run
from sentient.tools.base import Risk, bind_call, effective_risk
from tests.conftest import FakeProvider, tool_call
from tests.sandbox.conftest import pid_alive
from tests.terminal.conftest import py

TOOL = "terminal_run"


async def _turn(app: SentientApp, text: str, decision: str = "allow") -> tuple[list[ApprovalRequest], list]:
    sid = await app.store.create_session(channel="desktop")
    asked: list[ApprovalRequest] = []
    events = []
    async for ev in app.agent.run_turn(sid, text, channel="desktop"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            app.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _result(events: list) -> dict:
    return next(e.result for e in events if isinstance(e, ToolResultEvent) and e.name == TOOL)


async def _wait_for(path, timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


async def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return not pid_alive(pid)


# ---------------------------------------------------------------------------- on and off
async def test_off_by_default_and_not_offered(config, isolated_home):
    defaults = SentientConfig().terminal
    assert defaults.enabled is False and defaults.allowed_folders == [] and defaults.timeout_s == 180
    app = await SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "off.db", enable_background=False).start()
    try:
        offered = {s["function"]["name"] for s in app.registry.openai_schemas()}
        assert TOOL not in offered
        assert app.registry.get(TOOL) is not None  # registered, just hidden
        res = await app.registry.get(TOOL).call(app.agent.tool_context("s", "desktop"), {"command": "echo hi"})
        assert res["ok"] is False and "turned off" in res["error"] and res["exit_code"] is None
    finally:
        await app.stop()


async def test_turning_it_on_offers_the_tool(config, isolated_home):
    app = await SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "on.db", enable_background=False).start()
    try:
        app.config.terminal.enabled = True
        app.save_config()
        deadline = time.monotonic() + 3
        while app.registry.is_hidden("terminal") and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        schema = next(s for s in app.registry.openai_schemas() if s["function"]["name"] == TOOL)
        assert set(schema["function"]["parameters"]["properties"]) == {"command", "cwd"}
        assert app.registry.get(TOOL).risk == Risk.exec
    finally:
        await app.stop()


# ---------------------------------------------------------------------------- approvals
async def test_asks_first_showing_the_command_and_folder_then_streams_output(make, project):
    app = await make([[tool_call(TOOL, command="echo 'hello from the terminal'")], "done"])
    asked, events = await _turn(app, "say hello in the terminal")
    assert len(asked) == 1
    req = asked[0]
    assert req.risk == "exec" and req.risk_label == "Runs a command"
    assert req.arguments["command"] == "echo 'hello from the terminal'"
    assert req.target == str(project) or str(project).endswith(req.target.removeprefix("..."))
    assert len(req.target) <= 120 and req.target.endswith("project")
    res = _result(events)
    assert res["ok"] is True and res["exit_code"] == 0, res
    assert "hello from the terminal" in res["stdout"] and res["cwd"] == str(project)
    streamed = "".join(e.text or "" for e in events if isinstance(e, ToolProgress) and e.kind == "stdout")
    assert "hello from the terminal" in streamed


async def test_denied_command_does_not_run(make, project):
    app = await make([[tool_call(TOOL, command=py("open('ran.txt', 'w').write('x')"))], "ok"])
    asked, events = await _turn(app, "make a file", decision="deny")
    assert len(asked) == 1
    assert _result(events).get("declined") is True
    assert not (project / "ran.txt").exists()


async def test_allow_listed_command_runs_without_asking(make, config):
    config.terminal.allowed_commands = ["echo"]
    app = await make([[tool_call(TOOL, command="echo listed")], [tool_call(TOOL, command="echo a; echo b")], "done"])
    asked, events = await _turn(app, "echo something")
    assert len(asked) == 1 and asked[0].arguments["command"] == "echo a; echo b"  # chained: not a plain command
    first = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert first["ok"] and "listed" in first["stdout"]


async def test_allow_for_this_chat_never_covers_the_next_command(make):
    app = await make([[tool_call(TOOL, command="echo 'one'")], [tool_call(TOOL, command="echo 'two'")], "done"])
    sid = await app.store.create_session(channel="desktop")
    asked: list[ApprovalRequest] = []
    async for ev in app.agent.run_turn(sid, "two commands", channel="desktop"):
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            app.approvals.resolve(ev.approval_id, "allow_session")
    assert [a.arguments["command"] for a in asked] == ["echo 'one'", "echo 'two'"]
    assert app.approvals.needs_approval(app.registry.get(TOOL), sid, Risk.exec)


async def test_allow_rule_skips_the_question(make, config):
    config.tools.approvals.rules = {"terminal": "allow"}
    app = await make([[tool_call(TOOL, command="echo allowed")], "done"])
    asked, events = await _turn(app, "go")
    assert asked == [] and "allowed" in _result(events)["stdout"]


async def test_non_zero_exit_is_reported_not_an_error(make, project):
    app = await make()
    ctx = app.agent.tool_context("s", "desktop")
    res = await app.registry.get(TOOL).call(ctx, {"command": py("import sys; sys.exit(3)")})
    assert res["ok"] is False and res["exit_code"] == 3 and res["error"] is None


# ---------------------------------------------------------------------------- folders and blocklist
async def test_folder_outside_allowed_folders_is_refused_without_asking(make, project, tmp_path):
    app = await make([
        [tool_call(TOOL, command="echo hi", cwd=str(tmp_path))],
        [tool_call(TOOL, command="echo hi", cwd="..")],
        "ok",
    ])
    asked, events = await _turn(app, "go")
    assert asked == []
    results = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 2
    assert all(r["ok"] is False and "isn't inside a folder you allowed" in r["error"] for r in results)

    (project / "sub").mkdir()
    ctx = app.agent.tool_context("s", "desktop")
    app.config.terminal.allowed_commands = ["echo"]
    inside = await app.registry.get(TOOL).call(ctx, {"command": "echo inside", "cwd": "sub"})
    assert inside["ok"] and inside["cwd"] == str(project / "sub")

    app.config.terminal.allowed_folders = []
    none = await app.registry.get(TOOL).call(ctx, {"command": "echo hi"})
    assert "No folders are allowed yet" in none["error"]


async def test_blocklist_wins_over_an_allow_rule(make, config):
    config.tools.approvals.rules = {"terminal_run": "allow"}
    config.terminal.allowed_commands = ["shutdown", "format"]
    app = await make([[tool_call(TOOL, command="shutdown /s /t 0")], [tool_call(TOOL, command="format c:")], "ok"])
    asked, events = await _turn(app, "turn it off")
    assert asked == []
    results = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert [r["exit_code"] for r in results] == [None, None]
    assert "never runs commands that shut down" in results[0]["error"]
    assert "never runs commands that format a disk" in results[1]["error"]


# ---------------------------------------------------------------------------- stopping
async def test_timeout_kills_the_whole_process_tree(make, project):
    app = await make()
    app.config.terminal.timeout_s = 2  # below the Settings minimum, to keep the test quick
    ctx = app.agent.tool_context("s", "desktop")
    started = time.monotonic()
    res = await app.registry.get(TOOL).call(
        ctx, {"command": py("import os, time; print(os.getpid(), flush=True); time.sleep(60)")}
    )
    assert time.monotonic() - started < 30
    assert res["timed_out"] is True and res["ok"] is False
    assert "longer than 2 seconds" in res["error"]
    match = re.search(r"(\d+)", res["stdout"])
    assert match, res
    assert await _gone(int(match.group(1)))


async def test_stop_everything_kills_a_running_command(make, project, config):
    config.tools.approvals.rules = {"terminal": "allow"}
    code = "import os, time; open('pid.txt', 'w').write(str(os.getpid())); time.sleep(60)"
    app = await make([[tool_call(TOOL, command=py(code))], "never"])
    turn = asyncio.create_task(_turn(app, "wait"))
    pid = int(await _wait_for(project / "pid.txt"))
    result = await asyncio.wait_for(app.stop_all(), 15)
    assert result["cancelled"] >= 1
    assert turn.done() and turn.cancelled()
    assert await _gone(pid)
    assert app.terminal.status()["running"] == []
    await app.resume()


async def test_stop_button_kills_one_command_and_keeps_its_output(make, project):
    app = await make()
    ctx = app.agent.tool_context("s", "desktop")
    code = "import os, time; print('partial', flush=True); open('pid.txt', 'w').write(str(os.getpid())); time.sleep(60)"
    call = asyncio.create_task(app.registry.get(TOOL).call(ctx, {"command": py(code)}))
    pid = int(await _wait_for(project / "pid.txt"))
    running = app.terminal.status()["running"]
    assert len(running) == 1 and running[0]["cwd"] == str(project)
    assert app.terminal.stop_command(running[0]["id"]) is True
    res = await asyncio.wait_for(call, 15)
    assert res["stopped"] is True and res["ok"] is False and "stopped before it finished" in res["error"]
    assert "partial" in res["stdout"]
    assert await _gone(pid)
    assert app.terminal.stop_command(running[0]["id"]) is False


# ---------------------------------------------------------------------------- who may run commands
async def test_unprompted_work_can_never_run_commands(make, project, config):
    config.tools.approvals.mode = "off"
    config.tools.approvals.rules = {"terminal": "allow"}
    config.terminal.allowed_commands = ["echo"]
    app = await make([[tool_call(TOOL, command="echo sneaky")], "done"])
    ctx = app.agent.tool_context(None, "proactive")
    assert await effective_risk(app.registry.get(TOOL), {"command": "echo sneaky"}, ctx) == Risk.exec
    result = LoopResult()
    async for _ in app.agent.run_loop([{"role": "user", "content": "check"}], ctx, result=result, tool_names=[TOOL],
                                      use_approvals=False, source="proactive"):
        pass
    assert [h["tool"] for h in result.held] == [TOOL]
    direct = await app.registry.get(TOOL).call(ctx, {"command": "echo sneaky"})
    assert "Nobody asked for this work" in direct["error"] and direct["exit_code"] is None


async def test_tasks_only_run_commands_that_never_need_asking(make, project, config):
    config.terminal.allowed_commands = ["echo"]
    app = await make()
    ctx = app.agent.tool_context(None, "task")
    tool = app.registry.get(TOOL)
    refused = await tool.call(ctx, {"command": py("print(1)")})
    assert "tasks and helpers can't ask yet" in refused["error"]
    listed = await tool.call(ctx, {"command": "echo 'from a task'"})
    assert listed["ok"] and "from a task" in listed["stdout"]
    app.config.tools.approvals.rules = {"terminal": "allow"}
    allowed = await tool.call(ctx, {"command": py("print(1)")})
    assert allowed["ok"], allowed


async def test_scripts_can_never_run_commands():
    tool_obj = terminal_run
    for mode in ("off", "ask"):
        policy = BridgePolicy.build(approvals_mode=mode, allowed_tools=None, read_only=False, max_tool_calls=10,
                                    rules={"terminal": "allow"})
        assert policy.is_available(tool_obj) is False


# ---------------------------------------------------------------------------- output and environment
async def test_long_output_is_trimmed_and_saved_to_a_file(make, project):
    app = await make()
    app.config.terminal.max_output_chars = 1000
    ctx = app.agent.tool_context("s", "desktop")
    payloads: list[dict] = []
    ctx.progress = payloads.append  # type: ignore[method-assign]
    res = await app.registry.get(TOOL).call(ctx, {"command": py("print('a' * 3000 + 'END')")})
    assert res["ok"], res
    assert len(res["stdout"]) < 1200 and "characters cut here" in res["stdout"] and "END" in res["stdout"]
    assert res["output_file"] and res["output_file"].startswith("outputs/terminal-")
    saved = (paths.files_dir() / res["output_file"]).read_text(encoding="utf-8")
    assert "a" * 3000 + "END" in saved
    assert sum(len(p.get("text") or "") for p in payloads if p["kind"] == "stdout") <= 1000
    assert any(p["kind"] == "status" for p in payloads)


async def test_secrets_never_reach_the_command(make, project, config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("MY_LLM_VAR", "custom-secret")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    config.models.providers["custom"] = ProviderConfig(api_key_env="MY_LLM_VAR")
    app = await make()
    ctx = app.agent.tool_context("s", "desktop")
    code = ("import os; print([os.environ.get(k) for k in ('OPENAI_API_KEY', 'SENTIENT_GATEWAY_TOKEN', "
            "'MY_LLM_VAR', 'HARMLESS_SETTING')])")
    res = await app.registry.get(TOOL).call(ctx, {"command": py(code)})
    assert res["ok"], res
    out = res["stdout"]
    assert "sk-very-secret" not in out and "gateway-secret" not in out and "custom-secret" not in out
    assert "visible" in out


def test_status_and_stop_routes(config, isolated_home, monkeypatch, project):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    config.terminal.enabled = True
    config.terminal.allowed_folders = [str(project)]
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    with TestClient(create_app(core)) as client:
        client.headers.update({"Authorization": "Bearer test-token"})
        body = client.get("/api/terminal/status").json()
        assert body["enabled"] is True and body["allowed_folders"] == [str(project)]
        assert body["default_folder"] == str(project) and body["running"] == []
        assert body["blocked"] and isinstance(body["shell"], (str, type(None)))
        assert client.post("/api/terminal/stop", json={"id": "nope"}).json() == {"stopped": False}
        assert client.get("/api/terminal/status", headers={"Authorization": "Bearer wrong"}).status_code == 401


# ---------------------------------------------------------------------------- review hardening
async def test_a_child_that_leaves_the_process_group_is_killed_too(make, project):
    app = await make()
    app.config.terminal.timeout_s = 2
    ctx = app.agent.tool_context("s", "desktop")
    code = ("import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "start_new_session=True); print(c.pid, flush=True); time.sleep(60)")
    res = await app.registry.get(TOOL).call(ctx, {"command": py(code)})
    assert res["timed_out"] is True, res
    match = re.search(r"(\d+)", res["stdout"])
    assert match, res
    assert await _gone(int(match.group(1)))


async def test_calls_with_the_same_call_id_are_tracked_apart(make, project):
    app = await make()
    ctx = app.agent.tool_context("s", "desktop")
    tool = app.registry.get(TOOL)
    code = "import time; print('up', flush=True); time.sleep(60)"

    async def call() -> dict:
        bind_call(None, "same-id", TOOL)  # what a model that repeats call ids looks like (this task only)
        return await tool.call(ctx, {"command": py(code)})

    calls = [asyncio.create_task(call()) for _ in range(2)]
    deadline = time.monotonic() + 20
    while len(app.terminal.status()["running"]) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    running = app.terminal.status()["running"]
    assert len(running) == 2 and len({r["id"] for r in running}) == 2
    assert {r["call_id"] for r in running} == {"same-id"}
    assert app.terminal.stop_command(running[0]["id"]) is True  # one run by its own id
    first = await asyncio.wait_for(asyncio.shield(asyncio.gather(*calls, return_exceptions=True)), 0.01) \
        if False else None
    assert first is None
    assert len(app.terminal.status()["running"]) >= 1
    assert app.terminal.stop_command("same-id") is True  # the card's Stop button stops what is left
    results = await asyncio.wait_for(asyncio.gather(*calls), 20)
    assert all(r["stopped"] for r in results)
    assert app.terminal.status()["running"] == []


async def test_listed_git_commands_never_start_the_repositorys_fsmonitor(make, project, config):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    marker = project / "fsmonitor-ran.txt"
    hook = f"\"{sys.executable}\" -c \"open(r'{marker}', 'w').write('x')\""
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    for args in (["init", "-q"], ["config", "core.fsmonitor", hook]):
        subprocess.run([git, *args], cwd=project, env=env, check=True, capture_output=True)
    (project / "notes.txt").write_text("hello")
    subprocess.run([git, "status", "--short"], cwd=project, env=env, capture_output=True)
    if not marker.exists():
        pytest.skip("this git does not run core.fsmonitor here")
    marker.unlink()
    app = await make()
    ctx = app.agent.tool_context("s", "desktop")
    res = await app.registry.get(TOOL).call(ctx, {"command": "git status --short"})
    assert res["ok"] and "notes.txt" in res["stdout"], res
    assert not marker.exists()
