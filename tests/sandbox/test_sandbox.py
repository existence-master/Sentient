from __future__ import annotations

import asyncio
import json
import re
import textwrap
import time
import urllib.error
import urllib.request

import pytest

from sentient import paths
from sentient.sandbox import backends
from sentient.sandbox.bridge import ToolBridge
from sentient.sandbox.policy import BridgePolicy, Refused
from sentient.tools.base import Risk, Tool, ToolContext
from tests.sandbox.conftest import pid_alive

RESULT_KEYS = {"ok", "backend", "stdout", "stderr", "result", "files_created", "tool_calls", "duration_ms", "error"}

POLICY_SCRIPT = textwrap.dedent(
    """
    from sentient_tools import tools, result, ToolError, ToolRefused

    out = {}
    def attempt(key, fn):
        try:
            out[key] = {"value": fn()}
        except ToolRefused as exc:
            out[key] = {"refused": str(exc)}
        except ToolError as exc:
            out[key] = {"error": str(exc)}

    attempt("read", lambda: tools.kit_lookup(q="hello"))
    attempt("internal_write", lambda: tools.kit_note_save(text="n1"))
    attempt("external_write", lambda: tools.kit_calendar_create(title="standup"))
    attempt("send", lambda: tools.kit_email_send(to="a@b.c"))
    attempt("exec", lambda: tools.kit_shell(cmd="dir"))
    attempt("click_safe", lambda: tools.kit_click(label="Next page"))
    attempt("click_order", lambda: tools.kit_click(label="Place order"))
    attempt("code", lambda: tools.execute_code(code="print(1)", purpose="x"))
    attempt("unknown", lambda: tools.no_such_tool())
    attempt("boom", lambda: tools.kit_boom())
    attempt("soft_fail", lambda: tools.kit_soft_fail())
    out["available"] = tools.available()
    result(out)
    """
)


async def test_run_returns_sandbox_result(sandbox_app):
    res = await sandbox_app.sandbox.run(
        "import sys\nprint('hi there')\nprint('warn', file=sys.stderr)\n"
        "from sentient_tools import result\nresult({'answer': 6 * 7})\n"
    )
    assert set(res) == RESULT_KEYS
    assert res["ok"] is True, res
    assert res["backend"] == "process"
    assert res["stdout"].strip() == "hi there"
    assert "warn" in res["stderr"]
    assert res["result"] == {"answer": 42}
    assert res["error"] is None and res["tool_calls"] == 0 and res["duration_ms"] > 0
    # the working folder is removed afterwards
    assert not any((paths.home() / "sandbox").iterdir())


async def test_policy_in_ask_mode(sandbox_app, calls):
    res = await sandbox_app.sandbox.run(POLICY_SCRIPT, session_id="s1", channel="desktop")
    assert res["ok"], res
    out = res["result"]
    assert out["read"] == {"value": {"echo": "hello", "session_id": "s1"}}
    assert out["internal_write"] == {"value": {"saved": "n1"}}
    for key, tool_name in [("external_write", "kit_calendar_create"), ("send", "kit_email_send"), ("exec", "kit_shell")]:
        assert "refused" in out[key], key
        assert "directly" in out[key]["refused"] and tool_name in out[key]["refused"]
    assert out["click_safe"] == {"value": {"clicked": "Next page"}}
    assert "refused" in out["click_order"] and "risk send" in out["click_order"]["refused"]
    assert "not available inside scripts" in out["code"]["refused"]
    assert "no tool named" in out["unknown"]["refused"]
    assert "the kit exploded" in out["boom"]["error"]
    assert "not connected" in out["soft_fail"]["error"]
    assert "execute_code" not in out["available"] and "kit_lookup" in out["available"]
    ran = [name for name, _ in calls]
    assert "kit_calendar_create" not in ran and "kit_email_send" not in ran and "kit_shell" not in ran
    assert res["tool_calls"] == 5  # lookup, note_save, click_safe, boom, soft_fail


async def test_policy_off_allows_everything_but_code(sandbox_app, calls):
    sandbox_app.config.tools.approvals.mode = "off"
    res = await sandbox_app.sandbox.run(POLICY_SCRIPT)
    out = res["result"]
    assert out["send"] == {"value": {"sent": "a@b.c"}}
    assert out["exec"] == {"value": {"ran": "dir"}}
    assert out["click_order"] == {"value": {"clicked": "Place order"}}
    assert "refused" in out["code"]


async def test_read_only_and_allowed_tools(sandbox_app):
    script = textwrap.dedent(
        """
        from sentient_tools import tools, result, ToolRefused
        try:
            tools.kit_note_save(text="x")
            saved = "ran"
        except ToolRefused as exc:
            saved = str(exc)
        result({"saved": saved, "lookup": tools.kit_lookup(q="q"), "available": tools.available()})
        """
    )
    res = await sandbox_app.sandbox.run(script, read_only=True)
    assert "look things up" in res["result"]["saved"]
    assert res["result"]["lookup"]["echo"] == "q"

    res = await sandbox_app.sandbox.run(script, allowed_tools=["kit_lookup"])
    assert "not one of the tools" in res["result"]["saved"]
    assert res["result"]["available"] == ["kit_lookup"]


async def test_policy_unit_blocks_subagents_and_voice():
    policy = BridgePolicy(approvals_mode="off")
    ctx = ToolContext(store=None, config=None, llm=None)

    async def fn(ctx):
        return None

    for plugin, name in [("subagents", "delegate_task"), ("voice", "speak"), ("other", "delegate_tasks")]:
        t = Tool(name=name, description="", fn=fn, params_model=None, risk=Risk.read, plugin=plugin)  # type: ignore[arg-type]
        assert not policy.is_available(t)
        with pytest.raises(Refused):
            await policy.check(t, name, {}, ctx, 0)

    broken = Tool(name="b", description="", fn=fn, params_model=None, risk=Risk.read, plugin="x")  # type: ignore[arg-type]

    def bad_risk(arguments, ctx):
        raise ValueError("nope")

    broken.risk_fn = bad_risk  # type: ignore[attr-defined]
    with pytest.raises(Refused):
        await BridgePolicy(approvals_mode="ask").check(broken, "b", {}, ctx, 0)
    with pytest.raises(Refused, match="made 3 tool calls"):
        await BridgePolicy(approvals_mode="off", max_tool_calls=3).check(broken, "b", {}, ctx, 3)


async def test_policy_unit_follows_lasting_rules():
    ctx = ToolContext(store=None, config=None, llm=None)

    async def fn(ctx):
        return None

    def make(name: str, risk: Risk, plugin: str = "kit", label: str | None = None) -> Tool:
        t = Tool(name=name, description="", fn=fn, params_model=None, risk=risk, plugin=plugin)  # type: ignore[arg-type]
        if label:
            t.describe_fn = lambda a, c: {"risk_label": label}  # type: ignore[assignment]
        return t

    look, post, buy = make("kit_lookup", Risk.read), make("kit_post", Risk.send), make("kit_buy", Risk.send, label="Purchase")
    policy = BridgePolicy(approvals_mode="ask", rules={"kit_lookup": "never", "kit_post": "allow", "kit_buy": "allow"})
    assert not policy.is_available(look) and policy.is_available(post)
    with pytest.raises(Refused, match="never use \"Lookup\""):
        await policy.check(look, "kit_lookup", {}, ctx, 0)
    # "allow" never widens what scripts may do: they only read (ADR 0012), purchases included
    for t in (post, buy):
        with pytest.raises(Refused, match="cannot run from inside a script"):
            await policy.check(t, t.name, {}, ctx, 0)
    ask = BridgePolicy(approvals_mode="off", rules={"kit": "ask"})
    with pytest.raises(Refused, match="always ask before using kit"):
        await ask.check(look, "kit_lookup", {}, ctx, 0)


async def test_syntax_error_is_friendly_and_nothing_runs(sandbox_app):
    res = await sandbox_app.sandbox.run("x = 1\nprint(x\ny = 2\n")
    assert res["ok"] is False
    assert re.search(r"syntax error on line \d", res["error"])
    assert not (paths.home() / "sandbox").exists() or not any((paths.home() / "sandbox").iterdir())


async def test_runtime_error_reports_line(sandbox_app):
    res = await sandbox_app.sandbox.run("x = 1\ny = x / 0\n")
    assert res["ok"] is False
    assert "on line 2" in res["error"] and "ZeroDivisionError" in res["error"]
    assert "script.py" in res["stderr"] and "_sentient_run" not in res["stderr"]


async def test_refused_tool_uncaught_gives_clear_error(sandbox_app):
    res = await sandbox_app.sandbox.run("from sentient_tools import tools\n\ntools.kit_email_send(to='x')\n")
    assert res["ok"] is False
    assert "on line 3" in res["error"] and "refused" in res["error"] and "kit_email_send" in res["error"]


async def test_timeout_kills_whole_process_tree(sandbox_app):
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print("child", child.pid, flush=True)
        time.sleep(120)
        """
    )
    started = time.monotonic()
    res = await sandbox_app.sandbox.run(script, timeout_s=3)
    assert time.monotonic() - started < 20
    assert res["ok"] is False
    assert "longer than 3 seconds" in res["error"]
    match = re.search(r"child (\d+)", res["stdout"])
    assert match, res
    pid = int(match.group(1))
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not pid_alive(pid)


async def test_files_are_copied_to_outputs(sandbox_app):
    script = textwrap.dedent(
        """
        import os
        os.makedirs("charts", exist_ok=True)
        with open("report.txt", "w") as fh:
            fh.write("total: 3")
        with open(os.path.join("charts", "data.csv"), "w") as fh:
            fh.write("a,b\\n1,2\\n")
        """
    )
    res = await sandbox_app.sandbox.run(script)
    assert res["ok"], res
    names = res["files_created"]
    assert len(names) == 2
    assert all(n.startswith("outputs/") for n in names)
    report = next(n for n in names if n.endswith("report.txt"))
    assert (paths.files_dir() / report).read_text() == "total: 3"
    assert any(n.endswith("charts/data.csv") for n in names)
    assert not any("sentient_tools" in n or "script.py" in n for n in names)


async def test_output_streams_and_is_capped(sandbox_app):
    sandbox_app.config.sandbox.max_output_chars = 1000
    chunks: list[tuple[str, str]] = []

    async def on_output(kind, text):
        chunks.append((kind, text))

    script = "import sys\nprint('first', flush=True)\nprint('oops', file=sys.stderr, flush=True)\nprint('x' * 5000)\n"
    res = await sandbox_app.sandbox.run(script, on_output=on_output)
    assert res["ok"]
    assert "".join(t for k, t in chunks if k == "stdout").startswith("first\n")
    assert "\r" not in res["stdout"]
    assert any(k == "stderr" and "oops" in t for k, t in chunks)
    assert sum(len(t) for k, t in chunks if k == "stdout") <= 1000
    assert "[output cut after 1000 characters]" in res["stdout"]


async def test_execute_code_tool_streams_progress(sandbox_app):
    t = sandbox_app.registry.get("execute_code")
    assert t is not None and t.risk == Risk.exec and t.plugin == "code"
    assert sandbox_app.registry.plugin("code").selection_hint
    ctx = sandbox_app.agent.tool_context("s9", "desktop")
    payloads: list[dict] = []
    ctx.progress = payloads.append  # type: ignore[attr-defined]
    res = await t.call(ctx, {"code": "print('streamed')", "purpose": "say hi"})
    assert res["ok"] and res["stdout"].strip() == "streamed"
    assert payloads and all(p["kind"] == "stdout" for p in payloads)
    assert "".join(p["text"] for p in payloads) == "streamed\n"


async def test_environment_is_scrubbed(sandbox_app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("PYTHONPATH", "C:/nowhere")
    script = "import os, sys\nfrom sentient_tools import result\nresult({'key': os.environ.get('OPENAI_API_KEY'), 'cwd': os.getcwd(), 'isolated': sys.flags.isolated})\n"
    res = await sandbox_app.sandbox.run(script)
    assert res["ok"], res
    assert res["result"]["key"] is None
    assert res["result"]["isolated"] == 1
    assert "sandbox" in res["result"]["cwd"]
    env = backends.scrubbed_env(paths.home() / "sandbox" / "x")
    assert "OPENAI_API_KEY" not in env and "PYTHONPATH" not in env


async def test_mailbox_transport(sandbox_app, monkeypatch, calls):
    monkeypatch.setattr(sandbox_app.sandbox, "_transport_for", lambda backend: "mailbox")
    res = await sandbox_app.sandbox.run(
        "from sentient_tools import tools, result\nresult([tools.kit_lookup(q=str(i))['echo'] for i in range(3)])\n"
    )
    assert res["ok"], res
    assert res["result"] == ["0", "1", "2"] and res["tool_calls"] == 3


async def test_bridge_rejects_wrong_token(sandbox_app, tmp_path):
    bridge = ToolBridge(
        sandbox_app.registry, BridgePolicy(), sandbox_app.agent.tool_context(None, "system"), run_dir=tmp_path
    )
    endpoint = await bridge.start()

    def post(token: str) -> tuple[int, dict]:
        req = urllib.request.Request(
            endpoint["url"], data=json.dumps({"tool": "kit_lookup", "arguments": {"q": "a"}}).encode(),
            headers={"X-Sentient-Token": token}, method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    try:
        assert (await asyncio.to_thread(post, "wrong"))[0] == 403
        status, body = await asyncio.to_thread(post, bridge.token)
        assert status == 200 and body["result"]["echo"] == "a"
        old = bridge.token
    finally:
        await bridge.stop()
    assert bridge.token != old


async def test_disabled_and_missing_docker(sandbox_app, monkeypatch):
    sandbox_app.config.sandbox.enabled = False
    sandbox_app.sandbox._apply_enabled()
    assert sandbox_app.registry.is_hidden("code")
    res = await sandbox_app.sandbox.run("print(1)")
    assert res["ok"] is False and "Settings" in res["error"]

    sandbox_app.config.sandbox.enabled = True
    sandbox_app.config.sandbox.backend = "docker"

    async def no_docker():
        return False

    monkeypatch.setattr(sandbox_app.sandbox, "docker_available", no_docker)
    res = await sandbox_app.sandbox.run("print(1)")
    assert res["ok"] is False and "Docker is not running" in res["error"]
    sandbox_app.config.sandbox.backend = "auto"
    assert (await sandbox_app.sandbox.status())["backend"] == "process"
    res = await sandbox_app.sandbox.run("print(1)")
    assert res["ok"] and res["backend"] == "process"


@pytest.mark.skipif(not backends.docker_available_sync(), reason="Docker is not running")
async def test_docker_backend(sandbox_app):
    sandbox_app.config.sandbox.backend = "docker"
    sandbox_app.config.sandbox.timeout_s = 600
    res = await sandbox_app.sandbox.run(
        "from sentient_tools import tools, result\nopen('out.txt','w').write('hi')\n"
        "print('in docker')\nresult(tools.kit_lookup(q='d'))\n"
    )
    assert res["backend"] == "docker"
    assert res["ok"], res
    assert res["result"]["echo"] == "d" and res["files_created"]
