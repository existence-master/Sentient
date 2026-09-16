"""Code execution: short Python scripts that call Sentient tools (docs/API.md section 11).

``await app.sandbox.run(code, ...)`` writes the script into a fresh working folder under
``~/.sentient/sandbox/<run_id>``, starts a one-time tool bridge, runs the script in a local
process or a Docker container, copies the files it created to ``files/outputs/<run_id>/``
and returns a **SandboxResult**.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.sandbox import backends
from sentient.sandbox.bridge import ToolBridge
from sentient.sandbox.policy import BridgePolicy
from sentient.sandbox.runtime import (
    CLIENT_FILE,
    ERROR_FILE,
    HOME_DIR,
    INTERNAL_NAMES,
    RESULT_FILE,
    RUNNER_FILE,
    RUNNER_SOURCE,
    SCRIPT_FILE,
    TMP_DIR,
    client_source,
)
from sentient.services import Service
from sentient.store.db import new_id
from sentient.tools.base import ToolContext

log = logging.getLogger(__name__)

OnOutput = Callable[[str, str], Any]
CODE_PLUGIN_ID = "code"


def empty_result(backend: str = "process") -> dict:
    return {
        "ok": False,
        "backend": backend,
        "stdout": "",
        "stderr": "",
        "result": None,
        "files_created": [],
        "tool_calls": 0,
        "duration_ms": 0,
        "error": None,
    }


def syntax_error_message(exc: SyntaxError) -> str:
    where = f" on line {exc.lineno}" if exc.lineno else ""
    msg = f"The code has a syntax error{where}: {exc.msg}."
    text = (exc.text or "").strip()
    if text:
        msg += f" The line is: {text[:200]}"
    return msg


class SandboxService(Service):
    name = "sandbox"

    def __init__(self, app: Any):
        super().__init__(app)
        self._sem: asyncio.Semaphore | None = None
        self._sem_size = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        from sentient.sandbox.tool import CodePlugin

        if self.app.registry.plugin(CODE_PLUGIN_ID) is None:
            self.app.registry.register(CodePlugin())
        self._apply_enabled()
        self._loops.append(asyncio.create_task(self._watch_config(), name="sandbox:config"))

    def _apply_enabled(self) -> None:
        if self.app.registry.plugin(CODE_PLUGIN_ID) is not None:
            self.app.registry.set_hidden(CODE_PLUGIN_ID, not self.app.config.sandbox.enabled)

    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as queue:
            while True:
                event = await queue.get()
                if event.get("type") == "config.updated":
                    self._apply_enabled()

    # ------------------------------------------------------------------ status
    async def docker_available(self) -> bool:
        return await backends.docker_available()

    async def resolve_backend(self) -> tuple[str, str | None]:
        """``(backend, error)``: the backend a run uses now, or why it cannot run."""
        choice = self.app.config.sandbox.backend
        if choice == "process":
            return "process", None
        available = await self.docker_available()
        if available:
            return "docker", None
        if choice == "docker":
            return "docker", (
                "Docker is not running, so the script could not start. Start Docker Desktop, or set code "
                "execution to 'auto' in Settings to run scripts as a local process."
            )
        return "process", None

    async def status(self) -> dict:
        cfg = self.app.config.sandbox
        backend, _ = await self.resolve_backend()
        return {
            "enabled": cfg.enabled,
            "backend": backend,
            "docker_available": await self.docker_available(),
            "python_version": backends.python_version(),
        }

    # ------------------------------------------------------------------ running
    def _semaphore(self) -> asyncio.Semaphore:
        size = self.app.config.sandbox.max_concurrent_runs
        if self._sem is None or self._sem_size != size:
            self._sem = asyncio.Semaphore(size)
            self._sem_size = size
        return self._sem

    def _tool_context(self, session_id: str | None, channel: str) -> ToolContext:
        if self.app.agent is not None:
            return self.app.agent.tool_context(session_id, channel)
        return ToolContext(
            store=self.app.store, config=self.app.config, llm=self.app.llm, memory=self.app.memory,
            session_id=session_id, channel=channel, extra=self.app.tool_extra(),
        )

    def _transport_for(self, backend: str) -> str:
        if backend == "docker" and not self.app.config.sandbox.allow_network_in_docker:
            return "mailbox"
        return "tcp"

    async def run(
        self,
        code: str,
        *,
        session_id: str | None = None,
        channel: str = "system",
        timeout_s: float | None = None,
        allowed_tools: Iterable[str] | None = None,
        on_output: OnOutput | None = None,
        read_only: bool = False,
    ) -> dict:
        """Run ``code`` and return a SandboxResult (docs/API.md section 11).

        ``allowed_tools``: tool names or plugin ids the script may call (None: every tool the policy
        allows). ``on_output(kind, text)`` receives stdout/stderr chunks as they arrive (sync or
        async). ``read_only`` limits tool calls to effective risk ``read``.
        """
        cfg = self.app.config.sandbox
        started = time.perf_counter()
        if not cfg.enabled:
            res = empty_result()
            res["error"] = "Running code is turned off. Turn on code execution in Settings to use it."
            return res
        if not (code or "").strip():
            res = empty_result()
            res["error"] = "There is no code to run."
            return res
        try:
            compile(code, SCRIPT_FILE, "exec")
        except SyntaxError as exc:
            res = empty_result()
            res["error"] = syntax_error_message(exc)
            res["stderr"] = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            return res
        except ValueError as exc:
            res = empty_result()
            res["error"] = f"The code contains characters Python cannot read: {exc}."
            return res

        backend, backend_error = await self.resolve_backend()
        if backend_error:
            res = empty_result(backend)
            res["error"] = backend_error
            return res
        timeout = float(timeout_s if timeout_s is not None else cfg.timeout_s)

        async with self._semaphore():
            res = await self._run_in_folder(
                code, backend=backend, timeout=timeout, session_id=session_id, channel=channel,
                allowed_tools=allowed_tools, on_output=on_output, read_only=read_only,
            )
        res["duration_ms"] = int((time.perf_counter() - started) * 1000)
        return res

    async def _run_in_folder(
        self,
        code: str,
        *,
        backend: str,
        timeout: float,
        session_id: str | None,
        channel: str,
        allowed_tools: Iterable[str] | None,
        on_output: OnOutput | None,
        read_only: bool,
    ) -> dict:
        cfg = self.app.config.sandbox
        run_id = new_id()
        run_dir = (paths.home() / "sandbox" / run_id).resolve()
        res = empty_result(backend)
        for sub in (TMP_DIR, HOME_DIR):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        (run_dir / SCRIPT_FILE).write_text(code, encoding="utf-8")
        (run_dir / RUNNER_FILE).write_text(RUNNER_SOURCE, encoding="utf-8")

        policy = BridgePolicy.build(
            approvals_mode=self.app.config.tools.approvals.mode,
            allowed_tools=allowed_tools,
            read_only=read_only,
            max_tool_calls=cfg.max_tool_calls,
        )
        ctx = self._tool_context(session_id, channel)
        transport = self._transport_for(backend)
        bind = "0.0.0.0" if backend == "docker" and sys.platform.startswith("linux") else "127.0.0.1"
        bridge = ToolBridge(self.app.registry, policy, ctx, run_dir=run_dir, transport=transport, host=bind)
        try:
            endpoint = await bridge.start()
            url = endpoint["url"]
            if url and backend == "docker":
                url = f"http://host.docker.internal:{bridge.port}/call"
            names = [t.name for t in self.app.registry.tools(include_hidden=True) if policy.is_available(t)]
            (run_dir / CLIENT_FILE).write_text(
                client_source(url=url, token=bridge.token, mailbox=endpoint["mailbox"], tool_names=names,
                              timeout_s=timeout),
                encoding="utf-8",
            )
            if backend == "docker":
                name = f"sentient-sbx-{run_id[:12]}"
                argv = backends.docker_argv(
                    run_dir, name=name, image=cfg.docker_image, memory_mb=cfg.docker_memory_mb,
                    cpus=cfg.docker_cpus, network=cfg.allow_network_in_docker,
                )
                outcome = await backends.run_command(
                    argv, cwd=run_dir, env=None, timeout_s=timeout, max_chars=cfg.max_output_chars,
                    on_chunk=on_output, extra_kill=lambda: backends.docker_kill(name),
                )
            else:
                outcome = await backends.run_command(
                    backends.python_argv(run_dir), cwd=run_dir, env=backends.scrubbed_env(run_dir),
                    timeout_s=timeout, max_chars=cfg.max_output_chars, on_chunk=on_output,
                )
            res["tool_calls"] = bridge.tool_calls
            await bridge.stop()

            res["stdout"] = outcome.stdout + (
                f"\n[output cut after {cfg.max_output_chars} characters]" if outcome.stdout_truncated else ""
            )
            res["stderr"] = outcome.stderr + (
                f"\n[errors cut after {cfg.max_output_chars} characters]" if outcome.stderr_truncated else ""
            )
            res["result"] = _read_json(run_dir / RESULT_FILE)
            error_info = _read_json(run_dir / ERROR_FILE)
            res["ok"] = outcome.spawn_error is None and not outcome.timed_out and outcome.exit_code == 0
            res["error"] = None if res["ok"] else self._friendly_error(outcome, error_info, backend, timeout)
            res["files_created"] = await asyncio.to_thread(self._collect_files, run_dir, run_id)
        finally:
            await bridge.stop()
            if not cfg.keep_workdirs:
                shutil.rmtree(run_dir, ignore_errors=True)
        return res

    # ------------------------------------------------------------------ results
    def _friendly_error(self, outcome: backends.Outcome, info: Any, backend: str, timeout: float) -> str:
        cfg = self.app.config.sandbox
        if outcome.spawn_error:
            what = "Docker" if backend == "docker" else "Python"
            return f"Could not start {what} to run the script: {outcome.spawn_error}"
        if outcome.timed_out:
            return (
                f"The script took longer than {timeout:g} seconds and was stopped. "
                "Try a smaller piece of work, or split it into steps."
            )
        if isinstance(info, dict) and info.get("type"):
            kind, message, line = info["type"], info.get("message") or "", info.get("line")
            where = f" on line {line}" if line else ""
            if kind == "ToolRefused":
                return f"A tool call{where} was refused: {message}"
            if kind == "ToolError":
                return f"A tool call{where} failed: {message}"
            if kind in {"ModuleNotFoundError", "ImportError"}:
                return (
                    f"The script{where} needs a package that is not installed here ({message}). "
                    "Use the Python standard library instead."
                )
            if kind == "MemoryError":
                return f"The script{where} ran out of memory."
            if kind == "KeyboardInterrupt":
                return "The script was interrupted."
            return f"The script stopped with an error{where}: {kind}: {message}"
        last = next((ln.strip() for ln in reversed(outcome.stderr.splitlines()) if ln.strip()), "")
        if backend == "docker":
            if outcome.exit_code == 137:
                return f"The script used more than {cfg.docker_memory_mb} MB of memory and was stopped."
            if outcome.exit_code in {125, 126, 127}:
                return f"Docker could not start the script: {last or 'unknown Docker error'}"
        msg = f"The script exited with code {outcome.exit_code}."
        return f"{msg} {last}" if last else msg

    def _collect_files(self, run_dir: Path, run_id: str) -> list[str]:
        cfg = self.app.config.sandbox
        files_root = paths.files_dir().resolve()
        out_root = files_root / "outputs" / run_id
        limit = cfg.max_file_mb * 1024 * 1024
        created: list[str] = []
        for p in sorted(run_dir.rglob("*")):
            rel = p.relative_to(run_dir)
            if rel.parts[0] in INTERNAL_NAMES or "__pycache__" in rel.parts:
                continue
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                if run_dir not in p.resolve().parents:
                    continue
                if p.stat().st_size > limit:
                    log.info("sandbox: not copying %s (larger than %s MB)", rel, cfg.max_file_mb)
                    continue
                if len(created) >= cfg.max_files:
                    log.info("sandbox: file limit reached, not copying the rest")
                    break
                dest = out_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                created.append(dest.relative_to(files_root).as_posix())
            except OSError as exc:
                log.warning("sandbox: could not copy %s: %s", rel, exc)
        return created


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
