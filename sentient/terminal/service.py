"""Commands on this computer (docs/API.md section 18, ADR 0019).

``await app.terminal.run(command, cwd, ctx)`` checks the command in code (turned on, asked for by the user, not on
the blocklist, starting inside an allowed folder), runs it as a fresh shell process with secrets removed from its
environment, streams its output as tool progress and returns a **TerminalResult**. The timeout, the Stop button on
the command (``stop_command``) and Stop everything (``halt``) kill the whole process tree, using the sandbox's process
runner.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.sandbox import backends
from sentient.services import Service
from sentient.store.db import new_id, now_iso
from sentient.terminal import guard
from sentient.tools.base import Risk, ToolContext, is_unprompted
from sentient.tools.rules import rule_for, unprompted_message

log = logging.getLogger(__name__)

PLUGIN_ID = "terminal"
TOOL_NAME = "terminal_run"
# Runs nobody can be asked in yet (task runs, helpers, start-up): only commands that never need asking run there.
UNATTENDED_CHANNELS = frozenset({"task", "subagent", "system"})
TARGET_CHARS = 120  # approval cards show at most this much of the folder (``describe_call``)
FULL_OUTPUT_CHARS = 2_000_000  # kept per stream for the saved file; beyond this the output is dropped
BLOCKED_KINDS = [
    "format or wipe a disk",
    "shut down, restart or sign out",
    "delete from the registry",
    "delete a whole drive, system folder or home folder",
    "delete backups or change how the computer starts",
]


@dataclass
class Check:
    command: str
    error: str | None = None
    folder: Path | None = None
    allow_listed: bool = False
    unprompted: bool = False


@dataclass
class _Running:
    id: str
    command: str
    cwd: str
    started_at: str
    task: asyncio.Task


@dataclass
class _Stream:
    """All output up to ``FULL_OUTPUT_CHARS`` (for the file) and how much of it was streamed live."""

    show: int
    parts: list[str] = field(default_factory=list)
    size: int = 0
    shown: int = 0

    def add(self, text: str) -> tuple[str, bool]:
        """Keep ``text``; return the part to stream live and whether the live limit was just reached."""
        self.parts.append(text)
        self.size += len(text)
        if self.shown >= self.show:
            return "", False
        part = text[: self.show - self.shown]
        self.shown += len(part)
        return part, self.shown >= self.show

    def text(self) -> str:
        return "".join(self.parts)


def _trim(text: str, cap: int) -> tuple[str, bool]:
    """At most ``cap`` characters: the start and the end (where errors usually are), with a note in between."""
    if len(text) <= cap:
        return text, False
    head = cap // 4
    tail = cap - head
    return f"{text[:head]}\n[... {len(text) - cap:,} characters cut here ...]\n{text[-tail:]}", True


class TerminalService(Service):
    name = "terminal"

    def __init__(self, app: Any):
        super().__init__(app)
        self._running: dict[str, _Running] = {}
        self._shell: tuple[str, str] | None = None
        self._shell_checked = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        from sentient.terminal.tool import TerminalPlugin

        if self.app.registry.plugin(PLUGIN_ID) is None:
            self.app.registry.register(TerminalPlugin())
        self._apply_enabled()
        self._loops.append(asyncio.create_task(self._watch_config(), name="terminal:config"))

    def _apply_enabled(self) -> None:
        if self.app.registry.plugin(PLUGIN_ID) is not None:
            self.app.registry.set_hidden(PLUGIN_ID, not self.app.config.terminal.enabled)

    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as queue:
            while True:
                event = await queue.get()
                if event.get("type") == "config.updated":
                    self._apply_enabled()

    async def halt(self) -> int:
        """Stop everything: kill every running command (chat replies are cancelled first, which kills theirs)."""
        running = [r.task for r in self._running.values() if not r.task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.wait(running, timeout=10)
        return len(running)

    # ------------------------------------------------------------------ checks
    def shell(self) -> tuple[str, str] | None:
        if not self._shell_checked:
            self._shell = guard.find_shell()
            self._shell_checked = True
        return self._shell

    def check(self, command: str, cwd: str | None, ctx: ToolContext | None) -> Check:
        """Every rule that does not depend on who can be asked, in order. Never runs anything."""
        cfg = self.app.config.terminal
        command = str(command or "").strip()
        chk = Check(command=command)
        if not cfg.enabled:
            chk.error = f"Running commands on this computer is turned off. The user can turn it on in {guard.SETTINGS}."
            return chk
        if is_unprompted(getattr(ctx, "origin", None)):
            chk.unprompted = True
            chk.error = unprompted_message("Running a command")
            return chk
        if not command:
            chk.error = "There is no command to run."
            return chk
        if len(command) > guard.MAX_COMMAND_CHARS or "\x00" in command:
            chk.error = f"The command is too long or has characters a shell can't take (at most {guard.MAX_COMMAND_CHARS} characters)."
            return chk
        what = guard.blocked_reason(command)
        if what:
            chk.error = (f"Sentient never runs commands that {what}, so this one was not run. "
                         "This is built in and can't be changed in Settings.")
            return chk
        folder = guard.resolve_folder(cwd, cfg.allowed_folders, cfg.default_folder)
        if folder.error:
            chk.error = folder.error
            return chk
        chk.folder = folder.path
        chk.allow_listed = guard.is_allow_listed(command, cfg.allowed_commands)
        return chk

    def risk_for(self, arguments: dict, ctx: ToolContext) -> Risk:
        """``exec``, except: a command that never needs asking is ``read``, and a call the terminal refuses anyway
        (turned off, blocked, outside the allowed folders) is ``read`` too, since it runs nothing and the user
        should not be asked to approve a refusal. Work nobody asked for stays ``exec`` so ADR 0017 refuses it."""
        chk = self.check(arguments.get("command") or "", arguments.get("cwd"), ctx)
        if chk.unprompted:
            return Risk.exec
        if chk.error or chk.allow_listed:
            return Risk.read
        return Risk.exec

    def describe(self, arguments: dict, ctx: ToolContext) -> dict:
        """Approval wording: the folder the command runs in (its end, when it is long; the card shows the command)."""
        chk = self.check(arguments.get("command") or "", arguments.get("cwd"), ctx)
        folder = str(chk.folder) if chk.folder else None
        if folder and len(folder) > TARGET_CHARS:
            folder = "..." + folder[-(TARGET_CHARS - 3):]
        return {"risk_label": "Runs a command", "target": folder}

    def _unattended_refusal(self, chk: Check, ctx: ToolContext | None) -> str | None:
        """Task runs and helpers can't stop to ask yet: only commands that never need asking run there."""
        if getattr(ctx, "channel", None) not in UNATTENDED_CHANNELS or chk.allow_listed:
            return None
        approvals = self.app.config.tools.approvals
        tool = self.app.registry.get(TOOL_NAME)
        if approvals.mode == "off" or (tool is not None and rule_for(approvals.rules, tool) == "allow"):
            return None
        return ("Commands on this computer need the user's yes, and tasks and helpers can't ask yet, so this was "
                "not run. Run it from a chat instead, or the user can add it to the commands that never need "
                f"asking in {guard.SETTINGS}.")

    # ------------------------------------------------------------------ running
    async def run(self, command: str, cwd: str | None, ctx: ToolContext) -> dict:
        """Check and run one command. Returns a TerminalResult (docs/API.md section 18)."""
        started = time.perf_counter()
        chk = self.check(command, cwd, ctx)
        shell = self.shell()
        refusal = chk.error or self._unattended_refusal(chk, ctx)
        if not refusal and shell is None:
            refusal = "No command shell was found on this computer, so the command could not run."
        result: dict[str, Any] = {
            "ok": False, "command": chk.command, "cwd": str(chk.folder) if chk.folder else None,
            "shell": shell[0] if shell else None, "exit_code": None, "stdout": "", "stderr": "",
            "timed_out": False, "stopped": False, "duration_ms": 0, "output_file": None, "error": refusal,
        }
        if refusal:
            return result
        assert shell is not None and chk.folder is not None
        cfg = self.app.config.terminal
        streams = {"stdout": _Stream(cfg.max_output_chars), "stderr": _Stream(cfg.max_output_chars)}
        progress = getattr(ctx, "progress", None)

        def on_chunk(kind: str, text: str) -> None:
            live, full = streams[kind].add(text)
            if callable(progress):
                if live:
                    progress({"kind": kind, "text": live})
                if full:
                    progress({"kind": "status", "text": "There is more output. All of it will be saved to a file."})

        provider_keys = [p.api_key_env for p in self.app.config.models.providers.values() if p.api_key_env]
        run_id = ctx.call_id or new_id()
        task = asyncio.create_task(
            backends.run_command(
                guard.shell_argv(shell, chk.command), cwd=chk.folder, env=guard.command_env(provider_keys),
                timeout_s=float(cfg.timeout_s), max_chars=FULL_OUTPUT_CHARS, on_chunk=on_chunk,
            ),
            name=f"terminal:{run_id}",
        )
        self._running[run_id] = _Running(run_id, chk.command, str(chk.folder), now_iso(), task)
        outcome: backends.Outcome | None = None
        try:
            outcome = await task
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if me is not None and me.cancelling():
                raise  # the reply or run this belongs to was stopped: the process tree is already gone
            result["stopped"] = True  # the command's own Stop button
        finally:
            self._running.pop(run_id, None)

        stdout, stderr = streams["stdout"].text(), streams["stderr"].text()
        result["stdout"], cut_out = _trim(stdout, cfg.max_output_chars)
        result["stderr"], cut_err = _trim(stderr, cfg.max_output_chars)
        if outcome is not None:
            result["exit_code"] = outcome.exit_code
            result["timed_out"] = outcome.timed_out
            if outcome.spawn_error:
                result["error"] = f"Could not start {shell[0]}: {outcome.spawn_error}"
            elif outcome.timed_out:
                result["error"] = (f"The command took longer than {cfg.timeout_s} seconds and was stopped. "
                                   f"The user can change the time limit in {guard.SETTINGS}.")
            cut_out = cut_out or outcome.stdout_truncated
            cut_err = cut_err or outcome.stderr_truncated
        else:
            result["error"] = "The command was stopped before it finished."
        result["ok"] = outcome is not None and not result["error"] and outcome.exit_code == 0
        if cut_out or cut_err:
            result["output_file"] = await asyncio.to_thread(self._save_output, run_id, chk, shell[0], stdout, stderr)
        result["duration_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    def _save_output(self, run_id: str, chk: Check, shell: str, stdout: str, stderr: str) -> str | None:
        files_root = paths.files_dir()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)[:80] or new_id()
        rel = f"outputs/terminal-{safe}.txt"
        body = f"$ {chk.command}\n(in {chk.folder}, {shell})\n\n--- output ---\n{stdout}\n--- errors ---\n{stderr}\n"
        try:
            path = files_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            log.warning("terminal: could not save long output: %s", exc)
            return None
        return rel

    # ------------------------------------------------------------------ control
    def stop_command(self, run_id: str) -> bool:
        """Kill one running command (its card's Stop button). The tool returns what it printed so far."""
        running = self._running.get(run_id)
        if running is None or running.task.done():
            return False
        running.task.cancel()
        return True

    def status(self) -> dict:
        cfg = self.app.config.terminal
        shell = self.shell()
        return {
            "enabled": cfg.enabled,
            "shell": shell[0] if shell else None,
            "shell_path": shell[1] if shell else None,
            "allowed_folders": list(cfg.allowed_folders),
            "default_folder": str(guard.resolve_folder(None, cfg.allowed_folders, cfg.default_folder).path or "") or None,
            "blocked": list(BLOCKED_KINDS),
            "running": [
                {"id": r.id, "command": r.command, "cwd": r.cwd, "started_at": r.started_at}
                for r in self._running.values() if not r.task.done()
            ],
        }
