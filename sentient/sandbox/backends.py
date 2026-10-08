"""Where a script actually runs: a local process or a Docker container.

Both backends start one child process (``python`` or ``docker run``), stream its stdout and
stderr through ``on_chunk`` with size caps, and stop it at the deadline. Pipes are read on
threads so this works on any asyncio event loop (the Windows selector loop cannot spawn
asyncio subprocesses). On Windows the child is placed in a Job Object so the whole process tree
(including the venv launcher's real interpreter and anything the script spawned) dies together;
on POSIX the child leads its own process group, which is killed as a unit.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import inspect
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentient.sandbox.runtime import HOME_DIR, RUNNER_FILE, TMP_DIR

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
EXIT_GRACE_S = 1.0     # after the main process exits, wait this long for pipes held by stragglers
DRAIN_AFTER_KILL_S = 3.0

OnChunk = Callable[[str, str], Any]

# Variables a Python interpreter (and the Windows venv launcher) needs. Everything else, API keys
# included, is dropped.
_KEEP_ENV = {
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "OS",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
}


@dataclass
class Outcome:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    spawn_error: str | None = None


# ---------------------------------------------------------------------------- environment
def _interpreter() -> tuple[str, dict[str, str]]:
    """The engine's Python. In a Windows venv, ``python.exe`` is a launcher that starts the real
    interpreter as a second process; run the base interpreter directly and tell it about the venv
    with ``__PYVENV_LAUNCHER__`` (what the launcher does) so there is one process to contain."""
    base = getattr(sys, "_base_executable", None)
    if IS_WINDOWS and base and Path(base).is_file() and Path(base) != Path(sys.executable) and sys.prefix != sys.base_prefix:
        return base, {"__PYVENV_LAUNCHER__": sys.executable}
    return sys.executable, {}


def scrubbed_env(run_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in _KEEP_ENV}
    env.update(_interpreter()[1])
    tmp = str(run_dir / TMP_DIR)
    home = str(run_dir / HOME_DIR)
    env.update({
        "TEMP": tmp, "TMP": tmp, "TMPDIR": tmp,
        "HOME": home, "USERPROFILE": home,
        "MPLCONFIGDIR": str(run_dir / TMP_DIR / "matplotlib"),
        "MPLBACKEND": "Agg",
        "SENTIENT_SANDBOX": "1",
    })
    return env


def python_argv(run_dir: Path) -> list[str]:
    # -I: ignore PYTHON* variables and the user site; -u: unbuffered so output streams live
    return [_interpreter()[0], "-I", "-u", "-X", "utf8", str(run_dir / RUNNER_FILE)]


def docker_argv(
    run_dir: Path,
    *,
    name: str,
    image: str,
    memory_mb: int,
    cpus: float,
    network: bool,
) -> list[str]:
    argv = [
        "docker", "run", "--rm", "--name", name,
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--cpus", f"{cpus:g}", "--pids-limit", "256",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
        "-v", f"{run_dir}:/work", "-w", "/work",
        "-e", f"HOME=/work/{HOME_DIR}", "-e", f"TMPDIR=/work/{TMP_DIR}",
        "-e", f"MPLCONFIGDIR=/work/{TMP_DIR}/matplotlib", "-e", "MPLBACKEND=Agg", "-e", "SENTIENT_SANDBOX=1",
    ]
    if network:
        argv += ["--add-host", "host.docker.internal:host-gateway"]
    else:
        argv += ["--network", "none"]
    if not IS_WINDOWS and hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]  # files in the mount stay the user's
    argv += [image, "python", "-I", "-u", "-X", "utf8", f"/work/{RUNNER_FILE}"]
    return argv


# ---------------------------------------------------------------------------- docker detection
_docker_cache: tuple[float, bool] | None = None
DOCKER_CACHE_S = 30.0


def docker_available_sync() -> bool:
    global _docker_cache
    now = time.monotonic()
    if _docker_cache is not None and now - _docker_cache[0] < DOCKER_CACHE_S:
        return _docker_cache[1]
    ok = False
    if shutil.which("docker"):
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.OSType}}"],
                capture_output=True, timeout=8, creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
            # the sandbox image is Linux; Docker in Windows-container mode cannot run it
            ok = proc.returncode == 0 and proc.stdout.strip().lower() == b"linux"

        except (OSError, subprocess.SubprocessError):
            ok = False
    _docker_cache = (now, ok)
    return ok


async def docker_available() -> bool:
    return await asyncio.to_thread(docker_available_sync)


def docker_kill(name: str) -> None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["docker", "kill", name], capture_output=True, timeout=15,
            creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )


# ---------------------------------------------------------------------------- process tree control
class _WindowsJob:
    """A kill-on-close Job Object holding the child and everything it starts."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32 = k32

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> bool:
        handle = int(proc._handle)  # type: ignore[attr-defined]
        return bool(self._k32.AssignProcessToJobObject(self.handle, handle))

    def terminate(self) -> None:
        if self.handle:
            self._k32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def new_job() -> _WindowsJob | None:
    """Create the job before spawning so the child is assigned as early as possible."""
    if not IS_WINDOWS:
        return None
    try:
        return _WindowsJob()
    except OSError as exc:
        log.warning("could not create a job object for the sandbox: %s", exc)
        return None


class ProcessTree:
    def __init__(
        self,
        proc: subprocess.Popen,
        extra_kill: Callable[[], None] | None = None,
        job: _WindowsJob | None = None,
    ):
        self.proc = proc
        self.extra_kill = extra_kill
        self.job: _WindowsJob | None = None
        if job is not None:
            if job.assign(proc):
                self.job = job
            else:
                job.close()

    def kill(self) -> None:
        """Stop the child and everything it started. Safe to call more than once."""
        if self.extra_kill is not None:
            self.extra_kill()
        if IS_WINDOWS:
            if self.proc.poll() is None:
                # taskkill walks the parent/child tree while the root is still alive; the job then
                # catches anything that was reparented or escaped the walk
                taskkill = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(
                        [str(taskkill), "/F", "/T", "/PID", str(self.proc.pid)],
                        capture_output=True, timeout=10, creationflags=CREATE_NO_WINDOW,
                    )
            if self.job is not None:
                self.job.terminate()
        else:
            import signal

            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            self.proc.kill()

    def cleanup(self) -> None:
        """After the run: kill stragglers the script left behind and release the job."""
        if IS_WINDOWS:
            if self.job is not None:
                self.job.terminate()
                self.job.close()
        else:
            import signal

            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(self.proc.pid, signal.SIGKILL)


# ---------------------------------------------------------------------------- runner
class _Capture:
    def __init__(self, cap: int):
        self.cap = cap
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def add(self, text: str) -> str:
        """Keep up to ``cap`` characters; returns the part that was kept."""
        if self.truncated:
            return ""
        room = self.cap - self.size
        if len(text) > room:
            text = text[:room]
            self.truncated = True
        self.parts.append(text)
        self.size += len(text)
        return text

    def text(self) -> str:
        return "".join(self.parts)


async def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout_s: float,
    max_chars: int,
    on_chunk: OnChunk | None = None,
    extra_kill: Callable[[], None] | None = None,
) -> Outcome:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    job = new_job()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs,
        )
    except OSError as exc:
        if job is not None:
            job.close()
        return Outcome(None, False, "", "", spawn_error=str(exc))
    tree = ProcessTree(proc, extra_kill, job)

    def pump(stream: Any, kind: str) -> None:
        # universal newlines so Windows "\r\n" reaches the UI and the model as "\n"
        decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")("replace"), translate=True)
        try:
            while True:
                data = stream.read1(8192)
                if not data:
                    break
                text = decoder.decode(data)
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, text))
            tail = decoder.decode(b"", final=True)
            if tail:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, tail))
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(queue.put_nowait, (kind, None))

    for stream, kind in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        threading.Thread(target=pump, args=(stream, kind), daemon=True, name=f"sandbox-{kind}").start()

    captures = {"stdout": _Capture(max_chars), "stderr": _Capture(max_chars)}
    open_streams = 2
    deadline = loop.time() + timeout_s
    timed_out = False
    killed_at: float | None = None
    exited_at: float | None = None
    try:
        while open_streams:
            now = loop.time()
            if not timed_out and now >= deadline:
                timed_out = True
                killed_at = now
                await asyncio.to_thread(tree.kill)
            if killed_at is not None and now - killed_at > DRAIN_AFTER_KILL_S:
                break
            if exited_at is None and proc.poll() is not None:
                exited_at = now
            if exited_at is not None and killed_at is None and now - exited_at > EXIT_GRACE_S:
                # the script finished but something it started still holds the pipes
                killed_at = now
                await asyncio.to_thread(tree.kill)
            wait = 0.25 if timed_out else max(0.01, min(0.25, deadline - now))
            try:
                kind, text = await asyncio.wait_for(queue.get(), timeout=wait)
            except TimeoutError:
                continue
            if text is None:
                open_streams -= 1
                continue
            kept = captures[kind].add(text)
            if kept and on_chunk is not None:
                try:
                    maybe = on_chunk(kind, kept)
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception:
                    log.debug("sandbox output callback failed", exc_info=True)
        try:
            exit_code = await asyncio.wait_for(asyncio.to_thread(proc.wait), 10)
        except TimeoutError:
            await asyncio.to_thread(tree.kill)
            exit_code = proc.poll()
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.to_thread(tree.kill))
        raise
    finally:
        # Pipes are not closed here: a reader thread may still hold the stream's lock. They close
        # when the killed tree releases its handles and the threads finish.
        await asyncio.to_thread(tree.cleanup)
    out, err = captures["stdout"], captures["stderr"]
    return Outcome(
        exit_code=exit_code, timed_out=timed_out, stdout=out.text(), stderr=err.text(),
        stdout_truncated=out.truncated, stderr_truncated=err.truncated,
    )


def python_version() -> str:
    return platform.python_version()
