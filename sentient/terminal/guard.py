"""Deterministic checks for host commands (ADR 0019). Pure functions, no model involved.

- ``blocked_reason(command)``: the built-in blocklist (formatting disks, shutting down, deleting the registry,
  deleting a whole drive or home folder). It cannot be turned off or overridden by an Allow rule. It is a seatbelt
  against slips and obvious injected commands, not a security boundary: approval is the real gate.
- ``is_allow_listed(command, prefixes)``: commands that never need asking (``git status``). Only a plain command
  matches: no chaining, redirection, substitution or line breaks.
- ``resolve_folder(...)``: where a command starts; it must be inside one of the allowed folders.
- ``command_env(...)``: the engine's environment without secrets (keys, tokens, passwords, Sentient's own token).
- ``shell_argv(command)``: PowerShell on Windows (pwsh when installed), else bash, zsh or sh.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
MAX_COMMAND_CHARS = 8_000

# ---------------------------------------------------------------------------- blocklist
# Matched anywhere in the command: names that are hard to type by accident in an ordinary command.
_ANYWHERE = [
    (r"\bformat-volume\b|\bclear-disk\b|\binitialize-disk\b|\bremove-partition\b|\bdiskpart\b", "format or wipe a disk"),
    (r"\bmkfs(\.\w+)?\b|\bwipefs\b|\bsfdisk\b|\bgdisk\b|\bfdisk\b", "format or wipe a disk"),
    (r"\bdiskutil\s+(erase\w*|partitiondisk|zerodisk|securerase|secureerase|reformat)\b", "format or wipe a disk"),
    (r"\bstop-computer\b|\brestart-computer\b", "shut down or restart the computer"),
    (r"\bvssadmin(\.exe)?\s+delete\b|\bwmic(\.exe)?\s+shadowcopy\s+delete\b|\bwbadmin(\.exe)?\s+delete\b",
     "delete backups"),
    (r"\bbcdedit\b", "change how the computer starts"),
    (r"\breg(\.exe)?\s+delete\b", "delete from the registry"),
    (r"\b(remove-item\w*|clear-item\w*|ri|rm|del|erase|rd|rmdir|rp|cli|clp)\b[^\n;&|]*"
     r"(\b(hklm|hkcu|hkcr|hku|hkcc):|\bregistry::)", "delete from the registry"),
    (r"--no-preserve-root", "delete everything on the drive"),
    (r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:", "start a fork bomb"),
    (r"\bdd\b[^\n;&|]*\bof=/dev/", "write over a disk"),
    (r">\s*/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|mmcblk\d)", "write over a disk"),
    (r"\bcipher(\.exe)?\s+/w\b", "wipe free disk space"),
]

# Matched where a command starts (the beginning, after ; & | ( { ` or a wrapper such as sudo or bash -c), so
# ordinary words like "shutdown" in a commit message are not refused.
_AT_COMMAND = [
    (r"(shutdown|reboot|poweroff|halt|logoff|logout)(\.exe)?\b", "shut down, restart or sign out"),
    (r"init\s+[06]\b|telinit\s+[06]\b", "shut down or restart the computer"),
    (r"systemctl\s+(poweroff|reboot|halt|suspend|hibernate|kexec|emergency|rescue)\b",
     "shut down or restart the computer"),
    (r"format(\.com|\.exe)?\s+[a-z]:", "format a disk"),
]

_WRAPPER = re.compile(
    r"^(?:sudo|doas|nohup|exec|time|command|builtin|call|iex|invoke-expression|start-process|"
    r"env(?:\s+-\S+)*(?:\s+\w+=\S*)*|start(?:\s+/\w+)*|"
    r"cmd(?:\.exe)?(?:\s+/\w)*?\s+/[ck]|(?:ba|z|da|k)?sh(?:\s+-\w+)*?\s+-c|"
    r"(?:powershell|pwsh)(?:\.exe)?(?:\s+-\w+)*?\s+-c(?:ommand)?)\s+",
)
_DELETE_VERBS = {"rm", "del", "erase", "rd", "rmdir", "ri", "remove-item", "rimraf"}
_OWNER_VERBS = {"chmod", "chown", "chgrp", "icacls", "takeown"}
_PROTECTED_POSIX = re.compile(
    r"^/(?:bin|boot|dev|etc|home|lib|lib64|opt|proc|root|sbin|srv|sys|usr|var|system|users|applications|library|"
    r"private|volumes|home/[^/]+|users/[^/]+)?/?\*?$"
)
_PROTECTED_WINDOWS = re.compile(
    r"^[a-z]:[\\/]?(?:\*|windows|users|users[\\/][^\\/]+|program files(?: \(x86\))?|programdata)?[\\/]?\*?$"
)
_PROTECTED_HOME = {
    "~", "~/", "~/*", "~\\", "~\\*", "$home", "${home}", "$home/", "$home/*", "$env:userprofile", "%userprofile%",
    "$env:systemdrive", "%systemdrive%", "$env:systemroot", "%systemroot%", "$env:windir", "%windir%", "/*",
}


def _tokens(segment: str) -> list[str]:
    return [t.strip("\"'") for t in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', segment)]


def _split(command: str) -> list[str]:
    """Split at ; & | ( ) { } and line breaks outside quotes, and at $( and ` everywhere but single quotes (shells
    run those inside double quotes too)."""
    parts: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = ""
            cur.append(ch)
        elif ch == "`" or (ch == "$" and command[i + 1:i + 2] == "("):
            parts.append("".join(cur))
            cur = []
            i += 1 if ch == "`" else 2
            continue
        elif quote == '"':
            if ch == '"':
                quote = ""
            cur.append(ch)
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch in ";&|(){}\n\r":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _segments(command: str, depth: int = 0) -> list[str]:
    """Each command in ``command``, with wrappers such as ``sudo`` or ``bash -c "..."`` peeled off."""
    out = []
    for raw in _split(command):
        seg = raw.strip()
        peeled = False
        for _ in range(4):  # sudo bash -c "rm -rf /" peels in a few steps
            m = _WRAPPER.match(seg)
            if not m:
                break
            seg = seg[m.end():].strip()
            peeled = True
        if peeled and seg[:1] in {"'", '"'} and depth < 3:
            out.extend(_segments(seg.strip("\"'"), depth + 1))  # the quoted command a wrapper runs
        elif seg:
            out.append(" ".join(_tokens(seg)))
    return out


def _protected(target: str) -> bool:
    t = target.strip().lower().rstrip()
    if not t:
        return False
    if t in _PROTECTED_HOME:
        return True
    if _PROTECTED_WINDOWS.match(t):
        return True
    return bool(_PROTECTED_POSIX.match(t))


def _is_flag(token: str) -> bool:
    return token.startswith("-") or bool(re.fullmatch(r"/[a-z?]", token.lower()))  # -rf, --recurse, /s


def _wide_delete(tokens: list[str]) -> str | None:
    """Deleting (or changing the owner of) a whole drive, a system folder or a home folder."""
    if not tokens:
        return None
    verb = tokens[0].lower().removesuffix(".exe")
    flags = [t.lower() for t in tokens[1:] if _is_flag(t)]
    targets = [t for t in tokens[1:] if not _is_flag(t)]
    if not any(_protected(t) for t in targets):
        return None
    if verb in _DELETE_VERBS:
        # PowerShell's Remove-Item and cmd's del and rd delete a folder's contents even without a flag
        recursive = verb not in {"rm", "rimraf"} or any(
            f in {"--recursive", "/s"} or f.startswith("-rec") or re.fullmatch(r"-[a-z]*r[a-z]*", f) for f in flags
        )
        if recursive or verb == "rimraf":
            return "delete a whole drive, system folder or home folder"
    if verb in _OWNER_VERBS and any(f in {"--recursive", "/t"} or re.fullmatch(r"-[a-z]*r[a-z]*", f) for f in flags):
        return "change permissions on a whole drive, system folder or home folder"
    return None


def blocked_reason(command: str) -> str | None:
    """Plain words for what the command would do when it is on the built-in blocklist, else None."""
    text = (command or "").lower()
    for pattern, what in _ANYWHERE:
        if re.search(pattern, text):
            return what
    for seg in _segments(text):
        for pattern, what in _AT_COMMAND:
            if re.match(pattern, seg):
                return what
        what = _wide_delete(_tokens(seg))
        if what:
            return what
    return None


# ---------------------------------------------------------------------------- allow-list
# --ext-diff and --textconv make git run a program from its config
_NOT_PLAIN = re.compile(r"[;&|<>`$(){}\n\r]|(?:^|\s)--?(?:output|exec|ext-diff|textconv)\b", re.IGNORECASE)


def _norm(text: str) -> str:
    text = " ".join(text.split())
    return text.lower() if IS_WINDOWS else text


def is_allow_listed(command: str, prefixes: Iterable[str]) -> bool:
    """True when ``command`` is exactly one of ``prefixes`` or starts with one followed by a space, and is a plain
    command (no chaining, redirection, substitution, ``--output``, ``--ext-diff``, ``--textconv`` or line breaks).
    Case-insensitive on Windows."""
    if not command or _NOT_PLAIN.search(command):
        return False
    cmd = _norm(command)
    for raw in prefixes:
        prefix = _norm(str(raw or ""))
        if prefix and (cmd == prefix or cmd.startswith(prefix + " ")):
            return True
    return False


# ---------------------------------------------------------------------------- folders
@dataclass
class Folder:
    path: Path | None
    error: str | None = None


def _key(p: Path) -> str:
    return os.path.normcase(str(p))


def _inside(child: Path, parent: Path) -> bool:
    c, p = _key(child), _key(parent)
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


def _resolve(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw.strip()))).resolve()


def allowed_roots(folders: Iterable[str]) -> list[Path]:
    out = []
    for raw in folders:
        if str(raw or "").strip():
            try:
                out.append(_resolve(str(raw)))
            except (OSError, RuntimeError, ValueError):
                continue
    return out


SETTINGS = "Settings > Terminal"


def resolve_folder(cwd: str | None, allowed: Iterable[str], default: str = "") -> Folder:
    """The folder a command starts in. ``cwd`` may be absolute or relative to the default folder. It must exist and
    be inside an allowed folder (after following links), else ``error`` says why in plain words."""
    roots = allowed_roots(allowed)
    if not roots:
        return Folder(None, f"No folders are allowed yet. Add a folder in {SETTINGS} first.")
    try:
        base = _resolve(default) if str(default or "").strip() else roots[0]
    except (OSError, RuntimeError, ValueError):
        return Folder(None, f"The default folder in {SETTINGS} can't be read.")
    if not any(_inside(base, r) for r in roots):
        return Folder(None, f"The default folder isn't one of the allowed folders. Fix it in {SETTINGS}.")
    want = str(cwd or "").strip()
    try:
        if not want:
            target = base
        elif Path(os.path.expandvars(os.path.expanduser(want))).is_absolute():
            target = _resolve(want)
        else:
            target = (base / want).resolve()
    except (OSError, RuntimeError, ValueError):
        return Folder(None, f"The folder {want} can't be read.")
    if not any(_inside(target, r) for r in roots):
        return Folder(None, f"{target} isn't inside a folder you allowed, so the command was not run. "
                            f"You can add folders in {SETTINGS}.")
    if not target.is_dir():
        return Folder(None, f"The folder {target} doesn't exist.")
    return Folder(target)


# ---------------------------------------------------------------------------- environment
_SECRET_PARTS = {"KEY", "KEYS", "APIKEY", "AUTH", "PAT", "PASS", "COOKIE", "PRIVATE", "CREDS", "DSN"}
_SECRET_WORDS = ("PASSWORD", "PASSWD", "SECRET", "TOKEN", "CREDENTIAL", "API_KEY", "APIKEY", "PASSPHRASE")
_SECRET_NAMES = {"DATABASE_URL", "SENTRY_DSN", "NETRC"}
_KEEP = {"SSH_AUTH_SOCK", "SSH_AGENT_PID", "GPG_AGENT_INFO", "XAUTHORITY"}


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    if upper in _KEEP or re.fullmatch(r"GIT_CONFIG_KEY_\d+", upper):  # git setting names, not keys
        return False
    if upper.startswith(("SENTIENT_", "LITELLM_")) or upper in _SECRET_NAMES:
        return True
    if any(w in upper for w in _SECRET_WORDS):
        return True
    return any(part in _SECRET_PARTS for part in re.split(r"[_\-.]", upper))


RUN_MARKER = "SENTIENT_TERMINAL_RUN"
# For commands that run without asking: a repository's own config must not make git start a program. core.fsmonitor
# is the one plain git status and git diff run; external diff and textconv need flags the list refuses (or a
# diff.external that only someone who could already run commands can set).
_GIT_SAFE_CONFIG = (("core.fsmonitor", "false"),)


def command_env(extra_secret_names: Iterable[str] = (), *, run_id: str = "", listed: bool = False) -> dict[str, str]:
    """The engine's environment minus anything that looks like a key, token or password, Sentient's own
    variables and the provider key variables named in the config. Keychain secrets are never added. ``run_id``
    marks every process of the run (so ``kill_marked`` finds the ones that left its process group); ``listed``
    adds git settings that stop a repository's config from starting programs."""
    drop = {str(n).upper() for n in extra_secret_names if n}
    env = {k: v for k, v in os.environ.items() if not is_secret_name(k) and k.upper() not in drop}
    env["GIT_TERMINAL_PROMPT"] = "0"  # git asks for a password on a terminal nobody can type into: fail instead
    if run_id:
        env[RUN_MARKER] = run_id
    if listed:
        try:
            count = int(env.get("GIT_CONFIG_COUNT") or 0)
        except ValueError:
            count = 0
        for key, value in _GIT_SAFE_CONFIG:
            env[f"GIT_CONFIG_KEY_{count}"] = key
            env[f"GIT_CONFIG_VALUE_{count}"] = value
            count += 1
        env["GIT_CONFIG_COUNT"] = str(count)
    return env


def _marked_pids(marker: str) -> list[int]:
    """Processes of this user whose environment carries ``marker`` (Linux: /proc; macOS: ps -E)."""
    needle = marker.encode()
    pids: list[int] = []
    me = os.getpid()
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) == me:
                continue
            try:
                if needle in (entry / "environ").read_bytes().split(b"\0"):
                    pids.append(int(entry.name))
            except OSError:
                continue
        return pids
    try:
        out = subprocess.run(["ps", "-A", "-E", "-ww", "-o", "pid=,command="], capture_output=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return pids
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != me and needle in parts[1].split():
            pids.append(int(parts[0]))
    return pids


def kill_marked(run_id: str) -> int:
    """Kill every process started by run ``run_id``, including ones that left its process group (``setsid``) or
    were reparented. Windows needs nothing here: the Job Object holds the whole tree. Returns how many were killed."""
    if IS_WINDOWS or not run_id:
        return 0
    import signal

    killed = 0
    for pid in _marked_pids(f"{RUN_MARKER}={run_id}"):
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return killed


# ---------------------------------------------------------------------------- shell
_PS_PREAMBLE = (
    "$ProgressPreference = 'SilentlyContinue'; "
    "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}; $global:LASTEXITCODE = 0\n"
)
_PS_EPILOGUE = (
    "\n$__ok = $?; if ($global:LASTEXITCODE) { exit $global:LASTEXITCODE }; if (-not $__ok) { exit 1 }; exit 0"
)


def find_shell() -> tuple[str, str] | None:
    """``(name, path)`` of the shell commands run in, or None when none is installed."""
    if IS_WINDOWS:
        for name in ("pwsh", "powershell"):
            path = shutil.which(name)
            if path:
                return name, path
        fallback = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        return ("powershell", str(fallback)) if fallback.is_file() else None
    preferred = Path(os.environ.get("SHELL", "")).name
    for name in ([preferred] if preferred in {"bash", "zsh"} else []) + ["bash", "zsh", "sh"]:
        path = shutil.which(name)
        if path:
            return name, path
    return None


def shell_argv(shell: tuple[str, str], command: str) -> list[str]:
    name, path = shell
    if name in {"pwsh", "powershell"}:
        # exit with the native command's exit code, or 1 when the last PowerShell command failed
        script = _PS_PREAMBLE + command + _PS_EPILOGUE
        return [path, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]
    return [path, "-c", command]
