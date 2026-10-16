"""Skill checks (#386): a skill can say how to check that it works, and Sentient runs that before it proposes a
self-evolved skill and again before it saves one. ADR 0010 (amendment) has the format and the reasons.

In the SKILL.md header, ``validate`` holds one check or a list of up to five::

    validate:
      - file: notes/weekly-review.md   # this file must exist (in the skill's folder, or Sentient's files folder)
        expect: "## Next week"         # optional: text the file must contain
      - run: |                          # Python, run in the code sandbox like execute_code (section 11)
          print(sum([2, 3]))
        expect: "5"                     # optional: text its output must contain
        timeout_s: 30                   # optional, 5 to 120 seconds

It is the "how to validate" field of the six-field template (#186) in a form code can run. A ``file`` check only
reads. A ``run`` check runs code, so it follows the code rules: work Sentient started itself never runs code (ADR
0017), so a self-evolved skill's code check waits and runs when the user approves the skill (that click is the yes,
ADR 0012); a "Never" rule, a Read only code connection or code execution turned off means it doesn't run at all; and a
skill downloaded from outside (ADR 0027) never has its code check run.

A **report** is ``{status, results: [{kind, target, ok, detail}], checked_at}``. ``status``: ``passed`` (every check
ran and passed), ``failed`` (one failed: the skill isn't proposed, or isn't saved), ``waiting`` (the file checks
passed and a code check runs when the user approves) or ``not_run`` (a code check can't run here, ``detail`` says
why). ``ok`` is None for a check that didn't run.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.store.db import now_iso
from sentient.tools.base import Risk

log = logging.getLogger(__name__)

MAX_CHECKS = 5
MAX_CODE = 4000
MAX_EXPECT = 300
DEFAULT_TIMEOUT_S = 30
MIN_TIMEOUT_S, MAX_TIMEOUT_S = 5, 120
MAX_READ = 1_000_000
CODE_TOOL = "execute_code"
WAITING = "It runs code, so Sentient runs it when you approve the skill."


def parse(value: Any) -> tuple[list[dict], list[str]]:
    """The checks a ``validate`` header declares, cleaned, and plain sentences for what's wrong with it. Unknown keys
    are ignored; a check with neither ``file`` nor ``run`` (or both) is dropped with a problem."""
    if value in (None, "", []):
        return [], []
    items = value if isinstance(value, list) else [value]
    checks: list[dict] = []
    problems: list[str] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append(f"Check {i} should be a file: or run: entry.")
            continue
        file, run = item.get("file"), item.get("run")
        if bool(file) == bool(run):
            problems.append(f"Check {i} needs exactly one of file: (a file that must exist) or run: (Python to run).")
            continue
        check: dict[str, Any] = {}
        if file:
            path = str(file).strip().replace("\\", "/")
            if path.startswith(("/", "~")) or ":" in path or ".." in path.split("/"):
                problems.append(f"Check {i}: the file must be inside the skill's folder or Sentient's files folder.")
                continue
            check["file"] = path
        else:
            code = str(run)
            if len(code) > MAX_CODE:
                problems.append(f"Check {i}: the code is too long to be a quick check (at most {MAX_CODE} characters).")
                continue
            check["run"] = code
            try:
                timeout = int(item.get("timeout_s") or DEFAULT_TIMEOUT_S)
            except (TypeError, ValueError):
                timeout = DEFAULT_TIMEOUT_S
            check["timeout_s"] = max(MIN_TIMEOUT_S, min(timeout, MAX_TIMEOUT_S))
        expect = item.get("expect")
        if expect not in (None, ""):
            check["expect"] = str(expect)[:MAX_EXPECT]
        checks.append(check)
    if len(checks) > MAX_CHECKS:
        problems.append(f"Only the first {MAX_CHECKS} checks are used.")
        checks = checks[:MAX_CHECKS]
    return checks, problems


def has_code(checks: list[dict]) -> bool:
    return any("run" in c for c in checks or [])


def _resolve(rel: str, folders: list[Path]) -> Path | None:
    """The first existing file named ``rel`` inside one of ``folders`` (never outside them)."""
    for base in folders:
        try:
            root = base.resolve()
            path = (root / rel).resolve()
        except OSError:
            continue
        if path.is_relative_to(root) and path.is_file():
            return path
    return None


def _file_check(check: dict, folders: list[Path]) -> dict:
    out = {"kind": "file", "target": check["file"], "ok": False, "detail": ""}
    path = _resolve(check["file"], folders)
    if path is None:
        out["detail"] = f"{check['file']} doesn't exist"
        return out
    want = check.get("expect")
    if want:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_READ)
        except OSError as exc:
            out["detail"] = f"{check['file']} can't be read ({exc.strerror or exc})"
            return out
        if path.name == "SKILL.md" and text.startswith("---"):  # its steps: the header holds the expected text itself
            text = text.split("\n---", 1)[-1]
        if want not in text:
            out["detail"] = f"{check['file']} doesn't contain \"{want}\""
            return out
    out["ok"] = True
    return out


def code_blocked(app: Any) -> str | None:
    """Why a code check can't run at all with the user's settings, or None when it may (with the user's yes)."""
    sandbox_cfg = getattr(app.config, "sandbox", None)
    if getattr(app, "sandbox", None) is None or not getattr(sandbox_cfg, "enabled", True):
        return "Running code is turned off, so this check can't run."
    tool = app.registry.get(CODE_TOOL) if getattr(app, "registry", None) is not None else None
    if tool is None:
        return "Running code isn't available, so this check can't run."
    approvals = app.approvals
    if approvals.is_never(tool) or approvals.read_only_refusal(tool, Risk.exec, app.registry):
        return "You set Sentient never to run code, so this check can't run."
    return None


async def _code_check(app: Any, check: dict, skill: str) -> dict:
    out = {"kind": "run", "target": _first_line(check["run"]), "ok": False, "detail": ""}
    ctx = app.agent.tool_context(None, "skills")
    res = await app.sandbox.run(
        check["run"], channel="skills", timeout_s=check.get("timeout_s", DEFAULT_TIMEOUT_S), read_only=True,
        ctx=ctx, purpose=f"Check that the skill {skill} works",
    )
    printed = str(res.get("stdout") or "")
    if res.get("result") is not None:
        printed += "\n" + json.dumps(res.get("result"), ensure_ascii=False, default=str)
    if not res.get("ok"):
        why = str(res.get("error") or "").strip() or _last_line(str(res.get("stderr") or "")) or "it failed"
        out["detail"] = f"its code failed: {why.rstrip('.')}"
        return out
    want = check.get("expect")
    if want and want not in printed:
        shown = _last_line(printed) or "nothing"
        out["detail"] = f"it printed {shown!r}, not \"{want}\""
        return out
    out["ok"] = True
    return out


def _first_line(code: str) -> str:
    return next((ln.strip() for ln in code.splitlines() if ln.strip()), "")[:80]


def _last_line(text: str) -> str:
    return next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")[:200]


def merge(previous: list[dict], proposed: list[dict]) -> list[dict]:
    """A skill's checks after a change: every check it had, then the new ones (a repair or merge never drops one)."""
    out = list(previous or [])
    for check in proposed or []:
        if check not in out:  # one at a time, so a repeated new check never crowds out another
            out.append(check)
    return out[:MAX_CHECKS]


async def run(
    app: Any, skill: str, checks: list[dict], folders: list[Path], *, run_code: bool, outside: bool = False,
    candidate: str | None = None,
) -> dict | None:
    """Run a skill's checks and return the report, or None when it declares none. ``run_code``: the user just
    approved the skill (their yes for its code check); False for work Sentient started itself, where a code check
    waits. ``outside``: the skill came from outside (ADR 0027), so its code check never runs. File checks look in
    ``folders`` and then Sentient's files folder; ``candidate`` is the SKILL.md about to be saved, which a check of
    ``SKILL.md`` reads instead of the one on disk. Never raises: a check that breaks counts as failed."""
    if not checks:
        return None
    with tempfile.TemporaryDirectory(prefix="skill-check-") as tmp:
        if candidate is not None:
            (Path(tmp) / "SKILL.md").write_text(candidate, encoding="utf-8")
            folders = [Path(tmp), *folders]
        return await _run(app, skill, checks, [*folders, paths.files_dir()], run_code=run_code, outside=outside)


async def _run(app: Any, skill: str, checks: list[dict], folders: list[Path], *, run_code: bool,
               outside: bool) -> dict:
    results: list[dict] = []
    status = "passed"
    for check in checks:
        if "file" in check:
            result = _file_check(check, folders)
        elif outside:
            result = {"kind": "run", "target": _first_line(check["run"]), "ok": None,
                      "detail": "Sentient doesn't run code from skills it downloaded."}
        elif (blocked := code_blocked(app)) is not None:
            result = {"kind": "run", "target": _first_line(check["run"]), "ok": None, "detail": blocked}
        elif not run_code:
            result = {"kind": "run", "target": _first_line(check["run"]), "ok": None, "detail": WAITING}
        else:
            try:
                result = await _code_check(app, check, skill)
            except Exception as exc:  # a check that breaks is a check that didn't pass
                log.warning("skill check for %s failed to run: %s", skill, exc)
                result = {"kind": "run", "target": _first_line(check["run"]), "ok": False,
                          "detail": f"the check couldn't run ({type(exc).__name__})"}
        results.append(result)
        if result["ok"] is False:
            status = "failed"
            break  # the first failure decides; later checks may depend on it
        if result["ok"] is None and status == "passed":
            status = "waiting" if result["detail"] == WAITING else "not_run"
    return {"status": status, "results": results, "checked_at": now_iso()}


def failure(report: dict | None) -> str:
    """The plain reason a report failed ("notes/x.md doesn't exist"), or ""."""
    if not report or report.get("status") != "failed":
        return ""
    return next((r["detail"] for r in report.get("results") or [] if r.get("ok") is False), "")
