"""Safety check for skills from outside (#146): Hermes imports and skills added from a web address.

Before such a skill can be used it waits in pending with a plain report. The check is deterministic: the safety rules
of the linter (``hidden``, ``danger``, ``outbound``) run on the SKILL.md and on every other text file in its folder, and
any script or program it brings is listed. A ``high`` verdict means "decline unless you trust where it came from":
approving it then needs an explicit "approve anyway" (``accept_risk``).

The ``skill_imports`` table is the lock record: where the skill came from, the hash of its folder, the scanner version
and the verdict when it was imported, and the hash the user approved. The report is computed again from the files on
disk every time it is shown or approved, so an edit after import is checked too.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from sentient.skills.lint import SECURITY_RULES, lint_file, security_findings
from sentient.store.db import now_iso

SCANNER_VERSION = "1"
MAX_SCAN_BYTES = 512 * 1024
MAX_FETCH_BYTES = 256 * 1024
FETCH_TIMEOUT_S = 20.0
_TEXT = {
    ".md", ".markdown", ".txt", ".rst", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd", ".py", ".js",
    ".mjs", ".cjs", ".ts", ".rb", ".pl", ".php", ".lua", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".html",
    ".htm", ".xml", ".csv", ".applescript", ".scpt", ".vbs",
}
_PROGRAMS = {
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd", ".py", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl",
    ".php", ".lua", ".exe", ".dll", ".so", ".dylib", ".bin", ".jar", ".app", ".msi", ".vbs", ".applescript", ".scpt",
}
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
SUMMARY = {
    "high": "Sentient found signs that this skill could act against you. We suggest you decline it unless you trust "
            "where it came from.",
    "check": "Nothing clearly harmful, but read the notes below before you approve it.",
    "clear": "The safety check found nothing to worry about.",
}


def _files(folder: Path) -> list[Path]:
    """The files a skill brings: no dotfiles (they are never copied in), no symlinks, no caches. Secrets files such as
    ``.env`` and ``auth.json`` are never opened."""
    out: list[Path] = []
    for root, dirs, names in os.walk(folder, followlinks=False):
        base = Path(root)
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__" and not (base / d).is_symlink())
        for n in sorted(names):
            p = base / n
            if n.startswith(".") or n == "auth.json":
                continue
            if not p.is_symlink() and p.is_file():
                out.append(p)
    return out


def folder_hash(folder: Path) -> str:
    """SHA-256 over every file's relative path and bytes, in a fixed order (line endings normalized)."""
    h = hashlib.sha256()
    for f in _files(folder):
        h.update(f.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        try:
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            continue
        h.update(b"\0")
    return h.hexdigest()


def verdict_of(findings: list[dict]) -> str:
    if any(f.get("level") == "error" for f in findings):
        return "high"
    return "check" if findings else "clear"


def vet_folder(folder: Path, *, source: str | None = None, origin: str | None = None) -> dict:
    """The report for a skill folder: ``{verdict, summary, findings: [{rule, level, message, file}], files, programs,
    hash, scanner, source, origin}``. ``file`` is ``SKILL.md`` or the other file a finding is in."""
    findings: list[dict] = []
    skill_md = folder / "SKILL.md"
    for f in lint_file(skill_md, check_name=False):
        if f["rule"] in SECURITY_RULES:
            findings.append({**f, "file": "SKILL.md"})
    files = [f.relative_to(folder).as_posix() for f in _files(folder)]
    programs = [rel for rel in files if Path(rel).suffix.lower() in _PROGRAMS or _shebang(folder / rel)]
    for rel in files:
        if rel == "SKILL.md":
            continue
        path = folder / rel
        if path.suffix.lower() not in _TEXT and rel not in programs:
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                findings.append({"rule": "hidden", "level": "warning", "file": rel,
                                 "message": f"{rel} is too large to check, so it wasn't read."})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for f in security_findings(text):
            findings.append({**f, "file": rel})
    if programs:
        shown = ", ".join(programs[:4]) + (f" and {len(programs) - 4} more" if len(programs) > 4 else "")
        findings.append({"rule": "programs", "level": "warning", "file": programs[0],
                         "message": f"It brings {len(programs)} script{'s' if len(programs) != 1 else ''} or "
                                    f"program{'s' if len(programs) != 1 else ''} ({shown}). Sentient only runs them "
                                    "when it asks you first, but read them before you approve."})
    verdict = verdict_of(findings)
    return {
        "verdict": verdict,
        "summary": SUMMARY[verdict],
        "findings": sorted(findings, key=lambda f: f["level"] != "error"),
        "files": files,
        "programs": programs,
        "hash": folder_hash(folder),
        "scanner": SCANNER_VERSION,
        "source": source,
        "origin": origin,
    }


def _shebang(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"#!"
    except OSError:
        return False


# ---------------------------------------------------------------------------- lock records (skill_imports)
async def record_import(store: Any, name: str, report: dict) -> None:
    await store.execute(
        "INSERT INTO skill_imports(name, source, origin, hash, scanner, verdict, imported_at)"
        " VALUES(?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET source = excluded.source, origin = excluded.origin,"
        " hash = excluded.hash, scanner = excluded.scanner, verdict = excluded.verdict,"
        " imported_at = excluded.imported_at, decision = NULL, decided_at = NULL, approved_hash = NULL",
        (name, report.get("source") or "link", report.get("origin"), report["hash"], report["scanner"],
         report["verdict"], now_iso()),
    )


async def get_import(store: Any, name: str) -> dict | None:
    row = await store.fetchone("SELECT * FROM skill_imports WHERE name = ?", (name.strip().lower(),))
    return dict(row) if row else None


async def mark_approved(store: Any, name: str, folder: Path) -> None:
    await store.execute(
        "UPDATE skill_imports SET decision = 'approved', decided_at = ?, approved_hash = ? WHERE name = ?",
        (now_iso(), folder_hash(folder), name.strip().lower()),
    )


async def forget(store: Any, name: str) -> None:
    await store.execute("DELETE FROM skill_imports WHERE name = ?", (name.strip().lower(),))


async def report_for(store: Any, name: str, folder: Path) -> dict | None:
    """The live report for an imported skill (``None`` for skills that weren't imported), with its lock record:
    ``imported_at``, ``decision``, ``changed_since_import`` (the folder differs from what was checked at import) and
    ``changed_since_approval`` (an approved skill whose folder changed since)."""
    rec = await get_import(store, name)
    if rec is None or not (folder / "SKILL.md").exists():
        return None
    report = vet_folder(folder, source=rec["source"], origin=rec["origin"])
    report.update(
        imported_at=rec["imported_at"],
        decision=rec["decision"],
        changed_since_import=report["hash"] != rec["hash"],
        changed_since_approval=bool(rec["approved_hash"]) and report["hash"] != rec["approved_hash"],
        scanned_by=rec["scanner"],
    )
    return report


# ---------------------------------------------------------------------------- adding a skill from a web address
class SkillImportError(ValueError):
    """A plain sentence for the user."""


def raw_url(url: str) -> str:
    """Where the SKILL.md text is for a link the user pasted: GitHub page, folder and repository links point at the raw
    file; any other address is used as it is."""
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise SkillImportError("Use a web address that starts with https://.")
    if parts.netloc.lower() in {"github.com", "www.github.com"}:
        segs = [s for s in parts.path.split("/") if s]
        if len(segs) >= 2:
            owner, repo = segs[0], segs[1].removesuffix(".git")
            if len(segs) >= 4 and segs[2] in {"blob", "tree", "raw"}:
                ref, rest = segs[3], segs[4:]
            else:
                ref, rest = "HEAD", []
            path = "/".join(rest)
            if not path.lower().endswith(".md"):
                path = f"{path}/SKILL.md" if path else "SKILL.md"
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    return url


async def fetch_skill_text(url: str, *, transport: Any = None) -> tuple[str, str]:
    """Download a SKILL.md: ``(text, address it came from)``. Raises ``SkillImportError`` with a plain reason."""
    import httpx

    target = raw_url(url)
    try:
        async with (
            httpx.AsyncClient(timeout=FETCH_TIMEOUT_S, follow_redirects=True, max_redirects=5, transport=transport,
                              headers={"Accept": "text/markdown, text/plain;q=0.9, */*;q=0.5"}) as client,
            client.stream("GET", target) as resp,
        ):
            if resp.status_code == 404:
                raise SkillImportError("There is no SKILL.md at that address.")
            if resp.status_code >= 400:
                raise SkillImportError(f"The site answered with an error ({resp.status_code}).")
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) > MAX_FETCH_BYTES:
                    raise SkillImportError("That file is too large to be a skill.")
            final = str(resp.url)
    except httpx.HTTPError as exc:
        raise SkillImportError(f"Couldn't download it: {type(exc).__name__}.") from exc
    text = body.decode("utf-8", errors="replace").lstrip("﻿")
    if b"\0" in body or not _FRONTMATTER.match(text.replace("\r\n", "\n")):
        raise SkillImportError("That address doesn't hold a skill (a SKILL.md that starts with a --- header).")
    return text, final


def stage_text(pending_dir: Path, text: str, name: str) -> Path:
    """Write a downloaded SKILL.md into ``pending/<name>/``: the header gets the new name and ``author: community``,
    the steps are kept exactly as downloaded (so hidden text stays visible to the check)."""
    m = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    if not m:
        raise SkillImportError("That address doesn't hold a skill.")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillImportError("The skill's header can't be read.") from exc
    if not isinstance(meta, dict):
        raise SkillImportError("The skill's header can't be read.")
    meta = {**meta, "name": name, "author": "community"}
    meta["description"] = " ".join(str(meta.get("description") or "").split())
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    dst = pending_dir / name
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "SKILL.md").write_text(f"---\n{front}\n---\n\n{m.group(2).strip()}\n", encoding="utf-8")
    return dst


def suggested_name(text: str, url: str) -> str:
    m = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    meta: Any = {}
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
    name = str(meta.get("name") or "") if isinstance(meta, dict) else ""
    if not name:
        segs = [s for s in urlsplit(url).path.split("/") if s and s.lower() != "skill.md"]
        name = segs[-1] if segs else "imported-skill"
    return name
