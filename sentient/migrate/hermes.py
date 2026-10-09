"""Move from Hermes Agent in one step (issue #203, docs/API.md section 19).

``preview(app, path)`` reads a Hermes home folder (default ``~/.hermes``) and says what would happen to each thing
in it; ``apply(app, path, parts, skip)`` does it for the parts the user picked:

- ``skills/**/SKILL.md``         -> ``~/.sentient/skills/pending/<name>/`` (never active). Hermes' bundled skills
                                   the user never changed are skipped; names are made unique.
- ``memories/MEMORY.md``         -> facts with source ``import:hermes`` (``DELETE /api/import/hermes/memories``
                                   removes them again).
- ``memories/USER.md``           -> user-model insights with source ``import:hermes``.
- ``SOUL.md``                    -> Sentient's SOUL.md, only when ``persona`` is picked (the preview shows both).
- ``cron/jobs.json``             -> paused tasks. Resuming one plans it and asks for approval like any new task.
                                   A script job becomes a script task only when its Python script is there.
- ``config.yaml`` ``mcp_servers`` -> MCP servers, turned off, with no header, environment or sign-in values.
- ``config.yaml`` wake word and voice -> suggestions only.

Only the files above (and a job's script under ``scripts/``) are opened. ``auth.json``, ``.env``, sessions, logs
and state databases are never read, and skill folders are copied without dotfiles or symlinks.

Hermes formats relied on (github.com/NousResearch/hermes-agent, main, checked 2026-10-09):

- ``tools/memory_tool_store.py``: ``ENTRY_DELIMITER = "\\n§\\n"``; entries are stripped and empty ones dropped.
- ``tools/skills_sync.py``: ``skills/.bundled_manifest`` holds ``name:md5`` lines (older installs: bare names). The
  hash is MD5 over the skill folder's files in sorted path order, each adding its relative path and then its bytes,
  without ``__pycache__``-style caches. A bundled skill whose folder still has that hash was never changed.
- ``cron/jobs.py``: ``jobs.json`` is ``{"jobs": [...]}`` (a bare list or an id-keyed map also load). A job has
  ``id, name, prompt, skills, enabled, deliver`` (``local``, ``origin`` or ``<platform>:<chat>``, comma-separated
  for several), ``script`` / ``monitor_script`` (a path inside ``<home>/scripts/``) and ``schedule``:
  ``{kind: "cron", expr}``, ``{kind: "interval", minutes}`` or ``{kind: "once", run_at}``.
- ``website/docs/reference/mcp-config-reference.md``: ``mcp_servers: {name: {command, args, env} | {url, headers,
  auth: "oauth", transport}, enabled}``.
- ``hermes_cli/config_defaults.py``: ``wake_word.phrase``, ``tts.provider`` and ``tts.<provider>.voice``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sentient.skills.loader import slugify, valid_name
from sentient.tasks.schedule import DAY_NAMES, MIN_INTERVAL_MINUTES, normalize_schedule, parse_iso
from sentient.tasks.scripts import ScriptInvalid, validate_code

log = logging.getLogger(__name__)

SOURCE = "import:hermes"
PARTS = ("skills", "memory", "persona", "jobs", "mcp")
MEMORY_DELIMITER = "\n§\n"
# never opened, wherever they appear (also inside skill folders)
NEVER_READ = frozenset({"auth.json", ".env"})
_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_SKIP_DIRS = frozenset({"node_modules", "venv", ".venv", "site-packages"})
_FRONTMATTER = re.compile(r"^﻿?---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_USER_RE = re.compile(r"\b(?:[Tt]he user|User)\b")
# a key typed straight into an MCP command line or URL (it would land in config.yaml, not the keychain)
_SECRET_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "pat", "credentials"})
_SECRET_VALUE = re.compile(
    r"(?:^|[=\s])(?:sk-|sk_|ghp_|gho_|github_pat_|xox[abpr]-|AKIA|AIza|glpat-)[A-Za-z0-9_\-]{8,}"
    r"|[?&](?:api[_-]?key|key|token|access_token|secret)=[^&\s$]",
    re.IGNORECASE,
)
MAX_TEXT_BYTES = 1_000_000
MAX_SKILL_BYTES = 10_000_000
MAX_SKILL_DEPTH = 6
CHANNEL_NAMES = {"whatsapp": "WhatsApp", "telegram": "Telegram", "discord": "Discord"}
_DOW = {"sun": 6, "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5}


class HermesImportError(ValueError):
    """The folder can't be used (a plain sentence for the user)."""


def default_home() -> Path:
    return Path.home() / ".hermes"


def resolve_home(path: str | None) -> Path:
    home = Path(path).expanduser() if path and str(path).strip() else default_home()
    if not home.is_dir():
        raise HermesImportError(f"There's no Hermes folder at {home}.")
    if not any((home / n).exists() for n in ("config.yaml", "SOUL.md", "memories", "skills", "cron")):
        raise HermesImportError(f"{home} doesn't look like a Hermes folder. Pick the folder that has config.yaml in it.")
    return home


# ---------------------------------------------------------------------------- careful reading
def _read_text(path: Path, limit: int = MAX_TEXT_BYTES) -> str | None:
    """A small text file, or None. Never a secrets file and never through a symlink."""
    if path.name in NEVER_READ or path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None


def _files(folder: Path) -> tuple[list[Path], bool]:
    """Regular files under ``folder`` (no symlinks, no caches) in Hermes' hash order, and whether a secrets file
    was seen (its bytes are never read, so such a folder can't be proven unchanged)."""
    found: list[Path] = []
    secret = False
    for root, dirs, names in os.walk(folder, followlinks=False):
        base = Path(root)
        dirs[:] = [d for d in dirs if d not in _CACHE_DIRS and not (base / d).is_symlink()]
        for n in names:
            p = base / n
            if p.is_symlink() or not p.is_file():
                continue
            if n in NEVER_READ:
                secret = True
                continue
            if p.suffix in {".pyc", ".pyo"} and p.with_suffix(".py").is_file():
                continue
            found.append(p)
    return sorted(found), secret


def _dir_hashes(folder: Path, files: list[Path]) -> set[str]:
    """Hermes' folder hash, with this system's path separators and with "/" (a folder copied from another OS)."""
    native = hashlib.md5()  # MD5 because Hermes uses it; not for security
    posix = hashlib.md5()
    for f in files:
        rel = f.relative_to(folder)
        data = f.read_bytes()
        native.update(str(rel).encode("utf-8"))
        native.update(data)
        posix.update(rel.as_posix().encode("utf-8"))
        posix.update(data)
    return {native.hexdigest(), posix.hexdigest()}


def _frontmatter(text: str) -> tuple[dict | None, str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return None, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None, text
    return (meta if isinstance(meta, dict) else None), m.group(2)


def _config(home: Path) -> dict:
    """config.yaml, held in memory only: nothing but server shapes, a wake phrase and a voice name leave here."""
    raw = _read_text(home / "config.yaml")
    if not raw:
        return {}
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _item(key: str, action: str, note: str, **fields: Any) -> dict:
    return {"key": key, "action": action, "note": note, **fields}


# ---------------------------------------------------------------------------- skills
def _manifest(skills_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (_read_text(skills_root / ".bundled_manifest") or "").splitlines():
        line = line.strip()
        if line:
            name, _, digest = line.partition(":")
            out[name.strip()] = digest.strip()
    return out


def _skill_dirs(root: Path) -> list[Path]:
    out: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if depth > MAX_SKILL_DEPTH:
            return
        md = d / "SKILL.md"
        if d != root and md.is_file() and not md.is_symlink():
            out.append(d)  # a SKILL.md deeper inside a skill is part of that skill
            return
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        for e in entries:
            if e.name.startswith(".") or e.name in _SKIP_DIRS or e.is_symlink() or not e.is_dir():
                continue
            walk(e, depth + 1)

    if root.is_dir():
        walk(root, 0)
    return out


def _existing_skill(app: Any, name: str) -> bool:
    return app.skills.get_any(name) is not None or (app.skills.pending_dir / name).exists()


def _skill_bodies(app: Any) -> set[str]:
    lib = app.skills
    skills = [*lib.list(), *lib.list_pending(), *lib.list_archived()]
    return {" ".join(s.body.split()) for s in skills if s.body.strip()}


def _plan_skills(app: Any, home: Path) -> list[dict]:
    root = home / "skills"
    manifest = _manifest(root)
    bodies = _skill_bodies(app)
    taken: set[str] = set()
    items: list[dict] = []
    for folder in _skill_dirs(root):
        rel = folder.relative_to(root).as_posix()
        key = f"skill:{rel}"
        meta, body = _frontmatter(_read_text(folder / "SKILL.md") or "")
        if meta is None:
            items.append(_item(key, "skip", "Its SKILL.md has no readable header.", name=folder.name, folder=rel))
            continue
        raw_name = str(meta.get("name") or folder.name).strip()
        info = {"name": raw_name, "folder": rel, "description": " ".join(str(meta.get("description") or "").split())}
        files, secret = _files(folder)
        size = sum(f.stat().st_size for f in files)
        if size > MAX_SKILL_BYTES:
            items.append(_item(key, "skip", "This skill folder is over 10 MB.", **info))
            continue
        changed = False
        if raw_name in manifest or folder.name in manifest:
            origin = manifest.get(raw_name) or manifest.get(folder.name) or ""
            if not origin or (not secret and origin in _dir_hashes(folder, files)):
                items.append(_item(key, "skip", "Built into Hermes and never changed.", **info))
                continue
            changed = True
        if " ".join(body.split()) in bodies:
            items.append(_item(key, "skip", "Sentient already has this skill.", **info))
            continue
        base = slugify(raw_name)
        if not valid_name(base):
            base = slugify(folder.name)
        if not valid_name(base):
            items.append(_item(key, "skip", "Its name can't be used for a skill.", **info))
            continue
        target, n = base, 1
        while target in taken or _existing_skill(app, target):
            n += 1
            target = f"{base[:55]}-hermes" if n == 2 else f"{base[:52]}-hermes-{n - 1}"
        taken.add(target)
        note = "Goes to Skills to review. It stays off until you approve it."
        if target != base:
            note = f"Saved as {target} because Sentient already has a skill named {base}. " + note
        if changed:
            note = "You changed this built-in Hermes skill. " + note
        items.append(_item(key, "import", note, target=target, changed_builtin=changed, **info))
    return items


def _copy_skill(app: Any, home: Path, item: dict) -> str:
    folder = home / "skills" / item["folder"]
    dst = app.skills.pending_dir / item["target"]
    if dst.exists():
        raise FileExistsError(item["target"])
    files, _ = _files(folder)
    try:
        for f in files:
            rel = f.relative_to(folder)
            if any(part.startswith(".") for part in rel.parts):
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
        meta, body = _frontmatter((dst / "SKILL.md").read_text(encoding="utf-8-sig", errors="replace"))
        meta = dict(meta or {})
        meta["name"] = item["target"]
        meta["description"] = " ".join(str(meta.get("description") or "").split())
        hermes_meta = (meta.get("metadata") or {}).get("hermes") if isinstance(meta.get("metadata"), dict) else None
        if not meta.get("tags") and isinstance(hermes_meta, dict) and isinstance(hermes_meta.get("tags"), list):
            meta["tags"] = [str(t) for t in hermes_meta["tags"]]
        front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
        (dst / "SKILL.md").write_text(f"---\n{front}\n---\n\n{body.strip()}\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    return item["target"]


# ---------------------------------------------------------------------------- memory
def _entries(path: Path) -> list[str]:
    raw = (_read_text(path) or "").replace("\r\n", "\n")
    return [e for e in (x.strip() for x in raw.split(MEMORY_DELIMITER)) if e]


def personalize(text: str, user: str) -> str:
    """Hermes writes "User prefers ..."; Sentient says the user's name."""
    text = " ".join(text.split())
    if user:
        text = re.sub(r"\b(?:[Tt]he user|User)'s\b", f"{user}'s", text)
        text = _USER_RE.sub(user, text)
    return text[:1].upper() + text[1:]


async def _plan_memory(app: Any, home: Path) -> list[dict]:
    user = app.config.assistant.user_name.strip()
    mem = home / "memories"
    profile = mem / "USER.md" if (mem / "USER.md").is_file() else home / "USER.md"
    items: list[dict] = []
    seen: set[str] = set()
    sources = [("fact", e) for e in _entries(mem / "MEMORY.md")] + [("insight", e) for e in _entries(profile)]
    for n, (kind, entry) in enumerate(sources):
        text = personalize(entry, user)
        key = f"{kind}:{n}"
        where = "a memory" if kind == "fact" else "something Sentient knows about you"
        if text.lower() in seen:
            items.append(_item(key, "skip", "It's in the file twice.", kind=kind, text=text))
            continue
        seen.add(text.lower())
        if await _memory_exists(app, kind, text):
            items.append(_item(key, "skip", "Sentient already has this.", kind=kind, text=text))
            continue
        if kind == "fact" and app.memory is None:
            items.append(_item(key, "skip", "Memory is turned off on this computer.", kind=kind, text=text))
            continue
        items.append(_item(key, "import", f"Becomes {where}, marked as imported from Hermes.", kind=kind, text=text))
    return items


async def _memory_exists(app: Any, kind: str, text: str) -> bool:
    try:
        if kind == "fact":
            row = await app.store.fetchone("SELECT id FROM facts WHERE content = ? COLLATE NOCASE", (text,))
        else:
            row = await app.store.fetchone(
                "SELECT id FROM user_insights WHERE statement = ? COLLATE NOCASE AND status != 'retired'", (text,)
            )
    except Exception:
        return False
    return row is not None


# ---------------------------------------------------------------------------- persona
def _plan_persona(app: Any, home: Path) -> dict | None:
    proposed = (_read_text(home / "SOUL.md") or "").strip()
    if not proposed:
        return None
    current = app.workspace.read_full().get("soul", "")
    if proposed == current.strip():
        return _item("persona", "skip", "Same as Sentient's personality now.", current=current, proposed=proposed)
    note = "Replaces Sentient's personality (SOUL.md) after you confirm."
    if re.search(r"\bhermes\b", proposed, re.IGNORECASE):
        note += " It mentions Hermes by name; you can edit it in Settings > Personality."
    return _item("persona", "import", note, current=current, proposed=proposed)


# ---------------------------------------------------------------------------- scheduled jobs
def _int(field: str, lo: int, hi: int) -> int | None:
    return int(field) if field.isdigit() and lo <= int(field) <= hi else None


def _days(field: str) -> list[str] | None:
    """Cron day-of-week (0 or 7 = Sunday, names, lists, ranges) as Sentient day names, or None."""
    picked: set[int] = set()

    def num(token: str) -> int | None:
        t = token.strip().lower()[:3]
        if t in _DOW:
            return _DOW[t]
        if t.isdigit() and 0 <= int(t) <= 7:
            return (int(t) - 1) % 7  # cron 1 = Monday -> 0; 0 and 7 = Sunday -> 6
        return None

    for part in field.split(","):
        lo, sep, hi = part.partition("-")
        a = num(lo)
        if a is None:
            return None
        if not sep:
            picked.add(a)
            continue
        b = num(hi)
        if b is None:
            return None
        start, end = (int(lo) if lo.isdigit() else None), (int(hi) if hi.isdigit() else None)
        if start is not None and end is not None:  # numeric range in cron order (0 = Sunday)
            if start > end:
                return None
            picked.update((d - 1) % 7 for d in range(start, end + 1))
        else:
            i = a
            while True:
                picked.add(i)
                if i == b:
                    break
                i = (i + 1) % 7
    return [DAY_NAMES[i] for i in sorted(picked)]


def cron_to_schedule(expr: str) -> tuple[dict | None, str]:
    """A 5-field cron expression as a Sentient recurring schedule, or ``(None, reason)``.

    Sentient runs daily or weekly at one time, or every N minutes (at least 5). Anything else is skipped."""
    fields = expr.split()
    if len(fields) != 5:
        return None, f"Sentient can't run the schedule \"{expr}\" yet."
    minute, hour, dom, month, dow = fields
    if dom != "*" or month != "*":
        return None, f"\"{expr}\" runs on certain days of the month, which Sentient can't do yet."
    if hour == "*" and dow == "*":
        if minute == "*":
            return None, f"\"{expr}\" runs every minute; Sentient runs things at most every {MIN_INTERVAL_MINUTES} minutes."
        if minute.startswith("*/") and minute[2:].isdigit():
            every = int(minute[2:])
            if every < MIN_INTERVAL_MINUTES:
                return None, f"\"{expr}\" runs every {every} minutes; Sentient runs things at most every {MIN_INTERVAL_MINUTES} minutes."
            return {"type": "recurring", "frequency": "interval", "interval_minutes": every}, ""
        if _int(minute, 0, 59) is not None:
            return {"type": "recurring", "frequency": "interval", "interval_minutes": 60}, ""
    if dow == "*" and _int(minute, 0, 59) is not None and hour.startswith("*/") and hour[2:].isdigit() and int(hour[2:]) > 0:
        return {"type": "recurring", "frequency": "interval", "interval_minutes": int(hour[2:]) * 60}, ""
    m, h = _int(minute, 0, 59), _int(hour, 0, 23)
    if m is None or h is None:
        return None, f"\"{expr}\" runs at several times a day, which Sentient can't do in one task yet."
    time = f"{h:02d}:{m:02d}"
    if dow == "*":
        return {"type": "recurring", "frequency": "daily", "time": time}, ""
    days = _days(dow)
    if not days:
        return None, f"Sentient can't read the days in \"{expr}\"."
    return {"type": "recurring", "frequency": "weekly", "days": days, "time": time}, ""


def _job_schedule(raw: Any) -> tuple[dict | None, str, str]:
    """(Sentient schedule, Hermes' own wording, reason when it can't be used)."""
    if isinstance(raw, str):
        text = raw.strip()
        m = re.fullmatch(r"(?:every\s+)?(\d+)\s*(m|min|mins|minutes?|h|hr|hours?)", text, re.IGNORECASE)
        if m:
            minutes = int(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)
            raw = {"kind": "interval", "minutes": minutes, "display": text}
        else:
            raw = {"kind": "cron", "expr": text, "display": text}
    if not isinstance(raw, dict):
        return None, "", "This job has no schedule."
    kind = str(raw.get("kind") or "").lower()
    shown = str(raw.get("display") or raw.get("expr") or raw.get("run_at") or "").strip()
    if kind == "interval":
        try:
            minutes = int(float(raw.get("minutes") or 0))
        except (TypeError, ValueError):
            minutes = 0
        if minutes < MIN_INTERVAL_MINUTES:
            return None, shown, f"It runs every {minutes} minutes; Sentient runs things at most every {MIN_INTERVAL_MINUTES} minutes."
        return {"type": "recurring", "frequency": "interval", "interval_minutes": minutes}, shown or f"every {minutes}m", ""
    if kind == "cron":
        schedule, reason = cron_to_schedule(str(raw.get("expr") or ""))
        return schedule, shown, reason
    if kind == "once":
        when = parse_iso(raw.get("run_at"))
        if when is None or when <= datetime.now(UTC):
            return None, shown, "A one-time job whose time has passed."
        return {"type": "once", "run_at": when.isoformat(timespec="seconds")}, shown, ""
    return None, shown, "Sentient can't read this job's schedule."


def _jobs(home: Path) -> list[dict]:
    raw = _read_text(home / "cron" / "jobs.json", limit=5_000_000)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    jobs = data.get("jobs", []) if isinstance(data, dict) else data
    if isinstance(jobs, dict):
        jobs = [{**v, "id": v.get("id") or k} for k, v in jobs.items() if isinstance(v, dict)]
    return [j for j in jobs if isinstance(j, dict)] if isinstance(jobs, list) else []


def _script(home: Path, raw: str) -> tuple[Path | None, str, str]:
    """(script file, path to show, reason it can't be used)."""
    scripts = home / "scripts"
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else scripts / path
    try:
        shown = path.relative_to(home).as_posix()
    except ValueError:
        shown = str(path)
    if path.name in NEVER_READ:
        return None, shown, "Its script can't be imported."
    try:
        real = path.resolve()
        real.relative_to(scripts.resolve())
    except (OSError, ValueError):
        return None, shown, f"Its script {shown} is outside Hermes' scripts folder."
    if path.is_symlink() or not real.is_file():
        return None, shown, f"Its script {shown} isn't there."
    if real.suffix.lower() != ".py":
        return None, shown, f"Its script {shown} isn't Python; Sentient's check scripts are Python only."
    return real, shown, ""


async def _delivery(app: Any, deliver: Any, origin: Any = None) -> tuple[str, str]:
    """Where results go: a paired messaging app named by the job (or the chat it was made in), else this computer."""
    wanted = [d.split(":", 1)[0].strip().lower() for d in str(deliver or "").split(",") if d.strip()]
    if "origin" in wanted and isinstance(origin, dict):
        wanted.append(str(origin.get("platform") or "").strip().lower())
    asked = [p for p in wanted if p in CHANNEL_NAMES]
    for platform in asked:
        try:
            paired = await app.channels.store.chats(platform)
        except Exception:
            paired = []
        if paired:
            return platform, f"Results go to your paired {CHANNEL_NAMES[platform]} chat and this computer."
    if asked:
        name = CHANNEL_NAMES[asked[0]]
        return "desktop", f"{name} isn't set up in Sentient yet, so results show on this computer. Pair it in Channels."
    return "desktop", "Results show on this computer."


async def _plan_jobs(app: Any, home: Path) -> list[dict]:
    items: list[dict] = []
    tz = app.tasks.tz_name()
    done = {
        str((t.get("original_context") or {}).get("hermes_job"))
        for t in await app.tasks.repo.list_tasks()
        if (t.get("original_context") or {}).get("imported_from") == "hermes"
    }
    for n, job in enumerate(_jobs(home)):
        key = f"job:{job.get('id') or n}"
        prompt = str(job.get("prompt") or "").strip()
        skills = [str(s) for s in (job.get("skills") or ([job["skill"]] if job.get("skill") else [])) if str(s).strip()]
        name = str(job.get("name") or "").strip() or (prompt or (skills[0] if skills else "Hermes job"))[:50]
        schedule, shown, reason = _job_schedule(job.get("schedule"))
        info: dict[str, Any] = {"name": name, "prompt": prompt, "schedule_text": shown, "skills": skills, "script": None}
        if key.split(":", 1)[1] in done:
            items.append(_item(key, "skip", "Already brought over.", schedule=None, kind="task", delivery="desktop", **info))
            continue
        if schedule is None:
            items.append(_item(key, "skip", reason, schedule=None, kind="task", delivery="desktop", **info))
            continue
        schedule = normalize_schedule(schedule, tz)
        delivery, delivery_note = await _delivery(app, job.get("deliver"), job.get("origin"))
        info.update(schedule=schedule, delivery=delivery)
        raw_script = str(job.get("monitor_script") or job.get("script") or "").strip()
        kind = "task"
        notes = [delivery_note]
        if raw_script:
            path, shown_path, why = _script(home, raw_script)
            code = None
            if path is not None:
                try:
                    code = validate_code(_read_text(path))
                except ScriptInvalid as exc:
                    why = f"Its script {shown_path} can't be used: {exc}"
            if code is None:
                items.append(_item(key, "skip", why, kind="script", **info))
                continue
            kind = "script"
            info["script"] = {"path": shown_path, "code": code}
            then = "run" if prompt and not job.get("no_agent") else "notify"
            info["then"] = then
            notes.insert(0, f"Runs your script {shown_path} with no AI; when its output changes, "
                         + ("Sentient carries out the job." if then == "run" else "you get a notification."))
        elif not prompt and not skills:
            items.append(_item(key, "skip", "This job has nothing to do.", kind="task", **info))
            continue
        if skills:
            notes.append("It used the Hermes skills " + ", ".join(skills) + "; import them too so it can use them.")
        notes.insert(0, "Added paused. Resume it and Sentient makes a plan for you to approve.")
        items.append(_item(key, "import", " ".join(notes), kind=kind, **info))
    return items


# ---------------------------------------------------------------------------- MCP servers
def _plan_mcp(app: Any, cfg: dict) -> list[dict]:
    from sentient.integrations.mcp import plugin_id_for

    servers = cfg.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    existing = {plugin_id_for(n) for n in app.config.integrations.mcp_servers}
    taken: set[str] = set()
    items: list[dict] = []
    for raw_name, spec in servers.items():
        name = str(raw_name).strip()
        key = f"mcp:{name}"
        if not isinstance(spec, dict) or not name:
            continue
        url = str(spec.get("url") or "").strip()
        command = str(spec.get("command") or "").strip()
        headers = spec.get("headers") if isinstance(spec.get("headers"), dict) else {}
        env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
        transport = "http" if url else "stdio"
        auth = "none"
        if transport == "http":
            auth = "oauth" if str(spec.get("auth") or "").lower() == "oauth" else ("headers" if headers else "none")
        info = {
            "name": name,
            "transport": transport,
            "url": url or None,
            "command": command or None,
            "args": [str(a) for a in spec.get("args") or []] if transport == "stdio" else [],
            "auth": auth,
            "header_keys": sorted(str(k) for k in headers) if transport == "http" else [],
            "env_keys": sorted(str(k) for k in env) if transport == "stdio" else [],
        }
        pid = plugin_id_for(name)
        if _has_secret([url, command, *info["args"]]):
            items.append(_item(key, "skip", "Its address or command line has what looks like a key in it. Add it "
                               "yourself in Integrations so the key is kept in the system keychain.", **info))
            continue
        if not url and not command:
            items.append(_item(key, "skip", "It has no address or command.", **info))
            continue
        if pid in existing or pid in taken:
            items.append(_item(key, "skip", "Sentient already has a server with this name.", **info))
            continue
        taken.add(pid)
        notes = ["Added turned off. Turn it on in Integrations."]
        if auth == "oauth":
            notes.append("Then sign in again.")
        if info["header_keys"]:
            notes.append(f"Its header values ({', '.join(info['header_keys'])}) aren't copied; add the server again with them.")
        if info["env_keys"]:
            notes.append(f"Its settings ({', '.join(info['env_keys'])}) aren't copied; add the server again with them.")
        if "${" in " ".join([url, command, *info["args"]]):
            notes.append("It uses ${...} values from Hermes' settings; change them to real values when you add it again.")
        if str(spec.get("transport") or "").lower() == "sse":
            notes.append("It uses an older kind of connection that may not work.")
        items.append(_item(key, "import", " ".join(notes), **info))
    return items


def _secret_flag(arg: str) -> bool:
    """``--api-key``, ``--access_token``, ``-password``; not ``--keyboard-layout``."""
    if not arg.startswith("-"):
        return False
    return any(w in _SECRET_WORDS for w in re.split(r"[-_]", arg.lstrip("-").lower()))


def _has_secret(parts: list[str]) -> bool:
    """True when a URL or argument looks like it carries a key (``--api-key abc``, ``--token=abc``, ``ghp_...``,
    ``?api_key=...``). ``${VAR}`` references are fine: they hold no value."""
    after_flag = False
    for part in parts:
        if not part:
            continue
        if after_flag and "${" not in part:
            return True
        flag, eq, value = part.partition("=")
        after_flag = _secret_flag(part)
        if eq and _secret_flag(flag) and value and "${" not in value:
            return True
        if _SECRET_VALUE.search(part):
            return True
    return False


def _suggestions(cfg: dict) -> dict:
    wake = cfg.get("wake_word") if isinstance(cfg.get("wake_word"), dict) else {}
    tts = cfg.get("tts") if isinstance(cfg.get("tts"), dict) else {}
    provider = str(tts.get("provider") or "").strip() or None
    engine = tts.get(provider) if provider and isinstance(tts.get(provider), dict) else {}
    voice = str(engine.get("voice") or engine.get("voice_id") or "").strip() or None
    phrase = str(wake.get("phrase") or "").strip() or None
    return {"wake_word": phrase, "tts_provider": provider, "tts_voice": voice}


# ---------------------------------------------------------------------------- public API
async def preview(app: Any, path: str | None = None) -> dict:
    home = resolve_home(path)
    cfg = _config(home)
    plan = {
        "path": str(home),
        "skills": _plan_skills(app, home),
        "memory": await _plan_memory(app, home),
        "persona": _plan_persona(app, home),
        "jobs": await _plan_jobs(app, home),
        "mcp": _plan_mcp(app, cfg),
        "suggestions": _suggestions(cfg),
        "never_read": sorted(NEVER_READ),
    }
    persona = plan["persona"]
    plan["counts"] = {
        part: sum(1 for i in plan[part] if i["action"] == "import") for part in ("skills", "memory", "jobs", "mcp")
    } | {"persona": 1 if persona and persona["action"] == "import" else 0}
    return plan


def _picked(items: list[dict], skip: set[str]) -> list[dict]:
    return [i for i in items if i["action"] == "import" and i["key"] not in skip]


def _skipped(items: list[dict], skip: set[str]) -> list[dict]:
    return [
        {"key": i["key"], "name": i.get("name") or i.get("text") or i["key"],
         "note": "You left it out." if i["key"] in skip else i["note"]}
        for i in items if i["action"] != "import" or i["key"] in skip
    ]


async def apply(app: Any, path: str | None, parts: list[str] | None, skip: list[str] | None = None) -> dict:
    """Import the picked parts. The plan is worked out again from the folder, so only what a preview would show
    as "import" happens. ``persona`` is only applied when it is in ``parts`` (the user confirmed the diff)."""
    chosen = {p for p in (parts or []) if p in PARTS}
    if not chosen:
        raise HermesImportError("Pick at least one thing to import.")
    plan = await preview(app, path)
    home = Path(plan["path"])
    left_out = set(skip or [])
    result: dict[str, Any] = {"path": plan["path"]}

    if "skills" in chosen:
        imported: list[str] = []
        failed: list[dict] = []
        for item in _picked(plan["skills"], left_out):
            try:
                imported.append(_copy_skill(app, home, item))
            except Exception as exc:
                log.warning("could not import Hermes skill %s: %s", item["folder"], exc)
                failed.append({"key": item["key"], "name": item["name"], "note": "Couldn't copy it."})
        if imported:
            app.skills.reload({p.id for p in app.registry.plugins()})
            await app.skills.sync_stats(app.store)
            for name in imported:
                app.bus.publish("skill.updated", {"name": name, "state": "pending_review"})
        result["skills"] = {"imported": imported, "skipped": _skipped(plan["skills"], left_out) + failed}

    if "memory" in chosen:
        facts = insights = 0
        failed = []
        for item in _picked(plan["memory"], left_out):
            try:
                if item["kind"] == "fact":
                    r = await app.memory.remember(item["text"], source=SOURCE, use_llm=False)
                    facts += r["action"] == "ADD"
                else:
                    insights += await app.user_model.import_insight(item["text"], source=SOURCE) is not None
            except Exception as exc:
                log.warning("could not import a Hermes memory: %s", exc)
                failed.append({"key": item["key"], "name": item["text"], "note": "Couldn't save it."})
        if facts and app.memory is not None:
            app.memory.publish("ADD", None, None, source=SOURCE, count=facts)
        result["memory"] = {"facts": facts, "insights": insights, "skipped": _skipped(plan["memory"], left_out) + failed}

    if "persona" in chosen:
        persona = plan["persona"]
        updated = bool(persona and persona["action"] == "import" and "persona" not in left_out)
        if updated:
            app.workspace.write("soul", persona["proposed"].rstrip() + "\n")
        result["persona"] = {"updated": updated}

    if "jobs" in chosen:
        created: list[dict] = []
        failed = []
        for item in _picked(plan["jobs"], left_out):
            try:
                task = await app.tasks.create_imported(
                    name=item["name"],
                    prompt=_job_prompt(item),
                    schedule=item["schedule"],
                    script=({"code": item["script"]["code"], "condition": "changed", "then": item["then"]}
                            if item["kind"] == "script" else None),
                    context={"source": SOURCE, "imported_from": "hermes", "hermes_job": item["key"].split(":", 1)[1],
                             "hermes_schedule": item["schedule_text"], "deliver": item["delivery"],
                             **({"script_path": item["script"]["path"]} if item["script"] else {})},
                )
            except Exception as exc:
                log.warning("could not import Hermes job %s: %s", item["key"], exc)
                failed.append({"key": item["key"], "name": item["name"], "note": "Couldn't add it."})
                continue
            created.append({"task_id": task["task_id"], "name": task["name"]})
        result["jobs"] = {"created": created, "skipped": _skipped(plan["jobs"], left_out) + failed}

    if "mcp" in chosen:
        added: list[str] = []
        failed = []
        for item in _picked(plan["mcp"], left_out):
            try:
                await app.integrations.mcp.import_server(item["name"], {
                    k: item[k] for k in ("transport", "url", "command", "args", "auth", "header_keys", "env_keys")
                })
                added.append(item["name"])
            except ValueError as exc:
                failed.append({"key": item["key"], "name": item["name"], "note": str(exc)})
            except Exception as exc:
                log.warning("could not import Hermes MCP server %s: %s", item["name"], exc)
                failed.append({"key": item["key"], "name": item["name"], "note": "Couldn't add it."})
        result["mcp"] = {"added": added, "skipped": _skipped(plan["mcp"], left_out) + failed}
    return result


def _job_prompt(item: dict) -> str:
    prompt = item["prompt"] or f"Run the Hermes job '{item['name']}'."
    if item["skills"]:
        prompt += "\n\nUse these skills if they help: " + ", ".join(item["skills"]) + "."
    return prompt


async def remove_memories(app: Any) -> dict:
    """Undo the memory part: every fact and insight whose source is ``import:hermes``."""
    facts = await app.memory.forget_source(SOURCE) if app.memory is not None else 0
    insights = await app.user_model.delete_by_source(SOURCE)
    return {"facts": facts, "insights": insights}
