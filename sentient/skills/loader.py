"""Skills: folders containing a SKILL.md (AgentSkills standard, as used by Hermes and OpenClaw).

    ~/.sentient/skills/<name>/SKILL.md            active (or stale: state kept in skill_stats)
    ~/.sentient/skills/pending/<name>/SKILL.md    waiting for review (new skill, or a proposed
                                                  update of the active skill with the same name)
    ~/.sentient/skills/archived/<name>/SKILL.md   retired by the curator or the user
    ---
    name: weekly-review
    description: How to compile the user's weekly review from calendar and notes.
    version: 2
    author: user | assistant | community
    tags: [productivity]
    requires_tools: [gcalendar]     # optional: skill is hidden if a tool plugin is missing
    browser_profile: x-growth       # optional: browser tools use this profile once the skill is read
    created_by_review: false
    ---
    # Weekly review
    ## When to use ...
    ## Procedure ...

Progressive disclosure keeps prompts small: the system prompt lists only names
and descriptions; the model calls ``skill_view`` to read the body when needed.
Usage counters and lifecycle state live in the ``skill_stats`` table.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sentient import paths
from sentient.store.db import now_iso

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
RESERVED = {"pending", "archived"}
_UNSET: Any = object()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:64]


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name)) and name not in RESERVED


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    body: str
    tags: list[str] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)
    version: str = "1"
    author: str = "user"      # user | assistant | community
    state: str = "active"     # active | pending_review | archived  (stale comes from skill_stats)
    created_by_review: bool = False
    browser_profile: str | None = None

    @property
    def dir(self) -> Path:
        return self.path.parent

    def summary_line(self) -> str:
        return f"- {self.name}: {self.description}"

    @property
    def version_int(self) -> int:
        try:
            return int(float(self.version))
        except ValueError:
            return 1

    def to_dict(self, stats: dict | None = None, *, body: bool = False) -> dict:
        stats = stats or {}
        state = self.state
        if state == "active" and stats.get("state") == "stale":
            state = "stale"
        d = {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "state": state,
            "tags": self.tags,
            "requires_tools": self.requires_tools,
            "version": self.version,
            "use_count": int(stats.get("use_count") or 0),
            "view_count": int(stats.get("view_count") or 0),
            "patch_count": int(stats.get("patch_count") or 0),
            "last_used_at": stats.get("last_used_at"),
            "success_count": int(stats.get("success_count") or 0),
            "failure_count": int(stats.get("failure_count") or 0),
            "last_failure_at": stats.get("last_failure_at"),
            "created_by_review": self.created_by_review,
            "browser_profile": self.browser_profile,
        }
        if body:
            d["body"] = self.body
        return d


def parse_skill_file(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _FRONTMATTER.match(text)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    name = str(meta.get("name") or path.parent.name).strip().lower()
    if not valid_name(name):
        return None
    parent = path.parent.parent.name
    state = {"pending": "pending_review", "archived": "archived"}.get(parent, "active")
    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        path=path,
        body=m.group(2).strip(),
        tags=[str(t) for t in meta.get("tags", []) or []],
        requires_tools=[str(t) for t in meta.get("requires_tools", []) or []],
        version=str(meta.get("version", "1")),
        author=str(meta.get("author", "user")),
        state=state,
        created_by_review=bool(meta.get("created_by_review", False)),
        browser_profile=str(meta.get("browser_profile") or "").strip() or None,
    )


def render_skill(
    name: str,
    description: str,
    body: str,
    *,
    author: str,
    version: int,
    tags: list[str] | None,
    requires_tools: list[str] | None,
    created_by_review: bool = False,
    browser_profile: str | None = None,
) -> str:
    meta: dict[str, Any] = {
        "name": name,
        "description": " ".join(description.split()),
        "version": str(version),
        "author": author,
        "tags": tags or [],
        "requires_tools": requires_tools or [],
    }
    if browser_profile:
        meta["browser_profile"] = browser_profile
    if created_by_review:
        meta["created_by_review"] = True
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{body.strip()}\n"


class SkillLibrary:
    def __init__(self, dirs: list[Path] | None = None, store: Any = None):
        self.dirs = dirs or [paths.skills_dir()]
        self.store = store  # set by the evolution service; used for skill_stats
        self._skills: dict[str, Skill] = {}
        self._available: set[str] | None = None
        self._stats_ready = False

    # ------------------------------------------------------------------ layout
    @property
    def root(self) -> Path:
        return self.dirs[0]

    @property
    def pending_dir(self) -> Path:
        return self.root / "pending"

    @property
    def archived_dir(self) -> Path:
        return self.root / "archived"

    def _file(self, base: Path, name: str) -> Path:
        return base / name / "SKILL.md"

    # ------------------------------------------------------------------ reading
    def reload(self, available_plugins: set[str] | None = _UNSET) -> None:
        if available_plugins is not _UNSET:
            self._available = available_plugins
        self._skills.clear()
        for d in self.dirs:
            if not d.exists():
                continue
            for skill_file in sorted(d.glob("*/SKILL.md")):
                if skill_file.parent.name in RESERVED:
                    continue
                skill = parse_skill_file(skill_file)
                if not skill or skill.name in self._skills:
                    continue
                if self._available is not None and any(r not in self._available for r in skill.requires_tools):
                    continue
                self._skills[skill.name] = skill

    def list(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.strip().lower())

    def _list_dir(self, base: Path) -> list[Skill]:
        out = []
        if base.exists():
            for f in sorted(base.glob("*/SKILL.md")):
                s = parse_skill_file(f)
                if s:
                    out.append(s)
        return out

    def list_pending(self) -> list[Skill]:
        return self._list_dir(self.pending_dir)

    def list_archived(self) -> list[Skill]:
        return self._list_dir(self.archived_dir)

    def get_active_file(self, name: str) -> Skill | None:
        """The active skill on disk in the writable root, ignoring requires_tools filtering."""
        f = self._file(self.root, name)
        return parse_skill_file(f) if f.exists() else None

    def get_pending(self, name: str) -> Skill | None:
        f = self._file(self.pending_dir, name)
        return parse_skill_file(f) if f.exists() else None

    def get_archived(self, name: str) -> Skill | None:
        f = self._file(self.archived_dir, name)
        return parse_skill_file(f) if f.exists() else None

    def get_any(self, name: str) -> Skill | None:
        name = name.strip().lower()
        return self.get(name) or self.get_active_file(name) or self.get_pending(name) or self.get_archived(name)

    def prompt_index(self, limit_chars: int = 2000) -> str:
        lines = [s.summary_line() for s in self.list()]
        text = "\n".join(lines)
        return text[:limit_chars]

    # ------------------------------------------------------------------ writing (self-evolution)
    def write(
        self,
        name: str,
        description: str,
        body: str,
        *,
        author: str = "assistant",
        tags: list[str] | None = None,
        requires_tools: list[str] | None = None,
        pending: bool = False,
        created_by_review: bool = False,
        version: int | None = None,
    ) -> Path:
        """Create a skill. With ``pending`` and an active skill of the same name, this stores a
        proposed update next to it (see ``diff``)."""
        name = name.strip().lower()
        if not valid_name(name):
            raise ValueError("skill name must be lowercase letters, digits and dashes (2-64 chars)")
        current = self.get_active_file(name)
        if version is None:
            version = current.version_int + 1 if current else 1
        base = self.pending_dir if pending else self.root
        path = self._file(base, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_skill(
                name, description, body, author=author, version=version, tags=tags,
                requires_tools=requires_tools, created_by_review=created_by_review,
                browser_profile=current.browser_profile if current else None,
            ),
            encoding="utf-8",
        )
        return path

    def approve_pending(self, name: str) -> Path:
        name = name.strip().lower()
        src = self.pending_dir / name
        dst = self.root / name
        if not (src / "SKILL.md").exists():
            raise FileNotFoundError(name)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        return dst

    def reject_pending(self, name: str) -> None:
        src = self.pending_dir / name.strip().lower()
        if not src.exists():
            raise FileNotFoundError(name)
        shutil.rmtree(src)

    def edit(
        self,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
        requires_tools: list[str] | None = None,
        target: str | None = None,
    ) -> Skill:
        """Edit in place.

        ``target`` picks the copy: ``"active"``, ``"pending"`` (a proposal awaiting review, e.g.
        "edit then approve" of a proposed update) or ``"archived"``. Without it: the active skill,
        else the pending proposal, else the archived copy. Only active edits bump the version.
        """
        name = name.strip().lower()
        pickers = {
            "active": self.get_active_file,
            "pending": self.get_pending,
            "archived": self.get_archived,
        }
        if target is not None:
            if target not in pickers:
                raise ValueError("target must be active, pending or archived")
            skill = pickers[target](name)
        else:
            skill = self.get_active_file(name) or self.get_pending(name) or self.get_archived(name)
        if skill is None:
            raise FileNotFoundError(name)
        skill.path.write_text(
            render_skill(
                name,
                description if description is not None else skill.description,
                body if body is not None else skill.body,
                author=skill.author,
                version=skill.version_int + (1 if skill.state == "active" else 0),
                tags=tags if tags is not None else skill.tags,
                requires_tools=requires_tools if requires_tools is not None else skill.requires_tools,
                created_by_review=skill.created_by_review,
                browser_profile=skill.browser_profile,
            ),
            encoding="utf-8",
        )
        updated = parse_skill_file(skill.path)
        assert updated is not None
        return updated

    def delete(self, name: str) -> bool:
        name = name.strip().lower()
        self._skills.pop(name, None)
        found = False
        for base in (self.root, self.pending_dir, self.archived_dir):
            folder = base / name
            if name not in RESERVED and (folder / "SKILL.md").exists():
                shutil.rmtree(folder)
                found = True
        return found

    def archive(self, name: str) -> Path:
        name = name.strip().lower()
        src = self.root / name
        if name in RESERVED or not (src / "SKILL.md").exists():
            raise FileNotFoundError(name)
        dst = self.archived_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        self._skills.pop(name, None)
        return dst

    def restore(self, name: str) -> Path:
        name = name.strip().lower()
        src = self.archived_dir / name
        if not (src / "SKILL.md").exists():
            raise FileNotFoundError(name)
        dst = self.root / name
        if dst.exists():
            raise FileExistsError(name)
        shutil.move(str(src), str(dst))
        return dst

    def diff(self, name: str) -> dict:
        name = name.strip().lower()
        proposed = self._file(self.pending_dir, name)
        if not proposed.exists():
            raise FileNotFoundError(name)
        current = self._file(self.root, name)
        return {
            "current": current.read_text(encoding="utf-8") if current.exists() else "",
            "proposed": proposed.read_text(encoding="utf-8"),
        }

    # ------------------------------------------------------------------ stats (skill_stats table)
    async def _ensure_stats(self, store: Any) -> None:
        if self._stats_ready:
            return
        await store.ensure_column("skill_stats", "created_at", "TEXT")
        # usage outcomes recorded by self-evolution (skill repair) so the curator can prefer reliable skills
        await store.ensure_column("skill_stats", "success_count", "INTEGER NOT NULL DEFAULT 0")
        await store.ensure_column("skill_stats", "failure_count", "INTEGER NOT NULL DEFAULT 0")
        await store.ensure_column("skill_stats", "last_failure_at", "TEXT")
        await store.db.commit()
        self._stats_ready = True

    async def stats(self, store: Any = None) -> dict[str, dict]:
        store = store or self.store
        if store is None:
            return {}
        await self._ensure_stats(store)
        rows = await store.fetchall("SELECT * FROM skill_stats")
        return {r["name"]: dict(r) for r in rows}

    async def sync_stats(self, store: Any = None) -> None:
        """Make skill_stats mirror what is on disk (one row per known skill)."""
        store = store or self.store
        if store is None:
            return
        await self._ensure_stats(store)
        existing = await self.stats(store)
        on_disk: dict[str, str] = {}
        for s in self.list_archived():
            on_disk[s.name] = "archived"
        for s in self.list_pending():
            on_disk[s.name] = "pending_review"
        for s in self._list_dir(self.root):
            on_disk[s.name] = "active"
        for s in self.list():
            on_disk.setdefault(s.name, "active")
        ts = now_iso()
        for name, state in on_disk.items():
            row = existing.get(name)
            if row is None:
                await store.execute(
                    "INSERT INTO skill_stats(name, state, created_at) VALUES(?,?,?)", (name, state, ts)
                )
            else:
                new_state = state
                if state == "active" and row.get("state") == "stale":
                    new_state = "stale"
                if new_state != row.get("state") or not row.get("created_at"):
                    await store.execute(
                        "UPDATE skill_stats SET state = ?, created_at = COALESCE(created_at, ?) WHERE name = ?",
                        (new_state, ts, name),
                    )
        for name in set(existing) - set(on_disk):
            await store.execute("DELETE FROM skill_stats WHERE name = ?", (name,))

    async def mark_used(self, name: str, store: Any = None, *, viewed: bool = True, when: datetime | None = None) -> None:
        store = store or self.store
        if store is None:
            return
        await self._ensure_stats(store)
        ts = (when or datetime.now(UTC)).isoformat()
        await store.execute(
            "INSERT INTO skill_stats(name, use_count, view_count, last_used_at, state, created_at) VALUES(?, 1, ?, ?, 'active', ?)"
            " ON CONFLICT(name) DO UPDATE SET use_count = use_count + 1, view_count = view_count + excluded.view_count,"
            " last_used_at = excluded.last_used_at,"
            " state = CASE WHEN skill_stats.state = 'stale' THEN 'active' ELSE skill_stats.state END",
            (name.strip().lower(), 1 if viewed else 0, ts, ts),
        )

    async def record_outcome(
        self, name: str, success: bool, store: Any = None, *, revert_success: bool = False, when: datetime | None = None
    ) -> None:
        """Count one use of a skill as a success or a failure. ``revert_success`` turns a success that was
        already counted into a failure (the user corrected the result on the next message)."""
        store = store or self.store
        if store is None:
            return
        await self._ensure_stats(store)
        name = name.strip().lower()
        ts = (when or datetime.now(UTC)).isoformat()
        await store.execute(
            "INSERT INTO skill_stats(name, state, created_at) VALUES(?, 'active', ?) ON CONFLICT(name) DO NOTHING",
            (name, ts),
        )
        if success:
            await store.execute("UPDATE skill_stats SET success_count = success_count + 1 WHERE name = ?", (name,))
            return
        undo = 1 if revert_success else 0
        await store.execute(
            "UPDATE skill_stats SET failure_count = failure_count + 1, last_failure_at = ?,"
            " success_count = MAX(0, success_count - ?) WHERE name = ?",
            (ts, undo, name),
        )

    @staticmethod
    def reliability(stats: dict | None) -> float:
        """Share of recorded uses that went well; 0.5 when nothing is recorded yet."""
        stats = stats or {}
        ok, bad = int(stats.get("success_count") or 0), int(stats.get("failure_count") or 0)
        return ok / (ok + bad) if ok + bad else 0.5

    async def set_state(self, name: str, state: str, store: Any = None) -> None:
        store = store or self.store
        if store is None:
            return
        await self._ensure_stats(store)
        await store.execute(
            "INSERT INTO skill_stats(name, state, created_at) VALUES(?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET state = excluded.state",
            (name.strip().lower(), state, now_iso()),
        )

    async def record_patch(self, name: str, state: str, store: Any = None) -> None:
        store = store or self.store
        if store is None:
            return
        await self._ensure_stats(store)
        await store.execute(
            "INSERT INTO skill_stats(name, patch_count, state, created_at) VALUES(?, 1, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET patch_count = patch_count + 1,"
            " state = CASE WHEN skill_stats.state IN ('active', 'stale') AND excluded.state = 'pending_review'"
            " THEN skill_stats.state ELSE excluded.state END",
            (name.strip().lower(), state, now_iso()),
        )

    async def catalog(self, store: Any = None) -> dict[str, list[dict]]:
        """Shape of GET /api/skills."""
        stats = await self.stats(store)
        active = {s.name: s for s in self._list_dir(self.root)}
        for s in self.list():
            active.setdefault(s.name, s)
        return {
            "active": [s.to_dict(stats.get(n)) for n, s in sorted(active.items())],
            "pending": [s.to_dict(stats.get(s.name)) for s in self.list_pending()],
            "archived": [s.to_dict(stats.get(s.name)) for s in self.list_archived()],
        }
