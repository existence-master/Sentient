"""Skills: folders containing a SKILL.md (AgentSkills standard, as used by Hermes and OpenClaw).

    ~/.sentient/skills/<name>/SKILL.md            active (or stale: state kept in skill_stats)
    ~/.sentient/skills/pending/<name>/SKILL.md    waiting for review (new skill, or a proposed
                                                  update of the active skill with the same name)
    ~/.sentient/skills/archived/<name>/SKILL.md   retired by the curator or the user
    ~/.sentient/skills/<name>/references/*.md     optional reference files (a skill learned from long docs, #167):
                                                  ``skill_view`` reads one only when the skill is used
    ---
    name: weekly-review
    description: How to compile the user's weekly review from calendar and notes.
    version: 2
    author: user | assistant | community
    tags: [productivity]
    requires_tools: [gcalendar]     # optional: skill is hidden if a tool plugin is missing
    browser_profile: x-growth       # optional: browser tools use this profile once the skill is read
    validate:                       # optional: how to check it works (sentient/skills/checks.py, #386)
      - file: notes/weekly-review.md
        expect: "## Next week"
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

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sentient import paths
from sentient.skills import checks as skill_checks
from sentient.skills.lint import lint_file
from sentient.store.db import now_iso

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
RESERVED = {"pending", "archived"}
REFERENCE_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml"}
MAX_REFERENCE_FILES = 50
_UNSET: Any = object()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:64]


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name)) and name not in RESERVED


def content_hash(text: str) -> str:
    """Hash of a SKILL.md as written, ignoring line endings and surrounding blank space."""
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()


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
    # how to check that it works (``validate`` in the header, ``sentient.skills.checks``)
    validate: list[dict] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def references(self) -> list[str]:
        """The skill's reference files (``references/setup.md``, #167): the text files in its ``references`` folder,
        which ``skill_view`` reads one at a time when the skill is used. Dotfiles and links are left out."""
        out: list[str] = []
        base = self.dir / "references"
        if not base.is_dir() or base.is_symlink():
            return out
        root = self.dir.resolve()
        for f in sorted(base.rglob("*")):
            rel = f.relative_to(self.dir)
            if any(part.startswith(".") for part in rel.parts) or f.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            if f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root):
                out.append(rel.as_posix())
        return out[:MAX_REFERENCE_FILES]

    def reference_path(self, rel: str) -> Path | None:
        """The reference file ``rel`` names, or None when it isn't one of the skill's reference files."""
        want = str(rel or "").strip().replace("\\", "/").removeprefix("./")
        if want and "/" not in want:
            want = f"references/{want}"
        return self.dir / want if want in self.references else None

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
            "validate": self.validate,
            "references": self.references,
            "check": SkillLibrary.check_of(stats, pending=self.state == "pending_review") if self.validate else None,
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
        validate=skill_checks.parse(meta.get("validate"))[0],
    )


class _Dumper(yaml.SafeDumper):
    """Writes a multi-line value (a check's code) as a readable ``|`` block."""


def _str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value.strip() else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


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
    validate: list[dict] | None = None,
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
    if validate:
        meta["validate"] = validate
    front = yaml.dump(meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{body.strip()}\n"


class SkillLibrary:
    def __init__(self, dirs: list[Path] | None = None, store: Any = None):
        self.dirs = dirs or [paths.skills_dir()]
        self.store = store  # set by the evolution service; used for skill_stats
        self._skills: dict[str, Skill] = {}
        self._available: set[str] | None = None

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
        validate: list[dict] | None = None,
    ) -> Path:
        """Create a skill. With ``pending`` and an active skill of the same name, this stores a
        proposed update next to it (see ``diff``). ``validate``: its checks (None keeps the active skill's)."""
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
                validate=validate if validate is not None else (current.validate if current else None),
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
            # a proposal holds only its SKILL.md: keep the skill's other files (scripts, references) with it
            for f in dst.rglob("*"):
                if f.is_file() and not (src / f.relative_to(dst)).exists():
                    (src / f.relative_to(dst)).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, src / f.relative_to(dst))
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
                validate=skill.validate,
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
    async def stats(self, store: Any = None) -> dict[str, dict]:
        store = store or self.store
        if store is None:
            return {}
        rows = await store.fetchall("SELECT * FROM skill_stats")
        return {r["name"]: dict(r) for r in rows}

    async def sync_stats(self, store: Any = None) -> None:
        """Make skill_stats mirror what is on disk (one row per known skill)."""
        store = store or self.store
        if store is None:
            return
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
        for s in self._list_dir(self.root):  # first sight of an active skill: what is on disk is the baseline
            if not (existing.get(s.name) or {}).get("origin_hash"):
                await self.record_origin(s.name, store)

    async def mark_used(self, name: str, store: Any = None, *, viewed: bool = True, when: datetime | None = None) -> None:
        store = store or self.store
        if store is None:
            return
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
        await store.execute(
            "INSERT INTO skill_stats(name, state, created_at) VALUES(?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET state = excluded.state",
            (name.strip().lower(), state, now_iso()),
        )

    async def record_patch(self, name: str, state: str, store: Any = None) -> None:
        store = store or self.store
        if store is None:
            return
        await store.execute(
            "INSERT INTO skill_stats(name, patch_count, state, created_at) VALUES(?, 1, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET patch_count = patch_count + 1,"
            " state = CASE WHEN skill_stats.state IN ('active', 'stale') AND excluded.state = 'pending_review'"
            " THEN skill_stats.state ELSE excluded.state END",
            (name.strip().lower(), state, now_iso()),
        )
        if state == "pending_review":  # remember which version of the active skill the proposal was drafted from
            active = self._file(self.root, name.strip().lower())
            base = content_hash(active.read_text(encoding="utf-8", errors="replace")) if active.exists() else None
            await store.execute("UPDATE skill_stats SET proposal_base_hash = ? WHERE name = ?", (base, name.strip().lower()))

    async def record_check(self, name: str, report: dict | None, store: Any = None, *, pending: bool = False) -> None:
        """Keep the last report of the skill's checks (None: it declares none) for the Skills screen: the pending
        proposal's with ``pending``, else the active skill's."""
        store = store or self.store
        if store is None:
            return
        column = "pending_check_report" if pending else "check_report"
        await store.execute(
            f"INSERT INTO skill_stats(name, state, created_at, {column}) VALUES(?, 'pending_review', ?, ?)"
            f" ON CONFLICT(name) DO UPDATE SET {column} = excluded.{column}",
            (name.strip().lower(), now_iso(), json.dumps(report) if report else None),
        )

    @staticmethod
    def check_of(stats: dict | None, *, pending: bool = False) -> dict | None:
        """The last check report kept in a ``skill_stats`` row (the pending proposal's with ``pending``), or None."""
        raw = (stats or {}).get("pending_check_report" if pending else "check_report")
        try:
            report = json.loads(raw) if raw else None
        except ValueError:
            return None
        return report if isinstance(report, dict) else None

    # ------------------------------------------------------------------ edit protection (#164)
    async def record_origin(self, name: str, store: Any = None, *, text: str | None = None) -> None:
        """Remember the active SKILL.md (or ``text``) as the version Sentient wrote. Call it after every write
        Sentient makes to an active skill, so only the user's own edits change the hash afterwards."""
        store = store or self.store
        name = name.strip().lower()
        if store is None:
            return
        if text is None:
            f = self._file(self.root, name)
            if not f.exists():
                return
            text = f.read_text(encoding="utf-8", errors="replace")
        await store.execute(
            "INSERT INTO skill_stats(name, state, created_at, origin_hash, origin_text) VALUES(?, 'active', ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET origin_hash = excluded.origin_hash, origin_text = excluded.origin_text",
            (name, now_iso(), content_hash(text), text),
        )

    async def edit_state(self, name: str, store: Any = None) -> dict:
        """``{user_edited, can_reset, protected}`` for the active skill. ``user_edited``: its file differs from the
        version Sentient last wrote (an edit in the Skills screen or in any editor). ``protected``: Sentient must
        propose changes instead of writing them (edited by the user, or the user's own skill)."""
        store = store or self.store
        name = name.strip().lower()
        f = self._file(self.root, name)
        if store is None or not f.exists():
            return {"user_edited": False, "can_reset": False, "protected": False}
        text = f.read_text(encoding="utf-8", errors="replace")
        row = await store.fetchone("SELECT origin_hash, origin_text FROM skill_stats WHERE name = ?", (name,))
        if row is None or not row["origin_hash"]:
            await self.record_origin(name, store, text=text)  # first sight: this is the baseline
            edited = can_reset = False
        else:
            edited = content_hash(text) != row["origin_hash"]
            can_reset = edited and bool(row["origin_text"])
        skill = parse_skill_file(f)
        own = skill is not None and skill.author == "user"
        return {"user_edited": edited, "can_reset": can_reset, "protected": edited or own}

    async def reset_to_original(self, name: str, store: Any = None) -> Skill:
        """Put back the version Sentient last wrote, undoing the user's edits. FileNotFoundError when there is
        no active skill or nothing to go back to; ValueError when the skill has no edits."""
        store = store or self.store
        name = name.strip().lower()
        f = self._file(self.root, name)
        if store is None or not f.exists():
            raise FileNotFoundError(name)
        state = await self.edit_state(name, store)
        if not state["user_edited"]:
            raise ValueError("this skill has no edits to undo")
        row = await store.fetchone("SELECT origin_text FROM skill_stats WHERE name = ?", (name,))
        if row is None or not row["origin_text"]:
            raise FileNotFoundError(name)
        f.write_text(row["origin_text"], encoding="utf-8")
        skill = parse_skill_file(f)
        if skill is None:
            raise FileNotFoundError(name)
        return skill

    async def proposal_state(self, name: str, store: Any = None) -> dict:
        """For a pending proposal: ``{user_edited, changed_since_proposal}``. ``changed_since_proposal`` means the
        active skill changed after the proposal was drafted, so approving would undo those changes."""
        store = store or self.store
        name = name.strip().lower()
        f = self._file(self.root, name)
        if store is None or not f.exists():
            return {"user_edited": False, "changed_since_proposal": False}
        state = await self.edit_state(name, store)
        row = await store.fetchone("SELECT proposal_base_hash FROM skill_stats WHERE name = ?", (name,))
        base = row["proposal_base_hash"] if row else None
        current = content_hash(f.read_text(encoding="utf-8", errors="replace"))
        return {"user_edited": state["user_edited"], "changed_since_proposal": bool(base) and base != current}

    # ------------------------------------------------------------------ lint (#164)
    def lint(self, skill: Skill) -> list[dict]:
        return lint_file(skill.path, known_tools=self._available)

    def broken(self) -> list[dict]:
        """SKILL.md files in the writable library that can't load, with what is wrong (for the Skills screen)."""
        out = []
        for base, state in ((self.root, "active"), (self.pending_dir, "pending_review"), (self.archived_dir, "archived")):
            if not base.exists():
                continue
            for f in sorted(base.glob("*/SKILL.md")):
                if f.parent.name in RESERVED or parse_skill_file(f) is not None:
                    continue
                problems = lint_file(f, known_tools=self._available)
                if not any(p["level"] == "error" for p in problems):
                    problems.insert(0, {"rule": "required", "level": "error", "message": "Sentient can't load this skill."})
                out.append({"folder": f.parent.name, "state": state, "path": str(f), "lint": problems})
        return out

    async def catalog(self, store: Any = None) -> dict[str, list[dict]]:
        """Shape of GET /api/skills."""
        stats = await self.stats(store)
        active = {s.name: s for s in self._list_dir(self.root)}
        for s in self.list():
            active.setdefault(s.name, s)
        active_out = []
        for n, s in sorted(active.items()):
            item = s.to_dict(stats.get(n))
            if s.dir.parent == self.root:
                edit = await self.edit_state(n, store)
                item.update(user_edited=edit["user_edited"], can_reset=edit["can_reset"])
            item["lint"] = self.lint(s)
            active_out.append(item)
        pending_out = []
        for s in self.list_pending():
            item = s.to_dict(stats.get(s.name))
            item.update(await self.proposal_state(s.name, store))
            item["lint"] = self.lint(s)
            pending_out.append(item)
        return {
            "active": active_out,
            "pending": pending_out,
            "archived": [{**s.to_dict(stats.get(s.name)), "lint": self.lint(s)} for s in self.list_archived()],
            "broken": self.broken(),
        }
