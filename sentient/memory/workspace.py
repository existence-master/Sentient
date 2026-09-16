"""The human-readable memory tier: markdown files the user can open and edit.

- SOUL.md   : who the assistant is (persona, tone, values). Editable = configurable personality.
- USER.md   : who the user is, in the user's own words. Seeded by setup.
- MEMORY.md : curated long-term notes the assistant maintains (budgeted).
- notes/YYYY-MM-DD.md : daily notes; today and yesterday are loaded each turn.

Every file has a character budget so the system prompt stays predictable and
prompt caching stays effective (the snapshot is read once per turn, not per
model call).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from sentient import paths

DEFAULT_SOUL = """# Soul

You are {name}, a personal assistant that lives on {user}'s own computer.

## How you behave
- Warm, direct, and brief. You talk like a capable friend, not a corporate helpdesk.
- You act. When a request can be done with your tools, do it, then report the result in one or two sentences.
- You ask exactly one clarifying question when a request is genuinely ambiguous; otherwise you make the sensible call and say what you assumed.
- You remember. Facts about {user} that come up in conversation are worth saving; use the memory tools without being asked.
- You never invent tool results. If something failed, say what failed and what you will try next.
- You respect approvals. Anything that sends, deletes, or spends waits for a yes.

## Voice
Plain language, short sentences, no filler, no emoji unless {user} uses them first.
"""

DEFAULT_USER = """# About {user}

(Sentient fills this in over time. You can edit it freely; it is read every turn.)

- Name: {user}
- Timezone: {timezone}
"""

DEFAULT_MEMORY = """# Long-term memory

(Curated notes the assistant keeps about ongoing projects, preferences, and commitments.
Short, bulleted, and pruned. Atomic facts live in the database; this file holds the big picture.)
"""


class Workspace:
    def __init__(self, root: Path | None = None, budget_chars: int = 6000):
        self.root = root or paths.workspace_dir()
        self.budget = budget_chars

    # ------------------------------------------------------------------ files
    @property
    def soul(self) -> Path:
        return self.root / "SOUL.md"

    @property
    def user(self) -> Path:
        return self.root / "USER.md"

    @property
    def memory(self) -> Path:
        return self.root / "MEMORY.md"

    def note(self, day: date) -> Path:
        return self.root / "notes" / f"{day.isoformat()}.md"

    def ensure_defaults(self, name: str, user: str, timezone: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "notes").mkdir(exist_ok=True)
        user = user or "the user"
        if not self.soul.exists():
            self.soul.write_text(DEFAULT_SOUL.format(name=name, user=user), encoding="utf-8")
        if not self.user.exists():
            self.user.write_text(DEFAULT_USER.format(user=user, timezone=timezone), encoding="utf-8")
        if not self.memory.exists():
            self.memory.write_text(DEFAULT_MEMORY, encoding="utf-8")

    # ------------------------------------------------------------------ reading
    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > self.budget:
            text = text[: self.budget] + "\n\n[... truncated to budget; trim this file ...]"
        return text

    def snapshot(self, today: date | None = None) -> dict[str, str]:
        today = today or date.today()
        return {
            "soul": self._read(self.soul),
            "user": self._read(self.user),
            "memory": self._read(self.memory),
            "today": self._read(self.note(today)),
            "yesterday": self._read(self.note(today - timedelta(days=1))),
        }

    def read_full(self, today: date | None = None) -> dict[str, str]:
        """Unbudgeted contents for the editor UI (the prompt uses ``snapshot``)."""
        today = today or date.today()

        def raw(p: Path) -> str:
            return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

        return {
            "soul": raw(self.soul),
            "user": raw(self.user),
            "memory": raw(self.memory),
            "today": raw(self.note(today)),
            "yesterday": raw(self.note(today - timedelta(days=1))),
        }

    # ------------------------------------------------------------------ writing
    def append_learned(self, facts: list[str], heading: str = "## Learned") -> list[str]:
        """Append new bullet facts under a heading in USER.md without touching the user's own text.

        Returns the facts actually appended (ones already present anywhere in the file are skipped).
        """
        text = self.user.read_text(encoding="utf-8", errors="replace") if self.user.exists() else ""
        lower = text.lower()
        new = []
        for f in facts:
            f = f.strip()
            if f and f.lower() not in lower and f not in new:
                new.append(f)
        if not new:
            return []
        bullets = "\n".join(f"- {f}" for f in new)
        lines = text.splitlines()
        idx = next((i for i, line in enumerate(lines) if line.strip().lower() == heading.lower()), None)
        if idx is None:
            body = text.rstrip() + ("\n\n" if text.strip() else "") + f"{heading}\n\n{bullets}\n"
        else:
            # insert at the end of the Learned section (before the next heading of same or higher level)
            end = len(lines)
            for j in range(idx + 1, len(lines)):
                if lines[j].startswith("#") and len(lines[j]) - len(lines[j].lstrip("#")) <= 2:
                    end = j
                    break
            section = lines[idx + 1 : end]
            while section and not section[-1].strip():
                section.pop()
            merged = [*lines[: idx + 1], *(section or [""]), *bullets.splitlines()]
            rest = lines[end:]
            body = "\n".join(merged) + ("\n\n" + "\n".join(rest) if rest else "") + "\n"
        self.user.parent.mkdir(parents=True, exist_ok=True)
        self.user.write_text(body, encoding="utf-8")
        return new

    def append_note(self, text: str, day: date | None = None) -> Path:
        day = day or date.today()
        p = self.note(day)
        p.parent.mkdir(parents=True, exist_ok=True)
        is_new = not p.exists() or p.stat().st_size == 0
        with p.open("a", encoding="utf-8") as fh:
            if is_new:
                fh.write(f"# {day.isoformat()}\n\n")
            fh.write(text.rstrip() + "\n")
        return p

    def write(self, which: str, content: str) -> Path:
        target = {"soul": self.soul, "user": self.user, "memory": self.memory}[which]
        target.write_text(content, encoding="utf-8")
        return target
