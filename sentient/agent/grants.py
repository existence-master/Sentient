"""Scoped approval grants (#139, ADR 0016 amendment): how wide a "yes" to an approval request is.

When Sentient asks before a tool call, the person picks one of these answers (``Decision`` strings):

- ``allow``: this call only ("Allow once").
- ``allow_session``: the rest of this chat (``ApprovalBroker._session_allow``, unchanged from ADR 0008).
- ``allow_task``: later calls of the same tool in the same task, until the grant is revoked or the task deleted.
- ``allow_timed``: later calls of the same tool anywhere, for ``tools.approvals.grant_minutes`` ("Allow for 1 hour").
- ``allow_always``: an Allow rule for the tool (``tools.approvals.rules``, ADR 0016), noted in ``rule_origins``.
- ``deny``.

A task or timed grant covers calls up to the risk the person approved, and only calls a grant may cover at all
(``grantable``): never a purchase, a call held because outside content was read (ADR 0018), a tool that asks every
time (``allow_for_chat=False``, the terminal), a call a ``risk_fn`` raised to ``send`` or ``exec``, or anything that
runs code (effective risk ``exec``). For those only "Allow once" and "Deny" are offered (``choices``), and the broker
refuses any other answer. Never rules, Read only connections and the unprompted-work limit (ADR 0017) are checked
before any grant, so a grant can never reopen them. Grants are checked in code; no model sees or changes them.

Grants live in the ``approval_grants`` table and in memory (``covers`` is synchronous and cheap). A timed grant is
dropped once it expires; Settings > Approvals lists the live ones and can revoke each.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sentient.store.db import new_id, now_iso
from sentient.tools.base import Risk, Tool
from sentient.tools.rules import rule_label

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

log = logging.getLogger(__name__)

ALLOW, ALLOW_SESSION, ALLOW_TASK, ALLOW_TIMED, ALLOW_ALWAYS, DENY = (
    "allow", "allow_session", "allow_task", "allow_timed", "allow_always", "deny",
)
DECISIONS = (ALLOW, ALLOW_SESSION, ALLOW_TASK, ALLOW_TIMED, ALLOW_ALWAYS, DENY)
ALLOWING = frozenset(DECISIONS) - {DENY}
SCOPES = ("task", "timed")  # what a stored grant can be

_DENY_WORDS = frozenset({"no", "n", "deny", "denied", "dont", "don't", "stop", "cancel", "decline", "never", "nope"})
_ONCE_WORDS = frozenset({"yes", "y", "ok", "okay", "allow", "once", "go", "ahead", "sure", "approve", "it", "just"})
_WORD = re.compile(r"[a-z']+")


def duration_label(minutes: int) -> str:
    """ "1 hour", "2 hours", "30 minutes"."""
    minutes = max(1, int(minutes))
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def decision_label(decision: str, minutes: int = 60) -> str:
    """The plain words for a choice, as buttons and task question options show it."""
    return {
        ALLOW: "Allow once",
        ALLOW_SESSION: "Allow for this chat",
        ALLOW_TASK: "Allow for this task",
        ALLOW_TIMED: f"Allow for {duration_label(minutes)}",
        ALLOW_ALWAYS: "Always allow",
        DENY: "Deny",
    }.get(decision, decision)


def grantable(tool: Tool, risk: Risk, *, purchase: bool = False, untrusted: bool = False) -> bool:
    """True when a grant wider than one call may cover this call (see the module docstring)."""
    risk = Risk(risk)
    if purchase or untrusted or not getattr(tool, "allow_for_chat", True):
        return False
    if risk >= Risk.exec:  # running code keeps asking each time (or an Allow rule set in Settings)
        return False
    return not (risk >= Risk.send and risk > Risk(tool.risk))  # a click raised to "Place order" asks every time


def choices(
    tool: Tool, risk: Risk, *, purchase: bool = False, untrusted: bool = False, chat: bool = False,
    task: bool = False, remember_session: bool = True,
) -> list[str]:
    """The answers offered for one approval request, in button order."""
    if not grantable(tool, risk, purchase=purchase, untrusted=untrusted):
        return [ALLOW, DENY]
    out = [ALLOW]
    if chat and remember_session:
        out.append(ALLOW_SESSION)
    if task:
        out.append(ALLOW_TASK)
    return [*out, ALLOW_TIMED, ALLOW_ALWAYS, DENY]


def parse_answer(text: str, offered: list[str], minutes: int = 60) -> str | None:
    """The decision a typed answer means, or None when it is unclear or not one of ``offered``. Exact option text
    ("Allow for this task") or a key word: no/deny/stop, always, task, hour/minutes/while, yes/once/ok."""
    said = " ".join(str(text or "").lower().split()).strip(" .!")
    for decision in offered:
        if said == decision_label(decision, minutes).lower() or said == decision:
            return decision
    words = set(_WORD.findall(said))
    if not words:
        return None
    if words & _DENY_WORDS:
        found = DENY
    elif "always" in words:
        found = ALLOW_ALWAYS
    elif "task" in words:
        found = ALLOW_TASK
    elif "chat" in words:
        found = ALLOW_SESSION
    elif words & {"hour", "hours", "minute", "minutes", "while"}:
        found = ALLOW_TIMED
    elif words <= _ONCE_WORDS:
        found = ALLOW
    else:
        return None
    return found if found in offered else None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class GrantStore:
    """Live task and timed grants, mirrored in the ``approval_grants`` table."""

    def __init__(self, app: SentientApp):
        self.app = app
        self._grants: dict[str, dict[str, Any]] = {}
        self.clock = lambda: datetime.now(UTC)

    async def load(self) -> None:
        rows = await self.app.store.fetchall("SELECT * FROM approval_grants")
        self._grants = {r["id"]: dict(r) for r in rows}
        await self.purge()

    # ------------------------------------------------------------------ checks
    def _live(self, grant: dict[str, Any], now: datetime) -> bool:
        expires = _parse_time(grant.get("expires_at"))
        return expires is None or expires > now

    def covers(self, tool_name: str, risk: Risk, task_id: str | None = None) -> dict[str, Any] | None:
        """The grant that lets this call run without asking, or None. Synchronous: the broker calls it per call."""
        now = self.clock()
        level = int(Risk(risk))
        for grant in self._grants.values():
            if grant["tool"] != tool_name or int(grant["risk"]) < level or not self._live(grant, now):
                continue
            if grant["scope"] == "timed" or (grant["scope"] == "task" and task_id and grant["task_id"] == task_id):
                return grant
        return None

    # ------------------------------------------------------------------ changes
    async def add(
        self, tool: str, scope: str, risk: Risk, *, task_id: str | None = None, via: str = "desktop",
        minutes: int | None = None,
    ) -> dict[str, Any]:
        """Save a grant (a wider one for the same tool and scope replaces the old one)."""
        if scope not in SCOPES:
            raise ValueError(f"unknown grant scope {scope!r}")
        if scope == "task" and not task_id:
            raise ValueError("a task grant needs a task")
        now = self.clock()
        expires = None
        if scope == "timed":
            span = int(minutes if minutes is not None else self.app.config.tools.approvals.grant_minutes)
            expires = (now + timedelta(minutes=max(1, span))).isoformat()
        level = int(Risk(risk))
        for old in list(self._grants.values()):  # one grant per tool and scope (and task): keep the widest
            if old["tool"] == tool and old["scope"] == scope and old.get("task_id") == task_id:
                level = max(level, int(old["risk"])) if self._live(old, now) else level
                await self._delete(old["id"])
        grant = {
            "id": new_id(), "tool": tool, "scope": scope, "task_id": task_id, "risk": level,
            "via": via or "desktop", "created_at": now.isoformat(), "expires_at": expires,
        }
        await self.app.store.execute(
            "INSERT INTO approval_grants(id, tool, scope, task_id, risk, via, created_at, expires_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (grant["id"], tool, scope, task_id, level, grant["via"], grant["created_at"], expires),
        )
        self._grants[grant["id"]] = grant
        self._publish()
        return grant

    async def allow_always(self, tool_name: str, via: str = "desktop") -> None:
        """"Always allow": an Allow rule for this tool, saved like a rule set in Settings, with a note saying it came
        from an answer to an approval request."""
        from sentient.config.schema import RuleOrigin

        cfg = self.app.config.model_copy(deep=True)
        approvals = cfg.tools.approvals
        rules, origins = dict(approvals.rules), dict(approvals.rule_origins)
        rules[tool_name] = "allow"
        origins[tool_name] = RuleOrigin(rule="allow", source="approval", via=via or "desktop", at=now_iso())
        approvals.rules, approvals.rule_origins = rules, origins
        self.app.save_config(cfg)
        # the rule now covers it: drop narrower grants for the same tool so Settings shows one thing
        for grant in [g for g in self._grants.values() if g["tool"] == tool_name]:
            await self._delete(grant["id"])
        self._publish()

    async def revoke(self, grant_id: str) -> bool:
        if grant_id not in self._grants:
            return False
        await self._delete(grant_id)
        self._publish()
        return True

    async def revoke_task(self, task_id: str) -> int:
        """Drop every grant of a task (the task was deleted)."""
        gone = [g["id"] for g in self._grants.values() if g.get("task_id") == task_id]
        for gid in gone:
            await self._delete(gid)
        if gone:
            self._publish()
        return len(gone)

    async def purge(self) -> int:
        """Drop expired grants. Returns how many went."""
        now = self.clock()
        gone = [g["id"] for g in self._grants.values() if not self._live(g, now)]
        for gid in gone:
            await self._delete(gid)
        if gone:
            self._publish()
        return len(gone)

    async def _delete(self, grant_id: str) -> None:
        self._grants.pop(grant_id, None)
        await self.app.store.execute("DELETE FROM approval_grants WHERE id = ?", (grant_id,))

    def _publish(self) -> None:
        self.app.bus.publish("approval_grants.updated", {"count": len(self._grants)})

    # ------------------------------------------------------------------ listing
    async def list(self) -> list[dict[str, Any]]:
        """Live grants for Settings, newest first, with plain labels."""
        await self.purge()
        registry = self.app.registry
        out: list[dict[str, Any]] = []
        for grant in sorted(self._grants.values(), key=lambda g: g["created_at"], reverse=True):
            tool = registry.get(grant["tool"])
            plugin = registry.plugin(tool.plugin) if tool is not None else None
            task_name = None
            if grant["scope"] == "task" and grant.get("task_id"):
                task = await self.app.tasks.repo.get_task(grant["task_id"])
                if task is None:  # the task is gone: so is its grant
                    await self._delete(grant["id"])
                    continue
                task_name = task.get("name") or "Untitled task"
            out.append({
                "id": grant["id"],
                "tool": grant["tool"],
                "label": rule_label(tool, None, registry) if tool is not None else grant["tool"],
                "app": getattr(plugin, "display_name", None) or (tool.plugin if tool is not None else None),
                "scope": grant["scope"],
                "task_id": grant.get("task_id"),
                "task_name": task_name,
                "risk": Risk(int(grant["risk"])).name,
                "via": grant.get("via") or "desktop",
                "created_at": grant["created_at"],
                "expires_at": grant.get("expires_at"),
            })
        return out

