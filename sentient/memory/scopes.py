"""Memory scopes (#155): keep work, personal and project memories apart.

Every fact has a ``scope``: ``personal`` (the default, and where older memories went), ``work``, a project's own name
("Kestrel launch"), or ``always`` for the few things every chat should know (the user's name, how they like answers).
A chat (``sessions.memory_scope``), a task (``tasks.memory_scope``) or a messaging app
(``channels.<app>.memory_scope``) has a scope too. Recall then reads that scope plus ``always``, and saving puts a new
memory in it, so a work chat never brings up personal health or family memories unless the user asks
(``memory_recall(everywhere=true)``). Work not tied to a chat or task (proactive look-ups, dreaming, the user model)
reads every scope.

Personal chats keep the profile files (MEMORY.md, the learned part of USER.md, daily notes), which are built from
personal and always memories only; work and project chats leave them out.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

PERSONAL = "personal"
WORK = "work"
ALWAYS = "always"
BUILT_IN = (ALWAYS, PERSONAL, WORK)
LABELS = {ALWAYS: "Always", PERSONAL: "Personal", WORK: "Work"}
MAX_LEN = 40
# a chat, task or messaging app can't be "always": that scope is only for memories every chat reads
CHAT_SCOPES_EXCLUDE = frozenset({ALWAYS})
# insight dimensions a work or project chat leaves out of its prompt (family and friends are personal)
PERSONAL_DIMENSIONS = frozenset({"relationships"})
_LEARNED_RE = re.compile(r"^## Learned\s*$", re.IGNORECASE | re.MULTILINE)


def normalize(raw: Any, default: str | None = None) -> str | None:
    """A scope as stored: built-in names in lowercase ("Work" -> "work"), a project's name as typed with spaces
    tidied, at most 40 characters. ``default`` for an empty value."""
    text = " ".join(str(raw or "").split())[:MAX_LEN].strip()
    if not text:
        return default
    low = text.lower()
    return low if low in BUILT_IN else text


def label(scope: str | None) -> str:
    scope = normalize(scope, PERSONAL) or PERSONAL
    return LABELS.get(scope, scope)


def recall_scopes(scope: str | None) -> list[str] | None:
    """The scopes recall reads for a run in ``scope``: it plus ``always``. None (every scope) when ``scope`` is None."""
    if scope is None:
        return None
    scope = normalize(scope, PERSONAL) or PERSONAL
    return [ALWAYS] if scope == ALWAYS else [scope, ALWAYS]


def is_personal(scope: str | None) -> bool:
    return scope is None or normalize(scope, PERSONAL) == PERSONAL


def chat_scope(raw: Any) -> str | None:
    """A scope a chat, task or messaging app may use: None for "the default", ValueError for ``always``."""
    scope = normalize(raw)
    if scope in CHAT_SCOPES_EXCLUDE:
        raise ValueError("A chat or task can't use the Always scope. Pick Personal, Work or a project.")
    return scope


def channel_scope(config: Any, channel: str | None) -> str | None:
    """The scope set for a messaging app (``channels.<app>.memory_scope``), or None for other channels."""
    channels = getattr(config, "channels", None)
    app = getattr(channels, str(channel or ""), None) if channels is not None else None
    raw = getattr(app, "memory_scope", None)
    scope = normalize(raw) if isinstance(raw, str) else None
    return None if scope in CHAT_SCOPES_EXCLUDE else scope


def of_session(config: Any, session: dict | None) -> str:
    """A chat's scope: its own (``sessions.memory_scope``), else its messaging app's, else personal."""
    session = session or {}
    own = normalize(session.get("memory_scope"))
    if own and own not in CHAT_SCOPES_EXCLUDE:
        return own
    return channel_scope(config, session.get("channel")) or PERSONAL


def of_task(task: dict | None) -> str:
    scope = normalize((task or {}).get("memory_scope"))
    return scope if scope and scope not in CHAT_SCOPES_EXCLUDE else PERSONAL


async def for_context(ctx: Any) -> str | None:
    """The scope of a tool call: the run's (``ctx.memory_scope``, set for chat turns and task runs), else its chat's
    (a helper a chat started), else None: work not tied to a chat or task reads every scope."""
    scope = normalize(getattr(ctx, "memory_scope", None))
    if scope:
        return scope
    session_id = getattr(ctx, "session_id", None)
    get = getattr(getattr(ctx, "store", None), "get_session", None)
    if session_id and callable(get):
        try:
            return of_session(ctx.config, await get(session_id))
        except Exception:
            return PERSONAL
    return None


def without_learned(user_md: str) -> str:
    """USER.md without its "## Learned" section (facts Sentient appended from personal memories)."""
    m = _LEARNED_RE.search(user_md or "")
    if m is None:
        return user_md
    rest = user_md[m.end():]
    nxt = re.search(r"^#{1,2} ", rest, re.MULTILINE)
    return (user_md[: m.start()] + (rest[nxt.start():] if nxt else "")).strip()


def personal_summary_sql(config: Any) -> tuple[str, list[str]]:
    """SQL condition (with parameters) on ``summaries`` that keeps summaries of personal chats: the profile files never
    read a work or project chat's summary. A chat that still exists counts with its scope now (:func:`of_session`,
    messaging-app fallback included); for one that was deleted, the scope it had when the summary was written
    (``summaries.scope``) counts."""
    apps = [(app, scope) for app in ("telegram", "discord", "whatsapp") if (scope := channel_scope(config, app))]
    case = "CASE s.channel " + " ".join("WHEN ? THEN ?" for _ in apps) + " ELSE 'personal' END" if apps else "'personal'"
    params = [x for pair in apps for x in pair]
    sql = (
        "(CASE WHEN EXISTS (SELECT 1 FROM sessions s WHERE s.id = summaries.session_id)"
        " THEN NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = summaries.session_id"
        f" AND COALESCE(NULLIF(s.memory_scope, ''), {case}) NOT IN ('personal', 'always'))"
        " ELSE COALESCE(summaries.scope, 'personal') IN ('personal', 'always') END)"
    )
    return sql, params


def prompt_note(scope: str) -> str:
    name = label(scope)
    return (
        "## Memory scope\n"
        f"This chat uses the user's {name} memories plus the ones kept for every chat. Their personal memories (family, "
        "health, home) are left out. Don't guess about them; if the user asks about one, call memory_recall with "
        "everywhere=true."
    )


async def resolve(app: Any, raw: Any, default: str | None = None) -> str | None:
    """:func:`normalize`, then the spelling of a project already in use when one matches whatever the case ("atlas"
    is "Atlas" when a memory, chat, task, messaging app or folder uses "Atlas"), so one project keeps one name."""
    scope = normalize(raw, default)
    if not scope or scope in BUILT_IN:
        return scope
    want = scope.lower()
    return next((s["name"] for s in await list_scopes(app) if s["name"].lower() == want), scope)


async def list_scopes(app: Any) -> list[dict]:
    """``[{name, label, count, built_in}]``: Always, Personal and Work, then every project a memory, chat, task,
    messaging app or added folder (#392) uses (by name). ``count`` = active memories in it."""
    store = app.store
    counts: dict[str, int] = {}
    names: list[str] = []
    try:
        for r in await store.fetchall(
            "SELECT scope, COUNT(*) AS n FROM facts WHERE status = 'active' GROUP BY scope"
        ):
            name = normalize(r["scope"], PERSONAL) or PERSONAL
            counts[name] = counts.get(name, 0) + int(r["n"])
            names.append(name)
    except Exception:  # memory unavailable
        pass
    for sql in (
        "SELECT DISTINCT memory_scope AS s FROM sessions WHERE memory_scope IS NOT NULL",
        "SELECT DISTINCT memory_scope AS s FROM tasks WHERE memory_scope IS NOT NULL",
        "SELECT DISTINCT scope AS s FROM folders WHERE scope IS NOT NULL",
    ):
        with contextlib.suppress(Exception):  # an older database without the column
            names += [n for r in await store.fetchall(sql) if (n := normalize(r["s"]))]
    for app_name in ("telegram", "discord", "whatsapp"):
        if scope := channel_scope(app.config, app_name):
            names.append(scope)
    projects = sorted({n for n in names if n not in BUILT_IN}, key=str.lower)
    return [
        {"name": n, "label": label(n), "count": counts.get(n, 0), "built_in": n in BUILT_IN}
        for n in (*BUILT_IN, *projects)
    ]
