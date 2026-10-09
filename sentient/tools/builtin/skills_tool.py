"""Skill tools: progressive disclosure plus the write path that lets the assistant
teach itself repeatable procedures (staged for review when configured)."""

from __future__ import annotations

from sentient.evolution.log import log_event
from sentient.skills.loader import slugify, valid_name
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool


def _lib(ctx: ToolContext):
    return ctx.extra.get("skills")


@tool("skill_view", risk=Risk.read)
async def skill_view(ctx: ToolContext, name: str) -> dict:
    """Read the full procedure of a skill listed in your prompt. Do this before following a skill."""
    lib = _lib(ctx)
    if lib is None:
        return {"error": "skills unavailable"}
    s = lib.get(name)
    if not s:
        return {"error": f"no skill named {name}", "available": [k.name for k in lib.list()][:30]}
    if ctx.store is not None:
        await lib.mark_used(s.name, ctx.store, viewed=True)
    out = {"name": s.name, "description": s.description, "version": s.version, "body": s.body}
    if s.browser_profile:  # the skill's browser work happens in its own profile (docs/API.md section 12)
        ctx.extra["browser_profile"] = s.browser_profile
        out["browser_profile"] = s.browser_profile
    return out


@tool("skill_save", risk=Risk.write, internal=True)
async def skill_save(
    ctx: ToolContext,
    name: str,
    description: str,
    body: str,
    tags: list[str] | None = None,
    requires_tools: list[str] | None = None,
) -> dict:
    """Save a reusable skill you worked out: a multi-step procedure that succeeded after
    trial and error, or a workflow the user will want again. Saving an existing name patches it.
    Write the body as markdown with sections: When to use, Procedure (numbered), Pitfalls,
    Verification. The user may review it first."""
    lib = _lib(ctx)
    if lib is None:
        return {"error": "skills unavailable"}
    name = name.strip().lower()
    if not valid_name(name):
        name = slugify(name)
    if not valid_name(name):
        return {"error": "skill name must be lowercase letters, digits and dashes (2-64 chars)"}
    existing = lib.get_active_file(name)
    pending = bool(ctx.config.skills.write_approval)
    path = lib.write(
        name, description, body, author="assistant", tags=tags, requires_tools=requires_tools, pending=pending
    )
    lib.reload()
    state = "pending_review" if pending else "active"
    action = "patched" if existing else "created"
    app = ctx.extra.get("app")
    if ctx.store is not None:
        await lib.record_patch(name, state, ctx.store)
        await log_event(
            ctx.store, f"skill_{action}", {"name": name, "pending": pending, "via": "skill_save", "session_id": ctx.session_id}
        )
    if app is not None:
        app.bus.publish("skill.updated", {"name": name, "state": state})
        if pending:
            verb = "an update to the skill" if existing else "a new skill"
            await app.notify(
                "skill",
                f"Sentient wrote {verb} **{name}** and is waiting for your review.",
                title="Skill to review",
                payload={"skill": name, "action": "patch" if existing else "create"},
            )
    elif pending and ctx.store is not None:
        await ctx.store.add_notification(
            "skill", f"Sentient wrote a skill '{name}' and is waiting for your review.",
            title="Skill to review", payload={"skill": name},
        )
    out = {"saved": str(path), "name": name, "action": action, "pending_review": pending}
    if pending:
        out["note"] = (
            "Saved for the user's review. It is not active yet, so skill_view cannot load it. "
            "Carry on with the current work without it."
        )
    return out


class SkillsPlugin(ToolPlugin):
    id = "skills"
    display_name = "Skills"
    description = "Reusable procedures the assistant can follow and learn."
    category = "core"
    icon = "IconBook"
    selection_hint = "following or saving a documented procedure"
    tools = [skill_view, skill_save]


PLUGIN = SkillsPlugin()
