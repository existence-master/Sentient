"""Skill tools: progressive disclosure plus the write path that lets the assistant
teach itself repeatable procedures (staged for review when configured)."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit

from sentient.evolution.log import log_event
from sentient.skills import checks as skill_checks
from sentient.skills import vet
from sentient.skills.lint import has_danger, lint_text
from sentient.skills.loader import render_skill, slugify, valid_name
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import OUTSIDE_KEY

log = logging.getLogger(__name__)

INCOGNITO_REFUSAL = "This is an incognito chat, so nothing from it is saved as a skill. Carry on without saving."
MAX_REFERENCE_READ = 40_000


def _lib(ctx: ToolContext):
    return ctx.extra.get("skills")


@tool("skill_view", risk=Risk.read)
async def skill_view(ctx: ToolContext, name: str, file: str | None = None) -> dict:
    """Read the full procedure of a skill listed in your prompt. Do this before following a skill.
    Some skills list reference files: read one only when the task needs it, by calling skill_view again with
    `file` set to its path (for example references/setup.md)."""
    lib = _lib(ctx)
    if lib is None:
        return {"error": "skills unavailable"}
    s = lib.get(name)
    if not s:
        return {"error": f"no skill named {name}", "available": [k.name for k in lib.list()][:30]}
    if file:
        return await _reference(ctx, s, file)
    if ctx.store is not None and not await ctx.is_incognito():  # an incognito chat leaves no usage trail (#154)
        await lib.mark_used(s.name, ctx.store, viewed=True)
    out = {"name": s.name, "description": s.description, "version": s.version, "body": s.body}
    if refs := s.references:  # loaded one at a time, only when needed (#167)
        out["references"] = refs
    if s.browser_profile:  # the skill's browser work happens in its own profile (docs/API.md section 12)
        ctx.extra["browser_profile"] = s.browser_profile
        out["browser_profile"] = s.browser_profile
    return out


async def _reference(ctx: ToolContext, skill, file: str) -> dict:
    """One reference file of a skill (#167). A skill from outside (learned from docs, added from a link, brought over
    from Hermes) keeps text someone else wrote there, so its file comes back as outside content (ADR 0018): the run is
    marked and anything that sends data out asks first."""
    path = skill.reference_path(file)
    if path is None:
        return {"error": f"{skill.name} has no reference file {file}", "references": skill.references}
    limit = MAX_REFERENCE_READ if ctx.result_limit <= 0 else min(MAX_REFERENCE_READ, max(2000, ctx.result_limit - 500))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": f"{file} can't be read ({exc.strerror or exc})"}
    rel = path.relative_to(skill.dir).as_posix()
    out: dict = {}
    source = await _outside_source(ctx, skill)
    if source:
        out[OUTSIDE_KEY] = source  # first, so a result cut for length keeps it
    out.update(name=skill.name, file=rel, content=text[:limit])
    if len(text) > limit:
        out["truncated"] = True
    log.info("skill %s: read its reference file %s (%d characters)", skill.name, rel, len(text))
    return out


async def _outside_source(ctx: ToolContext, skill) -> str:
    if ctx.store is None:
        return ""
    rec = await vet.get_import(ctx.store, skill.name)
    if rec is None:
        return ""
    where = str(rec.get("origin") or "")
    short = urlsplit(where).hostname if where.startswith(("http://", "https://")) else Path(where).name
    return f"{short or 'its source'}, through the skill {skill.name}"


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
    if await ctx.is_incognito():
        return {"error": INCOGNITO_REFUSAL}
    name = name.strip().lower()
    if not valid_name(name):
        name = slugify(name)
    if not valid_name(name):
        return {"error": "skill name must be lowercase letters, digits and dashes (2-64 chars)"}
    existing = lib.get_active_file(name)
    pending = bool(ctx.config.skills.write_approval)
    # never overwrite a skill the user edited or wrote: propose the change instead (#164)
    protected = existing is not None and (await lib.edit_state(name, ctx.store))["protected"]
    findings = lint_text(
        render_skill(name, description, body, author="assistant", version=1, tags=tags, requires_tools=requires_tools)
    )
    risky = has_danger(findings)  # instructions to skip approvals, wipe files... always wait for the user
    pending = pending or protected or risky
    app = ctx.extra.get("app")
    report = None
    # the skill's own checks come first (#386): the active skill's and a waiting proposal's, none dropped
    proposal = lib.get_pending(name)
    validate = skill_checks.merge(existing.validate if existing else [], proposal.validate if proposal else [])
    if validate and app is not None:
        draft = render_skill(name, description, body, author="assistant", version=1, tags=tags,
                             requires_tools=requires_tools, validate=validate)
        folders = [existing.dir] if existing is not None else []
        report = await skill_checks.run(app, name, validate, folders, run_code=False, candidate=draft)
        if reason := skill_checks.failure(report):
            return {"error": f"The skill's own check failed, so it wasn't saved: {reason}. Fix the steps and try again."}
        pending = pending or report["status"] == "waiting"  # a code check runs when the user approves it
    path = lib.write(
        name, description, body, author=existing.author if existing else "assistant",
        tags=tags, requires_tools=requires_tools, pending=pending, validate=validate,
    )
    lib.reload()
    state = "pending_review" if pending else "active"
    action = "patched" if existing else "created"
    if ctx.store is not None:
        if not pending:
            await lib.record_origin(name, ctx.store)
        await lib.record_patch(name, state, ctx.store)
        if report is not None:
            await lib.record_check(name, report, ctx.store, pending=pending)
        detail = {"name": name, "pending": pending, "via": "skill_save", "session_id": ctx.session_id}
        if protected:
            detail["user_edited"] = True
        await log_event(ctx.store, f"skill_{action}", detail)
    if app is not None:
        app.bus.publish("skill.updated", {"name": name, "state": state})
        if pending:
            verb = "an update to the skill" if existing else "a new skill"
            message = f"Sentient wrote {verb} **{name}** and is waiting for your review."
            if protected:
                message = (
                    f"You changed **{name}** yourself, so Sentient didn't overwrite it. "
                    "It drafted an update instead. Review the change in Skills."
                )
            elif risky:
                message += " Check it carefully: it asks for something risky."
            await app.notify(
                "skill",
                message,
                title="Skill to review",
                payload={"skill": name, "action": "patch" if existing else "create", **({"user_edited": True} if protected else {})},
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
    if protected:
        out["note"] = (
            "The user edited this skill, so it was not changed. Your version waits for their review. "
            "Carry on with the current skill."
        )
    problems = [f["message"] for f in findings]
    if problems:
        out["problems"] = problems[:6]
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
