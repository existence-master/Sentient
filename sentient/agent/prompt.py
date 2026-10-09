"""System prompt assembly.

Everything the model needs to *be* the user's assistant is injected here every
turn: persona, who the user is, curated notes, recalled facts, the user model,
the skill index, and the clock. The legacy system relied on the model calling a
memory tool before it knew anything; here memory is pushed, and tools are the fallback.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from sentient.tools.builtin.time_tool import resolve_tz

TOOL_RULES = """## Working rules
- If a request can be done with a tool, do it and report the outcome briefly. Do not narrate what you are about to do.
- Never invent a tool result. If a tool fails, say what failed and try a sensible alternative once.
- Ask at most one clarifying question, and only when the answer changes what you would do.
- When you learn something lasting about the user (people, preferences, plans, routines), call memory_remember with one third-person fact. Do not announce that you did.
- For "what did we talk about" questions use history_semantic_search (by meaning) or history_time_search (by date range); for things from an imported document use memory_search_by_source.
- For anything that should happen later, repeatedly, or when something happens ("every morning", "when I get an email from"), create a task with create_task_from_prompt instead of doing it now.
- For multi-step procedures listed under Skills, call skill_view first and follow it.
- If you discover a repeatable procedure after some trial and error, offer to save it with skill_save.
- If the user adds a message while you are working, take it into account from that point on.
- Keep replies short. Plain language. Match the user's tone.
"""

# Extra rules shown only when the matching tools exist (prefix match on tool names).
CAPABILITY_RULES: tuple[tuple[str, str], ...] = (
    (
        "execute_code",
        "- Use execute_code for arithmetic, data wrangling, parsing or converting files, and jobs that call the same "
        "tool many times (scripts can call your tools). Do not use it for something one tool call already does.",
    ),
    (
        "browser_",
        "- Use the browser_* tools only for websites that have no integration. Read the page with browser_snapshot "
        "before clicking. Never type passwords, card numbers or one-time codes: when a site needs signing in, ask the "
        "user to sign in themselves with the Open browser button, then continue.",
    ),
    (
        "delegate_task",
        "- For work with many steps or several independent parts (research across sources, comparing options), use "
        "delegate_task or delegate_tasks and relay the summary. Put everything the subagent needs in goal and context: "
        "it cannot see this chat. Use background=true for long work the user should not wait for. Do small things yourself.",
    ),
    (
        "device_",
        "- The device_* tools reach the user's phone, glasses and other devices: take a photo or capture the screen when "
        "they ask about what they are looking at, get their location, or show, notify or speak something there.",
    ),
)


CHANNEL_GUIDANCE = {
    "voice": (
        "## You are speaking out loud\n"
        "- Your reply is converted to speech. Use short, natural spoken sentences.\n"
        "- No markdown, lists, tables, code, emoji or URLs. Say numbers and times the way a person would.\n"
        "- Keep it to one to three sentences unless asked for detail; offer to send details to the screen.\n"
    ),
    "glasses": (
        "## You are on the user's smart glasses\n"
        "- Replies are spoken or shown on a tiny display: at most two short sentences.\n"
        "- No markdown, lists, URLs or emoji.\n"
    ),
    "phone": (
        "## You are speaking through the user's phone\n"
        "- Your reply is converted to speech. Use short, natural spoken sentences.\n"
        "- No markdown, lists, tables, code, emoji or URLs. Say numbers and times the way a person would.\n"
        "- Keep it to one to three sentences unless asked for detail; offer to send details to the desktop app.\n"
    ),
}

_MESSAGING_GUIDANCE = (
    "## You are chatting in {app}\n"
    "- The user reads replies on a phone: keep them short and easy to scan, with short paragraphs or brief bullet lists.\n"
    "- No wide tables and no long code blocks; links are fine.\n"
    "- No LaTeX, since {app} shows it as raw symbols. Write math as plain text, like 2^100.\n"
    "- Images and files you create are sent as attachments, so mention them by name instead of pasting their contents.\n"
)
CHANNEL_GUIDANCE["telegram"] = _MESSAGING_GUIDANCE.format(app="Telegram")
CHANNEL_GUIDANCE["discord"] = _MESSAGING_GUIDANCE.format(app="Discord")
CHANNEL_GUIDANCE["whatsapp"] = _MESSAGING_GUIDANCE.format(app="WhatsApp")


def capability_rules(tool_names: Iterable[str] | None) -> str:
    if not tool_names:
        return ""
    names = list(tool_names)
    lines = [rule for prefix, rule in CAPABILITY_RULES if any(n.startswith(prefix) for n in names)]
    return "\n".join(lines)


def build_system_prompt(
    *,
    snapshot: dict[str, str],
    facts: list[dict],
    skills_index: str,
    assistant_name: str,
    user_name: str,
    timezone: str,
    channel: str,
    location: str = "",
    user_context: str = "",
    tool_names: Iterable[str] | None = None,
) -> str:
    tz = resolve_tz(timezone)
    now = datetime.now(tz)
    parts: list[str] = []

    soul = snapshot.get("soul") or f"You are {assistant_name}, a personal assistant."
    parts.append(soul)

    if snapshot.get("user"):
        parts.append(snapshot["user"])
    if snapshot.get("memory"):
        parts.append(snapshot["memory"])
    notes = "\n\n".join(x for x in (snapshot.get("yesterday"), snapshot.get("today")) if x)
    if notes:
        parts.append("## Recent daily notes\n" + notes)

    if facts:
        lines = "\n".join(f"- {f['content']} (id {f['id']})" for f in facts)
        parts.append("## Things you remember that may be relevant now\n" + lines)

    user_context = (user_context or "").strip()
    if user_context:
        if not user_context.startswith("#"):
            user_context = "## How the user likes things\n" + user_context
        parts.append(user_context)

    if skills_index:
        parts.append("## Skills (call skill_view to read one before following it)\n" + skills_index)

    parts.append(
        "## Context\n"
        f"- Now: {now.strftime('%A %Y-%m-%d %H:%M')} ({now.tzinfo})\n"
        f"- User: {user_name or 'unknown (ask once, then remember)'}\n"
        f"- User location: {location or 'unknown'}\n"
        f"- Channel: {channel}\n"
        f"- Assistant name: {assistant_name}"
    )
    rules = TOOL_RULES
    extra = capability_rules(tool_names)
    if extra:
        rules = rules.rstrip("\n") + "\n" + extra + "\n"
    parts.append(rules)
    if channel in CHANNEL_GUIDANCE:
        parts.append(CHANNEL_GUIDANCE[channel])
    return "\n\n".join(parts)
