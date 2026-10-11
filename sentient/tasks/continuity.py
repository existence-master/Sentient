"""Recurring tasks remember their last runs (docs/API.md section 4, "Recurring tasks remember").

Each run of a recurring task sees a short note of what its last few runs found, sent or decided, plus the task's
notepad: a few lines the runs keep for each other ("told Maya about the Lisbon fare on 12 Oct; tell again only if it
drops below 30k"). The run reports only what is new instead of repeating itself every day.

Kept compact for small local models: at most ``EARLIER_RUNS`` runs of ``RUN_NOTE_CHARS`` each and a notepad of
``MAX_NOTEPAD_CHARS``. The note is the run's result summary, or its final answer when the report is not written yet.
The run updates its notepad with ``task_notepad`` (plugin ``task_notepad``, scoped: offered only inside runs of
recurring tasks).
"""

from __future__ import annotations

from datetime import tzinfo
from typing import Any

from sentient.tasks.schedule import parse_iso
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

EARLIER_RUNS = 3
RUN_NOTE_CHARS = 400
MAX_NOTEPAD_CHARS = 1500
NOTEPAD_TOOL = "task_notepad"
PLUGIN_ID = "task_notepad"
FINISHED = ("completed", "completed_with_errors")


def applies(task: dict) -> bool:
    """Runs of recurring tasks carry what came before. One-off and triggered runs stand alone."""
    return (task.get("schedule") or {}).get("type") == "recurring"


def clip(text: Any, limit: int) -> str:
    """One paragraph of at most ``limit`` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def clean_notepad(text: Any) -> str | None:
    """The stored notepad: trimmed lines, at most ``MAX_NOTEPAD_CHARS``; None when empty."""
    lines = [line.rstrip() for line in str(text or "").strip().splitlines()]
    out = "\n".join(lines).strip()
    if not out:
        return None
    return out if len(out) <= MAX_NOTEPAD_CHARS else out[: MAX_NOTEPAD_CHARS - 3].rstrip() + "..."


async def earlier_runs(repo: Any, task_id: str, current_run_id: str, tz: tzinfo) -> list[dict]:
    """``[{when, note}]`` for the last finished runs before ``current_run_id``, oldest first."""
    runs = [r for r in await repo.runs_for(task_id) if r["id"] != current_run_id and r["status"] in FINISHED]
    out: list[dict] = []
    for run in runs[-EARLIER_RUNS:]:
        result = run.get("result") if isinstance(run.get("result"), dict) else {}
        note = str(result.get("summary") or "").strip()
        if not note:  # the report is written after the run settles; its final answer says the same
            for update in reversed(await repo.events(run["id"], limit=20)):
                msg = (update or {}).get("message") or {}
                if msg.get("type") == "final_answer" and str(msg.get("content") or "").strip():
                    note = str(msg["content"])
                    break
        if not note:
            continue
        started = parse_iso(run.get("started_at") or run.get("created_at"))
        when = started.astimezone(tz).strftime("%a %d %b %H:%M") if started else "earlier"
        out.append({"when": when, "note": clip(note, RUN_NOTE_CHARS)})
    return out


def prompt_section(runs: list[dict], notepad: str | None) -> str:
    """The executor prompt's section about earlier runs (a one-line note for the first run)."""
    if not runs and not notepad:
        return (
            "**Earlier runs:** This is the first run of this recurring task. Before your final answer, call "
            "`task_notepad` once with a short note for the next run (what you reported, what is still open).\n\n"
        )
    parts = ["**Earlier runs of this recurring task (most recent last):**"]
    parts += [f"- {r['when']}: {r['note']}" for r in runs] or ["- (no finished runs yet)"]
    if notepad:
        parts.append("**Your notepad from earlier runs:**\n" + notepad)
    parts.append(
        "Do not repeat what earlier runs already reported. Report what is new or changed since then; mention an old "
        "item again only if it is still open and needs attention, and say it was reported before. If nothing is new, "
        "say so in one sentence. Before your final answer, call `task_notepad` once with the updated note for the next run."
    )
    return "\n".join(parts) + "\n\n"


@tool(NOTEPAD_TOOL, risk=Risk.write, internal=True)
async def task_notepad(ctx: ToolContext, text: str) -> dict:
    """Replace this recurring task's notepad: a few short lines the next run will read, such as what you reported
    today and what is still open. Keep it under 1500 characters; it replaces the previous notepad."""
    task_id = ctx.extra.get("task_id")
    if not task_id or not ctx.extra.get(PLUGIN_ID):
        return {"error": "The notepad only works inside a run of a recurring task."}
    from sentient.tasks.repo import TaskRepo

    note = clean_notepad(text)
    await TaskRepo(ctx.store).update_task(str(task_id), {"notepad": note})
    return {"saved": True, "next": "Notepad saved. Do not call it again; reply now with your final answer for the user."}


class TaskNotepadPlugin(ToolPlugin):
    id = PLUGIN_ID
    display_name = "Task notepad"
    description = "Lets a recurring task keep short notes for its next run."
    category = "core"
    icon = "IconNotes"
    scoped = True
    tools = [task_notepad]
