"""Seed realistic demo tasks into a Sentient home (no LLM calls).

    .venv/Scripts/python.exe desktop/scripts/seed-tasks.py <SENTIENT_HOME>

Writes straight through the tasks repository: one task per interesting state
(planning, approval, clarification, recurring with run history, triggered by an
email and by a calendar event, a live run, a swarm, completed with files, errored,
scheduled, archived) plus a couple of notifications that point at tasks.

The demo user is Maya Rao, a product designer at Northwind Studio in Bengaluru.
Every name, client and address in the data is made up.

The profile's models are pointed at a black-holed address (10.255.255.1) with a long
timeout. `sentient serve` always resumes interrupted runs and re-plans `planning` tasks
on start; with an unreachable model those calls just wait, so no LLM traffic is
generated and the demo states survive long enough for screenshots. Re-run the script
to reset the data (existing tasks are wiped first).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_API_BASE = "http://10.255.255.1:11434"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("home", nargs="?", default=os.environ.get("SENTIENT_HOME"),
                   help="folder to seed (created if missing); defaults to SENTIENT_HOME")
    p.add_argument("--user", default="Maya", help="user name written to config")
    p.add_argument("--timezone", default="Asia/Kolkata", help="assistant timezone written to config")
    p.add_argument("--theme", choices=["dark", "light", "system"], default=None, help="set ui.theme (for screenshots)")
    p.add_argument("--config-only", action="store_true", help="only update config.yaml (theme, models); keep tasks")
    args = p.parse_args()
    if not args.home:
        p.error("pass a folder or set SENTIENT_HOME")
    return args


ARGS = _parse_args()
os.environ["SENTIENT_HOME"] = str(Path(ARGS.home).expanduser().resolve())
sys.path.insert(0, str(REPO_ROOT))

from sentient import paths  # noqa: E402
from sentient.app import SentientApp  # noqa: E402
from sentient.config.loader import load_config, save_config  # noqa: E402
from sentient.config.schema import ProviderConfig  # noqa: E402
from sentient.tasks.schedule import calculate_next_run, get_tz, iso  # noqa: E402


class OfflineProvider:
    """Stands in for the LLM while seeding. Seeding never calls it."""

    def model_for(self, role: str) -> str:
        return f"offline/{role}"

    async def stream(self, role, messages, tools=None, *, model=None):  # pragma: no cover - never called
        raise RuntimeError("seed-tasks.py never calls the LLM")
        yield

    async def complete_text(self, role, messages, *, model=None) -> str:  # pragma: no cover
        raise RuntimeError("seed-tasks.py never calls the LLM")

    async def complete_json(self, role, messages, *, model=None):  # pragma: no cover
        raise RuntimeError("seed-tasks.py never calls the LLM")

    async def embed(self, texts, *, model=None):  # pragma: no cover
        return [[0.0] * 16 for _ in texts]


NOW = datetime.now(UTC).replace(microsecond=0)
TZ = get_tz(ARGS.timezone)


def ts(delta: timedelta = timedelta(0)) -> str:
    return iso(NOW + delta) or ""


def local_at(days: int, hour: int, minute: int = 0) -> datetime:
    """UTC instant for `hour:minute` local time, `days` from today (user timezone)."""
    local_today = NOW.astimezone(TZ).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return (local_today + timedelta(days=days)).astimezone(UTC)


def naive_local(days: int, hour: int, minute: int = 0) -> str:
    return local_at(days, hour, minute).astimezone(TZ).strftime("%Y-%m-%dT%H:%M")


def configure() -> Any:
    cfg = load_config()
    cfg.assistant.user_name = ARGS.user
    cfg.assistant.timezone = ARGS.timezone
    cfg.assistant.location = cfg.assistant.location or "Bengaluru, India"
    cfg.assistant.onboarding_complete = True
    cfg.models.roles.primary = "ollama_chat/qwen3:8b"
    cfg.models.roles.fast = "ollama_chat/qwen3:4b"
    cfg.models.roles.embedding = "ollama/nomic-embed-text"
    cfg.models.fallbacks = {}
    cfg.models.request_timeout_s = 3600
    for prefix in ("ollama", "ollama_chat"):
        cfg.models.providers[prefix] = ProviderConfig(api_base=OFFLINE_API_BASE)
    cfg.memory.extract_after_turn = False
    cfg.chat.auto_title = False
    if ARGS.theme:
        cfg.ui.theme = ARGS.theme
    save_config(cfg)
    return cfg


# ---------------------------------------------------------------------------- helpers
async def add_task(repo, task_id: str, **fields: Any) -> str:
    created = fields.pop("created_at", ts(timedelta(hours=-2)))
    base = {
        "id": task_id,
        "status": "approval_pending",
        "priority": 1,
        "task_type": "single",
        "assignee": "ai",
        "enabled": True,
        "plan": [],
        "chat_history": [],
        "clarifying_questions": [],
        "original_context": {"source": "manual_creation"},
        "created_at": created,
        "updated_at": fields.pop("updated_at", created),
    }
    base.update(fields)
    base.setdefault("original_prompt", base.get("description") or base.get("name"))
    source = base["original_context"].get("source", "manual_creation")
    base.setdefault("source", "user" if source == "manual_creation" else source)
    await repo.insert_task(base)
    return task_id


async def add_run(
    repo,
    task_id: str,
    *,
    started: datetime,
    events: list[tuple[int, dict]],
    status: str = "completed",
    plan: list | None = None,
    trigger: dict | None = None,
    result: dict | None = None,
    error: str | None = None,
    duration_s: int | None = None,
    memory_sources: list[dict] | None = None,
) -> str:
    run_id = await repo.insert_run(task_id, now=iso(started), plan=plan or [], trigger_data=trigger)
    for offset_s, message in events:
        await repo.add_event(run_id, message, iso(started + timedelta(seconds=offset_s)))
    fields: dict[str, Any] = {"status": status, "error": error}
    if status not in {"processing", "waiting_for_user"}:
        last = max((o for o, _ in events), default=0)
        fields["finished_at"] = iso(started + timedelta(seconds=duration_s or last + 4))
    if result is not None:
        fields["result"] = result
    if memory_sources:
        fields["memory_sources"] = memory_sources
    await repo.update_run(run_id, fields)
    return run_id


async def seeded_sources(repo, texts: list[str]) -> list[dict]:
    """Memory sources pointing at facts seed-memory-skills.py made (the first as in the prompt, the rest as looked
    up), so "This is wrong" edits real memories. Facts it has not made are left out."""
    out = []
    for n, text in enumerate(texts):
        try:
            row = await repo.store.fetchone("SELECT id, source FROM facts WHERE content = ?", (text,))
        except Exception:  # no memory table yet
            row = None
        if row:
            out.append({"kind": "fact", "id": int(row["id"]), "text": text, "source": row["source"],
                        "via": "prompt" if n == 0 else "tool"})
    return out


def info(text: str) -> dict:
    return {"type": "info", "content": text}


def thought(text: str) -> dict:
    return {"type": "thought", "content": text}


def call(tool: str, **params: Any) -> dict:
    return {"type": "tool_call", "tool_name": tool, "parameters": params}


def result_msg(tool: str, value: Any, is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_name": tool, "result": value, "is_error": is_error}


def final(text: str) -> dict:
    return {"type": "final_answer", "content": text}


def err(text: str) -> dict:
    return {"type": "error", "content": text}


# ---------------------------------------------------------------------------- the demo tasks
DIGEST_PLAN = [
    {"tool": "gmail", "description": "Fetch unread emails received since yesterday's digest"},
    {"tool": "gmail", "description": "Group them by sender and urgency and pick out anything that needs a reply"},
    {"tool": "files", "description": "Save the digest as a markdown note and summarise it here"},
]


def digest_events(day_label: str, unread: int, *, rate_limited: bool = False) -> list[tuple[int, dict]]:
    ev: list[tuple[int, dict]] = [
        (0, info("Executor has picked up the task and is starting execution.")),
        (3, thought(f"I need yesterday's unread mail. I'll search Gmail for `is:unread newer_than:1d`, then group the {unread} results by sender.")),
        (5, call("gmail_search", query="is:unread newer_than:1d", max_results=50)),
    ]
    if rate_limited:
        ev += [
            (7, result_msg("gmail_search", "429 Too Many Requests: User-rate limit exceeded. Retry after 30s.", is_error=True)),
            (9, thought("Gmail rate-limited the first search. I'll wait briefly and retry with a smaller page size.")),
            (40, call("gmail_search", query="is:unread newer_than:1d", max_results=20)),
        ]
    ev += [
        (
            44 if rate_limited else 8,
            result_msg(
                "gmail_search",
                {
                    "count": unread,
                    "messages": [
                        {"from": "Kavya Iyer <kavya@northwind.example>", "subject": "Re: Lumen Health onboarding, feedback round 2", "labels": ["IMPORTANT"]},
                        {"from": "Figma <notifications@figma.example>", "subject": "4 new comments on Lumen Health / Onboarding v3"},
                        {"from": "City Bank <alerts@citybank.example>", "subject": "Credit card statement for September"},
                    ],
                },
            ),
        ),
        (52 if rate_limited else 15, thought("Kavya's thread is the only one that needs a reply today. The Figma comments and the bank statement are FYI.")),
        (
            55 if rate_limited else 18,
            call("file_write", path=f"digests/inbox-{day_label}.md", content=f"# Inbox digest {day_label}\n\n- Reply to Kavya (onboarding feedback)\n- 4 new Figma comments\n- Card statement arrived"),
        ),
        (57 if rate_limited else 19, result_msg("file_write", {"ok": True, "path": f"digests/inbox-{day_label}.md", "bytes": 214})),
        (
            61 if rate_limited else 24,
            final(
                f"**{unread} unread emails** since yesterday.\n\n"
                "1. **Kavya Iyer (Northwind)** wants the revised onboarding screens for Lumen Health by Thursday. *Needs a reply.*\n"
                "2. **Figma**: 4 new comments on *Lumen Health / Onboarding v3*.\n"
                "3. **City Bank**: your September credit card statement is ready (due Oct 28)."
            ),
        ),
        (62 if rate_limited else 25, info("Execution finished. Generating final report...")),
    ]
    return ev


def digest_result(day_label: str, unread: int) -> dict:
    return {
        "summary": (
            f"### Inbox digest for {day_label}\n\n"
            f"You had **{unread} unread emails**. One needs a reply today:\n\n"
            "- **Kavya Iyer** asked for the revised Lumen Health onboarding screens by Thursday.\n\n"
            "FYI: 4 new Figma comments came in and your September card statement arrived."
        ),
        "links_created": [],
        "links_found": [
            {"url": "https://mail.example/inbox/18f2a9c", "description": "Re: Lumen Health onboarding, feedback round 2"},
            {"url": "https://figma.example/file/lumen-onboarding-v3", "description": "Lumen Health / Onboarding v3"},
        ],
        "files_created": [{"filename": f"digests/inbox-{day_label}.md", "description": "Markdown digest"}],
        "tools_used": ["gmail", "files"],
    }


async def seed(app: SentientApp) -> dict[str, str]:
    repo = app.tasks.repo
    for t in await repo.list_tasks():
        await repo.delete_task(t["id"])
    await app.store.execute("DELETE FROM seen_events")
    await app.store.execute("DELETE FROM notifications")
    tz = ARGS.timezone
    ids: dict[str, str] = {}

    # 1. planning ------------------------------------------------------------
    ids["planning"] = await add_task(
        repo, "demo-planning",
        name="Find birthday gift ideas for Anika under ₹5,000",
        description="Find three birthday gift ideas for Anika under ₹5,000 that can be delivered in Bengaluru before the 20th",
        status="planning", schedule=None, created_at=ts(timedelta(minutes=-1)),
    )

    # 2. approval_pending: 4-step plan with gmail + gcalendar ----------------
    ids["approval"] = await add_task(
        repo, "demo-approval",
        name="Prepare for Thursday's design critique",
        description=(
            "Before Thursday's design critique for the Lumen Health app, pull together the client feedback thread, "
            "the meeting details and the open comments in our design notes, then email me a one-page prep note."
        ),
        status="approval_pending", priority=0,
        schedule={"type": "once", "run_at": naive_local(2, 8, 30), "timezone": tz},
        plan=[
            {"tool": "gmail", "description": "Find the latest feedback emails from Lumen Health and summarise the open questions"},
            {"tool": "gcalendar", "description": "Look up Thursday's critique: time, agenda and who is attending"},
            {"tool": "notion", "description": "Collect the unresolved comments from the Lumen Health design notes"},
            {"tool": "gmail", "description": "Draft a one-page prep note and email it to me"},
        ],
        original_context={"source": "chat", "session_id": "demo-session"},
        created_at=ts(timedelta(minutes=-18)),
    )

    # 3. clarification_pending ------------------------------------------------
    ids["clarify"] = await add_task(
        repo, "demo-clarify",
        name="Book a table for Rohan's farewell dinner",
        description="Book a nice restaurant for Rohan's farewell dinner and add it to my calendar",
        status="clarification_pending", priority=1, schedule={"type": "once", "run_at": None, "timezone": tz},
        clarifying_questions=[
            {"question_id": "q1", "text": "Which date is the dinner, and roughly what time?", "answer": None},
            {"question_id": "q2", "text": "How many people are coming, and is there a cuisine Rohan prefers?", "answer": None},
            {"question_id": "q3", "text": "Should the restaurant be near home in Indiranagar or somewhere else?", "answer": None},
        ],
        original_context={"source": "proactive", "suggestion_type": "plan_event"},
        created_at=ts(timedelta(minutes=-42)),
    )

    # 3b. waiting_for_user: a running task paused to ask a question (ask_user) --
    train_plan = [
        {"tool": "internet_search", "description": "Find Saturday morning trains from Bengaluru to Mysuru with seats left"},
        {"tool": "files", "description": "Save the chosen train and timings to a trip note"},
    ]
    train_question = "Two morning trains still have seats on Saturday. Which one should I note down for you?"
    train_options = ["Shatabdi at 11:00 (2h, ₹745)", "Chamundi Express at 06:15 (3h, ₹180)"]
    ids["waiting"] = await add_task(
        repo, "demo-waiting",
        name="Plan Saturday's train to Mysuru",
        description="Find a good morning train from Bengaluru to Mysuru this Saturday and save the timings to my trip note.",
        status="waiting_for_user", priority=0, schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=train_plan, created_at=ts(timedelta(minutes=-25)), last_execution_at=ts(timedelta(minutes=-12)),
    )
    waiting_run = await add_run(
        repo, ids["waiting"], started=NOW - timedelta(minutes=12), status="waiting_for_user", plan=train_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (4, call("web_search", query="Bengaluru to Mysuru trains Saturday morning seats")),
            (9, result_msg("web_search", [
                {"title": "12007 Shatabdi Express", "departs": "11:00", "arrives": "13:00", "fare": "₹745"},
                {"title": "16216 Chamundi Express", "departs": "06:15", "arrives": "09:15", "fare": "₹180"},
            ])),
            (12, thought("Both trains fit a morning trip. The choice depends on what Maya prefers, so I'll ask her.")),
            (14, call("ask_user", question=train_question, options=train_options)),
            (15, result_msg("ask_user", {"status": "waiting_for_user"})),
            (15, info(f"Waiting for your answer: {train_question}")),
        ],
    )
    await repo.update_run(waiting_run, {"pending_question": {
        "question": train_question, "options": train_options, "tool_call_id": "call_demo_ask",
        "asked_at": iso(NOW - timedelta(minutes=12) + timedelta(seconds=15)),
    }})

    # 4. daily recurring digest with 3 past runs -------------------------------
    daily = {"type": "recurring", "frequency": "daily", "time": "08:00", "timezone": tz}
    ids["daily"] = await add_task(
        repo, "demo-daily",
        name="Morning inbox digest",
        description="Every morning at 8, summarise my unread email and tell me what needs a reply.",
        status="active", priority=1, schedule=daily, plan=DIGEST_PLAN,
        next_execution_at=iso(calculate_next_run(daily, NOW)),
        last_execution_at=iso(local_at(-1, 8)),
        created_at=iso(local_at(-9, 21, 14)),
    )
    for days, status, unread in ((-3, "error", 14), (-2, "completed_with_errors", 9), (-1, "completed", 11)):
        started = local_at(days, 8)
        label = started.astimezone(TZ).strftime("%Y-%m-%d")
        if status == "error":
            await add_run(
                repo, ids["daily"], started=started, status="error", plan=DIGEST_PLAN,
                events=[
                    (0, info("Executor has picked up the task and is starting execution.")),
                    (2, call("gmail_search", query="is:unread newer_than:1d", max_results=50)),
                    (4, result_msg("gmail_search", "401 Unauthorized: Token has been expired or revoked.", is_error=True)),
                    (6, err("Gmail rejected the saved credentials. Reconnect Gmail in Integrations.")),
                ],
                error="Gmail rejected the saved credentials. Reconnect Gmail in Integrations.",
            )
        else:
            await add_run(
                repo, ids["daily"], started=started, status=status, plan=DIGEST_PLAN,
                events=digest_events(label, unread, rate_limited=status == "completed_with_errors"),
                result=digest_result(label, unread),
            )

    # 5. weekly recurring -----------------------------------------------------
    weekly = {"type": "recurring", "frequency": "weekly", "days": ["Monday", "Thursday"], "time": "17:30", "timezone": tz}
    weekly_plan = [
        {"tool": "gmail", "description": "Collect new feedback emails from Lumen Health and Paperkite"},
        {"tool": "slack", "description": "Post a short summary to #design-feedback"},
    ]
    ids["weekly"] = await add_task(
        repo, "demo-weekly",
        name="Client feedback roundup",
        description="Every Monday and Thursday evening, summarise new client feedback and post it to #design-feedback on Slack.",
        status="active", priority=2, schedule=weekly, plan=weekly_plan,
        next_execution_at=iso(calculate_next_run(weekly, NOW)),
        last_execution_at=iso(local_at(-4, 17, 30)),
        created_at=iso(local_at(-20, 11)),
    )
    await add_run(
        repo, ids["weekly"], started=local_at(-4, 17, 30), plan=weekly_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("gmail_search", query="from:(lumenhealth.example OR paperkite.example) newer_than:4d")),
            (5, result_msg("gmail_search", {"count": 2, "messages": [
                {"from": "Nisha Menon <nisha@lumenhealth.example>", "subject": "Onboarding v3: a few small asks"},
                {"from": "Leela Thomas <leela@paperkite.example>", "subject": "Journal screen looks great"},
            ]})),
            (8, call("slack_post_message", channel="#design-feedback", text="Lumen Health wants a shorter sign-up step. Paperkite approved the journal screen.")),
            (10, result_msg("slack_post_message", {"ok": True, "ts": "1726151400.000200"})),
            (13, final("Posted the roundup to **#design-feedback**: one change request from Lumen Health and one approval from Paperkite.")),
        ],
        result={
            "summary": "Two clients sent feedback. **Lumen Health** wants a shorter sign-up step and **Paperkite** approved the journal screen. I posted the summary to **#design-feedback**.",
            "links_created": [{"url": "https://chat.northwind.example/archives/C07/p1726151400000200", "description": "Roundup message in #design-feedback"}],
            "links_found": [{"url": "https://mail.example/inbox/19a3b7d", "description": "Onboarding v3: a few small asks"}],
            "files_created": [],
            "tools_used": ["gmail", "slack"],
        },
    )

    # 6a. triggered by gmail ---------------------------------------------------
    receipt_filter = {
        "from": {"$in": ["rent@harborhomes.example", "billing@citypower.example"]},
        "subject": {"$contains": "receipt"},
    }
    receipt_plan = [
        {"tool": "gmail", "description": "Download the receipt PDF attached to the email"},
        {"tool": "gdrive", "description": "Save it to Home/Receipts/2026 in Google Drive"},
        {"tool": "gsheets", "description": "Add the payee, amount and month to the household budget sheet"},
    ]
    ids["trigger_mail"] = await add_task(
        repo, "demo-trigger-mail",
        name="File rent and electricity receipts",
        description="Whenever my landlord or the electricity company emails a receipt, save the PDF to Drive and log it in the household budget sheet.",
        status="active", priority=1,
        schedule={"type": "triggered", "source": "gmail", "event": "new_email", "filter": receipt_filter, "timezone": tz},
        plan=receipt_plan, last_execution_at=ts(timedelta(hours=-5)), created_at=ts(timedelta(days=-12)),
    )
    email = {
        "id": "191e3c5f7a2b4d10",
        "thread_id": "191e3c5f7a2b4d10",
        "from": "Harbor Homes <rent@harborhomes.example>",
        "sender_email": "rent@harborhomes.example",
        "to": "maya@northwind.example",
        "subject": "Rent receipt for October, Flat 304",
        "snippet": "We received your rent of ₹32,000.00 for October. Your receipt RC-1041 is attached.",
        "date": ts(timedelta(hours=-5, minutes=-1)),
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "url": "https://mail.example/inbox/191e3c5f7a2b4d10",
    }
    await add_run(
        repo, ids["trigger_mail"], started=NOW - timedelta(hours=5), plan=receipt_plan, trigger=email,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, thought("New rent receipt RC-1041 from Harbor Homes. I'll grab the attachment first.")),
            (3, call("gmail_get_attachment", message_id="191e3c5f7a2b4d10", filename="Receipt-RC-1041.pdf")),
            (6, result_msg("gmail_get_attachment", {"saved_to": "files/downloads/Receipt-RC-1041.pdf", "size": 48213})),
            (8, call("gdrive_upload", path="files/downloads/Receipt-RC-1041.pdf", folder="Home/Receipts/2026")),
            (12, result_msg("gdrive_upload", {"id": "1AbC", "url": "https://drive.example/file/1AbC"})),
            (14, call("gsheets_append_row", sheet="Household 2026", values=["Harbor Homes", "RC-1041", "32000", "2026-10"])),
            (17, result_msg("gsheets_append_row", {"updated_range": "Household 2026!A42:D42"})),
            (20, final("Saved **RC-1041** (₹32,000, October rent) to *Home/Receipts/2026* and logged it in the household budget sheet.")),
        ],
        result={
            "summary": "Filed rent receipt **RC-1041** from Harbor Homes (₹32,000 for **October**) and added it to *Household 2026*.",
            "links_created": [{"url": "https://drive.example/file/1AbC", "description": "Receipt-RC-1041.pdf in Drive"}],
            "links_found": [],
            "files_created": [{"filename": "downloads/Receipt-RC-1041.pdf", "description": "Receipt PDF"}],
            "tools_used": ["gmail", "gdrive", "gsheets"],
        },
    )

    # 6b. triggered by a calendar event ------------------------------------------
    cal_plan = [
        {"tool": "gcalendar", "description": "Read the new event's details and attendees"},
        {"tool": "memory", "description": "Recall what I know about the attendees"},
        {"tool": "notion", "description": "Create a prep page linked from the event"},
    ]
    ids["trigger_cal"] = await add_task(
        repo, "demo-trigger-cal",
        name="Prep notes for new client meetings",
        description="When a meeting with someone outside the studio lands on my calendar, create a prep page in Notion.",
        status="active", priority=2,
        schedule={"type": "triggered", "source": "gcalendar", "event": "new_event", "filter": {"attendees": {"$regex": "^(?!.*@northwind\\.example).*$"}}, "timezone": tz},
        plan=cal_plan, enabled=False, last_execution_at=ts(timedelta(days=-2)), created_at=ts(timedelta(days=-30)),
    )
    event = {
        "id": "7kq2m1r0evnt",
        "summary": "Kickoff with Paperkite",
        "description": "Walk through the redesign scope and the timeline for the first round of screens.",
        "start": iso(local_at(1, 15)),
        "end": iso(local_at(1, 15, 45)),
        "location": "Google Meet",
        "attendees": ["leela@paperkite.example", "arjun@paperkite.example", "maya@northwind.example"],
        "organizer_email": "leela@paperkite.example",
        "url": "https://calendar.example/event?eid=7kq2m1r0evnt",
        "status": "confirmed",
    }
    await add_run(
        repo, ids["trigger_cal"], started=NOW - timedelta(days=2), plan=cal_plan, trigger=event,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("memory_search", query="Paperkite Leela Arjun")),
            (3, result_msg("memory_search", ["Met Leela at a design meetup in March.", "Paperkite makes a journaling app for students."])),
            (6, call("notion_create_page", parent="Meetings", title="Prep: Kickoff with Paperkite")),
            (9, result_msg("notion_create_page", {"url": "https://notion.example/northwind/Prep-Paperkite-3f2a"})),
            (11, final("Created a prep page for **Kickoff with Paperkite** with background on both attendees.")),
        ],
        result={
            "summary": "Prep page ready for **Kickoff with Paperkite**, including what we know about Leela and Arjun.",
            "links_created": [{"url": "https://notion.example/northwind/Prep-Paperkite-3f2a", "description": "Prep page in Notion"}],
            "links_found": [],
            "files_created": [],
            "tools_used": ["gcalendar", "memory", "notion"],
        },
        memory_sources=await seeded_sources(repo, [
            "Maya is waiting for Paperkite to approve her quote for the website",
            "Maya chose a warm serif font and soft pastel colours for the Paperkite website",
        ]),
    )

    # 7. running now ------------------------------------------------------------
    running_plan = [
        {"tool": "internet_search", "description": "Find onboarding walkthroughs for five popular wellness apps"},
        {"tool": "web", "description": "Read each walkthrough and note the steps, permission asks and tone"},
        {"tool": "gsheets", "description": "Build a comparison sheet and share the link"},
    ]
    ids["running"] = await add_task(
        repo, "demo-running",
        name="Collect onboarding examples from wellness apps",
        description="Compare how five wellness apps handle onboarding and put it in a Google Sheet for the Lumen Health project.",
        status="processing", priority=0, schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=running_plan, last_execution_at=ts(timedelta(minutes=-3)), created_at=ts(timedelta(minutes=-9)),
    )
    await add_run(
        repo, ids["running"], started=NOW - timedelta(minutes=3), status="processing", plan=running_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (4, thought("I'll look for full screen-by-screen walkthroughs rather than store listings, so the steps are complete.")),
            (6, call("web_search", query="wellness app onboarding walkthrough screens 2026", max_results=10)),
            (11, result_msg("web_search", [
                {"title": "Calmly onboarding, every screen", "url": "https://calmly.example/onboarding"},
                {"title": "Stride: first run experience", "url": "https://stride.example/first-run"},
                {"title": "Bloom Journal setup flow", "url": "https://bloomjournal.example/setup"},
                {"title": "Pulse Track welcome tour", "url": "https://pulsetrack.example/tour"},
                {"title": "Sleepwell getting started", "url": "https://sleepwell.example/start"},
            ])),
            (18, thought("Five walkthroughs found. Reading Calmly first; it has the longest flow.")),
            (20, call("web_fetch", url="https://calmly.example/onboarding")),
            (31, result_msg("web_fetch", "6 screens: pick a goal, choose a reminder time, try a 1-minute session, then sign up. Skip is on every step.")),
            (38, call("web_fetch", url="https://stride.example/first-run")),
            (49, result_msg("web_fetch", "3 screens: sign up first, then health permissions, then a short goal quiz.")),
            (57, thought("Two of five done. Bloom Journal next, then Pulse Track and Sleepwell.")),
            (60, call("web_fetch", url="https://bloomjournal.example/setup")),
        ],
    )

    # 8. swarm with sub-agent progress and aggregated results ------------------
    treks = ["Skandagiri", "Savandurga", "Makalidurga", "Nandi Hills", "Kumara Parvatha", "Tadiandamol"]
    swarm_updates: list[dict] = []
    for i, trek in enumerate(treks):
        swarm_updates.append({"worker_id": f"agent-{i + 1}", "timestamp": ts(timedelta(minutes=-26, seconds=i * 5)), "status": "processing", "message": f"Starting work on item: {trek}"})
    for i, trek in enumerate(treks):
        swarm_updates.append({"worker_id": f"agent-{i + 1}", "timestamp": ts(timedelta(minutes=-22, seconds=i * 41)), "status": "completed", "message": f"Finished work. Result: {{'name': '{trek}', 'difficulty': ..."})
    swarm_updates.append({"worker_id": "aggregator", "timestamp": ts(timedelta(minutes=-17)), "status": "aggregating", "message": "Aggregating results from 6 agents."})
    aggregated = [
        {"name": "Skandagiri", "distance": "About 70 km", "difficulty": "Easy to moderate", "duration": "3 hours", "highlight": "Sunrise above the clouds, start before 4 AM"},
        {"name": "Savandurga", "distance": "About 60 km", "difficulty": "Moderate", "duration": "4 hours", "highlight": "Steep climb up a huge granite hill"},
        {"name": "Makalidurga", "distance": "About 60 km", "difficulty": "Easy to moderate", "duration": "3 to 4 hours", "highlight": "Quiet fort ruins and lake views"},
        {"name": "Nandi Hills", "distance": "About 60 km", "difficulty": "Easy", "duration": "2 hours", "highlight": "Relaxed morning walk, busy on weekends"},
        {"name": "Kumara Parvatha", "distance": "About 280 km", "difficulty": "Hard", "duration": "2 days", "highlight": "Long forest trail, needs a camping stop"},
        {"name": "Tadiandamol", "distance": "About 250 km (Coorg)", "difficulty": "Moderate", "duration": "6 hours", "highlight": "Highest peak in Coorg, grassland ridges"},
    ]
    ids["swarm"] = await add_task(
        repo, "demo-swarm",
        name="Compare six weekend treks near Bengaluru",
        description="Swarm task to achieve the goal: Research Skandagiri, Savandurga, Makalidurga, Nandi Hills, Kumara Parvatha and Tadiandamol and compare them",
        status="completed", task_type="swarm", priority=1, schedule=None,
        swarm_details={
            "goal": "Research Skandagiri, Savandurga, Makalidurga, Nandi Hills, Kumara Parvatha and Tadiandamol: distance from Bengaluru, difficulty, time needed and what each is best for.",
            "items": treks, "total_agents": 6, "completed_agents": 6,
            "progress_updates": swarm_updates, "aggregated_results": aggregated,
        },
        last_execution_at=ts(timedelta(minutes=-27)), created_at=ts(timedelta(minutes=-28)),
    )
    worker_configs = [{"item_indices": [i], "worker_prompt": f"Research the {trek} trek: distance from Bengaluru, difficulty, time needed and what it is best for.", "required_tools": ["internet_search", "web"]} for i, trek in enumerate(treks)]
    await add_run(
        repo, ids["swarm"], started=NOW - timedelta(minutes=27), plan=worker_configs,
        events=[
            (0, info("Resource manager created a plan for 6 agents.")),
            (2, info("Executor has picked up the task and is starting execution.")),
            (560, info("All 6 agents finished. Aggregating results.")),
            (600, final("Compared 6 treks. **Tadiandamol** fits the Coorg weekend best; **Skandagiri** is the easiest short trip from the city.")),
        ],
        result={
            "summary": (
                "| Trek | Distance | Difficulty | Time | Best for |\n|---|---|---|---|---|\n"
                + "\n".join(f"| {a['name']} | {a['distance']} | {a['difficulty']} | {a['duration']} | {a['highlight']} |" for a in aggregated)
                + "\n\n**Recommendation:** pick **Tadiandamol** for the Coorg weekend; choose **Skandagiri** for a quick sunrise trip from the city."
            ),
            "links_created": [],
            "links_found": [
                {"url": "https://trails.example/karnataka/tadiandamol", "description": "Tadiandamol trail guide"},
                {"url": "https://trails.example/karnataka/skandagiri", "description": "Skandagiri trail guide"},
            ],
            "files_created": [{"filename": "outputs/weekend-treks.md", "description": "Comparison table"}],
            "tools_used": ["internet_search", "web"],
        },
        duration_s=640,
    )

    # 9. completed one-off with files and a change-request conversation ---------
    run_plan = [
        {"tool": "memory", "description": "Recall my current running pace, gym days and race date"},
        {"tool": "files", "description": "Build the training plan spreadsheet and a printable PDF"},
    ]
    ids["completed"] = await add_task(
        repo, "demo-completed",
        name="Make a 10-week half marathon plan",
        description="Build a 10-week training plan for the half marathon that fits around my gym days.",
        status="completed", priority=1, schedule={"type": "once", "run_at": None, "timezone": tz}, plan=run_plan,
        chat_history=[
            {"role": "user", "content": "Can you keep Wednesdays for the gym and move long runs to Sunday?", "timestamp": ts(timedelta(hours=-26))},
            {"role": "assistant", "content": "I've updated the plan (2 steps) based on your request. Review it and approve to run it.", "timestamp": ts(timedelta(hours=-26, seconds=20))},
        ],
        last_execution_at=ts(timedelta(hours=-25)), created_at=ts(timedelta(hours=-27)),
    )
    await add_run(
        repo, ids["completed"], started=NOW - timedelta(hours=25), plan=run_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (3, call("memory_search", query="running pace gym days half marathon")),
            (9, result_msg("memory_search", ["Maya runs 5 km in about 31 minutes.", "Gym on Mondays and Wednesdays.", "Half marathon on December 14."])),
            (15, thought("Ten weeks to race day. Four runs a week, long run on Sunday, Wednesday kept for the gym.")),
            (40, call("file_write", path="outputs/half-marathon-plan.xlsx")),
            (44, result_msg("file_write", {"ok": True})),
            (46, call("file_write", path="outputs/half-marathon-plan.pdf")),
            (50, result_msg("file_write", {"ok": True})),
            (54, final("Your 10-week plan is ready: **4 runs a week**, building the Sunday long run from 8 km to 18 km.")),
        ],
        result={
            "summary": "Your half marathon plan has **4 runs a week** for **10 weeks**. Sunday long runs build from 8 km to 18 km, Wednesdays stay free for the gym and the last week is an easy taper.",
            "links_created": [],
            "links_found": [],
            "files_created": [
                {"filename": "outputs/half-marathon-plan.xlsx", "description": "Week-by-week plan with paces"},
                {"filename": "outputs/half-marathon-plan.pdf", "description": "One-page printable version"},
            ],
            "tools_used": ["memory", "files"],
        },
    )

    # 10. errored one-off ----------------------------------------------------
    slack_error = "Slack isn't connected. Connect Slack in Integrations and run the task again."
    ids["error"] = await add_task(
        repo, "demo-error",
        name="Post the weekly design update to #studio",
        description="Summarise the design work we finished this week and post it to #studio on Slack.",
        status="error", priority=1, error=slack_error,
        schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=[
            {"tool": "notion", "description": "List design tasks marked done this week"},
            {"tool": "slack", "description": "Post the update to #studio"},
        ],
        last_execution_at=ts(timedelta(hours=-3)), created_at=ts(timedelta(hours=-3, minutes=-5)),
    )
    await add_run(
        repo, ids["error"], started=NOW - timedelta(hours=3), status="error", error=slack_error,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("notion_query_database", database="Design tasks", filter="Done this week")),
            (6, result_msg("notion_query_database", {"total": 6})),
            (9, call("slack_post_message", channel="#studio", text="This week we finished 6 design tasks.")),
            (10, result_msg("slack_post_message", "Slack is not connected.", is_error=True)),
            (12, err(slack_error)),
        ],
    )

    # 11. scheduled one-off (pending) -----------------------------------------
    check_at = local_at(4, 7)
    ids["scheduled"] = await add_task(
        repo, "demo-scheduled",
        name="Check the weather before the Coorg trek",
        description="On Saturday morning, check the weather in Coorg for the weekend and email me if rain is likely.",
        status="pending", priority=2,
        schedule={"type": "once", "run_at": naive_local(4, 7), "timezone": tz},
        plan=[
            {"tool": "internet_search", "description": "Look up the weekend forecast for Madikeri and Tadiandamol"},
            {"tool": "gmail", "description": "Email me a short summary and whether to pack rain gear"},
        ],
        next_execution_at=iso(check_at), created_at=ts(timedelta(days=-1)),
    )

    # 12. archived ------------------------------------------------------------
    ids["archived"] = await add_task(
        repo, "demo-archived",
        name="Find a plumber for the kitchen sink",
        description="Find a well-reviewed plumber in Indiranagar who can come this week.",
        status="archived", priority=2, schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=[{"tool": "internet_search", "description": "Search for plumbers in Indiranagar with 4.5+ ratings"}],
        created_at=ts(timedelta(days=-15)), last_execution_at=ts(timedelta(days=-15)),
    )
    await add_run(
        repo, ids["archived"], started=NOW - timedelta(days=15),
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (3, call("web_search", query="plumber Indiranagar Bengaluru reviews")),
            (7, result_msg("web_search", [{"title": "FlowRight Plumbing Indiranagar", "rating": 4.7}])),
            (10, final("**FlowRight Plumbing** (4.7★) can visit on Wednesday between 10 and 1.")),
        ],
        result={"summary": "**FlowRight Plumbing** in Indiranagar (4.7★) has a Wednesday morning slot.", "links_created": [], "links_found": [], "files_created": [], "tools_used": ["internet_search"]},
    )

    # notifications that deep-link into tasks -----------------------------------
    await app.notify("approval", "I've created a new plan for you: 'Prepare for Thursday's design critique'",
                     title="Plan ready for approval", payload={"task_id": ids["approval"], "event": "approval_needed"})
    await app.notify("task", "I need a bit more information to plan 'Book a table for Rohan's farewell dinner'. Please answer the questions in the task.",
                     title="Clarification needed", payload={"task_id": ids["clarify"], "event": "clarification_needed"})
    waiting = (await repo.waiting_runs(ids["waiting"]))[0]
    question = waiting["pending_question"]
    await app.notify("task", question["question"], title="Plan Saturday's train to Mysuru needs your answer",
                     payload={"task_id": ids["waiting"], "event": "question", "run_id": waiting["id"],
                              "question": question["question"], "options": question["options"]})
    await app.notify("task", f"Task 'Post the weekly design update to #studio' has finished with status: error.\n\n{slack_error}",
                     title="Task failed", payload={"task_id": ids["error"], "event": "run_failed"})
    return ids


async def main() -> None:
    paths.ensure_layout()
    cfg = configure()
    if ARGS.config_only:
        print(f"Updated {paths.config_file()}")
        return
    app =SentientApp(cfg, llm=OfflineProvider(), db_path=paths.db_file(), enable_background=False)  # type: ignore[arg-type]
    await app.start()
    try:
        ids = await seed(app)
    finally:
        await app.stop()
    print(f"Seeded {len(ids)} tasks into {paths.home()}")
    for key, task_id in ids.items():
        print(f"  {key:<13} {task_id}")


if __name__ == "__main__":
    asyncio.run(main())
