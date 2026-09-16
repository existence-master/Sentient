"""Seed realistic demo tasks into a Sentient home (no LLM calls).

    .venv/Scripts/python.exe desktop/scripts/seed-tasks.py <SENTIENT_HOME>

Writes straight through the tasks repository: one task per interesting state
(planning, approval, clarification, recurring with run history, triggered by an
email and by a calendar event, a live run, a swarm, completed with files, errored,
scheduled, archived) plus a couple of notifications that point at tasks.

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
    p.add_argument("home", help="SENTIENT_HOME folder to seed (created if missing)")
    p.add_argument("--user", default="Sarthak", help="user name written to config")
    p.add_argument("--timezone", default="Asia/Kolkata", help="assistant timezone written to config")
    p.add_argument("--theme", choices=["dark", "light", "system"], default=None, help="set ui.theme (for screenshots)")
    p.add_argument("--config-only", action="store_true", help="only update config.yaml (theme, models); keep tasks")
    return p.parse_args()


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
        yield  # noqa: B901

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
    cfg.assistant.location = cfg.assistant.location or "Pune, India"
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
) -> str:
    run_id = await repo.insert_run(task_id, now=iso(started), plan=plan or [], trigger_data=trigger)
    for offset_s, message in events:
        await repo.add_event(run_id, message, iso(started + timedelta(seconds=offset_s)))
    fields: dict[str, Any] = {"status": status, "error": error}
    if status != "processing":
        last = max((o for o, _ in events), default=0)
        fields["finished_at"] = iso(started + timedelta(seconds=duration_s or last + 4))
    if result is not None:
        fields["result"] = result
    await repo.update_run(run_id, fields)
    return run_id


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
                        {"from": "Priya Nair <priya@northwind.vc>", "subject": "Re: Term sheet follow-ups", "labels": ["IMPORTANT"]},
                        {"from": "GitHub <noreply@github.com>", "subject": "[sentient] 3 PRs need your review"},
                        {"from": "HDFC Bank <alerts@hdfcbank.net>", "subject": "Credit card statement for August"},
                    ],
                },
            ),
        ),
        (52 if rate_limited else 15, thought("Priya's thread is the only one that needs a reply today. GitHub and the bank statement are FYI.")),
        (
            55 if rate_limited else 18,
            call("file_write", path=f"digests/inbox-{day_label}.md", content=f"# Inbox digest {day_label}\n\n- Reply to Priya (term sheet)\n- 3 PRs waiting on GitHub\n- Card statement arrived"),
        ),
        (57 if rate_limited else 19, result_msg("file_write", {"ok": True, "path": f"digests/inbox-{day_label}.md", "bytes": 212})),
        (
            61 if rate_limited else 24,
            final(
                f"**{unread} unread emails** since yesterday.\n\n"
                "1. **Priya Nair (Northwind)** wants the revised term sheet by Thursday. *Needs a reply.*\n"
                "2. **GitHub**: 3 pull requests are waiting for your review on `sentient`.\n"
                "3. **HDFC Bank**: August credit card statement is ready (due Sep 28)."
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
            "- **Priya Nair** (Northwind Ventures) asked for the revised term sheet by Thursday.\n\n"
            "FYI: 3 GitHub pull requests are waiting for review and your August card statement arrived."
        ),
        "links_created": [],
        "links_found": [
            {"url": "https://mail.google.com/mail/u/0/#inbox/18f2a9c", "description": "Re: Term sheet follow-ups"},
            {"url": "https://github.com/pulls/review-requested", "description": "Pull requests waiting for your review"},
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
        name="Find three standing desks under ₹30,000 with good reviews",
        description="Find three standing desks under ₹30,000 with good reviews and delivery to Pune",
        status="planning", schedule=None, created_at=ts(timedelta(minutes=-1)),
    )

    # 2. approval_pending: 4-step plan with gmail + gcalendar ----------------
    ids["approval"] = await add_task(
        repo, "demo-approval",
        name="Prepare a brief for Thursday's investor call",
        description=(
            "Before Thursday's call with Northwind Ventures, pull together the recent email thread, "
            "the meeting details and a short background on the partners, then email me a one-page brief."
        ),
        status="approval_pending", priority=0,
        schedule={"type": "once", "run_at": naive_local(2, 8, 30), "timezone": tz},
        plan=[
            {"tool": "gmail", "description": "Find the latest email threads with Northwind Ventures and summarise open questions"},
            {"tool": "gcalendar", "description": "Look up Thursday's call: time, agenda and who is attending"},
            {"tool": "internet_search", "description": "Research the attending partners and their recent investments"},
            {"tool": "gmail", "description": "Draft a one-page brief and email it to me"},
        ],
        original_context={"source": "chat", "session_id": "demo-session"},
        created_at=ts(timedelta(minutes=-18)),
    )

    # 3. clarification_pending ------------------------------------------------
    ids["clarify"] = await add_task(
        repo, "demo-clarify",
        name="Book a table for Mom's birthday dinner",
        description="Book a nice restaurant for Mom's birthday dinner and add it to my calendar",
        status="clarification_pending", priority=1, schedule={"type": "once", "run_at": None, "timezone": tz},
        clarifying_questions=[
            {"question_id": "q1", "text": "Which date is the dinner, and roughly what time?", "answer": None},
            {"question_id": "q2", "text": "How many people are coming, and is there a cuisine Mom prefers?", "answer": None},
            {"question_id": "q3", "text": "Should the restaurant be near home in Pune or somewhere else?", "answer": None},
        ],
        original_context={"source": "proactive", "suggestion_type": "plan_event"},
        created_at=ts(timedelta(minutes=-42)),
    )

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
                    (6, err("Executor agent failed: Gmail rejected the saved credentials. Reconnect Gmail in Integrations.")),
                ],
                error="Executor agent failed: Gmail rejected the saved credentials. Reconnect Gmail in Integrations.",
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
        {"tool": "github", "description": "List open pull requests that are waiting on my review"},
        {"tool": "slack", "description": "Post a short summary to #eng-reviews"},
    ]
    ids["weekly"] = await add_task(
        repo, "demo-weekly",
        name="Pull request review roundup",
        description="Every Monday and Thursday evening, list PRs waiting on my review and post a summary to Slack.",
        status="active", priority=2, schedule=weekly, plan=weekly_plan,
        next_execution_at=iso(calculate_next_run(weekly, NOW)),
        last_execution_at=iso(local_at(-4, 17, 30)),
        created_at=iso(local_at(-20, 11)),
    )
    await add_run(
        repo, ids["weekly"], started=local_at(-4, 17, 30), plan=weekly_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("github_search_pull_requests", query="is:open review-requested:@me")),
            (5, result_msg("github_search_pull_requests", {"total": 2, "items": [{"title": "Tasks board view", "repo": "existence/sentient"}, {"title": "Fix tray icon on Windows", "repo": "existence/sentient"}]})),
            (8, call("slack_post_message", channel="#eng-reviews", text="2 PRs waiting on Sarthak: Tasks board view, Fix tray icon on Windows")),
            (10, result_msg("slack_post_message", {"ok": True, "ts": "1726151400.000200"})),
            (13, final("Posted the roundup to **#eng-reviews**: 2 pull requests are waiting on your review.")),
        ],
        result={
            "summary": "Two pull requests are waiting on your review. I posted the list to **#eng-reviews**.",
            "links_created": [{"url": "https://existence.slack.com/archives/C07/p1726151400000200", "description": "Roundup message in #eng-reviews"}],
            "links_found": [{"url": "https://github.com/existence/sentient/pull/412", "description": "Tasks board view"}],
            "files_created": [],
            "tools_used": ["github", "slack"],
        },
    )

    # 6a. triggered by gmail ---------------------------------------------------
    invoice_filter = {
        "from": {"$in": ["invoices@stripe.com", "billing@aws.amazon.com"]},
        "subject": {"$contains": "invoice"},
    }
    invoice_plan = [
        {"tool": "gmail", "description": "Download the invoice PDF attached to the email"},
        {"tool": "gdrive", "description": "Save it to Finance/Invoices/2026 in Google Drive"},
        {"tool": "gsheets", "description": "Add the vendor, amount and due date to the expenses sheet"},
    ]
    ids["trigger_mail"] = await add_task(
        repo, "demo-trigger-mail",
        name="File vendor invoices to Drive",
        description="Whenever Stripe or AWS emails me an invoice, save the PDF to Drive and log it in the expenses sheet.",
        status="active", priority=1,
        schedule={"type": "triggered", "source": "gmail", "event": "new_email", "filter": invoice_filter, "timezone": tz},
        plan=invoice_plan, last_execution_at=ts(timedelta(hours=-5)), created_at=ts(timedelta(days=-12)),
    )
    email = {
        "id": "191e3c5f7a2b4d10",
        "thread_id": "191e3c5f7a2b4d10",
        "from": "Stripe <invoices@stripe.com>",
        "sender_email": "invoices@stripe.com",
        "to": "sarthak@existence.technology",
        "subject": "Your invoice from Acme Cloud Hosting #INV-2041",
        "snippet": "Invoice INV-2041 for ₹18,450.00 is due on September 30, 2026. Download the PDF or pay online.",
        "date": ts(timedelta(hours=-5, minutes=-1)),
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "url": "https://mail.google.com/mail/u/0/#inbox/191e3c5f7a2b4d10",
    }
    await add_run(
        repo, ids["trigger_mail"], started=NOW - timedelta(hours=5), plan=invoice_plan, trigger=email,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, thought("New Stripe invoice INV-2041. I'll grab the attachment first.")),
            (3, call("gmail_get_attachment", message_id="191e3c5f7a2b4d10", filename="Invoice-INV-2041.pdf")),
            (6, result_msg("gmail_get_attachment", {"saved_to": "files/downloads/Invoice-INV-2041.pdf", "size": 48213})),
            (8, call("gdrive_upload", path="files/downloads/Invoice-INV-2041.pdf", folder="Finance/Invoices/2026")),
            (12, result_msg("gdrive_upload", {"id": "1AbC", "url": "https://drive.google.com/file/d/1AbC/view"})),
            (14, call("gsheets_append_row", sheet="Expenses 2026", values=["Acme Cloud Hosting", "INV-2041", "18450", "2026-09-30"])),
            (17, result_msg("gsheets_append_row", {"updated_range": "Expenses 2026!A88:D88"})),
            (20, final("Saved **INV-2041** (₹18,450, due Sep 30) to *Finance/Invoices/2026* and logged it in the expenses sheet.")),
        ],
        result={
            "summary": "Filed invoice **INV-2041** from Acme Cloud Hosting (₹18,450, due **Sep 30**) and added it to *Expenses 2026*.",
            "links_created": [{"url": "https://drive.google.com/file/d/1AbC/view", "description": "Invoice-INV-2041.pdf in Drive"}],
            "links_found": [],
            "files_created": [{"filename": "downloads/Invoice-INV-2041.pdf", "description": "Invoice PDF"}],
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
        name="Prep notes for new external meetings",
        description="When a meeting with someone outside the company lands on my calendar, create a prep page in Notion.",
        status="active", priority=2,
        schedule={"type": "triggered", "source": "gcalendar", "event": "new_event", "filter": {"attendees": {"$regex": "^(?!.*@existence\\.technology).*$"}}, "timezone": tz},
        plan=cal_plan, enabled=False, last_execution_at=ts(timedelta(days=-2)), created_at=ts(timedelta(days=-30)),
    )
    event = {
        "id": "7kq2m1r0evnt",
        "summary": "Partnership sync with Lumen Health",
        "description": "Walk through the pilot scope and data-sharing agreement.",
        "start": iso(local_at(1, 15)),
        "end": iso(local_at(1, 15, 45)),
        "location": "Google Meet",
        "attendees": ["ananya@lumenhealth.in", "rohit@lumenhealth.in", "sarthak@existence.technology"],
        "organizer_email": "ananya@lumenhealth.in",
        "url": "https://calendar.google.com/calendar/event?eid=7kq2m1r0evnt",
        "status": "confirmed",
    }
    await add_run(
        repo, ids["trigger_cal"], started=NOW - timedelta(days=2), plan=cal_plan, trigger=event,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("memory_search", query="Lumen Health Ananya Rohit")),
            (3, result_msg("memory_search", ["Met Ananya at the HealthTech summit in March.", "Lumen Health runs clinics in Pune and Bengaluru."])),
            (6, call("notion_create_page", parent="Meetings", title="Prep: Partnership sync with Lumen Health")),
            (9, result_msg("notion_create_page", {"url": "https://notion.so/existence/Prep-Lumen-3f2a"})),
            (11, final("Created a prep page for **Partnership sync with Lumen Health** with background on both attendees.")),
        ],
        result={
            "summary": "Prep page ready for **Partnership sync with Lumen Health**, including what we know about Ananya and Rohit.",
            "links_created": [{"url": "https://notion.so/existence/Prep-Lumen-3f2a", "description": "Prep page in Notion"}],
            "links_found": [],
            "files_created": [],
            "tools_used": ["gcalendar", "memory", "notion"],
        },
    )

    # 7. running now ------------------------------------------------------------
    running_plan = [
        {"tool": "internet_search", "description": "Find the pricing pages of the five closest competitors"},
        {"tool": "web", "description": "Read each pricing page and extract plans, prices and limits"},
        {"tool": "gsheets", "description": "Build a comparison sheet and share the link"},
    ]
    ids["running"] = await add_task(
        repo, "demo-running",
        name="Compile a competitor pricing sheet",
        description="Compare pricing for the five closest personal-assistant apps and put it in a Google Sheet.",
        status="processing", priority=0, schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=running_plan, last_execution_at=ts(timedelta(minutes=-3)), created_at=ts(timedelta(minutes=-9)),
    )
    await add_run(
        repo, ids["running"], started=NOW - timedelta(minutes=3), status="processing", plan=running_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (4, thought("I'll start by finding the official pricing pages rather than review sites, so the numbers are current.")),
            (6, call("web_search", query="personal AI assistant app pricing plans 2026", max_results=10)),
            (11, result_msg("web_search", [
                {"title": "Pricing - Lindy", "url": "https://www.lindy.ai/pricing"},
                {"title": "Plans - Motion", "url": "https://www.usemotion.com/pricing"},
                {"title": "Reclaim.ai Pricing", "url": "https://reclaim.ai/pricing"},
                {"title": "Superhuman plans", "url": "https://superhuman.com/plans"},
                {"title": "Martin AI pricing", "url": "https://www.trymartin.com/pricing"},
            ])),
            (18, thought("Five official pages found. Reading Lindy's first; it has the most tiers.")),
            (20, call("web_fetch", url="https://www.lindy.ai/pricing")),
            (31, result_msg("web_fetch", "Free: 400 tasks/mo · Pro $49.99/mo: 5,000 tasks · Business $299.99/mo: 30,000 tasks, priority support")),
            (38, call("web_fetch", url="https://www.usemotion.com/pricing")),
            (49, result_msg("web_fetch", "Individual $29/mo billed monthly ($19 annually) · Business $12/user/mo annually")),
            (57, thought("Two of five done. Reclaim next, then Superhuman and Martin.")),
            (60, call("web_fetch", url="https://reclaim.ai/pricing")),
        ],
    )

    # 8. swarm with sub-agent progress and aggregated results ------------------
    dbs = ["Qdrant", "Weaviate", "Milvus", "Chroma", "LanceDB", "pgvector"]
    swarm_updates: list[dict] = []
    for i, db in enumerate(dbs):
        swarm_updates.append({"worker_id": f"agent-{i + 1}", "timestamp": ts(timedelta(minutes=-26, seconds=i * 5)), "status": "processing", "message": f"Starting work on item: {db}"})
    for i, db in enumerate(dbs):
        swarm_updates.append({"worker_id": f"agent-{i + 1}", "timestamp": ts(timedelta(minutes=-22, seconds=i * 41)), "status": "completed", "message": f"Finished work. Result: {{'name': '{db}', 'license': ..."})
    swarm_updates.append({"worker_id": "aggregator", "timestamp": ts(timedelta(minutes=-17)), "status": "aggregating", "message": "Aggregating results from 6 agents."})
    aggregated = [
        {"name": "Qdrant", "language": "Rust", "license": "Apache-2.0", "hosting": "Self-host or Qdrant Cloud", "highlight": "Fast filtered search, strong payload indexing"},
        {"name": "Weaviate", "language": "Go", "license": "BSD-3-Clause", "hosting": "Self-host or Weaviate Cloud", "highlight": "Built-in hybrid search and modules for vectorisers"},
        {"name": "Milvus", "language": "Go / C++", "license": "Apache-2.0", "hosting": "Self-host or Zilliz Cloud", "highlight": "Scales to billions of vectors with GPU indexes"},
        {"name": "Chroma", "language": "Rust / Python", "license": "Apache-2.0", "hosting": "Embedded or Chroma Cloud", "highlight": "Simplest developer experience for prototypes"},
        {"name": "LanceDB", "language": "Rust", "license": "Apache-2.0", "hosting": "Embedded, serverless", "highlight": "Columnar Lance format, zero-copy on object storage"},
        {"name": "pgvector", "language": "C", "license": "PostgreSQL", "hosting": "Any Postgres", "highlight": "Vectors next to relational data, HNSW + IVFFlat"},
    ]
    ids["swarm"] = await add_task(
        repo, "demo-swarm",
        name="Compare six open-source vector databases",
        description="Swarm task to achieve the goal: Research Qdrant, Weaviate, Milvus, Chroma, LanceDB and pgvector and compare them",
        status="completed", task_type="swarm", priority=1, schedule=None,
        swarm_details={
            "goal": "Research Qdrant, Weaviate, Milvus, Chroma, LanceDB and pgvector: language, license, hosting options and what each is best at.",
            "items": dbs, "total_agents": 6, "completed_agents": 6,
            "progress_updates": swarm_updates, "aggregated_results": aggregated,
        },
        last_execution_at=ts(timedelta(minutes=-27)), created_at=ts(timedelta(minutes=-28)),
    )
    worker_configs = [{"item_indices": [i], "worker_prompt": f"Research {db}: language, license, hosting and best use.", "required_tools": ["internet_search", "web"]} for i, db in enumerate(dbs)]
    await add_run(
        repo, ids["swarm"], started=NOW - timedelta(minutes=27), plan=worker_configs,
        events=[
            (0, info("Resource manager created a plan for 6 agents.")),
            (2, info("Executor has picked up the task and is starting execution.")),
            (560, info("All 6 agents finished. Aggregating results.")),
            (600, final("Compared 6 vector databases. **pgvector** fits best if you already run Postgres; **Qdrant** is the strongest standalone option.")),
        ],
        result={
            "summary": (
                "| Database | Language | License | Best at |\n|---|---|---|---|\n"
                + "\n".join(f"| {a['name']} | {a['language']} | {a['license']} | {a['highlight']} |" for a in aggregated)
                + "\n\n**Recommendation:** start with **pgvector** if Postgres is already in the stack; choose **Qdrant** for a dedicated, filter-heavy workload."
            ),
            "links_created": [],
            "links_found": [
                {"url": "https://qdrant.tech/documentation/", "description": "Qdrant documentation"},
                {"url": "https://github.com/pgvector/pgvector", "description": "pgvector on GitHub"},
            ],
            "files_created": [{"filename": "outputs/vector-db-comparison.md", "description": "Comparison table"}],
            "tools_used": ["internet_search", "web"],
        },
        duration_s=640,
    )

    # 9. completed one-off with files and a change-request conversation ---------
    report_plan = [
        {"tool": "gmail", "description": "Collect receipts emailed between July and September"},
        {"tool": "files", "description": "Build the expense report spreadsheet and a PDF summary"},
    ]
    ids["completed"] = await add_task(
        repo, "demo-completed",
        name="Draft the Q3 expense report",
        description="Collect Q3 receipts from my inbox and prepare an expense report I can submit.",
        status="completed", priority=1, schedule={"type": "once", "run_at": None, "timezone": tz}, plan=report_plan,
        chat_history=[
            {"role": "user", "content": "Can you also split travel and software into separate sections?", "timestamp": ts(timedelta(hours=-26))},
            {"role": "assistant", "content": "I've updated the plan (2 steps) based on your request. Review it and approve to run it.", "timestamp": ts(timedelta(hours=-26, seconds=20))},
        ],
        last_execution_at=ts(timedelta(hours=-25)), created_at=ts(timedelta(hours=-27)),
    )
    await add_run(
        repo, ids["completed"], started=NOW - timedelta(hours=25), plan=report_plan,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (3, call("gmail_search", query="receipt OR invoice after:2026/07/01 before:2026/10/01", max_results=100)),
            (9, result_msg("gmail_search", {"count": 37})),
            (15, thought("37 receipts. Grouping into travel, software and other.")),
            (40, call("file_write", path="outputs/q3-expense-report.xlsx")),
            (44, result_msg("file_write", {"ok": True})),
            (46, call("file_write", path="outputs/q3-expense-summary.pdf")),
            (50, result_msg("file_write", {"ok": True})),
            (54, final("Your Q3 expense report is ready: **₹1,42,380** across 37 receipts.")),
        ],
        result={
            "summary": "Q3 expenses total **₹1,42,380** across **37 receipts**: travel ₹86,200, software ₹41,930 and other ₹14,250.",
            "links_created": [],
            "links_found": [],
            "files_created": [
                {"filename": "outputs/q3-expense-report.xlsx", "description": "Itemised report with travel and software sections"},
                {"filename": "outputs/q3-expense-summary.pdf", "description": "One-page summary for finance"},
            ],
            "tools_used": ["gmail", "files"],
        },
    )

    # 10. errored one-off ----------------------------------------------------
    slack_error = "Executor agent failed: Slack isn't connected. Connect Slack in Integrations and run the task again."
    ids["error"] = await add_task(
        repo, "demo-error",
        name="Post the weekly product update to #team",
        description="Summarise this week's shipped work from GitHub and post it to #team on Slack.",
        status="error", priority=1, error=slack_error,
        schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=[
            {"tool": "github", "description": "List pull requests merged this week"},
            {"tool": "slack", "description": "Post the update to #team"},
        ],
        last_execution_at=ts(timedelta(hours=-3)), created_at=ts(timedelta(hours=-3, minutes=-5)),
    )
    await add_run(
        repo, ids["error"], started=NOW - timedelta(hours=3), status="error", error=slack_error,
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (2, call("github_search_pull_requests", query="is:merged merged:>=2026-09-08")),
            (6, result_msg("github_search_pull_requests", {"total": 6})),
            (9, call("slack_post_message", channel="#team", text="This week we shipped…")),
            (10, result_msg("slack_post_message", "Slack is not connected.", is_error=True)),
            (12, err(slack_error)),
        ],
    )

    # 11. scheduled one-off (pending) -----------------------------------------
    renew_at = local_at(4, 10)
    ids["scheduled"] = await add_task(
        repo, "demo-scheduled",
        name="Remind me to renew my passport",
        description="On Saturday morning, check the passport renewal slots in Pune and email me the earliest ones.",
        status="pending", priority=2,
        schedule={"type": "once", "run_at": naive_local(4, 10), "timezone": tz},
        plan=[
            {"tool": "internet_search", "description": "Look up available Passport Seva appointment slots in Pune"},
            {"tool": "gmail", "description": "Email me the three earliest slots"},
        ],
        next_execution_at=iso(renew_at), created_at=ts(timedelta(days=-1)),
    )

    # 12. archived ------------------------------------------------------------
    ids["archived"] = await add_task(
        repo, "demo-archived",
        name="Find a plumber for the kitchen sink",
        description="Find a well-reviewed plumber in Baner who can come this week.",
        status="archived", priority=2, schedule={"type": "once", "run_at": None, "timezone": tz},
        plan=[{"tool": "internet_search", "description": "Search for plumbers in Baner with 4.5+ ratings"}],
        created_at=ts(timedelta(days=-15)), last_execution_at=ts(timedelta(days=-15)),
    )
    await add_run(
        repo, ids["archived"], started=NOW - timedelta(days=15),
        events=[
            (0, info("Executor has picked up the task and is starting execution.")),
            (3, call("web_search", query="plumber Baner Pune reviews")),
            (7, result_msg("web_search", [{"title": "QuickFix Plumbing Baner", "rating": 4.7}])),
            (10, final("**QuickFix Plumbing** (4.7★) can visit on Wednesday between 10 and 1.")),
        ],
        result={"summary": "**QuickFix Plumbing** in Baner (4.7★) has a Wednesday morning slot.", "links_created": [], "links_found": [], "files_created": [], "tools_used": ["internet_search"]},
    )

    # notifications that deep-link into tasks -----------------------------------
    await app.notify("approval", "I've created a new plan for you: 'Prepare a brief for Thursday's investor call'",
                     title="Plan ready for approval", payload={"task_id": ids["approval"], "event": "approval_needed"})
    await app.notify("task", "I need a bit more information to plan 'Book a table for Mom's birthday dinner'. Please answer the questions in the task.",
                     title="Clarification needed", payload={"task_id": ids["clarify"], "event": "clarification_needed"})
    await app.notify("task", f"Task 'Post the weekly product update to #team' has finished with status: error.\n\n{slack_error}",
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
