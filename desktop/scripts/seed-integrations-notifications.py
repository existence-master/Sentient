"""Seed a throwaway SENTIENT_HOME with demo data for the Integrations, Notifications and Voice UI.

    .venv/Scripts/python.exe desktop/scripts/seed-integrations-notifications.py [SENTIENT_HOME] [--keep-db]

Without a folder it seeds the folder in the SENTIENT_HOME variable, like the other seed scripts.

No LLM calls (a tiny fake provider is injected), no network, and nothing is written to the
real OS keychain (keychain helpers are swapped for an in-memory dict while this runs).
By default the database is recreated on every run so screenshots always start from the same state.

--keep-db keeps the existing sentient.db (and its -wal/-shm files) so this script can add its
data to a home that other seed scripts already filled. Tasks, notifications and other rows from
those scripts are left alone; only the rows this script created on an earlier run are replaced.

The demo user is Maya Rao, a product designer at Northwind Studio in Bengaluru.
Every name, client and address in the data is made up.

Creates:
- integration state: GitHub + Google Calendar connected, Gmail in an error state with privacy filters
- one MCP server config entry whose command doesn't exist (shows the error state when the engine runs)
- notifications of every kind, including 4 proactive suggestions (approved, pending gmail, pending gcalendar,
  pending follow-up with a draft reply)
- proactive preference rows, proactive source poll state and suggestion rows
- the Daily Brief task (weekdays at 07:30), the Evening Brief task (daily at 21:00) and today's morning brief
- assistant.onboarding_complete = true
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("home", nargs="?", default=os.environ.get("SENTIENT_HOME") or None,
                   help="SENTIENT_HOME folder to seed (created if missing); defaults to the SENTIENT_HOME variable")
    p.add_argument("--keep-db", action="store_true",
                   help="keep the existing sentient.db and add to it instead of starting from an empty database")
    args = p.parse_args()
    if not args.home:
        p.error("pass a folder or set SENTIENT_HOME")
    return args


ARGS = _parse_args()
HOME = Path(ARGS.home).expanduser()
KEEP_DB: bool = ARGS.keep_db
os.environ["SENTIENT_HOME"] = str(HOME)

# Import after SENTIENT_HOME is set.
from sentient.app import SentientApp  # noqa: E402
from sentient.config.loader import load_config, save_config  # noqa: E402
from sentient.integrations import common, google, mcp  # noqa: E402
from sentient.integrations import service as integrations_service  # noqa: E402
from sentient.llm.provider import StreamChunk  # noqa: E402

# ----------------------------------------------------------------------------- in-memory keychain
_MEMORY_KEYCHAIN: dict[str, dict] = {}


def _load(name: str) -> dict | None:
    return _MEMORY_KEYCHAIN.get(name)


def _store(name: str, data: dict) -> bool:
    _MEMORY_KEYCHAIN[name] = dict(data)
    return True


def _delete(name: str) -> bool:
    return _MEMORY_KEYCHAIN.pop(name, None) is not None


for module in (common, integrations_service, mcp):
    for attr, fn in (("load_secret_json", _load), ("store_secret_json", _store), ("delete_secret", _delete)):
        if hasattr(module, attr):
            setattr(module, attr, fn)
google.get_client = lambda: None  # type: ignore[assignment]


# ----------------------------------------------------------------------------- fake LLM
class TinyFakeProvider:
    def model_for(self, role: str) -> str:
        return f"fake/{role}"

    async def stream(self, role, messages, tools=None, *, model=None) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="ok", model="fake")
        yield StreamChunk(done=True, usage={"prompt_tokens": 1, "completion_tokens": 1}, model="fake")

    async def complete_text(self, role, messages, *, model=None) -> str:
        return "Demo"

    async def complete_json(self, role, messages, *, model=None):
        return {}

    async def embed(self, texts: list[str], *, model=None) -> list[list[float]]:
        return [[1.0] + [0.0] * 15 for _ in texts]


# ----------------------------------------------------------------------------- helpers
NOW = datetime.now(UTC)


def ago(**kw: float) -> str:
    return (NOW - timedelta(**kw)).isoformat()


async def backdate(app: SentientApp, note: dict, when: str, *, read: bool = False) -> None:
    await app.store.execute("UPDATE notifications SET created_at = ?, read = ? WHERE id = ?", (when, int(read), note["id"]))


async def make_task(app: SentientApp, task_id: str, name: str, prompt: str, status: str, when: str, **extra) -> str:
    """Insert a task with a fixed id so a --keep-db re-run replaces it instead of adding a copy."""
    if KEEP_DB:
        await app.tasks.repo.delete_task(task_id)
    fields = {
        "id": task_id,
        "name": name, "description": prompt, "status": status, "priority": 1, "assignee": "ai",
        "original_prompt": prompt, "source": extra.pop("source", "user"), "enabled": True, "model": None,
        "original_context": extra.pop("original_context", {"source": "manual_creation"}), "plan": [],
        "chat_history": [], "clarifying_questions": [], "created_at": when, "updated_at": when,
        "task_type": "single", "schedule": None, **extra,
    }
    return await app.tasks.repo.insert_task(fields)


async def notify(app: SentientApp, kind: str, body: str, **kw) -> dict:
    """Create a notification. With --keep-db, first drop this script's copy from an earlier run
    (same kind, title and body) and its suggestion row, leaving every other notification alone."""
    if KEEP_DB:
        rows = await app.store.fetchall(
            "SELECT id FROM notifications WHERE kind = ? AND body = ? AND IFNULL(title, '') = ?",
            (kind, body, kw.get("title") or ""),
        )
        for row in rows:
            await app.store.execute("DELETE FROM proactive_suggestions WHERE notification_id = ?", (row["id"],))
            await app.store.execute("DELETE FROM notifications WHERE id = ?", (row["id"],))
    return await app.notifications.create(kind, body, **kw)


async def suggestion_row(app: SentientApp, note: dict, payload: dict, when: str, threshold: float) -> None:
    s = payload["suggestion"]
    ev = s["source_event"]
    await app.store.execute(
        "INSERT INTO proactive_suggestions(id, notification_id, suggestion_type, description, status, confidence, threshold,"
        " source, event_type, item_id, payload, context, task_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"sug_{note['id']}", note["id"], s["suggestion_type"], s["description"], payload["status"], s["confidence"],
         threshold, ev["source"], ev["event_type"], ev.get("item_id"), json.dumps(payload), "{}", payload.get("task_id"), when),
    )


# ----------------------------------------------------------------------------- seed
async def seed() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    if not KEEP_DB:
        for suffix in ("", "-wal", "-shm"):
            p = HOME / f"sentient.db{suffix}"
            if p.exists():
                p.unlink()

    cfg = load_config()
    cfg.assistant.onboarding_complete = True
    cfg.assistant.user_name = cfg.assistant.user_name or "Maya"
    cfg.memory.extract_after_turn = False
    cfg.chat.auto_title = False
    cfg.proactivity.enabled = True
    cfg.proactivity.quiet_hours = "22:30-07:00"
    cfg.integrations.mcp_servers = {
        "linear": {
            "transport": "stdio",
            "command": "linear-mcp-server-not-installed",
            "args": ["--stdio"],
            "url": None,
            "env_keys": ["LINEAR_API_KEY"],
            "enabled": True,
        }
    }
    save_config(cfg)

    app = SentientApp(cfg, llm=TinyFakeProvider(), db_path=HOME / "sentient.db", enable_background=False)
    await app.start()
    try:
        await seed_integrations(app)
        await seed_notifications(app)
        await seed_proactivity(app)
        await seed_brief(app)
    finally:
        await app.stop()
    print(f"seeded {HOME}" + (" (kept existing database)" if KEEP_DB else ""))


async def seed_integrations(app: SentientApp) -> None:
    mgr = app.integrations
    await mgr._save_state("github", connected=True, account_label="maya-northwind", status="connected", error=None,
                          connected_at=ago(days=12))
    await mgr._save_state("gcalendar", connected=True, account_label="maya@northwind.example", status="connected",
                          error=None, connected_at=ago(days=30))
    await mgr._save_state("gmail", connected=False, account_label=None, status="error",
                          error="Your Google sign-in expired. Reconnect so Sentient can keep watching your inbox.")
    await mgr.set_privacy_filters("gmail", {
        "keywords": ["bank statement", "OTP", "salary slip"],
        "emails": ["hr@northwind.example", "alerts@citybank.example"],
        "labels": ["Finance", "Personal/Health"],
    })
    await mgr.set_privacy_filters("gcalendar", {"keywords": ["therapy"], "emails": [], "labels": []})


async def seed_notifications(app: SentientApp) -> None:
    # 1. info (3 days ago, read)
    note = await notify(app, "info", "Sentient is set up and running on your computer. Connect your apps from **Integrations** to unlock suggestions.",
                        title="Welcome to Sentient")
    await backdate(app, note, ago(days=3, hours=2), read=True)

    # 2. error (2 days ago, read)
    note = await notify(app, "error", "Google sign-in for **Gmail** expired, so new email isn't being checked. Reconnect it from Integrations.",
                        title="Gmail needs attention", payload={"integration": "gmail"})
    await backdate(app, note, ago(days=2, hours=5), read=True)

    # 3. skill (yesterday)
    note = await notify(app, "skill", "I noticed a repeatable way you turn client feedback into design to-dos and saved it as **client-feedback-triage**. Review it before I use it.",
                        title="New skill ready for review",
                        payload={"skill": "client-feedback-triage", "action": "create", "origin": "background_review"})
    await backdate(app, note, ago(days=1, hours=3))

    # 4. approved proactive suggestion (yesterday) with a real task row
    task_id = await make_task(
        app, "seed-notif-dentist", "Add dentist appointment to calendar",
        "Add the dentist appointment on Friday at 5:30 PM to your calendar", "completed", ago(days=1, hours=6),
        source="proactive", original_context={"source": "proactive", "suggestion_type": "add_calendar_event"},
    )
    approved = {
        "suggestion": {
            "suggestion_type": "add_calendar_event",
            "description": "Add the dentist appointment on Friday at 5:30 PM to your calendar",
            "action_details": {"action_type": "create_event", "summary": "Dentist with Dr. Mehta", "start": "Fri 17:30", "duration_minutes": 45},
            "reasoning": "The clinic confirmed a Friday 5:30 PM slot and there's nothing on your calendar at that time yet.",
            "confidence": 0.88,
            "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Bright Smile Dental: Your appointment is confirmed",
                             "item_id": "18f1c2d9a7b3e001", "url": "https://mail.example/inbox/18f1c2d9a7b3e001"},
        },
        "status": "approved",
        "task_id": task_id,
    }
    note = await notify(app, "proactive", approved["suggestion"]["description"], title="Suggestion from Gmail", payload=approved)
    await backdate(app, note, ago(days=1, hours=7), read=True)
    await suggestion_row(app, note, approved, ago(days=1, hours=7), 0.6)

    # 5. task: plan ready for approval (today)
    plan_task = await make_task(app, "seed-notif-client-update", "Weekly update for Lumen Health",
                                "Draft and send the weekly project update to Lumen Health every Friday",
                                "approval_pending", ago(hours=4),
                                plan=[{"tool": "gmail", "description": "Collect this week's notes and Figma comments for Lumen Health"},
                                      {"tool": "gdocs", "description": "Draft a short update with progress, next steps and open questions"},
                                      {"tool": "gmail", "description": "Send the update to clients@lumenhealth.example after you approve it"}])
    note = await notify(app, "task", "I've created a new plan for you: 'Weekly update for Lumen Health'", title="Plan ready for approval",
                        payload={"task_id": plan_task, "event": "approval_needed"})
    await backdate(app, note, ago(hours=3, minutes=40))

    # 6. task completed (today)
    done_task = await make_task(app, "seed-notif-newsletters", "Summarise unread design newsletters",
                                "Summarise my unread design newsletters into five bullet points", "completed", ago(hours=3))
    note = await notify(app, "task", "Task 'Summarise unread design newsletters' has finished with status: completed.", title="Task completed",
                        payload={"task_id": done_task, "event": "run_completed"})
    await backdate(app, note, ago(hours=2, minutes=15))

    # 7. pending gcalendar suggestion (today)
    cal = {
        "suggestion": {
            "suggestion_type": "prepare_meeting_brief",
            "description": "Prepare a one-page brief for tomorrow's kickoff with Paperkite",
            "action_details": {"action_type": "create_document", "title": "Paperkite kickoff brief",
                               "include": ["last email from Leela", "open questions", "screens to show"]},
            "reasoning": "The kickoff is tomorrow at 10:00 and the invite links the project brief. You usually review notes before client meetings, and there's no prep doc yet.",
            "confidence": 0.74,
            "source_event": {"source": "gcalendar", "event_type": "new_event", "summary": "Kickoff with Paperkite (tomorrow, 10:00)",
                             "item_id": "5k2v9d0qpaperkite", "url": "https://calendar.example/event?eid=5k2v9d0qpaperkite"},
        },
        "status": "pending",
        "task_id": None,
    }
    note = await notify(app, "proactive", cal["suggestion"]["description"], title="Suggestion from Calendar", payload=cal)
    await backdate(app, note, ago(minutes=58))
    await suggestion_row(app, note, cal, ago(minutes=58), 0.7)

    # 8. approval (today)
    note = await notify(
        app, "approval",
        "Your task **Weekly update for Lumen Health** wants to send an email to **clients@lumenhealth.example**.",
        title="Approval needed",
        payload={"approval_id": "apr_demo_weekly_update", "call_id": "call_demo_1", "name": "gmail_send", "risk": "send",
                 "reason": "Send the weekly project update", "arguments": {"to": "clients@lumenhealth.example", "subject": "Weekly update, week 41"},
                 "task_id": plan_task},
    )
    await backdate(app, note, ago(minutes=31))

    # 9. pending gmail suggestion, high confidence (most recent)
    mail = {
        "suggestion": {
            "suggestion_type": "draft_email_reply",
            "description": "Draft a reply to Kavya confirming Thursday's design review at 3 PM",
            "action_details": {"action_type": "draft_email", "to": "kavya@northwind.example", "subject": "Re: Design review moved to Thursday?",
                               "thread_id": "18f2a7c1be44d0a2", "points": ["Thursday 3 PM works", "share the Figma link beforehand"]},
            "reasoning": "Kavya asked whether Thursday 3 PM works. Your calendar is free then, and you've accepted every design review this month.",
            "confidence": 0.91,
            "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Kavya Iyer: Design review moved to Thursday?",
                             "item_id": "18f2a7c1be44d0a2", "url": "https://mail.example/inbox/18f2a7c1be44d0a2"},
        },
        "status": "pending",
        "task_id": None,
    }
    note = await notify(app, "proactive", mail["suggestion"]["description"], title="Suggestion from Gmail", payload=mail)
    await backdate(app, note, ago(minutes=12))
    await suggestion_row(app, note, mail, ago(minutes=12), 0.55)

    # 10. pending follow-up: an email waiting on Maya's reply for 4 days, with a ready draft (most recent)
    draft = ("Hi Leela,\n\nThanks for the revised quote, and sorry for the slow reply. The new scope looks right to me. "
             "Could you send the updated timeline as well? Then I can confirm the project.\n\nMaya")
    follow_up = {
        "suggestion": {
            "suggestion_type": "follow_up_reply",
            "description": "Leela is waiting for your reply about the revised quote",
            "action_details": {"action_type": "send_email_reply", "to": "leela@paperkite.example",
                               "subject": "Re: Revised quote for the Paperkite site", "body": draft},
            "reasoning": "Leela Menon wrote to you directly 4 days ago and you have not replied yet.",
            "confidence": 0.86,
            "source_event": {"source": "gmail", "event_type": "follow_up", "summary": "Leela Menon: Revised quote for the Paperkite site",
                             "item_id": "18f2b9e05c1a7f33:<quote-v2@paperkite.example>",
                             "url": "https://mail.example/inbox/18f2b9e05c1a7f33"},
            "follow_up": {"kind": "waiting_on_you", "person": "Leela Menon", "person_email": "leela@paperkite.example",
                          "to": "leela@paperkite.example", "subject": "Re: Revised quote for the Paperkite site",
                          "draft": draft, "days_waiting": 4, "thread_id": "18f2b9e05c1a7f33", "message_id": "18f2b9e05c1a7f33"},
        },
        "status": "pending",
        "task_id": None,
    }
    quoted = "\n".join(f"> {line}" if line else ">" for line in draft.split("\n"))
    note = await notify(app, "proactive", f"{follow_up['suggestion']['description']}\n\n{quoted}",
                        title="Gmail: Leela Menon: Revised quote for the Paperkite site", payload=follow_up)
    await backdate(app, note, ago(minutes=5))
    await suggestion_row(app, note, follow_up, ago(minutes=5), 0.7)


async def seed_proactivity(app: SentientApp) -> None:
    store = app.store
    prefs = [
        ("draft_email_reply", 3, 4, 1),
        ("add_calendar_event", 2, 2, 0),
        ("prepare_meeting_brief", -1, 1, 2),
        ("unsubscribe_newsletter", -3, 0, 3),
    ]
    for stype, score, approvals, dismissals in prefs:
        await store.execute(
            "INSERT OR REPLACE INTO proactive_preferences(suggestion_type, score, approvals, dismissals, updated_at) VALUES(?,?,?,?,?)",
            (stype, score, approvals, dismissals, ago(hours=1)),
        )
    await store.execute(
        "INSERT OR REPLACE INTO proactive_sources(source, connected, last_poll_at, last_success_at, last_error, items_seen, updated_at)"
        " VALUES(?,?,?,?,?,?,?)",
        ("gmail", 0, ago(minutes=14), ago(hours=26), "Your Google sign-in expired, so the last check could not run.", 212, ago(minutes=14)),
    )
    await store.execute(
        "INSERT OR REPLACE INTO proactive_sources(source, connected, last_poll_at, last_success_at, last_error, items_seen, updated_at)"
        " VALUES(?,?,?,?,?,?,?)",
        ("gcalendar", 1, ago(minutes=4), ago(minutes=4), None, 57, ago(minutes=4)),
    )


def _brief_item(section: str, n: int, text: str, why: str, link: str | None = None, feedback: str | None = None) -> dict:
    return {"id": f"{section}-demo{n}", "section": section, "text": text, "link": link, "why": why, "feedback": feedback}


async def seed_brief(app: SentientApp) -> None:
    """The Daily Brief and Evening Brief tasks (normal recurring tasks) and the brief delivered this morning."""
    state = await app.proactivity.brief.setup({})
    await app.proactivity.brief.setup({"kind": "evening"})
    items = [
        _brief_item("calendar", 1, "09:30 Design critique with the Paperkite team", "On your calendar today",
                    "https://calendar.google.com/calendar/r/day"),
        _brief_item("calendar", 2, "13:00 Lunch with Ishaan (Indiranagar)", "On your calendar today",
                    "https://calendar.google.com/calendar/r/day"),
        _brief_item("calendar", 3, "16:00 Client review: Northwind onboarding flow", "On your calendar today",
                    "https://calendar.google.com/calendar/r/day"),
        _brief_item("email", 1, "Leela Menon is waiting for your reply about the revised quote", "No reply for 4 days",
                    "https://mail.google.com/mail/u/0/#inbox", "up"),
        _brief_item("email", 2, "Rohan Das asks for the final icon set by Thursday", "Unread and marked important in Gmail",
                    "https://mail.google.com/mail/u/0/#inbox"),
        _brief_item("tasks", 1, "Book a venue for the team offsite: plan waiting for your approval",
                    "Waiting for you in Tasks", "/tasks"),
        _brief_item("weather", 1, "Bengaluru: partly cloudy, 24°C now, high 29°C, 40% chance of rain",
                    "Weather for Bengaluru, India, your city in Settings"),
    ]
    labels = {"calendar": "Calendar", "email": "Email", "tasks": "Tasks", "weather": "Weather"}
    brief = {
        "kind": "morning", "day": NOW.astimezone().date().isoformat(), "title": f"Your Daily Brief for {NOW.astimezone().strftime('%A')}",
        "sections": [{"id": k, "label": v, "feedback": None} for k, v in labels.items()],
        "items": items, "skipped": [], "expires_at": app.proactivity.brief._expires_at(NOW),
    }
    rows = await app.store.fetchall("SELECT id FROM notifications WHERE kind = 'brief'")
    for row in rows:  # a re-run replaces this script's brief
        await app.store.execute("DELETE FROM notifications WHERE id = ?", (row["id"],))
    note = await app.notifications.create("brief", app.proactivity.brief.as_text(brief), title=brief["title"],
                                          payload={"brief": brief, "status": "active", "task_id": state["task_id"]})
    await backdate(app, note, ago(minutes=50))


if __name__ == "__main__":
    asyncio.run(seed())
