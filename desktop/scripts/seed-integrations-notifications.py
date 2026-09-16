"""Seed a throwaway SENTIENT_HOME with demo data for the Integrations, Notifications and Voice UI.

    .venv/Scripts/python.exe desktop/scripts/seed-integrations-notifications.py <SENTIENT_HOME>

No LLM calls (a tiny fake provider is injected), no network, and nothing is written to the
real OS keychain (keychain helpers are swapped for an in-memory dict while this runs).
The database is recreated on every run so screenshots always start from the same state.

Creates:
- integration state: GitHub + Google Calendar connected, Gmail in an error state with privacy filters
- one MCP server config entry whose command doesn't exist (shows the error state when the engine runs)
- notifications of every kind, including 3 proactive suggestions (approved, pending gmail, pending gcalendar)
- proactive preference rows, proactive source poll state and suggestion rows
- assistant.onboarding_complete = true
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_HOME = (
    "C:/Users/SARTHA~1/AppData/Local/Temp/claude/D--Career-Technology-Startup-Existence-Products-Sentient/"
    "3e60fc94-b13a-4e3b-96de-d57037ba3e5b/scratchpad/ui-integrations-home"
)
HOME = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOME).expanduser()
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


async def make_task(app: SentientApp, name: str, prompt: str, status: str, when: str, **extra) -> str:
    fields = {
        "name": name, "description": prompt, "status": status, "priority": 1, "assignee": "ai",
        "original_prompt": prompt, "source": extra.pop("source", "user"), "enabled": True, "model": None,
        "original_context": extra.pop("original_context", {"source": "manual_creation"}), "plan": [],
        "chat_history": [], "clarifying_questions": [], "created_at": when, "updated_at": when,
        "task_type": "single", "schedule": None, **extra,
    }
    return await app.tasks.repo.insert_task(fields)


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
    for suffix in ("", "-wal", "-shm"):
        p = HOME / f"sentient.db{suffix}"
        if p.exists():
            p.unlink()

    cfg = load_config()
    cfg.assistant.onboarding_complete = True
    cfg.assistant.user_name = cfg.assistant.user_name or "Sarthak"
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
    finally:
        await app.stop()
    print(f"seeded {HOME}")


async def seed_integrations(app: SentientApp) -> None:
    mgr = app.integrations
    await mgr._save_state("github", connected=True, account_label="sarthak-k", status="connected", error=None,
                          connected_at=ago(days=12))
    await mgr._save_state("gcalendar", connected=True, account_label="sarthak@example.com", status="connected",
                          error=None, connected_at=ago(days=30))
    await mgr._save_state("gmail", connected=False, account_label=None, status="error",
                          error="Google sign-in expired (invalid_grant). Reconnect so Sentient can keep watching your inbox.")
    await mgr.set_privacy_filters("gmail", {
        "keywords": ["bank statement", "OTP", "salary slip"],
        "emails": ["hr@acme.com", "noreply@mybank.com"],
        "labels": ["Finance", "Personal/Health"],
    })
    await mgr.set_privacy_filters("gcalendar", {"keywords": ["therapy"], "emails": [], "labels": []})


async def seed_notifications(app: SentientApp) -> None:
    n = app.notifications

    # 1. info (3 days ago, read)
    note = await n.create("info", "Sentient is set up and running locally. Connect your apps from **Integrations** to unlock suggestions.",
                          title="Welcome to Sentient")
    await backdate(app, note, ago(days=3, hours=2), read=True)

    # 2. error (2 days ago, read)
    note = await n.create("error", "Google sign-in for **Gmail** expired, so new email isn't being checked. Reconnect it from Integrations.",
                          title="Gmail needs attention", payload={"integration": "gmail"})
    await backdate(app, note, ago(days=2, hours=5), read=True)

    # 3. skill (yesterday)
    note = await n.create("skill", "I noticed a repeatable way you triage GitHub issues and saved it as **github-issue-triage**. Review it before I use it.",
                          title="New skill ready for review",
                          payload={"skill": "github-issue-triage", "action": "create", "origin": "background_review"})
    await backdate(app, note, ago(days=1, hours=3))

    # 4. approved proactive suggestion (yesterday) with a real task row
    task_id = await make_task(
        app, "Add dentist appointment to calendar",
        "Add the dentist appointment on Friday at 5:30 PM to your calendar", "completed", ago(days=1, hours=6),
        source="proactive", original_context={"source": "proactive", "suggestion_type": "add_calendar_event"},
    )
    approved = {
        "suggestion": {
            "suggestion_type": "add_calendar_event",
            "description": "Add the dentist appointment on Friday at 5:30 PM to your calendar",
            "action_details": {"action_type": "create_event", "summary": "Dentist - Dr. Mehta", "start": "Fri 17:30", "duration_minutes": 45},
            "reasoning": "The clinic confirmed a Friday 5:30 PM slot and there's nothing on your calendar at that time yet.",
            "confidence": 0.88,
            "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Smile Dental: Your appointment is confirmed",
                             "item_id": "18f1c2d9a7b3e001", "url": "https://mail.google.com/mail/u/0/#inbox/18f1c2d9a7b3e001"},
        },
        "status": "approved",
        "task_id": task_id,
    }
    note = await n.create("proactive", approved["suggestion"]["description"], title="Suggestion from Gmail", payload=approved)
    await backdate(app, note, ago(days=1, hours=7), read=True)
    await suggestion_row(app, note, approved, ago(days=1, hours=7), 0.6)

    # 5. task: plan ready for approval (today)
    plan_task = await make_task(app, "Weekly investor update", "Draft and send the weekly investor update every Friday",
                                "approval_pending", ago(hours=4))
    note = await n.create("task", "I've created a new plan for you: 'Weekly investor update'", title="Plan ready for approval",
                          payload={"task_id": plan_task, "event": "approval_needed"})
    await backdate(app, note, ago(hours=3, minutes=40))

    # 6. task completed (today)
    done_task = await make_task(app, "Summarise unread newsletters", "Summarise my unread newsletters into five bullet points",
                                "completed", ago(hours=3))
    note = await n.create("task", "Task 'Summarise unread newsletters' has finished with status: completed.", title="Task completed",
                          payload={"task_id": done_task, "event": "run_completed"})
    await backdate(app, note, ago(hours=2, minutes=15))

    # 7. pending gcalendar suggestion (today)
    cal = {
        "suggestion": {
            "suggestion_type": "prepare_meeting_brief",
            "description": "Prepare a one-page brief for tomorrow's investor call with Northwind Capital",
            "action_details": {"action_type": "create_document", "title": "Northwind Capital - call brief",
                               "include": ["last update sent", "open questions", "metrics since last call"]},
            "reasoning": "The call is tomorrow at 10:00 and the invite links last month's deck. You usually review notes before investor calls, and there's no prep doc yet.",
            "confidence": 0.74,
            "source_event": {"source": "gcalendar", "event_type": "new_event", "summary": "Investor call - Northwind Capital (tomorrow, 10:00)",
                             "item_id": "5k2v9d0qnorthwind", "url": "https://calendar.google.com/calendar/event?eid=5k2v9d0qnorthwind"},
        },
        "status": "pending",
        "task_id": None,
    }
    note = await n.create("proactive", cal["suggestion"]["description"], title="Suggestion from Calendar", payload=cal)
    await backdate(app, note, ago(minutes=58))
    await suggestion_row(app, note, cal, ago(minutes=58), 0.7)

    # 8. approval (today)
    note = await n.create(
        "approval",
        "Your task **Weekly investor update** wants to send an email to **investors@northwind.vc**.",
        title="Approval needed",
        payload={"approval_id": "apr_demo_weekly_update", "call_id": "call_demo_1", "name": "gmail_send", "risk": "send",
                 "reason": "Send the weekly investor update", "arguments": {"to": "investors@northwind.vc", "subject": "Weekly update - week 37"},
                 "task_id": plan_task},
    )
    await backdate(app, note, ago(minutes=31))

    # 9. pending gmail suggestion, high confidence (most recent)
    mail = {
        "suggestion": {
            "suggestion_type": "draft_email_reply",
            "description": "Draft a reply to Priya confirming Thursday's design review at 3 PM",
            "action_details": {"action_type": "draft_email", "to": "priya@acme.com", "subject": "Re: Design review moved to Thursday?",
                               "thread_id": "18f2a7c1be44d0a2", "points": ["Thursday 3 PM works", "share the Figma link beforehand"]},
            "reasoning": "Priya asked whether Thursday 3 PM works. Your calendar is free then, and you've accepted every design review this month.",
            "confidence": 0.91,
            "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Priya Sharma: Design review moved to Thursday?",
                             "item_id": "18f2a7c1be44d0a2", "url": "https://mail.google.com/mail/u/0/#inbox/18f2a7c1be44d0a2"},
        },
        "status": "pending",
        "task_id": None,
    }
    note = await n.create("proactive", mail["suggestion"]["description"], title="Suggestion from Gmail", payload=mail)
    await backdate(app, note, ago(minutes=12))
    await suggestion_row(app, note, mail, ago(minutes=12), 0.55)


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
        ("gmail", 0, ago(minutes=14), ago(hours=26), "Gmail poll failed: invalid_grant (token expired or revoked)", 212, ago(minutes=14)),
    )
    await store.execute(
        "INSERT OR REPLACE INTO proactive_sources(source, connected, last_poll_at, last_success_at, last_error, items_seen, updated_at)"
        " VALUES(?,?,?,?,?,?,?)",
        ("gcalendar", 1, ago(minutes=4), ago(minutes=4), None, 57, ago(minutes=4)),
    )


if __name__ == "__main__":
    asyncio.run(seed())
