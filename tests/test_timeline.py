"""The activity timeline (#147): one list of what Sentient did and what it will do, read from existing tables."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from sentient.activity.timeline import decode_cursor
from sentient.app import SentientApp
from sentient.evolution.log import log_event
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest
from sentient.store.db import new_id
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

NOW = datetime.now(UTC).replace(microsecond=0)


def ago(**kw) -> str:
    return (NOW - timedelta(**kw)).isoformat()


def ahead(**kw) -> str:
    return (NOW + timedelta(**kw)).isoformat()


def _mail(sent: list[str]) -> ToolPlugin:
    @tool("mail_send", risk=Risk.send, describe_fn=lambda args, ctx: {"target": f"to {args.get('to')}"})
    async def mail_send(ctx: ToolContext, to: str, body: str) -> dict:
        """Send an email."""
        sent.append(to)
        return {"ok": True}

    @tool("mail_search", risk=Risk.read)
    async def mail_search(ctx: ToolContext, query: str) -> dict:
        """Search emails."""
        return {"results": []}

    class Mail(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [mail_send, mail_search]

    return Mail()


async def _app(config, isolated_home, llm=None) -> SentientApp:
    config.chat.tool_selection = "all"
    config.tools.approvals.mode = "ask"
    config.assistant.timezone = "UTC"
    return await SentientApp(config, llm=llm or FakeProvider(), db_path=isolated_home / "t.db",
                             enable_background=False).start()


async def _turn(s: SentientApp, sid: str, decision: str) -> None:
    async for ev in s.agent.run_turn(sid, "go", channel="cli"):
        if isinstance(ev, ApprovalRequest):
            assert s.approvals.resolve(ev.approval_id, decision)


async def _task(s: SentientApp, name: str, *, schedule: dict | None = None, context: dict | None = None,
                task_type: str = "single", enabled: int = 1, status: str = "active", next_at: str | None = None,
                script: dict | None = None) -> str:
    tid = new_id()
    await s.store.execute(
        "INSERT INTO tasks(id, name, status, task_type, schedule, original_context, enabled, next_execution_at,"
        " script, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (tid, name, status, task_type, json.dumps(schedule) if schedule else None,
         json.dumps(context) if context else None, enabled, next_at, json.dumps(script) if script else None,
         ago(days=30), ago(days=30)),
    )
    return tid


async def _run(s: SentientApp, task_id: str, at: str, status: str = "completed", **fields) -> str:
    rid = new_id()
    await s.store.execute(
        "INSERT INTO task_runs(id, task_id, status, result, error, trigger_data, created_at) VALUES(?,?,?,?,?,?,?)",
        (rid, task_id, status, json.dumps(fields.get("result")) if fields.get("result") else None, fields.get("error"),
         json.dumps(fields.get("trigger")) if fields.get("trigger") else None, at),
    )
    return rid


async def _note(s: SentientApp, kind: str, title: str, body: str, payload: dict, at: str) -> str:
    nid = new_id()
    await s.store.execute(
        "INSERT INTO notifications(id, kind, title, body, payload, created_at) VALUES(?,?,?,?,?,?)",
        (nid, kind, title, body, json.dumps(payload), at),
    )
    return nid


async def _all(s: SentientApp, **kw) -> list[dict]:
    out: list[dict] = []
    before = None
    for _ in range(100):
        page = await s.timeline.past(before=before, **kw)
        out += page["items"]
        before = page["next"]
        if before is None:
            return out
    raise AssertionError("paging never ended")


async def test_chat_tool_calls_and_approvals_say_what_and_why(config, isolated_home):
    sent: list[str] = []
    llm = FakeProvider(replies=[
        [tool_call("mail_search", query="Saturday plans")], [tool_call("mail_send", to="Arjun", body="See you")], "done",
        [tool_call("mail_send", to="Riya", body="No")], "ok",
    ])
    s = await _app(config, isolated_home, llm)
    try:
        s.registry.register(_mail(sent))
        sid = await s.store.create_session(channel="web", title="Weekend plans")
        await _turn(s, sid, "allow")
        await _turn(s, sid, "deny")
        assert sent == ["Arjun"]
        items = (await s.timeline.past())["items"]
        titles = [i["title"] for i in items]
        assert titles == [
            'Asked you before "Send" in Mail (to Riya)',
            'Asked you before "Send" in Mail (to Arjun)',
            "Used Mail: Search",
        ]
        declined, approved, search = items
        assert declined["status"] == "declined" and declined["kind"] == "approvals"
        assert declined["detail"] == "You said no, so it was not done."
        assert approved["status"] == "done" and approved["detail"] == "You chose Allow once."
        assert approved["why"].startswith("Sentient read content from Mail in this chat")  # the search read mail
        assert search["kind"] == "chat" and search["why"] == 'You asked in the chat "Weekend plans"'
        assert search["detail"] == "query: Saturday plans" and search["app"] == "mail" and search["app_label"] == "Mail"
        assert search["ref"]["type"] == "chat" and search["ref"]["id"] == sid
        stored = await s.store.fetchone("SELECT approval FROM messages WHERE role = 'tool' AND approval IS NOT NULL")
        assert json.loads(stored["approval"])["decision"] in {"allow", "deny"}
        # filters
        assert [i["title"] for i in (await s.timeline.past(kinds=["chat"]))["items"]] == ["Used Mail: Search"]
        assert len((await s.timeline.past(kinds=["approvals"]))["items"]) == 2
        assert len((await s.timeline.past(q="riya"))["items"]) == 1
        assert (await s.timeline.past(app="gmail"))["items"] == []
        assert len((await s.timeline.past(app="mail"))["items"]) == 3
        assert {"id": "mail", "label": "Mail"} in await s.timeline.apps()
    finally:
        await s.stop()


async def test_a_calls_details_survive_a_shared_timestamp(config, isolated_home):
    # Windows' clock (Python 3.12) ticks every 15.6 ms: a tool result and the next call often share created_at
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="web", title="Weekend plans")
        search = {"id": "c1", "type": "function",
                  "function": {"name": "mail_search", "arguments": json.dumps({"query": "Saturday plans"})}}
        send = {"id": "c2", "type": "function",
                "function": {"name": "mail_send", "arguments": json.dumps({"to": "Arjun", "body": "See you"})}}
        await s.store.add_message(sid, "assistant", "", tool_calls=[search])
        await s.store.add_message(sid, "tool", json.dumps({"results": []}), tool_call_id="c1", name="mail_search")
        await s.store.add_message(sid, "assistant", "", tool_calls=[send])
        at = "2026-10-20T10:00:00+00:00"
        await s.store.execute("UPDATE messages SET created_at = ? WHERE session_id = ?", (at, sid))
        assert await s.timeline._call_arguments(sid, "c1", at) == {"query": "Saturday plans"}
        assert await s.timeline._call_arguments(sid, "c2", at) == {"to": "Arjun", "body": "See you"}
    finally:
        await s.stop()


async def test_incognito_chats_never_show(config, isolated_home):
    s = await _app(config, isolated_home, FakeProvider(replies=[[tool_call("mail_search", query="secret")], "ok"]))
    try:
        s.registry.register(_mail([]))
        sid = await s.store.create_session(channel="web", incognito=True)
        await _turn(s, sid, "allow")
        assert await s.store.fetchone("SELECT 1 FROM messages WHERE role = 'tool'") is not None
        assert (await s.timeline.past())["items"] == []
        assert await s.timeline.apps() == []  # not even which apps an incognito chat used
    finally:
        await s.stop()


async def test_runs_briefs_checks_notifications_memories_and_skills(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        brief = await _task(s, "Daily Brief", context={"source": "brief", "brief": "morning"},
                            schedule={"type": "recurring", "frequency": "weekly", "days": ["Monday", "Tuesday",
                                      "Wednesday", "Thursday", "Friday"], "time": "07:30", "timezone": "UTC"})
        await _run(s, brief, ago(hours=5), result={"summary": "3 events today"})
        inbox = await _task(s, "Sort my inbox", schedule={"type": "triggered", "source": "gmail", "event": "new_email"})
        failed = await _run(s, inbox, ago(hours=4), status="error", error="Gmail said no")
        await _note(s, "task", "Task failed", "Task 'Sort my inbox' failed", {"task_id": inbox, "event": "run_failed"},
                    ago(hours=4))  # the run already shows it
        await _task(s, "Price of the bike", task_type="script",
                            schedule={"type": "recurring", "frequency": "interval", "interval_minutes": 30},
                            script={"code": "x", "checks": [
                                {"at": ago(hours=3), "outcome": "baseline"},
                                {"at": ago(hours=2), "outcome": "nothing_new", "count": 3, "last_at": ago(minutes=90)},
                                {"at": ago(hours=1), "outcome": "changed", "note": "The price dropped to 9,000"},
                            ]})
        await _note(s, "task", "Send it?", "This task wants to use Send", {
            "task_id": inbox, "run_id": failed, "event": "question", "status": "answered", "answer": "Allow for this task",
            "approval": {"tool": "gmail_send", "label": '"Send" in Gmail', "choices": ["allow", "deny"]},
        }, ago(minutes=50))
        await _note(s, "proactive", "Gmail: Jane", "Draft a reply to Jane", {
            "suggestion": {"description": "Draft a reply to Jane", "source_event": {"source": "gmail",
                           "summary": "Jane: Tuesday?"}}, "status": "dismissed"}, ago(minutes=40))
        await s.store.execute(
            "INSERT INTO facts(content, source, created_at, updated_at) VALUES(?,?,?,?)",
            ("Maya's sister is Riya", "conversation", ago(minutes=30), ago(minutes=30)),
        )
        await log_event(s.store, "skill_created", {"name": "weekly-report", "pending": True}, ts=ago(minutes=20))
        await s.store.execute(
            "INSERT INTO dreams(id, started_at, finished_at, status, trigger, journal_md) VALUES(?,?,?,?,?,?)",
            ("d1", ago(minutes=10), ago(minutes=9), "completed", "schedule", "Tonight I merged 2 memories."),
        )
        await s.store.execute(
            "INSERT INTO usage(model, source, prompt_tokens, completion_tokens, cost_usd, created_at)"
            " VALUES(?,?,?,?,?,?)", ("openai/gpt-4o-mini", "chat", 10, 10, 0.25, ago(minutes=5)),
        )
        items = await _all(s, limit=3)
        by_title = {i["title"]: i for i in items}
        assert [i["at"] for i in items] == sorted((i["at"] for i in items), reverse=True)
        run = by_title["Made your Daily Brief"]
        assert run["kind"] == "briefs" and run["why"] == "It runs every weekday at 7:30 AM"
        assert run["detail"] == "3 events today" and run["ref"] == {"type": "task", "id": brief, "run_id": run["ref"]["run_id"]}
        fail = by_title["Sort my inbox didn't finish"]
        assert fail["status"] == "failed" and fail["why"] == "It runs when a new email arrives in Gmail"
        assert fail["app"] == "gmail" and fail["detail"] == "Gmail said no"
        assert not any(i["title"].startswith("Told you: Task failed") for i in items)
        assert by_title["Price of the bike changed. You were told"]["detail"] == "The price dropped to 9,000"
        nothing = by_title["Checked Price of the bike: nothing new (3 checks)"]
        assert nothing["at"] == ago(minutes=90) and nothing["why"] == "It checks every 30 minutes"
        approval = by_title['Asked you before Send it? used "Send" in Gmail']
        assert approval["kind"] == "approvals" and approval["detail"] == "You answered: Allow for this task"
        suggestion = by_title["Suggested: Draft a reply to Jane"]
        assert suggestion["status"] == "declined" and suggestion["why"] == "Something in Gmail: Jane: Tuesday?"
        assert by_title["Remembered something about you"]["detail"] == "Maya's sister is Riya"
        assert by_title["Drafted a new skill for your review: weekly-report"]["status"] == "waiting"
        assert by_title["Tidied up your memories"]["detail"] == "Tonight I merged 2 memories."
        assert by_title["Spent $0.25 on cloud models"]["detail"] == "1 paid call, most on openai/gpt-4o-mini"
        assert len(items) == len({i["id"] for i in items}) == 11
        kinds = {i["kind"] for i in (await _all(s, kinds=["memory", "skills"]))}
        assert kinds == {"memory", "skills"}
        assert [i["title"] for i in await _all(s, app="gmail")] == [
            "Suggested: Draft a reply to Jane", 'Asked you before Send it? used "Send" in Gmail',
            "Sort my inbox didn't finish",
        ]
    finally:
        await s.stop()


async def test_paging_never_skips_or_repeats_rows_saved_in_the_same_moment(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        same = ago(minutes=1)
        task = await _task(s, "Tidy files")
        for i in range(23):
            await _run(s, task, same if i % 2 else ago(minutes=i + 2))
            await s.store.execute(
                "INSERT INTO facts(content, source, created_at, updated_at) VALUES(?,?,?,?)",
                (f"fact {i}", "manual", same if i % 3 else ago(minutes=i + 2), same),
            )
            await _note(s, "info", f"Note {i}", "Hello", {}, same if i % 4 else ago(minutes=i + 2))
            await _note(s, "task", "done", "done", {"event": "run_completed"}, same)  # never shown
        whole = await _all(s, limit=200)
        assert len(whole) == 69
        for limit in (1, 4, 7, 50):
            paged = await _all(s, limit=limit)
            assert [i["id"] for i in paged] == [i["id"] for i in whole]
        # a filter that drops most notification rows still pages to the end without repeats
        infos = await _all(s, limit=5, kinds=["notifications"])
        assert len(infos) == len({i["id"] for i in infos}) == 23
        assert decode_cursor("nonsense") is None
        # one spending line per day, also when pages end inside a day
        for day in range(5):
            for hour in (1, 5, 9):
                await s.store.execute(
                    "INSERT INTO usage(model, source, prompt_tokens, completion_tokens, cost_usd, created_at)"
                    " VALUES(?,?,?,?,?,?)", ("openai/gpt-4o", "task", 1, 1, 0.5, ago(days=day + 1, hours=hour)),
                )
        for limit in (1, 2, 50):
            cost = await _all(s, limit=limit, kinds=["cost"])
            assert len(cost) == len({i["id"] for i in cost}) >= 5
            assert all(i["title"].startswith("Spent $") for i in cost)
        assert sum(int(i["detail"].split()[0]) for i in await _all(s, kinds=["cost"])) == 15
    finally:
        await s.stop()


async def test_upcoming_lists_waiting_scheduled_paused_backup_and_dream(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        s.config.backups.schedule = "daily"
        daily = await _task(s, "Daily Brief", context={"source": "brief"}, next_at=ahead(hours=2),
                            schedule={"type": "recurring", "frequency": "daily", "time": "06:00", "timezone": "UTC"})
        await _task(s, "Water plants", next_at=ahead(days=1), status="pending",
                    schedule={"type": "once", "run_at": ahead(days=1)})
        await _task(s, "Weekly report", enabled=0, next_at=ahead(days=2),
                    schedule={"type": "recurring", "frequency": "weekly", "days": ["Friday"], "time": "09:00"})
        waiting = await _task(s, "Pay rent", status="waiting_for_user")
        run = await _run(s, waiting, ago(minutes=3), status="waiting_for_user")
        await s.store.execute("UPDATE task_runs SET pending_question = ? WHERE id = ?", (json.dumps({
            "question": "OK to pay?", "options": ["Allow once", "Deny"], "asked_at": ago(minutes=2),
            "approval": {"tool": "pay", "choices": ["allow", "deny"]}, "expires_at": ahead(hours=4)}), run))
        out = await s.timeline.upcoming()
        items = out["items"]
        assert items[0]["title"] == "Pay rent is waiting for your OK" and items[0]["status"] == "waiting"
        brief = [i for i in items if i["ref"] and i["ref"].get("id") == daily]
        assert len(brief) == 7 and brief[0]["title"] == "Will make your Daily Brief"
        assert brief[0]["why"] == "It runs every day at 6:00 AM"
        assert [i["at"] for i in brief] == sorted(i["at"] for i in brief)
        once = next(i for i in items if i["title"] == "Will run Water plants")
        assert once["why"].startswith("You set it for ")
        assert any(i["title"] == "Will back up everything" for i in items)
        assert out["paused"][0]["title"] == "Weekly report is paused" and out["paused"][0]["why"] == "You paused it"
        assert all(i["upcoming"] for i in items)
        only = await s.timeline.upcoming(kinds=["briefs"])
        assert {i["kind"] for i in only["items"]} == {"briefs"} and only["paused"] == []
    finally:
        await s.stop()


def test_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        assert c.get("/api/timeline").status_code == 401
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/timeline", params={"limit": 5, "kinds": "tasks,memory"}).json() == {"items": [], "next": None}
        up = c.get("/api/timeline/upcoming").json()
        assert set(up) == {"items", "paused"}
        assert c.get("/api/timeline/apps").json() == {"apps": []}
