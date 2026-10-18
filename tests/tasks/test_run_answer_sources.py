"""Sources for task runs (#291): the pages, emails, files and apps a run used are recorded on it like a chat reply's
(#174), kept across a pause, and the run API returns them."""

from __future__ import annotations

import httpx
import respx

from sentient import paths
from sentient.integrations.plugins import web as web_mod
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import RESULT

PAGE = (
    "<html><head><title>Greyhound care guide</title></head><body><h1>First week</h1>"
    "<p>Retired racers need a soft bed and quiet days.</p></body></html>"
)
ASK = [tool_call("ask_user", question="Which bed should I order?", options=["Foam", "Orthopaedic"])]


async def _start(app, plan: list[dict], name: str = "Read up on greyhound care") -> tuple[str, str]:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": "approval_pending", "schedule": {"type": "once", "run_at": None},
        "plan": plan, "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    return task_id, task["runs"][-1]["run_id"]


async def test_a_run_records_the_page_and_file_it_read(make_app, monkeypatch):
    monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.215.14", 0))])
    notes = paths.files_dir() / "greyhound-notes.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("Maya's greyhound Biscuit sleeps on the left side.", encoding="utf-8")
    llm = FakeProvider(
        replies=[
            [tool_call("web_fetch", url="https://example.org/greyhounds")],
            [tool_call("file_read", name="greyhound-notes.md")],
            "Biscuit needs a soft bed.",
        ],
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    with respx.mock() as router:
        router.get("https://example.org/greyhounds").mock(
            return_value=httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
        )
        task_id, run_id = await _start(app, [{"tool": "web", "description": "Read the care guide"},
                                             {"tool": "files", "description": "Read Maya's notes"}])

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed", run["error"]
    assert [(s["kind"], s["title"]) for s in run["sources"]] == [
        ("web", "Greyhound care guide"), ("file", "greyhound-notes.md"),
    ]
    assert run["sources"][0]["url"] == "https://example.org/greyhounds" and "soft bed" in run["sources"][0]["snippet"]
    assert (await app.tasks.repo.get_run(run_id))["sources"] == run["sources"]


async def test_sources_survive_a_pause_and_a_run_without_any_has_none(make_app, monkeypatch):
    monkeypatch.setattr(web_mod.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.215.14", 0))])
    llm = FakeProvider(replies=[[tool_call("web_fetch", url="https://example.org/greyhounds")], ASK])
    app = await make_app(llm)
    with respx.mock() as router:
        router.get("https://example.org/greyhounds").mock(
            return_value=httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
        )
        task_id, run_id = await _start(app, [{"tool": "web", "description": "Read the care guide"}])
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "waiting_for_user" and len(run["sources"]) == 1

    llm.replies.append("Ordered the orthopaedic bed.")
    llm.json_replies.append(dict(RESULT))
    await app.tasks.answer_question(task_id, run_id, "Orthopaedic")
    await app.tasks.drain()
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed", run["error"]
    assert [s["url"] for s in run["sources"]] == ["https://example.org/greyhounds"]  # kept across the pause

    llm.replies.append("It is 9 am.")
    llm.json_replies.append(dict(RESULT))
    other, _ = await _start(app, [{"tool": "time", "description": "Check the time"}], name="Check the time")
    assert (await app.tasks.get(other))["runs"][-1]["sources"] == []
