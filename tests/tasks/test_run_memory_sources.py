"""Memory sources on task runs (issue #136): a run records the memories it had in mind, keeps them across a pause
and a restart, and the run API returns them."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory import review
from tests.conftest import FakeProvider, tool_call
from tests.memory.helpers import add_fact
from tests.tasks.conftest import RESULT, stream_calls

ASK = [tool_call("ask_user", question="Which hotel should I book?", options=["Taj", "Oberoi"])]


async def _start(app, name: str = "Plan the Goa trip") -> tuple[str, str]:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": "approval_pending", "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "time", "description": "Check today's date"}], "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    return task_id, task["runs"][-1]["run_id"]


def _ids(sources: list[dict]) -> list[tuple]:
    return [(s["kind"], s["id"], s["via"]) for s in sources]


async def test_prompt_facts_and_recalled_facts_are_recorded_once(make_app, config):
    config.memory.min_similarity = 0.0
    config.memory.facts_top_k = 1
    llm = FakeProvider(json_replies=[dict(RESULT)])
    app = await make_app(llm)
    trip = await add_fact(app.memory, "Maya is planning a Goa trip in December", source="manual")
    budget = await add_fact(app.memory, "Maya keeps hotel budgets under 8000 rupees a night", source="file:notes.md")
    await app.memory.remember(
        "Maya wants a sea view room", source="gmail", use_llm=False, review=review.note("Gmail", "sea view please")
    )
    [held] = await app.memory.pending_facts()
    llm.replies.extend([[tool_call("memory_recall", query="Goa trip hotel budget sea view")], "Shortlisted two hotels."])

    task_id, run_id = await _start(app)

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed", run["error"]
    system = stream_calls(llm)[0]["messages"][0]["content"]
    assert "Goa trip in December" in system and "8000 rupees" not in system  # one fact in the prompt
    assert _ids(run["memory_sources"]) == [("fact", trip, "prompt"), ("fact", budget, "tool")]
    assert run["memory_sources"][1] == {
        "kind": "fact", "id": budget, "text": "Maya keeps hotel budgets under 8000 rupees a night",
        "source": "file:notes.md", "via": "tool",
    }
    assert held["id"] not in [s["id"] for s in run["memory_sources"]]  # pending memories are never used
    assert (await app.tasks.repo.get_run(run_id))["memory_sources"] == run["memory_sources"]


async def test_only_rows_the_model_read_count(make_app, config):
    config.memory.facts_top_k = 0  # nothing in the prompt: only the tool finds them
    config.chat.tool_result_max_chars = 400
    llm = FakeProvider(json_replies=[dict(RESULT)])
    app = await make_app(llm)
    for i in range(30):
        await add_fact(app.memory, f"Maya noted detail number {i} about the Goa trip plans", source="manual")
    llm.replies.extend([[tool_call("memory_recall", query="Goa trip", limit=20)], "Noted."])

    task_id, _ = await _start(app)

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed", run["error"]
    assert 0 < len(run["memory_sources"]) < 20
    tool_msg = next(m for m in stream_calls(llm)[-1]["messages"] if m["role"] == "tool")
    for s in run["memory_sources"]:
        assert s["via"] == "tool" and s["text"] in tool_msg["content"]


async def test_sources_survive_a_pause_and_a_restart(make_app, config):
    config.memory.min_similarity = 0.0
    config.memory.facts_top_k = 1
    llm = FakeProvider(replies=[[tool_call("memory_recall", query="hotel budget")], ASK])
    app = await make_app(llm, db_name="sources-restart.db")
    first = await add_fact(app.memory, "Maya is planning a Goa trip in December", source="manual")
    budget = await add_fact(app.memory, "Maya keeps hotel budgets under 8000 rupees a night", source="manual")
    later = await add_fact(app.memory, "Maya's partner Arjun is allergic to shellfish", source="conversation")
    task_id, run_id = await _start(app)

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "waiting_for_user"
    before = _ids(run["memory_sources"])
    assert ("fact", first, "prompt") in before and ("fact", budget, "tool") in before
    await app.stop()  # quit while the run waits for the answer

    llm2 = FakeProvider(
        replies=[[tool_call("memory_recall", query="Arjun allergy shellfish")], "Booked the Taj."],
        json_replies=[dict(RESULT)],
    )
    app2 = await make_app(llm2, db_name="sources-restart.db")
    await app2.tasks.answer_question(task_id, run_id, "Taj")
    await app2.tasks.drain()

    run = (await app2.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed", run["error"]
    after = _ids(run["memory_sources"])
    assert after[: len(before)] == before  # kept, in order, and merged with what the resumed part looked up
    assert ("fact", later, "tool") in after and len(after) == len(set(after))


async def test_a_retry_keeps_the_sources_of_the_transcript_it_continues(make_app, config):
    config.memory.min_similarity = 0.0
    config.memory.facts_top_k = 1
    llm = FakeProvider(replies=[ASK])
    app = await make_app(llm)
    fid = await add_fact(app.memory, "Maya is planning a Goa trip in December", source="manual")
    task_id, run_id = await _start(app)
    await app.tasks.cancel_run(task_id, run_id)
    llm.replies.append("Done.")
    llm.json_replies.append(dict(RESULT))

    await app.tasks.retry_run(task_id, run_id)
    await app.tasks.drain()

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["retry_of"] == run_id and run["status"] == "completed", run["error"]
    assert _ids(run["memory_sources"]) == [("fact", fid, "prompt")]


async def test_a_run_without_memories_has_none(make_app):
    llm = FakeProvider(replies=["Done."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, _ = await _start(app)
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed" and run["memory_sources"] == []


def test_the_run_api_returns_sources(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    config.memory.min_similarity = 0.0
    llm = FakeProvider(replies=["Done."], json_replies=[dict(RESULT)])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "api.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        fid = c.portal.call(add_fact, core.memory, "Maya is planning a Goa trip in December")
        task_id = c.portal.call(_start, core)[0]
        source = {"kind": "fact", "id": fid, "text": "Maya is planning a Goa trip in December", "source": "conversation",
                  "via": "prompt"}
        [run] = c.get(f"/api/tasks/{task_id}").json()["runs"]
        assert run["memory_sources"] == [source]
