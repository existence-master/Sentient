"""Recurring tasks remember their last runs: earlier summaries and the notepad go into the next run (#162)."""

from datetime import UTC

from sentient.tasks import continuity
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import PLAN_TIME, REFINE_DAILY, REFINE_ONCE, stream_calls


def _result(summary: str) -> dict:
    return {"summary": summary, "links_created": [], "links_found": [], "files_created": [], "tools_used": ["time"]}


async def _recurring(app, prompt="Every day at 9am tell me the date"):
    created = await app.tasks.create_task(prompt)
    await app.tasks.drain()
    return await app.tasks.approve(created["task_id"])


async def _run(app, task_id):
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    return await app.tasks.get(task_id)


def _system(call) -> str:
    return call["messages"][0]["content"]


def _tool_names(call) -> set[str]:
    return {t["function"]["name"] for t in call["tools"]}


async def test_second_run_sees_what_the_first_run_said_and_its_notepad(make_app):
    llm = FakeProvider(
        replies=[
            [tool_call("task_notepad", text="Told the user about the Lisbon fare at 32k.")],
            "The Lisbon fare is 32k.",
            "Nothing new since the last run: the Lisbon fare is still 32k, as reported before.",
        ],
        json_replies=[REFINE_DAILY, PLAN_TIME, _result("Found the Lisbon fare at 32k and told the user."),
                      _result("Nothing new.")],
    )
    app = await make_app(llm)
    task = await _recurring(app)
    task = await _run(app, task["task_id"])
    assert task["notepad"] == "Told the user about the Lisbon fare at 32k."
    assert task["runs"][0]["result"]["tools_used"] == ["time"]  # the notepad is the run's own helper, not a tool used

    task = await _run(app, task["task_id"])
    first, second = stream_calls(llm)[0], stream_calls(llm)[-1]
    assert "task_notepad" in _tool_names(first) and "task_notepad" in _tool_names(second)
    assert "this is the first run of this recurring task" in _system(first)
    system = _system(second)
    assert "What earlier runs of this recurring task told the user" in system
    assert "The Lisbon fare is 32k." in system  # what run 1 told the user
    assert "Told the user about the Lisbon fare at 32k." in system  # its notepad
    assert "Report only what is new." in system
    log = [u["message"].get("content") for u in task["runs"][1]["progress_updates"]]
    assert "Remembering 1 earlier run and its notepad, so it reports only what is new." in log


async def test_one_off_tasks_get_no_memory_of_runs_or_notepad(make_app):
    llm = FakeProvider(replies=["It is Tuesday."], json_replies=[REFINE_ONCE, PLAN_TIME, _result("Said the date.")])
    app = await make_app(llm)
    created = await app.tasks.create_task("Tell me the date")
    await app.tasks.drain()
    await app.tasks.approve(created["task_id"])
    await app.tasks.drain()
    call = stream_calls(llm)[0]
    assert "task_notepad" not in _tool_names(call)
    assert "Earlier runs" not in _system(call)


async def test_notepad_tool_refuses_outside_a_recurring_run(make_app):
    app = await make_app()
    tool = app.registry.get("task_notepad")
    ctx = app.agent.tool_context(None, "task")
    ctx.extra["task_id"] = "nope"
    assert "error" in await tool.call(ctx, {"text": "x"})
    # scoped: never offered in chat
    assert "task_notepad" not in {t.name for t in app.registry.tools()}


async def test_a_run_is_remembered_by_what_it_told_the_user_else_its_summary(make_app):
    llm = FakeProvider(replies=["The fare is 32k."], json_replies=[REFINE_DAILY, PLAN_TIME, _result("Checked fares.")])
    app = await make_app(llm)
    task = await _recurring(app)
    task = await _run(app, task["task_id"])
    runs = await continuity.earlier_runs(app.tasks.repo, task["task_id"], "other", UTC)
    assert [r["note"] for r in runs] == ["The fare is 32k."]
    await app.store.execute("DELETE FROM run_events WHERE run_id = ?", (task["runs"][0]["run_id"],))
    runs = await continuity.earlier_runs(app.tasks.repo, task["task_id"], "other", UTC)
    assert [r["note"] for r in runs] == ["Checked fares."]


async def test_earlier_runs_stay_compact(make_app):
    replies = [f"Run {i} " + "found a lot. " * 80 for i in range(5)]
    llm = FakeProvider(replies=replies, json_replies=[REFINE_DAILY, PLAN_TIME, *[_result(f"Run {i}.") for i in range(5)]])
    app = await make_app(llm)
    task = await _recurring(app)
    for _ in range(5):
        task = await _run(app, task["task_id"])
    system = _system(stream_calls(llm)[-1])
    assert "Run 1 found" in system and "Run 3 found" in system and "Run 0 found" not in system  # the last three
    section = system.split("told the user (most recent last):**", 1)[1].split("**Report only", 1)[0]
    assert len(section) < 3 * continuity.RUN_NOTE_CHARS + 200


async def test_user_can_edit_or_clear_the_notepad(make_app):
    llm = FakeProvider(json_replies=[REFINE_DAILY, PLAN_TIME])
    app = await make_app(llm)
    task = await _recurring(app)
    task = await app.tasks.update(task["task_id"], {"notepad": "  Skip the weekend fares.  \n"})
    assert task["notepad"] == "Skip the weekend fares."
    task = await app.tasks.update(task["task_id"], {"notepad": "x" * 5000})
    assert len(task["notepad"]) == continuity.MAX_NOTEPAD_CHARS
    task = await app.tasks.update(task["task_id"], {"notepad": ""})
    assert task["notepad"] is None
