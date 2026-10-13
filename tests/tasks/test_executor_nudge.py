"""Regression from the real qwen3:8b run: a run that only announces its next step is nudged to finish,
and the executor never gets skill_save (it looped saving a pending skill it could not load). A run that answers
nothing after its tool calls is asked once more, then answers from what the tools found (#322)."""

from sentient import paths
from sentient.agent import findings
from sentient.agent.loop import FINAL_ANSWER_NUDGE
from sentient.tasks.executor import announces_unfinished_work
from tests.conftest import FakeProvider
from tests.memory.helpers import add_fact
from tests.tasks.conftest import PLAN_FILES, REFINE_ONCE, RESULT, stream_calls


def tc(tool_name: str, **arguments):
    return {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}


def test_announcement_detection():
    assert announces_unfinished_work("I will now proceed to write the poem about the Pune monsoon.")
    assert announces_unfinished_work("Let's create a four-line poem that captures the season.")
    assert not announces_unfinished_work("I saved the poem to monsoon.txt.")
    assert not announces_unfinished_work("Here is the summary of your inbox: two invoices and a meeting request.")
    assert not announces_unfinished_work("")


async def test_run_that_only_announces_is_nudged_to_finish(make_app):
    llm = FakeProvider(
        replies=[
            "I will now proceed to write the haiku and save it.",
            [tc("file_write", name="haiku.txt", content="rain on tin roofs")],
            "I saved the haiku to haiku.txt.",
        ],
        json_replies=[REFINE_ONCE, PLAN_FILES, RESULT],
    )
    app = await make_app(llm)
    created = await app.tasks.create_task("Write a haiku and save it to haiku.txt")
    await app.tasks.drain()
    await app.tasks.approve(created["task_id"])
    await app.tasks.drain()

    task = await app.tasks.get(created["task_id"])
    run = task["runs"][0]
    assert run["status"] == "completed", run["error"]
    assert (paths.files_dir() / "haiku.txt").read_text(encoding="utf-8") == "rain on tin roofs"
    infos = [u["message"]["content"] for u in run["progress_updates"] if u["message"]["type"] == "info"]
    assert any("carry on" in m for m in infos)
    finals = [u["message"]["content"] for u in run["progress_updates"] if u["message"]["type"] == "final_answer"]
    assert finals == ["I saved the haiku to haiku.txt."]

    tools = {t["function"]["name"] for t in stream_calls(llm)[0]["tools"]}
    assert "skill_view" in tools and "skill_save" not in tools


class LastSaid(FakeProvider):
    """Keeps what the last message was at each model call (the loop removes the nudge from its list later)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_said: list[str] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        last = messages[-1]
        self.last_said.append("tool" if last["role"] == "tool" else str(last.get("content")))
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


async def _run_with(make_app, config, replies: list, trigger_data: dict | None = None) -> tuple[LastSaid, dict]:
    """A recurring price check (the real qwen3:8b case of #322) run once with scripted executor replies."""
    config.memory.facts_top_k = 0  # only the tool finds the fact
    llm = LastSaid(json_replies=[dict(RESULT)])
    app = await make_app(llm)
    await add_fact(app.memory, "Last time Maya checked, Sapiens cost Rs. 599 at Lakeside Books", source="manual")
    llm.replies.extend(replies)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Sapiens price", "description": "Remind me what the Sapiens price was last time", "status": "active",
        "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 30},
        "plan": [{"tool": "memory", "description": "Recall the Sapiens price from last time"}],
        "created_at": now, "updated_at": now,
    })
    task = await app.tasks.repo.get_task(task_id)
    await app.tasks._start_run(task, trigger_data=trigger_data)
    await app.tasks.drain()
    return llm, (await app.tasks.get(task_id))["runs"][-1]


def _updates(run: dict, kind: str) -> list[str]:
    return [u["message"]["content"] for u in run["progress_updates"] if u["message"]["type"] == kind]


async def test_empty_answer_after_a_memory_look_up_is_nudged_then_answered(make_app, config):
    llm, run = await _run_with(make_app, config, [[tc("memory_recall", query="Sapiens price")], "", "It was Rs. 599."])
    assert run["status"] == "completed", run["error"]
    assert _updates(run, "final_answer") == ["It was Rs. 599."]
    assert llm.last_said[1:] == ["tool", FINAL_ANSWER_NUDGE]  # the look-up, the empty answer, then the nudge


async def test_still_empty_after_the_nudge_completes_with_what_the_look_up_found(make_app, config):
    llm, run = await _run_with(make_app, config, [[tc("memory_recall", query="Sapiens price")], "", ""])
    assert run["status"] == "completed", run["error"]
    [final] = _updates(run, "final_answer")
    fact = "- Memory: Last time Maya checked, Sapiens cost Rs. 599 at Lakeside Books"
    assert final.splitlines() == [findings.INTRO, fact]
    assert any("lists what its steps found" in m for m in _updates(run, "info"))
    assert len(llm.last_said) == 3  # the look-up, the empty answer, the nudge: no continue nudge after it


async def test_a_check_scripts_result_is_part_of_the_answer(make_app, config):
    trigger = {"script_result": {"price": "Rs. 549"}, "message": "The check found a result."}
    _, run = await _run_with(make_app, config, [[tc("memory_recall", query="Sapiens price")], "", ""], trigger)
    assert run["status"] == "completed", run["error"]
    [final] = _updates(run, "final_answer")
    assert "- The check: price: Rs. 549" in final.splitlines()
    assert "- Memory: Last time Maya checked, Sapiens cost Rs. 599 at Lakeside Books" in final.splitlines()
