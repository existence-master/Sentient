"""Regression from the real qwen3:8b run: a run that only announces its next step is nudged to finish,
and the executor never gets skill_save (it looped saving a pending skill it could not load)."""

from sentient import paths
from sentient.tasks.executor import announces_unfinished_work
from tests.conftest import FakeProvider
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
