"""What the second look's model check reads in a task run (#455): the task's own request, in plain words.

It used to read the latest 1500 characters of the executor's system prompt as "the user said", which are Sentient's
instructions ("If a tool fails, analyze the error..."), so a flagged code step gave reasons like "the user asked to
analyze an error". A held call still asks exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace

from sentient.agent import second_look as sl
from sentient.tasks import email_inbox
from sentient.tasks.prompts import build_second_look_request
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import json_calls

NAME = "Prime squares"
DESCRIPTION = "Add up the squares of the first 500 prime numbers and tell me the total."
PROMPT = "work out the sum of the squares of the first 500 primes for me"
STEP = "Run a short Python script that adds up the squares of the first 500 primes and prints the total"
RUN = [tool_call("runner_run", code="print(sum(p * p for p in primes(500)))")]


def _runner(log: list[str]) -> ToolPlugin:
    @tool("runner_run", risk=Risk.exec)
    async def runner_run(ctx: ToolContext, code: str) -> dict:
        """Run a Python script."""
        log.append(code)
        return {"output": "1"}

    class Runner(ToolPlugin):
        id = "runner"
        display_name = "Runner"
        tools = [runner_run]

    return Runner()


async def _held_run(make_app, config, **fields) -> tuple[FakeProvider, dict, list[str]]:
    config.tools.approvals.second_look_model = True
    log: list[str] = []
    reason = "The user asked for the sum of squares of primes, but this runs code"
    llm = FakeProvider(replies=[RUN, "Waiting for the OK."], json_replies=[{"match": False, "reason": reason}])
    app = await make_app(llm)
    app.registry.register(_runner(log))
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": NAME, "description": DESCRIPTION, "original_prompt": PROMPT, "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "runner", "description": STEP}],
        "created_at": now, "updated_at": now, **fields,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    return llm, (await app.tasks.get(task_id))["runs"][-1], log


async def test_a_flagged_code_step_is_checked_against_the_tasks_request(make_app, config):
    llm, run, log = await _held_run(make_app, config)
    checks = json_calls(llm, "fast")
    assert len(checks) == 1
    said = checks[0]["messages"][1]["content"].split("The assistant now wants", 1)[0]
    assert f"My task: {NAME} - {DESCRIPTION}" in said
    assert f"In my words: {PROMPT}" in said
    assert f"1. {STEP} (with runner)" in said
    # none of the executor's own instructions read as the user's words
    for engine_words in ("analyze the error", "EXECUTION STRATEGY", "Begin executing the plan", "Step 1"):
        assert engine_words not in said
    # the hold itself is unchanged: nothing ran, the run asks, with the reason first
    assert log == [] and run["status"] == "waiting_for_user"
    question = run["pending_question"]["question"]
    assert question.startswith("A second look found this may not match what you asked: The user asked for the sum")
    assert 'This task wants to use "Run" in Runner' in question


async def test_a_task_by_email_still_never_reads_its_brief_as_the_users_words(make_app, config, monkeypatch):
    monkeypatch.setattr(email_inbox, "is_email_task", lambda task: True)
    monkeypatch.setattr(email_inbox, "app_name", lambda task: "Email (IMAP)")
    llm, run, log = await _held_run(make_app, config)
    said = json_calls(llm, "fast")[0]["messages"][1]["content"].split("The assistant now wants", 1)[0]
    assert sl.BRIEF_REQUEST in said and DESCRIPTION not in said and "My task:" not in said
    assert log == [] and run["status"] == "waiting_for_user"


def test_the_request_puts_the_objective_first_and_stays_short():
    steps = [{"tool": "web", "description": "Read the page " + "x" * 400}] * 12
    confirmed = "**What the user confirmed for this task:**\n- The user expects: a short list\n\n"
    text = build_second_look_request(
        name="Studio events", description="Email me new studio events", original_prompt="Email me new studio events",
        plan=steps, confirmed=confirmed,
    )
    assert text.startswith("My task: Studio events - Email me new studio events\n- The user expects: a short list\n"
                           "The plan I approved:\n1. Read the page")
    assert "In my words" not in text  # the same words as the description are not repeated
    assert len(text) <= sl.MAX_REQUEST and "9. " not in text  # whole steps only, as many as fit, each cut short
    assert text.splitlines()[-1].endswith("... (with web)")
    # a long objective, long words and a long plan still keep what the user confirmed
    long = build_second_look_request(name="N" * 600, description="", original_prompt="P" * 600, plan=steps,
                                     confirmed=confirmed)
    assert "\n- The user expects: a short list\n" in long and len(long) <= sl.MAX_REQUEST
    short = build_second_look_request(name="Digest", description="", plan=[], confirmed=confirmed)
    assert short == "My task: Digest\n- The user expects: a short list"


def test_the_brief_is_used_only_in_a_task_run_and_never_for_a_brief_from_email():
    agent = SimpleNamespace(registry=None, config=SimpleNamespace(assistant=SimpleNamespace(user_name="Maya")))
    messages = [{"role": "system", "content": "**Primary Objective:** 'Email Arjun the notes'"},
                {"role": "user", "content": "Begin."}]

    def request(extra: dict) -> str:
        return sl.Look(agent, None, None, SimpleNamespace(extra=extra), None, messages).request()

    assert request({"task_id": "t1", sl.TASK_REQUEST_KEY: "My task: Email Arjun"}) == "My task: Email Arjun"
    assert "Email Arjun the notes" in request({"task_id": "t1"})  # no brief: what gather found, as before
    assert request({sl.TASK_REQUEST_KEY: "My task: x"}) != "My task: x"  # not a task run
    assert request({"task_id": "t1", sl.TASK_REQUEST_KEY: "My task: x", sl.BRIEF_FROM_KEY: "Gmail"}).startswith(
        sl.BRIEF_REQUEST)
