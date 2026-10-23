"""Memory jobs with Claude Code as the fast model (#474).

Claude Code only answers chats (ADR 0022), so with the fast role on it the add-or-update decision (CUD) was refused
and every memory was added, never updated or skipped. The fast role's jobs now go to another model the user set up
(``claude_code.stand_in``); when there is none, Sentient says so in the log and once in a notification, which
Activity shows under Memory. The model is faked at ``litellm.acompletion``: Claude Code itself is never started,
because a refused call never reaches it.
"""

from __future__ import annotations

import json
import logging

import litellm
import pytest

from sentient.app import SentientApp
from sentient.llm import claude_code
from sentient.llm.provider import LiteLLMProvider, ModelRefused
from tests.conftest import FakeProvider

CC = "claude-code/sonnet"
LOCAL = "ollama_chat/qwen3:8b"


class Provider(LiteLLMProvider):
    """The real provider (roles, fallbacks, Claude Code refusals); embeddings are the fake bag of words."""

    async def embed(self, texts, *, model=None):
        return await FakeProvider().embed(texts)


@pytest.fixture
def cc_config(config):
    config.models.experimental_claude_code = True
    config.models.roles.primary = CC
    config.models.roles.fast = "claude-code/haiku"
    config.memory.duplicate_similarity = 0.999  # near repeats reach the decision
    return config


@pytest.fixture
def model(monkeypatch):
    """Answers every ``litellm.acompletion`` with the next scripted JSON; records which models were asked."""

    class Script:
        asked: list[str] = []
        answers: list[dict] = []

    async def acompletion(model, messages, **kwargs):
        Script.asked.append(model)
        answer = Script.answers.pop(0) if Script.answers else {"topics": ["Personal Identity"], "memory_type": "long-term"}

        class Msg:
            content = json.dumps(answer)

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]
            usage = None

        return Resp()

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return Script


@pytest.fixture
async def make_app(cc_config, isolated_home):
    apps: list[SentientApp] = []

    async def factory() -> SentientApp:
        app = SentientApp(cc_config, llm=Provider(cc_config), db_path=isolated_home / "cc.db", enable_background=False)
        await app.start()
        apps.append(app)
        return app

    yield factory
    for app in apps:
        await app.stop()


def test_stand_in_is_another_model_the_user_set_up(cc_config):
    chain = ["claude-code/haiku"]
    assert claude_code.stand_in(cc_config, "fast", chain) is None  # nothing else set up: the plain refusal stays
    cc_config.models.roles.planner = "ollama_chat/qwen3:14b"
    assert claude_code.stand_in(cc_config, "fast", chain) == "ollama_chat/qwen3:14b"
    cc_config.models.roles.executor = LOCAL
    assert claude_code.stand_in(cc_config, "fast", chain) == LOCAL  # the executor before the planner
    cc_config.models.fallbacks["primary"] = ["claude-code/opus", "openai/gpt-5-mini"]
    assert claude_code.stand_in(cc_config, "fast", chain) == "openai/gpt-5-mini"  # never another Claude Code model
    # only the fast role, and only when every model it has is Claude Code
    assert claude_code.stand_in(cc_config, "planner", chain) is None
    assert claude_code.stand_in(cc_config, "fast", ["claude-code/haiku", "openai/gpt-5-mini"]) is None
    # Claude Code doing everything (ADR 0032) does these jobs itself
    cc_config.models.claude_code_everywhere = True
    assert claude_code.stand_in(cc_config, "fast", chain) is None


async def test_the_decision_runs_on_another_model_and_updates(make_app, cc_config, model, caplog):
    """Chat on Claude Code, the fast role on Claude Code too, and qwen3:8b as the executor: "moved" updates."""
    cc_config.models.roles.executor = LOCAL
    app = await make_app()
    mem = app.memory
    with caplog.at_level(logging.INFO, logger="sentient"):
        first = await mem.remember("Maya's favourite food is biryani.")
        assert first["action"] == "ADD"
        model.answers = [{"action": "UPDATE", "fact_id": first["id"], "content": "Maya's favourite food is dosa.",
                          "topics": ["Preferences"], "memory_type": "long-term"}]
        second = await mem.remember("Maya's favourite food is dosa now.")
    assert second["action"] == "UPDATE" and second["id"] == first["id"]
    assert second["previous"] == "Maya's favourite food is biryani."
    assert set(model.asked) == {LOCAL}  # never Claude Code, never the chat's model
    assert caplog.text.count("so ollama_chat/qwen3:8b does its jobs") == 1  # said once, not per call
    assert "failed for role fast" not in caplog.text  # Claude Code isn't tried just to be refused
    assert "defaulting to ADD" not in caplog.text and "no model could" not in caplog.text
    assert await app.store.fetchall("SELECT id FROM notifications") == []


async def test_a_repeat_is_skipped_on_the_other_model(make_app, cc_config, model):
    cc_config.models.fallbacks["primary"] = [LOCAL]
    app = await make_app()
    first = await app.memory.remember("Maya enjoys hiking on weekends.")
    model.answers = [{"action": "SKIP", "fact_id": first["id"]}]
    again = await app.memory.remember("Maya likes going hiking at weekends.")
    assert again["action"] == "SKIP" and again["id"] == first["id"]
    assert model.asked[-1] == LOCAL
    rows = await app.store.fetchall("SELECT content FROM facts")
    assert [r["content"] for r in rows] == ["Maya enjoys hiking on weekends."]


async def test_no_model_says_so_plainly_once(make_app, model, caplog):
    """Only Claude Code is set up: the memory is still saved, and the log and Activity say no model checked it."""
    app = await make_app()
    with caplog.at_level(logging.INFO, logger="sentient"):
        a = await app.memory.remember("Maya enjoys hiking on weekends.")
        b = await app.memory.remember("Maya likes going hiking at weekends.")
    assert a["action"] == b["action"] == "ADD"
    assert model.asked == []  # nothing was sent anywhere
    assert caplog.text.count("no model could check") == 2  # every time in the log
    assert "Claude Code only answers your chats" in caplog.text
    assert "defaulting to ADD" not in caplog.text
    notes = await app.store.fetchall("SELECT title, body, payload FROM notifications")
    assert len(notes) == 1  # once, not per memory
    assert notes[0]["title"] == "Memories saved without a check"
    assert "may repeat or contradict older ones" in notes[0]["body"]
    assert claude_code.CHATS_ONLY in notes[0]["body"]
    assert json.loads(notes[0]["payload"]) == {"kind": "memory_no_model", "job": "check"}
    items = (await app.timeline.past(kinds=["memory"]))["items"]
    assert any(i["title"] == "Told you: Memories saved without a check" for i in items)


async def test_no_model_keeps_the_same_person_guards(make_app, model):
    """#364 and #471 decide without a model: a move still updates, a sister's home is still added beside Maya's."""
    app = await make_app()
    home = await app.memory.remember("Maya lives in Pune.")
    moved = await app.memory.remember("Maya moved to Mumbai.")
    assert moved["action"] == "UPDATE" and moved["id"] == home["id"]
    sister = await app.memory.remember("Maya's sister Priya lives in Pune.")
    assert sister["action"] == "ADD" and sister["id"] != home["id"]
    row = await app.store.fetchone("SELECT content FROM facts WHERE id = ?", (home["id"],))
    assert row["content"] == "Maya moved to Mumbai."
    assert model.asked == []


async def test_extraction_without_a_model_says_so(make_app, model, caplog):
    app = await make_app()
    with caplog.at_level(logging.WARNING, logger="sentient"):
        assert await app.memory.extract_and_store("I started learning the violin this month.", "Maya") == []
    assert "no model could pick out memories" in caplog.text
    notes = await app.store.fetchall("SELECT title FROM notifications")
    assert [n["title"] for n in notes] == ["Memories not saved from your chats"]


async def test_the_fast_role_still_refuses_with_nothing_else(make_app, model):
    app = await make_app()
    with pytest.raises(ModelRefused, match="only answers your chats"):
        await app.llm.complete_json("fast", [{"role": "user", "content": "hi"}])
    assert model.asked == []
