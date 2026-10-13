"""The cost meter and the monthly budget (#148).

Every call the provider makes is counted with its price, plan models are kept apart from dollars, a notification goes
out at 80% and at 100%, and once the budget is used up a paid cloud model is refused before anything is sent (ask:
until the user chooses Keep going; stop: until next month or a bigger budget).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import litellm
import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm import claude_code
from sentient.llm.locality import usage_source
from sentient.llm.provider import BudgetReached, LiteLLMProvider, StreamChunk
from sentient.llm.spending import BudgetError, period_starts

PRICED = "openai/gpt-priced"
PRICE = 0.0042
USAGE = SimpleNamespace(prompt_tokens=1000, completion_tokens=200)


@pytest.fixture
def sent(monkeypatch):
    """litellm calls; streams and JSON jobs report usage, and only PRICED has a price."""
    calls: list[str] = []

    async def acompletion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs.get("stream"):
            async def chunks():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", reasoning_content=None))])

            return chunks()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))], usage=USAGE)

    def stream_chunk_builder(chunks, messages=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))], usage=USAGE)

    def completion_cost(completion_response=None, model=None):
        if model != PRICED:
            raise ValueError("not mapped")
        return PRICE

    async def plan_stream(config, model, role, messages, tools):
        calls.append(model)
        yield StreamChunk(text="from the plan", model=model)
        yield StreamChunk(done=True, usage={"prompt_tokens": 500, "completion_tokens": 50}, model=model)

    async def no_limit(*a, **k):
        return None

    async def aembedding(**kwargs):  # memory recall in a chat turn: never a real model
        return SimpleNamespace(data=[{"embedding": [0.1] * 8} for _ in kwargs["input"]], usage=None)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "aembedding", aembedding)
    monkeypatch.setattr(litellm, "stream_chunk_builder", stream_chunk_builder)
    monkeypatch.setattr(litellm, "completion_cost", completion_cost)
    monkeypatch.setattr(claude_code, "stream", plan_stream)
    monkeypatch.setattr(LiteLLMProvider, "_model_max_context", no_limit)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: "sk-test")
    return calls


@pytest.fixture
async def app(config, isolated_home, sent):
    config.models.roles.primary = PRICED
    s = await SentientApp(config, db_path=isolated_home / "cost.db", enable_background=False).start()
    try:
        yield s
    finally:
        await s.stop()


async def chat(app, model=None, source="chat"):
    with usage_source(source):
        return [c async for c in app.llm.stream("primary", [{"role": "user", "content": "hi"}], model=model)]


async def test_every_call_is_counted_with_its_price_and_plans_stay_apart(app, sent):
    await chat(app)
    await chat(app, model="claude-code/sonnet")
    await chat(app, model="ollama_chat/qwen3:8b")
    await app.llm.complete_json("planner", [{"role": "user", "content": "plan"}])  # background work counts too
    rows = await app.store.fetchall("SELECT model, role, source, cost_usd FROM usage ORDER BY id")
    assert [(r["model"], r["source"], r["cost_usd"]) for r in rows] == [
        (PRICED, "chat", PRICE), ("claude-code/sonnet", "chat", None), ("ollama_chat/qwen3:8b", "chat", None),
        (PRICED, "background", PRICE),
    ]
    report = await app.spending.report()
    assert [p["key"] for p in report["periods"]] == ["today", "week", "month"]
    today = report["periods"][0]
    assert today["cost_usd"] == pytest.approx(2 * PRICE)
    assert today["plan_calls"] == 1 and today["plan_tokens"] == 550 and today["local_calls"] == 1
    by_model = {m["model"]: m for m in today["by_model"]}
    assert by_model[PRICED]["billing"] == "paid" and by_model[PRICED]["cost_usd"] == pytest.approx(2 * PRICE)
    assert by_model["claude-code/sonnet"]["billing"] == "plan" and by_model["claude-code/sonnet"]["cost_usd"] is None
    by_role = {r["role"]: r for r in today["by_role"]}
    assert by_role["planner"]["cost_usd"] == pytest.approx(PRICE) and by_role["primary"]["plan_calls"] == 1
    assert report["budget"]["state"] == "off" and report["local_only"] is False


async def test_the_agent_loop_does_not_count_a_call_twice(app):
    sid = await app.store.create_session(channel="desktop")
    [ev async for ev in app.agent.run_turn(sid, "hi", channel="desktop")]
    rows = await app.store.fetchall("SELECT source, cost_usd FROM usage")
    assert [(r["source"], r["cost_usd"]) for r in rows] == [("chat", PRICE)]


async def test_budget_warns_at_80_then_asks_at_100(app, sent):
    app.config.models.monthly_budget_usd = 0.01  # 2 calls = 84%, 3 calls = 126%
    await chat(app)
    await chat(app)
    notes = await app.store.fetchall("SELECT kind, body FROM notifications ORDER BY created_at")
    assert [n["kind"] for n in notes] == ["budget"]
    assert notes[0]["body"].startswith("You've used 84% of this month's $0.01 budget")
    await chat(app)
    notes = await app.store.fetchall("SELECT body FROM notifications ORDER BY created_at")
    assert len(notes) == 2 and "budget is used up" in notes[1]["body"] and "Choose Keep going" in notes[1]["body"]
    state = await app.spending.budget()
    assert state["state"] == "reached" and state["percent"] == 126

    calls_before = len(sent)
    with pytest.raises(BudgetReached) as refused:
        await chat(app)
    assert refused.value.code == "budget_reached" and "nothing was sent to gpt-priced" in str(refused.value)
    assert len(sent) == calls_before  # refused before anything was sent
    await chat(app, model="claude-code/sonnet")  # plan and local models still answer
    await chat(app, model="ollama_chat/qwen3:8b")

    state = await app.spending.keep_going()
    assert state["kept_going"] is True
    await chat(app)  # allowed again for the rest of the month
    assert len(await app.store.fetchall("SELECT id FROM notifications")) == 2  # no repeat notifications


async def test_stop_mode_refuses_keep_going(app):
    app.config.models.monthly_budget_usd = 0.004
    app.config.models.budget_at_limit = "stop"
    await chat(app)
    with pytest.raises(BudgetReached) as refused:
        await chat(app)
    assert "Raise the budget in Settings > Usage" in str(refused.value)
    with pytest.raises(BudgetError):
        await app.spending.keep_going()
    app.config.models.monthly_budget_usd = 1.0  # a bigger budget lets it through
    await chat(app)


async def test_a_budgeted_cloud_role_falls_back_to_a_local_model(app, sent):
    app.config.models.monthly_budget_usd = 0.004
    app.config.models.fallbacks = {"primary": ["ollama_chat/qwen3:8b"]}
    await chat(app)
    out = await chat(app)
    assert out[-1].model == "ollama_chat/qwen3:8b" and sent[-1] == "ollama_chat/qwen3:8b"


def test_periods_follow_the_local_clock():
    now = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)  # a Wednesday
    starts = period_starts(now)
    local = {k: v.astimezone() for k, v in starts.items()}
    assert local["today"].hour == 0 and local["week"].weekday() == 0 and local["month"].day == 1
    assert starts["month"] <= starts["week"] <= starts["today"] <= now


def test_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "t")
    s = SentientApp(config, db_path=isolated_home / "routes.db", enable_background=False)
    with TestClient(create_app(s)) as client:
        h = {"Authorization": "Bearer t"}
        report = client.get("/api/usage/cost", headers=h).json()
        assert report["budget"]["state"] == "off" and len(report["periods"]) == 3
        r = client.post("/api/usage/budget/keep-going", headers=h)
        assert r.status_code == 409 and "no monthly budget" in r.json()["detail"]
        assert client.patch("/api/config", headers=h, json={"models": {"monthly_budget_usd": 5}}).status_code == 200
        assert client.post("/api/usage/budget/keep-going", headers=h).json()["kept_going"] is True
        assert "cost_usd" in client.get("/api/usage", headers=h).json()["totals"]
