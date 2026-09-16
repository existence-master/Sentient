from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentient.memory.usermodel import normalize_dimension, parse_confidence, parse_operations
from sentient.tools.base import Risk
from tests.memory.helpers import add_fact, drain


async def _say(app, *texts: str) -> str:
    sid = await app.store.create_session()
    for t in texts:
        await app.store.add_message(sid, "user", t)
    return sid


async def _seed(app) -> dict[str, dict]:
    """Two inferred insights via a refresh. Returns them keyed by a short name."""
    await _say(app, "keep it short please, no essays", "I'm still coding at 2am again")
    app.fake.json_replies.append(
        {
            "operations": [
                {"op": "add", "dimension": "communication", "statement": "Sarthak prefers short, direct answers.",
                 "confidence": 0.6, "evidence": ["m1"]},
                {"op": "add", "dimension": "routines", "statement": "The user works late into the night.",
                 "confidence": "high", "evidence": "m2, f999"},
            ],
            "summary": "Sarthak is a night owl who likes brevity.",
        }
    )
    counts = await app.user_model.refresh()
    assert counts == {"added": 2, "updated": 0, "disputed": 0, "questions": 0}
    state = await app.user_model.get_state()
    by = {i["dimension"]: i for i in state["insights"]}
    return {"short": by["communication"], "late": by["routines"]}


def test_tolerant_parsers():
    assert normalize_dimension("Work Style") == "work_style"
    assert normalize_dimension("habits") == "routines"
    assert normalize_dimension("???") == "context"
    assert parse_confidence("high") == 0.7 and parse_confidence(70) == 0.7 and parse_confidence(None) == 0.5
    ops = parse_operations(
        [{"action": "create", "statement": "x"}, "garbage", {"op": "explode"}, {"type": "Reinforce", "id": "i1"}]
    )
    assert [o["op"] for o in ops] == ["add", "support"]
    assert parse_operations({"op": "retire", "id": "i2"})[0]["op"] == "retire"
    assert parse_operations("not json") == [] and parse_operations({"operations": "nope"}) == []


async def test_refresh_adds_insights_with_evidence_and_summary(app):
    async with app.bus.subscribe() as q:
        ins = await _seed(app)
        events = [e for e in drain(q) if e["type"] == "user_model.updated"]
    short, late = ins["short"], ins["late"]
    assert short["status"] == "active" and short["source"] == "inferred" and short["confidence"] == 0.6
    assert short["evidence"][0]["kind"] == "message" and "short" in short["evidence"][0]["quote"]
    assert late["statement"] == "Sarthak works late into the night."  # "The user" personalised
    assert late["confidence"] == 0.7 and len(late["evidence"]) == 1  # unknown ref f999 ignored
    state = await app.user_model.get_state()
    assert state["summary"] == "Sarthak is a night owl who likes brevity." and state["updated_at"]
    assert events and events[-1]["data"]["summary_changed"] is True and events[-1]["data"]["insights"] == 2
    system = app.fake.calls[-1]["messages"][0]["content"]
    assert "work_style" in system and "Sarthak" in system

    # nothing new since the last refresh: no model call
    calls = len(app.fake.calls)
    assert await app.user_model.refresh() == {"added": 0, "updated": 0, "disputed": 0, "questions": 0}
    assert len(app.fake.calls) == calls


async def test_support_contradict_dispute_retire(app):
    um = app.user_model
    ins = await _seed(app)
    short, late = ins["short"], ins["late"]
    await _say(app, "ok that answer was way too long", "give me more detail this time")
    app.fake.json_replies.append(
        {"operations": [
            {"op": "support", "id": short["id"], "evidence": ["m1"]},
            {"op": "add", "dimension": "communication", "statement": "Sarthak prefers short direct answers",
             "evidence": ["m1"]},  # duplicate of an existing insight -> treated as support
        ]}
    )
    counts = await um.refresh()
    assert counts["added"] == 0 and counts["updated"] == 1
    assert (await um.get_insight(short["id"]))["confidence"] == 0.8

    await _say(app, "actually I like detailed explanations now")
    app.fake.json_replies.append({"operations": [{"op": "contradict", "id": short["id"], "evidence": ["m1"]}]})
    assert (await um.refresh())["updated"] == 1
    assert (await um.get_insight(short["id"]))["status"] == "active"

    await _say(app, "please explain in depth")
    app.fake.json_replies.append(
        {"operations": [
            {"op": "contradict", "id": short["id"], "evidence": ["m1"], "question": "Do you want longer answers now?"},
            {"op": "contradict", "id": short["id"], "evidence": ["m1"]},
        ]}
    )
    counts = await um.refresh()
    assert counts["disputed"] == 1 and counts["questions"] == 1
    got = await um.get_insight(short["id"])
    assert got["status"] == "disputed" and got["confidence"] < app.config.user_model.dispute_below
    questions = (await um.get_state())["questions"]
    assert len(questions) == 1 and questions[0]["question"] == "Do you want longer answers now?"
    assert questions[0]["insight_id"] == short["id"]

    await _say(app, "I sleep by ten these days")
    app.fake.json_replies.append({"operations": [{"op": "retire", "id": "i1", "reason": "sleeps early now"}]})
    # i1 is the highest-confidence active insight: "works late" (0.7)
    assert (await um.refresh())["updated"] == 1
    assert (await um.get_insight(late["id"]))["status"] == "retired"
    # a retired insight is not re-added by a later refresh
    await _say(app, "coding at night again")
    app.fake.json_replies.append(
        {"operations": [{"op": "add", "dimension": "routines", "statement": "Sarthak works late into the night.",
                         "evidence": ["m1"]}]}
    )
    assert (await um.refresh())["added"] == 0


async def test_confirmed_and_user_insights_are_only_questioned(app):
    um = app.user_model
    ins = await _seed(app)
    mine = await um.add_insight("I prefer tea over coffee", "preferences")
    assert mine["status"] == "confirmed" and mine["source"] == "user" and mine["confidence"] == 1.0
    confirmed = await um.update_insight(ins["short"]["id"], status="confirmed")
    await _say(app, "grabbing a coffee", "long answers are fine")
    app.fake.json_replies.append(
        {"operations": [
            {"op": "contradict", "id": mine["id"], "evidence": ["m1"], "question": "Have you switched to coffee?"},
            {"op": "retire", "id": mine["id"]},
            {"op": "support", "id": confirmed["id"], "evidence": ["m2"]},
            {"op": "retire", "id": confirmed["id"]},
        ]}
    )
    counts = await um.refresh()
    assert counts["questions"] == 2 and counts["updated"] == 0 and counts["disputed"] == 0
    for before in (mine, confirmed):
        after = await um.get_insight(before["id"])
        assert (after["status"], after["confidence"], after["statement"]) == (
            before["status"], before["confidence"], before["statement"]
        )
    assert {q["insight_id"] for q in (await um.get_state())["questions"]} == {mine["id"], confirmed["id"]}


async def test_answering_questions_updates_insight_and_stores_fact(app):
    um, llm = app.user_model, app.fake
    ins = await _seed(app)
    for key in ("short", "late"):
        i = await um._get(ins[key]["id"])
        await um._ask(i)
    q_short, q_late = sorted((await um.get_state())["questions"], key=lambda q: q["insight_id"] != ins["short"]["id"])

    llm.json_replies.append({"verdict": "rewrite", "statement": "Sarthak wants detail on technical topics.",
                             "fact": "Sarthak wants detailed answers on technical topics."})
    out = await um.answer_question(q_short["id"], "Only for casual stuff; go deep on technical things")
    assert out["verdict"] == "rewrite"
    got = await um.get_insight(ins["short"]["id"])
    assert got["statement"] == "Sarthak wants detail on technical topics."
    assert (got["status"], got["source"]) == ("confirmed", "user")
    assert got["evidence"][-1]["kind"] == "feedback"
    facts = await app.memory.list_facts(source="user_model")
    assert [f["content"] for f in facts] == ["Sarthak wants detailed answers on technical topics."]

    class Broken(Exception):
        pass

    async def boom(*a, **k):
        raise Broken("model down")

    llm.complete_json = boom  # the answer is still understood from a plain "no"
    out = await um.answer_question(q_late["id"], "No, not anymore")
    assert out["verdict"] == "retire"
    assert (await um.get_insight(ins["late"]["id"]))["status"] == "retired"
    assert (await um.get_state())["questions"] == []
    try:
        await um.answer_question(q_late["id"], "again")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("answered question must not be answerable twice")


async def test_malformed_refresh_replies_are_tolerated(app):
    um, llm = app.user_model, app.fake
    await _say(app, "I always batch my email twice a day")
    llm.json_replies.append(
        [{"action": "create", "dimension": "Work Style", "insight": "Sarthak batches email twice a day.",
          "confidence": "95%", "evidence": [{"ref": "M1"}]},
         "garbage", {"op": "explode"}, {"op": "support", "id": "nope"}, {"op": "add", "statement": "hi"}]
    )
    counts = await um.refresh()
    assert counts["added"] == 1
    [ins] = (await um.get_state())["insights"]
    assert ins["dimension"] == "work_style" and ins["confidence"] == 0.8 and ins["evidence"][0]["kind"] == "message"

    await _say(app, "another message")
    llm.json_replies.append("the model rambled instead of JSON")
    assert await um.refresh() == {"added": 0, "updated": 0, "disputed": 0, "questions": 0}

    async def boom(*a, **k):
        raise ValueError("Model did not return JSON")

    llm.complete_json = boom
    await _say(app, "and another")
    assert await um.refresh() == {"added": 0, "updated": 0, "disputed": 0, "questions": 0}


async def test_context_for_budget_and_relevance(app):
    um, cfg = app.user_model, app.config.user_model
    vocab = ["hiking", "mountains", "weekends", "tea", "coffee", "vegetarian", "plans"]

    async def embed(texts, *, model=None):  # hashed toy vectors collide; use one dimension per word
        out = []
        for t in texts:
            words = {w.strip(".,!?'\"").lower() for w in t.split()}
            vec = [1.0 if v in words else 0.0 for v in vocab] + [0.1]
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / norm for x in vec])
        return out

    app.fake.embed = embed
    assert await um.context_for("anything") == ""
    await um.add_insight("Sarthak is vegetarian", "preferences")
    for statement in ("Sarthak loves hiking mountains on weekends", "Sarthak prefers green tea over coffee"):
        await um._insert(statement, "preferences", confidence=0.5, status="active", source="inferred", evidence=[])
    cfg.context_min_similarity = 0.5
    calls = len(app.fake.calls) + len(app.fake.text_calls)
    block = await um.context_for("any hiking mountains weekends plans?")
    assert len(app.fake.calls) + len(app.fake.text_calls) == calls  # no model call
    assert block.startswith("## What I have learned about Sarthak")
    assert "vegetarian" in block and "hiking" in block and "green tea" not in block
    assert "(likely)" in block
    assert len(block) <= cfg.context_max_chars

    cfg.context_max_chars = 80
    small = await um.context_for("hiking mountains weekends")
    assert len(small) <= 80 and "vegetarian" in small and "hiking" not in small

    cfg.enabled = False
    assert await um.context_for("hiking") == ""


async def test_turn_trigger_respects_count_and_daily_limit(app):
    um, cfg = app.user_model, app.config.user_model
    cfg.refresh_after_turns, cfg.min_refresh_hours = 2, 24
    base = datetime.now(UTC) - timedelta(days=3)
    await _say(app, "I like to plan my week on Sunday evenings")
    reply = {"operations": [], "summary": "Sarthak plans ahead."}
    app.fake.json_replies.append(reply)
    assert await um.on_turn_completed({"session_id": "s"}, now=base) is None
    assert await um.on_turn_completed({"session_id": "s"}, now=base) is not None
    assert (await um.get_state())["summary"] == "Sarthak plans ahead."
    calls = len(app.fake.calls)
    for _ in range(3):
        assert await um.on_turn_completed({}, now=base + timedelta(hours=1)) is None
    assert len(app.fake.calls) == calls  # daily limit
    app.fake.json_replies.append(reply)
    assert await um.on_turn_completed({}, now=base + timedelta(hours=25)) is not None
    assert len(app.fake.calls) == calls + 1
    cfg.enabled = False
    assert await um.on_turn_completed({}, now=base + timedelta(days=2)) is None


async def test_user_model_ask_tool(app):
    await app.user_model.add_insight("Sarthak prefers window seats on flights", "preferences")
    await add_fact(app.memory, "Sarthak is flying to Berlin in October")
    app.fake.text_replies.append("A window seat, based on his stated preference.")
    ctx = app.agent.tool_context(None, "desktop")
    out = await app.registry.get("user_model_ask").call(ctx, {"question": "Which seat should I book on the flight?"})
    assert out["answer"].startswith("A window seat")
    assert "Sarthak prefers window seats on flights" in out["insights"]
    prompt = app.fake.text_calls[-1]["messages"][1]["content"]
    assert "window seats" in prompt and "Which seat" in prompt
    assert app.registry.get("user_model_ask").risk == Risk.read
