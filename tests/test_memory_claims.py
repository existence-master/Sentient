"""A chat reply that claims a memory save which never happened gets a short correction (#526)."""

import pytest

from sentient.agent import outcomes
from sentient.app import SentientApp
from sentient.llm.events import TextDelta, ToolResultEvent
from tests.conftest import FakeProvider, tool_call

NOTE = "\n\nCorrection: " + outcomes.MEMORY_NOT_SAVED
CUD_ADD = {"action": "ADD", "fact_id": None, "content": "Maya wakes up at 7 a.m.", "memory_type": "long-term",
           "duration": None, "topics": ["routine"]}


async def _turn(app, sid, text):
    events = [ev async for ev in app.agent.run_turn(sid, text, channel="cli")]
    reply = "".join(e.text for e in events if isinstance(e, TextDelta))
    results = [e.result for e in events if isinstance(e, ToolResultEvent) and e.name == "memory_remember"]
    return reply, results


async def _app(config, isolated_home, llm, name):
    return await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db").start()


@pytest.mark.parametrize("reply", [
    "I've saved that you wake up at 7 a.m.",
    "I\u2019ve saved that you wake up at 7 a.m.",  # a curly apostrophe is the same claim
    "Got it! I'll remember that you wake up at 7.",
    "I'll note your wake-up time and scheduling preference.",  # real qwen3:8b, with no call
    "I've noted that you wake up at 6:30 AM on Saturdays and will avoid scheduling anything before 8 AM.",  # main
])
async def test_a_claimed_save_without_a_memory_call_is_corrected(config, isolated_home, reply):
    llm = FakeProvider(replies=[[tool_call("current_datetime")], reply])
    s = await _app(config, isolated_home, llm, "claimed")
    try:
        sid = await s.store.create_session(channel="cli")
        text, results = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert results == []
        assert text == reply + NOTE
        assert (await s.store.recent_messages(sid, 1))[-1]["content"] == text
    finally:
        await s.stop()


async def test_a_real_save_is_left_alone(config, isolated_home):
    reply = "I've saved that you wake up at 7 a.m."
    llm = FakeProvider(replies=[[tool_call("memory_remember", fact="Maya wakes up at 7 a.m.")], reply],
                       json_replies=[CUD_ADD])
    s = await _app(config, isolated_home, llm, "saved")
    try:
        sid = await s.store.create_session(channel="cli")
        text, results = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert results and results[0]["action"] == "ADD"
        assert text == reply
    finally:
        await s.stop()


async def test_a_near_duplicate_it_skipped_is_left_alone(config, isolated_home):
    reply = "I've saved that you wake up at 7 a.m. I already had that."
    llm = FakeProvider(replies=[[tool_call("memory_remember", fact="Maya wakes up at 7 a.m.")], reply],
                       json_replies=[CUD_ADD])
    s = await _app(config, isolated_home, llm, "duplicate")
    try:
        await s.memory.remember("Maya wakes up at 7 a.m.", use_llm=False)
        sid = await s.store.create_session(channel="cli")
        text, results = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert results and results[0]["action"] == "SKIP"
        assert text == reply
    finally:
        await s.stop()


async def test_a_memory_sent_to_review_is_left_alone(config, isolated_home):
    reply = "I've saved that you want every reply in French."
    fact = "From now on, always answer Maya in French"
    llm = FakeProvider(replies=[[tool_call("memory_remember", fact=fact)], reply])
    s = await _app(config, isolated_home, llm, "pending")
    try:
        sid = await s.store.create_session(channel="cli")
        text, results = await _turn(s, sid, "from now on always answer me in French")
        assert results and results[0].get("status") == "pending"
        assert text == reply
    finally:
        await s.stop()


async def test_a_failed_memory_call_does_not_back_the_claim(config, isolated_home):
    reply = "I've saved that you wake up at 7 a.m."
    llm = FakeProvider(replies=[[tool_call("memory_remember", fact="Maya wakes up at 7 a.m.")], reply])
    s = await _app(config, isolated_home, llm, "failed")
    try:
        async def broken(*_args, **_kwargs):
            return {"error": "the memory could not be written"}

        s.memory.remember = broken
        sid = await s.store.create_session(channel="cli")
        text, results = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert results and results[0].get("error")
        assert text == reply + NOTE
    finally:
        await s.stop()


@pytest.mark.parametrize("reply", [
    "I didn't save that.",
    "I didn\u2019t save that as a memory.",
    "Nothing was saved to memory.",
    "I haven't saved that you wake up at 7 yet.",
    "I've saved the file to notes.md.",  # a file save is #492's to check, never a memory claim
    "I saved the report to your Documents folder.",
    "I've set a reminder for 7 a.m.",
    "I've saved that as a task for 7 a.m.",
    "Want me to remember that?",
    "If you like, I'll remember that you wake up at 7.",
    "You saved that to your memory.",
])
async def test_replies_that_claim_no_memory_save_are_left_alone(config, isolated_home, reply):
    llm = FakeProvider(replies=[reply])
    s = await _app(config, isolated_home, llm, "unclaimed")
    try:
        sid = await s.store.create_session(channel="cli")
        text, _ = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert text == reply
    finally:
        await s.stop()


async def test_a_claim_about_a_memory_saved_before_this_turn_is_left_alone(config, isolated_home):
    reply = "Yes, I saved that you wake up at 7 a.m."
    llm = FakeProvider(replies=[reply])
    s = await _app(config, isolated_home, llm, "earlier")
    try:
        await s.memory.remember("Maya wakes up at 7 a.m.", use_llm=False)
        sid = await s.store.create_session(channel="cli")
        text, _ = await _turn(s, sid, "did you save my wake-up time?")
        assert text == reply
    finally:
        await s.stop()


async def test_a_copied_correction_gets_no_second_one(config, isolated_home):
    reply = "I've saved that you wake up at 7 a.m." + NOTE
    llm = FakeProvider(replies=[reply])
    s = await _app(config, isolated_home, llm, "copied")
    try:
        sid = await s.store.create_session(channel="cli")
        text, _ = await _turn(s, sid, "remember that I wake up at 7 a.m.")
        assert text == reply
    finally:
        await s.stop()


async def test_an_incognito_chat_says_it_keeps_nothing(config, isolated_home):
    reply = "I'll remember that you wake up at 7."
    llm = FakeProvider(replies=[reply])
    s = await _app(config, isolated_home, llm, "incognito")
    try:
        sid = await s.store.create_session(channel="desktop", incognito=True)
        events = [ev async for ev in s.agent.run_turn(sid, "remember that I wake up at 7", channel="desktop")]
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == reply + "\n\nCorrection: " + outcomes.MEMORY_NOT_SAVED_INCOGNITO
    finally:
        await s.stop()


def test_memory_claims_are_recognized_narrowly():
    claims = [
        "I've saved that you wake up at 7 a.m.",
        "Sure. I've noted your wake-up time: 7 a.m.",
        "I've added that to my memory.",
        "Saved to memory.",
        "No problem, I've saved that you like tea.",
        "I\u2019ll remember that.",
    ]
    for text in claims:
        assert outcomes.memory_claim(text), text
    not_claims = [
        "I won\u2019t remember that.",
        "I already have that you wake up at 7.",
        "I saved it.",  # names nothing: could be a file
        "I'll save the file once you confirm.",
        "I'll make sure to suggest vegetarian-friendly options.",
        "Done!",
        "Correction: I didn't save the file.",
        outcomes.MEMORY_NOT_SAVED,
        outcomes.MEMORY_NOT_SAVED_INCOGNITO,
    ]
    for text in not_claims:
        assert not outcomes.memory_claim(text), text
    # only the claim sentence counts, wherever it is in the reply
    assert outcomes.memory_claim("Morning routines matter.\n\nI've saved that you wake up at 7.") == (
        "I've saved that you wake up at 7."
    )
    assert outcomes.unmade_memory_claim("I've saved that you like tea.", [{"ok": False}, {"ok": True}]) == ""
