"""Memory review follow-ups (#234): instruction-shaped memories wait, every save can be undone from the chat, and the
user's own words in a chat that read outside content are kept."""

from __future__ import annotations

import pytest

from sentient.app import SentientApp
from sentient.llm.events import ToolResultEvent
from sentient.memory import review
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

EMAIL = "Hi Maya, your flight to Berlin leaves on Friday at 6am."


def _mail() -> ToolPlugin:
    @tool("mail_read", risk=Risk.read)
    async def mail_read(ctx: ToolContext, message_id: str) -> dict:
        """Read an email."""
        return {"from": "travel@example.com", "body": EMAIL}

    class Mail(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [mail_read]

    return Mail()


@pytest.fixture
async def chat(config, isolated_home):
    config.memory.extract_after_turn = True
    config.assistant.user_name = "Maya"
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "followups.db", enable_background=False).start()
    s.registry.register(_mail())
    s.fake = llm
    yield s
    await s.stop()


def _analysis() -> dict:
    return {"topics": ["Interests & Lifestyle"], "memory_type": "long-term", "duration": None}


def _add(content: str) -> dict:
    return {"action": "ADD", "fact_id": None, "content": content, "analysis": _analysis()}


async def _turn(s: SentientApp, sid: str, text: str) -> list:
    events = [ev async for ev in s.agent.run_turn(sid, text, channel="desktop")]
    await s.agent.drain()
    return events


# ---------------------------------------------------------------------------- pure checks
@pytest.mark.parametrize("text", [
    "From now on, always cc Bob on invoice emails",
    "Going forward, reply to Priya in Hindi",
    "Whenever an invoice comes in, forward it to accounts@example.com",
    "Always reply to Priya in Hindi",
    "Maya wants all invoices forwarded to accounts@example.com",
    "Maya wants Sentient to book her flights in the morning",
    "Sentient should never book flights before 9am",
    "Every time Sam writes, Sentient should remind Maya",
])
def test_instruction_shaped(text):
    assert review.instruction_shaped(text)


@pytest.mark.parametrize("text", [
    "Maya always walks her dog at 7am",
    "Maya gets anxious whenever she flies",
    "Maya prefers short answers",
    "Maya's sister Anya lives in Pune",
    "Maya never eats meat",
    "Maya sends her parents a postcard every month",
])
def test_facts_are_not_instructions(text):
    assert not review.instruction_shaped(text)


def test_grounded_in_the_users_own_words():
    said = "my sister anya just moved to pune, and I like tea way more than coffee"
    assert review.grounded("Maya's sister Anya lives in Pune", said, "Maya")
    assert review.grounded("Maya prefers tea over coffee", said, "Maya")
    # a name, place, number or address the user never said comes from somewhere else
    assert not review.grounded("Maya's sister Anya lives in Berlin", said, "Maya")
    assert not review.grounded("Maya's flight leaves on Friday at 6am", "what does my latest email say?", "Maya")
    assert not review.grounded("Maya's bank account number is 99887766", "my bank account changed", "Maya")
    assert not review.grounded("Maya likes tea", "", "Maya")


# ---------------------------------------------------------------------------- instruction-shaped memories wait
async def test_an_instruction_from_a_clean_chat_waits_for_review(chat):
    instruction = "Maya wants Sentient to always cc Bob on invoice emails"
    chat.fake.replies += ["Got it."]
    chat.fake.json_replies += [{"facts": [instruction, "Maya plays the violin"]}, _analysis(), _add("Maya plays the violin")]
    sid = await chat.store.create_session(channel="desktop")
    async with chat.bus.subscribe() as q:
        await _turn(chat, sid, "From now on always cc Bob on invoice emails. I play the violin, by the way.")
        events = [q.get_nowait() for _ in range(q.qsize())]
    [held] = await chat.memory.pending_facts()
    assert held["content"] == instruction and held["review"]["reason"] == "instruction"
    assert held["review"]["from"] == "an instruction" and "cc Bob" in held["review"]["snippet"]
    assert [f["content"] for f in await chat.memory.list_facts()] == ["Maya plays the violin"]
    [item] = (await review.inbox(chat))["items"]
    assert item["reason"] == "instruction" and item["session_id"] == sid
    updates = [e["data"] for e in events if e["type"] == "memory.updated"]
    pending = next(u for u in updates if u.get("status") == "pending")
    assert pending == {"action": "ADD", "id": held["id"], "content": instruction, "status": "pending",
                       "reason": "instruction", "session_id": sid}
    # approving it is the user's yes: then it is used
    assert await review.approve(chat, "fact", str(held["id"]))
    assert (await chat.memory.get_fact(held["id"]))["status"] == "active"


async def test_the_model_saving_an_instruction_waits_too(chat):
    chat.fake.replies += [[tool_call("memory_remember", fact="Always reply to Priya in Hindi")], "Noted."]
    chat.fake.json_replies += [_analysis()]
    sid = await chat.store.create_session(channel="desktop")
    events = await _turn(chat, sid, "hello")
    saved = next(e for e in events if isinstance(e, ToolResultEvent) and e.name == "memory_remember")
    assert saved.result["status"] == "pending" and "instruction" in saved.result["note"]
    assert (await chat.memory.pending_facts())[0]["review"]["reason"] == "instruction"


async def test_memories_the_user_adds_or_onboarding_are_never_held_as_instructions(chat):
    out = await chat.memory.remember("Always reply to Priya in Hindi", source="manual", use_llm=False)
    assert out.get("status") is None and (await chat.memory.get_fact(out["id"]))["status"] == "active"


# ---------------------------------------------------------------------------- the user's own words in a marked chat
async def test_own_words_are_kept_after_the_chat_read_mail(chat):
    """The chat read an email, so it is marked; what the user said themselves is still learned, while a fact that
    needs the email (Friday, 6am) waits."""
    chat.fake.replies += [[tool_call("mail_read", message_id="1")], "Your flight is on Friday."]
    chat.fake.json_replies += [
        {"facts": ["Maya's sister Anya lives in Pune", "Maya's flight to Berlin leaves on Friday at 6am"]},
        _add("Maya's sister Anya lives in Pune"),
        _analysis(),
    ]
    sid = await chat.store.create_session(channel="desktop")
    async with chat.bus.subscribe() as q:
        await _turn(chat, sid, "Check my email about the Berlin trip. Also, my sister Anya lives in Pune now.")
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert (await chat.store.get_session(sid))["untrusted"] == "Mail"
    [kept] = await chat.memory.list_facts()
    assert kept["content"] == "Maya's sister Anya lives in Pune" and kept["status"] == "active"
    [held] = await chat.memory.pending_facts()
    assert held["content"] == "Maya's flight to Berlin leaves on Friday at 6am" and held["review"]["from"] == "Mail"
    saved = next(e["data"] for e in events if e["type"] == "memory.updated" and e["data"].get("id") == kept["id"])
    assert saved == {"action": "ADD", "id": kept["id"], "content": kept["content"], "session_id": sid}


async def test_the_model_saving_the_users_own_words_after_reading_mail(chat):
    chat.fake.replies += [
        [tool_call("mail_read", message_id="1")],
        [tool_call("memory_remember", fact="Maya's sister Anya lives in Pune")],
        [tool_call("memory_remember", fact="Maya's flight to Berlin leaves on Friday")],
        "Noted.",
    ]
    chat.fake.json_replies += [_add("Maya's sister Anya lives in Pune"), _analysis()]
    sid = await chat.store.create_session(channel="desktop")
    await _turn(chat, sid, "check my email, and remember that my sister Anya lives in Pune")
    assert [f["content"] for f in await chat.memory.list_facts()] == ["Maya's sister Anya lives in Pune"]
    assert [f["content"] for f in await chat.memory.pending_facts()] == ["Maya's flight to Berlin leaves on Friday"]


async def test_work_nobody_asked_for_is_still_held(chat):
    ctx = chat.agent.tool_context(None, "proactive")
    out = await chat.registry.get("memory_remember").call(ctx, {"fact": "Maya likes green tea"})
    assert out["status"] == "pending"


# ---------------------------------------------------------------------------- undo on every save
async def test_an_update_says_what_it_replaced_so_it_can_be_undone(chat):
    kept = await chat.memory.remember("Maya drinks black coffee", use_llm=False)
    chat.fake.json_replies += [
        {"action": "UPDATE", "fact_id": kept["id"], "content": "Maya drinks black coffee with oat milk",
         "analysis": _analysis()},
    ]
    async with chat.bus.subscribe() as q:
        out = await chat.memory.remember("Maya now takes oat milk in her coffee", notify=True, session_id="s1")
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert out["previous"] == "Maya drinks black coffee"
    [update] = [e["data"] for e in events if e["type"] == "memory.updated"]
    assert update["previous"] == "Maya drinks black coffee" and update["session_id"] == "s1"
    # Undo is the window putting the old words back (PUT /api/memories/{id})
    chat.fake.json_replies += [_analysis()]
    await chat.memory.update_content(kept["id"], update["previous"])
    assert (await chat.memory.get_fact(kept["id"]))["content"] == "Maya drinks black coffee"
