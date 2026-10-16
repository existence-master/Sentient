"""/scope in a messaging app (#392): pick which memories a chat uses, answered without a model."""

from __future__ import annotations

SISTER = "Maya's sister Meera lives in Lisbon"
KESTREL = "Maya leads Project Kestrel, a payments launch"


async def _scope_of(app, chat_id: str = "42") -> str | None:
    chat = await app.channels.store.chat("telegram", chat_id)
    return (await app.store.get_session(chat["session_id"]))["memory_scope"]


async def test_scope_shows_and_changes_what_the_chat_recalls(tg):
    app = tg.app
    app.config.memory.min_similarity = 0.0
    await app.memory.remember(SISTER, use_llm=False, scope="personal")
    await app.memory.remember(KESTREL, use_llm=False, scope="work")
    await tg.pair(42)
    calls = len(app.llm.calls)

    await tg.say(42, "/scope")
    assert tg.api.screen()[-1].startswith("This chat uses your Personal memories, plus the ones kept for every chat.")
    await tg.say(42, "/scope work")
    reply = tg.api.screen()[-1]
    assert reply.startswith("Done. This chat now uses your Work memories") and "no longer brings up your Personal" in reply
    assert await _scope_of(app) == "work"
    await tg.say(42, "/scope Work")
    assert tg.api.screen()[-1] == "This chat already uses your Work memories."
    assert len(app.llm.calls) == calls  # no model call for any of it

    app.llm.replies = ["You lead Kestrel."]
    await tg.say(42, "what do you know about me?")
    system = app.llm.calls[-1]["messages"][0]["content"]
    assert KESTREL in system and SISTER not in system

    await tg.say(42, "/scope personal")
    assert await _scope_of(app) == "personal"
    app.llm.replies = ["Your sister lives in Lisbon."]
    await tg.say(42, "what do you know about me?")
    system = app.llm.calls[-1]["messages"][0]["content"]
    assert SISTER in system and KESTREL not in system


async def test_projects_new_chats_default_and_always(tg):
    app = tg.app
    await app.memory.remember("Project Atlas ships the new onboarding flow", use_llm=False, scope="Atlas launch")
    await tg.pair(42)
    await tg.say(42, "/scope")
    assert "Your projects: Atlas launch." in tg.api.screen()[-1]
    await tg.say(42, "/scope atlas   LAUNCH")  # an existing project keeps its spelling
    assert await _scope_of(app) == "Atlas launch"
    await tg.say(42, "/scope Garden")
    assert "Garden is a new project, so it has no memories yet." in tg.api.screen()[-1]
    await tg.say(42, "/scope always")
    assert "can't use only the memories kept for every chat" in tg.api.screen()[-1]
    assert await _scope_of(app) == "Garden"

    await tg.say(42, "/new")  # a fresh chat keeps the scope
    assert await _scope_of(app) == "Garden"

    app.config.channels.telegram.memory_scope = "work"
    await tg.say(42, "/scope default")  # follows the app's setting again
    assert await _scope_of(app) is None
    assert tg.api.screen()[-1].startswith("Done. This chat now uses your Work memories")
    await tg.say(42, "/help")
    assert "/scope" in tg.api.screen()[-1]


async def test_unpaired_chats_cannot_change_a_scope(tg):
    await tg.pair(42)
    await tg.say(7, "/scope work")
    assert "isn't paired" in tg.api.screen()[-1]
    assert await _scope_of(tg.app) is None
