"""/learn in a messaging app (#167): the chat's owner turns a page or a file into a skill that waits for review."""

from __future__ import annotations

from pathlib import Path

from tests.evolution.test_learn_skill import DRAFT, PACKING


async def test_learn_a_file_path_and_a_sent_file(tg, llm, tmp_path: Path):
    await tg.pair(42)
    await tg.say(42, "/learn")
    assert tg.api.screen()[-1].startswith("Send /learn followed by a link")

    f = tmp_path / "packing.md"
    f.write_text(PACKING, encoding="utf-8")
    llm.json_replies = [DRAFT]
    await tg.say(42, f"/learn {f}")
    screen = tg.api.screen()
    assert screen[-2].startswith("Reading it now.")
    assert "drafted the skill" in screen[-1] and "work-trip-packing" in screen[-1]
    assert "waits for your review" in screen[-1]
    assert tg.app.skills.get_pending("work-trip-packing") is not None
    log = await tg.app.store.fetchall("SELECT detail FROM evolution_log WHERE kind = 'skill_learned'")
    assert '"via": "telegram"' in log[0]["detail"]

    # a file sent with /learn as its caption
    tg.api.files["d1"] = PACKING.encode()
    llm.json_replies = [{**DRAFT, "name": "carry-on-checklist"}]
    doc = {"document": {"file_id": "d1", "file_unique_id": "doc", "file_name": "trip.md", "mime_type": "text/markdown",
                        "file_size": len(PACKING)}, "caption": "/learn"}
    await tg.say(42, None, **doc)
    assert "carry-on-checklist" in tg.api.screen()[-1]
    assert tg.app.skills.get_pending("carry-on-checklist") is not None

    await tg.say(42, f"/learn {tmp_path / 'missing.md'}")
    assert tg.api.screen()[-1].startswith("I couldn't learn from that. There is no file or folder")
    await tg.say(42, "/help")
    assert "/learn" in tg.api.screen()[-1]


async def test_in_a_group_only_the_owner_can_learn(tg, llm, tmp_path: Path):
    from tests.channels.test_group_chats import OWNER, RAVI, _group, say

    await _group(tg)
    f = tmp_path / "packing.md"
    f.write_text(PACKING, encoding="utf-8")
    shown = len(tg.api.screen())
    await say(tg, RAVI, f"/learn {f}")
    assert len(tg.api.screen()) == shown and tg.app.skills.list_pending() == []
    llm.json_replies = [DRAFT]
    await say(tg, OWNER, f"/learn {f}")
    assert "work-trip-packing" in tg.api.screen()[-1]


async def test_stop_everything_stops_a_learn_draft(tg, llm, tmp_path: Path):
    import asyncio

    f = tmp_path / "packing.md"
    f.write_text(PACKING, encoding="utf-8")
    await tg.pair(42)
    gate = asyncio.Event()
    started = asyncio.Event()

    async def slow_json(role, messages, *, model=None):
        started.set()
        await gate.wait()
        return DRAFT

    llm.complete_json = slow_json
    tg.ch.dispatch(__import__("tests.channels.conftest", fromlist=["update"]).update(42, f"/learn {f}"))
    await asyncio.wait_for(started.wait(), 10)
    assert await tg.ch.stop_all_turns() == 1
    await tg.ch.wait_idle()
    assert tg.api.screen()[-1] == "Stopped. I didn't finish drafting the skill, so nothing was saved."
    assert tg.app.skills.list_pending() == []
