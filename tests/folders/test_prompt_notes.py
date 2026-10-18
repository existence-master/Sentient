"""Matching pieces of the user's notes go with a chat message (#354), so small models answer from notes without
searching them, and the notes the reply drew on show as sources. Offline: FakeProvider makes the embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentient.agent.prompt import notes_block
from sentient.agent.sources import AnswerSources, note_used
from sentient.app import SentientApp
from sentient.folders.service import content_words, excerpt
from sentient.llm.events import Done
from tests.conftest import FakeProvider

INSURANCE = "Health insurance: Star Health, policy SH-4471-09. Renews on 1 March. Covers Maya and Priya.\n"
LEASE = "Flat lease for 304, Lavender Court. Rent 38,000 rupees a month. Lease ends 31 May 2027.\n"
CLIP = "---\nsource: https://example.org/insurance\n---\n# Insurance tips\n\nAlways renew insurance early.\n"


@pytest.fixture
def docs(tmp_path) -> Path:
    d = tmp_path / "Maya Docs"
    (d / "Clippings").mkdir(parents=True)
    (d / "insurance.txt").write_text(INSURANCE, encoding="utf-8")
    (d / "lease.txt").write_text(LEASE, encoding="utf-8")
    (d / "Clippings" / "Insurance tips.md").write_text(CLIP, encoding="utf-8")
    return d


TOPICS = ("insur", "renew", "polic", "health", "lease", "rent", "flat", "sea", "haiku", "wave")


async def topic_embed(texts, *, model=None):
    """Embeddings by topic words, plus a little of everything, so similar meaning scores high and the rest low."""
    out = []
    for t in texts:
        low = t.lower()
        vec = [float(low.count(w)) for w in TOPICS] + [0.3]
        norm = sum(v * v for v in vec) ** 0.5
        out.append([v / norm for v in vec])
    return out


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    llm.embed = topic_embed
    a = SentientApp(config, llm=llm, db_path=isolated_home / "notes.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()


def test_excerpt_and_words():
    assert content_words("When does my insurance renew?") == ["insurance", "renew"]
    text = "Intro sentence about nothing. " * 20 + "The policy renews on 1 March every year. " + "Filler words here. " * 20
    cut = excerpt(text, ["renew"], 120)
    assert "renews on 1 March" in cut and cut.startswith("…") and cut.endswith("…") and len(cut) <= 122
    assert excerpt("short", ["x"], 120) == "short"


def test_note_used_is_decided_from_the_reply():
    note = {"title": "insurance", "file": "/d/insurance.txt", "text": INSURANCE}
    assert note_used(note, "Your Star Health policy renews on 1 March.", "when does my insurance renew?")
    assert note_used(note, "It's policy SH-4471-09.", "what is my policy?")  # a number from the note
    assert not note_used(note, "I don't know when that renews.", "when does my insurance renew?")
    assert not note_used(note, "", "when does my insurance renew?")
    lease = {"title": "Lease", "file": "/d/lease.txt", "text": LEASE}
    assert note_used(lease, "Your lease runs out in May.", "q")  # it names the note
    assert not note_used(lease, "I can't tell when your lease ends.", "when does my lease end?")  # only echoes it


def test_the_prompt_block_is_short_and_names_each_note():
    block = notes_block([{"title": "insurance", "folder": "Maya Docs", "path": "insurance.txt", "text": "Renews on 1 March."}])
    assert block.startswith("## From the user's notes")
    assert '- From "insurance" (Maya Docs/insurance.txt): Renews on 1 March.' in block
    assert len(block) < 300


async def test_matching_pieces_go_in_and_unrelated_questions_get_none(app, docs):
    await app.folders.add(str(docs))
    await app.folders.idle()
    pieces = await app.folders.prompt_pieces("when does my insurance renew?")
    assert [p["path"] for p in pieces] == ["insurance.txt"]  # never the clipped web page
    p = pieces[0]
    assert p["folder"] == "Maya Docs" and p["file"] == str(docs / "insurance.txt") and "Renews on 1 March" in p["text"]
    assert await app.folders.prompt_pieces("write a haiku about the sea") == []
    app.config.folders.prompt_notes = 0
    assert await app.folders.prompt_pieces("when does my insurance renew?") == []


async def test_without_search_by_meaning_most_words_must_match(app, docs):
    async def broken(texts, *, model=None):
        raise RuntimeError("model not found")

    app.fake.embed = broken
    await app.folders.add(str(docs))
    await app.folders.idle()
    assert [p["path"] for p in await app.folders.prompt_pieces("when does my lease end?")] == ["lease.txt"]
    assert await app.folders.prompt_pieces("is the lease on my car good value for money?") == []


async def test_a_chat_gets_the_pieces_and_shows_the_note_it_used(app, docs):
    await app.folders.add(str(docs))
    await app.folders.idle()
    sid = await app.store.create_session(channel="desktop")
    app.fake.replies.append("Your Star Health insurance renews on 1 March.")
    events = [ev async for ev in app.agent.run_turn(sid, "when does my insurance renew?", channel="desktop")]
    system = app.fake.calls[-1]["messages"][0]["content"]
    assert "## From the user's notes" in system and "Renews on 1 March" in system
    assert "Always renew insurance early" not in system
    [done] = [e for e in events if isinstance(e, Done)]
    [src] = done.sources
    assert src["kind"] == "file" and src["title"] == "insurance" and src["via"] == "prompt"
    assert src["path"] == str(docs / "insurance.txt") and src["folder"] == "Maya Docs"
    stored = (await app.store.recent_messages(sid, 5))[-1]
    assert stored["sources"] == done.sources

    # a reply that didn't draw on the note shows no chip for it
    sid2 = await app.store.create_session(channel="desktop")
    app.fake.replies.append("I'm not sure, sorry.")
    events = [ev async for ev in app.agent.run_turn(sid2, "when does my insurance renew?", channel="desktop")]
    assert next(e for e in events if isinstance(e, Done)).sources == []

    # an unrelated question gets no pieces
    app.fake.replies.append("Waves fold on the shore.")
    _ = [ev async for ev in app.agent.run_turn(sid2, "write a haiku about the sea", channel="desktop")]
    assert "## From the user's notes" not in app.fake.calls[-1]["messages"][0]["content"]


async def test_incognito_without_memory_gets_no_pieces(app, docs):
    await app.folders.add(str(docs))
    await app.folders.idle()
    app.config.chat.incognito_uses_memory = False
    session = {"incognito": 1}
    prompt, _, notes = await app.agent._system_prompt_with_sources("when does my insurance renew?", "desktop", session)
    assert notes == [] and "From the user's notes" not in prompt


def test_stored_prompt_sources_survive():
    used = AnswerSources([{"kind": "file", "title": "insurance", "path": "/d/i.txt", "via": "prompt", "app": "folders"}])
    assert used.items()[0]["via"] == "prompt"


async def test_no_folders_means_no_lookup(config, isolated_home):
    a = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "none.db", enable_background=False)
    await a.start()
    try:
        assert await a.agent._prompt_notes("when does my insurance renew?") == []
    finally:
        await a.stop()
