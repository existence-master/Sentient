"""Importing memory from ChatGPT and Claude data exports and an OpenClaw workspace (issue #156)."""

from __future__ import annotations

import asyncio
import shutil
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient.gateway.app import create_app
from sentient.memory import review
from sentient.migrate import exports, memory_import
from sentient.migrate.exports import ExportError, directive_to_insight, saved_lines
from sentient.migrate.memory_import import _Known

FIXTURES = Path(__file__).parent / "fixtures"
TRAP = "TRAP-VALUE-THAT-MUST-NEVER-BE-READ"


def _zip(folder: Path, out: Path, extra: dict[str, str] | None = None, inside: str = "") -> Path:
    with zipfile.ZipFile(out, "w") as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, inside + f.relative_to(folder).as_posix())
        for name, text in (extra or {}).items():
            z.writestr(inside + name, text)
    return out


@pytest.fixture
def chatgpt_zip(tmp_path) -> Path:
    """A ChatGPT export: conversations.json plus the files Sentient never opens."""
    return _zip(FIXTURES / "chatgpt", tmp_path / "f3a9c2-2026-10-01-09-00-00.zip",
                {"user.json": TRAP, "chat.html": "<html></html>", "message_feedback.json": "[]"})


@pytest.fixture
def claude_zip(tmp_path) -> Path:
    return _zip(FIXTURES / "claude", tmp_path / "data-2026-10-01-09-00-00.zip", {"users.json": TRAP})


@pytest.fixture
def openclaw_home(tmp_path) -> Path:
    home = tmp_path / ".openclaw"
    shutil.copytree(FIXTURES / "openclaw", home)
    (home / "openclaw.json").write_text(TRAP, encoding="utf-8")
    (home / "workspace" / ".env").write_text(TRAP, encoding="utf-8")
    return home


@pytest.fixture
def reads(monkeypatch) -> list[str]:
    """Every file an export reader opens (zip members and folder files)."""
    seen: list[str] = []
    real = exports._Files.read

    def spy(self, name, limit):
        data = real(self, name, limit)
        if data is not None:
            seen.append(name)
            assert TRAP.encode() not in data, f"{name} must never be read"
        return data

    monkeypatch.setattr(exports._Files, "read", spy)
    return seen


async def run_import(app, kind: str, path: Path, parts: list[str], skip: list[str] | None = None) -> dict:
    job = await app.memory_imports.start(kind, str(path), parts, skip)
    assert job["status"] == "running"
    await app.memory_imports._task
    return app.memory_imports.current()


async def inbox_by_text(app) -> dict[str, dict]:
    return {i["text"]: i for i in (await review.inbox(app))["items"]}


# ---------------------------------------------------------------------------- reading the formats
def test_chatgpt_export(chatgpt_zip, reads):
    e = exports.read("chatgpt", str(chatgpt_zip), user_name="Maya")
    assert [(s.text, s.dated) for s in e.saved] == [
        ("User is allergic to cashews.", "2026-08-02"),
        ("User has a cat named Pixel.", "2026-08-20"),
        ("User's pottery class is on Saturdays at Clayworks in Indiranagar.", "2026-09-02"),
    ]
    convs = [s for s in e.sources if s.part == "conversations"]
    assert [c.title for c in convs] == ["Pottery class and a snack", "10K training plan"]
    text = " ".join(t for s in e.sources for t in s.texts)
    assert "Chennai" not in text  # the edited-away branch is left out
    assert "Roasted chana" not in text and "Model set context" not in text  # only the user's own words
    profile = next(s for s in e.sources if s.part == "profile")
    assert profile.texts == ["I'm Maya, a product designer at Northwind Studio in Bengaluru. I'm vegetarian.",
                             "How I'd like answers: Keep answers short. Use metric units."]
    assert e.stats == {"conversations": 2, "messages": 3, "saved_memories": 3}
    assert reads == ["conversations.json"]


def test_chatgpt_export_inside_a_folder_and_as_a_json_file(tmp_path, chatgpt_zip):
    nested = _zip(FIXTURES / "chatgpt", tmp_path / "again.zip", inside="export/")
    assert len(exports.read("chatgpt", str(nested)).saved) == 3
    assert len(exports.read("chatgpt", str(FIXTURES / "chatgpt" / "conversations.json")).saved) == 3


def test_claude_export(claude_zip, reads):
    e = exports.read("claude", str(claude_zip), user_name="Maya")
    convs = [s for s in e.sources if s.part == "conversations"]
    assert len(convs) == 1  # the empty one is skipped
    assert convs[0].texts == [
        "I'm rewriting my portfolio around the Northwind design system work. I want it live before my Japan trip in April.",
        "Can you tighten this paragraph?",
    ]
    memories = [s for s in e.sources if s.part == "memories"]
    assert [m.title for m in memories] == ["Claude's memory", "Claude's memory for Design critique"]
    assert memories[0].texts[0] == ("Maya is a product designer at Northwind Studio in Bengaluru, where she looks after "
                                    "the design system.")
    assert "Recent months" not in memories[0].texts and len(memories[0].texts) == 4  # headings are not memories
    projects = [s for s in e.sources if s.part == "projects"]
    assert [p.title for p in projects] == ["Project Design critique"]  # the starter project is skipped
    text = " ".join(t for s in e.sources for t in s.texts)
    assert "Kavya's resume" not in text and "Kavya presents first" not in text  # attachments and project files
    assert "Maya wants an outline" not in text  # thinking blocks
    assert e.stats == {"conversations": 1, "messages": 2, "projects": 1, "project_files": 1}
    assert "users.json" not in reads


def test_openclaw_workspace(openclaw_home, reads):
    e = exports.read("openclaw", str(openclaw_home), user_name="Maya")
    assert e.path == openclaw_home / "workspace"
    facts = [s.text for s in e.saved if s.kind == "fact"]
    assert facts == [
        "Maya's sister Anika is a school teacher in Mysuru.",
        "Maya's colleague Kavya builds the components Maya designs.",
        "Maya decided to move her design newsletter from Substack to Buttondown in September 2026.",
        "Maya is allergic to cashews.",
    ]
    insights = [(s.text, s.dated, s.dimension) for s in e.saved if s.kind == "insight"]
    assert insights == [  # the superseded directive, the template's example and its links are left out
        ("Maya prefers short answers with the key point first.", "2026-09-02", "preferences"),
        ("Maya wants the assistant to never schedule meetings before 10am.", "2026-09-10", "preferences"),
    ]
    notes = [(s.title, s.texts) for s in e.sources if s.part == "notes"]
    assert notes == [
        ("Daily note 2026-09-14", ["Maya booked flights to Osaka for 3 April 2027.", "She asked for a packing list (done)."]),
        ("Daily note 2026-09-20 (veena lessons)", ["Maya's veena teacher is Mrs. Iyer; lessons are on Sunday mornings."]),
    ]
    assert e.stats["superseded"] == 1
    assert sorted(reads) == ["MEMORY.md", "USER.md", "memory/2026-09-14.md", "memory/2026-09-20-veena-lessons.md"]


@pytest.mark.parametrize(
    ("kind", "pick", "message"),
    [
        ("chatgpt", "claude", "isn't from ChatGPT"),
        ("claude", "chatgpt", "isn't from Claude"),
        ("chatgpt", "missing", "There's nothing at"),
        ("openclaw", "claude", "OpenClaw workspace folder"),
        ("chatgpt", "empty", "no conversations.json"),
    ],
)
def test_wrong_files_are_plain_errors(tmp_path, chatgpt_zip, claude_zip, kind, pick, message):
    empty = tmp_path / "empty"
    empty.mkdir()
    paths = {"claude": claude_zip, "chatgpt": chatgpt_zip, "missing": tmp_path / "nope.zip", "empty": empty}
    with pytest.raises(ExportError, match=message):
        exports.read(kind, str(paths[pick]))


def test_a_huge_member_is_refused(tmp_path, monkeypatch):
    big = _zip(FIXTURES / "chatgpt", tmp_path / "big.zip")
    monkeypatch.setattr(exports, "MAX_JSON_BYTES", 100)
    with pytest.raises(ExportError, match="too big"):
        exports.read("chatgpt", str(big))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1. [2026-08-02]. User is allergic to cashews.\n2. Lives in Pune.",
         [("User is allergic to cashews.", "2026-08-02"), ("Lives in Pune.", None)]),
        ("- **Has a cat** named Pixel\n\n# Notes\n* ok", [("Has a cat named Pixel", None)]),
    ],
)
def test_saved_lines(raw, expected):
    assert saved_lines(raw) == expected


@pytest.mark.parametrize(
    ("directive", "expected"),
    [
        ("Prefer concise updates", "Maya prefers concise updates."),
        ("Always reply in English.", "Maya wants the assistant to always reply in English."),
        ("Don't use emojis", "Maya wants the assistant not to use emojis."),
        ("Do not book red-eye flights", "Maya wants the assistant not to book red-eye flights."),
        ("Maya's partner is Arjun", "Maya's partner is Arjun."),
    ],
)
def test_directive_to_insight(directive, expected):
    assert directive_to_insight(directive, "Maya") == expected


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        ("Is allergic to cashews.", "Maya is allergic to cashews."),
        ("Works as a product designer.", "Maya works as a product designer."),
        ("User's name is Maya Rao.", "Maya's name is Maya Rao."),
        ("Pixel is Maya's cat.", "Pixel is Maya's cat."),
    ],
)
def test_saved_memories_get_a_subject(saved, expected):
    assert memory_import._clean_fact(saved, "Maya") == expected


# ---------------------------------------------------------------------------- preview
async def test_preview_has_counts_and_samples_without_a_model(app, chatgpt_zip):
    plan = await memory_import.preview(app, "chatgpt", str(chatgpt_zip))
    assert plan["label"] == "ChatGPT" and plan["export"] == chatgpt_zip.name
    assert plan["counts"] == {"memories": 3, "profile": 1, "conversations": 2}
    memories = next(p for p in plan["parts"] if p["id"] == "memories")
    assert [i["text"] for i in memories["items"]] == [
        "Maya is allergic to cashews.", "Maya has a cat named Pixel.",
        "Maya's pottery class is on Saturdays at Clayworks in Indiranagar.",
    ]
    convs = next(p for p in plan["parts"] if p["id"] == "conversations")
    assert convs["messages"] == 3 and convs["batches"] == 1
    assert convs["sources"][0]["title"] == "10K training plan"  # newest first
    assert convs["sources"][0]["snippet"].startswith("I'm training for the TCS World 10K")
    assert app.fake.calls == []  # a preview never calls the model
    assert "user.json" in plan["never_read"]


async def test_preview_marks_what_sentient_already_knows(app, chatgpt_zip):
    await app.memory.remember("Maya has a cat named Pixel and she loves her.", source="manual", use_llm=False)
    await app.memory.remember("Maya is allergic to cashews.", source="manual", use_llm=False)
    plan = await memory_import.preview(app, "chatgpt", str(chatgpt_zip))
    items = {i["text"]: i for i in next(p for p in plan["parts"] if p["id"] == "memories")["items"]}
    assert items["Maya is allergic to cashews."]["action"] == "skip"
    assert items["Maya has a cat named Pixel."]["action"] == "skip"  # an existing fact already says it
    assert items["Maya is allergic to cashews."]["note"] == "Sentient already knows this."
    assert plan["counts"]["memories"] == 1


# ---------------------------------------------------------------------------- importing
async def test_chatgpt_import_fills_the_review_inbox_with_sources(app, chatgpt_zip):
    app.fake.json_replies = [
        {"facts": ["Maya is vegetarian.", "Maya prefers short answers."]},  # custom instructions
        {"facts": [  # conversations
            "Maya is training for the TCS World 10K in Bengaluru next May.",
            "Maya's friend Rohan runs with her on weekend mornings.",
            "Maya is allergic to cashews.",  # a saved memory already says it
            "Maya's sister Riya lives in Pune.",  # in no line: copied from the prompt's example
            "The user asked about the weather.",  # a request, not a fact
        ]},
    ]
    job = await run_import(app, "chatgpt", chatgpt_zip, ["memories", "profile", "conversations"])
    assert job["status"] == "done" and job["total"] == 2 and job["done"] == 2
    assert job["facts"] == 7  # 3 saved memories + 4 from the model
    reads = [c for c in app.fake.calls if c.get("json") and "decomposition" in c["messages"][0]["content"]]
    assert len(reads) == 2  # a batch per part; saved memories are kept as they are
    sorts = [c for c in app.fake.calls if c.get("json") and c["messages"][1]["content"].startswith("Analyze")]
    assert len(sorts) == 7  # then every new memory is sorted into topics
    inbox = await inbox_by_text(app)
    assert set(inbox) == {
        "Maya is allergic to cashews.", "Maya has a cat named Pixel.",
        "Maya's pottery class is on Saturdays at Clayworks in Indiranagar.",
        "Maya is training for the TCS World 10K in Bengaluru next May.",
        "Maya's friend Rohan runs with her on weekend mornings.", "Maya is vegetarian.", "Maya prefers short answers.",
    }
    run = inbox["Maya is training for the TCS World 10K in Bengaluru next May."]
    assert run["from"] == "ChatGPT" and run["source"] == "import:chatgpt"
    assert run["title"] == "10K training plan" and run["dated"] == "2026-09-16"
    assert run["export"] == chatgpt_zip.name
    assert run["snippet"].startswith("I'm training for the TCS World 10K")
    veg = inbox["Maya is vegetarian."]
    assert veg["title"] == "Custom instructions" and "vegetarian" in veg["snippet"]
    saved = inbox["Maya has a cat named Pixel."]
    assert saved["title"] == "Saved memories" and saved["dated"] == "2026-08-20"
    prompts_sent = [c["messages"][1]["content"] for c in reads]
    assert 'From "Custom instructions" (2026-09-02):' in prompts_sent[0]
    prompt = prompts_sent[1]
    assert 'Conversation "10K training plan" (2026-09-16):' in prompt
    assert prompt.index("10K training plan") < prompt.index("Pottery class")  # newest first
    assert "Roasted chana" not in prompt
    # held, so nothing reaches recall or the memory list until approved
    assert await app.memory.list_facts(100, 0, source="import:chatgpt") == []


def test_batches_keep_parts_apart_and_stay_small():
    src = exports.Source(key="c", part="conversations", title="t", dated=None, texts=[f"line {n}" for n in range(11)])
    note = exports.Source(key="p", part="profile", title="p", dated=None, texts=["about me"])
    out = memory_import.batches([note, src], 4000)
    assert [len(b) for b in out] == [1, memory_import.BATCH_LINES, 11 - memory_import.BATCH_LINES]
    assert [len(b) for b in memory_import.batches([src], 20)] == [3, 3, 3, 2]  # by characters too


async def test_importing_again_reads_only_what_is_new(app, chatgpt_zip):
    app.fake.json_replies = [{"facts": ["Maya is vegetarian."]}]
    await run_import(app, "chatgpt", chatgpt_zip, ["memories", "profile", "conversations"])
    plan = await memory_import.preview(app, "chatgpt", str(chatgpt_zip))
    assert plan["counts"] == {"memories": 0, "profile": 0, "conversations": 0}
    assert {s["note"] for p in plan["parts"] for s in p["sources"]} == {"Read in an earlier import."}
    calls = len(app.fake.calls)
    job = await run_import(app, "chatgpt", chatgpt_zip, ["memories", "conversations"])
    assert job["facts"] == 0 and job["total"] == 0 and len(app.fake.calls) == calls


async def test_no_duplicates_across_assistants_and_fuller_wording_fills_a_waiting_card(app, openclaw_home, chatgpt_zip):
    await run_import(app, "openclaw", openclaw_home, ["memories", "profile"])
    before = await inbox_by_text(app)
    assert "Maya is allergic to cashews." in before
    app.fake.json_replies = [{"facts": [
        "Maya's pottery class is on Saturdays.",  # the saved memory already says it, with more
        "Maya's pottery class is on Saturdays at Clayworks in Indiranagar, Bengaluru.",  # says more than it
        "Maya's friend Rohan runs with her on weekend mornings.",
    ]}]
    job = await run_import(app, "chatgpt", chatgpt_zip, ["memories", "conversations"])
    after = await inbox_by_text(app)
    assert list(after).count("Maya is allergic to cashews.") == 1  # ChatGPT's saved memory is OpenClaw's fact
    assert after["Maya is allergic to cashews."]["from"] == "OpenClaw"
    assert "Maya's pottery class is on Saturdays." not in after
    assert "Maya's pottery class is on Saturdays at Clayworks in Indiranagar." not in after
    fuller = after["Maya's pottery class is on Saturdays at Clayworks in Indiranagar, Bengaluru."]
    assert fuller["title"] == "Pottery class and a snack"  # the card points at where the fuller wording came from
    assert job["merged"] == 1 and job["skipped"] == 1 and job["facts"] == 3  # the cashews one was left out in the plan
    assert len(after) == len(before) + job["facts"]


async def test_imported_memories_are_sorted_into_topics(app, openclaw_home):
    app.fake.json_replies = [
        {"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": None},
        {"topics": ["Work & Learning", "Relationships & Social Life"], "memory_type": "long-term", "duration": None},
        {"topics": ["Work & Learning"], "memory_type": "long-term", "duration": None},
        {"topics": ["Health & Wellbeing"], "memory_type": "long-term", "duration": None},
    ]
    job = await run_import(app, "openclaw", openclaw_home, ["memories"])
    assert job["facts"] == 4 and job["status"] == "done"
    rows = await app.memory.pending_facts()
    topics = {f["content"]: f["topics"] for f in rows}
    assert topics["Maya is allergic to cashews."] == ["Health & Wellbeing"]
    assert topics["Maya's sister Anika is a school teacher in Mysuru."] == ["Relationships & Social Life"]
    assert all(f["status"] == "pending" for f in rows)  # sorting never approves anything


async def test_openclaw_profile_becomes_insights_to_review(app, openclaw_home):
    job = await run_import(app, "openclaw", openclaw_home, ["profile"])
    assert job["insights"] == 2 and job["facts"] == 0 and job["total"] == 0
    inbox = await inbox_by_text(app)
    item = inbox["Maya wants the assistant to never schedule meetings before 10am."]
    assert item["kind"] == "insight" and item["from"] == "OpenClaw" and item["title"] == "USER.md"
    assert item["dated"] == "2026-09-10" and item["export"] == ".openclaw/workspace"
    assert item["snippet"] is None  # it would only repeat the memory word for word


async def test_an_insight_takes_the_place_of_a_waiting_fact_it_says_in_full(app, openclaw_home):
    await app.memory.remember("Maya prefers short answers.", source="import:chatgpt", use_llm=False,
                              review=review.note("ChatGPT", "Keep answers short."))
    job = await run_import(app, "openclaw", openclaw_home, ["profile"])
    assert job["insights"] == 2 and job["merged"] == 1
    inbox = await inbox_by_text(app)
    assert "Maya prefers short answers." not in inbox
    assert inbox["Maya prefers short answers with the key point first."]["kind"] == "insight"


async def test_claude_import(app, claude_zip):
    app.fake.json_replies = [
        {"facts": ["Maya is a product designer at Northwind Studio in Bengaluru."]},  # Claude's memory
        {"facts": ["Maya prefers feedback as numbered lists."]},  # project instructions
        {"facts": ["Maya is rewriting her portfolio around the Northwind design system work."]},  # conversations
    ]
    job = await run_import(app, "claude", claude_zip, ["memories", "projects", "conversations"])
    assert job["status"] == "done" and job["facts"] == 3 and job["total"] == 3
    inbox = await inbox_by_text(app)
    assert inbox["Maya is a product designer at Northwind Studio in Bengaluru."]["title"] == "Claude's memory"
    portfolio = inbox["Maya is rewriting her portfolio around the Northwind design system work."]
    assert portfolio["title"] == "Portfolio case study outline" and portfolio["dated"] == "2026-09-05"
    assert portfolio["export"] == claude_zip.name and portfolio["from"] == "Claude"


async def test_an_approved_import_is_recalled(app, chatgpt_zip):
    await run_import(app, "chatgpt", chatgpt_zip, ["memories"])
    item = (await inbox_by_text(app))["Maya is allergic to cashews."]
    assert await review.approve(app, "fact", str(item["id"]))
    hits = await app.memory.recall("Is Maya allergic to anything like cashews?", min_similarity=0.0)
    assert hits[0]["content"] == "Maya is allergic to cashews."


async def test_a_failed_batch_is_read_again_next_time(app, chatgpt_zip, monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("model went away")

    monkeypatch.setattr(app.fake, "complete_json", broken)
    job = await run_import(app, "chatgpt", chatgpt_zip, ["conversations"])
    assert job["status"] == "done" and job["failed"] == 1 and job["facts"] == 0
    assert "couldn't be read" in job["message"]
    plan = await memory_import.preview(app, "chatgpt", str(chatgpt_zip))
    assert plan["counts"]["conversations"] == 2


async def test_stopping_an_import(app, chatgpt_zip, monkeypatch):
    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return {"facts": []}

    monkeypatch.setattr(app.fake, "complete_json", slow)
    await app.memory_imports.start("chatgpt", str(chatgpt_zip), ["conversations"])
    with pytest.raises(memory_import.ImportBusy):
        await app.memory_imports.start("chatgpt", str(chatgpt_zip), ["conversations"])
    await asyncio.sleep(0)
    assert app.memory_imports.cancel()["message"] == "Stopping after this batch."
    gate.set()
    await app.memory_imports._task
    assert app.memory_imports.current()["status"] in {"cancelled", "done"}


async def test_removing_an_import(app, openclaw_home):
    app.fake.json_replies = [{"facts": ["Maya booked flights to Osaka for 3 April 2027."]}]
    await run_import(app, "openclaw", openclaw_home, ["memories", "profile", "notes"])
    item = (await inbox_by_text(app))["Maya is allergic to cashews."]
    await review.approve(app, "fact", str(item["id"]))
    result = await memory_import.remove(app, "openclaw")
    assert result == {"facts": 5, "insights": 2}
    assert (await review.inbox(app))["count"] == 0
    assert await app.memory.list_facts(100, 0, source="import:openclaw") == []
    plan = await memory_import.preview(app, "openclaw", str(openclaw_home))
    assert plan["counts"]["notes"] == 2  # read again after an undo


def test_known_matching():
    known = _Known("Maya")
    known.add(1, "Maya lives in Bengaluru.", waiting_import=False)
    known.add(2, "Maya is training for a 10K.", waiting_import=True)
    assert known.match("maya lives in bengaluru.")[0] == "same"
    assert known.match("Maya lives in a city.")[0] == "same"
    assert known.match("Maya is training for a 10K in May.") == ("fuller", known.rows[2])
    assert known.match("Maya lives in Bengaluru with her cat.")[0] == ""  # active facts are never reworded
    assert known.match("Maya has a dog.") == ("", None)


@pytest.mark.parametrize(
    ("waiting", "new", "expected"),
    [
        ("Maya runs a weekly design critique on Wednesdays.", "Maya runs the Wednesday design critique at Northwind.", "fuller"),
        ("Maya works at Northwind Studio as a designer.", "Maya works as a designer at Northwind Studio.", "same"),
        ("Maya runs 5 km on Mondays.", "Maya runs 10 km on Mondays.", ""),  # numbers disagree
        ("Maya's sister lives in Pune.", "Maya's brother lives in Pune.", ""),  # different people
        ("Maya likes masala chai.", "Maya dislikes masala chai.", ""),
        ("Maya does not eat eggs.", "Maya eats eggs on Sundays.", ""),  # a negation on one side
        ("Maya's veena teacher is Mrs. Iyer.", "Maya's veena lessons are on Sunday mornings with Mrs. Iyer.", "fuller"),
        ("Maya's mother loves gardening.", "Maya's mother's birthday is on 2 November.", ""),  # another thing
        ("Maya is training for the TCS World 10K in Bengaluru.", "Maya lives in Bengaluru.", ""),  # too short to judge
        ("Maya is based in Bengaluru.", "Maya lives in Bengaluru.", "same"),  # same kind of fact, same place
        ("Maya lives in Bengaluru.", "Maya moved out of Bengaluru.", ""),
    ],
)
def test_paraphrases_of_a_waiting_import(waiting, new, expected):
    known = _Known("Maya")
    known.add(1, waiting, waiting_import=True)
    assert known.match(new)[0] == expected


def test_a_remembered_fact_is_never_reworded_by_a_paraphrase():
    known = _Known("Maya")
    known.add(1, "Maya runs a weekly design critique on Wednesdays.", waiting_import=False)
    assert known.match("Maya runs the Wednesday design critique at Northwind.") == ("", None)


# ---------------------------------------------------------------------------- REST
def test_routes(app, chatgpt_zip, openclaw_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    monkeypatch.setattr(exports.Path, "home", classmethod(lambda cls: openclaw_home.parent))
    with TestClient(create_app(app)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/import/openclaw").json() == {"path": str(openclaw_home / "workspace"), "exists": True}
        assert c.get("/api/import/chatgpt").json() == {"path": None, "exists": False}
        assert c.get("/api/import/gemini").status_code == 404
        assert c.get("/api/import/hermes").json()["path"].endswith(".hermes")  # still Hermes' own route
        assert c.get("/api/import/memory/job").json() is None
        bad = c.post("/api/import/chatgpt/preview", json={"path": str(openclaw_home)})
        assert bad.status_code == 400 and "ChatGPT export" in bad.json()["detail"]
        plan = c.post("/api/import/chatgpt/preview", json={"path": str(chatgpt_zip)}).json()
        assert plan["counts"]["memories"] == 3
        assert c.post("/api/import/chatgpt/apply", json={"path": str(chatgpt_zip), "parts": []}).status_code == 400
        job = c.post("/api/import/chatgpt/apply", json={"path": str(chatgpt_zip), "parts": ["memories"]}).json()
        assert job["app"] == "chatgpt" and job["status"] == "running" and job["export"] == chatgpt_zip.name
        deadline = time.monotonic() + 30  # a busy Windows runner can take far longer than a few seconds: #422
        while (now := c.get("/api/import/memory/job").json())["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert now["id"] == job["id"] and now["status"] == "done" and now["facts"] == 3
        items = c.get("/api/memories/review").json()["items"]
        assert {i["title"] for i in items} == {"Saved memories"} and {i["export"] for i in items} == {chatgpt_zip.name}
        assert c.post("/api/import/memory/job/cancel").json()["status"] == "done"
        assert c.delete("/api/import/chatgpt/memories").json() == {"facts": 3, "insights": 0}


async def test_stop_everything_stops_an_import_and_blocks_new_ones(app, chatgpt_zip, monkeypatch):
    gate = asyncio.Event()
    reading = asyncio.Event()

    async def slow(*a, **k):
        reading.set()  # the saved memories are in; it now waits on the model for the conversations
        await gate.wait()
        return {"facts": []}

    monkeypatch.setattr(app.fake, "complete_json", slow)
    await app.memory_imports.start("chatgpt", str(chatgpt_zip), ["memories", "conversations"])
    await asyncio.wait_for(reading.wait(), 10)  # not a fixed sleep: slow CI machines took longer than 50 ms
    out = await app.stop_all("test")
    assert out["cancelled"] >= 1
    assert app.memory_imports.current()["status"] == "cancelled"
    assert app.memory_imports.current()["facts"] == 3  # what it saved stays for review
    with pytest.raises(memory_import.ImportBusy, match="Resume"):
        await app.memory_imports.start("chatgpt", str(chatgpt_zip), ["conversations"])
    await app.resume("test")


async def test_stop_everything_while_an_import_starts_and_removing_while_it_runs(app, chatgpt_zip, monkeypatch):
    real = memory_import._plan

    async def slow_plan(*a, **k):
        await asyncio.sleep(0.05)
        return await real(*a, **k)

    monkeypatch.setattr(memory_import, "_plan", slow_plan)
    starting = asyncio.create_task(app.memory_imports.start("chatgpt", str(chatgpt_zip), ["memories"]))
    await asyncio.sleep(0.01)
    assert await app.memory_imports.halt() == 1
    with pytest.raises(memory_import.ImportBusy):
        await starting
    assert app.memory_imports.current() is None and not app.memory_imports.running()

    gate = asyncio.Event()

    async def slow(*a, **k):
        await gate.wait()
        return {"facts": []}

    monkeypatch.setattr(app.fake, "complete_json", slow)
    await app.memory_imports.start("chatgpt", str(chatgpt_zip), ["conversations"])
    with pytest.raises(memory_import.ImportBusy):
        await memory_import.remove(app, "chatgpt")
    gate.set()
    await app.memory_imports._task


async def test_two_starts_at_once_run_one_import(app, chatgpt_zip):
    results = await asyncio.gather(
        app.memory_imports.start("chatgpt", str(chatgpt_zip), ["memories"]),
        app.memory_imports.start("chatgpt", str(chatgpt_zip), ["memories"]),
        return_exceptions=True,
    )
    assert sum(isinstance(r, memory_import.ImportBusy) for r in results) == 1
    await app.memory_imports._task
