"""Seed a throwaway SENTIENT_HOME with demo memory, summaries, skills and evolution log.

No LLM traffic: a tiny fake provider supplies deterministic embeddings.

    .venv/Scripts/python.exe desktop/scripts/seed-memory-skills.py --home <dir> [--reset] [--theme dark|light]
    .venv/Scripts/python.exe desktop/scripts/seed-memory-skills.py --home <dir> --theme light --theme-only

The real engine reads the same database. Its embedding role is set to a model that does not
exist, so semantic search falls back to keyword matching instead of re-indexing the vectors.
Background jobs (reviewer, curator, profile updates, summaries, proactivity) are switched off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DIM = 64
MARKER = ".seeded-demo"
TOPICS = [
    "Personal Identity", "Interests & Lifestyle", "Work & Learning", "Health & Wellbeing",
    "Relationships & Social Life", "Financial", "Goals & Challenges", "Miscellaneous",
]
STOP = set(
    "a an the and or of to in on at for with is are was be has have his her he she it its as by from that this"
    " sarthak sarthak's".split()
)


def bow(text: str) -> list[float]:
    """Hashed bag of words in dims 8..DIM (dims 0..7 are topic anchors)."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        w = word.strip(".,!?'\"()")
        if not w or w in STOP:
            continue
        vec[8 + zlib.crc32(w.encode()) % (DIM - 8)] += 1.0
    n = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / n for v in vec]


def fact_vector(content: str, topics: list[str]) -> list[float]:
    vec = [x * 0.45 for x in bow(content)]
    for i, t in enumerate(topics):
        vec[TOPICS.index(t)] += 1.0 if i == 0 else 0.55
    n = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / n for v in vec]


class FakeProvider:
    """Deterministic, offline stand-in for app.llm (pattern from tests/conftest.py)."""

    def model_for(self, role: str) -> str:
        return f"fake/{role}"

    async def stream(self, role, messages, tools=None, *, model=None):
        from sentient.llm.provider import StreamChunk

        yield StreamChunk(text="ok", model="fake")
        yield StreamChunk(done=True, model="fake")

    async def complete_text(self, role, messages, *, model=None) -> str:
        return "ok"

    async def complete_json(self, role, messages, *, model=None):
        return {}

    async def embed(self, texts, *, model=None):
        return [bow(t) for t in texts]


# (content, topics, source, memory_type, duration, days_ago)
FACTS = [
    ("Sarthak is the founder of Existence, a startup building Sentient", ["Work & Learning", "Personal Identity"], "onboarding", "long-term", None, 41),
    ("Sarthak lives in Pune, India", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Sarthak's timezone is Asia/Kolkata", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Sarthak prefers short, direct answers without filler", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Sarthak values privacy and wants his assistant to run locally", ["Personal Identity", "Work & Learning"], "onboarding", "long-term", None, 41),
    ("Sarthak is an early riser and does deep work before 10am", ["Interests & Lifestyle", "Work & Learning"], "conversation", "long-term", None, 38),
    ("Sarthak writes most of his code in Python and TypeScript", ["Work & Learning"], "conversation", "long-term", None, 37),
    ("Sarthak uses a Windows 11 desktop with an 8 GB NVIDIA GPU for development", ["Work & Learning"], "conversation", "long-term", None, 36),
    ("Sarthak runs local models through Ollama for Sentient development", ["Work & Learning"], "conversation", "long-term", None, 35),
    ("Sarthak is planning a smart glasses companion for Sentient", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 33),
    ("Sarthak wants to launch Sentient v3 publicly in September 2026", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 30),
    ("Sarthak studied computer engineering", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Sarthak previously built Sentient v2 with FastAPI, Celery and Next.js", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Sarthak has experience with LLM agents, MCP servers and vector databases", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Sarthak gave a talk on local-first AI assistants at a Pune meetup", ["Work & Learning", "Relationships & Social Life"], "file:resume.pdf", "long-term", None, 29),
    ("Sarthak is reading Designing Data-Intensive Applications", ["Work & Learning", "Interests & Lifestyle"], "conversation", "long-term", None, 26),
    ("Sarthak is learning Rust on weekends", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 24),
    ("Sarthak enjoys playing chess online in the evenings", ["Interests & Lifestyle"], "conversation", "long-term", None, 23),
    ("Sarthak likes filter coffee and drinks two cups a day", ["Interests & Lifestyle", "Health & Wellbeing"], "conversation", "long-term", None, 22),
    ("Sarthak goes trekking in the Sahyadri hills during monsoon", ["Interests & Lifestyle", "Health & Wellbeing"], "conversation", "long-term", None, 21),
    ("Sarthak listens to lo-fi music while coding", ["Interests & Lifestyle"], "conversation", "long-term", None, 20),
    ("Sarthak is a fan of science fiction, especially The Expanse", ["Interests & Lifestyle"], "manual", "long-term", None, 19),
    ("Sarthak photographs street scenes with a Fujifilm camera", ["Interests & Lifestyle"], "manual", "long-term", None, 18),
    ("Sarthak runs 5 km three times a week", ["Health & Wellbeing", "Interests & Lifestyle"], "conversation", "long-term", None, 18),
    ("Sarthak is trying to sleep before midnight on weekdays", ["Health & Wellbeing", "Goals & Challenges"], "conversation", "long-term", None, 17),
    ("Sarthak is lactose intolerant", ["Health & Wellbeing"], "manual", "long-term", None, 16),
    ("Sarthak meditates for ten minutes after waking up", ["Health & Wellbeing"], "conversation", "long-term", None, 15),
    ("Sarthak has an annual health checkup due in October", ["Health & Wellbeing"], "conversation", "long-term", None, 14),
    ("Sarthak's sister Aditi is a doctor in Bengaluru", ["Relationships & Social Life"], "conversation", "long-term", None, 14),
    ("Sarthak's parents live in Nashik", ["Relationships & Social Life"], "conversation", "long-term", None, 13),
    ("Sarthak calls his parents every Sunday evening", ["Relationships & Social Life", "Interests & Lifestyle"], "conversation", "long-term", None, 13),
    ("Sarthak's cofounder Rohan handles hardware for the glasses project", ["Relationships & Social Life", "Work & Learning"], "conversation", "long-term", None, 12),
    ("Sarthak's close friend Neha is a product designer who reviews Sentient's UI", ["Relationships & Social Life", "Work & Learning"], "conversation", "long-term", None, 12),
    ("Sarthak's mother's birthday is on 2 November", ["Relationships & Social Life"], "manual", "long-term", None, 11),
    ("Sarthak mentors two engineering students every month", ["Relationships & Social Life", "Work & Learning"], "conversation", "long-term", None, 10),
    ("Sarthak bootstraps Existence from savings and consulting income", ["Financial", "Work & Learning"], "conversation", "long-term", None, 10),
    ("Sarthak tracks expenses in a monthly spreadsheet", ["Financial"], "conversation", "long-term", None, 9),
    ("Sarthak invests a fixed amount in index funds every month", ["Financial"], "conversation", "long-term", None, 9),
    ("Sarthak's cloud budget for Sentient is capped at 50 dollars a month", ["Financial", "Work & Learning"], "conversation", "long-term", None, 8),
    ("Sarthak wants to reach 1000 beta users by the end of the year", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 8),
    ("Sarthak finds it hard to stop working late at night", ["Goals & Challenges", "Health & Wellbeing"], "conversation", "long-term", None, 7),
    ("Sarthak wants to write a weekly founder newsletter", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 7),
    ("Sarthak is preparing a pitch deck for a pre-seed round", ["Goals & Challenges", "Financial"], "conversation", "long-term", None, 6),
    ("Sarthak wants to run a half marathon next year", ["Goals & Challenges", "Health & Wellbeing"], "manual", "long-term", None, 6),
    ("Sarthak prefers meetings after 2pm", ["Work & Learning", "Personal Identity"], "conversation", "long-term", None, 5),
    ("Sarthak uses Gmail and Google Calendar for work", ["Work & Learning", "Miscellaneous"], "conversation", "long-term", None, 5),
    ("Sarthak keeps project notes in Notion", ["Work & Learning", "Miscellaneous"], "conversation", "long-term", None, 4),
    ("Sarthak's laptop is a ThinkPad X1 Carbon", ["Miscellaneous"], "conversation", "long-term", None, 4),
    ("Sarthak's favourite restaurant in Pune is Vaishali on FC Road", ["Interests & Lifestyle", "Miscellaneous"], "conversation", "long-term", None, 3),
    ("Sarthak drives a grey Maruti Swift", ["Miscellaneous"], "manual", "long-term", None, 3),
    ("Sarthak switched Sentient's database from MongoDB to SQLite", ["Work & Learning"], "conversation", "long-term", None, 2),
    ("Sarthak decided the v3 UI uses React, Tailwind and Electron", ["Work & Learning"], "conversation", "long-term", None, 2),
    ("Sarthak is reviewing Neha's designs for the memory graph", ["Work & Learning", "Relationships & Social Life"], "conversation", "long-term", None, 1),
    ("Sarthak wants weekly reviews compiled every Friday evening", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 1),
    ("Sarthak prefers dark mode in every app", ["Personal Identity", "Miscellaneous"], "conversation", "long-term", None, 0),
    # short-term (a few expire soon)
    ("Sarthak has a call with Rohan tomorrow at 3pm about the glasses prototype", ["Work & Learning", "Relationships & Social Life"], "conversation", "short-term", "1 day", 0),
    ("Sarthak is travelling to Mumbai this weekend", ["Interests & Lifestyle"], "conversation", "short-term", "3 days", 0),
    ("Sarthak needs to renew his car insurance this week", ["Financial", "Miscellaneous"], "conversation", "short-term", "5 days", 1),
    ("Sarthak is fasting today and skipping lunch", ["Health & Wellbeing"], "conversation", "short-term", "6 hours", 0),
    ("Sarthak is waiting for a reply from an investor about the pitch deck", ["Goals & Challenges", "Financial"], "conversation", "short-term", "2 weeks", 2),
]

UPDATED = {
    "Sarthak runs 5 km three times a week": "Sarthak runs 3 km twice a week",
    "Sarthak's cloud budget for Sentient is capped at 50 dollars a month": "Sarthak's cloud budget for Sentient is capped at 30 dollars a month",
}

SUMMARIES = [
    ("Planning the Sentient v3 launch", 1,
     "I helped Sarthak turn the September launch into a checklist: finish the memory and skills screens, record a demo video, "
     "and invite the first 50 beta users from the waitlist. He wants the release notes written in plain language."),
    ("Weekly review, week 36", 3,
     "Sarthak and I went through his week. He shipped the SQLite migration, missed two runs because of late nights, and "
     "asked me to remind him about sleep before midnight. We moved the investor follow-ups to Monday."),
    ("Glasses prototype with Rohan", 6,
     "We compared microphones and battery packs for the smart glasses. Rohan prefers a bone-conduction speaker; Sarthak "
     "wants voice latency under 800 ms. I drafted an email to two suppliers and saved their quotes."),
    ("Trek planning for Rajmachi", 11,
     "Sarthak asked for a monsoon trek near Pune. I suggested Rajmachi, checked the weather for Saturday, and listed what to pack. "
     "He invited Neha and two friends."),
    ("Budget and pre-seed deck", 16,
     "I summarised Sarthak's monthly spending sheet and helped outline a pre-seed deck: problem, local-first privacy, traction, "
     "and a 12-month plan. He wants to keep cloud costs under 30 dollars a month."),
]

SOUL = """# Soul

You are Sentient, a personal assistant that lives on Sarthak's own computer.

## How you behave
- Warm, direct, and brief. You talk like a capable friend.
- You act. When a request can be done with your tools, do it, then report the result in one or two sentences.
- You remember. Facts about Sarthak that come up are worth saving.
- You respect approvals. Anything that sends, deletes, or spends waits for a yes.

## Voice
Plain language, short sentences, no filler.
"""

USER = """# About Sarthak

- Founder of **Existence**, building Sentient, a local-first personal assistant.
- Lives in Pune, India (Asia/Kolkata). Early riser; deep work before 10am.
- Prefers short, direct answers. Meetings after 2pm, please.

## Work
- Python and TypeScript. Windows 11 desktop, 8 GB GPU, Ollama for local models.
- Launching Sentient v3 in September 2026; smart glasses companion is next.

## Learned

- Sarthak is learning Rust on weekends
- Sarthak's sister Aditi is a doctor in Bengaluru
- Sarthak wants weekly reviews compiled every Friday evening
"""

MEMORY_MD = """# Long-term memory

## Ongoing projects
- **Sentient v3 launch** (Sept 2026): memory + skills screens, demo video, first 50 beta invites.
- **Smart glasses** with Rohan: voice latency target under 800 ms; supplier quotes saved.
- **Pre-seed round**: deck outlined; waiting on one investor reply.

## Preferences
- Short answers, dark mode, meetings after 2pm.
- Keep cloud spend under $30/month.

## Commitments
- Weekly review every Friday evening.
- Call parents on Sunday evenings.
"""

SKILLS = [
    dict(name="weekly-review", author="user", tags=["productivity", "planning"], requires_tools=["gcalendar", "gmail"],
         description="Compile Sarthak's weekly review from calendar, email and task results every Friday.",
         uses=14, views=22, patches=2, last_used_days=1,
         body="""# Weekly review

## When to use
Every Friday evening, or when Sarthak asks "how did my week go?".

## Procedure
1. Pull this week's calendar events and count meetings vs. focus blocks.
2. Search Gmail for threads Sarthak replied to and anything still waiting on him.
3. List completed and failed tasks from the task log.
4. Write three sections: **Shipped**, **Slipped**, **Next week**.
5. End with one question about priorities for Monday.

## Pitfalls
- Skip newsletters and automated notifications.
- Keep it under 250 words.

## Verification
- Every item links back to an email, event or task.
"""),
    dict(name="investor-follow-up", author="assistant", tags=["fundraising", "email"], requires_tools=["gmail"],
         description="Draft a polite follow-up to investors who have not replied in five business days.",
         uses=6, views=9, patches=1, last_used_days=3, created_by_review=True,
         body="""# Investor follow-up

## When to use
An investor thread has had no reply for five business days.

## Procedure
1. Find the last message Sarthak sent in the thread.
2. Draft a two-sentence follow-up with one new piece of traction.
3. Ask for approval before sending.

## Pitfalls
- Never follow up more than twice.

## Verification
- The draft references the original deck or meeting.
"""),
    dict(name="trek-planner", author="assistant", tags=["travel", "weather"], requires_tools=["weather", "maps"],
         description="Plan a day trek near Pune with weather, travel time and a packing list.",
         uses=3, views=4, patches=0, last_used_days=11, created_by_review=True,
         body="""# Trek planner

## When to use
Sarthak wants to go trekking this weekend.

## Procedure
1. Check the weather for Saturday and Sunday at 2-3 candidate forts.
2. Estimate drive time from Pune with maps.
3. Pick the best option and list a monsoon packing list.

## Pitfalls
- Avoid routes with red rainfall alerts.

## Verification
- Forecast fetched within the last 12 hours.
"""),
    dict(name="meeting-prep", author="user", tags=["meetings"], requires_tools=["gcalendar"],
         description="Prepare a one-page brief before any external meeting.",
         uses=9, views=12, patches=1, last_used_days=2,
         body="""# Meeting prep

## When to use
30 minutes before an external meeting on the calendar.

## Procedure
1. Read the event description and attendees.
2. Search email for the last thread with each attendee.
3. Summarise context, open questions and a proposed agenda.

## Pitfalls
- Don't include private notes about attendees.

## Verification
- Brief fits on one screen.
"""),
    dict(name="expense-summary", author="community", tags=["finance"], requires_tools=[],
         description="Summarise a monthly expense spreadsheet into categories and trends.",
         uses=4, views=5, patches=0, last_used_days=9,
         body="""# Expense summary

## When to use
Sarthak shares or mentions his monthly spending sheet.

## Procedure
1. Group rows by category.
2. Compare with the previous month and flag changes above 20%.
3. Produce a short table and two suggestions.

## Pitfalls
- Never move money or change the sheet.

## Verification
- Totals match the sheet.
"""),
    dict(name="release-notes", author="assistant", tags=["writing", "product"], requires_tools=["github"],
         description="Turn merged pull requests into plain-language release notes.",
         uses=1, views=2, patches=0, last_used_days=21, created_by_review=True, stale=True,
         body="""# Release notes

## When to use
Before tagging a Sentient release.

## Procedure
1. List merged pull requests since the last tag.
2. Group them into New, Improved and Fixed.
3. Rewrite each in one plain sentence a non-technical user understands.

## Pitfalls
- Leave out internal refactors.

## Verification
- Every line maps to a merged PR.
"""),
]

PENDING_NEW = dict(
    name="supplier-quote-compare", tags=["hardware", "email"], requires_tools=["gmail"],
    description="Compare hardware supplier quotes from email into a table with price, lead time and MOQ.",
    reason="In the glasses prototype chat you asked me to collect three supplier quotes and compare them. "
    "I used 6 tool calls to search Gmail, open attachments and build a table; this is likely to repeat.",
    body="""# Supplier quote comparison

## When to use
Sarthak asks to compare quotes from hardware suppliers.

## Procedure
1. Search Gmail for recent emails with "quote" or "quotation" from the named suppliers.
2. Open PDF or spreadsheet attachments and extract unit price, currency, lead time and minimum order quantity.
3. Convert prices to INR using today's rate.
4. Present a table sorted by total cost for the requested quantity.
5. Recommend one supplier in a single sentence and list open questions.

## Pitfalls
- Quotes often exclude shipping and GST; call that out.
- Don't reply to suppliers without approval.

## Verification
- Every row cites the email it came from.
""")

PENDING_UPDATE_BODY = """# Weekly review

## When to use
Every Friday evening, or when Sarthak asks "how did my week go?".

## Procedure
1. Pull this week's calendar events and count meetings vs. focus blocks.
2. Search Gmail for threads Sarthak replied to and anything still waiting on him.
3. List completed and failed tasks from the task log.
4. Check health goals: runs logged and nights he slept before midnight.
5. Write four sections: **Shipped**, **Slipped**, **Health**, **Next week**.
6. End with one question about priorities for Monday.

## Pitfalls
- Skip newsletters and automated notifications.
- Keep it under 300 words.
- Don't nag about missed runs; state them once.

## Verification
- Every item links back to an email, event or task.
"""

ARCHIVED = dict(
    name="standup-notes", tags=["meetings"], requires_tools=["slack"],
    description="Post daily stand-up notes to Slack from yesterday's commits.",
    body="""# Stand-up notes

## When to use
Each weekday at 10am.

## Procedure
1. Collect yesterday's commits.
2. Summarise into three bullets.
3. Post to the team channel after approval.

## Pitfalls
- Skip weekends.

## Verification
- Posted once per day.
""")


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


async def seed(home: Path, theme: str) -> None:
    from sentient.app import SentientApp
    from sentient.config import SentientConfig, save_config
    from sentient.evolution.log import log_event

    now = datetime.now(UTC)
    cfg = SentientConfig()
    cfg.assistant.name = "Sentient"
    cfg.assistant.user_name = "Sarthak"
    cfg.assistant.timezone = "Asia/Kolkata"
    cfg.assistant.location = "Pune, India"
    cfg.assistant.onboarding_complete = True
    cfg.ui.theme = theme
    # keep the real engine from calling models while the demo is open
    cfg.models.roles.embedding = "ollama/seeded-demo-no-embeddings"
    cfg.memory.extract_after_turn = False
    cfg.memory.summaries_enabled = False
    cfg.evolution.review_enabled = False
    cfg.evolution.curator_enabled = False
    cfg.evolution.user_profile_updates = False
    cfg.proactivity.enabled = False
    save_config(cfg)

    app = SentientApp(cfg, llm=FakeProvider(), db_path=home / "sentient.db", enable_background=False)
    await app.start()
    try:
        store, mem = app.store, app.memory
        assert mem is not None, "sqlite-vec is required"
        await mem.vec.ensure(DIM)

        # ---------------------------------------------------------------- facts
        for content, topics, source, mtype, duration, days_ago in FACTS:
            fid = await mem._insert(
                content, fact_vector(content, topics), source=source, topics=topics, memory_type=mtype, duration=duration
            )
            created = now - timedelta(days=days_ago, hours=(fid * 7) % 11, minutes=(fid * 13) % 50)
            if days_ago == 0:
                created = now - timedelta(minutes=20 + fid % 90)
            await store.execute(
                "UPDATE facts SET created_at = ?, updated_at = ? WHERE id = ?", (iso(created), iso(created), fid)
            )
            if content in UPDATED:
                new = UPDATED[content]
                await store.execute(
                    "UPDATE facts SET content = ?, previous_content = ?, updated_at = ? WHERE id = ?",
                    (new, content, iso(now - timedelta(hours=5 + fid % 20)), fid),
                )
                await mem.vec.upsert(fid, fact_vector(new, topics))

        # ---------------------------------------------------------------- sessions + episodic summaries
        glasses_sid = None
        for title, days_ago, summary in SUMMARIES:
            sid = await store.create_session("web", title)
            if title.startswith("Glasses"):
                glasses_sid = sid
            start = now - timedelta(days=days_ago, hours=3)
            chunk = []
            for i in range(6):
                mid = f"seed{zlib.crc32((title + str(i)).encode()):08x}"
                ts = iso(start + timedelta(minutes=4 * i))
                role = "user" if i % 2 == 0 else "assistant"
                text = f"(demo) {title}: message {i + 1}" if role == "user" else f"(demo) Working on {title.lower()}."
                await store.execute(
                    "INSERT INTO messages(id, session_id, role, content, created_at, summarized) VALUES(?,?,?,?,?,1)",
                    (mid, sid, role, text, ts),
                )
                chunk.append({"id": mid, "created_at": ts})
            await store.execute(
                "UPDATE sessions SET created_at = ?, updated_at = ? WHERE id = ?",
                (chunk[0]["created_at"], chunk[-1]["created_at"], sid),
            )
            s = await mem.episodic.add_summary(sid, summary, chunk)
            await log_event(
                store, "summary_created",
                {"id": s["id"], "session_id": sid, "start_at": s["start_at"], "end_at": s["end_at"]},
                ts=iso(start + timedelta(hours=1, minutes=5)),
            )

        # ---------------------------------------------------------------- workspace
        app.workspace.write("soul", SOUL)
        app.workspace.write("user", USER)
        app.workspace.write("memory", MEMORY_MD)

        # ---------------------------------------------------------------- skills
        lib = app.skills
        for sk in SKILLS:
            lib.write(
                sk["name"], sk["description"], sk["body"], author=sk["author"], tags=sk["tags"],
                requires_tools=sk["requires_tools"], created_by_review=sk.get("created_by_review", False),
                version=1 + sk["patches"],
            )
            await log_event(
                store, "skill_created",
                {"name": sk["name"], "author": sk["author"], **({"approved": True} if sk["author"] == "assistant" else {})},
                ts=iso(now - timedelta(days=sk["last_used_days"] + 10)),
            )
        lib.write(
            ARCHIVED["name"], ARCHIVED["description"], ARCHIVED["body"], author="assistant",
            tags=ARCHIVED["tags"], requires_tools=ARCHIVED["requires_tools"], created_by_review=True,
        )
        lib.reload({p.id for p in app.registry.plugins()})
        await lib.sync_stats(store)
        for sk in SKILLS:
            await store.execute(
                "UPDATE skill_stats SET use_count = ?, view_count = ?, patch_count = ?, last_used_at = ?, state = ?"
                " WHERE name = ?",
                (
                    sk["uses"], sk["views"], sk["patches"], iso(now - timedelta(days=sk["last_used_days"], hours=2)),
                    "stale" if sk.get("stale") else "active", sk["name"],
                ),
            )

        lib.archive(ARCHIVED["name"])
        await lib.set_state(ARCHIVED["name"], "archived", store)
        await store.execute(
            "UPDATE skill_stats SET use_count = 2, view_count = 3, last_used_at = ? WHERE name = ?",
            (iso(now - timedelta(days=48)), ARCHIVED["name"]),
        )

        # pending: a brand-new skill
        lib.write(
            PENDING_NEW["name"], PENDING_NEW["description"], PENDING_NEW["body"], author="assistant",
            tags=PENDING_NEW["tags"], requires_tools=PENDING_NEW["requires_tools"], pending=True, created_by_review=True,
        )
        await lib.record_patch(PENDING_NEW["name"], "pending_review", store)
        await log_event(
            store, "skill_created",
            {"name": PENDING_NEW["name"], "pending": True, "reason": PENDING_NEW["reason"], "session_id": glasses_sid},
            ts=iso(now - timedelta(hours=20)),
        )
        await app.notify(
            "skill",
            f"I learned a repeatable procedure and saved it as **{PENDING_NEW['name']}**: "
            f"{PENDING_NEW['description']} Review it in Skills.",
            title="New skill to review",
            payload={"skill": PENDING_NEW["name"], "action": "create", "origin": {"session_id": glasses_sid}},
        )

        # pending: a proposed update of weekly-review (diff works)
        wr = SKILLS[0]
        lib.write(
            "weekly-review",
            "Compile Sarthak's weekly review from calendar, email, tasks and health goals every Friday.",
            PENDING_UPDATE_BODY, author="user", tags=[*wr["tags"], "health"], requires_tools=wr["requires_tools"],
            pending=True, created_by_review=True,
        )
        await lib.record_patch("weekly-review", "pending_review", store)
        reason = (
            "During the last two weekly reviews you asked me to add how many runs you logged and how often you slept "
            "before midnight. Adding a Health section makes that automatic."
        )
        await log_event(
            store, "skill_patched",
            {"name": "weekly-review", "pending": True, "reason": reason, "task_id": "demo-weekly-review"},
            ts=iso(now - timedelta(hours=9)),
        )
        await app.notify(
            "skill",
            "I found a better way to do **weekly-review** and drafted an update. Review the change in Skills.",
            title="Skill update to review",
            payload={"skill": "weekly-review", "action": "patch", "origin": {"task_id": "demo-weekly-review"}},
        )

        # ---------------------------------------------------------------- more evolution entries
        await log_event(store, "skill_archived", {"name": ARCHIVED["name"], "by": "curator", "idle_days": 34},
                        ts=iso(now - timedelta(days=4, hours=6)))
        await log_event(
            store, "curator_run", {"staled": ["release-notes"], "archived": [ARCHIVED["name"]], "merge_proposals": []},
            ts=iso(now - timedelta(days=4, hours=5, minutes=59)),
        )
        await log_event(
            store, "profile_updated",
            {"learned_appended": 3, "facts_considered": 18, "summaries_considered": 2, "memory_chars": len(MEMORY_MD)},
            ts=iso(now - timedelta(days=1, hours=4)),
        )
        await log_event(
            store, "profile_updated",
            {"learned_appended": 0, "facts_considered": 6, "summaries_considered": 1, "memory_chars": len(MEMORY_MD) - 120},
            ts=iso(now - timedelta(days=8, hours=2)),
        )
        # mark every periodic job as just run, so nothing is due even if re-enabled
        for key in ("memory.purge_last_run", "memory.summaries_last_run", "evolution.curator_last_run",
                    "evolution.profile_last_run", "proactivity.poll_last_run"):
            await store.set_meta(key, iso(now))

        graph = await mem.graph()
        cat = await lib.catalog(store)
        print(json.dumps({
            "home": str(home),
            "facts": await mem.count(),
            "graph_links": len(graph["links"]),
            "summaries": len(await mem.episodic.list()),
            "skills": {k: [f"{s['name']} ({s['state']})" for s in v] for k, v in cat.items()},
        }, indent=2))
    finally:
        await app.stop()


def set_theme(home: Path, theme: str) -> None:
    import yaml

    path = home / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("ui", {})["theme"] = theme
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"ui.theme -> {theme}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("SENTIENT_HOME"), help="target SENTIENT_HOME (required)")
    ap.add_argument("--reset", action="store_true", help=f"wipe the home first (only if it contains {MARKER})")
    ap.add_argument("--theme", choices=["dark", "light", "system"], default="dark")
    ap.add_argument("--theme-only", action="store_true", help="only change ui.theme in an already seeded home")
    args = ap.parse_args()
    if not args.home:
        ap.error("--home or SENTIENT_HOME is required")
    home = Path(args.home).expanduser().resolve()
    if args.theme_only:
        set_theme(home, args.theme)
        return
    if home.exists() and any(home.iterdir()):
        if not args.reset:
            ap.error(f"{home} is not empty; pass --reset to re-seed it")
        if not (home / MARKER).exists():
            ap.error(f"refusing to wipe {home}: it was not created by this script")
        shutil.rmtree(home)
    home.mkdir(parents=True, exist_ok=True)
    (home / MARKER).write_text("created by desktop/scripts/seed-memory-skills.py\n", encoding="utf-8")
    os.environ["SENTIENT_HOME"] = str(home)
    asyncio.run(seed(home, args.theme))


if __name__ == "__main__":
    main()
