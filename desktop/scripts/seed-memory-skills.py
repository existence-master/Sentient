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
STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "is", "are", "was", "be", "has",
    "have", "his", "her", "he", "she", "it", "its", "as", "by", "from", "that", "this", "maya", "maya's",
}


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
    ("Maya is a product designer at Northwind Studio, a small design studio", ["Work & Learning", "Personal Identity"], "onboarding", "long-term", None, 41),
    ("Maya lives in Bengaluru, India", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Maya's timezone is Asia/Kolkata", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Maya prefers short, concise answers without filler", ["Personal Identity"], "onboarding", "long-term", None, 41),
    ("Maya values privacy and likes that her assistant keeps her data on her own computer", ["Personal Identity", "Work & Learning"], "onboarding", "long-term", None, 41),
    ("Maya is a morning person and does her best design work before 11am", ["Interests & Lifestyle", "Work & Learning"], "conversation", "long-term", None, 38),
    ("Maya does most of her design work in Figma", ["Work & Learning"], "conversation", "long-term", None, 37),
    ("Maya looks after the design system at Northwind Studio", ["Work & Learning"], "conversation", "long-term", None, 36),
    ("Maya leads a design critique every Wednesday afternoon", ["Work & Learning"], "conversation", "long-term", None, 35),
    ("Maya is planning a refresh of her design portfolio", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 33),
    ("Maya wants to finish her portfolio refresh by the end of November", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 30),
    ("Maya studied communication design", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Maya previously worked as a visual designer at a small branding agency", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Maya has experience with user research, prototyping and accessibility reviews", ["Work & Learning"], "file:resume.pdf", "long-term", None, 29),
    ("Maya gave a talk on accessible colour palettes at a Bengaluru design meetup", ["Work & Learning", "Relationships & Social Life"], "file:resume.pdf", "long-term", None, 29),
    ("Maya is reading a book about how everyday objects are designed", ["Work & Learning", "Interests & Lifestyle"], "conversation", "long-term", None, 26),
    ("Maya is learning the veena on weekends", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 24),
    ("Maya enjoys pottery classes on Thursday evenings", ["Interests & Lifestyle"], "conversation", "long-term", None, 23),
    ("Maya likes masala chai and drinks two cups a day", ["Interests & Lifestyle", "Health & Wellbeing"], "conversation", "long-term", None, 22),
    ("Maya goes on sunrise hikes near Bengaluru after the monsoon", ["Interests & Lifestyle", "Health & Wellbeing"], "conversation", "long-term", None, 21),
    ("Maya listens to instrumental music while designing", ["Interests & Lifestyle"], "conversation", "long-term", None, 20),
    ("Maya is a fan of cosy mystery novels", ["Interests & Lifestyle"], "manual", "long-term", None, 19),
    ("Maya sketches street scenes in a pocket sketchbook", ["Interests & Lifestyle"], "manual", "long-term", None, 18),
    ("Maya runs 5 km three times a week", ["Health & Wellbeing", "Interests & Lifestyle"], "conversation", "long-term", None, 18),
    ("Maya is trying to keep screens off after 10pm", ["Health & Wellbeing", "Goals & Challenges"], "conversation", "long-term", None, 17),
    ("Maya is vegetarian", ["Health & Wellbeing"], "manual", "long-term", None, 16),
    ("Maya does ten minutes of stretching after waking up", ["Health & Wellbeing"], "conversation", "long-term", None, 15),
    ("Maya has an eye checkup due in October", ["Health & Wellbeing"], "conversation", "long-term", None, 14),
    ("Maya's sister Anika is a school teacher in Mysuru", ["Relationships & Social Life"], "conversation", "long-term", None, 14),
    ("Maya's parents live in Mangaluru", ["Relationships & Social Life"], "conversation", "long-term", None, 13),
    ("Maya calls her parents every Sunday evening", ["Relationships & Social Life", "Interests & Lifestyle"], "conversation", "long-term", None, 13),
    ("Maya's friend Rohan is her running partner on weekend mornings", ["Relationships & Social Life", "Health & Wellbeing"], "conversation", "long-term", None, 12),
    ("Maya's colleague Kavya is a developer who builds the components Maya designs", ["Relationships & Social Life", "Work & Learning"], "conversation", "long-term", None, 12),
    ("Maya's mother's birthday is on 2 November", ["Relationships & Social Life"], "manual", "long-term", None, 11),
    ("Maya mentors two design students from a local college every month", ["Relationships & Social Life", "Work & Learning"], "conversation", "long-term", None, 10),
    ("Maya is saving for a two-week trip to Japan next spring", ["Financial", "Goals & Challenges"], "conversation", "long-term", None, 10),
    ("Maya tracks expenses in a monthly spreadsheet", ["Financial"], "conversation", "long-term", None, 9),
    ("Maya puts a fixed amount into a recurring deposit every month", ["Financial"], "conversation", "long-term", None, 9),
    ("Maya's monthly budget for eating out is 4000 rupees", ["Financial", "Interests & Lifestyle"], "conversation", "long-term", None, 8),
    ("Maya wants her design newsletter to reach 500 readers by the end of the year", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 8),
    ("Maya finds it hard to stop tweaking designs late at night", ["Goals & Challenges", "Health & Wellbeing"], "conversation", "long-term", None, 7),
    ("Maya wants to write a short case study for every client project", ["Goals & Challenges", "Work & Learning"], "conversation", "long-term", None, 7),
    ("Maya is preparing a rate card for freelance illustration work", ["Goals & Challenges", "Financial"], "conversation", "long-term", None, 6),
    ("Maya wants to run a 10 km race next year", ["Goals & Challenges", "Health & Wellbeing"], "manual", "long-term", None, 6),
    ("Maya prefers meetings after 2pm", ["Work & Learning", "Personal Identity"], "conversation", "long-term", None, 5),
    ("Maya uses Gmail and Google Calendar for work", ["Work & Learning", "Miscellaneous"], "conversation", "long-term", None, 5),
    ("Maya keeps meeting notes in Google Docs", ["Work & Learning", "Miscellaneous"], "conversation", "long-term", None, 4),
    ("Maya has a ginger cat named Biscuit", ["Miscellaneous", "Interests & Lifestyle"], "conversation", "long-term", None, 4),
    ("Maya's favourite place to eat in Bengaluru is a small dosa cafe near her flat", ["Interests & Lifestyle", "Miscellaneous"], "conversation", "long-term", None, 3),
    ("Maya rides a blue scooter to the studio", ["Miscellaneous"], "manual", "long-term", None, 3),
    ("Maya is moving the Lumen Health app redesign onto the new design system", ["Work & Learning"], "conversation", "long-term", None, 2),
    ("Maya chose a warm serif font and soft pastel colours for the Paperkite website", ["Work & Learning"], "conversation", "long-term", None, 2),
    ("Maya is reviewing Kavya's build of the new button components", ["Work & Learning", "Relationships & Social Life"], "conversation", "long-term", None, 1),
    ("Maya wants weekly reviews compiled every Friday evening", ["Work & Learning", "Goals & Challenges"], "conversation", "long-term", None, 1),
    ("Maya prefers dark mode in every app", ["Personal Identity", "Miscellaneous"], "conversation", "long-term", None, 0),
    # short-term (a few expire soon)
    ("Maya has a call with the Lumen Health team tomorrow at 3pm about the onboarding screens", ["Work & Learning", "Relationships & Social Life"], "conversation", "short-term", "1 day", 0),
    ("Maya is visiting Anika in Mysuru this weekend", ["Interests & Lifestyle"], "conversation", "short-term", "3 days", 0),
    ("Maya needs to renew her scooter insurance this week", ["Financial", "Miscellaneous"], "conversation", "short-term", "5 days", 1),
    ("Maya has a dentist appointment this afternoon", ["Health & Wellbeing"], "conversation", "short-term", "6 hours", 0),
    ("Maya is waiting for Paperkite to approve her quote for the website", ["Goals & Challenges", "Financial"], "conversation", "short-term", "2 weeks", 2),
]

UPDATED = {
    "Maya runs 5 km three times a week": "Maya runs 5 km on Saturday and Sunday mornings",
    "Maya's monthly budget for eating out is 4000 rupees": "Maya's monthly budget for eating out is 3000 rupees",
}

SUMMARIES = [
    ("Planning the portfolio refresh", 1,
     "I helped Maya turn her portfolio refresh into a checklist: pick four case studies, rewrite the about page, "
     "and ask Rohan and Anika for feedback before the end of November. She wants each case study to fit on one screen."),
    ("Weekly review, week 40", 3,
     "Maya and I went through her week. She shipped the Lumen Health onboarding screens, missed one weekend run because "
     "of rain, and asked me to remind her to keep screens off after 10pm. We moved the Paperkite follow-up to Monday."),
    ("Glasses: reading a recipe card", 6,
     "Maya used her glasses to read her mother's handwritten recipe card for bisi bele bath. I read out each step, "
     "scaled it for six people, converted cups to grams and turned the ingredients into a shopping list."),
    ("Sunrise hike at Nandi Hills", 11,
     "Maya asked for an easy sunrise hike near Bengaluru. I suggested Nandi Hills, checked the weather for Saturday, "
     "and listed what to pack. She invited Rohan and two friends from her pottery class."),
    ("Budget and Japan trip savings", 16,
     "I summarised Maya's monthly spending sheet and helped plan savings for her Japan trip: flights, a rail pass and "
     "a 12-day budget. She wants to keep eating out under 3000 rupees a month."),
]

SOUL = """# Soul

You are Sentient, a personal assistant that lives on Maya's own computer.

## How you behave
- Warm, direct, and brief. You talk like a capable friend.
- You act. When a request can be done with your tools, do it, then report the result in one or two sentences.
- You remember. Facts about Maya that come up are worth saving.
- You respect approvals. Anything that sends, deletes, or spends waits for a yes.

## Voice
Plain language, short sentences, no filler.
"""

USER = """# About Maya

- Product designer at **Northwind Studio**, a small design studio in Bengaluru.
- Lives in Bengaluru, India (Asia/Kolkata). Morning person; best design work before 11am.
- Prefers short, concise answers. Meetings after 2pm, please.

## Work
- Designs in Figma and looks after the studio's design system. Leads the design critique on Wednesdays.
- Current clients: Lumen Health (app redesign) and Paperkite (new website). Portfolio refresh due in November.

## Learned

- Maya is learning the veena on weekends
- Maya's sister Anika is a school teacher in Mysuru
- Maya wants weekly reviews compiled every Friday evening
"""

MEMORY_MD = """# Long-term memory

## Ongoing projects
- **Portfolio refresh** (end of November): four case studies, a new about page, feedback from Rohan and Anika.
- **Lumen Health app**: onboarding screens shipped; moving the rest onto the new design system.
- **Paperkite website**: homepage designs sent; waiting on approval of the quote.

## Preferences
- Short answers, dark mode, meetings after 2pm.
- Vegetarian. Keep eating out under 3000 rupees a month; saving for Japan next spring.

## Commitments
- Weekly review every Friday evening.
- Call parents on Sunday evenings.
"""

SKILLS = [
    dict(name="weekly-review", author="user", tags=["productivity", "planning"], requires_tools=["gcalendar", "gmail"],
         description="Compile Maya's weekly review from calendar, email and task results every Friday.",
         uses=14, views=22, patches=2, last_used_days=1,
         body="""# Weekly review

## When to use
Every Friday evening, or when Maya asks "how did my week go?".

## Procedure
1. Pull this week's calendar events and count client meetings vs. design time.
2. Search Gmail for client threads Maya replied to and anything still waiting on her.
3. List completed and failed tasks from the task log.
4. Write three sections: **Shipped**, **Slipped**, **Next week**.
5. End with one question about priorities for Monday.

## Pitfalls
- Skip newsletters and automated notifications.
- Keep it under 250 words.

## Verification
- Every item links back to an email, event or task.
"""),
    dict(name="client-feedback-digest", author="assistant", tags=["clients", "email"], requires_tools=["gmail"],
         description="Collect client feedback from email into one list of changes per screen.",
         uses=6, views=9, patches=1, last_used_days=3, created_by_review=True,
         body="""# Client feedback digest

## When to use
A client has sent feedback on designs across several emails, or Maya asks "what did they want changed?".

## Procedure
1. Search Gmail for the client's messages since the last design review.
2. Pull out every requested change and group it by screen or page.
3. Mark each change as clear, unclear or conflicting with earlier feedback.
4. Draft a short reply that confirms the list and asks about the unclear items, and wait for approval before sending.

## Pitfalls
- Don't treat a question as a change request.
- Never reply to a client without approval.

## Verification
- Every change cites the email it came from.
"""),
    dict(name="hike-planner", author="assistant", tags=["travel", "weather"], requires_tools=["weather", "maps"],
         description="Plan an easy day hike near Bengaluru with weather, travel time and a packing list.",
         uses=3, views=4, patches=0, last_used_days=11, created_by_review=True,
         body="""# Hike planner

## When to use
Maya wants to go hiking this weekend.

## Procedure
1. Check the weather for Saturday and Sunday at 2-3 candidate trails.
2. Estimate travel time from Bengaluru with maps, aiming to arrive before sunrise.
3. Pick the best option and list what to pack, including vegetarian snacks.

## Pitfalls
- Avoid trails with heavy rain alerts.

## Verification
- Forecast fetched within the last 12 hours.
"""),
    dict(name="critique-prep", author="user", tags=["meetings", "design"], requires_tools=["gcalendar"],
         description="Prepare a short agenda before the Wednesday design critique.",
         uses=9, views=12, patches=1, last_used_days=2,
         body="""# Critique prep

## When to use
On Wednesday morning, before the design critique on the calendar.

## Procedure
1. Read the event description and the list of attendees.
2. Search email for designs people asked to have reviewed this week.
3. Write an agenda with one slot per design, the question each designer wants answered, and the time allowed.

## Pitfalls
- Keep feedback about the work, never about the person.

## Verification
- The agenda fits on one screen.
"""),
    dict(name="expense-summary", author="community", tags=["finance"], requires_tools=[],
         description="Summarise a monthly expense spreadsheet into categories and trends.",
         uses=4, views=5, patches=0, last_used_days=9,
         body="""# Expense summary

## When to use
Maya shares or mentions her monthly spending sheet.

## Procedure
1. Group rows by category.
2. Compare with the previous month and flag changes above 20%.
3. Produce a short table and two suggestions, including progress toward the Japan trip savings.

## Pitfalls
- Never move money or change the sheet.

## Verification
- Totals match the sheet.
"""),
    dict(name="case-study-draft", author="assistant", tags=["writing", "portfolio"], requires_tools=[],
         description="Turn a finished client project into a short portfolio case study.",
         uses=1, views=2, patches=0, last_used_days=21, created_by_review=True, stale=True,
         body="""# Case study draft

## When to use
A client project has wrapped up and Maya wants it in her portfolio.

## Procedure
1. Ask which screens and results she wants to show.
2. Write four short parts: the problem, what she tried, what shipped, and what changed for the client.
3. Suggest three images to go with it.

## Pitfalls
- Leave out anything the client asked to keep private.

## Verification
- The draft fits on one screen and names no private client details.
"""),
]

PENDING_NEW = dict(
    name="recipe-shopping-list", tags=["cooking", "groceries"], requires_tools=[],
    description="Turn a recipe into a scaled shopping list grouped by shop section.",
    reason="While reading your mother's recipe card you asked me to scale it for six people and turn it into a "
    "shopping list. I used 5 tool calls to read the card, convert units and group the items; this is likely to repeat.",
    body="""# Recipe to shopping list

## When to use
Maya shares a recipe (a photo, a card read through her glasses, or a link) and wants to cook it.

## Procedure
1. Read the recipe and list every ingredient with its quantity.
2. Scale the quantities to the number of people she names.
3. Convert cups and spoons to grams where it helps.
4. Leave out items she has said are already at home.
5. Group the list by shop section: vegetables, dairy, grains and spices.

## Pitfalls
- Maya is vegetarian; flag any ingredient that is not.
- Don't order groceries without approval.

## Verification
- Every item on the list appears in the recipe.
""")

PENDING_UPDATE_BODY = """# Weekly review

## When to use
Every Friday evening, or when Maya asks "how did my week go?".

## Procedure
1. Pull this week's calendar events and count client meetings vs. design time.
2. Search Gmail for client threads Maya replied to and anything still waiting on her.
3. List completed and failed tasks from the task log.
4. Check health goals: weekend runs logged and nights she kept screens off after 10pm.
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
    name="inspiration-roundup", tags=["design"], requires_tools=["slack"],
    description="Post a weekly roundup of saved design inspiration to the studio's Slack.",
    body="""# Inspiration roundup

## When to use
Each Monday at 10am.

## Procedure
1. Collect the links and screenshots Maya saved last week.
2. Pick the five best and write one line about each.
3. Post to the studio channel after approval.

## Pitfalls
- Skip weeks with fewer than three saves.

## Verification
- Posted once per week.
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
    cfg.assistant.user_name = "Maya"
    cfg.assistant.timezone = "Asia/Kolkata"
    cfg.assistant.location = "Bengaluru, India"
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
            "Compile Maya's weekly review from calendar, email, tasks and health goals every Friday.",
            PENDING_UPDATE_BODY, author="user", tags=[*wr["tags"], "health"], requires_tools=wr["requires_tools"],
            pending=True, created_by_review=True,
        )
        await lib.record_patch("weekly-review", "pending_review", store)
        reason = (
            "During the last two weekly reviews you asked me to add how many runs you logged and how often you kept "
            "screens off after 10pm. Adding a Health section makes that automatic."
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
            store, "curator_run", {"staled": ["case-study-draft"], "archived": [ARCHIVED["name"]], "merge_proposals": []},
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
