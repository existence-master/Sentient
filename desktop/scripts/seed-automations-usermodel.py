"""Seed demo data for the automation and About you screens (no LLM calls, no network, no real keychain).

    .venv/Scripts/python.exe desktop/scripts/seed-automations-usermodel.py <SENTIENT_HOME> [--reset] [--theme dark|light]

Covers: notifications (script alert / failure / recovery, failed run, helper finished, dream, skill fix),
a skill repair proposal with a diff, a script job task, webhooks, the user model and dreams.

Each block first checks that the engine supports it (a service method, or a table/column) and is skipped with a
note otherwise. The desktop screens stay friendly when data is missing. To design screens before an engine piece
exists, use the renderer's dev-only demo mode instead (`#/about?demo=1`, see desktop/src/lib/demo.ts; never active
in packaged builds).

The profile's models point at a black-holed address so `sentient serve` on this home makes no model calls;
start it with SENTIENT_DISABLE_BACKGROUND=1 for screenshots.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sqlite3
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_API_BASE = "http://10.255.255.1:11434"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("home", help="SENTIENT_HOME folder to seed (created if missing)")
    p.add_argument("--reset", action="store_true", help="delete sentient.db in that home first")
    p.add_argument("--theme", choices=["dark", "light", "system"], default=None)
    return p.parse_args()


ARGS = _args()
HOME = Path(ARGS.home).expanduser().resolve()
os.environ["SENTIENT_HOME"] = str(HOME)
sys.path.insert(0, str(REPO_ROOT))

from sentient.app import SentientApp  # noqa: E402
from sentient.config.loader import load_config, save_config  # noqa: E402
from sentient.config.schema import ProviderConfig  # noqa: E402

try:  # noqa: SIM105
    from sentient.llm.provider import StreamChunk  # noqa: E402
except Exception:  # pragma: no cover - older engines
    StreamChunk = None  # type: ignore[assignment]

NOW = datetime.now(UTC).replace(microsecond=0)
REPORT: dict[str, str] = {}


def ago(**kw: float) -> str:
    return (NOW - timedelta(**kw)).isoformat()


class OfflineProvider:
    """Stands in for the LLM. Seeding never needs a real answer."""

    def model_for(self, role: str) -> str:
        return f"offline/{role}"

    async def stream(self, role, messages, tools=None, *, model=None) -> AsyncIterator[Any]:
        if StreamChunk is not None:
            yield StreamChunk(text="ok", model="offline")
            yield StreamChunk(done=True, usage={"prompt_tokens": 0, "completion_tokens": 0}, model="offline")

    async def complete_text(self, role, messages, *, model=None) -> str:
        return "Demo"

    async def complete_json(self, role, messages, *, model=None):
        return {}

    async def embed(self, texts, *, model=None):
        return [[1.0] + [0.0] * 15 for _ in texts]


def configure():
    cfg = load_config()
    cfg.assistant.user_name = cfg.assistant.user_name or "Maya"
    cfg.assistant.timezone = "Asia/Kolkata"
    cfg.assistant.location = cfg.assistant.location or "Bengaluru, India"
    cfg.assistant.onboarding_complete = True
    cfg.models.roles.primary = "ollama_chat/qwen3:8b"
    cfg.models.roles.fast = "ollama_chat/qwen3:4b"
    cfg.models.roles.embedding = "ollama/seeded-demo-no-embeddings"
    cfg.models.fallbacks = {}
    cfg.models.request_timeout_s = 3600
    for prefix in ("ollama", "ollama_chat"):
        cfg.models.providers[prefix] = ProviderConfig(api_base=OFFLINE_API_BASE)
    cfg.memory.extract_after_turn = False
    cfg.chat.auto_title = False
    cfg.proactivity.enabled = False
    cfg.evolution.review_enabled = False
    cfg.evolution.curator_enabled = False
    if ARGS.theme:
        cfg.ui.theme = ARGS.theme
    save_config(cfg)
    return cfg


def columns(table: str) -> set[str]:
    db = HOME / "sentient.db"
    if not db.exists():
        return set()
    con = sqlite3.connect(db)
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


async def call(fn, *args, **kwargs):
    """Call a sync or async engine method, dropping keyword arguments it does not accept."""
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    res = fn(*args, **kwargs)
    return await res if inspect.isawaitable(res) else res


# ----------------------------------------------------------------------------- blocks
async def seed_notifications(app: SentientApp, script_task_id: str | None) -> None:
    n = app.notifications
    task_ref = {"task_id": script_task_id} if script_task_id else {}
    await n.create("task", "The Aura X2 is now **₹24,490**, below your ₹25,000 target.",
                   title="Tell me when the Aura X2 headphones drop below ₹25,000",
                   payload={**task_ref, "event": "script_alert", "result": {"alert": True, "price": 24490}})
    await n.create("task", "The price check couldn't reach GadgetBay, so I'll try again at the next check.",
                   title="A watcher couldn't check", payload={**task_ref, "event": "script_failed"})
    await n.create("task", "The price check is working again.", title="A watcher is back on track",
                   payload={**task_ref, "event": "script_recovered"})
    await n.create("task", "**Morning inbox digest** stopped: Gmail rejected the saved sign-in.",
                   title="A task run failed", payload={"task_id": "demo-daily", "run_id": "demo-run", "event": "run_failed"})
    await n.create("info", "I compared four portfolio site builders. **Folio** and **Canvasly** suit a designer best: clean templates and easy case studies.",
                   title="Your helper finished",
                   payload={"event": "subagent_completed", "subagent_id": "sa_demo", "goal": "Compare four portfolio site builders for a designer"})
    await n.create("info", "I reviewed 146 memories, merged 6 duplicates and settled 2 contradictions.",
                   title="I tidied up my memory overnight",
                   payload={"event": "dream_completed", "dream_id": "demo-dream",
                            "stats": {"facts_reviewed": 146, "merged": 6, "contradictions_resolved": 2}})
    await n.create("skill", "Filing an invoice failed because a Drive folder was missing. I drafted a fix to **invoice-filing**.",
                   title="A fix for invoice-filing",
                   payload={"skill": "invoice-filing", "action": "patch", "origin": {"repair": True, "task_id": "demo-invoices"}})
    REPORT["notifications"] = "7 automation, helper, dream and skill notifications"


CURRENT = """# Invoice filing

## Procedure
1. Search Gmail for the invoice email and download the PDF attachment.
2. Upload it to Finance/Invoices in Google Drive.
3. Add vendor, amount and due date to the Expenses sheet.

## Pitfalls
- Some vendors send a link instead of an attachment.
"""
FIXED = """# Invoice filing

## Procedure
1. Search Gmail for the invoice email and download the PDF attachment.
2. Upload it to Finance/Invoices/<year> in Google Drive, creating the year folder if it is missing.
3. Add vendor, amount and due date to the Expenses sheet.

## Pitfalls
- Some vendors send a link instead of an attachment.
- Drive answers "File not found" when the year folder does not exist yet. Create it first, then upload.
"""


async def seed_skill_repair(app: SentientApp) -> None:
    lib = getattr(app, "skills", None)
    if lib is None or not hasattr(lib, "write"):
        REPORT["skill repair"] = "skipped (skills library not available)"
        return
    from sentient.evolution.log import log_event

    desc = "Save vendor invoices from Gmail to Drive and log them in the expenses sheet."
    tools = ["gmail", "gdrive", "gsheets"]
    lib.write("invoice-filing", desc, CURRENT, author="assistant", tags=["finance"], requires_tools=tools, created_by_review=True)
    if hasattr(lib, "reload"):
        lib.reload({p.id for p in app.registry.plugins()})
    if hasattr(lib, "sync_stats"):
        await lib.sync_stats(app.store)
    lib.write("invoice-filing", desc, FIXED, author="assistant", tags=["finance"], requires_tools=tools, pending=True, created_by_review=True)
    if hasattr(lib, "record_patch"):
        await lib.record_patch("invoice-filing", "pending_review", app.store)
    reason = ("The last run of “File vendor invoices to Drive” failed: Drive said “File not found” because the 2026 folder "
              "did not exist. This fix creates the year folder first.")
    detail = {"name": "invoice-filing", "pending": True, "origin": "repair", "failure": "run_failed", "reason": reason,
              "task_id": "demo-invoices"}
    await log_event(app.store, "skill_repair_proposed", detail, ts=ago(hours=3))
    REPORT["skill repair"] = "invoice-filing (pending fix with diff)"


SCRIPT = '''"""Watch the price of the Aura X2 headphones and alert below the target."""
from sentient_tools import tools, result

TARGET = 25_000
page = tools.web_fetch(url="https://gadgetbay.example/p/aura-x2")
text = page.get("text", "")
price = None
for line in text.splitlines():
    if "₹" in line:
        digits = "".join(ch for ch in line if ch.isdigit())
        if digits:
            price = int(digits[:6])
            break
if price is None:
    raise RuntimeError("Couldn't find the price on the page")
print(f"Current price: ₹{price:,}")
result({"alert": price < TARGET, "price": price,
        "message": f"The Aura X2 is now ₹{price:,}, below your ₹{TARGET:,} target."})
'''


async def seed_script_job(app: SentientApp) -> str | None:
    if "script" not in columns("tasks"):
        REPORT["script job"] = "skipped (tasks table has no script column yet)"
        return None
    repo = app.tasks.repo
    task_id = "demo-script-watch"
    try:
        await repo.delete_task(task_id)
    except Exception:
        pass
    fields = {
        "id": task_id, "name": "Tell me when the Aura X2 headphones drop below ₹25,000",
        "description": "Check the GadgetBay price of the Aura X2 headphones every hour and tell me when it drops below ₹25,000.",
        "status": "active", "priority": 1, "task_type": "script", "assignee": "ai", "enabled": True,
        "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "Asia/Kolkata"},
        "plan": [], "chat_history": [], "clarifying_questions": [],
        "script": {"code": SCRIPT, "condition": "alert", "then": "notify",
                   "last_result": {"alert": False, "price": 26490}, "last_run_at": ago(minutes=40), "last_error": None},
        "original_context": {"source": "chat"}, "original_prompt": "Tell me when the Aura X2 headphones drop below ₹25,000",
        "source": "chat", "next_execution_at": (NOW + timedelta(minutes=20)).isoformat(), "last_execution_at": ago(minutes=40),
        "created_at": ago(days=6), "updated_at": ago(minutes=40),
    }
    try:
        await repo.insert_task(fields)
        REPORT["script job"] = task_id
        return task_id
    except Exception as exc:  # engine shape changed
        REPORT["script job"] = f"skipped ({exc})"
        return None


async def seed_hooks(app: SentientApp) -> None:
    store = getattr(getattr(app, "integrations", None), "hooks", None)
    for owner, attr in ((store, "create"), (getattr(app, "integrations", None), "create_hook")):
        fn = getattr(owner, attr, None) if owner is not None else None
        if fn:
            try:
                for existing in await store.list() if store is not None and hasattr(store, "list") else []:
                    if existing.get("name") in {"Shopify orders", "Home Assistant", "Zapier: new form entry"} and hasattr(store, "delete"):
                        await store.delete(existing["id"])
            except Exception:
                pass
            made = []
            for name in ("Shopify orders", "Home Assistant", "Zapier: new form entry"):
                try:
                    made.append((await call(fn, name))["id"])
                except Exception as exc:
                    REPORT["webhooks"] = f"partly skipped ({exc})"
                    break
            REPORT.setdefault("webhooks", f"{len(made)} created")
            return
    REPORT["webhooks"] = "skipped (engine has no create_hook yet)"


INSIGHTS = [
    ("preferences", "Prefers short, direct answers with the key point first."),
    ("communication", "Writes to clients formally but keeps studio chat casual and brief."),
    ("goals", "Wants to finish her portfolio refresh before the end of October."),
    ("routines", "Does deep work early in the morning and keeps client calls after lunch."),
    ("values", "Cares about privacy and prefers tools that keep data on her own computer."),
    ("work_style", "Likes plans broken into small steps she can approve."),
    ("relationships", "Calls her sister Anika most Sundays."),
]


async def seed_user_model(app: SentientApp) -> None:
    um = getattr(app, "user_model", None)
    add = getattr(um, "add_insight", None) or getattr(um, "create_insight", None)
    if not add:
        REPORT["user model"] = "skipped (engine has no add_insight yet)"
        return
    count = 0
    made: list[str] = []
    for dimension, statement in INSIGHTS:
        try:
            ins = await call(add, statement=statement, dimension=dimension, source="user")
            made.append(ins["id"] if isinstance(ins, dict) else None)
            count += 1
        except Exception as exc:
            REPORT["user model"] = f"partly skipped ({exc})"
            break
    # Make most of them look learned (inferred, with evidence and varied confidence), as the engine would.
    if {"confidence", "evidence", "source"} <= columns("user_insights"):
        shape = [
            (0.92, "confirmed", "inferred", [("feedback", "You said: “Just give me the answer, skip the preamble.”", 3)]),
            (0.78, "active", "inferred", [("message", "“Hi Priya, thank you for the thoughtful notes on the deck.”", 2), ("message", "“yo, sending the new mockups tonight”", 1)]),
            (1.0, "confirmed", "user", []),
            (0.81, "active", "inferred", [("summary", "Most focused sessions start between 6 and 7 AM", 4)]),
            (0.9, "confirmed", "inferred", [("message", "“I don’t want my notes going to someone else’s server.”", 20)]),
            (0.64, "active", "inferred", [("summary", "Asked for step-by-step plans in 6 recent tasks", 3)]),
            (0.42, "disputed", "inferred", [("message", "“Remind me to call Anika on Sunday evening”", 7)]),
        ]
        for iid, (conf, status, source, ev) in zip(made, shape):
            if not iid:
                continue
            evidence = [{"kind": k, "ref": f"{iid}-{i}", "quote": q, "at": ago(days=d)} for i, (k, q, d) in enumerate(ev)]
            await app.store.execute(
                "UPDATE user_insights SET confidence = ?, status = ?, source = ?, evidence = ? WHERE id = ?",
                (conf, status, source, json.dumps(evidence, ensure_ascii=False), iid),
            )
        if "question" in columns("user_questions") and len(made) >= 7:
            questions = [
                ("demo-q1", "Should I keep suggestions quiet until 10 AM so your mornings stay free for deep work?", made[3]),
                ("demo-q2", "Is calling Anika on Sundays still a routine I should plan around?", made[6]),
            ]
            for qid, text, iid in questions:
                await app.store.execute("DELETE FROM user_questions WHERE id = ?", (qid,))
                await app.store.execute(
                    "INSERT INTO user_questions(id, question, insight_id, status, created_at) VALUES(?,?,?,?,?)",
                    (qid, text, iid, "open", ago(hours=5)),
                )
    if hasattr(app.store, "set_meta"):
        await app.store.set_meta(
            "user_model.summary",
            "Maya is a product designer at Northwind Studio in Bengaluru, and her portfolio refresh is front of mind. "
            "She does her best thinking early in the morning, likes answers short and plans in small steps she can approve, "
            "and protects both her data and her focus. Outside work she stays close to her sister Anika.",
        )
    REPORT.setdefault("user model", f"{count} insights, 2 questions, summary")


DREAMS = [
    ("demo-dream-1", 9, "schedule", {"facts_reviewed": 146, "merged": 6, "contradictions_resolved": 2, "promoted": 3, "expired": 5, "insights_updated": 4},
     "Tonight I went through **146 memories** from the past week.\n\n"
     "I had written down your Thursday client review three different ways, so I merged them into one. "
     "You told me in April that you live in Chennai, but everything since says **Bengaluru**, so I kept Bengaluru and let the old note go.\n\n"
     "Your morning runs have become a habit, so I now keep “training for a 10 km” in mind. "
     "I also let five short-term reminders fade now that their dates have passed.", None),
    ("demo-dream-2", 33, "manual", {"facts_reviewed": 38, "merged": 1, "promoted": 1, "expired": 2, "insights_updated": 1},
     "You asked me to tidy up after importing your resume. I found **38 new memories**, merged a duplicate about your time at "
     "Northwind Studio, and noted that you now lead the design systems work.", None),
    ("demo-dream-3", 57, "schedule", {"facts_reviewed": 121, "merged": 3, "contradictions_resolved": 1, "expired": 7},
     "A quiet night. I merged three notes about the Goa trip, settled when your sister’s birthday is (the 14th, not the 4th), "
     "and cleared seven reminders that were done.", None),
    ("demo-dream-4", 81, "schedule", {}, "", "The memory model was not reachable, so I will try again tonight."),
]


async def seed_dreams(app: SentientApp) -> None:
    cols = columns("dreams")
    if not {"id", "started_at", "status", "stats", "journal_md"} <= cols:
        REPORT["dreams"] = "skipped (engine has no dreams table yet)"
        return
    for did, hours, trigger, stats, journal, error in DREAMS:
        await app.store.execute("DELETE FROM dreams WHERE id = ?", (did,))
        await app.store.execute(
            "INSERT INTO dreams(id, started_at, finished_at, status, trigger, stats, journal_md, error) VALUES(?,?,?,?,?,?,?,?)",
            (did, ago(hours=hours), ago(hours=hours, minutes=-4), "error" if error else "completed", trigger,
             json.dumps(stats), journal, error),
        )
    REPORT["dreams"] = f"{len(DREAMS)} dreams"


async def main() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    if ARGS.reset:
        for suffix in ("", "-wal", "-shm"):
            p = HOME / f"sentient.db{suffix}"
            if p.exists():
                p.unlink()
    cfg = configure()
    app = SentientApp(cfg, llm=OfflineProvider(), db_path=HOME / "sentient.db", enable_background=False)
    await app.start()
    try:
        script_task = await seed_script_job(app)
        await seed_skill_repair(app)
        await seed_hooks(app)
        await seed_user_model(app)
        await seed_dreams(app)
        await seed_notifications(app, script_task)
    finally:
        await app.stop()
    print(json.dumps({"home": str(HOME), **REPORT}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
