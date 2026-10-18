"""Deleting one of two similar memories leaves the other's derived data alone (#456).

The cascading delete (#152, #436) finds MEMORY.md and USER.md lines, journal lines, summaries and evidence quotes by
their words. A line that a remaining current memory says as well or better (or that names a remaining memory's id) is
that memory's line and stays; a memory's evidence in the user model goes by its id only.
"""

from __future__ import annotations

import json

from sentient.memory import origins
from sentient.store.db import now_iso

PUNE = "Maya lives in Pune"
MOVED = "Maya moved from Pune to Bengaluru"
HONDA = "Maya drives a red Honda City"
ATHER = "Maya sold her red Honda City and now rides a blue Ather"


async def _ids(app, sql: str, params: tuple = ()) -> list:
    return [r[0] for r in await app.store.fetchall(sql, params)]


async def _fact(app, content: str) -> int:
    """A current memory with exactly this wording. The save path would rightly replace one of a similar pair (the
    newer residence wins), so it is saved under a stand-in wording and then given its own: the pair a user has after
    keeping both, or after a replaced memory's history (#453) is deleted."""
    res = await app.memory.remember(f"Stand-in note number {len(content)}", source="manual", use_llm=False)
    assert res["action"] == "ADD", res
    await app.store.execute("UPDATE facts SET content = ? WHERE id = ?", (content, res["id"]))
    return int(res["id"])


def test_a_line_a_remaining_memory_says_as_well_stays():
    gone = origins.Erased([HONDA], [(7, ATHER)], "Maya")
    assert gone("- Maya drives a red Honda City")
    assert gone("- Maya's car: a red Honda City")
    assert not gone("- Maya sold her red Honda City and now rides a blue Ather")  # the remaining one's line
    assert not gone("- Maya's car (id 7) was a red Honda City")  # names the remaining memory
    assert gone("- Maya's car (id 8) is a red Honda City")  # names a memory that isn't there
    assert not gone("- Maya plays the veena")
    # only a written reference counts, and never a number the deleted memory itself says
    assert gone("- Maya drives a red Honda City, memory 7 of her notes")
    phone = origins.Erased(["Maya's phone has memory 7 GB"], [(7, "Maya likes filter coffee")], "Maya")
    assert phone("- Maya's phone has memory 7 GB") and phone("- Maya's phone (memory 7) has 7 GB")
    # a number only says nothing for the deleted wording that says it
    two = origins.Erased([HONDA, "Maya's flat is number 7"], [(7, ATHER)], "Maya")
    assert not two("- Maya drives a red Honda City (memory 7)")
    # naming a remaining memory never keeps a quote of the deleted one
    assert gone(f'- (memory 7) is new; I used to remember "{HONDA}"')
    # a tie goes to the remaining memory
    pune = origins.Erased([PUNE], [(3, MOVED)], "Maya")
    assert not pune("- Maya moved from Pune to Bengaluru and lives there now")
    assert pune("- Maya lives in Pune")
    # a quoted part is weighed on its own: a journal line about a change goes when either side is deleted
    line = f'- I used to remember "{PUNE}", but now "{MOVED}"'
    assert pune(line) and origins.Erased([MOVED], [(1, PUNE)], "Maya")(line)
    assert not pune(f'- "{MOVED}" keeps coming up, so I will keep it.')
    # with nothing left, it is the plain word match it always was
    assert origins.Erased([HONDA], [], "Maya")("- Maya sold her red Honda City and now rides a blue Ather")
    # an exact USER.md line a remaining memory says word for word is that memory's too
    assert origins.Erased([HONDA, PUNE], [(2, "maya  drives a red honda city")], "Maya").wordings == [PUNE]


async def test_deleting_one_of_two_similar_memories_keeps_the_others_derived_data(app):
    mem, store, ws, users = app.memory, app.store, app.workspace, app.user_model
    tg = await store.create_session(channel="telegram")
    said = await store.add_message(tg, "user", "I drive a red Honda City to work")
    pune, moved = await _fact(app, PUNE), await _fact(app, MOVED)
    first = await mem.remember(HONDA, use_llm=False, session_id=tg)  # the first car memory: saved as said
    honda = int(first["id"])
    assert (await mem.get_fact(honda))["ref"]["message_id"] == said
    later = await store.add_message(tg, "user", "Update: I sold my red Honda City and now ride a blue Ather")
    ather = await _fact(app, ATHER)
    assert {r["state"] for r in await store.fetchall("SELECT state FROM facts")} == {"current"}

    ws.append_learned([PUNE, MOVED, HONDA, ATHER])
    ws.memory.write_text(
        "# Memory\n"
        f"- {PUNE}\n"
        "- Maya moved from Pune to Bengaluru and lives there now\n"
        f"- {HONDA}\n"
        f"- {ATHER}\n"
        f"- Maya's first car (memory {ather}) was a red Honda City\n",
        encoding="utf-8",
    )
    msgs = await store.recent_messages(tg, 10)
    gone_summary = await mem.episodic.add_summary(tg, "Maya drives a red Honda City to the office.", msgs)
    kept_summary = await mem.episodic.add_summary(tg, "Maya sold her red Honda City and now rides a blue Ather.", msgs)
    await store.execute(
        "INSERT INTO dreams(id, started_at, status, trigger, stats, journal_md) VALUES('d1', ?, 'completed', 'manual',"
        " '{}', ?)",
        (now_iso(), f'Tonight I tidied up.\n- I used to remember "{PUNE}", but now "{MOVED}".\n'
                    f'- "{ATHER}" keeps coming up, so I will keep it.\n- "{HONDA}" keeps coming up, so I will keep it.'),
    )

    def fact_ev(fid: int, quote: str) -> dict:
        return {"kind": "fact", "ref": str(fid), "quote": quote}

    sibling_only = await users._insert(
        "Maya switched to an electric scooter", "lifestyle", confidence=0.6, status="active", source="inferred",
        evidence=[fact_ev(ather, ATHER)],
    )
    mixed = await users._insert(
        "Maya gets around on two or four wheels", "lifestyle", confidence=0.6, status="active", source="inferred",
        evidence=[fact_ev(honda, HONDA), fact_ev(ather, ATHER), fact_ev(moved, MOVED), fact_ev(pune, PUNE)],
    )
    deleted_only = await users._insert(
        "Maya commutes by car", "lifestyle", confidence=0.6, status="active", source="inferred",
        evidence=[fact_ev(honda, HONDA)],
    )
    chats = await users._insert(
        "Maya talks about her vehicles", "lifestyle", confidence=0.6, status="active", source="inferred",
        evidence=[{"kind": "message", "ref": said, "quote": "I drive a red Honda City to work"},
                  {"kind": "message", "ref": later, "quote": "Update: I sold my red Honda City and now ride a blue Ather"}],
    )

    assert await mem.erase(honda, notify=True)
    assert await mem.erase(pune, notify=True)

    # the deleted memories' own lines are gone, the remaining ones' stay
    memory_md = ws.memory.read_text(encoding="utf-8")
    assert f"- {PUNE}\n" not in memory_md and f"- {HONDA}\n" not in memory_md
    assert "Bengaluru and lives there now" in memory_md and ATHER in memory_md
    assert f"(memory {ather})" in memory_md
    user_md = ws.user.read_text(encoding="utf-8")
    assert f"- {PUNE}\n" not in user_md and f"- {HONDA}\n" not in user_md
    assert MOVED in user_md and ATHER in user_md
    journal = (await store.fetchone("SELECT journal_md FROM dreams WHERE id = 'd1'"))["journal_md"]
    assert f'"{PUNE}"' not in journal and f'"{HONDA}"' not in journal and f'"{ATHER}" keeps coming up' in journal
    assert journal.count("(a memory you deleted)") == 2
    summaries = await _ids(app, "SELECT id FROM summaries")
    assert gone_summary["id"] not in summaries and kept_summary["id"] in summaries

    insights = {r["id"]: json.loads(r["evidence"]) for r in await store.fetchall("SELECT id, evidence FROM user_insights")}
    assert [e["ref"] for e in insights[sibling_only["id"]]] == [str(ather)]
    assert [e["ref"] for e in insights[mixed["id"]]] == [str(ather), str(moved)]
    assert deleted_only["id"] not in insights
    assert [e["ref"] for e in insights[chats["id"]]] == [later]
    for fid in (moved, ather):
        assert (await mem.get_fact(fid))["state"] == "current"


async def test_a_disputed_memory_leaves_a_similar_memorys_lines(app):
    mem, ws = app.memory, app.workspace
    honda, ather = await _fact(app, HONDA), await _fact(app, ATHER)
    ws.append_learned([HONDA, ATHER])
    ws.memory.write_text(f"# Memory\n- {HONDA}\n- {ATHER}\n", encoding="utf-8")
    await mem.set_state(honda, "disputed")
    for f in (ws.memory, ws.user):
        text = f.read_text(encoding="utf-8")
        assert f"- {HONDA}\n" not in text and ATHER in text
    # superseding the other way round takes the replaced one's lines, not the newer one's
    await mem.set_state(honda, "current")
    ws.memory.write_text(f"# Memory\n- {HONDA}\n- {ATHER}\n", encoding="utf-8")
    await mem.supersede(honda, ather)
    assert ws.memory.read_text(encoding="utf-8") == f"# Memory\n- {HONDA}\n"
