from __future__ import annotations

import pytest

from sentient.app import SentientApp
from tests.conftest import FakeProvider

SKILL_BODY = """# Weekly inbox digest
## When to use
- The user asks for a digest of important unread email.
## Procedure
1. Call `gmail_search` with `is:unread newer_than:7d`.
2. Read the top threads with `gmail_read`.
3. Summarize by sender and urgency.
## Pitfalls
- Skip newsletters.
## Verification
- Every thread in the digest links back to Gmail.
"""


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "evo.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()


async def add_tool_chat(app, tool_calls: int = 3) -> str:
    store = app.store
    sid = await store.create_session()
    await store.add_message(sid, "user", "Make me a digest of my unread email from this week")
    for i in range(tool_calls):
        call = {"id": f"c{i}", "type": "function", "function": {"name": "gmail_search", "arguments": '{"query": "is:unread"}'}}
        await store.add_message(sid, "assistant", None, tool_calls=[call])
        await store.add_message(sid, "tool", '{"threads": 4}', tool_call_id=f"c{i}", name="gmail_search")
    await store.add_message(sid, "assistant", "Here is your digest: 4 important threads.")
    return sid
