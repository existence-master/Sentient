"""Interrupted API approval publication must leave neither broker requests nor pending notifications."""

from __future__ import annotations

import asyncio

import pytest

from sentient.app import SentientApp
from sentient.gateway.routes import openai_api
from tests.conftest import FakeProvider, tool_call


@pytest.mark.parametrize("outcome", ["stop", "answered-stop", "publication-error"])
async def test_committed_approval_is_cleaned_when_publication_does_not_return(config, isolated_home, monkeypatch, outcome):
    config.tools.approvals.rules = {"file_write": "ask"}
    llm = FakeProvider(replies=[
        [tool_call("file_write", name="publication-probe.txt", content="must not be written")],
        "Finished.",
    ])
    core = await SentientApp(
        config, llm=llm, db_path=isolated_home / "approval-publication.db", enable_background=False,
    ).start()
    committed, release = asyncio.Event(), asyncio.Event()
    execute = core.store.execute
    run_tool = core.agent._run_tool
    tool_calls = []

    async def observe_tool(call, *args, **kwargs):
        tool_calls.append(call.name)
        return await run_tool(call, *args, **kwargs)

    async def pause_pruning(sql, params=()):
        if sql.startswith("DELETE FROM notifications WHERE id NOT IN"):
            # NotificationService.create has committed its INSERT, but has not returned the note to the API bridge.
            committed.set()
            await release.wait()
            if outcome == "publication-error":
                raise RuntimeError("The scripted notification publication failed after its insert.")
        return await execute(sql, params)

    monkeypatch.setattr(core.store, "execute", pause_pruning)
    monkeypatch.setattr(core.agent, "_run_tool", observe_tool)
    try:
        sid = await openai_api._session(core, "Publication probe")
        started = openai_api._start(core, sid, "Save the publication probe note.")
        assert started is not None
        out, _ = started
        await asyncio.wait_for(committed.wait(), timeout=5)
        [note] = [note for note in await core.notifications.list() if note["kind"] == "approval"]
        approval_id = note["payload"]["approval_id"]
        assert approval_id in core.approvals.pending_ids()
        assert not note["payload"].get("status")

        if outcome == "answered-stop":
            assert core.approvals.resolve(approval_id, "allow", via="telegram")
        if outcome in {"stop", "answered-stop"}:
            stopped = await asyncio.wait_for(core.stop_all(), timeout=5)
            assert stopped["cancelled"] >= 1 and not release.is_set()
        else:
            release.set()

        events = []
        while (event := await asyncio.wait_for(out.get(), timeout=5)) is not None:
            events.append(event)
        if outcome in {"stop", "answered-stop"}:
            assert any(isinstance(event, asyncio.CancelledError) for event in events)
        else:
            assert any(isinstance(event, RuntimeError) for event in events)
        assert (await core.notifications.get(note["id"]))["payload"].get("status") == "expired"
        assert core.approvals.pending_ids() == []
        assert core.approvals._pending == core.approvals._choices == core.approvals._via == {}
        assert not core.approvals.resolve(approval_id, "allow")
        assert await core.approvals.grants.list() == []
        assert core.approvals._session_allow == {}
        assert tool_calls == []
        assert not (isolated_home / "files" / "publication-probe.txt").exists()
        assert len([call for call in llm.calls if not call.get("json")]) == 1
        assert sid not in core._openai_api_turns
    finally:
        release.set()
        tasks = list(getattr(core, "_openai_api_turns", {}).values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await core.stop()
