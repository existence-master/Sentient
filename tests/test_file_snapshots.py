"""File snapshots before code writes, put back from Activity (#142)."""

from __future__ import annotations

import json

import pytest

from sentient import paths
from sentient.app import SentientApp
from sentient.snapshots import service as snap_mod
from sentient.terminal import guard
from tests.conftest import FakeProvider
from tests.terminal.conftest import py


@pytest.fixture
def scratch(tmp_path):
    folder = (tmp_path / "scratch").resolve()
    folder.mkdir()
    return folder


@pytest.fixture
async def app(config, isolated_home, scratch):
    config.terminal.enabled = True
    config.terminal.allowed_folders = [str(scratch)]
    config.sandbox.backend = "process"
    a = await SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "snaps.db", enable_background=False).start()
    yield a
    await a.stop()


async def _rows(app) -> list[dict]:
    rows = await app.store.fetchall("SELECT id FROM file_snapshots ORDER BY created_at")
    return [await app.snapshots.get(r["id"]) for r in rows]


async def test_changed_and_deleted_files_are_kept_and_put_back(app, scratch):
    (scratch / "notes.txt").write_text("original notes", encoding="utf-8")
    (scratch / "old.txt").write_text("old file", encoding="utf-8")
    (scratch / "same.txt").write_text("unchanged", encoding="utf-8")
    ctx = app.agent.tool_context("chat-1", "desktop")

    async with app.snapshots.guard([scratch], ctx, source="terminal", what="tidy up") as cap:
        assert cap is not None
        (scratch / "notes.txt").write_text("overwritten!", encoding="utf-8")
        (scratch / "old.txt").unlink()
        (scratch / "new.txt").write_text("brand new", encoding="utf-8")

    [snap] = await _rows(app)
    assert snap["source"] == "terminal" and snap["session_id"] == "chat-1" and snap["what"] == "tidy up"
    changes = {f["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: f["change"] for f in snap["files"]}
    assert changes == {"notes.txt": "changed", "old.txt": "deleted"}
    assert [p.endswith("new.txt") for p in snap["created"]] == [True]
    folder = snap_mod.store_dir() / snap["id"]
    assert len(list(folder.iterdir())) == 2  # the unchanged file's copy was dropped

    out = await app.snapshots.restore(snap["id"])
    assert out == {"ok": True, "restored": 2, "message": "Put back 2 files."}
    assert (scratch / "notes.txt").read_text(encoding="utf-8") == "original notes"
    assert (scratch / "old.txt").read_text(encoding="utf-8") == "old file"
    assert (scratch / "new.txt").exists()  # files it made are left alone
    assert (await app.snapshots.restore(snap["id"]))["ok"] is False  # once

    # Restore kept a copy of what it replaced, so it can be undone too
    undo = (await _rows(app))[-1]
    assert undo["source"] == "restore" and undo["session_id"] == "chat-1"
    assert [f["change"] for f in undo["files"]] == ["changed"]  # notes.txt; old.txt didn't exist then
    assert (await app.snapshots.restore(undo["id"]))["ok"] is True
    assert (scratch / "notes.txt").read_text(encoding="utf-8") == "overwritten!"


async def test_nothing_is_kept_when_nothing_changed(app, scratch):
    (scratch / "a.txt").write_text("a", encoding="utf-8")
    async with app.snapshots.guard([scratch], None, source="code"):
        pass
    assert await _rows(app) == []
    assert not any(snap_mod.store_dir().iterdir()) if snap_mod.store_dir().exists() else True


async def test_big_and_binary_files_get_a_plain_note_instead_of_a_copy(app, scratch, config):
    config.tools.snapshots.max_file_mb = 0.1
    (scratch / "big.txt").write_text("x" * 200_000, encoding="utf-8")
    (scratch / "pic.bin").write_bytes(b"\x89PNG\0\0data")
    (scratch / "ok.txt").write_text("fine", encoding="utf-8")
    async with app.snapshots.guard([scratch], None, source="code"):
        for name in ("big.txt", "pic.bin", "ok.txt"):
            (scratch / name).write_text("changed", encoding="utf-8")
    [snap] = await _rows(app)
    notes = {f["path"].replace("\\", "/").rsplit("/", 1)[-1]: (f["copy"] is not None, f["note"]) for f in snap["files"]}
    assert notes["big.txt"] == (False, "Too big to keep a copy (over 0.1 MB).")
    assert notes["pic.bin"] == (False, "Not a text file, so no copy was kept.")
    assert notes["ok.txt"] == (True, None)
    assert snap["bytes"] == len("fine")
    assert (await app.snapshots.restore(snap["id"]))["restored"] == 1


async def test_skipped_folders_and_the_copy_limit(app, scratch, config):
    config.tools.snapshots.max_files = 1
    (scratch / "node_modules").mkdir()
    (scratch / "node_modules" / "lib.js").write_text("x", encoding="utf-8")
    (scratch / "a.txt").write_text("a", encoding="utf-8")
    (scratch / "b.txt").write_text("b", encoding="utf-8")
    async with app.snapshots.guard([scratch], None, source="code"):
        for name in ("a.txt", "b.txt", "node_modules/lib.js"):
            (scratch / name).write_text("changed", encoding="utf-8")
    [snap] = await _rows(app)
    kept = sorted((f["path"].replace("\\", "/").rsplit("/", 1)[-1], bool(f["copy"])) for f in snap["files"])
    assert kept == [("a.txt", True), ("b.txt", False)]  # node_modules is never looked at
    assert snap["files"][1]["note"] == "No copy: the copy limit for one run was reached."


async def test_turned_off_and_incognito_keep_nothing(app, scratch, config):
    (scratch / "a.txt").write_text("a", encoding="utf-8")
    config.tools.snapshots.enabled = False
    async with app.snapshots.guard([scratch], None, source="code") as cap:
        (scratch / "a.txt").write_text("b", encoding="utf-8")
    assert cap is None
    config.tools.snapshots.enabled = True
    sid = await app.store.create_session(channel="desktop", incognito=True)
    async with app.snapshots.guard([scratch], app.agent.tool_context(sid, "desktop"), source="code") as cap:
        (scratch / "a.txt").write_text("c", encoding="utf-8")
    assert cap is None and await _rows(app) == []


async def test_a_command_that_overwrites_a_file_can_be_put_back_from_activity(app, scratch):
    if guard.find_shell() is None:
        pytest.skip("no command shell on this machine")
    (scratch / "report.txt").write_text("the real report", encoding="utf-8")
    ctx = app.agent.tool_context("chat-2", "desktop")
    res = await app.terminal.run(py("open('report.txt','w').write('oops')"), None, ctx)
    assert res["ok"] is True
    assert (scratch / "report.txt").read_text(encoding="utf-8") == "oops"

    page = await app.timeline.past(kinds=["chat"])
    item = next(i for i in page["items"] if i["id"].startswith("snap:"))
    assert item["title"] == "A command changed 1 file" and item["undo"] == {"action": "restore", "label": "Restore"}
    assert item["app"] == "terminal" and item["ref"] == {"type": "chat", "id": "chat-2"}
    assert "report.txt" in item["detail"]

    out = await app.timeline.restore(item["id"])
    assert out["ok"] is True
    assert (scratch / "report.txt").read_text(encoding="utf-8") == "the real report"
    again = next(i for i in (await app.timeline.past(kinds=["chat"]))["items"] if i["id"] == item["id"])
    assert again["status"] == "restored" and again["undo"] is None


async def test_a_command_that_never_needs_asking_keeps_no_copies(app, scratch, config):
    if guard.find_shell() is None:
        pytest.skip("no command shell on this machine")
    config.terminal.allowed_commands = ["echo"]
    (scratch / "a.txt").write_text("a", encoding="utf-8")
    calls: list = []
    real = app.snapshots.guard

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    app.snapshots.guard = spy
    await app.terminal.run("echo hi", None, app.agent.tool_context("s", "desktop"))
    assert calls == []


async def test_a_local_script_that_changes_an_allowed_folder_is_kept(app, scratch):
    target = scratch / "data.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    code = f"open({json.dumps(str(target))}, 'w').write('wiped')\nprint('done')"
    res = await app.sandbox.run(code, ctx=app.agent.tool_context("chat-3", "desktop"), purpose="Clean the data")
    assert res["ok"] is True, res
    [snap] = await _rows(app)
    assert snap["source"] == "code" and snap["what"] == "Clean the data" and snap["session_id"] == "chat-3"
    await app.snapshots.restore(snap["id"])
    assert target.read_text(encoding="utf-8") == "a,b\n1,2\n"


async def test_file_write_over_an_existing_file_keeps_the_old_one(app):
    tool = app.registry.get("file_write")
    ctx = app.agent.tool_context("chat-4", "desktop")
    await tool.call(ctx, {"name": "draft.md", "content": "first"})
    assert await _rows(app) == []  # a new file: nothing to keep
    await tool.call(ctx, {"name": "draft.md", "content": "second"})
    [snap] = await _rows(app)
    assert snap["source"] == "file_write" and snap["what"] == "draft.md"
    item = next(i for i in (await app.timeline.past())["items"] if i["id"] == f"snap:{snap['id']}")
    assert item["title"] == "Wrote over draft.md" and item["app"] == "files"
    await app.timeline.restore(item["id"])
    assert (paths.files_dir() / "draft.md").read_text(encoding="utf-8") == "first"


async def test_old_copies_are_pruned(app, scratch):
    (scratch / "a.txt").write_text("a", encoding="utf-8")
    async with app.snapshots.guard([scratch], None, source="code"):
        (scratch / "a.txt").write_text("b", encoding="utf-8")
    [snap] = await _rows(app)
    await app.store.execute("UPDATE file_snapshots SET created_at = '2000-01-01T00:00:00+00:00'")
    assert await app.snapshots.prune() == 1
    assert await _rows(app) == [] and not (snap_mod.store_dir() / snap["id"]).exists()
