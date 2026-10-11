"""Backup, restore and snapshots of the data folder (issue #151, docs/API.md section 20)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient import paths, secrets
from sentient.app import SentientApp
from sentient.backup import BackupError, apply_pending_restore, archive
from sentient.gateway.app import create_app
from sentient.migrate import hermes
from sentient.store.db import SCHEMA_VERSION
from tests.conftest import FakeProvider


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    """In-memory keychain: a backup lists sign-ins, it never reads the real one."""
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    return store


async def _app(config) -> SentientApp:
    return await SentientApp(config, llm=FakeProvider(), enable_background=False).start()


async def _seed(app: SentientApp) -> dict:
    fact = await app.memory.remember("Maya Rao's sister Priya lives in Lisbon", use_llm=False)
    sid = await app.store.create_session(title="Trip planning")
    await app.store.add_message(sid, "user", "Book the Lisbon trip for March")
    task = await app.tasks.create_task("Draft the weekly product update")
    (paths.files_dir() / "notes.txt").write_text("budget v1", encoding="utf-8")
    return {"fact": fact["id"], "session": sid, "task": task["task_id"]}


async def test_backup_restores_an_identical_working_state(config, isolated_home, keychain):
    keychain["openai"] = "sk-test"
    app = await _app(config)
    seeded = await _seed(app)
    made = await app.backups.backup_now()
    assert made["kind"] == "manual" and made["includes_files"]
    assert made["counts"]["chats"] == 1 and made["counts"]["memories"] == 1 and made["counts"]["tasks"] == 1
    assert {"kind": "model", "id": "openai", "label": "OpenAI"} in made["sign_ins"]
    with zipfile.ZipFile(made["path"]) as zf:
        names = set(zf.namelist())
    assert {"manifest.json", "sentient.db", "files/notes.txt", "workspace/SOUL.md"} <= names
    assert not any(n.startswith(("backups/", "logs/")) or n.endswith(("-wal", "-shm", "gateway.token")) for n in names)

    # lose and change things after the backup
    await app.memory.forget(seeded["fact"])
    await app.store.delete_session(seeded["session"])
    await app.tasks.delete(seeded["task"])
    (paths.files_dir() / "notes.txt").write_text("budget v2", encoding="utf-8")
    (paths.files_dir() / "later.txt").write_text("made after the backup", encoding="utf-8")
    staged = await app.backups.restore(made["path"])
    assert staged["restart_required"] and "sentient.db" in staged["contents"]
    assert (await app.backups.overview())["pending_restore"]["source"] == made["path"]
    await app.stop()

    keychain.clear()  # as on a new computer: the key is not in the backup
    result = apply_pending_restore()
    assert result["error"] is None and result["snapshot"].startswith("sentient-snapshot-")
    assert not archive.pending_dir().exists()

    app = await _app(config)
    try:
        facts = await app.memory.list_facts()
        assert [f["content"] for f in facts] == ["Maya Rao's sister Priya lives in Lisbon"]
        assert await app.store.get_session(seeded["session"]) is not None
        assert (await app.store.recent_messages(seeded["session"], 5))[0]["content"] == "Book the Lisbon trip for March"
        assert (await app.tasks.get(seeded["task"]))["description"] == "Draft the weekly product update"
        assert (paths.files_dir() / "notes.txt").read_text(encoding="utf-8") == "budget v1"
        assert not (paths.files_dir() / "later.txt").exists()
        overview = await app.backups.overview()
        last = overview["last_restore"]
        assert last["source"] == made["path"]
        assert last["sign_ins"] == [{"kind": "model", "id": "openai", "label": "OpenAI", "signed_in": False}]
        # the data from before the restore is a snapshot, so the restore can be undone
        assert [s["reason"] for s in overview["snapshots"]] == ["before restoring a backup"]
        assert not overview["snapshots"][0]["includes_files"]
    finally:
        await app.stop()


async def test_snapshot_before_a_schema_update_and_only_the_last_n_kept(config, isolated_home):
    config.backups.snapshots_keep = 2
    app = await _app(config)  # a new database: nothing to protect
    await app.stop()
    assert archive.list_archives(archive.snapshots_dir()) == []
    app = await _app(config)  # same version again: no snapshot
    await app.stop()
    assert archive.list_archives(archive.snapshots_dir()) == []
    assert archive.database_version(paths.db_file()) == SCHEMA_VERSION

    for _ in range(3):  # an older database (a previous build) each time
        con = sqlite3.connect(paths.db_file())
        con.execute("PRAGMA user_version = 0")
        con.commit()
        con.close()
        app = await _app(config)
        await app.stop()
    snaps = archive.list_archives(archive.snapshots_dir())
    assert len(snaps) == 2
    assert snaps[0]["reason"] == f"before updating the database (version 0 to {SCHEMA_VERSION})"
    assert snaps[0]["schema_version"] == 0  # the copy is from before the update
    assert archive.database_version(paths.db_file()) == SCHEMA_VERSION


async def test_snapshot_before_a_new_column_on_an_existing_database(config, isolated_home):
    app = await _app(config)
    await app.store.ensure_column("sessions", "colour", "TEXT")  # what a later build adding a column does
    await app.store.ensure_column("sessions", "shape", "TEXT")  # one snapshot per start, not per column
    await app.stop()
    assert archive.list_archives(archive.snapshots_dir()) == []  # it was a new database
    app = await _app(config)
    try:
        await app.store.ensure_column("sessions", "size", "TEXT")
        await app.store.ensure_column("sessions", "weight", "TEXT")
        snaps = archive.list_archives(archive.snapshots_dir())
        assert [s["reason"] for s in snaps] == ["before adding size to sessions"]
    finally:
        await app.stop()


async def test_hermes_import_takes_a_snapshot_first(config, isolated_home, tmp_path):
    folder = tmp_path / "hermes"
    (folder / "memories").mkdir(parents=True)
    (folder / "memories" / "MEMORY.md").write_text("Maya likes window seats", encoding="utf-8")
    app = await _app(config)
    try:
        await hermes.apply(app, str(folder), ["memory"])
        snaps = archive.list_archives(archive.snapshots_dir())
        assert [s["reason"] for s in snaps] == ["before importing from Hermes"]
    finally:
        await app.stop()


async def test_a_snapshot_restore_leaves_files_alone(config, isolated_home):
    app = await _app(config)
    (paths.files_dir() / "keep.txt").write_text("still here", encoding="utf-8")
    snap = await app.backups.snapshot("before trying something")
    sid = await app.store.create_session(title="After the snapshot")
    await app.backups.restore(snap["path"])
    await app.stop()
    assert apply_pending_restore()["error"] is None
    app = await _app(config)
    try:
        assert await app.store.get_session(sid) is None
        assert (paths.files_dir() / "keep.txt").read_text(encoding="utf-8") == "still here"
    finally:
        await app.stop()


def _zip(path: Path, entries: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _manifest(contents=("sentient.db",)) -> str:
    return json.dumps({"format": 1, "kind": "manual", "contents": list(contents), "created_at": "2026-10-11T10:00:00"})


def test_bad_archives_are_refused_and_nothing_is_staged(tmp_path, isolated_home):
    good_db = tmp_path / "ok.db"
    con = sqlite3.connect(good_db)
    con.execute("CREATE TABLE t(x)")
    con.commit()
    con.close()
    cases = {
        "not a zip": (tmp_path / "a.zip").write_bytes(b"hello") and tmp_path / "a.zip",
        "no manifest": _zip(tmp_path / "b.zip", {"sentient.db": good_db.read_bytes()}),
        "escapes": _zip(tmp_path / "c.zip", {"manifest.json": _manifest(), "sentient.db": good_db.read_bytes(),
                                             "../evil.txt": "x"}),
        "backslash": _zip(tmp_path / "d.zip", {"manifest.json": _manifest(("sentient.db", "files")),
                                               "sentient.db": good_db.read_bytes(), "files/..\\..\\evil.txt": "x"}),
        "not listed": _zip(tmp_path / "e.zip", {"manifest.json": _manifest(), "sentient.db": good_db.read_bytes(),
                                                "logs/x.log": "x"}),
        "damaged db": _zip(tmp_path / "f.zip", {"manifest.json": _manifest(), "sentient.db": b"not a database" * 100}),
        "no db": _zip(tmp_path / "g.zip", {"manifest.json": _manifest(("config.yaml",)), "config.yaml": "x: 1"}),
        "future format": _zip(tmp_path / "h.zip", {"manifest.json": json.dumps({"format": 9, "contents": []})}),
    }
    for label, path in cases.items():
        with pytest.raises(BackupError):
            archive.stage_restore(path)
        assert archive.pending_restore() is None, label
        assert not (tmp_path / "evil.txt").exists() and not (isolated_home / "evil.txt").exists()
    assert apply_pending_restore() is None


async def test_scheduled_backups_keep_the_last_n_and_never_touch_manual_ones(config, isolated_home, tmp_path):
    folder = tmp_path / "elsewhere"
    config.backups.folder = str(folder)
    config.backups.schedule = "daily"
    config.backups.keep = 2
    app = await _app(config)
    try:
        manual = await app.backups.backup_now()
        await app.backups._scheduled()
        await app.backups._scheduled()  # not due yet
        assert len([b for b in (await app.backups.overview())["backups"] if b["kind"] == "scheduled"]) == 1
        for _ in range(3):
            await app.store.set_meta("backup.last_scheduled", "2020-01-01T00:00:00+00:00")
            await app.backups._scheduled()
        kinds = [b["kind"] for b in (await app.backups.overview())["backups"]]
        assert kinds.count("scheduled") == 2 and kinds.count("manual") == 1
        assert Path(manual["path"]).exists()
        config.backups.schedule = "off"
        await app.store.set_meta("backup.last_scheduled", "2020-01-01T00:00:00+00:00")
        await app.backups._scheduled()
        assert len((await app.backups.overview())["backups"]) == 3
    finally:
        await app.stop()


def test_backup_routes(config, isolated_home, monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        first = c.get("/api/backups").json()
        assert first["backups"] == [] and first["pending_restore"] is None and first["schedule"] == "off"
        made = c.post("/api/backups", json={}).json()
        assert made["name"].startswith("sentient-backup-")
        elsewhere = c.post("/api/backups", json={"folder": str(tmp_path / "usb")}).json()
        assert Path(elsewhere["path"]).parent == tmp_path / "usb"
        assert c.post("/api/backups", json={"folder": "relative/folder"}).status_code == 400
        seen = c.post("/api/backups/inspect", json={"path": made["path"]}).json()
        assert seen["contents"][0] == "sentient.db" and seen["newer"] is False
        assert c.post("/api/backups/inspect", json={"path": str(tmp_path / "nope.zip")}).status_code == 400
        staged = c.post("/api/backups/restore", json={"path": made["path"]}).json()
        assert staged["restart_required"] is True
        assert c.get("/api/backups").json()["pending_restore"]["source"] == made["path"]
        assert c.delete("/api/backups/restore").json() == {"cancelled": True}
        assert c.get("/api/backups").json()["pending_restore"] is None
        assert [b["name"] for b in c.get("/api/backups").json()["backups"]] == [made["name"]]
        assert c.delete("/api/backups/last-restore").json() == {"ok": True}


def test_a_staged_restore_is_applied_when_the_engine_starts(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    monkeypatch.setenv("SENTIENT_DISABLE_BACKGROUND", "1")

    async def setup():
        app = await _app(config)
        made = await app.backups.backup_now()
        await app.store.create_session(title="Made after the backup")
        await app.backups.restore(made["path"])
        await app.stop()

    asyncio.run(setup())
    with TestClient(create_app()) as c:  # the desktop's engine: lifespan applies the restore first
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/sessions").json() == []
        last = c.get("/api/backups").json()["last_restore"]
        assert last["error"] is None and last["snapshot"]


async def test_a_failed_restore_puts_the_current_data_back(config, isolated_home, monkeypatch):
    app = await _app(config)
    (paths.files_dir() / "mine.txt").write_text("in the backup", encoding="utf-8")
    made = await app.backups.backup_now()
    sid = await app.store.create_session(title="Made after the backup")
    (paths.files_dir() / "mine.txt").write_text("keep me", encoding="utf-8")
    await app.backups.restore(made["path"])
    await app.stop()

    real = archive._move

    def flaky(src, dst, tries=6):
        if Path(src).name == "files" and archive.PENDING in str(src):
            raise PermissionError("in use")
        real(src, dst, tries=1)

    monkeypatch.setattr(archive, "_move", flaky)
    result = apply_pending_restore()
    assert result["error"] and "left as it was" in result["error"]
    assert not archive.pending_dir().exists()
    app = await _app(config)
    try:
        assert await app.store.get_session(sid) is not None
        assert (paths.files_dir() / "mine.txt").read_text(encoding="utf-8") == "keep me"
        assert (await app.backups.overview())["last_restore"]["error"] == result["error"]
    finally:
        await app.stop()


async def test_a_failed_scheduled_backup_is_retried_and_told_once(config, isolated_home, tmp_path, monkeypatch):
    config.backups.schedule = "daily"
    app = await _app(config)
    try:
        calls = []

        async def broken(folder=None, *, kind="manual"):
            calls.append(kind)
            raise BackupError("Sentient can't write to that folder. Pick another one.")

        monkeypatch.setattr(app.backups, "backup_now", broken)
        await app.backups._scheduled()
        await app.backups._scheduled()  # an hour later: tried again, not told again
        assert calls == ["scheduled", "scheduled"]
        notes = [n for n in await app.notifications.list() if n["kind"] == "backup"]
        assert len(notes) == 1 and "didn't work" in json.dumps(notes[0])
        assert await app.store.get_meta("backup.last_scheduled") is None
        monkeypatch.undo()
        await app.backups._scheduled()
        assert await app.store.get_meta("backup.last_scheduled")
    finally:
        await app.stop()


async def test_a_rollback_that_fails_keeps_the_data_aside(config, isolated_home, monkeypatch):
    app = await _app(config)
    (paths.files_dir() / "mine.txt").write_text("in the backup", encoding="utf-8")
    made = await app.backups.backup_now()
    (paths.files_dir() / "mine.txt").write_text("only copy", encoding="utf-8")
    await app.backups.restore(made["path"])
    await app.stop()

    real = archive._move

    def stuck(src, dst, tries=6):
        if Path(src).name == "files" and (archive.PENDING in str(src) or "restore-old" in str(src)):
            raise PermissionError("in use")
        real(src, dst, tries=1)

    monkeypatch.setattr(archive, "_move", stuck)
    result = apply_pending_restore()
    assert "couldn't be put back" in result["error"]
    kept = [p for p in archive.backups_root().glob("restore-old-*") if (p / "files" / "mine.txt").exists()]
    assert len(kept) == 1 and (kept[0] / "files" / "mine.txt").read_text(encoding="utf-8") == "only copy"
