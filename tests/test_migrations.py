"""Numbered migration steps (ADR 0030): their numbering, and that an old database ends up shaped like a new one."""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path

import pytest

from sentient.store import migrations
from sentient.store.db import SCHEMA_VERSION, Store
from sentient.store.migrations import MigrationError

SCHEMAS = Path(__file__).with_name("schemas")


# ---------------------------------------------------------------- numbering
def test_the_steps_run_one_two_three_and_the_last_is_the_schema_version():
    steps = migrations.discover()
    assert [s.number for s in steps] == list(range(1, len(steps) + 1))
    assert steps[-1].number == SCHEMA_VERSION
    assert all(re.fullmatch(r"\d{4}_[a-z0-9_]+\.(py|sql)", s.file) for s in steps)


def _package(tmp_path, monkeypatch, files: dict[str, str]) -> str:
    """A throwaway migrations package holding ``files``; returns its import name."""
    name = f"steps_{uuid.uuid4().hex[:8]}"
    folder = tmp_path / name
    folder.mkdir()
    (folder / "__init__.py").write_text("", encoding="utf-8")
    for file, text in files.items():
        (folder / file).write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


STEP = "async def migrate(store):\n    pass\n"


def test_two_steps_with_one_number_fail(tmp_path, monkeypatch):
    pkg = _package(tmp_path, monkeypatch, {
        "0001_first.py": STEP, "0002_mine.py": STEP, "0002_theirs.sql": "SELECT 1;",
    })
    with pytest.raises(MigrationError, match=r"share a number: 0002_mine.py, 0002_theirs.sql"):
        migrations.discover(pkg)


def test_a_gap_in_the_numbers_fails(tmp_path, monkeypatch):
    pkg = _package(tmp_path, monkeypatch, {"0001_first.py": STEP, "0003_third.py": STEP})
    with pytest.raises(MigrationError, match=r"no gaps; missing \[2\]"):
        migrations.discover(pkg)


@pytest.mark.parametrize("file", ["13_short.py", "0013-dash.py", "0013_Upper.py", "notes.sql"])
def test_a_misnamed_step_fails(tmp_path, monkeypatch, file):
    pkg = _package(tmp_path, monkeypatch, {"0001_first.py": STEP, file: STEP})
    with pytest.raises(MigrationError, match="named NNNN_"):
        migrations.discover(pkg)


def test_a_python_step_needs_migrate(tmp_path, monkeypatch):
    pkg = _package(tmp_path, monkeypatch, {"0001_first.py": "X = 1\n"})
    with pytest.raises(MigrationError, match="has no async def migrate"):
        migrations.discover(pkg)


async def test_steps_above_the_version_run_in_order_and_each_is_stamped(tmp_path, monkeypatch, isolated_home):
    pkg = _package(tmp_path, monkeypatch, {
        "0001_table.sql": "CREATE TABLE IF NOT EXISTS t (a TEXT);",
        "0002_column.py": "async def migrate(store):\n    await store.ensure_column('t', 'b', 'TEXT')\n",
        "0003_fails.py": "async def migrate(store):\n    raise RuntimeError('half way')\n",
    })
    steps = migrations.discover(pkg)
    store = await Store(tmp_path / "steps.db").open()
    try:
        await store.execute("PRAGMA user_version = 0")
        with pytest.raises(RuntimeError, match="half way"):
            await migrations.migrate(store, 0, steps)
        assert await store.columns("t") == {"a", "b"}
        row = await store.fetchone("PRAGMA user_version")
        assert row[0] == 2  # the next start runs step 3 again, and only step 3
        assert await migrations.migrate(store, 2, steps[:2]) == []
    finally:
        await store.close()


# ---------------------------------------------------------------- an old database ends up like a new one
def _shape(path: Path) -> dict:
    """Tables with their columns (order aside), indexes and triggers: what a step or a schema.sql decides."""
    con = sqlite3.connect(path)
    try:
        shape: dict = {}
        for kind, name, table, sql in con.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ):
            if kind == "table":
                cols = con.execute(f"PRAGMA table_xinfo('{name}')").fetchall()
                shape[f"table {name}"] = sorted((c[1], c[2].upper(), c[3], c[4], c[5]) for c in cols)
            elif sql:  # an index or trigger written out (not one SQLite makes for a key)
                text = re.sub(r"\s+", " ", sql.replace("IF NOT EXISTS ", "")).strip()
                shape[f"{kind} {name}"] = (table, text)
        return shape
    finally:
        con.close()


def _old_database(path: Path, schema: str) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript((SCHEMAS / schema).read_text(encoding="utf-8"))
        con.execute(
            "INSERT INTO sessions(id, title, channel, created_at, updated_at)"
            " VALUES('s1', 'Plans for Lisbon', 'web', '2026-10-01T09:00:00', '2026-10-01T09:00:00')"
        )
        con.execute(
            "INSERT INTO summaries(id, session_id, content, start_at, end_at, message_ids, created_at)"
            " VALUES('m1', 's1', 'Maya planned a trip to Lisbon', '2026-10-01T09:00:00', '2026-10-01T09:30:00', '[]',"
            " '2026-10-01T09:30:00')"
        )
        con.commit()
    finally:
        con.close()


@pytest.mark.parametrize(("schema", "version"), [("schema_v1.sql", 1), ("schema_v0.sql", 0)])
async def test_an_old_database_migrated_matches_a_new_one(tmp_path, isolated_home, schema, version):
    new = await Store(tmp_path / "new.db").open()
    await new.close()

    old_path = tmp_path / "old.db"
    _old_database(old_path, schema)
    assert sqlite3.connect(old_path).execute("PRAGMA user_version").fetchone()[0] == version
    old = await Store(old_path).open()
    try:
        row = await old.fetchone("SELECT title, incognito, model, memory_scope FROM sessions WHERE id = 's1'")
        assert dict(row) == {"title": "Plans for Lisbon", "incognito": 0, "model": None, "memory_scope": None}
        summary = await old.fetchone("SELECT untrusted, scope FROM summaries WHERE id = 'm1'")
        assert (summary["untrusted"], summary["scope"]) == (None, None)
    finally:
        await old.close()

    assert sqlite3.connect(old_path).execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    fresh, migrated = _shape(tmp_path / "new.db"), _shape(old_path)
    missing = {k: v for k, v in fresh.items() if migrated.get(k) != v}
    extra = {k: v for k, v in migrated.items() if k not in fresh}
    assert not missing and not extra, (
        "a new database (the schema.sql files) and one migrated from version "
        f"{version} differ. Add a step in sentient/store/migrations or fix the schema.sql.\n"
        f"new database: {missing}\nmigrated database: { {k: migrated.get(k) for k in missing} | extra}"
    )


async def test_a_new_database_runs_no_steps(tmp_path, isolated_home, monkeypatch):
    ran = []

    async def spy(store, version, steps=None):
        ran.append(version)
        return []

    monkeypatch.setattr(migrations, "migrate", spy)
    store = await Store(tmp_path / "new.db").open()
    await store.close()
    assert ran == []
    store = await Store(tmp_path / "new.db").open()  # opened again at the same version
    await store.close()
    assert ran == []
