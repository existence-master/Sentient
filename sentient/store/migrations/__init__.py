"""Numbered steps that bring a database made by an older build up to date (ADR 0030).

One file per step, named ``NNNN_<what>.py`` (or ``NNNN_<what>.sql``), numbered 1, 2, 3... with no gaps or repeats.
The highest number is the schema version (``sentient.store.db.SCHEMA_VERSION``), kept in ``PRAGMA user_version``;
nobody types it by hand.

- A ``.py`` step defines ``async def migrate(store)`` and adds columns with ``store.ensure_column``.
- A ``.sql`` step is a script of statements that are safe to run twice (``ALTER TABLE ... ADD COLUMN`` is not).

A step runs on an existing database whose version is lower than its number, before the ``schema.sql`` files,
which then create any missing tables and indexes. So a step never creates a table or an index: it only adds
columns to tables that already exist and fixes their rows. A table it names may be missing (a very old database),
so it checks first. Every step can run twice: a crash between a step and its version stamp runs it again.

A new database never runs them: the ``schema.sql`` files alone describe the whole current shape, and
``tests/test_migrations.py`` checks that both ways end with the same tables, columns and indexes.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentient.store.db import Store

FILE_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.(py|sql)$")


class MigrationError(RuntimeError):
    """The migration files are numbered wrong (a repeat, a gap or a bad name)."""


@dataclass(frozen=True)
class Migration:
    number: int
    name: str
    file: str  # "0012_folder_scopes.py"
    run: Callable[[Store], Awaitable[Any]]


def _sql_step(sql: str) -> Callable[[Store], Awaitable[Any]]:
    async def run(store: Store) -> None:
        await store.db.executescript(sql)

    return run


def _files(package: str) -> list[str]:
    """Every step file in ``package``: Python modules through pkgutil (which also works in the frozen engine),
    SQL scripts as package data."""
    mod = importlib.import_module(package)
    names = {f"{m.name}.py" for m in pkgutil.iter_modules(mod.__path__) if not m.ispkg}
    with contextlib.suppress(FileNotFoundError, NotADirectoryError):  # a frozen package with no data files
        names |= {p.name for p in resources.files(package).iterdir() if p.name.endswith(".sql")}
    return sorted(names)


def discover(package: str = __name__) -> list[Migration]:
    """The steps in ``package``, in order. Raises ``MigrationError`` when two share a number, a number is skipped
    or a file is misnamed, so a bad numbering stops the build's tests instead of silently skipping a step."""
    found: dict[int, list[str]] = {}
    steps: dict[int, Migration] = {}
    for file in _files(package):
        m = FILE_NAME.match(file)
        if m is None:
            raise MigrationError(f"{file}: a migration file is named NNNN_<what>.py or NNNN_<what>.sql")
        number, name, kind = int(m[1]), m[2], m[3]
        found.setdefault(number, []).append(file)
        if kind == "py":
            module = importlib.import_module(f"{package}.{file[:-3]}")
            run = getattr(module, "migrate", None)
            if run is None:
                raise MigrationError(f"{file}: has no async def migrate(store)")
        else:
            run = _sql_step(resources.files(package).joinpath(file).read_text(encoding="utf-8"))
        steps[number] = Migration(number, name, file, run)
    repeats = {n: files for n, files in found.items() if len(files) > 1}
    if repeats:
        listed = "; ".join(", ".join(files) for files in repeats.values())
        raise MigrationError(f"two migrations share a number: {listed}. Renumber yours to the next free number.")
    numbers = sorted(steps)
    missing = sorted(set(range(1, (numbers[-1] if numbers else 0) + 1)) - set(numbers))
    if missing:
        raise MigrationError(f"migration numbers must run 1, 2, 3... with no gaps; missing {missing}")
    return [steps[n] for n in numbers]


async def migrate(store: Store, from_version: int, steps: list[Migration] | None = None) -> list[int]:
    """Run every step above ``from_version`` in order, stamping the version after each. Returns the numbers run."""
    done = []
    for step in steps if steps is not None else discover():
        if step.number <= from_version:
            continue
        await step.run(store)
        await store.db.execute(f"PRAGMA user_version = {step.number}")
        await store.db.commit()
        done.append(step.number)
    return done
