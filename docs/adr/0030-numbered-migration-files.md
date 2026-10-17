# 0030. Change the database shape through numbered migration files

- **Status:** Accepted
- **Date:** 2026-10-17

## Context

[ADR 0003](0003-single-sqlite-database.md) keeps all state in one SQLite file. Each package owned an idempotent
`schema.sql` and added columns with `store.ensure_column`, called from `Store.open` and from a dozen services'
`start()` (tasks, memory, skills, channels, reminders...). Issue #151 added a `SCHEMA_VERSION` constant, so a
database opened at another version gets a snapshot first.

That constant was one hand-typed line with a growing comment. Two pull requests that each changed the schema both
edited that line and silently picked the same number (#401 took 12 while another change was in flight). Columns
also kept arriving without a bump (`messages.checks`, group chats, skill checks), and nothing checked that a new
database and an updated old one ended up with the same shape: several columns existed only because a service added
them when it started, not in any `schema.sql`.

## Decision

The database shape is described in two places, each with one job:

- The `schema.sql` files hold the **whole current shape** (`CREATE ... IF NOT EXISTS`). A new database is made from
  them alone and runs no steps.
- `sentient/store/migrations/` holds **one file per step**, `NNNN_<what>.py` (`async def migrate(store)`) or
  `NNNN_<what>.sql`, numbered 1, 2, 3... The highest number is `SCHEMA_VERSION`; nobody types it.

`Store.open` on an existing database below that version takes the snapshot, runs the steps above its version in
order (stamping `PRAGMA user_version` after each), then runs the `schema.sql` files, which add new tables and
indexes. Steps only add columns and fix rows, check that their table exists, and can run twice.

Packages no longer call `store.ensure_column` when they start. Steps 1 to 12 port what `Store.open` and the services
did before; columns that had arrived without a bump went into the step after the version they arrived at, and the
skill checks columns (#386) into a new step 13.

`tests/test_migrations.py` fails when two files share a number, a number is skipped or a file is misnamed, and when
a database made by an old build (frozen schemas in `tests/schemas/`) and migrated differs from a new one in its
tables, columns, indexes or triggers.

## Consequences

Two changes that add a step no longer edit the same line. They may still pick the same number, but git shows no
conflict, so the test catches it: whoever merges second renumbers their file. The version can't be forgotten: adding
a column means adding a step, which raises the version and takes the snapshot.

A step is frozen once it is on `main`, because databases that ran it never run it again. `schema.sql` and the steps
must agree; the test enforces it. The frozen build collects the step modules through `collect_submodules("sentient")`
and finds them with `pkgutil.iter_modules`; `.sql` steps are shipped as data files.

## Alternatives considered

A version table per package, or Alembic: more machinery than one file and one writer need, and Alembic brings
SQLAlchemy. Keeping `ensure_column` on every start without versions: no conflicts, but no record of what changed when
and no check that old and new databases match. Running every step on every start: simpler, but data fixes would run
again and again, and a new database would no longer be described by `schema.sql` alone.
