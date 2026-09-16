"""Evolution log: an append-only record of what self-evolution changed."""

from __future__ import annotations

import json
import logging
from typing import Any

from sentient.store.db import Store, now_iso

log = logging.getLogger(__name__)

KINDS = {
    "skill_created", "skill_patched", "skill_repair_proposed", "skill_archived", "profile_updated", "curator_run",
    "summary_created",
}


async def log_event(store: Store, kind: str, detail: dict[str, Any] | None = None, *, ts: str | None = None) -> None:
    try:
        await store.execute(
            "INSERT INTO evolution_log(ts, kind, detail) VALUES(?,?,?)",
            (ts or now_iso(), kind, json.dumps(detail or {}, default=str)),
        )
    except Exception as exc:  # the log must never break the action it records
        log.warning("evolution log write failed: %s", exc)


async def read_log(store: Store, limit: int = 100) -> list[dict]:
    rows = await store.fetchall("SELECT ts, kind, detail FROM evolution_log ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
    return [{"ts": r["ts"], "kind": r["kind"], "detail": json.loads(r["detail"] or "{}")} for r in rows]
