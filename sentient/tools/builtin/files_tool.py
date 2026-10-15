"""Scratch files the assistant may read and write on the user's behalf.

Confined to ~/.sentient/files with a path-traversal guard (ported from the
legacy file_management MCP, minus textract/OCR which will return as an optional extra).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from sentient import paths
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

_WHOLE_TEXT_BYTES = 4_000_000  # a text file up to this size is read whole for each part, so its total is known
_MAX_TEXT = _WHOLE_TEXT_BYTES + 1  # characters: never reached by a file read whole (one byte holds at most one)
_DEFAULT_PART = 16_000  # one part when nothing says how much the model reads at once
_MIN_PART = 1_000


def _safe(name: str) -> Path:
    root = paths.files_dir().resolve()
    target = (root / name).resolve()
    if root not in target.parents and target != root:
        raise ValueError("path escapes the files directory")
    return target


@tool("file_list", risk=Risk.read)
async def file_list(ctx: ToolContext, subdir: str = "") -> list[dict]:
    """List files in the assistant's files folder (where saved outputs live)."""
    base = _safe(subdir) if subdir else paths.files_dir()
    if not base.exists():
        return []
    return [
        {"name": str(p.relative_to(paths.files_dir())), "bytes": p.stat().st_size}
        for p in sorted(base.rglob("*"))
        if p.is_file()
    ]


@tool("file_read", risk=Risk.read)
async def file_read(ctx: ToolContext, name: str, offset: int = 0, limit: int = 0) -> dict:
    """Read a file from the files folder by relative name (text, markdown, CSV, JSON, PDF, DOCX).
    Uploaded attachments live under uploads/. A long file comes in parts: `offset` is the character to start at
    (0 = the start) and `limit` the most characters to return (0 = as much as fits). When `more` is true, call
    again with offset = `next_offset` for the next part."""
    import asyncio

    from sentient.files.extract import extract_text, is_text_like

    p = _safe(name)
    if not p.is_file():
        return {"error": "not found"}
    offset = max(0, int(offset or 0))
    budget = part_budget(ctx)
    want = min(int(limit) if limit and int(limit) > 0 else budget, budget)
    try:
        if is_text_like(p) and p.stat().st_size > _WHOLE_TEXT_BYTES:
            # a big text file: read only as far as this part needs, never the whole file for every part
            text, cut = await asyncio.to_thread(_read_upto, p, offset + want + 1)
        else:
            # a small text file is read whole, so the total is known; a PDF or document only as far as this part needs
            cap = _MAX_TEXT if is_text_like(p) else offset + want + 1
            text = await asyncio.to_thread(extract_text, p, cap)
            cut = len(text) > cap  # extract_text added its own note after the first ``cap`` characters
            if cut:
                text = text[:cap]
    except ValueError as exc:
        return {"error": str(exc), "name": name, "bytes": p.stat().st_size}
    if offset >= len(text) and not cut:
        return {"name": name, "offset": offset, "content": "", "total": len(text), "more": False,
                "note": f"The file has {len(text)} characters, so there is nothing at offset {offset}."}
    return _fit_part(name, text, offset, want, budget, None if cut else len(text))


def _read_upto(path: Path, count: int) -> tuple[str, bool]:
    """The first ``count`` characters of a text file, and True when the file goes on after them."""
    with path.open(encoding="utf-8", errors="replace") as f:
        text = f.read(count)
        return text, bool(f.read(1))


def part_budget(ctx: ToolContext) -> int:
    """The most characters one tool result may hold for the model reading it: the run's own limit (set by the
    agent loop), else ``chat.tool_result_max_chars``."""
    limit = int(getattr(ctx, "result_limit", 0) or 0)
    if not limit:
        chat = getattr(getattr(ctx, "config", None), "chat", None)
        limit = int(getattr(chat, "tool_result_max_chars", 0) or _DEFAULT_PART)
    return max(_MIN_PART, limit)


def _fit_part(name: str, text: str, offset: int, want: int, budget: int, total: int | None) -> dict:
    """The part of ``text`` from ``offset``, at most ``want`` characters, cut further so the whole result (as JSON,
    where line breaks and quotes take two characters) fits in ``budget``. It says where the next part starts."""

    def build(k: int) -> dict:
        end = offset + k
        more = end < len(text) or total is None
        out: dict = {"name": name, "offset": offset, "content": text[offset:end]}
        if total is not None:
            out["total"] = total
        out["more"] = more
        if more:
            out["next_offset"] = end
        return out

    k = max(0, min(want, len(text) - offset))
    res = build(k)
    size = len(json.dumps(res, ensure_ascii=False))
    while size > budget and k > 1:
        k = max(1, min(k - 1, int(k * budget / size) - 16))
        res = build(k)
        size = len(json.dumps(res, ensure_ascii=False))
    return res


def _lines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _wrote(arguments: dict, result: object, ctx: ToolContext) -> dict | None:
    """Read a written file back (#165): it exists and holds what was written (line endings aside)."""
    name = str((arguments or {}).get("name") or "")
    if not name:
        return None
    p = _safe(name)
    if not p.is_file():
        return {"ok": False, "detail": f"the file {name} isn't in the files folder"}
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"ok": False, "detail": f"the file {name} couldn't be read back ({type(exc).__name__})"}
    want = _lines(str((arguments or {}).get("content") or ""))
    if _lines(text) != want:
        return {"ok": False, "detail": f"the file {name} doesn't hold what was written"}
    return {"ok": True}


@tool("file_write", risk=Risk.write, internal=True, outcome="file", verify=_wrote)
async def file_write(ctx: ToolContext, name: str, content: str) -> dict:
    """Write a text file into the files folder. Tell the user the name so they can open it."""
    p = _safe(name)
    # a file written over keeps a copy, so it can be put back from Activity (#142)
    snaps = getattr((getattr(ctx, "extra", None) or {}).get("app"), "snapshots", None)
    keep = snaps.guard([p], ctx, source="file_write", what=name) if snaps is not None else contextlib.nullcontext()
    async with keep:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return {"saved": name, "path": str(p), "bytes": len(content.encode("utf-8"))}


class FilesPlugin(ToolPlugin):
    id = "files"
    display_name = "Files"
    description = "Read and write files in Sentient's own folder."
    category = "core"
    icon = "IconFolder"
    selection_hint = "saving outputs, reading documents the user dropped in the files folder"
    tools = [file_list, file_read, file_write]


PLUGIN = FilesPlugin()
