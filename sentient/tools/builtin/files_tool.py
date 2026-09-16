"""Scratch files the assistant may read and write on the user's behalf.

Confined to ~/.sentient/files with a path-traversal guard (ported from the
legacy file_management MCP, minus textract/OCR which will return as an optional extra).
"""

from __future__ import annotations

from pathlib import Path

from sentient import paths
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

_MAX_READ = 200_000


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
async def file_read(ctx: ToolContext, name: str) -> dict:
    """Read a file from the files folder by relative name (text, markdown, CSV, JSON, PDF, DOCX).
    Uploaded attachments live under uploads/."""
    import asyncio

    from sentient.files.extract import extract_text

    p = _safe(name)
    if not p.is_file():
        return {"error": "not found"}
    try:
        text = await asyncio.to_thread(extract_text, p, _MAX_READ + 1)
    except ValueError as exc:
        return {"error": str(exc), "name": name, "bytes": p.stat().st_size}
    return {"name": name, "content": text[:_MAX_READ], "truncated": len(text) > _MAX_READ}


@tool("file_write", risk=Risk.write, internal=True)
async def file_write(ctx: ToolContext, name: str, content: str) -> dict:
    """Write a text file into the files folder. Tell the user the name so they can open it."""
    p = _safe(name)
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
