"""Tools for the folders and Obsidian vaults the user added (#176). Read only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentient.folders.notes import obsidian_url
from sentient.folders.service import CLIPPED_SOURCE, folder_scope
from sentient.memory import scopes as memory_scopes
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.builtin.files_tool import _fit_part, part_budget


def _service(ctx: ToolContext) -> Any:
    return getattr((getattr(ctx, "extra", None) or {}).get("app"), "folders", None)


# The user's own notes are not outside content (ADR 0018); a clipped web page among them is, per result (ADR 0025).
@tool("folder_search", risk=Risk.read, untrusted_output=False)
async def folder_search(ctx: ToolContext, query: str, folder: str = "", limit: int = 0, everywhere: bool = False) -> dict:
    """Search the notes and documents in the folders the user added (an Obsidian vault, a documents folder).
    Use it when the user asks about their notes, projects, people, plans or anything they may have written down.
    `query` is what to look for in plain words. `folder` limits it to one folder by name (empty = all).
    It searches the folders this chat uses (for example Work). Set everywhere=true only when the user asks about
    notes from another part of their life.
    Each result has the note's `title`, `path`, `heading` and the matching `text`. Answer from them and name the
    notes you used. To read a whole note, call folder_read with its `file`."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Searching your folders isn't available right now."}
    scope = await memory_scopes.for_context(ctx)
    return await svc.search(query, limit=limit or None, folder=folder, scope=scope, everywhere=bool(everywhere))


@tool("folder_read", risk=Risk.read, untrusted_output=False)
async def folder_read(ctx: ToolContext, path: str, offset: int = 0, limit: int = 0, everywhere: bool = False) -> dict:
    """Read a whole note or document from the user's folders. Pass the `file` a folder_search result gave (its full
    path), so the right folder is used. Set everywhere=true for a note folder_search found with everywhere=true.
    A long one comes in parts: when `more` is true, call again with offset = `next_offset`."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Reading your folders isn't available right now."}
    scope = None if everywhere else await memory_scopes.for_context(ctx)
    try:
        found = await svc.read(path, scope)
    except (OSError, ValueError) as exc:  # also a file that became a link out of the folder
        return {"error": f"That file can't be read: {exc}"}
    if found is None:
        other = await svc.resolve(path) if scope is not None else None
        if other is not None:  # in a folder kept for other chats (#392)
            return {"error": (
                f"That note is in a folder kept for {memory_scopes.label(folder_scope(other[0].get('scope')))} chats. "
                "Read it with everywhere=true only if the user asked about it."
            )}
        return {"error": "That isn't a file in the user's folders. Use the `file` a folder_search result gave."}
    folder, row, text = found
    budget = max(1000, part_budget(ctx) - 600)  # room for the title, path and link around the part
    want = min(int(limit) if limit and int(limit) > 0 else budget, budget)
    offset = max(0, int(offset or 0))
    part = _fit_part(row["rel"], text, offset, want, budget, len(text))
    out: dict = {}
    if row.get("clipped") and ctx.config.folders.clippings_are_outside:
        out["outside_content"] = CLIPPED_SOURCE  # first, so it survives a cut (ADR 0025)
    out.update({"title": row.get("title") or row["rel"], "folder": folder["name"], "path": row["rel"]})
    out["file"] = str(Path(folder["path"]) / row["rel"])
    if folder.get("kind") == "obsidian":
        out["link"] = obsidian_url(folder["name"], row["rel"])
    if row.get("clipped"):
        out["clipped_from"] = row["clipped"]
    part.pop("name", None)
    out.update(part)
    return out


class FoldersPlugin(ToolPlugin):
    id = "folders"
    display_name = "Your folders"
    description = "Search and read the notes and documents in folders the user added, such as an Obsidian vault."
    category = "core"
    icon = "IconFolders"
    selection_hint = (
        "questions about the user's own notes, Obsidian vault, journal, documents, projects, people or plans they "
        "wrote down"
    )
    tools = [folder_search, folder_read]
