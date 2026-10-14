"""Answer sources (#174): what a chat reply was based on, shown as chips under it.

Collected deterministically from the tool results the model read during the turn, like memory sources
(``sentient.memory.sources``): the model is never asked what it used. A source is
``{kind, title, url?, file?, from?, snippet?, app, tool}``:

- ``web``: a page web_fetch or the browser opened (``url``, the page ``title``), or one of the first results of a
  web search;
- ``email``: an email a mail tool returned (``title`` is its subject, ``from``, ``url`` when the mail app has a link);
- ``file``: a file file_read opened (``file`` is its name under the files folder), or a note from a folder the user
  added (#176): ``path`` is its full path, ``folder`` the folder's name, ``url`` an ``obsidian://`` link for a note in
  an Obsidian vault;
- ``app``: a connected app whose tools ran and returned something else (a calendar, a task list...), ``title`` its
  name and ``snippet`` what was done.

Memories stay in ``memory_sources``. ``app`` is the plugin id, ``tool`` the tool that found it. First mention wins,
at most ``MAX_SOURCES``; snippets are short excerpts of what the model read.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

MAX_SOURCES = 12
SNIPPET_CHARS = 280
SEARCH_RESULTS_PER_CALL = 3
WEB_PLUGINS = frozenset({"web", "browser"})
SEARCH_PLUGINS = frozenset({"internet_search"})
FILE_TOOLS = frozenset({"file_read"})
FOLDER_TOOLS = frozenset({"folder_search", "folder_read"})
FOLDER_RESULTS_PER_CALL = 3


_MARKER = re.compile(r"^\s*(?:#{1,6}|[-*+>]|\d+[.)])\s+")


def _snip(text: Any) -> str:
    """A short excerpt on one line, without markdown heading and list markers (web_fetch returns markdown)."""
    lines = (_MARKER.sub("", line) for line in str(text or "").splitlines())
    s = " ".join(" ".join(lines).replace("**", "").split())
    return s if len(s) <= SNIPPET_CHARS else s[: SNIPPET_CHARS - 1].rstrip() + "…"


def _http(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    try:
        parts = urlparse(url.strip())
    except ValueError:
        return None
    return url.strip() if parts.scheme in {"http", "https"} and parts.netloc else None


def _emails(value: Any, depth: int = 0) -> list[dict]:
    """Email-shaped dicts (a subject and a sender) anywhere in a mail tool's result."""
    if depth > 4:
        return []
    if isinstance(value, dict):
        if "subject" in value and ("from" in value or "sender_email" in value):
            return [value]
        return [e for v in value.values() for e in _emails(v, depth + 1)]
    if isinstance(value, list):
        return [e for v in value for e in _emails(v, depth + 1)]
    return []


class AnswerSources:
    """Collects one reply's sources, in order. ``stored`` is a list saved earlier, merged in first."""

    def __init__(self, stored: list[dict] | None = None) -> None:
        self._items: list[dict] = []
        self._keys: dict[tuple[str, str], dict] = {}
        for s in stored if isinstance(stored, list) else []:
            if isinstance(s, dict) and s.get("kind") in {"web", "email", "file", "app"}:
                self._add(dict(s), self._key(s))

    @staticmethod
    def _key(s: dict) -> tuple[str, str]:
        kind = str(s.get("kind"))
        if kind == "web":
            return kind, str(s.get("url"))
        if kind == "email":
            return kind, f"{s.get('app')}:{s.get('id') or s.get('url') or s.get('title')}"
        if kind == "file":
            return kind, str(s.get("path") or s.get("file"))
        return kind, str(s.get("app"))

    def _add(self, item: dict, key: tuple[str, str]) -> dict | None:
        if key in self._keys or len(self._items) >= MAX_SOURCES:
            return self._keys.get(key)
        item = {k: v for k, v in item.items() if v not in (None, "")}
        self._keys[key] = item
        self._items.append(item)
        return item

    def add_tool_result(self, name: str, result: Any, *, plugin: str = "", app_name: str = "", is_app: bool = False) -> None:
        """One successful tool result. ``plugin`` is the tool's plugin id, ``app_name`` its display name, ``is_app``
        true for a connected app (an integration, not Sentient's own memory, files or skills)."""
        if isinstance(result, dict) and (result.get("error") or result.get("declined")):
            return
        before = len(self._items)
        if plugin in WEB_PLUGINS and isinstance(result, dict):
            url = _http(result.get("url"))
            if url and (result.get("title") or result.get("content") or result.get("text")):
                title = str(result.get("title") or "").strip() or urlparse(url).netloc
                snippet = _snip(result.get("content") or result.get("text"))
                self._add({"kind": "web", "title": title, "url": url, "snippet": snippet, "app": plugin, "tool": name},
                          ("web", url))
            return
        if plugin in SEARCH_PLUGINS and isinstance(result, dict):
            for r in (result.get("results") or [])[:SEARCH_RESULTS_PER_CALL]:
                url = _http(r.get("url")) if isinstance(r, dict) else None
                if url:
                    title = str(r.get("title") or "").strip() or urlparse(url).netloc
                    self._add({"kind": "web", "title": title, "url": url, "snippet": _snip(r.get("snippet")),
                               "app": plugin, "tool": name}, ("web", url))
            return
        if name in FOLDER_TOOLS and isinstance(result, dict):
            found = result.get("results") if name == "folder_search" else [result]
            files: set[str] = set()  # the first notes of a search (a note can come with two pieces)
            for r in found if isinstance(found, list) else []:
                path = r.get("file") if isinstance(r, dict) else None
                if not isinstance(path, str) or not path or (path not in files and len(files) >= FOLDER_RESULTS_PER_CALL):
                    continue
                files.add(path)
                link = r.get("link")
                self._add({
                    "kind": "file", "title": str(r.get("title") or r.get("path") or path), "path": path,
                    "folder": r.get("folder"), "url": link if isinstance(link, str) and link.startswith("obsidian://")
                    else None, "snippet": _snip(r.get("text") or r.get("content")), "app": plugin, "tool": name,
                }, ("file", path))
            return
        if name in FILE_TOOLS and isinstance(result, dict) and result.get("name"):
            self._add({"kind": "file", "title": str(result["name"]).rsplit("/", 1)[-1], "file": str(result["name"]),
                       "snippet": _snip(result.get("content")), "app": plugin, "tool": name},
                      ("file", str(result["name"])))
            return
        if is_app:
            for e in _emails(result):
                key = ("email", f"{plugin}:{e.get('id') or e.get('url') or e.get('subject')}")
                self._add({
                    "kind": "email", "title": str(e.get("subject") or "").strip() or "(no subject)",
                    "from": str(e.get("from") or e.get("sender_email") or "").strip(), "id": e.get("id"),
                    "url": _http(e.get("url")), "snippet": _snip(e.get("snippet") or e.get("body")),
                    "app": plugin, "tool": name,
                }, key)
            if len(self._items) == before:  # nothing more specific: the app itself is the source
                item = self._add({"kind": "app", "title": app_name or plugin, "app": plugin, "tool": name,
                                  "tools": []}, ("app", plugin))
                if item is not None and name not in item.setdefault("tools", []):
                    item["tools"].append(name)

    def items(self) -> list[dict]:
        return [dict(s) for s in self._items]

    def __len__(self) -> int:
        return len(self._items)
