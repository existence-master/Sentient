"""Reading memory out of other assistants' exports (issue #156, docs/API.md section 19).

``read(app, path)`` turns a ChatGPT or Claude data export (the zip, its unzipped folder or its ``conversations.json``)
or an OpenClaw workspace folder into an :class:`Export`: memories that are already one fact each (``saved``) and text
the fast model reads for facts (``sources``). Nothing here writes anything or talks to the network; the files are only
opened for reading, and only the ones named below.

Formats relied on (checked 2026-10-12):

**ChatGPT data export** (Settings > Data controls > Export data). A zip with ``conversations.json``, ``chat.html``,
``user.json``, ``message_feedback.json`` and more; only ``conversations.json`` is read. It is a list of
``{id | conversation_id, title, create_time, update_time, current_node, mapping: {node_id: {id, parent, children,
message: {author: {role}, create_time, recipient, content: {content_type, parts | text | ...}}}}}`` (times are Unix
seconds). The open branch is the chain from ``current_node`` up through ``parent``; edited-away branches are left out.
Saved memories show up in three ways (the shapes ChatGPT's own web client uses, github.com/pionxzh/chatgpt-exporter
``src/api.ts``):

- ``content_type: "model_editable_context"`` with ``model_set_context``: the saved memories the model saw when that
  conversation started, one per line (often ``1. [2025-03-02]. User lives in Pune.``). The newest one is used.
- an assistant message with ``recipient: "bio"``: a memory being saved ("User lives in Pune.").
- ``content_type: "user_editable_context"`` with ``user_profile`` and ``user_instructions``: custom instructions.

**Claude data export** (Settings > Privacy > Export data). A zip with ``conversations.json`` (a list of ``{uuid, name,
summary, created_at, updated_at, chat_messages: [{uuid, text, content: [{type: "text", text} | thinking | tool_use |
tool_result | ...], sender: "human" | "assistant", created_at, attachments, files}]}``), ``memories.json``
(``[{conversations_memory, project_memories: {project_uuid: text}, account_uuid}]``), ``projects/<uuid>.json`` (older
exports: one ``projects.json`` list) with ``{uuid, name, description, prompt_template, is_starter_project, docs:
[{filename, content}]}``, and ``users.json``. ``users.json`` is never read; project files (``docs``) are counted but
not read. Large accounts get several batch zips; each is imported on its own.

**OpenClaw workspace** (docs.openclaw.ai/concepts/memory and /concepts/user-model; default
``~/.openclaw/workspace``). ``MEMORY.md`` holds curated long-term memory (Markdown; each bullet or paragraph is one
memory), ``USER.md`` holds directives (a ``<!-- observed: YYYY-MM-DD | status: active | superseded -->`` line, then a
bullet such as ``- Prefer concise updates.``; superseded ones are skipped), and ``memory/YYYY-MM-DD.md`` or
``memory/YYYY-MM-DD-<slug>.md`` are daily notes. ``DREAMS.md``, ``memory/dreaming/``, ``memory/.dreams/``,
``memory/imports/``, ``openclaw.json`` and every other file are never read.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

APPS = {"chatgpt": "ChatGPT", "claude": "Claude", "openclaw": "OpenClaw"}
MAX_JSON_BYTES = 400_000_000  # conversations.json of a very large account; parsed in memory
MAX_TEXT_BYTES = 2_000_000
MESSAGE_CHARS = 1000  # of each message: long ones are mostly pasted documents, code or emails
SAVED_CHARS = 300  # a curated memory longer than this is read by the model instead of kept as one fact
NEVER_READ = ("user.json", "users.json", "openclaw.json", ".env")

_DAILY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:-(.+))?\.md$", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,4}[.)])\s+")
_DATE_PREFIX_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2})\]?[.:]?\s*")
_META_RE = re.compile(r"<!--\s*(.*?)\s*-->")
_FENCE_RE = re.compile(r"```[a-z]*\n?(.*?)```", re.DOTALL)
_MD_MARKS_RE = re.compile(r"\*\*|__|`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# a line that is only bold or italic text is a heading: "**Work context**", "*Recent months*" in Claude's memory
_BOLD_HEADING_RE = re.compile(r"^(\*\*|__|\*|_)[^*_]{1,80}\1:?$")
_KEY_VALUE_RE =re.compile(r"^([A-Za-z][\w /'-]{0,40}):\s*(.*)$")
_PREAMBLE_ENDS = ("User profile:", "about themselves:", "would like you to respond:")


class ExportError(ValueError):
    """The file or folder can't be used (a plain sentence for the user)."""


@dataclass
class Saved:
    """One memory as the other assistant kept it: imported as it is, without the model."""

    key: str
    kind: str  # "fact" | "insight"
    text: str
    part: str
    title: str  # where it sits: "Saved memories", "MEMORY.md", "USER.md"
    dated: str | None = None
    dimension: str = "context"  # for insights


@dataclass
class Source:
    """Text the fast model reads for lasting facts: a conversation's own messages, a note, a memory summary."""

    key: str  # stable across exports: remembered once read, so a later import skips it
    part: str
    title: str
    dated: str | None
    texts: list[str]


@dataclass
class Export:
    app: str
    path: Path
    saved: list[Saved] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return APPS[self.app]

    @property
    def name(self) -> str:
        """The export's file name, or for a folder its last two parts (".openclaw/workspace")."""
        if self.path.is_dir() and self.path.parent.name:
            return f"{self.path.parent.name}/{self.path.name}"
        return self.path.name


# ---------------------------------------------------------------------------- careful reading
class _Files:
    """Read-only access to the few files an export may hold, in a zip or a folder. Never through a symlink."""

    def __init__(self, path: Path):
        self.path = path
        self.zip: zipfile.ZipFile | None = None
        self.root: Path | None = None
        self.names: list[str] = []
        if path.is_file() and zipfile.is_zipfile(path):
            try:
                self.zip = zipfile.ZipFile(path)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ExportError(f"{path.name} can't be opened as a zip file.") from exc
            infos = [i for i in self.zip.infolist() if not i.is_dir()]
            # an export unzipped and zipped again often sits in one folder: read inside it
            self.names = [i.filename for i in infos]
            top = {PurePosixPath(n).parts[0] for n in self.names if len(PurePosixPath(n).parts) > 1}
            if len(top) == 1 and not any(len(PurePosixPath(n).parts) == 1 for n in self.names):
                self.prefix = f"{top.pop()}/"
            else:
                self.prefix = ""
        elif path.is_dir():
            self.root = path
            self.prefix = ""
        else:
            self.prefix = ""

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()

    def exists(self, name: str) -> bool:
        if self.zip is not None:
            return f"{self.prefix}{name}" in self.names
        return self.root is not None and self._file(name) is not None

    def _file(self, name: str) -> Path | None:
        assert self.root is not None
        p = self.root / name
        return p if p.is_file() and not p.is_symlink() and PurePosixPath(name).name not in NEVER_READ else None

    def list(self, folder: str) -> list[str]:
        """Names of the files directly inside ``folder`` (relative to the export)."""
        folder = folder.rstrip("/") + "/"
        if self.zip is not None:
            full = f"{self.prefix}{folder}"
            return sorted(n[len(self.prefix):] for n in self.names if n.startswith(full) and "/" not in n[len(full):])
        if self.root is None or not (self.root / folder).is_dir() or (self.root / folder).is_symlink():
            return []
        return sorted(f"{folder}{p.name}" for p in (self.root / folder).iterdir() if p.is_file() and not p.is_symlink())

    def read(self, name: str, limit: int) -> bytes | None:
        if PurePosixPath(name).name in NEVER_READ:
            return None
        if self.zip is not None:
            full = f"{self.prefix}{name}"
            if full not in self.names:
                return None
            info = self.zip.getinfo(full)
            if info.file_size > limit:
                raise ExportError(f"{PurePosixPath(name).name} in this export is too big to read ({info.file_size // 1_000_000} MB).")
            with self.zip.open(info) as fh:
                data = fh.read(limit + 1)
            if len(data) > limit:  # the zip lied about the size
                raise ExportError(f"{PurePosixPath(name).name} in this export is too big to read.")
            return data
        if self.root is None:
            return None
        p = self._file(name)
        if p is None:
            return None
        if p.stat().st_size > limit:
            raise ExportError(f"{p.name} is too big to read ({p.stat().st_size // 1_000_000} MB).")
        return p.read_bytes()

    def json(self, name: str) -> Any:
        raw = self.read(name, MAX_JSON_BYTES)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8-sig", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ExportError(f"{PurePosixPath(name).name} in this export isn't readable JSON.") from exc

    def text(self, name: str) -> str | None:
        try:
            raw = self.read(name, MAX_TEXT_BYTES)
        except ExportError:
            return None
        return raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n") if raw is not None else None


def _date(value: Any) -> str | None:
    """YYYY-MM-DD from Unix seconds or an ISO time, else None."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, int | float):
            return datetime.fromtimestamp(float(value), UTC).date().isoformat()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def clip(text: str, limit: int = MESSAGE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _paragraphs(text: str) -> list[str]:
    """Markdown as plain blocks: one per bullet or paragraph, headings and comments dropped."""
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(" ".join(buf))
            buf.clear()

    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if (in_fence or not stripped or stripped.startswith("#") or set(stripped) <= {"-", "=", "*", "_"}
                or _BOLD_HEADING_RE.match(stripped)):
            flush()
            continue
        if _LIST_RE.match(line):
            flush()
            buf.append(_LIST_RE.sub("", line, count=1).strip())
        else:
            buf.append(stripped)
    flush()
    return [_LINK_RE.sub(r"\1", _MD_MARKS_RE.sub("", p)).strip() for p in out if p.strip()]


def _frontmatter_free(text: str) -> str:
    return re.sub(r"^\ufeff?---\s*\n.*?\n---\s*\n", "", text or "", count=1, flags=re.DOTALL)


# ---------------------------------------------------------------------------- ChatGPT
def _chatgpt_text(content: dict) -> str:
    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join(p for p in parts if isinstance(p, str)).strip()
    return str(content.get("text") or "").strip()


def _branch(conv: dict) -> list[dict]:
    """The messages on the conversation's open branch, oldest first."""
    mapping = conv.get("mapping") if isinstance(conv.get("mapping"), dict) else {}
    node = conv.get("current_node")
    ids: list[str] = []
    seen: set[str] = set()
    if node in mapping:
        while node and node in mapping and node not in seen:
            seen.add(node)
            ids.append(node)
            node = (mapping[node] or {}).get("parent")
        ids.reverse()
    else:
        ids = sorted(mapping, key=lambda k: ((mapping[k] or {}).get("message") or {}).get("create_time") or 0)
    out = []
    for nid in ids:
        msg = (mapping.get(nid) or {}).get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), dict):
            out.append(msg)
    return out


def _after_preamble(text: str) -> str:
    """Custom instructions as the user typed them, without ChatGPT's wrapper around them."""
    text = text or ""
    fenced = _FENCE_RE.findall(text)
    if fenced:
        return "\n".join(f.strip() for f in fenced if f.strip())
    for marker in _PREAMBLE_ENDS:
        if marker in text:
            text = text.split(marker, 1)[1]
    return text.strip()


def saved_lines(text: str) -> list[tuple[str, str | None]]:
    """``model_set_context`` (or a pasted memory list) as ``(memory, date)`` pairs: numbering, bullets and a leading
    ``[YYYY-MM-DD].`` are taken off."""
    out: list[tuple[str, str | None]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _LIST_RE.sub("", line, count=1).strip()
        dated = None
        m = _DATE_PREFIX_RE.match(line)
        if m:
            dated, line = m.group(1), line[m.end():].strip()
        line = _MD_MARKS_RE.sub("", line).strip()
        if len(line) >= 6:
            out.append((line, dated))
    return out


def _read_chatgpt(files: _Files, export: Export) -> None:
    data = files.json("conversations.json")
    if data is None:
        raise ExportError("This doesn't look like a ChatGPT export: there's no conversations.json in it.")
    convs = data if isinstance(data, list) else data.get("conversations") if isinstance(data, dict) else None
    if not isinstance(convs, list) or (convs and not any(isinstance(c, dict) and "mapping" in c for c in convs)):
        raise ExportError("This conversations.json isn't from ChatGPT. Pick the ChatGPT export zip.")
    memory_ctx: tuple[float, str] | None = None
    profile: tuple[float, str, str] | None = None
    bio: list[tuple[str, str | None]] = []
    messages = 0
    for n, conv in enumerate(c for c in convs if isinstance(c, dict)):
        cid = str(conv.get("conversation_id") or conv.get("id") or f"n{n}")
        title = " ".join(str(conv.get("title") or "").split()) or "Untitled conversation"
        stamp = float(conv.get("update_time") or conv.get("create_time") or 0)
        texts: list[str] = []
        for msg in _branch(conv):
            content = msg["content"]
            kind = content.get("content_type")
            role = (msg.get("author") or {}).get("role")
            at = float(msg.get("create_time") or stamp or 0)
            if kind == "model_editable_context":
                ctx = str(content.get("model_set_context") or "").strip()
                if ctx and (memory_ctx is None or at >= memory_ctx[0]):
                    memory_ctx = (at, ctx)
            elif kind == "user_editable_context":
                about = _after_preamble(str(content.get("user_profile") or ""))
                how = _after_preamble(str(content.get("user_instructions") or ""))
                if (about or how) and (profile is None or at >= profile[0]):
                    profile = (at, about, how)
            elif role == "assistant" and msg.get("recipient") == "bio":
                text = _chatgpt_text(content)
                if text:
                    bio.append((text, _date(at)))
            elif role == "user" and kind in ("text", "multimodal_text"):
                text = _chatgpt_text(content)
                if text:
                    texts.append(clip(text))
        messages += len(texts)
        if texts:
            export.sources.append(Source(
                key=f"conv:{cid}:{len(texts)}", part="conversations", title=title,
                dated=_date(conv.get("create_time")), texts=texts,
            ))
    seen: set[str] = set()
    found = (saved_lines(memory_ctx[1]) if memory_ctx else []) + [(t, d) for b, d in bio for t, _ in saved_lines(b) or [(b, d)]]
    for text, dated in found:
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        export.saved.append(Saved(key=f"saved:{len(export.saved)}", kind="fact", text=text, part="memories",
                                  title="Saved memories", dated=dated))
    if profile is not None:
        parts = [p for p in (profile[1], f"How I'd like answers: {profile[2]}" if profile[2] else "") if p]
        export.sources.append(Source(key=f"profile:{_digest(' '.join(parts))}", part="profile",
                                     title="Custom instructions", dated=_date(profile[0]), texts=[clip(p, 3000) for p in parts]))
    export.stats.update(conversations=sum(1 for s in export.sources if s.part == "conversations"), messages=messages,
                        saved_memories=len(export.saved))


# ---------------------------------------------------------------------------- Claude
def _claude_text(msg: dict) -> str:
    blocks = msg.get("content")
    if isinstance(blocks, list):
        text = "\n".join(str(b.get("text") or "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            return text.strip()
    return str(msg.get("text") or "").strip()


def _claude_projects(files: _Files) -> list[dict]:
    found: list[dict] = []
    for name in files.list("projects"):
        if name.lower().endswith(".json"):
            data = files.json(name)
            found.extend(d for d in (data if isinstance(data, list) else [data]) if isinstance(d, dict))
    if files.exists("projects.json"):
        data = files.json("projects.json")
        found.extend(d for d in (data if isinstance(data, list) else [data]) if isinstance(d, dict))
    return found


def _read_claude(files: _Files, export: Export) -> None:
    convs = files.json("conversations.json") if files.exists("conversations.json") else None
    memories = files.json("memories.json") if files.exists("memories.json") else None
    projects = _claude_projects(files)
    if convs is None and memories is None and not projects:
        raise ExportError("This doesn't look like a Claude export: there's no conversations.json or memories.json in it.")
    if convs is not None and (not isinstance(convs, list) or (convs and not any(
        isinstance(c, dict) and "chat_messages" in c for c in convs
    ))):
        raise ExportError("This conversations.json isn't from Claude. Pick the Claude export zip.")
    names = {str(p.get("uuid")): " ".join(str(p.get("name") or "").split()) for p in projects}
    messages = 0
    for n, conv in enumerate(c for c in convs or [] if isinstance(c, dict)):
        cid = str(conv.get("uuid") or f"n{n}")
        title = " ".join(str(conv.get("name") or "").split()) or "Untitled conversation"
        texts = [clip(t) for t in (_claude_text(m) for m in conv.get("chat_messages") or []
                                    if isinstance(m, dict) and m.get("sender") == "human") if t]
        messages += len(texts)
        if texts:
            export.sources.append(Source(key=f"conv:{cid}:{len(texts)}", part="conversations", title=title,
                                         dated=_date(conv.get("created_at")), texts=texts))
    for record in memories if isinstance(memories, list) else [memories] if isinstance(memories, dict) else []:
        if not isinstance(record, dict):
            continue
        general = str(record.get("conversations_memory") or "").strip()
        if general:
            export.sources.append(Source(key=f"memory:{_digest(general)}", part="memories", title="Claude's memory",
                                         dated=None, texts=[clip(p, 1500) for p in _paragraphs(general)]))
        per_project = record.get("project_memories")
        for pid, text in (per_project.items() if isinstance(per_project, dict) else []):
            text = str(text or "").strip()
            if text:
                name = names.get(str(pid)) or "a project"
                export.sources.append(Source(key=f"memory:{pid}:{_digest(text)}", part="memories",
                                             title=f"Claude's memory for {name}", dated=None,
                                             texts=[clip(p, 1500) for p in _paragraphs(text)]))
    files_count = 0
    for p in projects:
        if p.get("is_starter_project"):
            continue
        docs = p.get("docs") if isinstance(p.get("docs"), list) else []
        files_count += len(docs)
        parts = [str(p.get(k) or "").strip() for k in ("description", "prompt_template")]
        parts = [x for x in parts if x]
        if parts:
            name = " ".join(str(p.get("name") or "").split()) or "Untitled project"
            export.sources.append(Source(key=f"project:{p.get('uuid')}:{_digest(' '.join(parts))}", part="projects",
                                         title=f"Project {name}", dated=_date(p.get("created_at")),
                                         texts=[clip(x, 2000) for x in parts]))
    export.stats.update(
        conversations=sum(1 for s in export.sources if s.part == "conversations"), messages=messages,
        projects=sum(1 for p in projects if not p.get("is_starter_project")), project_files=files_count,
    )


# ---------------------------------------------------------------------------- OpenClaw
_DIRECTIVE_VERBS = {
    "prefer": None, "prefers": None, "always": "always", "never": "never", "avoid": "avoid", "use": "use",
    "keep": "keep", "call": "call", "address": "address", "write": "write", "reply": "reply", "respond": "respond",
    "ask": "ask", "remind": "remind", "default": "default to", "don't": "not", "do": None, "skip": "skip",
    "include": "include", "send": "send", "check": "check", "mention": "mention", "suggest": "suggest",
    "speak": "speak", "summarize": "summarize", "format": "format", "start": "start", "end": "end", "stop": "stop",
}


def directive_to_insight(text: str, user: str) -> str:
    """An OpenClaw directive in third person: "Prefer concise updates" -> "Maya prefers concise updates.",
    "Never schedule calls before 10am" -> "Maya wants her assistant to never schedule calls before 10am."
    Text that isn't an instruction is kept as it is."""
    text = " ".join(text.split()).rstrip()
    name = user.strip() or "The user"
    first, _, rest = text.partition(" ")
    verb = first.lower().rstrip(",:")
    if verb not in _DIRECTIVE_VERBS or not rest:
        return text if text.endswith((".", "!", "?")) else text + "."
    if verb in ("prefer", "prefers"):
        out = f"{name} prefers {rest}"
    elif verb == "do" and rest.lower().startswith("not "):
        out = f"{name} wants the assistant not to {rest[4:]}"
    elif verb == "do":
        out = f"{name} wants the assistant to {rest}"
    elif verb == "don't":
        out = f"{name} wants the assistant not to {rest}"
    else:
        out = f"{name} wants the assistant to {_DIRECTIVE_VERBS[verb]} {rest}"
    return out if out.endswith((".", "!", "?")) else out + "."


def _openclaw_root(path: Path) -> Path:
    if (path / "workspace").is_dir() and not (path / "MEMORY.md").exists() and not (path / "USER.md").exists():
        return path / "workspace"
    return path


def _placeholder(entry: str) -> bool:
    """Template leftovers: "Prefer ...", "YYYY-MM-DD", a bare link."""
    return entry.rstrip().endswith("...") or "YYYY" in entry or bool(re.fullmatch(r"\[[^\]]*\]\([^)]*\)\.?", entry))


def _user_entries(text: str) -> list[tuple[str, str | None, bool]]:
    """USER.md bullets as ``(entry, observed date, active)``. A ``<!-- observed: ... | status: ... -->`` line applies
    to the bullets after it. Prose (the template's own explanation) and code blocks (its example) are not entries."""
    out: list[tuple[str, str | None, bool]] = []
    dated: str | None = None
    active = True
    item: list[str] = []
    meta_now: tuple[str | None, bool] = (None, True)

    def flush() -> None:
        if item:
            raw = _MD_MARKS_RE.sub("", " ".join(item)).strip()
            entry = _LINK_RE.sub(r"\1", raw)
            if entry and not _placeholder(raw):
                out.append((entry, *meta_now))
            item.clear()

    in_fence = False
    for line in _frontmatter_free(text).split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        meta = _META_RE.search(stripped)
        if meta:
            flush()
            fields = {k.strip().lower(): v.strip().lower() for k, _, v in
                      (part.partition(":") for part in meta.group(1).split("|"))}
            observed = fields.get("observed", "")
            dated = observed if re.fullmatch(r"\d{4}-\d{2}-\d{2}", observed) else None
            active = fields.get("status", "active") != "superseded"
            continue
        if _LIST_RE.match(line):
            flush()
            meta_now = (dated, active)
            item.append(_LIST_RE.sub("", line, count=1).strip())
        elif stripped and item and line[:1].isspace():
            item.append(stripped)  # a wrapped bullet
        else:
            flush()
    flush()
    return out


def _read_openclaw(path: Path, export: Export, user: str) -> None:
    if not path.is_dir():
        raise ExportError("Pick your OpenClaw workspace folder (it has MEMORY.md in it), not a file.")
    root = _openclaw_root(path)
    files = _Files(root)
    memory_name = "MEMORY.md" if files.exists("MEMORY.md") else "memory.md" if files.exists("memory.md") else None
    daily = [n for n in files.list("memory") if _DAILY_RE.match(PurePosixPath(n).name)]
    if memory_name is None and not files.exists("USER.md") and not daily:
        raise ExportError(f"{path} doesn't look like an OpenClaw workspace. Pick the folder that has MEMORY.md in it "
                          "(usually .openclaw/workspace in your home folder).")
    export.path = root
    if memory_name:
        long_texts: list[str] = []
        for entry in _paragraphs(_frontmatter_free(files.text(memory_name) or "")):
            if _placeholder(entry):
                continue
            if len(entry) > SAVED_CHARS:
                long_texts.append(clip(entry, 1500))
            else:
                export.saved.append(Saved(key=f"memory:{len(export.saved)}", kind="fact", text=entry, part="memories",
                                          title="MEMORY.md"))
        if long_texts:
            export.sources.append(Source(key=f"memory:{_digest(' '.join(long_texts))}", part="memories",
                                         title="MEMORY.md", dated=None, texts=long_texts))
    superseded = 0
    profile_texts: list[str] = []
    for n, (entry, dated, active) in enumerate(_user_entries(files.text("USER.md") or "")):
        if not active:
            superseded += 1
            continue
        directive = entry.split(" ", 1)[0].lower().rstrip(",:") in _DIRECTIVE_VERBS
        kv = None if directive else _KEY_VALUE_RE.match(entry)
        if kv:  # an older USER.md: "Name: Maya", "Timezone: Asia/Kolkata"; empty ones are template leftovers
            if kv.group(2).strip():
                profile_texts.append(entry)
            continue
        export.saved.append(Saved(key=f"user:{n}", kind="insight", text=directive_to_insight(entry, user),
                                  part="profile", title="USER.md", dated=dated,
                                  dimension="preferences" if directive else "context"))
    if profile_texts:
        export.sources.append(Source(key=f"profile:{_digest(' '.join(profile_texts))}", part="profile",
                                     title="USER.md", dated=None, texts=profile_texts))
    for name in daily:
        m = _DAILY_RE.match(PurePosixPath(name).name)
        assert m is not None
        text = files.text(name) or ""
        lines = [p for p in _paragraphs(_frontmatter_free(text)) if not p.lower().startswith("current time:")]
        if lines:
            slug = f" ({m.group(2).replace('-', ' ')})" if m.group(2) else ""
            export.sources.append(Source(key=f"note:{PurePosixPath(name).name}:{_digest(text)}", part="notes",
                                         title=f"Daily note {m.group(1)}{slug}", dated=m.group(1),
                                         texts=[clip(x) for x in lines]))
    export.stats.update(memories=sum(1 for s in export.saved if s.part == "memories"), superseded=superseded,
                        notes=sum(1 for s in export.sources if s.part == "notes"))


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]  # a change marker, not security


# ---------------------------------------------------------------------------- public
def default_path(app: str) -> Path | None:
    """Where the app keeps its memory on this computer, when it does (only OpenClaw; the others are downloads)."""
    return Path.home() / ".openclaw" / "workspace" if app == "openclaw" else None


def read(app: str, path: str | None, *, user_name: str = "") -> Export:
    """Parse an export. Raises :class:`ExportError` with a plain sentence when it can't be used."""
    if app not in APPS:
        raise ExportError("Pick ChatGPT, Claude or OpenClaw.")
    raw = str(path or "").strip()
    target = Path(raw).expanduser() if raw else default_path(app)
    if target is None:
        raise ExportError(f"Pick your {APPS[app]} export file.")
    if not target.exists():
        raise ExportError(f"There's nothing at {target}.")
    export = Export(app=app, path=target)
    if app == "openclaw":
        _read_openclaw(target, export, user_name)
        return export
    if target.is_file() and target.name.lower().endswith(".json") and not zipfile.is_zipfile(target):
        target = target.parent  # their conversations.json: read the folder it sits in
    files = _Files(target)
    try:
        if files.zip is None and files.root is None:
            raise ExportError(f"Pick the {APPS[app]} export zip (or the folder you unzipped it into).")
        (_read_chatgpt if app == "chatgpt" else _read_claude)(files, export)
    finally:
        files.close()
    return export
