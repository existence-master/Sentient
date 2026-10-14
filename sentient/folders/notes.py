"""Read one file from a folder the user added and cut it into searchable pieces (#176).

Pure functions that run in a worker thread. Markdown is split at its headings, then into pieces of about
``chunk_chars``; front matter becomes the note's tags, aliases and source. Every piece starts with the note's title
and heading, so a piece found on its own still says where it came from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml

from sentient.files.extract import extract_text
from sentient.folders.scan import MARKDOWN

MAX_TEXT_CHARS = 2_000_000  # one file's text; a longer one is read only this far
MAX_CHUNKS = 400  # pieces from one file
_FRONT = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TAG = re.compile(r"(?<![\w/&#])#([A-Za-z][\w/-]*)")
_WIKILINK = re.compile(r"!?\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|([^\]]+))?\]\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
CLIP_TAGS = frozenset({"clippings", "clipping", "clipped", "webclip", "web-clip", "web-clipping", "readlater"})
CLIP_FOLDERS = frozenset({"clippings", "web clippings", "webclips"})
SOURCE_KEYS = ("source", "url", "clipped_from", "source_url", "original_url", "link")


@dataclass
class Note:
    title: str
    text: str  # the body without front matter
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    clipped: str = ""  # the address a clipped web page came from, "web" when only its tag or folder says so


@dataclass
class Piece:
    heading: str
    text: str


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,\s]+", value) if v.strip()]
    if isinstance(value, list | tuple | set):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        data = None
    return (data if isinstance(data, dict) else {}), text[m.end():]


def _clip_source(front: dict, tags: list[str], rel: str) -> str:
    for key in SOURCE_KEYS:
        v = front.get(key)
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, str) and re.match(r"https?://", v.strip(), re.I):
            return v.strip()
    if {t.lower().rsplit("/", 1)[-1] for t in tags} & CLIP_TAGS:
        return "web"
    parts = rel.lower().split("/")[:-1]
    return "web" if any(p in CLIP_FOLDERS for p in parts) else ""


def read_note(path: Path, rel: str) -> Note:
    """The note's text, title, tags, aliases, links and whether it is a clipped web page. Raises ``ValueError`` or
    ``OSError`` when it can't be read."""
    suffix = path.suffix.lower()
    raw = extract_text(path, MAX_TEXT_CHARS)
    title = path.stem
    if suffix not in MARKDOWN:
        return Note(title=title, text=raw, clipped=_clip_source({}, [], rel))
    front, body = _front_matter(raw)
    tags = [t.lstrip("#") for t in _as_list(front.get("tags") or front.get("tag"))]
    in_body = [m.group(1) for m in _TAG.finditer(_strip_code(body))]
    tags = list(dict.fromkeys(tags + in_body))[:40]
    aliases = _as_list(front.get("aliases") or front.get("alias")) if not isinstance(front.get("aliases"), str) \
        else [front["aliases"].strip()]
    links = list(dict.fromkeys(m.group(1).strip() for m in _WIKILINK.finditer(body)))[:100]
    if isinstance(front.get("title"), str) and front["title"].strip():
        title = front["title"].strip()
    return Note(title=title, text=body, tags=tags, aliases=aliases[:10], links=links,
                clipped=_clip_source(front, tags, rel))


def _strip_code(text: str) -> str:
    out, fenced = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def readable(text: str) -> str:
    """Markdown as plain reading text: ``[[Note|label]]`` becomes ``label``, ``[[Note]]`` ``Note``."""
    return _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)


def _sections(text: str, markdown: bool) -> list[tuple[str, str]]:
    """``[(heading path, text)]``: markdown split at headings outside code blocks; other text is one section."""
    if not markdown:
        return [("", text)]
    out: list[tuple[str, str]] = []
    trail: list[tuple[int, str]] = []
    buf: list[str] = []
    fenced = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((" > ".join(h for _, h in trail), body))
        buf.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            trail = [(lv, h) for lv, h in trail if lv < level] + [(level, m.group(2).strip())]
            continue
        buf.append(line)
    flush()
    return out


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    out, cur = [], ""
    for sentence in _SENTENCE.split(paragraph):
        while len(sentence) > size:  # one very long "sentence" (a table, a minified line)
            if cur:
                out.append(cur)
                cur = ""
            out.append(sentence[:size])
            sentence = sentence[size:]
        if cur and len(cur) + 1 + len(sentence) > size:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def pieces(note: Note, *, markdown: bool, size: int = 1200) -> list[Piece]:
    """The note cut into pieces of about ``size`` characters, at most ``MAX_CHUNKS``."""
    out: list[Piece] = []
    for heading, body in _sections(readable(note.text) if markdown else note.text, markdown):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        cur = ""
        for para in paragraphs:
            for part in _split_long(para, size):
                if cur and len(cur) + 2 + len(part) > size:
                    out.append(Piece(heading, cur))
                    cur = part
                else:
                    cur = f"{cur}\n\n{part}" if cur else part
        if cur:
            out.append(Piece(heading, cur))
        if len(out) >= MAX_CHUNKS:
            return out[:MAX_CHUNKS]
    return out


def embed_text(title: str, piece: Piece, tags: list[str] | None = None) -> str:
    """What is embedded for one piece: where it is, then the text."""
    head = title + (f" > {piece.heading}" if piece.heading else "")
    tag_line = f"\nTags: {', '.join(tags)}" if tags else ""
    return f"{head}{tag_line}\n{piece.text}"


def obsidian_url(vault: str, rel: str) -> str:
    """``obsidian://open`` for a note in a vault; Obsidian takes the path without ``.md``."""
    file = rel[:-3] if rel.lower().endswith(".md") else rel
    return f"obsidian://open?vault={quote(vault, safe='')}&file={quote(file, safe='')}"
