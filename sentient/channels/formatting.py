"""Text formatting for messaging apps.

- ``markdown_to_telegram_html``: the model writes markdown; Telegram's HTML parse mode supports a small
  tag set (b, i, s, u, code, pre, a, blockquote). Everything that is not a recognised construct is escaped,
  and the output is checked for balanced tags (callers fall back to plain text otherwise).
- ``split_markdown``: cuts long replies on paragraph and line boundaries, keeping code fences balanced.
- ``markdown_to_whatsapp``: WhatsApp's own markup (*bold*, _italic_, ~strike~, ```code```); links become
  "label (url)" because WhatsApp has no link syntax.
- ``summary_text``: short, link-free text for delivered notifications.
"""

from __future__ import annotations

import html
import re

TELEGRAM_LIMIT = 4096
DISCORD_LIMIT = 2000
WHATSAPP_LIMIT = 4000  # WhatsApp allows far more, but long messages are hard to read on a phone

_ALLOWED_TAGS = {"b", "i", "s", "u", "code", "pre", "a", "blockquote"}
_SAFE_SCHEMES = ("http://", "https://", "mailto:", "tg://")

_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([\w+#.-]*)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_TASK_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")

_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.S)
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
_BOLD_STAR_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S)
_BOLD_UNDER_RE = re.compile(r"(?<![\w])__(?=\S)(.+?)(?<=\S)__(?![\w])", re.S)
_ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*(?=[^\s*])([^*\n<>]+?)(?<=[^\s*])\*(?![*\w])")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w])_(?=[^\s_])([^_\n<>]+?)(?<=[^\s_])_(?![\w])")
_STRIKE_RE = re.compile(r"~~(?=\S)([^~\n<>]+?)(?<=\S)~~")
_PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")
_TAG_RE = re.compile(r"<(/?)([a-z]+)(?:\s[^>]*)?>")
_URL_RE = re.compile(r"https?://\S+")


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def _inline(text: str) -> str:
    """Convert inline markdown in one block of text to Telegram HTML."""
    holders: list[str] = []

    def hold(fragment: str) -> str:
        holders.append(fragment)
        return f"\x00{len(holders) - 1}\x00"

    text = text.replace("\x00", "")
    text = _CODE_SPAN_RE.sub(lambda m: hold(f"<code>{escape(m.group(2).strip() or m.group(2))}</code>"), text)

    def link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if not url.lower().startswith(_SAFE_SCHEMES):
            return hold(escape(label))
        return hold(f'<a href="{html.escape(url, quote=True)}">{_emphasis(escape(label))}</a>')

    text = _LINK_RE.sub(link, text)
    text = _AUTOLINK_RE.sub(lambda m: hold(f'<a href="{html.escape(m.group(1), quote=True)}">{escape(m.group(1))}</a>'), text)
    text = _emphasis(escape(text))
    for _ in range(3):  # placeholders may nest (a link inside bold text)
        text = _PLACEHOLDER_RE.sub(lambda m: holders[int(m.group(1))], text)
    return text


def _emphasis(text: str) -> str:
    text = _BOLD_STAR_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BOLD_UNDER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _ITALIC_STAR_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _ITALIC_UNDER_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    return _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", text)


def markdown_to_telegram_html(md: str) -> str:
    lines = (md or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_RE.match(line)
        if fence:
            marker, lang = fence.group(1), fence.group(2)
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence (or end of text while streaming)
            code = escape("\n".join(body))
            cls = f' class="language-{escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{code}</code></pre>")
            continue
        if _TABLE_RE.match(line):
            rows = []
            while i < len(lines) and _TABLE_RE.match(lines[i]):
                if not re.fullmatch(r"\s*\|[\s:|-]+\|\s*", lines[i]):
                    rows.append(lines[i].strip())
                i += 1
            out.append("<pre>" + escape("\n".join(rows)) + "</pre>")
            continue
        quote = _QUOTE_RE.match(line)
        if quote:
            quoted = []
            while i < len(lines) and (m := _QUOTE_RE.match(lines[i])):
                quoted.append(_inline(m.group(1)))
                i += 1
            out.append("<blockquote>" + "\n".join(quoted) + "</blockquote>")
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            out.append(f"<b>{_inline(heading.group(1))}</b>")
        elif _HR_RE.match(line):
            out.append("──────────")
        elif bullet := _BULLET_RE.match(line):
            indent, item = bullet.group(1), bullet.group(2)
            task = _TASK_RE.match(item)
            mark = ("☑" if task.group(1).lower() == "x" else "☐") if task else "•"
            depth = len(indent.replace("\t", "  ")) // 2
            out.append("  " * depth + f"{mark} {_inline(task.group(2) if task else item)}")
        else:
            out.append(_inline(line))
        i += 1
    rendered = "\n".join(out)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered if tags_balanced(rendered) else escape(md or "").strip()


def tags_balanced(fragment: str) -> bool:
    stack: list[str] = []
    for m in _TAG_RE.finditer(fragment):
        closing, name = m.group(1) == "/", m.group(2)
        if name not in _ALLOWED_TAGS:
            return False
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def html_to_plain(fragment: str) -> str:
    """Strip tags from Telegram HTML (fallback when Telegram rejects the markup)."""
    text = re.sub(r'<a href="([^"]*)">(.*?)</a>', lambda m: f"{m.group(2)} ({html.unescape(m.group(1))})", fragment)
    return html.unescape(_TAG_RE.sub("", text))


def split_markdown(md: str, limit: int) -> list[str]:
    """Split markdown into pieces of at most ``limit`` characters, preferring paragraph, then line,
    then word boundaries. Code fences cut in the middle are closed and reopened."""
    md = (md or "").strip()
    if len(md) <= limit:
        return [md] if md else []
    limit = max(limit, 40)
    pieces: list[str] = []
    rest = md
    reopen = ""
    while rest:
        rest = reopen + rest
        reopen = ""
        if len(rest) <= limit:
            pieces.append(rest)
            break
        budget = limit - 8  # room to close a fence
        cut = rest.rfind("\n\n", 0, budget)
        if cut < budget // 3:
            cut = rest.rfind("\n", 0, budget)
        if cut < budget // 3:
            cut = rest.rfind(" ", 0, budget)
        if cut < budget // 3:
            cut = budget
        head, rest = rest[:cut].rstrip(), rest[cut:].lstrip("\n")
        if rest.startswith(" "):
            rest = rest.lstrip(" ")
        fences = [m for m in re.finditer(r"^\s*(```|~~~)([\w+#.-]*)\s*$", head, re.M)]
        if len(fences) % 2 == 1:  # inside a code block: close it here, reopen in the next piece
            opener = fences[-1]
            head += "\n" + opener.group(1)
            reopen = f"{opener.group(1)}{opener.group(2)}\n"
        pieces.append(head)
    return [p for p in pieces if p.strip()]


def render_telegram_chunks(md: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Markdown to one or more Telegram HTML messages, each within ``limit`` characters."""
    chunks: list[str] = []
    for piece in split_markdown(md, limit - 400):
        rendered = markdown_to_telegram_html(piece)
        if len(rendered) <= limit:
            if rendered:
                chunks.append(rendered)
            continue
        # escaping grew the text past the limit: split this piece smaller
        smaller = max(200, len(piece) * limit // (len(rendered) + 1) - 100)
        if smaller >= len(piece):
            smaller = len(piece) // 2
        chunks.extend(render_telegram_chunks(piece, limit) if smaller < 200 else
                      [c for p in split_markdown(piece, smaller) for c in render_telegram_chunks(p, limit)])
    return chunks


def strip_links(md: str) -> str:
    md = _LINK_RE.sub(lambda m: m.group(1), md or "")
    md = _AUTOLINK_RE.sub("", md)
    return re.sub(r"[ \t]{2,}", " ", _URL_RE.sub("", md)).strip()


def summary_text(md: str, max_chars: int = 700) -> str:
    """A short, link-free version of a notification message (markdown kept)."""
    text = strip_links(md)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    text = text[: cut if cut > max_chars // 2 else max_chars].rstrip()
    if text.count("```") % 2 == 1:
        text += "\n```"
    return text + " …"


def escape_discord(text: str) -> str:
    return re.sub(r"([\\*_~`|>])", r"\\\1", text or "")


def _whatsapp_inline(text: str) -> str:
    holders: list[str] = []

    def hold(fragment: str) -> str:
        holders.append(fragment)
        return f"\x00{len(holders) - 1}\x00"

    text = text.replace("\x00", "").replace("\x01", "")
    text = _CODE_SPAN_RE.sub(lambda m: hold(f"`{m.group(2).strip() or m.group(2)}`"), text)

    def link(m: re.Match) -> str:
        label, url = m.group(1).strip(), m.group(2)
        return hold(url if label == url else f"{label} ({url})")

    text = _LINK_RE.sub(link, text)
    text = _AUTOLINK_RE.sub(lambda m: hold(m.group(1)), text)
    text = _BOLD_STAR_RE.sub(lambda m: f"\x01{m.group(1)}\x01", text)  # bold is *one* star in WhatsApp
    text = _BOLD_UNDER_RE.sub(lambda m: f"\x01{m.group(1)}\x01", text)
    text = _ITALIC_STAR_RE.sub(lambda m: f"_{m.group(1)}_", text)
    text = _STRIKE_RE.sub(lambda m: f"~{m.group(1)}~", text)
    text = text.replace("\x01", "*")
    return _PLACEHOLDER_RE.sub(lambda m: holders[int(m.group(1))], text)


def markdown_to_whatsapp(md: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append("```")  # WhatsApp would show a language tag as text: drop it
            continue
        if in_fence:
            out.append(line)
        elif heading := _HEADING_RE.match(line):
            out.append(f"*{_whatsapp_inline(heading.group(1))}*")
        elif _HR_RE.match(line):
            out.append("──────────")
        elif bullet := _BULLET_RE.match(line):
            out.append(f"{bullet.group(1)}- {_whatsapp_inline(bullet.group(2))}")
        else:
            out.append(_whatsapp_inline(line))
    return "\n".join(out).strip()


def render_whatsapp_chunks(md: str, limit: int = WHATSAPP_LIMIT) -> list[str]:
    """Markdown to one or more WhatsApp messages (links grow a little, so split with some room)."""
    return [c for c in (markdown_to_whatsapp(p) for p in split_markdown(md, limit - 300)) if c]
