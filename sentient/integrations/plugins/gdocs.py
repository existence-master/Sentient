"""Google Docs: create from markdown/plain text, read, append, replace."""

from __future__ import annotations

import re
from typing import Any

from sentient.integrations.base import IntegrationError, itool
from sentient.integrations.common import truncate
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext

API = "https://docs.googleapis.com/v1/documents"
PID = "gdocs"


def u16(s: str) -> int:
    """Docs indexes count UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2


def markdown_to_requests(markdown: str, start: int, *, as_markdown: bool = True) -> list[dict]:
    """Build batchUpdate requests inserting ``markdown`` at ``start`` with headings, bullets and bold."""
    if not as_markdown:
        text = markdown if markdown.endswith("\n") else markdown + "\n"
        return [{"insertText": {"location": {"index": start}, "text": text}}]
    lines_out: list[str] = []
    paragraph_styles: list[tuple[int, int, str]] = []
    bullets: list[tuple[int, int, str]] = []
    bold: list[tuple[int, int]] = []
    offset = 0
    for raw in markdown.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        style = None
        bullet = None
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            style = "TITLE" if len(m.group(1)) == 1 and offset == 0 else f"HEADING_{min(len(m.group(1)), 3)}"
            line = m.group(2)
        elif re.match(r"^\s*[-*+]\s+", line):
            bullet, line = "BULLET_DISC_CIRCLE_SQUARE", re.sub(r"^\s*[-*+]\s+", "", line)
        elif re.match(r"^\s*\d+[.)]\s+", line):
            bullet, line = "NUMBERED_DECIMAL_ALPHA_ROMAN", re.sub(r"^\s*\d+[.)]\s+", "", line)
        plain = ""
        pos = 0
        for bm in re.finditer(r"\*\*(.+?)\*\*", line):
            plain += line[pos:bm.start()]
            b0 = offset + u16(plain)
            plain += bm.group(1)
            bold.append((b0, offset + u16(plain)))
            pos = bm.end()
        plain += line[pos:]
        seg = plain + "\n"
        length = u16(seg)
        if style:
            paragraph_styles.append((offset, offset + length, style))
        if bullet:
            bullets.append((offset, offset + length, bullet))
        lines_out.append(seg)
        offset += length
    text = "".join(lines_out)
    reqs: list[dict] = [{"insertText": {"location": {"index": start}, "text": text}}]
    for a, b, style in paragraph_styles:
        reqs.append({"updateParagraphStyle": {"range": {"startIndex": start + a, "endIndex": start + b},
                                              "paragraphStyle": {"namedStyleType": style}, "fields": "namedStyleType"}})
    for a, b in bold:
        reqs.append({"updateTextStyle": {"range": {"startIndex": start + a, "endIndex": start + b},
                                         "textStyle": {"bold": True}, "fields": "bold"}})
    # bullets last and in reverse so earlier ranges keep their indexes
    for a, b, preset in reversed(bullets):
        reqs.append({"createParagraphBullets": {"range": {"startIndex": start + a, "endIndex": start + b},
                                                "bulletPreset": preset}})
    return reqs


def document_text(doc: dict) -> str:
    out: list[str] = []

    def walk(content: list[dict]) -> None:
        for el in content or []:
            if "paragraph" in el:
                para = el["paragraph"]
                text = "".join((e.get("textRun") or {}).get("content", "") for e in para.get("elements") or [])
                style = (para.get("paragraphStyle") or {}).get("namedStyleType", "")
                if style.startswith("HEADING_") and text.strip():
                    text = "#" * int(style[-1]) + " " + text
                elif style == "TITLE" and text.strip():
                    text = "# " + text
                elif para.get("bullet") and text.strip():
                    text = "- " + text
                out.append(text)
            elif "table" in el:
                for row in el["table"].get("tableRows") or []:
                    cells = []
                    for cell in row.get("tableCells") or []:
                        sub: list[str] = []
                        for c in cell.get("content") or []:
                            for e in (c.get("paragraph") or {}).get("elements") or []:
                                sub.append((e.get("textRun") or {}).get("content", ""))
                        cells.append("".join(sub).strip())
                    out.append(" | ".join(cells) + "\n")
            elif "tableOfContents" in el:
                walk(el["tableOfContents"].get("content") or [])

    walk((doc.get("body") or {}).get("content") or [])
    return "".join(out).strip()


def _url(doc_id: str) -> str:
    return f"https://docs.google.com/document/d/{doc_id}/edit"


async def _end_index(ctx: ToolContext, document_id: str) -> int:
    doc = await gapi(ctx, PID, "GET", f"{API}/{document_id}", params={"fields": "body(content(endIndex))"})
    content = (doc.get("body") or {}).get("content") or []
    return max(1, int(content[-1].get("endIndex", 2)) - 1) if content else 1


@itool(PID, "gdocs_create_document", risk=Risk.write)
async def gdocs_create_document(ctx: ToolContext, title: str, content: str = "", markdown: bool = True) -> dict:
    """Create a Google Doc. `content` may use simple markdown (# headings, - bullets, 1. numbered lists, **bold**),
    which becomes real Docs formatting. Returns the document id and link."""
    doc = await gapi(ctx, PID, "POST", API, json={"title": title})
    doc_id = doc["documentId"]
    if content.strip():
        await gapi(ctx, PID, "POST", f"{API}/{doc_id}:batchUpdate",
                   json={"requests": markdown_to_requests(content, 1, as_markdown=markdown)})
    return {"document_id": doc_id, "title": title, "url": _url(doc_id)}


@itool(PID, "gdocs_read_document")
async def gdocs_read_document(ctx: ToolContext, document_id: str, max_chars: int = 30000) -> dict:
    """Read a Google Doc as text (headings shown with #, bullets with -, tables as | rows)."""
    doc = await gapi(ctx, PID, "GET", f"{API}/{document_id}")
    text, cut = truncate(document_text(doc), max(1000, int(max_chars or 30000)))
    return {"document_id": document_id, "title": doc.get("title"), "content": text, "truncated": cut,
            "url": _url(document_id)}


@itool(PID, "gdocs_append_text", risk=Risk.write)
async def gdocs_append_text(ctx: ToolContext, document_id: str, text: str, markdown: bool = True) -> dict:
    """Add text to the end of a Google Doc (simple markdown supported, like gdocs_create_document)."""
    if not text.strip():
        raise IntegrationError("There's no text to add.")
    start = await _end_index(ctx, document_id)
    body = text if text.startswith("\n") else "\n" + text
    await gapi(ctx, PID, "POST", f"{API}/{document_id}:batchUpdate",
               json={"requests": markdown_to_requests(body, start, as_markdown=markdown)})
    return {"document_id": document_id, "appended_chars": len(text), "url": _url(document_id)}


@itool(PID, "gdocs_replace_text", risk=Risk.write)
async def gdocs_replace_text(ctx: ToolContext, document_id: str, find: str, replace_with: str,
                             match_case: bool = False) -> dict:
    """Replace every occurrence of `find` with `replace_with` in a Google Doc."""
    res = await gapi(ctx, PID, "POST", f"{API}/{document_id}:batchUpdate", json={"requests": [{
        "replaceAllText": {"containsText": {"text": find, "matchCase": bool(match_case)}, "replaceText": replace_with}}]})
    replies: list[Any] = res.get("replies") or [{}]
    changed = ((replies[0] or {}).get("replaceAllText") or {}).get("occurrencesChanged", 0)
    return {"document_id": document_id, "occurrences_changed": changed, "url": _url(document_id)}


class GDocsPlugin(GooglePlugin):
    id = PID
    display_name = "Google Docs"
    description = (
        "Write and edit Google Docs. Sentient can create formatted documents from its drafts (headings, bullet "
        "lists, bold), read existing documents, add sections to the end, and find-and-replace text."
    )
    category = "productivity"
    icon = "google-docs"
    api_name = "Google Docs API"
    api_slug = "docs.googleapis.com"
    selection_hint = "Use to create, read or edit documents in Google Docs."
    tools = [gdocs_create_document, gdocs_read_document, gdocs_append_text, gdocs_replace_text]


PLUGIN = GDocsPlugin()
