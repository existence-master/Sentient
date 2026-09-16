"""Fetch a web page and return its readable text."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from sentient.files.extract import extract_text
from sentient.integrations.base import IntegrationError, IntegrationPlugin, itool, manager_from
from sentient.integrations.common import http_client, truncate
from sentient.tools.base import ToolContext

SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe",
             "template", "button", "select", "canvas"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "main", "h1", "h2", "h3",
              "h4", "h5", "h6", "blockquote", "pre", "hr", "dd", "dt", "figcaption"}
MAX_DOWNLOAD = 15 * 1024 * 1024


class _Readable(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
            if tag in {"h1", "h2", "h3"}:
                self.parts.append("#" * int(tag[1]) + " ")
            elif tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.split("\n")]
        out: list[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip()


def html_to_text(html: str) -> tuple[str, str]:
    p = _Readable()
    p.feed(html)
    p.close()
    return p.title.strip(), p.text()


def _check_url(url: str) -> str:
    parts = urlsplit(url if "://" in url else f"https://{url}")
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise IntegrationError("Only http and https web addresses can be fetched.")
    host = parts.hostname
    if host in {"localhost"} or host.endswith(".localhost"):
        raise IntegrationError("Fetching addresses on this computer isn't allowed.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise IntegrationError(f"Couldn't find the website {host}.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            raise IntegrationError("Fetching addresses on this computer or network internals isn't allowed.")
    return parts.geturl()


@itool("web", "web_fetch")
async def web_fetch(ctx: ToolContext, url: str, max_chars: int | None = None) -> dict:
    """Open a web page (or an online PDF/text file) and return its readable text without menus, ads and scripts.
    Use after web_search to read a result, or when the user gives you a link."""
    mgr = manager_from(ctx)
    limit = int(max_chars or mgr.app.config.integrations.web_fetch_max_chars)
    target = await asyncio.to_thread(_check_url, url.strip())
    headers = {"Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5",
               "Accept-Language": "en;q=0.9"}
    async with http_client(timeout=30, headers=headers) as http:
        r = await http.get(target)
    r.raise_for_status()
    if len(r.content) > MAX_DOWNLOAD:
        raise IntegrationError("That file is too large to read.")
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    title = ""
    if "html" in ctype or (not ctype and r.text.lstrip().startswith("<")):
        title, text = html_to_text(r.text)
    elif ctype == "application/pdf" or str(r.url).lower().endswith(".pdf"):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "page.pdf"
            p.write_bytes(r.content)
            text = await asyncio.to_thread(extract_text, p, limit + 1000)
    elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
        text = r.text
    else:
        raise IntegrationError(f"That link is a {ctype or 'binary'} file, which can't be read as text.")
    text, cut = truncate(text, limit)
    return {"url": str(r.url), "title": title, "content": text, "truncated": cut}


class WebPlugin(IntegrationPlugin):
    id = "web"
    display_name = "Web Pages"
    description = (
        "Opens any web page or online document and reads it for you, skipping menus, ads and scripts. "
        "Built in, nothing to set up."
    )
    category = "knowledge"
    icon = "globe"
    auth_type = "builtin"
    selection_hint = "Use to open and read a specific web page or link (for example a search result)."
    tools = [web_fetch]


PLUGIN = WebPlugin()
