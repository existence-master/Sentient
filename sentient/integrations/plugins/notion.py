"""Notion with an internal integration token."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from sentient.integrations.base import IntegrationError, IntegrationPlugin, SetupField, creds, itool
from sentient.integrations.common import http_client, truncate
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://api.notion.com/v1"
PID = "notion"
VERSION = "2022-06-28"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Notion-Version": VERSION}


async def notion(ctx: ToolContext | None, method: str, path: str, *, token: str | None = None,
                 params: dict | None = None, json: Any = None) -> dict:
    if token is None:
        assert ctx is not None
        token = (await creds(ctx, PID))["token"]
    async with http_client(headers=_headers(token)) as http:
        r = await http.request(method, f"{API}{path}", params=params, json=json)
    if r.status_code == 404:
        raise IntegrationError("Notion couldn't find that page. Share it with the Sentient integration "
                               "(page menu → Connections → Sentient) and try again.")
    if r.status_code == 401:
        raise IntegrationError("Notion rejected the saved token. Reconnect Notion.")
    if r.status_code == 400:
        raise IntegrationError(f"Notion refused the request: {r.json().get('message', r.text[:200])}")
    r.raise_for_status()
    return r.json()


def _plain(rich: list[dict] | None) -> str:
    return "".join(t.get("plain_text", "") for t in rich or [])


def _rich(text: str) -> list[dict]:
    return [{"type": "text", "text": {"content": text[i:i + 2000]}} for i in range(0, len(text), 2000)] or [
        {"type": "text", "text": {"content": ""}}]


def page_title(page: dict) -> str:
    for prop in (page.get("properties") or {}).values():
        if prop.get("type") == "title":
            return _plain(prop.get("title"))
    return _plain(page.get("title")) if isinstance(page.get("title"), list) else ""


def property_value(prop: dict) -> Any:
    t = prop.get("type")
    v = prop.get(t)
    if t in {"title", "rich_text"}:
        return _plain(v)
    if t in {"select", "status"}:
        return (v or {}).get("name")
    if t == "multi_select":
        return [o.get("name") for o in v or []]
    if t == "date":
        return None if not v else (f"{v.get('start')} → {v['end']}" if v.get("end") else v.get("start"))
    if t == "people":
        return [p.get("name") or p.get("id") for p in v or []]
    if t == "relation":
        return [r.get("id") for r in v or []]
    if t == "formula":
        return (v or {}).get((v or {}).get("type"))
    if t in {"files"}:
        return [f.get("name") for f in v or []]
    if t in {"created_by", "last_edited_by"}:
        return (v or {}).get("name")
    return v


def markdown_to_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    in_code = False
    code: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                blocks.append({"object": "block", "type": "code",
                               "code": {"rich_text": _rich("\n".join(code)), "language": "plain text"}})
                code, in_code = [], False
            else:
                in_code = True
            continue
        if in_code:
            code.append(raw)
            continue
        if not line.strip():
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            kind = f"heading_{len(m.group(1))}"
            blocks.append({"object": "block", "type": kind, kind: {"rich_text": _rich(m.group(2))}})
        elif re.match(r"^\s*[-*]\s+\[( |x|X)\]\s+", line):
            checked = bool(re.match(r"^\s*[-*]\s+\[(x|X)\]", line))
            blocks.append({"object": "block", "type": "to_do", "to_do": {
                "rich_text": _rich(re.sub(r"^\s*[-*]\s+\[( |x|X)\]\s+", "", line)), "checked": checked}})
        elif re.match(r"^\s*[-*+]\s+", line):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": _rich(re.sub(r"^\s*[-*+]\s+", "", line))}})
        elif re.match(r"^\s*\d+[.)]\s+", line):
            blocks.append({"object": "block", "type": "numbered_list_item",
                           "numbered_list_item": {"rich_text": _rich(re.sub(r"^\s*\d+[.)]\s+", "", line))}})
        elif line.startswith("> "):
            blocks.append({"object": "block", "type": "quote", "quote": {"rich_text": _rich(line[2:])}})
        else:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich(line)}})
    if in_code and code:
        blocks.append({"object": "block", "type": "code", "code": {"rich_text": _rich("\n".join(code)), "language": "plain text"}})
    return blocks


def block_text(block: dict) -> str:
    t = block.get("type", "")
    body = block.get(t) or {}
    text = _plain(body.get("rich_text")) if isinstance(body, dict) else ""
    prefix = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### ", "bulleted_list_item": "- ",
              "numbered_list_item": "1. ", "quote": "> "}.get(t, "")
    if t == "to_do":
        prefix = "- [x] " if body.get("checked") else "- [ ] "
    if t == "code":
        return f"```\n{text}\n```"
    if t == "child_page":
        return f"[subpage: {body.get('title')} ({block.get('id')})]"
    if t == "child_database":
        return f"[database: {body.get('title')} ({block.get('id')})]"
    if t in {"divider"}:
        return "---"
    return prefix + text if text or prefix else ""


def _id(value: str) -> str:
    m = re.search(r"([0-9a-f]{32})(?:\?|$)", value.replace("-", ""))
    return m.group(1) if m else value.strip()


@itool(PID, "notion_search")
async def notion_search(ctx: ToolContext, query: str = "", kind: str | None = None, max_results: int = 10) -> dict:
    """Search Notion pages and databases shared with Sentient by title. `kind`: "page" or "database" (optional)."""
    body: dict[str, Any] = {"query": query, "page_size": max(1, min(int(max_results or 10), 50))}
    if kind in {"page", "database"}:
        body["filter"] = {"property": "object", "value": kind}
    res = await notion(ctx, "POST", "/search", json=body)
    rows = []
    for r in res.get("results") or []:
        title = page_title(r) if r.get("object") == "page" else _plain(r.get("title"))
        rows.append({"id": r.get("id"), "type": r.get("object"), "title": title, "url": r.get("url"),
                     "last_edited": r.get("last_edited_time")})
    return {"query": query, "results": rows}


@itool(PID, "notion_read_page")
async def notion_read_page(ctx: ToolContext, page_id: str, max_chars: int = 30000) -> dict:
    """Read a Notion page as markdown-like text (id or page URL)."""
    pid = _id(page_id)
    page = await notion(ctx, "GET", f"/pages/{pid}")
    lines: list[str] = []

    async def walk(block_id: str, depth: int) -> None:
        cursor = None
        for _ in range(10):
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            res = await notion(ctx, "GET", f"/blocks/{block_id}/children", params=params)
            for b in res.get("results") or []:
                t = block_text(b)
                if t:
                    lines.append("  " * depth + t)
                if b.get("has_children") and depth < 2 and b.get("type") not in {"child_page", "child_database"}:
                    await walk(b["id"], depth + 1)
            cursor = res.get("next_cursor")
            if not res.get("has_more"):
                break

    await walk(pid, 0)
    text, cut = truncate("\n".join(lines), max(1000, int(max_chars or 30000)))
    props = {k: property_value(v) for k, v in (page.get("properties") or {}).items() if v.get("type") != "title"}
    return {"id": pid, "title": page_title(page), "url": page.get("url"), "properties": props, "content": text,
            "truncated": cut}


@itool(PID, "notion_create_page", risk=Risk.write)
async def notion_create_page(ctx: ToolContext, title: str, content: str = "", parent_page_id: str | None = None,
                             parent_database_id: str | None = None, properties: dict | None = None) -> dict:
    """Create a Notion page under a parent page or inside a database. `content` may use markdown (# headings,
    - bullets, 1. lists, - [ ] to-dos, ``` code). Find parent ids with notion_search. For databases, `properties`
    may set other columns in Notion API format."""
    if not parent_page_id and not parent_database_id:
        raise IntegrationError("Say where to create the page: give parent_page_id or parent_database_id (use notion_search).")
    blocks = markdown_to_blocks(content) if content.strip() else []
    body: dict[str, Any]
    if parent_database_id:
        dbid = _id(parent_database_id)
        db = await notion(ctx, "GET", f"/databases/{dbid}")
        title_prop = next((k for k, v in (db.get("properties") or {}).items() if v.get("type") == "title"), "Name")
        body = {"parent": {"database_id": dbid}, "properties": {**(properties or {}), title_prop: {"title": _rich(title)}}}
    else:
        body = {"parent": {"page_id": _id(parent_page_id or "")}, "properties": {"title": {"title": _rich(title)}}}
    body["children"] = blocks[:100]
    page = await notion(ctx, "POST", "/pages", json=body)
    for i in range(100, len(blocks), 100):
        await notion(ctx, "PATCH", f"/blocks/{page['id']}/children", json={"children": blocks[i:i + 100]})
    return {"created": True, "id": page.get("id"), "url": page.get("url")}


@itool(PID, "notion_append_blocks", risk=Risk.write)
async def notion_append_blocks(ctx: ToolContext, page_id: str, content: str) -> dict:
    """Add content (markdown) to the end of a Notion page."""
    blocks = markdown_to_blocks(content)
    if not blocks:
        raise IntegrationError("There's no content to add.")
    pid = _id(page_id)
    for i in range(0, len(blocks), 100):
        await notion(ctx, "PATCH", f"/blocks/{pid}/children", json={"children": blocks[i:i + 100]})
    return {"appended_blocks": len(blocks), "page_id": pid}


@itool(PID, "notion_query_database")
async def notion_query_database(ctx: ToolContext, database_id: str, filter: dict | None = None,
                                sorts: list[dict] | None = None, max_results: int = 25) -> dict:
    """List rows of a Notion database with their column values. Optional `filter`/`sorts` use the Notion API
    format, e.g. filter {"property": "Status", "status": {"equals": "Done"}}."""
    body: dict[str, Any] = {"page_size": max(1, min(int(max_results or 25), 100))}
    if filter:
        body["filter"] = filter
    if sorts:
        body["sorts"] = sorts
    res = await notion(ctx, "POST", f"/databases/{_id(database_id)}/query", json=body)
    rows = [{"id": p.get("id"), "title": page_title(p), "url": p.get("url"),
             "properties": {k: property_value(v) for k, v in (p.get("properties") or {}).items()}}
            for p in res.get("results") or []]
    return {"database_id": database_id, "count": len(rows), "has_more": res.get("has_more"), "rows": rows}


class NotionPlugin(IntegrationPlugin):
    id = PID
    display_name = "Notion"
    description = (
        "Use your Notion workspace as Sentient's notebook: search pages and databases, read pages, write new "
        "pages with headings, lists and to-dos, add to existing pages, and query databases."
    )
    category = "knowledge"
    icon = "notion"
    auth_type = "api_key"
    selection_hint = "Use to search, read, create or add to Notion pages, or query Notion databases."
    setup_fields = [SetupField("token", "Internal integration secret", secret=True, required=True,
                               help="From notion.so/profile/integrations → your integration → Configuration.",
                               placeholder="ntn_...")]
    docs_url = "https://developers.notion.com/docs/create-a-notion-integration"
    instructions_md = (
        "1. Open https://www.notion.so/profile/integrations and click **New integration**.\n"
        "2. Name it `Sentient`, choose your workspace, keep the type **Internal**, and click **Save**.\n"
        "3. On the **Configuration** tab, make sure **Read content**, **Update content** and **Insert content** "
        "are ticked.\n"
        "4. Click **Show** next to **Internal Integration Secret**, then copy it.\n"
        "5. Paste it here and click **Connect**.\n"
        "6. Important: in Notion, open each top-level page Sentient should use, click **•••** (top right) → "
        "**Connections** → **Sentient**. Sub-pages are included automatically.\n"
    )
    tools = [notion_search, notion_read_page, notion_create_page, notion_append_blocks, notion_query_database]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        token = str(fields.get("token", "")).strip()
        if not token:
            raise IntegrationError("Please paste the Notion integration secret.")
        async with http_client(headers=_headers(token)) as http:
            r = await http.get(f"{API}/users/me")
        if r.status_code in (401, 403):
            raise IntegrationError("Notion didn't accept that secret. Copy it again from the integration page.")
        r.raise_for_status()
        me = r.json()
        label = (me.get("bot") or {}).get("workspace_name") or me.get("name")
        return {"token": token}, label


PLUGIN = NotionPlugin()
