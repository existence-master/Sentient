"""Gmail over the REST API with the user's own OAuth token."""

from __future__ import annotations

import asyncio
import base64
import re
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import getaddresses
from typing import TYPE_CHECKING, Any

import httpx

from sentient.integrations.base import FeedBatch, IntegrationError, itool, manager_from
from sentient.integrations.common import email_blocked, email_of, truncate
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.integrations.plugins.web import html_to_text
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://gmail.googleapis.com/gmail/v1/users/me"
PID = "gmail"
LIST_BODY_CHARS = 1500
READ_BODY_CHARS = 20000
CATEGORIES = {"primary", "social", "promotions", "updates", "forums"}


# ---------------------------------------------------------------------------- MIME helpers
def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _part_text(part: dict) -> tuple[str, str]:
    """Return (plain, html) text found in a payload part, recursively."""
    mime = part.get("mimeType", "")
    data = (part.get("body") or {}).get("data")
    plain, html = "", ""
    if data and not (part.get("filename")):
        text = _b64decode(data).decode("utf-8", errors="replace")
        if mime == "text/plain":
            plain = text
        elif mime == "text/html":
            html = text
    for sub in part.get("parts") or []:
        p, h = _part_text(sub)
        plain += p
        html += h
    return plain, html


def decode_body(payload: dict) -> str:
    plain, html = _part_text(payload or {})
    if plain.strip():
        return plain.strip()
    if html.strip():
        return html_to_text(html)[1]
    return ""


def _attachments(payload: dict) -> list[dict]:
    out = []
    for part in (payload or {}).get("parts") or []:
        if part.get("filename"):
            out.append({"filename": part["filename"], "mime": part.get("mimeType"),
                        "size": (part.get("body") or {}).get("size")})
        out.extend(_attachments(part))
    return out


def _headers(msg: dict) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in (msg.get("payload") or {}).get("headers") or []}


def normalize_message(msg: dict, body_chars: int | None = LIST_BODY_CHARS) -> dict:
    h = _headers(msg)
    body = decode_body(msg.get("payload") or {})
    if body_chars:
        body, _ = truncate(body, body_chars)
    date = None
    if msg.get("internalDate"):
        date = datetime.fromtimestamp(int(msg["internalDate"]) / 1000, UTC).isoformat()
    return {
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": h.get("from"),
        "sender_email": email_of(h.get("from")),
        "to": h.get("to"),
        "subject": h.get("subject") or "",
        "snippet": msg.get("snippet") or "",
        "body": body,
        "date": date or h.get("date"),
        "labels": list(msg.get("labelIds") or []),
        "url": f"https://mail.google.com/mail/u/0/#all/{msg.get('id')}",
    }


def build_raw(to: str, subject: str, body: str, *, cc: str | None = None, bcc: str | None = None,
              in_reply_to: str | None = None, references: str | None = None) -> str:
    m = EmailMessage()
    m["To"] = to
    if cc:
        m["Cc"] = cc
    if bcc:
        m["Bcc"] = bcc
    m["Subject"] = subject
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
        m["References"] = f"{references} {in_reply_to}".strip() if references else in_reply_to
    m.set_content(body)
    return base64.urlsafe_b64encode(m.as_bytes()).decode()


# ---------------------------------------------------------------------------- API helpers
async def _get_messages(ctx: ToolContext | None, ids: list[str], mgr: IntegrationManager,
                        body_chars: int | None = LIST_BODY_CHARS) -> list[dict]:
    sem = asyncio.Semaphore(8)

    async def one(mid: str) -> dict:
        async with sem:
            return await gapi(ctx, PID, "GET", f"{API}/messages/{mid}", params={"format": "full"}, mgr=mgr)

    raw = await asyncio.gather(*(one(i) for i in ids))
    return [normalize_message(m, body_chars) for m in raw]


async def _search(ctx: ToolContext | None, query: str, max_results: int, mgr: IntegrationManager) -> list[dict]:
    n = max(1, min(int(max_results or 10), 50))
    listing = await gapi(ctx, PID, "GET", f"{API}/messages", params={"q": query, "maxResults": n}, mgr=mgr)
    ids = [m["id"] for m in listing.get("messages") or []]
    return await _get_messages(ctx, ids, mgr) if ids else []


async def _filtered(ctx: ToolContext, query: str, max_results: int) -> dict:
    mgr = manager_from(ctx)
    items = await _search(ctx, query, max_results, mgr)
    kept = await mgr.filter_items(PID, items, "email")
    out: dict[str, Any] = {"query": query, "count": len(kept), "messages": kept}
    if len(kept) < len(items):
        out["hidden_by_privacy_filters"] = len(items) - len(kept)
    return out


def _category_query(base: str, category: str | None) -> str:
    cat = (category or "primary").lower().strip()
    return f"{base} category:{cat}" if cat in CATEGORIES else base


async def _label_ids(ctx: ToolContext, names: list[str] | None) -> list[str]:
    if not names:
        return []
    listing = await gapi(ctx, PID, "GET", f"{API}/labels")
    by_name = {lab["name"].lower(): lab["id"] for lab in listing.get("labels") or []}
    by_id = {lab["id"] for lab in listing.get("labels") or []}
    out = []
    for n in names:
        if n in by_id:
            out.append(n)
        elif n.lower() in by_name:
            out.append(by_name[n.lower()])
        elif n.upper() in by_id:
            out.append(n.upper())
        else:
            raise IntegrationError(f"There is no Gmail label called '{n}'. Use gmail_list_labels to see them.")
    return out


async def _modify(ctx: ToolContext, message_id: str, add: list[str] | None = None, remove: list[str] | None = None) -> dict:
    body = {"addLabelIds": await _label_ids(ctx, add), "removeLabelIds": await _label_ids(ctx, remove)}
    res = await gapi(ctx, PID, "POST", f"{API}/messages/{message_id}/modify", json=body)
    return {"id": res.get("id", message_id), "labels": res.get("labelIds", [])}


# ---------------------------------------------------------------------------- tools
@itool(PID, "gmail_search")
async def gmail_search(ctx: ToolContext, query: str, max_results: int = 10) -> dict:
    """Search emails with Gmail search syntax, e.g. "from:jane@acme.com", "subject:invoice newer_than:7d",
    "has:attachment", "in:sent", "after:2026/09/01 before:2026/09/10", "label:work is:unread"."""
    return await _filtered(ctx, query, max_results)


@itool(PID, "gmail_get_unread")
async def gmail_get_unread(ctx: ToolContext, max_results: int = 10, category: str = "primary") -> dict:
    """Get unread emails in the inbox. `category`: primary, social, promotions, updates, forums, or "all"."""
    return await _filtered(ctx, _category_query("in:inbox is:unread", category), max_results)


@itool(PID, "gmail_get_latest")
async def gmail_get_latest(ctx: ToolContext, max_results: int = 10, category: str = "primary") -> dict:
    """Get the most recent emails in the inbox (read or unread). `category`: primary, social, promotions, updates, forums, or "all"."""
    return await _filtered(ctx, _category_query("in:inbox", category), max_results)


@itool(PID, "gmail_read_message")
async def gmail_read_message(ctx: ToolContext, message_id: str) -> dict:
    """Read one email in full (headers, plain-text body, attachment names) by its message id."""
    mgr = manager_from(ctx)
    msg = await gapi(ctx, PID, "GET", f"{API}/messages/{message_id}", params={"format": "full"})
    item = normalize_message(msg, READ_BODY_CHARS)
    if email_blocked(item, await mgr.get_privacy_filters(PID)):
        return {"error": "This email is hidden by your privacy filters."}
    h = _headers(msg)
    item.update({"cc": h.get("cc"), "reply_to": h.get("reply-to"), "attachments": _attachments(msg.get("payload") or {})})
    return item


@itool(PID, "gmail_read_thread")
async def gmail_read_thread(ctx: ToolContext, thread_id: str) -> dict:
    """Read a whole email conversation (every message in the thread, oldest first)."""
    mgr = manager_from(ctx)
    thread = await gapi(ctx, PID, "GET", f"{API}/threads/{thread_id}", params={"format": "full"})
    items = [normalize_message(m, 6000) for m in thread.get("messages") or []]
    kept = await mgr.filter_items(PID, items, "email")
    out: dict[str, Any] = {"thread_id": thread_id, "messages": kept}
    if len(kept) < len(items):
        out["hidden_by_privacy_filters"] = len(items) - len(kept)
    return out


@itool(PID, "gmail_send", risk=Risk.send)
async def gmail_send(ctx: ToolContext, to: str, subject: str, body: str, cc: str | None = None,
                     bcc: str | None = None) -> dict:
    """Send a new email. `to`, `cc`, `bcc` accept comma-separated addresses. Body is plain text."""
    raw = build_raw(to, subject, body, cc=cc, bcc=bcc)
    res = await gapi(ctx, PID, "POST", f"{API}/messages/send", json={"raw": raw})
    return {"sent": True, "id": res.get("id"), "thread_id": res.get("threadId")}


async def _reply_parts(ctx: ToolContext, message_id: str, reply_all: bool) -> dict:
    mgr = manager_from(ctx)
    orig = await gapi(ctx, PID, "GET", f"{API}/messages/{message_id}", params={
        "format": "metadata", "metadataHeaders": ["From", "To", "Cc", "Subject", "Message-ID", "References", "Reply-To"]})
    h = _headers(orig)
    me = (mgr._get_state(PID).get("account_label") or "").lower()
    to = h.get("reply-to") or h.get("from") or ""
    if "SENT" in (orig.get("labelIds") or []) or (me and email_of(h.get("from")) == me):
        to = h.get("to") or to  # replying to your own message (a nudge) goes to its recipients, like Gmail does
    cc = None
    if reply_all:
        others = [f"{n} <{a}>" if n else a for n, a in getaddresses([h.get("to", ""), h.get("cc", "")])
                  if a and a.lower() != me and a.lower() != email_of(to)]
        cc = ", ".join(others) or None
    subject = h.get("subject") or ""
    if not re.match(r"^\s*re:", subject, re.I):
        subject = f"Re: {subject}"
    return {"to": to, "cc": cc, "subject": subject, "in_reply_to": h.get("message-id"),
            "references": h.get("references"), "thread_id": orig.get("threadId")}


@itool(PID, "gmail_reply", risk=Risk.send)
async def gmail_reply(ctx: ToolContext, message_id: str, body: str, reply_all: bool = False) -> dict:
    """Reply to an email in its thread (to the sender, or everyone with reply_all=true). Body is plain text."""
    p = await _reply_parts(ctx, message_id, reply_all)
    raw = build_raw(p["to"], p["subject"], body, cc=p["cc"], in_reply_to=p["in_reply_to"], references=p["references"])
    res = await gapi(ctx, PID, "POST", f"{API}/messages/send", json={"raw": raw, "threadId": p["thread_id"]})
    return {"sent": True, "id": res.get("id"), "thread_id": res.get("threadId"), "to": p["to"], "cc": p["cc"]}


@itool(PID, "gmail_create_draft", risk=Risk.write)
async def gmail_create_draft(ctx: ToolContext, to: str, subject: str, body: str,
                             reply_to_message_id: str | None = None) -> dict:
    """Save an email as a draft (not sent) so the user can review it in Gmail. Set reply_to_message_id to draft a reply."""
    message: dict[str, Any]
    if reply_to_message_id:
        p = await _reply_parts(ctx, reply_to_message_id, False)
        raw = build_raw(to or p["to"], subject or p["subject"], body, in_reply_to=p["in_reply_to"],
                        references=p["references"])
        message = {"raw": raw, "threadId": p["thread_id"]}
    else:
        message = {"raw": build_raw(to, subject, body)}
    res = await gapi(ctx, PID, "POST", f"{API}/drafts", json={"message": message})
    return {"draft_id": res.get("id"), "message_id": (res.get("message") or {}).get("id"),
            "url": "https://mail.google.com/mail/u/0/#drafts"}


@itool(PID, "gmail_list_labels")
async def gmail_list_labels(ctx: ToolContext) -> dict:
    """List the Gmail labels (system and the user's own)."""
    res = await gapi(ctx, PID, "GET", f"{API}/labels")
    return {"labels": [{"id": lab["id"], "name": lab["name"], "type": lab.get("type")} for lab in res.get("labels") or []]}


@itool(PID, "gmail_modify_labels", risk=Risk.write)
async def gmail_modify_labels(ctx: ToolContext, message_id: str, add_labels: list[str] | None = None,
                              remove_labels: list[str] | None = None) -> dict:
    """Add and/or remove labels on an email. Labels can be names ("Work") or ids ("STARRED", "IMPORTANT")."""
    return await _modify(ctx, message_id, add_labels, remove_labels)


@itool(PID, "gmail_mark_read", risk=Risk.write)
async def gmail_mark_read(ctx: ToolContext, message_id: str) -> dict:
    """Mark an email as read."""
    return await _modify(ctx, message_id, remove=["UNREAD"])


@itool(PID, "gmail_mark_unread", risk=Risk.write)
async def gmail_mark_unread(ctx: ToolContext, message_id: str) -> dict:
    """Mark an email as unread."""
    return await _modify(ctx, message_id, add=["UNREAD"])


@itool(PID, "gmail_archive", risk=Risk.write)
async def gmail_archive(ctx: ToolContext, message_id: str) -> dict:
    """Archive an email (remove it from the inbox without deleting it)."""
    return await _modify(ctx, message_id, remove=["INBOX"])


@itool(PID, "gmail_trash", risk=Risk.send)
async def gmail_trash(ctx: ToolContext, message_id: str) -> dict:
    """Move an email to the trash."""
    res = await gapi(ctx, PID, "POST", f"{API}/messages/{message_id}/trash")
    return {"trashed": True, "id": res.get("id", message_id)}


# ---------------------------------------------------------------------------- change feed (users.history)
HISTORY_PAGES = 10
FEED_MAX_MESSAGES = 50
FEED_SKIP_LABELS = {"DRAFT", "SENT", "SPAM", "TRASH", "CHAT"}


async def _history_baseline(mgr: IntegrationManager) -> str:
    profile = await gapi(None, PID, "GET", f"{API}/profile", mgr=mgr)
    hid = profile.get("historyId")
    if not hid:
        raise IntegrationError("Gmail didn't report a mailbox history id.")
    return str(hid)


async def _get_existing(mgr: IntegrationManager, ids: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(8)

    async def one(mid: str) -> dict | None:
        async with sem:
            try:
                return await gapi(None, PID, "GET", f"{API}/messages/{mid}", params={"format": "full"}, mgr=mgr)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:  # deleted before we fetched it
                    return None
                raise

    return [m for m in await asyncio.gather(*(one(i) for i in ids)) if m]


async def gmail_change_feed(mgr: IntegrationManager, cursor: str | None) -> FeedBatch:
    """New INBOX messages since the stored historyId.

    No cursor: baseline from users.getProfile (nothing emitted). 404 (history too old): re-baseline from now
    without emitting, so a long offline period never floods triggers.
    """
    if not cursor:
        return FeedBatch(cursor=await _history_baseline(mgr), rebaselined=True)
    ids: list[str] = []
    latest = cursor
    page_token: str | None = None
    for _ in range(HISTORY_PAGES):
        params: dict[str, Any] = {"startHistoryId": cursor, "historyTypes": "messageAdded", "labelId": "INBOX",
                                  "maxResults": 500}
        if page_token:
            params["pageToken"] = page_token
        try:
            res = await gapi(None, PID, "GET", f"{API}/history", params=params, mgr=mgr)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return FeedBatch(cursor=await _history_baseline(mgr), rebaselined=True,
                                 note="Gmail's change history had expired, so watching restarted from now.")
            raise
        for h in res.get("history") or []:
            for added in h.get("messagesAdded") or []:
                m = added.get("message") or {}
                labels = set(m.get("labelIds") or [])
                if m.get("id") and (not labels or ("INBOX" in labels and not labels & FEED_SKIP_LABELS)):
                    ids.append(str(m["id"]))
        latest = str(res.get("historyId") or latest)
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    ids = list(dict.fromkeys(ids))[-FEED_MAX_MESSAGES:]
    items: list[dict] = []
    for msg in await _get_existing(mgr, ids) if ids else []:
        item = normalize_message(msg)
        labels = set(item["labels"])
        if "INBOX" not in labels or labels & FEED_SKIP_LABELS:
            continue
        item["_key"] = item["id"]
        items.append(item)
    return FeedBatch(cursor=latest, items=items)


# ---------------------------------------------------------------------------- recent threads (follow-ups)
THREAD_HEADERS = ("list-unsubscribe", "list-id", "precedence", "auto-submitted", "content-type")
THREAD_SKIP_LABELS = {"DRAFT", "SPAM", "TRASH", "CHAT"}
THREAD_BODY_CHARS = 4000
THREAD_SCAN_MAX = 50
THREAD_PAGE_SIZE = 50
THREAD_PAGES = 5  # listing pages read per check at most


def normalize_thread_message(msg: dict, body_chars: int = THREAD_BODY_CHARS) -> dict:
    """``normalize_message`` plus ``cc``, ``message_id`` and the headers that mark bulk or automated mail."""
    item = normalize_message(msg, body_chars)
    h = _headers(msg)
    item["cc"] = h.get("cc")
    item["message_id"] = h.get("message-id")
    item["headers"] = {k: h[k] for k in THREAD_HEADERS if k in h}
    return item


async def gmail_recent_threads(mgr: IntegrationManager, *, newer_than_days: int, idle_days: int,
                               limit: int = 40) -> dict:
    """Inbox and sent threads with activity in the last ``newer_than_days`` days whose newest message is at least
    ``idle_days`` old. Bulk categories are left out by the search; every message carries ``from_me``."""
    me = {(mgr._get_state(PID).get("account_label") or "").lower()} - {""}
    query = (f"{{in:inbox in:sent}} newer_than:{max(1, int(newer_than_days))}d -in:chats "
             "-category:promotions -category:social -category:updates -category:forums")
    if idle_days > 0:
        query += f" older_than:{int(idle_days)}d"
    n = max(1, min(int(limit or 40), THREAD_SCAN_MAX))
    idle_cutoff_ms = (datetime.now(UTC).timestamp() - max(0, int(idle_days)) * 86400) * 1000
    sem = asyncio.Semaphore(8)

    async def one(tid: str, fmt: str) -> dict:
        async with sem:
            return await gapi(None, PID, "GET", f"{API}/threads/{tid}", params={"format": fmt}, mgr=mgr)

    def newest_ms(raw: dict) -> int:
        stamps = [int(m.get("internalDate") or 0) for m in raw.get("messages") or []
                  if not set(m.get("labelIds") or []) & THREAD_SKIP_LABELS]
        return max(stamps, default=0)

    # `older_than` matches threads with ANY old message and the listing is newest-activity first, so active
    # threads can fill whole pages: page on, and keep only threads whose newest message is past the idle cutoff
    eligible: list[str] = []
    page_token: str | None = None
    for _ in range(THREAD_PAGES):
        params: dict[str, Any] = {"q": query, "maxResults": THREAD_PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token
        listing = await gapi(None, PID, "GET", f"{API}/threads", params=params, mgr=mgr)
        ids = [str(t["id"]) for t in listing.get("threads") or [] if t.get("id")]
        for raw in await asyncio.gather(*(one(i, "minimal") for i in ids)):
            if 0 < newest_ms(raw) <= idle_cutoff_ms and str(raw.get("id")) not in eligible:
                eligible.append(str(raw.get("id")))
        page_token = listing.get("nextPageToken")
        if len(eligible) >= n or not page_token:
            break

    threads = []
    for raw in await asyncio.gather(*(one(i, "full") for i in eligible[:n])):
        msgs = [normalize_thread_message(m) for m in raw.get("messages") or []
                if not set(m.get("labelIds") or []) & THREAD_SKIP_LABELS]
        if not msgs:
            continue
        me.update(m["sender_email"] for m in msgs if "SENT" in m["labels"] and m["sender_email"])
        tid = str(raw.get("id") or msgs[0]["thread_id"])
        threads.append({"source": PID, "thread_id": tid, "url": f"https://mail.google.com/mail/u/0/#all/{tid}",
                        "messages": msgs})
    for t in threads:
        for m in t["messages"]:
            m["from_me"] = "SENT" in m["labels"] or m["sender_email"] in me
    return {"addresses": sorted(me), "threads": threads}


class GmailPlugin(GooglePlugin):
    id = PID
    display_name = "Gmail"
    description = (
        "Read, search, send and organise your email. Sentient can catch you up on unread mail, draft and send "
        "replies in the right thread, label, archive and trash messages, and watch for new email to suggest "
        "actions or trigger tasks. Privacy filters keep chosen senders, labels or keywords out of its sight."
    )
    category = "communication"
    icon = "gmail"
    api_name = "Gmail API"
    api_slug = "gmail.googleapis.com"
    selection_hint = "Use to read, search, send, reply to, draft or organise emails in Gmail."
    privacy_fields = ["keywords", "emails", "labels"]
    privacy_kind = "email"
    feed_kind = "gmail_history"
    triggers = [{"event": "new_email", "label": "New email"}]
    tools = [gmail_search, gmail_get_unread, gmail_get_latest, gmail_read_message, gmail_read_thread, gmail_send,
             gmail_reply, gmail_create_draft, gmail_list_labels, gmail_modify_labels, gmail_mark_read,
             gmail_mark_unread, gmail_archive, gmail_trash]

    async def change_feed(self, mgr: IntegrationManager, cursor: str | None) -> FeedBatch:
        return await gmail_change_feed(mgr, cursor)

    async def recent_threads(self, mgr: IntegrationManager, *, newer_than_days: int, idle_days: int,
                             limit: int = 40) -> dict:
        return await gmail_recent_threads(mgr, newer_than_days=newer_than_days, idle_days=idle_days, limit=limit)

    async def poll(self, mgr: IntegrationManager, since: datetime) -> list[dict]:
        """New inbox messages since ``since`` (normalized, unfiltered; the manager filters)."""
        query = f"in:inbox after:{int(since.timestamp())}"
        items = await _search(None, query, 25, mgr)
        for i in items:
            i["_key"] = i["id"]
        return items


PLUGIN = GmailPlugin()
