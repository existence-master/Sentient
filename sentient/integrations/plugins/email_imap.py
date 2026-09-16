"""Any email account over IMAP (reading + IDLE push) and SMTP (sending), signed in with an app password.

Works with Gmail, iCloud, Yahoo, Fastmail and most providers. Items use the Gmail item shape
(docs/API.md section 5) plus ``message_id``; ``id`` is the message's IMAP UID in its mailbox.
New inbox mail is pushed with IMAP IDLE (periodic NOOP checks when the server has no IDLE) and
published as ``source.items`` ``{source: "email_imap", event: "new_email", origin: "feed"}``.
"""

from __future__ import annotations

import asyncio
import contextlib
import email
import json
import logging
import re
import smtplib
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from email import policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import aioimaplib

from sentient.integrations.base import (
    IntegrationError,
    IntegrationPlugin,
    SetupField,
    creds,
    itool,
    manager_from,
)
from sentient.integrations.common import email_blocked, email_of, truncate
from sentient.integrations.plugins.web import html_to_text
from sentient.tools.base import Risk, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

PID = "email_imap"
IMAP_TIMEOUT_S = 30.0
SMTP_TIMEOUT_S = 30.0
IDLE_SECONDS = 25 * 60  # servers drop IDLE after ~29 minutes
BACKOFF_BASE_S = 15.0
WATCH_MAX_NEW = 50
LIST_BODY_CHARS = 1500
READ_BODY_CHARS = 20000
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_APP_PASSWORD_GROUPS = re.compile(r"^[a-zA-Z]{4}( [a-zA-Z]{4}){3}$")

# (smtp host, port) for well-known IMAP hosts, used when the SMTP fields are left empty
KNOWN_SMTP = {
    "imap.gmail.com": ("smtp.gmail.com", 465),
    "imap.mail.me.com": ("smtp.mail.me.com", 587),
    "outlook.office365.com": ("smtp-mail.outlook.com", 587),
    "imap-mail.outlook.com": ("smtp-mail.outlook.com", 587),
    "imap.mail.yahoo.com": ("smtp.mail.yahoo.com", 465),
    "imap.fastmail.com": ("smtp.fastmail.com", 465),
}

INSTRUCTIONS = """Connect almost any email account (Gmail, iCloud, Yahoo, Outlook and others) with an **app password**.
An app password is a separate password made just for Sentient. Your real password stays private, and you can
cancel the app password at any time from your email account's security page.

**Gmail**
1. Turn on 2-Step Verification: open https://myaccount.google.com/signinoptions/twosv and follow the steps.
2. Open https://myaccount.google.com/apppasswords, type `Sentient` as the app name and click **Create**.
3. Copy the 16-letter password Google shows you (the spaces don't matter).
4. Fill in here: IMAP server `imap.gmail.com`, port `993`, your full Gmail address, and the app password.
   To let Sentient send email, also fill in SMTP server `smtp.gmail.com` and SMTP port `465`.

**iCloud Mail**
1. Open https://account.apple.com and sign in. Go to **Sign-In and Security**, then **App-Specific Passwords**.
2. Click **Generate** (or **+**), name it `Sentient` and copy the password (it looks like `abcd-efgh-ijkl-mnop`).
3. Fill in here: IMAP server `imap.mail.me.com`, port `993`, your iCloud address (for example `name@icloud.com`),
   and the app password. For sending: SMTP server `smtp.mail.me.com`, SMTP port `587`.

**Outlook.com / Hotmail**
1. Open https://account.microsoft.com/security, choose **Advanced security options** and turn on two-step verification.
2. On the same page, under **App passwords**, click **Create a new app password** and copy it.
3. Fill in here: IMAP server `outlook.office365.com`, port `993`, your full Outlook address, and the app password.
   For sending: SMTP server `smtp-mail.outlook.com`, SMTP port `587`.
   Note: Microsoft is moving many accounts to its own sign-in and may refuse app passwords. If connecting fails
   with a sign-in error even though the password is right, this account can't be connected this way yet.

**Yahoo Mail**
1. Open https://login.yahoo.com/account/security and click **Generate app password**. Name it `Sentient`.
2. Fill in here: IMAP server `imap.mail.yahoo.com`, port `993`, your Yahoo address, and the app password.
   For sending: SMTP server `smtp.mail.yahoo.com`, SMTP port `465`.

**Other providers**: search your provider's help pages for "IMAP settings" and "app password".

Click **Connect**. Sentient checks the sign-in right away. The app password is kept in your system keychain, never
in a file. New email reaches Sentient within seconds (IMAP push), and watching your inbox uses no AI.
"""


class ImapAuthError(IntegrationError):
    """The server refused the username/app password."""


# ---------------------------------------------------------------------------- parsing helpers
def imap_quote(value: str) -> str:
    v = re.sub(r"[\r\n]+", " ", str(value))
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def imap_date(d: datetime) -> str:
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}"


def build_search(text: str = "", from_address: str | None = None, subject: str | None = None,
                 unread_only: bool = False, newer_than_days: int | None = None) -> list[str]:
    criteria: list[str] = []
    if text and text.strip():
        criteria += ["TEXT", imap_quote(text.strip())]
    if from_address and from_address.strip():
        criteria += ["FROM", imap_quote(from_address.strip())]
    if subject and subject.strip():
        criteria += ["SUBJECT", imap_quote(subject.strip())]
    if unread_only:
        criteria.append("UNSEEN")
    if newer_than_days:
        criteria += ["SINCE", imap_date(datetime.now(UTC) - timedelta(days=max(0, int(newer_than_days))))]
    return criteria or ["ALL"]


def parse_search(lines: list[Any]) -> list[int]:
    uids: list[int] = []
    for line in lines or []:
        if not isinstance(line, bytes | bytearray):
            continue
        parts = bytes(line).split()
        if parts and parts[0].upper() == b"SEARCH" and all(p.isdigit() for p in parts[1:]):
            uids.extend(int(p) for p in parts[1:])
    return sorted(set(uids))


def parse_uidvalidity(lines: list[Any]) -> str | None:
    for line in lines or []:
        if isinstance(line, bytes | bytearray):
            m = re.search(rb"UIDVALIDITY (\d+)", bytes(line))
            if m:
                return m.group(1).decode()
    return None


def parse_fetch(lines: list[Any]) -> list[dict]:
    """aioimaplib FETCH response -> [{uid, flags, raw}]. A literal follows a line ending in ``{n}``."""
    out: list[dict] = []
    i = 0
    while i < len(lines or []):
        line = lines[i]
        if isinstance(line, bytes | bytearray) and b"FETCH (" in bytes(line):
            head = bytes(line)
            raw = b""
            tail = b""
            if head.rstrip().endswith(b"}") and i + 1 < len(lines):
                raw = bytes(lines[i + 1])
                i += 1
                if i + 1 < len(lines) and isinstance(lines[i + 1], bytes | bytearray):
                    tail = bytes(lines[i + 1])
            meta = head + b" " + tail
            uid = re.search(rb"UID (\d+)", meta)
            flags = re.search(rb"FLAGS \(([^)]*)\)", meta)
            if uid:
                out.append({"uid": int(uid.group(1)), "flags": flags.group(1).decode(errors="ignore").split() if flags else [],
                            "raw": raw})
        i += 1
    return out


def parse_message(raw: bytes) -> EmailMessage:
    return email.message_from_bytes(raw or b"", policy=policy.default)  # type: ignore[return-value]


def _header(msg: EmailMessage, name: str) -> str:
    try:
        value = msg.get(name)
    except Exception:
        return ""
    return str(value) if value is not None else ""


def _body_text(msg: EmailMessage) -> str:
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
    except Exception:
        part = None
    if part is None:
        return ""
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    if part.get_content_type() == "text/html":
        return html_to_text(str(content))[1]
    return str(content).strip()


def attachments(msg: EmailMessage) -> list[dict]:
    out = []
    with contextlib.suppress(Exception):
        for part in msg.iter_attachments():
            payload = part.get_payload(decode=True) or b""
            out.append({"filename": part.get_filename(), "mime": part.get_content_type(), "size": len(payload)})
    return out


def normalize_raw(uid: int, raw: bytes, flags: list[str], *, mailbox: str = "INBOX",
                  body_chars: int | None = LIST_BODY_CHARS) -> dict:
    msg = parse_message(raw)
    body = _body_text(msg)
    snippet = " ".join(body.split())[:200]
    if body_chars:
        body, _ = truncate(body, body_chars)
    date = None
    with contextlib.suppress(Exception):
        dt = parsedate_to_datetime(_header(msg, "Date"))
        date = (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    message_id = _header(msg, "Message-ID").strip()
    refs = _header(msg, "References").split()
    thread_id = refs[0] if refs else (_header(msg, "In-Reply-To").strip() or message_id or str(uid))
    frm = _header(msg, "From")
    labels = [mailbox.upper() if mailbox.upper() == "INBOX" else mailbox]
    lowered = {f.lower() for f in flags}
    if "\\seen" not in lowered:
        labels.append("UNREAD")
    if "\\flagged" in lowered:
        labels.append("STARRED")
    return {
        "id": str(uid), "thread_id": thread_id, "from": frm, "sender_email": email_of(frm), "to": _header(msg, "To"),
        "subject": _header(msg, "Subject"), "snippet": snippet, "body": body, "date": date, "labels": labels,
        "url": None, "message_id": message_id or None,
    }


def _port(value: Any, default: int) -> int:
    s = str(value if value is not None else "").strip()
    if not s:
        return default
    if not s.isdigit() or not 0 < int(s) < 65536:
        raise IntegrationError(f"'{s}' isn't a valid port number.")
    return int(s)


# ---------------------------------------------------------------------------- IMAP session
class ImapSession:
    """A small async wrapper over aioimaplib (IMAP over SSL only); tests replace ``open_session``."""

    def __init__(self, c: dict):
        self.host = str(c.get("host") or "")
        self.port = int(c.get("port") or 993)
        self.username = str(c.get("username") or "")
        self.password = str(c.get("password") or "")
        self.client: aioimaplib.IMAP4_SSL | None = None
        self.uidvalidity: str | None = None
        self.mailbox = "INBOX"

    @property
    def _c(self) -> aioimaplib.IMAP4_SSL:
        if self.client is None:
            raise IntegrationError("The mail server connection is closed.")
        return self.client

    async def open(self, mailbox: str = "INBOX") -> None:
        try:
            self.client = aioimaplib.IMAP4_SSL(host=self.host, port=self.port, timeout=IMAP_TIMEOUT_S)
            await self.client.wait_hello_from_server()
        except Exception as exc:
            raise IntegrationError(
                f"Couldn't reach the mail server {self.host}:{self.port} ({type(exc).__name__}). "
                "Check the IMAP server name and port (usually 993)."
            ) from exc
        res = await self._c.login(self.username, self.password)
        if res.result != "OK":
            raise ImapAuthError("The mail server didn't accept the username and app password. Use an app password, "
                                "not your normal password (see the steps).")
        await self.select(mailbox)

    async def select(self, mailbox: str) -> None:
        res = await self._c.select(imap_quote(mailbox))
        if res.result != "OK":
            raise IntegrationError(f"There is no mailbox called '{mailbox}'.")
        self.mailbox = mailbox
        self.uidvalidity = parse_uidvalidity(res.lines)

    @property
    def supports_idle(self) -> bool:
        return bool(self._c.has_capability("IDLE"))

    async def uid_search(self, *criteria: str) -> list[int]:
        charset = None if all(c.isascii() for c in criteria) else "utf-8"
        res = await self._c.uid_search(*criteria, charset=charset)
        if res.result != "OK":
            raise IntegrationError("The mail server couldn't run that search.")
        return parse_search(res.lines)

    async def fetch(self, uids: list[int]) -> list[dict]:
        if not uids:
            return []
        res = await self._c.uid("fetch", ",".join(str(u) for u in uids), "(UID FLAGS BODY.PEEK[])")
        if res.result != "OK":
            raise IntegrationError("The mail server couldn't return those messages.")
        return parse_fetch(res.lines)

    async def idle_wait(self, timeout: float) -> bool:
        """Wait in IDLE until the server reports new mail (True) or the timeout passes (False)."""
        c = self._c
        idle = await c.idle_start(timeout=timeout)
        try:
            push = await c.wait_server_push(timeout=timeout + 30)
        except TimeoutError:
            push = None
        finally:
            with contextlib.suppress(Exception):
                c.idle_done()
                await asyncio.wait_for(idle, 10)
        if push is None or push == aioimaplib.STOP_WAIT_SERVER_PUSH:
            return False
        lines = push if isinstance(push, list) else [push]
        return any(isinstance(line, bytes | bytearray) and (b"EXISTS" in line or b"RECENT" in line) for line in lines)

    async def noop(self) -> None:
        await self._c.noop()

    async def close(self) -> None:
        if self.client is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.client.logout(), 5)
        self.client = None


async def open_session(c: dict, mailbox: str = "INBOX") -> Any:
    s = ImapSession(c)
    try:
        await s.open(mailbox)
    except IntegrationError:
        await s.close()
        raise
    except Exception as exc:
        await s.close()
        raise IntegrationError(f"The mail server connection failed ({type(exc).__name__}). Try again in a moment.") from exc
    return s


@contextlib.asynccontextmanager
async def _session(c: dict, mailbox: str = "INBOX") -> AsyncIterator[Any]:
    s = await open_session(c, mailbox)
    try:
        yield s
    except IntegrationError:
        raise
    except Exception as exc:
        raise IntegrationError(f"The mail server connection failed ({type(exc).__name__}). Try again in a moment.") from exc
    finally:
        await s.close()


# ---------------------------------------------------------------------------- SMTP
def smtp_settings(c: dict) -> tuple[str, int]:
    host = str(c.get("smtp_host") or "").strip()
    port = int(c.get("smtp_port") or 0)
    if not host:
        guess = KNOWN_SMTP.get(str(c.get("host") or "").strip().lower())
        if guess is None:
            raise IntegrationError("Sending needs the outgoing (SMTP) server. Reconnect Email (IMAP) and fill in the "
                                   "SMTP server and port.")
        host, port = guess[0], port or guess[1]
    return host, port or 465


def smtp_connect(host: str, port: int, timeout: float) -> smtplib.SMTP:
    context = ssl.create_default_context()
    if port == 465:
        return smtplib.SMTP_SSL(host, port, timeout=timeout, context=context)
    server = smtplib.SMTP(host, port, timeout=timeout)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()
    return server


def smtp_send(c: dict, message: EmailMessage | None) -> None:
    """Sign in to SMTP and send ``message`` (only check the sign-in when it is None). Blocking: run in a thread."""
    host, port = smtp_settings(c)
    try:
        server = smtp_connect(host, port, SMTP_TIMEOUT_S)
    except (OSError, smtplib.SMTPException) as exc:
        raise IntegrationError(f"Couldn't reach the outgoing mail server {host}:{port} ({type(exc).__name__}). "
                               "Check the SMTP server and port.") from exc
    try:
        server.login(str(c.get("username") or ""), str(c.get("password") or ""))
        if message is not None:
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise IntegrationError("The outgoing mail server didn't accept the username and app password.") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise IntegrationError("The mail server refused the recipient address. Check the email address.") from exc
    except (OSError, smtplib.SMTPException) as exc:
        raise IntegrationError(f"Sending failed ({type(exc).__name__}: {exc}).") from exc
    finally:
        with contextlib.suppress(Exception):
            server.quit()


# ---------------------------------------------------------------------------- tools
@itool(PID, "email_imap_search")
async def email_imap_search(ctx: ToolContext, text: str = "", from_address: str | None = None,
                            subject: str | None = None, unread_only: bool = False,
                            newer_than_days: int | None = None, mailbox: str = "INBOX", max_results: int = 10) -> dict:
    """Search the email account connected over IMAP (newest first). `text` matches anywhere in a message;
    `from_address`, `subject`, `unread_only` and `newer_than_days` narrow it down. `mailbox` defaults to INBOX."""
    c = await creds(ctx, PID)
    mgr = manager_from(ctx)
    n = max(1, min(int(max_results or 10), 50))
    criteria = build_search(text, from_address, subject, unread_only, newer_than_days)
    async with _session(c, mailbox) as s:
        uids = (await s.uid_search(*criteria))[-n:]
        rows = await s.fetch(uids)
    rows.sort(key=lambda r: r["uid"], reverse=True)
    items = [normalize_raw(r["uid"], r["raw"], r["flags"], mailbox=mailbox) for r in rows]
    kept = await mgr.filter_items(PID, items, "email")
    out: dict[str, Any] = {"mailbox": mailbox, "count": len(kept), "messages": kept}
    if len(kept) < len(items):
        out["hidden_by_privacy_filters"] = len(items) - len(kept)
    return out


async def _fetch_one(c: dict, message_id: str, mailbox: str) -> dict:
    uid = str(message_id).strip()
    if not uid.isdigit():
        raise IntegrationError("message_id is the number shown as `id` by email_imap_search.")
    async with _session(c, mailbox) as s:
        rows = await s.fetch([int(uid)])
    rows = [r for r in rows if r["uid"] == int(uid)]
    if not rows:
        raise IntegrationError("That email wasn't found (it may have been moved or deleted).")
    return rows[0]


@itool(PID, "email_imap_read")
async def email_imap_read(ctx: ToolContext, message_id: str, mailbox: str = "INBOX") -> dict:
    """Read one email in full (headers, plain-text body, attachment names) by the `id` from email_imap_search."""
    c = await creds(ctx, PID)
    mgr = manager_from(ctx)
    row = await _fetch_one(c, message_id, mailbox)
    item = normalize_raw(row["uid"], row["raw"], row["flags"], mailbox=mailbox, body_chars=READ_BODY_CHARS)
    if email_blocked(item, await mgr.get_privacy_filters(PID)):
        return {"error": "This email is hidden by your privacy filters."}
    msg = parse_message(row["raw"])
    item.update({"cc": _header(msg, "Cc") or None, "reply_to": _header(msg, "Reply-To") or None,
                 "attachments": attachments(msg)})
    return item


@itool(PID, "email_imap_send", risk=Risk.send)
async def email_imap_send(ctx: ToolContext, to: str, subject: str, body: str, cc: str | None = None,
                          bcc: str | None = None, reply_to_message_id: str | None = None) -> dict:
    """Send an email from the IMAP-connected account. `to`, `cc`, `bcc` accept comma-separated addresses; body is
    plain text. Set reply_to_message_id (an INBOX `id` from email_imap_search) to reply in that conversation."""
    c = await creds(ctx, PID)
    m = EmailMessage()
    m["From"] = str(c.get("username") or "")
    m["To"] = to
    if cc:
        m["Cc"] = cc
    if bcc:
        m["Bcc"] = bcc
    if reply_to_message_id:
        orig = parse_message((await _fetch_one(c, reply_to_message_id, "INBOX"))["raw"])
        orig_id = _header(orig, "Message-ID").strip()
        if orig_id:
            refs = _header(orig, "References").strip()
            m["In-Reply-To"] = orig_id
            m["References"] = f"{refs} {orig_id}".strip()
        if not subject:
            subject = _header(orig, "Subject")
        if subject and not re.match(r"^\s*re:", subject, re.IGNORECASE):
            subject = f"Re: {subject}"
    m["Subject"] = subject
    m["Date"] = formatdate(localtime=True)
    domain = str(c.get("username") or "").rpartition("@")[2] or None
    m["Message-ID"] = make_msgid(domain=domain)
    m.set_content(body)
    await asyncio.to_thread(smtp_send, c, m)
    return {"sent": True, "message_id": m["Message-ID"], "to": to, "cc": cc}


# ---------------------------------------------------------------------------- plugin
class EmailImapPlugin(IntegrationPlugin):
    id = PID
    display_name = "Email (IMAP)"
    description = (
        "Connect any email account (iCloud, Yahoo, Outlook, Fastmail, Gmail and more) with an app password. "
        "Sentient can search, read and send email, and new mail arrives within seconds to suggest actions or "
        "trigger tasks. Privacy filters keep chosen senders or keywords out of its sight."
    )
    category = "communication"
    icon = "mail"
    auth_type = "manual"
    selection_hint = "Use to search, read or send email in the account connected over IMAP."
    privacy_fields = ["keywords", "emails", "labels"]
    privacy_kind = "email"
    feed_kind = "imap_idle"
    triggers = [{"event": "new_email", "label": "New email"}]
    instructions_md = INSTRUCTIONS
    docs_url = "https://support.google.com/mail/answer/185833"
    setup_fields = [
        SetupField("host", "IMAP server", placeholder="imap.gmail.com",
                   help="Your provider's incoming mail server. See the steps for common providers."),
        SetupField("port", "IMAP port", required=False, placeholder="993", help="Almost always 993."),
        SetupField("username", "Email address", placeholder="you@example.com",
                   help="Usually your full email address."),
        SetupField("password", "App password", secret=True,
                   help="An app password made for Sentient (not your normal password). Stored in your system keychain."),
        SetupField("smtp_host", "SMTP server (for sending)", required=False, placeholder="smtp.gmail.com",
                   help="Optional. Needed to send email. Left empty, Sentient guesses it for well-known providers."),
        SetupField("smtp_port", "SMTP port", required=False, placeholder="465",
                   help="Usually 465 or 587."),
    ]
    tools = [email_imap_search, email_imap_read, email_imap_send]

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        await super().validate(fields, mgr)  # required fields
        host = re.sub(r"^imaps?://", "", str(fields.get("host", "")).strip().lower()).strip("/")
        password = str(fields.get("password", ""))
        if _APP_PASSWORD_GROUPS.match(password.strip()):
            password = password.replace(" ", "")  # Google shows app passwords in groups of four
        c = {
            "host": host,
            "port": _port(fields.get("port"), 993),
            "username": str(fields.get("username", "")).strip(),
            "password": password.strip(),
            "smtp_host": re.sub(r"^smtps?://", "", str(fields.get("smtp_host", "")).strip().lower()).strip("/"),
            "smtp_port": _port(fields.get("smtp_port"), 0) or None,
        }
        s = await open_session(c, "INBOX")
        await s.close()
        if c["smtp_host"]:
            await asyncio.to_thread(smtp_send, c, None)
        return c, c["username"]

    async def test(self, credentials: dict | None, mgr: IntegrationManager) -> str:
        c = credentials or {}
        s = await open_session(c, "INBOX")
        await s.close()
        return f"Signed in to {c.get('host')} as {c.get('username')}."

    # ------------------------------------------------------------------ push watcher
    async def check_new(self, mgr: IntegrationManager, session: Any) -> int:
        """Emit INBOX messages newer than the stored UID. The first check only records a baseline."""
        st = await mgr.feeds.state(PID)
        try:
            cursor = json.loads(st["cursor"]) if st["cursor"] else {}
        except json.JSONDecodeError:
            cursor = {}
        validity = session.uidvalidity
        if not cursor or cursor.get("uidvalidity") != validity or "last_uid" not in cursor:
            last = max(await session.uid_search("UID", "*"), default=0)
            await mgr.feeds.record_success(PID, cursor=json.dumps({"uidvalidity": validity, "last_uid": last}),
                                           note="Watching for new email from now." if not cursor else
                                           "The mailbox was reset on the server, so watching restarted from now.")
            return 0
        last = int(cursor["last_uid"])
        new = [u for u in await session.uid_search("UID", f"{last + 1}:*") if u > last]
        kept: list[dict] = []
        if new:
            rows = await session.fetch(new[-WATCH_MAX_NEW:])
            items = []
            for r in sorted(rows, key=lambda r: r["uid"]):
                item = normalize_raw(r["uid"], r["raw"], r["flags"])
                item["_key"] = item.get("message_id") or f"{validity}:{r['uid']}"
                items.append(item)
            kept = await mgr.emit_items(PID, "feed", items, event="new_email")
            last = max(new)
        await mgr.feeds.record_success(PID, cursor=json.dumps({"uidvalidity": validity, "last_uid": last}),
                                       emitted=len(kept))
        return len(kept)

    async def watch(self, mgr: IntegrationManager) -> None:
        """IDLE push loop with reconnect and exponential backoff; NOOP checks when IDLE is unsupported."""
        while True:
            session = None
            try:
                c = await mgr.get_credentials(PID)
                if not c:
                    return
                session = await open_session(c, "INBOX")
                await self.check_new(mgr, session)
                while True:
                    if session.supports_idle:
                        await session.idle_wait(IDLE_SECONDS)
                    else:
                        await asyncio.sleep(float(mgr.app.config.integrations.fast_sync_seconds or 300))
                        await session.noop()
                    await self.check_new(mgr, session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                res = await mgr.feeds.record_failure(PID, exc, base=BACKOFF_BASE_S)
                if isinstance(exc, ImapAuthError):
                    await mgr.set_error(PID, str(exc))
                await asyncio.sleep(res["retry_in_s"])
            finally:
                if session is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(session.close(), 5)


PLUGIN = EmailImapPlugin()
