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

from sentient.integrations import redact
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

**Self-hosted or no SSL port?** Set connection security to `starttls` and the port to the one your server gives for
STARTTLS (often `143`). Leave it as `ssl` for the providers above.

**Watching more than your inbox?** List extra mailbox names in Folders, comma-separated (for example
`INBOX, Work`), to also get new-mail pushes from them.

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


_LIST_RE = re.compile(rb'^(?:LIST\s+)?\((?P<flags>[^)]*)\)\s+(?:"(?:[^"\\]|\\.)*"|NIL)\s+(?P<name>.+)$', re.IGNORECASE)
SENT_NAMES = ("Sent", "Sent Items", "[Gmail]/Sent Mail", "Sent Messages", "Sent Mail", "INBOX.Sent")


def parse_list(lines: list[Any]) -> list[tuple[set[str], str]]:
    """aioimaplib LIST response -> [(lowercase flags, mailbox name)]. Quoted, atom and literal names."""
    out: list[tuple[set[str], str]] = []
    rows = [bytes(x) for x in lines or [] if isinstance(x, bytes | bytearray)]
    i = 0
    while i < len(rows):
        m = _LIST_RE.match(rows[i].strip())
        i += 1
        if not m:
            continue
        name = m.group("name").strip()
        if re.fullmatch(rb"\{\d+\}", name) and i < len(rows):
            name = rows[i].strip()
            i += 1
        elif name.startswith(b'"') and name.endswith(b'"') and len(name) >= 2:
            name = re.sub(rb"\\(.)", rb"\1", name[1:-1])
        flags = {f.lower() for f in m.group("flags").decode(errors="ignore").split()}
        out.append((flags, name.decode("utf-8", errors="replace")))
    return out


def find_sent_mailbox(entries: list[tuple[set[str], str]]) -> str | None:
    """The SPECIAL-USE ``\\Sent`` mailbox, else one with a common sent-mail name, else None."""
    for flags, name in entries:
        if "\\sent" in flags:
            return name
    by_lower = {name.lower(): name for _, name in entries}
    for candidate in SENT_NAMES:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


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
                  body_chars: int | None = LIST_BODY_CHARS, hide_codes: bool = True) -> dict:
    """The gmail item shape plus ``message_id``. ``hide_codes`` masks one-time codes and sign-in or reset links."""
    msg = parse_message(raw)
    frm = _header(msg, "From")
    subject = _header(msg, "Subject")
    body = _body_text(msg)
    if hide_codes:
        body = redact.hide_secrets(body, subject=subject, sender=email_of(frm))
        subject = redact.hide_secrets(subject, subject=subject, sender=email_of(frm))
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
    labels = [mailbox.upper() if mailbox.upper() == "INBOX" else mailbox]
    lowered = {f.lower() for f in flags}
    if "\\seen" not in lowered:
        labels.append("UNREAD")
    if "\\flagged" in lowered:
        labels.append("STARRED")
    return {
        "id": str(uid), "thread_id": thread_id, "from": frm, "sender_email": email_of(frm), "to": _header(msg, "To"),
        "subject": subject, "snippet": snippet, "body": body, "date": date, "labels": labels,
        "url": None, "message_id": message_id or None,
    }


def _port(value: Any, default: int) -> int:
    s = str(value if value is not None else "").strip()
    if not s:
        return default
    if not s.isdigit() or not 0 < int(s) < 65536:
        raise IntegrationError(f"'{s}' isn't a valid port number.")
    return int(s)


def _security(value: Any) -> str:
    s = str(value or "").strip().lower() or "ssl"
    if s not in ("ssl", "starttls"):
        raise IntegrationError("Connection security must be 'ssl' or 'starttls'.")
    return s


def parse_folders(value: Any) -> list[str]:
    """Comma-separated mailbox names, trimmed and de-duplicated in order; ``INBOX`` alone when empty."""
    seen: dict[str, None] = {}
    for part in str(value or "").split(","):
        name = part.strip()
        if name:
            seen.setdefault(name, None)
    return list(seen) or ["INBOX"]


async def _starttls(client: aioimaplib.IMAP4, host: str) -> None:
    """Upgrade a plaintext IMAP connection to TLS in place (RFC 3501 6.2.1).

    aioimaplib has no STARTTLS helper (only ``IMAP4_SSL``, TLS from the first byte), so this sends the
    command and upgrades the transport by hand. The server sends no new greeting after STARTTLS, so the
    protocol's state is set back to ``NONAUTH`` directly, and capabilities learned before TLS are discarded
    and re-fetched, since a pre-TLS CAPABILITY response could have been tampered with in transit.
    """
    protocol = client.protocol
    res = await protocol.execute(aioimaplib.Command("STARTTLS", protocol.new_tag(), loop=protocol.loop))
    if res.result != "OK":
        raise IntegrationError("The mail server refused to start TLS (STARTTLS).")
    loop = asyncio.get_running_loop()
    # start_tls returns a NEW transport; the protocol must switch to it immediately, or every command
    # after this point -- including LOGIN -- keeps going out over the raw, unencrypted socket.
    protocol.transport = await loop.start_tls(
        protocol.transport, protocol, ssl.create_default_context(), server_hostname=host
    )
    protocol.state = aioimaplib.NONAUTH
    protocol.capabilities = set()
    await protocol.execute(aioimaplib.Command("CAPABILITY", protocol.new_tag(), loop=protocol.loop))


# ---------------------------------------------------------------------------- IMAP session
class ImapSession:
    """A small async wrapper over aioimaplib, over SSL or STARTTLS; tests replace ``open_session``."""

    def __init__(self, c: dict):
        self.host = str(c.get("host") or "")
        self.security = _security(c.get("security"))
        self.port = int(c.get("port") or (993 if self.security == "ssl" else 143))
        self.username = str(c.get("username") or "")
        self.password = str(c.get("password") or "")
        self.client: aioimaplib.IMAP4 | None = None
        self.uidvalidity: str | None = None
        self.mailbox = "INBOX"

    @property
    def _c(self) -> aioimaplib.IMAP4:
        if self.client is None:
            raise IntegrationError("The mail server connection is closed.")
        return self.client

    async def open(self, mailbox: str = "INBOX") -> None:
        try:
            if self.security == "starttls":
                self.client = aioimaplib.IMAP4(host=self.host, port=self.port, timeout=IMAP_TIMEOUT_S)
                await self.client.wait_hello_from_server()
                await _starttls(self.client, self.host)
            else:
                self.client = aioimaplib.IMAP4_SSL(host=self.host, port=self.port, timeout=IMAP_TIMEOUT_S)
                await self.client.wait_hello_from_server()
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(
                f"Couldn't reach the mail server {self.host}:{self.port} ({type(exc).__name__}). Check the IMAP "
                f"server name, port and connection security (ssl is usually 993, starttls is usually 143)."
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

    async def fetch_headers(self, uids: list[int]) -> list[dict]:
        """Like ``fetch`` but only the header block (``raw`` holds the headers)."""
        if not uids:
            return []
        res = await self._c.uid("fetch", ",".join(str(u) for u in uids), "(UID FLAGS BODY.PEEK[HEADER])")
        if res.result != "OK":
            raise IntegrationError("The mail server couldn't return those messages.")
        return parse_fetch(res.lines)

    async def list_mailboxes(self) -> list[tuple[set[str], str]]:
        res = await self._c.list('""', "*")
        if res.result != "OK":
            return []
        return parse_list(res.lines)

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
    hide = redact.enabled(mgr)
    items = [normalize_raw(r["uid"], r["raw"], r["flags"], mailbox=mailbox, hide_codes=hide) for r in rows]
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
    item = normalize_raw(row["uid"], row["raw"], row["flags"], mailbox=mailbox, body_chars=READ_BODY_CHARS,
                         hide_codes=redact.enabled(mgr))
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


# ---------------------------------------------------------------------------- recent threads (follow-ups)
THREAD_HEADER_SCAN = 150  # newest quiet messages (max age .. idle cutoff) per mailbox whose headers are read
THREAD_RECENT_SCAN = 400  # newest recent messages (after the idle cutoff) per mailbox, to spot active threads
THREAD_HEADERS = ("list-unsubscribe", "list-id", "precedence", "auto-submitted", "content-type")


def _thread_message(row: dict, mailbox: str, *, body: bool, hide_codes: bool = True) -> dict:
    item = normalize_raw(row["uid"], row["raw"], row["flags"], mailbox=mailbox,
                         body_chars=4000 if body else LIST_BODY_CHARS, hide_codes=hide_codes)
    msg = parse_message(row["raw"])
    item["mailbox"] = mailbox
    item["cc"] = _header(msg, "Cc") or None
    item["headers"] = {k: _header(msg, k) for k in THREAD_HEADERS if _header(msg, k)}
    refs = _header(msg, "References").split() + _header(msg, "In-Reply-To").split()
    item["_refs"] = [r.strip() for r in refs if r.strip()]
    return item


def group_threads(messages: list[dict]) -> list[list[dict]]:
    """Group messages into conversations by Message-ID, References and In-Reply-To (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def key(m: dict) -> str:
        return m.get("message_id") or f"{m['mailbox']}:{m['id']}"

    for m in messages:
        k = find(key(m))
        for ref in m.get("_refs") or []:
            r = find(ref)
            if r != k:
                parent[r] = k
    groups: dict[str, list[dict]] = {}
    for m in messages:
        groups.setdefault(find(key(m)), []).append(m)
    return list(groups.values())


def _ts(m: dict) -> str:
    return str(m.get("date") or "")


async def imap_recent_threads(mgr: IntegrationManager, *, newer_than_days: int, idle_days: int,
                              limit: int = 40) -> dict:
    """INBOX and Sent conversations active in the last ``newer_than_days`` days whose newest message is at least
    ``idle_days`` old. Returns no threads (quietly) when the account has no recognisable Sent mailbox."""
    c = await mgr.get_credentials(PID)
    if not c:
        return {"addresses": [], "threads": []}
    me = {str(c.get("username") or "").strip().lower()} - {""}
    hide = redact.enabled(mgr)
    now = datetime.now(UTC)
    since = imap_date(now - timedelta(days=max(1, int(newer_than_days))))
    idle_at = now - timedelta(days=max(0, int(idle_days)))
    idle_before = idle_at.isoformat()

    async def scan(s: Any) -> list[dict]:
        # two capped searches, so a busy last few days can't push the quiet candidates out of the scan:
        # (a) quiet ones, from the max age up to the idle cutoff; (b) recent ones, to see which threads are active
        quiet = (await s.uid_search("SINCE", since, "BEFORE", imap_date(idle_at)))[-THREAD_HEADER_SCAN:]
        recent = (await s.uid_search("SINCE", imap_date(idle_at)))[-THREAD_RECENT_SCAN:]
        return await s.fetch_headers(sorted(set(quiet) | set(recent)))

    async with _session(c, "INBOX") as s:
        sent_box = find_sent_mailbox(await s.list_mailboxes())
        if not sent_box:
            log.info("no Sent mailbox found, so follow-ups skip this IMAP account")
            return {"addresses": sorted(me), "threads": [], "note": "no_sent_mailbox"}
        inbox_rows = await scan(s)
        await s.select(sent_box)
        sent_rows = await scan(s)
        sent = [_thread_message(r, sent_box, body=False, hide_codes=hide) for r in sent_rows]
        me.update(m["sender_email"] for m in sent if m["sender_email"])
        by_id: dict[str, dict] = {}
        for m in sent + [_thread_message(r, "INBOX", body=False, hide_codes=hide) for r in inbox_rows]:
            # a message in both mailboxes (mail to yourself) keeps its Sent copy
            by_id.setdefault(m.get("message_id") or f"{m['mailbox']}:{m['id']}", m)
        groups = []
        for msgs in group_threads(list(by_id.values())):
            msgs.sort(key=_ts)
            if msgs[-1].get("date") and _ts(msgs[-1]) <= idle_before:
                groups.append(msgs)
        groups.sort(key=lambda g: _ts(g[-1]), reverse=True)
        groups = groups[: max(1, min(int(limit or 40), 50))]
        # full text only for each conversation's newest message, one mailbox at a time
        wanted: dict[str, list[int]] = {}
        for g in groups:
            wanted.setdefault(g[-1]["mailbox"], []).append(int(g[-1]["id"]))
        full: dict[tuple[str, int], dict] = {}
        for box in sorted(wanted, key=lambda b: b != sent_box):  # Sent is selected already
            if box != s.mailbox:
                await s.select(box)
            for r in await s.fetch(wanted[box]):
                full[(box, r["uid"])] = _thread_message(r, box, body=True, hide_codes=hide)
    threads = []
    for g in groups:
        last = g[-1]
        g[-1] = full.get((last["mailbox"], int(last["id"])), last)
        for m in g:
            m["from_me"] = m["mailbox"] == sent_box or m["sender_email"] in me
            m.pop("_refs", None)
        threads.append({"source": PID, "thread_id": g[0].get("message_id") or f"{g[0]['mailbox']}:{g[0]['id']}",
                        "url": None, "messages": g})
    return {"addresses": sorted(me), "threads": threads}


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
        SetupField("security", "Connection security", required=False, placeholder="ssl",
                   help="`ssl` (default; connects already encrypted, usually port 993) or `starttls` (connects in "
                        "plain text, then upgrades; usually port 143). Use starttls for a provider or self-hosted "
                        "server that offers no SSL port."),
        SetupField("port", "IMAP port", required=False, placeholder="993",
                   help="993 for ssl, 143 for starttls, unless your provider says otherwise."),
        SetupField("username", "Email address", placeholder="you@example.com",
                   help="Usually your full email address."),
        SetupField("password", "App password", secret=True,
                   help="An app password made for Sentient (not your normal password). Stored in your system keychain."),
        SetupField("folders", "Folders to watch", required=False, placeholder="INBOX",
                   help="Comma-separated mailbox names to watch for new mail, for example `INBOX, Work`. "
                        "Defaults to INBOX alone; search and read still take any mailbox by name."),
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
        security = _security(fields.get("security"))
        folders = parse_folders(fields.get("folders"))
        c = {
            "host": host,
            "security": security,
            "port": _port(fields.get("port"), 993 if security == "ssl" else 143),
            "username": str(fields.get("username", "")).strip(),
            "password": password.strip(),
            "folders": folders,
            "smtp_host": re.sub(r"^smtps?://", "", str(fields.get("smtp_host", "")).strip().lower()).strip("/"),
            "smtp_port": _port(fields.get("smtp_port"), 0) or None,
        }
        s = await open_session(c, folders[0])
        for mailbox in folders[1:]:  # fail setup, not the watcher, if a watched folder doesn't exist
            await s.select(mailbox)
        await s.close()
        if c["smtp_host"]:
            await asyncio.to_thread(smtp_send, c, None)
        return c, c["username"]

    async def test(self, credentials: dict | None, mgr: IntegrationManager) -> str:
        c = credentials or {}
        s = await open_session(c, (c.get("folders") or ["INBOX"])[0])  # already a list: validate() parsed it once
        await s.close()
        return f"Signed in to {c.get('host')} as {c.get('username')}."

    async def recent_threads(self, mgr: IntegrationManager, *, newer_than_days: int, idle_days: int,
                             limit: int = 40) -> dict:
        return await imap_recent_threads(mgr, newer_than_days=newer_than_days, idle_days=idle_days, limit=limit)

    # ------------------------------------------------------------------ push watcher
    async def check_new(self, mgr: IntegrationManager, session: Any, mailbox: str) -> int:
        """Emit *mailbox* messages newer than its stored UID. The first check of a mailbox only baselines it."""
        st = await mgr.feeds.state(PID)
        try:
            cursors = json.loads(st["cursor"]) if st["cursor"] else {}
        except json.JSONDecodeError:
            cursors = {}
        if "uidvalidity" in cursors or "last_uid" in cursors:
            cursors = {"INBOX": cursors}  # pre-#92 shape: one mailbox, always INBOX; keep its progress
        cursor = cursors.get(mailbox) or {}
        validity = session.uidvalidity
        if not cursor or cursor.get("uidvalidity") != validity or "last_uid" not in cursor:
            last = max(await session.uid_search("UID", "*"), default=0)
            cursors[mailbox] = {"uidvalidity": validity, "last_uid": last}
            await mgr.feeds.record_success(PID, cursor=json.dumps(cursors),
                                           note=f"Watching {mailbox} for new email from now." if not cursor else
                                           f"{mailbox} was reset on the server, so watching restarted from now.")
            return 0
        last = int(cursor["last_uid"])
        new = [u for u in await session.uid_search("UID", f"{last + 1}:*") if u > last]
        kept: list[dict] = []
        if new:
            rows = await session.fetch(new[-WATCH_MAX_NEW:])
            items = []
            for r in sorted(rows, key=lambda r: r["uid"]):
                item = normalize_raw(r["uid"], r["raw"], r["flags"], mailbox=mailbox, hide_codes=redact.enabled(mgr))
                item["_key"] = item.get("message_id") or f"{mailbox}:{validity}:{r['uid']}"
                items.append(item)
            kept = await mgr.emit_items(PID, "feed", items, event="new_email")
            last = max(new)
        cursors[mailbox] = {"uidvalidity": validity, "last_uid": last}
        await mgr.feeds.record_success(PID, cursor=json.dumps(cursors), emitted=len(kept))
        return len(kept)

    async def watch(self, mgr: IntegrationManager) -> None:
        """IDLE push loop with reconnect and exponential backoff; NOOP checks when IDLE is unsupported.

        IDLE only pushes for the selected mailbox, so the first ``folders`` entry stays selected between
        waits; the rest are checked right after each wake, by ``select``-ing over to them in turn.
        """
        while True:
            session = None
            try:
                c = await mgr.get_credentials(PID)
                if not c:
                    return
                folders = c.get("folders") or ["INBOX"]  # already a list: validate() parsed it once
                primary = folders[0]
                session = await open_session(c, primary)
                for mailbox in folders:
                    if session.mailbox != mailbox:
                        await session.select(mailbox)
                    await self.check_new(mgr, session, mailbox)
                if session.mailbox != primary:
                    await session.select(primary)
                while True:
                    if session.supports_idle:
                        await session.idle_wait(IDLE_SECONDS)
                    else:
                        await asyncio.sleep(float(mgr.app.config.integrations.fast_sync_seconds or 300))
                        await session.noop()
                    for mailbox in folders:
                        if session.mailbox != mailbox:
                            await session.select(mailbox)
                        await self.check_new(mgr, session, mailbox)
                    if session.mailbox != primary:
                        await session.select(primary)
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
