"""IMAP email plugin: validation, IDLE push loop with reconnect, NOOP fallback, tools and SMTP sending."""

from __future__ import annotations

import asyncio
import json
import smtplib
from datetime import UTC, datetime

import aioimaplib
import pytest

from sentient.integrations import feeds as feeds_mod
from sentient.integrations.base import IntegrationError
from sentient.integrations.plugins import email_imap
from sentient.integrations.plugins.email_imap import (
    ImapAuthError,
    build_search,
    imap_date,
    normalize_raw,
    parse_fetch,
    parse_folders,
    parse_search,
    parse_uidvalidity,
)

GOOD_PASSWORD = "abcdefghijklmnop"
CREDS = {"host": "imap.gmail.com", "security": "ssl", "port": 993, "username": "me@gmail.com", "password": GOOD_PASSWORD,
         "folders": ["INBOX"], "smtp_host": "smtp.gmail.com", "smtp_port": 465}
GMAIL_KEYS = {"id", "thread_id", "from", "sender_email", "to", "subject", "snippet", "body", "date", "labels", "url"}


def raw_email(frm: str, subject: str, body: str, mid: str, *, extra: str = "") -> bytes:
    return (f"From: {frm}\r\nTo: me@gmail.com\r\nSubject: {subject}\r\nMessage-ID: <{mid}@x>\r\n{extra}"
            f"Date: Tue, 15 Sep 2026 09:00:00 +0000\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}\r\n").encode()


class FakeMailbox:
    def __init__(self, messages: dict[int, bytes], *, idle: bool = True, uidvalidity: str = "7"):
        self.messages = dict(messages)
        self.idle = idle
        self.uidvalidity = uidvalidity
        self.pushes: asyncio.Queue = asyncio.Queue()
        self.fail_opens = 0
        self.opened = 0
        self.closed = 0
        self.noops = 0
        self.searches: list[tuple] = []
        self.flags: dict[int, list[str]] = {}


class FakeSession:
    def __init__(self, box: FakeMailbox):
        self.box = box
        self.mailbox = "INBOX"  # a single-mailbox fake: select() is a no-op, like one folder's own session

    @property
    def uidvalidity(self) -> str:
        return self.box.uidvalidity

    @property
    def supports_idle(self) -> bool:
        return self.box.idle

    async def select(self, mailbox: str) -> None:
        self.mailbox = mailbox

    async def uid_search(self, *criteria: str) -> list[int]:
        self.box.searches.append(criteria)
        uids = sorted(self.box.messages)
        if criteria == ("UID", "*"):
            return uids[-1:]
        if criteria and criteria[0] == "UID":
            low = int(criteria[1].split(":")[0])
            return [u for u in uids if u >= low] or uids[-1:]  # like real servers, "n:*" returns the last uid
        return uids

    async def fetch(self, uids: list[int]) -> list[dict]:
        return [{"uid": u, "flags": self.box.flags.get(u, []), "raw": self.box.messages[u]} for u in uids
                if u in self.box.messages]

    async def idle_wait(self, timeout: float) -> bool:
        push = await self.box.pushes.get()
        if isinstance(push, BaseException):
            raise push
        return bool(push)

    async def noop(self) -> None:
        self.box.noops += 1

    async def close(self) -> None:
        self.box.closed += 1


def install_fake_imap(monkeypatch, box: FakeMailbox) -> None:
    async def open_session(c: dict, mailbox: str = "INBOX"):
        if c.get("password") != GOOD_PASSWORD:
            raise ImapAuthError("The mail server didn't accept the username and app password.")
        if box.fail_opens > 0:
            box.fail_opens -= 1
            raise IntegrationError("Couldn't reach the mail server imap.gmail.com:993 (ConnectionResetError).")
        box.opened += 1
        return FakeSession(box)

    monkeypatch.setattr(email_imap, "open_session", open_session)


class FakeMultiMailbox:
    """Like FakeMailbox, but one message set and UIDVALIDITY per folder, for the `folders` tests."""

    def __init__(self, folders: dict[str, dict[int, bytes]], *, idle: bool = True,
                uidvalidity: dict[str, str] | None = None):
        self.folders = {name: dict(messages) for name, messages in folders.items()}
        self.uidvalidity = {name: (uidvalidity or {}).get(name, "7") for name in self.folders}
        self.idle = idle
        self.pushes: asyncio.Queue = asyncio.Queue()
        self.selected: list[str] = []
        self.flags: dict[str, dict[int, list[str]]] = {name: {} for name in self.folders}


class FakeMultiSession:
    def __init__(self, box: FakeMultiMailbox, mailbox: str):
        self.box = box
        self.mailbox = mailbox

    @property
    def uidvalidity(self) -> str:
        return self.box.uidvalidity[self.mailbox]

    @property
    def supports_idle(self) -> bool:
        return self.box.idle

    async def select(self, mailbox: str) -> None:
        self.box.selected.append(mailbox)
        self.mailbox = mailbox

    async def uid_search(self, *criteria: str) -> list[int]:
        uids = sorted(self.box.folders[self.mailbox])
        if criteria == ("UID", "*"):
            return uids[-1:]
        if criteria and criteria[0] == "UID":
            low = int(criteria[1].split(":")[0])
            return [u for u in uids if u >= low] or uids[-1:]
        return uids

    async def fetch(self, uids: list[int]) -> list[dict]:
        messages, flags = self.box.folders[self.mailbox], self.box.flags[self.mailbox]
        return [{"uid": u, "flags": flags.get(u, []), "raw": messages[u]} for u in uids if u in messages]

    async def idle_wait(self, timeout: float) -> bool:
        push = await self.box.pushes.get()
        if isinstance(push, BaseException):
            raise push
        return bool(push)

    async def noop(self) -> None:
        pass

    async def close(self) -> None:
        pass


def install_fake_multi_imap(monkeypatch, box: FakeMultiMailbox) -> None:
    async def open_session(c: dict, mailbox: str = "INBOX"):
        box.selected.append(mailbox)
        return FakeMultiSession(box, mailbox)

    monkeypatch.setattr(email_imap, "open_session", open_session)


class FakeSMTP:
    instances: list[FakeSMTP] = []

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.login_as: tuple[str, str] | None = None
        self.sent: list = []
        self.quit_called = False
        FakeSMTP.instances.append(self)

    def login(self, user: str, password: str) -> None:
        if password != GOOD_PASSWORD:
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
        self.login_as = (user, password)

    def send_message(self, message) -> None:
        self.sent.append(message)

    def quit(self) -> None:
        self.quit_called = True


@pytest.fixture
def smtp(monkeypatch) -> type[FakeSMTP]:
    FakeSMTP.instances = []
    monkeypatch.setattr(email_imap, "smtp_connect", lambda host, port, timeout: FakeSMTP(host, port))
    return FakeSMTP


async def connect_imap(app, creds: dict | None = None) -> None:
    await app.integrations.store_credentials("email_imap", dict(creds or CREDS))
    await app.integrations.mark_connected("email_imap", "me@gmail.com")


async def next_items(q, timeout: float = 5.0) -> dict:
    while True:
        ev = await asyncio.wait_for(q.get(), timeout)
        if ev["type"] == "source.items":
            return ev["data"]


async def wait_for_cursor(mgr, source: str) -> dict:
    for _ in range(500):
        st = await mgr.feeds.state(source)
        if st["cursor"]:
            return json.loads(st["cursor"])
        await asyncio.sleep(0.01)
    raise AssertionError("the watcher never recorded a baseline")


# ----------------------------------------------------------------------------- connect / validation
async def test_connect_validates_imap_and_smtp(app, keychain, monkeypatch, smtp):
    mgr = app.integrations
    install_fake_imap(monkeypatch, FakeMailbox({1: raw_email("a@b.com", "Hi", "Hello", "m1")}))
    with pytest.raises(IntegrationError, match="Please fill in"):
        await mgr.connect("email_imap", {"host": "imap.gmail.com"})
    with pytest.raises(IntegrationError, match="app password"):
        await mgr.connect("email_imap", {"host": "imap.gmail.com", "username": "me@gmail.com", "password": "wrong"})
    assert (await mgr.integration("email_imap"))["status"] == "error"
    with pytest.raises(IntegrationError, match="valid port"):
        await mgr.connect("email_imap", {"host": "imap.gmail.com", "port": "abc", "username": "me@gmail.com",
                                         "password": GOOD_PASSWORD})
    integ = await mgr.connect("email_imap", {
        "host": "IMAPS://imap.gmail.com/", "port": "", "username": " me@gmail.com ", "password": "abcd efgh ijkl mnop",
        "smtp_host": "smtp.gmail.com", "smtp_port": "465"})
    assert integ["connected"] is True and integ["account_label"] == "me@gmail.com" and integ["status"] == "connected"
    assert json.loads(keychain["integration:email_imap"]) == CREDS
    assert [(s.host, s.port, s.login_as) for s in smtp.instances] == [("smtp.gmail.com", 465, ("me@gmail.com", GOOD_PASSWORD))]
    assert smtp.instances[0].sent == [] and smtp.instances[0].quit_called
    assert integ["auth_type"] == "manual" and integ["triggers"] == [{"event": "new_email", "label": "New email"}]
    assert {t["name"]: t["risk"] for t in integ["tools"]} == {
        "email_imap_search": "read", "email_imap_read": "read", "email_imap_send": "send"}
    assert {f["key"] for f in integ["setup"]["fields"]} == {
        "host", "security", "port", "username", "password", "folders", "smtp_host", "smtp_port"}
    assert next(f for f in integ["setup"]["fields"] if f["key"] == "password")["secret"] is True
    assert "apppasswords" in integ["setup"]["instructions_md"] and "iCloud" in integ["setup"]["instructions_md"]
    assert (await mgr.test("email_imap"))["ok"] is True


async def test_bad_smtp_password_fails_connect(app, monkeypatch, smtp):
    install_fake_imap(monkeypatch, FakeMailbox({}))
    smtp_login_fails = {**CREDS, "password": GOOD_PASSWORD}

    def connect(host, port, timeout):
        s = FakeSMTP(host, port)
        s.login = lambda u, p: (_ for _ in ()).throw(smtplib.SMTPAuthenticationError(535, b"no"))
        return s

    monkeypatch.setattr(email_imap, "smtp_connect", connect)
    with pytest.raises(IntegrationError, match="outgoing mail server didn't accept"):
        await app.integrations.connect("email_imap", {k: str(v) for k, v in smtp_login_fails.items()})
    assert (await app.integrations.integration("email_imap"))["connected"] is False


# ----------------------------------------------------------------------------- push loop
async def test_idle_loop_baseline_push_privacy_and_reconnect(app, keychain, monkeypatch):
    mgr = app.integrations
    monkeypatch.setattr(email_imap, "BACKOFF_BASE_S", 0.01)
    monkeypatch.setattr(feeds_mod, "MIN_INTERVAL_S", 0.0)
    box = FakeMailbox({1: raw_email("Old <old@x.com>", "Old news", "Seen before", "m1")})
    install_fake_imap(monkeypatch, box)
    await connect_imap(app)
    await mgr.set_privacy_filters("email_imap", {"keywords": ["salary"]})
    failures: list[str] = []
    record_failure = mgr.feeds.record_failure

    async def spy(source, exc, **kw):
        res = await record_failure(source, exc, **kw)
        failures.append(res["error"])
        return res

    monkeypatch.setattr(mgr.feeds, "record_failure", spy)
    async with app.bus.subscribe() as q:
        mgr.feeds.start_watch("email_imap")
        try:
            assert await wait_for_cursor(mgr, "email_imap") == {"INBOX": {"uidvalidity": "7", "last_uid": 1}}
            assert mgr.feed_active("email_imap") is True
            box.messages[2] = raw_email("Jane Doe <jane@y.com>", "Lunch?", "Want lunch at 1?", "m2")
            box.messages[3] = raw_email("Boss <boss@x.com>", "Review", "About your salary", "m3")
            box.flags[2] = ["\\Flagged"]
            await box.pushes.put(True)
            batch = await next_items(q)
            assert (batch["source"], batch["event"], batch["origin"]) == ("email_imap", "new_email", "feed")
            assert [i["id"] for i in batch["items"]] == ["2"]
            item = batch["items"][0]
            assert set(item) == GMAIL_KEYS | {"message_id"}
            assert item["sender_email"] == "jane@y.com" and item["subject"] == "Lunch?"
            assert item["body"] == "Want lunch at 1?" and item["message_id"] == "<m2@x>"
            assert item["labels"] == ["INBOX", "UNREAD", "STARRED"] and item["date"].startswith("2026-09-15T09:00")

            # the connection drops and the first reconnect fails: back off, reconnect, catch up
            box.messages[4] = raw_email("Ravi <ravi@z.com>", "Invoice", "Attached", "m4")
            box.fail_opens = 1
            await box.pushes.put(ConnectionResetError("connection lost"))
            batch = await next_items(q)
            assert [i["id"] for i in batch["items"]] == ["4"]
            for _ in range(500):  # the cursor is saved right after publishing
                st = await mgr.feeds.state("email_imap")
                if st["failures"] == 0 and json.loads(st["cursor"])["INBOX"]["last_uid"] == 4:
                    break
                await asyncio.sleep(0.01)
        finally:
            await mgr.feeds.stop_watch("email_imap")
    assert len(failures) == 2 and "connection lost" in failures[0] and "Couldn't reach the mail server" in failures[1]
    st = await mgr.feeds.state("email_imap")
    assert st["failures"] == 0 and st["status"] == "ok" and json.loads(st["cursor"])["INBOX"]["last_uid"] == 4
    assert st["emitted"] == 2 and box.opened == 2 and box.closed >= 2
    assert mgr.feed_active("email_imap") is False  # the watcher stopped


async def test_noop_fallback_when_idle_unsupported_and_uidvalidity_reset(app, monkeypatch):
    mgr = app.integrations
    app.config.integrations.fast_sync_seconds = 0.01  # the NOOP check interval
    box = FakeMailbox({5: raw_email("a@b.com", "First", "x", "m5")}, idle=False)
    install_fake_imap(monkeypatch, box)
    await connect_imap(app)
    async with app.bus.subscribe() as q:
        mgr.feeds.start_watch("email_imap")
        try:
            await wait_for_cursor(mgr, "email_imap")
            box.messages[6] = raw_email("c@d.com", "Second", "y", "m6")
            batch = await next_items(q)
            assert [i["id"] for i in batch["items"]] == ["6"] and box.noops >= 1
            # the server reset the mailbox (new UIDVALIDITY): re-baseline, don't re-emit
            box.uidvalidity = "8"
            for _ in range(500):
                st = await mgr.feeds.state("email_imap")
                if json.loads(st["cursor"])["INBOX"]["uidvalidity"] == "8":
                    break
                await asyncio.sleep(0.01)
            assert "reset" in st["note"]
        finally:
            await mgr.feeds.stop_watch("email_imap")
    assert q.empty() or all(e["type"] != "source.items" for e in [q.get_nowait() for _ in range(q.qsize())])


async def test_auth_failure_in_watch_marks_integration_error(app, monkeypatch):
    mgr = app.integrations
    monkeypatch.setattr(email_imap, "BACKOFF_BASE_S", 3600)
    install_fake_imap(monkeypatch, FakeMailbox({}))
    await connect_imap(app, {**CREDS, "password": "revoked"})
    mgr.feeds.start_watch("email_imap")
    try:
        for _ in range(500):
            if (await mgr.feeds.state("email_imap"))["failures"]:
                break
            await asyncio.sleep(0.01)
        integ = await mgr.integration("email_imap")
        assert integ["status"] == "error" and "app password" in integ["error"]
        st = await mgr.feeds.state("email_imap")
        assert st["status"] == "error" and st["next_attempt_at"]
    finally:
        await mgr.disconnect("email_imap")
    assert not mgr.feeds.watching("email_imap")
    assert (await mgr.feeds.state("email_imap"))["cursor"] is None


# ----------------------------------------------------------------------------- tools
async def test_tools_search_read_and_send_reply(app, ctx, monkeypatch, smtp):
    mgr = app.integrations
    box = FakeMailbox({
        1: raw_email("Jane Doe <jane@y.com>", "Lunch?", "Want lunch at 1?", "m1"),
        2: raw_email("Boss <boss@x.com>", "Review", "Salary", "m2"),
        3: raw_email("Ravi <ravi@z.com>", "Re: Lunch?", "Count me in", "m3", extra="References: <m1@x>\r\n"),
    })
    box.flags[1] = ["\\Seen"]
    install_fake_imap(monkeypatch, box)
    await connect_imap(app)
    await mgr.set_privacy_filters("email_imap", {"emails": ["boss@x.com"]})

    res = await app.registry.get("email_imap_search").call(ctx, {"text": "lunch", "unread_only": True,
                                                                "newer_than_days": 7, "max_results": 5})
    criteria = box.searches[-1]
    assert criteria[:2] == ("TEXT", '"lunch"') and "UNSEEN" in criteria and "SINCE" in criteria
    assert res["count"] == 2 and res["hidden_by_privacy_filters"] == 1
    assert [m["id"] for m in res["messages"]] == ["3", "1"]  # newest first
    assert res["messages"][0]["thread_id"] == "<m1@x>" and res["messages"][1]["labels"] == ["INBOX"]

    read = await app.registry.get("email_imap_read").call(ctx, {"message_id": "1"})
    assert read["body"] == "Want lunch at 1?" and read["attachments"] == [] and read["cc"] is None
    assert (await app.registry.get("email_imap_read").call(ctx, {"message_id": "2"}))["error"].startswith("This email is hidden")
    assert "wasn't found" in (await app.registry.get("email_imap_read").call(ctx, {"message_id": "99"}))["error"]
    assert "email_imap_search" in (await app.registry.get("email_imap_read").call(ctx, {"message_id": "abc"}))["error"]

    sent = await app.registry.get("email_imap_send").call(ctx, {"to": "jane@y.com", "subject": "", "body": "Sure!",
                                                               "bcc": "me2@x.com", "reply_to_message_id": "1"})
    assert sent["sent"] is True and sent["to"] == "jane@y.com"
    msg = smtp.instances[-1].sent[0]
    assert (smtp.instances[-1].host, smtp.instances[-1].port) == ("smtp.gmail.com", 465)
    assert msg["From"] == "me@gmail.com" and msg["Subject"] == "Re: Lunch?" and msg["In-Reply-To"] == "<m1@x>"
    assert msg["References"] == "<m1@x>" and msg["Bcc"] == "me2@x.com" and "Sure!" in msg.get_content()


async def test_send_guesses_smtp_and_reports_friendly_errors(app, ctx, monkeypatch, smtp):
    install_fake_imap(monkeypatch, FakeMailbox({}))
    await connect_imap(app, {**CREDS, "host": "imap.mail.me.com", "smtp_host": "", "smtp_port": None})
    res = await app.registry.get("email_imap_send").call(ctx, {"to": "a@b.com", "subject": "Hi", "body": "Hello"})
    assert res["sent"] is True and (smtp.instances[-1].host, smtp.instances[-1].port) == ("smtp.mail.me.com", 587)

    await connect_imap(app, {**CREDS, "host": "mail.unknown.org", "smtp_host": "", "smtp_port": None})
    res = await app.registry.get("email_imap_send").call(ctx, {"to": "a@b.com", "subject": "Hi", "body": "Hello"})
    assert "SMTP server" in res["error"]

    await app.integrations.disconnect("email_imap")
    res = await app.registry.get("email_imap_search").call(ctx, {"text": "x"})
    assert res == {"error": "Email (IMAP) isn't connected yet. Connect it from Integrations."}


# ----------------------------------------------------------------------------- parsing (real aioimaplib shapes)
def test_parse_imap_responses():
    lines = [b"1 FETCH (UID 41 FLAGS (\\Seen) BODY[] {12}", bytearray(b"Subject: x\r\n"), b")",
             b"2 FETCH (BODY[] {5}", bytearray(b"hello"), b" UID 42 FLAGS ())", b"Fetch completed (0.001 + 0.000 secs)."]
    assert parse_fetch(lines) == [{"uid": 41, "flags": ["\\Seen"], "raw": b"Subject: x\r\n"},
                                  {"uid": 42, "flags": [], "raw": b"hello"}]
    assert parse_search([b"SEARCH 3 1 2", b"SEARCH completed (Success)"]) == [1, 2, 3]
    assert parse_search([b"SEARCH", b"UID SEARCH completed"]) == []
    assert parse_uidvalidity([b"FLAGS (\\Seen)", b"OK [UIDVALIDITY 1234] UIDs valid", b"[READ-WRITE] SELECT completed"]) == "1234"
    assert build_search(text='say "hi"') == ["TEXT", '"say \\"hi\\""']
    assert build_search() == ["ALL"]
    assert imap_date(datetime(2026, 9, 5, tzinfo=UTC)) == "05-Sep-2026"


def test_normalize_html_only_email():
    raw = (b"From: News <news@x.com>\r\nSubject: Weekly\r\nMIME-Version: 1.0\r\n"
           b"Content-Type: multipart/alternative; boundary=b1\r\n\r\n--b1\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
           b"<p>Big <b>news</b> today</p>\r\n--b1--\r\n")
    item = normalize_raw(9, raw, ["\\Seen"])
    assert item["body"] == "Big news today" and item["labels"] == ["INBOX"] and item["thread_id"] == "9"
    assert item["sender_email"] == "news@x.com" and item["message_id"] is None


async def test_imap_hides_codes_and_magic_links(app, ctx, monkeypatch):
    """#128: tool results and pushed feed items carry placeholders, ordinary numbers stay."""
    mgr = app.integrations
    monkeypatch.setattr(feeds_mod, "MIN_INTERVAL_S", 0.0)
    body = ("Use code 730215 to sign in, or click this link:\r\n"
            "https://app.example.com/l/9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c\r\n"
            "Questions? Call 415-555-0134. Invoice 2026-0042.")
    box = FakeMailbox({1: raw_email("Old <old@x.com>", "Old news", "Seen before", "m1")})
    install_fake_imap(monkeypatch, box)
    await connect_imap(app)

    box.messages[2] = raw_email("Acme <hello@acme.example>", "Your sign-in link", body, "m2")
    found = (await app.registry.get("email_imap_search").call(ctx, {"text": "acme"}))["messages"]
    read = await app.registry.get("email_imap_read").call(ctx, {"message_id": "2"})
    item = next(m for m in found if m["id"] == "2")
    for m in (item, read):
        assert "730215" not in m["body"] and "9f8e7d6c5b4a" not in m["body"] and "730215" not in m["snippet"]
        assert "[one-time code hidden]" in m["body"] and "[sign-in link hidden]" in m["body"]
        assert "415-555-0134" in m["body"] and "Invoice 2026-0042" in m["body"]

    # pushed new mail is masked before it is published to proactivity and triggered tasks
    session = await email_imap.open_session(CREDS, "INBOX")
    await email_imap.PLUGIN.check_new(mgr, session, "INBOX")  # baseline
    box.messages[3] = raw_email("Acme <hello@acme.example>", "Code", "Your verification code is 118822", "m3")
    async with app.bus.subscribe() as q:
        await email_imap.PLUGIN.check_new(mgr, session, "INBOX")
        batch = await next_items(q)
    assert batch["items"][0]["body"] == "Your verification code is [one-time code hidden]"


# ----------------------------------------------------------------------------- security (SSL / STARTTLS, #92)
def test_parse_folders_trims_dedupes_and_defaults_to_inbox():
    assert parse_folders("") == ["INBOX"]
    assert parse_folders("   ") == ["INBOX"]
    assert parse_folders("INBOX, Work, , Work") == ["INBOX", "Work"]
    assert parse_folders("Alpha,Beta,Alpha") == ["Alpha", "Beta"]


async def test_security_field_defaults_to_ssl_and_rejects_an_unknown_value(app, monkeypatch, smtp):
    install_fake_imap(monkeypatch, FakeMailbox({}))
    with pytest.raises(IntegrationError, match="security must be 'ssl' or 'starttls'"):
        await app.integrations.connect("email_imap", {"host": "imap.gmail.com", "username": "me@gmail.com",
                                                       "password": GOOD_PASSWORD, "security": "tls1.2"})
    integ = await app.integrations.connect("email_imap", {"host": "imap.gmail.com", "username": "me@gmail.com",
                                                           "password": GOOD_PASSWORD})
    assert integ["connected"] is True  # security left blank: defaults to ssl, same as before #92


class FakeProtocol:
    """Stands in for aioimaplib's IMAP4ClientProtocol, just enough for _starttls to run against."""

    def __init__(self):
        self.transport = object()
        self.loop = asyncio.get_event_loop()
        self.state = aioimaplib.CONNECTED
        self.capabilities = {"IMAP4rev1"}
        self.calls: list[str] = []
        self._tag = 0

    def new_tag(self) -> str:
        self._tag += 1
        return f"A{self._tag}"

    async def execute(self, command) -> aioimaplib.Response:
        self.calls.append(command.name)
        return aioimaplib.Response("OK", [])


class FakeLowClient:
    """Stands in for aioimaplib.IMAP4 / IMAP4_SSL, just enough for ImapSession.open to run against."""

    def __init__(self, kind: str, recorder: list[FakeLowClient], host: str, port: int, timeout: float):
        self.kind, self.host, self.port = kind, host, port
        self.protocol = FakeProtocol()
        recorder.append(self)

    async def wait_hello_from_server(self) -> None:
        pass

    async def login(self, user: str, password: str) -> aioimaplib.Response:
        return aioimaplib.Response("OK" if password == GOOD_PASSWORD else "NO", [])

    async def select(self, mailbox: str) -> aioimaplib.Response:
        return aioimaplib.Response("OK", [b"OK [UIDVALIDITY 9] UIDs valid"])

    def has_capability(self, name: str) -> bool:
        return False

    async def logout(self) -> aioimaplib.Response:
        return aioimaplib.Response("OK", [])


async def test_open_uses_imap4_ssl_by_default_and_plain_imap4_plus_starttls_when_asked(monkeypatch):
    created: list[FakeLowClient] = []
    starttls_calls: list[tuple] = []

    monkeypatch.setattr(email_imap.aioimaplib, "IMAP4_SSL", lambda host, port, timeout: FakeLowClient("ssl", created, host, port, timeout))
    monkeypatch.setattr(email_imap.aioimaplib, "IMAP4", lambda host, port, timeout: FakeLowClient("plain", created, host, port, timeout))

    async def fake_starttls(client, host):
        starttls_calls.append((client, host))

    monkeypatch.setattr(email_imap, "_starttls", fake_starttls)

    s = email_imap.ImapSession({"host": "imap.example.com", "username": "me@example.com", "password": GOOD_PASSWORD})
    assert s.security == "ssl" and s.port == 993  # the default, unchanged from before #92
    await s.open("INBOX")
    assert created[-1].kind == "ssl" and created[-1].port == 993 and starttls_calls == []
    await s.close()

    created.clear()
    s = email_imap.ImapSession({"host": "imap.example.com", "username": "me@example.com", "password": GOOD_PASSWORD,
                                "security": "starttls"})
    assert s.port == 143  # the STARTTLS default, distinct from the SSL default
    await s.open("INBOX")
    assert created[-1].kind == "plain" and created[-1].port == 143
    assert starttls_calls == [(s.client, "imap.example.com")]
    await s.close()

    created.clear()
    s = email_imap.ImapSession({"host": "imap.example.com", "port": 1143, "username": "me@example.com",
                                "password": GOOD_PASSWORD, "security": "STARTTLS"})  # case-insensitive, explicit port kept
    assert s.port == 1143
    await s.open("INBOX")
    assert created[-1].kind == "plain" and created[-1].port == 1143
    await s.close()


async def test_starttls_sends_the_command_upgrades_the_transport_and_refreshes_capabilities(monkeypatch):
    protocol = FakeProtocol()
    original_transport = protocol.transport
    tls_transport = object()  # a distinct object: start_tls returns a NEW transport, never the old one
    captured: dict = {}

    async def fake_start_tls(transport, proto, ssl_context, server_hostname=None):
        captured["args"] = (transport, proto, server_hostname)
        return tls_transport

    monkeypatch.setattr(asyncio.get_running_loop(), "start_tls", fake_start_tls)
    client = type("Client", (), {"protocol": protocol})()

    await email_imap._starttls(client, "imap.example.com")

    assert protocol.calls == ["STARTTLS", "CAPABILITY"]  # STARTTLS first, then a fresh CAPABILITY over the upgraded link
    assert protocol.state == aioimaplib.NONAUTH  # the server sends no new greeting after STARTTLS
    assert protocol.capabilities == set()  # pre-TLS capabilities are discarded, not trusted
    assert captured["args"] == (original_transport, protocol, "imap.example.com")
    assert protocol.transport is tls_transport  # every command after this must go out over the upgraded transport


async def test_starttls_raises_when_the_server_refuses(monkeypatch):
    protocol = FakeProtocol()

    async def refuse(command) -> aioimaplib.Response:
        protocol.calls.append(command.name)
        return aioimaplib.Response("NO", [b"STARTTLS not supported"])

    protocol.execute = refuse
    client = type("Client", (), {"protocol": protocol})()
    with pytest.raises(IntegrationError, match="refused to start TLS"):
        await email_imap._starttls(client, "imap.example.com")


# ----------------------------------------------------------------------------- folders (watch other mailboxes, #92)
async def test_connect_fails_fast_when_a_watched_folder_does_not_exist(app, monkeypatch, smtp):
    box = FakeMailbox({})
    real_select = FakeSession.select

    async def select(self, mailbox: str) -> None:
        if mailbox == "Ghost":
            raise IntegrationError("There is no mailbox called 'Ghost'.")
        await real_select(self, mailbox)

    monkeypatch.setattr(FakeSession, "select", select)
    install_fake_imap(monkeypatch, box)
    with pytest.raises(IntegrationError, match="no mailbox called 'Ghost'"):
        await app.integrations.connect("email_imap", {"host": "imap.gmail.com", "username": "me@gmail.com",
                                                       "password": GOOD_PASSWORD, "folders": "INBOX, Ghost"})


async def test_watch_checks_every_configured_folder_with_its_own_cursor(app, monkeypatch):
    mgr = app.integrations
    box = FakeMultiMailbox(
        {"INBOX": {1: raw_email("a@b.com", "Hi", "x", "m1")}, "Work": {5: raw_email("c@d.com", "Job", "y", "m5")}},
        uidvalidity={"INBOX": "7", "Work": "9"},
    )
    install_fake_multi_imap(monkeypatch, box)
    await connect_imap(app, {**CREDS, "folders": ["INBOX", "Work"]})
    async with app.bus.subscribe() as q:
        mgr.feeds.start_watch("email_imap")
        try:
            cursor = await wait_for_cursor(mgr, "email_imap")
            for _ in range(500):
                cursor = json.loads((await mgr.feeds.state("email_imap"))["cursor"])
                if "Work" in cursor:
                    break
                await asyncio.sleep(0.01)
            assert cursor == {"INBOX": {"uidvalidity": "7", "last_uid": 1}, "Work": {"uidvalidity": "9", "last_uid": 5}}
            assert box.selected[:2] == ["INBOX", "Work"]  # the baseline pass visits both, primary first

            box.folders["Work"][6] = raw_email("e@f.com", "Second", "z", "m6")
            box.pushes.put_nowait(True)
            batch = await next_items(q)
            assert batch["source"] == "email_imap" and batch["event"] == "new_email"
            item = batch["items"][0]
            assert item["subject"] == "Second" and "Work" in item["labels"]
            for _ in range(500):  # the IDLE wait is on INBOX; Work's wake-check is right after
                cursor = json.loads((await mgr.feeds.state("email_imap"))["cursor"])
                if cursor.get("Work", {}).get("last_uid") == 6:
                    break
                await asyncio.sleep(0.01)
            assert cursor["Work"] == {"uidvalidity": "9", "last_uid": 6}
            assert cursor["INBOX"] == {"uidvalidity": "7", "last_uid": 1}  # untouched by the other folder's new mail
        finally:
            await mgr.feeds.stop_watch("email_imap")


async def test_check_new_migrates_the_pre_folders_flat_cursor_shape(app, monkeypatch):
    mgr = app.integrations
    box = FakeMailbox({3: raw_email("a@b.com", "Hi", "x", "m1")})
    install_fake_imap(monkeypatch, box)
    await connect_imap(app)
    await mgr.feeds.record_success("email_imap", cursor=json.dumps({"uidvalidity": "7", "last_uid": 2}))

    n = await email_imap.PLUGIN.check_new(mgr, FakeSession(box), "INBOX")

    assert n == 1  # continues from the old last_uid=2: uid 3 is new, not re-baselined
    cursor = json.loads((await mgr.feeds.state("email_imap"))["cursor"])
    assert cursor == {"INBOX": {"uidvalidity": "7", "last_uid": 3}}
