"""Follow-ups (#110): dropped email threads become suggestions with a draft; answered ones do not.

Gmail is mocked with respx, IMAP with a fake session, the model with FakeProvider. Nothing is ever sent.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from sentient import secrets
from sentient.app import SentientApp
from sentient.integrations.plugins import email_imap
from sentient.integrations.plugins.email_imap import find_sent_mailbox, group_threads, parse_list
from sentient.proactivity import followups
from sentient.proactivity import service as pro_service
from tests.conftest import FakeProvider
from tests.proactivity.conftest import FakeTasks

GMAIL = "gmail.googleapis.com"
API = "/gmail/v1/users/me"
ME = "maya@example.com"
NOW = datetime.now(UTC)


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


@pytest.fixture
async def mail_app(config, isolated_home, monkeypatch):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "followups.db", enable_background=False)
    await a.start()
    monkeypatch.setattr(a, "tasks", FakeTasks())
    a.config.assistant.user_name = "Maya Rao"
    a.config.assistant.timezone = "UTC"
    a.fake = llm
    yield a
    await a.stop()


async def connect_gmail(app, keychain) -> None:
    keychain["google_oauth_client"] = json.dumps({"client_id": "cid", "client_secret": "csec"})
    await app.integrations.store_credentials("gmail", {"access_token": "at1", "refresh_token": "rt1",
                                                       "expires_at": time.time() + 3600})
    await app.integrations.mark_connected("gmail", ME)


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def gmsg(mid: str, tid: str, frm: str, to: str, subject: str, body: str, *, days: float,
         labels=("INBOX",), headers: dict | None = None) -> dict:
    hs = [{"name": "From", "value": frm}, {"name": "To", "value": to}, {"name": "Subject", "value": subject},
          {"name": "Message-ID", "value": f"<{mid}@mail.example>"},
          *[{"name": k, "value": v} for k, v in (headers or {}).items()]]
    when = NOW - timedelta(days=days)
    return {"id": mid, "threadId": tid, "snippet": body[:80], "labelIds": list(labels),
            "internalDate": str(int(when.timestamp() * 1000)),
            "payload": {"mimeType": "text/plain", "headers": hs, "body": {"data": b64(body)}}}


PRIYA = "Priya Shah <priya@acme.example>"
ROHAN = "Rohan Mehta <rohan@studio.example>"


def base_threads() -> dict[str, list[dict]]:
    return {
        # waiting on you: sent to Maya directly 4 days ago, no answer -> suggested
        "t-direct": [gmsg("m1", "t-direct", PRIYA, ME, "Invoice for September",
                          "Hi Maya, could you confirm the invoice amount by Friday? Thanks, Priya", days=4)],
        # same, but Maya answered 2 days ago -> not
        "t-replied": [
            gmsg("m2", "t-replied", "Arjun Nair <arjun@acme.example>", ME, "Venue for the offsite",
                 "Maya, can you check whether the venue is free on the 14th?", days=5),
            gmsg("m3", "t-replied", f"Maya Rao <{ME}>", "arjun@acme.example", "Re: Venue for the offsite",
                 "Yes, it is free. I booked it.", days=2, labels=("SENT",)),
        ],
        # newsletter -> not
        "t-news": [gmsg("m4", "t-news", "Weekly Digest <hello@digest.example>", ME, "This week in design",
                        "Ten things to read this week. What did you think?", days=4,
                        headers={"List-Unsubscribe": "<mailto:unsub@digest.example>"})],
        # no-reply sender -> not
        "t-noreply": [gmsg("m5", "t-noreply", "Shop <no-reply@shop.example>", ME, "Your order",
                           "Can you rate your order? Tell us how we did.", days=4)],
        # quiet for longer than the 21 day cap -> not
        "t-old": [gmsg("m6", "t-old", "Kabir Das <kabir@old.example>", ME, "Old question",
                       "Could you send me the deck from March?", days=30)],
        # waiting on them: Maya asked Rohan something 5 days ago, no reply -> suggested
        "t-sent": [gmsg("m7", "t-sent", f"Maya Rao <{ME}>", ROHAN, "Saturday",
                        "Hi Rohan, are we still on for Saturday? Let me know. Maya", days=5, labels=("SENT",))],
    }


def mock_gmail(router, threads: dict[str, list[dict]]) -> None:
    router.get(host=GMAIL, path=f"{API}/threads").mock(
        return_value=httpx.Response(200, json={"threads": [{"id": t} for t in threads]}))
    for tid, msgs in threads.items():
        router.get(host=GMAIL, path=f"{API}/threads/{tid}").mock(
            return_value=httpx.Response(200, json={"id": tid, "messages": msgs}))


def yes(about: str, draft: str, confidence: float = 0.85) -> str:
    return "<think>short</think>```json\n" + json.dumps(
        {"needs_follow_up": True, "about": about, "draft": draft, "confidence": confidence}) + "\n```"


NO = '{"needs_follow_up": false}'


async def notes(app) -> list[dict]:
    return [n for n in await app.notifications.list(limit=50) if n["kind"] == "proactive"]


# ----------------------------------------------------------------------------- gmail, both directions
async def test_gmail_both_directions_with_drafts(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    app.fake.text_replies += [
        yes("the invoice", "Hi Priya,\nThe amount is right. I will confirm by Friday.\nMaya"),
        yes("Saturday", "Hi Rohan, just checking in: are we still on for Saturday?\nMaya"),
    ]
    with respx.mock() as router:  # any unmocked request (like a send) fails the test
        mock_gmail(router, base_threads())
        out = await app.proactivity.run_followups(now=NOW)
        q = router.calls[0].request.url.params["q"]
    assert "newer_than:21d" in q and "older_than:3d" in q and "-category:promotions" in q
    assert len(app.fake.text_calls) == 2  # only the two real candidates reached the model
    assert all(c["role"] == "fast" for c in app.fake.text_calls)
    assert all(len(c["messages"][1]["content"]) < 2500 for c in app.fake.text_calls)
    assert [r["suggestion"]["description"] for r in out] == [
        "Priya is waiting for your reply about the invoice",
        "No reply from Rohan about Saturday yet",
    ]
    reply, nudge = (r["suggestion"] for r in out)
    assert reply["suggestion_type"] == "follow_up_reply" and nudge["suggestion_type"] == "follow_up_nudge"
    assert reply["follow_up"] == {
        "kind": "waiting_on_you", "person": "Priya Shah", "person_email": "priya@acme.example",
        "to": "priya@acme.example", "subject": "Re: Invoice for September",
        "draft": "Hi Priya,\nThe amount is right. I will confirm by Friday.\nMaya", "days_waiting": 4,
        "thread_id": "t-direct", "message_id": "m1",
    }
    assert nudge["follow_up"]["kind"] == "waiting_on_them" and nudge["follow_up"]["to"] == ROHAN
    assert nudge["follow_up"]["days_waiting"] == 5 and nudge["follow_up"]["message_id"] == "m7"
    assert reply["source_event"]["source"] == "gmail" and reply["source_event"]["event_type"] == "follow_up"
    assert reply["source_event"]["item_id"] == "t-direct:<m1@mail.example>"
    assert reply["source_event"]["url"].endswith("#all/t-direct")
    assert nudge["source_event"]["summary"] == "You to Rohan Mehta: Saturday"
    ns = await notes(app)
    assert len(ns) == 2
    by_type = {n["payload"]["suggestion"]["suggestion_type"]: n for n in ns}
    assert by_type["follow_up_reply"]["title"] == "Gmail: Priya Shah: Invoice for September"
    assert "> The amount is right." in by_type["follow_up_reply"]["message"]
    assert by_type["follow_up_reply"]["payload"]["status"] == "pending"


async def test_second_run_dedupes_without_model_calls(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    app.fake.text_replies += [yes("the invoice", "Sure, confirming by Friday. Maya"), yes("Saturday", "Still on? Maya")]
    with respx.mock() as router:
        mock_gmail(router, base_threads())
        assert len(await app.proactivity.run_followups(now=NOW)) == 2
        calls = len(app.fake.text_calls)
        assert await app.proactivity.run_followups(now=NOW) == []
    assert len(app.fake.text_calls) == calls
    assert len(await notes(app)) == 2


async def test_dismissed_thread_is_not_suggested_again_and_tunes_type(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    threads = {"t-direct": base_threads()["t-direct"]}
    app.fake.text_replies.append(yes("the invoice", "Confirming by Friday. Maya"))
    with respx.mock() as router:
        mock_gmail(router, threads)
        [rec] = await app.proactivity.run_followups(now=NOW)
        assert await app.proactivity.act_on_suggestion(rec["notification_id"], "dismiss") == {"ok": True}
        app.fake.text_replies.append(yes("the invoice", "Confirming by Friday. Maya"))
        assert await app.proactivity.run_followups(now=NOW) == []
    assert len(app.fake.text_calls) == 1
    prefs = {p["suggestion_type"]: p for p in await app.proactivity.preferences()}
    assert prefs["follow_up_reply"]["dismissals"] == 1 and prefs["follow_up_reply"]["threshold"] == 0.75


async def test_a_new_message_in_a_suggested_thread_is_judged_again(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    threads = {"t-direct": base_threads()["t-direct"]}
    app.fake.text_replies.append(NO)
    with respx.mock() as router:
        mock_gmail(router, threads)
        assert await app.proactivity.run_followups(now=NOW) == []
    threads["t-direct"].append(gmsg("m9", "t-direct", PRIYA, ME, "Re: Invoice for September",
                                    "Maya, could you also send the PO number?", days=3.5))
    app.fake.text_replies.append(yes("the PO number", "I will send the PO number this afternoon. Maya"))
    with respx.mock() as router:
        mock_gmail(router, threads)
        [rec] = await app.proactivity.run_followups(now=NOW)
    assert rec["suggestion"]["follow_up"]["message_id"] == "m9"
    assert rec["suggestion"]["description"] == "Priya is waiting for your reply about the PO number"


async def test_model_says_no_need_and_is_not_asked_again(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    app.fake.text_replies += [NO, NO]
    with respx.mock() as router:
        mock_gmail(router, base_threads())
        assert await app.proactivity.run_followups(now=NOW) == []
        assert await app.proactivity.run_followups(now=NOW) == []
    assert len(app.fake.text_calls) == 2
    assert await notes(app) == []


async def test_malformed_model_json_is_skipped_and_retried_next_time(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    threads = {"t-direct": base_threads()["t-direct"]}
    app.fake.text_replies += ["Sure! I think Maya should reply to Priya.", '{"needs_follow_up": true, "draft": ""}']
    with respx.mock() as router:
        mock_gmail(router, threads)
        assert await app.proactivity.run_followups(now=NOW) == []
        assert await app.proactivity.run_followups(now=NOW) == []
        app.fake.text_replies.append('{"needs_follow_up": "yes", "about": "the invoice", "draft": "Confirmed. Maya"')
        [rec] = await app.proactivity.run_followups(now=NOW)
    assert len(app.fake.text_calls) == 3
    assert rec["suggestion"]["confidence"] == followups.DEFAULT_CONFIDENCE


async def test_prompt_forbids_placeholders(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    app.fake.text_replies.append(NO)
    with respx.mock() as router:
        mock_gmail(router, {"t-direct": base_threads()["t-direct"]})
        await app.proactivity.run_followups(now=NOW)
    system = app.fake.text_calls[0]["messages"][0]["content"]
    assert "Never use placeholders, brackets or blanks" in system and "Leave out anything you do not know" in system
    assert "[day]" not in system


async def test_draft_with_a_placeholder_is_dropped_and_retried(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    threads = {"t-direct": base_threads()["t-direct"]}
    app.fake.text_replies += [yes("the invoice", "Hi Priya, I can confirm by [day]. Maya"),
                              yes("the invoice", "Hi Priya, the amount is TBD. Maya")]
    with respx.mock() as router:
        mock_gmail(router, threads)
        assert await app.proactivity.run_followups(now=NOW) == []
        assert await app.proactivity.run_followups(now=NOW) == []  # not marked as decided: asked again
        app.fake.text_replies.append(yes("the invoice", "Hi Priya, the amount is right. I will confirm on Friday. Maya"))
        [rec] = await app.proactivity.run_followups(now=NOW)
    assert len(app.fake.text_calls) == 3
    assert rec["suggestion"]["follow_up"]["draft"] == "Hi Priya, the amount is right. I will confirm on Friday. Maya"
    assert len(await notes(app)) == 1


@pytest.mark.parametrize("draft,found", [
    ("I can confirm by [day].", True),
    ("Hi [name], thanks!", True),
    ("See you on {date}.", True),
    ("See you on <date>.", True),
    ("The total is XX rupees.", True),
    ("Timing is TBD for now.", True),
    ("Thanks, (your name)", True),
    ("Write to me at <maya@example.com> any time.", False),
    ("Sounds good, see you Saturday at 10. Maya", False),
    ("Is the 2x price right?", False),
])
def test_has_placeholder(draft, found):
    assert followups.has_placeholder(draft) is found


async def test_only_listed_accounts_are_read(mail_app, keychain, monkeypatch):
    app = mail_app
    await connect_gmail(app, keychain)
    fake = FakeImap(imap_boxes(), LISTING)
    await connect_imap(app, monkeypatch, fake)
    app.config.proactivity.followups.sources = ["gmail"]
    app.fake.text_replies += [NO, NO]
    with respx.mock() as router:
        mock_gmail(router, base_threads())
        await app.proactivity.run_followups(now=NOW)
    assert fake.selected == []  # the connected IMAP account was never opened
    assert len(app.fake.text_calls) == 2  # only the two Gmail candidates

    app.config.proactivity.followups.sources = ["email_imap"]
    app.fake.text_replies += [NO, NO]
    with respx.mock() as router:  # any Gmail request would be unmocked and fail
        await app.proactivity.run_followups(now=NOW)
    assert "Sent Messages" in fake.selected and len(app.fake.text_calls) == 4

    app.config.proactivity.followups.sources = []
    with respx.mock():
        assert await app.proactivity.run_followups(now=NOW) == []
    assert len(app.fake.text_calls) == 4


async def test_caps_on_candidates_and_suggestions(mail_app, keychain, monkeypatch):
    app = mail_app
    await connect_gmail(app, keychain)
    many = {f"t{i}": [gmsg(f"x{i}", f"t{i}", f"Person {i} <p{i}@acme.example>", ME, f"Question {i}",
                           f"Could you look at item {i}?", days=4 + i * 0.1)] for i in range(6)}
    app.fake.text_replies += [yes(f"item {i}", f"Looking now. Maya {i}") for i in range(6)]
    with respx.mock() as router:
        mock_gmail(router, many)
        out = await app.proactivity.run_followups(now=NOW)
    assert len(out) == 3 and len(app.fake.text_calls) == 3  # followups.max_suggestions
    # newest first: the threads waiting 4.0, 4.1 and 4.2 days
    assert [r["suggestion"]["follow_up"]["thread_id"] for r in out] == ["t0", "t1", "t2"]

    monkeypatch.setattr(pro_service, "FOLLOWUP_MAX_CANDIDATES", 2)
    app.fake.text_replies[:] = [NO] * 6
    with respx.mock() as router:
        mock_gmail(router, many)
        assert await app.proactivity.run_followups(now=NOW) == []
    assert len(app.fake.text_calls) == 5  # three new threads were left, only two reached the model


async def test_privacy_filters_apply(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    await app.integrations.set_privacy_filters("gmail", {"emails": ["priya@acme.example", "rohan@studio.example"]})
    with respx.mock() as router:
        mock_gmail(router, base_threads())
        assert await app.proactivity.run_followups(now=NOW) == []
    assert app.fake.text_calls == []


async def test_off_when_proactivity_or_followups_off(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    with respx.mock(assert_all_called=False) as router:
        mock_gmail(router, base_threads())
        app.config.proactivity.enabled = False
        assert await app.proactivity.run_followups(now=NOW) == []
        app.config.proactivity.enabled = True
        app.config.proactivity.followups.enabled = False
        assert await app.proactivity.run_followups(now=NOW) == []
        assert not router.calls
    assert (await app.proactivity.status())["followups"] == {"enabled": False, "last_run_at": None}


async def test_runs_once_a_day_in_the_morning(mail_app):
    pro = mail_app.proactivity
    mail_app.config.assistant.timezone = "Asia/Kolkata"
    early = datetime(2026, 10, 9, 1, 0, tzinfo=UTC)    # 06:30 in Kolkata
    morning = datetime(2026, 10, 9, 3, 0, tzinfo=UTC)  # 08:30
    assert await pro._followups_due(early) is False
    assert await pro._followups_due(morning) is True
    await pro.run_followups(now=morning)  # nothing connected: records the run anyway
    assert await pro._followups_due(morning + timedelta(hours=8)) is False
    assert await pro._followups_due(morning + timedelta(days=1)) is True
    assert (await pro.status())["followups"]["last_run_at"] == morning.isoformat()


async def test_approving_creates_a_task_that_sends_exactly_the_draft(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    app.fake.text_replies += [yes("the invoice", "Hi Priya, confirming by Friday. Maya"), NO]
    with respx.mock() as router:
        mock_gmail(router, base_threads())
        [rec] = await app.proactivity.run_followups(now=NOW)
    res = await app.proactivity.act_on_suggestion(rec["notification_id"], "approve")
    assert res == {"ok": True, "task_id": "task1"}
    prompt = app.tasks.created[0]["prompt"]
    assert prompt.startswith('Send this reply to Priya Shah about "Invoice for September".')
    assert 'gmail_reply with message_id "m1"' in prompt and "Hi Priya, confirming by Friday. Maya" in prompt


async def test_gmail_nudge_reply_goes_to_the_original_recipients(mail_app, keychain):
    app = mail_app
    await connect_gmail(app, keychain)
    ctx = app.agent.tool_context(None, "test")
    sent_msg = gmsg("m7", "t-sent", f"Maya Rao <{ME}>", ROHAN, "Saturday", "Still on?", days=5, labels=("SENT",))
    with respx.mock() as router:
        router.get(host=GMAIL, path=f"{API}/messages/m7").mock(return_value=httpx.Response(200, json=sent_msg))
        send = router.post(host=GMAIL, path=f"{API}/messages/send").mock(
            return_value=httpx.Response(200, json={"id": "s1", "threadId": "t-sent"}))
        res = await app.registry.get("gmail_reply").call(ctx, {"message_id": "m7", "body": "Just checking in."})
        raw = base64.urlsafe_b64decode(json.loads(send.calls.last.request.content)["raw"]).decode()
    assert res["sent"] is True and res["to"] == ROHAN
    assert "To: Rohan Mehta <rohan@studio.example>" in raw and "Subject: Re: Saturday" in raw


# ----------------------------------------------------------------------------- IMAP
def raw_email(frm: str, to: str, subject: str, body: str, mid: str, *, days: float, extra: str = "") -> bytes:
    date = format_datetime(NOW - timedelta(days=days))
    return (f"From: {frm}\r\nTo: {to}\r\nSubject: {subject}\r\nMessage-ID: <{mid}@mail.example>\r\n{extra}"
            f"Date: {date}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}\r\n").encode()


class FakeImap:
    """Mailboxes {name: {uid: raw}} behind the ImapSession methods follow-ups use."""

    def __init__(self, boxes: dict[str, dict[int, bytes]], listing: list[bytes]):
        self.boxes = boxes
        self.listing = listing
        self.mailbox = "INBOX"
        self.selected: list[str] = []

    async def list_mailboxes(self):
        return parse_list(self.listing)

    async def select(self, mailbox: str) -> None:
        self.mailbox = mailbox
        self.selected.append(mailbox)

    async def uid_search(self, *criteria: str) -> list[int]:
        return sorted(self.boxes.get(self.mailbox, {}))

    async def fetch_headers(self, uids: list[int]) -> list[dict]:
        box = self.boxes.get(self.mailbox, {})
        return [{"uid": u, "flags": ["\\Seen"], "raw": box[u].split(b"\r\n\r\n")[0] + b"\r\n\r\n"} for u in uids if u in box]

    async def fetch(self, uids: list[int]) -> list[dict]:
        box = self.boxes.get(self.mailbox, {})
        return [{"uid": u, "flags": ["\\Seen"], "raw": box[u]} for u in uids if u in box]

    async def close(self) -> None:
        return None


IMAP_ME = "maya@mail.example"
LISTING = [b'LIST (\\HasNoChildren) "/" "INBOX"', b'LIST (\\HasNoChildren \\Sent) "/" "Sent Messages"',
           b'LIST (\\HasNoChildren \\Trash) "/" Trash', b"LIST completed"]


def imap_boxes() -> dict[str, dict[int, bytes]]:
    return {
        "INBOX": {
            1: raw_email(PRIYA, IMAP_ME, "Invoice for September", "Maya, could you confirm the invoice amount?", "i1", days=4),
            2: raw_email("Arjun Nair <arjun@acme.example>", IMAP_ME, "Venue", "Can you check the venue?", "i2", days=5),
            3: raw_email("Digest <hello@digest.example>", IMAP_ME, "Weekly news", "What did you think?", "i3", days=4,
                         extra="List-Id: <news.digest.example>\r\n"),
        },
        "Sent Messages": {
            7: raw_email(f"Maya Rao <{IMAP_ME}>", "arjun@acme.example", "Re: Venue", "It is free.", "s7", days=2,
                         extra="In-Reply-To: <i2@mail.example>\r\nReferences: <i2@mail.example>\r\n"),
            8: raw_email(f"Maya Rao <{IMAP_ME}>", ROHAN, "Saturday", "Are we still on for Saturday?", "s8", days=5),
        },
    }


async def connect_imap(app, monkeypatch, fake: FakeImap) -> None:
    async def open_session(c: dict, mailbox: str = "INBOX"):
        fake.mailbox = mailbox
        return fake

    monkeypatch.setattr(email_imap, "open_session", open_session)
    await app.integrations.store_credentials("email_imap", {"host": "imap.mail.example", "port": 993,
                                                            "username": IMAP_ME, "password": "app-password"})
    await app.integrations.mark_connected("email_imap", IMAP_ME)


async def test_imap_both_directions_via_special_use_sent(mail_app, monkeypatch):
    app = mail_app
    fake = FakeImap(imap_boxes(), LISTING)
    await connect_imap(app, monkeypatch, fake)
    app.fake.text_replies += [yes("the invoice", "Confirming today. Maya"), yes("Saturday", "Still on? Maya")]
    out = await app.proactivity.run_followups(now=NOW)
    assert [r["suggestion"]["description"] for r in out] == [
        "Priya is waiting for your reply about the invoice", "No reply from Rohan about Saturday yet"]
    reply, nudge = (r["suggestion"]["follow_up"] for r in out)
    assert reply["mailbox"] == "INBOX" and reply["message_id"] == "1"
    assert nudge["mailbox"] == "Sent Messages" and nudge["message_id"] == "8"
    assert out[0]["suggestion"]["source_event"]["source"] == "email_imap"
    assert len(app.fake.text_calls) == 2  # the answered venue thread and the list mail never reach the model
    assert "Sent Messages" in fake.selected
    app.tasks.created.clear()
    await app.proactivity.act_on_suggestion(out[0]["notification_id"], "approve")
    await app.proactivity.act_on_suggestion(out[1]["notification_id"], "approve")
    first, second = (t["prompt"] for t in app.tasks.created)
    assert 'reply_to_message_id "1"' in first and "Confirming today. Maya" in first
    assert "reply_to_message_id" not in second and f'to "{ROHAN}"' in second


async def test_imap_without_a_sent_mailbox_is_skipped_quietly(mail_app, monkeypatch):
    app = mail_app
    fake = FakeImap(imap_boxes(), [b'LIST (\\HasNoChildren) "/" "INBOX"', b'LIST (\\HasNoChildren) "/" "Archive"'])
    await connect_imap(app, monkeypatch, fake)
    assert await app.proactivity.run_followups(now=NOW) == []
    assert app.fake.text_calls == []
    data = await app.integrations.recent_threads("email_imap", newer_than_days=21, idle_days=3)
    assert data["threads"] == [] and data["note"] == "no_sent_mailbox"


def test_parse_list_and_find_sent_mailbox():
    entries = parse_list([b'LIST (\\HasNoChildren) "/" "INBOX"', b'LIST (\\HasNoChildren \\Sent) "/" "[Gmail]/Sent Mail"',
                          b'LIST (\\Noselect) "/" {7}', b"[Gmail]", b'(\\HasNoChildren) "." Sent', b"LIST completed"])
    assert entries[1] == ({"\\hasnochildren", "\\sent"}, "[Gmail]/Sent Mail")
    assert entries[2][1] == "[Gmail]" and entries[3][1] == "Sent"
    assert find_sent_mailbox(entries) == "[Gmail]/Sent Mail"
    assert find_sent_mailbox([({"\\hasnochildren"}, "INBOX"), (set(), "Sent Items")]) == "Sent Items"
    assert find_sent_mailbox([(set(), "INBOX"), (set(), "[gmail]/sent mail")]) == "[gmail]/sent mail"
    assert find_sent_mailbox([(set(), "INBOX"), (set(), "Archive")]) is None


def test_group_threads_links_replies_and_references():
    msgs = [
        {"id": "1", "mailbox": "INBOX", "message_id": "<a@x>", "_refs": []},
        {"id": "2", "mailbox": "Sent", "message_id": "<b@x>", "_refs": ["<a@x>"]},
        {"id": "3", "mailbox": "INBOX", "message_id": "<c@x>", "_refs": ["<a@x>", "<b@x>"]},
        {"id": "4", "mailbox": "INBOX", "message_id": "<d@x>", "_refs": []},
    ]
    groups = sorted(sorted(m["id"] for m in g) for g in group_threads(msgs))
    assert groups == [["1", "2", "3"], ["4"]]


# ----------------------------------------------------------------------------- deterministic pieces
class Cfg:
    waiting_on_you_days = 3
    waiting_on_them_days = 4
    max_age_days = 21


def m(**kw) -> dict:
    base = {"id": "x", "message_id": "<x@y>", "from": PRIYA, "sender_email": "priya@acme.example", "to": ME,
            "subject": "Hello", "body": "Could you look at this?", "date": (NOW - timedelta(days=4)).isoformat(),
            "labels": ["INBOX"], "headers": {}, "from_me": False}
    return {**base, **kw}


@pytest.mark.parametrize("msg,reason", [
    (m(to="team@acme.example", cc=ME), "not sent to you directly"),
    (m(headers={"Precedence": "bulk"}), "mailing list or bulk mail"),
    (m(labels=["INBOX", "CATEGORY_UPDATES"]), "mailing list or bulk mail"),
    (m(headers={"Auto-Submitted": "auto-generated"}), "mailing list or bulk mail"),
    (m(sender_email="notifications@github.example"), "automated sender"),
    (m(subject="Invitation: Design review @ Thu"), "calendar invite"),
    (m(sender_email=ME), "sent from your own address"),
    (m(body="Thanks!"), "nothing to answer"),
    (m(date=(NOW - timedelta(days=2)).isoformat()), "not waiting long enough"),
])
def test_classify_skips(msg, reason):
    c, why = followups.classify("gmail", {"thread_id": "t", "messages": [msg]}, {ME}, Cfg, NOW)
    assert c is None and why == reason


def test_classify_waiting_on_them_needs_a_question():
    mine = m(from_me=True, sender_email=ME, to=ROHAN, body="Here are the photos from Saturday.",
             date=(NOW - timedelta(days=5)).isoformat())
    assert followups.classify("gmail", {"thread_id": "t", "messages": [mine]}, {ME}, Cfg, NOW) == (None, "no question asked")
    asked = {**mine, "body": "Could you send me the final cut?"}
    c, _ = followups.classify("gmail", {"thread_id": "t", "messages": [asked]}, {ME}, Cfg, NOW)
    assert c is not None and c.kind == "waiting_on_them" and c.person == "Rohan" and c.days == 5
    to_self = {**asked, "to": ME}
    assert followups.classify("gmail", {"thread_id": "t", "messages": [to_self]}, {ME}, Cfg, NOW) == (None, "a note to yourself")


def test_parse_decision_is_tolerant():
    d = followups.parse_decision('<think>hmm</think>\n{"needs_follow_up": true, "about": "About: the invoice.", '
                                 '"draft": "Subject: Re: Invoice\\nHi Priya, done.", "confidence": 90}')
    assert d is not None and d.needed and d.about == "the invoice" and d.draft == "Hi Priya, done." and d.confidence == 0.9
    assert followups.parse_decision('[{"needs_follow_up": "no"}]').needed is False  # type: ignore[union-attr]
    assert followups.parse_decision("no json here") is None
    assert followups.parse_decision('{"about": "x"}') is None


def test_config_schema_describes_followups():
    from sentient.config.schema import SentientConfig

    schema = SentientConfig.model_json_schema()
    fu = schema["$defs"]["FollowUpsConfig"]["properties"]
    assert set(fu) == {"enabled", "sources", "waiting_on_you_days", "waiting_on_them_days", "max_age_days",
                       "max_suggestions"}
    assert all(p.get("description") for p in fu.values())
    cfg = SentientConfig()
    assert cfg.proactivity.followups.sources == ["gmail", "email_imap"]
    assert cfg.proactivity.sources == ["gmail", "gcalendar"]  # the watched-apps list is unchanged
    assert (cfg.proactivity.followups.waiting_on_you_days, cfg.proactivity.followups.waiting_on_them_days,
            cfg.proactivity.followups.max_age_days, cfg.proactivity.followups.max_suggestions) == (3, 4, 21, 3)
