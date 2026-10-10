"""WhatsApp channel (issue #106) driven through a fake bridge: no WhatsApp, no network, no real model."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import respx

from sentient import paths
from sentient.channels import ChannelError
from sentient.channels.formatting import markdown_to_whatsapp
from sentient.channels.whatsapp import WAMedia, WAMessage
from tests.channels.conftest import GatedProvider, until
from tests.channels.test_task_questions import (
    ASK_FLIGHT,
    ASK_HOTEL,
    FLIGHTS,
    RESULT,
    _executor_answer,
    _waiting_task,
)
from tests.channels.test_telegram import FakeVoice
from tests.conftest import tool_call

ME = "15550001111@s.whatsapp.net"
MY_LID = "99887766@lid"
FRIEND = "15559990000@s.whatsapp.net"
SIGNED = "*Sentient:* "  # how Sentient's messages start in the self chat
_ids = itertools.count(1)


class FakeBridge:
    """One fake WhatsApp Web connection. The test feeds it events; what Sentient sends is kept on the hub."""

    def __init__(self, hub: Hub, session_dir: Path):
        self.hub, self.session_dir = hub, session_dir
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.connected = False
        self.closed = False

    async def run(self, emit) -> None:
        while True:
            item = await self.inbox.get()
            if item is None:  # the connection dropped
                self.connected = False
                return
            kind, data = item
            if kind == "connected":
                self.session_dir.mkdir(parents=True, exist_ok=True)
                (self.session_dir / "session.sqlite3").write_bytes(b"keys")
                self.connected = True
            await emit(kind, data)
            if kind in {"logged_out", "failed"}:
                self.connected = False
                return

    def _new(self, op: str, chat: str, **extra: Any) -> str:
        mid = f"S{next(_ids)}"
        self.hub.sent.append({"op": op, "chat": chat, "id": mid, **extra})
        return mid

    async def send_text(self, chat: str, text: str) -> str:
        return self._new("text", chat, text=text)

    async def edit_text(self, chat: str, message_id: str, text: str) -> None:
        self.hub.sent.append({"op": "edit", "chat": chat, "id": message_id, "text": text})

    async def revoke(self, chat: str, message_id: str) -> None:
        self.hub.sent.append({"op": "revoke", "chat": chat, "id": message_id})

    async def send_voice(self, chat: str, ogg: bytes, seconds: int) -> str:
        return self._new("voice", chat, data=ogg, seconds=seconds)

    async def send_document(self, chat: str, data: bytes, filename: str, mime: str) -> str:
        return self._new("document", chat, data=data, filename=filename, mime=mime)

    async def typing(self, chat: str) -> None:
        self.hub.typing += 1

    async def download(self, media: WAMedia) -> bytes:
        return self.hub.files.get(media.ref, b"")

    async def logout(self) -> None:
        self.hub.logged_out = True

    async def close(self) -> None:
        self.closed = True


class Hub:
    def __init__(self) -> None:
        self.bridges: list[FakeBridge] = []
        self.sent: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.typing = 0
        self.logged_out = False
        self.signed = SIGNED

    def factory(self, session_dir: Path) -> FakeBridge:
        bridge = FakeBridge(self, session_dir)
        self.bridges.append(bridge)
        return bridge

    @property
    def bridge(self) -> FakeBridge:
        return self.bridges[-1]

    def screen(self, chat: str = ME) -> list[str]:
        """Current text of every message Sentient sent to ``chat``, oldest first.

        In the self chat every message and every edit must start with "*Sentient:* " (checked here, then left out).
        """
        msgs: dict[str, str] = {}
        for s in self.sent:
            if s["chat"] != chat:
                continue
            if s["op"] == "text" or (s["op"] == "edit" and s["id"] in msgs):
                text = s["text"]
                if chat == ME:
                    assert text.startswith(self.signed), text
                    text = text.removeprefix(self.signed)
                else:
                    assert not text.startswith(self.signed), text
                msgs[s["id"]] = text
            elif s["op"] == "revoke":
                msgs.pop(s["id"], None)
        return list(msgs.values())

    def last(self, chat: str = ME) -> str:
        return self.screen(chat)[-1]

    def message_with(self, needle: str, chat: str = ME) -> dict | None:
        return next((s for s in self.sent if s["op"] == "text" and s["chat"] == chat and needle in s["text"]), None)


@dataclass
class Env:
    app: Any
    ch: Any
    hub: Hub

    async def link(self) -> None:
        await self.app.channels.connect("whatsapp", {})
        self.ch.start_runtime()
        await until(lambda: bool(self.hub.bridges))
        await self.hub.bridge.inbox.put(("qr", {"code": "2@first-code"}))
        await self.hub.bridge.inbox.put(("connected", {"jid": "15550001111:7@s.whatsapp.net", "lid": MY_LID, "name": "Maya"}))
        await until(lambda: self.ch.linked and bool(self.hub.screen()))
        await self.ch.wait_idle()

    async def say(self, text: str = "", *, chat: str = ME, from_me: bool | None = None, reply_to: str | None = None,
                  media: list[WAMedia] | None = None, push_name: str = "Maya", wait: bool = True) -> None:
        message = WAMessage(
            id=f"U{next(_ids)}", chat=chat, from_me=chat in (ME, MY_LID) if from_me is None else from_me,
            push_name=push_name, text=text, reply_to=reply_to, media=media or [],
        )
        await self.ch.on_event("message", {"message": message})
        if wait:
            await self.ch.wait_idle()


async def _no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


@pytest.fixture
async def wa(app):
    hub = Hub()
    ch = app.channels.channels["whatsapp"]
    ch.bridge_factory = hub.factory
    ch.available = lambda: True
    ch.sleep = _no_sleep
    env = Env(app, ch, hub)
    try:
        yield env
    finally:
        await ch.stop_runtime()


# ---------------------------------------------------------------------------- linking


async def test_linking_shows_the_qr_code_then_pairs_the_self_chat(wa):
    app, ch, hub = wa.app, wa.ch, wa.hub
    channel = await app.channels.connect("whatsapp", {})
    assert channel["status"] == "linking" and channel["qr"] is None and channel["setup"]["fields"] == []
    assert "Linked devices" in channel["setup"]["instructions_md"] and "ban" in channel["setup"]["instructions_md"]
    updates: list[dict] = []
    async with app.bus.subscribe() as q:
        ch.start_runtime()
        await until(lambda: bool(hub.bridges))
        await hub.bridge.inbox.put(("qr", {"code": "2@first-code"}))
        await until(lambda: ch.qr == "2@first-code")
        await hub.bridge.inbox.put(("qr", {"code": "2@second-code"}))  # WhatsApp rotates the code
        await until(lambda: ch.qr == "2@second-code")
        await hub.bridge.inbox.put(("connected", {"jid": "15550001111:7@s.whatsapp.net", "lid": MY_LID, "name": "Maya"}))
        await until(lambda: ch.linked and bool(hub.screen()))
        await ch.wait_idle()
        while not q.empty():
            e = q.get_nowait()
            if e["type"] == "channel.updated":
                updates.append(e["data"])
    assert [u["qr"] for u in updates if u["status"] == "linking"][-2:] == ["2@first-code", "2@second-code"]
    channel = await app.channels.channel_dict("whatsapp")
    assert channel["status"] == "connected" and channel["account_label"] == "+15550001111" and channel["qr"] is None
    assert [(p["chat_id"], p["label"], p["deliver"]) for p in channel["paired"]] == [(ME, "Message yourself", True)]
    assert hub.last().startswith("Linked! This chat is where you talk to")
    assert (ch.session_dir() / "session.sqlite3").exists()
    assert ch.session_dir().parent.name == "home"  # under SENTIENT_HOME, never in config


async def test_connect_without_the_whatsapp_extra_is_a_plain_error(wa):
    wa.ch.available = lambda: False
    with pytest.raises(ChannelError, match="isn't installed"):
        await wa.app.channels.connect("whatsapp", {})


async def test_an_unscanned_code_is_not_retried_forever(wa):
    await wa.app.channels.connect("whatsapp", {})
    wa.ch.start_runtime()
    await until(lambda: bool(wa.hub.bridges))
    await wa.hub.bridge.inbox.put(("qr", {"code": "2@code"}))
    await wa.hub.bridge.inbox.put(None)  # WhatsApp stops offering codes
    await until(lambda: not wa.ch.running)
    state = await wa.app.channels.store.state("whatsapp")
    assert state["status"] == "error" and "expired before it was scanned" in state["error"]
    assert len(wa.hub.bridges) == 1 and wa.ch.qr is None


# ---------------------------------------------------------------------------- chatting


async def test_self_chat_message_gets_a_reply(wa, llm):
    await wa.link()
    llm.replies = ["It's **sunny** in Pune."]
    await wa.say("weather in pune?")
    assert wa.hub.last() == "It's *sunny* in Pune."
    chat = await wa.app.channels.store.chat("whatsapp", ME)
    history = await wa.app.store.recent_messages(chat["session_id"], 10)
    assert [m["content"] for m in history if m["role"] == "user"] == ["weather in pune?"]
    session = await wa.app.store.get_session(chat["session_id"])
    assert session["channel"] == "whatsapp"
    assert not any(s["op"] == "revoke" for s in wa.hub.sent)  # no temporary status lines on WhatsApp

    llm.replies = ["Sure."]
    await wa.say("same chat, addressed by LID", chat=MY_LID)  # newer WhatsApp addresses the self chat by LID
    assert wa.hub.last() == "Sure."


async def test_other_chats_are_ignored_and_never_answered(wa, llm):
    await wa.link()
    before = len(wa.hub.sent)
    await wa.say("hey, are you free tonight?", chat=FRIEND, push_name="Ravi")  # a friend writing to the user
    await wa.say("yes, see you at 8", chat=FRIEND, from_me=True)  # the user answering the friend
    await wa.say("/stopall", chat=FRIEND, push_name="Ravi")
    await wa.say("hello group", chat="120363000000@g.us", push_name="Ravi")
    await wa.say("status", chat="status@broadcast", push_name="Ravi")
    assert len(wa.hub.sent) == before and not wa.app.stopped
    assert not llm.calls
    # Sentient's own messages coming back are never read as the user's
    own = WAMessage(id=wa.hub.sent[-1]["id"], chat=ME, from_me=True, text="Linked!")
    await wa.ch.on_event("message", {"message": own})
    await wa.ch.wait_idle()
    assert len(wa.hub.sent) == before


async def test_another_chat_can_be_paired_with_a_code(wa, llm):
    await wa.link()
    await wa.say("/pair 000000", chat=FRIEND, push_name="Ravi")  # no code is active: wrong code
    assert "didn't work" in wa.hub.last(FRIEND)
    code = (await wa.app.channels.create_pairing("whatsapp"))
    assert code["instructions"].startswith("From the other WhatsApp chat, send this to +15550001111: /pair ")
    await wa.say(f"/pair {code['code']}", chat=FRIEND, push_name="Ravi")
    assert wa.hub.screen(FRIEND)[-1].startswith("Paired!")
    chat = await wa.app.channels.store.chat("whatsapp", FRIEND)
    assert chat["label"] == "Ravi"
    llm.replies = ["Hello Ravi."]
    await wa.say("hi", chat=FRIEND, push_name="Ravi")
    assert wa.hub.last(FRIEND) == "Hello Ravi."
    await wa.say("not for Sentient", chat=FRIEND, from_me=True)  # the user's own messages there stay ignored
    assert wa.hub.last(FRIEND) == "Hello Ravi."


async def test_a_task_set_to_the_self_chat_reaches_only_it(wa):
    """``{"channel": "whatsapp", "chat_id": "self"}`` (a Hermes job that delivered to WhatsApp) is the Message
    yourself chat of whatever number is linked; another paired chat with delivery on gets nothing."""
    await wa.link()
    code = await wa.app.channels.create_pairing("whatsapp")
    await wa.say(f"/pair {code['code']}", chat=FRIEND, push_name="Ravi")
    await wa.app.channels.set_deliver("whatsapp", FRIEND, True)
    now = wa.app.tasks.now_iso()
    task_id = await wa.app.tasks.repo.insert_task({
        "name": "Morning brief", "description": "Morning brief", "status": "active", "created_at": now, "updated_at": now,
    })
    await wa.app.tasks.update(task_id, {"deliver_to": [{"channel": "whatsapp", "chat_id": "self"}]})
    friend_before = len(wa.hub.screen(FRIEND))
    await wa.app.notify("task", "Task 'Morning brief' has finished with status: completed.", title="Task completed",
                        payload={"task_id": task_id, "event": "run_completed"})
    await until(lambda: "Task completed" in wa.hub.last())
    await asyncio.sleep(0.2)
    assert "Morning brief" in wa.hub.last()
    assert len(wa.hub.screen(FRIEND)) == friend_before


async def test_photo_and_document_become_attachments(wa, llm):
    await wa.link()
    wa.hub.files.update({"img": b"\xff\xd8\xff\xe0fakejpeg", "doc": b"hello from a text file"})
    llm.replies = ["A photo.", "Got it."]
    await wa.say("what is this", media=[WAMedia(kind="image", name="whatsapp-image-1.jpg", mime="image/jpeg", size=12, ref="img")])
    assert (paths.files_dir() / "uploads" / "whatsapp-image-1.jpg").read_bytes() == b"\xff\xd8\xff\xe0fakejpeg"
    content = llm.calls[-1]["messages"][-1]["content"]
    assert isinstance(content, list) and "what is this" in content[0]["text"]
    await wa.say(media=[WAMedia(kind="document", name="notes.txt", mime="text/plain", size=22, ref="doc")])
    assert "hello from a text file" in llm.calls[-1]["messages"][-1]["content"]
    await wa.say(media=[WAMedia(kind="document", name="huge.zip", size=50 * 1024 * 1024, ref="big")])
    assert "larger than 20 MB" in wa.hub.last()


# ---------------------------------------------------------------------------- approvals and questions


async def test_approval_by_numbered_reply(wa, llm):
    wa.app.config.tools.approvals.mode = "always"
    await wa.link()
    llm.replies = [[tool_call("current_datetime")], "It is noon."]
    await wa.say("what time is it", wait=False)
    await until(lambda: wa.hub.message_with("Approval needed") is not None)
    msg = wa.hub.message_with("Approval needed")
    assert msg["text"].endswith("Reply with a number:\n*1* Allow\n*2* Allow for this chat\n*3* Deny")
    await wa.say("7", reply_to=msg["id"], wait=False)
    await until(lambda: wa.hub.last() == "Reply with a number from 1 to 3.")
    await wa.say("1", reply_to=msg["id"])
    await wa.ch.wait_idle()
    edits = [s for s in wa.hub.sent if s["op"] == "edit" and s["id"] == msg["id"]]
    assert edits and edits[-1]["text"].endswith("*Allowed*") and "Reply with a number" not in edits[-1]["text"]
    assert "_Allowed._" in wa.hub.screen()
    assert wa.hub.last() == "It is noon."


async def test_a_bare_number_answers_the_waiting_approval(wa, llm):
    wa.app.config.tools.approvals.mode = "always"
    await wa.link()
    llm.replies = [[tool_call("current_datetime")], "Okay, I won't."]
    await wa.say("what time is it", wait=False)
    await until(lambda: wa.hub.message_with("Approval needed") is not None)
    await wa.say("3")
    await wa.ch.wait_idle()
    assert "_Denied._" in wa.hub.screen() and wa.hub.last() == "Okay, I won't."
    llm.replies = ["Three it is."]
    await wa.say("3")  # nothing is waiting any more: a bare number is just a message
    assert wa.hub.last() == "Three it is."


async def test_task_question_answered_by_number_or_by_reply(wa, llm):
    await wa.link()
    llm.replies = [ASK_FLIGHT, "Done with the task."]
    llm.json_replies = [dict(RESULT)]
    task_id, _run_id = await _waiting_task(wa.app, "Book a flight to Goa")
    await until(lambda: wa.hub.message_with("Which flight should I book?") is not None)
    msg = wa.hub.message_with("Which flight should I book?")
    assert "Reply with an option's number, or reply to this message with your answer." in msg["text"]
    assert msg["text"].endswith(f"Reply to this message with a number:\n*1* {FLIGHTS[0]}\n*2* {FLIGHTS[1]}")
    llm.replies.insert(0, "Two of what?")
    await wa.say("2")  # without replying to it, a number is normal chat
    assert wa.hub.last() == "Two of what?"
    assert not wa.hub.message_with("_Answered")
    await wa.say("2", reply_to=msg["id"])
    await wa.app.tasks.drain()
    assert f"_Answered: {FLIGHTS[1]}._" in wa.hub.screen()
    assert (await wa.app.tasks.get(task_id))["status"] == "completed"

    llm.replies = [ASK_HOTEL, "Done."]
    llm.json_replies = [dict(RESULT)]
    task_id, _run_id = await _waiting_task(wa.app, "Find a hotel")
    await until(lambda: wa.hub.message_with("Which hotel area") is not None)
    msg = wa.hub.message_with("Which hotel area")
    await wa.say("Near the beach, please", reply_to=msg["id"])
    await wa.app.tasks.drain()
    assert wa.hub.last().startswith("Thanks! I passed your answer to 'Find a hotel'.")
    assert (await wa.app.tasks.get(task_id))["status"] == "completed"


async def test_a_numbered_reply_still_answers_after_a_restart(wa, llm):
    await wa.link()
    llm.replies = [ASK_FLIGHT, "Done with the task."]
    llm.json_replies = [dict(RESULT)]
    task_id, _run_id = await _waiting_task(wa.app, "Book a flight to Goa")
    await until(lambda: wa.hub.message_with("Which flight should I book?") is not None)
    msg = wa.hub.message_with("Which flight should I book?")
    wa.ch._choices.clear()  # the numbered options lived in memory; the question itself is in SQLite
    await wa.say("1", reply_to=msg["id"])
    await wa.app.tasks.drain()
    assert wa.hub.last().startswith("Thanks! I passed your answer to 'Book a flight to Goa'.")
    task = await wa.app.tasks.get(task_id)
    assert task["status"] == "completed" and _executor_answer(llm, 1) == FLIGHTS[0]  # "1" became the option itself


async def test_model_presets_as_numbered_options(wa, keychain):
    keychain["anthropic"] = "sk-ant"
    await wa.link()
    await wa.say("/model")
    msg = wa.hub.message_with("Pick a setup")
    assert msg["text"].endswith("Reply to this message with a number:\n*1* Local only\n*2* Cloud\n*3* Mixed")
    with respx.mock() as mock:  # the switch checks which local models are downloaded: never a real Ollama
        mock.get("http://localhost:11434/api/tags").respond(200, json={"models": [{"name": "nomic-embed-text:latest"}]})
        await wa.say("2", reply_to=msg["id"])
    assert wa.app.config.models.active_preset == "Cloud"
    assert "_Switched to Cloud._" in wa.hub.screen()


# ---------------------------------------------------------------------------- voice, commands


async def test_voice_note_in_and_voice_note_out(wa, llm, monkeypatch):
    voice = FakeVoice()
    monkeypatch.setattr(wa.app, "voice", voice)
    wa.app.config.channels.whatsapp.voice_replies = True
    await wa.link()
    wa.hub.files["v1"] = b"OggS-voice"
    llm.replies = ["It's sunny."]
    await wa.say(media=[WAMedia(kind="voice", name="whatsapp-voice-ABC.ogg", mime="audio/ogg; codecs=opus", size=10, ref="v1")])
    assert voice.received == [(b"OggS-voice", "whatsapp-voice-ABC.ogg")]
    assert "_Heard:_ what's the weather in Pune" in wa.hub.screen()
    assert llm.calls[-1]["messages"][-1]["content"] == "what's the weather in Pune"
    assert wa.hub.last() == "It's sunny."
    notes = [s for s in wa.hub.sent if s["op"] == "voice"]
    assert len(notes) == 1 and notes[0]["data"].startswith(b"OggS") and notes[0]["seconds"] >= 1


async def test_stopall_and_resume_from_whatsapp(wa):
    gated = GatedProvider(["first reply"], block_on_call=1)
    wa.app.agent.llm = gated
    await wa.link()
    await wa.say("tell me a long story", wait=False)
    await asyncio.wait_for(gated.blocked.wait(), 5)
    await wa.say("/stopall")
    assert wa.hub.last().startswith("Stopped everything (1 running job cancelled).")
    assert wa.app.stopped and wa.app.stop_state["source"] == "whatsapp"
    await wa.say("/resume")
    assert wa.hub.last().startswith("Resumed.") and not wa.app.stopped
    await wa.say("/help")
    assert "/stopall" in wa.hub.last() and "reply with an option's number" in wa.hub.last()


# ---------------------------------------------------------------------------- connection


async def channel_updates(q, enough) -> list[dict]:
    """``channel.updated`` payloads from ``q``, read until ``enough(payloads)`` is true."""
    updates: list[dict] = []
    async with asyncio.timeout(30):
        while not enough(updates):
            event = await q.get()
            if event["type"] == "channel.updated":
                updates.append(event["data"])
    return updates


async def test_reconnects_after_a_dropped_connection(wa):
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        await asyncio.sleep(0)

    wa.ch.sleep = fake_sleep
    await wa.link()
    async with wa.app.bus.subscribe() as q:
        await wa.hub.bridge.inbox.put(None)  # dropped
        await until(lambda: len(wa.hub.bridges) == 2)
        await wa.hub.bridge.inbox.put(("connected", {"jid": ME, "lid": MY_LID, "name": "Maya"}))
        await until(lambda: wa.hub.bridge.connected)
        updates = await channel_updates(  # until it shows as connected again after reconnecting
            q, lambda ups: "connecting" in [u["status"] for u in ups] and ups[-1]["status"] == "connected"
        )
    statuses = [u["status"] for u in updates]
    assert waits == [1.0]
    assert "connecting" in statuses and statuses[-1] == "connected"
    assert wa.hub.bridges[0].closed
    assert len(await wa.app.channels.store.chats("whatsapp")) == 1  # no second self chat
    assert len([s for s in wa.hub.sent if "Linked!" in s.get("text", "")]) == 1


async def test_logged_out_on_the_phone_needs_a_new_scan(wa):
    await wa.link()
    assert wa.ch.session_dir().exists()
    await wa.hub.bridge.inbox.put(("logged_out", {"reason": "401"}))
    await until(lambda: not wa.ch.running)
    channel = await wa.app.channels.channel_dict("whatsapp")
    assert channel["status"] == "error" and "scan the new code" in channel["error"]
    assert channel["account_label"] is None and not wa.ch.session_dir().exists()
    assert not await wa.app.channels.is_connected("whatsapp")


async def test_disconnect_unlinks_and_forgets_the_session(wa):
    await wa.link()
    channel = await wa.app.channels.disconnect("whatsapp")
    assert wa.hub.logged_out and not wa.ch.session_dir().exists()
    assert channel["status"] == "disconnected" and channel["paired"]  # the chat and its history stay


async def test_start_restores_a_linked_session(app, monkeypatch):
    ch = app.channels.channels["whatsapp"]
    ch.session_dir().mkdir(parents=True)
    await app.channels.store.set_state("whatsapp", enabled=1, status="connected", account_label="+15550001111")
    app.enable_background = True
    started: list[str] = []
    monkeypatch.setattr(ch, "start_runtime", lambda: started.append("whatsapp"))
    await app.channels.stop()
    await app.channels.start()
    assert started == ["whatsapp"] and ch.linked
    ch.forget_session()  # the session folder is gone (wiped by hand)
    await app.channels.stop()
    await app.channels.start()
    state = await app.channels.store.state("whatsapp")
    assert state["status"] == "error" and "linked again" in state["error"]


def test_markdown_to_whatsapp():
    md = "# Plan\n**bold**, *italic*, ~~old~~ and `a **b**`. See [the docs](https://example.com).\n- one\n```py\nx = **1**\n```"
    assert markdown_to_whatsapp(md) == (
        "*Plan*\n*bold*, _italic_, ~old~ and `a **b**`. See the docs (https://example.com).\n- one\n```\nx = **1**\n```"
    )


async def test_replies_in_the_self_chat_carry_the_assistant_name(wa, llm):
    wa.app.config.assistant.name = "Juno"
    wa.hub.signed = "*Juno:* "
    await wa.link()
    llm.replies = ["A reply long enough to be streamed in more than one piece."]
    wa.ch.cfg.edit_interval_s = 0.3
    await wa.say("hi")
    mine = [s for s in wa.hub.sent if s["chat"] == ME and s["op"] in {"text", "edit"}]
    assert mine and all(s["text"].startswith("*Juno:* ") for s in mine)  # edits keep it too
    assert mine[-1]["text"] == "*Juno:* A reply long enough to be streamed in more than one piece."


async def test_a_crash_inside_the_whatsapp_library_never_stops_the_engine(wa, llm, caplog):
    class Exploding(FakeBridge):
        async def run(self, emit) -> None:
            raise RuntimeError("panic: runtime error in whatsmeow")

    real = wa.hub.factory
    wa.ch.bridge_factory = lambda d: Exploding(wa.hub, d)
    await wa.app.channels.store.set_state("whatsapp", enabled=1, account_label="+15550001111")
    wa.ch.session_dir().mkdir(parents=True)
    assert await wa.ch.restore()
    retries: list[float] = []

    async def slow_sleep(seconds: float) -> None:
        retries.append(seconds)
        await asyncio.sleep(0.01)

    wa.ch.sleep = slow_sleep
    async with wa.app.bus.subscribe() as q:
        wa.ch.start_runtime()
        statuses = await channel_updates(q, bool)
        await until(lambda: len(retries) >= 2)
    assert statuses[0]["status"] == "error" and "stopped working unexpectedly" in statuses[0]["error"]
    assert "whatsapp bridge failed" in caplog.text
    assert wa.ch.running and retries[:2] == [1.0, 2.0]  # still trying, with backoff; the engine carries on
    assert await wa.app.store.create_session(channel="desktop")  # the engine itself is untouched

    wa.ch.bridge_factory = real  # the library behaves again: it reconnects
    await until(lambda: bool(wa.hub.bridges))
    await wa.hub.bridge.inbox.put(("connected", {"jid": ME, "lid": MY_LID, "name": "Maya"}))
    await until(lambda: wa.hub.bridge.connected)
    for _ in range(300):
        if (await wa.app.channels.store.state("whatsapp"))["status"] == "connected":
            break
        await asyncio.sleep(0.01)
    assert (await wa.app.channels.channel_dict("whatsapp"))["status"] == "connected"
    # an event handler that blows up is logged, not raised
    await wa.ch.on_event("message", {"message": None})
    assert wa.ch.running


async def test_only_ascii_digits_pick_an_option(wa, llm):
    await wa.link()
    llm.replies = [ASK_FLIGHT, "Done with the task."]
    llm.json_replies = [dict(RESULT)]
    await _waiting_task(wa.app, "Book a flight to Goa")
    await until(lambda: wa.hub.message_with("Which flight should I book?") is not None)
    msg = wa.hub.message_with("Which flight should I book?")
    wa.ch._choices.clear()  # go through the stored question, as after a restart
    await wa.say("²", reply_to=msg["id"])  # a Unicode digit is an answer in words, not option 2 (and no crash)
    await wa.app.tasks.drain()
    assert wa.hub.last().startswith("Thanks! I passed your answer to 'Book a flight to Goa'.")
    assert _executor_answer(llm, 1) == "²"


# ---------------------------------------------------------------------------- rules from chat (#130)
async def test_rule_proposal_by_numbered_reply(wa, llm):
    await wa.link()
    llm.replies, llm.json_replies = ["Understood."], [{"keys": ["gmail_trash"], "rule": "never"}]
    await wa.say("never delete my emails")
    await until(lambda: wa.hub.message_with("Make this a rule?") is not None)
    msg = wa.hub.message_with("Make this a rule?")
    assert "Never: Gmail > Trash" in msg["text"]
    assert msg["text"].endswith("Reply to this message with a number:\n*1* Make it a rule\n*2* Not now")
    await wa.say("1", reply_to=msg["id"])
    assert wa.app.config.tools.approvals.rules == {"gmail_trash": "never"}
    assert any(line.startswith("_Made it a rule") for line in wa.hub.screen())
    edits = [s for s in wa.hub.sent if s["op"] == "edit" and s["id"] == msg["id"]]
    assert edits and "Reply to this message" not in edits[-1]["text"]


async def test_rule_proposal_declined_by_numbered_reply(wa, llm):
    await wa.link()
    llm.replies, llm.json_replies = ["Understood."], [{"keys": ["gmail_trash"], "rule": "never"}]
    await wa.say("never delete my emails")
    await until(lambda: wa.hub.message_with("Make this a rule?") is not None)
    await wa.say("2", reply_to=wa.hub.message_with("Make this a rule?")["id"])
    assert wa.app.config.tools.approvals.rules == {}
    assert "_Not now._" in wa.hub.screen()
