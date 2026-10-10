from __future__ import annotations

import asyncio
import io
import itertools
import json
import logging
import wave
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from sentient import paths
from sentient.channels import ChannelError
from sentient.llm.provider import StreamChunk
from tests.channels.conftest import (
    TOKEN,
    FakeTelegram,
    GatedProvider,
    callback,
    tg_error,
    until,
    update,
)
from tests.conftest import FakeProvider, tool_call

# ---------------------------------------------------------------------------- connect


async def test_connect_validates_token_and_keeps_it_in_keychain(app, keychain):
    fake = FakeTelegram()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        with pytest.raises(ChannelError, match="doesn't look like"):
            await app.channels.connect("telegram", {"bot_token": "nope"})
        with pytest.raises(ChannelError, match="didn't accept"):
            await app.channels.connect("telegram", {"bot_token": "987654321:" + "B" * 35})
        async with app.bus.subscribe() as q:
            channel = await app.channels.connect("telegram", {"bot_token": TOKEN})
            events = [q.get_nowait() for _ in range(q.qsize())]
    assert channel["status"] == "connected" and channel["account_label"] == "@sentient_test_bot"
    assert channel["setup"]["fields"][0]["key"] == "bot_token" and "@BotFather" in channel["setup"]["instructions_md"]
    assert keychain["channel_telegram_token"] == TOKEN
    assert any(e["type"] == "channel.updated" and e["data"]["id"] == "telegram" for e in events)
    rows = await app.store.fetchall("SELECT * FROM channel_state")
    assert TOKEN not in json.dumps([dict(r) for r in rows])
    assert TOKEN not in app.config.model_dump_json()


async def test_disconnect_removes_token_and_stops(tg, keychain):
    await tg.pair(42)
    tg.ch.start_runtime()
    await until(lambda: bool(tg.api.sent("getUpdates")))
    channel = await tg.app.channels.disconnect("telegram")
    assert channel["status"] == "disconnected" and channel["account_label"] is None
    assert "channel_telegram_token" not in keychain
    assert tg.ch.token is None and not tg.ch.running and tg.ch.http is None
    assert [c["chat_id"] for c in channel["paired"]] == ["42"]  # pairings are kept for a reconnect
    with pytest.raises(ChannelError) as exc:
        await tg.app.channels.create_pairing("telegram")
    assert exc.value.status == 409


async def test_token_is_redacted_from_http_logs(caplog, app):
    caplog.set_level(logging.INFO, logger="httpx")
    logging.getLogger("httpx").info('HTTP Request: POST https://api.telegram.org/bot%s/getUpdates "200"', TOKEN)
    assert TOKEN not in caplog.text and "bot<token>" in caplog.text


# ---------------------------------------------------------------------------- pairing


async def test_pairing_flow(tg):
    app, api = tg.app, tg.api
    code = (await app.channels.create_pairing("telegram"))["code"]
    assert len(code) == 6 and code.isdigit()
    wrong = "000000" if code != "000000" else "111111"
    await tg.say(42, f"/pair {wrong}")
    assert "didn't work" in api.screen()[-1]
    await tg.say(42, "/pair@sentient_test_bot " + code)
    assert "Paired!" in api.screen()[-1]
    chat = await app.channels.store.chat("telegram", "42")
    assert chat["label"] == "Sarthak (@sk)" and chat["deliver"] is True
    session = await app.store.get_session(chat["session_id"])
    assert session["channel"] == "telegram"
    # single use: the same code no longer works for another chat
    await tg.say(43, f"/pair {code}")
    assert "didn't work" in api.screen()[-1]
    assert await app.channels.store.chat("telegram", "43") is None


async def test_pairing_code_expires(tg):
    code = (await tg.app.channels.create_pairing("telegram"))["code"]
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    await tg.app.store.execute("UPDATE channel_pairing_codes SET expires_at = ?", (past,))
    await tg.say(42, f"/pair {code}")
    assert "expired" in tg.api.screen()[-1]


async def test_pairing_attempt_limits(tg):
    app, api = tg.app, tg.api
    app.config.channels.pairing_max_attempts = 3
    code = (await app.channels.create_pairing("telegram"))["code"]
    wrong = "000000" if code != "000000" else "111111"
    for chat_id in (51, 52, 53):  # the code is burned after 3 wrong tries, from any chat
        await tg.say(chat_id, f"/pair {wrong}")
    assert "Too many wrong codes" in api.screen()[-1]
    await tg.say(54, f"/pair {code}")
    assert "didn't work" in api.screen()[-1]
    # one chat is paused after too many wrong codes, even with a fresh code
    code = (await app.channels.create_pairing("telegram"))["code"]
    for _ in range(3):
        await tg.say(60, "/pair 999999" if code != "999999" else "/pair 888888")
    before = len(api.sent("sendMessage"))
    await tg.say(60, f"/pair {code}")
    assert len(api.sent("sendMessage")) == before
    assert await app.channels.store.chat("telegram", "60") is None


async def test_unpaired_chats_get_one_refusal(tg, llm):
    await tg.say(7, "hello")
    await tg.say(7, "hello again")
    await tg.say(7, "/help")
    texts = [p["text"] for p in tg.api.sent("sendMessage")]
    assert len(texts) == 1 and "isn't paired" in texts[0]
    assert llm.calls == []
    # group chats are ignored entirely
    group = update(-100, "hi")
    group["message"]["chat"]["type"] = "group"
    tg.ch.dispatch(group)
    await tg.ch.wait_idle()
    assert len(tg.api.sent("sendMessage")) == 1


async def test_deliver_toggle_remove_and_test_message(tg):
    app = tg.app
    await tg.pair(42)
    channel = await app.channels.set_deliver("telegram", "42", False)
    assert channel["paired"][0]["deliver"] is False
    assert (await app.channels.test("telegram", "42")) == {"ok": True}
    assert "test message" in tg.api.screen()[-1]
    assert (await app.channels.test("telegram")) == {"ok": True}
    assert (await app.channels.test("telegram", "999"))["ok"] is False
    channel = await app.channels.remove_chat("telegram", "42")
    assert channel["paired"] == []
    with pytest.raises(ChannelError) as exc:
        await app.channels.set_deliver("telegram", "42", True)
    assert exc.value.status == 404


# ---------------------------------------------------------------------------- chat


async def test_message_runs_a_turn_and_replies(tg, llm):
    chat = await tg.pair(42)
    llm.replies = ["Hello **there** & <friends>"]
    async with tg.app.bus.subscribe() as q:
        await tg.say(42, "hi sentient")
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert tg.api.screen()[-1] == "Hello <b>there</b> &amp; &lt;friends&gt;"
    assert tg.api.sent("sendChatAction")[0]["action"] == "typing"
    assert all(p.get("parse_mode") == "HTML" for p in tg.api.sent("sendMessage"))
    msgs = [e["data"] for e in events if e["type"] == "channel.message"]
    assert msgs[0] == {"channel": "telegram", "chat_id": "42", "session_id": chat["session_id"], "direction": "in", "text": "hi sentient"}
    assert msgs[-1]["direction"] == "out" and msgs[-1]["text"] == "Hello **there** & <friends>"
    history = await tg.app.store.recent_messages(chat["session_id"], 10)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "Channel: telegram" in llm.calls[0]["messages"][0]["content"]


async def test_long_reply_is_split(tg, llm):
    await tg.pair(42)
    llm.replies = ["\n\n".join(f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 20 for i in range(20))]
    await tg.say(42, "write a lot")
    screen = tg.api.screen()
    reply = screen[1:]  # after the pairing message
    assert len(reply) >= 3
    assert all(len(p["text"]) <= 4096 for p in tg.api.sent("sendMessage") + tg.api.sent("editMessageText"))
    assert reply[0].startswith("Paragraph 0") and reply[-1].rstrip().endswith("amet")
    assert "…" not in reply[-1]


class ClockProvider(FakeProvider):
    def __init__(self, clock: list[float], text: str, step: float):
        super().__init__()
        self.clock, self.text, self.step = clock, text, step

    async def stream(self, role, messages, tools=None, *, model=None):
        for i in range(0, len(self.text), 5):
            self.clock[0] += self.step
            yield StreamChunk(text=self.text[i : i + 5], model="fake")
            await asyncio.sleep(0)
        yield StreamChunk(done=True, model="fake")


async def test_streaming_edits_are_throttled(tg):
    await tg.pair(42)
    clock = [0.0]
    tg.ch.clock = lambda: clock[0]
    tg.app.agent.llm = ClockProvider(clock, "abcde" * 40, step=0.25)  # 40 deltas over 10 simulated seconds
    times: list[tuple[str, float]] = []
    tg.api.on_call = lambda method, params: times.append((method, clock[0])) if method in {"sendMessage", "editMessageText"} else None
    await tg.say(42, "stream please")
    updates = [t for m, t in times]
    assert 6 <= len(updates) <= 12
    gaps = [b - a for a, b in itertools.pairwise(updates[:-1])]  # the final flush may come sooner
    assert all(g >= 1.0 - 1e-9 for g in gaps[1:]), updates
    assert tg.api.screen()[-1] == "abcde" * 40


async def test_stream_edits_off_sends_once(tg, llm):
    await tg.pair(42)
    tg.app.config.channels.telegram.stream_edits = False
    llm.replies = ["one two three four five six"]
    before = len(tg.api.sent("sendMessage"))
    await tg.say(42, "hi")
    assert len(tg.api.sent("sendMessage")) == before + 1 and tg.api.sent("editMessageText") == []


async def test_html_rejected_falls_back_to_plain_text(tg, llm):
    await tg.pair(42)
    tg.app.config.channels.telegram.stream_edits = False
    tg.api.queue["sendMessage"] = [tg_error(400, "Bad Request: can't parse entities: unsupported start tag")]
    llm.replies = ["**Hi** there"]
    await tg.say(42, "hi")
    last = tg.api.sent("sendMessage")[-1]
    assert "parse_mode" not in last and last["text"] == "Hi there"


async def test_tool_activity_is_shown_then_removed(tg, llm):
    await tg.pair(42)
    llm.replies = [[tool_call("current_datetime")], "It is noon."]
    await tg.say(42, "what time is it")
    sent = [p["text"] for p in tg.api.sent("sendMessage")]
    assert "<i>Checking the time...</i>" in sent
    assert tg.api.sent("deleteMessage")
    assert tg.api.screen()[-1] == "It is noon." and "<i>Checking the time...</i>" not in tg.api.screen()


async def test_commands_new_help_unknown(tg):
    app = tg.app
    chat = await tg.pair(42)
    await tg.say(42, "/help")
    assert "/new" in tg.api.screen()[-1] and "/stop" in tg.api.screen()[-1]
    await tg.say(42, "/new")
    assert "fresh chat" in tg.api.screen()[-1]
    fresh = await app.channels.store.chat("telegram", "42")
    assert fresh["session_id"] != chat["session_id"]
    assert (await app.store.get_session(fresh["session_id"]))["channel"] == "telegram"
    await tg.say(42, "/dance")
    assert "don't know that command" in tg.api.screen()[-1]
    await tg.say(42, "/stop")
    assert "Nothing is running" in tg.api.screen()[-1]


@pytest.fixture
def gated() -> GatedProvider:
    return GatedProvider(["first reply", "second reply"], block_on_call=1)


@pytest.fixture
def llm_gated(gated):
    return gated


async def test_stop_cancels_the_running_reply(tg, gated):
    tg.app.agent.llm = gated
    chat = await tg.pair(42)
    tg.ch.dispatch(update(42, "tell me a long story"))
    await asyncio.wait_for(gated.blocked.wait(), 5)
    await tg.say(42, "/stop")
    assert tg.api.screen()[-1] == "Stopped."
    assert not tg.ch.runtime("42").running
    history = await tg.app.store.recent_messages(chat["session_id"], 10)
    assert history[-1]["role"] == "assistant" and "stopped" in history[-1]["content"]


async def test_message_while_running_is_queued_without_steer(tg, gated, monkeypatch):
    tg.app.agent.llm = gated
    monkeypatch.setattr(tg.app.agent, "steer", None, raising=False)  # an engine without steering
    await tg.pair(42)
    tg.ch.dispatch(update(42, "one"))
    await asyncio.wait_for(gated.blocked.wait(), 5)
    tg.ch.dispatch(update(42, "two"))
    await until(lambda: any("as soon as I finish" in t for t in tg.api.screen()))
    gated.gate.set()
    await tg.ch.wait_idle()
    assert gated.stream_calls == 2
    content = gated.calls[-1]["messages"][-1]["content"]
    assert "[Current time:" in content
    assert content.endswith("two")
    assert tg.api.screen()[-1] == "second reply"


async def test_message_while_running_steers_when_available(tg, gated, monkeypatch):
    tg.app.agent.llm = gated
    chat = await tg.pair(42)
    steered: list[tuple[str, str]] = []

    async def steer(session_id: str, text: str) -> bool:
        steered.append((session_id, text))
        return True

    monkeypatch.setattr(tg.app.agent, "steer", steer, raising=False)
    tg.ch.dispatch(update(42, "one"))
    await asyncio.wait_for(gated.blocked.wait(), 5)
    tg.ch.dispatch(update(42, "actually make it short"))
    await until(lambda: bool(steered))
    gated.gate.set()
    await tg.ch.wait_idle()
    assert steered == [(chat["session_id"], "actually make it short")]
    assert gated.stream_calls == 1
    assert not any("as soon as I finish" in t for t in tg.api.screen())


# ---------------------------------------------------------------------------- voice and files


class FakeVoice:
    def __init__(self, text: str = "what's the weather in Pune", error: Exception | None = None):
        self.text, self.error = text, error
        self.received: list[tuple[bytes, str]] = []

    async def transcribe_bytes(self, data: bytes, filename: str) -> str:
        self.received.append((data, filename))
        if self.error:
            raise self.error
        return self.text

    async def speak(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x01" * 1600)
        return buf.getvalue()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


VOICE = {"voice": {"file_id": "v1", "file_unique_id": "u1", "duration": 2, "mime_type": "audio/ogg", "file_size": 1000}}


async def test_voice_note_is_transcribed_and_answered(tg, llm, monkeypatch):
    voice = FakeVoice()
    monkeypatch.setattr(tg.app, "voice", voice)
    tg.app.config.channels.telegram.voice_replies = True
    tg.api.files["v1"] = b"OggS-voice"
    await tg.pair(42)
    llm.replies = ["It's sunny."]
    sent: list[tuple[str, dict]] = []
    call = tg.ch.call

    async def spy(method, **kwargs):
        sent.append((method, kwargs))
        return await call(method, **kwargs)

    monkeypatch.setattr(tg.ch, "call", spy)
    await tg.say(42, None, **VOICE)
    assert voice.received == [(b"OggS-voice", "telegram-voice-u1.ogg")]
    assert any("<i>Heard:</i> what&#x27;s the weather" in t or "<i>Heard:</i> what's the weather" in t for t in tg.api.screen())
    content = llm.calls[-1]["messages"][-1]["content"]
    assert "[Current time:" in content
    assert content.endswith("what's the weather in Pune")
    assert tg.api.screen()[-1] == "It's sunny."
    voices = [kwargs for method, kwargs in sent if method == "sendVoice"]
    assert len(voices) == 1
    name, data, mime = voices[0]["files"]["voice"]
    assert name.endswith(".ogg") and mime == "audio/ogg" and data.startswith(b"OggS")
    assert not any(method == "sendAudio" for method, _ in sent)


async def test_undecodable_audio_falls_back_to_send_audio(tg, monkeypatch):
    sent: list[tuple[str, dict]] = []
    call = tg.ch.call

    async def spy(method, **kwargs):
        sent.append((method, kwargs))
        return await call(method, **kwargs)

    monkeypatch.setattr(tg.ch, "call", spy)
    await tg.ch.send_audio("42", b"not-a-wav", "reply.wav")
    audio = [kwargs for method, kwargs in sent if method == "sendAudio"]
    assert len(audio) == 1
    assert audio[0]["files"]["audio"] == ("reply.wav", b"not-a-wav", "audio/wav")
    assert not any(method == "sendVoice" for method, _ in sent)


async def test_voice_note_without_voice_engine(tg, llm, monkeypatch):
    class NoVoice:
        async def stop(self):
            return None

    monkeypatch.setattr(tg.app, "voice", NoVoice())
    tg.api.files["v1"] = b"OggS"
    await tg.pair(42)
    await tg.say(42, None, **VOICE)
    assert "can't understand voice notes yet" in tg.api.screen()[-1]
    assert llm.calls == []


async def test_voice_error_is_friendly(tg, llm, monkeypatch):
    from sentient.voice.base import VoiceError

    monkeypatch.setattr(tg.app, "voice", FakeVoice(error=VoiceError("The speech model isn't downloaded yet.")))
    tg.api.files["v1"] = b"OggS"
    await tg.pair(42)
    await tg.say(42, None, **VOICE)
    assert "couldn't transcribe" in tg.api.screen()[-1] and "isn&#x27;t downloaded" not in tg.api.screen()[-1]
    assert llm.calls == []


async def test_photo_and_document_become_attachments(tg, llm):
    tg.api.files["p2"] = b"\xff\xd8\xff\xe0fakejpeg"
    tg.api.files["d1"] = b"hello from a text file"
    await tg.pair(42)
    photo = {"photo": [{"file_id": "p1", "file_unique_id": "small", "file_size": 10, "width": 90},
                       {"file_id": "p2", "file_unique_id": "big", "file_size": 100, "width": 800}],
             "caption": "what is this"}
    await tg.say(42, None, **photo)
    saved = paths.files_dir() / "uploads" / "telegram-photo-big.jpg"
    assert saved.read_bytes() == b"\xff\xd8\xff\xe0fakejpeg"
    content = llm.calls[-1]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert "[Current time:" in content[0]["text"]
    assert any(
        block.get("type") == "text" and "what is this" in block.get("text", "")
        for block in content
    )
    assert any(block.get("type") == "image_url" for block in content)
    doc = {"document": {"file_id": "d1", "file_unique_id": "doc", "file_name": "notes.txt", "mime_type": "text/plain", "file_size": 22}}
    await tg.say(42, None, **doc)
    assert "hello from a text file" in llm.calls[-1]["messages"][-1]["content"]
    big = {"document": {"file_id": "d2", "file_unique_id": "big", "file_name": "huge.zip", "file_size": 50 * 1024 * 1024}}
    await tg.say(42, None, **big)
    assert "larger than 20 MB" in tg.api.screen()[-1] or "larger than 20 MB" in "".join(tg.api.screen())


# ---------------------------------------------------------------------------- approvals


async def test_approval_buttons_allow(tg, llm):
    tg.app.config.tools.approvals.mode = "always"
    await tg.pair(42)
    llm.replies = [[tool_call("current_datetime")], "It is noon."]
    tg.ch.dispatch(update(42, "what time is it"))
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    keyboard = msg["reply_markup"]["inline_keyboard"]
    assert [b["text"] for row in keyboard for b in row] == ["Allow", "Allow for this chat", "Deny"]
    assert "Approval needed" in msg["text"] and "current datetime" in msg["text"]
    tg.ch.dispatch(callback(42, msg["_id"], keyboard[0][0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Allowed"
    edit = [p for p in tg.api.sent("editMessageText") if p["message_id"] == msg["_id"]][-1]
    assert "<b>Allowed</b>" in edit["text"] and "reply_markup" not in edit
    assert tg.api.screen()[-1] == "It is noon."


async def test_approval_buttons_deny_and_stale(tg, llm):
    tg.app.config.tools.approvals.mode = "always"
    await tg.pair(42)
    llm.replies = [[tool_call("current_datetime")], "Okay, I won't."]
    tg.ch.dispatch(update(42, "what time is it"))
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    deny = msg["reply_markup"]["inline_keyboard"][1][0]["callback_data"]
    tg.ch.dispatch(callback(42, msg["_id"], deny))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Denied"
    history = await tg.app.store.recent_messages((await tg.app.channels.store.chat("telegram", "42"))["session_id"], 10)
    assert any(m["role"] == "tool" and "declined" in (m["content"] or "") for m in history)
    # pressing again: nothing is waiting any more
    tg.ch.dispatch(callback(42, msg["_id"], deny))
    await tg.ch.wait_idle()
    assert "no longer waiting" in tg.api.sent("answerCallbackQuery")[-1]["text"]
    # an unpaired chat cannot press buttons
    tg.ch.dispatch(callback(77, msg["_id"], deny))
    await tg.ch.wait_idle()
    assert "isn't paired" in tg.api.sent("answerCallbackQuery")[-1]["text"]


# ---------------------------------------------------------------------------- polling


async def test_polling_backoff_recovery_and_revoked_token(tg):
    app, ch, api = tg.app, tg.ch, tg.api
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    ch.sleep = fake_sleep
    api.queue["getUpdates"] = [
        httpx.ConnectError("offline"),
        tg_error(502, "Bad Gateway"),
        tg_error(500, "Internal Server Error"),
        tg_error(429, "Too Many Requests: retry after 7", retry_after=7),
        [update(7, "hello from before")],
        tg_error(401, "Unauthorized"),
    ]
    statuses: list[str] = []
    async with app.bus.subscribe() as q:
        await asyncio.wait_for(ch.run(), 5)
        while not q.empty():
            e = q.get_nowait()
            if e["type"] == "channel.updated":
                statuses.append(e["data"]["status"])
    await ch.wait_idle()
    assert waits == [1.0, 2.0, 4.0, 7.0]
    transitions = [s for i, s in enumerate(statuses) if i == 0 or statuses[i - 1] != s]  # error text may change
    assert transitions == ["error", "connected", "error"]
    state = await app.channels.store.state("telegram")
    assert state["status"] == "error" and "rejected the bot token" in state["error"]
    params = api.sent("getUpdates")
    assert params[0]["timeout"] == 30 and params[0]["allowed_updates"] == ["message", "callback_query"]
    assert params[-1]["offset"] == int(state["cursor"])
    assert "isn't paired" in api.screen()[-1]  # the update was dispatched


async def test_runtime_starts_and_stops_gracefully(tg):
    tg.ch.start_runtime()
    await until(lambda: len(tg.api.sent("getUpdates")) >= 2)
    assert tg.ch.running
    await tg.ch.stop_runtime()
    assert not tg.ch.running and tg.ch.http is None


async def test_start_reconnects_saved_channel(app, keychain, monkeypatch):
    keychain["channel_telegram_token"] = TOKEN
    await app.channels.store.set_state("telegram", enabled=1, status="connected", account_label="@old")
    app.enable_background = True
    started: list[str] = []
    monkeypatch.setattr(app.channels.channels["telegram"], "start_runtime", lambda: started.append("telegram"))
    await app.channels.stop()
    await app.channels.start()
    assert started == ["telegram"]
    assert (await app.channels.store.state("telegram"))["status"] == "connecting"
    keychain.clear()
    await app.channels.stop()
    await app.channels.start()
    state = await app.channels.store.state("telegram")
    assert state["status"] == "error" and "missing" in state["error"]
