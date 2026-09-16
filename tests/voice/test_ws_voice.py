import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import tool_call
from tests.voice.conftest import SlowProvider, collect_until, frames, recv, silence, tone, types


def _start(ws, **extra):
    ws.send_json({"type": "start", "sample_rate": 16000, **extra})
    ready = recv(ws)
    assert ready["type"] == "ready"
    state = recv(ws)
    assert state == {"type": "state", "state": "listening", "session_id": ready["session_id"]}
    return ready


def _speak(ws, ms: int = 700, tail_ms: int = 1000):
    for f in frames(silence(200) + tone(ms) + silence(tail_ms)):
        ws.send_bytes(f)


def test_rejects_bad_token(voice_client):
    client = voice_client()
    client.headers.pop("Authorization")  # the fixture's bearer header would otherwise authenticate
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/voice?token=nope") as ws:
        ws.receive_json()


def test_full_voice_turn(voice_client):
    client = voice_client(stt_texts=["what can you do"])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ready = _start(ws)
        assert ready["stt"] == "faster_whisper" and ready["tts"] == "system" and ready["sample_rate"] == 16000
        _speak(ws)
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
        items += collect_until(ws, lambda m: m["type"] == "state")

    msgs = [m for m in items if isinstance(m, dict)]
    states = [m["state"] for m in msgs if m["type"] == "state"]
    assert states == ["transcribing", "thinking", "speaking", "listening"]
    transcript = next(m for m in msgs if m["type"] == "transcript")
    assert transcript["text"] == "what can you do" and transcript["final"] is True
    assert "".join(m["text"] for m in msgs if m["type"] == "text_delta") == "Hello there. How can I help?"
    done = next(m for m in msgs if m["type"] == "done")
    assert done["content"] == "Hello there. How can I help?" and done["session_id"] == ready["session_id"]

    # every audio header is immediately followed by one binary WAV frame
    audio = []
    for i, item in enumerate(items):
        if isinstance(item, dict) and item["type"] == "audio":
            assert item["format"] == "wav"
            assert isinstance(items[i + 1], bytes) and items[i + 1][:4] == b"RIFF"
            audio.append(item)
    assert [a["sentence_index"] for a in audio] == [0, 1]
    assert [a["text"] for a in audio] == ["Hello there.", "How can I help?"]
    assert client.tts.texts == ["Hello there.", "How can I help?"]
    end = next(m for m in msgs if m["type"] == "audio_end")
    assert end["sentences"] == 2 and "first_audio_ms" in end["metrics"]

    # the STT saw the utterance (tone + pre-roll + short tail), not the whole stream
    (n_bytes, rate), = client.stt.calls
    assert rate == 16000 and 700 <= n_bytes / 32 <= 1300

    # the turn was persisted on a voice-channel session
    import asyncio

    session = asyncio.run(_get_session(client, ready["session_id"]))
    assert session["channel"] == "voice"


async def _get_session(client, sid):
    return await client.core.store.get_session(sid)


def test_voice_state_published_on_bus(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws?token={client.token}") as main_ws:
        assert main_ws.receive_json()["type"] == "hello"
        with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
            _start(ws)
        seen = collect_until(main_ws, lambda m: m["type"] == "voice.state" and m["data"]["state"] == "idle")
    states = [m["data"]["state"] for m in seen if m["type"] == "voice.state"]
    assert states[0] == "listening" and states[-1] == "idle"


def test_push_to_talk_end_utterance(voice_client):
    client = voice_client(stt_texts=["push to talk works"])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        for f in frames(tone(500)):  # no trailing silence: server VAD alone would keep waiting
            ws.send_bytes(f)
        ws.send_json({"type": "end_utterance"})
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    assert any(isinstance(m, dict) and m.get("text") == "push to talk works" for m in items)


def test_empty_transcript_returns_to_listening(voice_client):
    client = voice_client(stt_texts=[""])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        _speak(ws)
        items = collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "listening")
    assert types(items) == ["state", "state"]
    assert client.tts.texts == []


def test_text_input_and_markdown_is_not_spoken(voice_client):
    client = voice_client(replies=["**Sure.** Run `ls`:\n```bash\nls -la\n```\nSee https://x.io for more."])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "how do I list files"})
        collect_until(ws, lambda m: m["type"] == "audio_end")
    assert client.tts.texts == ["Sure.", "Run ls:", "See for more."]
    assert client.stt.calls == []


def test_interrupt_stops_audio_but_text_completes(voice_client):
    client = voice_client(replies=["One is here. Two is here. Three is here. Four is here."], tts_delay=0.3)
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "count"})
        first = collect_until(ws, lambda m: m["type"] == "audio")
        assert isinstance(recv(ws), bytes)
        ws.send_json({"type": "interrupt"})
        rest = collect_until(ws, lambda m: m["type"] == "audio_end")
        ws.send_json({"type": "ping"})
        tail = collect_until(ws, lambda m: m["type"] == "pong")
    end = rest[-1]
    assert end["interrupted"] is True and end["reason"] == "client"
    everything = first + rest + tail
    assert any(isinstance(m, dict) and m["type"] == "done" and not m.get("cancelled") for m in everything)
    after_end = [m for m in tail if isinstance(m, dict | bytes)]
    assert "audio" not in types(after_end) and "<bytes>" not in types(after_end)
    assert types(everything).count("audio_end") == 1
    assert len(client.tts.texts) < 4


def test_new_utterance_while_generating_cancels_turn(voice_client):
    llm = SlowProvider(replies=["This is a long story that keeps going and going. " * 3, "Short answer."], chunk_delay=0.03)
    client = voice_client(llm=llm, stt_texts=["actually, what time is it"])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "tell me a story"})
        collect_until(ws, lambda m: m["type"] == "text_delta")
        _speak(ws, ms=400, tail_ms=900)
        items = collect_until(ws, lambda m: m["type"] == "done" and m.get("cancelled"))
        items += collect_until(ws, lambda m: m["type"] == "audio_end" and not m.get("interrupted"))
    msgs = [m for m in items if isinstance(m, dict)]
    transcripts = [m["text"] for m in msgs if m["type"] == "transcript"]
    assert transcripts == ["actually, what time is it"]
    final_done = [m for m in msgs if m["type"] == "done" and not m.get("cancelled")]
    assert final_done and final_done[-1]["content"] == "Short answer."
    assert client.tts.texts[-1] == "Short answer."


def test_approval_passthrough_on_voice_socket(voice_client):
    client = voice_client(replies=[[tool_call("current_datetime")], "It is noon."], approvals="always")
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "what time is it"})
        items = collect_until(ws, lambda m: m["type"] == "approval_request")
        req = items[-1]
        assert req["name"] == "current_datetime"
        prompt = collect_until(ws, lambda m: m["type"] == "audio")[-1]
        assert "approval" in prompt["text"]
        ws.send_json({"type": "approval.respond", "approval_id": req["approval_id"], "decision": "allow"})
        rest = collect_until(ws, lambda m: m["type"] == "audio_end")
    msgs = [m for m in rest if isinstance(m, dict)]
    assert {"type": "approval.ack", "approval_id": req["approval_id"], "resolved": True} in msgs
    result = next(m for m in msgs if m["type"] == "tool_result")
    assert result["is_error"] is False
    assert client.tts.texts[-1] == "It is noon."


def test_spoken_yes_answers_pending_approval(voice_client):
    client = voice_client(
        replies=[[tool_call("current_datetime")], "It is noon."], approvals="always", stt_texts=["Yes, go ahead."]
    )
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "what time is it"})
        req = collect_until(ws, lambda m: m["type"] == "approval_request")[-1]
        _speak(ws, ms=500, tail_ms=900)
        rest = collect_until(ws, lambda m: m["type"] == "audio_end")
    msgs = [m for m in rest if isinstance(m, dict)]
    assert {"type": "approval.ack", "approval_id": req["approval_id"], "resolved": True} in msgs
    assert next(m for m in msgs if m["type"] == "tool_result")["is_error"] is False
    assert not any(m["type"] == "done" and m.get("cancelled") for m in msgs)


def test_stop_closes_cleanly(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "stop"})
        assert recv(ws) == {"type": "state", "state": "idle", "session_id": ws_session(client)}
        with pytest.raises((WebSocketDisconnect, AssertionError)):
            recv(ws)
    assert client.core.voice.sessions == set()


def ws_session(client):
    # the only session created in this test
    import asyncio

    rows = asyncio.run(client.core.store.list_sessions())
    return rows[0]["id"]


def test_audio_before_start_is_rejected(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ws.send_bytes(tone(100))
        err = recv(ws)
        assert err["type"] == "error" and "start" in err["message"]


def test_resume_existing_session(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        first = _start(ws)
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        again = _start(ws, session_id=first["session_id"])
    assert again["session_id"] == first["session_id"]
