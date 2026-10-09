"""Stop everything during voice: a spoken answer to an approval must not start a new turn afterwards."""

from __future__ import annotations

import time

from tests.conftest import tool_call
from tests.voice.conftest import collect_until
from tests.voice.test_ws_voice import _speak, _start


def test_stop_cancels_a_spoken_answer_still_being_transcribed(voice_client):
    client = voice_client(
        replies=[[tool_call("current_datetime")], "It is noon.", "Here is a joke."],
        approvals="always", stt_texts=["tell me a joke"],
    )
    client.stt.delay = 1.0  # still transcribing when the stop lands
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_json({"type": "text", "text": "what time is it"})
        collect_until(ws, lambda m: m["type"] == "approval_request")
        _speak(ws, ms=500, tail_ms=900)
        collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "transcribing")
        assert client.post("/api/stop-all").json()["stopped"] is True
        collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "listening")
        time.sleep(1.5)  # longer than the transcription would have taken
    assert len(client.core.llm.calls) == 1  # the transcript never became a new request
