"""Push to talk and dictation into any app (#169): cleanup, polish guard, the REST endpoint, Stop everything."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from sentient.app import SentientApp
from sentient.voice.audio import pcm16_to_wav
from sentient.voice.base import STTProvider
from sentient.voice.dictation import clean_dictation, is_faithful, tidy
from sentient.voice.service import DictationStopped
from sentient.voice.stt import FasterWhisperSTT
from tests.conftest import FakeProvider
from tests.voice.conftest import tone


class FileSTT(STTProvider):
    """Returns ``text`` for any file; with ``gate`` it waits until the gate opens (still transcribing)."""

    name = "file_stt"
    model = "fake"

    def __init__(self, text: str, gate: asyncio.Event | None = None):
        self.text = text
        self.gate = gate
        self.started = asyncio.Event()
        self.files: list[str] = []

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        return self.text

    async def transcribe_file(self, path) -> str:
        self.files.append(Path(path).suffix)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return self.text


WAV = pcm16_to_wav(tone(300), 16000)


# ----------------------------------------------------------------------------- tidy
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("um so I think we should uh meet tomorrow", "So I think we should meet tomorrow."),
        ("Um, I think, uh, it works", "I think, it works."),
        ("hello world . how are you ?", "Hello world. How are you?"),
        ("the umbrella is under her error log", "The umbrella is under her error log."),
        ("uh-huh, that works for me", "Uh-huh, that works for me."),
        ("erm... see you at 3:30 pm", "See you at 3:30 pm."),
        ("first point. second point", "First point. Second point."),
        ("ok", "Ok"),
        ("  um uh  ", ""),
        ("", ""),
    ],
)
def test_tidy_removes_fillers_and_fixes_punctuation(raw, expected):
    assert tidy(raw) == expected


def test_tidy_never_changes_the_words():
    raw = "so the the report is due on Friday and Maya said that that was fine"
    assert is_faithful(raw, tidy(raw))
    assert tidy(raw).lower().rstrip(".") == raw.lower()  # repeated words stay: "that that" can be meant


# ----------------------------------------------------------------------------- polish guard
@pytest.mark.parametrize(
    "candidate",
    [
        "So, I think we should meet tomorrow at 10.",
        "so I think we should meet tomorrow at 10",
        "Um, so I think we should meet tomorrow at 10!",
    ],
)
def test_faithful_when_only_punctuation_capitals_or_fillers_change(candidate):
    assert is_faithful("So I think we should meet tomorrow at 10.", candidate)
    assert is_faithful("So I think we we should meet tomorrow at 10.", "So I think we should meet tomorrow at 10.")


@pytest.mark.parametrize(
    "candidate",
    [
        "So I think we should cancel tomorrow at 10.",  # a word with another meaning
        "So I think we should meet tomorrow at 11.",  # a different number
        "So I think we should not meet tomorrow at 10.",  # one word flips the meaning of a short sentence
        "Sure! Here is a plan for your meeting tomorrow at 10.",  # an answer instead of the text
        "So I think we should meet tomorrow at 10. Let me know if that works for you.",  # an added sentence
        "",
    ],
)
def test_not_faithful_when_the_meaning_could_change(candidate):
    assert not is_faithful("So I think we should meet tomorrow at 10.", candidate)


async def test_raw_and_tidy_never_call_a_model():
    llm = FakeProvider()
    raw = await clean_dictation(llm, " um send   the file ", "raw")
    assert raw == {"text": "um send the file", "raw": "um send the file", "cleanup": "raw", "polished": False}
    tidied = await clean_dictation(llm, "um send the file", "tidy")
    assert tidied["text"] == "Send the file." and tidied["polished"] is False
    assert llm.text_calls == [] and llm.calls == []


async def test_polish_uses_the_fast_model_when_it_keeps_the_words():
    llm = FakeProvider()
    llm.text_replies = ['"Hi Maya, the report is ready, and I sent it at 9."']
    out = await clean_dictation(llm, "hi maya the report is ready and uh I sent it at 9", "polish")
    assert out == {
        "text": "Hi Maya, the report is ready, and I sent it at 9.",
        "raw": "hi maya the report is ready and uh I sent it at 9",
        "cleanup": "polish",
        "polished": True,
    }
    assert [c["role"] for c in llm.text_calls] == ["fast"]
    assert llm.text_calls[0]["messages"][-1]["content"] == "Hi maya the report is ready and I sent it at 9."


@pytest.mark.parametrize(
    "reply",
    [
        "Here's a reply to Maya: Thanks, I'll read it now.",  # followed the text instead of cleaning it
        "Hi Maya, the report is ready, and I sent it at 10.",  # changed a number
        "Ignore that. The report is not ready.",  # changed the meaning
    ],
)
async def test_polish_falls_back_to_tidy_text_when_the_model_changes_the_meaning(reply):
    llm = FakeProvider()
    llm.text_replies = [reply]
    out = await clean_dictation(llm, "hi maya the report is ready and I sent it at 9", "polish")
    assert out["text"] == "Hi maya the report is ready and I sent it at 9."
    assert out["polished"] is False


async def test_polish_falls_back_when_the_model_fails():
    class Broken(FakeProvider):
        async def complete_text(self, role, messages, *, model=None):
            raise RuntimeError("model is down")

    out = await clean_dictation(Broken(), "um call me back", "polish")
    assert out == {"text": "Call me back.", "raw": "um call me back", "cleanup": "polish", "polished": False}


# ----------------------------------------------------------------------------- STT choice
def test_dictation_stays_on_this_computer_with_a_cloud_stt(config, isolated_home):
    config.voice.stt_provider = "openai"
    config.voice.stt_model = "gpt-4o-transcribe"
    config.voice.dictation.language = "de-DE"
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "d.db", enable_background=False)
    stt = core.voice.dictation_stt
    assert isinstance(stt, FasterWhisperSTT)  # built lazily: nothing is downloaded or loaded here
    assert (stt.model, stt.language) == ("base", "de")
    assert core.voice.dictation_stt is stt


def test_dictation_reuses_the_local_chat_model(config, isolated_home):
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "d.db", enable_background=False)
    assert core.voice.dictation_stt is core.voice.stt
    core.config.voice.dictation.language = "auto"
    assert core.voice.dictation_stt is not core.voice.stt
    assert core.voice.dictation_stt.language is None


# ----------------------------------------------------------------------------- REST
def test_dictate_endpoint_transcribes_locally_and_tidies(voice_client):
    client = voice_client()
    stt = FileSTT("um so the meeting moved to Thursday")
    client.core.voice.use_providers(stt=stt, tts=client.tts)
    r = client.post("/api/voice/dictate", files={"file": ("clip.webm", b"\x1aE\xdf\xa3", "audio/webm;codecs=opus")})
    assert r.status_code == 200
    assert r.json() == {
        "text": "So the meeting moved to Thursday.",
        "raw": "um so the meeting moved to Thursday",
        "cleanup": "tidy",
        "polished": False,
    }
    assert stt.files == [".webm"]
    assert client.core.llm.text_calls == []  # tidy is local: the text went nowhere


def test_dictate_endpoint_polishes_with_the_fast_model(voice_client):
    client = voice_client()
    client.core.config.voice.dictation.cleanup = "polish"
    client.core.voice.use_providers(stt=FileSTT("thanks see you then"), tts=client.tts)
    client.core.llm.text_replies = ["Thanks, see you then."]
    r = client.post("/api/voice/dictate", files={"file": ("clip.wav", WAV, "audio/wav")})
    assert r.json() == {"text": "Thanks, see you then.", "raw": "thanks see you then", "cleanup": "polish", "polished": True}


def test_push_to_talk_asks_for_tidy_text_whatever_the_setting(voice_client):
    client = voice_client()
    client.core.config.voice.dictation.cleanup = "polish"
    client.core.voice.use_providers(stt=FileSTT("um what is on my calendar today"), tts=client.tts)
    r = client.post("/api/voice/dictate", files={"file": ("clip.wav", WAV, "audio/wav")}, data={"cleanup": "tidy"})
    assert r.json()["text"] == "What is on my calendar today." and r.json()["cleanup"] == "tidy"
    assert client.core.llm.text_calls == []
    bad = client.post("/api/voice/dictate", files={"file": ("clip.wav", WAV, "audio/wav")}, data={"cleanup": "shout"})
    assert bad.status_code == 422


def test_dictate_endpoint_rejects_bad_uploads(voice_client):
    client = voice_client()
    assert client.post("/api/voice/dictate", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/api/voice/dictate", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400


# ----------------------------------------------------------------------------- Stop everything
async def test_stop_everything_cancels_an_active_dictation(config, isolated_home):
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "stop.db", enable_background=False)
    await core.start()
    try:
        stt = FileSTT("never typed", gate=asyncio.Event())
        core.voice.use_providers(stt=stt)
        job = asyncio.create_task(core.voice.dictate(WAV, "clip.wav"))
        await asyncio.wait_for(stt.started.wait(), 5)
        state = await core.stop_all(source="hotkey")
        assert state["cancelled"] >= 1
        with pytest.raises(DictationStopped):
            await asyncio.wait_for(job, 5)
        assert not core.voice._dictations
        assert core.llm.text_calls == []
    finally:
        await core.stop()


def test_stop_everything_answers_a_waiting_dictate_request_with_409(voice_client):
    client = voice_client()
    gate = asyncio.Event()  # never set: the transcription only ends when Stop everything cancels it
    stt = FileSTT("never typed", gate=gate)
    client.core.voice.use_providers(stt=stt, tts=client.tts)
    result: dict = {}

    def dictate() -> None:
        r = client.post("/api/voice/dictate", files={"file": ("clip.wav", WAV, "audio/wav")})
        result["status"], result["body"] = r.status_code, r.json()

    worker = threading.Thread(target=dictate)
    worker.start()
    for _ in range(500):
        if stt.started.is_set():
            break
        threading.Event().wait(0.01)
    assert stt.started.is_set()
    assert client.post("/api/stop-all", json={"source": "hotkey"}).json()["stopped"] is True
    worker.join(10)
    assert result == {"status": 409, "body": {"detail": "Stopped by Stop everything."}}
