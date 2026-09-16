from __future__ import annotations

import json
import zlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sentient.config.schema import SentientConfig
from sentient.llm.provider import StreamChunk, ToolCall


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.sentient."""
    monkeypatch.setenv("SENTIENT_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


class FakeProvider:
    """Scripted LLM: a queue of replies, each either text or a list of tool calls.
    Embeddings are deterministic bag-of-words vectors so similarity is meaningful."""

    def __init__(self, replies: list[Any] | None = None, json_replies: list[Any] | None = None):
        self.replies = list(replies or [])
        self.json_replies = list(json_replies or [])
        self.calls: list[dict] = []
        self.text_calls: list[dict] = []
        self.text_replies: list[str] = []
        self.dim = 16

    def model_for(self, role: str) -> str:
        return f"fake/{role}"

    async def stream(self, role, messages, tools=None, *, model=None) -> AsyncIterator[StreamChunk]:
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, list):
            yield StreamChunk(done=True, tool_calls=[ToolCall(**tc) for tc in reply], model="fake")
            return
        for i in range(0, len(reply), 5):
            yield StreamChunk(text=reply[i : i + 5], model="fake")
        yield StreamChunk(done=True, usage={"prompt_tokens": 1, "completion_tokens": 1}, model="fake")

    async def complete_text(self, role, messages, *, model=None) -> str:
        self.text_calls.append({"role": role, "messages": messages})
        return self.text_replies.pop(0) if self.text_replies else "Untitled chat"

    async def complete_json(self, role, messages, *, model=None):
        self.calls.append({"role": role, "messages": messages, "json": True})
        return self.json_replies.pop(0) if self.json_replies else {"facts": []}

    async def embed(self, texts: list[str], *, model=None) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for word in t.lower().split():
                # crc32, not hash(): str hashing is randomized per process, which made similarity flaky
                vec[zlib.crc32(word.strip(".,!?'\"").encode()) % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def config():
    cfg = SentientConfig()
    cfg.assistant.user_name = "Sarthak"
    cfg.tools.approvals.mode = "off"
    cfg.memory.extract_after_turn = False
    cfg.chat.auto_title = False
    return cfg


def tool_call(tool_name: str, /, **arguments) -> dict:
    return {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}


def dumps(x) -> str:
    return json.dumps(x)
