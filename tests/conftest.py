from __future__ import annotations

import json
import os
import sys
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from sentient.config.schema import SentientConfig
from sentient.llm.provider import StreamChunk, ToolCall
from sentient.paths import _windows_documents

# The real user's folders a test must never write to (resolved once, before any test patches the home folder).
_REAL_HOME = Path(os.path.expanduser("~"))
REAL_GUARDED = [
    _REAL_HOME / ".sentient" / "backups",
    _REAL_HOME / "Documents" / "Sentient Backups",
    *([Path(os.path.expandvars(os.environ["XDG_DOCUMENTS_DIR"])).expanduser() / "Sentient Backups"]
      if os.environ.get("XDG_DOCUMENTS_DIR") else []),
    # Windows' real Documents (it may be redirected, e.g. to OneDrive)
    *([known / "Sentient Backups"] if sys.platform == "win32" and (known := _windows_documents()) is not None else []),
]


def _guarded_state() -> list:
    out = []
    for folder in REAL_GUARDED:
        try:
            out.append(sorted((p.name, p.stat().st_mtime_ns) for p in folder.iterdir()) if folder.exists() else None)
        except OSError:
            out.append("unreadable")
    return out


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.sentient and its own Documents folder (where backups go by default), and fails if
    it wrote backups to the real user's ones (issue #151)."""
    before = _guarded_state()
    monkeypatch.setenv("SENTIENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SENTIENT_DOCUMENTS_DIR", str(tmp_path / "documents"))
    yield tmp_path / "home"
    assert _guarded_state() == before, f"a test wrote to the real user's backup folders: {REAL_GUARDED}"


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
    cfg.assistant.user_name = "Maya"
    cfg.tools.approvals.mode = "off"
    # the second look's model check (#143) is a model call: tests that cover it turn it on and script the answer
    cfg.tools.approvals.second_look_model = False
    # the undo window (#142) holds a desktop chat's sends for seconds: tests that cover it turn it on
    cfg.tools.approvals.undo_seconds = 0
    cfg.memory.extract_after_turn = False
    cfg.chat.auto_title = False
    return cfg


def tool_call(tool_name: str, /, **arguments) -> dict:
    return {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}


def dumps(x) -> str:
    return json.dumps(x)
