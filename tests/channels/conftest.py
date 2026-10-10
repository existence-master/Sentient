"""Fixtures for channels: in-memory keychain, a started app, and a fake Telegram Bot API on respx."""

from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx

from sentient import secrets
from sentient.app import SentientApp
from tests.conftest import FakeProvider

TOKEN = "123456789:" + "A" * 35
_update_ids = itertools.count(1)


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


def ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


def tg_error(code: int, description: str, retry_after: int | None = None) -> httpx.Response:
    body: dict[str, Any] = {"ok": False, "error_code": code, "description": description}
    if retry_after is not None:
        body["parameters"] = {"retry_after": retry_after}
    return httpx.Response(code, json=body)


class FakeTelegram:
    """Records every Bot API call. ``queue[method]`` holds responses/exceptions returned before the defaults."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.queue: dict[str, list[Any]] = {}
        self.files: dict[str, bytes] = {}
        self.next_id = 1000
        self.on_call = None

    def mount(self, router: respx.MockRouter) -> None:
        router.route(host="api.telegram.org", path__startswith="/file/").mock(side_effect=self.download)
        router.route(host="api.telegram.org", path__startswith=f"/bot{TOKEN}/").mock(side_effect=self.handle)
        router.route(host="api.telegram.org").mock(return_value=tg_error(401, "Unauthorized"))

    def handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        ctype = request.headers.get("content-type", "")
        params = json.loads(request.content or b"{}") if "json" in ctype else {"_multipart": True}
        self.calls.append((method, params))
        if self.on_call is not None:
            self.on_call(method, params)
        if self.queue.get(method):
            item = self.queue[method].pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, httpx.Response) else ok(item)
        if method == "getMe":
            return ok({"id": 1, "is_bot": True, "username": "sentient_test_bot", "first_name": "Sentient"})
        if method == "sendMessage":
            self.next_id += 1
            params["_id"] = self.next_id
            return ok({"message_id": self.next_id, "chat": {"id": params.get("chat_id")}, "text": params.get("text")})
        if method == "getUpdates":
            return ok([])
        if method == "getFile":
            return ok({"file_id": params["file_id"], "file_path": f"docs/{params['file_id']}"})
        return ok(True)

    def download(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name not in self.files:
            return httpx.Response(404)
        return httpx.Response(200, content=self.files[name])

    # ------------------------------------------------------------------ assertions
    def sent(self, method: str) -> list[dict]:
        return [p for m, p in self.calls if m == method]

    def screen(self) -> list[str]:
        """Current text of every visible message, oldest first."""
        msgs: dict[int, str] = {}
        for m, p in self.calls:
            if m == "sendMessage":
                msgs[p["_id"]] = p["text"]
            elif m == "editMessageText":
                msgs[int(p["message_id"])] = p["text"]
            elif m == "deleteMessage":
                msgs.pop(int(p["message_id"]), None)
        return [msgs[k] for k in sorted(msgs)]

    def button_messages(self) -> list[dict]:
        return [p for p in self.sent("sendMessage") if "reply_markup" in p]


def update(chat_id: int, text: str | None = None, **extra: Any) -> dict:
    uid = next(_update_ids)
    message: dict[str, Any] = {
        "message_id": uid,
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": chat_id, "is_bot": False, "first_name": "Sarthak", "username": "sk"},
        **extra,
    }
    if text is not None:
        message["text"] = text
    return {"update_id": uid, "message": message}


def callback(chat_id: int, message_id: int, data: str) -> dict:
    uid = next(_update_ids)
    return {
        "update_id": uid,
        "callback_query": {
            "id": f"cq{uid}", "data": data, "from": {"id": chat_id},
            "message": {"message_id": message_id, "chat": {"id": chat_id, "type": "private"}},
        },
    }


async def until(predicate, timeout: float = 30.0) -> None:
    """Wait for ``predicate``; the generous ceiling only matters when it never comes true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class GatedProvider(FakeProvider):
    """FakeProvider whose Nth ``stream`` call waits for ``gate``."""

    def __init__(self, *args, block_on_call: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_on_call = block_on_call
        self.stream_calls = 0
        self.gate = asyncio.Event()
        self.blocked = asyncio.Event()

    async def stream(self, role, messages, tools=None, *, model=None):
        self.stream_calls += 1
        if self.block_on_call is not None and self.stream_calls == self.block_on_call:
            self.blocked.set()
            await self.gate.wait()
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


@dataclass
class Env:
    app: SentientApp
    ch: Any
    api: FakeTelegram

    async def pair(self, chat_id: int = 42) -> dict:
        code = (await self.app.channels.create_pairing("telegram"))["code"]
        self.ch.dispatch(update(chat_id, f"/pair {code}"))
        await self.ch.wait_idle()
        chat = await self.app.channels.store.chat("telegram", str(chat_id))
        assert chat is not None
        return chat

    async def say(self, chat_id: int, text: str | None = None, **extra: Any) -> None:
        self.ch.dispatch(update(chat_id, text, **extra))
        await self.ch.wait_idle()


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
async def app(config, isolated_home, llm):
    config.chat.tool_selection = "all"
    a = SentientApp(config, llm=llm, db_path=isolated_home / "channels.db", enable_background=False)
    await a.start()
    try:
        yield a
    finally:
        await a.stop()


async def _no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


@pytest.fixture
async def tg(app, keychain):
    fake = FakeTelegram()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        await app.channels.connect("telegram", {"bot_token": TOKEN})
        ch = app.channels.channels["telegram"]
        ch.sleep = _no_sleep
        await ch.open()
        try:
            yield Env(app, ch, fake)
        finally:
            await ch.stop_runtime()
