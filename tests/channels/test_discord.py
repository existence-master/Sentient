from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

import httpx
import pytest
import respx
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from sentient.channels import ChannelError
from sentient.channels.discord import API, INTENTS
from tests.channels.conftest import until
from tests.conftest import tool_call

DTOKEN = "MTE" + "x" * 60


class FakeDiscordAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.ids = itertools.count(1)

    def mount(self, router: respx.MockRouter) -> None:
        router.route(host="discord.com").mock(side_effect=self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v10")
        body = json.loads(request.content) if request.content and "json" in request.headers.get("content-type", "") else None
        self.calls.append((request.method, path, body))
        auth = request.headers.get("authorization", "")
        if path == "/users/@me":
            if auth != f"Bot {DTOKEN}":
                return httpx.Response(401, json={"message": "401: Unauthorized", "code": 0})
            return httpx.Response(200, json={"id": "900", "username": "sentient", "bot": True})
        if request.method == "POST" and path.endswith("/messages"):
            return httpx.Response(200, json={"id": f"m{next(self.ids)}"})
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": path.rsplit("/", 1)[-1]})
        return httpx.Response(204)

    def screen(self, channel_id: str = "dm1") -> list[str]:
        msgs: dict[str, str] = {}
        order: list[str] = []
        created = itertools.count(1)
        for method, path, body in self.calls:
            if method == "POST" and path == f"/channels/{channel_id}/messages":
                mid = f"m{next(created)}"
                msgs[mid] = body["content"]
                order.append(mid)
            elif method == "POST" and path.endswith("/messages"):
                next(created)
            elif method == "PATCH" and path.startswith(f"/channels/{channel_id}/messages/") and "content" in (body or {}):
                msgs[path.rsplit("/", 1)[-1]] = body["content"]
            elif method == "DELETE" and path.startswith(f"/channels/{channel_id}/messages/"):
                msgs.pop(path.rsplit("/", 1)[-1], None)
        return [msgs[m] for m in order if m in msgs]


class FakeWS:
    def __init__(self, frames: list[Any]):
        self.incoming: asyncio.Queue = asyncio.Queue()
        for f in frames:
            self.incoming.put_nowait(f)
        self.sent: list[dict] = []
        self.closed: int | None = None

    async def __aenter__(self) -> FakeWS:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def recv(self) -> str:
        item = await asyncio.wait_for(self.incoming.get(), 5)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = code
        self.incoming.put_nowait(ConnectionClosedError(Close(code, reason), None))


def closed(code: int) -> ConnectionClosedError:
    return ConnectionClosedError(Close(code, "bye"), None)


HELLO = {"op": 10, "d": {"heartbeat_interval": 45000}}


def dm(seq: int, content: str, **extra: Any) -> dict:
    d = {"id": f"msg{seq}", "channel_id": "dm1", "author": {"id": "5", "username": "sk", "global_name": "Sarthak"},
         "content": content, "attachments": [], **extra}
    return {"op": 0, "t": "MESSAGE_CREATE", "s": seq, "d": d}


@pytest.fixture
async def dc(app, keychain):
    fake = FakeDiscordAPI()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        await app.channels.connect("discord", {"bot_token": f"Bot {DTOKEN}"})
        ch = app.channels.channels["discord"]

        async def no_sleep(_s: float) -> None:
            await asyncio.sleep(0)

        ch.sleep = no_sleep
        await ch.open()
        try:
            yield app, ch, fake
        finally:
            await ch.stop_runtime()


async def test_validate_token(app, keychain):
    fake = FakeDiscordAPI()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        with pytest.raises(ChannelError, match="paste the bot token"):
            await app.channels.connect("discord", {"bot_token": "short"})
        with pytest.raises(ChannelError, match="didn't accept"):
            await app.channels.connect("discord", {"bot_token": "MTE" + "y" * 60})
        channel = await app.channels.connect("discord", {"bot_token": DTOKEN})
    assert channel["status"] == "connected" and channel["account_label"] == "sentient"
    assert keychain["channel_discord_token"] == DTOKEN


async def test_identify_ready_pair_resume_and_reply(dc, llm):
    app, ch, api = dc
    code = (await app.channels.create_pairing("discord"))["code"]
    urls: list[str] = []
    ready = {"op": 0, "t": "READY", "s": 1, "d": {"session_id": "sess1", "resume_gateway_url": "wss://resume.discord.gg",
                                                   "user": {"id": "900", "username": "sentient"}}}
    ws1 = FakeWS([HELLO, ready, dm(2, f"/pair {code}"), dm(3, "from a server", guild_id="g1"),
                  {"op": 0, "t": "MESSAGE_CREATE", "s": 4, "d": {"channel_id": "dm1", "author": {"id": "900", "bot": True}, "content": "echo"}},
                  closed(1006)])
    ch.connect_ws = lambda url: urls.append(url) or ws1
    await ch.session()
    await ch.wait_idle()
    assert ws1.sent[0]["op"] == 2 and ws1.sent[0]["d"]["intents"] == INTENTS and ws1.sent[0]["d"]["token"] == DTOKEN
    assert ch.session_id == "sess1" and ch.seq == 4
    assert "Paired!" in api.screen()[-1]
    assert len(api.screen()) == 1  # server message and the bot's own message ignored
    chat = await app.channels.store.chat("discord", "dm1")
    assert chat["label"] == "Sarthak" and (await app.store.get_session(chat["session_id"]))["channel"] == "discord"

    llm.replies = ["Hi from **Discord**"]
    ws2 = FakeWS([HELLO, {"op": 0, "t": "RESUMED", "s": 5, "d": {}}, {"op": 1, "d": None}, {"op": 11},
                  dm(6, "hello"), {"op": 7, "d": None}])
    ch.connect_ws = lambda url: urls.append(url) or ws2
    await ch.session()
    await ch.wait_idle()
    assert urls == ["wss://gateway.discord.gg/?v=10&encoding=json", "wss://resume.discord.gg/?v=10&encoding=json"]
    assert ws2.sent[0] == {"op": 6, "d": {"token": DTOKEN, "session_id": "sess1", "seq": 4}}
    assert {"op": 1, "d": 5} in ws2.sent  # heartbeat on request
    assert ws2.closed == 4000  # reconnect keeps the session resumable
    assert api.screen()[-1] == "Hi from **Discord**"
    assert all(b is None or b.get("allowed_mentions") == {"parse": []} for m, p, b in api.calls if m == "POST" and p.endswith("/messages"))
    state = await app.channels.store.state("discord")
    assert state["status"] == "connected"


async def test_fatal_close_stops_with_error(dc):
    app, ch, _ = dc
    ch.connect_ws = lambda url: FakeWS([HELLO, closed(4004)])
    await asyncio.wait_for(ch.run(), 5)
    state = await app.channels.store.state("discord")
    assert state["status"] == "error" and "rejected the bot token" in state["error"]


async def test_invalid_session_resets_and_reconnects(dc):
    _, ch, _ = dc
    ch.session_id, ch.seq, ch.resume_url = "old", 9, "wss://resume.discord.gg/?v=10&encoding=json"
    ws = FakeWS([HELLO, {"op": 9, "d": False}])
    ch.connect_ws = lambda url: ws
    await ch.session()
    assert ch.session_id is None and ch.seq is None and ws.closed == 4000


async def test_button_interaction_resolves_approval(dc, llm):
    app, ch, api = dc
    app.config.tools.approvals.mode = "always"
    sid = await app.store.create_session(channel="discord")
    await app.channels.store.add_chat("discord", "dm1", "Sarthak", deliver=True, session_id=sid)
    llm.replies = [[tool_call("current_datetime")], "It is noon."]
    ch.on_message(dm(1, "what time is it")["d"])
    await until(lambda: any(b and b.get("components") for m, p, b in api.calls if m == "POST"))
    method, path, body = next((m, p, b) for m, p, b in api.calls if m == "POST" and b and b.get("components"))
    buttons = body["components"][0]["components"]
    assert [b["label"] for b in buttons] == ["Allow", "Allow for this chat"] and buttons[0]["style"] == 3
    posted = [c for c in api.calls if c[0] == "POST" and c[1].endswith("/messages")]
    message_id = f"m{posted.index((method, path, body)) + 1}"
    await ch.on_interaction({"type": 3, "id": "i1", "token": "itok", "channel_id": "dm1",
                             "message": {"id": message_id}, "data": {"custom_id": buttons[0]["custom_id"]}})
    await ch.wait_idle()
    assert ("POST", "/interactions/i1/itok/callback", {"type": 6}) in api.calls
    patch = [b for m, p, b in api.calls if m == "PATCH" and p.endswith(message_id)][-1]
    assert "**Allowed**" in patch["content"] and patch["components"] == []
    assert api.screen()[-1] == "It is noon."
    assert API.endswith("/v10")


async def test_reply_reference_is_parsed(dc):
    _app, ch, _api = dc
    seen: list = []

    async def capture(msg) -> None:
        seen.append(msg)

    ch.handle_incoming = capture
    ch.on_message(dm(1, "The early one", message_reference={"message_id": "q77", "channel_id": "dm1"})["d"])
    ch.on_message(dm(2, "Just chatting")["d"])
    await ch.wait_idle()
    assert [m.reply_to for m in seen] == ["q77", None]
