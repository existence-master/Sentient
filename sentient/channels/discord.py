"""Discord bot channel: direct messages to the bot over the Discord gateway websocket.

Identify, heartbeat (with zombie detection), resume, and REST calls for sending, editing and
component buttons. Only DMs are handled; server messages are ignored. DM message content is
delivered to bots without the privileged Message Content intent, so only DIRECT_MESSAGES is requested.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import random
import sys
from typing import Any

import httpx

from sentient.channels.base import (
    MAX_DOWNLOAD_BYTES,
    Button,
    Channel,
    ChannelError,
    FileFetcher,
    Incoming,
)
from sentient.channels.formatting import DISCORD_LIMIT, split_markdown

log = logging.getLogger(__name__)

API = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
INTENT_DIRECT_MESSAGES = 1 << 12
INTENTS = INTENT_DIRECT_MESSAGES
BUTTON_STYLES = {"primary": 1, "secondary": 2, "success": 3, "danger": 4}
FATAL_CLOSE_CODES = {
    4004: "Discord rejected the bot token. Reset it in the Developer Portal and connect again.",
    4010: "Discord rejected the connection (invalid shard).",
    4011: "This bot is in too many servers for a single connection.",
    4012: "Discord no longer supports this gateway version. Update Sentient.",
    4013: "Discord rejected the requested permissions (invalid intents).",
    4014: "Discord refused the requested permissions (intents). Check the bot's settings in the Developer Portal.",
}
RESET_SESSION_CODES = {4007, 4009}

INSTRUCTIONS = """\
**What you need:** a Discord account and a server you own (you can create a private one in a few taps).
Your computer must be on with Sentient running for the bot to answer.

1. Open https://discord.com/developers/applications in a browser, sign in and click **New Application**.
   Name it (for example `My Sentient`) and accept the terms.
2. Open the **Bot** tab on the left. Click **Reset Token**, confirm, and **Copy** the token.
3. On the same tab you'll see **Message Content Intent**. Sentient only reads direct messages sent to the bot,
   which Discord allows without it, so you can leave it off. Turning it on does no harm.
4. Open **OAuth2**, then **URL Generator**. Tick **bot**, and under Bot Permissions tick **Send Messages**.
   Copy the URL at the bottom of the page.
5. Open that URL, choose your server and click **Authorize**. The bot now appears in your server's member list.
6. Paste the token above and click **Connect**.
7. Click **Pair a chat** to get a 6-digit code.
8. In Discord, click the bot in your server's member list, choose **Message**, and send `/pair` followed by
   the code, for example `/pair 123456`.

Now send the bot a direct message to talk to Sentient: text, voice messages, images and files all work.
Commands: `/new` starts a fresh chat, `/stop` stops a reply, `/stopall` stops everything Sentient is doing
(`/resume` starts it again), `/model` switches between local and cloud models, `/help` shows help.

**Keep the token private.** If it leaks, click **Reset Token** again and connect with the new one.
"""


class DiscordError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class _Fatal(Exception):
    pass


def _close_code(exc: BaseException) -> int | None:
    rcvd = getattr(exc, "rcvd", None)
    return getattr(rcvd, "code", None) if rcvd is not None else getattr(exc, "code", None)


class DiscordChannel(Channel):
    id = "discord"
    display_name = "Discord"
    secret_name = "channel_discord_token"
    message_limit = DISCORD_LIMIT
    setup_fields = [
        {
            "key": "bot_token",
            "label": "Bot token",
            "secret": True,
            "required": True,
            "help": "Developer Portal, your application, Bot tab, Reset Token.",
            "placeholder": "MTE...",
        }
    ]
    instructions_md = INSTRUCTIONS

    def __init__(self, service):
        super().__init__(service)
        self.http: httpx.AsyncClient | None = None
        self.session_id: str | None = None
        self.seq: int | None = None
        self.resume_url: str | None = None
        self.bot_id: str | None = None
        self._acked = True
        self.connect_ws = self._default_connect

    @staticmethod
    def _default_connect(url: str):
        from websockets.asyncio.client import connect

        return connect(url, max_size=2**22, open_timeout=20)

    # ------------------------------------------------------------------ HTTP
    @staticmethod
    def _client(token: str | None = None) -> httpx.AsyncClient:
        headers = {"User-Agent": "DiscordBot (https://sentient.local, 3.0)"}
        if token:
            headers["Authorization"] = f"Bot {token}"
        return httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0), headers=headers)

    async def open(self) -> None:
        if self.http is None:
            self.http = self._client(self.token)

    async def close(self) -> None:
        if self.http is not None:
            http, self.http = self.http, None
            await http.aclose()

    async def api(self, method: str, path: str, *, token: str | None = None, **kwargs: Any) -> Any:
        token = token or self.token
        if not token:
            raise ChannelError("Discord isn't connected.", 409)
        own = self.http is None or token != self.token
        http = self._client(token) if own else self.http
        assert http is not None
        try:
            for attempt in range(3):
                try:
                    r = await http.request(method, f"{API}{path}", **kwargs)
                except httpx.HTTPError as exc:
                    raise DiscordError(0, f"can't reach Discord ({type(exc).__name__})") from None
                if r.status_code == 429 and attempt < 2:
                    try:
                        retry = float(r.json().get("retry_after", 1))
                    except ValueError:
                        retry = 1.0
                    if retry <= 30:
                        await self.sleep(retry)
                        continue
                if r.status_code >= 400:
                    try:
                        message = str(r.json().get("message") or r.text)
                    except ValueError:
                        message = r.text
                    raise DiscordError(r.status_code, message[:300])
                if r.status_code == 204 or not r.content:
                    return {}
                return r.json()
        finally:
            if own:
                await http.aclose()
        return {}  # pragma: no cover

    # ------------------------------------------------------------------ connect
    async def validate(self, fields: dict[str, Any]) -> tuple[str, str]:
        token = str(fields.get("bot_token") or fields.get("token") or "").strip().removeprefix("Bot ").strip()
        if len(token) < 30 or " " in token:
            raise ChannelError("Please paste the bot token from the Discord Developer Portal (Bot tab, Reset Token).")
        try:
            me = await self.api("GET", "/users/@me", token=token)
        except DiscordError as exc:
            if exc.status in (401, 403):
                raise ChannelError(
                    "Discord didn't accept that bot token. Click Reset Token in the Developer Portal and paste the new one."
                ) from None
            raise ChannelError(f"Couldn't check the token with Discord: {exc.message}") from None
        if not me.get("bot"):
            raise ChannelError("That token doesn't belong to a Discord bot.")
        return token, str(me.get("username") or "Discord bot")

    async def check(self) -> None:
        await self.api("GET", "/users/@me")

    # ------------------------------------------------------------------ gateway
    async def run(self) -> None:
        backoff = 1.0
        failures = 0
        while True:
            try:
                await self.session()
                failures, backoff = 0, 1.0
                wait = 1.0
            except asyncio.CancelledError:
                raise
            except _Fatal as exc:
                await self.service.set_status(self.id, "error", error=str(exc))
                return
            except Exception as exc:
                failures += 1
                wait = backoff
                backoff = min(backoff * 2, 60.0)
                log.warning("discord gateway failed: %s", self.redact(repr(exc)))
                if failures >= 3:
                    await self.service.set_status(
                        self.id, "error", error="Can't reach Discord. Check your internet connection."
                    )
            await self.sleep(wait)

    async def session(self) -> None:
        """One websocket connection. Returns when Discord asks to reconnect; raises on fatal errors."""
        url = self.resume_url if (self.session_id and self.resume_url) else GATEWAY_URL
        async with self.connect_ws(url) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("op") != 10:
                raise RuntimeError("unexpected first gateway message")
            interval = float(hello["d"]["heartbeat_interval"]) / 1000.0
            self._acked = True
            heartbeat = asyncio.create_task(self._heartbeat(ws, interval))
            try:
                if self.session_id and self.seq is not None:
                    await ws.send(json.dumps({"op": 6, "d": {"token": self.token, "session_id": self.session_id, "seq": self.seq}}))
                else:
                    await self._identify(ws)
                while True:
                    try:
                        raw = await ws.recv()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        code = _close_code(exc)
                        if code in FATAL_CLOSE_CODES:
                            raise _Fatal(FATAL_CLOSE_CODES[code]) from None
                        if code in RESET_SESSION_CODES:
                            self._reset_session()
                        return
                    if await self.on_payload(ws, json.loads(raw)) == "reconnect":
                        return
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat

    def _reset_session(self) -> None:
        self.session_id = None
        self.seq = None
        self.resume_url = None

    async def _identify(self, ws: Any) -> None:
        await ws.send(json.dumps({
            "op": 2,
            "d": {
                "token": self.token,
                "intents": INTENTS,
                "properties": {"os": sys.platform, "browser": "sentient", "device": "sentient"},
            },
        }))

    async def _heartbeat(self, ws: Any, interval: float) -> None:
        await asyncio.sleep(interval * random.random())
        while True:
            if not self._acked:  # zombie connection: reconnect and resume
                with contextlib.suppress(Exception):
                    await ws.close(code=4000)
                return
            self._acked = False
            await ws.send(json.dumps({"op": 1, "d": self.seq}))
            await asyncio.sleep(interval)

    async def on_payload(self, ws: Any, p: dict) -> str | None:
        op = p.get("op")
        if p.get("s") is not None:
            self.seq = int(p["s"])
        if op == 0:
            kind, d = p.get("t"), p.get("d") or {}
            if kind == "READY":
                self.session_id = d.get("session_id")
                resume = str(d.get("resume_gateway_url") or "").rstrip("/")
                self.resume_url = f"{resume}/?v=10&encoding=json" if resume else None
                user = d.get("user") or {}
                self.bot_id = str(user.get("id") or "") or None
                await self.service.set_status(self.id, "connected", error=None, account_label=user.get("username"))
            elif kind == "RESUMED":
                await self.service.set_status(self.id, "connected", error=None)
            elif kind == "MESSAGE_CREATE":
                self.on_message(d)
            elif kind == "INTERACTION_CREATE":
                self.spawn(self.on_interaction(d))
        elif op == 1:
            await ws.send(json.dumps({"op": 1, "d": self.seq}))
        elif op == 7:
            with contextlib.suppress(Exception):
                await ws.close(code=4000)
            return "reconnect"
        elif op == 9:
            if not p.get("d"):
                self._reset_session()
            await self.sleep(random.uniform(1, 5))
            with contextlib.suppress(Exception):
                await ws.close(code=4000)
            return "reconnect"
        elif op == 11:
            self._acked = True
        return None

    def on_message(self, d: dict) -> None:
        if d.get("guild_id"):
            return  # direct messages only
        author = d.get("author") or {}
        if author.get("bot") or (self.bot_id and str(author.get("id")) == self.bot_id):
            return
        chat_id = str(d.get("channel_id") or "")
        if not chat_id:
            return
        audio: FileFetcher | None = None
        files: list[FileFetcher] = []
        for att in d.get("attachments") or []:
            ctype = att.get("content_type") or mimetypes.guess_type(att.get("filename") or "")[0] or ""
            fetch = self._fetcher(att)
            if ctype.startswith("audio/") and audio is None:
                audio = fetch
            else:
                files.append(fetch)
        text = str(d.get("content") or "")
        ref = d.get("message_reference") if isinstance(d.get("message_reference"), dict) else {}
        reply_to = ref.get("message_id") or (d.get("referenced_message") or {}).get("id")
        self.spawn(self.handle_incoming(Incoming(
            chat_id=chat_id,
            label=str(author.get("global_name") or author.get("username") or chat_id),
            text=text,
            audio=audio,
            files=files,
            unsupported=not (text or audio or files),
            reply_to=str(reply_to) if reply_to else None,
        )))

    def _fetcher(self, att: dict) -> FileFetcher:
        async def fetch() -> tuple[bytes, str]:
            if (att.get("size") or 0) > MAX_DOWNLOAD_BYTES:
                raise ChannelError("That file is larger than 20 MB, which is more than I can take.")
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
                try:
                    r = await http.get(str(att.get("url")))
                except httpx.HTTPError:
                    raise ChannelError("I couldn't download that file from Discord. Please try again.") from None
            if r.status_code != 200:
                raise ChannelError("I couldn't download that file from Discord. Please try again.")
            return r.content, str(att.get("filename") or "discord-file")

        return fetch

    async def on_interaction(self, d: dict) -> None:
        if d.get("type") != 3:  # message component
            return
        try:
            await self.api("POST", f"/interactions/{d.get('id')}/{d.get('token')}/callback", json={"type": 6})
        except Exception as exc:
            log.debug("discord interaction ack failed: %s", self.redact(str(exc)))
        chat_id = str(d.get("channel_id") or "")
        message_id = str((d.get("message") or {}).get("id") or "")
        custom_id = str((d.get("data") or {}).get("custom_id") or "")
        if chat_id and message_id:
            await self.handle_button(chat_id, message_id, custom_id)

    # ------------------------------------------------------------------ sending
    def render(self, md: str) -> list[str]:
        return split_markdown(md, DISCORD_LIMIT)

    @staticmethod
    def _components(buttons: list[list[Button]] | None) -> list[dict]:
        return [
            {"type": 1, "components": [
                {"type": 2, "style": BUTTON_STYLES.get(b.style, 1), "label": b.label[:80], "custom_id": b.data[:100]}
                for b in row[:5]
            ]}
            for row in (buttons or [])[:5]
        ]

    async def send_chunk(self, chat_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> str:
        body: dict[str, Any] = {"content": chunk, "allowed_mentions": {"parse": []}}
        if buttons:
            body["components"] = self._components(buttons)
        m = await self.api("POST", f"/channels/{chat_id}/messages", json=body)
        return str(m.get("id"))

    async def edit_chunk(self, chat_id: str, message_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> None:
        await self.api("PATCH", f"/channels/{chat_id}/messages/{message_id}",
                       json={"content": chunk, "components": self._components(buttons)})

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        await self.api("DELETE", f"/channels/{chat_id}/messages/{message_id}")

    async def clear_buttons(self, chat_id: str, message_id: str) -> None:
        await self.api("PATCH", f"/channels/{chat_id}/messages/{message_id}", json={"components": []})

    async def send_typing(self, chat_id: str) -> None:
        await self.api("POST", f"/channels/{chat_id}/typing")

    async def send_audio(self, chat_id: str, data: bytes, filename: str) -> None:
        payload = {"attachments": [{"id": 0, "filename": filename}], "allowed_mentions": {"parse": []}}
        await self.api("POST", f"/channels/{chat_id}/messages",
                       data={"payload_json": json.dumps(payload)}, files={"files[0]": (filename, data, "audio/wav")})

    def pairing_instructions(self, code: str, account_label: str | None) -> str:
        bot = account_label or "your bot"
        return f"Send a direct message to {bot} on Discord: /pair {code}"
