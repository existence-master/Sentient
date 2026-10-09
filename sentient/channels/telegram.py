"""Telegram bot channel: Bot API over HTTPS long polling (no webhook, no public URL needed)."""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import re
from pathlib import Path
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
from sentient.channels.formatting import TELEGRAM_LIMIT, html_to_plain, render_telegram_chunks

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")

INSTRUCTIONS = """\
**What you need:** the Telegram app and about two minutes. Your computer must be on with Sentient running
for the bot to answer.

1. In Telegram, search for **@BotFather** (it has a blue check mark) and open the chat.
2. Tap **Start**, then send `/newbot`.
3. BotFather asks for a **name**. Type anything you like, for example `My Sentient`.
4. Then it asks for a **username**. It must end in `bot`, for example `maya_sentient_bot`. If it is taken, try another.
5. BotFather replies with your **token**. It looks like `123456789:AAF...`. Tap and hold the message and copy the token.
6. Paste the token above and click **Connect**. Sentient checks it and shows your bot's name.
7. Click **Pair a chat**. Sentient shows a 6-digit code that works for 10 minutes.
8. In Telegram, open your new bot (tap the `t.me/...` link in BotFather's message), tap **Start**, and send
   `/pair` followed by the code, for example `/pair 123456`.

Now just message your bot to talk to Sentient. You can send text, voice notes, photos and files.
Commands: `/new` starts a fresh chat, `/stop` stops a reply, `/stopall` stops everything Sentient is doing
(`/resume` starts it again), `/help` shows help.

**Keep the token private.** Anyone who has it can use your bot. If it leaks, send `/revoke` to BotFather,
then connect again here with the new token.
"""


class TelegramError(Exception):
    def __init__(self, code: int, description: str, retry_after: float | None = None):
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description
        self.retry_after = retry_after


def _chat(chat_id: str) -> int | str:
    return int(chat_id) if re.fullmatch(r"-?\d+", str(chat_id)) else str(chat_id)


def _wav_to_ogg_opus(data: bytes) -> bytes:
    """Convert WAV bytes to a mono 48 kHz OGG/Opus voice note. Raises on failure."""
    import av

    with av.open(io.BytesIO(data)) as src:
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="ogg") as dst:
            stream = dst.add_stream("libopus", rate=48000)
            stream.layout = "mono"
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
            for frame in src.decode(audio=0):
                for resampled in resampler.resample(frame):
                    for packet in stream.encode(resampled):
                        dst.mux(packet)
            for resampled in resampler.resample(None):
                for packet in stream.encode(resampled):
                    dst.mux(packet)
            for packet in stream.encode(None):
                dst.mux(packet)
        ogg = buf.getvalue()
    if not ogg.startswith(b"OggS"):
        raise ValueError("voice conversion did not produce Ogg")
    return ogg


class TelegramChannel(Channel):
    id = "telegram"
    display_name = "Telegram"
    secret_name = "channel_telegram_token"
    message_limit = TELEGRAM_LIMIT
    poll_timeout = 30
    setup_fields = [
        {
            "key": "bot_token",
            "label": "Bot token",
            "secret": True,
            "required": True,
            "help": "The token @BotFather sends you after /newbot.",
            "placeholder": "123456789:AAF...",
        }
    ]
    instructions_md = INSTRUCTIONS

    def __init__(self, service):
        super().__init__(service)
        self.http: httpx.AsyncClient | None = None
        self.offset: int | None = None
        self.bot_username: str | None = None

    # ------------------------------------------------------------------ HTTP
    @staticmethod
    def _client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0), headers={"User-Agent": "Sentient"})

    async def open(self) -> None:
        if self.http is None:
            self.http = self._client()

    async def close(self) -> None:
        if self.http is not None:
            http, self.http = self.http, None
            await http.aclose()

    async def call(
        self,
        method: str,
        *,
        token: str | None = None,
        files: dict | None = None,
        http_timeout: float | None = None,
        **params: Any,
    ) -> Any:
        token = token or self.token
        if not token:
            raise ChannelError("Telegram isn't connected.", 409)
        params = {k: v for k, v in params.items() if v is not None}
        url = f"{API}/bot{token}/{method}"
        http = self.http or self._client()
        try:
            if files:
                data = {k: str(v) for k, v in params.items()}
                r = await http.post(url, data=data, files=files, timeout=http_timeout or 120.0)
            else:
                r = await http.post(url, json=params, timeout=http_timeout or 30.0)
        except httpx.HTTPError as exc:
            raise TelegramError(0, f"can't reach Telegram ({type(exc).__name__})") from None
        finally:
            if http is not self.http:
                await http.aclose()
        try:
            data = r.json()
        except ValueError:
            raise TelegramError(r.status_code, f"unexpected response (HTTP {r.status_code})") from None
        if not isinstance(data, dict) or not data.get("ok"):
            data = data if isinstance(data, dict) else {}
            retry = (data.get("parameters") or {}).get("retry_after")
            raise TelegramError(int(data.get("error_code") or r.status_code), str(data.get("description") or ""), retry)
        return data.get("result")

    async def send_call(self, method: str, **params: Any) -> Any:
        """``call`` with flood-control retries (HTTP 429 retry_after)."""
        for attempt in range(3):
            try:
                return await self.call(method, **params)
            except TelegramError as exc:
                if exc.code == 429 and attempt < 2 and (exc.retry_after or 1) <= 30:
                    await self.sleep(float(exc.retry_after or 1))
                    continue
                raise
        return None  # pragma: no cover

    # ------------------------------------------------------------------ connect
    async def validate(self, fields: dict[str, Any]) -> tuple[str, str]:
        token = str(fields.get("bot_token") or fields.get("token") or "").strip()
        if not TOKEN_RE.match(token):
            raise ChannelError(
                "That doesn't look like a Telegram bot token. It looks like 123456789:AAF... and comes from @BotFather."
            )
        try:
            me = await self.call("getMe", token=token)
        except TelegramError as exc:
            if exc.code in (401, 404):
                raise ChannelError(
                    "Telegram didn't accept that token. Copy it again from @BotFather, or send /token there to see it."
                ) from None
            raise ChannelError(f"Couldn't check the token with Telegram: {exc.description}") from None
        if not isinstance(me, dict) or not me.get("is_bot"):
            raise ChannelError("That token doesn't belong to a Telegram bot.")
        return token, f"@{me.get('username')}"

    async def check(self) -> None:
        await self.call("getMe")

    # ------------------------------------------------------------------ receive loop
    async def run(self) -> None:
        state = await self.service.store.state(self.id)
        cursor = state.get("cursor")
        self.offset = int(cursor) if cursor and str(cursor).lstrip("-").isdigit() else None
        backoff = 1.0
        failures = 0
        ready = False
        while True:
            try:
                if not ready:
                    me = await self.call("getMe")
                    self.bot_username = me.get("username")
                    await self.service.set_status(self.id, "connected", error=None, account_label=f"@{self.bot_username}")
                    ready = True
                started = asyncio.get_running_loop().time()
                updates = await self.call(
                    "getUpdates",
                    offset=self.offset,
                    timeout=self.poll_timeout,
                    allowed_updates=["message", "callback_query"],
                    http_timeout=self.poll_timeout + 15,
                )
                if failures:
                    failures, backoff = 0, 1.0
                    await self.service.set_status(self.id, "connected", error=None)
                for update in updates or []:
                    self.offset = int(update["update_id"]) + 1
                    self.dispatch(update)
                if updates:
                    await self.service.store.set_state(self.id, cursor=str(self.offset))
                elif asyncio.get_running_loop().time() - started < 1.0:
                    # a long poll that returns empty at once (proxy, mock): don't spin
                    await self.sleep(1.0)
                await asyncio.sleep(0)  # always let dispatched updates and other work run
                continue
            except asyncio.CancelledError:
                raise
            except TelegramError as exc:
                if exc.code == 401:
                    await self.service.set_status(
                        self.id, "error",
                        error="Telegram rejected the bot token (it may have been revoked in @BotFather). "
                        "Connect again with a new token.",
                    )
                    return
                failures += 1
                wait = float(exc.retry_after or backoff)
                if exc.code == 409 and "webhook" in exc.description.lower():
                    try:
                        await self.call("deleteWebhook")
                        continue
                    except Exception:
                        pass
                if exc.code == 409:
                    error = ("Another program is reading this bot's messages. Close other copies of Sentient "
                             "or other apps that use this bot token.")
                elif exc.code == 0:
                    error = "Can't reach Telegram. Check your internet connection."
                else:
                    error = f"Telegram error: {exc.description}"
            except Exception as exc:
                failures += 1
                wait = backoff
                error = f"Receiving Telegram messages failed: {self.redact(str(exc))}"
                log.warning("telegram polling failed: %s", self.redact(repr(exc)))
            if failures >= 3:
                await self.service.set_status(self.id, "error", error=error)
            await self.sleep(wait)
            backoff = min(backoff * 2, 60.0)

    def dispatch(self, update: dict) -> None:
        if isinstance(update.get("callback_query"), dict):
            self.spawn(self._on_callback(update["callback_query"]))
            return
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        if chat.get("type") != "private" or (message.get("from") or {}).get("is_bot"):
            return  # direct chats only: a bot in a group would see other people's messages
        self.spawn(self.handle_incoming(self.to_incoming(message)))

    def to_incoming(self, m: dict) -> Incoming:
        sender = m.get("from") or {}
        label = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x)
        if sender.get("username"):
            label = f"{label} (@{sender['username']})".strip()
        text = m.get("text") or m.get("caption") or ""
        reply = m.get("reply_to_message")
        audio: FileFetcher | None = None
        files: list[FileFetcher] = []
        voice = m.get("voice") or m.get("audio")
        if isinstance(voice, dict):
            ext = ".ogg" if m.get("voice") else (
                Path(voice.get("file_name") or "").suffix or mimetypes.guess_extension(voice.get("mime_type") or "") or ".mp3"
            )
            name = voice.get("file_name") or f"telegram-voice-{voice.get('file_unique_id', 'note')}{ext}"
            audio = self._fetcher(voice, name)
        photos = m.get("photo")
        if isinstance(photos, list) and photos:
            best = max(photos, key=lambda p: (p.get("file_size") or 0, p.get("width") or 0))
            files.append(self._fetcher(best, f"telegram-photo-{best.get('file_unique_id', 'image')}.jpg"))
        doc = m.get("document")
        if isinstance(doc, dict):
            ext = mimetypes.guess_extension(doc.get("mime_type") or "") or ""
            files.append(self._fetcher(doc, doc.get("file_name") or f"telegram-file-{doc.get('file_unique_id', 'file')}{ext}"))
        return Incoming(
            chat_id=str((m.get("chat") or {}).get("id")),
            label=label or str((m.get("chat") or {}).get("id")),
            text=text,
            audio=audio,
            files=files,
            unsupported=not (text or audio or files),
            reply_to=str(reply["message_id"]) if isinstance(reply, dict) and reply.get("message_id") is not None else None,
        )

    def _fetcher(self, obj: dict, name: str) -> FileFetcher:
        async def fetch() -> tuple[bytes, str]:
            if (obj.get("file_size") or 0) > MAX_DOWNLOAD_BYTES:
                raise ChannelError("That file is larger than 20 MB, the most Telegram lets bots download.")
            info = await self.call("getFile", file_id=obj.get("file_id"))
            path = (info or {}).get("file_path")
            if not path:
                raise ChannelError("Telegram didn't give me that file. Please send it again.")
            http = self.http or self._client()
            try:
                r = await http.get(f"{API}/file/bot{self.token}/{path}", timeout=120.0)
            except httpx.HTTPError:
                raise ChannelError("I couldn't download that file from Telegram. Please try again.") from None
            finally:
                if http is not self.http:
                    await http.aclose()
            if r.status_code != 200:
                raise ChannelError("I couldn't download that file from Telegram. Please try again.")
            return r.content, name

        return fetch

    async def _on_callback(self, cq: dict) -> None:
        message = cq.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id") or "")
        message_id = str(message.get("message_id") or "")
        toast = "This button has expired."
        try:
            if chat_id and message_id:
                toast = await self.handle_button(chat_id, message_id, str(cq.get("data") or ""))
        finally:
            try:
                await self.send_call("answerCallbackQuery", callback_query_id=cq.get("id"), text=toast[:190])
            except Exception as exc:
                log.debug("answerCallbackQuery failed: %s", self.redact(str(exc)))

    # ------------------------------------------------------------------ sending
    def render(self, md: str) -> list[str]:
        return render_telegram_chunks(md, TELEGRAM_LIMIT)

    @staticmethod
    def _keyboard(buttons: list[list[Button]]) -> dict:
        return {"inline_keyboard": [[{"text": b.label, "callback_data": b.data} for b in row] for row in buttons]}

    async def _with_plain_fallback(self, method: str, params: dict) -> Any:
        try:
            return await self.send_call(method, **params)
        except TelegramError as exc:
            if exc.code == 400 and "parse" in exc.description.lower():
                params = {**params, "text": html_to_plain(params["text"])[:TELEGRAM_LIMIT]}
                params.pop("parse_mode", None)
                return await self.send_call(method, **params)
            raise

    async def send_chunk(self, chat_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> str:
        params: dict[str, Any] = {
            "chat_id": _chat(chat_id), "text": chunk, "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if buttons:
            params["reply_markup"] = self._keyboard(buttons)
        result = await self._with_plain_fallback("sendMessage", params)
        return str(result["message_id"])

    async def edit_chunk(self, chat_id: str, message_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> None:
        params: dict[str, Any] = {
            "chat_id": _chat(chat_id), "message_id": int(message_id), "text": chunk, "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if buttons:
            params["reply_markup"] = self._keyboard(buttons)
        try:
            await self._with_plain_fallback("editMessageText", params)
        except TelegramError as exc:
            if "not modified" not in exc.description.lower():
                raise

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        await self.send_call("deleteMessage", chat_id=_chat(chat_id), message_id=int(message_id))

    async def clear_buttons(self, chat_id: str, message_id: str) -> None:
        try:
            await self.send_call("editMessageReplyMarkup", chat_id=_chat(chat_id), message_id=int(message_id))
        except TelegramError as exc:
            if "not modified" not in exc.description.lower():
                raise

    async def send_typing(self, chat_id: str) -> None:
        await self.call("sendChatAction", chat_id=_chat(chat_id), action="typing")

    async def send_audio(self, chat_id: str, data: bytes, filename: str) -> None:
        try:
            ogg = await asyncio.to_thread(_wav_to_ogg_opus, data)
        except Exception as exc:
            log.info("%s: voice note conversion failed: %s", self.id, exc)
            await self.call("sendAudio", chat_id=_chat(chat_id), files={"audio": (filename, data, "audio/wav")})
            return
        await self.call("sendVoice", chat_id=_chat(chat_id), files={"voice": ("reply.ogg", ogg, "audio/ogg")})

    def pairing_instructions(self, code: str, account_label: str | None) -> str:
        bot = account_label or "your bot"
        return f"Open {bot} in Telegram, tap Start, and send: /pair {code}"
