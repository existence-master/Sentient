"""WhatsApp Web bridge on neonize (Python bindings for whatsmeow), see ADR 0020.

Runs inside the engine: neonize ships whatsmeow as a shared library in its wheel, so users install
nothing else. Only this module imports neonize, lazily, so the engine starts without it. The linked
session (keys, not messages) is whatsmeow's own SQLite file under ``~/.sentient/whatsapp``.

Kept thin on purpose: everything here is translation between neonize objects and ``WAMessage``.
The channel's tests use a fake bridge; tests/channels/test_whatsapp_web.py checks the message
translation when neonize is installed. Nothing here is exercised against a live WhatsApp in CI.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any

from sentient.channels.whatsapp import EventSink, WAMedia, WAMessage

log = logging.getLogger(__name__)

_WRAPPERS = ("ephemeralMessage", "viewOnceMessage", "viewOnceMessageV2", "viewOnceMessageV2Extension",
             "documentWithCaptionMessage")
_MEDIA = ("audioMessage", "imageMessage", "videoMessage", "documentMessage")


def available() -> bool:
    return importlib.util.find_spec("neonize") is not None


def _ensure_magic() -> None:
    """neonize imports python-magic, which needs the libmagic system library (missing on most Macs).
    Sentient always passes the file type itself, so a stand-in is enough when libmagic is absent."""
    try:
        import magic  # noqa: F401
    except ImportError:
        stub = types.ModuleType("magic")
        stub.from_buffer = lambda _buffer, mime=False: "application/octet-stream"  # type: ignore[attr-defined]
        sys.modules["magic"] = stub


def _jid_text(jid: Any) -> str:
    return f"{jid.User}@{jid.Server}" if jid.User else str(jid.Server)


def _jid(text: str) -> Any:
    _ensure_magic()
    from neonize.utils.jid import build_jid

    user, _, server = text.partition("@")
    return build_jid(user, server or "s.whatsapp.net")


def _unwrap(msg: Any) -> Any:
    for _ in range(3):
        for name in _WRAPPERS:
            if msg.HasField(name):
                msg = getattr(msg, name).message
                break
        else:
            return msg
    return msg


def to_message(event: Any) -> WAMessage | None:
    """A neonize ``Message`` event as a ``WAMessage`` (None for edits, reactions and protocol messages)."""
    info, src = event.Info, event.Info.MessageSource
    msg = _unwrap(event.Message)
    if event.IsEdit or msg.HasField("protocolMessage") or msg.HasField("reactionMessage"):
        return None
    text, reply_to, media = "", None, []
    if msg.conversation:
        text = msg.conversation
    elif msg.HasField("extendedTextMessage"):
        text = msg.extendedTextMessage.text
        reply_to = msg.extendedTextMessage.contextInfo.stanzaID or None
    for name in _MEDIA:
        if not msg.HasField(name):
            continue
        part = getattr(msg, name)
        reply_to = reply_to or part.contextInfo.stanzaID or None
        text = text or getattr(part, "caption", "")
        ext = {"audioMessage": ".ogg", "imageMessage": ".jpg", "videoMessage": ".mp4"}.get(name, "")
        if name == "audioMessage":
            kind = "voice" if part.PTT else "audio"
        else:
            kind = {"imageMessage": "image", "videoMessage": "video"}.get(name, "document")
        filename = getattr(part, "fileName", "") or f"whatsapp-{kind}-{info.ID}{ext}"
        media.append(WAMedia(kind=kind, name=filename, mime=part.mimetype, size=int(part.fileLength), ref=msg))
    return WAMessage(
        id=info.ID, chat=_jid_text(src.Chat), sender=_jid_text(src.Sender), from_me=src.IsFromMe,
        push_name=info.Pushname, text=text, reply_to=reply_to, media=media,
        unsupported=not (text or media),
    )


class NeonizeBridge:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.client: Any = None
        self.connected = False
        self.me: Any = None
        self._done: asyncio.Event | None = None

    async def run(self, emit: EventSink) -> None:
        _ensure_magic()
        from neonize.aioze.client import NewAClient
        from neonize.aioze.events import (
            ConnectedEv,
            ConnectFailureEv,
            DisconnectedEv,
            LoggedOutEv,
            MessageEv,
            StreamReplacedEv,
            TemporaryBanEv,
        )
        from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import DeviceProps

        for name in ("neonize", "whatsmeow", "Whatsmeow"):  # chatty, and its debug lines carry phone numbers
            logging.getLogger(name).setLevel(logging.WARNING)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        done = self._done = asyncio.Event()
        client = self.client = NewAClient(
            str(self.session_dir / "session.sqlite3"),
            props=DeviceProps(os="Sentient", platformType=DeviceProps.DESKTOP),
        )

        async def on_qr(_client: Any, code: bytes) -> None:
            await emit("qr", {"code": code.decode() if isinstance(code, bytes) else str(code)})

        async def on_connected(_client: Any, _ev: Any) -> None:
            self.connected = True
            self.me = await client.get_me()
            lid = _jid_text(self.me.LID) if self.me.LID.User else ""
            await emit("connected", {"jid": _jid_text(self.me.JID), "lid": lid, "name": self.me.PushName})

        async def on_disconnected(_client: Any, _ev: Any) -> None:
            self.connected = False
            await emit("disconnected", {})

        async def on_logged_out(_client: Any, ev: Any) -> None:
            self.connected = False
            await emit("logged_out", {"reason": str(ev.Reason)})
            done.set()

        async def fatal(error: str) -> None:
            self.connected = False
            await emit("failed", {"error": error})
            done.set()

        async def on_replaced(_client: Any, _ev: Any) -> None:
            await fatal("Another copy of Sentient is using this WhatsApp link. Close it, then click Reconnect.")

        async def on_ban(_client: Any, _ev: Any) -> None:
            await fatal("WhatsApp has temporarily blocked linked devices on this account. Try again later.")

        async def on_failure(_client: Any, ev: Any) -> None:
            await fatal(f"WhatsApp refused the connection: {ev.Message or ev.Reason}")

        async def on_message(_client: Any, ev: Any) -> None:
            try:
                message = to_message(ev)
            except Exception:
                log.exception("could not read a WhatsApp message")
                return
            if message is not None:
                await emit("message", {"message": message})

        client.event.qr(on_qr)
        client.event(ConnectedEv)(on_connected)
        client.event(DisconnectedEv)(on_disconnected)
        client.event(LoggedOutEv)(on_logged_out)
        client.event(StreamReplacedEv)(on_replaced)
        client.event(TemporaryBanEv)(on_ban)
        client.event(ConnectFailureEv)(on_failure)
        client.event(MessageEv)(on_message)

        task = await client.connect()  # whatsmeow reconnects by itself while this runs
        waiter = asyncio.create_task(done.wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        finally:
            waiter.cancel()
            self.connected = False

    async def send_text(self, chat: str, text: str) -> str:
        return (await self.client.send_message(_jid(chat), text)).ID

    async def edit_text(self, chat: str, message_id: str, text: str) -> None:
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import Message

        await self.client.edit_message(_jid(chat), message_id, Message(conversation=text))

    async def revoke(self, chat: str, message_id: str) -> None:
        await self.client.revoke_message(_jid(chat), self.me.JID, message_id)

    async def send_voice(self, chat: str, ogg: bytes, seconds: int) -> str:
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import AudioMessage, Message
        from neonize.utils.enum import MediaType

        up = await self.client.upload(ogg, MediaType.MediaAudio)
        message = Message(audioMessage=AudioMessage(
            URL=up.url, directPath=up.DirectPath, fileEncSHA256=up.FileEncSHA256, fileLength=up.FileLength,
            fileSHA256=up.FileSHA256, mediaKey=up.MediaKey, mimetype="audio/ogg; codecs=opus", PTT=True,
            seconds=seconds,
        ))
        return (await self.client.send_message(_jid(chat), message)).ID

    async def send_document(self, chat: str, data: bytes, filename: str, mime: str) -> str:
        return (await self.client.send_document(_jid(chat), data, filename=filename, mimetype=mime)).ID

    async def typing(self, chat: str) -> None:
        from neonize.utils.enum import ChatPresence, ChatPresenceMedia

        await self.client.send_chat_presence(
            _jid(chat), ChatPresence.CHAT_PRESENCE_COMPOSING, ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
        )

    async def download(self, media: WAMedia) -> bytes:
        return await self.client.download_any(media.ref)

    async def logout(self) -> None:
        await self.client.logout()

    async def close(self) -> None:
        client, self.client = self.client, None
        self.connected = False
        if self._done is not None:
            self._done.set()
        if client is not None:
            with contextlib.suppress(Exception):
                await client.stop()
