"""WhatsApp channel: Sentient as a linked device on the user's own WhatsApp (ADR 0020).

The user scans a QR code once (WhatsApp, Linked devices) and then talks to Sentient in their own
"Message yourself" chat. No business account, no webhook, no server. Other chats are ignored unless
the user pairs one with a code. The WhatsApp Web protocol lives behind ``WhatsAppBridge``
(``whatsapp_web.NeonizeBridge`` in the app, a fake in tests); this module only sees plain messages.

WhatsApp has no buttons for personal accounts, so options are numbered: "Reply 1 to allow". A reply
(or, for an approval that is holding up a reply, a bare number) picks the option.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import re
import shutil
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sentient import paths
from sentient.channels.base import (
    MAX_DOWNLOAD_BYTES,
    Button,
    Channel,
    ChannelError,
    FileFetcher,
    Incoming,
)
from sentient.channels.formatting import WHATSAPP_LIMIT, render_whatsapp_chunks

log = logging.getLogger(__name__)

PAIR_RE = re.compile(r"^/pair\s+[0-9]{6}$", re.I)
CHOICE_RE = re.compile(r"^\s*([0-9]{1,2})\s*[.)]?\s*$")  # ASCII digits only: "\d" also matches "٢" or "²"
IGNORED_SERVERS = {"g.us", "broadcast", "newsletter"}  # groups, status updates, channels
MAX_CHOICES = 200

INSTRUCTIONS = """\
**What you need:** WhatsApp on your phone. Your computer must be on with Sentient running for it to answer.

Sentient joins your WhatsApp as a linked device, the same way WhatsApp on a computer does. You talk to it in
your own **Message yourself** chat. Nobody else can talk to Sentient, and it never answers your other chats.

1. Click **Show QR code** below.
2. On your phone, open WhatsApp and go to **Linked devices**: on Android tap the three dots at the top right,
   on iPhone open **Settings**.
3. Tap **Link a device** and point your phone at the code.
4. Open your chat with yourself (search for your own name, or "Message yourself") and say hi.

You can send text, voice notes, photos and files. When Sentient needs a yes or no, it lists numbered
options: reply with the number. Commands: `/new` starts a fresh chat, `/stop` stops a reply, `/stopall`
stops everything Sentient is doing (`/resume` starts it again), `/model` switches between local and cloud
models, `/help` shows help.

**Good to know:** this uses the same linking as WhatsApp Web, through an unofficial app. WhatsApp does not
officially support assistants on personal accounts and could limit or ban an account that uses one. That is
rare for normal personal use, but it can happen, so use a number you can afford to have restricted if you
are worried. Messages go through WhatsApp's servers like any other WhatsApp message. To stop, click
**Disconnect** here, or remove Sentient under **Linked devices** on your phone.
"""


# ---------------------------------------------------------------------------- bridge contract
@dataclass
class WAMedia:
    kind: str  # "voice" | "audio" | "image" | "video" | "document"
    name: str
    mime: str = ""
    size: int = 0
    ref: Any = None  # whatever the bridge needs to download it


@dataclass
class WAMessage:
    id: str
    chat: str  # "<user>@<server>" without a device part
    sender: str = ""
    from_me: bool = False
    push_name: str = ""
    text: str = ""
    reply_to: str | None = None  # id of the quoted message
    media: list[WAMedia] = field(default_factory=list)
    unsupported: bool = False


EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


class WhatsAppBridge(Protocol):
    """One WhatsApp Web connection. ``run`` emits events until the connection ends:

    ``qr`` {code}, ``connected`` {jid, lid, name}, ``disconnected`` {}, ``logged_out`` {reason},
    ``failed`` {error} (fatal: do not reconnect), ``message`` {message: WAMessage}.
    """

    connected: bool

    async def run(self, emit: EventSink) -> None: ...
    async def send_text(self, chat: str, text: str) -> str: ...
    async def edit_text(self, chat: str, message_id: str, text: str) -> None: ...
    async def revoke(self, chat: str, message_id: str) -> None: ...
    async def send_voice(self, chat: str, ogg: bytes, seconds: int) -> str: ...
    async def send_document(self, chat: str, data: bytes, filename: str, mime: str) -> str: ...
    async def typing(self, chat: str) -> None: ...
    async def download(self, media: WAMedia) -> bytes: ...
    async def logout(self) -> None: ...
    async def close(self) -> None: ...


def default_bridge(session_dir: Path) -> WhatsAppBridge:
    from sentient.channels.whatsapp_web import NeonizeBridge

    return NeonizeBridge(session_dir)


def bridge_available() -> bool:
    from sentient.channels.whatsapp_web import available

    return available()


def _user(jid: str) -> str:
    return jid.split("@", 1)[0].split(":", 1)[0]


def _server(jid: str) -> str:
    return jid.split("@", 1)[1] if "@" in jid else ""


def _wav_seconds(data: bytes) -> int:
    with contextlib.suppress(Exception), wave.open(io.BytesIO(data)) as w:
        return max(1, round(w.getnframes() / float(w.getframerate() or 1)))
    return 1


class WhatsAppChannel(Channel):
    id = "whatsapp"
    display_name = "WhatsApp"
    secret_name = ""  # no token: the linked session lives under ~/.sentient/whatsapp
    message_limit = WHATSAPP_LIMIT
    setup_fields: list[dict] = []
    instructions_md = INSTRUCTIONS
    uses_token = False
    status_lines = False  # WhatsApp leaves "This message was deleted" behind
    choice_hint = "reply with an option's number"

    def __init__(self, service):
        super().__init__(service)
        self.bridge_factory: Callable[[Path], WhatsAppBridge] = default_bridge
        self.available: Callable[[], bool] = bridge_available
        self.bridge: WhatsAppBridge | None = None
        self.linked = False
        self.self_ids: set[str] = set()  # the user's phone number and LID (user parts)
        self.self_chat: str | None = None  # "<number>@s.whatsapp.net"
        self._sent: list[str] = []
        self._choices: dict[tuple[str, str], list[str]] = {}  # (chat, message) -> button data, in order
        self._fresh_link = False  # a QR code was shown: pair the self chat once it is scanned
        self._up = False  # this connection reached "connected"
        self._outcome: tuple[str, str] | None = None  # set when the connection must not be retried

    # ------------------------------------------------------------------ session
    @staticmethod
    def session_dir() -> Path:
        return paths.home() / "whatsapp"

    @property
    def ready(self) -> bool:
        return self.linked

    async def restore(self) -> bool:
        """At startup: linked before and the session is still on disk."""
        state = await self.service.store.state(self.id)
        self.linked = bool(state.get("account_label")) and self.session_dir().exists()
        return self.linked

    async def begin_link(self) -> None:
        if not self.available():
            raise ChannelError(
                "WhatsApp support isn't installed in this copy of Sentient. "
                "Install Sentient with the 'whatsapp' extra, or use the installer."
            )
        await self.stop_runtime()
        self.qr = None

    async def unlink(self) -> None:
        """Log out (removes Sentient from Linked devices on the phone) and delete the saved session."""
        bridge = self.bridge
        if bridge is not None and bridge.connected:
            try:
                await asyncio.wait_for(bridge.logout(), 10)
            except Exception as exc:
                log.info("whatsapp logout failed: %s", exc)
        await self.stop_runtime()
        self.forget_session()

    def forget_session(self) -> None:
        self.linked = False
        self.qr = None
        self.self_ids.clear()
        self.self_chat = None
        shutil.rmtree(self.session_dir(), ignore_errors=True)

    async def check(self) -> None:
        if self.bridge is None or not self.bridge.connected:
            raise ChannelError("WhatsApp isn't connected right now. Check that your phone is online.", 409)

    def pairing_instructions(self, code: str, account_label: str | None) -> str:
        number = account_label or "your number"
        return f"From the other WhatsApp chat, send this to {number}: /pair {code}"

    # ------------------------------------------------------------------ connection loop
    async def close(self) -> None:
        if self.bridge is not None:
            bridge, self.bridge = self.bridge, None
            with contextlib.suppress(Exception):
                await bridge.close()

    async def run(self) -> None:
        backoff = 1.0
        failures = 0
        while True:
            self._outcome, self._up = None, False
            error = "Can't reach WhatsApp. Check your internet connection."
            crashed = False
            try:  # the bridge boundary: nothing the WhatsApp library raises may stop the engine
                self.bridge = self.bridge_factory(self.session_dir())
                await self.bridge.run(self.on_event)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("whatsapp bridge failed")
                crashed = True
                error = ("WhatsApp stopped working unexpectedly. Sentient keeps trying to reconnect; "
                         "if this stays, click Reconnect.")
            finally:
                await self.close()
            if self._outcome is not None:
                if not self.linked:  # logged out: the keys are useless now (deleted once the bridge let go of them)
                    self.forget_session()
                status, message = self._outcome
                await self.service.set_status(self.id, status, error=message)
                return
            if not self.linked:  # never retry linking on its own: each try asks WhatsApp for new codes
                if self._fresh_link:
                    error = "The code expired before it was scanned. Click Reconnect to get a new one."
                self.qr, self._fresh_link = None, False
                await self.service.set_status(self.id, "error", error=error)
                return
            if self._up:
                failures, backoff = 0, 1.0
            failures += 1
            attention = crashed or failures >= 3
            await self.service.set_status(self.id, "error" if attention else "connecting",
                                          error=error if attention else None)
            await self.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def on_event(self, kind: str, data: dict[str, Any]) -> None:
        try:
            if kind == "message":
                self.on_message(data["message"])
            elif kind == "qr":
                self.qr = str(data.get("code") or "") or None
                self._fresh_link = True
                await self.service.store.set_state(self.id, status="linking", error=None)
                await self.service.publish_channel(self.id)
            elif kind == "connected":
                await self._on_connected(data)
            elif kind == "disconnected":
                await self.service.set_status(self.id, "connecting", error=None)
            elif kind == "logged_out":
                self.linked = False
                await self.service.store.set_state(self.id, account_label=None)
                self._outcome = ("error", "WhatsApp unlinked Sentient (it was removed from Linked devices, or the "
                                          "link expired). Click Reconnect and scan the new code.")
            elif kind == "failed":
                self._outcome = ("error", str(data.get("error") or "WhatsApp stopped the connection."))
        except Exception:
            log.exception("whatsapp event %s failed", kind)

    async def _on_connected(self, data: dict[str, Any]) -> None:
        jid, lid = str(data.get("jid") or ""), str(data.get("lid") or "")
        if not jid:
            return
        self.self_ids = {u for u in (_user(jid), _user(lid)) if u}
        self.self_chat = f"{_user(jid)}@s.whatsapp.net"
        self.linked, self._up = True, True
        self.qr = None
        await self.service.set_status(self.id, "connected", error=None, account_label=f"+{_user(jid)}")
        if self._fresh_link:
            self._fresh_link = False
            await self._pair_self_chat()
        await self.service.publish_channel(self.id)

    async def _pair_self_chat(self) -> None:
        assert self.self_chat is not None
        store = self.service.store
        if await store.chat(self.id, self.self_chat) is not None:
            return
        session_id = await self.app.store.create_session(channel=self.id)
        chat = await store.add_chat(self.id, self.self_chat, "Message yourself", deliver=self.cfg.deliver_default,
                                    session_id=session_id)
        name = self.app.config.assistant.name
        await self.reply(self.self_chat, f"Linked! This chat is where you talk to {name}.\n\n" + self.help_text(chat))

    # ------------------------------------------------------------------ inbound
    def chat_id_for(self, m: WAMessage) -> str | None:
        """The chat Sentient answers in, or None when the message is not for Sentient."""
        if m.id in self._sent or _server(m.chat) in IGNORED_SERVERS or not m.chat:
            return None
        if _user(m.chat) in self.self_ids:
            return self.self_chat if m.from_me else None
        if m.from_me:
            return None  # the user writing to someone else
        return f"{_user(m.chat)}@{_server(m.chat)}"

    def on_message(self, m: WAMessage) -> None:
        chat_id = self.chat_id_for(m)
        if chat_id is None:
            return
        self.spawn(self._route(chat_id, m))

    async def _route(self, chat_id: str, m: WAMessage) -> None:
        if (chat_id != self.self_chat and not PAIR_RE.match(m.text.strip())
                and await self.service.store.chat(self.id, chat_id) is None):
            return  # someone else writing to the user: never Sentient's business, never answered
        if not m.media and await self._pick_choice(chat_id, m):
            return
        await self.handle_incoming(self.to_incoming(chat_id, m))

    def to_incoming(self, chat_id: str, m: WAMessage) -> Incoming:
        audio: FileFetcher | None = None
        files: list[FileFetcher] = []
        for media in m.media:
            if media.kind in {"voice", "audio"} and audio is None:
                audio = self._fetcher(media)
            else:
                files.append(self._fetcher(media))
        label = "Message yourself" if chat_id == self.self_chat else (m.push_name or f"+{_user(chat_id)}")
        return Incoming(chat_id=chat_id, label=label, text=m.text, audio=audio, files=files,
                        unsupported=m.unsupported and not (m.text or m.media), reply_to=m.reply_to)

    def _fetcher(self, media: WAMedia) -> FileFetcher:
        async def fetch() -> tuple[bytes, str]:
            if media.size > MAX_DOWNLOAD_BYTES:
                raise ChannelError("That file is larger than 20 MB, which is more than I can take.")
            if self.bridge is None:
                raise ChannelError("WhatsApp isn't connected right now. Please send it again in a moment.")
            try:
                data = await self.bridge.download(media)
            except Exception as exc:
                log.warning("whatsapp download failed: %s", exc)
                raise ChannelError("I couldn't download that from WhatsApp. Please try again.") from None
            return data, media.name

        return fetch

    # ------------------------------------------------------------------ numbered options
    async def _pick_choice(self, chat_id: str, m: WAMessage) -> bool:
        """A number that answers a message with options. Returns True when handled."""
        match = CHOICE_RE.match(m.text or "")
        if match is None:
            return False
        if m.reply_to:
            message_id = m.reply_to if (chat_id, m.reply_to) in self._choices else None
        else:  # a bare number only answers an approval that is holding up the reply
            pending = [mid for (c, mid), data in self._choices.items() if c == chat_id and data[0].startswith("ap:")]
            message_id = pending[-1] if pending else None
        if message_id is None:
            return False
        options = self._choices[(chat_id, message_id)]
        number = int(match.group(1))
        if not 1 <= number <= len(options):
            await self.reply(chat_id, f"Reply with a number from 1 to {len(options)}.")
            return True
        outcome = await self.handle_button(chat_id, message_id, options[number - 1])
        await self.reply(chat_id, f"_{outcome.rstrip('.')}._")
        return True

    @staticmethod
    def _options_text(buttons: list[list[Button]]) -> str:
        labels = [b.label for row in buttons for b in row]
        approval = buttons[0][0].data.startswith("ap:") if labels else False
        lead = "Reply with a number:" if approval else "Reply to this message with a number:"
        return "\n\n" + lead + "\n" + "\n".join(f"*{i}* {label}" for i, label in enumerate(labels, 1))

    def _remember_choices(self, chat_id: str, message_id: str, buttons: list[list[Button]]) -> None:
        self._choices[(str(chat_id), str(message_id))] = [b.data for row in buttons for b in row]
        while len(self._choices) > MAX_CHOICES:
            self._choices.pop(next(iter(self._choices)))

    async def settle_buttons(self, chat_id: str, message_id: str, outcome: str) -> None:
        self._choices.pop((str(chat_id), str(message_id)), None)
        await super().settle_buttons(chat_id, message_id, outcome)

    # ------------------------------------------------------------------ sending
    def render(self, md: str) -> list[str]:
        return render_whatsapp_chunks(md, WHATSAPP_LIMIT)

    def _bridge(self) -> WhatsAppBridge:
        if self.bridge is None:
            raise ChannelError("WhatsApp isn't connected right now.", 409)
        return self.bridge

    async def send_markdown(self, chat_id: str, md: str, buttons: list[list[Button]] | None = None) -> list[str]:
        if buttons:
            md = md.replace("Tap an option, or reply", "Reply with an option's number, or reply")
        return await super().send_markdown(chat_id, md, buttons)

    def _signed(self, chat_id: str, text: str) -> str:
        """In the self chat every message shows as the user's own, so Sentient's start with its name."""
        if chat_id != self.self_chat:
            return text
        return f"*{self.app.config.assistant.name}:* {text}"

    async def send_chunk(self, chat_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> str:
        text = self._signed(chat_id, chunk + (self._options_text(buttons) if buttons else ""))
        message_id = self._track(await self._bridge().send_text(chat_id, text))
        if buttons:
            self._remember_choices(chat_id, message_id, buttons)
        return message_id

    def _track(self, message_id: str) -> str:
        """Remember what Sentient sent, so its own messages are never read back as the user's."""
        self._sent = [*self._sent[-499:], message_id]
        return message_id

    async def edit_chunk(self, chat_id: str, message_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> None:
        await self._bridge().edit_text(chat_id, message_id, self._signed(chat_id, chunk + (self._options_text(buttons) if buttons else "")))

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        await self._bridge().revoke(chat_id, message_id)

    async def clear_buttons(self, chat_id: str, message_id: str) -> None:
        self._choices.pop((str(chat_id), str(message_id)), None)  # the text stays; its numbers no longer count

    async def send_typing(self, chat_id: str) -> None:
        await self._bridge().typing(chat_id)

    async def send_audio(self, chat_id: str, data: bytes, filename: str) -> None:
        from sentient.channels.telegram import _wav_to_ogg_opus

        bridge = self._bridge()
        try:
            ogg = await asyncio.to_thread(_wav_to_ogg_opus, data)
        except Exception as exc:
            log.info("whatsapp: voice note conversion failed: %s", exc)
            self._track(await bridge.send_document(chat_id, data, filename, "audio/wav"))
            return
        self._track(await bridge.send_voice(chat_id, ogg, _wav_seconds(data)))
