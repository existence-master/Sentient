"""Channel base class: everything a messaging app does that is not transport.

A concrete channel (Telegram, Discord, WhatsApp) implements a handful of transport primitives
(``validate``, ``run``, ``render``, ``send_chunk``, ``edit_chunk``, ``delete_message``,
``send_typing``, ``clear_buttons``, ``send_audio``) and turns inbound updates into
``Incoming`` objects. This class handles pairing, commands, running chat turns with
progressive message edits, tool activity, approvals, voice notes, attachments,
steering and button callbacks, identically for every app.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentient import paths
from sentient.channels.store import PairingResult
from sentient.llm import presets as model_presets
from sentient.llm.events import (
    ApprovalRequest,
    Done,
    Error,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:  # pragma: no cover
    from sentient.channels.service import ChannelService

log = logging.getLogger(__name__)

FileFetcher = Callable[[], Awaitable[tuple[bytes, str]]]
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
TYPING_EVERY_S = 4.5

ACTIVITY = {
    "internet_search": "Searching the web",
    "web": "Reading a web page",
    "weather": "Checking the weather",
    "maps": "Looking at the map",
    "news": "Checking the news",
    "charts": "Making a chart",
    "files": "Working with files",
    "memory": "Checking my memory",
    "skills": "Looking through my skills",
    "time": "Checking the time",
    "tasks": "Working on your tasks",
    "gmail": "Checking Gmail",
    "gcalendar": "Checking your calendar",
    "gdrive": "Looking in Google Drive",
    "browser": "Using the browser",
    "sandbox": "Running some code",
    "subagents": "Handing work to a helper",
    "github": "Checking GitHub",
    "notion": "Checking Notion",
    "slack": "Checking Slack",
    "discord": "Checking Discord",
}

APPROVAL_DECISIONS = {"a": "allow", "s": "allow_session", "d": "deny"}
DECISION_LABEL = {"allow": "Allowed", "allow_session": "Allowed for this chat", "deny": "Denied"}
_COMMAND_RE = re.compile(r"^/([A-Za-z_]+)(?:@\w+)?(?:\s+(.*))?$", re.S)


class ChannelError(Exception):
    """A user-facing problem (bad token, not connected...). ``status`` is the HTTP status for routes."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class Button:
    label: str
    data: str
    style: str = "primary"  # primary | success | danger | secondary


@dataclass
class Incoming:
    chat_id: str
    label: str
    text: str = ""
    audio: FileFetcher | None = None
    files: list[FileFetcher] = field(default_factory=list)
    unsupported: bool = False
    reply_to: str | None = None  # id of the message this one replies to (Telegram reply, Discord reference)


@dataclass
class ChatRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn: asyncio.Task | None = None
    queued: list[tuple[str, list[str]]] = field(default_factory=list)
    pair_failures: list[float] = field(default_factory=list)
    stopped: bool = False

    @property
    def running(self) -> bool:
        return self.turn is not None and not self.turn.done()


async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def parse_command(text: str) -> tuple[str | None, str]:
    m = _COMMAND_RE.match(text.strip())
    if not m:
        return None, text
    return m.group(1).lower(), (m.group(2) or "").strip()


def humanize_tool(name: str) -> str:
    return name.replace("_", " ").strip()


def safe_filename(name: str) -> str:
    base = Path(name or "file").name
    base = re.sub(r"[^\w.\- ()]+", "_", base).strip() or "file"
    return base[:160]


def save_upload(name: str, data: bytes) -> str:
    """Save bytes under files/uploads and return the Files API name (``uploads/<name>``)."""
    folder = paths.files_dir() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(name)
    stem, suffix = target.stem, target.suffix
    i = 1
    while target.exists():
        target = folder / f"{stem} ({i}){suffix}"
        i += 1
    target.write_bytes(data)
    return f"uploads/{target.name}"


class ReplyStream:
    """One segment of assistant text shown in one or more messages, edited as it grows."""

    def __init__(self, channel: Channel, chat_id: str, *, streaming: bool, interval: float):
        self.channel = channel
        self.chat_id = chat_id
        self.streaming = streaming
        self.interval = interval
        self.text = ""
        self.ids: list[str] = []
        self.shown: list[str] = []
        self._last = float("-inf")
        self._lock = asyncio.Lock()
        self._later: asyncio.Task | None = None

    async def add(self, delta: str) -> None:
        self.text += delta
        if not self.streaming or not self.text.strip():
            return
        wait = self.interval - (self.channel.clock() - self._last)
        if wait <= 0:
            if self._later is not None and not self._later.done():
                self._later.cancel()
            await self.flush(cursor=True)
        elif self._later is None or self._later.done():
            self._later = asyncio.create_task(self._flush_later(wait))

    async def _flush_later(self, wait: float) -> None:
        await asyncio.sleep(wait)
        await self.flush(cursor=True)

    async def flush(self, *, cursor: bool = False) -> None:
        async with self._lock:
            body = self.text.strip()
            if not body:
                return
            chunks = self.channel.render(body + (" …" if cursor else ""))
            for idx, chunk in enumerate(chunks):
                if idx < len(self.ids):
                    if self.shown[idx] != chunk:
                        await self.channel.edit_chunk(self.chat_id, self.ids[idx], chunk)
                        self.shown[idx] = chunk
                else:
                    mid = await self.channel.send_chunk(self.chat_id, chunk)
                    self.ids.append(mid)
                    self.shown.append(chunk)
            self._last = self.channel.clock()

    async def finish(self, replacement: str | None = None, suffix: str = "") -> str:
        if self._later is not None and not self._later.done():
            self._later.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._later
        if replacement is not None and not self.text.strip():
            self.text = replacement
        if suffix and self.text.strip():
            self.text = self.text.rstrip() + suffix
        await self.flush(cursor=False)
        return self.text.strip()


class StatusLine:
    """A short, temporary 'Searching the web...' message."""

    def __init__(self, channel: Channel, chat_id: str):
        self.channel = channel
        self.chat_id = chat_id
        self.message_id: str | None = None
        self.label = ""

    async def show(self, label: str) -> None:
        if label == self.label:
            return
        chunk = self.channel.render(f"_{label}..._")[0]
        try:
            if self.message_id is None:
                self.message_id = await self.channel.send_chunk(self.chat_id, chunk)
            else:
                await self.channel.edit_chunk(self.chat_id, self.message_id, chunk)
            self.label = label
        except Exception as exc:
            log.debug("status line failed: %s", exc)

    async def clear(self) -> None:
        if self.message_id is None:
            return
        mid, self.message_id, self.label = self.message_id, None, ""
        with contextlib.suppress(Exception):
            await self.channel.delete_message(self.chat_id, mid)


class Channel:
    id: str = ""
    display_name: str = ""
    secret_name: str = ""
    message_limit: int = 4000
    setup_fields: list[dict] = []
    instructions_md: str = ""
    uses_token = True  # False: the channel keeps its own session (WhatsApp) instead of a keychain token
    status_lines = True  # short "Searching the web..." messages that are deleted again
    choice_hint = "tap an option"  # how a person picks one of a message's options

    def __init__(self, service: ChannelService):
        self.service = service
        self.app = service.app
        self.token: str | None = None
        self.clock: Callable[[], float] = time.monotonic
        self.sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
        self._chats: dict[str, ChatRuntime] = {}
        self._tasks: set[asyncio.Task] = set()
        self._runtime: asyncio.Task | None = None
        self._button_text: dict[tuple[str, str], str] = {}
        self.qr: str | None = None  # a code to scan while linking (WhatsApp), else None

    # ------------------------------------------------------------------ config
    @property
    def cfg(self):
        return getattr(self.app.config.channels, self.id)

    @property
    def ready(self) -> bool:
        """Connected by the user and able to send (a token, or a linked session)."""
        return bool(self.token)

    # ------------------------------------------------------------------ transport (override)
    async def validate(self, fields: dict[str, Any]) -> tuple[str, str]:
        """Check credentials with the service. Returns ``(token, account_label)``; raises ChannelError."""
        raise NotImplementedError

    async def open(self) -> None:
        """Create clients for ``self.token``."""

    async def close(self) -> None:
        """Release clients."""

    async def run(self) -> None:
        """Receive updates until cancelled (or a fatal error)."""
        raise NotImplementedError

    def render(self, md: str) -> list[str]:
        raise NotImplementedError

    async def send_chunk(self, chat_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> str:
        raise NotImplementedError

    async def edit_chunk(self, chat_id: str, message_id: str, chunk: str, buttons: list[list[Button]] | None = None) -> None:
        raise NotImplementedError

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        raise NotImplementedError

    async def clear_buttons(self, chat_id: str, message_id: str) -> None:
        raise NotImplementedError

    async def send_typing(self, chat_id: str) -> None:
        return None

    async def send_audio(self, chat_id: str, data: bytes, filename: str) -> None:
        return None

    # ------------------------------------------------------------------ runtime lifecycle
    @property
    def running(self) -> bool:
        return self._runtime is not None and not self._runtime.done()

    def start_runtime(self) -> None:
        if self.running:
            return
        self._runtime = asyncio.create_task(self._run_runtime(), name=f"channels:{self.id}")

    async def _run_runtime(self) -> None:
        await self.open()
        try:
            await self.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("%s channel stopped", self.id)
            await self.service.set_status(self.id, "error", error=f"{self.display_name} stopped: {self.redact(str(exc))}")
        finally:
            await self.close()

    async def stop_runtime(self) -> None:
        if self._runtime is not None:
            self._runtime.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._runtime
            self._runtime = None
        for rt in self._chats.values():
            if rt.running:
                assert rt.turn is not None
                rt.turn.cancel()
        tasks = [t for t in self._tasks if not t.done()]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for rt in self._chats.values():
            if rt.turn is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await rt.turn
        self._chats.clear()
        await self.close()

    def spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            log.error("%s channel task failed: %s", self.id, self.redact(repr(exc)))

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """Wait for in-flight updates and turns (tests and graceful shutdown)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            pending = [t for t in self._tasks if not t.done()]
            pending += [rt.turn for rt in self._chats.values() if rt.turn is not None and not rt.turn.done()]
            if not pending:
                return
            await asyncio.wait(pending, timeout=max(0.01, deadline - loop.time()))

    def redact(self, text: str) -> str:
        if self.token:
            text = text.replace(self.token, "<token>")
        return re.sub(r"bot\d+:[A-Za-z0-9_-]{20,}", "bot<token>", text)

    # ------------------------------------------------------------------ sending helpers
    async def send_markdown(self, chat_id: str, md: str, buttons: list[list[Button]] | None = None) -> list[str]:
        chunks = self.render(md) or self.render("(empty)")
        ids: list[str] = []
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            ids.append(await self.send_chunk(chat_id, chunk, buttons if last else None))
        if buttons and ids:
            self._button_text[(str(chat_id), ids[-1])] = md
        return ids

    async def settle_buttons(self, chat_id: str, message_id: str, outcome: str) -> None:
        """Replace a message's buttons with the outcome ("Allowed", "Plan approved"...)."""
        md = self._button_text.pop((str(chat_id), str(message_id)), None)
        try:
            if md is not None:
                chunks = self.render(f"{md}\n\n**{outcome}**")
                await self.edit_chunk(chat_id, message_id, chunks[-1] if len(chunks) == 1 else self.render(f"**{outcome}**")[0])
            else:
                await self.clear_buttons(chat_id, message_id)
        except Exception as exc:
            log.debug("could not update button message: %s", self.redact(str(exc)))

    def publish_message(self, chat_id: str, session_id: str | None, direction: str, text: str) -> None:
        self.app.bus.publish(
            "channel.message",
            {"channel": self.id, "chat_id": str(chat_id), "session_id": session_id, "direction": direction, "text": text},
        )

    async def reply(self, chat_id: str, md: str) -> None:
        try:
            await self.send_markdown(chat_id, md)
        except Exception as exc:
            log.warning("%s: reply failed: %s", self.id, self.redact(str(exc)))

    # ------------------------------------------------------------------ inbound
    def runtime(self, chat_id: str) -> ChatRuntime:
        return self._chats.setdefault(str(chat_id), ChatRuntime())

    async def handle_incoming(self, msg: Incoming) -> None:
        rt = self.runtime(msg.chat_id)
        async with rt.lock:
            try:
                await self._handle_incoming(rt, msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("%s: handling a message failed", self.id)
                await self.reply(msg.chat_id, f"Sorry, something went wrong: {self.redact(str(exc))[:300]}")

    async def _handle_incoming(self, rt: ChatRuntime, msg: Incoming) -> None:
        store = self.service.store
        chat_id = str(msg.chat_id)
        cmd, arg = parse_command(msg.text) if msg.text.startswith("/") else (None, msg.text)
        chat = await store.chat(self.id, chat_id)
        if cmd == "pair":
            await self._pair(rt, msg, chat, arg)
            return
        if chat is None:
            if await store.refuse_once(self.id, chat_id):
                await self.reply(chat_id, self.refusal_text())
            return
        if cmd in {"start", "help"}:
            await self.reply(chat_id, self.help_text(chat))
            return
        if cmd == "new":
            await self.cancel_turn(chat_id)
            rt.queued.clear()
            session_id = await self.app.store.create_session(channel=self.id)
            await store.update_chat(self.id, chat_id, session_id=session_id)
            await self.service.publish_channel(self.id)
            await self.reply(chat_id, "Started a fresh chat. What's next?")
            return
        if cmd == "stopall" or (cmd == "stop" and arg.lower() == "all"):
            result = await self.app.stop_all(source=self.id)
            jobs = int(result.get("cancelled") or 0)
            done = f"Stopped everything ({jobs} running job{'s' if jobs != 1 else ''} cancelled)." if jobs else "Stopped everything."
            await self.reply(chat_id, f"{done} Nothing new will start until you resume. Send /resume or press Resume in Sentient.")
            return
        if cmd == "resume":
            if not self.app.stopped:
                await self.reply(chat_id, "Sentient isn't stopped. Everything is running as usual.")
                return
            await self.app.resume(source=self.id)
            await self.reply(chat_id, "Resumed. Scheduled tasks and suggestions are back on.")
            return
        if cmd == "model":
            await self._model_command(chat_id, arg)
            return
        if cmd == "stop":
            rt.queued.clear()
            if await self.cancel_turn(chat_id):
                await self.reply(chat_id, "Stopped.")
            else:
                await self.reply(chat_id, "Nothing is running right now.")
            return
        if cmd is not None:
            await self.reply(chat_id, "I don't know that command. Send /help to see what I can do.")
            return
        if msg.unsupported and not msg.text and not msg.files and msg.audio is None:
            await self.reply(chat_id, "I can read text, voice notes, photos and files. That kind of message isn't supported yet.")
            return

        text = msg.text.strip()
        attachments: list[str] = []
        for fetch in msg.files:
            try:
                data, name = await fetch()
                attachments.append(await asyncio.to_thread(save_upload, name, data))
            except ChannelError as exc:
                await self.reply(chat_id, exc.message)
            except Exception as exc:
                log.warning("%s: download failed: %s", self.id, self.redact(str(exc)))
                await self.reply(chat_id, "I couldn't download that file. Please try sending it again.")
        voice_in = False
        if msg.audio is not None:
            transcript = await self._transcribe(chat_id, msg.audio)
            if transcript is None:
                return
            voice_in = True
            text = f"{transcript}\n\n{text}".strip() if text else transcript
        if not text and not attachments:
            return
        if msg.reply_to and text and not attachments and await self.service.answer_reply(self, chat, msg.reply_to, text):
            return  # a reply to a task's question message: it is the answer, not a chat message

        session_id = await self.ensure_session(chat)
        if rt.running:
            steer = getattr(self.app.agent, "steer", None)
            if steer is not None and not attachments:
                try:
                    accepted = await maybe_await(steer(session_id, text))
                except Exception as exc:
                    log.debug("steer failed: %s", exc)
                    accepted = False
                if accepted is not False:
                    self.publish_message(chat_id, session_id, "in", text)
                    return
            rt.queued.append((text, attachments))
            await self.reply(chat_id, "Got it. I'll read that as soon as I finish this reply.")
            return
        rt.stopped = False
        rt.turn = asyncio.create_task(self._turn_worker(rt, chat_id, text, attachments, voice_in))

    async def _transcribe(self, chat_id: str, fetch: FileFetcher) -> str | None:
        transcribe = getattr(self.app.voice, "transcribe_bytes", None)
        if transcribe is None:
            await self.reply(chat_id, "I can't understand voice notes yet. Please type your message instead.")
            return None
        try:
            data, name = await fetch()
            with contextlib.suppress(Exception):
                await self.send_typing(chat_id)
            text = str(await transcribe(data, name) or "").strip()
        except ChannelError as exc:
            await self.reply(chat_id, exc.message)
            return None
        except Exception as exc:
            if type(exc).__name__ == "VoiceError":
                await self.reply(chat_id, f"I couldn't transcribe that voice note: {exc}")
            else:
                log.warning("%s: voice note failed: %s", self.id, self.redact(str(exc)))
                await self.reply(chat_id, "I couldn't transcribe that voice note. Please try again or type it.")
            return None
        if not text:
            await self.reply(chat_id, "I couldn't hear anything in that voice note.")
            return None
        await self.reply(chat_id, f"_Heard:_ {text}")
        return text

    async def ensure_session(self, chat: dict) -> str:
        sid = chat.get("session_id")
        if sid and await self.app.store.get_session(sid) is not None:
            return sid
        sid = await self.app.store.create_session(channel=self.id)
        await self.service.store.update_chat(self.id, chat["chat_id"], session_id=sid)
        chat["session_id"] = sid
        return sid

    async def stop_all_turns(self) -> int:
        """Stop everything: stop the reply in every chat and drop messages queued behind it (and tell the chat)."""
        stopped = 0
        dropped_steers = getattr(self.app, "stop_dropped", {}) or {}
        for chat_id, rt in list(self._chats.items()):
            dropped = len(rt.queued)
            rt.queued.clear()
            if await self.cancel_turn(chat_id):
                stopped += 1
            chat = await self.service.store.chat(self.id, chat_id)
            if chat is not None and chat.get("session_id") in dropped_steers:
                dropped += len(dropped_steers[chat["session_id"]])
            if dropped:
                await self.reply(chat_id, "Stopped. Your queued message wasn't sent." if dropped == 1
                                 else f"Stopped. Your {dropped} queued messages weren't sent.")
        return stopped

    async def cancel_turn(self, chat_id: str) -> bool:
        rt = self.runtime(chat_id)
        if not rt.running:
            return False
        assert rt.turn is not None
        rt.stopped = True
        rt.turn.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await rt.turn
        return True

    # ------------------------------------------------------------------ pairing
    def refusal_text(self) -> str:
        return (
            "Hi! This is a private Sentient assistant and this chat isn't paired with it.\n\n"
            "If this is your Sentient, open Sentient on your computer, go to **Channels**, "
            "and send `/pair` followed by the 6-digit code shown there."
        )

    def help_text(self, chat: dict | None) -> str:
        lines = [
            f"**{self.app.config.assistant.name} on {self.display_name}**",
            "Send a message, a voice note, a photo or a file and I'll reply here.",
            "",
            "/new - start a fresh chat",
            "/stop - stop the reply in progress",
            "/stopall - stop everything Sentient is doing, on every device",
            "/resume - start scheduled tasks and suggestions again after /stopall",
            "/model - switch between local and cloud models",
            "/help - show this message",
        ]
        if chat and chat.get("deliver"):
            lines += ["", "Task results, plans to approve, questions from your tasks and suggestions are also sent here. "
                          f"To answer a task's question, {self.choice_hint} or reply to its message. "
                          "You can turn that off in Sentient under Channels."]
        return "\n".join(lines)

    async def _pair(self, rt: ChatRuntime, msg: Incoming, chat: dict | None, arg: str) -> None:
        chat_id = str(msg.chat_id)
        cfg = self.app.config.channels
        if chat is not None:
            await self.reply(chat_id, "This chat is already paired. Just send a message.")
            return
        now = self.clock()
        rt.pair_failures = [t for t in rt.pair_failures if now - t < cfg.pairing_code_minutes * 60]
        if len(rt.pair_failures) >= cfg.pairing_max_attempts:
            return  # paused after too many wrong codes; already told once
        code = re.sub(r"\s+", "", arg)
        if not re.fullmatch(r"\d{6}", code):
            await self.reply(chat_id, "Send `/pair` followed by the 6-digit code shown in Sentient under **Channels**.")
            return
        result = await self.service.store.redeem_code(self.id, code, cfg.pairing_max_attempts)
        if result == PairingResult.OK:
            rt.pair_failures.clear()
            session_id = await self.app.store.create_session(channel=self.id)
            chat = await self.service.store.add_chat(
                self.id, chat_id, msg.label or chat_id, deliver=self.cfg.deliver_default, session_id=session_id
            )
            await self.service.publish_channel(self.id)
            name = self.app.config.assistant.name
            await self.reply(chat_id, f"Paired! You can talk to {name} here now.\n\n" + self.help_text(chat))
            return
        if result == PairingResult.EXPIRED:
            await self.reply(chat_id, "That code has expired. Create a new one in Sentient under **Channels**.")
            return
        rt.pair_failures.append(now)
        if result == PairingResult.BURNED or len(rt.pair_failures) >= cfg.pairing_max_attempts:
            await self.reply(chat_id, "Too many wrong codes. For your safety that code was cancelled. "
                                      "Create a new one in Sentient and try again in a few minutes.")
            return
        await self.reply(chat_id, "That code didn't work. Check the code shown in Sentient and try again.")

    # ------------------------------------------------------------------ chat turns
    async def _turn_worker(self, rt: ChatRuntime, chat_id: str, text: str, attachments: list[str], voice_in: bool) -> None:
        while True:
            await self.run_reply(chat_id, text, attachments, voice_in=voice_in)
            if rt.stopped or not rt.queued:
                return
            batch, rt.queued = rt.queued, []
            text = "\n\n".join(t for t, _ in batch if t)
            attachments = [a for _, atts in batch for a in atts]
            voice_in = False

    def activity_label(self, tool_name: str) -> str:
        tool = self.app.registry.get(tool_name)
        plugin = getattr(tool, "plugin", "") or ""
        if plugin in ACTIVITY:
            return ACTIVITY[plugin]
        integ = self.app.integrations
        display = None
        if plugin and hasattr(integ, "plugin"):
            with contextlib.suppress(Exception):
                display = getattr(integ.plugin(plugin), "display_name", None)
        return f"Using {display or humanize_tool(plugin or tool_name)}"

    async def run_reply(self, chat_id: str, text: str, attachments: list[str], *, voice_in: bool = False) -> None:
        chat = await self.service.store.chat(self.id, chat_id)
        if chat is None or self.app.agent is None:
            return
        session_id = await self.ensure_session(chat)
        self.publish_message(chat_id, session_id, "in", text)
        cfg = self.cfg
        typing = asyncio.create_task(self._typing_loop(chat_id))
        status = StatusLine(self, chat_id)
        stream = ReplyStream(self, chat_id, streaming=cfg.stream_edits, interval=cfg.edit_interval_s)
        approval_msgs: dict[str, tuple[str, str]] = {}  # call_id -> (approval_id, message_id)
        replies: list[str] = []
        agen = self.app.agent.run_turn(session_id, text, channel=self.id, attachments=attachments)
        try:
            async for event in agen:
                if isinstance(event, TextDelta):
                    if status.message_id is not None:
                        await status.clear()
                    await stream.add(event.text)
                elif isinstance(event, ToolCallEvent):
                    if part := await stream.finish():
                        replies.append(part)
                    stream = ReplyStream(self, chat_id, streaming=cfg.stream_edits, interval=cfg.edit_interval_s)
                    if cfg.show_tool_activity and self.status_lines:
                        await status.show(self.activity_label(event.name))
                elif isinstance(event, ApprovalRequest):
                    await status.clear()
                    mid = await self.send_approval(chat_id, event)
                    if mid:
                        approval_msgs[event.call_id] = (event.approval_id, mid)
                elif isinstance(event, ToolResultEvent):
                    if event.call_id in approval_msgs:
                        _, mid = approval_msgs.pop(event.call_id)
                        if (str(chat_id), mid) in self._button_text:  # answered elsewhere (desktop)
                            outcome = "Denied" if event.is_error and "declined" in json.dumps(event.result) else "Answered"
                            await self.settle_buttons(chat_id, mid, outcome)
                elif isinstance(event, Error):
                    if event.recoverable:
                        await self.reply(chat_id, f"_{event.message}_")
                    else:
                        await stream.finish()
                        await self.reply(chat_id, f"Sorry, something went wrong: {self.redact(event.message)[:500]}")
                elif getattr(event, "type", None) == "user_interjection":
                    # a steer was applied: keep what was written so far as its own message, continue below it
                    if part := await stream.finish():
                        replies.append(part)
                    stream = ReplyStream(self, chat_id, streaming=cfg.stream_edits, interval=cfg.edit_interval_s)
                elif isinstance(event, Done):
                    await status.clear()
                    final = await stream.finish(replacement=event.content or "")
                    if final:
                        replies.append(final)
                    if replies:
                        self.publish_message(chat_id, session_id, "out", "\n\n".join(replies))
                    if voice_in and cfg.voice_replies and final:
                        await self._voice_reply(chat_id, final)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await status.clear()
                if stream.text.strip():
                    await stream.finish(suffix="\n\n_(stopped)_")
            raise
        except Exception as exc:
            log.exception("%s: chat turn failed", self.id)
            await self.reply(chat_id, f"Sorry, something went wrong: {self.redact(str(exc))[:300]}")
        finally:
            typing.cancel()
            with contextlib.suppress(Exception):
                await agen.aclose()
            for _, mid in approval_msgs.values():
                if (str(chat_id), mid) in self._button_text:
                    await self.settle_buttons(chat_id, mid, "No longer needed")

    async def _typing_loop(self, chat_id: str) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                with contextlib.suppress(Exception):
                    await self.send_typing(chat_id)
                await asyncio.sleep(TYPING_EVERY_S)

    async def _voice_reply(self, chat_id: str, text: str) -> None:
        speak = getattr(self.app.voice, "speak", None)
        if speak is None:
            return
        try:
            audio = await speak(text)
            if audio:
                await self.send_audio(chat_id, audio, "reply.wav")
        except Exception as exc:
            log.info("%s: voice reply skipped: %s", self.id, exc)

    async def send_approval(self, chat_id: str, event: ApprovalRequest) -> str | None:
        args = json.dumps(event.arguments, ensure_ascii=False, indent=1, default=str)
        if len(args) > 700:
            args = args[:700] + "\n…"
        md = (
            f"**Approval needed**\n{self.app.config.assistant.name} wants to use **{humanize_tool(event.name)}** "
            f"(risk: {event.risk}).\n```json\n{args}\n```"
        )
        first = [Button("Allow", f"ap:a:{event.approval_id}", "success")]
        if event.untrusted:  # it read outside content: say why; "for this chat" would not cover the next one anyway
            md += f"\n{event.untrusted}"
        else:
            first.append(Button("Allow for this chat", f"ap:s:{event.approval_id}", "primary"))
        buttons = [first, [Button("Deny", f"ap:d:{event.approval_id}", "danger")]]
        try:
            ids = await self.send_markdown(chat_id, md, buttons)
            return ids[-1] if ids else None
        except Exception as exc:
            log.warning("%s: approval message failed: %s", self.id, self.redact(str(exc)))
            return None

    # ------------------------------------------------------------------ buttons
    async def handle_button(self, chat_id: str, message_id: str, data: str) -> str:
        """Run a button action. Returns a short toast text; updates the message itself."""
        chat_id, message_id = str(chat_id), str(message_id)
        if await self.service.store.chat(self.id, chat_id) is None:
            return "This chat isn't paired with Sentient."
        parts = (data or "").split(":", 2)
        if len(parts) != 3:
            return "Unknown button."
        kind, code, ref = parts
        if kind == "ap":
            decision = APPROVAL_DECISIONS.get(code)
            if decision is None:
                return "Unknown button."
            if not self.app.approvals.resolve(ref, decision):
                await self.settle_buttons(chat_id, message_id, "This request is no longer waiting")
                return "This request is no longer waiting."
            await self.settle_buttons(chat_id, message_id, DECISION_LABEL[decision])
            return DECISION_LABEL[decision]
        if kind == "tp":
            return await self.service.act_on_plan(self, chat_id, message_id, ref, approve=code == "a")
        if kind == "sg":
            return await self.service.act_on_suggestion(self, chat_id, message_id, ref, approve=code == "a")
        if kind == "tq":
            return await self.service.act_on_question(self, chat_id, message_id, ref, code)
        if kind == "rp":  # "Make this a rule?" (#130)
            if code not in {"a", "d"}:
                return "Unknown button."
            return await self.service.act_on_rule_proposal(self, chat_id, message_id, ref, accept=code == "a")
        if kind == "mp":
            return await self._preset_button(chat_id, message_id, ref)
        return "Unknown button."

    # ------------------------------------------------------------------ /model
    @staticmethod
    def preset_ref(name: str) -> str:
        """Short, stable button data for a preset (Telegram allows 64 bytes)."""
        return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]

    async def _model_command(self, chat_id: str, arg: str) -> None:
        """``/model`` lists the model setups as buttons; ``/model <number or name>`` and ``/model undo`` act directly."""
        listing = await model_presets.listing(self.app)
        usable = [p for p in listing["presets"] if p["available"]]
        arg = arg.strip()
        if arg.lower() == "undo":
            await self.reply(chat_id, await self._preset_outcome(None))
            return
        if arg:
            pick = usable[int(arg) - 1] if arg.isdigit() and 1 <= int(arg) <= len(usable) else None
            if pick is None:
                pick = next((p for p in listing["presets"] if p["name"].lower() == arg.lower()), None)
            if pick is None:
                await self.reply(chat_id, f"I don't have a model setup called {arg!r}. Send /model to see them.")
                return
            await self.reply(chat_id, await self._preset_outcome(pick["name"]))
            return
        primary = self.app.config.models.roles.primary
        if listing["active"]:
            changed = " (changed since)" if listing["modified"] else ""
            lines = [f"Models: **{listing['active']}**{changed}, chatting with {primary}."]
        else:
            lines = [f"Models: your own setup, chatting with {primary}."]
        for p in listing["presets"]:
            if not p["available"]:
                lines.append(f"{p['name']}: {p['reason']} Add one in Sentient under Settings, Models.")
        lines.append("Pick a setup to switch every model at once." + (" Send /model undo to go back." if listing["can_undo"] else ""))
        buttons = [[Button(p["name"] + (" (current)" if p["active"] else ""), f"mp:a:{self.preset_ref(p['name'])}",
                           "success" if p["active"] else "primary")] for p in usable]
        try:
            await self.send_markdown(chat_id, "\n\n".join(lines), buttons)
        except Exception as exc:
            log.warning("%s: /model failed: %s", self.id, self.redact(str(exc)))

    async def _preset_outcome(self, name: str | None) -> str:
        """Apply a preset (None = undo the last switch) and describe the result in plain words."""
        try:
            result = await (model_presets.undo(self.app) if name is None else model_presets.apply(self.app, name))
        except model_presets.PresetError as exc:
            return exc.message
        if name is None:
            head = f"Back to {result['preset']}." if result["preset"] else "Back to your previous models."
        else:
            head = f"Switched to {result['preset']}." if result["changed"] else f"Already using {result['preset']}."
        lines = [head] + [f"Still needed: {item['detail']} {item['fix']}" for item in result["missing"]]
        if result["missing"]:
            lines.append("You can do that in Sentient on your computer.")
        return "\n".join(lines)

    async def _preset_button(self, chat_id: str, message_id: str, ref: str) -> str:
        name = next((p["name"] for p in model_presets.presets(self.app.config, model_presets.hardware(self.app)) if self.preset_ref(p["name"]) == ref), None)
        if name is None:
            await self.settle_buttons(chat_id, message_id, "That setup no longer exists")
            return "That setup no longer exists."
        text = await self._preset_outcome(name)
        head, _, rest = text.partition("\n")
        await self.settle_buttons(chat_id, message_id, head.rstrip("."))
        if rest:
            await self.reply(chat_id, rest)
        return head
