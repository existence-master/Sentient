"""Messaging channels (docs/API.md section 14): Telegram and Discord bots paired with pairing codes, and
WhatsApp linked to the user's own account (their "Message yourself" chat).

- Tokens live in the OS keychain (``channel_<id>_token``); config and logs never see them. WhatsApp has no
  token: its linked session is kept under ``~/.sentient/whatsapp``.
- A paired chat is a normal Sentient chat whose session has ``channel`` = the channel id.
- Delivery: ``notification.new`` events (task results, plans awaiting approval, questions from running
  tasks, proactive suggestions, subagent completions) are forwarded to paired chats with ``deliver`` on,
  with action buttons. A reply to a delivered task question message is the answer to that question.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

from sentient import secrets as keychain
from sentient.channels.base import Button, Channel, ChannelError
from sentient.channels.discord import DiscordChannel
from sentient.channels.formatting import summary_text
from sentient.channels.store import ChannelStore
from sentient.channels.telegram import TelegramChannel
from sentient.channels.whatsapp import WhatsAppChannel
from sentient.proactivity.brief import DailyBrief
from sentient.services import Service

log = logging.getLogger(__name__)

_UNSET: Any = object()
TASK_RESULT_EVENTS = {"run_completed", "run_failed", "planning_failed", "clarification_needed", "disabled"}
STATUS_LABELS = {
    "approved": "Approved", "declined": "Declined", "dismissed": "Dismissed", "answered": "Answered", "cancelled": "Cancelled",
}
QUESTION_ROW = 2  # option buttons per row (Discord allows 5 per row and 5 rows)
_TOKEN_LOG_RE = re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}")


class _RedactTokens(logging.Filter):
    """httpx logs request URLs at INFO; Telegram puts the bot token in the URL path."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = _TOKEN_LOG_RE.sub("bot<token>", message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def install_log_redaction() -> None:
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _RedactTokens) for f in logger.filters):
            logger.addFilter(_RedactTokens())


class ChannelService(Service):
    name = "channels"

    def __init__(self, app):
        super().__init__(app)
        self.store = ChannelStore(app.store)
        self.channels: dict[str, Channel] = {
            "telegram": TelegramChannel(self), "discord": DiscordChannel(self), "whatsapp": WhatsAppChannel(self),
        }
        # notification id -> [(channel, chat_id, message_id)] for messages with action buttons
        self._delivered: dict[str, list[tuple[str, str, str]]] = {}
        self._subagents_sent: set[tuple[str, str, str]] = set()
        self._listening = asyncio.Event()
        install_log_redaction()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.store.migrate()
        self._loops.append(asyncio.create_task(self._listen(), name="channels:delivery"))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._listening.wait(), timeout=2)
        if not self.app.config.channels.enabled:
            return
        for ch in self.channels.values():
            state = await self.store.state(ch.id)
            if not state["enabled"]:
                continue
            if not ch.uses_token:
                if not await ch.restore():  # type: ignore[attr-defined]
                    await self.set_status(ch.id, "error", error=f"{ch.display_name} needs to be linked again. "
                                          "Click Reconnect and scan the code.")
                    continue
            elif token := keychain.get_secret(ch.secret_name):
                ch.token = token
            else:
                await self.set_status(ch.id, "error", error="The bot token is missing from the system keychain. Connect again.")
                continue
            if self.app.enable_background and ch.cfg.enabled:
                await self.set_status(ch.id, "connecting", error=None)
                ch.start_runtime()

    async def stop(self) -> None:
        for ch in self.channels.values():
            with contextlib.suppress(Exception):
                await ch.stop_runtime()
        await super().stop()

    async def halt(self) -> int:
        """Stop everything: stop the reply in progress in every chat and drop queued messages."""
        stopped = 0
        for ch in self.channels.values():
            stopped += await ch.stop_all_turns()
        return stopped

    # ------------------------------------------------------------------ helpers
    def get(self, channel_id: str) -> Channel:
        ch = self.channels.get(channel_id)
        if ch is None:
            raise ChannelError(f"Unknown channel '{channel_id}'.", 404)
        return ch

    async def channel_dict(self, channel_id: str) -> dict:
        ch = self.get(channel_id)
        state = await self.store.state(channel_id)
        return {
            "id": ch.id,
            "display_name": ch.display_name,
            "status": state.get("status") or "disconnected",
            "account_label": state.get("account_label"),
            "error": state.get("error"),
            "qr": ch.qr,
            "paired": await self.store.chats(channel_id),
            "setup": {"fields": ch.setup_fields, "instructions_md": ch.instructions_md},
        }

    async def list(self) -> list[dict]:
        return [await self.channel_dict(cid) for cid in self.channels]

    async def publish_channel(self, channel_id: str) -> None:
        self.app.bus.publish("channel.updated", await self.channel_dict(channel_id))

    async def set_status(self, channel_id: str, status: str, *, error: Any = _UNSET, account_label: Any = _UNSET) -> None:
        state = await self.store.state(channel_id)
        fields: dict[str, Any] = {}
        if state.get("status") != status:
            fields["status"] = status
        if error is not _UNSET and state.get("error") != error:
            fields["error"] = error
        if account_label is not _UNSET and account_label and state.get("account_label") != account_label:
            fields["account_label"] = account_label
        if fields:
            await self.store.set_state(channel_id, **fields)
            await self.publish_channel(channel_id)

    async def is_connected(self, channel_id: str) -> bool:
        ch = self.channels.get(channel_id)
        if ch is None or not ch.ready:
            return False
        return bool((await self.store.state(channel_id))["enabled"])

    # ------------------------------------------------------------------ REST operations
    async def connect(self, channel_id: str, fields: dict[str, Any]) -> dict:
        ch = self.get(channel_id)
        if not self.app.config.channels.enabled or not ch.cfg.enabled:
            raise ChannelError(f"{ch.display_name} is turned off in Settings (Channels).")
        if not ch.uses_token:
            return await self._link(ch)
        token, label = await ch.validate(fields or {})
        if not keychain.set_secret(ch.secret_name, token):
            raise ChannelError("Couldn't save the token in the system keychain.", 500)
        await ch.stop_runtime()
        previous = await self.store.state(channel_id)
        same_bot = previous.get("account_label") == label
        ch.token = token
        await self.store.set_state(
            channel_id, enabled=1, status="connected", account_label=label, error=None,
            cursor=previous.get("cursor") if same_bot else None,
        )
        if self.app.enable_background:
            ch.start_runtime()
        await self.publish_channel(channel_id)
        return await self.channel_dict(channel_id)

    async def _link(self, ch: Channel) -> dict:
        """Start (or restart) a channel that links a session instead of using a token: WhatsApp shows a QR
        code (``qr`` on the Channel, updated through ``channel.updated``) until the phone scans it."""
        await ch.begin_link()  # type: ignore[attr-defined]
        await self.store.set_state(ch.id, enabled=1, status="connecting" if ch.ready else "linking", error=None)
        if self.app.enable_background:
            ch.start_runtime()
        await self.publish_channel(ch.id)
        return await self.channel_dict(ch.id)

    async def disconnect(self, channel_id: str) -> dict:
        ch = self.get(channel_id)
        if not ch.uses_token:
            await ch.unlink()  # type: ignore[attr-defined]
        await ch.stop_runtime()
        ch.token = None
        if ch.uses_token:
            keychain.delete_secret(ch.secret_name)
        await self.store.clear_code(channel_id)
        await self.store.set_state(channel_id, enabled=0, status="disconnected", error=None, account_label=None)
        await self.publish_channel(channel_id)
        return await self.channel_dict(channel_id)

    async def create_pairing(self, channel_id: str) -> dict:
        ch = self.get(channel_id)
        if not await self.is_connected(channel_id):
            raise ChannelError(f"Connect {ch.display_name} before pairing a chat.", 409)
        minutes = self.app.config.channels.pairing_code_minutes
        code, expires_at = await self.store.new_code(channel_id, minutes)
        state = await self.store.state(channel_id)
        instructions = ch.pairing_instructions(code, state.get("account_label"))  # type: ignore[attr-defined]
        return {"code": code, "expires_at": expires_at, "instructions": f"{instructions} (valid for {minutes} minutes)"}

    async def set_deliver(self, channel_id: str, chat_id: str, deliver: bool) -> dict:
        self.get(channel_id)
        if not await self.store.update_chat(channel_id, chat_id, deliver=bool(deliver)):
            raise ChannelError("That chat isn't paired.", 404)
        await self.publish_channel(channel_id)
        return await self.channel_dict(channel_id)

    async def remove_chat(self, channel_id: str, chat_id: str) -> dict:
        ch = self.get(channel_id)
        if await self.store.chat(channel_id, chat_id) is None:
            raise ChannelError("That chat isn't paired.", 404)
        await ch.cancel_turn(chat_id)
        await self.store.remove_chat(channel_id, chat_id)
        await self.publish_channel(channel_id)
        return await self.channel_dict(channel_id)

    async def test(self, channel_id: str, chat_id: str | None = None) -> dict:
        ch = self.get(channel_id)
        if not await self.is_connected(channel_id):
            return {"ok": False, "error": f"{ch.display_name} isn't connected."}
        try:
            if chat_id:
                if await self.store.chat(channel_id, str(chat_id)) is None:
                    return {"ok": False, "error": "That chat isn't paired."}
                name = self.app.config.assistant.name
                await ch.send_markdown(str(chat_id), f"This is a test message from **{name}**. Everything is working.")
            else:
                await ch.check()  # type: ignore[attr-defined]
        except ChannelError as exc:
            return {"ok": False, "error": exc.message}
        except Exception as exc:
            return {"ok": False, "error": ch.redact(str(exc))[:300]}
        return {"ok": True}

    # ------------------------------------------------------------------ delivery
    async def _listen(self) -> None:
        async with self.app.bus.subscribe() as queue:
            self._listening.set()
            while True:
                event = await queue.get()
                try:
                    kind, data = event.get("type"), event.get("data")
                    if not isinstance(data, dict):
                        continue
                    if kind == "notification.new":
                        await self.deliver_notification(data)
                    elif kind == "notification.updated":
                        await self.update_delivered(data)
                    elif kind == "subagent.updated":
                        await self.deliver_subagent(data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("channel delivery failed")

    async def _delivery_targets(self) -> list[tuple[Channel, dict]]:
        targets = []
        for ch in self.channels.values():
            if not await self.is_connected(ch.id):
                continue
            for chat in await self.store.chats(ch.id):
                if chat["deliver"]:
                    targets.append((ch, chat))
        return targets

    async def _task_result(self, task_id: str | None) -> str:
        getter = getattr(self.app.tasks, "get", None)
        if not task_id or getter is None:
            return ""
        with contextlib.suppress(Exception):
            task = await getter(task_id)
            runs = (task or {}).get("runs") or []
            result = (runs[-1] or {}).get("result") if runs else None
            if isinstance(result, dict):
                return str(result.get("summary") or "")
        return ""

    async def format_notification(self, note: dict) -> tuple[str, list[list[Button]] | None] | None:
        cfg = self.app.config.channels
        kind = note.get("kind")
        payload = note.get("payload") or {}
        title = (note.get("title") or "").strip()
        message = summary_text(note.get("message") or "")
        if kind == "task":
            event, task_id = payload.get("event"), payload.get("task_id") or note.get("task_id")
            if event == "approval_needed":
                if not cfg.deliver_plans or not task_id or payload.get("status"):
                    return None
                md = f"**{title or 'Plan ready for approval'}**\n{message}"
                return md, [[Button("Approve plan", f"tp:a:{task_id}", "success"), Button("Decline", f"tp:d:{task_id}", "danger")]]
            if event == "question":
                return self._format_question(note) if cfg.deliver_task_results else None
            if event in TASK_RESULT_EVENTS and cfg.deliver_task_results:
                md = f"**{title or 'Task update'}**\n{message}"
                if event == "run_completed":
                    summary = summary_text(await self._task_result(task_id), 900)
                    if summary:
                        md += f"\n\n{summary}"
                return md, None
            return None
        if kind == "proactive":
            nid = note.get("id")
            if not cfg.deliver_suggestions or not isinstance(payload.get("suggestion"), dict) or not nid:
                return None
            if payload.get("status", "pending") != "pending":
                return None
            md = f"**{title or 'Suggestion'}**\n{message}"
            return md, [[Button("Approve", f"sg:a:{nid}", "success"), Button("Dismiss", f"sg:d:{nid}", "secondary")]]
        if kind == "brief":
            brief = payload.get("brief")
            if not cfg.deliver_briefs or not isinstance(brief, dict) or payload.get("status", "active") != "active":
                return None
            return f"**{title or 'Your Daily Brief'}**\n{DailyBrief.as_text(brief, links=True)}", None
        if payload.get("subagent_id") and cfg.deliver_subagents:
            return f"**{title or 'Background work finished'}**\n{message}", None
        return None

    @staticmethod
    def _format_question(note: dict) -> tuple[str, list[list[Button]] | None] | None:
        payload = note.get("payload") or {}
        run_id, question = payload.get("run_id"), str(payload.get("question") or note.get("message") or "").strip()
        if not run_id or not question or payload.get("status"):
            return None
        title = (note.get("title") or "").strip() or "A task needs your answer"
        options = [str(o) for o in payload.get("options") or [] if str(o).strip()]
        hint = "Tap an option, or reply to this message with your answer." if options else "Reply to this message with your answer."
        md = f"**{title}**\n{summary_text(question, 900)}\n\n_{hint}_"
        if not options:
            return md, None
        buttons = [Button(label[:60], f"tq:{i}:{run_id}", "primary") for i, label in enumerate(options)]
        return md, [buttons[i:i + QUESTION_ROW] for i in range(0, len(buttons), QUESTION_ROW)]

    async def deliver_notification(self, note: dict) -> int:
        formatted = await self.format_notification(note)
        if formatted is None:
            return 0
        md, buttons = formatted
        subagent_id = (note.get("payload") or {}).get("subagent_id")
        sent = 0
        for ch, chat in await self._delivery_targets():
            if subagent_id:
                key = (str(subagent_id), ch.id, chat["chat_id"])
                if key in self._subagents_sent:
                    continue
                self._subagents_sent.add(key)
            try:
                ids = await ch.send_markdown(chat["chat_id"], md, buttons)
            except Exception as exc:
                log.warning("delivering to %s failed: %s", ch.id, ch.redact(str(exc)))
                continue
            sent += 1
            ch.publish_message(chat["chat_id"], None, "out", md)
            payload = note.get("payload") or {}
            if note.get("kind") == "task" and payload.get("event") == "question" and ids:
                try:  # persisted, so a reply still answers the question after a restart
                    await self.store.add_question(
                        ch.id, chat["chat_id"], [str(i) for i in ids],
                        task_id=str(payload.get("task_id") or note.get("task_id") or ""), run_id=str(payload.get("run_id")),
                    )
                except Exception:
                    log.exception("could not remember the delivered question")
            if buttons and ids and note.get("id"):
                self._delivered.setdefault(str(note["id"]), []).append((ch.id, chat["chat_id"], ids[-1]))
        return sent

    async def update_delivered(self, note: dict) -> None:
        status = (note.get("payload") or {}).get("status")
        if status not in STATUS_LABELS:
            return
        targets = self._delivered.pop(str(note.get("id")), [])
        label = STATUS_LABELS[status]
        payload = note.get("payload") or {}
        if note.get("kind") == "task" and payload.get("event") == "question":
            answer = summary_text(str(payload.get("answer") or ""), 80)
            label = f"Answered: {answer}" if status == "answered" and answer else label
        elif note.get("kind") == "task":
            label = f"Plan {status}"
        for channel_id, chat_id, message_id in targets:
            ch = self.channels.get(channel_id)
            if ch is not None and (chat_id, message_id) in ch._button_text:
                await ch.settle_buttons(chat_id, message_id, label)

    async def deliver_subagent(self, data: dict) -> bool:
        if not data.get("background") or data.get("status") not in {"completed", "error"}:
            return False
        if not self.app.config.channels.deliver_subagents:
            return False
        session_id, subagent_id = data.get("session_id"), data.get("subagent_id")
        if not session_id or not subagent_id:
            return False
        found = await self.store.chat_for_session(str(session_id))
        if found is None:
            return False
        channel_id, chat = found
        ch = self.channels.get(channel_id)
        key = (str(subagent_id), channel_id, chat["chat_id"])
        if ch is None or key in self._subagents_sent or not await self.is_connected(channel_id):
            return False
        self._subagents_sent.add(key)
        goal = summary_text(str(data.get("goal") or ""), 200)
        if data.get("status") == "completed":
            md = f"**Background work finished**\n{goal}\n\n{summary_text(str(data.get('summary') or ''), 1500)}".strip()
        else:
            md = f"**Background work failed**\n{goal}\n\n{summary_text(str(data.get('error') or ''), 500)}".strip()
        try:
            await ch.send_markdown(chat["chat_id"], md)
        except Exception as exc:
            log.warning("subagent delivery failed: %s", ch.redact(str(exc)))
            return False
        ch.publish_message(chat["chat_id"], session_id, "out", md)
        return True

    # ------------------------------------------------------------------ button actions
    async def act_on_plan(self, ch: Channel, chat_id: str, message_id: str, task_id: str, *, approve: bool) -> str:
        fn = getattr(self.app.tasks, "approve" if approve else "decline", None)
        if fn is None:
            return "Tasks aren't available right now."
        self._forget_delivered(ch.id, chat_id, message_id)
        try:
            await fn(task_id)
        except Exception as exc:
            text = str(exc) or "That didn't work."
            await ch.settle_buttons(chat_id, message_id, text[:120])
            return text[:190]
        outcome = "Plan approved" if approve else "Plan declined"
        await ch.settle_buttons(chat_id, message_id, outcome)
        return outcome

    async def act_on_suggestion(self, ch: Channel, chat_id: str, message_id: str, notification_id: str, *, approve: bool) -> str:
        fn = getattr(self.app.proactivity, "act_on_suggestion", None)
        if fn is None:
            return "Suggestions aren't available right now."
        self._forget_delivered(ch.id, chat_id, message_id)
        try:
            result = await fn(notification_id, "approve" if approve else "dismiss")
        except Exception as exc:
            text = str(getattr(exc, "detail", "") or exc) or "That didn't work."
            await ch.settle_buttons(chat_id, message_id, text[:1].upper() + text[1:120])
            return text[:190]
        if approve:
            outcome = "Approved. I've started a task for it." if (result or {}).get("task_id") else "Approved"
        else:
            outcome = "Dismissed"
        await ch.settle_buttons(chat_id, message_id, outcome)
        return outcome

    async def act_on_question(self, ch: Channel, chat_id: str, message_id: str, run_id: str, index: str) -> str:
        """An option button under a task's question: answer it with that option."""
        found = await self._waiting_question(run_id)
        if found is None:
            await ch.settle_buttons(chat_id, message_id, "This question is no longer waiting")
            return "This question is no longer waiting."
        options = found.get("options") or []
        if not index.isdigit() or int(index) >= len(options):
            return "Unknown button."
        self._forget_delivered(ch.id, chat_id, message_id)
        answer = str(options[int(index)])
        try:
            await self.app.tasks.answer_question(found["task_id"], run_id, answer)
        except Exception as exc:
            text = str(exc) or "That didn't work."
            await ch.settle_buttons(chat_id, message_id, text[:120])
            return text[:190]
        outcome = f"Answered: {summary_text(answer, 80)}"
        await ch.settle_buttons(chat_id, message_id, outcome)
        return outcome

    async def _waiting_question(self, run_id: str) -> dict | None:
        fn = getattr(self.app.tasks, "waiting_questions", None)
        if fn is None:
            return None
        with contextlib.suppress(Exception):
            for q in await fn():
                if q.get("run_id") == run_id:
                    return q
        return None

    async def answer_reply(self, ch: Channel, chat: dict, reply_to: str, text: str) -> bool:
        """A message that replies to a delivered task question is the answer to that question.

        Returns True when ``reply_to`` is one of this chat's question messages (answered now, or already
        handled); False for a reply to anything else, which is then normal chat.
        """
        chat_id = str(chat["chat_id"])
        found = await self.store.question_for(ch.id, chat_id, str(reply_to))
        if found is None:
            return False
        ch.publish_message(chat_id, chat.get("session_id"), "in", text)
        waiting = await self._waiting_question(found["run_id"])
        if waiting is None:
            reply = "That question has already been handled, so I didn't pass this on."
        else:
            options = [str(o) for o in waiting.get("options") or []]
            choice = text.strip()
            if re.fullmatch(r"[0-9]{1,3}", choice) and 1 <= int(choice) <= len(options):
                text = options[int(choice) - 1]  # "2" picks the second option (ASCII digits only)
            try:
                await self.app.tasks.answer_question(waiting["task_id"], waiting["run_id"], text)
                reply = f"Thanks! I passed your answer to '{summary_text(waiting['task_name'], 80)}'. It's carrying on now."
            except Exception as exc:
                reply = f"I couldn't pass that on: {str(exc)[:200] or 'something went wrong'}"
        await ch.reply(chat_id, reply)
        ch.publish_message(chat_id, None, "out", reply)
        return True

    def _forget_delivered(self, channel_id: str, chat_id: str, message_id: str) -> None:
        for nid, targets in list(self._delivered.items()):
            kept = [t for t in targets if t != (channel_id, chat_id, message_id)]
            if kept:
                self._delivered[nid] = kept
            else:
                self._delivered.pop(nid, None)
