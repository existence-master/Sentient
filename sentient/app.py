"""Composition root: builds every subsystem once.

The desktop app launches ``sentient serve``; the gateway owns one SentientApp
for the life of the process. Tests construct SentientApp directly with a fake
LLM provider and a temporary database.

Start order (stop runs in reverse):
    store -> memory -> notifications -> integrations (registers plugins)
    -> builtin tools -> skills -> agent -> subagents -> sandbox -> terminal -> browser -> nodes
    -> tasks -> reminders -> proactivity -> evolution -> user_model -> dreaming -> voice -> channels -> backups

Stop everything (``stop_all`` / ``resume``, docs/API.md section 17) is deterministic: it never
asks the model. It cancels running chat replies, task runs, helpers, scripts, commands and browser
actions, and pauses scheduled and triggered tasks, proactivity, learning and dreaming until
the user resumes. The stopped state is kept in the ``meta`` table, so it survives a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from sentient import paths, secrets
from sentient.activity import Timeline
from sentient.agent import incognito as incognito_outputs
from sentient.agent import selections
from sentient.agent.approvals import ApprovalBroker
from sentient.agent.chat_rules import ChatRules
from sentient.agent.grants import GrantStore
from sentient.agent.incognito import IncognitoSweeper
from sentient.agent.loop import Agent
from sentient.agent.subagents import SubagentManager
from sentient.agent.undo import UndoBroker
from sentient.backup import BackupService
from sentient.browser import BrowserService
from sentient.channels import ChannelService
from sentient.config import SentientConfig, load_config, save_config
from sentient.events import EventBus
from sentient.evolution import EvolutionService
from sentient.folders import FolderService
from sentient.integrations import IntegrationManager
from sentient.llm import claude_code
from sentient.llm.connect import ProviderConnections
from sentient.llm.hardware import HardwareProbe
from sentient.llm.jobs import ModelJobs, as_kind
from sentient.llm.provider import LiteLLMProvider, LLMProvider
from sentient.llm.spending import Spending
from sentient.memory.dreaming import DreamingService
from sentient.memory.facts import FactMemory
from sentient.memory.usermodel import UserModelService
from sentient.memory.workspace import Workspace
from sentient.migrate.memory_import import MemoryImports
from sentient.nodes import NodeService
from sentient.notifications import NotificationService
from sentient.presence import Presence
from sentient.proactivity import ProactiveEngine
from sentient.reminders import ReminderService
from sentient.sandbox import SandboxService
from sentient.sensors import SensorRegistry
from sentient.services import Service
from sentient.skills.hub import SkillsHub
from sentient.skills.loader import SkillLibrary
from sentient.snapshots import FileSnapshots
from sentient.store.db import Store, now_iso
from sentient.tasks import TaskService
from sentient.terminal import TerminalService
from sentient.tools.registry import ToolRegistry
from sentient.voice import VoiceService

log = logging.getLogger(__name__)

# Background work gets this long to finish when Sentient shuts down; then it is cancelled.
SHUTDOWN_GRACE_S = 15.0
STOP_META_KEY = "stop.state"  # {"stopped", "stopped_at", "source"} while Stop everything is on


class SentientApp:
    def __init__(
        self,
        config: SentientConfig | None = None,
        *,
        llm: LLMProvider | None = None,
        db_path: Path | None = None,
        enable_background: bool = True,
    ):
        self.config = config or load_config()
        self._claude_code_everywhere = claude_code.everywhere(self.config)  # to notice it being turned off (#444)
        self._notes: set[asyncio.Task] = set()
        self.enable_background = enable_background  # tests turn schedulers/pollers off
        self.bus = EventBus()
        self.store = Store(db_path, before_migration=self._snapshot_before_migration)
        self.llm: LLMProvider = llm or LiteLLMProvider(self.config)
        # the cost meter and the monthly budget (#148): the provider checks and counts every call through it
        self.spending = Spending(self)
        if isinstance(self.llm, LiteLLMProvider) and self.llm.meter is None:
            self.llm.meter = self.spending
        # one local model job at a time, chats first (#149): the provider's scheduler, or an idle one for status
        jobs = getattr(self.llm, "jobs", None)
        self.model_jobs: ModelJobs = jobs if isinstance(jobs, ModelJobs) else ModelJobs(lambda: self.config.models)
        self.model_jobs.on_change = lambda state: self.bus.publish("model.busy", state)
        self.workspace = Workspace(budget_chars=self.config.memory.workspace_budget_chars)
        self.skills = SkillLibrary(
            [paths.skills_dir(), *[Path(d).expanduser() for d in self.config.skills.extra_dirs]]
        )
        self.skills_hub = SkillsHub(self.skills)  # finding and installing skills others wrote (#210)
        self.registry = ToolRegistry(disabled=self.config.tools.disabled)
        self.approvals = ApprovalBroker(self.config.tools.approvals, timeout_s=self.config.tools.approvals.timeout_s)
        # Read only connections (#141) are read from the live config on every check, so a switch applies at once
        self.approvals.read_only_source = lambda: self.config.integrations.read_only
        # "never" rules and the write tools of Read only connections are hidden from the model (ADR 0016)
        self.registry.set_blocked(self.approvals.is_hidden)
        self.chat_rules = ChatRules(self)  # "never delete my emails" in chat -> a proposed rule (#130)
        # how wide a yes to an approval request is: this task, a while, or an Allow rule (#139)
        self.approvals.grants = GrantStore(self)
        self.undo = UndoBroker()  # emails, messages and posts waiting in their undo window (#142)
        self.presence = Presence(self.bus)  # is the person looking at the window? (#139)
        self.sensors = SensorRegistry(self.bus)  # live microphones and cameras, here and on devices (#170)
        self.memory: FactMemory | None = None
        self.agent: Agent | None = None

        # feature services
        self.notifications = NotificationService(self)
        self.integrations = IntegrationManager(self)
        self.connections = ProviderConnections(self)  # OpenRouter sign-in, provider key checks
        self.hardware = HardwareProbe(self)  # memory and graphics card, for sizing local models (#131)
        self._hardware_warmup: asyncio.Task | None = None
        self.integrations.oauth_owners.append(self.connections)
        self.tasks = TaskService(self)
        self.reminders = ReminderService(self)  # timers and one-off reminders (#150)
        self.proactivity = ProactiveEngine(self)
        self.evolution = EvolutionService(self)
        self.voice = VoiceService(self)
        self.subagents = SubagentManager(self)
        self.snapshots = FileSnapshots(self)  # copies of files kept before code changes them (#142)
        self.sandbox = SandboxService(self)
        self.terminal = TerminalService(self)
        self.browser = BrowserService(self)
        self.nodes = NodeService(self)
        self.channels = ChannelService(self)
        self.user_model = UserModelService(self)
        self.dreaming = DreamingService(self)
        self.backups = BackupService(self)
        self.folders = FolderService(self)  # folders and Obsidian vaults as searchable notes (#176)
        self.memory_imports = MemoryImports(self)  # memory from ChatGPT, Claude and OpenClaw exports (#156)
        self.incognito = IncognitoSweeper(self)  # incognito chats are deleted when closed (#154)
        self.timeline = Timeline(self)  # what Sentient did and will do, read from the other packages' tables (#147)
        self._started = False
        self.stop_state: dict[str, Any] = {"stopped": False, "stopped_at": None, "source": None}
        # bumped by every stop_all: work queued before it (e.g. a chat message waiting behind a reply) is dropped
        self.stop_generation = 0
        # messages dropped by the last stop_all, {session_id: [text]}; services read it in halt()
        self.stop_dropped: dict[str, list[str]] = {}
        self._stop_lock = asyncio.Lock()

    # ------------------------------------------------------------------ helpers for services
    @property
    def services(self) -> list[Service]:
        return [
            self.notifications, self.integrations, self.snapshots, self.subagents, self.sandbox, self.terminal, self.browser, self.nodes,
            self.tasks, self.reminders, self.proactivity, self.evolution, self.user_model, self.dreaming, self.voice, self.channels,
            self.backups, self.folders,
            self.incognito,
        ]

    async def notify(self, kind: str, message: str, *, title: str | None = None, payload: dict | None = None) -> dict:
        return await self.notifications.create(kind, message, title=title, payload=payload)

    async def user_seen(self, surface: str) -> None:
        """The user did something (a chat message anywhere, the app in use, a paired chat): resets the away timer
        that pauses routines (#163). Never raises."""
        try:
            await self.tasks.away.seen(surface)
        except Exception:
            log.exception("could not record activity from %s", surface)

    def save_config(self, config: SentientConfig | None = None) -> None:
        """Persist and hot-apply configuration."""
        if config is not None:
            self.config = config
        moved = self._claude_code_switched()
        save_config(self.config)
        if self.config.integrations.mcp_server_enabled:
            # turning the MCP server on issues its own narrow token in the keychain (#443, ADR 0034);
            # a keychain problem must never break saving settings, the status endpoint reports it
            try:
                from sentient.gateway import mcp_token

                mcp_token.ensure_token()
            except Exception:
                pass
        if hasattr(self.llm, "config"):
            self.llm.config = self.config  # type: ignore[attr-defined]
        if self.agent is not None:
            self.agent.config = self.config
        if self.memory is not None:  # the user's name, recall and dedup settings take effect at once (#152)
            self.memory.config = self.config
            self.memory.episodic.config = self.config
        self.approvals.config = self.config.tools.approvals
        self.registry.set_hidden("subagents", not self.config.subagents.enabled)
        self.bus.publish("config.updated", {"sections": list(self.config.model_dump().keys())})
        if moved:
            self._note_claude_code(moved)

    def _claude_code_switched(self) -> list[dict]:
        """Claude Code doing everything (#444, ADR 0032) was just turned off: move the roles still on it that refuse it
        now to another model (in place, before saving). Returns what moved, or [] when nothing changed."""
        now = claude_code.everywhere(self.config)
        was, self._claude_code_everywhere = self._claude_code_everywhere, now
        if now != was:
            log.warning("Claude Code for everything turned %s", "on" if now else "off")
        if not was or now:
            return []
        moved = claude_code.move_leftovers(self.config)
        for item in moved:
            log.info("Claude Code for everything is off: role %s moves from %s to %s", item["role"], item["model"],
                     item["to"] or "nothing (no other model)")
        return moved

    def _note_claude_code(self, moved: list[dict]) -> None:
        """One notification saying which roles left Claude Code and which still need a model."""
        off = not self.config.models.experimental_claude_code  # turned off altogether, not just back to chats
        title = "Claude Code is turned off" if off else "Claude Code is back to chats only"
        try:
            task = asyncio.get_running_loop().create_task(
                self.notify("info", claude_code.leftover_note(moved, off=off), title=title,
                            payload={"kind": "claude_code_roles", "moved": moved})
            )
        except RuntimeError:  # no running loop (a command-line change): the log above says it
            return
        self._notes.add(task)
        task.add_done_callback(self._notes.discard)

    # ------------------------------------------------------------------ stop everything
    @property
    def stopped(self) -> bool:
        """True after Stop everything until Resume: nothing scheduled, triggered or proactive starts."""
        return bool(self.stop_state["stopped"])

    async def _load_stop_state(self) -> None:
        try:
            raw = json.loads(await self.store.get_meta(STOP_META_KEY) or "null")
        except ValueError:
            raw = None
        if isinstance(raw, dict) and raw.get("stopped"):
            self.stop_state = {"stopped": True, "stopped_at": raw.get("stopped_at"), "source": raw.get("source")}

    async def _set_stop_state(self, state: dict[str, Any]) -> None:
        self.stop_state = state
        await self.store.set_meta(STOP_META_KEY, json.dumps(state))
        self.bus.publish("stop.updated", dict(state))

    async def _tell_devices(self) -> None:
        try:
            await self.nodes.send_stop_state()  # bounded per device (nodes.STOP_STATE_SEND_S)
        except Exception:
            log.exception("could not tell devices about the stop state")

    async def stop_all(self, source: str = "desktop") -> dict[str, Any]:
        """Stop everything now, without the model: pause first so nothing new starts, then cancel what runs.

        Safe to call again while stopped (it cancels anything started since). Returns the stop state
        plus ``cancelled``, the number of running jobs that were cancelled."""
        async with self._stop_lock:
            changed = not self.stopped
            if changed:
                await self._set_stop_state({"stopped": True, "stopped_at": now_iso(), "source": source})
                log.warning("stop everything (from %s)", source)
            self.stop_generation += 1
            self.stop_dropped = self.agent.drop_queued() if self.agent is not None else {}
            cancelled = 0
            claude_code.hold()  # a Claude Code call waiting its turn never starts during the stop (#444)
            try:
                if self.agent is not None:
                    cancelled += await self.agent.halt()
                for svc in self.services:
                    try:
                        cancelled += await svc.halt()
                    except Exception:
                        log.exception("service %s failed to halt", svc.name)
                try:  # an import from ChatGPT, Claude or OpenClaw stops too; what it saved stays for review
                    cancelled += await self.memory_imports.halt()
                except Exception:
                    log.exception("stopping the memory import failed")
            finally:
                # a Claude Code reply left running by anything above (ADR 0022), and every call still waiting
                claude_code.kill_all()
            if changed:  # devices hear about it after the work is cancelled, never before
                await self._tell_devices()
            return {**self.stop_state, "cancelled": cancelled}

    async def resume(self, source: str = "desktop") -> dict[str, Any]:
        """Undo Stop everything: schedules, triggers and proactivity run again; interrupted work is picked up."""
        async with self._stop_lock:
            if self.stopped:
                await self._set_stop_state({"stopped": False, "stopped_at": None, "source": source})
                log.warning("resumed (from %s)", source)
                await self._tell_devices()
                if self._started and self.enable_background:
                    try:
                        await self.tasks.recover_interrupted()
                    except Exception:
                        log.exception("resuming interrupted tasks failed")
            return dict(self.stop_state)

    async def _snapshot_before_migration(self, reason: str) -> None:
        """Store hook: a snapshot of the data folder before the database changes shape (issue #151)."""
        await self.backups.snapshot(reason, db_path=self.store.path)

    async def delete_chat(self, session_id: str, *, reason: str | None = None) -> None:
        """Delete a chat: its running replies and helpers are stopped first, so none keeps writing to or working for a
        chat that is gone (a stopped reply saves what it kept, then goes with the chat)."""
        try:
            if self.agent is not None:
                await self.agent.cancel_session(session_id)
            await self.subagents.cancel_for_session(session_id)
        except Exception:
            log.exception("stopping the work of chat %s failed", session_id)
        incognito = await self.store.is_incognito(session_id)
        selected = await selections.attached(self.store, session_id)
        await self.store.delete_session(session_id)
        if incognito:  # the long tool results it saved go with it (#291); files it wrote on request stay
            try:
                await asyncio.to_thread(incognito_outputs.delete_outputs, session_id)
            except OSError:
                log.exception("removing the saved results of incognito chat %s failed", session_id)
        try:  # text selected in another app and asked about goes with its chat (#332)
            await selections.delete_with_chat(self.store, selected, incognito=incognito)
        except OSError:
            log.exception("removing the selected text of chat %s failed", session_id)
        self.bus.publish("session.deleted", {"session_id": session_id, **({"reason": reason} if reason else {})})

    async def _memory_without_model(self, title: str, message: str, job: str) -> None:
        """No model can do a memory job (#474): one notification, which Activity shows under Memory."""
        await self.notify("info", message, title=title, payload={"kind": "memory_no_model", "job": job})

    def _after_memory_erased(self, erased: Any) -> None:
        """A memory was deleted (#152): its words leave USER.md and MEMORY.md (not a remaining memory's similar lines,
        #456), and the user model reads its insights again (some may be gone with it)."""
        self.workspace.scrub(erased.wordings, self.config.assistant.user_name, repeats=erased)
        model = getattr(self, "user_model", None)
        if model is not None:
            model._vectors = None

    def tool_extra(self) -> dict[str, Any]:
        return {"app": self, "skills": self.skills, "workspace": self.workspace, "registry": self.registry}

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> SentientApp:
        if self._started:
            return self
        paths.ensure_layout()
        log.info("data folder %s, %s", paths.home(), secrets.describe())  # #302: secrets are per data folder
        await asyncio.to_thread(claude_code.sweep_scratch)  # folders an earlier run couldn't remove (ADR 0022)
        await self.store.open()
        await self._load_stop_state()  # before services start: a stopped Sentient starts paused
        assert self.approvals.grants is not None
        await self.approvals.grants.load()
        if self.config.integrations.mcp_server_enabled:
            # the MCP key is issued when the feature is turned on in Settings; a config edited by
            # hand reaches start with the feature on and no key, so issue it here too (#443)
            try:
                from sentient.gateway import mcp_token

                mcp_token.ensure_token()
            except Exception:
                log.exception("could not issue the MCP key")
        self.workspace.ensure_defaults(
            self.config.assistant.name, self.config.assistant.user_name, self.config.assistant.timezone
        )
        self.memory = FactMemory(self.store, self.llm, self.config) if self.store.vec_available else None
        if self.memory is None:
            log.warning("sqlite-vec unavailable: semantic memory disabled")
        else:
            self.memory.on_erase = self._after_memory_erased
            self.memory.on_no_model = self._memory_without_model

        with as_kind(self.notifications.model_kind):
            await self.notifications.start()
        with as_kind(self.integrations.model_kind):
            await self.integrations.start()  # registers integration plugins into self.registry
        self.registry.load_builtin()
        self.skills.reload(self._available_tools())
        self.agent = Agent(
            config=self.config,
            store=self.store,
            llm=self.llm,
            registry=self.registry,
            memory=self.memory,
            workspace=self.workspace,
            skills=self.skills,
            approvals=self.approvals,
            app=self,
        )
        await self.registry.setup_all(self.agent.tool_context(None, "system"))
        for svc in self.services[2:]:  # notifications and integrations started above
            try:
                with as_kind(svc.model_kind):  # the loops and listeners it starts call the model as this kind
                    await svc.start()
            except Exception:
                log.exception("service %s failed to start", svc.name)
        # services register plugins in start() (browser, code, devices, channels): re-check skills' requires_tools
        self.skills.reload(self._available_tools())
        if self.enable_background:  # the "Local only" preset and the check-up size models from it
            self._hardware_warmup = asyncio.create_task(self.hardware.get())
        self._started = True
        return self

    def _available_tools(self) -> set[str]:
        """Plugin ids and tool names a skill's ``requires_tools`` may name."""
        return {p.id for p in self.registry.plugins()} | {t.name for t in self.registry.tools(include_hidden=True)}

    async def stop(self) -> None:
        if self._hardware_warmup is not None and not self._hardware_warmup.done():
            self._hardware_warmup.cancel()
        await self.memory_imports.stop()
        for svc in reversed(self.services):
            try:
                await svc.stop()
            except Exception:
                log.exception("service %s failed to stop", svc.name)
        if self.agent is not None:
            await self.agent.drain(timeout=SHUTDOWN_GRACE_S)
        try:  # incognito chats are deleted when Sentient closes (#154), once nothing writes to them any more
            await self.incognito.purge(skip_running=False)
        except Exception:
            log.exception("deleting incognito chats at shutdown failed")
        claude_code.kill_all()
        await self.store.close()
        self._started = False
