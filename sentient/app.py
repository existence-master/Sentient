"""Composition root: builds every subsystem once.

The desktop app launches ``sentient serve``; the gateway owns one SentientApp
for the life of the process. Tests construct SentientApp directly with a fake
LLM provider and a temporary database.

Start order (stop runs in reverse):
    store -> memory -> notifications -> integrations (registers plugins)
    -> builtin tools -> skills -> agent -> subagents -> sandbox -> browser -> nodes
    -> tasks -> proactivity -> evolution -> user_model -> dreaming -> voice -> channels
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.agent.approvals import ApprovalBroker
from sentient.agent.loop import Agent
from sentient.agent.subagents import SubagentManager
from sentient.browser import BrowserService
from sentient.channels import ChannelService
from sentient.config import SentientConfig, load_config, save_config
from sentient.events import EventBus
from sentient.evolution import EvolutionService
from sentient.integrations import IntegrationManager
from sentient.llm.provider import LiteLLMProvider, LLMProvider
from sentient.memory.dreaming import DreamingService
from sentient.memory.facts import FactMemory
from sentient.memory.usermodel import UserModelService
from sentient.memory.workspace import Workspace
from sentient.nodes import NodeService
from sentient.notifications import NotificationService
from sentient.proactivity import ProactiveEngine
from sentient.sandbox import SandboxService
from sentient.services import Service
from sentient.skills.loader import SkillLibrary
from sentient.store.db import Store
from sentient.tasks import TaskService
from sentient.tools.registry import ToolRegistry
from sentient.voice import VoiceService

log = logging.getLogger(__name__)

# Background work gets this long to finish when Sentient shuts down; then it is cancelled.
SHUTDOWN_GRACE_S = 15.0


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
        self.enable_background = enable_background  # tests turn schedulers/pollers off
        self.bus = EventBus()
        self.store = Store(db_path)
        self.llm: LLMProvider = llm or LiteLLMProvider(self.config)
        self.workspace = Workspace(budget_chars=self.config.memory.workspace_budget_chars)
        self.skills = SkillLibrary(
            [paths.skills_dir(), *[Path(d).expanduser() for d in self.config.skills.extra_dirs]]
        )
        self.registry = ToolRegistry(disabled=self.config.tools.disabled)
        self.approvals = ApprovalBroker(self.config.tools.approvals, timeout_s=self.config.tools.approvals.timeout_s)
        self.registry.set_blocked(self.approvals.is_never)  # "never" rules hide tools from the model (ADR 0016)
        self.memory: FactMemory | None = None
        self.agent: Agent | None = None

        # feature services
        self.notifications = NotificationService(self)
        self.integrations = IntegrationManager(self)
        self.tasks = TaskService(self)
        self.proactivity = ProactiveEngine(self)
        self.evolution = EvolutionService(self)
        self.voice = VoiceService(self)
        self.subagents = SubagentManager(self)
        self.sandbox = SandboxService(self)
        self.browser = BrowserService(self)
        self.nodes = NodeService(self)
        self.channels = ChannelService(self)
        self.user_model = UserModelService(self)
        self.dreaming = DreamingService(self)
        self._started = False

    # ------------------------------------------------------------------ helpers for services
    @property
    def services(self) -> list[Service]:
        return [
            self.notifications, self.integrations, self.subagents, self.sandbox, self.browser, self.nodes,
            self.tasks, self.proactivity, self.evolution, self.user_model, self.dreaming, self.voice, self.channels,
        ]

    async def notify(self, kind: str, message: str, *, title: str | None = None, payload: dict | None = None) -> dict:
        return await self.notifications.create(kind, message, title=title, payload=payload)

    def save_config(self, config: SentientConfig | None = None) -> None:
        """Persist and hot-apply configuration."""
        if config is not None:
            self.config = config
        save_config(self.config)
        if hasattr(self.llm, "config"):
            self.llm.config = self.config  # type: ignore[attr-defined]
        if self.agent is not None:
            self.agent.config = self.config
        self.approvals.config = self.config.tools.approvals
        self.registry.set_hidden("subagents", not self.config.subagents.enabled)
        self.bus.publish("config.updated", {"sections": list(self.config.model_dump().keys())})

    def tool_extra(self) -> dict[str, Any]:
        return {"app": self, "skills": self.skills, "workspace": self.workspace, "registry": self.registry}

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> SentientApp:
        if self._started:
            return self
        paths.ensure_layout()
        await self.store.open()
        self.workspace.ensure_defaults(
            self.config.assistant.name, self.config.assistant.user_name, self.config.assistant.timezone
        )
        self.memory = FactMemory(self.store, self.llm, self.config) if self.store.vec_available else None
        if self.memory is None:
            log.warning("sqlite-vec unavailable: semantic memory disabled")

        await self.notifications.start()
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
                await svc.start()
            except Exception:
                log.exception("service %s failed to start", svc.name)
        # services register plugins in start() (browser, code, devices, channels): re-check skills' requires_tools
        self.skills.reload(self._available_tools())
        self._started = True
        return self

    def _available_tools(self) -> set[str]:
        """Plugin ids and tool names a skill's ``requires_tools`` may name."""
        return {p.id for p in self.registry.plugins()} | {t.name for t in self.registry.tools(include_hidden=True)}

    async def stop(self) -> None:
        for svc in reversed(self.services):
            try:
                await svc.stop()
            except Exception:
                log.exception("service %s failed to stop", svc.name)
        if self.agent is not None:
            await self.agent.drain(timeout=SHUTDOWN_GRACE_S)
        await self.store.close()
        self._started = False
