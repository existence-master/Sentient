"""Human-in-the-loop gate for risky tool calls.

The agent loop emits an ``ApprovalRequest`` event and then awaits the broker.
Whichever client is attached (CLI prompt, web UI dialog, a phone notification,
the glasses) resolves it with ``allow``, ``allow_session`` or ``deny``.

Decisions use the call's *effective* risk (``Tool.risk_fn``). "Allow for this chat"
covers the tool up to the risk level that was approved: allowing ordinary browser
clicks never covers a call whose ``risk_fn`` raised it to ``send`` or ``exec`` ("Place order" asks every time).
A tool declared with ``allow_for_chat=False`` (commands on the host) is never covered by it.

Lasting rules (``tools.approvals.rules``, ADR 0016) are checked first, in code, never by a model. A key is a tool
name or a plugin id, and a tool's own rule beats its plugin's rule:

- ``never``: the tool is not offered to the model, and a call is refused without running.
- ``ask``: always ask, even in mode "off", after "Allow for this chat" and for read-only tools.
- ``allow``: run without asking, except purchases, which still ask whenever approvals are on.

Work nobody asked for (``ToolContext.origin`` in ``UNPROMPTED_ORIGINS``, ADR 0017) is checked before all of this:
it may only read and make Sentient-internal changes, whatever the mode or rules say (``unprompted_refusal``).

Once a run has read outside content (``ToolContext.untrusted``, ADR 0018), the agent loop asks for every call that
can send data out (``rules.sends_out``) after "never" rules and before this broker's modes and rules, so an Allow
rule, mode "off" or "Allow for this chat" never skip that question.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from sentient.config.schema import ApprovalsConfig
from sentient.tools.base import Risk, Tool, ToolContext, effective_risk
from sentient.tools.rules import (
    is_purchase,
    rule_for,
    rule_label,
    unprompted_allows,
    unprompted_message,
)

Decision = str  # "allow" | "allow_session" | "deny"


def _sync_effective_risk(tool: Tool, arguments: dict, ctx: Any) -> Risk:
    try:
        value: Any = tool.risk_fn(arguments or {}, ctx)  # type: ignore[misc]
    except Exception:
        return Risk.exec
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        return max(tool.risk, Risk.send)
    if value is None:
        return tool.risk
    if isinstance(value, str):
        return Risk[value] if value in Risk.__members__ else Risk.exec
    try:
        return Risk(value)
    except (TypeError, ValueError):
        return Risk.exec


@dataclass
class ApprovalBroker:
    config: ApprovalsConfig
    timeout_s: float = 600.0
    _pending: dict[str, asyncio.Future] = field(default_factory=dict)
    # (session_id, tool name) -> highest risk the user allowed for this chat
    _session_allow: dict[tuple[str, str], Risk] = field(default_factory=dict)

    # ------------------------------------------------------------------ lasting rules
    def rule(self, tool: Tool) -> str | None:
        """The lasting rule for ``tool`` ("allow", "ask", "never") or None (see ``sentient.tools.rules``)."""
        return rule_for(getattr(self.config, "rules", None), tool)

    def is_never(self, tool: Tool) -> bool:
        """True when a lasting rule says Sentient must never use ``tool`` (the registry hides it)."""
        return self.rule(tool) == "never"

    def label(self, tool: Tool, registry: Any = None) -> str:
        return rule_label(tool, getattr(self.config, "rules", None), registry)

    def unprompted_refusal(self, tool: Tool, risk: Risk, registry: Any = None) -> str | None:
        """For work nobody asked for: None when the call may run (a look-up or a Sentient-internal change), else
        the plain refusal. Checked before modes and rules, so an Allow rule never lifts it (ADR 0017)."""
        if unprompted_allows(tool, risk):
            return None
        return unprompted_message(self.label(tool, registry))

    async def requires_approval(
        self, tool: Tool, arguments: dict, ctx: ToolContext, session_id: str | None = None
    ) -> tuple[bool, Risk]:
        """Evaluate ``tool.risk_fn`` (sync or async) and decide. Returns ``(needs_approval, effective_risk)``."""
        risk = await effective_risk(tool, arguments, ctx)
        sid = session_id if session_id is not None else getattr(ctx, "session_id", None)
        return await self.decide(tool, sid, risk, arguments, ctx), risk

    async def decide(self, tool: Tool, session_id: str | None, risk: Risk, arguments: dict, ctx: Any) -> bool:
        """``needs_approval`` for a call whose effective risk is known. It also checks whether the call is
        a purchase: purchases always ask, whatever the mode, rules or "allow for this chat" say."""
        purchase = await is_purchase(tool, arguments, ctx, risk)
        return self.needs_approval(tool, session_id, risk, purchase=purchase)

    def needs_approval(
        self,
        tool: Tool,
        session_id: str | None,
        risk: Risk | None = None,
        *,
        arguments: dict | None = None,
        ctx: ToolContext | None = None,
        purchase: bool = False,
    ) -> bool:
        """``risk`` is the call's effective risk. Without it, pass ``arguments`` (and ``ctx``) so a synchronous
        ``tool.risk_fn`` is evaluated here; an async ``risk_fn`` cannot be awaited in this sync method and is
        treated as at least ``send`` (use ``await requires_approval(...)`` instead). ``purchase`` marks a call
        that spends money (``decide`` works it out); it always asks, even with approvals mode "off"."""
        if risk is None and arguments is not None and getattr(tool, "risk_fn", None) is not None:
            risk = _sync_effective_risk(tool, arguments, ctx)
        risk = tool.risk if risk is None else Risk(risk)
        mode = self.config.mode
        rule = self.rule(tool)
        if rule in {"ask", "never"}:  # "never" is refused before this; asking is the safe answer anyway
            return True
        if purchase:  # spending money always asks; browser.confirm_purchases is the only switch for it
            return True
        if rule == "allow":
            return False
        if mode == "off":
            return False
        # a tool with ``allow_for_chat=False`` (the terminal, ADR 0019) asks for every call
        if self.config.remember_session and session_id and getattr(tool, "allow_for_chat", True):
            allowed = self._session_allow.get((session_id, tool.name))
            # a call the tool's risk_fn raised to send/exec (a click on "Place order") always asks again
            escalated = risk >= Risk.send and risk > tool.risk
            if allowed is not None and allowed >= risk and not escalated:
                return False
        if mode == "always":
            return True
        # "ask": confirm things that leave Sentient or cannot be undone; the assistant's own
        # memory, skills (reviewed separately), files folder and task list do not interrupt.
        return risk >= Risk.write and not (tool.internal and risk < Risk.send)

    def create(self, approval_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[approval_id] = fut
        return fut

    async def wait(
        self, approval_id: str, session_id: str | None, tool_name: str, risk: Risk | None = None
    ) -> Decision:
        fut = self._pending.get(approval_id) or self.create(approval_id)
        try:
            decision: Decision = await asyncio.wait_for(fut, timeout=self.timeout_s)
        except TimeoutError:
            decision = "deny"
        finally:
            self._pending.pop(approval_id, None)
        if decision == "allow_session" and session_id:
            level = Risk.exec if risk is None else Risk(risk)
            key = (session_id, tool_name)
            self._session_allow[key] = max(level, self._session_allow.get(key, Risk.read))
        return decision

    def resolve(self, approval_id: str, decision: Decision) -> bool:
        fut = self._pending.get(approval_id)
        if fut is None or fut.done():
            return False
        fut.set_result(decision)
        return True

    def pending_ids(self) -> list[str]:
        return list(self._pending)
