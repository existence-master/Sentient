"""Which tools a script may call through the tool bridge (docs/API.md section 11).

Scripts run after the user (or approvals mode) allowed ``execute_code`` once; they cannot stop
to ask again. So from inside a script only calls that need no approval run: effective risk
``read`` and ``internal`` ``write`` tools. Anything else is refused with a message telling the
model to make that call directly, where approvals can ask the user. Approvals mode ``off``
allows everything except tools that would recurse (code, subagents, voice), commands on this
computer (terminal, ADR 0019) and purchases, which always ask.

Lasting rules (ADR 0016) only ever take away here: a "never" tool is not listed and is refused,
and an "ask" tool is refused because a script cannot stop to ask. An "allow" rule changes nothing:
scripts still only read (ADR 0012, docs/PRIVACY.md). A connection set to Read only (#141) takes away the same way,
even with approvals mode ``off``: its tools that can change things are not listed and calls above ``read`` are refused.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field

from sentient.tools.base import Risk, Tool, ToolContext, effective_risk
from sentient.tools.rules import (
    SETTINGS_HINT,
    is_purchase,
    never_message,
    read_only_blocks,
    read_only_hides,
    read_only_message,
    rule_for,
    rule_label,
)

BLOCKED_PLUGINS = frozenset({"code", "subagents", "voice", "terminal"})
BLOCKED_TOOLS = frozenset({"execute_code", "delegate_task", "delegate_tasks", "terminal_run"})


class Refused(Exception):
    """A tool call from a script that the policy does not allow."""


@dataclass
class BridgePolicy:
    approvals_mode: str = "ask"
    allowed_tools: frozenset[str] | None = None  # tool names or plugin ids; None = every tool
    read_only: bool = False                      # POST /api/sandbox/run: only risk read
    max_tool_calls: int = 200
    rules: Mapping[str, str] = field(default_factory=dict)  # tools.approvals.rules
    # read the owner's rules on every call, so a rule changed while a script runs applies at once
    rules_source: Callable[[], Mapping[str, str] | None] | None = None
    # app ids of Read only connections (``integrations.read_only``), read on every call like the rules
    read_only_apps_source: Callable[[], Collection[str] | None] | None = None

    @classmethod
    def build(
        cls,
        *,
        approvals_mode: str,
        allowed_tools: Iterable[str] | None,
        read_only: bool,
        max_tool_calls: int,
        rules: Mapping[str, str] | None = None,
        rules_source: Callable[[], Mapping[str, str] | None] | None = None,
        read_only_apps_source: Callable[[], Collection[str] | None] | None = None,
    ) -> BridgePolicy:
        return cls(
            approvals_mode=approvals_mode,
            allowed_tools=None if allowed_tools is None else frozenset(allowed_tools),
            read_only=read_only,
            max_tool_calls=max_tool_calls,
            rules=dict(rules or {}),
            rules_source=rules_source,
            read_only_apps_source=read_only_apps_source,
        )

    def current_rules(self) -> Mapping[str, str]:
        if self.rules_source is not None:
            return self.rules_source() or {}
        return self.rules

    def read_only_apps(self) -> Collection[str]:
        return (self.read_only_apps_source() if self.read_only_apps_source is not None else None) or ()

    def is_available(self, tool: Tool) -> bool:
        """Tools a script can see at all (listed in the generated sentient_tools module)."""
        if tool.plugin in BLOCKED_PLUGINS or tool.name in BLOCKED_TOOLS:
            return False
        if rule_for(self.current_rules(), tool) == "never" or read_only_hides(self.read_only_apps(), tool):
            return False
        return self.allowed_tools is None or tool.name in self.allowed_tools or tool.plugin in self.allowed_tools

    async def check(self, tool: Tool | None, name: str, arguments: dict, ctx: ToolContext, calls_so_far: int) -> Risk:
        """Return the effective risk when the call may run, else raise ``Refused``."""
        if tool is None:
            raise Refused(f"There is no tool named '{name}'. Use tools.available() to list the tools you can call.")
        if tool.plugin in BLOCKED_PLUGINS or tool.name in BLOCKED_TOOLS:
            raise Refused(f"'{name}' is not available inside scripts. Call it directly as a normal tool call instead.")
        rules = self.current_rules()
        rule = rule_for(rules, tool)
        label = rule_label(tool, rules, (getattr(ctx, "extra", None) or {}).get("registry"))
        if rule == "never":
            raise Refused(never_message(label))
        if rule == "ask":
            raise Refused(
                f"You've set Sentient to always ask before using {label}, and a script cannot stop to ask. "
                f"Call '{name}' directly as a normal tool call instead. {SETTINGS_HINT}"
            )
        if read_only_hides(self.read_only_apps(), tool):
            raise Refused(read_only_message(tool, (getattr(ctx, "extra", None) or {}).get("registry")))
        if not self.is_available(tool):
            raise Refused(f"'{name}' is not one of the tools this script may use.")
        if calls_so_far >= self.max_tool_calls:
            raise Refused(
                f"This script already made {self.max_tool_calls} tool calls, the most one script may make. "
                "Split the work into smaller scripts."
            )
        risk = await effective_risk(tool, arguments, ctx)
        if read_only_blocks(self.read_only_apps(), tool, risk):
            raise Refused(read_only_message(tool, (getattr(ctx, "extra", None) or {}).get("registry")))
        if self.read_only:
            if risk > Risk.read:
                raise Refused(
                    f"'{name}' can change things (risk {risk.name}); scripts run from here may only look things up."
                )
            return risk
        if await is_purchase(tool, arguments, ctx, risk):
            raise Refused(
                f"'{name}' would spend money, which always needs the user's yes. Call '{name}' directly as a "
                "normal tool call instead."
            )
        if self.approvals_mode == "off":
            return risk
        if risk == Risk.read or (risk == Risk.write and tool.internal):
            return risk
        raise Refused(
            f"'{name}' can {_verb(risk)} (risk {risk.name}), so it cannot run from inside a script where the user "
            f"cannot approve it. Call '{name}' directly as a normal tool call instead."
        )


def _verb(risk: Risk) -> str:
    return {
        Risk.write: "change things outside Sentient",
        Risk.send: "send, delete or spend",
        Risk.exec: "run code or commands",
    }.get(risk, "change things")
