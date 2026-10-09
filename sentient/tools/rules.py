"""Lasting approval rules per tool or app (``tools.approvals.rules``, ADR 0016).

A rule key is a tool name (``gmail_send_email``, ``mcp_files_delete``) or a plugin id (``gmail``,
``mcp_files``) meaning every tool of that plugin. A tool's own rule beats its plugin's rule. Values:

- ``allow``: run without asking. Purchases still ask.
- ``ask``: always ask first, even when approvals are off, after "Allow for this chat" and for look-ups.
- ``never``: the tool is not offered to the model, and a call made anyway is refused without running.

Rules are applied in code by the approvals broker, the agent loop and the sandbox bridge; a model never
decides them. These helpers are pure so every one of those places can share them without import cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sentient.tools.base import Risk, Tool, describe_call

RULE_CHOICES = ("allow", "ask", "never")
# describe_fn label for money leaving the user (browser: buy, pay, order, subscribe, donate, transfer ...)
PURCHASE_LABEL = "Purchase"
SETTINGS_HINT = "Change this in Settings > Approvals & safety."


def rule_for(rules: Mapping[str, str] | None, tool: Tool) -> str | None:
    """The lasting rule for ``tool``: its own rule, else its plugin's rule, else None."""
    if not rules:
        return None
    rule = rules.get(tool.name)
    if rule is None:
        rule = rules.get(tool.plugin)
    return rule


def rule_label(tool: Tool, rules: Mapping[str, str] | None = None, registry: Any = None) -> str:
    """Plain name for refusal messages: the app's name when the rule is the app's, else the tool's name."""
    if rules and tool.name not in rules and tool.plugin in rules:
        plugin = registry.plugin(tool.plugin) if registry is not None else None
        return getattr(plugin, "display_name", None) or tool.plugin
    return tool.name


def never_message(label: str) -> str:
    return f"You've set Sentient to never use {label}. {SETTINGS_HINT}"


def unattended_ask_message(label: str) -> str:
    return (
        f"You've set Sentient to always ask before using {label}, and there is no one to ask while this runs "
        f"in the background, so it did not run. {SETTINGS_HINT}"
    )


async def is_purchase(tool: Tool, arguments: dict, ctx: Any, risk: Risk) -> bool:
    """True when the call spends money: effective risk ``send`` or higher and approval wording "Purchase"."""
    if Risk(risk) < Risk.send:
        return False
    wording = await describe_call(tool, arguments, ctx, Risk(risk))
    return wording.get("risk_label") == PURCHASE_LABEL
