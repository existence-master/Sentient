"""Tool registry: every tool the model can call, grouped by plugin.

Plugins can be registered at startup (builtin tools, integrations), at runtime
(MCP servers connecting, tasks service) and removed again (MCP server deleted).
A plugin can be *hidden*: its tools stay registered (so an explicit call still
reaches the tool, which reports "not connected") but they are not offered to
the model and not listed in the catalog. Integrations hide disconnected apps so
small local models are not flooded with dozens of unusable tools.

A single tool can be *blocked* by a lasting "never" rule (``set_blocked``, ADR 0016): it is
not offered to the model and not listed for planners, but the Settings catalog still shows it
so the rule can be changed back.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterable

from sentient.tools.base import Tool, ToolContext, ToolPlugin

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, disabled: Iterable[str] = ()):
        self._plugins: dict[str, ToolPlugin] = {}
        self._tools: dict[str, Tool] = {}
        self._disabled = set(disabled)
        self._hidden: set[str] = set()
        self._blocked: Callable[[Tool], bool] | None = None

    def set_blocked(self, fn: Callable[[Tool], bool] | None) -> None:
        """``fn(tool) -> True`` hides that tool from the model (the approvals broker's "never" rules)."""
        self._blocked = fn

    def is_blocked(self, t: Tool) -> bool:
        return self._blocked is not None and bool(self._blocked(t))

    # ------------------------------------------------------------------ loading
    def register(self, plugin: ToolPlugin, *, replace: bool = False) -> None:
        """Add a plugin and its tools. Duplicate tool names raise unless ``replace`` is set
        and the existing tool belongs to the same plugin id."""
        if plugin.id in self._disabled:
            log.info("plugin %s disabled by config", plugin.id)
            return
        if replace and plugin.id in self._plugins:
            self.unregister(plugin.id)
        for t in plugin.tools:
            if t.name in self._disabled:
                continue
            if t.name in self._tools:
                raise ValueError(f"duplicate tool name {t.name} from plugin {plugin.id}")
        self._plugins[plugin.id] = plugin
        for t in plugin.tools:
            if t.name in self._disabled:
                continue
            t.plugin = plugin.id
            self._tools[t.name] = t

    def unregister(self, plugin_id: str) -> ToolPlugin | None:
        """Remove a plugin and every tool it registered."""
        plugin = self._plugins.pop(plugin_id, None)
        for name in [n for n, t in self._tools.items() if t.plugin == plugin_id]:
            self._tools.pop(name, None)
        self._hidden.discard(plugin_id)
        return plugin

    def set_hidden(self, plugin_id: str, hidden: bool) -> None:
        """Hide or show a plugin's tools from the model and the catalog."""
        if hidden:
            self._hidden.add(plugin_id)
        else:
            self._hidden.discard(plugin_id)

    def is_hidden(self, plugin_id: str) -> bool:
        return plugin_id in self._hidden

    def load_builtin(self) -> None:
        """Import every module under sentient.tools.builtin and register its PLUGIN."""
        from sentient.tools import builtin

        for mod in pkgutil.iter_modules(builtin.__path__):
            module = importlib.import_module(f"{builtin.__name__}.{mod.name}")
            plugin = getattr(module, "PLUGIN", None)
            if isinstance(plugin, ToolPlugin) and plugin.id not in self._plugins:
                self.register(plugin)

    async def setup_all(self, ctx: ToolContext) -> None:
        for p in list(self._plugins.values()):
            try:
                await p.setup(ctx)
            except Exception:  # a broken plugin must not take the assistant down
                log.exception("plugin %s failed to set up", p.id)

    # ------------------------------------------------------------------ queries
    def get(self, name: str) -> Tool | None:
        """Any registered tool, including tools of hidden plugins."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def plugin(self, plugin_id: str) -> ToolPlugin | None:
        return self._plugins.get(plugin_id)

    def _visible(self, t: Tool) -> bool:
        return t.plugin not in self._hidden and not self.is_blocked(t)

    def tools(self, *, include_hidden: bool = False) -> list[Tool]:
        return [t for t in self._tools.values() if include_hidden or self._visible(t)]

    def plugins(self) -> list[ToolPlugin]:
        return list(self._plugins.values())

    def openai_schemas(self, names: Iterable[str] | None = None) -> list[dict]:
        """Schemas offered to the model. Hidden plugins are never offered."""
        wanted = None if names is None else set(names)
        return [
            t.openai_schema()
            for n, t in self._tools.items()
            if self._visible(t) and (wanted is None or n in wanted)
        ]

    def catalog(self, *, include_hidden: bool = False, include_blocked: bool = False) -> list[dict]:
        """Serializable description for the UI and for the planner's tool list. Tools blocked by a
        "never" rule are left out unless ``include_blocked`` (the Settings screen needs them)."""
        out = []
        for p in self._plugins.values():
            hidden = p.id in self._hidden
            if hidden and not include_hidden:
                continue
            out.append(
                {
                    "id": p.id,
                    "display_name": p.display_name,
                    "description": p.description,
                    "category": p.category,
                    "icon": p.icon,
                    "auth": p.auth,
                    "selection_hint": p.selection_hint,
                    "hidden": hidden,
                    "tools": [
                        {"name": t.name, "description": t.description, "risk": t.risk.name}
                        for t in p.tools
                        if self._tools.get(t.name) is t and (include_blocked or not self.is_blocked(t))
                    ],
                }
            )
        return out
