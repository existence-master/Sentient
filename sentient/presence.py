"""Whether the person is looking at Sentient's window right now (#139).

The window reports ``{"type": "presence", "active": bool}`` over ``/ws`` when it gains or loses focus or is hidden.
A connection that never reported counts as away, so an older window, a closed window or the engine on its own
never hold anything back. ``presence.changed`` ``{in_app}`` is published when the overall answer flips; channels
use it to send approval requests to the paired phone chat only while the person is away from the app.
"""

from __future__ import annotations

from typing import Any


class Presence:
    def __init__(self, bus: Any):
        self.bus = bus
        self._clients: dict[str, bool] = {}

    def in_app(self) -> bool:
        """True while at least one window says it is focused and visible."""
        return any(self._clients.values())

    def set(self, client: str, active: bool) -> None:
        before = self.in_app()
        self._clients[client] = bool(active)
        self._changed(before)

    def drop(self, client: str) -> None:
        before = self.in_app()
        self._clients.pop(client, None)
        self._changed(before)

    def _changed(self, before: bool) -> None:
        now = self.in_app()
        if now != before:
            self.bus.publish("presence.changed", {"in_app": now})
