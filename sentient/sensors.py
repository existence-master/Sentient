"""Which microphones and cameras are live right now, on this computer and on paired devices (#170).

Voice (``sentient.voice``) and devices (``sentient.nodes``) report every microphone or camera they turn on or off
here; nothing else decides it. Each change publishes the domain event ``sensors.updated`` with the whole
**SensorState**, so the window's title bar and the tray can show an indicator within a second (docs/API.md
section 24). A use can carry a ``stop`` callback: the indicator's Turn off button ends it through
:meth:`SensorRegistry.stop`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from sentient.store.db import now_iso

log = logging.getLogger(__name__)

Sensor = Literal["mic", "camera"]
# a stop callback answers True when the microphone or camera is off now, False when the device was asked to turn
# it off and has not said so yet
StopFn = Callable[[], Awaitable[bool]]


@dataclass
class SensorUse:
    id: str
    sensor: Sensor
    device_id: str
    device: str
    device_kind: str
    purpose: str
    since: str
    stop: StopFn | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sensor": self.sensor,
            "device_id": self.device_id,
            "device": self.device,
            "device_kind": self.device_kind,
            "purpose": self.purpose,
            "since": self.since,
            "can_stop": self.stop is not None,
        }


class SensorRegistry:
    def __init__(self, bus: Any):
        self.bus = bus
        self._uses: dict[str, SensorUse] = {}
        self._camera_off: dict[str, str] = {}  # device_id -> device name
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # ------------------------------------------------------------------ reading
    def state(self) -> dict[str, Any]:
        """**SensorState** ``{mic, camera, active: [SensorUse], camera_off: [{device_id, device}]}``."""
        active = sorted(self._uses.values(), key=lambda u: (u.since, u.id))
        return {
            "mic": any(u.sensor == "mic" for u in active),
            "camera": any(u.sensor == "camera" for u in active),
            "active": [u.public() for u in active],
            "camera_off": [{"device_id": d, "device": name} for d, name in sorted(self._camera_off.items())],
        }

    def is_on(self, use_id: str) -> bool:
        return use_id in self._uses

    def get(self, use_id: str) -> SensorUse | None:
        return self._uses.get(use_id)

    # ------------------------------------------------------------------ reporting
    def on_change(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Call ``listener(state)`` after every change (the nodes service tells the desktop app this way)."""
        self._listeners.append(listener)

    def start(
        self,
        use_id: str,
        sensor: Sensor,
        *,
        device_id: str,
        device: str,
        device_kind: str,
        purpose: str,
        stop: StopFn | None = None,
    ) -> None:
        """A microphone or camera turned on (or a live one changed what it is for)."""
        old = self._uses.get(use_id)
        if old is not None:
            if (old.sensor, old.device, old.purpose) == (sensor, device, purpose):
                old.stop = stop or old.stop
                return
            self._uses[use_id] = SensorUse(use_id, sensor, device_id, device, device_kind, purpose, old.since, stop or old.stop)
        else:
            self._uses[use_id] = SensorUse(use_id, sensor, device_id, device, device_kind, purpose, now_iso(), stop)
        self._changed()

    def update(self, use_id: str, *, purpose: str) -> None:
        use = self._uses.get(use_id)
        if use is not None and use.purpose != purpose:
            use.purpose = purpose
            self._changed()

    def end(self, use_id: str) -> None:
        if self._uses.pop(use_id, None) is not None:
            self._changed()

    def end_prefix(self, prefix: str) -> None:
        """End every use whose id starts with ``prefix`` (a device that disconnected)."""
        gone = [k for k in self._uses if k.startswith(prefix)]
        for k in gone:
            del self._uses[k]
        if gone:
            self._changed()

    def set_camera_off(self, device_id: str, device: str, off: bool) -> None:
        """The device says its camera is switched off on the device itself (or back on)."""
        if off and self._camera_off.get(device_id) != device:
            self._camera_off[device_id] = device
            self._changed()
        elif not off and self._camera_off.pop(device_id, None) is not None:
            self._changed()

    def _changed(self) -> None:
        state = self.state()
        self.bus.publish("sensors.updated", state)
        for listener in list(self._listeners):
            try:
                listener(state)
            except Exception:
                log.exception("sensors: a listener failed")

    # ------------------------------------------------------------------ turning off
    async def stop(self, use_id: str | None = None) -> dict[str, list[str]]:
        """Turn off one use (``use_id``) or all of them. ``stopped``: off now; ``asked``: the device was asked."""
        ids = [use_id] if use_id is not None else list(self._uses)
        out: dict[str, list[str]] = {"stopped": [], "asked": []}
        for uid in ids:
            use = self._uses.get(uid)
            if use is None or use.stop is None:
                continue
            done = False
            try:
                done = await use.stop()
            except Exception:
                log.exception("sensors: turning off %s failed", uid)
            if done:
                self.end(uid)
                out["stopped"].append(uid)
            else:
                out["asked"].append(uid)
        return out
