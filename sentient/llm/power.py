"""Is this computer running on battery, and how full is it? (#149, #296) Sentient's own upkeep goes easy on battery.

No dependency: Windows asks ``GetSystemPowerStatus``, macOS reads ``pmset -g ps``, Linux reads
``/sys/class/power_supply``. ``None`` means unknown (a desktop without a battery reads as plugged in). The answer is
remembered for ``TTL_S`` seconds because the scheduler asks often.

**Faking it (tests and debugging only).** ``SENTIENT_DEBUG_BATTERY`` replaces the reading: ``ac`` means plugged in,
a number from 0 to 100 means on battery at that percent (``35``), and ``battery`` means on battery at an unknown
percent. Anything else is ignored. It exists so the battery rules can be checked on a computer that is plugged in;
nobody needs it in normal use.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

TTL_S = 20.0
DEBUG_ENV = "SENTIENT_DEBUG_BATTERY"


class PowerState(NamedTuple):
    on_battery: bool | None  # None: unknown (counts as plugged in)
    percent: int | None  # 0-100, None: unknown or no battery


UNKNOWN = PowerState(None, None)
_cache: tuple[float, PowerState] | None = None
_refreshing = False
_warned: str | None = None


def read() -> PowerState:
    """The power state right now. Never blocks for long: the scheduler asks from the event loop, so macOS's ``pmset``
    runs in a thread and the last reading is used meanwhile."""
    global _refreshing
    if (fake := _debug_override()) is not None:
        return fake
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < TTL_S:
        return _cache[1]
    if sys.platform == "darwin":
        if not _refreshing:
            _refreshing = True
            threading.Thread(target=_refresh, name="sentient-power", daemon=True).start()
        return _cache[1] if _cache is not None else UNKNOWN
    _refresh()
    return _cache[1] if _cache is not None else UNKNOWN


def on_battery() -> bool:
    """True when the computer runs on battery right now (unknown counts as plugged in)."""
    return bool(read().on_battery)


def percent() -> int | None:
    """How full the battery is (0-100) while on battery; None when plugged in or unknown."""
    state = read()
    return state.percent if state.on_battery else None


def _debug_override() -> PowerState | None:
    global _warned
    raw = os.environ.get(DEBUG_ENV, "").strip().lower()
    if not raw:
        return None
    if raw in {"ac", "plugged", "charging"}:
        state = PowerState(False, None)
    elif raw == "battery":
        state = PowerState(True, None)
    elif raw.isascii() and raw.isdigit() and 0 <= int(raw) <= 100:
        state = PowerState(True, int(raw))
    else:
        return None
    if _warned != raw:
        _warned = raw
        log.warning("battery reading faked by %s=%s (tests and debugging only)", DEBUG_ENV, raw)
    return state


def _refresh() -> None:
    global _cache, _refreshing
    try:
        value = _probe()
    except Exception as exc:  # never let a power reading break a model call
        log.debug("battery check failed: %s", exc)
        value = UNKNOWN
    if not isinstance(value, PowerState):  # a bare bool (older probes, tests)
        value = PowerState(value, None)
    _cache = (time.monotonic(), value)
    _refreshing = False


def _probe() -> PowerState:
    if sys.platform == "win32":
        return _windows()
    if sys.platform == "darwin":
        return _macos()
    return _linux(Path("/sys/class/power_supply"))


def _windows() -> PowerState:
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return UNKNOWN
    if status.battery_flag == 128:  # no system battery
        return PowerState(False, None)
    pct = int(status.percent) if status.percent <= 100 else None  # 255: unknown
    return PowerState({0: True, 1: False}.get(status.ac_line), pct)  # ac_line 255: unknown


def _macos() -> PowerState:
    out = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=2).stdout
    return parse_pmset(out)


def parse_pmset(out: str) -> PowerState:
    m = re.search(r"(\d{1,3})%", out)
    pct = min(100, int(m.group(1))) if m else None
    if "Battery Power" in out:
        return PowerState(True, pct)
    if "AC Power" in out:
        return PowerState(False, pct)
    return PowerState(None, pct)


def _linux(root: Path) -> PowerState:
    if not root.is_dir():
        return UNKNOWN
    discharging = False
    levels: list[int] = []
    for supply in root.iterdir():
        kind = _read(supply / "type")
        if kind in {"Mains", "USB", "USB_C", "USB_PD"} and _read(supply / "online") == "1":
            return PowerState(False, None)
        if kind == "Battery" and _read(supply / "status") == "Discharging":
            discharging = True
            cap = _read(supply / "capacity")
            if cap.isascii() and cap.isdigit():
                levels.append(min(100, int(cap)))
    # several discharging batteries: the lowest is the careful answer
    return PowerState(discharging, min(levels) if levels else None)


def _read(path: Path) -> str:
    with contextlib.suppress(OSError):
        return path.read_text().strip()
    return ""
