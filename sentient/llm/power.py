"""Is this computer running on battery? (#149) Background work waits for the charger when it is.

No dependency: Windows asks ``GetSystemPowerStatus``, macOS reads ``pmset -g ps``, Linux reads
``/sys/class/power_supply``. ``None`` means unknown (a desktop without a battery reads as plugged in). The answer is
remembered for ``TTL_S`` seconds because the scheduler asks often.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

TTL_S = 20.0
_cache: tuple[float, bool | None] | None = None


def on_battery() -> bool:
    """True when the computer runs on battery right now (unknown counts as plugged in)."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < TTL_S:
        return bool(_cache[1])
    try:
        value = _probe()
    except Exception as exc:  # never let a power reading break a model call
        log.debug("battery check failed: %s", exc)
        value = None
    _cache = (now, value)
    return bool(value)


def _probe() -> bool | None:
    if sys.platform == "win32":
        return _windows()
    if sys.platform == "darwin":
        return _macos()
    return _linux(Path("/sys/class/power_supply"))


def _windows() -> bool | None:
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    if status.battery_flag == 128:  # no system battery
        return False
    return {0: True, 1: False}.get(status.ac_line)  # 255: unknown


def _macos() -> bool | None:
    out = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=2).stdout
    if "Battery Power" in out:
        return True
    if "AC Power" in out:
        return False
    return None


def _linux(root: Path) -> bool | None:
    if not root.is_dir():
        return None
    discharging = False
    for supply in root.iterdir():
        kind = _read(supply / "type")
        if kind in {"Mains", "USB", "USB_C", "USB_PD"} and _read(supply / "online") == "1":
            return False
        if kind == "Battery" and _read(supply / "status") == "Discharging":
            discharging = True
    return discharging


def _read(path: Path) -> str:
    with contextlib.suppress(OSError):
        return path.read_text().strip()
    return ""
