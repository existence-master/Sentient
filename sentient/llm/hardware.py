"""What this computer can run: memory, graphics card and the local model that fits (#131).

Detection uses the standard library and tools that are already there (``nvidia-smi``, ``sysctl``, the Windows
registry, ``/sys`` on Linux, and Ollama's ``/api/ps`` as a hint). Every probe is best effort: one that fails reads as
unknown and detection never raises. :func:`recommend` picks a row of ``LOCAL_MODEL_TIERS`` from config/schema.py.
:class:`HardwareProbe` keeps the result for the app (``GET /api/system/hardware``, onboarding, the "Local only"
preset and the model check-up).
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import httpx

from sentient.config.schema import LOCAL_MODEL_TIERS, ModelRoles, ModelsConfig

log = logging.getLogger(__name__)

GIB = 1024**3
PROBE_TIMEOUT_S = 5
APPLE_GPU_SHARE = 2 / 3  # macOS lets the graphics side of Apple silicon use about this much of the memory
MIN_AMD_VRAM_GB = 2.0  # below this an AMD chip is a built-in one sharing system memory
CREATE_NO_WINDOW = 0x08000000
NVIDIA_SMI_ARGS = ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
WINDOWS_GPU_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _gb(n: float | None) -> float | None:
    return None if n is None else round(n, 1)


def _run(cmd: list[str]) -> str | None:
    """Output of a short command, or None when it is missing, fails or takes too long."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def vendor_of(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "quadro", "tesla")):
        return "nvidia"
    if any(k in n for k in ("amd", "radeon")):
        return "amd"
    if "intel" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return "other"


def _gpu(name: str, vram_gb: float | None, vendor: str | None = None) -> dict[str, Any]:
    vendor = vendor or vendor_of(name)
    # Ollama runs models on NVIDIA and AMD cards and Apple silicon; built-in Intel graphics share system memory
    usable = vendor in {"nvidia", "apple"} or (vendor == "amd" and (vram_gb or 0) >= MIN_AMD_VRAM_GB)
    return {"name": name.strip(), "vendor": vendor, "vram_gb": _gb(vram_gb), "usable": usable}


# ----------------------------------------------------------------------------- probes
def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Cards from ``nvidia-smi --query-gpu=name,memory.total --format=csv``: lines like
    ``NVIDIA GeForce RTX 4060, 8188 MiB`` (memory in MiB; a header line and the unit are optional)."""
    gpus: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        if "," not in line:
            continue
        name, _, mem = line.rpartition(",")
        mem = mem.strip().lower().removesuffix("mib").strip()
        if not name.strip() or name.strip().lower() == "name":
            continue
        try:
            vram = float(mem) / 1024
        except ValueError:
            vram = None
        gpus.append(_gpu(name, vram, "nvidia"))
    return gpus


def nvidia_gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        legacy = os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI",
                              "nvidia-smi.exe")
        exe = legacy if os.path.exists(legacy) else None
    if exe is None:
        return []
    return parse_nvidia_smi(_run([exe, *NVIDIA_SMI_ARGS]) or "")


def windows_gpus() -> list[dict[str, Any]]:
    """Graphics adapters from the registry, with their dedicated memory when the driver records it."""
    import winreg

    gpus: list[dict[str, Any]] = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, WINDOWS_GPU_CLASS) as root:
        for i in range(64):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    size = None
                    for value in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        with contextlib.suppress(OSError):
                            raw = winreg.QueryValueEx(key, value)[0]
                            size = int.from_bytes(raw[:8], "little") if isinstance(raw, bytes) else int(raw)
                            break
            except OSError:
                continue
            gpus.append(_gpu(name, size / GIB if size else None))
    return gpus


def linux_amd_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        device = os.path.dirname(path)
        try:
            with open(os.path.join(device, "vendor")) as f:
                if f.read().strip() != "0x1002":
                    continue
            with open(path) as f:
                vram = int(f.read().strip()) / GIB
        except (OSError, ValueError):
            continue
        name = "AMD graphics card"
        with contextlib.suppress(OSError), open(os.path.join(device, "product_name")) as f:
            name = f.read().strip() or name
        gpus.append(_gpu(name, vram, "amd"))
    return gpus


def total_ram_gb() -> float | None:
    try:
        import psutil  # not a dependency; used when something else installed it

        return psutil.virtual_memory().total / GIB
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                        ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                        ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return status.total / GIB
        return None
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out.strip()) / GIB if out and out.strip().isdigit() else None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (ValueError, OSError, AttributeError):
        return None


def _safe(probe, default):
    try:
        return probe()
    except Exception as exc:  # one failed probe never stops the others
        log.debug("hardware probe %s failed: %s", getattr(probe, "__name__", probe), exc)
        return default


def detect() -> dict[str, Any]:
    """This computer's memory and graphics cards. Never raises; unknown values are None."""
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux" if sys.platform.startswith("linux")
                                                          else sys.platform)
    ram = _safe(total_ram_gb, None)
    gpus: list[dict[str, Any]] = _safe(nvidia_gpus, [])
    unified = False
    if system == "windows":
        have_nvidia = bool(gpus)
        gpus += [g for g in _safe(windows_gpus, []) if not (have_nvidia and g["vendor"] == "nvidia")]
    elif system == "linux":
        gpus += _safe(linux_amd_gpus, [])
    elif system == "macos" and platform.machine() == "arm64":
        unified = True
        name = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or "Apple silicon"
        gpus.append(_gpu(name, ram * APPLE_GPU_SHARE if ram else None, "apple"))
    usable = [g["vram_gb"] for g in gpus if g["usable"] and g["vram_gb"]]
    return {
        "os": system,
        "ram_gb": _gb(ram),
        "gpus": gpus,
        "unified_memory": unified,
        "usable_vram_gb": max(usable) if usable else None,
        "ollama_vram_gb": None,
    }


async def ollama_vram_gb(base: str) -> float | None:
    """Graphics memory the models Ollama has loaded use right now (``/api/ps``): proof of at least that much."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{base.rstrip('/')}/api/ps")
            r.raise_for_status()
            total = sum(int(m.get("size_vram") or 0) for m in r.json().get("models") or [])
    except Exception:
        return None
    return _gb(total / GIB) if total > 0 else None


# ----------------------------------------------------------------------------- recommendation
def _fmt(n: float) -> str:
    return f"{n:.0f}" if abs(n - round(n)) < 0.05 else f"{n:.1f}"


def summary(hw: dict[str, Any]) -> str:
    parts: list[str] = []
    best = max((g for g in hw.get("gpus") or [] if g.get("usable")), key=lambda g: g.get("vram_gb") or 0, default=None)
    if best and hw.get("unified_memory"):
        parts.append(best["name"])
    elif best:
        vram = f" with {_fmt(best['vram_gb'])} GB of graphics memory" if best.get("vram_gb") else ""
        parts.append(best["name"] + vram)
    elif hw.get("ollama_vram_gb"):
        parts.append(f"a graphics card Ollama uses (at least {_fmt(hw['ollama_vram_gb'])} GB)")
    else:
        parts.append("no graphics card a local model can use")
    if hw.get("ram_gb"):
        parts.append(f"{_fmt(hw['ram_gb'])} GB of memory")
    return ", ".join(parts)


NOTES = {
    "graphics": "Fits on the graphics card, so replies stay quick.",
    "processor": "No graphics card a local model can use was found, so it runs on the processor. Replies will be "
    "slow; a cloud model is much faster.",
    "small": "This computer has less than 12 GB of memory, so only a small model fits, and it can't use tools "
    "reliably. A cloud model is the better choice here.",
    "unknown": "Sentient couldn't check this computer's memory, so this is the usual starting point.",
}


def recommend(hw: dict[str, Any] | None) -> dict[str, Any]:
    """The local model and context length for this computer: the first ``LOCAL_MODEL_TIERS`` row it meets."""
    hw = hw or {}
    vram = max(hw.get("usable_vram_gb") or 0, hw.get("ollama_vram_gb") or 0) or None
    ram = hw.get("ram_gb")
    if vram is None and ram is None:
        row = {"id": "unknown", "model": ModelRoles.model_fields["primary"].default,
               "context_length": ModelsConfig.model_fields["context_length"].default}
        runs_on = "unknown"
    else:
        row = next(
            r for r in LOCAL_MODEL_TIERS
            if ("min_vram_gb" in r and vram is not None and vram >= r["min_vram_gb"])
            or ("min_ram_gb" in r and (ram is None or ram >= r["min_ram_gb"]))
        )
        runs_on = "graphics" if "min_vram_gb" in row else "processor"
    name = row["model"].split("/", 1)[1]
    note_key = "small" if row["id"] == "small" else runs_on
    return {
        "tier": row["id"],
        "model": row["model"],
        "name": name,
        "context_length": row["context_length"],
        "runs_on": runs_on,
        "summary": f"{name}, reading {row['context_length']:,} tokens at a time",
        "note": NOTES[note_key],
    }


class HardwareProbe:
    """Detects once and keeps the answer (hardware doesn't change while Sentient runs; ``refresh`` checks again)."""

    def __init__(self, app: Any):
        self.app = app
        self._info: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    def peek(self) -> dict[str, Any] | None:
        """The last answer without detecting: None until :meth:`get` has run."""
        return self._info

    def _ollama_base(self) -> str:
        providers = self.app.config.models.providers
        pc = providers.get("ollama_chat") or providers.get("ollama")
        return (pc.api_base if pc and pc.api_base else "http://localhost:11434").rstrip("/")

    async def get(self, refresh: bool = False) -> dict[str, Any]:
        """``detect()`` plus ``summary`` and ``recommendation``. Never raises."""
        async with self._lock:
            if self._info is None or refresh:
                try:
                    info = await asyncio.to_thread(detect)
                except Exception as exc:
                    log.debug("hardware detection failed: %s", exc)
                    info = {"os": None, "ram_gb": None, "gpus": [], "unified_memory": False,
                            "usable_vram_gb": None, "ollama_vram_gb": None}
                info["ollama_vram_gb"] = await ollama_vram_gb(self._ollama_base())
                known = info["ram_gb"] is not None or bool(info["gpus"]) or info["ollama_vram_gb"] is not None
                info["summary"] = summary(info) if known else "unknown"
                info["recommendation"] = recommend(info)
                self._info = info
            return self._info
