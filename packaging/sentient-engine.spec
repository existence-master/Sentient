# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Sentient engine.

Produces a *onedir* bundle so the user never installs Python:

    packaging/dist/sentient-engine/sentient-engine.exe
        serve --host 127.0.0.1 --port <port>

Build it with ``python packaging/build_engine.py`` (or ``npm run package:engine``
from ``desktop/``), which also copies the result to ``desktop/build/engine`` where
electron-builder picks it up as ``extraResources``.

What a naive freeze misses, and is therefore listed explicitly below:
  * ``sqlite_vec/vec0.dll``  - loaded as a SQLite extension by path, never imported
  * ``sentient/**/schema.sql`` and ``sentient/nodes/web/*`` - read with ``Path(__file__)``
  * litellm + tiktoken data files, certifi's CA bundle, the tzdata database
  * keyring's Windows backend, which is only ever found dynamically
  * every ``sentient.*`` submodule, because tools and integration plugins are
    imported by name at runtime (``pkgutil.iter_modules`` / ``importlib``)

Heavy optional extras are excluded on purpose (see EXCLUDES): voice models are
downloaded on demand and the browser drives the user's own Edge/Chrome.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

REPO = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected by PyInstaller
PKG = REPO / "sentient"

datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []


def add_all(name: str) -> None:
    """Data files + binaries + submodules of a package that resists static analysis."""
    try:
        d, b, h = collect_all(name)
    except Exception as exc:  # pragma: no cover - a missing optional package is fine
        print(f"[spec] skipping {name}: {exc}")
        return
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


def add_data(name: str) -> None:
    try:
        datas.extend(collect_data_files(name))
    except Exception as exc:  # pragma: no cover
        print(f"[spec] no data files for {name}: {exc}")


def add_meta(name: str) -> None:
    try:
        datas.extend(copy_metadata(name))
    except Exception as exc:  # pragma: no cover
        print(f"[spec] no metadata for {name}: {exc}")


# --------------------------------------------------------------- Sentient's own data files
for sql in sorted(PKG.rglob("schema.sql")):
    datas.append((str(sql), str(sql.parent.relative_to(REPO))))
for asset in sorted((PKG / "nodes" / "web").rglob("*")):
    if asset.is_file():
        datas.append((str(asset), str(asset.parent.relative_to(REPO))))

# Tools, integration plugins and channels are imported by name at runtime.
hiddenimports += collect_submodules("sentient")

# --------------------------------------------------------------- third-party collection
# Packages with data files, plugin folders or dynamic imports.
for name in (
    "litellm",          # model price table, tokenizers, provider modules
    "tiktoken_ext",     # BPE registry looked up by entry-point-ish scan
    "sqlite_vec",       # vec0.dll
    "playwright",       # bundled driver (node) - the browsers stay the user's own Edge/Chrome
    "zeroconf",         # mDNS for device discovery
    "mcp",              # external MCP servers
    "keyring",          # OS keychain backends
    "pyttsx3",          # system TTS drivers, chosen by name
    "certifi",
    "tzdata",
):
    add_all(name)

for name in ("tiktoken", "docx", "ddgs", "feedparser", "segno", "pypdf", "truststore", "soundfile"):
    add_data(name)

# Version lookups at import time (importlib.metadata) need the .dist-info.
for name in (
    "litellm", "openai", "mcp", "ddgs", "httpx", "fastapi", "starlette", "uvicorn",
    "sqlite-vec", "keyring", "playwright", "tiktoken", "pypdf", "python-docx", "sentient",
):
    add_meta(name)

hiddenimports += [
    # uvicorn resolves these through "auto" indirections
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "websockets",
    "websockets.legacy",
    "httptools",
    # uvicorn.run(..., factory=True) imports this from a string
    "sentient.gateway.app",
    # tiktoken's encodings register themselves on import
    "tiktoken_ext.openai_public",
    # keyring only ever finds its backend dynamically
    "keyring.backends.Windows",
    "keyring.backends.SecretService",
    "keyring.backends.macOS",
    "win32ctypes.core",
    "win32ctypes.core.cffi",
    "win32ctypes.core.ctypes",
    # imported lazily or by name elsewhere
    "aioimaplib",
    "cryptography",
    "cryptography.hazmat.backends.openssl",
    "cryptography.x509",
    "feedparser",
    "ddgs",
    "segno",
    "pypdf",
    "docx",
    "json_repair",
    "multipart",
    "python_multipart",
    "dateutil.tz",
    "dateutil.rrule",
    "tzlocal",
    "aiosqlite",
    "encodings.idna",
    "numpy",  # sentient.voice.audio needs it even when the speech pack is absent
]

# --------------------------------------------------------------- what we leave out
# Voice models download on demand and the browser uses the user's installed
# Edge/Chrome, so none of this belongs in a first-run installer. The runtime hook
# turns an attempt to import them into a sentence the user can act on.
EXCLUDES = [
    # NOTE: litellm.proxy looks like 36 MB of dead weight (we use litellm as a library, never its
    # proxy server), but parts of litellm's normal completion path import litellm.proxy._types.
    # Excluding it saved 5 MB in the installer and is not worth the risk - leave it in.
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "onnxruntime_gpu",
    "kokoro_onnx",
    "openwakeword",
    "espeakng_loader",
    "phonemizer",
    "torch",
    "torchaudio",
    "torchvision",
    "cv2",
    "av",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "hf_xet",
    "sklearn",
    "scipy",
    "pandas",
    "matplotlib",
    "boto3",
    "botocore",
    "s3transfer",
    "nvidia",
    "tkinter",
    "IPython",
    "pytest",
    "_pytest",
    "respx",
    "ruff",
    "PyInstaller",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "notebook",
    "sympy",
    "numba",
]

a = Analysis(  # noqa: F821
    [str(REPO / "packaging" / "engine_entry.py")],
    pathex=[str(REPO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(REPO / "packaging" / "runtime_hooks" / "excluded_extras.py")],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sentient-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed DLLs trip antivirus heuristics; not worth the megabytes
    console=True,  # keeps stdout/stderr pipes; the shell spawns it with windowsHide
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(REPO / "desktop" / "resources" / "icon.ico"),
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="sentient-engine",
)
